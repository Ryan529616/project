#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py

Main trainer for hardware-aware detector.
Canonical names:
- detector: hardware-aware detection model
"""

from __future__ import annotations

import argparse
import faulthandler
import glob
import json
import logging
import math
import os
import re
import shutil
import signal
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import amp as torch_amp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import Cfg
from data.dataloaders import build_dataloaders
from engine.step import train_one_epoch, validate
from models.builder import (
    attach_bridge_with_cfg,
    build_detector,
    build_detector_loss,
    build_optimizer_for,
    build_schedulers,
    dump_model_param_report,
    freeze_module,
    maybe_channels_last,
    maybe_compile,
)
from utils.checkpoint import load_dual_ckpt, load_weights, save_dual_ckpt
from utils.ema import EMA
from utils.trainlog import StepLogger


def set_seed(seed: int = 1337) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _coerce_atom(s: str):
    t = s.strip()
    tl = t.lower()
    if tl in ("true", "false"):
        return tl == "true"
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        try:
            return json.loads(t)
        except Exception:
            pass
    try:
        if t.isdigit() or (t[0] == "-" and t[1:].isdigit()):
            return int(t)
    except Exception:
        pass
    try:
        return float(t)
    except Exception:
        return t


def _coerce_val(v: str):
    if "," in v and not (v.startswith("[") and v.endswith("]")):
        return [_coerce_atom(x) for x in v.split(",")]
    return _coerce_atom(v)


def _load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_state(obj):
    try:
        return obj.state_dict() if obj is not None else None
    except Exception:
        return None


def _resolve_resume_path(
    path_like: Optional[str], default_dir: Optional[str] = None, logger: Optional[logging.Logger] = None
) -> Optional[str]:
    log = logger or logging.getLogger("resume")
    if not path_like and not default_dir:
        return None

    def _pick_latest_in_dir(d: str) -> Optional[str]:
        if not d or not os.path.isdir(d):
            return None
        candidates: List[Tuple[int, float, str]] = []
        for p in glob.glob(os.path.join(d, "epoch_*.pt")):
            m_ep = re.match(r".*epoch_(\d+)\.pt$", p)
            if m_ep:
                candidates.append((1, float(m_ep.group(1)), p))
        for p in glob.glob(os.path.join(d, "best_*.pt")):
            try:
                candidates.append((2, float(os.path.getmtime(p)), p))
            except Exception:
                pass
        steps_dir = os.path.join(d, "steps")
        if os.path.isdir(steps_dir):
            for p in glob.glob(os.path.join(steps_dir, "*.pt")):
                m_st = re.match(r".*gstep(\d+).*\.pt$", p)
                if m_st:
                    candidates.append((0, float(m_st.group(1)), p))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], -x[1]))
        return candidates[0][2]

    s = (path_like or "").strip().lower()
    if s in ("auto", "latest"):
        best = _pick_latest_in_dir(default_dir or "")
        if best:
            log.info(f"[resume] auto -> {best}")
        return best
    if os.path.isfile(path_like or ""):
        return path_like
    if os.path.isdir(path_like or ""):
        return _pick_latest_in_dir(path_like or "")
    return None


def collect_loss_overrides(unknown_argv: list) -> Dict[str, Any]:
    dl: Dict[str, Any] = {}
    for a in unknown_argv:
        if a.startswith("--dl.") and "=" in a:
            k, v = a.split("=", 1)
            dl[k.split(".", 1)[1]] = _coerce_val(v)
    i = 0
    while i < len(unknown_argv):
        a = unknown_argv[i]
        if a.startswith("--dl.") and ("=" not in a):
            val = "true"
            if i + 1 < len(unknown_argv) and not unknown_argv[i + 1].startswith("--"):
                val = unknown_argv[i + 1]
                i += 1
            dl[a.split(".", 1)[1]] = _coerce_val(val)
        i += 1

    def _maybe_json(flags: List[str]) -> Dict[str, Any]:
        for flag in flags:
            if flag in unknown_argv:
                idx = unknown_argv.index(flag)
                if idx + 1 < len(unknown_argv):
                    val = unknown_argv[idx + 1]
                    try:
                        return _load_json_file(val) if isinstance(val, str) and os.path.isfile(val) else json.loads(val)
                    except Exception:
                        return {}
        return {}

    dl.update(_maybe_json(["--detector-loss-json", "--detr-loss-json"]))
    return dl


def _apply_detector_loss_debug_env(detector_loss_overrides: Dict[str, Any]) -> None:
    if "debug" in detector_loss_overrides:
        os.environ["DETR_LOSS_DEBUG"] = "1" if bool(detector_loss_overrides["debug"]) else "0"
    if "debug_every" in detector_loss_overrides:
        os.environ["DETR_LOSS_DEBUG_EVERY"] = str(int(detector_loss_overrides["debug_every"]))
    if "debug_verbosity" in detector_loss_overrides:
        os.environ["DETR_LOSS_DEBUG_VERBOSITY"] = str(int(detector_loss_overrides["debug_verbosity"]))
    if "debug_assert_finite" in detector_loss_overrides:
        os.environ["DETR_LOSS_DEBUG_ASSERT"] = "1" if bool(detector_loss_overrides["debug_assert_finite"]) else "0"


def parse_known_args_with_env(ap: argparse.ArgumentParser) -> Tuple[argparse.Namespace, List[str]]:
    args, unknown = ap.parse_known_args()
    env_set = os.environ.get("TRAIN_SET", "")
    if env_set:
        unknown = list(unknown) + sum([["--set", kv] for kv in env_set.split(";") if kv.strip()], [])
    return args, unknown


def parse_args() -> Tuple[argparse.Namespace, list]:
    ap = argparse.ArgumentParser("Hardware-aware detector trainer")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--mixed-precision", type=str, default=None, choices=["none", "fp16", "bf16"])
    ap.add_argument("--channels-last", type=int, default=None, help="1/0")
    ap.add_argument("--log-interval", type=int, default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile-backend", type=str, default=None)
    ap.add_argument("--compile-mode", type=str, default=None)
    ap.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["detector_only"],
        help="Only detector_only is supported.",
    )
    ap.add_argument("--cfg", type=str, default=None, help="Path to JSON config file")
    ap.add_argument("--cfg-json", type=str, default=None, help="JSON string or path (back-compat)")
    ap.add_argument("--set", action="append", default=[], help="--set a.b.c=val")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--init-dual", type=str, default=None)
    ap.add_argument("--init-detector", type=str, default=None)
    ap.add_argument("--init-cast", type=str, default=None, choices=["fp32", "bf16", "fp16", "none"])
    ap.add_argument("--init-strip", type=str, default=None)
    ap.add_argument("--init-prefer-ema", action="store_true")
    ap.add_argument("--debug", action="store_true")
    return parse_known_args_with_env(ap)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _force_math_sdp_if_needed(cfg: Cfg, log: logging.Logger) -> None:
    try:
        flag = bool(getattr(cfg.trainopt, "force_math_sdp", False))
        if not flag or not torch.cuda.is_available():
            return
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
        log.info("[runtime] force_math_sdp=True")
    except Exception as e:
        log.warning(f"[runtime] force_math_sdp failed: {e}")


_FAULT_DUMP_FH = None


def _enable_runtime_stack_dump(log: logging.Logger, output_dir: str) -> None:
    global _FAULT_DUMP_FH
    dump_file = os.path.join(output_dir, "faulthandler_stacks.log")
    try:
        os.makedirs(output_dir, exist_ok=True)
        _FAULT_DUMP_FH = open(dump_file, "a", encoding="utf-8")
    except Exception as e:
        log.warning(f"[debug] cannot open stack dump file: {e}")
        _FAULT_DUMP_FH = None

    try:
        if _FAULT_DUMP_FH is not None:
            faulthandler.enable(file=_FAULT_DUMP_FH, all_threads=True)
        else:
            faulthandler.enable(all_threads=True)
    except Exception as e:
        log.warning(f"[debug] faulthandler enable failed: {e}")
        return

    sig = getattr(signal, "SIGUSR1", None)
    if sig is None:
        return
    try:
        if _FAULT_DUMP_FH is not None:
            faulthandler.register(sig, file=_FAULT_DUMP_FH, all_threads=True, chain=False)
            log.info(f"[debug] faulthandler enabled (SIGUSR1 -> {dump_file})")
        else:
            faulthandler.register(sig, all_threads=True, chain=False)
            log.info("[debug] faulthandler enabled (send SIGUSR1 to dump Python stacks)")
    except Exception as e:
        log.warning(f"[debug] faulthandler register(SIGUSR1) failed: {e}")


def _attach_bridge_safe(
    detector: nn.Module,
    cfg: Cfg,
    device: torch.device,
    dtype: torch.dtype,
    second_pass: bool = False,
    log: Optional[logging.Logger] = None,
):
    _log = log or logging.getLogger("bridge")
    try:
        return attach_bridge_with_cfg(
            detector,
            cfg,
            device=device,
            dtype=dtype,
            rebuild_if_mismatch=bool(second_pass),
        )
    except TypeError:
        try:
            return attach_bridge_with_cfg(detector, cfg, device=device, dtype=dtype)
        except Exception as e:
            _log.warning(f"[bridge] attach failed: {e}")
            return None
    except Exception as e:
        _log.warning(f"[bridge] attach failed: {e}")
        return None


def _normalize_train_mode(mode: str) -> str:
    m = str(mode or "detector_only").lower()
    if m in ("detector_only",):
        return m
    return "detector_only"


def _apply_train_mode_freeze(cfg: Cfg, detector: nn.Module, log: logging.Logger) -> None:
    mode = _normalize_train_mode(getattr(cfg.trainopt, "train_mode", "detector_only"))
    setattr(cfg.trainopt, "train_mode", mode)

    if mode != "detector_only":
        log.warning(f"[mode] unknown train_mode={mode}, fallback to detector_only")
        setattr(cfg.trainopt, "train_mode", "detector_only")
    setattr(cfg.trainopt, "enable_detector", True)
    if bool(getattr(cfg.trainopt, "freeze_detector", False)):
        for p in detector.parameters():
            p.requires_grad = False
        log.info("[mode] detector_only: detector frozen")
    else:
        for p in detector.parameters():
            p.requires_grad = True
        log.info("[mode] detector_only: train detector")


def _auto_toggle_compile_by_mode(cfg: Cfg, log: logging.Logger, *, force_all: bool = False) -> None:
    mode = _normalize_train_mode(getattr(cfg.trainopt, "train_mode", "detector_only"))
    old_det = bool(getattr(cfg.trainopt, "compile_detector", False))
    new_det = old_det
    if force_all:
        if mode == "detector_only":
            new_det = True
    setattr(cfg.trainopt, "compile_detector", bool(new_det))
    if old_det != new_det:
        log.info(f"[compile] auto-toggle by mode: detector {old_det}->{new_det}")
    if new_det and not hasattr(torch, "compile"):
        log.warning("[compile] torch.compile not available; ignored.")


def _dump_model_params_if_enabled(cfg: Cfg, log: logging.Logger, detector: nn.Module) -> None:
    if not bool(getattr(cfg.trainopt, "dump_model_params", True)):
        return
    out_dir = os.path.join(cfg.output_dir, "model_params")
    os.makedirs(out_dir, exist_ok=True)
    rep_det = dump_model_param_report(detector, out_dir, name="detector")

    summary = {
        "detector": rep_det,
        "total_params_all": int(rep_det["params_all"]),
        "total_params_trainable": int(rep_det["params_trainable"]),
    }
    summary["total_params_frozen"] = int(summary["total_params_all"] - summary["total_params_trainable"])
    summary_path = os.path.join(out_dir, "model_params_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info(
        "[params] dumped model parameter reports: "
        f"detector={rep_det['params_all']:,} (trainable {rep_det['params_trainable']:,}), "
        f"total={summary['total_params_all']:,} -> {summary_path}"
    )


def _load_weights_with_alias(
    model: nn.Module,
    path: str,
    *,
    prefer_keys: List[str],
    map_location: str,
    strict: bool,
    shape_filter: bool,
    cast_dtype: Optional[str],
    strip_prefixes: Tuple[str, ...],
) -> bool:
    for k in prefer_keys:
        try:
            load_weights(
                model,
                path,
                prefer_key=k,
                map_location=map_location,
                strict=strict,
                shape_filter=shape_filter,
                cast_dtype=cast_dtype,
                strip_prefixes=strip_prefixes,
                log_mismatch=True,
            )
            return True
        except Exception:
            continue
    return False


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    set_seed(1337)
    device = _select_device()

    try:
        if device.type == "cuda":
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    args, unknown = parse_args()
    detector_loss_overrides = collect_loss_overrides(unknown)

    cfg = Cfg()
    cfg.set_meta_run_context()

    if args.cfg and os.path.isfile(args.cfg):
        try:
            cfg.apply_overrides(_load_json_file(args.cfg), source="json")
        except Exception as e:
            logging.getLogger("cfg").warning(f"[warn] --cfg load failed: {e}")
    if args.cfg_json:
        try:
            cfg.apply_overrides(
                _load_json_file(args.cfg_json) if os.path.isfile(args.cfg_json) else json.loads(args.cfg_json),
                source="json",
            )
        except Exception as e:
            logging.getLogger("cfg").warning(f"[warn] --cfg-json parse failed: {e}")

    dot_overrides: Dict[str, Any] = {}
    if args.epochs is not None:
        dot_overrides["epochs"] = int(args.epochs)
    if args.mixed_precision is not None:
        dot_overrides["trainopt.mixed_precision"] = args.mixed_precision
    if args.channels_last is not None:
        dot_overrides["trainopt.channels_last"] = bool(args.channels_last)
    if args.log_interval is not None:
        dot_overrides["log_interval"] = int(args.log_interval)
    if args.compile is True:
        dot_overrides["trainopt.compile_detector"] = True
    if args.compile_backend:
        dot_overrides["trainopt.compile_backend"] = args.compile_backend
    if args.compile_mode:
        dot_overrides["trainopt.compile_mode"] = args.compile_mode
    if getattr(args, "mode", None) is not None:
        dot_overrides["trainopt.train_mode"] = args.mode
    if args.resume:
        dot_overrides["ckpt.resume"] = args.resume
    if args.init_dual:
        dot_overrides["ckpt.init_dual"] = args.init_dual
    if args.init_detector:
        dot_overrides["ckpt.init_detector"] = args.init_detector
    if args.init_cast is not None:
        dot_overrides["ckpt.init_cast"] = args.init_cast
    if args.init_strip:
        dot_overrides["ckpt.init_strip"] = args.init_strip
    if args.init_prefer_ema:
        dot_overrides["ckpt.init_prefer_ema"] = True

    for s in (args.set or []):
        if "=" in s:
            k, v = s.split("=", 1)
            dot_overrides[k.strip()] = _coerce_val(v.strip())

    if dot_overrides:
        cfg.apply_overrides(dot_overrides, source="cli")

    cfg.setup_logging(cfg.output_dir)
    log = logging.getLogger("train")
    _enable_runtime_stack_dump(log, cfg.output_dir)

    step_logger = StepLogger(
        output_dir=cfg.output_dir,
        run_name=getattr(cfg, "run_name", "exp"),
        enable_tb=True,
        enable_jsonl=True,
        enable_console=False,
        metadata=cfg.meta.to_dict() if hasattr(cfg.meta, "to_dict") else {},
    )

    if getattr(cfg.trainopt, "deterministic", False):
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            log.info("[deterministic] enabled (cudnn.benchmark=False)")
        except Exception as e:
            log.warning(f"[deterministic] enable failed: {e}")
    else:
        torch.backends.cudnn.benchmark = True
        log.info("[cudnn] benchmark enabled for faster training")

    cfg.validate(strict=True)
    cfg.dump_all(cfg.output_dir)

    run_seed = getattr(cfg, "seed", 1337)
    set_seed(int(run_seed))

    try:
        log.info(
            f"device={device.type} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}) | "
            f"mp={cfg.trainopt.mixed_precision}"
        )
    except Exception:
        pass
    _force_math_sdp_if_needed(cfg, log)

    loaders = build_dataloaders(cfg)
    train_loader = loaders.get("train_loader", None) or loaders.get("train", None)
    if train_loader is None:
        raise RuntimeError("build_dataloaders() did not provide 'train_loader'")
    steps_per_epoch = max(1, len(train_loader))

    detector = build_detector(cfg.detector).to(device)
    _attach_bridge_safe(detector, cfg, device=device, dtype=torch.float32, second_pass=False, log=log)

    _unwrap_detector = detector.module if hasattr(detector, "module") else detector
    setattr(_unwrap_detector, "has_background", True)
    setattr(_unwrap_detector, "num_classes", getattr(_unwrap_detector, "num_classes", 5))

    global_step = 0
    start_epoch = 0
    if not getattr(cfg.ckpt, "ckpt_dir", None):
        cfg.ckpt.ckpt_dir = os.path.join(cfg.output_dir, "ckpt")
    strips = tuple([x.strip() for x in (cfg.ckpt.init_strip or "").split(",") if x and x.strip()]) or (
        "module.",
        "model.",
        "_orig_mod.",
    )
    cast_dtype = None if (cfg.ckpt.init_cast or "none").lower() == "none" else (cfg.ckpt.init_cast or "none").lower()
    resume_path = _resolve_resume_path((cfg.ckpt.resume or "").strip(), default_dir=getattr(cfg.ckpt, "ckpt_dir", None), logger=log)
    extra: Dict[str, Any] = {}

    if resume_path:
        step_loaded, extra = load_dual_ckpt(
            resume_path,
            detector=detector,
            map_location="cpu",
            strict=False,
            shape_filter=True,
            cast_dtype=None,
            strip_prefixes=strips,
            log_mismatch=True,
        )
        global_step = max(0, int(step_loaded))
        ep_loaded = None
        if isinstance(extra, dict):
            ep_loaded = extra.get("epoch", None)
            if not isinstance(ep_loaded, (int, float)):
                nested_extra = extra.get("extra", None)
                if isinstance(nested_extra, dict):
                    ep_loaded = nested_extra.get("epoch", None)
            if isinstance(ep_loaded, (int, float)):
                start_epoch = max(0, int(ep_loaded) + 1)
        log.info(f"[resume] {resume_path} (step={global_step}, start_epoch={start_epoch})")
    else:
        if cfg.ckpt.init_dual and os.path.isfile(cfg.ckpt.init_dual):
            _step, _ = load_dual_ckpt(
                cfg.ckpt.init_dual,
                detector=detector,
                map_location="cpu",
                strict=False,
                shape_filter=True,
                cast_dtype=cast_dtype,
                strip_prefixes=strips,
                log_mismatch=True,
            )
            log.info(f"[init] dual ckpt loaded: {cfg.ckpt.init_dual}")

        if cfg.ckpt.init_detector and os.path.isfile(cfg.ckpt.init_detector):
            ok_det = _load_weights_with_alias(
                detector,
                cfg.ckpt.init_detector,
                prefer_keys=["detector", "detr", "model", "state_dict"],
                map_location="cpu",
                strict=False,
                shape_filter=True,
                cast_dtype=cast_dtype,
                strip_prefixes=strips,
            )
            if ok_det:
                log.info("[init] detector loaded")

    _attach_bridge_safe(detector, cfg, device=device, dtype=torch.float32, second_pass=True, log=log)
    _apply_train_mode_freeze(cfg, detector, log)

    if device.type in ("cuda", "mps") and getattr(cfg.trainopt, "channels_last", False):
        maybe_channels_last(detector, True, device)
        log.info("[runtime] channels_last enabled")

    if getattr(cfg.trainopt, "freeze_detector", False):
        freeze_module(detector, True)

    try:
        _dump_model_params_if_enabled(cfg, log, detector)
    except Exception as e:
        log.warning(f"[params] dump failed: {e}")

    _auto_toggle_compile_by_mode(cfg, log, force_all=bool(getattr(args, "compile", False)))
    if getattr(cfg.trainopt, "compile_detector", False) and hasattr(torch, "compile"):
        detector = maybe_compile(
            detector,
            True,
            backend=getattr(cfg.trainopt, "compile_backend", "inductor"),
            mode=getattr(cfg.trainopt, "compile_mode", "default"),
        )

    _apply_detector_loss_debug_env(detector_loss_overrides)
    loss_detector = build_detector_loss(
        model_cfg=cfg.detector,
        loss_cfg=cfg.loss_detector,
        **detector_loss_overrides,
    )

    mp_str = (getattr(cfg.trainopt, "mixed_precision", "none") or "none").lower()
    use_fp16_scaler = (mp_str in ("fp16", "half", "float16")) and (device.type == "cuda")
    scal_detector = torch_amp.GradScaler(enabled=use_fp16_scaler) if use_fp16_scaler else None

    ema_detector = (
        EMA(detector, decay=cfg.ema.decay, device=cfg.ema.device or "cpu", include_buffers=cfg.ema.include_buffers)
        if cfg.ema.enable_detector
        else None
    )
    try:
        detector._ema = ema_detector
    except Exception:
        pass

    lr_det = float(getattr(cfg.optim, "lr_detector", getattr(cfg.optim, "lr", 2e-4)))
    opt_detector = build_optimizer_for(detector, cfg.optim, lr_override=lr_det, role="detector")
    accum = max(1, int(getattr(cfg.trainopt, "grad_accum_steps", 1)))
    steps_per_epoch_accum = max(1, int(math.ceil(steps_per_epoch / float(accum))))
    _, sched_detector = build_schedulers(
        cfg,
        opt_u=None,
        opt_detector=opt_detector,
        steps_per_epoch=steps_per_epoch_accum,
    )

    if resume_path:
        try:
            if isinstance(extra, dict):
                scal_det_state = (
                    extra.get("scaler_state_detector")
                    or (extra.get("scaler", {}) or {}).get("detector")
                    or extra.get("scaler_detector")
                    or (extra.get("scaler", {}) or {}).get("detr")
                    or extra.get("scaler_detr")
                )
                if use_fp16_scaler:
                    if scal_det_state and scal_detector:
                        try:
                            scal_detector.load_state_dict(scal_det_state)
                        except Exception:
                            pass

                ema_det_state = (
                    extra.get("ema_state_detector")
                    or (extra.get("ema", {}) or {}).get("detector")
                    or extra.get("ema_detector")
                    or (extra.get("ema", {}) or {}).get("detr")
                    or extra.get("ema_detr")
                )
                if ema_det_state and ema_detector:
                    try:
                        ema_detector.load_state_dict(ema_det_state)
                    except Exception:
                        pass

                opt_det_state = (
                    extra.get("opt_state_detector")
                    or (extra.get("opt", {}) or {}).get("detector")
                    or extra.get("optimizer_detector")
                    or (extra.get("opt", {}) or {}).get("detr")
                    or extra.get("optimizer_detr")
                )
                if opt_det_state and opt_detector:
                    try:
                        opt_detector.load_state_dict(opt_det_state)
                    except Exception:
                        pass

                sched_det_state = (
                    extra.get("sched_state_detector")
                    or (extra.get("sched", {}) or {}).get("detector")
                    or extra.get("scheduler_detector")
                    or (extra.get("sched", {}) or {}).get("detr")
                    or extra.get("scheduler_detr")
                )
                if sched_det_state and sched_detector:
                    try:
                        sched_detector.load_state_dict(sched_det_state)
                    except Exception:
                        pass
        except Exception as e:
            log.warning(f"[resume] state restore failed: {e}")

    ckpt_dir = cfg.ckpt.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    step_ckpt_dir = os.path.join(ckpt_dir, "steps")
    os.makedirs(step_ckpt_dir, exist_ok=True)

    eval_every = max(1, int(getattr(cfg.eval, "interval_epochs", 1)))
    _best_metric_raw = str(getattr(cfg.eval, "save_best_metric", "mAP50"))
    best_metric_name = _best_metric_raw.replace("det/", "") if _best_metric_raw.startswith("det/") else _best_metric_raw
    best_metric_ckpt_name = best_metric_name.replace("/", "_")
    best_mode = str(getattr(cfg.eval, "best_metric_mode", "max")).lower()
    min_delta = float(getattr(cfg.eval, "early_stop_min_delta", 0.0))
    patience = getattr(cfg.eval, "early_stop_patience", None)
    topk = int(getattr(cfg.eval, "save_best_topk", 1))

    def _is_better(new: float, best: float) -> bool:
        if math.isnan(new) or math.isinf(new):
            return False
        return (new > best + min_delta) if best_mode == "max" else (new < best - min_delta)

    def _select_metric(log_dict: Dict[str, Any]) -> float:
        if "det/mAP50" in log_dict and "mAP50" not in log_dict:
            log_dict["mAP50"] = log_dict["det/mAP50"]
        if "det/mAP" in log_dict and "mAP" not in log_dict:
            log_dict["mAP"] = log_dict["det/mAP"]
        if best_metric_name in log_dict and isinstance(log_dict[best_metric_name], (int, float)):
            return float(log_dict[best_metric_name])
        for key in ["loss_total", "mAP50", "mAP"]:
            if key in log_dict and isinstance(log_dict[key], (int, float)):
                return float(log_dict[key])
        for v in log_dict.values():
            if isinstance(v, (int, float)):
                return float(v)
        return float("inf") if best_mode == "min" else -float("inf")

    def _load_ckpt_dict(path: str) -> Optional[Dict[str, Any]]:
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[arg-type]
        except TypeError:
            blob = torch.load(path, map_location="cpu")
        except Exception:
            try:
                blob = torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
            except TypeError:
                blob = torch.load(path, map_location="cpu")
            except Exception:
                return None
        return blob if isinstance(blob, dict) else None

    def _extract_val_metric_from_ckpt(path: str) -> Optional[float]:
        if not path or (not os.path.isfile(path)):
            return None
        blob = _load_ckpt_dict(path)
        if not isinstance(blob, dict):
            return None
        ex = blob.get("extra", None)
        if not isinstance(ex, dict):
            return None
        val_log = ex.get("val", None)
        if not isinstance(val_log, dict):
            return None
        score = _select_metric(dict(val_log))
        if not math.isfinite(score):
            return None
        return float(score)

    best_score = float("-inf") if best_mode == "max" else float("inf")
    bad_eval_count = 0
    topk_list: List[Tuple[float, str]] = []

    if resume_path:
        restored_score = None
        restored_from = None

        best_ckpt_path = os.path.join(ckpt_dir, f"best_{best_metric_ckpt_name}.pt")
        s = _extract_val_metric_from_ckpt(best_ckpt_path)
        if s is not None:
            restored_score = s
            restored_from = best_ckpt_path

        if restored_score is None:
            for ep_path in sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt"))):
                s_ep = _extract_val_metric_from_ckpt(ep_path)
                if s_ep is None:
                    continue
                if (restored_score is None) or _is_better(s_ep, restored_score):
                    restored_score = s_ep
                    restored_from = ep_path

        if restored_score is None and isinstance(extra, dict):
            cand_val = extra.get("val", None)
            if not isinstance(cand_val, dict):
                nested = extra.get("extra", None)
                if isinstance(nested, dict):
                    cand_val = nested.get("val", None)
            if isinstance(cand_val, dict):
                s_cur = _select_metric(dict(cand_val))
                if math.isfinite(s_cur):
                    restored_score = float(s_cur)
                    restored_from = "resume_payload"

        if restored_score is not None:
            best_score = float(restored_score)
            log.info(f"[resume] restore best {best_metric_name}={best_score:.6g} from {restored_from}")

    keep_last = max(1, int(getattr(cfg.ckpt, "keep_last", 5)))
    step_interval = max(1, int(getattr(cfg.ckpt, "interval_steps", 2000)))
    keep_last_steps = int(getattr(cfg.ckpt, "keep_last_steps", 8))

    try:
        for epoch in range(start_epoch, int(cfg.epochs)):
            global_step, train_log = train_one_epoch(
                epoch,
                cfg,
                device,
                detector=detector,
                loss_detector=loss_detector,
                opt_detector=opt_detector,
                scal_detector=scal_detector,
                sched_detector=sched_detector,
                loaders=loaders,
                ema_detector=ema_detector,
                global_step=global_step,
                step_ckpt_config=dict(
                    every_n_steps=step_interval,
                    out_dir=step_ckpt_dir,
                    tag_prefix="g",
                    keep_last_k=keep_last_steps,
                    save_optim=True,
                    save_sched=True,
                    save_scaler=True,
                    save_ema=True,
                ),
                logger=step_logger,
            )

            run_val = epoch % eval_every == 0
            if run_val:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                val_log = validate(
                    epoch,
                    cfg,
                    device,
                    detector=detector,
                    loss_detector=loss_detector,
                    loaders=loaders,
                    use_ema=True,
                    logger=step_logger,
                    global_step=global_step,
                )
            else:
                val_log = {"skipped": True}

            t_total = train_log.get("loss_total", float("nan"))
            det_eval_enabled = bool(getattr(cfg.trainopt, "enable_detector", True)) and (detector is not None)
            if det_eval_enabled:
                if run_val:
                    v_mAP50 = val_log.get("det/mAP50", val_log.get("mAP50", float("nan")))
                    v_mAP = val_log.get("det/mAP", val_log.get("mAP", float("nan")))
                    logging.getLogger("eval").info(
                        f"[epoch {epoch:03d}] train loss_total={t_total:.4f} | val mAP50={v_mAP50:.4f} mAP={v_mAP:.4f}"
                    )
                else:
                    logging.getLogger("eval").info(
                        f"[epoch {epoch:03d}] train loss_total={t_total:.4f} | val skipped (interval={eval_every})"
                    )
            else:
                if run_val:
                    v_loss_total = val_log.get("loss_total", float("nan"))
                    logging.getLogger("eval").info(
                        f"[epoch {epoch:03d}] train loss_total={t_total:.4f} | val loss_total={v_loss_total:.4f} (detector disabled)"
                    )
                else:
                    logging.getLogger("eval").info(
                        f"[epoch {epoch:03d}] train loss_total={t_total:.4f} | val skipped (interval={eval_every})"
                    )

            ep_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            save_dual_ckpt(
                ep_path,
                step=global_step,
                detector=detector,
                opt={"detector": _safe_state(opt_detector)},
                sched={"detector": _safe_state(sched_detector)},
                scaler={"detector": (scal_detector.state_dict() if scal_detector else None)},
                ema={"detector": (ema_detector.state_dict() if ema_detector else None)},
                extra={
                    "epoch": epoch,
                    "train": train_log,
                    "val": val_log,
                    "cfg_trace": {
                        "argv": cfg.meta.argv,
                        "cli_overrides": cfg.meta.cli_overrides,
                        "cfg_json_overrides": cfg.meta.cfg_json_overrides,
                    },
                },
            )

            try:
                ep_files = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
                if len(ep_files) > keep_last:
                    for v in ep_files[: max(0, len(ep_files) - keep_last)]:
                        try:
                            os.remove(v)
                        except Exception:
                            pass
            except Exception:
                pass

            if run_val:
                score = _select_metric(val_log)
                if _is_better(score, best_score):
                    best_score = score
                    bad_eval_count = 0
                    best_path = os.path.join(ckpt_dir, f"best_{best_metric_ckpt_name}.pt")
                    try:
                        shutil.copy2(ep_path, best_path)
                        logging.getLogger("best").info(f"[best] {best_metric_name}={best_score:.6g} @ epoch {epoch}")
                    except Exception:
                        pass

                    topk_list.append((score, ep_path))
                    topk_list.sort(reverse=(best_mode == "max"), key=lambda x: x[0])
                    while len(topk_list) > max(1, topk):
                        topk_list.pop()
                else:
                    bad_eval_count += 1
                    if patience is not None and bad_eval_count >= patience:
                        logging.getLogger("best").info(
                            f"[early-stop] no improvement on {best_metric_name} for {bad_eval_count} evals (>= {patience})."
                        )
                        break

        log.info("[done] training finished.")

    except torch.cuda.OutOfMemoryError:
        log.error(
            "CUDA OOM: try 1) --set data.batch_size=1; 2) trainopt.mixed_precision='bf16'; 3) disable compile/channels_last."
        )
        raise
    except Exception:
        log.exception("Fatal error during training")
        raise
    finally:
        try:
            if step_logger:
                step_logger.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger("signal").info("[signal] interrupted by user.")
        sys.stderr.write("\n[signal] interrupted by user.\n")
        os._exit(130)
