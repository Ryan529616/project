# -*- coding: utf-8 -*-
"""
step.py

Training/validation step core for detector-only pipeline.
Canonical names:
- detector: hardware-aware object detector
"""

from __future__ import annotations

import contextlib
import glob
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.amp as torch_amp
import torch.nn as nn

try:
    import torch._functorch.config as _fc

    _fc.donated_buffer = False
except Exception:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.metrics.detection_metrics import CocoAveragePrecisionMeter
from engine.metrics.detection_metrics import compute_detection_batch_metrics as det_batch_metrics
from engine.metrics.hardware_aware_metrics import compute_hardware_aware_batch_metrics

try:
    from utils.trainlog import StepLogger
except Exception:
    StepLogger = None  # type: ignore

try:
    from utils.checkpoint import save_dual_ckpt
except Exception:
    save_dual_ckpt = None  # type: ignore


def unwrap(m: nn.Module) -> nn.Module:
    seen_ids = set()
    cur = m
    while True:
        cur_id = id(cur)
        if cur_id in seen_ids:
            break
        seen_ids.add(cur_id)
        if hasattr(cur, "module"):
            cur = getattr(cur, "module")
            continue
        if hasattr(cur, "_orig_mod"):
            cur = getattr(cur, "_orig_mod")
            continue
        break
    return cur


def to_device(x: Any, device: torch.device) -> Any:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(xx, device) for xx in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def get_dtype_tag(mp: Optional[str]) -> Optional[torch.dtype]:
    if not mp:
        return None
    mp = str(mp).lower()
    if mp in ("none", "fp32", "float32"):
        return None
    if mp in ("fp16", "float16", "half"):
        return torch.float16
    if mp in ("bf16", "bfloat16"):
        return torch.bfloat16
    return None


def _resolve_amp_dtype(device: torch.device, cfg: Any) -> Tuple[bool, Optional[torch.dtype]]:
    if device.type not in ("cuda", "mps"):
        return False, None
    trainopt = getattr(cfg, "trainopt", cfg)
    mp_str = getattr(trainopt, "mixed_precision", getattr(trainopt, "mp", getattr(cfg, "mixed_precision", None)))
    if mp_str is None:
        return True, torch.float16
    mp = str(mp_str).lower()
    if mp in ("off", "none", "fp32", "float32"):
        return False, None
    dt = get_dtype_tag(mp)
    if dt is None:
        return False, None
    return True, dt


def _safe(x: torch.Tensor, *, nan: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=nan, posinf=0.0, neginf=0.0)


@contextlib.contextmanager
def autocast_if_available(device: torch.device, dtype: Optional[torch.dtype], enabled: bool):
    if enabled and device.type in ("cuda", "mps") and dtype is not None:
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            yield
    else:
        yield


def _scaler_active(scaler: Optional[torch_amp.GradScaler]) -> bool:
    if scaler is None:
        return False
    try:
        return bool(scaler.is_enabled())
    except Exception:
        return True


def _ctx_str(ctx: Optional[Dict[str, Any]]) -> str:
    if not ctx:
        return ""
    parts = [f"{k}={ctx[k]}" for k in ("phase", "epoch", "iter", "global_step") if k in ctx]
    return " | " + " ".join(parts) if parts else ""


def _fail(msg: str, *, print_trace: bool = True):
    if print_trace:
        import traceback

        traceback.print_stack(limit=32)
    raise SystemExit(msg)


def _assert_finite_tensor(
    t: torch.Tensor,
    where: str,
    name: str,
    ctx: Optional[Dict[str, Any]],
    fail_fast: bool = True,
    skip: bool = False,
):
    if skip or t is None or not torch.is_tensor(t):
        return
    if not torch.isfinite(t).all():
        msg = f"[FAIL] Non-finite tensor at {where}:{name}{_ctx_str(ctx)}"
        if fail_fast:
            _fail(msg)
        else:
            raise RuntimeError(msg)


def _set_bn_eval_only(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()


def _safe_state(obj: Any):
    try:
        return obj.state_dict() if obj is not None else None
    except Exception:
        return None


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _maybe_save_step_ckpt(
    *,
    step_ckpt_config: Optional[Dict[str, Any]],
    cfg: Any,
    epoch: int,
    it: int,
    global_step: int,
    detector: nn.Module,
    opt_detector: Optional[torch.optim.Optimizer],
    sched_detector: Optional[Any],
    scal_detector: Optional[torch_amp.GradScaler],
    ema_detector: Optional[Any],
) -> None:
    if not step_ckpt_config or save_dual_ckpt is None:
        return

    every_n = int(step_ckpt_config.get("every_n_steps", 0) or 0)
    if every_n <= 0 or global_step <= 0 or (global_step % every_n) != 0:
        return

    out_dir = str(step_ckpt_config.get("out_dir", "") or "")
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)

    tag_prefix = str(step_ckpt_config.get("tag_prefix", "g") or "g")
    keep_last_k = int(step_ckpt_config.get("keep_last_k", 0) or 0)
    save_optim = bool(step_ckpt_config.get("save_optim", True))
    save_sched = bool(step_ckpt_config.get("save_sched", True))
    save_scaler = bool(step_ckpt_config.get("save_scaler", True))
    save_ema = bool(step_ckpt_config.get("save_ema", True))

    ckpt_name = f"{tag_prefix}step{global_step:07d}.pt"
    ckpt_path = os.path.join(out_dir, ckpt_name)

    save_dual_ckpt(
        ckpt_path,
        step=global_step,
        detector=detector,
        opt={"detector": _safe_state(opt_detector)} if save_optim else None,
        sched={"detector": _safe_state(sched_detector)} if save_sched else None,
        scaler={"detector": (scal_detector.state_dict() if scal_detector else None)} if save_scaler else None,
        ema={"detector": (ema_detector.state_dict() if ema_detector else None)} if save_ema else None,
        extra={"epoch": int(epoch), "iter": int(it), "kind": "step"},
    )

    run_dir = getattr(cfg, "output_dir", None)
    if isinstance(run_dir, str) and run_dir:
        idx_path = os.path.join(run_dir, "logs", "ckpt_index.jsonl")
        _append_jsonl(
            idx_path,
            {
                "ts": time.time(),
                "event": "step_ckpt",
                "step": int(global_step),
                "epoch": int(epoch),
                "iter": int(it),
                "path": ckpt_path,
            },
        )

    if keep_last_k > 0:
        pat = os.path.join(out_dir, f"{tag_prefix}step*.pt")
        files = sorted(glob.glob(pat))
        if len(files) > keep_last_k:
            for p in files[: len(files) - keep_last_k]:
                try:
                    os.remove(p)
                except Exception:
                    pass


def safe_sum_losses(loss_dict: Dict[str, torch.Tensor], skip_safe: bool = False) -> Tuple[torch.Tensor, List[str]]:
    total = None
    bad: List[str] = []
    for k, v in loss_dict.items():
        if "loss" not in k:
            continue
        if not torch.is_tensor(v):
            v = torch.tensor(float(v))
        if not skip_safe:
            if not torch.isfinite(v):
                bad.append(k)
            v_clean = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            v_clean = v
        total = v_clean if total is None else (total + v_clean)
    if total is None:
        device = next((v.device for v in loss_dict.values() if torch.is_tensor(v)), None)
        total = torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
    return total, bad


def aggregate_losses_with_guard(
    loss_dict: Dict[str, torch.Tensor], *, policy: str = "fail", skip_safe: bool = False
) -> Tuple[torch.Tensor, List[str], bool]:
    total, bad = safe_sum_losses(loss_dict, skip_safe=skip_safe)
    if not bad:
        return total, bad, False
    if policy == "fail":
        _fail(f"[loss-guard] non-finite losses: {bad}")
    if policy == "zero":
        return total, bad, False
    return total.detach() * 0.0, bad, True


def forward_detector(
    detector: nn.Module,
    raw4: torch.Tensor,
    cfg: Any,
    *,
    amp_enabled: bool,
    amp_dtype_: Optional[torch.dtype],
    ctx: Optional[Dict[str, Any]] = None,
    skip_safe: bool = False,
) -> Dict[str, torch.Tensor]:
    device = raw4.device
    trainopt = getattr(cfg, "trainopt", cfg)
    fail_fast = bool(getattr(trainopt, "fail_fast", True))

    det_dtype = amp_dtype_ if (amp_enabled and amp_dtype_ is not None) else None
    det_autocast = bool(amp_enabled and (det_dtype is not None) and (device.type in ("cuda", "mps")))

    try:
        with torch.no_grad():
            detector.ensure_bridge(raw4, None)
    except Exception:
        pass

    with autocast_if_available(device, det_dtype, det_autocast and amp_enabled):
        outputs = detector(raw4)
        if isinstance(outputs, dict):
            if "pred_logits" in outputs:
                _assert_finite_tensor(outputs["pred_logits"], "forward_detector", "pred_logits", ctx, fail_fast=fail_fast, skip=skip_safe)
            if "pred_boxes" in outputs:
                _assert_finite_tensor(outputs["pred_boxes"], "forward_detector", "pred_boxes", ctx, fail_fast=fail_fast, skip=skip_safe)
            if "pred_boxes" in outputs and bool(getattr(trainopt, "clamp_boxes01", False)):
                outputs["pred_boxes"] = outputs["pred_boxes"].clamp(0.0, 1.0)
        else:
            _fail(f"[FAIL] detector returned non-dict outputs{_ctx_str(ctx)}")

    return outputs


def _build_empty_targets(B: int, device: torch.device) -> List[Dict[str, torch.Tensor]]:
    return [
        {
            "labels": torch.zeros(0, dtype=torch.long, device=device),
            "boxes": torch.zeros(0, 4, device=device),
            "size": torch.tensor([0.0, 0.0], device=device),
        }
        for _ in range(B)
    ]


def _build_detector_targets_from_batch(
    batch: Dict[str, Any], B: int, device: torch.device
) -> List[Dict[str, torch.Tensor]]:
    H = batch.get("H", torch.zeros(B, device=device, dtype=torch.float32)).to(device=device, dtype=torch.float32)
    W = batch.get("W", torch.zeros(B, device=device, dtype=torch.float32)).to(device=device, dtype=torch.float32)

    targets_in = batch.get("targets", None)
    if isinstance(targets_in, list) and len(targets_in) == B:
        targets: List[Dict[str, torch.Tensor]] = []
        for i in range(B):
            t = targets_in[i]
            Hin, Win = batch.get("Hin", H), batch.get("Win", W)
            if t is None:
                targets.append(
                    {
                        "labels": torch.zeros(0, device=device, dtype=torch.long),
                        "boxes": torch.zeros(0, 4, device=device, dtype=torch.float32),
                        "size": torch.stack([Hin[i], Win[i]], dim=0),
                    }
                )
                continue
            targets.append(
                {
                    "labels": t.get("labels", torch.zeros(0, device=device)).to(device).long(),
                    "boxes": t.get("boxes", torch.zeros(0, 4, device=device)).to(device).float(),
                    "size": torch.stack([Hin[i], Win[i]], dim=0),
                }
            )
        return targets

    targets = _build_empty_targets(B, device)
    for i in range(B):
        targets[i]["size"] = torch.stack([batch.get("Hin", H)[i], batch.get("Win", W)[i]], dim=0)
    return targets


def train_micro_step(
    *,
    detector,
    opt_detector,
    loss_detector,
    scaler_detector,
    batch,
    cfg,
    device,
    amp_dtype_=None,
    ctx=None,
    compute_metrics=False,
) -> Dict[str, float]:
    trainopt = getattr(cfg, "trainopt", cfg)
    enable_detector = bool(getattr(trainopt, "enable_detector", True))
    freeze_detector = bool(getattr(trainopt, "freeze_detector", False))
    accum = max(1, int(getattr(trainopt, "grad_accum_steps", 1)))
    fail_fast = bool(getattr(trainopt, "fail_fast", True))

    skip_safe = bool(getattr(trainopt, "skip_finite_checks", True))

    mp_enabled, mp_dtype = _resolve_amp_dtype(device, cfg) if amp_dtype_ is None else (device.type in ("cuda", "mps"), amp_dtype_)

    if not skip_safe:
        if "raw4" in batch and not torch.isfinite(batch["raw4"]).all():
            print("[FAIL] Input batch['raw4'] contains NaN/Inf")
            return {"loss_total": 0.0, "skipped": 1.0}

    if "raw4" not in batch:
        raise KeyError("batch must contain 'raw4'.")

    raw4 = batch["raw4"].float()

    stats: Dict[str, float] = {
        "loss_detector": 0.0,
        "loss_total": 0.0,
    }
    if (not enable_detector) or (detector is None) or (loss_detector is None):
        return stats

    wants_backward_detector = (not freeze_detector) and (opt_detector is not None)
    mp_detector_mode = str(getattr(trainopt, "mixed_precision_detector", "bf16")).lower()
    if mp_detector_mode in ("fp32", "float32", "none"):
        detector_amp_enabled, detector_amp_dtype = False, None
    elif mp_detector_mode in ("fp16", "float16", "half"):
        detector_amp_enabled, detector_amp_dtype = (device.type == "cuda"), torch.float16
    elif mp_detector_mode in ("bf16", "bfloat16"):
        detector_amp_enabled, detector_amp_dtype = (device.type == "cuda"), torch.bfloat16
    else:
        detector_amp_enabled, detector_amp_dtype = mp_enabled, mp_dtype

    with torch.enable_grad() if wants_backward_detector else torch.no_grad():
        dout = forward_detector(
            detector,
            raw4,
            cfg,
            amp_enabled=detector_amp_enabled,
            amp_dtype_=detector_amp_dtype,
            ctx=ctx,
            skip_safe=skip_safe,
        )

    pred_logits, pred_boxes = dout["pred_logits"], dout["pred_boxes"]
    targets = _build_detector_targets_from_batch(batch, int(pred_logits.shape[0]), device)
    with torch.no_grad():
        C_used = getattr(unwrap(detector), "num_classes", None) or (
            int(pred_logits.shape[2]) - 1 if int(pred_logits.shape[2]) > 1 else int(pred_logits.shape[2])
        )
        for t in targets:
            if "labels" in t and torch.is_tensor(t["labels"]) and t["labels"].numel() > 0:
                t["labels"] = t["labels"].to(pred_logits.device).clamp_(0, max(0, C_used - 1))

    loss_inputs = dict(dout)
    loss_inputs["pred_logits"] = pred_logits.float()
    loss_inputs["pred_boxes"] = pred_boxes.float()

    with torch.autocast(device_type=device.type, enabled=False):
        ldict_detector = loss_detector(loss_inputs, targets)

    total_detector, _, should_skip = aggregate_losses_with_guard(
        ldict_detector if isinstance(ldict_detector, dict) else {"loss_total": _safe(ldict_detector)},
        policy=str(getattr(trainopt, "loss_guard_policy", "fail")).lower(),
        skip_safe=skip_safe,
    )

    if should_skip:
        if opt_detector:
            opt_detector.zero_grad(set_to_none=True)
        stats["loss_total"] = 0.0
        return stats

    _assert_finite_tensor(total_detector, "loss_detector", "total", ctx, fail_fast=fail_fast, skip=skip_safe)
    loss_detector_val = total_detector / accum

    if wants_backward_detector:
        if mp_enabled and _scaler_active(scaler_detector):
            scaler_detector.scale(loss_detector_val).backward()
        else:
            loss_detector_val.backward()

    stats["loss_detector"] = float(total_detector.detach().item())
    stats["loss_total"] = float(stats["loss_detector"])

    if compute_metrics:
        det_model = unwrap(detector)
        mcfg = getattr(cfg, "metrics_detector", None)
        score_thr = float(getattr(mcfg, "score_thr", 0.05))
        max_dets = int(getattr(mcfg, "max_dets", 300))
        nms_iou = getattr(mcfg, "nms_iou", 0.5)
        if nms_iou is not None:
            nms_iou = float(nms_iou)
        try:
            detm = det_batch_metrics(
                {"pred_logits": pred_logits.detach(), "pred_boxes": pred_boxes.detach()},
                targets,
                score_thr=score_thr,
                max_dets=max_dets,
                num_classes=int(getattr(det_model, "num_classes", pred_logits.shape[-1] - 1)),
                has_background=bool(getattr(det_model, "has_bg", True)),
                nms_iou=nms_iou,
                iou_thr=0.5,
            )
            for key in (
                "precision@0.5",
                "recall@0.5",
                "f1@0.5",
                "mean_iou@0.5",
                "tp@0.5",
                "fp@0.5",
                "fn@0.5",
            ):
                if key in detm:
                    stats[f"det/{key}"] = float(detm[key])
            for key in ("precision", "recall", "f1", "mean_iou"):
                if key in detm:
                    v = detm[key]
                    if torch.is_tensor(v):
                        v = float(v.detach().item())
                    stats[f"det/{key}"] = float(v)
        except Exception:
            pass

        try:
            hw = compute_hardware_aware_batch_metrics(
                {
                    "pred_logits": pred_logits.detach(),
                    "object_tokens": dout.get("object_tokens", None).detach()
                    if isinstance(dout.get("object_tokens", None), torch.Tensor)
                    else dout.get("object_tokens", None),
                    "tokens_per_level": [
                        t.detach() if isinstance(t, torch.Tensor) else t for t in (dout.get("tokens_per_level", []) or [])
                    ],
                }
            )
            for k, v in hw.items():
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    stats[k] = float(v)
        except Exception:
            pass

    return stats


def step_optim(
    optim: Optional[torch.optim.Optimizer],
    scaler: Optional[torch_amp.GradScaler],
    *,
    mixed_precision: bool,
    model: Optional[nn.Module],
    clip_norm: Optional[float] = None,
    zero_grad: bool = True,
):
    if optim is None:
        return
    if model is not None:
        params_with_grad = [p for p in model.parameters() if getattr(p, "grad", None) is not None]
    else:
        params_with_grad = [p for g in optim.param_groups for p in g.get("params", []) if getattr(p, "grad", None) is not None]

    if not params_with_grad:
        if zero_grad:
            try:
                optim.zero_grad(set_to_none=True)
            except Exception:
                pass
        return

    scaler_ok = mixed_precision and _scaler_active(scaler)
    if scaler_ok:
        try:
            scaler.unscale_(optim)
        except Exception:
            pass

    if clip_norm is not None:
        try:
            torch.nn.utils.clip_grad_norm_(params_with_grad, float(clip_norm))
        except Exception:
            pass

    if scaler_ok:
        try:
            scaler.step(optim)
            try:
                scaler.update()
            except AssertionError:
                pass
        except Exception:
            optim.step()
    else:
        optim.step()

    if zero_grad:
        try:
            optim.zero_grad(set_to_none=True)
        except Exception:
            pass


@torch.no_grad()
def _grad_stats_from_module(model: Optional[nn.Module]) -> Dict[str, float]:
    if model is None:
        return {"l2": 0.0, "mean_abs": 0.0, "tensor_count": 0.0, "elem_count": 0.0}

    sq_sum_t: Optional[torch.Tensor] = None
    abs_sum_t: Optional[torch.Tensor] = None
    elem_count = 0
    tensor_count = 0
    for p in model.parameters():
        g = getattr(p, "grad", None)
        if g is None:
            continue
        gg = g.detach().float()
        cur_sq = torch.sum(gg * gg)
        cur_abs = torch.sum(torch.abs(gg))
        sq_sum_t = cur_sq if sq_sum_t is None else (sq_sum_t + cur_sq)
        abs_sum_t = cur_abs if abs_sum_t is None else (abs_sum_t + cur_abs)
        elem_count += int(gg.numel())
        tensor_count += 1

    if elem_count <= 0 or sq_sum_t is None or abs_sum_t is None:
        return {"l2": 0.0, "mean_abs": 0.0, "tensor_count": float(tensor_count), "elem_count": 0.0}
    sq_sum = float(sq_sum_t.item())
    abs_sum = float(abs_sum_t.item())
    l2 = math.sqrt(max(0.0, sq_sum))
    mean_abs = abs_sum / max(1, elem_count)
    return {
        "l2": float(l2),
        "mean_abs": float(mean_abs),
        "tensor_count": float(tensor_count),
        "elem_count": float(elem_count),
    }


def _apply_epoch_train_mode(cfg: Any, detector: nn.Module):
    trainopt = getattr(cfg, "trainopt", cfg)
    mode = str(getattr(trainopt, "train_mode", "detector_only") or "detector_only").lower()
    freeze_det = bool(getattr(trainopt, "freeze_detector", False))
    enable_det = bool(getattr(trainopt, "enable_detector", True))

    if mode != "detector_only":
        logging.getLogger("mode").warning(f"[mode] unknown train_mode={mode}; forcing detector_only.")
        setattr(trainopt, "train_mode", "detector_only")
    if enable_det and not freeze_det:
        detector.train()
    else:
        detector.eval()


def train_one_epoch(
    epoch: int,
    cfg: Any,
    device: torch.device,
    *,
    detector,
    loss_detector,
    opt_detector,
    scal_detector,
    sched_detector,
    loaders,
    ema_detector,
    global_step,
    step_ckpt_config=None,
    logger=None,
) -> Tuple[int, Dict[str, float]]:
    trainopt = getattr(cfg, "trainopt", cfg)
    accum = max(1, int(getattr(trainopt, "grad_accum_steps", 1)))
    mp_enabled, mp_dtype = _resolve_amp_dtype(device, cfg)
    log_every = int(getattr(cfg, "log_interval", 50))
    log_detail = getattr(cfg, "log_detail", None)
    do_grad_log = bool(getattr(log_detail, "grad_norms", True))

    _apply_epoch_train_mode(cfg, detector)

    if opt_detector:
        opt_detector.zero_grad(set_to_none=True)

    loss_meter = {"loss_detector": 0.0, "loss_total": 0.0}
    nsteps = 0
    train_loader = loaders["train_loader"]

    if loaders.get("set_epoch"):
        loaders["set_epoch"](epoch)

    for it, batch in enumerate(train_loader):
        ctx = {"phase": "train", "epoch": epoch, "iter": it, "global_step": global_step}
        log_this_step = (global_step % log_every) == 0
        try:
            batch = to_device(batch, device)
            if not isinstance(batch, dict):
                raise TypeError(f"train_loader must yield dict batches, got {type(batch).__name__}")
            if getattr(trainopt, "channels_last", False) and isinstance(batch.get("raw4", None), torch.Tensor):
                batch["raw4"] = batch["raw4"].contiguous(memory_format=torch.channels_last)

            stats = train_micro_step(
                detector=detector,
                opt_detector=opt_detector,
                loss_detector=loss_detector,
                scaler_detector=scal_detector,
                batch=batch,
                cfg=cfg,
                device=device,
                amp_dtype_=mp_dtype,
                ctx=ctx,
                compute_metrics=log_this_step,
            )
        except SystemExit:
            raise
        except Exception as e:
            if (isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()) and device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            batch_desc = f"type={type(batch).__name__}"
            if isinstance(batch, dict):
                try:
                    batch_desc += f" keys={sorted(list(batch.keys()))}"
                except Exception:
                    pass
            raise RuntimeError(
                f"[FAIL] Exception during train_micro_step{_ctx_str(ctx)}: {type(e).__name__}: {e} | batch={batch_desc}"
            ) from e

        loss_meter["loss_detector"] += float(stats.get("loss_detector", 0.0))
        loss_meter["loss_total"] += float(stats.get("loss_total", 0.0))
        nsteps += 1

        if log_this_step and do_grad_log:
            g_det = _grad_stats_from_module(detector)
            stats["grad/detector_l2"] = float(g_det["l2"])
            stats["grad/detector_mean_abs"] = float(g_det["mean_abs"])
            stats["grad/detector_grad_tensors"] = float(g_det["tensor_count"])

            det_unwrapped = unwrap(detector)
            bb = getattr(det_unwrapped, "backbone", None)
            if isinstance(bb, nn.Module):
                g_bb = _grad_stats_from_module(bb)
                stats["grad/backbone_l2"] = float(g_bb["l2"])
                stats["grad/backbone_mean_abs"] = float(g_bb["mean_abs"])
                stats["grad/backbone_grad_tensors"] = float(g_bb["tensor_count"])

        made_opt_step = ((it + 1) % accum) == 0
        if made_opt_step:
            if not getattr(trainopt, "freeze_detector", False) and opt_detector:
                step_optim(
                    opt_detector,
                    scal_detector,
                    mixed_precision=mp_enabled,
                    model=detector,
                    clip_norm=getattr(getattr(cfg, "optim", object()), "clip_norm_detector", None),
                )

            if ema_detector and not getattr(trainopt, "freeze_detector", False):
                ema_detector.update(unwrap(detector))

            try:
                if sched_detector:
                    sched_detector.step()
            except Exception:
                pass

        if log_this_step:
            lr_det = opt_detector.param_groups[0]["lr"] if opt_detector else 0.0
            print(
                f"[epoch {epoch:03d} | step {global_step:07d}] "
                f"lr_det={lr_det:.2e} loss={stats.get('loss_total', 0.0):.4f}"
                + (f" g_backbone={stats.get('grad/backbone_l2', 0.0):.3e}" if do_grad_log else "")
            )
            if logger:
                logger.log_metrics(
                    phase="train",
                    global_step=global_step,
                    epoch=epoch,
                    it=it,
                    metrics={**stats, "lr_detector": lr_det},
                )

        global_step += 1

        try:
            _maybe_save_step_ckpt(
                step_ckpt_config=step_ckpt_config,
                cfg=cfg,
                epoch=epoch,
                it=it,
                global_step=global_step,
                detector=detector,
                opt_detector=opt_detector,
                sched_detector=sched_detector,
                scal_detector=scal_detector,
                ema_detector=ema_detector,
            )
        except Exception as e:
            logging.getLogger("ckpt").warning(f"[step-ckpt] save failed at step={global_step}: {e}")

    if (len(train_loader) % accum) != 0:
        if not getattr(trainopt, "freeze_detector", False) and opt_detector:
            step_optim(
                opt_detector,
                scal_detector,
                mixed_precision=mp_enabled,
                model=detector,
                clip_norm=getattr(getattr(cfg, "optim", object()), "clip_norm_detector", None),
            )
        if ema_detector and not getattr(trainopt, "freeze_detector", False):
            ema_detector.update(unwrap(detector))
        try:
            if sched_detector:
                sched_detector.step()
        except Exception:
            pass

    if nsteps > 0:
        for k in loss_meter:
            loss_meter[k] /= nsteps
    return global_step, loss_meter


@torch.no_grad()
def validate(
    epoch: int,
    cfg: Any,
    device: torch.device,
    *,
    detector,
    loss_detector,
    loaders,
    use_ema: bool = False,
    logger: Optional[Any] = None,
    global_step: Optional[int] = None,
) -> Dict[str, float]:
    trainopt = getattr(cfg, "trainopt", cfg)
    val_max_batches = int(getattr(trainopt, "val_max_batches", 0) or 0)
    val_log_every = int(getattr(trainopt, "val_log_every", 50) or 50)

    calc_val_loss = bool(getattr(trainopt, "calc_val_loss", False))
    calc_large_ap = bool(getattr(trainopt, "calc_large_ap", False))

    enable_detector = bool(getattr(trainopt, "enable_detector", True))
    run_det_eval = bool(enable_detector and detector is not None)

    backup_det = None
    if use_ema:
        if hasattr(detector, "_ema") and getattr(detector, "_ema"):
            backup_det = {k: v.to("cpu", copy=True, non_blocking=True) for k, v in detector.state_dict().items()}
            detector._ema.copy_to(detector)

    if detector is not None:
        detector.eval()

    amp_enabled, amp_dtype = _resolve_amp_dtype(device, cfg)
    ap_meter = None
    ap_meter_large = None
    if run_det_eval:
        det_model = unwrap(detector)
        num_classes = getattr(det_model, "num_classes", 4)
        has_bg = getattr(det_model, "has_bg", True)
        ap_meter = CocoAveragePrecisionMeter(num_classes=num_classes, has_background=has_bg, min_area_ratio=0.0)
        ap_meter_large = (
            CocoAveragePrecisionMeter(num_classes=num_classes, has_background=has_bg, min_area_ratio=0.05)
            if calc_large_ap
            else None
        )
        mcfg = getattr(cfg, "metrics_detector", None)
        det_score_thr = float(getattr(mcfg, "score_thr", 0.05))
        det_max_dets = int(getattr(mcfg, "max_dets", 300))
        det_nms_iou = getattr(mcfg, "nms_iou", 0.5)
        if det_nms_iou is not None:
            det_nms_iou = float(det_nms_iou)

    meter: Dict[str, float] = {"loss_total": 0.0, "det/skipped": 0.0}
    loss_detector_sum = 0.0
    detector_steps = 0
    det_batch_steps = 0
    det_batch_sum: Dict[str, float] = {}
    hw_steps = 0
    hw_sum: Dict[str, float] = {}
    n_steps = 0
    loss_steps = 0
    nonfinite_detector_batches = 0
    nonfinite_total_batches = 0

    val_loader = loaders.get("val_loader") or loaders.get("val") or loaders.get("valid_loader")
    t0 = time.time()

    for it, batch in enumerate(val_loader):
        if val_max_batches > 0 and it >= val_max_batches:
            break
        ctx = {"phase": "val", "epoch": epoch, "iter": it}

        batch = to_device(batch, device)
        if "raw4" not in batch:
            continue
        raw4 = batch["raw4"].float()

        loss_detector_val = torch.tensor(0.0, device=device)
        skipped_frac = 0.0

        if run_det_eval:
            mp_detector_mode = str(getattr(trainopt, "mixed_precision_detector", "bf16")).lower()
            if mp_detector_mode in ("fp32", "float32", "none"):
                detector_amp_enabled, detector_amp_dtype = False, None
            elif mp_detector_mode in ("fp16", "float16", "half"):
                detector_amp_enabled, detector_amp_dtype = True, torch.float16
            elif mp_detector_mode in ("bf16", "bfloat16"):
                detector_amp_enabled, detector_amp_dtype = True, torch.bfloat16
            else:
                detector_amp_enabled, detector_amp_dtype = amp_enabled, amp_dtype

            d_out = forward_detector(
                detector,
                raw4,
                cfg,
                amp_enabled=detector_amp_enabled,
                amp_dtype_=detector_amp_dtype,
                ctx=ctx,
                skip_safe=True,
            )
            pred_logits, pred_boxes = d_out["pred_logits"], d_out["pred_boxes"]

            targets = _build_detector_targets_from_batch(batch, int(pred_logits.shape[0]), device)

            if calc_val_loss and loss_detector is not None:
                loss_inputs = dict(d_out)
                loss_inputs["pred_logits"] = pred_logits.float()
                loss_inputs["pred_boxes"] = pred_boxes.float()
                with torch.autocast(device_type=device.type, enabled=False):
                    ldict_detector = loss_detector(loss_inputs, targets)
                total_detector, _, _ = aggregate_losses_with_guard(
                    ldict_detector if isinstance(ldict_detector, dict) else {"loss_total": _safe(ldict_detector)},
                    policy="zero",
                    skip_safe=True,
                )
                loss_detector_val = total_detector
                if torch.isfinite(loss_detector_val).all():
                    loss_detector_sum += float(loss_detector_val.item())
                    detector_steps += 1
                else:
                    nonfinite_detector_batches += 1
                    loss_detector_val = torch.zeros_like(loss_detector_val)

            if ap_meter is not None:
                update_args = {
                    "outputs": {"pred_logits": pred_logits.detach(), "pred_boxes": pred_boxes.detach()},
                    "targets": targets,
                    "score_thr": det_score_thr,
                    "max_dets": det_max_dets,
                    "nms_iou": det_nms_iou,
                }
                ap_meter.update(**update_args)
                if ap_meter_large:
                    ap_meter_large.update(**update_args)

            try:
                detm = det_batch_metrics(
                    {"pred_logits": pred_logits.detach(), "pred_boxes": pred_boxes.detach()},
                    targets,
                    score_thr=det_score_thr,
                    max_dets=det_max_dets,
                    num_classes=num_classes,
                    has_background=has_bg,
                    nms_iou=det_nms_iou,
                    iou_thr=0.5,
                )
                used = False
                for key in ("precision@0.5", "recall@0.5", "f1@0.5", "mean_iou@0.5", "precision", "recall", "f1", "mean_iou"):
                    if key not in detm:
                        continue
                    v = detm[key]
                    if torch.is_tensor(v):
                        v = float(v.detach().item())
                    if isinstance(v, (int, float)) and math.isfinite(float(v)):
                        det_batch_sum[key] = det_batch_sum.get(key, 0.0) + float(v)
                        used = True
                if used:
                    det_batch_steps += 1
            except Exception:
                pass

            try:
                hw = compute_hardware_aware_batch_metrics(
                    {
                        "pred_logits": pred_logits.detach(),
                        "object_tokens": d_out.get("object_tokens", None).detach()
                        if isinstance(d_out.get("object_tokens", None), torch.Tensor)
                        else d_out.get("object_tokens", None),
                        "tokens_per_level": [
                            t.detach() if isinstance(t, torch.Tensor) else t for t in (d_out.get("tokens_per_level", []) or [])
                        ],
                    }
                )
                used = False
                for k, v in hw.items():
                    if isinstance(v, (int, float)) and math.isfinite(float(v)):
                        hw_sum[k] = hw_sum.get(k, 0.0) + float(v)
                        used = True
                if used:
                    hw_steps += 1
            except Exception:
                pass

        loss_step = loss_detector_val
        if torch.isfinite(loss_step).all():
            meter["loss_total"] += float(loss_step.detach().item())
            loss_steps += 1
        else:
            nonfinite_total_batches += 1
            if nonfinite_total_batches <= 5:
                bid = batch.get("id", None)
                if bid is None:
                    bid = batch.get("ids", None)
                if isinstance(bid, (list, tuple)):
                    bid = bid[:2]
                print(f"[val e{epoch:03d}] non-finite total loss at it={it:05d}; batch ignored; id={bid}")
        meter["det/skipped"] += float(skipped_frac)
        n_steps += 1

        if (it % val_log_every) == 0:
            print(f"[val e{epoch:03d}] it={it:05d} loss={meter['loss_total']/max(1,loss_steps):.4f} elapsed={(time.time() - t0)/60:.1f}m")

    if loss_steps > 0:
        meter["loss_total"] /= loss_steps
        meter["val/loss_all_nonfinite"] = 0.0
    else:
        meter["loss_total"] = 0.0
        meter["val/loss_all_nonfinite"] = 1.0
        print(f"[val e{epoch:03d}] warning: all validation loss steps are non-finite; forcing loss_total=0.0")

    if n_steps > 0:
        meter["det/skipped"] /= n_steps

    meter["loss_detector"] = (loss_detector_sum / detector_steps) if detector_steps > 0 else 0.0

    meter["val/nonfinite_detector_batches"] = float(nonfinite_detector_batches)
    meter["val/nonfinite_total_batches"] = float(nonfinite_total_batches)
    meter["val/loss_steps"] = float(loss_steps)

    if det_batch_steps > 0:
        for k, v in det_batch_sum.items():
            meter[f"det/{k}"] = float(v) / float(det_batch_steps)
    if hw_steps > 0:
        for k, v in hw_sum.items():
            meter[k] = float(v) / float(hw_steps)

    if run_det_eval and ap_meter is not None:
        try:
            ap_res = ap_meter.compute() or {}
            map_vec = ap_res.get("map", None)
            if isinstance(map_vec, torch.Tensor):
                map_vec = torch.nan_to_num(map_vec, nan=0.0)
                if map_vec.numel() > 0:
                    meter["det/mAP"] = float(map_vec.item() if map_vec.ndim == 0 else map_vec.mean().item())
                    meter["det/mAP50"] = float(map_vec.item() if map_vec.ndim == 0 else map_vec[0].item())
        except Exception:
            pass

        if ap_meter_large:
            try:
                ap_res_large = ap_meter_large.compute() or {}
                map_vec_l = ap_res_large.get("map", None)
                if isinstance(map_vec_l, torch.Tensor):
                    map_vec_l = torch.nan_to_num(map_vec_l, nan=0.0)
                    if map_vec_l.numel() > 0:
                        meter["det/mAP_large"] = float(map_vec_l.item() if map_vec_l.ndim == 0 else map_vec_l.mean().item())
                        meter["det/mAP50_large"] = float(map_vec_l.item() if map_vec_l.ndim == 0 else map_vec_l[0].item())
            except Exception:
                pass

        meter.setdefault("det/mAP", 0.0)
        meter.setdefault("det/mAP50", 0.0)
        meter.setdefault("det/mAP_large", 0.0)
        meter.setdefault("det/mAP50_large", 0.0)

    if backup_det is not None:
        detector.load_state_dict({k: v.to(device) for k, v in backup_det.items()}, strict=False)

    if run_det_eval:
        det_f1 = meter.get("det/f1", float("nan"))
        det_iou = meter.get("det/mean_iou", float("nan"))
        hw_tok = meter.get("hw/object_token_offdiag_cos2", float("nan"))
        msg = (
            f"[Validation Epoch {epoch}] loss_total={meter['loss_total']:.4f} "
            f"mAP={meter['det/mAP']:.4f} mAP50={meter['det/mAP50']:.4f} "
            f"mAP_L={meter['det/mAP_large']:.4f} mAP50_L={meter['det/mAP50_large']:.4f} "
        )
        if math.isfinite(det_f1):
            msg += f"f1@0.5={det_f1:.4f} "
        if math.isfinite(det_iou):
            msg += f"miou@0.5={det_iou:.4f} "
        if math.isfinite(hw_tok):
            msg += f"tok_offdiag={hw_tok:.4f} "
        msg += f"skipped={meter['det/skipped']:.3f}"
        print(msg)
    else:
        print(f"[Validation Epoch {epoch}] loss_total={meter['loss_total']:.4f} (detector disabled; mAP skipped)")

    if logger is not None and hasattr(logger, "log_metrics"):
        try:
            logger.log_metrics(
                phase="val",
                global_step=int(global_step if global_step is not None else 0),
                epoch=int(epoch),
                it=int(max(0, n_steps - 1)),
                metrics=meter,
            )
        except Exception:
            pass
    return meter


__all__ = [
    "to_device",
    "unwrap",
    "get_dtype_tag",
    "_resolve_amp_dtype",
    "forward_detector",
    "train_micro_step",
    "step_optim",
    "train_one_epoch",
    "validate",
]
