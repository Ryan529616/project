# -*- coding: utf-8 -*-
"""
builder.py

Detector-only builders for the hardware-aware detector stack.
"""

from __future__ import annotations

import inspect
import json
import os
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from configs.config import Cfg, DetectorCfg
from engine.losses.hardware_aware_detection_losses import (
    build_hardware_aware_detection_loss as _build_detector_loss,
)
import engine.optim_sched as osched
from models.hardware_aware_detector import DualStageBackboneDetector


def _asdict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    raise TypeError("cfg must be None, a dataclass, or a dict-like object")


def _filter_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(callable_obj)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _merge(base: Dict[str, Any], *adds: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for a in adds:
        if a:
            out.update(a)
    return out


def sanity_check_cfg(cfg: Cfg) -> None:
    d_model = int(getattr(cfg.detector, "d_model", 256))
    nhead = int(getattr(cfg.detector, "nhead", 8))
    if d_model % nhead != 0:
        raise ValueError(f"[builder] detector.d_model({d_model}) must be divisible by nhead({nhead})")


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: nn.Module) -> int:
    return count_params(model, trainable_only=True)


def has_trainable_params(model: nn.Module) -> bool:
    return any(p.requires_grad for p in model.parameters())


def model_report(model: nn.Module, name: str = "model") -> Dict[str, Any]:
    n_all = count_params(model, trainable_only=False)
    n_tr = count_params(model, trainable_only=True)
    return {
        "name": name,
        "params_all": int(n_all),
        "params_trainable": int(n_tr),
        "ratio_trainable": float(n_tr) / max(1, n_all),
    }


def collect_model_param_report(model: nn.Module, name: str = "model") -> Dict[str, Any]:
    param_rows: List[Dict[str, Any]] = []
    layer_map: Dict[str, Dict[str, Any]] = {}
    total = 0
    total_trainable = 0

    for p_name, p in model.named_parameters():
        n = int(p.numel())
        tr = bool(p.requires_grad)
        total += n
        if tr:
            total_trainable += n

        shape = [int(s) for s in p.shape]
        layer = p_name.rsplit(".", 1)[0] if "." in p_name else "<root>"
        row = {
            "name": p_name,
            "layer": layer,
            "shape": shape,
            "numel": n,
            "trainable": tr,
            "dtype": str(p.dtype).replace("torch.", ""),
        }
        param_rows.append(row)

        if layer not in layer_map:
            layer_map[layer] = {
                "layer": layer,
                "params_all": 0,
                "params_trainable": 0,
                "tensor_count": 0,
            }
        layer_map[layer]["params_all"] += n
        if tr:
            layer_map[layer]["params_trainable"] += n
        layer_map[layer]["tensor_count"] += 1

    layer_rows = sorted(layer_map.values(), key=lambda x: (-int(x["params_all"]), x["layer"]))
    summary = {
        "name": name,
        "params_all": int(total),
        "params_trainable": int(total_trainable),
        "params_frozen": int(total - total_trainable),
        "ratio_trainable": float(total_trainable) / max(1, int(total)),
        "parameter_tensor_count": len(param_rows),
        "layer_count": len(layer_rows),
    }
    return {
        "summary": summary,
        "layers": layer_rows,
        "parameters": param_rows,
    }


def dump_model_param_report(model: nn.Module, out_dir: str, *, name: str = "model") -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    report = collect_model_param_report(model, name=name)

    json_path = os.path.join(out_dir, f"{name}_params.json")
    txt_path = os.path.join(out_dir, f"{name}_params.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        s = report["summary"]
        f.write(f"# {name} parameter summary\n")
        f.write(f"params_all={s['params_all']}\n")
        f.write(f"params_trainable={s['params_trainable']}\n")
        f.write(f"params_frozen={s['params_frozen']}\n")
        f.write(f"ratio_trainable={s['ratio_trainable']:.6f}\n")
        f.write(f"layer_count={s['layer_count']}\n")
        f.write(f"parameter_tensor_count={s['parameter_tensor_count']}\n\n")

        f.write("[layers]\n")
        for layer in report["layers"]:
            f.write(
                f"{layer['layer']}\tparams_all={layer['params_all']}\t"
                f"params_trainable={layer['params_trainable']}\ttensors={layer['tensor_count']}\n"
            )
        f.write("\n[parameters]\n")
        for p in report["parameters"]:
            shape_s = "x".join(str(int(v)) for v in p["shape"])
            f.write(
                f"{p['name']}\tshape={shape_s}\tnumel={p['numel']}\t"
                f"trainable={int(bool(p['trainable']))}\tdtype={p['dtype']}\n"
            )

    ret = dict(report["summary"])
    ret.update({"json_path": json_path, "txt_path": txt_path})
    return ret


def build_detector(cfg: Optional[DetectorCfg] = None, **overrides) -> nn.Module:
    params = _merge(_asdict(cfg), overrides)
    _ = params.pop("arch", None)

    # Support compact config keys prefixed by dual_*
    key_map = {
        "dual_backbone_base_c": "backbone_base_c",
        "dual_token_counts": "token_counts",
        "dual_pe_max_h": "max_input_h",
        "dual_pe_max_w": "max_input_w",
        "dual_bbox_hidden_dim": "bbox_hidden_dim",
        "dual_token_ffn_mult": "token_ffn_mult",
        "dual_query_prior": "query_prior",
        "dual_query_prior_init": "query_prior_init",
        "dual_anchor_wh_prior": "anchor_wh_prior",
        "dual_backbone_stem_kind": "backbone_stem_kind",
        "dual_backbone_act": "backbone_act",
        "dual_backbone_bn_eps": "backbone_bn_eps",
        "dual_backbone_bottleneck_local_blocks": "backbone_bottleneck_local_blocks",
        "dual_backbone_gc_reduction": "backbone_gc_reduction",
        "dual_backbone_gc_gate": "backbone_gc_gate",
        "dual_backbone_gc_clamp_alpha": "backbone_gc_clamp_alpha",
        "dual_backbone_gc_clamp_beta": "backbone_gc_clamp_beta",
    }
    for src, dst in key_map.items():
        if dst not in params and src in params:
            params[dst] = params[src]

    params = _filter_kwargs(DualStageBackboneDetector, params)
    return DualStageBackboneDetector(**params)


def build_detector_loss(
    num_classes: Optional[int] = None,
    loss_cfg: Optional[Any] = None,
    model_cfg: Optional[Any] = None,
    **overrides,
) -> nn.Module:
    ncls = (
        int(num_classes)
        if num_classes is not None
        else int(overrides.get("num_classes"))
        if "num_classes" in overrides
        else int(getattr(model_cfg, "num_classes", None))
    )
    if ncls is None:
        raise RuntimeError("[builder] build_detector_loss needs num_classes")

    overrides = {k: v for k, v in overrides.items() if k != "num_classes"}
    kwargs = _merge(_asdict(loss_cfg), overrides)
    kwargs = _filter_kwargs(_build_detector_loss, kwargs)
    return _build_detector_loss(ncls, **kwargs)


@torch.no_grad()
def prime_detector_bridge(
    detector: nn.Module,
    *,
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> None:
    _ = detector, H, W, device, dtype


def attach_bridge_with_cfg(
    detector: nn.Module,
    cfg: Cfg,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    rebuild_if_mismatch: bool = False,
) -> None:
    _ = detector, cfg, device, dtype, rebuild_if_mismatch


def freeze_module(
    model: nn.Module,
    freeze: bool = True,
    exclude: Optional[Iterable[Union[str, re.Pattern]]] = None,
) -> None:
    if not freeze:
        return

    patterns: List[re.Pattern] = []
    for s in (exclude or []):
        if isinstance(s, re.Pattern):
            patterns.append(s)
            continue
        if isinstance(s, str) and s.startswith("re:"):
            try:
                patterns.append(re.compile(s[3:]))
                continue
            except Exception:
                pass
        try:
            patterns.append(re.compile(re.escape(str(s))))
        except Exception:
            pass

    def _excluded(name: str) -> bool:
        return any(p.search(name) is not None for p in patterns)

    for n, p in model.named_parameters():
        p.requires_grad = _excluded(n)


def maybe_channels_last(model: nn.Module, enable: bool, device: torch.device) -> None:
    if enable and device.type == "cuda":
        model.to(memory_format=torch.channels_last)


def maybe_compile(model: nn.Module, enable: bool, backend: str = "inductor", mode: str = "default") -> nn.Module:
    if not enable:
        return model
    if not hasattr(torch, "compile"):
        print("[warn] torch.compile unavailable, fallback to eager")
        return model
    try:
        return torch.compile(model, backend=backend, mode=mode, fullgraph=True, dynamic=True)
    except Exception as e:
        print(f"[warn] torch.compile failed: {e}; fallback to eager")
        return model


def build_param_groups(model: nn.Module, optim_cfg: Any) -> List[Dict]:
    return osched.param_groups(
        model=model,
        weight_decay=float(getattr(optim_cfg, "weight_decay", 0.01)),
        weight_decay_norm=float(getattr(optim_cfg, "weight_decay_norm", 0.0)),
        weight_decay_bias=float(getattr(optim_cfg, "weight_decay_bias", 0.0)),
        skip_names=getattr(optim_cfg, "skip_names", None),
        skip_regex=getattr(optim_cfg, "skip_regex", None),
        norm_keywords=getattr(
            optim_cfg,
            "norm_keywords",
            ("norm", "bn", "gn", "ln", "layernorm", "batchnorm", "groupnorm"),
        ),
        bias_keyword=str(getattr(optim_cfg, "bias_keyword", ".bias")),
        layer_decay=getattr(optim_cfg, "layer_decay", None),
        layer_map=getattr(optim_cfg, "layer_map", None),
        layer_id_fn=getattr(optim_cfg, "layer_id_fn", None),
    )


def build_optimizer_for(
    model: nn.Module,
    optim_cfg: Any,
    *,
    lr_override: Optional[float] = None,
    role: Optional[str] = None,
) -> Optional[torch.optim.Optimizer]:
    n_tr = count_trainable_params(model)
    if n_tr == 0:
        name = getattr(model, "__class__", type("obj", (), {})).__name__
        print(f"[optim] skip optimizer for {name}: no trainable params")
        return None

    if role == "detector" and hasattr(optim_cfg, "lr_detector"):
        lr_from_cfg = float(getattr(optim_cfg, "lr_detector"))
    elif hasattr(optim_cfg, "lr"):
        lr_from_cfg = float(getattr(optim_cfg, "lr"))
    else:
        lr_from_cfg = 2e-4

    base_lr = float(lr_override if lr_override is not None else lr_from_cfg)

    return osched.build_optimizer(
        model,
        lr=base_lr,
        weight_decay=float(getattr(optim_cfg, "weight_decay", 0.01)),
        betas=tuple(getattr(optim_cfg, "betas", (0.9, 0.999))),
        eps=float(getattr(optim_cfg, "eps", 1e-8)),
        optim_name=str(getattr(optim_cfg, "name", "adamw")).lower(),
        weight_decay_norm=float(getattr(optim_cfg, "weight_decay_norm", 0.0)),
        weight_decay_bias=float(getattr(optim_cfg, "weight_decay_bias", 0.0)),
        skip_names=getattr(optim_cfg, "skip_names", None),
        skip_regex=getattr(optim_cfg, "skip_regex", None),
        norm_keywords=getattr(
            optim_cfg,
            "norm_keywords",
            ("norm", "bn", "gn", "ln", "layernorm", "batchnorm", "groupnorm"),
        ),
        layer_decay=getattr(optim_cfg, "layer_decay", None),
        layer_map=getattr(optim_cfg, "layer_map", None),
        layer_id_fn=getattr(optim_cfg, "layer_id_fn", None),
        fused=getattr(optim_cfg, "fused", None),
        foreach=getattr(optim_cfg, "foreach", None),
    )


def build_schedulers(
    cfg: Cfg,
    *,
    opt_detector: Optional[torch.optim.Optimizer] = None,
    steps_per_epoch: int,
    opt_u: Optional[torch.optim.Optimizer] = None,
) -> Tuple[None, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    _ = opt_u
    sched_root = getattr(cfg, "sched", None)

    side_obj = getattr(sched_root, "detector", None) if sched_root is not None else None

    def _get(key: str, default: Any = None) -> Any:
        if side_obj is not None and hasattr(side_obj, key):
            v = getattr(side_obj, key)
            if v is not None:
                return v
        if sched_root is not None and hasattr(sched_root, key):
            v = getattr(sched_root, key)
            if v is not None:
                return v
        return default

    if opt_detector is None:
        return None, None

    kind = str(_get("type", getattr(sched_root, "type", "cosine"))).lower()
    warmup = int(_get("warmup_steps", 0) or 0)
    total_steps = int(_get("total_steps", 0) or 0)
    if total_steps <= 0:
        total_steps = int(steps_per_epoch) * int(getattr(cfg, "epochs", 1))
    min_lr = float(_get("min_lr", 1e-6))
    warmup_init_lr = _get("warmup_init_lr", None)

    if kind in ("holdcos", "warmup_hold_cosine", "hold_cosine"):
        hold_steps = int(_get("hold_steps", 0) or 0)
        sched_det = osched.build_warmup_hold_cosine(
            opt_detector,
            warmup_steps=warmup,
            hold_steps=hold_steps,
            total_steps=total_steps,
            hold_lr_scale=float(_get("hold_lr_scale", 0.5)),
            min_lr=min_lr,
            warmup_init_lr=warmup_init_lr,
        )
        return None, sched_det

    if kind in ("onecycle", "one_cycle", "onecyclelr", "1cycle") and hasattr(osched, "build_onecycle"):
        max_lr = float(_get("max_lr", getattr(cfg.optim, "lr_detector", getattr(cfg.optim, "lr", 1e-3))))
        sched_det = osched.build_onecycle(
            opt_detector,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=float(_get("pct_start", 0.3)),
            start_lr=float(_get("start_lr", 0.0)),
            final_lr=float(_get("final_lr", 0.0)),
        )
        return None, sched_det

    sched_det = osched.build_cosine_with_warmup(
        opt_detector,
        warmup_steps=warmup,
        total_steps=total_steps,
        min_lr=min_lr,
        warmup_init_lr=warmup_init_lr,
    )
    return None, sched_det


def build_grad_clipper(cfg: Cfg) -> osched.GradClipper:
    max_norm = getattr(cfg.trainopt, "clip_grad_norm", None)
    clip_value = getattr(cfg.trainopt, "clip_grad_value", None)
    return osched.GradClipper(max_norm=max_norm, clip_value=clip_value)


def summarize_param_groups(optim: torch.optim.Optimizer) -> Dict[str, int]:
    return osched.summarize_param_groups(optim)


__all__ = [
    "build_detector",
    "build_detector_loss",
    "prime_detector_bridge",
    "attach_bridge_with_cfg",
    "freeze_module",
    "maybe_channels_last",
    "maybe_compile",
    "count_params",
    "count_trainable_params",
    "has_trainable_params",
    "model_report",
    "collect_model_param_report",
    "dump_model_param_report",
    "sanity_check_cfg",
    "build_param_groups",
    "build_optimizer_for",
    "build_schedulers",
    "build_grad_clipper",
    "summarize_param_groups",
]
