# -*- coding: utf-8 -*-
"""
checkpoint.py

Detector checkpoint utilities.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

LOG = logging.getLogger("ckpt")


def _add_safe_globals() -> None:
    try:
        if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    except Exception:
        pass


def _torch_load(path: str, map_location: Union[str, torch.device, None] = "cpu") -> Dict[str, Any]:
    _add_safe_globals()
    try:
        return torch.load(path, map_location=map_location)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        pass

    try:
        return torch.load(path, map_location=map_location, weights_only=False)  # type: ignore[arg-type]
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[arg-type]


def _safe_torch_save(obj: Dict[str, Any], path: str, sync: bool = False) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

    if sync:
        try:
            with open(path, "rb") as f:
                os.fsync(f.fileno())
            if d:
                dfd = os.open(d, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
        except Exception:
            pass


def _strip_prefixes(sd: Dict[str, Any], prefixes: Tuple[str, ...]) -> Dict[str, Any]:
    if not prefixes:
        return sd
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
                break
        out[nk] = v
    return out


def _cast_state_dict_dtype_(sd: Dict[str, Any], cast_dtype: Optional[str]) -> None:
    if not cast_dtype:
        return
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    tgt = mapping.get(str(cast_dtype).lower())
    if tgt is None:
        return
    for k, v in list(sd.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            sd[k] = v.to(dtype=tgt)


def _filter_state_dict_for_model(sd: Dict[str, Any], model_sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], List[Tuple]]:
    filtered: Dict[str, Any] = {}
    skipped: List[Tuple] = []
    for k, v in sd.items():
        if k not in model_sd:
            skipped.append((k, "not_in_model"))
            continue
        try:
            vv = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            if tuple(vv.shape) != tuple(model_sd[k].shape):
                skipped.append((k, "shape_mismatch", tuple(vv.shape), tuple(model_sd[k].shape)))
                continue
            filtered[k] = vv
        except Exception as e:
            skipped.append((k, f"convert_error:{type(e).__name__}:{e}"))
    return filtered, skipped


def _summarize_kv_counts(sd: Dict[str, Any]) -> Dict[str, int]:
    t, p = 0, 0
    for v in sd.values():
        if isinstance(v, torch.Tensor):
            t += 1
            p += v.numel()
    return {"num_tensors": t, "num_params": p}


def _maybe_state(x: Any) -> Optional[Any]:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if hasattr(x, "state_dict"):
        try:
            return x.state_dict()
        except Exception:
            return None
    return None


def _extract_state_dict_like(blob: Any, prefer_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(blob, dict):
        return None

    if prefer_key and prefer_key in blob and isinstance(blob[prefer_key], dict):
        return blob[prefer_key]

    for k in ("detector", "detr", "state_dict", "model"):
        if isinstance(blob.get(k, None), dict):
            return blob[k]

    cnt = 0
    for k, v in blob.items():
        if isinstance(k, str) and (isinstance(v, torch.Tensor) or hasattr(v, "shape")):
            cnt += 1
            if cnt >= 8:
                return blob
    if cnt >= 1:
        return blob
    return None


def save_dual_ckpt(
    path: str,
    step: int,
    *,
    detector: Optional[nn.Module] = None,
    detr: Optional[nn.Module] = None,
    opt: Optional[Any] = None,
    sched: Optional[Any] = None,
    scaler: Optional[Any] = None,
    ema: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    sync: bool = False,
) -> None:
    model_det = detector if detector is not None else detr

    blob: Dict[str, Any] = {
        "step": int(step),
        "meta": meta or {},
        "extra": extra or {},
    }
    if model_det is not None:
        sd_det = model_det.state_dict()
        blob["detector"] = sd_det

    d_opt = _maybe_state(opt)
    d_sched = _maybe_state(sched)
    d_scaler = _maybe_state(scaler)
    d_ema = _maybe_state(ema)
    if d_opt is not None:
        blob["opt"] = d_opt
    if d_sched is not None:
        blob["sched"] = d_sched
    if d_scaler is not None:
        blob["scaler"] = d_scaler
    if d_ema is not None:
        blob["ema"] = d_ema

    _safe_torch_save(blob, path, sync=sync)
    LOG.info(f"[ckpt] saved: {path} (step={step})")


def load_dual_ckpt(
    path: str,
    *,
    detector: Optional[nn.Module] = None,
    detr: Optional[nn.Module] = None,
    map_location: Union[str, torch.device, None] = "cpu",
    strict: bool = False,
    shape_filter: bool = True,
    cast_dtype: Optional[str] = None,
    strip_prefixes: Tuple[str, ...] = ("module.", "model.", "_orig_mod."),
    log_mismatch: bool = True,
) -> Tuple[int, Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ckpt = _torch_load(path, map_location=map_location)
    step = int(ckpt.get("step", -1))

    payload: Dict[str, Any] = {
        "step": step,
        "meta": ckpt.get("meta", {}),
        "extra": ckpt.get("extra", {}),
    }

    model_det = detector if detector is not None else detr

    if model_det is not None:
        sd0 = None
        picked = None
        for k in ("detector", "detr"):
            if isinstance(ckpt.get(k, None), dict):
                sd0 = ckpt[k]
                picked = k
                break
        if sd0 is None:
            sd0 = _extract_state_dict_like(ckpt)
            picked = "state_dict_like"

        if isinstance(sd0, dict):
            sd = _strip_prefixes(sd0, strip_prefixes)
            _cast_state_dict_dtype_(sd, cast_dtype)

            skipped: List[Tuple] = []
            if (not strict) and shape_filter:
                filtered, skipped = _filter_state_dict_for_model(sd, model_det.state_dict())
                inc = model_det.load_state_dict(filtered, strict=False)
            else:
                inc = model_det.load_state_dict(sd, strict=bool(strict))

            missing = list(getattr(inc, "missing_keys", []))
            unexpected = list(getattr(inc, "unexpected_keys", []))

            payload["detector_missing"] = missing
            payload["detector_unexpected"] = unexpected
            payload["detector_skipped"] = skipped
            payload["detector_stats"] = _summarize_kv_counts(sd)

            if log_mismatch:
                ok = len(model_det.state_dict()) - len(missing)
                LOG.info(
                    f"[ckpt] loaded detector from '{picked}': "
                    f"ok={ok} missing={len(missing)} unexpected={len(unexpected)} skipped={len(skipped)}"
                )
        else:
            LOG.warning("[ckpt] detector weights not found in checkpoint")

    for k in ("opt", "sched", "scaler", "ema"):
        if k in ckpt and isinstance(ckpt[k], dict):
            payload[k] = ckpt[k]

    return step, payload


def load_weights(
    model: nn.Module,
    path: str,
    *,
    prefer_key: Optional[str] = None,
    map_location: Union[str, torch.device, None] = "cpu",
    strict: bool = False,
    shape_filter: bool = True,
    cast_dtype: Optional[str] = None,
    strip_prefixes: Tuple[str, ...] = ("module.", "model.", "_orig_mod."),
    log_mismatch: bool = True,
) -> Dict[str, Any]:
    blob = _torch_load(path, map_location=map_location)
    sd0 = _extract_state_dict_like(blob, prefer_key=prefer_key)
    if sd0 is None or not isinstance(sd0, dict):
        raise RuntimeError(f"checkpoint has no usable state_dict: {path}")

    sd = _strip_prefixes(sd0, strip_prefixes)
    _cast_state_dict_dtype_(sd, cast_dtype)

    skipped: List[Tuple] = []
    if (not strict) and shape_filter:
        filtered, skipped = _filter_state_dict_for_model(sd, model.state_dict())
        inc = model.load_state_dict(filtered, strict=False)
    else:
        inc = model.load_state_dict(sd, strict=bool(strict))

    missing = list(getattr(inc, "missing_keys", []))
    unexpected = list(getattr(inc, "unexpected_keys", []))
    if log_mismatch:
        ok = len(model.state_dict()) - len(missing)
        LOG.info(
            f"[init] loaded {model.__class__.__name__}: "
            f"ok={ok} missing={len(missing)} unexpected={len(unexpected)} skipped={len(skipped)}"
        )

    return {
        "missing": missing,
        "unexpected": unexpected,
        "skipped": skipped,
        "stats": _summarize_kv_counts(sd),
    }


__all__ = [
    "save_dual_ckpt",
    "load_dual_ckpt",
    "load_weights",
]
