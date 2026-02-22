#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RT-DETR (RGGB) standalone inference.

CLI style follows existing project tools:
- --input --ckpt --out --device
- Supports file/dir/glob .npy input.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RTDETR_ROOT = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(RTDETR_ROOT)
TRANSFORMER_ROOT = os.path.join(PROJECT_ROOT, "transformer")

if RTDETR_ROOT not in sys.path:
    sys.path.insert(0, RTDETR_ROOT)
if TRANSFORMER_ROOT not in sys.path:
    sys.path.insert(0, TRANSFORMER_ROOT)

from engine.metrics.detection_metrics import decode_detections_for_image
from models.rtdetr_rggb import RTDETRRGGB

_OFFSETS = {
    "RGGB": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "GRBG": ((0, 1), (0, 0), (1, 1), (1, 0)),
    "GBRG": ((1, 0), (0, 0), (1, 1), (0, 1)),
    "BGGR": ((1, 1), (1, 0), (0, 1), (0, 0)),
}

_DEFAULT_CLASS_NAMES = ["vehicle", "human", "traffic light", "traffic sign"]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _list_npy(input_arg: str) -> List[str]:
    if os.path.isdir(input_arg):
        return sorted(glob.glob(os.path.join(input_arg, "*.npy")))
    if os.path.isfile(input_arg) and input_arg.endswith(".npy"):
        return [input_arg]
    return sorted(glob.glob(input_arg))


def _mosaic_to_4half(mono_chw: np.ndarray, pattern: str) -> np.ndarray:
    if mono_chw.ndim != 3 or mono_chw.shape[0] != 1:
        raise ValueError(f"expect (1,H,W), got {tuple(mono_chw.shape)}")
    x = mono_chw[0]
    h2, w2 = x.shape[0] // 2, x.shape[1] // 2
    x = x[: 2 * h2, : 2 * w2].astype(np.float32, copy=False)
    pat = pattern.upper()
    if pat not in _OFFSETS:
        raise ValueError(f"unsupported Bayer pattern: {pattern}")
    o_r, o_g1, o_g2, o_b = _OFFSETS[pat]
    r = x[o_r[0] :: 2, o_r[1] :: 2]
    g1 = x[o_g1[0] :: 2, o_g1[1] :: 2]
    g2 = x[o_g2[0] :: 2, o_g2[1] :: 2]
    b = x[o_b[0] :: 2, o_b[1] :: 2]
    return np.stack([r, g1, g2, b], axis=0)


def _normalize_raw4_auto(raw4_chw: np.ndarray) -> np.ndarray:
    x = raw4_chw.astype(np.float32, copy=False)
    x_max = float(np.max(x)) if x.size else 1.0
    if x_max <= 1.5:
        return np.clip(x, 0.0, 1.0)
    if x_max <= 1200.0:
        den = 1023.0
    elif x_max <= 5000.0:
        den = 4095.0
    elif x_max <= 70000.0:
        den = 65535.0
    else:
        den = max(1.0, x_max)
    return np.clip(x / den, 0.0, 1.0)


def _load_raw4(path: str, pattern: Optional[str]) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise RuntimeError(f"{path}: expect 3D array, got {arr.shape}")

    if arr.shape[0] == 4:
        chw = arr
    elif arr.shape[-1] == 4:
        chw = np.transpose(arr, (2, 0, 1))
    elif arr.shape[0] == 1:
        if not pattern:
            raise RuntimeError(f"{path}: single-channel input needs --pattern")
        chw = _mosaic_to_4half(arr, pattern)
    elif arr.shape[-1] == 1:
        if not pattern:
            raise RuntimeError(f"{path}: single-channel input needs --pattern")
        chw = _mosaic_to_4half(np.transpose(arr, (2, 0, 1)), pattern)
    else:
        raise RuntimeError(f"{path}: unsupported shape {arr.shape}")

    chw = chw.astype(np.float32, copy=False)
    return _normalize_raw4_auto(chw)


def _pad_to_multiple(raw4_chw: np.ndarray, multiple: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    c, h, w = raw4_chw.shape
    h2 = ((h + multiple - 1) // multiple) * multiple
    w2 = ((w + multiple - 1) // multiple) * multiple
    if h2 == h and w2 == w:
        return raw4_chw, (h, w)
    out = np.pad(raw4_chw, ((0, 0), (0, h2 - h), (0, w2 - w)), mode="edge")
    return out, (h, w)


def _to_gray_bgr_u8(raw4_chw: np.ndarray) -> np.ndarray:
    gray = np.mean(raw4_chw, axis=0)
    gray = np.clip(gray, 0.0, 1.0)
    u8 = (gray * 255.0 + 0.5).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def _parse_class_names(names_arg: Optional[str], num_classes: int) -> List[str]:
    if names_arg:
        if os.path.isfile(names_arg):
            try:
                with open(names_arg, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return [str(x) for x in data]
            except Exception:
                pass
        parts = [x.strip() for x in names_arg.split(",") if x.strip()]
        if parts:
            return parts
    if num_classes == 4:
        return list(_DEFAULT_CLASS_NAMES)
    return [f"class_{i}" for i in range(num_classes)]


def _color_for_label(lbl: int) -> Tuple[int, int, int]:
    palette = [
        (0, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]
    return palette[int(lbl) % len(palette)]


def _draw_boxes(
    bgr_u8: np.ndarray,
    boxes_xyxy: Sequence[Sequence[float]],
    labels: Sequence[int],
    scores: Sequence[float],
    class_names: Sequence[str],
) -> np.ndarray:
    img = bgr_u8.copy()
    h, w = img.shape[:2]
    scale = max(0.8, max(h, w) / 640.0)
    thickness = max(1, int(round(2 * scale)))
    font_scale = 0.5 * scale

    for box, label, score in zip(boxes_xyxy, labels, scores):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = _color_for_label(int(label))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        cls = str(label)
        if 0 <= int(label) < len(class_names):
            cls = class_names[int(label)]
        text = f"{cls} {score:.3f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1))
        y_text_top = max(0, y1 - th - bl - 2)
        cv2.rectangle(img, (x1, y_text_top), (min(w - 1, x1 + tw + 2), y1), color, -1)
        cv2.putText(
            img,
            text,
            (x1 + 1, max(0, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return img


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_amp_mode(amp_arg: str, device: torch.device) -> str:
    if amp_arg != "auto":
        return amp_arg
    if device.type == "cuda":
        return "bf16"
    return "none"


def _autocast_ctx(device: torch.device, amp_mode: str):
    if device.type != "cuda":
        return contextlib.nullcontext()
    mode = str(amp_mode).lower()
    if mode in ("none", "off", "fp32"):
        return contextlib.nullcontext()
    if mode in ("fp16", "float16", "half"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _prepare_outputs_for_decode(outputs: Dict[str, Any], num_classes: int) -> Dict[str, Any]:
    """
    Align decode behavior with training loss when model outputs no background logit.
    If logits shape is [B,Q,C] (no-bg), append a zero background logit -> [B,Q,C+1].
    """
    if not isinstance(outputs, dict):
        return outputs

    logits = outputs.get("pred_logits", None)
    if not torch.is_tensor(logits) or logits.ndim != 3:
        return outputs

    has_bg = bool(outputs.get("has_bg", False))
    if has_bg:
        return outputs

    c_logit = int(logits.shape[-1])
    c_expect = int(num_classes)
    if c_logit != c_expect:
        return outputs

    bg = torch.zeros((logits.shape[0], logits.shape[1], 1), dtype=logits.dtype, device=logits.device)
    out = dict(outputs)
    out["pred_logits"] = torch.cat([logits, bg], dim=-1)
    out["has_bg"] = True
    return out


def _torch_load_compat(path: str, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)  # type: ignore[arg-type]
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        try:
            return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[arg-type]
        except TypeError:
            return torch.load(path, map_location=map_location)


def _nested_get_dict(blob: Dict[str, Any], key_path: str) -> Optional[Dict[str, Any]]:
    cur: Any = blob
    for part in str(key_path).split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part, None)
        if cur is None:
            return None
    return cur if isinstance(cur, dict) else None


def _dict_tensor_count(d: Dict[str, Any]) -> int:
    return sum(1 for v in d.values() if torch.is_tensor(v))


def _discover_state_dict_like(blob: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    preferred_roots = ("model", "state_dict", "module", "detector", "detr")
    for k in preferred_roots:
        v = blob.get(k, None)
        if isinstance(v, dict) and _dict_tensor_count(v) > 0:
            return v

    ema = blob.get("ema", None)
    if isinstance(ema, dict):
        for k in ("module", "model", "state_dict"):
            v = ema.get(k, None)
            if isinstance(v, dict) and _dict_tensor_count(v) > 0:
                return v

    if _dict_tensor_count(blob) > 0:
        return blob
    return None


def _extract_state_dict(blob: Any, prefer_key: Optional[str]) -> Dict[str, torch.Tensor]:
    if not isinstance(blob, dict):
        raise RuntimeError("checkpoint is not a dict")

    picked: Optional[Dict[str, Any]] = None
    if prefer_key:
        picked = _nested_get_dict(blob, str(prefer_key))
        if not (isinstance(picked, dict) and _dict_tensor_count(picked) > 0):
            picked = None
    if picked is None:
        picked = _discover_state_dict_like(blob)
    if picked is None:
        raise RuntimeError("checkpoint does not contain a usable state_dict")

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in picked.items():
        if not torch.is_tensor(v):
            continue
        kk = str(k)
        if kk.startswith("module."):
            kk = kk[len("module.") :]
        if kk.startswith("model."):
            kk = kk[len("model.") :]
        if kk.startswith("_orig_mod."):
            kk = kk[len("_orig_mod.") :]
        cleaned[kk] = v
    return cleaned


def _get_arch_value(ckpt_args: Dict[str, Any], key: str, cli_value: Optional[Any], default_value: Any, cast):
    if cli_value is not None:
        return cast(cli_value)
    if key in ckpt_args and ckpt_args[key] is not None:
        return cast(ckpt_args[key])
    return cast(default_value)


def _build_model_from_ckpt_args(blob: Dict[str, Any], cli: argparse.Namespace) -> Tuple[RTDETRRGGB, Dict[str, Any], int]:
    ckpt_args = blob.get("args", {}) if isinstance(blob.get("args"), dict) else {}

    num_classes = _get_arch_value(ckpt_args, "num_classes", cli.num_classes, 4, int)
    num_queries = _get_arch_value(ckpt_args, "num_queries", cli.num_queries, 300, int)
    d_model = _get_arch_value(ckpt_args, "d_model", cli.d_model, 256, int)
    nhead = _get_arch_value(ckpt_args, "nhead", cli.nhead, 8, int)
    aifi_layers = _get_arch_value(
        ckpt_args,
        "aifi_layers",
        cli.aifi_layers,
        _get_arch_value(ckpt_args, "enc_layers", None, 1, int),
        int,
    )
    dec_layers = _get_arch_value(ckpt_args, "dec_layers", cli.dec_layers, 6, int)
    dim_feedforward = _get_arch_value(ckpt_args, "dim_feedforward", cli.dim_feedforward, 1024, int)
    dropout = _get_arch_value(ckpt_args, "dropout", cli.dropout, 0.1, float)
    backbone = _get_arch_value(ckpt_args, "backbone", cli.backbone, "resnet50", str)

    model = RTDETRRGGB(
        num_classes=num_classes,
        num_queries=num_queries,
        d_model=d_model,
        nhead=nhead,
        aifi_layers=aifi_layers,
        dec_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        backbone=backbone,
    )

    pad_multiple = int(cli.pad_multiple) if cli.pad_multiple is not None else int(ckpt_args.get("pad_multiple", 8))
    pad_multiple = max(1, int(pad_multiple))

    model_cfg = {
        "num_classes": num_classes,
        "num_queries": num_queries,
        "d_model": d_model,
        "nhead": nhead,
        "aifi_layers": aifi_layers,
        "dec_layers": dec_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "backbone": backbone,
        "pad_multiple": pad_multiple,
    }
    return model, model_cfg, pad_multiple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Standalone RT-DETR (RGGB) inference")
    ap.add_argument("--input", required=True, help=".npy file/dir/glob")
    ap.add_argument("--ckpt", required=True, help="checkpoint path (best_mAP50.pt or last.pt)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda[:id]/mps")
    ap.add_argument("--amp", choices=["auto", "none", "bf16", "fp16"], default="auto")
    ap.add_argument("--ckpt-key", default=None, help="Optional checkpoint state_dict key path, e.g. model / ema.module")

    ap.add_argument("--pattern", choices=["RGGB", "GRBG", "GBRG", "BGGR"], default=None)
    ap.add_argument("--pad-multiple", type=int, default=None, help="override ckpt args pad_multiple")

    ap.add_argument("--score-thr", type=float, default=0.30)
    ap.add_argument("--nms-iou", type=float, default=0.60, help="set <0 to disable NMS")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--names", default=None, help="class names json list or comma-separated string")

    # Optional architecture overrides (default: read from ckpt args)
    ap.add_argument("--num-classes", type=int, default=None)
    ap.add_argument("--num-queries", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--nhead", type=int, default=None)
    ap.add_argument("--aifi-layers", type=int, default=None)
    ap.add_argument("--dec-layers", type=int, default=None)
    ap.add_argument("--dim-feedforward", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--backbone", type=str, default=None, choices=["resnet50", "r50", "r50vd"])

    ap.add_argument("--timeit", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dir(args.out)

    paths = _list_npy(args.input)
    if not paths:
        raise SystemExit(f"no .npy found for input: {args.input}")

    device = _resolve_device(args.device)
    amp_mode = _resolve_amp_mode(args.amp, device)

    blob = _torch_load_compat(args.ckpt, map_location="cpu")
    if not isinstance(blob, dict):
        raise SystemExit(f"checkpoint format error: {args.ckpt}")

    model, model_cfg, pad_multiple = _build_model_from_ckpt_args(blob, args)
    state_dict = _extract_state_dict(blob, args.ckpt_key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[ckpt] loaded missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"[ckpt] missing_examples={missing[:8]}", flush=True)
    if unexpected:
        print(f"[ckpt] unexpected_examples={unexpected[:8]}", flush=True)

    model = model.to(device).eval()

    num_classes = int(model_cfg["num_classes"])
    class_names = _parse_class_names(args.names, num_classes=num_classes)

    print(
        f"[init] device={device} amp={amp_mode} files={len(paths)} pad_multiple={pad_multiple} num_classes={num_classes}",
        flush=True,
    )

    summary: Dict[str, Any] = {
        "model": "rtdetr_rggb",
        "ckpt": os.path.abspath(args.ckpt),
        "device": str(device),
        "amp": amp_mode,
        "num_files": len(paths),
        "model_cfg": model_cfg,
        "files": [],
    }

    nms_iou = None if float(args.nms_iou) < 0 else float(args.nms_iou)

    for idx, path in enumerate(paths, start=1):
        name = os.path.splitext(os.path.basename(path))[0]
        t0 = time.perf_counter()

        raw4 = _load_raw4(path, args.pattern)
        raw4_pad, (h, w) = _pad_to_multiple(raw4, pad_multiple)
        hp, wp = raw4_pad.shape[-2:]

        inp = torch.from_numpy(raw4_pad).unsqueeze(0).to(device, non_blocking=True)
        with torch.inference_mode():
            with _autocast_ctx(device, amp_mode):
                outputs = model(inp)
        decode_outputs = _prepare_outputs_for_decode(outputs, num_classes=num_classes)

        size_hw = torch.tensor([hp, wp], device=device, dtype=torch.long)
        det = decode_detections_for_image(
            decode_outputs,
            b=0,
            size_hw=size_hw,
            score_thr=float(args.score_thr),
            max_dets=int(args.topk),
            num_classes=int(decode_outputs.get("num_classes", num_classes)),
            has_background=bool(decode_outputs.get("has_bg", True)),
            nms_iou=nms_iou,
        )

        boxes = det["boxes_px"].detach().cpu().float()
        labels = det["labels"].detach().cpu().long()
        scores = det["scores"].detach().cpu().float()

        if boxes.numel() > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, max(0, w - 1))
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, max(0, h - 1))

        boxes_list = boxes.tolist()
        labels_list = [int(x) for x in labels.tolist()]
        scores_list = [float(x) for x in scores.tolist()]

        vis_raw = _to_gray_bgr_u8(raw4)
        vis_det = _draw_boxes(vis_raw, boxes_list, labels_list, scores_list, class_names)

        raw_out = os.path.join(args.out, f"{name}_raw.png")
        det_out = os.path.join(args.out, f"{name}_det.png")
        json_out = os.path.join(args.out, f"{name}_dets.json")

        cv2.imwrite(raw_out, vis_raw)
        cv2.imwrite(det_out, vis_det)

        row = {
            "file": path,
            "size": [int(h), int(w)],
            "size_padded": [int(hp), int(wp)],
            "boxes_xyxy": boxes_list,
            "labels": labels_list,
            "scores": scores_list,
            "num_dets": int(len(scores_list)),
        }
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

        ms = (time.perf_counter() - t0) * 1000.0
        if args.timeit:
            row["time_ms"] = float(ms)
        summary["files"].append(row)

        suffix = f" {ms:.1f}ms" if args.timeit else ""
        print(f"[infer][{idx}/{len(paths)}] {name} dets={len(scores_list)}{suffix}", flush=True)

    if args.timeit and summary["files"]:
        times = [float(x.get("time_ms", 0.0)) for x in summary["files"]]
        summary["time_ms"] = {
            "mean": float(np.mean(times)),
            "p50": float(np.percentile(times, 50)),
            "p90": float(np.percentile(times, 90)),
            "max": float(np.max(times)),
        }

    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
