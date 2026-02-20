#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware-aware detector standalone inference.

Adapted from the legacy infer CLI style:
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import Cfg
from engine.metrics.detection_metrics import decode_detections_for_image
from models.builder import build_detector
from utils.checkpoint import load_dual_ckpt, load_weights

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


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _guess_cfg_path(ckpt_path: str) -> Optional[str]:
    p = Path(ckpt_path).resolve()
    cands: List[Path] = []

    # .../run/ckpt/best.pt -> .../run/config_dump.json
    if p.parent.name == "ckpt":
        cands.append(p.parent.parent / "config_dump.json")
        cands.append(p.parent.parent / "config_dump.yaml")

    # .../run/ckpt/steps/*.pt
    if p.parent.name == "steps" and p.parent.parent.name == "ckpt":
        cands.append(p.parent.parent.parent / "config_dump.json")
        cands.append(p.parent.parent.parent / "config_dump.yaml")

    cands.append(Path(PROJECT_ROOT) / "configs" / "cfg_hardware_aware_bdd100k_v1.json")

    for c in cands:
        if c.is_file() and c.suffix.lower() == ".json":
            return str(c)
    return None


def _load_cfg(cfg_path: Optional[str]) -> Tuple[Cfg, Optional[str]]:
    cfg = Cfg()
    if cfg_path and os.path.isfile(cfg_path):
        cfg.apply_overrides(_load_json(cfg_path), source="json")
        return cfg, cfg_path
    return cfg, None


def _load_detector_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    try:
        step, payload = load_dual_ckpt(
            ckpt_path,
            detector=model,
            map_location="cpu",
            strict=False,
            shape_filter=True,
            cast_dtype=None,
            strip_prefixes=("module.", "model.", "_orig_mod."),
            log_mismatch=True,
        )
        missing = len(payload.get("detector_missing", [])) if isinstance(payload, dict) else -1
        unexpected = len(payload.get("detector_unexpected", [])) if isinstance(payload, dict) else -1
        print(f"[ckpt] load_dual_ckpt ok: step={step} missing={missing} unexpected={unexpected}", flush=True)
        return
    except Exception as e:
        print(f"[ckpt] load_dual_ckpt failed, fallback to load_weights: {type(e).__name__}: {e}", flush=True)

    loaded = False
    for key in ("detector", "detr", "model", "state_dict"):
        try:
            load_weights(
                model,
                ckpt_path,
                prefer_key=key,
                map_location="cpu",
                strict=False,
                shape_filter=True,
                cast_dtype=None,
                strip_prefixes=("module.", "model.", "_orig_mod."),
                log_mismatch=True,
            )
            print(f"[ckpt] load_weights ok (prefer_key={key})", flush=True)
            loaded = True
            break
        except Exception:
            continue
    if not loaded:
        raise RuntimeError(f"unable to load detector weights from checkpoint: {ckpt_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Standalone hardware-aware detector inference")
    ap.add_argument("--input", required=True, help=".npy file/dir/glob")
    ap.add_argument("--ckpt", required=True, help="checkpoint path")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--cfg", default=None, help="detector config json; if omitted, auto-guess from ckpt run dir")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda[:id]/mps")
    ap.add_argument("--amp", choices=["auto", "none", "bf16", "fp16"], default="auto")

    ap.add_argument("--pattern", choices=["RGGB", "GRBG", "GBRG", "BGGR"], default=None)
    ap.add_argument("--pad-multiple", type=int, default=8, help="pad input to multiple of this value")

    ap.add_argument("--score-thr", type=float, default=0.30)
    ap.add_argument("--nms-iou", type=float, default=0.60, help="set <0 to disable NMS")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--num-classes", type=int, default=None)
    ap.add_argument("--names", default=None, help="class names json list or comma-separated string")

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

    cfg_path = args.cfg or _guess_cfg_path(args.ckpt)
    cfg, used_cfg_path = _load_cfg(cfg_path)
    if used_cfg_path:
        print(f"[cfg] {used_cfg_path}", flush=True)
    else:
        print("[cfg] using default detector config dataclass values", flush=True)

    model = build_detector(cfg.detector).to(device).eval()
    _load_detector_weights(model, args.ckpt)

    pad_multiple = int(args.pad_multiple)
    if pad_multiple <= 0:
        pad_multiple = int(getattr(cfg.data, "pad_multiple", 8) or 8)
    pad_multiple = max(1, pad_multiple)

    num_classes = int(args.num_classes) if args.num_classes is not None else int(getattr(cfg.detector, "num_classes", 4))
    class_names = _parse_class_names(args.names, num_classes=num_classes)

    print(
        f"[init] device={device} amp={amp_mode} files={len(paths)} pad_multiple={pad_multiple} num_classes={num_classes}",
        flush=True,
    )

    summary: Dict[str, Any] = {
        "model": "hardware_aware_detector",
        "ckpt": os.path.abspath(args.ckpt),
        "cfg": os.path.abspath(used_cfg_path) if used_cfg_path else None,
        "device": str(device),
        "amp": amp_mode,
        "num_files": len(paths),
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

        size_hw = torch.tensor([hp, wp], device=device, dtype=torch.long)
        det = decode_detections_for_image(
            outputs,
            b=0,
            size_hw=size_hw,
            score_thr=float(args.score_thr),
            max_dets=int(args.topk),
            num_classes=int(outputs.get("num_classes", num_classes)),
            has_background=bool(outputs.get("has_bg", True)),
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

        print(f"[infer][{idx}/{len(paths)}] {name} dets={len(scores_list)}" + (f" {ms:.1f}ms" if args.timeit else ""), flush=True)

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
