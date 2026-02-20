#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check consistency between raw10 Bayer PNG + sidecar meta and generated raw10_npy tensors.

Usage example:
  python tools/check_raw10_npy_consistency.py \
    --merged-json configs/train_merged.json \
    --raw-root /home/ryan529/project/bdd100k/raw10/train \
    --npy-root /home/ryan529/project/bdd100k/raw10_npy/train \
    --samples 200
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v3 as iio
import numpy as np


def _to_4ch(
    raw_u16: np.ndarray,
    *,
    pattern: str = "RGGB",
    black_level: float = 64.0,
    white_total: float = 1087.0,
) -> np.ndarray:
    x = np.asarray(raw_u16)
    if x.ndim == 3:
        x = x[..., 0]
    if x.ndim != 2:
        raise ValueError(f"Expect raw shape (H,W), got {x.shape}")

    h2, w2 = x.shape[0] // 2, x.shape[1] // 2
    x = x[: h2 * 2, : w2 * 2].astype(np.float32, copy=False)
    denom = max(1.0, float(white_total) - float(black_level))
    x = np.clip((x - float(black_level)) / denom, 0.0, 1.0)

    pat = str(pattern).upper()
    if pat == "RGGB":
        r, g1, g2, b = x[0::2, 0::2], x[0::2, 1::2], x[1::2, 0::2], x[1::2, 1::2]
    elif pat == "BGGR":
        r, g1, g2, b = x[1::2, 1::2], x[0::2, 1::2], x[1::2, 0::2], x[0::2, 0::2]
    elif pat == "GRBG":
        r, g1, g2, b = x[0::2, 1::2], x[0::2, 0::2], x[1::2, 1::2], x[1::2, 0::2]
    elif pat == "GBRG":
        r, g1, g2, b = x[1::2, 0::2], x[0::2, 0::2], x[1::2, 1::2], x[0::2, 1::2]
    else:
        raise ValueError(f"Unsupported pattern: {pattern}")

    return np.stack([r, g1, g2, b], axis=0).astype(np.float32, copy=False)


def _load_items(path: Path) -> List[Dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("items", "data", "images"):
            v = obj.get(k)
            if isinstance(v, list):
                return v
    raise ValueError(f"Unsupported merged json format: {path}")


def _to_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"npy must be 3D, got {arr.shape}")
    if arr.shape[0] == 4:
        return arr
    if arr.shape[-1] == 4:
        return np.ascontiguousarray(arr.transpose(2, 0, 1))
    raise ValueError(f"npy must be 4-channel, got {arr.shape}")


def main() -> int:
    ap = argparse.ArgumentParser("Check raw10->raw10_npy consistency")
    ap.add_argument("--merged-json", required=True, type=Path)
    ap.add_argument("--raw-root", required=True, type=Path)
    ap.add_argument("--npy-root", required=True, type=Path)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--mae-thr", type=float, default=5e-4)
    ap.add_argument("--max-thr", type=float, default=5e-3)
    args = ap.parse_args()

    items = _load_items(args.merged_json)
    random.Random(int(args.seed)).shuffle(items)

    checked = 0
    missing = 0
    bad_shape = 0
    bad_value = 0
    maes: List[float] = []
    maxes: List[float] = []
    sat0: List[float] = []
    sat1: List[float] = []

    for it in items:
        sid = str(it.get("id", "")).strip()
        if not sid:
            continue
        raw_path = args.raw_root / f"{sid}.png"
        npy_path = args.npy_root / f"{sid}.npy"
        meta_path = args.raw_root / f"{sid}.json"
        if (not raw_path.is_file()) or (not npy_path.is_file()):
            missing += 1
            continue

        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        pattern = str(meta.get("pattern", "RGGB")).upper()
        black = float(meta.get("black_level", 64.0))
        white_total = black + float(meta.get("white_level", 1023.0))

        raw = iio.imread(raw_path)
        ref = _to_4ch(raw, pattern=pattern, black_level=black, white_total=white_total)

        arr = np.load(npy_path, mmap_mode="r")
        arr = _to_chw(arr).astype(np.float32, copy=False)
        sat0.append(float((arr <= 0.0).mean()))
        sat1.append(float((arr >= 1.0).mean()))

        if arr.shape != ref.shape:
            bad_shape += 1
        else:
            diff = np.abs(arr - ref)
            mae = float(diff.mean())
            mxe = float(diff.max())
            maes.append(mae)
            maxes.append(mxe)
            if mae > float(args.mae_thr) or mxe > float(args.max_thr):
                bad_value += 1

        checked += 1
        if checked >= max(1, int(args.samples)):
            break

    if checked == 0:
        print("[check] no comparable pairs found")
        return 2

    print(f"[check] checked={checked} missing_pairs={missing}")
    print(f"[check] bad_shape={bad_shape} bad_value={bad_value}")
    if maes:
        print(
            "[check] mae(mean/p95/max)=%.6g / %.6g / %.6g"
            % (float(np.mean(maes)), float(np.percentile(maes, 95)), float(np.max(maes)))
        )
        print(
            "[check] max_abs(mean/p95/max)=%.6g / %.6g / %.6g"
            % (float(np.mean(maxes)), float(np.percentile(maxes, 95)), float(np.max(maxes)))
        )
    print(
        "[check] sat0(mean/p95)=%.6g / %.6g, sat1(mean/p95)=%.6g / %.6g"
        % (
            float(np.mean(sat0)),
            float(np.percentile(sat0, 95)),
            float(np.mean(sat1)),
            float(np.percentile(sat1, 95)),
        )
    )

    return 1 if (bad_shape > 0 or bad_value > 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())

