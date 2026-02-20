#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_human_masks.py — from weighted_saliency_masks only
- 來源：/home/ryan529/project/bdd100k/labels/sem_seg/weighted_saliency_masks/{split}/*.png
- 僅保留「有人類(==2)」的樣本並輸出：
    * binary_saliency_masks_human/{split}/xxx.png （0/255）
    * weighted_saliency_masks_human/{split}/xxx.png（0/1/2）
- 近期鄰縮放到 --width x --height（預設 640x384，32 倍數友善）
- 多進程處理 + 稽核 CSV + kept 名單
- RGB 稽核路徑寫死：/home/ryan529/project/bdd100k/images/10k/{split}/xxx.jpg/png

用法範例：
python build_human_masks.py \
  --splits train val test \
  --out-root /home/ryan529/project/bdd100k/labels/sem_seg \
  --report-root /home/ryan529/project/bdd100k/labels/sem_seg/reports \
  --width 640 --height 384 \
  --workers 8 --compress 3
"""

from __future__ import annotations
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

# 來源與 RGB 稽核路徑（寫死）
WEIGHTED_SRC_BASE = Path("/home/ryan529/project/bdd100k/labels/sem_seg/weighted_saliency_masks")
RGB_BASE = Path("/home/ryan529/project/bdd100k/images/10k")

# 值域檢查
VALID_WEIGHTED_VALUES = {0, 1, 2}
WEIGHTED_PERSON_VALUE = 2

def _load_png_gray(fp: Path) -> np.ndarray:
    im = Image.open(fp)
    arr = np.array(im)
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr

def _resize_nearest(mask: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return np.array(Image.fromarray(mask, mode="L").resize((out_w, out_h), resample=Image.NEAREST), dtype=np.uint8)

def _find_rgb_size(split: str, stem: str) -> Tuple[Optional[int], Optional[int]]:
    for ext in ("jpg", "png", "jpeg", "JPG", "PNG", "JPEG"):
        p = RGB_BASE / split / f"{stem}.{ext}"
        if p.exists():
            with Image.open(p) as im:
                w, h = im.size
            return w, h
    return None, None

def _quick_vals(a: np.ndarray) -> str:
    vals = np.unique(a).tolist()
    return "|".join(map(str, vals))

def _process_one(
    fp: Path,
    split: str,
    out_bin_dir: Path,
    out_wgt_dir: Path,
    out_w: int,
    out_h: int,
    compress: int,
) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        stem = fp.stem
        wgt = _load_png_gray(fp).astype(np.uint8)

        # 值域檢查
        vals = set(np.unique(wgt).tolist())
        unknown_vals = sorted(list(vals - VALID_WEIGHTED_VALUES))
        suspicious = 1 if unknown_vals else 0

        # 人類像素計數
        person_px = int((wgt == WEIGHTED_PERSON_VALUE).sum())
        if person_px <= 0:
            # 無人 → 不輸出檔案，只回報 CSV 行
            row = {
                "file": f"{stem}.png",
                "h": int(wgt.shape[0]),
                "w": int(wgt.shape[1]),
                "rgb_w": None, "rgb_h": None,
                "values": _quick_vals(wgt),
                "unknown_values": "|".join(map(str, unknown_vals)) if unknown_vals else "",
                "suspicious": suspicious,
                "kept": 0,
                "drop_reason": "no_human",
                "sal_px": int((wgt > 0).sum()),
                "person_px": 0,
                "out_w": out_w, "out_h": out_h,
            }
            return row, None

        # RGB 尺寸稽核
        rgb_w, rgb_h = _find_rgb_size(split, stem)
        if rgb_w is not None and (wgt.shape[1] != rgb_w or wgt.shape[0] != rgb_h):
            suspicious = 1

        # 生成 binary（0/255）
        biny = (wgt > 0).astype(np.uint8) * 255

        # 近期鄰縮放到固定大小
        biny_r = _resize_nearest(biny, out_w, out_h)
        wgt_r  = _resize_nearest(wgt,  out_w, out_h)

        # 輸出
        out_bin = out_bin_dir / f"{stem}.png"
        out_wgt = out_wgt_dir / f"{stem}.png"
        Image.fromarray(biny_r, mode="L").save(out_bin, compress_level=compress)
        Image.fromarray(wgt_r,  mode="L").save(out_wgt, compress_level=compress)

        row = {
            "file": f"{stem}.png",
            "h": int(wgt.shape[0]),
            "w": int(wgt.shape[1]),
            "rgb_w": rgb_w, "rgb_h": rgb_h,
            "values": _quick_vals(wgt),
            "unknown_values": "|".join(map(str, unknown_vals)) if unknown_vals else "",
            "suspicious": suspicious,
            "kept": 1,
            "drop_reason": "ok" if suspicious == 0 else "size_mismatch_or_values",
            "sal_px": int((wgt_r > 0).sum()),
            "person_px": int((wgt_r == WEIGHTED_PERSON_VALUE).sum()),
            "out_w": out_w, "out_h": out_h,
        }
        return row, None
    except Exception as e:
        return None, f"{fp}: {type(e).__name__}: {e}"

def _process_split(
    split: str,
    out_root: Path,
    report_root: Path,
    width: int,
    height: int,
    workers: int,
    compress: int,
) -> Tuple[int, int]:
    src_dir = WEIGHTED_SRC_BASE / split
    out_bin = out_root / "binary_saliency_masks_human" / split
    out_wgt = out_root / "weighted_saliency_masks_human" / split
    out_bin.mkdir(parents=True, exist_ok=True)
    out_wgt.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.png"))
    rows: List[Dict] = []
    errors: List[str] = []
    kept: List[str] = []

    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [
            ex.submit(_process_one, f, split, out_bin, out_wgt, width, height, compress)
            for f in files
        ]
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=f"[{split}] human-filtered build+resize {width}x{height}"):
            row, err = fut.result()
            if err:
                errors.append(err)
            elif row:
                rows.append(row)
                if row.get("kept", 0) == 1:
                    kept.append(row["file"])

    # CSV
    rpt_csv = report_root / f"{split}_human_filtered_audit.csv"
    fields = ["file","h","w","rgb_w","rgb_h","values","unknown_values",
              "suspicious","kept","drop_reason","sal_px","person_px","out_w","out_h"]
    with open(rpt_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # kept 名單
    kept_txt = report_root / f"{split}_human_filtered_kept.txt"
    with open(kept_txt, "w", encoding="utf-8") as f:
        for n in kept:
            f.write(n + "\n")

    # errors
    if errors:
        err_txt = report_root / f"{split}_human_filtered_errors.txt"
        with open(err_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(errors))

    print(f"[{split}] kept {len(kept)} / total {len(files)}  ({len(kept)/max(1,len(files)):.1%})")
    print(f"-> saved to: {out_bin} / {out_wgt}")
    print(f"-> report:   {rpt_csv}")
    print(f"-> kept list:{kept_txt}")
    if errors:
        print(f"-> errors:   {err_txt}  ({len(errors)})")

    return len(kept), len(errors)

def parse_args():
    ap = argparse.ArgumentParser("Human-filtered saliency from weighted masks (0/1/2), with fixed-size nearest resize.")
    ap.add_argument("--splits", nargs="+", default=["train","val"], help="train/val")
    ap.add_argument("--out-root", type=Path, default=Path("/home/ryan529/project/bdd100k/labels/sem_seg"),
                    help="輸出根目錄（會建立 binary_saliency_masks_human / weighted_saliency_masks_human）")
    ap.add_argument("--report-root", type=Path, default=Path("/home/ryan529/project/bdd100k/labels/sem_seg/reports"),
                    help="稽核報告輸出目錄")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--compress", type=int, default=3, help="PNG compress_level 0~9")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.width % 8 or args.height % 8:
        raise ValueError(f"width/height 必須是 8 的倍數，收到 {args.width}x{args.height}")

    total_kept = total_err = 0
    for sp in args.splits:
        kept, errs = _process_split(sp, args.out_root, args.report_root, args.width, args.height, args.workers, args.compress)
        total_kept += kept
        total_err  += errs
    print(f"[done] kept={total_kept}, errors={total_err}")

if __name__ == "__main__":
    main()
