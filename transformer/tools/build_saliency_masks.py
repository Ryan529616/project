#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_saliency_masks.py — BDD100K trainId → saliency masks（硬寫 RGB 路徑）
- 產生兩種輸出：
    1) binary_saliency_masks: 0/1
    2) weighted_saliency_masks: 0/1/2（2=person/rider）
- 固定把輸出「最近鄰」縮到 --width x --height（預設 640x384，皆為 32 倍數）
- 多進程處理；輸出每個 split 的稽核報告 CSV（含 unknown ids、尺寸比對等）
- RGB 路徑寫死為：/home/ryan529/project/bdd100k/images/10k/{train,val}

用法範例：
python /home/ryan529/project/bdd100k/build_saliency_masks.py \
  --seg-root /home/ryan529/project/bdd100k/labels/sem_seg/masks \
  --out-root /home/ryan529/project/bdd100k/labels/sem_seg \
  --report-root /home/ryan529/project/bdd100k/labels/sem_seg/reports \
  --splits train val \
  --width 640 --height 384 \
  --workers 8 --compress 3
"""

from __future__ import annotations
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# =========================
# 硬寫的 RGB 影像根目錄
# =========================
RGB_BASE = Path("/home/ryan529/project/bdd100k/images/10k")

# =========================
# 類別定義（trainId）
# =========================
GENERAL_SALIENCY_TRAINIDS = {6, 7, 13, 14, 15, 17, 18}   # 不含 person/rider
PERSON_TRAINIDS = {11, 12}                                # person, rider
VALID_IDS = GENERAL_SALIENCY_TRAINIDS | PERSON_TRAINIDS | {0}  # 保留背景 0

# =========================
# 低階 I/O
# =========================
def load_label_png(fp: Path) -> np.ndarray:
    """讀取語意分割標註（保持 indexed，不做 'L' 轉換），回傳 HxW。"""
    im = Image.open(fp)
    arr = np.array(im)
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr

def save_png(arr: np.ndarray, path: Path, compress_level: int = 3) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path, compress_level=compress_level)

# =========================
# 邏輯
# =========================
def generate_masks(seg_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """trainId → (binary 0/1, weighted 0/1/2)"""
    weighted = np.zeros_like(seg_mask, dtype=np.uint8)
    if seg_mask.size == 0:
        return weighted.copy(), weighted

    # 逐類別指派（省記憶體）
    for cls in GENERAL_SALIENCY_TRAINIDS:
        weighted[seg_mask == cls] = 1
    for cls in PERSON_TRAINIDS:
        weighted[seg_mask == cls] = 2

    binary = (weighted > 0).astype(np.uint8)
    return binary, weighted

def resize_nearest(mask: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """最近鄰縮放（保持 mask 的離散性）"""
    return np.array(Image.fromarray(mask, mode="L").resize((out_w, out_h), resample=Image.NEAREST), dtype=np.uint8)

def quick_stats(a: np.ndarray) -> dict[int, int]:
    vals, cnts = np.unique(a, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}

def _find_rgb_size(split: str, stem: str) -> tuple[int | None, int | None]:
    """用硬寫的 RGB_BASE 去找原圖尺寸；找不到就回 None。"""
    for ext in ("jpg", "png", "jpeg", "JPG", "PNG", "JPEG"):
        p = RGB_BASE / split / f"{stem}.{ext}"
        if p.exists():
            with Image.open(p) as im:
                w, h = im.size
            return w, h
    return None, None

def _process_one(seg_file: Path, split: str, out_bin: Path, out_wgt: Path,
                 out_w: int, out_h: int, compress: int):
    """子行程：處理單一檔案，回傳 (row 或 None, error 或 None)。"""
    try:
        seg_mask = load_label_png(seg_file)
        stem = seg_file.stem

        # 值域檢查
        vals = set(np.unique(seg_mask).tolist())
        unknown = sorted(list(vals - VALID_IDS))
        suspicious = len(unknown) > 0

        # 尺寸稽核（若能找到原圖）
        rgb_w, rgb_h = _find_rgb_size(split, stem)
        if rgb_w is not None and (seg_mask.shape[1] != rgb_w or seg_mask.shape[0] != rgb_h):
            suspicious = True

        # 產生 saliency
        bin_mask, wgt_mask = generate_masks(seg_mask)

        # 尺寸固定到 out_w/out_h（nearest）
        bin_mask_resized = resize_nearest(bin_mask, out_w, out_h)
        wgt_mask_resized = resize_nearest(wgt_mask, out_w, out_h)

        # 輸出
        out_bin_path = out_bin / f"{stem}.png"
        out_wgt_path = out_wgt / f"{stem}.png"
        save_png(bin_mask_resized, out_bin_path, compress_level=compress)
        save_png(wgt_mask_resized, out_wgt_path, compress_level=compress)

        stat = quick_stats(seg_mask)
        row = {
            "file": f"{stem}.png",
            "h": seg_mask.shape[0], "w": seg_mask.shape[1],
            "rgb_w": rgb_w, "rgb_h": rgb_h,
            "unique_count": len(vals),
            "values": "|".join(map(str, sorted(vals))),
            "unknown_values": "|".join(map(str, unknown)) if unknown else "",
            "suspicious": int(suspicious),
            "bg_px": stat.get(0, 0),
            "sal_px": int((wgt_mask_resized > 0).sum()),
            "person_px": int((wgt_mask_resized == 2).sum()),
            "out_w": out_w, "out_h": out_h,
        }
        return row, None
    except Exception as e:
        return None, f"{seg_file}: {type(e).__name__}: {e}"

def process_split(split: str, seg_root: Path, out_root: Path, report_root: Path,
                  out_w: int, out_h: int, workers: int, compress: int):
    in_dir = seg_root / split
    out_bin = out_root / "binary_saliency_masks" / split
    out_wgt = out_root / "weighted_saliency_masks" / split
    out_bin.mkdir(parents=True, exist_ok=True)
    out_wgt.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.png"))
    rows, errors = [], []

    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [
            ex.submit(_process_one, f, split, out_bin, out_wgt, out_w, out_h, compress)
            for f in files
        ]
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=f"[{split}] build+resize {out_w}x{out_h}"):
            row, err = fut.result()
            if err:
                errors.append(err)
            elif row:
                rows.append(row)

    # 報告
    rpt_csv = report_root / f"{split}_label_audit.csv"
    fields = ["file","h","w","rgb_w","rgb_h","unique_count","values",
              "unknown_values","suspicious","bg_px","sal_px","person_px","out_w","out_h"]
    with open(rpt_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if errors:
        err_txt = report_root / f"{split}_errors.txt"
        with open(err_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(errors))

    return len(rows), len(errors)

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser("Build saliency masks (binary/weighted) and resize to fixed size. RGB path is hardcoded.")
    ap.add_argument("--seg-root", required=True, type=Path, help="原始 trainId 標註資料夾根目錄（含 train/val 子資料夾）")
    ap.add_argument("--out-root", required=True, type=Path, help="輸出根目錄（會在底下建立 binary/weighted 資料夾）")
    ap.add_argument("--report-root", required=True, type=Path, help="輸出 CSV 報告位置")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], help="要處理的 split 列表（train/val）")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--compress", type=int, default=3, help="PNG compress_level (0~9)")
    return ap.parse_args()

def main():
    args = parse_args()

    if args.width % 8 or args.height % 8:
        raise ValueError(f"width/height 必須為 8 倍數，收到 {args.width}x{args.height}")

    total_ok = total_err = 0
    for sp in args.splits:
        ok, err = process_split(
            sp, args.seg_root, args.out_root, args.report_root,
            args.width, args.height, args.workers, args.compress
        )
        total_ok += ok; total_err += err
    print(f"[done] rows={total_ok}, errors={total_err}")

if __name__ == "__main__":
    main()
