#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pack_bayer_npy.py — 把 RAW16 PNG(存RAW10) 或 MIPI RAW10 批量轉成 4ch half 尺寸 .npy（獨立版，無白平衡）
================================================================================================
- 讀 raw_generator.py 產出的單通道 RAW（PNG/TIFF/PGM）或 MIPI RAW10（.raw10/.raw/.bin）
- 如有 sidecar .json，僅採用 {pattern, black_level, white_level}，完全忽略 wb（因為 RAW 不該有 WB）
- Bayer：內建 CFA-aware 拆分 + 正規化（去黑位/除白位），不做 WB
- MONO：把 2×2 stride 切成 4 平面以維持下游 (4, H/2, W/2) 介面一致
- 預設輸出 float32，避免半精度量化吃光暗通道
- 列印每檔通道 min/max/mean 與飽和率，提早發現爆框與全 1.0 慘案

注意：
- 這版把 CLI 的 --white 當「sensor white_level」（例如 1023）。實際正規化使用 white_total = black + white_level。
"""

from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from typing import Sequence, Tuple, List
from functools import partial

import numpy as np
import imageio.v3 as iio

RAW10_EXTS = {".raw10", ".raw", ".bin"}
IMG_EXTS   = {".png", ".tif", ".tiff", ".pgm"}


# ------------------------------------------------------------
# 內建：Bayer -> 4ch half，不做白平衡
# ------------------------------------------------------------

_BAYER_OFFSETS = {
    "RGGB": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "GRBG": ((0, 1), (0, 0), (1, 1), (1, 0)),
    "GBRG": ((1, 0), (1, 1), (0, 0), (0, 1)),
    "BGGR": ((1, 1), (1, 0), (0, 1), (0, 0)),
}

def _normalize01(x: np.ndarray, black: float, white_total: float) -> np.ndarray:
    wl = max(float(white_total) - float(black), 1.0)
    return np.clip((x.astype(np.float32) - float(black)) / wl, 0.0, 1.0)

def bayer_to_4ch_half(
    bayer: np.ndarray,
    pattern: str = "RGGB",
    *,
    normalize: bool = True,
    black_level: float = 64.0,
    white_total: float = 1087.0
) -> np.ndarray:
    """
    numpy 單通道 Bayer → 四平面 (4, H/2, W/2)，可選正規化（無白平衡）。
    pattern 支援：RGGB / BGGR / GRBG / GBRG / MONO
    """
    x = np.asarray(bayer)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"expect (H,W) gray, got {x.shape}")

    H, W = x.shape
    H2, W2 = (H // 2), (W // 2)
    x = x[: H2 * 2, : W2 * 2].astype(np.float32, copy=False)

    if normalize:
        x = _normalize01(x, black=black_level, white_total=white_total)

    pat = pattern.upper()
    if pat == "MONO":
        R  = x[0::2, 0::2]
        G1 = x[0::2, 1::2]
        G2 = x[1::2, 0::2]
        B  = x[1::2, 1::2]
    else:
        if pat not in _BAYER_OFFSETS:
            raise ValueError(f"Unsupported Bayer pattern: {pattern}")
        oR, oG1, oG2, oB = _BAYER_OFFSETS[pat]
        R  = x[oR[0] :: 2,  oR[1] :: 2]
        G1 = x[oG1[0] :: 2, oG1[1] :: 2]
        G2 = x[oG2[0] :: 2, oG2[1] :: 2]
        B  = x[oB[0] :: 2,  oB[1] :: 2]

    ch4 = np.stack([R, G1, G2, B], axis=0).astype(np.float32, copy=False)
    return ch4  # (4, H/2, W/2)


# ------------------------------------------------------------
# 低階 I/O
# ------------------------------------------------------------

def unpack_mipi_raw10(buf: bytes, width: int) -> np.ndarray:
    """解包 MIPI RAW10：每 5 bytes → 4 pixels (10bit)。輸出 uint16，值域 0..1023。"""
    if width is None:
        raise ValueError("Reading RAW10 requires --width")
    w4 = (width + 3) // 4
    row_bytes = w4 * 5
    if len(buf) % row_bytes != 0:
        raise ValueError(f"RAW10 length {len(buf)} not divisible by {row_bytes}; wrong --width?")
    height = len(buf) // row_bytes
    out = np.empty((height, width), dtype=np.uint16)
    mv = memoryview(buf)
    for y in range(height):
        r = np.frombuffer(mv[y*row_bytes:(y+1)*row_bytes], dtype=np.uint8).reshape(-1, 5)
        msb0, msb1, msb2, msb3, lsb = [r[:, i].astype(np.uint16) for i in range(5)]
        p0 = (msb0 << 2) | ((lsb >> 0) & 0x3)
        p1 = (msb1 << 2) | ((lsb >> 2) & 0x3)
        p2 = (msb2 << 2) | ((lsb >> 4) & 0x3)
        p3 = (msb3 << 2) | ((lsb >> 6) & 0x3)
        row_vals = np.empty(w4 * 4, dtype=np.uint16)
        row_vals[0::4], row_vals[1::4], row_vals[2::4], row_vals[3::4] = p0, p1, p2, p3
        out[y, :] = row_vals[:width]
    return out

def read_bayer_or_raw10(p: Path, width: int | None) -> np.ndarray:
    """讀取單通道 Bayer (PNG/TIFF/PGM) 或 MIPI RAW10。回傳 uint16。"""
    ext = p.suffix.lower()
    if ext in RAW10_EXTS:
        if width is None:
            raise ValueError("Reading RAW10 requires --width")
        with open(p, "rb") as f:
            return unpack_mipi_raw10(f.read(), width)
    elif ext in IMG_EXTS:
        img = iio.imread(p)
        if img.ndim == 3:
            img = img[..., 0]
        if img.ndim != 2:
            raise ValueError(f"{p} not single-channel, got {img.shape}")
        return img.astype(np.uint16, copy=False)
    else:
        raise ValueError(f"Unsupported ext: {ext}")

def mono_to_4ch_half(x01: np.ndarray) -> np.ndarray:
    """把 MONO (H,W) float[0,1] 切成 4 個 2x2 stride 平面 → (4,h2,w2)。"""
    H, W = x01.shape
    H2, W2 = H // 2, W // 2
    x = x01[:H2 * 2, :W2 * 2]
    R  = x[0::2, 0::2]
    G1 = x[0::2, 1::2]
    G2 = x[1::2, 0::2]
    B  = x[1::2, 1::2]
    return np.stack([R, G1, G2, B], axis=0).astype(np.float32, copy=False)

def scan(root: Path, exts: Sequence[str]) -> list[Path]:
    S = {e.lower() for e in exts}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in S]


# ------------------------------------------------------------
# 診斷與統計
# ------------------------------------------------------------

def stats_line(arr: np.ndarray) -> str:
    chs: List[str] = []
    for i in range(arr.shape[0]):
        a = arr[i]
        ch_min = float(a.min()); ch_max = float(a.max()); ch_mean = float(a.mean())
        sat0 = float((a <= 0.0).mean())
        sat1 = float((a >= 1.0).mean())
        chs.append(f"ch{i}: min={ch_min:.4f} max={ch_max:.4f} mean={ch_mean:.4f} sat0={sat0*100:.2f}% sat1={sat1*100:.2f}%")
    return " | ".join(chs)

def warn_if_weird(arr: np.ndarray, tag: str, verbose: bool = True) -> None:
    if not verbose:
        return
    if not np.isfinite(arr).all():
        print(f"[WARN:{tag}] detected NaN/Inf")
    for i in range(arr.shape[0]):
        a = arr[i]
        if float(a.max()) - float(a.min()) < 1e-6:
            print(f"[WARN:{tag}] channel {i} almost constant: {float(a.min()):.6f}")
        sat1 = float((a >= 1.0).mean())
        if sat1 > 0.20:
            print(f"[WARN:{tag}] channel {i} high saturation at 1.0 → {sat1*100:.2f}%（可能曝光或白點界限偏置）")


# ------------------------------------------------------------
# 單檔處理（支援 multiprocessing；保證回傳 tuple）
# ------------------------------------------------------------

def process_one(p: Path, args, src: Path, dst: Path) -> tuple[str, Path]:
    try:
        rel = p.relative_to(src)
        outp = (dst / rel).with_suffix(".npy")
        outp.parent.mkdir(parents=True, exist_ok=True)

        # 讀 sidecar meta（只取 pattern/black/white；忽略 wb）
        meta = None
        meta_path = p.with_suffix(".json")
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = None

        pattern = (meta.get("pattern") if meta else None) or args.pattern
        black   = float((meta.get("black_level") if meta else None) or args.black)

        # meta.white_level 為 sensor white；正規化使用 white_total = black + white_level
        if meta and ("white_level" in meta):
            white_total = float(black + float(meta["white_level"]))
        else:
            # CLI 的 --white 視為 sensor white_level，與 black 相加
            white_total = float(args.black + args.white)

        if (not args.overwrite) and outp.exists() and outp.stat().st_mtime >= p.stat().st_mtime:
            return "skip", outp

        raw = read_bayer_or_raw10(p, width=args.width)  # uint16

        if pattern == "MONO":
            x01 = _normalize01(raw, black=black, white_total=white_total)
            arr = mono_to_4ch_half(x01)  # (4,h2,w2)
        else:
            arr = bayer_to_4ch_half(
                raw, pattern=pattern, normalize=True,
                black_level=black, white_total=white_total
            )
            arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)

        if args.verbose:
            print(f"[{p.name}] {stats_line(arr)}")

        # dtype
        if args.store == "float16":
            warn_if_weird(arr, "pre-save", verbose=True)
            print("[WARN] saving dataset as float16. 半精度可能吃掉暗區細節，請自行承擔。")
            arr = arr.astype(np.float16, copy=False)
        else:
            arr = arr.astype(np.float32, copy=False)

        np.save(outp, arr, allow_pickle=False)
        return "ok", outp

    except Exception as e:
        # 把任何 worker 內部錯誤回報給主進程，不讓它回 None
        return f"err:{type(e).__name__}:{e}", (dst / p.name).with_suffix(".npy")


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("Pack RAW to 4ch half .npy (no white balance, standalone)")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--pattern", type=str, default="RGGB",
                    help="預設 Bayer pattern（無 meta 時使用；'MONO' 代表灰階源）")
    ap.add_argument("--black", type=float, default=64.0,
                    help="black_level（無 meta 時使用）")
    ap.add_argument("--white", type=float, default=1023.0,
                    help="sensor white_level（不含黑位；無 meta 時使用）。實際正規化用 (black + white)。")
    ap.add_argument("--width", type=int, default=None,
                    help="讀 MIPI RAW10 需要指定寬度")
    ap.add_argument("--store", type=str, default="float32", choices=["float16","float32"],
                    help="輸出 dtype（皆已正規化到 [0,1]；預設 float32）")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--exts", type=str, nargs="+",
                    default=[".png",".tif",".tiff",".pgm",".raw10",".raw",".bin"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--verbose", action="store_true", help="列印每檔通道統計與飽和率")
    args = ap.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    files = scan(src, args.exts)
    if not files:
        print(f"[warn] no files under {src}")
        return 0

    white_total_cli = args.black + args.white
    print(f"[info] src={src}\n[info] dst={dst}\n[info] n={len(files)}"
          f"\n[info] store={args.store}\n[info] pattern_default={args.pattern}"
          f"\n[info] black={args.black} white_level={args.white} white_total={white_total_cli}")

    worker = partial(process_one, args=args, src=src, dst=dst)

    if args.workers > 0:
        from multiprocessing import Pool
        with Pool(processes=args.workers) as pool:
            for status, outp in pool.imap_unordered(worker, files, chunksize=8):
                print(f"[{status.upper():4}] {outp}")
    else:
        # 單執行緒路線也統一印同樣格式
        for p in files:
            status, outp = worker(p)
            print(f"[{status.upper():4}] {outp}")

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
