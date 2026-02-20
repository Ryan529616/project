#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raw_generator.py — Synthetic RAW10 generator (RGB -> RGGB, inverse-WB only)
============================================================================
固定行為（不給你選）：
- sRGB → linear
- sensor_ccm（可選 3x3）把 linear RGB 投到感測器原色域
- AE 一律在 pre-WB 空間（避免被 WB 扭曲統計）
- 反白平衡：直接 / (wb_r, wb_g, wb_b)
- Poisson shot + Gaussian read noise
- PRNU/DSNU
- hot pixels 在 white-referenced 域注入
- 加黑位，打包 RGGB，輸出單通道 uint16 PNG（值域：black..black+white）

你可以調：wb 值、AE 目標、noise、PRNU/DSNU、hot pixel、黑白位、ccm、並行相關參數。
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Iterable, List

# 抑制每個進程內 BLAS/OMP 的過度多線程
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    cv2.setNumThreads(0)
except Exception:
    pass

# ---------------- color helpers ----------------
_A = 0.055

def srgb_to_linear_rgb(rgb: np.ndarray) -> np.ndarray:
    """rgb float32 [0,1], shape HxWx3."""
    out = np.empty_like(rgb, dtype=np.float32)
    mask = rgb <= 0.04045
    out[mask]  = rgb[mask] / 12.92
    out[~mask] = ((rgb[~mask] + _A) / (1 + _A)) ** 2.4
    return out

def apply_matrix3x3(rgb_lin: np.ndarray, m: Optional[np.ndarray]) -> np.ndarray:
    """對 HxWx3 的 linear RGB 右乘 3x3 矩陣。"""
    if m is None:
        return rgb_lin
    H, W, _ = rgb_lin.shape
    x = rgb_lin.reshape(-1, 3)
    y = (x @ m.T).reshape(H, W, 3)
    return np.clip(y, 0.0, 1.0)

# ---------------- sensor-ish noise ----------------
def add_prnu_dsnu(signal: np.ndarray, prnu_std: float, dsnu_std: float, rng: np.random.Generator) -> np.ndarray:
    """Apply multiplicative PRNU and additive DSNU in signal units (pre-quant)."""
    if prnu_std > 0:
        prnu = rng.normal(0.0, prnu_std, size=signal.shape).astype(np.float32)
        signal = signal * (1.0 + prnu)
    if dsnu_std > 0:
        dsnu = rng.normal(0.0, dsnu_std, size=signal.shape).astype(np.float32)
        signal = signal + dsnu
    return signal

def add_hot_pixels(raw_wo_bl: np.ndarray, prob: float, extra: float,
                   rng: np.random.Generator, white_level: int) -> np.ndarray:
    """Operate in white-referenced space (i.e., after removing black)."""
    if prob <= 0:
        return raw_wo_bl
    H, W = raw_wo_bl.shape
    n = int(H * W * prob)
    if n <= 0:
        return raw_wo_bl
    ys = rng.integers(0, H, size=n)
    xs = rng.integers(0, W, size=n)
    raw_wo_bl[ys, xs] = np.clip(
        raw_wo_bl[ys, xs].astype(np.int32) + int(round(extra)), 0, white_level
    ).astype(np.uint16)
    return raw_wo_bl

# ---------------- RGGB packing ----------------
# positions for (R, G1, G2, B) under RGGB
_RGGG_POS = ((0, 0), (0, 1), (1, 0), (1, 1))

def pack_rggb_from_rgb10(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """r/g/b uint16, same HxW; return single-channel uint16 RGGB mosaic of same HxW."""
    H, W = r.shape
    out = np.empty((H, W), dtype=np.uint16)
    (rpos, g1pos, g2pos, bpos) = _RGGG_POS
    out[rpos[0]::2,  rpos[1]::2]  = r[rpos[0]::2,  rpos[1]::2]
    out[g1pos[0]::2, g1pos[1]::2] = g[g1pos[0]::2, g1pos[1]::2]
    out[g2pos[0]::2, g2pos[1]::2] = g[g2pos[0]::2, g2pos[1]::2]
    out[bpos[0]::2,  bpos[1]::2]  = b[bpos[0]::2,  bpos[1]::2]
    return out

# ---------------- exposure ----------------
def auto_exposure_scale_rgb(rgb_lin: np.ndarray, target_pct: float, target_level: float) -> float:
    """scale s so percentile(target_pct) of max(R,G,B) => target_level."""
    mx = rgb_lin.max(axis=2)
    thr = float(np.percentile(mx, target_pct))
    if thr < 1e-8:
        return 1.0
    return float(target_level / thr)

# ---------------- RAW synthesis ----------------
def linrgb_to_raw10(rgb_lin: np.ndarray, gain: float, black_level: int, white_level: int,
                    read_noise: float, prnu_std: float, dsnu_std: float,
                    rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-channel uint16 raw planes (R,G,B) in 0..white_level; add black later."""
    H, W, _ = rgb_lin.shape
    sig = rgb_lin * (gain * white_level)  # pre-black signal in ADU

    # shot noise (Poisson)
    r = rng.poisson(sig[..., 0]).astype(np.float32)
    g = rng.poisson(sig[..., 1]).astype(np.float32)
    b = rng.poisson(sig[..., 2]).astype(np.float32)

    # read noise (Gaussian)
    if read_noise > 0:
        rn = float(read_noise)
        r += rng.normal(0.0, rn, size=(H, W)).astype(np.float32)
        g += rng.normal(0.0, rn, size=(H, W)).astype(np.float32)
        b += rng.normal(0.0, rn, size=(H, W)).astype(np.float32)

    # PRNU/DSNU
    r = add_prnu_dsnu(r, prnu_std, dsnu_std, rng)
    g = add_prnu_dsnu(g, prnu_std, dsnu_std, rng)
    b = add_prnu_dsnu(b, prnu_std, dsnu_std, rng)

    # quantize/clip
    r = np.clip(np.round(r), 0, white_level).astype(np.uint16)
    g = np.clip(np.round(g), 0, white_level).astype(np.uint16)
    b = np.clip(np.round(b), 0, white_level).astype(np.uint16)

    # add black
    if black_level != 0:
        maxv = int(white_level + black_level)
        r = np.clip(r.astype(np.int32) + black_level, 0, maxv).astype(np.uint16)
        g = np.clip(g.astype(np.int32) + black_level, 0, maxv).astype(np.uint16)
        b = np.clip(b.astype(np.int32) + black_level, 0, maxv).astype(np.uint16)
    return r, g, b

def deterministic_rng(seed: int, rel_path: Path) -> np.random.Generator:
    """Stable per-file seed = seed ^ blake2b(path)."""
    h = hashlib.blake2b(str(rel_path).encode('utf-8'), digest_size=8)
    val = int.from_bytes(h.digest(), 'little')
    return np.random.default_rng(seed ^ val)

# ---------------- diagnostics ----------------
def saturation_stats(raw_with_black: np.ndarray, black_level: int, white_level: int) -> Tuple[float, float]:
    total = float(raw_with_black.size)
    if total <= 0:
        return 0.0, 0.0
    sat0 = float((raw_with_black <= black_level).sum()) / total * 100.0
    sat1 = float((raw_with_black >= (white_level + black_level)).sum()) / total * 100.0
    return sat0, sat1

# ---------------- per-file pipeline ----------------
def process_one(rgb_path: Path, in_root: Path, out_root: Path, cfg) -> Tuple[bool, Optional[Path], Optional[str]]:
    """單檔處理。回傳 (ok, out_png_path, err_msg)。"""
    try:
        rel = rgb_path.relative_to(in_root)
        rng = deterministic_rng(cfg.seed, rel)

        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return False, None, f"Cannot read {rgb_path}"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        H, W, _ = rgb.shape

        # 1) sRGB -> linear
        lin = srgb_to_linear_rgb(rgb)

        # 2) optional sensor CCM（linear RGB → sensor domain）
        sensor_ccm = None
        if cfg.sensor_ccm is not None:
            vals = [v for v in cfg.sensor_ccm.split(',') if v.strip() != '']
            if len(vals) == 9:
                sensor_ccm = np.array([float(x) for x in vals], dtype=np.float32).reshape(3, 3)
        lin = apply_matrix3x3(lin, sensor_ccm)

        # 3) AE 在 pre-WB 空間
        scale = auto_exposure_scale_rgb(lin, cfg.target_pct, cfg.target_level)
        jitter = 1.0 + rng.uniform(-cfg.gain_jitter, cfg.gain_jitter)

        # 4) 反白平衡：除以 (wr,wg,wb)
        wb = np.array([cfg.wb_r, cfg.wb_g, cfg.wb_b], dtype=np.float32).reshape(1, 1, 3)
        lin = np.clip(lin / wb * scale * jitter, 0.0, 1.0)

        # 5) 合成 RAW 並打包 RGGB
        r, g, b = linrgb_to_raw10(
            lin,
            gain=1.0,
            black_level=cfg.black_level,
            white_level=cfg.white_level,
            read_noise=cfg.read_noise,
            prnu_std=cfg.prnu,
            dsnu_std=cfg.dsnu,
            rng=rng,
        )
        raw = pack_rggb_from_rgb10(r=r, g=g, b=b)

        # 6) hot pixels（white-referenced 域）
        raw_wo_bl = np.clip(raw.astype(np.int32) - cfg.black_level, 0, cfg.white_level).astype(np.uint16)
        raw_wo_bl = add_hot_pixels(raw_wo_bl, prob=cfg.hot_prob, extra=cfg.hot_extra, rng=rng, white_level=cfg.white_level)
        raw = np.clip(raw_wo_bl.astype(np.int32) + cfg.black_level, 0, cfg.white_level + cfg.black_level).astype(np.uint16)

        # 7) 統計與輸出
        sat0, sat1 = saturation_stats(raw, cfg.black_level, cfg.white_level)

        out_png = (out_root / rel).with_suffix('.png')
        out_png.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_png), raw):
            return False, None, f"Failed to write {out_png}"

        if cfg.write_meta:
            meta = {
                "src": str(rel).replace('\\', '/'),
                "H": int(H), "W": int(W),
                "pattern": "RGGB",
                "black_level": int(cfg.black_level),
                "white_level": int(cfg.white_level),
                "read_noise": float(cfg.read_noise),
                "prnu": float(cfg.prnu),
                "dsnu": float(cfg.dsnu),
                "hot_prob": float(cfg.hot_prob),
                "hot_extra": float(cfg.hot_extra),
                "wb": [float(cfg.wb_r), float(cfg.wb_g), float(cfg.wb_b)],
                "sensor_ccm": [float(x) for x in cfg.sensor_ccm.split(',')] if cfg.sensor_ccm else None,
                "ae_target_pct": float(cfg.target_pct),
                "ae_target_level": float(cfg.target_level),
                "ae_scale": float(scale),
                "gain_jitter": float(cfg.gain_jitter),
                "seed": int(cfg.seed),
                "sat0_pct": float(sat0),
                "sat1_pct": float(sat1),
                "note": "AE in pre-WB space; inverse white balance applied (lin / wb).",
            }
            meta_path = out_png.with_suffix('.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        if getattr(cfg, "verbose", False):
            print(f"[+] {rel} → {out_png.relative_to(out_root)} | {H}x{W} | ae={scale:.3f} | sat0={sat0:.2f}% sat1={sat1:.2f}%")
        return True, out_png, None

    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser("RGB->RGGB RAW10 generator — inverse-WB only (parallel)")

    # IO
    p.add_argument('--in_root',  required=True, help='Input RGB folder (jpg/png/jpeg/...)')
    p.add_argument('--out_root', required=True, help='Output root for RAW16 PNGs (RAW10 values + black)')
    p.add_argument('--exts',   default='.jpg,.jpeg,.png', help='comma-separated extensions')
    p.add_argument('--seed',   type=int, default=1234)
    p.add_argument('--write_meta', action='store_true')
    p.add_argument('--verbose',    action='store_true')

    # sensor / CCM
    p.add_argument('--sensor_ccm', type=str, default="1.12,-0.08,-0.04,-0.06,1.06,0.00,-0.03,-0.04,1.07",
                   help='3x3 row-major, comma-separated. 映射 linear RGB -> sensor RGB（預設輕微偏移）')

    # AE（固定 pre-WB）
    p.add_argument('--target_pct',   type=float, default=98.5)
    p.add_argument('--target_level', type=float, default=0.75)
    p.add_argument('--gain_jitter',  type=float, default=0.05)

    # 固定反白平衡的 WB 值
    p.add_argument('--wb_r', type=float, default=2.0)
    p.add_argument('--wb_g', type=float, default=1.0)
    p.add_argument('--wb_b', type=float, default=1.7)

    # sensor limits / noise
    p.add_argument('--black_level',  type=int,   default=64)
    p.add_argument('--white_level',  type=int,   default=1023)
    p.add_argument('--read_noise',   type=float, default=2.0)
    p.add_argument('--prnu',         type=float, default=0.02)
    p.add_argument('--dsnu',         type=float, default=0.5)
    p.add_argument('--hot_prob',     type=float, default=1e-5)
    p.add_argument('--hot_extra',    type=float, default=80.0)

    # parallel
    p.add_argument('-j', '--workers', type=int, default=min(16, max(1, os.cpu_count() or 1)))
    p.add_argument('--chunksize', type=int, default=8)
    p.add_argument('--progress', type=int, default=1)
    return p.parse_args()

def _iter_images(in_root: Path, exts: Iterable[str]) -> List[Path]:
    return [p for p in in_root.rglob('*') if p.suffix.lower() in exts]

def main():
    cfg = parse_args()
    in_root  = Path(cfg.in_root)
    out_root = Path(cfg.out_root)
    if not in_root.exists():
        raise FileNotFoundError(f"in_root not found: {in_root}")
    exts = tuple(e if e.startswith('.') else '.'+e for e in cfg.exts.lower().split(','))
    imgs = _iter_images(in_root, exts)
    if not imgs:
        raise RuntimeError('No input images found under in_root with given extensions')

    total = len(imgs)
    use_mp = int(cfg.workers) > 1

    if not use_mp:
        kept = 0
        iterator = imgs
        if cfg.progress and tqdm is not None:
            iterator = tqdm(iterator, total=total, dynamic_ncols=True)
        for pth in iterator:
            ok, _, err = process_one(pth, in_root, out_root, cfg)
            if ok:
                kept += 1
            elif cfg.verbose and err:
                print(f"[!] {err}")
        print(f"Done. Generated {kept}/{total} RAW files → {out_root}")
        return

    from concurrent.futures import ProcessPoolExecutor
    kept = 0
    errors = 0

    tasks = [(p, in_root, out_root, cfg) for p in imgs]
    with ProcessPoolExecutor(max_workers=int(cfg.workers)) as ex:
        futures = [ex.submit(process_one, *args) for args in tasks]
        iterator = futures
        if cfg.progress and tqdm is not None:
            iterator = tqdm(iterator, total=total, dynamic_ncols=True)
        for fut in iterator:
            try:
                ok, _, err = fut.result()
                if ok:
                    kept += 1
                else:
                    errors += 1
                    if cfg.verbose and err:
                        print(f"[!] {err}")
            except Exception as e:
                errors += 1
                if cfg.verbose:
                    print(f"[!] worker exception: {type(e).__name__}: {e}")

    print(f"Done. Generated {kept}/{total} RAW files (fail={errors}) → {out_root}")

if __name__ == '__main__':
    main()
