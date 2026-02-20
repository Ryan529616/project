# -*- coding: utf-8 -*-
"""
transforms.py — RAW4 dict-style augments in NVIDIA DALI (CUDA 13.0-safe)

設計目標：
- 保留原本 transforms.py 的所有「概念功能」，只是換成：
  - 部分在 DALI/GPU 上跑
  - 部分在 python_function/NumPy 上跑（lens shading / stripe / cutout）
- 輸入：HWC, float32, [0,1] 的 RAW4（C=4）+ mask + human
- 輸出：CHW, float32
"""

from __future__ import annotations

from nvidia.dali import fn, types, pipeline_def
import nvidia.dali.math as dmath

import numpy as np


# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------

def _clamp01(x):
    """Clamp DALI tensor to [0, 1]."""
    return dmath.clamp(x, 0.0, 1.0)


# ---------------------------------------------------------
# CPU-side augments via python_function
#   - RandomLensShading
#   - RandomStripeNoise
#   - RandomCutout (含 affect_gt 行為)
# ---------------------------------------------------------

def build_cpu_sensor_aug(
    lens_p: float = 0.3,
    lens_strength=(0.1, 0.35),
    stripe_p: float = 0.3,
    stripe_strength: float = 0.03,
    stripe_axis: str = "row",  # "row" or "col"
    cutout_p: float = 0.5,
    cutout_holes: int = 2,
    cutout_max_frac: float = 0.2,
    cutout_affect_gt: str = "none",  # "none" | "mask" | "both"
):
    """
    回傳一個給 DALI python_function 用的 callable：
    只處理 image/mask/human 的
      - LensShading
      - StripeNoise
      - Cutout
    其它東西交給 GPU / DALI。
    image/mask/human 皆為 numpy array，image 為 HWC float32 [0,1]。
    """
    assert stripe_axis in ("row", "col")
    assert cutout_affect_gt in ("none", "mask", "both")

    lens_p = float(lens_p)
    stripe_p = float(stripe_p)
    cutout_p = float(cutout_p)
    lens_strength = tuple(lens_strength)
    cutout_max_frac = float(cutout_max_frac)
    stripe_strength = float(stripe_strength)

    def _aug(image, mask, human):
        img = image.astype(np.float32, copy=True)
        m = None if mask is None else mask.astype(np.float32, copy=True)
        h = None if human is None else human.astype(np.float32, copy=True)

        if img.ndim != 3:
            return img, m, h

        H, W, C = img.shape

        # ---------- RandomStripeNoise (CPU版, 原邏輯) ----------
        if stripe_strength > 0 and np.random.rand() < stripe_p and H > 0 and W > 0:
            if stripe_axis == "row":
                offsets = (np.random.rand(H).astype(np.float32) * 2.0 - 1.0) * stripe_strength
                img = img + offsets[:, None, None]
            else:
                offsets = (np.random.rand(W).astype(np.float32) * 2.0 - 1.0) * stripe_strength
                img = img + offsets[None, :, None]
            img = np.clip(img, 0.0, 1.0)

        # ---------- RandomLensShading ----------
        if np.random.rand() < lens_p and H > 1 and W > 1:
            k = np.random.uniform(lens_strength[0], lens_strength[1])
            yy, xx = np.meshgrid(
                np.linspace(-1.0, 1.0, H, dtype=np.float32),
                np.linspace(-1.0, 1.0, W, dtype=np.float32),
                indexing="ij",
            )
            rr = np.sqrt(xx * xx + yy * yy)
            rr /= (rr.max() + 1e-6)
            mask_v = 1.0 - k * (rr ** 2)
            mask_v = np.clip(mask_v, 0.0, 1.0)
            img = img * mask_v[..., None]
            img = np.clip(img, 0.0, 1.0)

        # ---------- RandomCutout ----------
        if (
            cutout_holes > 0
            and cutout_max_frac > 0.0
            and H > 1
            and W > 1
            and np.random.rand() < cutout_p
        ):
            for _ in range(int(cutout_holes)):
                rh = max(1, int(cutout_max_frac * H * np.random.rand()))
                rw = max(1, int(cutout_max_frac * W * np.random.rand())) if False else max(1, int(cutout_max_frac * W * np.random.rand()))  # typo 防呆
                cy = np.random.randint(0, H)
                cx = np.random.randint(0, W)
                y0 = max(0, cy - rh // 2)
                y1 = min(H, y0 + rh)
                x0 = max(0, cx - rw // 2)
                x1 = min(W, x0 + rw)
                img[y0:y1, x0:x1, :] = 0.0

                if cutout_affect_gt in ("mask", "both") and m is not None:
                    if m.ndim == 2:
                        m[y0:y1, x0:x1] = 0.0
                    elif m.ndim == 3:
                        m[y0:y1, x0:x1, :] = 0.0

                if cutout_affect_gt in ("both",) and h is not None:
                    if h.ndim == 2:
                        h[y0:y1, x0:x1] = 0.0
                    elif h.ndim == 3:
                        h[y0:y1, x0:x1, :] = 0.0

            img = np.clip(img, 0.0, 1.0)

        return img, m, h

    return _aug


# ---------------------------------------------------------
# GPU-side DALI augments
#   幾何 & 大部分強度域
# ---------------------------------------------------------

def dali_random_shift_scale_crop_bottom_biased(
    img, mask, human,
    out_h: int,
    out_w: int,
    scale_range=(0.9, 1.1),
    bottom_bias: float = 0.45,
):
    """
    DALI 版 RandomShiftScaleCropBottomBiased
    - 使用 normalized slice + resize
    - image/mask/human 同步裁切
    """
    # scale factor
    s = fn.random.uniform(range=scale_range)  # ~ (0.9, 1.1)

    # max(s, 1.0) 用 dmath.max（不要用 fn.maximum，舊版 DALI 沒有）
    s_eff = dmath.max(s, 1.0)

    roi_h = 1.0 / s_eff
    roi_w = 1.0 / s_eff

    # bottom-biased anchor (y)
    y_max = 1.0 - roi_h
    base = bottom_bias * y_max
    jitter = fn.random.uniform(range=(0.0, 1.0))
    pos_y = base + jitter * jitter * (y_max - base)

    # x uniform
    x_max = 1.0 - roi_w
    pos_x = fn.random.uniform(range=(0.0, 1.0)) * x_max

    anchor = fn.stack(pos_y, pos_x)   # [2] (y, x)
    shape = fn.stack(roi_h, roi_w)    # [2] (h, w)

    # slice on H/W using normalized coordinates, layout=HWC
    img_c = fn.slice(
        img,
        anchor, shape,
        axis_names="HW",
        normalized_anchor=True,
        normalized_shape=True,
    )
    mask_c = fn.slice(
        mask,
        anchor, shape,
        axis_names="HW",
        normalized_anchor=True,
        normalized_shape=True,
    )
    human_c = fn.slice(
        human,
        anchor, shape,
        axis_names="HW",
        normalized_anchor=True,
        normalized_shape=True,
    )

    img_c = fn.resize(img_c, resize_y=out_h, resize_x=out_w, interp_type=types.INTERP_LINEAR)
    mask_c = fn.resize(mask_c, resize_y=out_h, resize_x=out_w, interp_type=types.INTERP_NN)
    human_c = fn.resize(human_c, resize_y=out_h, resize_x=out_w, interp_type=types.INTERP_NN)

    return img_c, mask_c, human_c


def dali_random_hflip(img, mask, human, p: float = 0.5):
    """Random horizontal flip for image/mask/human (同步)."""
    coin = fn.random.coin_flip(probability=p)

    flipped_img = fn.flip(img, horizontal=1)
    flipped_mask = fn.flip(mask, horizontal=1)
    flipped_human = fn.flip(human, horizontal=1)

    img_out = coin * flipped_img + (1.0 - coin) * img
    mask_out = coin * flipped_mask + (1.0 - coin) * mask
    human_out = coin * flipped_human + (1.0 - coin) * human

    return img_out, mask_out, human_out


def dali_random_exposure_gamma(
    img,
    p: float = 0.7,
    gain_range=(0.8, 1.25),
    bias_range=(-0.05, 0.05),
    gamma_range=(0.9, 1.2),
):
    coin = fn.random.coin_flip(probability=p)
    gain = fn.random.uniform(range=gain_range)
    bias = fn.random.uniform(range=bias_range)
    gamma = fn.random.uniform(range=gamma_range)

    x = img * gain + bias
    x = dmath.clamp(x, 0.0, 1.0)
    x = dmath.pow(x, gamma)
    x = _clamp01(x)
    return coin * x + (1.0 - coin) * img


def dali_random_gaussian_noise(img, p: float = 0.6, sigma=(0.0, 0.015)):
    coin = fn.random.coin_flip(probability=p)
    stddev = fn.random.uniform(range=sigma)
    noisy = fn.noise.gaussian(img, stddev=stddev)
    noisy = _clamp01(noisy)
    return coin * noisy + (1.0 - coin) * img


def dali_random_poisson_shot_noise(img, p: float = 0.5, lam_range=(200.0, 1200.0)):
    """
    原本是 lam in [200,1200] 的 Poisson shot noise.
    DALI 沒有原生 Poisson，所以用 noise.shot + 1/lam 製造近似。
    """
    coin = fn.random.coin_flip(probability=p)
    lam_lo, lam_hi = lam_range
    factor_lo = 1.0 / float(lam_hi)
    factor_hi = 1.0 / float(lam_lo)
    factor = fn.random.uniform(range=(factor_lo, factor_hi))
    shot = fn.noise.shot(img, factor=factor)
    shot = _clamp01(shot)
    return coin * shot + (1.0 - coin) * img


def dali_random_black_level_shift(img, p: float = 0.5, shift_range=(-0.02, 0.02)):
    coin = fn.random.coin_flip(probability=p)
    delta = fn.random.uniform(range=shift_range)
    shifted = img + delta
    shifted = _clamp01(shifted)
    return coin * shifted + (1.0 - coin) * img


def dali_random_gaussian_blur(img, p: float = 0.4, sigma=(0.0, 1.2)):
    coin = fn.random.coin_flip(probability=p)
    sig = fn.random.uniform(range=sigma)
    blurred = fn.gaussian_blur(img, sigma=sig)
    blurred = _clamp01(blurred)
    return coin * blurred + (1.0 - coin) * img


def dali_random_cfa_jitter(
    img,
    p: float = 0.3,
    swap_g_prob: float = 0.5,
    swap_rb_prob: float = 0.3,
):
    """
    DALI 版 RandomCFAJitter
    - 假設 image layout = HWC，且 C=4，順序為 R,G1,G2,B
    - 使用 slice + cat 實作 channel 重新排列
    """
    coin_main = fn.random.coin_flip(probability=p)

    # 從 channel 維度切出 4 個通道 (axis=2, layout=HWC)
    c0 = fn.slice(img, start=(0,), shape=(1,), axes=[2])
    c1 = fn.slice(img, start=(1,), shape=(1,), axes=[2])
    c2 = fn.slice(img, start=(2,), shape=(1,), axes=[2])
    c3 = fn.slice(img, start=(3,), shape=(1,), axes=[2])

    img_id = fn.cat(c0, c1, c2, c3, axis=2)    # [0,1,2,3]
    img_G = fn.cat(c0, c2, c1, c3, axis=2)     # swap G1/G2
    img_RB = fn.cat(c3, c1, c2, c0, axis=2)    # swap R/B
    img_both = fn.cat(c3, c2, c1, c0, axis=2)  # G swap + RB swap

    g_coin = fn.random.coin_flip(probability=swap_g_prob)
    rb_coin = fn.random.coin_flip(probability=swap_rb_prob)

    one = 1.0
    g = g_coin
    rb = rb_coin

    out = (
        (one - g) * (one - rb) * img_id +
        g * (one - rb) * img_G +
        (one - g) * rb * img_RB +
        g * rb * img_both
    )

    out = coin_main * out + (one - coin_main) * img
    return out


def dali_random_quantize(img, p: float = 0.5, bits_range=(6, 10)):
    """
    RandomQuantize:
    - 隨機整數 bits in [bits_range[0], bits_range[1]]（含端點）
    - q = round(x * levels) / levels ；round 用 floor(x+0.5) 近似
    """
    coin = fn.random.coin_flip(probability=p)
    lo, hi = bits_range

    bits_f = fn.random.uniform(range=(float(lo), float(hi) + 1.0))
    bits_i = dmath.floor(bits_f)
    levels = dmath.pow(2.0, bits_i) - 1.0

    q = dmath.floor(img * levels + 0.5) / levels
    q = _clamp01(q)
    return coin * q + (1.0 - coin) * img


# ---------------------------------------------------------
# Pipelines
# ---------------------------------------------------------

@pipeline_def
def raw4_train_pipeline(
    source,
    in_h: int,
    in_w: int,
    # 幾何參數
    hflip_p: float = 0.5,
    scale_range=(0.9, 1.1),
    bottom_bias: float = 0.45,
    # 強度域
    exp_p: float = 0.7,
    gain_range=(0.8, 1.25),
    bias_range=(-0.05, 0.05),
    gamma_range=(0.9, 1.2),
    gauss_p: float = 0.6,
    gauss_sigma=(0.0, 0.015),
    shot_p: float = 0.5,
    shot_lambda=(200.0, 1200.0),
    blur_p: float = 0.4,
    blur_sigma=(0.0, 1.2),
    blackshift_p: float = 0.5,
    blackshift_range=(-0.02, 0.02),
    cfa_p: float = 0.3,
    cfa_swap_g_prob: float = 0.5,
    cfa_swap_rb_prob: float = 0.3,
    quant_p: float = 0.5,
    quant_bits=(6, 10),
    # CPU sensor aug (lens / stripe / cutout)
    lens_p: float = 0.3,
    lens_strength=(0.1, 0.35),
    stripe_p: float = 0.3,
    stripe_strength: float = 0.03,
    stripe_axis: str = "row",
    cutout_p: float = 0.5,
    cutout_holes: int = 2,
    cutout_max_frac: float = 0.2,
    cutout_affect_gt: str = "none",
):
    """
    DALI train pipeline:
    - external_source:
        source: Raw4DaliSource(ds, ...)
        會回傳 (image, mask, human)，皆為 HWC float32 [0,1]，且 C=4
    - 輸出：
        image, mask, human -> CHW float32
    """
    out_h = in_h
    out_w = in_w

    img, mask, human = fn.external_source(
        source=source,
        num_outputs=3,
        batch=False,
        layout="HWC",
    )

    cpu_aug = build_cpu_sensor_aug(
        lens_p=lens_p,
        lens_strength=lens_strength,
        stripe_p=stripe_p,
        stripe_strength=stripe_strength,
        stripe_axis=stripe_axis,
        cutout_p=cutout_p,
        cutout_holes=cutout_holes,
        cutout_max_frac=cutout_max_frac,
        cutout_affect_gt=cutout_affect_gt,
    )
    img, mask, human = fn.python_function(
        img, mask, human,
        function=cpu_aug,
        num_outputs=3
    )

    img = fn.cast(img, dtype=types.FLOAT)
    mask = fn.cast(mask, dtype=types.FLOAT)
    human = fn.cast(human, dtype=types.FLOAT)

    img = img.gpu()
    mask = mask.gpu()
    human = human.gpu()

    # [FIX] Removed random crop/scale to keep boxes aligned with CPU dataset
    # img, mask, human = dali_random_shift_scale_crop_bottom_biased(
    #     img, mask, human,
    #     out_h=out_h, out_w=out_w,
    #     scale_range=scale_range,
    #     bottom_bias=bottom_bias,
    # )
    img = fn.resize(img, resize_y=out_h, resize_x=out_w, interp_type=types.INTERP_LINEAR)
    mask = fn.resize(mask, resize_y=out_h, resize_x=out_w, interp_type=types.INTERP_NN)
    human = fn.resize(human, resize_y=out_h, resize_x=out_w, interp_type=types.INTERP_NN)

    # [FIX] Removed random hflip (handled by dataset.py for box sync)
    # img, mask, human = dali_random_hflip(img, mask, human, p=hflip_p)

    img = dali_random_exposure_gamma(
        img,
        p=exp_p,
        gain_range=gain_range,
        bias_range=bias_range,
        gamma_range=gamma_range,
    )
    img = dali_random_gaussian_noise(img, p=gauss_p, sigma=gauss_sigma)
    img = dali_random_poisson_shot_noise(img, p=shot_p, lam_range=shot_lambda)
    img = dali_random_black_level_shift(img, p=blackshift_p, shift_range=blackshift_range)
    img = dali_random_gaussian_blur(img, p=blur_p, sigma=blur_sigma)
    img = dali_random_cfa_jitter(
        img,
        p=cfa_p,
        swap_g_prob=cfa_swap_g_prob,
        swap_rb_prob=cfa_swap_rb_prob,
    )
    img = dali_random_quantize(img, p=quant_p, bits_range=quant_bits)

    # Already resized above, no need to resize again here unless we want to be super safe
    # img = fn.resize(img, resize_y=out_h, resize_x=out_w, interp_type=types.INTERP_LINEAR)
    # mask = fn.resize(mask, resize_y=out_h, resize_x=out_w, interp_type=types.INTERP_NN)
    # human = fn.resize(human, resize_y=out_h, resize_x=out_w, interp_type=types.INTERP_NN)

    img = fn.transpose(img, perm=[2, 0, 1])
    mask = fn.transpose(mask, perm=[2, 0, 1])
    human = fn.transpose(human, perm=[2, 0, 1])

    return img, mask, human


@pipeline_def
def raw4_val_pipeline(
    source,
    in_h: int,
    in_w: int,
):
    """
    驗證版：只做 resize + layout 轉換，不做任何隨機 aug。
    仍假設輸入為 HWC float32 [0,1] 的 RAW4 + mask + human。
    """
    out_h = in_h
    out_w = in_w

    img, mask, human = fn.external_source(
        source=source,
        num_outputs=3,
        batch=False,
        layout="HWC",
    )

    img = fn.cast(img, dtype=types.FLOAT)
    mask = fn.cast(mask, dtype=types.FLOAT)
    human = fn.cast(human, dtype=types.FLOAT)

    img = img.gpu()
    mask = mask.gpu()
    human = human.gpu()

    img = fn.resize(img, resize_y=out_h, resize_x=out_w, interp_type=types.INTERP_LINEAR)
    mask = fn.resize(mask, resize_y=out_h, resize_x=out_w, interp_type=types.INTERP_NN)
    human = fn.resize(human, resize_y=out_h, resize_x=out_w, interp_type=types.INTERP_NN)

    img = fn.transpose(img, perm=[2, 0, 1])
    mask = fn.transpose(mask, perm=[2, 0, 1])
    human = fn.transpose(human, perm=[2, 0, 1])

    return img, mask, human


__all__ = [
    "raw4_train_pipeline",
    "raw4_val_pipeline",
]
