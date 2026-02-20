#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rggb_to_gray.py — 將 10-bit RGGB 圖轉換為人眼可見灰階圖

說明：
- 自動偵測單通道 Bayer 圖（RGGB pattern）
- 對應 RGGB 結構：
      R G
      G B
- 平均兩個 G，再用 RGB 光度公式轉灰階
- 自動正規化與 gamma 校正（模擬 sRGB）
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ======= 修改這裡的檔案路徑 =======
img_path = "/home/ryan529/project/bdd100k/raw10/train/0004a4c0-d4dff0ad.png"
# ==================================

# 讀取圖片
raw = np.array(Image.open(img_path))

# 若為三通道，取第一通道
if raw.ndim == 3:
    raw = raw[..., 0]

# 轉成 float 並歸一化到 0~1（10-bit → 1023）
raw = raw.astype(np.float32) / 1023.0

# 提取 RGGB pattern
R = raw[0::2, 0::2]
G1 = raw[0::2, 1::2]
G2 = raw[1::2, 0::2]
B = raw[1::2, 1::2]

# 對齊大小與平均兩個 G
H, W = R.shape
G = ((G1[:H, :W] + G2[:H, :W]) / 2.0)

# RGB → 光度灰階
gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

# 正規化 + gamma 校正（模擬 sRGB）
gray = np.clip(gray, 0, 1)
gray_gamma = np.power(gray, 1 / 2.2)

# 轉成 8-bit 顯示
gray_8bit = (gray_gamma * 255).astype(np.uint8)

# 顯示結果
plt.figure(figsize=(8, 5))
plt.imshow(gray_8bit, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title("Visible Grayscale from 10-bit RGGB")
plt.show()
