# -*- coding: utf-8 -*-
"""
dataset.py — RAW4 detection dataset — v2 fast path
--------------------------------------------------
設計目標：
- 保持舊版 API 與 key 介面不變（raw4/image/mask/human/boxes/...）。
- 對「每張 sample 的 CPU 開銷」做全面瘦身。
- 對 5090 友善：DataLoader + mmap + minimal CPU work。
"""

from __future__ import annotations
import os
import json
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover
    import imageio as iio  # type: ignore

__all__ = [
    "JointRaw4Dataset",
    "load_items_json",
    "merge_items_detr_json",
    "stack_collate_fn",
    "build_detr_targets_from_batch",
    "Raw4DaliSource",
]

# -----------------------
# helpers
# -----------------------

def _to_even(v: int) -> int:
    return v if (v % 2 == 0) else (v - 1)

def _safe_interpolate(
    img: torch.Tensor,
    size: Tuple[int, int],
    mode: str,
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    if img.ndim == 3:
        img4d = img.unsqueeze(0)
        if align_corners is not None:
            out = F.interpolate(img4d, size=size, mode=mode, align_corners=align_corners)
        else:
            out = F.interpolate(img4d, size=size, mode=mode)
        return out.squeeze(0)
    if align_corners is not None:
        return F.interpolate(img, size=size, mode=mode, align_corners=align_corners)
    return F.interpolate(img, size=size, mode=mode)

def _pad_to_multiple(t: torch.Tensor, mul: int = 32) -> Tuple[torch.Tensor, int, int]:
    if t.ndim != 3:
        raise ValueError(f"expect (C,H,W), got {tuple(t.shape)}")
    _, H, W = t.shape
    Hn = (H + mul - 1) // mul * mul
    Wn = (W + mul - 1) // mul * mul
    if (Hn, Wn) == (H, W):
        return t, H, W
    pad = (0, Wn - W, 0, Hn - H)  # pad right, bottom
    return F.pad(t, pad, mode="replicate"), Hn, Wn

def _pad_to_size(t: torch.Tensor, Hn: int, Wn: int) -> torch.Tensor:
    _, H, W = t.shape
    ph = max(0, Hn - H)
    pw = max(0, Wn - W)
    if ph == 0 and pw == 0:
        return t
    return F.pad(t, (0, pw, 0, ph), mode="replicate")

def _first_nonempty(d: Dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k, None)
        if isinstance(v, str) and v:
            return v
    return None

def _std_item_keys(it: Dict) -> Dict:
    out = dict(it)
    img = _first_nonempty(it, ["raw4_path", "raw", "raw_path", "file_name"])
    if img is not None:
        out["raw4_path"] = img
    if "mask_path" in it and isinstance(it["mask_path"], str):
        out["mask_path"] = it["mask_path"]
    if "id" in it and isinstance(it["id"], (str, int)):
        out["id"] = str(it["id"])
    if "annotations" in it and isinstance(it["annotations"], list):
        out["annotations"] = it["annotations"]
    if ("img_path" not in out) and ("id" not in out or "annotations" not in out):
        out["img_path"] = out.get("img_path", "")
    return out


# -----------------------
# DETR json parsing & merge
# -----------------------

def _parse_det_json(path: str) -> Dict[str, List[Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, List[Dict]] = {}

    if isinstance(data, list):
        for it in data:
            if not isinstance(it, dict): continue
            sid = str(it.get("id", _first_nonempty(it, ["file_name"]) or ""))
            if not sid: continue
            bag: List[Dict] = []
            for a in (it.get("annotations", []) or []):
                bb = a.get("bbox")
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    bag.append({"bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])], "category_id": int(a.get("category_id", 0))})
            out[sid] = bag
        return out

    if isinstance(data, dict) and "images" in data and "annotations" in data:
        id2name: Dict[str, str] = {}
        for im in data.get("images", []):
            if not isinstance(im, dict): continue
            sid = str(im.get("id", ""))
            name = os.path.splitext(os.path.basename(im.get("file_name", sid)))[0]
            id2name[sid] = name
        for ann in data.get("annotations", []):
            if not isinstance(ann, dict): continue
            img_id = str(ann.get("image_id", ""))
            sid = id2name.get(img_id, img_id)
            bb = ann.get("bbox")
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4): continue
            out.setdefault(sid, []).append({"bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])], "category_id": int(ann.get("category_id", 0))})
        return out

    if isinstance(data, dict) and all(isinstance(k, (str, int)) for k in data.keys()):
        for k, v in data.items():
            sid = str(k)
            bag: List[Dict] = []
            for a in (v or []):
                bb = a.get("bbox")
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    bag.append({"bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])], "category_id": int(a.get("category_id", 0))})
            out[sid] = bag
        return out

    return {}

def load_items_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError("items json must be a list")
    return [_std_item_keys(it) for it in items if isinstance(it, dict)]

def merge_items_detr_json(items_json: str, detr_json: Optional[str] = None, *, strict: bool = False) -> List[Dict]:
    items = load_items_json(items_json)
    if not detr_json: return items
    detmap = _parse_det_json(detr_json)
    merged: List[Dict] = []
    for it in items:
        sid = str(it.get("id", ""))
        anns = detmap.get(sid, [])
        if strict and not anns: continue
        mit = dict(it)
        mit["annotations"] = anns
        merged.append(mit)
    return merged


# -----------------------
# Dataset
# -----------------------

class JointRaw4Dataset(torch.utils.data.Dataset):
    def __init__(
        self, items: List[Dict], *, in_h: Optional[int] = None, in_w: Optional[int] = None,
        keep_aspect: bool = False, pad_multiple: int = 32, split: Optional[str] = None,
        raw_root: Optional[str] = None, raw_train: Optional[str] = None, raw_val: Optional[str] = None,
        mask_root: Optional[str] = None, transforms: Optional[Iterable[Any]] = None,
        hflip_p: float = 0.0, human_value: int = 2, saliency_threshold: float = 0.5,
        ann_src_size: Optional[Tuple[int, int]] = (1280, 720), ann_src_is_hw: bool = False,
        human_aux_weight: float = 0.3, debug: bool = False, cache_masks: bool = False
    ) -> None:
        super().__init__()
        self.items = [_std_item_keys(it) for it in items if isinstance(it, dict)]

        if in_h is not None and in_w is not None:
            in_h = _to_even(int(in_h))
            in_w = _to_even(int(in_w))
        self.in_h, self.in_w = in_h, in_w
        self.keep_aspect = bool(keep_aspect)
        self.pad_multiple = int(pad_multiple)

        self.split = (split or "").lower().strip()
        self.raw_train, self.raw_val, self.raw_root, self.mask_root = raw_train, raw_val, raw_root, mask_root
        self.human_value, self.sal_thr, self.human_aux_weight = int(human_value), float(saliency_threshold), float(human_aux_weight)
        self.tfms = list(transforms) if transforms is not None else None
        self.hflip_p, self._debug = float(hflip_p), bool(debug)

        if ann_src_size is not None:
            self.ann_src_size = (int(ann_src_size[1]), int(ann_src_size[0])) if ann_src_is_hw else (int(ann_src_size[0]), int(ann_src_size[1]))
        else:
            self.ann_src_size = None

        self._cat2idx: Optional[Dict[int, int]] = None
        self.num_classes: int = 0
        all_ids: List[int] = []
        for it in self.items:
            for a in (it.get("annotations") or []):
                if isinstance(a, dict) and "category_id" in a:
                    all_ids.append(int(a["category_id"]))
        if all_ids:
            uniq = sorted(set(all_ids))
            self._cat2idx = {cid: i for i, cid in enumerate(uniq)}
            self.num_classes = len(uniq)

        self._cache_masks = cache_masks
        self._mask_cache: Dict[str, torch.Tensor] = {}

    def _active_root(self) -> Optional[str]:
        if self.split in ("train", "training"): return self.raw_train or self.raw_root
        if self.split in ("val", "valid", "validation", "eval"): return self.raw_val or self.raw_root
        return self.raw_root

    def _resolve_raw4_path(self, it: Dict) -> str:
        p0 = it.get("raw4_path", "")
        if isinstance(p0, str) and p0 and os.path.isabs(p0) and os.path.isfile(p0): return p0
        root = self._active_root()
        sid = str(it.get("id", os.path.splitext(os.path.basename(p0))[0] or ""))
        if root:
            if isinstance(p0, str) and p0 and os.path.isfile(os.path.join(root, p0)): return os.path.join(root, p0)
            if os.path.isfile(os.path.join(root, f"{sid}.npy")): return os.path.join(root, f"{sid}.npy")
        if self.raw_root and self.split in ("train", "training", "val", "valid", "validation", "eval"):
            sub = "train" if self.split.startswith("train") else "val"
            if os.path.isfile(os.path.join(self.raw_root, sub, f"{sid}.npy")): return os.path.join(self.raw_root, sub, f"{sid}.npy")
        raise FileNotFoundError(f"Cannot resolve image path for id={sid}.")

    def _resolve_mask_path(self, it: Dict) -> Optional[str]:
        p0 = it.get("mask_path", "")
        if isinstance(p0, str) and p0:
            if os.path.isabs(p0) and os.path.isfile(p0): return p0
            if self.mask_root and os.path.isfile(os.path.join(self.mask_root, p0)): return os.path.join(self.mask_root, p0)
        return None

    @staticmethod
    def _load_image_npy(path: str, *, mmap: bool = True) -> torch.Tensor:
        if os.path.splitext(path)[1].lower() != ".npy": raise ValueError(f"image must be .npy, got: {path}")
        arr = np.load(path, mmap_mode="r" if mmap else None)
        if arr.ndim != 3: raise ValueError(f"npy must be 3D, got {arr.shape}")
        if arr.dtype != np.float32: arr = arr.astype(np.float32, copy=False)
        if arr.shape[0] == 4:
            chw = arr
        elif arr.shape[-1] == 4:
            chw = np.ascontiguousarray(arr.transpose(2, 0, 1))
        else:
            raise ValueError(f"npy must have 4 channels, got {arr.shape}")

        # np.load(..., mmap_mode='r') may be non-writable; torch.from_numpy warns on it.
        if not chw.flags.writeable:
            chw = np.array(chw, dtype=np.float32, copy=True)
        return torch.from_numpy(chw)

    def _parse_mask_tensor(self, m: torch.Tensor, like_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        Ht, Wt = like_hw
        if m.ndim == 2: m = m.unsqueeze(0)
        if m.shape[0] == 2:
            sal = m[0:1].float()
            hum = (m[1:2] > 0).float()
        else:
            if m.dtype.is_floating_point:
                sal = m[0:1].clamp(0, 1).float()
                if self.sal_thr > 0: sal = (sal > self.sal_thr).float()
                hum = torch.zeros_like(sal)
            else:
                sal = (m[0:1] > 0).float()
                hum = (m[0:1] == self.human_value).float()
        if sal.shape[-2:] != (Ht, Wt): sal = _safe_interpolate(sal, (Ht, Wt), mode="nearest")
        if hum.shape[-2:] != (Ht, Wt): hum = _safe_interpolate(hum, (Ht, Wt), mode="nearest")
        return sal, hum

    def _load_masks(self, mask_path: Optional[str], like_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        Ht, Wt = like_hw
        if mask_path is None or not os.path.isfile(mask_path):
            return torch.zeros(1, Ht, Wt, dtype=torch.float32), torch.zeros(1, Ht, Wt, dtype=torch.float32)

        if self._cache_masks and mask_path in self._mask_cache:
            m_t = self._mask_cache[mask_path]
        else:
            ext = os.path.splitext(mask_path)[1].lower()
            if ext == ".npy":
                arr = np.load(mask_path)
                if arr.ndim == 2: arr = arr[None, ...]
                elif arr.ndim == 3 and arr.shape[0] not in (1, 2) and arr.shape[-1] in (1, 2): arr = np.transpose(arr, (2, 0, 1))
                m_t = torch.from_numpy(np.array(arr, copy=False))
            else:
                m = iio.imread(mask_path)
                if getattr(m, "ndim", 0) == 3 and m.shape[-1] == 3: m = m[..., 0]
                m_t = torch.from_numpy(np.array(m, copy=False))
                if m_t.ndim == 3 and m_t.shape[0] != 1 and m_t.shape[-1] in (1, 2): m_t = m_t.permute(2, 0, 1).contiguous()
                elif m_t.ndim == 2: m_t = m_t.unsqueeze(0)

            m_t = m_t.float() if np.issubdtype(m_t.numpy().dtype, np.floating) else m_t.to(torch.int64)
            if self._cache_masks: self._mask_cache[mask_path] = m_t

        return self._parse_mask_tensor(m_t, like_hw)

    def _resize_content(self, img4: torch.Tensor, sal: torch.Tensor, hum: torch.Tensor, Ht: int, Wt: int, keep_aspect: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        _, H0, W0 = img4.shape
        if (H0, W0) == (Ht, Wt): return img4, sal, hum, H0, W0
        if not keep_aspect:
            img4 = _safe_interpolate(img4, (Ht, Wt), mode="bilinear", align_corners=False)
            sal = _safe_interpolate(sal, (Ht, Wt), mode="nearest")
            hum = _safe_interpolate(hum, (Ht, Wt), mode="nearest")
            return img4, sal, hum, Ht, Wt
        scale = min(Ht / H0, Wt / W0)
        Hr, Wr = max(2, _to_even(int(round(H0 * scale)))), max(2, _to_even(int(round(W0 * scale))))
        img4_s = _safe_interpolate(img4, (Hr, Wr), mode="bilinear", align_corners=False)
        sal_s = _safe_interpolate(sal, (Hr, Wr), mode="nearest")
        hum_s = _safe_interpolate(hum, (Hr, Wr), mode="nearest")
        return img4_s, sal_s, hum_s, Hr, Wr

    def __len__(self) -> int: return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        sid = str(it.get("id", "")) or os.path.splitext(os.path.basename(it.get("img_path", "")))[0]

        img4 = self._load_image_npy(self._resolve_raw4_path(it), mmap=True)
        H0, W0 = img4.shape[-2], img4.shape[-1]
        sal0, hum0 = self._load_masks(self._resolve_mask_path(it), like_hw=(H0, W0))

        Ht, Wt = (int(self.in_h), int(self.in_w)) if self.in_h and self.in_w else (H0, W0)

        if (Ht, Wt) == (H0, W0) and not self.keep_aspect:
            img4_c, sal_c, hum_c, Hr, Wr = img4, sal0, hum0, H0, W0
        else:
            img4_c, sal_c, hum_c, Hr, Wr = self._resize_content(img4, sal0, hum0, Ht, Wt, self.keep_aspect)

        flipped = False
        if self.hflip_p > 0.0 and random.random() < self.hflip_p:
            flipped = True
            img4_c = torch.flip(img4_c, dims=[2])
            sal_c = torch.flip(sal_c, dims=[2])
            hum_c = torch.flip(hum_c, dims=[2])

        pad = (0, Wt - Wr, 0, Ht - Hr)
        if pad[1] > 0 or pad[3] > 0:
            img4_p = F.pad(img4_c, pad, mode="replicate")
            sal_p = F.pad(sal_c, pad, mode="replicate")
            hum_p = F.pad(hum_c, pad, mode="replicate")
        else:
            img4_p, sal_p, hum_p = img4_c, sal_c, hum_c

        if self.tfms is not None:
            H_before, W_before = img4_p.shape[-2], img4_p.shape[-1]
            sample = {"image": img4_p, "mask": sal_p, "human": hum_p}
            for t in self.tfms: sample = t(sample)
            img4_p, sal_p, hum_p = sample["image"], sample["mask"], sample.get("human", torch.zeros_like(sample["mask"]))
            if (img4_p.shape[-2], img4_p.shape[-1]) != (H_before, W_before):
                img4_p = _safe_interpolate(img4_p, (H_before, W_before), mode="bilinear", align_corners=False)
                sal_p = _safe_interpolate(sal_p, (H_before, W_before), mode="nearest")
                hum_p = _safe_interpolate(hum_p, (H_before, W_before), mode="nearest")

        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.long)
        anns = it.get("annotations") or []

        if isinstance(anns, list) and len(anns) > 0:
            src_w, src_h = float(max(1.0, self.ann_src_size[0] if self.ann_src_size else W0)), float(max(1.0, self.ann_src_size[1] if self.ann_src_size else H0))
            bxs, lbs = [], []
            for a in anns:
                if not isinstance(a, dict): continue
                bb = a.get("bbox")
                if not (isinstance(bb, (list, tuple)) and len(bb) == 4): continue
                x, y, w, h = map(float, bb)
                if w <= 0 or h <= 0: continue

                if (x > 1.0) or (y > 1.0) or (w > 1.0) or (h > 1.0):
                    sx, sy = float(Wr) / src_w, float(Hr) / src_h
                    x1, y1, x2, y2 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
                elif 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
                    x1, y1, x2, y2 = x * Wr, y * Hr, (x + w) * Wr, (y + h) * Hr
                    if (x2 <= x1) or (y2 <= y1):
                        if ((x + w) * Wr > x * Wr) and ((y + h) * Hr > y * Hr): pass # Handled by clamping later
                else: continue

                if flipped: x1, x2 = Wr - x2, Wr - x1
                if (x2 <= x1) or (y2 <= y1): continue

                bxs.append([x1, y1, x2, y2])
                lbs.append(self._cat2idx.get(int(a.get("category_id", 0)), -1) if self._cat2idx is not None else int(a.get("category_id", 0)))

            if bxs:
                boxes = torch.tensor(bxs, dtype=torch.float32)
                labels = torch.tensor(lbs, dtype=torch.long)
                keep_l = labels >= 0
                boxes, labels = boxes[keep_l], labels[keep_l]

                boxes[:, 0] = boxes[:, 0].clamp(0, Wr - 1)
                boxes[:, 1] = boxes[:, 1].clamp(0, Hr - 1)
                boxes[:, 2] = boxes[:, 2].clamp(0, Wr - 1)
                boxes[:, 3] = boxes[:, 3].clamp(0, Hr - 1)

                wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)
                keep = (wh[:, 0] > 1e-3) & (wh[:, 1] > 1e-3)
                if keep.numel() and (~keep).any(): boxes, labels = boxes[keep], labels[keep]

        img4_pad, Hin, Win = _pad_to_multiple(img4_p, self.pad_multiple)
        sal = _pad_to_size(sal_p, Hin, Win)
        hum = _pad_to_size(hum_p, Hin, Win)

        pixel_weight = torch.ones_like(sal)
        if (self.human_aux_weight if (hum_c.sum().item() > 0.0) else 0.0) > 0:
            pixel_weight = pixel_weight + self.human_aux_weight * (hum > 0).float()

        return {
            "raw4": img4_pad, "image": img4_pad, "mask": sal, "human": hum, "pixel_weight": pixel_weight,
            "H": torch.tensor(Hr, dtype=torch.long), "W": torch.tensor(Wr, dtype=torch.long),
            "Hin": torch.tensor(Hin, dtype=torch.long), "Win": torch.tensor(Win, dtype=torch.long),
            "id": sid, "boxes": boxes, "labels": labels,
        }


# -----------------------
# DALI external source wrapper
# -----------------------

class Raw4DaliSource:
    def __init__(self, dataset, shuffle: bool = True, mmap: bool = True):
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.mmap = bool(mmap)
        self._indices = list(range(len(self.dataset)))
        self._rng = random.Random(1234)
        self._pos = 0
        self._epoch = 0
        self._reshuffle_if_needed()

    def __len__(self): return len(self.dataset)

    def _reshuffle_if_needed(self):
        if self.shuffle: self._rng.shuffle(self._indices)
        self._pos = 0

    def reset(self):
        self._epoch += 1
        self._reshuffle_if_needed()

    def __call__(self, sample_info):
        if self._pos >= len(self._indices): self.reset()
        idx = self._indices[self._pos]
        self._pos += 1
        ex = self.dataset[idx]

        img = self._to_hwc_float_np(ex.get("raw4", ex.get("image")))
        H, W, _ = img.shape
        mask = self._to_hw1_float_np(ex.get("mask", None), H, W)
        human = self._to_hw1_float_np(ex.get("human", None), H, W)

        return np.ascontiguousarray(img, dtype=np.float32), np.ascontiguousarray(mask, dtype=np.float32), np.ascontiguousarray(human, dtype=np.float32)

    def _to_hwc_float_np(self, x):
        if isinstance(x, torch.Tensor): x = x.detach().cpu().float().numpy()
        elif isinstance(x, np.ndarray): x = x.astype(np.float32, copy=False)
        else: x = np.array(x, dtype=np.float32)

        if x.ndim == 2: x = x[..., None]
        elif x.ndim == 3:
            if x.shape[0] in (1, 3, 4) and x.shape[0] <= x.shape[-1]: x = np.transpose(x, (1, 2, 0))
        return np.ascontiguousarray(x, dtype=np.float32)

    def _to_hw1_float_np(self, x, H: int, W: int):
        if x is None: return np.zeros((H, W, 1), dtype=np.float32)
        if isinstance(x, torch.Tensor): x = x.detach().cpu().float().numpy()
        elif isinstance(x, np.ndarray): x = x.astype(np.float32, copy=False)
        else: x = np.array(x, dtype=np.float32)

        if x.ndim == 2: x = x[..., None]
        elif x.ndim == 3:
            if x.shape[0] == 1 and (x.shape[1] == H and x.shape[2] == W): x = np.transpose(x, (1, 2, 0))
            elif x.shape[-1] == 1 and (x.shape[0] == H and x.shape[1] == W): pass
            else:
                if x.shape[0] > 1 and (x.shape[1] == H and x.shape[2] == W): x = np.transpose(x[0:1, ...], (1, 2, 0))
                elif x.shape[-1] > 1 and (x.shape[0] == H and x.shape[1] == W): x = x[..., :1]
        return np.ascontiguousarray(x, dtype=np.float32)


# -----------------------
# Collate + DETR targets
# -----------------------

def build_detr_targets_from_batch(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    B = batch["image"].shape[0]
    # 【關鍵修復】這裡必須使用 Padded 後的畫布尺寸 (Hin, Win)，
    # 因為 DETR 模型預測出的框是基於 Padded Image，Loss 內部也需要用 Hin/Win 來正規化。
    H_in = batch["Hin"]
    W_in = batch["Win"]
    out: List[Dict[str, Any]] = []
    for i in range(B):
        out.append({
            "boxes": batch["boxes"][i].float(),
            "labels": batch["labels"][i].long(),
            "size": torch.stack([H_in[i], W_in[i]]).long(),
            "id": batch["ids"][i],
        })
    return out

def stack_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    humans = torch.stack([b["human"] for b in batch], dim=0)
    pwmap = torch.stack([b["pixel_weight"] for b in batch], dim=0)
    ids = [b.get("id", str(i)) for i, b in enumerate(batch)]

    Hc = torch.stack([b.get("H", torch.tensor(imgs.shape[-2])) for b in batch]).long()
    Wc = torch.stack([b.get("W", torch.tensor(imgs.shape[-1])) for b in batch]).long()
    Hin = torch.stack([b.get("Hin", torch.tensor(imgs.shape[-2])) for b in batch]).long()
    Win = torch.stack([b.get("Win", torch.tensor(imgs.shape[-1])) for b in batch]).long()

    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]

    collated: Dict[str, Any] = {
        "raw4": imgs,
        "image": imgs,
        "mask": masks,
        "human": humans,
        "pixel_weight": pwmap,
        "ids": ids,
        "H": Hc,
        "W": Wc,
        "Hin": Hin,
        "Win": Win,
        "boxes": boxes,
        "labels": labels,
    }
    # 建立 DETR 需要的 targets 格式
    collated["targets"] = build_detr_targets_from_batch(collated)
    return collated
