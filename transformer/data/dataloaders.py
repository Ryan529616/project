# -*- coding: utf-8 -*-
"""
dataloaders.py — Detector dataloaders wired to dataset.py/transforms.py
(Rev.2025-11-XX, 5090-tuned + optional DALI)

功能總結：
- 預設走傳統 PyTorch DataLoader + JointRaw4Dataset。
- 若 cfg.data.use_dali = true 且安裝了 NVIDIA DALI：
    → 仍然用 PyTorch DataLoader 給 DETR（含 boxes / labels / targets）。
    → 額外建立 DALI loader：
        result["train_dali_loader"] / result["val_dali_loader"]
    → 訓練 loop 每個 iter 用 DALI 的 image/mask/human 覆蓋 PyTorch batch。
"""

from __future__ import annotations
import os, json, random, logging, inspect, time, sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

# ==========================================
# 模組定位魔法：確保能找到 data.dataset 與 data.transforms
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import data.dataset as dsmod
try:
    import data.transforms as tfm
except ImportError:
    tfm = None

# ---------------------------------------------------------------------
# Optional DALI support
# ---------------------------------------------------------------------

try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator  # type: ignore
    _HAS_DALI = True
except Exception:
    DALIGenericIterator = None  # type: ignore
    _HAS_DALI = False

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _get(d: Any, key: str, default=None):
    try: return getattr(d, key)
    except Exception:
        try: return d[key]
        except Exception: return default

def _json_load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _is_path(p: Optional[str]) -> bool:
    return isinstance(p, str) and len(p) > 0

def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**31
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

def _parse_ann_src_size(val: Any, fallback: Tuple[int,int]=(1280,720)) -> Tuple[int,int]:
    if isinstance(val, (list, tuple)) and len(val) == 2:
        try: return (int(val[0]), int(val[1]))
        except Exception: return fallback
    if isinstance(val, str) and "x" in val.lower():
        try:
            w, h = val.lower().split("x")
            return (int(w), int(h))
        except Exception: return fallback
    return fallback

def _ctor_accepts_kw(callable_obj, kw: str) -> bool:
    try:
        sig = inspect.signature(callable_obj)
        return (kw in sig.parameters) or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    except Exception:
        return False

def _fresh_seed(base: Optional[int] = None) -> int:
    s = int(time.time_ns() & 0x7FFFFFFF)
    try: s ^= int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF
    except Exception: pass
    if base is not None: s ^= int(base) & 0x7FFFFFFF
    return s & 0x7FFFFFFF

# ---------------------------------------------------------------------
# Resolve transforms
# ---------------------------------------------------------------------

def resolve_transforms(in_h: int, in_w: int, cfg_data: Any) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    if tfm is None:
        return None, None

    has_dali_train = hasattr(tfm, "raw4_train_pipeline")
    has_dali_val   = hasattr(tfm, "raw4_val_pipeline")
    if has_dali_train or has_dali_val:
        return None, None

    build_train = getattr(tfm, "build_train_transforms", None)
    build_val   = getattr(tfm, "build_val_transforms", None)

    if callable(build_train) and callable(build_val):
        aug = _get(cfg_data, "augment", True)
        kwargs = dict(
            hflip_p=0.5, scale_range=(0.9, 1.1), bottom_bias=0.45, exp_p=0.7,
            gain_range=(0.8, 1.25), bias_range=(-0.05, 0.05), gamma_range=(0.9, 1.2),
            gauss_p=0.6, gauss_sigma=(0.0, 0.015), shot_p=0.5, shot_lambda=(200.0, 1200.0),
            stripe_p=0.3, stripe_strength=0.03, stripe_axis="row", blur_p=0.4,
            blur_sigma=(0.0, 1.2), cutout_p=0.5, cutout_holes=2, cutout_max_frac=0.2,
            cutout_affect_gt="none",
        )
        tcfg = _get(cfg_data, "transforms", None)
        if tcfg and isinstance(tcfg, dict):
            for k, v in tcfg.items(): kwargs[k] = v

        train_tfms = build_train(in_h, in_w, **kwargs) if aug else []
        val_tfms   = build_val(in_h, in_w)
        return train_tfms, val_tfms

    return None, None

# ---------------------------------------------------------------------
# Resolve dataset class and collate
# ---------------------------------------------------------------------

def resolve_dataset_class():
    for name in ("JointRaw4Dataset", "DetrRaw4Dataset", "SaliencyDetrDataset", "SaliencyDatasetV2", "SaliencyDataset"):
        if hasattr(dsmod, name): return getattr(dsmod, name)
    raise ImportError("[dataloaders] No compatible dataset class found in data.dataset")

def resolve_collate_fn():
    fn = getattr(dsmod, "stack_collate_fn", None)
    if callable(fn): return fn
    return None

def _default_collate_safe(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) == 0: return {}
    out: Dict[str, Any] = {}
    keys = set().union(*(b.keys() for b in batch))

    def _maybe_stack(key: str, strict_ndim: Optional[int] = None):
        xs = [b[key] for b in batch if key in b and b[key] is not None]
        if len(xs) != len(batch): return
        if torch.is_tensor(xs[0]):
            tlist = []
            for x in xs:
                if strict_ndim is not None and x.ndim != strict_ndim: return
                tlist.append(x)
            try: out[key] = torch.stack(tlist, dim=0)
            except Exception: out[key] = tlist
        else: out[key] = xs

    for k in ("raw4", "image", "mask", "human", "pixel_weight"):
        if k in keys: _maybe_stack(k, strict_ndim=3)

    if "targets" in keys: out["targets"] = [b.get("targets") for b in batch]
    if "ids" in keys: out["ids"] = [b.get("ids") for b in batch]
    if "id" in keys and "ids" not in out: out["id"] = [b.get("id") for b in batch]
    if "size" in keys: out["size"] = [b.get("size") for b in batch]

    for k in keys:
        if k not in out: out[k] = [b.get(k) for b in batch]
    if "raw4" not in out and "image" in out: out["raw4"] = out["image"]
    return out

# ---------------------------------------------------------------------
# JSON/item helpers
# ---------------------------------------------------------------------

def _find_first(cfg_data: Any, keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        v = _get(cfg_data, k, None)
        if _is_path(v) and os.path.isfile(v): return v
    return None

def _gather_items_from_raw(root: Optional[str]) -> List[Dict[str, Any]]:
    if not _is_path(root) or not os.path.isdir(root): return []
    out: List[Dict[str, Any]] = []
    try:
        for fn in os.listdir(root):
            if fn.lower().endswith(".npy"): out.append({"id": os.path.splitext(fn)[0]})
    except Exception: pass
    seen = set()
    uniq = []
    for it in out:
        sid = it["id"]
        if sid not in seen:
            uniq.append(it); seen.add(sid)
    return uniq

# ---------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------

def _build_dataset(split: str, cfg: Any, *, in_h: int, in_w: int, tfms: Optional[List[Any]]) -> Tuple[torch.utils.data.Dataset, Dict[str, Any]]:
    data_cfg = _get(cfg, "data", {})
    raw_root  = _get(data_cfg, "raw_root", None)
    raw_train = _get(data_cfg, "raw_train", _get(data_cfg, "train_root", None))
    raw_val   = _get(data_cfg, "raw_val",   _get(data_cfg, "val_root", None))
    mask_root    = _get(data_cfg, "mask_root", None)
    mask_train   = _get(data_cfg, "train_mask_root", None)
    mask_val     = _get(data_cfg, "val_mask_root", None)
    active_mask_root = mask_train if split == "train" else (mask_val if split == "val" else mask_root)

    tcfg    = _get(data_cfg, "transforms", {}) or {}
    hflip_p = float(_get(tcfg, "hflip_p", 0.0)) if split == "train" else 0.0
    ann_src_size  = _parse_ann_src_size(_get(data_cfg, "ann_src_size", (1280, 720)))
    ann_src_is_hw = bool(_get(data_cfg, "ann_src_is_hw", False))

    items_json = _find_first(data_cfg, [f"{split}_items_json", f"{split}_items", "items_json"])
    detr_json  = _find_first(data_cfg, [f"{split}_detr_json", f"detr_{split}_json", f"json_{split}_det", f"{split}_det_json", "detr_json"])
    merged_json = _find_first(data_cfg, [f"{split}_merged_json", "merged_json"])

    DS = resolve_dataset_class()
    base_kwargs = dict(
        split=split, raw_root=raw_root, raw_train=raw_train, raw_val=raw_val, mask_root=active_mask_root,
        in_h=in_h, in_w=in_w, keep_aspect=bool(_get(data_cfg, "keep_aspect", False)),
        pad_multiple=int(_get(data_cfg, "pad_multiple", 8)), hflip_p=hflip_p, transforms=tfms,
        ann_src_size=ann_src_size, ann_src_is_hw=ann_src_is_hw,
    )

    ds = None
    items_used = "items(merged_json)"

    if _is_path(merged_json):
        items = _json_load(merged_json)
        if not isinstance(items, list): raise ValueError("merged_json is not a list")
        init_obj = getattr(DS, "__init__", DS)
        init_sig = inspect.signature(init_obj)
        allowed = set(init_sig.parameters.keys())
        ctor_kwargs = {k: v for k, v in base_kwargs.items() if (k in allowed) or _ctor_accepts_kw(init_obj, k)}
        ctor_kwargs["items"] = items
        ds = DS(**ctor_kwargs)
        items_used = "items(merged_json)"

    if ds is None and _is_path(items_json):
        items = _json_load(items_json)
        if not isinstance(items, list): raise ValueError("items_json is not a list")
        init_obj = getattr(DS, "__init__", DS)
        init_sig = inspect.signature(init_obj)
        allowed = set(init_sig.parameters.keys())
        ctor_kwargs = {k: v for k, v in base_kwargs.items() if (k in allowed) or _ctor_accepts_kw(init_obj, k)}
        ctor_kwargs["items"] = items
        ds = DS(**ctor_kwargs)
        items_used = "items(items_json)"

    if ds is None:
        active_root = raw_train if split == "train" else (raw_val if split == "val" else raw_root)
        items = _gather_items_from_raw(active_root)
        if not items: raise RuntimeError(f"[dataloaders] cannot build items for split={split}.")
        init_obj = getattr(DS, "__init__", DS)
        init_sig = inspect.signature(init_obj)
        allowed = set(init_sig.parameters.keys())
        ctor_kwargs = {k: v for k, v in base_kwargs.items() if (k in allowed) or _ctor_accepts_kw(init_obj, k)}
        ctor_kwargs["items"] = items
        ds = DS(**ctor_kwargs)
        items_used = "auto(active_root)"

    meta = {
        "split": split, "items_mode": items_used, "raw_root": raw_root, "raw_train": raw_train,
        "raw_val": raw_val, "mask_root": active_mask_root, "items_json": items_json,
        "detr_json": detr_json, "merged_json": merged_json, "ann_src_size": ann_src_size,
        "ann_src_is_hw": ann_src_is_hw, "hflip_p": hflip_p,
    }
    return ds, meta

def _count_items_with_boxes(ds, sample_k: int = 64) -> Tuple[int, int]:
    n = len(ds)
    if n == 0: return 0, 0
    step = max(1, n // min(n, sample_k))
    idxs = list(range(0, n, step))[:sample_k]
    ok = 0
    for i in idxs:
        try:
            ex = ds[i]
            boxes = ex.get("boxes") if isinstance(ex, dict) else None
            if torch.is_tensor(boxes) and boxes.numel() >= 4: ok += 1
        except Exception: pass
    return ok, len(idxs)

def _resolve_mp_context(cfg: Any):
    import multiprocessing as mp
    data_cfg = _get(cfg, "data", {})
    meth = str(_get(data_cfg, "mp_start_method", "spawn")).lower().strip()
    try: return mp.get_context(meth)
    except Exception:
        try: return mp.get_context("spawn")
        except Exception: return None

# ---------------------------------------------------------------------
# DALI wrapper & builders
# ---------------------------------------------------------------------

class DALIDataLoaderWrapper:
    def __init__(self, dali_iter, human_aux_weight: float = 0.3, channel_last: bool = True):
        self.dali_iter = dali_iter
        self.human_aux_weight = float(human_aux_weight)
        self.channel_last = bool(channel_last)
        self._len_cache: Optional[int] = None

    def __len__(self):
        if self._len_cache is not None: return self._len_cache
        try:
            size = int(getattr(self.dali_iter, "_size", -1))
            bs = int(getattr(self.dali_iter, "batch_size", -1))
            if size > 0 and bs > 0:
                self._len_cache = (size + bs - 1) // bs
                return self._len_cache
        except Exception: pass
        return -1

    def __iter__(self):
        for data in self.dali_iter:
            batch = data[0]
            img = batch.get("image", None)
            if img is None: img = batch.get("img", None)
            if img is None: img = batch.get("raw4", None)

            if isinstance(img, torch.Tensor):
                if self.channel_last and img.ndim == 4 and img.shape[-1] in (1, 3, 4):
                    img = img.permute(0, 3, 1, 2).contiguous()
                batch["image"] = img
                batch["raw4"] = img
            else: raise RuntimeError("[DALI wrapper] batch did not contain 'image'/'img'/'raw4' tensor")

            mask = batch.get("mask", None)
            if isinstance(mask, torch.Tensor):
                if self.channel_last and mask.ndim == 4 and mask.shape[-1] == 1:
                    mask = mask.permute(0, 3, 1, 2).contiguous()
                batch["mask"] = mask

            human = batch.get("human", None)
            if isinstance(human, torch.Tensor):
                if self.channel_last and human.ndim == 4 and human.shape[-1] == 1:
                    human = human.permute(0, 3, 1, 2).contiguous()
                batch["human"] = human

            if "pixel_weight" not in batch and mask is not None and isinstance(mask, torch.Tensor):
                pw = torch.ones_like(mask)
                if human is not None and isinstance(human, torch.Tensor):
                    pw = pw + self.human_aux_weight * (human > 0).float()
                batch["pixel_weight"] = pw

            img = batch["image"]
            if isinstance(img, torch.Tensor):
                B, C, H, W = img.shape
                device = img.device
                if "Hin" not in batch: batch["Hin"] = torch.full((B,), H, dtype=torch.long, device=device)
                if "Win" not in batch: batch["Win"] = torch.full((B,), W, dtype=torch.long, device=device)
                if "H" not in batch: batch["H"] = batch["Hin"]
                if "W" not in batch: batch["W"] = batch["Win"]
                if "boxes" not in batch: batch["boxes"] = [torch.zeros((0, 4), dtype=torch.float32, device=device) for _ in range(B)]
                if "labels" not in batch: batch["labels"] = [torch.zeros((0,), dtype=torch.long, device=device) for _ in range(B)]
                if "ids" not in batch: batch["ids"] = [str(i) for i in range(B)]

            yield batch

    def reset(self):
        try: self.dali_iter.reset()
        except Exception: pass

def _build_dali_loader(split: str, ds: torch.utils.data.Dataset, cfg: Any, *, in_h: int, in_w: int, batch_size: int, shuffle_source: bool) -> DALIDataLoaderWrapper:
    if not _HAS_DALI: raise RuntimeError("[dataloaders] cfg.data.use_dali=True 但系統沒有安裝 NVIDIA DALI")
    if tfm is None: raise RuntimeError("[dataloaders] data.transforms not found")
    pipe_name = "raw4_train_pipeline" if split == "train" else "raw4_val_pipeline"
    pipe_def = getattr(tfm, pipe_name, None)
    if pipe_def is None or not callable(pipe_def): raise RuntimeError(f"[dataloaders] use_dali=True but transforms.{pipe_name} not found")

    if not hasattr(dsmod, "Raw4DaliSource"): raise RuntimeError("[dataloaders] dataset.Raw4DaliSource not found")
    SourceCls = getattr(dsmod, "Raw4DaliSource")

    dali_source = SourceCls(ds, shuffle=shuffle_source, mmap=True)
    data_cfg = _get(cfg, "data", {})
    nw = int(_get(data_cfg, "num_workers", 4))
    device_id = int(_get(data_cfg, "dali_device_id", 0))
    global_seed = int(_get(cfg, "seed", 0) or 0)
    seed = _fresh_seed(global_seed ^ (1 if split == "train" else 2))

    pipe = pipe_def(batch_size=batch_size, num_threads=max(1, nw), device_id=device_id, seed=seed, source=dali_source, in_h=in_h, in_w=in_w)
    pipe.build()

    try: n_out = len(pipe.outputs)
    except Exception: n_out = 3
    if n_out == 3: output_map = ["image", "mask", "human"]
    elif n_out == 4: output_map = ["image", "mask", "human", "pixel_weight"]
    else: output_map = [f"out{i}" for i in range(n_out)]

    dali_iter = DALIGenericIterator([pipe], output_map, size=len(ds), auto_reset=True, dynamic_shape=True)
    human_aux_weight = float(_get(data_cfg, "human_aux_weight", 0.3))
    return DALIDataLoaderWrapper(dali_iter, human_aux_weight=human_aux_weight, channel_last=True)

# ---------------------------------------------------------------------
# Public: build_dataloaders
# ---------------------------------------------------------------------

def build_dataloaders(cfg: Any) -> Dict[str, Any]:
    logging.getLogger(__name__).setLevel(logging.INFO)
    data_cfg = _get(cfg, "data", {})
    in_h = int(_get(data_cfg, "in_h", 360))
    in_w = int(_get(data_cfg, "in_w", 640))
    use_dali = bool(_get(data_cfg, "use_dali", False))

    train_tfms, val_tfms = resolve_transforms(in_h, in_w, data_cfg)
    train_ds, meta_tr = _build_dataset("train", cfg, in_h=in_h, in_w=in_w, tfms=train_tfms)
    val_ds,   meta_va = _build_dataset("val",   cfg, in_h=in_h, in_w=in_w, tfms=val_tfms)

    allow_empty_val = bool(_get(data_cfg, "allow_empty_val", False))
    ok_cnt, tried = _count_items_with_boxes(val_ds, sample_k=64)
    if ok_cnt == 0 and not allow_empty_val:
        raise RuntimeError(
            "[dataloaders] VAL split appears to have no usable detection boxes in sampled items "
            f"(checked {tried} samples). Hints:\n"
            f"  - Ensure IDs in cfg.data.val_merged_json align with cfg.data.val_detr_json (if both are used)\n"
            f"  - Verify ann_src_size & orientation: got {meta_va['ann_src_size']} with ann_src_is_hw={meta_va['ann_src_is_hw']}\n"
            "To bypass temporarily, set data.allow_empty_val=true."
        )

    collate_fn = resolve_collate_fn() or _default_collate_safe
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
    except Exception: pass
    ddp = (world_size > 1)

    if use_dali:
        if not _HAS_DALI: raise RuntimeError("[dataloaders] cfg.data.use_dali=True but NVIDIA DALI is not installed")
        if ddp: raise RuntimeError("[dataloaders] DALI integration currently only supports single GPU (WORLD_SIZE=1)")

    shuffle_train_cfg = bool(_get(data_cfg, "shuffle_train", True))
    shuffle_val_cfg   = bool(_get(data_cfg, "shuffle_val", False))
    global_seed   = int(_get(cfg, "seed", 0) or 0)
    user_seed_opt = _get(data_cfg, "shuffle_seed", None)
    base_seed     = _fresh_seed(global_seed) if (user_seed_opt is None or str(user_seed_opt).lower() in ("auto", "random")) else int(user_seed_opt)

    if use_dali:
        shuffle_train = False
        shuffle_val = False
        print("[INFO] DALI enabled: train/val DataLoader 使用順序 sampler（不洗牌），隨機性交給 DALI pipeline。")
    else:
        shuffle_train = shuffle_train_cfg
        shuffle_val = shuffle_val_cfg

    if ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=shuffle_train, seed=base_seed ^ 0x13572468)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=shuffle_val, seed=base_seed ^ 0x24681357)
    else:
        if shuffle_train:
            gen_tr = torch.Generator()
            gen_tr.manual_seed((base_seed ^ 0xA5A5A5A5) & 0x7FFFFFFF)
            train_sampler = RandomSampler(train_ds, generator=gen_tr)
        else: train_sampler = SequentialSampler(train_ds)

        if shuffle_val:
            gen_va = torch.Generator()
            gen_va.manual_seed((base_seed ^ 0x5A5A5A5A) & 0x7FFFFFFF)
            val_sampler = RandomSampler(val_ds, generator=gen_va)
        else: val_sampler = SequentialSampler(val_ds)

    bs      = int(_get(data_cfg, "batch_size", 8))
    bs_val  = int(_get(data_cfg, "val_batch_size", bs))
    nw      = int(_get(data_cfg, "num_workers", 12))
    pin_mem = bool(_get(data_cfg, "pin_memory", True))
    pf      = int(_get(data_cfg, "prefetch_factor", 2))
    drop_last_train = bool(_get(data_cfg, "drop_last", True))
    timeout_s = float(_get(data_cfg, "timeout", 120.0))
    if nw <= 0:
        timeout_s = 0.0

    mp_ctx = _resolve_mp_context(cfg)
    base_kwargs = dict(
        num_workers=nw,
        pin_memory=pin_mem,
        persistent_workers=(nw > 0) and bool(_get(data_cfg, "persistent_workers", True)),
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        timeout=timeout_s,
    )
    if nw > 0:
        base_kwargs["prefetch_factor"] = pf
        base_kwargs["multiprocessing_context"] = mp_ctx

    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=bs, drop_last=drop_last_train, **base_kwargs)
    val_loader = DataLoader(val_ds, sampler=val_sampler, batch_size=bs_val, drop_last=False, **base_kwargs)

    def set_epoch(epoch: int) -> None:
        try:
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"): train_loader.sampler.set_epoch(epoch)
        except Exception: pass
        try:
            if hasattr(val_loader, "sampler") and hasattr(val_loader.sampler, "set_epoch"): val_loader.sampler.set_epoch(epoch)
        except Exception: pass

    if use_dali:
        train_dali_loader = _build_dali_loader("train", train_ds, cfg, in_h=in_h, in_w=in_w, batch_size=bs, shuffle_source=False)
        val_dali_loader = _build_dali_loader("val", val_ds, cfg, in_h=in_h, in_w=in_w, batch_size=bs_val, shuffle_source=False)
    else:
        train_dali_loader, val_dali_loader = None, None

    meta = {
        "train_size": len(train_ds), "val_size": len(val_ds), "train_meta": meta_tr, "val_meta": meta_va,
        "shuffle_train": shuffle_train, "shuffle_val": shuffle_val, "sampler_seed_base": base_seed,
        "ddp": ddp, "world_size": world_size, "rank": rank, "dali": use_dali,
    }

    return dict(train_loader=train_loader, val_loader=val_loader, train_dali_loader=train_dali_loader, val_dali_loader=val_dali_loader, set_epoch=set_epoch, meta=meta)

__all__ = [
    "build_dataloaders",
    "seed_worker",
    "resolve_transforms",
    "resolve_dataset_class",
    "resolve_collate_fn",
]
