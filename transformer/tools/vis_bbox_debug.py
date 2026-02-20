#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vis_bbox_debug.py — Visualize BDD/COCO-style bboxes on your RAW4 (640x360) images.

Changes:
- Draw label as contiguous class index 0/1/2/3... instead of raw category_id or "xywh#i".
- Mapping rule (category_id -> class index):
    * If --class-id-order is given, use that exact order (e.g., "3,1,10,11" -> 3→0, 1→1, 10→2, 11→3).
    * Else if JSON has "categories": use sorted unique category ids from categories.
    * Else: scan all annotations, collect unique category_id and sort.
- Works for three JSON shapes:
    1) list of {id, annotations:[{bbox,category_id},...]}
    2) COCO-like {"images":..., "annotations":..., "categories":[{id,name},...]}
    3) dict[id] -> [{bbox,category_id}, ...]

Typical use (your case: RAW4 = 640x360, ann defined on 1280x720):
python vis_bbox_debug.py \
  --json /home/ryan529/project/bdd100k/labels/sem_seg/train_det.json \
  --raw_root /home/ryan529/project/bdd100k/raw10_npy \
  --idx 0 \
  --bbox_mode xywh \
  --src_size 1280x720 \
  --out vis_000.png
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# I/O helpers
# -------------------------------
def _load_raw_npy(npy_path: str) -> np.ndarray:
    arr = np.load(npy_path)
    if arr.ndim != 3:
        raise ValueError(f"Expect 3D array in {npy_path}, got {arr.shape}")
    if arr.shape[0] == 4:  # (4,H,W) -> (H,W,4)
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape[-1] != 4:
        raise ValueError(f"Expect 4 channels, got {arr.shape}")
    rgb = arr[..., :3].astype(np.float32)
    # normalize for display
    mn, mx = float(rgb.min()), float(rgb.max())
    if mx > mn:
        rgb = (rgb - mn) / (mx - mn)
    else:
        rgb = np.zeros_like(rgb)
    return rgb


def _find_npy(raw_root: str, img_id: str) -> str:
    # try raw_root/train/ID.npy, then raw_root/ID.npy
    p = os.path.join(raw_root, "train", f"{img_id}.npy")
    if os.path.isfile(p):
        return p
    p = os.path.join(raw_root, f"{img_id}.npy")
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"Cannot find NPY for id={img_id} under {raw_root}")


# -------------------------------
# JSON parsing
# -------------------------------
def _parse_coco_like(data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    # returns {id(str): [ {bbox:[x,y,w,h], category_id:int}, ... ]}
    id2name: Dict[str, str] = {}
    for im in data.get("images", []):
        if not isinstance(im, dict):
            continue
        sid = str(im.get("id", ""))
        name = os.path.splitext(os.path.basename(im.get("file_name", sid)))[0]
        id2name[sid] = name
    out: Dict[str, List[Dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        img_id = str(ann.get("image_id", ""))
        sid = id2name.get(img_id, img_id)
        bb = ann.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            bag = out.setdefault(sid, [])
            bag.append({"bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                        "category_id": int(ann.get("category_id", 0))})
    return out


def load_id_to_anns(json_path: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 3: dict[id] -> anns
    if isinstance(data, dict) and "images" not in data and all(
        isinstance(k, (str, int)) for k in data.keys()
    ):
        out: Dict[str, List[Dict[str, Any]]] = {}
        for k, v in data.items():
            sid = str(k)
            bag = []
            if isinstance(v, list):
                for a in v:
                    if not isinstance(a, dict):
                        continue
                    bb = a.get("bbox")
                    if isinstance(bb, (list, tuple)) and len(bb) == 4:
                        bag.append({"bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                                    "category_id": int(a.get("category_id", 0))})
            out[sid] = bag
        return out

    # Case 2: COCO-like
    if isinstance(data, dict) and "images" in data and "annotations" in data:
        return _parse_coco_like(data)

    # Case 1: list of {id, annotations:[...]}
    if isinstance(data, list):
        out: Dict[str, List[Dict[str, Any]]] = {}
        for it in data:
            if not isinstance(it, dict):
                continue
            sid = str(it.get("id", "") or os.path.splitext(os.path.basename(it.get("file_name", "")))[0])
            if not sid:
                continue
            bag = []
            for a in it.get("annotations", []):
                if not isinstance(a, dict):
                    continue
                bb = a.get("bbox")
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    bag.append({"bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                                "category_id": int(a.get("category_id", 0))})
            out[sid] = bag
        return out

    raise ValueError("Unsupported JSON structure")


def load_id_list(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids: List[str] = []
    if isinstance(data, dict) and "images" in data and "annotations" in data:
        for im in data.get("images", []):
            if isinstance(im, dict):
                sid = str(im.get("id", "")) or os.path.splitext(os.path.basename(im.get("file_name", "")))[0]
                ids.append(sid)
    elif isinstance(data, dict):
        ids = [str(k) for k in data.keys()]
    elif isinstance(data, list):
        for it in data:
            if isinstance(it, dict):
                sid = str(it.get("id", "")) or os.path.splitext(os.path.basename(it.get("file_name", "")))[0]
                if sid:
                    ids.append(sid)
    return ids


# -------------------------------
# Category-id → class-index mapping
# -------------------------------
def _collect_unique_cat_ids_from_annotations(id2anns: Dict[str, List[Dict[str, Any]]]) -> List[int]:
    s = set()
    for anns in id2anns.values():
        for a in anns:
            if isinstance(a, dict) and "category_id" in a:
                try:
                    s.add(int(a["category_id"]))
                except Exception:
                    pass
    return sorted(s)


def _load_categories_list(json_path: str) -> Optional[List[int]]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "categories" in data and isinstance(data["categories"], list):
            ids = []
            for c in data["categories"]:
                if isinstance(c, dict) and "id" in c:
                    ids.append(int(c["id"]))
            if ids:
                return sorted(set(ids))
    except Exception:
        pass
    return None


def _parse_id_order(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    try:
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        vals = [int(p) for p in parts]
        if not vals:
            return None
        return vals
    except Exception:
        return None


def build_catid_to_index_map(
    json_path: str,
    id2anns: Dict[str, List[Dict[str, Any]]],
    class_id_order: Optional[str],
) -> Dict[int, int]:
    """
    Returns a dict mapping category_id -> contiguous class index (0..K-1).
    Priority:
      1) --class-id-order "a,b,c,..."  → a→0, b→1, c→2 ...
      2) JSON.categories ids (sorted)
      3) unique category_id from annotations (sorted)
    """
    # 1) explicit order
    order = _parse_id_order(class_id_order)
    if order:
        return {cid: i for i, cid in enumerate(order)}

    # 2) categories
    cat_ids = _load_categories_list(json_path)
    if cat_ids:
        return {cid: i for i, cid in enumerate(cat_ids)}

    # 3) scan annotations
    uniq = _collect_unique_cat_ids_from_annotations(id2anns)
    return {cid: i for i, cid in enumerate(uniq)}


# -------------------------------
# Geometry
# -------------------------------
def _to_xyxy(
    bbox: List[float],
    mode: str,                 # "xywh" or "cxcywh"
    normalized: bool,
    src_wh: Tuple[int, int],   # e.g., (1280,720)
) -> Tuple[float, float, float, float]:
    x, y, w, h = map(float, bbox)
    srcW, srcH = src_wh

    if normalized:
        # values in 0..1 relative to src size
        x *= srcW; y *= srcH; w *= srcW; h *= srcH

    if mode == "xywh":
        x1, y1 = x, y
        x2, y2 = x + w, y + h
    elif mode == "cxcywh":
        x1, y1 = x - w / 2.0, y - h / 2.0
        x2, y2 = x + w / 2.0, y + h / 2.0
    else:
        raise ValueError("mode must be 'xywh' or 'cxcywh'")

    # clamp to src canvas just in case
    x1 = max(0.0, min(x1, srcW - 1.0))
    y1 = max(0.0, min(y1, srcH - 1.0))
    x2 = max(0.0, min(x2, srcW - 1.0))
    y2 = max(0.0, min(y2, srcH - 1.0))
    return x1, y1, x2, y2


def _scale_xyxy_to_target(
    xyxy: Tuple[float, float, float, float],
    src_wh: Tuple[int, int],
    tgt_wh: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    srcW, srcH = src_wh
    tgtW, tgtH = tgt_wh
    sx = tgtW / float(srcW)
    sy = tgtH / float(srcH)
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


# -------------------------------
# Drawing
# -------------------------------
def _draw_rect(ax, xyxy, color="r", label: Optional[str] = None, lw=2):
    x1, y1, x2, y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    rect = plt.Rectangle((x1, y1), w, h, edgecolor=color, facecolor="none", linewidth=lw)
    ax.add_patch(rect)
    if label is not None:
        ax.text(x1, max(0, y1 - 5), label, color=color, fontsize=9, weight="bold")


# -------------------------------
# Main visualize
# -------------------------------
def visualize(
    json_path: str,
    raw_root: str,
    img_id: Optional[str],
    idx: Optional[int],
    bbox_mode: str,
    normalized: bool,
    src_wh: Tuple[int, int],
    compare: bool,
    out_path: Optional[str],
    class_id_order: Optional[str],
):
    id2anns = load_id_to_anns(json_path)

    # build category_id -> contiguous index map
    catid2idx = build_catid_to_index_map(json_path, id2anns, class_id_order)
    print("[info] category_id → class_index mapping:")
    for cid in sorted(catid2idx.keys()):
        print(f"  {cid} → {catid2idx[cid]}")

    ids = list(id2anns.keys()) if img_id is None else [img_id]
    if not ids:
        raise RuntimeError("No ids found in json.")

    if img_id is None:
        # Map idx to actual id order
        all_ids = load_id_list(json_path)
        if not all_ids:
            all_ids = list(id2anns.keys())
        pick = all_ids[idx or 0]
    else:
        pick = img_id
    anns = id2anns.get(pick, [])
    if not anns:
        print(f"[WARN] No annotations for id={pick}")
    npy_path = _find_npy(raw_root, pick)
    img = _load_raw_npy(npy_path)
    H, W = img.shape[:2]
    print(f"[info] id={pick}  npy={os.path.basename(npy_path)}  img={W}x{H}  anns={len(anns)}")
    print(f"[info] src_wh={src_wh}, bbox_mode={bbox_mode}, normalized={normalized}, compare={compare}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    colors = ["r", "lime", "dodgerblue", "orange", "magenta", "cyan", "yellow", "white", "tab:purple", "tab:brown"]

    for i, a in enumerate(anns):
        if not isinstance(a, dict):
            continue
        bb = a.get("bbox", None)
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        cat_id = int(a.get("category_id", 0))
        cls_idx = catid2idx.get(cat_id, -1)   # -1 表示未被映射（顯示為 -1）

        if compare:
            # draw both interpretations for quick sanity
            xyxy_xy = _to_xyxy(bb, "xywh", normalized, src_wh)
            xyxy_xy = _scale_xyxy_to_target(xyxy_xy, src_wh, (W, H))
            _draw_rect(ax, xyxy_xy, color="r", label=f"{cls_idx}", lw=2)

            xyxy_c = _to_xyxy(bb, "cxcywh", normalized, src_wh)
            xyxy_c = _scale_xyxy_to_target(xyxy_c, src_wh, (W, H))
            _draw_rect(ax, xyxy_c, color="lime", label=f"{cls_idx}", lw=2)

            print(f"[box#{i}] raw={bb} | cat_id={cat_id} → idx={cls_idx} | "
                  f"xywh→xyxy@tgt={tuple(round(v,1) for v in xyxy_xy)} | "
                  f"cxcywh→xyxy@tgt={tuple(round(v,1) for v in xyxy_c)}")
        else:
            xyxy = _to_xyxy(bb, bbox_mode, normalized, src_wh)
            xyxy = _scale_xyxy_to_target(xyxy, src_wh, (W, H))
            color = colors[cls_idx % len(colors)] if cls_idx >= 0 else "white"
            _draw_rect(ax, xyxy, color=color, label=f"{cls_idx}", lw=2)
            print(f"[box#{i}] raw={bb} | cat_id={cat_id} → idx={cls_idx} | "
                  f"{bbox_mode}→xyxy@tgt={tuple(round(v,1) for v in xyxy)}")

    ax.set_title(f"id={pick}  ({W}x{H}), src={src_wh[0]}x{src_wh[1]}  mode={bbox_mode}{' (compare)' if compare else ''}")
    ax.axis("off")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"[save] {out_path}")
    else:
        plt.show()


def parse_wh(s: str) -> Tuple[int, int]:
    # "1280x720" -> (1280,720)
    if "x" not in s:
        raise argparse.ArgumentTypeError("Expected WxH, e.g., 1280x720")
    w, h = s.lower().split("x")
    return int(w), int(h)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, type=str, help="Path to train_det.json")
    p.add_argument("--raw_root", required=True, type=str, help="Root of raw10_npy (contains train/)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--id", type=str, help="Specific image id to visualize")
    g.add_argument("--idx", type=int, default=0, help="Index in json to visualize (if --id not given)")
    p.add_argument("--bbox_mode", type=str, default="xywh", choices=["xywh", "cxcywh"],
                   help="Interpretation of bbox in json")
    p.add_argument("--normalized", action="store_true", help="If bbox is in 0..1 normalized coordinates")
    p.add_argument("--src_size", type=parse_wh, default=(1280, 720),
                   help="Source resolution that bbox is defined on. Default 1280x720.")
    p.add_argument("--compare", action="store_true",
                   help="Draw both xywh (red) and cxcywh (lime) to compare")
    p.add_argument("--out", type=str, default=None, help="Save to path instead of showing")

    # NEW: mapping control
    p.add_argument("--class-id-order", type=str, default=None,
                   help="Comma list of category_id order mapping to indices 0..K-1, "
                        "e.g., '3,1,10,11' → 3→0,1→1,10→2,11→3. "
                        "If omitted, will auto-build from JSON.")

    args = p.parse_args()

    visualize(
        json_path=args.json,
        raw_root=args.raw_root,
        img_id=args.id,
        idx=args.idx,
        bbox_mode=args.bbox_mode,
        normalized=args.normalized,
        src_wh=args.src_size,
        compare=args.compare,
        out_path=args.out,
        class_id_order=args.class_id_order,
    )


if __name__ == "__main__":
    main()
