# -*- coding: utf-8 -*-
"""
detection_metrics.py - Detection metrics (hardware-aware detector)

提供：
- decode_detections_for_image()
- compute_detection_batch_metrics()
- AveragePrecisionMeter / CocoAveragePrecisionMeter / COCOMAPMeter

特點：
- 無外部依賴；尊重輸入 tensor 的 device/dtype；函式皆 @torch.no_grad。
- 統一座標系：GT 會被轉成「xyxy 像素座標、尺寸=size_hw」，避免 IoU 因解析度或格式不一致而失真。
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union, Iterable, Any

import torch
import torch.nn.functional as F

try:
    from torchvision.ops import batched_nms as _tv_batched_nms
    _HAS_TV_NMS = True
except Exception:
    _tv_batched_nms = None
    _HAS_TV_NMS = False

__all__ = [
    "decode_detections_for_image",
    "compute_detection_batch_metrics",
    "AveragePrecisionMeter",
    "CocoAveragePrecisionMeter",
    "COCOMAPMeter",
]

# =============================================================================
# Small helpers
# =============================================================================

@torch.no_grad()
def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    a = a.float(); b = b.float()
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp_min(0) * (a[:, 3] - a[:, 1]).clamp_min(0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0))[None, :]
    union = area_a + area_b - inter
    return inter / (union + 1e-7)


@torch.no_grad()
def _cxcywh_to_xyxy_norm(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=-1)


@torch.no_grad()
def _scale_to_pixels(xyxy_norm: torch.Tensor, size_hw: torch.Tensor) -> torch.Tensor:
    H = float(size_hw[0].item()); W = float(size_hw[1].item())
    out = xyxy_norm.clone()
    out[:, [0, 2]] *= W
    out[:, [1, 3]] *= H
    return out


def _infer_has_bg_and_C(
    logits_bq: torch.Tensor,
    num_classes: Optional[int],
    has_background: Optional[bool]
) -> Tuple[int, bool]:
    Clog = int(logits_bq.shape[-1])
    if num_classes is not None:
        if has_background is None:
            has_background = (Clog == num_classes + 1)
        C = int(num_classes)
        if Clog == num_classes:
            has_background = bool(False) if has_background is None else bool(has_background)
        return C, bool(has_background)
    if has_background is None:
        has_background = True if Clog >= 2 else False
    C = Clog - 1 if has_background else Clog
    return C, bool(has_background)


@torch.no_grad()
def _classwise_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """回傳輸入座標系的保留索引（原始索引）"""
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    if _HAS_TV_NMS:
        try:
            return _tv_batched_nms(
                boxes.float(),
                scores.float(),
                labels.to(dtype=torch.int64),
                float(iou_thr),
            )
        except Exception:
            pass

    order = torch.argsort(scores, descending=True)
    b_sorted = boxes[order]
    s_sorted = scores[order]
    l_sorted = labels[order]
    keep_abs: List[int] = []
    # 逐類別抑制
    for c in torch.unique(l_sorted).tolist():
        mask = (l_sorted == c)
        idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
        b = b_sorted[idxs]
        s = s_sorted[idxs]
        idxs_rel = idxs
        while b.numel() > 0:
            i = int(torch.argmax(s).item())
            keep_abs.append(int(order[idxs_rel[i]].item()))
            if b.shape[0] == 1:
                break
            iou = _box_iou_xyxy(b[i:i+1], b).squeeze(0)
            remain = iou <= float(iou_thr)
            b = b[remain]
            s = s[remain]
            idxs_rel = idxs_rel[remain]
    keep_abs = torch.tensor(keep_abs, device=boxes.device, dtype=torch.long)
    keep_abs = keep_abs[torch.argsort(scores[keep_abs], descending=True)]
    return keep_abs


def _normalize_iou_thrs(
    iou_thr: Union[float, Iterable[float]] = 0.5,
    iou_thresholds: Optional[Iterable[float]] = None
) -> List[float]:
    if iou_thresholds is not None:
        thrs = list(map(float, iou_thresholds))
    else:
        thrs = [float(iou_thr)] if isinstance(iou_thr, (int, float)) else list(map(float, iou_thr))
    return sorted(set(thrs))


@torch.no_grad()
def _to_hw_tuple(x: Any) -> Tuple[int, int]:
    """將 targets[...] 內可能是 list/tuple/tensor 的 size 轉成 (H, W)"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return int(x[0]), int(x[1])
    raise ValueError(f"Invalid size spec: {x}")


@torch.no_grad()
def _ensure_gt_xyxy_pixels(
    gt_boxes: torch.Tensor,
    *,
    gt_mode: str = "xyxy",          # "xyxy" | "xywh" | "cxcywh"
    gt_normed: bool = False,        # True 表示 [0,1]
    gt_size_hw: Optional[Tuple[int, int]] = None,  # 原始 GT 尺寸 (H,W)，若 gt_normed=False 且與 size_hw 不同就等比縮放
    size_hw: torch.Tensor           # 目標尺寸 (H,W) tensor on same device
) -> torch.Tensor:
    """
    將 GT 轉換到「xyxy 像素座標，尺寸=size_hw」。不在這裡做 clamp，保留原始標註資訊。
    """
    boxes = gt_boxes.float().clone()
    # 格式轉換到 xyxy（仍在原單位）
    if gt_mode == "xywh":
        x, y, w, h = boxes.unbind(-1)
        boxes = torch.stack([x, y, x + w, y + h], dim=-1)
    elif gt_mode == "cxcywh":
        cx, cy, w, h = boxes.unbind(-1)
        boxes = torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)
    elif gt_mode == "xyxy":
        pass
    else:
        raise ValueError(f"Unknown gt_mode: {gt_mode}")

    H = float(size_hw[0].item()); W = float(size_hw[1].item())
    if gt_normed:
        # [0,1] 正規化 → 像素
        boxes[:, [0, 2]] *= W
        boxes[:, [1, 3]] *= H
    elif gt_size_hw is not None:
        # 來源像素解析度若與目標不同，做線性縮放
        H0, W0 = float(gt_size_hw[0]), float(gt_size_hw[1])
        if abs(H0 - H) > 1e-3 or abs(W0 - W) > 1e-3:
            sx, sy = W / W0, H / H0
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
    return boxes


# =============================================================================
# 1) Decoding and batch metrics
# =============================================================================

@torch.no_grad()
def decode_detections_for_image(
    outputs: Dict[str, torch.Tensor],
    b: int,
    size_hw: torch.Tensor,
    *,
    score_thr: float = 0.05,
    max_dets: int = 300,
    num_classes: Optional[int] = None,
    has_background: Optional[bool] = None,
    nms_iou: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Return:
      boxes_px: (M,4) xyxy in pixels
      labels:   (M,)
      scores:   (M,)
    """
    logits = outputs["pred_logits"][b]  # (Q,C or C+1)
    boxes = outputs["pred_boxes"][b]    # (Q,4) cxcywh in [0,1]

    # 嘗試從 outputs metadata 推測
    if num_classes is None and "num_classes" in outputs:
        try:
            num_classes = int(outputs["num_classes"])
        except Exception:
            num_classes = None
    if has_background is None and "has_bg" in outputs:
        try:
            has_background = bool(outputs["has_bg"])
        except Exception:
            has_background = None

    C, has_bg = _infer_has_bg_and_C(logits, num_classes, has_background)

    probs = F.softmax(logits, dim=-1)
    if has_bg:
        conf, labels = probs[..., :C].max(dim=-1)
    else:
        conf, labels = probs.max(dim=-1)

    keep = conf > float(score_thr)
    if not torch.any(keep):
        device = boxes.device
        return {
            "boxes_px": boxes.new_zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.long, device=device),
            "scores": boxes.new_zeros((0,)),
        }

    conf = conf[keep]
    labels = labels[keep]
    boxes = boxes[keep]

    xyxy_norm = _cxcywh_to_xyxy_norm(boxes).clamp(0.0, 1.0)
    boxes_px = _scale_to_pixels(xyxy_norm, size_hw)

    if nms_iou is not None and boxes_px.numel() > 0:
        keep_nms = _classwise_nms(boxes_px, conf, labels, float(nms_iou))
        conf = conf[keep_nms]
        labels = labels[keep_nms]
        boxes_px = boxes_px[keep_nms]

    if max_dets and boxes_px.size(0) > max_dets:
        idx = torch.topk(conf, k=max_dets, dim=0).indices
        conf = conf[idx]
        labels = labels[idx]
        boxes_px = boxes_px[idx]

    return {"boxes_px": boxes_px, "labels": labels, "scores": conf}


@torch.no_grad()
def compute_detection_batch_metrics(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    *,
    iou_thr: Union[float, Iterable[float]] = 0.5,
    iou_thresholds: Optional[Iterable[float]] = None,
    score_thr: float = 0.05,
    max_dets: int = 300,
    num_classes: Optional[int] = None,
    has_background: Optional[bool] = None,
    nms_iou: Optional[float] = None,
    min_area_ratio: Optional[float] = None,
    area_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Union[float, torch.Tensor]]:
    """
    即時計算一個 batch 的 P/R/F1/mean-IoU（以分數門檻過濾）。
    多個 IoU 門檻也可以同時算，輸出鍵如 "precision@0.50"。
    需求：
      targets[b] 需提供：
        - "boxes": (Ng,4)
        - "labels": (Ng,)
        - "size": (2,) 目標影像尺寸 (H,W) 供 decode 對齊
      可選：
        - "box_mode": "xyxy" | "xywh" | "cxcywh"
        - "box_normalized": bool, 若 True 表示 GT 為 [0,1]
        - "source_size": (2,) 原始 GT 尺寸，若與 "size" 不同會線性縮放

      min_area_ratio / area_range (COCO-style):
        - 若 area_range is not None，使用 area_range = (amin, amax) in [0,1] for area/(H*W)。
        - 否則若 min_area_ratio is not None，使用 (min_area_ratio, 1.0)。
        - 否則不做面積篩選。
        - 行為仿 COCO：
            * 只將 area 在範圍內的 GT 視為正樣本。
            * 預測框若匹配到「範圍外」的 GT，會被忽略（不算 TP/FP）。
            * 未匹配的預測框，僅當其本身 area_ratio 在範圍內時才算 FP，否則忽略。
    """
    B = outputs["pred_logits"].shape[0]
    dev = outputs["pred_boxes"].device
    iou_thrs = _normalize_iou_thrs(iou_thr, iou_thresholds)

    if area_range is not None:
        amin, amax = float(area_range[0]), float(area_range[1])
    elif min_area_ratio is not None:
        amin, amax = float(min_area_ratio), 1.0
    else:
        amin, amax = None, None  # type: ignore[assignment]

    tp = {t: 0.0 for t in iou_thrs}
    fp = {t: 0.0 for t in iou_thrs}
    fn = {t: 0.0 for t in iou_thrs}
    iou_sum = {t: 0.0 for t in iou_thrs}
    iou_cnt = {t: 0.0 for t in iou_thrs}

    for b in range(B):
        size_hw = targets[b]["size"].to(dev)

        pred = decode_detections_for_image(
            outputs, b, size_hw,
            score_thr=score_thr, max_dets=max_dets,
            num_classes=num_classes, has_background=has_background,
            nms_iou=nms_iou
        )

        # ---- 統一 GT 座標系 ----
        gt_boxes_raw = targets[b]["boxes"].to(dev)
        gt_mode = targets[b].get("box_mode", "xyxy")
        gt_normed = bool(targets[b].get("box_normalized", False))
        src_hw = targets[b].get("source_size", None)
        if src_hw is not None:
            src_hw = _to_hw_tuple(src_hw)
        gt_boxes = _ensure_gt_xyxy_pixels(
            gt_boxes_raw,
            gt_mode=gt_mode,
            gt_normed=gt_normed,
            gt_size_hw=src_hw,
            size_hw=size_hw,
        )
        gt_labels = targets[b]["labels"].to(dev).long()

        # 面積比例（GT）
        if gt_boxes.numel() > 0:
            H = float(size_hw[0].item()); W = float(size_hw[1].item())
            gt_areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]).clamp_min(0) *
                        (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp_min(0))
            denom = max(1.0, H * W)
            gt_area_ratio = gt_areas / denom
        else:
            gt_area_ratio = torch.zeros(0, device=dev)

        if amin is not None:
            gt_valid = (gt_area_ratio >= amin) & (gt_area_ratio < amax)
        else:
            gt_valid = torch.ones_like(gt_area_ratio, dtype=torch.bool)

        # 面積比例（pred）
        if pred["boxes_px"].numel() > 0:
            H = float(size_hw[0].item()); W = float(size_hw[1].item())
            p_boxes_all = pred["boxes_px"]
            p_labels_all = pred["labels"]
            p_scores_all = pred["scores"]
            p_areas = ((p_boxes_all[:, 2] - p_boxes_all[:, 0]).clamp_min(0) *
                       (p_boxes_all[:, 3] - p_boxes_all[:, 1]).clamp_min(0))
            denom = max(1.0, H * W)
            p_area_ratio = p_areas / denom

            # 依分數排序
            order = torch.argsort(p_scores_all, descending=True)
            p_boxes = p_boxes_all[order]
            p_labels = p_labels_all[order]
            p_scores = p_scores_all[order]
            p_area_ratio = p_area_ratio[order]
        else:
            p_boxes = pred["boxes_px"]
            p_labels = pred["labels"]
            p_scores = pred["scores"]
            p_area_ratio = torch.zeros(0, device=dev)

        if p_boxes.numel() == 0 and gt_boxes.numel() == 0:
            continue

        for thr in iou_thrs:
            used = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=dev)

            for i in range(p_boxes.shape[0]):
                # 若有面積範圍，且 pred 自身不在範圍內 → 忽略
                if amin is not None:
                    if not (amin <= float(p_area_ratio[i].item()) < amax):
                        continue

                cls = int(p_labels[i].item())
                gt_mask_cls = (gt_labels == cls)
                if not gt_mask_cls.any():
                    # 該類別沒有 GT：若 pred 在範圍內，算 FP
                    fp[thr] += 1.0
                    continue

                cand_idx = torch.nonzero(gt_mask_cls, as_tuple=False).squeeze(1)
                free_mask = ~used[cand_idx]
                if not torch.any(free_mask):
                    fp[thr] += 1.0
                    continue
                cand_idx = cand_idx[free_mask]
                ious = _box_iou_xyxy(p_boxes[i:i+1], gt_boxes[cand_idx]).squeeze(0)
                j_rel = int(torch.argmax(ious).item())
                best = float(ious[j_rel].item())
                j = int(cand_idx[j_rel].item())
                if best >= float(thr):
                    # 若匹配到的是範圍外的 GT，則忽略這個 pred
                    if not gt_valid[j]:
                        continue
                    tp[thr] += 1.0
                    used[j] = True
                    iou_sum[thr] += best
                    iou_cnt[thr] += 1.0
                else:
                    fp[thr] += 1.0

            # FN：未被匹配且屬於範圍內的 GT
            if gt_boxes.numel() > 0:
                if amin is not None:
                    fn_mask = (~used) & gt_valid
                else:
                    fn_mask = ~used
                fn[thr] += float(fn_mask.sum().item())

    out: Dict[str, Union[float, torch.Tensor]] = {}
    for thr in iou_thrs:
        p = tp[thr] / (tp[thr] + fp[thr] + 1e-9)
        r = tp[thr] / (tp[thr] + fn[thr] + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        miou = (iou_sum[thr] / max(1.0, iou_cnt[thr])) if iou_cnt[thr] > 0 else 0.0
        tag = f"{thr:.2f}".rstrip("0").rstrip(".")
        out[f"precision@{tag}"] = float(p)
        out[f"recall@{tag}"] = float(r)
        out[f"f1@{tag}"] = float(f1)
        out[f"mean_iou@{tag}"] = float(miou)
        out[f"tp@{tag}"] = float(tp[thr])
        out[f"fp@{tag}"] = float(fp[thr])
        out[f"fn@{tag}"] = float(fn[thr])

    base = iou_thrs[0]
    base_tag = f"{base:.2f}".rstrip("0").rstrip(".")
    out.update({
        "precision": torch.tensor(out[f"precision@{base_tag}"]),
        "recall": torch.tensor(out[f"recall@{base_tag}"]),
        "f1": torch.tensor(out[f"f1@{base_tag}"]),
        "mean_iou": torch.tensor(out[f"mean_iou@{base_tag}"]),
        "score_thr": float(score_thr),
    })
    if amin is not None:
        out["min_area_ratio"] = float(amin)
        out["area_range"] = (float(amin), float(amax))
    else:
        out["min_area_ratio"] = 0.0
        out["area_range"] = None
    return out


# =============================================================================
# 2) mAP over a dataset
# =============================================================================

@torch.no_grad()
def _voc_ap(rec: torch.Tensor, prec: torch.Tensor) -> float:
    """連續積分 AP（非 11-point）"""
    mrec = torch.cat([torch.tensor([0.0], device=rec.device), rec, torch.tensor([1.0], device=rec.device)])
    mpre = torch.cat([torch.tensor([0.0], device=prec.device), prec, torch.tensor([0.0], device=prec.device)])
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
    idx = torch.nonzero(mrec[1:] != mrec[:-1], as_tuple=False).squeeze(1)
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap.item())


class AveragePrecisionMeter:
    """
    累積整個驗證集的預測，再計算 AP / mAP。
    使用：
        meter = AveragePrecisionMeter(num_classes=9, has_background=True)
        for batch:
            meter.update(outputs, targets, score_thr=0.05, nms_iou=0.5)
        res = meter.compute(iou_thrs=0.5 或 [0.5,0.75] 或 iou_thresholds=[...])
        -> {
             "ap_per_class": Tensor[T, C],
             "map": Tensor[T],
             "num_gts": Tensor[C],
             "num_preds": Tensor[C],
             "iou_thrs": Tensor[T],
             "area_range": (amin, amax) or None,
           }

    注意：
        - 本類別假設 label 已是連續 0..C-1 索引。若資料集原始 ID 稀疏，請先在 dataloader 做映射。
        - area_range / min_area_ratio 行為仿 COCO：
            * 只計算 area_ratio in [amin, amax) 的 GT。
            * 預測框若匹配到範圍外 GT，則不計入 TP/FP（ignored）。
            * 未匹配的預測框僅在自身 area_ratio ∈ [amin, amax) 時算 FP。
    """
    def __init__(
        self,
        num_classes: int,
        has_background: bool = True,
        *,
        min_area_ratio: Optional[float] = None,
        area_range: Optional[Tuple[float, float]] = None,
    ):
        self.C = int(num_classes)
        self.has_bg = bool(has_background)

        if area_range is not None:
            amin, amax = float(area_range[0]), float(area_range[1])
            self.area_range: Optional[Tuple[float, float]] = (amin, amax)
        elif min_area_ratio is not None:
            amin = float(min_area_ratio)
            self.area_range = (amin, 1.0)
        else:
            self.area_range = None

        # per class: list of (img_id, score, box_xyxy_px[tensor(4)], area_ratio[float])
        self.preds: List[list] = [[] for _ in range(self.C)]
        # per class: img_id -> {"boxes": [Tensor[Ni,4], ...], "areas": [Tensor[Ni], ...]}
        self.gts: List[dict] = [dict() for _ in range(self.C)]
        self._img_uid = 0

    @torch.no_grad()
    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        *,
        score_thr: float = 0.0,
        max_dets: int = 300,
        nms_iou: Optional[float] = None,
    ):
        B = outputs["pred_logits"].shape[0]
        dev = outputs["pred_boxes"].device
        for b in range(B):
            size_hw = targets[b]["size"].to(dev)

            pred = decode_detections_for_image(
                outputs, b, size_hw,
                score_thr=score_thr, max_dets=max_dets,
                num_classes=self.C, has_background=self.has_bg,
                nms_iou=nms_iou
            )

            H = float(size_hw[0].item()); W = float(size_hw[1].item())
            denom = max(1.0, H * W)

            # image id
            if "id" in targets[b]:
                img_id = str(targets[b]["id"])
            elif "meta" in targets[b] and "id" in targets[b]["meta"]:
                img_id = str(targets[b]["meta"]["id"])
            else:
                img_id = f"img_{self._img_uid}"
                self._img_uid += 1

            # store preds
            if pred["boxes_px"].numel() > 0:
                p_boxes = pred["boxes_px"]
                p_labels = pred["labels"]
                p_scores = pred["scores"]
                p_areas = ((p_boxes[:, 2] - p_boxes[:, 0]).clamp_min(0) *
                           (p_boxes[:, 3] - p_boxes[:, 1]).clamp_min(0))
                p_area_ratio = (p_areas / denom).tolist()
                for box, label, score, ar in zip(p_boxes, p_labels, p_scores, p_area_ratio):
                    c = int(max(0, min(self.C - 1, int(label.item()))))
                    self.preds[c].append((img_id, float(score.item()), box.detach().cpu(), float(ar)))

            # ---- 統一 GT 座標系並存入 ----
            gt_boxes_raw = targets[b]["boxes"]
            gt_mode = targets[b].get("box_mode", "xyxy")
            gt_normed = bool(targets[b].get("box_normalized", False))
            src_hw = _to_hw_tuple(targets[b].get("source_size", targets[b]["size"]))
            gt_boxes_px = _ensure_gt_xyxy_pixels(
                gt_boxes_raw,
                gt_mode=gt_mode,
                gt_normed=gt_normed,
                gt_size_hw=src_hw,
                size_hw=size_hw,
            )
            gt_labels = targets[b]["labels"].to(gt_boxes_px.device).long()

            if gt_boxes_px.numel() > 0:
                gt_areas = ((gt_boxes_px[:, 2] - gt_boxes_px[:, 0]).clamp_min(0) *
                            (gt_boxes_px[:, 3] - gt_boxes_px[:, 1]).clamp_min(0))
                gt_area_ratio = gt_areas / denom
            else:
                gt_area_ratio = torch.zeros(0, device=gt_boxes_px.device)

            gt_boxes_px = gt_boxes_px.detach().cpu()
            gt_area_ratio = gt_area_ratio.detach().cpu()
            gt_labels = gt_labels.detach().cpu().long()

            for c in range(self.C):
                mask = (gt_labels == c)
                if mask.any():
                    boxes_c = gt_boxes_px[mask]
                    areas_c = gt_area_ratio[mask]
                    entry = self.gts[c].setdefault(img_id, {"boxes": [], "areas": []})
                    entry["boxes"].append(boxes_c)
                    entry["areas"].append(areas_c)

    @torch.no_grad()
    def compute(
        self,
        iou_thrs: Union[float, Iterable[float]] = 0.5,
        *,
        iou_thresholds: Optional[Iterable[float]] = None
    ) -> Dict[str, object]:
        iou_thrs_l = _normalize_iou_thrs(iou_thrs, iou_thresholds)
        T = len(iou_thrs_l)
        ap = torch.zeros((T, self.C), dtype=torch.float32)
        num_gts = torch.zeros(self.C, dtype=torch.int64)
        num_preds = torch.zeros(self.C, dtype=torch.int64)

        if self.area_range is not None:
            amin, amax = float(self.area_range[0]), float(self.area_range[1])
        else:
            amin, amax = None, None  # type: ignore[assignment]

        for c in range(self.C):
            # gather gt per image
            gt_by_img: Dict[str, Dict[str, torch.Tensor]] = {}
            for img_id, data in self.gts[c].items():
                boxes_list: List[torch.Tensor] = data.get("boxes", [])
                areas_list: List[torch.Tensor] = data.get("areas", [])
                if len(boxes_list) == 0:
                    continue
                boxes_img = torch.cat(boxes_list, dim=0)  # (Ng,4)
                areas_img = torch.cat(areas_list, dim=0)  # (Ng,)
                if amin is not None:
                    valid_mask = (areas_img >= amin) & (areas_img < amax)
                else:
                    valid_mask = torch.ones_like(areas_img, dtype=torch.bool)

                if valid_mask.any():
                    gt_by_img[img_id] = {
                        "boxes": boxes_img,
                        "valid": valid_mask,
                    }
                    num_gts[c] += int(valid_mask.sum().item())

            # sort preds by score
            preds = self.preds[c]
            if len(preds) == 0 or num_gts[c].item() == 0:
                # 若該類別沒有 GT 或沒有 pred，則 AP=0
                ap[:, c] = torch.tensor(0.0)
                continue

            det_img_ids = [p[0] for p in preds]
            det_scores = torch.tensor([p[1] for p in preds], dtype=torch.float32)
            det_boxes = torch.stack([p[2] for p in preds], dim=0).float()
            det_areas = torch.tensor([p[3] for p in preds], dtype=torch.float32)

            # 面積範圍內的 pred 數量（僅作為統計資訊）
            if amin is not None:
                dt_mask_range = (det_areas >= amin) & (det_areas < amax)
                num_preds[c] = int(dt_mask_range.sum().item())
            else:
                num_preds[c] = len(preds)

            # 依分數排序
            order = torch.argsort(det_scores, descending=True)
            det_img_ids = [det_img_ids[i] for i in order.tolist()]
            det_scores_sorted = det_scores[order]
            det_boxes_sorted = det_boxes[order]
            det_areas_sorted = det_areas[order]

            for ti, thr in enumerate(iou_thrs_l):
                tp = torch.zeros(len(preds), dtype=torch.float32)
                fp = torch.zeros(len(preds), dtype=torch.float32)
                matched: Dict[str, set] = {}  # per-image used gt indices

                for rank_idx, det_idx in enumerate(order.tolist()):
                    img_id = det_img_ids[rank_idx]
                    b = det_boxes_sorted[rank_idx:rank_idx+1]
                    dt_area = float(det_areas_sorted[rank_idx].item())

                    gdata = gt_by_img.get(img_id, None)
                    if gdata is None:
                        # 此 image 在本 area_range 下沒有 GT
                        if amin is not None:
                            if amin <= dt_area < amax:
                                fp[det_idx] = 1.0
                        else:
                            fp[det_idx] = 1.0
                        continue

                    gts = gdata["boxes"]
                    valid_mask = gdata["valid"]
                    used = matched.setdefault(img_id, set())

                    if gts.numel() == 0:
                        if amin is not None:
                            if amin <= dt_area < amax:
                                fp[det_idx] = 1.0
                        else:
                            fp[det_idx] = 1.0
                        continue

                    ious = _box_iou_xyxy(b, gts).squeeze(0)

                    # 排除已配對 GT
                    if len(used) > 0:
                        used_idx = torch.tensor(list(used), dtype=torch.long, device=ious.device)
                        ious[used_idx] = -1.0

                    j = int(torch.argmax(ious).item())
                    best = float(ious[j].item())
                    if best < float(thr):
                        # IoU 不足：僅當 pred 在 area_range 內才算 FP
                        if amin is not None:
                            if amin <= dt_area < amax:
                                fp[det_idx] = 1.0
                        else:
                            fp[det_idx] = 1.0
                        continue

                    # IoU >= thr，檢查 GT 是否在 area_range 內
                    if amin is not None and not bool(valid_mask[j].item()):
                        # 匹配到的 GT 不在範圍內 → 忽略此 pred
                        continue

                    # 正常 TP
                    tp[det_idx] = 1.0
                    used.add(j)

                # 累積並算 AP
                idx = torch.argsort(det_scores, descending=True)
                tp_cum = torch.cumsum(tp[idx], dim=0)
                fp_cum = torch.cumsum(fp[idx], dim=0)
                rec = tp_cum / (float(num_gts[c].item()) + 1e-9)
                prec = tp_cum / (tp_cum + fp_cum + 1e-9)
                ap[ti, c] = torch.tensor(_voc_ap(rec, prec))

        map_per_thr = ap.mean(dim=1) if self.C > 0 else torch.zeros(T)
        return {
            "ap_per_class": ap,         # [T, C]
            "map": map_per_thr,         # [T]
            "num_gts": num_gts,         # [C]
            "num_preds": num_preds,     # [C]
            "iou_thrs": torch.tensor(iou_thrs_l, dtype=torch.float32),
            "area_range": self.area_range,
        }


class CocoAveragePrecisionMeter(AveragePrecisionMeter):
    """COCO 風格 mAP：IoU=0.50:0.95, step=0.05"""
    def compute(self) -> Dict[str, object]:  # type: ignore[override]
        iou_thrs = [0.5 + 0.05 * i for i in range(10)]
        return super().compute(iou_thrs=iou_thrs)


# 舊名別名（在偵測這側保留一下，方便切檔後不動呼叫端）
COCOMAPMeter = CocoAveragePrecisionMeter


# =============================================================================
# quick self-test
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    # --- Detection quick test ---
    B, Q, C = 2, 100, 5
    H_src, W_src = 720, 1280     # 假設 GT 來源解析度
    H_tgt, W_tgt = 360, 640      # 模型實際輸入/輸出解析度
    det_out = {
        "pred_logits": torch.randn(B, Q, C + 1),  # with background
        "pred_boxes": torch.rand(B, Q, 4),        # cxcywh in [0,1]
        "has_bg": True,
        "num_classes": C,
    }
    targets = []
    for b in range(B):
        n = 6
        # 先在來源解析度產生 xyxy 像素
        boxes_src = torch.rand(n, 4)
        boxes_src[:, 2:] = boxes_src[:, :2] + boxes_src[:, 2:] * 0.2
        boxes_src = (boxes_src * torch.tensor([W_src, H_src, W_src, H_src])).clamp(0, max(H_src, W_src) - 1)
        labels = torch.randint(0, C, (n,))
        targets.append({
            "boxes": boxes_src, "labels": labels,
            "size": torch.tensor([H_tgt, W_tgt]),            # 模型輸入/輸出尺寸
            "source_size": torch.tensor([H_src, W_src]),     # GT 來源尺寸
            "box_mode": "xyxy", "box_normalized": False,
            "id": f"img{b}"
        })

    m_det = compute_detection_batch_metrics(
        det_out, targets,
        iou_thr=[0.5, 0.75],
        score_thr=0.05,
        nms_iou=0.5,
    )
    print("[det] sample:", {k: (float(v) if isinstance(v, torch.Tensor) else v) for k, v in list(m_det.items())[:6]})

    # 一般 mAP
    meter = AveragePrecisionMeter(num_classes=C, has_background=True)
    meter.update(det_out, targets, score_thr=0.05, nms_iou=0.5)
    print("[mAP] keys:", list(meter.compute(iou_thrs=[0.5, 0.75]).keys()))

    # 大物體 mAP50（示意：area_ratio >= 5%）
    meter_large = AveragePrecisionMeter(num_classes=C, has_background=True, min_area_ratio=0.05)
    meter_large.update(det_out, targets, score_thr=0.05, nms_iou=0.5)
    print("[mAP-large@0.5] map:", meter_large.compute(iou_thrs=0.5)["map"])
