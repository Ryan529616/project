# -*- coding: utf-8 -*-
"""
detection_losses_core.py - Hybrid matching detection loss core
===================================================================================
Features:
1. Batched Hungarian Matching: 消除逐張圖片的 GPU->CPU 死亡同步，整個 Batch 只同步一次。
2. OneToManyMatcher (Hybrid Matching) for faster convergence.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "box_xyxy_to_cxcywh",
    "box_cxcywh_to_xyxy",
    "OneToOneHungarianMatcher",
    "OneToManyMatcher",
    "DETRDetectionLoss",
    "build_detr_detection_loss",
]

_EPS = 1e-6


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = x.unbind(-1)
    w  = (x2 - x1).clamp_min(1e-5)
    h  = (y2 - y1).clamp_min(1e-5)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return torch.stack((cx, cy, w, h), dim=-1)


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    hw = 0.5 * w
    hh = 0.5 * h
    return torch.stack((cx - hw, cy - hh, cx + hw, cy + hh), dim=-1)


def _pairwise_giou(q_xyxy: torch.Tensor, t_xyxy: torch.Tensor) -> torch.Tensor:
    if q_xyxy.numel() == 0 or t_xyxy.numel() == 0:
        return q_xyxy.new_zeros((q_xyxy.size(0), t_xyxy.size(0)))

    q = q_xyxy.float()[:, None, :]
    t = t_xyxy.float()[None, :, :]

    lt = torch.maximum(q[..., :2], t[..., :2])
    rb = torch.minimum(q[..., 2:], t[..., 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]

    area_q = (q[..., 2] - q[..., 0]).clamp_min(0) * (q[..., 3] - q[..., 1]).clamp_min(0)
    area_t = (t[..., 2] - t[..., 0]).clamp_min(0) * (t[..., 3] - t[..., 1]).clamp_min(0)

    union = (area_q + area_t - inter).clamp_min(_EPS)
    iou = inter / union

    lt_en = torch.minimum(q[..., :2], t[..., :2])
    rb_en = torch.maximum(q[..., 2:], t[..., 2:])
    wh_en = (rb_en - lt_en).clamp_min(0)
    area_c = (wh_en[..., 0] * wh_en[..., 1]).clamp_min(_EPS)

    giou = iou - (area_c - union) / area_c
    return giou.clamp(-1.0, 1.0)


def _pairwise_iou(q_xyxy: torch.Tensor, t_xyxy: torch.Tensor) -> torch.Tensor:
    if q_xyxy.numel() == 0 or t_xyxy.numel() == 0:
        return q_xyxy.new_zeros((q_xyxy.size(0), t_xyxy.size(0)))

    q = q_xyxy.float()[:, None, :]
    t = t_xyxy.float()[None, :, :]

    lt = torch.maximum(q[..., :2], t[..., :2])
    rb = torch.minimum(q[..., 2:], t[..., 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]

    area_q = (q[..., 2] - q[..., 0]).clamp_min(0) * (q[..., 3] - q[..., 1]).clamp_min(0)
    area_t = (t[..., 2] - t[..., 0]).clamp_min(0) * (t[..., 3] - t[..., 1]).clamp_min(0)
    union = (area_q + area_t - inter).clamp_min(_EPS)
    return (inter / union).clamp(0.0, 1.0)


def _to_size_hw_tensor(
    size_hw: Optional[Union[torch.Tensor, List[int], Tuple[int, int]]],
    device: torch.device,
) -> torch.Tensor:
    if isinstance(size_hw, torch.Tensor):
        t = size_hw.to(device=device, dtype=torch.float32).view(-1)
        if t.numel() >= 2:
            return t[:2].contiguous()
        return torch.tensor([1.0, 1.0], device=device, dtype=torch.float32)
    if isinstance(size_hw, (list, tuple)) and len(size_hw) >= 2:
        return torch.tensor([float(size_hw[0]), float(size_hw[1])], device=device, dtype=torch.float32)
    return torch.tensor([1.0, 1.0], device=device, dtype=torch.float32)


def _normalize_targets_xyxy(tgt_xyxy: torch.Tensor, size_hw: torch.Tensor) -> torch.Tensor:
    if tgt_xyxy.numel() == 0:
        return tgt_xyxy
    dev = tgt_xyxy.device
    t = tgt_xyxy.to(dev).float()

    if t.max() <= 1.05:
        return t.clamp(0.0, 1.0)

    H = size_hw[0]
    W = size_hw[1] if size_hw.numel() > 1 else H
    scale = torch.stack([W, H, W, H]).clamp_min(1.0)
    return (t / scale).clamp(0.0, 1.0)


def _boxes_iou_diag(b1_xyxy: torch.Tensor, b2_xyxy: torch.Tensor) -> torch.Tensor:
    if b1_xyxy.numel() == 0:
        return b1_xyxy.new_zeros((0,))
    x1_1, y1_1, x2_1, y2_1 = b1_xyxy.unbind(-1)
    x1_2, y1_2, x2_2, y2_2 = b2_xyxy.unbind(-1)

    xx1 = torch.maximum(x1_1, x1_2)
    yy1 = torch.maximum(y1_1, y1_2)
    xx2 = torch.minimum(x2_1, x2_2)
    yy2 = torch.minimum(y2_1, y2_2)

    inter_w = (xx2 - xx1).clamp_min(0)
    inter_h = (yy2 - yy1).clamp_min(0)
    inter = inter_w * inter_h

    area1 = (x2_1 - x1_1).clamp_min(0) * (y2_1 - y1_1).clamp_min(0)
    area2 = (x2_2 - x1_2).clamp_min(0) * (y2_2 - y1_2).clamp_min(0)
    union = (area1 + area2 - inter).clamp_min(_EPS)
    iou = inter / union
    return iou.clamp(0.0, 1.0)


def _boxes_siou_diag(b1_xyxy: torch.Tensor, b2_xyxy: torch.Tensor) -> torch.Tensor:
    if b1_xyxy.numel() == 0:
        return b1_xyxy.new_zeros((0,))

    x1_1, y1_1, x2_1, y2_1 = b1_xyxy.unbind(-1)
    x1_2, y1_2, x2_2, y2_2 = b2_xyxy.unbind(-1)

    xx1 = torch.maximum(x1_1, x1_2)
    yy1 = torch.maximum(y1_1, y1_2)
    xx2 = torch.minimum(x2_1, x2_2)
    yy2 = torch.minimum(y2_1, y2_2)

    inter_w = (xx2 - xx1).clamp_min(0)
    inter_h = (yy2 - yy1).clamp_min(0)
    inter = inter_w * inter_h

    area1 = (x2_1 - x1_1).clamp_min(0) * (y2_1 - y1_1).clamp_min(0)
    area2 = (x2_2 - x1_2).clamp_min(0) * (y2_2 - y1_2).clamp_min(0)
    union = (area1 + area2 - inter).clamp_min(_EPS)
    iou = (inter / union).clamp(0.0, 1.0)

    cx1 = 0.5 * (x1_1 + x2_1)
    cy1 = 0.5 * (y1_1 + y2_1)
    cx2 = 0.5 * (x1_2 + x2_2)
    cy2 = 0.5 * (y1_2 + y2_2)

    center_dist = torch.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    c_x1 = torch.minimum(x1_1, x1_2)
    c_y1 = torch.minimum(y1_1, y1_2)
    c_x2 = torch.maximum(x2_1, x2_2)
    c_y2 = torch.maximum(y2_1, y2_2)
    diag_len = torch.sqrt((c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2).clamp_min(_EPS)

    center_term = (center_dist / diag_len).clamp(0.0, 1.0)

    w1 = (x2_1 - x1_1).clamp_min(_EPS)
    h1 = (y2_1 - y1_1).clamp_min(_EPS)
    w2 = (x2_2 - x1_2).clamp_min(_EPS)
    h2 = (y2_2 - y1_2).clamp_min(_EPS)

    w_min = torch.minimum(w1, w2)
    w_max = torch.maximum(w1, w2)
    h_min = torch.minimum(h1, h2)
    h_max = torch.maximum(h1, h2)

    shape_score = (w_min / w_max) * (h_min / h_max)
    shape_term = (1.0 - shape_score).clamp(0.0, 1.0)

    siou = iou - 0.5 * (center_term + shape_term)
    return siou.clamp(0.0, 1.0)


class OneToOneHungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        *,
        box_encoding: str = "sigmoid",
        iou_type: str = "giou",
        logit_temp: float = 1.0,
        batch_cpu_copy_max_numel: int = 8_000_000,
    ):
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.box_encoding = str(box_encoding).lower()
        self.iou_type = str(iou_type).lower()
        self.logit_temp = float(max(logit_temp, 1e-6))
        self.batch_cpu_copy_max_numel = int(max(1, batch_cpu_copy_max_numel))

        try:
            from scipy.optimize import linear_sum_assignment
            self.linear_sum_assignment = linear_sum_assignment
            self._has_scipy = True
        except Exception:
            self._has_scipy = False

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        logits = torch.nan_to_num(outputs["pred_logits"], nan=0.0).float()
        boxes_raw = outputs["pred_boxes"].float()

        if self.box_encoding == "logits":
            pb_cxcywh = torch.sigmoid(boxes_raw)
        else:
            pb_cxcywh = boxes_raw.clamp(0.0, 1.0)

        logp = F.log_softmax(logits / self.logit_temp, dim=-1)
        B, Q, C_plus_bg = logits.shape

        out: List[Tuple[torch.Tensor, torch.Tensor]] = [(None, None)] * B
        
        # 【效能優化】：Batched CPU Sync
        cmats_list = []
        valid_indices = []

        dev = pb_cxcywh.device

        for b in range(B):
            lp = logp[b]
            pb = pb_cxcywh[b]

            tgt_ids = targets[b]["labels"].to(dev).long()
            if tgt_ids.numel() == 0:
                empty = lp.new_empty((0,), dtype=torch.long)
                out[b] = (empty, empty)
                continue

            tgt_ids = tgt_ids.clamp(0, C_plus_bg - 2)
            tgt_px = targets[b]["boxes"].to(dev).float()
            size_hw = _to_size_hw_tensor(targets[b].get("size", None), device=dev)

            txy_norm = _normalize_targets_xyxy(tgt_px, size_hw)
            t_cxcywh = box_xyxy_to_cxcywh(txy_norm)

            c_cls = -lp[:, tgt_ids]
            c_l1 = torch.cdist(pb, t_cxcywh, p=1)

            q_xyxy = box_cxcywh_to_xyxy(pb).clamp(0.0, 1.0)
            if self.iou_type == "iou":
                c_iou = 1.0 - _pairwise_iou(q_xyxy, txy_norm)
            else:
                c_iou = 1.0 - _pairwise_giou(q_xyxy, txy_norm)

            Cmat = (self.cost_class * c_cls + self.cost_bbox * c_l1 + self.cost_giou * c_iou)
            Cmat = torch.nan_to_num(Cmat, nan=1e6, posinf=1e6, neginf=1e6)

            cmats_list.append(Cmat)
            valid_indices.append(b)

        if not cmats_list:
            return out

        # 小矩陣走單次扁平拷貝；大矩陣分段拷貝，避免一次性記憶體尖峰
        if self._has_scipy:
            sizes = [c.numel() for c in cmats_list]
            total_numel = sum(sizes)
            if total_numel <= self.batch_cpu_copy_max_numel:
                flat_cmat = torch.cat([c.reshape(-1) for c in cmats_list], dim=0).detach().float().cpu().numpy()

                offset = 0
                for i, b in enumerate(valid_indices):
                    shape = cmats_list[i].shape
                    numel = sizes[i]
                    cmat_b_cpu = flat_cmat[offset: offset + numel].reshape(shape)
                    offset += numel

                    ri, ci = self.linear_sum_assignment(cmat_b_cpu)
                    out[b] = (
                        torch.as_tensor(ri, dtype=torch.long, device=dev),
                        torch.as_tensor(ci, dtype=torch.long, device=dev),
                    )
            else:
                for i, b in enumerate(valid_indices):
                    cmat_b_cpu = cmats_list[i].detach().float().cpu().numpy()
                    ri, ci = self.linear_sum_assignment(cmat_b_cpu)
                    out[b] = (
                        torch.as_tensor(ri, dtype=torch.long, device=dev),
                        torch.as_tensor(ci, dtype=torch.long, device=dev),
                    )
        else:
            for i, b in enumerate(valid_indices):
                out[b] = self._greedy(cmats_list[i])

        return out

    @staticmethod
    def _greedy(C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Q, N = C.shape
        dev = C.device
        used_q = torch.zeros(Q, dtype=torch.bool, device=dev)
        used_t = torch.zeros(N, dtype=torch.bool, device=dev)
        qs, ts = [], []
        for _ in range(min(Q, N)):
            masked = C.masked_fill(used_q[:, None], float("inf")).masked_fill(used_t[None, :], float("inf"))
            if not torch.isfinite(masked).any(): break
            idx = masked.argmin()
            q, t = (idx // N).item(), (idx % N).item()
            used_q[q], used_t[t] = True, True
            qs.append(q)
            ts.append(t)
        if not qs:
            empty = torch.empty(0, dtype=torch.long, device=dev)
            return empty, empty
        return torch.as_tensor(qs, dtype=torch.long, device=dev), torch.as_tensor(ts, dtype=torch.long, device=dev)


class OneToManyMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        k: int = 6,
        *,
        box_encoding: str = "sigmoid",
        iou_type: str = "giou",
        logit_temp: float = 1.0,
    ):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou, self.k = float(cost_class), float(cost_bbox), float(cost_giou), int(k)
        self.box_encoding = str(box_encoding).lower()
        self.iou_type = str(iou_type).lower()
        self.logit_temp = float(max(logit_temp, 1e-6))

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        logits = torch.nan_to_num(outputs["pred_logits"], nan=0.0).float()
        boxes_raw = outputs["pred_boxes"].float()

        pb_cxcywh = torch.sigmoid(boxes_raw) if self.box_encoding == "logits" else boxes_raw.clamp(0.0, 1.0)
        out_prob = (logits / self.logit_temp).sigmoid()
        
        B, Q, _ = logits.shape
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for b in range(B):
            dev = pb_cxcywh.device
            prob, pb = out_prob[b], pb_cxcywh[b]

            tgt_ids = targets[b]["labels"].to(dev).long()
            if tgt_ids.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=dev)
                out.append((empty, empty))
                continue
            
            alpha, gamma = 0.25, 2.0
            neg_cost_class = (1 - alpha) * (prob ** gamma) * (-(1 - prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - prob) ** gamma) * (-(prob + 1e-8).log())
            c_cls = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            tgt_px = targets[b]["boxes"].to(dev).float()
            size_hw = _to_size_hw_tensor(targets[b].get("size", None), device=dev)
            txy_norm = _normalize_targets_xyxy(tgt_px, size_hw)
            t_cxcywh = box_xyxy_to_cxcywh(txy_norm)

            c_l1 = torch.cdist(pb, t_cxcywh, p=1)
            q_xyxy = box_cxcywh_to_xyxy(pb).clamp(0.0, 1.0)
            if self.iou_type == "iou":
                c_iou = 1.0 - _pairwise_iou(q_xyxy, txy_norm)
            else:
                c_iou = 1.0 - _pairwise_giou(q_xyxy, txy_norm)

            Cmat = self.cost_class * c_cls + self.cost_bbox * c_l1 + self.cost_giou * c_iou

            k_eff = min(self.k, Q)
            _, topk_idx = torch.topk(Cmat, k=k_eff, dim=0, largest=False) 
            idx_q = topk_idx.flatten()
            idx_t = torch.arange(tgt_ids.numel(), device=dev).repeat_interleave(k_eff)
            out.append((idx_q, idx_t))

        return out


class DETRDetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: OneToOneHungarianMatcher,
        *,
        box_encoding: str = "sigmoid",
        cls_loss_weight: float = 1.0,
        bbox_loss_weight: float = 5.0,
        giou_loss_weight: float = 2.0,
        eos_coef: float = 0.1,
        focal_ce: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        huber_delta: float = 0.0,
        iou_type: str = "giou",
        alpha_iou: float = 1.0,
        qfl_align: bool = False,
        qfl_gamma: float = 2.0,
        qfl_weight: float = 0.0,
        aux_loss_weight: float = 1.0,
        aux_loss_decay: float = 0.7,
        saliency_push_weight: float = 0.0,
        saliency_key: str = "saliency_prob",
        min_box_wh: float = 0.0,
        min_box_penalty: float = 0.0,
        wh_log_l1_weight: float = 0.0,
        cls_normalize: str = "num_pos",
        logit_temp: float = 1.0,
        itc_weight: float = 0.0,
        itc_beta: float = 1.0,
        saliency_consistency_weight: float = 0.0,
        saliency_consistency_tau: float = 0.3,
        saliency_consistency_lambda: float = 10.0,
        span_l1_beta1: float = 0.0,
        span_l1_beta2: float = 0.0,
        frr_gamma: float = 0.0,
        k_one2many: int = 0,
        lambda_one2many: float = 1.0,
        aux_apply_one2many: bool = False,
        debug: bool = False,
        debug_every: int = 50,
        debug_verbosity: int = 1,
        debug_assert_finite: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.matcher = matcher
        self.box_encoding = str(box_encoding).lower()

        self.lw_cls, self.lw_l1, self.lw_iou = float(cls_loss_weight), float(bbox_loss_weight), float(giou_loss_weight)
        self.eos_coef, self.focal_ce, self.focal_alpha, self.focal_gamma = float(eos_coef), bool(focal_ce), float(focal_alpha), float(focal_gamma)
        self.label_smoothing, self.huber_delta = float(label_smoothing), float(huber_delta)
        self.iou_type, self.alpha_iou = str(iou_type).lower(), float(alpha_iou)
        self.qfl_align, self.qfl_gamma, self.qfl_weight = bool(qfl_align), float(qfl_gamma), float(qfl_weight)
        self.aux_w, self.aux_decay = float(aux_loss_weight), float(aux_loss_decay)
        self.saliency_push_weight, self.saliency_key = float(saliency_push_weight), str(saliency_key)
        self.min_box_wh, self.min_box_penalty, self.wh_log_l1_weight = float(min_box_wh), float(min_box_penalty), float(wh_log_l1_weight)
        self.cls_normalize, self.logit_temp = str(cls_normalize).lower(), float(max(logit_temp, 1e-6))

        self.itc_weight, self.itc_beta = float(itc_weight), float(itc_beta)
        self.saliency_consistency_weight, self.saliency_consistency_tau, self.saliency_consistency_lambda = float(saliency_consistency_weight), float(saliency_consistency_tau), float(saliency_consistency_lambda)
        self.span_l1_beta1, self.span_l1_beta2, self.frr_gamma = float(span_l1_beta1), float(span_l1_beta2), float(frr_gamma)

        self.k_one2many, self.lambda_one2many = int(k_one2many), float(lambda_one2many)
        self.aux_apply_one2many = bool(aux_apply_one2many)
        if self.k_one2many > 0:
            self.matcher_o2m = OneToManyMatcher(
                cost_class=matcher.cost_class, cost_bbox=matcher.cost_bbox, cost_giou=matcher.cost_giou,
                k=self.k_one2many,
                box_encoding=self.box_encoding,
                iou_type=self.iou_type,
                logit_temp=self.logit_temp,
            )
        else:
            self.matcher_o2m = None

        self.debug, self.debug_every, self.debug_verbosity, self.debug_assert_finite = bool(debug), int(debug_every), int(debug_verbosity), bool(debug_assert_finite)
        self._debug_step = 0

        self.register_buffer("class_weights_fg", self._sanitize_class_weights(class_weights, self.num_classes), persistent=False)
        base_empty = torch.ones(self.num_classes + 1, dtype=torch.float32)
        base_empty[-1] = self.eos_coef
        self.register_buffer("empty_weight", base_empty, persistent=False)

    @staticmethod
    def _sanitize_class_weights(cw: Optional[torch.Tensor], C: int) -> torch.Tensor:
        if cw is None: return torch.ones(C, dtype=torch.float32)
        cw = torch.as_tensor(cw, dtype=torch.float32).view(-1)
        if cw.numel() < C:
            out = torch.ones(C, dtype=torch.float32)
            out[: cw.numel()] = cw
            return out
        return cw[:C].contiguous()

    def _runtime_weight(self, logits_bg: torch.Tensor) -> torch.Tensor:
        Cbg = logits_bg.size(-1)
        if self.empty_weight.numel() != Cbg:
            w = torch.ones(Cbg, device=logits_bg.device)
            w[:-1] = 1.0
            w[-1] = self.eos_coef
            return w
        w = self.empty_weight.to(logits_bg.device).clone()
        w[:-1] *= self.class_weights_fg.to(logits_bg.device)
        return w

    def _targets_per_query(self, logits_bg: torch.Tensor, targets: List[Dict[str, torch.Tensor]], indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        B, Q, Clog = logits_bg.shape
        all_t = []
        dev = logits_bg.device
        for b in range(B):
            tgt = targets[b]["labels"].to(dev).long()
            full = torch.full((Q,), Clog - 1, dtype=torch.long, device=dev)
            iq, it = indices[b]
            if iq.numel() > 0: full[iq] = tgt[it].clamp(0, Clog - 2)
            all_t.append(full)
        return torch.stack(all_t, dim=0)

    def _loss_cls(self, logits_bg: torch.Tensor, t_idx: torch.Tensor, denom: float, sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits32 = logits_bg.float().clamp(-50, 50)
        weight32 = self._runtime_weight(logits_bg).to(device=logits32.device, dtype=logits32.dtype)

        if self.focal_ce:
            logp = F.log_softmax(logits32, dim=-1)
            p = logp.exp().clamp(_EPS, 1.0 - _EPS)
            tgt = F.one_hot(t_idx, num_classes=logits32.size(-1)).to(dtype=logits32.dtype)
            ce = -(tgt * logp)
            mod = self.focal_alpha * (1.0 - p).pow(self.focal_gamma)
            loss = (ce * mod * weight32.view(1, 1, -1)).sum(-1)
        else:
            loss = F.cross_entropy(logits32.transpose(1, 2), t_idx, weight=weight32, reduction="none")

        if sample_weight is not None: loss = loss * sample_weight.to(device=loss.device, dtype=loss.dtype)
        return loss.sum() / max(1.0, denom)

    def _loss_boxes(self, pred_boxes_norm: torch.Tensor, targets: List[Dict[str, torch.Tensor]], indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        l1_list: List[torch.Tensor] = []
        iou_list: List[torch.Tensor] = []
        B, dev = pred_boxes_norm.size(0), pred_boxes_norm.device

        for b in range(B):
            iq, it = indices[b]
            if iq.numel() == 0: continue

            pb = pred_boxes_norm[b, iq, :].float()
            tgt_px = targets[b]["boxes"].to(dev).float()
            size_hw = _to_size_hw_tensor(targets[b].get("size", None), device=dev)
            tb_xy = _normalize_targets_xyxy(tgt_px, size_hw)
            tb = box_xyxy_to_cxcywh(tb_xy)[it, :].float()

            if self.span_l1_beta1 > 0.0 and self.span_l1_beta2 > 0.0:
                diff = torch.abs(pb - tb)
                l1 = ((diff * diff) / (self.span_l1_beta1 + diff).pow(self.span_l1_beta2)).sum(-1)
            elif self.huber_delta > 0: l1 = F.smooth_l1_loss(pb, tb, beta=self.huber_delta, reduction="none").sum(-1)
            else: l1 = F.l1_loss(pb, tb, reduction="none").sum(-1)

            if self.wh_log_l1_weight > 0.0:
                l1 = l1 + self.wh_log_l1_weight * F.l1_loss(pb[..., 2:].clamp_min(_EPS).log(), tb[..., 2:].clamp_min(_EPS).log(), reduction="none").sum(-1)

            l1_list.append(l1.mean())

            pb_xy, tb_xy_m = box_cxcywh_to_xyxy(pb).clamp(0.0, 1.0), box_cxcywh_to_xyxy(tb).clamp(0.0, 1.0)
            if self.iou_type == "siou": base = 1.0 - _boxes_siou_diag(pb_xy, tb_xy_m).clamp(0.0, 1.0)
            elif self.iou_type == "iou": base = 1.0 - _boxes_iou_diag(pb_xy, tb_xy_m).clamp(0.0, 1.0)
            else: base = 1.0 - _pairwise_giou(pb_xy, tb_xy_m).diag().clamp(-1.0, 1.0)

            base = base.clamp_min(0.0)
            if self.alpha_iou != 1.0: base = base.pow(self.alpha_iou)
            iou_list.append(base.mean())

        if not l1_list:
            zero = pred_boxes_norm.sum() * 0.0
            return zero, zero

        return torch.stack(l1_list).mean(), torch.stack(iou_list).mean()

    def _loss_qfl(self, logits_bg: torch.Tensor, pred_boxes_norm: torch.Tensor, targets: List[Dict[str, torch.Tensor]], indices: List[Tuple[torch.Tensor, torch.Tensor]], num_pos: int) -> torch.Tensor:
        if num_pos <= 0: return logits_bg.sum() * 0.0
        probs, B, C, dev = F.softmax(logits_bg.float(), dim=-1), logits_bg.size(0), self.num_classes, logits_bg.device
        loss_list: List[torch.Tensor] = []

        for b in range(B):
            iq, it = indices[b]
            if iq.numel() == 0: continue

            p_fg = probs[b][iq, targets[b]["labels"].to(dev).long().clamp(0, C - 1)[it]].clamp(_EPS, 1.0 - _EPS)
            pb = pred_boxes_norm[b, iq, :].float()
            tgt_px = targets[b]["boxes"].to(dev).float()
            size_hw = _to_size_hw_tensor(targets[b].get("size", None), device=dev)
            tb = box_xyxy_to_cxcywh(_normalize_targets_xyxy(tgt_px, size_hw))[it, :].float()
            q = _boxes_iou_diag(box_cxcywh_to_xyxy(pb).clamp(0.0, 1.0), box_cxcywh_to_xyxy(tb).clamp(0.0, 1.0))

            loss_list.append((F.binary_cross_entropy(p_fg, q, reduction="none") * torch.abs(q - p_fg).pow(self.qfl_gamma)).mean())

        return (torch.stack(loss_list).mean() / max(1.0, float(num_pos))) if loss_list else (logits_bg.sum() * 0.0)

    def _loss_itc(self, logits_bg: torch.Tensor, pred_boxes_norm: torch.Tensor, targets: List[Dict[str, torch.Tensor]], indices: List[Tuple[torch.Tensor, torch.Tensor]], num_pos: int) -> torch.Tensor:
        if num_pos <= 0: return logits_bg.sum() * 0.0
        B, C, dev = logits_bg.size(0), self.num_classes, logits_bg.device
        logits_fg, loss_list = logits_bg[..., :C].float(), []

        for b in range(B):
            iq, it = indices[b]
            if iq.numel() == 0: continue

            t_lab = targets[b]["labels"].to(dev).long().clamp(0, C - 1)[it]
            pb = pred_boxes_norm[b, iq, :].float()
            tgt_px = targets[b]["boxes"].to(dev).float()
            size_hw = _to_size_hw_tensor(targets[b].get("size", None), device=dev)
            tb = box_xyxy_to_cxcywh(_normalize_targets_xyxy(tgt_px, size_hw))[it, :].float()
            q = _boxes_iou_diag(box_cxcywh_to_xyxy(pb).clamp(0.0, 1.0), box_cxcywh_to_xyxy(tb).clamp(0.0, 1.0))

            logit_b = logits_fg[b, iq, :].clone()
            row_idx = torch.arange(t_lab.numel(), device=dev)
            logit_b[row_idx, t_lab] = logit_b[row_idx, t_lab] + self.itc_beta * q

            loss_list.append(F.cross_entropy(logit_b, t_lab, reduction="mean"))

        return (torch.stack(loss_list).mean() / max(1.0, float(num_pos))) if loss_list else (logits_bg.sum() * 0.0)

    def _loss_saliency_push(self, saliency: torch.Tensor, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        s = saliency[..., 0] if saliency.dim() == 3 and saliency.size(-1) == 1 else saliency
        if s.dim() != 2: return s.sum() * 0.0

        s = s.float().clamp(0.0, 1.0)
        tgt = torch.zeros_like(s, device=s.device)
        for b in range(min(s.size(0), len(indices))):
            iq, _ = indices[b]
            if iq.numel() > 0: tgt[b, iq] = 1.0

        pos_mask, neg_mask = tgt > 0.5, ~ (tgt > 0.5)
        loss_pos = F.binary_cross_entropy(s[pos_mask], torch.ones_like(s[pos_mask]), reduction="mean") if pos_mask.any() else s.new_tensor(0.0)
        loss_neg = F.binary_cross_entropy(s[neg_mask], torch.zeros_like(s[neg_mask]), reduction="mean") if neg_mask.any() else s.new_tensor(0.0)
        return loss_pos + loss_neg

    def _loss_saliency_consistency(self, saliency: torch.Tensor, pred_boxes_norm: torch.Tensor, targets: List[Dict[str, torch.Tensor]], indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        s = saliency[..., 0] if saliency.dim() == 3 and saliency.size(-1) == 1 else saliency
        if s.dim() != 2: return s.sum() * 0.0

        s = s.float().clamp(0.0, 1.0)
        dev = s.device
        tgt = torch.zeros_like(s, device=dev)

        for b in range(min(s.size(0), len(indices))):
            iq, it = indices[b]
            if iq.numel() == 0: continue

            pb = pred_boxes_norm[b, iq, :].float()
            tgt_px = targets[b]["boxes"].to(dev).float()
            size_hw = _to_size_hw_tensor(targets[b].get("size", None), device=dev)
            tb = box_xyxy_to_cxcywh(_normalize_targets_xyxy(tgt_px, size_hw))[it, :].float()
            iou = _boxes_iou_diag(box_cxcywh_to_xyxy(pb).clamp(0.0, 1.0), box_cxcywh_to_xyxy(tb).clamp(0.0, 1.0))

            soft = iou / (1.0 + torch.exp(-self.saliency_consistency_lambda * (iou - self.saliency_consistency_tau)))
            tgt[b, iq] = soft.clamp(0.0, 1.0)

        return F.binary_cross_entropy(s, tgt, reduction="mean")

    def _compute_criterion_losses(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], indices: List[Tuple[torch.Tensor, torch.Tensor]], logits_bg: torch.Tensor, pred_boxes_norm: torch.Tensor, num_boxes: int) -> Dict[str, torch.Tensor]:
        B, Q = logits_bg.shape[:2]
        num_pos = sum(len(iq) for iq, _ in indices)

        if self.cls_normalize == "num_boxes": denom = max(2.0, float(num_boxes))
        elif self.cls_normalize == "num_pos": denom = max(1.0, float(num_pos))
        else: denom = float(B * Q)

        sample_weight = None
        if self.frr_gamma != 0.0 and self.saliency_key in outputs:
            s = outputs[self.saliency_key]
            s = s[..., 0] if s.dim() == 3 and s.size(-1) == 1 else s
            if s.dim() == 2 and s.shape[:2] == (B, Q): sample_weight = 1.0 + self.frr_gamma * s.float().clamp(0.0, 1.0)

        t_idx = self._targets_per_query(logits_bg, targets, indices)
        loss_dict: Dict[str, torch.Tensor] = {
            "loss_cls": self.lw_cls * self._loss_cls(logits_bg, t_idx, denom=denom, sample_weight=sample_weight)
        }

        l_l1, l_iou = self._loss_boxes(pred_boxes_norm, targets, indices)
        loss_dict["loss_bbox"] = self.lw_l1 * l_l1
        loss_dict["loss_iou"] = self.lw_iou * l_iou

        if self.qfl_weight > 0.0 and self.qfl_align and num_pos > 0: loss_dict["loss_qfl"] = self.qfl_weight * self._loss_qfl(logits_bg, pred_boxes_norm, targets, indices, num_pos)
        if self.itc_weight > 0.0 and num_pos > 0: loss_dict["loss_cls_itc"] = self.itc_weight * self._loss_itc(logits_bg, pred_boxes_norm, targets, indices, num_pos)
        if self.min_box_wh > 0.0 and self.min_box_penalty > 0.0:
            too_small = (pred_boxes_norm[..., 2:4] < self.min_box_wh).float()
            loss_dict["loss_min_wh"] = self.min_box_penalty * (too_small.sum() / max(1.0, float(B * Q)))
        if self.saliency_push_weight > 0.0 and self.saliency_key in outputs: loss_dict["loss_saliency"] = self.saliency_push_weight * self._loss_saliency_push(outputs[self.saliency_key], indices)
        if self.saliency_consistency_weight > 0.0 and self.saliency_key in outputs: loss_dict["loss_saliency_consistency"] = self.saliency_consistency_weight * self._loss_saliency_consistency(outputs[self.saliency_key], pred_boxes_norm, targets, indices)
        
        return loss_dict

    def _forward_single_pass(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        *,
        enable_one2many: bool = True,
    ) -> Dict[str, torch.Tensor]:
        raw_logits = torch.nan_to_num(outputs["pred_logits"], nan=0.0).float()
        raw_boxes = outputs["pred_boxes"].float()

        B, Q, Cin = raw_logits.shape
        logits_temp = (raw_logits / self.logit_temp).clamp(-50, 50)

        if Cin == self.num_classes + 1 or Cin > self.num_classes: logits_bg = logits_temp
        else: logits_bg = torch.cat([logits_temp[..., :self.num_classes], logits_temp.new_zeros((B, Q, 1))], dim=-1)

        pred_boxes_norm = torch.sigmoid(raw_boxes) if self.box_encoding == "logits" else raw_boxes.clamp(0.0, 1.0)
        match_in = {"pred_logits": logits_bg, "pred_boxes": pred_boxes_norm}
        num_boxes = sum(len(t["labels"]) for t in targets)

        with torch.no_grad(): indices = self.matcher(match_in, targets)
        loss_dict = self._compute_criterion_losses(outputs, targets, indices, logits_bg, pred_boxes_norm, num_boxes)

        if enable_one2many and self.matcher_o2m is not None and self.lambda_one2many > 0.0:
            with torch.no_grad(): indices_o2m = self.matcher_o2m(match_in, targets)
            loss_dict_o2m = self._compute_criterion_losses(outputs, targets, indices_o2m, logits_bg, pred_boxes_norm, num_boxes)
            for k, v in loss_dict_o2m.items(): loss_dict[f"{k}_o2m"] = v * self.lambda_one2many

        if self.debug:
            self._debug_step += 1
            if self._debug_step % max(1, self.debug_every) == 0:
                with torch.no_grad():
                    def _to_float(x: Union[torch.Tensor, float, int]) -> float:
                        if isinstance(x, torch.Tensor):
                            xv = x.detach()
                            if xv.numel() == 0:
                                return 0.0
                            return float(xv.mean().item()) if xv.numel() > 1 else float(xv.item())
                        return float(x)

                    total = sum(_to_float(v) for v in loss_dict.values() if isinstance(v, torch.Tensor))
                    if self.debug_verbosity > 0:
                        print(
                            f"[DETRLoss] step={self._debug_step} total={total:.4f} "
                            f"o2o_cls={_to_float(loss_dict.get('loss_cls', 0.0)):.4f} "
                            f"o2m_cls={_to_float(loss_dict.get('loss_cls_o2m', 0.0)):.4f}",
                            flush=True,
                        )
                if self.debug_assert_finite:
                    for k, v in loss_dict.items():
                        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all(): raise RuntimeError(f"Non-finite loss term: {k}")

        return loss_dict

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        loss_dict = self._forward_single_pass(outputs, targets, enable_one2many=True)
        aux = outputs.get("aux_outputs", None)
        if aux is not None and self.aux_w > 0.0:
            aux = [aux[k] for k in sorted(aux.keys())] if isinstance(aux, dict) else aux
            if isinstance(aux, (list, tuple)):
                for i, aux_out in enumerate(aux):
                    if aux_out is None: continue
                    sub = self._forward_single_pass(aux_out, targets, enable_one2many=self.aux_apply_one2many)
                    w = self.aux_w * (self.aux_decay ** i)
                    for k, v in sub.items(): loss_dict[f"{k}_aux{i}"] = v * w

        # RT-DETR encoder supervision branch.
        enc_aux = outputs.get("enc_aux_outputs", None)
        if isinstance(enc_aux, dict) and self.aux_w > 0.0:
            sub_enc = self._forward_single_pass(enc_aux, targets, enable_one2many=False)
            for k, v in sub_enc.items():
                loss_dict[f"{k}_enc"] = v * self.aux_w
        return loss_dict


def build_detr_detection_loss(cfg_or_num_classes, **kwargs) -> DETRDetectionLoss:
    if isinstance(cfg_or_num_classes, int):
        num_classes = int(cfg_or_num_classes)
    else:
        cfg = cfg_or_num_classes
        num_classes = int(getattr(cfg, "num_classes"))
        for name in ["cls_loss_weight", "bbox_loss_weight", "giou_loss_weight", "iou_type", "box_encoding", "alpha_iou", "qfl_align", "qfl_gamma", "qfl_weight", "aux_loss_weight", "aux_loss_decay", "saliency_push_weight", "saliency_key", "min_box_wh", "min_box_penalty", "wh_log_l1_weight", "cls_normalize", "logit_temp", "itc_weight", "itc_beta", "saliency_consistency_weight", "saliency_consistency_tau", "saliency_consistency_lambda", "span_l1_beta1", "span_l1_beta2", "frr_gamma", "k_one2many", "lambda_one2many", "aux_apply_one2many", "debug", "debug_every", "debug_verbosity", "debug_assert_finite", "matcher_batch_cpu_copy_max_numel"]:
            if hasattr(cfg, name) and name not in kwargs: kwargs[name] = getattr(cfg, name)
        for name in ["cost_class", "cost_bbox", "cost_giou"]:
            if hasattr(cfg, name) and name not in kwargs: kwargs[name] = getattr(cfg, name)

    matcher_batch_cpu_copy_max_numel = kwargs.pop("matcher_batch_cpu_copy_max_numel", 8_000_000)
    matcher = OneToOneHungarianMatcher(
        cost_class=kwargs.get("cost_class", 2.0),
        cost_bbox=kwargs.get("cost_bbox", 5.0),
        cost_giou=kwargs.get("cost_giou", 2.0),
        box_encoding=kwargs.get("box_encoding", "sigmoid"),
        iou_type=kwargs.get("iou_type", "giou"),
        logit_temp=kwargs.get("logit_temp", 1.0),
        batch_cpu_copy_max_numel=matcher_batch_cpu_copy_max_numel,
    )

    return DETRDetectionLoss(num_classes=num_classes, matcher=matcher, **kwargs)
