# -*- coding: utf-8 -*-
"""
hardware_aware_detection_losses.py

Detector-only loss stack for DualStageBackboneDetector:
- Core matching loss: DETRDetectionLoss
- Optional hardware-aware regularizers over latent/query tokens
"""

from __future__ import annotations

from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.losses.detection_losses_core import build_detr_detection_loss


class HardwareAwareDetectionLoss(nn.Module):
    """
    Wrap detection core loss and add optional token/query regularizers.

    All regularizer weights default to 0.0, so behavior remains identical to
    core matching loss unless explicitly enabled.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        object_token_l2_weight: float = 0.0,
        object_token_diversity_weight: float = 0.0,
        object_token_diversity_margin: float = 0.0,
        object_token_var_weight: float = 0.0,
        object_token_var_target: float = 0.25,
        scale_token_balance_weight: float = 0.0,
        query_bg_balance_weight: float = 0.0,
        query_bg_target: float = 0.75,
        **detr_kwargs: Any,
    ):
        super().__init__()
        self.det_loss = build_detr_detection_loss(int(num_classes), **detr_kwargs)

        self.object_token_l2_weight = float(object_token_l2_weight)
        self.object_token_diversity_weight = float(object_token_diversity_weight)
        self.object_token_diversity_margin = float(object_token_diversity_margin)
        self.object_token_var_weight = float(object_token_var_weight)
        self.object_token_var_target = float(object_token_var_target)
        self.scale_token_balance_weight = float(scale_token_balance_weight)
        self.query_bg_balance_weight = float(query_bg_balance_weight)
        self.query_bg_target = float(query_bg_target)

    @staticmethod
    def _zero_like(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        t = outputs.get("pred_logits", None)
        if isinstance(t, torch.Tensor):
            return t.sum() * 0.0
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                return v.sum() * 0.0
        return torch.tensor(0.0)

    @staticmethod
    def _object_token_l2(obj_tokens: torch.Tensor) -> torch.Tensor:
        return obj_tokens.float().pow(2).mean()

    @staticmethod
    def _object_token_diversity(obj_tokens: torch.Tensor, margin: float) -> torch.Tensor:
        # Encourage low off-diagonal cosine similarity among object queries.
        if obj_tokens.ndim != 3:
            return obj_tokens.sum() * 0.0
        b, q, _ = obj_tokens.shape
        if q <= 1:
            return obj_tokens.sum() * 0.0

        z = F.normalize(obj_tokens.float(), dim=-1, eps=1e-6)
        sim = torch.matmul(z, z.transpose(1, 2))
        eye = torch.eye(q, device=sim.device, dtype=sim.dtype).unsqueeze(0)
        off_abs = (sim * (1.0 - eye)).abs()
        if float(margin) > 0.0:
            off_abs = F.relu(off_abs - float(margin))
        denom = max(1, q * (q - 1))
        return off_abs.square().sum(dim=(1, 2)).mean() / float(denom)

    @staticmethod
    def _object_token_var_floor(obj_tokens: torch.Tensor, target_var: float) -> torch.Tensor:
        # Prevent query collapse by ensuring per-dimension variance across queries.
        if obj_tokens.ndim != 3:
            return obj_tokens.sum() * 0.0
        var_q = obj_tokens.float().var(dim=1, unbiased=False).mean(dim=1)
        return F.relu(float(target_var) - var_q).mean()

    @staticmethod
    def _scale_token_balance(tokens_per_level: List[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
        if not isinstance(tokens_per_level, (list, tuple)) or len(tokens_per_level) <= 1:
            return zero
        per_level = []
        for t in tokens_per_level:
            if isinstance(t, torch.Tensor) and t.ndim == 3:
                per_level.append(t.float().pow(2).mean(dim=(1, 2)))
        if len(per_level) <= 1:
            return zero
        e = torch.stack(per_level, dim=1)
        return e.var(dim=1, unbiased=False).mean()

    @staticmethod
    def _query_bg_balance(pred_logits: torch.Tensor, target_bg: float) -> torch.Tensor:
        if pred_logits.ndim != 3:
            return pred_logits.sum() * 0.0
        probs = F.softmax(pred_logits.float(), dim=-1)
        bg = probs[..., -1]
        bg_mean = bg.mean()
        return (bg_mean - float(target_bg)).pow(2)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        det_loss = self.det_loss(outputs, targets)
        if isinstance(det_loss, dict):
            out = dict(det_loss)
        else:
            t = det_loss if torch.is_tensor(det_loss) else torch.as_tensor(det_loss, device=outputs["pred_logits"].device)
            out = {"loss_det_total": t}

        zero = self._zero_like(outputs)
        obj_tokens = outputs.get("object_tokens", None)
        tokens_per_level = outputs.get("tokens_per_level", None)
        pred_logits = outputs.get("pred_logits", None)

        if self.object_token_l2_weight > 0.0 and isinstance(obj_tokens, torch.Tensor):
            out["loss_obj_token_l2"] = self.object_token_l2_weight * self._object_token_l2(obj_tokens)

        if self.object_token_diversity_weight > 0.0 and isinstance(obj_tokens, torch.Tensor):
            out["loss_obj_token_div"] = self.object_token_diversity_weight * self._object_token_diversity(
                obj_tokens, self.object_token_diversity_margin
            )

        if self.object_token_var_weight > 0.0 and isinstance(obj_tokens, torch.Tensor):
            out["loss_obj_token_var"] = self.object_token_var_weight * self._object_token_var_floor(
                obj_tokens, self.object_token_var_target
            )

        if self.scale_token_balance_weight > 0.0:
            out["loss_scale_token_balance"] = self.scale_token_balance_weight * self._scale_token_balance(tokens_per_level, zero)

        if self.query_bg_balance_weight > 0.0 and isinstance(pred_logits, torch.Tensor):
            out["loss_query_bg_balance"] = self.query_bg_balance_weight * self._query_bg_balance(pred_logits, self.query_bg_target)

        return out


def build_hardware_aware_detection_loss(cfg_or_num_classes, **kwargs) -> HardwareAwareDetectionLoss:
    if isinstance(cfg_or_num_classes, int):
        num_classes = int(cfg_or_num_classes)
    else:
        cfg = cfg_or_num_classes
        num_classes = int(getattr(cfg, "num_classes"))

        regularizer_fields = (
            "object_token_l2_weight",
            "object_token_diversity_weight",
            "object_token_diversity_margin",
            "object_token_var_weight",
            "object_token_var_target",
            "scale_token_balance_weight",
            "query_bg_balance_weight",
            "query_bg_target",
        )
        for key in regularizer_fields:
            if (key not in kwargs) and hasattr(cfg, key):
                kwargs[key] = getattr(cfg, key)

    return HardwareAwareDetectionLoss(num_classes=num_classes, **kwargs)


__all__ = ["HardwareAwareDetectionLoss", "build_hardware_aware_detection_loss"]
