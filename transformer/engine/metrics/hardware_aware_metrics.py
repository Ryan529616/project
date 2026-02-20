# -*- coding: utf-8 -*-
"""
hardware_aware_metrics.py

Detector-only lightweight metrics for DualStageBackboneDetector.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def _object_token_metrics(obj_tokens: torch.Tensor) -> Dict[str, float]:
    m: Dict[str, float] = {}
    if obj_tokens.ndim != 3 or obj_tokens.numel() == 0:
        return m

    norms = obj_tokens.float().norm(dim=-1)
    m["hw/object_token_norm_mean"] = float(norms.mean().item())
    m["hw/object_token_norm_std"] = float(norms.std(unbiased=False).item())

    b, q, _ = obj_tokens.shape
    if q > 1:
        z = F.normalize(obj_tokens.float(), dim=-1, eps=1e-6)
        sim = torch.matmul(z, z.transpose(1, 2))
        eye = torch.eye(q, device=sim.device, dtype=sim.dtype).unsqueeze(0)
        off = sim * (1.0 - eye)
        m["hw/object_token_offdiag_cos2"] = float(off.square().sum(dim=(1, 2)).mean().item() / float(q * (q - 1)))
        m["hw/object_token_abs_cos"] = float(off.abs().sum(dim=(1, 2)).mean().item() / float(q * (q - 1)))

    return m


@torch.no_grad()
def _scale_token_metrics(tokens_per_level) -> Dict[str, float]:
    m: Dict[str, float] = {}
    if not isinstance(tokens_per_level, (list, tuple)):
        return m

    level_energies = []
    for i, t in enumerate(tokens_per_level):
        if not isinstance(t, torch.Tensor) or t.ndim != 3:
            continue
        norm = t.float().norm(dim=-1)
        m[f"hw/level{i}_token_norm_mean"] = float(norm.mean().item())
        energy = t.float().pow(2).mean(dim=(1, 2))
        level_energies.append(energy)

    if len(level_energies) > 1:
        e = torch.stack(level_energies, dim=1)
        m["hw/scale_energy_var"] = float(e.var(dim=1, unbiased=False).mean().item())

    return m


@torch.no_grad()
def _query_logit_metrics(pred_logits: torch.Tensor) -> Dict[str, float]:
    m: Dict[str, float] = {}
    if pred_logits.ndim != 3 or pred_logits.numel() == 0:
        return m

    probs = F.softmax(pred_logits.float(), dim=-1)
    bg_prob = probs[..., -1]
    fg_prob = probs[..., :-1]

    m["det/bg_prob_mean"] = float(bg_prob.mean().item())

    if fg_prob.numel() > 0:
        fg_conf = fg_prob.max(dim=-1).values
        m["det/fg_conf_mean"] = float(fg_conf.mean().item())
        m["det/fg_query_ratio@0.3"] = float((fg_conf > 0.3).float().mean().item())
        m["det/fg_query_ratio@0.5"] = float((fg_conf > 0.5).float().mean().item())

    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    m["det/query_entropy"] = float(entropy.mean().item())
    return m


@torch.no_grad()
def compute_hardware_aware_batch_metrics(outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Collect lightweight detector diagnostics from model outputs only."""
    m: Dict[str, float] = {}
    if not isinstance(outputs, dict):
        return m

    obj_tokens = outputs.get("object_tokens", None)
    if isinstance(obj_tokens, torch.Tensor):
        m.update(_object_token_metrics(obj_tokens))

    t_per_level = outputs.get("tokens_per_level", None)
    m.update(_scale_token_metrics(t_per_level))

    pred_logits = outputs.get("pred_logits", None)
    if isinstance(pred_logits, torch.Tensor):
        m.update(_query_logit_metrics(pred_logits))

    return m


__all__ = ["compute_hardware_aware_batch_metrics"]
