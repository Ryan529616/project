# -*- coding: utf-8 -*-
"""
hardware_aware_detector.py

End-to-end hardware-aware object detector:
1) Hybrid multi-scale backbone (CFA-aware stem + DW-separable residual blocks + GC bottleneck)
2) Intra-scale latent tokenization
3) Inter-scale fusion + global object query extraction
4) DETR-style class / bbox MLP heads
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_act(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported act: {name}")


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(min=float(eps), max=1.0 - float(eps))
    return torch.log(x / (1.0 - x))


class _NormAct(nn.Module):
    def __init__(self, c: int, *, act: str, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm2d(c, eps=eps, affine=affine)
        self.act = _make_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(x))


class _ConvDW2d(nn.Module):
    def __init__(self, c: int, k: int = 3, s: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=k, stride=s, padding=k // 2, groups=c, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _ConvPW(nn.Module):
    def __init__(self, c_in: int, c_out: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _DWSep(nn.Module):
    def __init__(self, c_in: int, c_out: int, *, k: int, s: int, act: str, eps: float, affine: bool):
        super().__init__()
        self.dw = _ConvDW2d(c_in, k=k, s=s, bias=False)
        self.na1 = _NormAct(c_in, act=act, eps=eps, affine=affine)
        self.pw = _ConvPW(c_in, c_out, bias=False)
        self.na2 = _NormAct(c_out, act=act, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.na2(self.pw(self.na1(self.dw(x))))


class _ResDWBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, *, k: int, act: str, eps: float, affine: bool):
        super().__init__()
        self.conv1 = _DWSep(c_in, c_out, k=k, s=1, act=act, eps=eps, affine=affine)
        self.conv2 = _DWSep(c_out, c_out, k=k, s=1, act=act, eps=eps, affine=affine)

        self.use_short = c_in == c_out
        self.shortcut = None
        if not self.use_short:
            self.shortcut = nn.Sequential(
                _ConvPW(c_in, c_out, bias=False),
                nn.BatchNorm2d(c_out, eps=eps, affine=affine),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        if self.use_short:
            return y + x
        return y + self.shortcut(x)


class _Down(nn.Module):
    def __init__(self, c_in: int, c_out: int, *, act: str, eps: float, affine: bool):
        super().__init__()
        self.dw = _ConvDW2d(c_in, k=3, s=2, bias=False)
        self.na1 = _NormAct(c_in, act=act, eps=eps, affine=affine)
        self.pw = _ConvPW(c_in, c_out, bias=False)
        self.na2 = _NormAct(c_out, act=act, eps=eps, affine=affine)
        self.block = _ResDWBlock(c_out, c_out, k=3, act=act, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.na2(self.pw(self.na1(self.dw(x)))))


class _UpAdd(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int, *, act: str, eps: float, affine: bool):
        super().__init__()
        self.reduce = nn.Sequential(
            _ConvPW(c_in, c_out, bias=False),
            nn.BatchNorm2d(c_out, eps=eps, affine=affine),
            _make_act(act),
        )
        self.skip_proj = nn.Sequential(
            _ConvPW(c_skip, c_out, bias=False),
            nn.BatchNorm2d(c_out, eps=eps, affine=affine),
        )
        self.fuse = _ResDWBlock(c_out, c_out, k=3, act=act, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.fuse(self.reduce(x) + self.skip_proj(skip))


class _GlobalContextBlock(nn.Module):
    def __init__(
        self,
        c: int,
        *,
        reduction: int,
        act: str,
        gate: str,
        clamp_alpha: float,
        clamp_beta: float,
    ):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        c_mid = max(1, int(c) // max(1, int(reduction)))
        self.mlp1 = nn.Conv2d(c, c_mid, 1, bias=True)
        self.act = _make_act(act)
        self.mlp2 = nn.Conv2d(c_mid, c, 1, bias=True)
        self.gate_kind = str(gate)
        self.clamp_alpha = float(clamp_alpha)
        self.clamp_beta = float(clamp_beta)
        self.hsig = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = self.mlp2(self.act(self.mlp1(self.gap(x))))
        if self.gate_kind == "hard_sigmoid":
            g = self.hsig(ctx)
        else:
            g = torch.clamp(self.clamp_alpha * ctx + self.clamp_beta, 0.0, 1.0)
        return x * g


class _HybridDualPathBottleneck(nn.Module):
    def __init__(
        self,
        c: int,
        *,
        act: str,
        eps: float,
        affine: bool,
        local_blocks: int,
        gc_reduction: int,
        gc_gate: str,
        gc_clamp_alpha: float,
        gc_clamp_beta: float,
    ):
        super().__init__()
        c_s = int(c) // 2
        c_l = int(c) - c_s
        self.c_s = c_s
        self.local = nn.Sequential(*[
            _ResDWBlock(c_s, c_s, k=3, act=act, eps=eps, affine=affine)
            for _ in range(max(1, int(local_blocks)))
        ])
        self.global_ctx = _GlobalContextBlock(
            c_l,
            reduction=int(gc_reduction),
            act=act,
            gate=gc_gate,
            clamp_alpha=gc_clamp_alpha,
            clamp_beta=gc_clamp_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_s = x[:, : self.c_s, :, :]
        x_l = x[:, self.c_s :, :, :]
        return torch.cat([self.local(x_s), self.global_ctx(x_l)], dim=1)


class _HybridFourScaleBackbone(nn.Module):
    """
    Hybrid multi-scale backbone:
      - CFA-aware stem, DW-separable residual path, GC bottleneck, add-fuse decoder
      - new detector strengths: stable 4-scale decoder output for latent tokenization

    Returns decoder features [F1, F2, F3, F4]:
      - F1: deepest / lowest resolution
      - F4: shallowest / highest resolution
    """

    def __init__(
        self,
        *,
        in_ch: int = 4,
        base_c: int = 64,
        stem_kind: str = "cfa_grouped",
        act: str = "relu",
        bn_eps: float = 1e-5,
        bn_affine: bool = True,
        bottleneck_local_blocks: int = 2,
        gc_reduction: int = 4,
        gc_gate: str = "hard_sigmoid",
        gc_clamp_alpha: float = 0.2,
        gc_clamp_beta: float = 0.5,
    ):
        super().__init__()
        c1 = int(base_c)
        c2, c3, c4, c5 = c1 * 2, c1 * 4, c1 * 8, c1 * 16

        if str(stem_kind) == "cfa_grouped" and int(in_ch) == 4:
            self.stem = nn.Sequential(
                nn.Conv2d(int(in_ch), c1, kernel_size=3, stride=1, padding=1, groups=int(in_ch), bias=False),
                nn.BatchNorm2d(c1, eps=bn_eps, affine=bn_affine),
                _make_act(act),
                nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(c1, eps=bn_eps, affine=bn_affine),
                _make_act(act),
            )
            self.enc1 = _ResDWBlock(c1, c1, k=1, act=act, eps=bn_eps, affine=bn_affine)
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(int(in_ch), c1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c1, eps=bn_eps, affine=bn_affine),
                _make_act(act),
            )
            self.enc1 = _ResDWBlock(c1, c1, k=3, act=act, eps=bn_eps, affine=bn_affine)

        self.down12 = _Down(c1, c2, act=act, eps=bn_eps, affine=bn_affine)
        self.down23 = _Down(c2, c3, act=act, eps=bn_eps, affine=bn_affine)
        self.down34 = _Down(c3, c4, act=act, eps=bn_eps, affine=bn_affine)
        self.down45 = _Down(c4, c5, act=act, eps=bn_eps, affine=bn_affine)

        self.bottleneck = _HybridDualPathBottleneck(
            c5,
            act=act,
            eps=bn_eps,
            affine=bn_affine,
            local_blocks=bottleneck_local_blocks,
            gc_reduction=gc_reduction,
            gc_gate=gc_gate,
            gc_clamp_alpha=gc_clamp_alpha,
            gc_clamp_beta=gc_clamp_beta,
        )

        self.up54 = _UpAdd(c5, c4, c4, act=act, eps=bn_eps, affine=bn_affine)  # F1
        self.up43 = _UpAdd(c4, c3, c3, act=act, eps=bn_eps, affine=bn_affine)  # F2
        self.up32 = _UpAdd(c3, c2, c2, act=act, eps=bn_eps, affine=bn_affine)  # F3
        self.up21 = _UpAdd(c2, c1, c1, act=act, eps=bn_eps, affine=bn_affine)  # F4

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(self.stem(x))
        e2 = self.down12(e1)
        e3 = self.down23(e2)
        e4 = self.down34(e3)
        e5 = self.down45(e4)

        b = self.bottleneck(e5)
        f1 = self.up54(b, e4)
        f2 = self.up43(f1, e3)
        f3 = self.up32(f2, e2)
        f4 = self.up21(f3, e1)
        return [f1, f2, f3, f4]


class DualStageBackboneDetector(nn.Module):
    """
    Hardware-aware end-to-end detector:
      Phase-1: Hybrid backbone + intra-scale latent query tokenization
      Phase-2: inter-scale token fusion + object queries
      Phase-3: DETR-style class/bbox heads (without transformer decoder)
    """

    def __init__(
        self,
        *,
        in_ch: int = 4,
        num_classes: int = 4,
        d_model: int = 256,
        nhead: int = 8,
        num_queries: int = 100,
        backbone_base_c: int = 64,
        token_counts: Tuple[int, int, int, int] = (32, 64, 128, 256),
        max_input_h: int = 360,
        max_input_w: int = 640,
        bbox_hidden_dim: int = 256,
        dropout: float = 0.0,
        token_ffn_mult: float = 2.0,
        pred_box_space: str = "normalized",
        query_prior: str = "none",
        query_prior_init: str = "grid",
        anchor_wh_prior: float = 0.2,
        backbone_stem_kind: str = "cfa_grouped",
        backbone_act: str = "relu",
        backbone_bn_eps: float = 1e-5,
        backbone_bottleneck_local_blocks: int = 2,
        backbone_gc_reduction: int = 4,
        backbone_gc_gate: str = "hard_sigmoid",
        backbone_gc_clamp_alpha: float = 0.2,
        backbone_gc_clamp_beta: float = 0.5,
    ):
        super().__init__()
        if int(d_model) % int(nhead) != 0:
            raise ValueError(f"d_model({d_model}) must be divisible by nhead({nhead})")
        if len(tuple(token_counts)) != 4:
            raise ValueError("token_counts must contain 4 scales, e.g. (32,64,128,256)")

        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.has_bg = True
        self.num_classes_with_bg = self.num_classes + 1
        self.pred_box_space = str(pred_box_space)
        self.token_counts = tuple(int(v) for v in token_counts)
        self.token_ffn_mult = float(token_ffn_mult)
        self.query_prior = str(query_prior).lower()
        self.query_prior_init = str(query_prior_init).lower()
        self.anchor_wh_prior = float(anchor_wh_prior)

        if self.query_prior not in ("none", "reference", "anchor"):
            raise ValueError(f"query_prior must be one of ['none', 'reference', 'anchor'], got {query_prior!r}")
        if self.query_prior_init not in ("grid", "random"):
            raise ValueError(f"query_prior_init must be one of ['grid', 'random'], got {query_prior_init!r}")
        if not (0.0 < self.anchor_wh_prior < 1.0):
            raise ValueError(f"anchor_wh_prior must be in (0, 1), got {anchor_wh_prior}")

        self.backbone = _HybridFourScaleBackbone(
            in_ch=int(in_ch),
            base_c=int(backbone_base_c),
            stem_kind=backbone_stem_kind,
            act=backbone_act,
            bn_eps=float(backbone_bn_eps),
            bottleneck_local_blocks=int(backbone_bottleneck_local_blocks),
            gc_reduction=int(backbone_gc_reduction),
            gc_gate=backbone_gc_gate,
            gc_clamp_alpha=float(backbone_gc_clamp_alpha),
            gc_clamp_beta=float(backbone_gc_clamp_beta),
        )

        c1 = int(backbone_base_c)
        feat_ch = [c1 * 8, c1 * 4, c1 * 2, c1]
        self.scale_proj = nn.ModuleList([
            nn.Conv2d(cin, self.d_model, kernel_size=1, bias=False) for cin in feat_ch
        ])

        level_strides = (8, 4, 2, 1)
        max_rows = [max(1, (int(max_input_h) + s - 1) // s) for s in level_strides]
        max_cols = [max(1, (int(max_input_w) + s - 1) // s) for s in level_strides]
        self.row_embeds = nn.ModuleList([nn.Embedding(r, self.d_model) for r in max_rows])
        self.col_embeds = nn.ModuleList([nn.Embedding(c, self.d_model) for c in max_cols])

        self.latent_queries = nn.ParameterList([
            nn.Parameter(torch.empty(m, self.d_model)) for m in self.token_counts
        ])
        self.intra_scale_attn = nn.ModuleList([
            nn.MultiheadAttention(self.d_model, self.nhead, dropout=float(dropout), batch_first=True)
            for _ in range(4)
        ])
        self.intra_q_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(4)])
        self.intra_kv_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(4)])
        self.intra_post_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(4)])
        self.intra_attn_drops = nn.ModuleList([nn.Dropout(float(dropout)) for _ in range(4)])
        ffn_hidden = max(self.d_model, int(round(self.d_model * self.token_ffn_mult)))
        self.intra_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, ffn_hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(ffn_hidden, self.d_model),
                nn.Dropout(float(dropout)),
            )
            for _ in range(4)
        ])

        self.scale_embed = nn.Embedding(4, self.d_model)
        level_ids = []
        for i, m in enumerate(self.token_counts):
            level_ids.extend([i] * int(m))
        self.register_buffer("scale_level_ids", torch.tensor(level_ids, dtype=torch.long), persistent=False)

        self.object_queries = nn.Parameter(torch.empty(self.num_queries, self.d_model))
        if self.query_prior == "none":
            self.query_refpoints = None
            self.query_prior_proj = None
        else:
            ref_dim = 2 if self.query_prior == "reference" else 4
            self.query_refpoints = nn.Parameter(torch.empty(self.num_queries, ref_dim))
            self.query_prior_proj = nn.Sequential(
                nn.Linear(ref_dim, self.d_model),
                nn.ReLU(inplace=True),
                nn.Linear(self.d_model, self.d_model),
            )
        self.inter_scale_attn = nn.MultiheadAttention(self.d_model, self.nhead, dropout=float(dropout), batch_first=True)
        self.inter_q_norm = nn.LayerNorm(self.d_model)
        self.inter_kv_norm = nn.LayerNorm(self.d_model)
        self.inter_post_norm = nn.LayerNorm(self.d_model)
        self.inter_attn_drop = nn.Dropout(float(dropout))
        self.inter_ffn = nn.Sequential(
            nn.Linear(self.d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_hidden, self.d_model),
            nn.Dropout(float(dropout)),
        )

        self.class_head = nn.Linear(self.d_model, self.num_classes_with_bg)
        self.bbox_head = nn.Sequential(
            nn.Linear(self.d_model, int(bbox_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(bbox_hidden_dim), 4),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for emb in self.row_embeds:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        for emb in self.col_embeds:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.scale_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.object_queries, mean=0.0, std=0.02)
        for q in self.latent_queries:
            nn.init.normal_(q, mean=0.0, std=0.02)
        self._init_query_priors()

    @staticmethod
    def _make_query_grid(num_queries: int) -> torch.Tensor:
        side = max(1, int(math.ceil(math.sqrt(max(1, int(num_queries))))))
        ys = torch.linspace(0.5 / side, 1.0 - 0.5 / side, side)
        xs = torch.linspace(0.5 / side, 1.0 - 0.5 / side, side)
        grid_yx = torch.cartesian_prod(ys, xs)
        centers = torch.stack([grid_yx[:, 1], grid_yx[:, 0]], dim=-1)
        return centers[: int(num_queries)].contiguous()

    def _init_query_priors(self) -> None:
        if self.query_prior == "none" or self.query_refpoints is None:
            return

        if self.query_prior_init == "random":
            centers = torch.rand(self.num_queries, 2, device=self.query_refpoints.device, dtype=self.query_refpoints.dtype)
        else:
            centers = self._make_query_grid(self.num_queries).to(
                device=self.query_refpoints.device, dtype=self.query_refpoints.dtype
            )

        if self.query_prior == "reference":
            prior = centers
        else:
            wh = torch.full(
                (self.num_queries, 2),
                float(self.anchor_wh_prior),
                device=self.query_refpoints.device,
                dtype=self.query_refpoints.dtype,
            )
            prior = torch.cat([centers, wh], dim=-1)

        with torch.no_grad():
            self.query_refpoints.copy_(_inverse_sigmoid(prior))

    @staticmethod
    def _take_or_interp_1d(emb: nn.Embedding, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        w = emb.weight
        if int(n) <= int(w.size(0)):
            out = w[: int(n), :]
        else:
            out = F.interpolate(
                w.transpose(0, 1).unsqueeze(0),
                size=int(n),
                mode="linear",
                align_corners=False,
            ).squeeze(0).transpose(0, 1)
        return out.to(device=device, dtype=dtype)

    def _build_2d_spatial_pe(self, level: int, h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        row = self._take_or_interp_1d(self.row_embeds[int(level)], int(h), device=device, dtype=dtype)
        col = self._take_or_interp_1d(self.col_embeds[int(level)], int(w), device=device, dtype=dtype)
        pe2d = row[:, None, :] + col[None, :, :]
        return pe2d.reshape(1, int(h) * int(w), self.d_model)

    def ensure_bridge(self, raw4: torch.Tensor) -> None:
        _ = raw4
        return

    def forward(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]):
        if isinstance(x, dict):
            raw4 = x.get("raw4", None)
            if raw4 is None:
                raise KeyError("DualStageBackboneDetector expects tensor input or dict containing key 'raw4'")
        else:
            raw4 = x

        image_hw = (int(raw4.shape[-2]), int(raw4.shape[-1]))
        feats = self.backbone(raw4)

        # Phase 1: per-scale intra-tokenization
        tokens_per_level = []
        for i, feat in enumerate(feats):
            feat_p = self.scale_proj[i](feat)
            bsz, _, h, w = feat_p.shape
            feat_seq = feat_p.flatten(2).transpose(1, 2).contiguous()
            pe2d = self._build_2d_spatial_pe(i, h, w, device=feat_seq.device, dtype=feat_seq.dtype)
            q = self.latent_queries[i].unsqueeze(0).expand(bsz, -1, -1).to(device=feat_seq.device, dtype=feat_seq.dtype)
            q_n = self.intra_q_norms[i](q)
            feat_n = self.intra_kv_norms[i](feat_seq)
            k_n = feat_n + pe2d
            v_n = feat_n
            attn_out = self.intra_scale_attn[i](q_n, k_n, v_n, need_weights=False)[0]
            t_i = q + self.intra_attn_drops[i](attn_out)
            t_i = t_i + self.intra_ffns[i](self.intra_post_norms[i](t_i))
            tokens_per_level.append(t_i)

        # Phase 2: inter-scale fusion + global object extraction
        t_concat = torch.cat(tokens_per_level, dim=1)
        lid = self.scale_level_ids.to(device=t_concat.device, non_blocking=True)
        scale_pe = self.scale_embed(lid).unsqueeze(0).to(dtype=t_concat.dtype)

        obj_q = self.object_queries.unsqueeze(0).expand(t_concat.size(0), -1, -1).to(device=t_concat.device, dtype=t_concat.dtype)
        ref_logits = None
        ref_points = None
        if self.query_refpoints is not None and self.query_prior_proj is not None:
            ref_logits = self.query_refpoints.unsqueeze(0).expand(t_concat.size(0), -1, -1).to(
                device=t_concat.device, dtype=t_concat.dtype
            )
            ref_points = ref_logits.sigmoid()
            obj_q = obj_q + self.query_prior_proj(ref_points)
        obj_q_n = self.inter_q_norm(obj_q)
        t_concat_n = self.inter_kv_norm(t_concat)
        k_n = t_concat_n + scale_pe
        v_n = t_concat_n
        obj_attn = self.inter_scale_attn(obj_q_n, k_n, v_n, need_weights=False)[0]
        obj_tokens = obj_q + self.inter_attn_drop(obj_attn)
        obj_tokens = obj_tokens + self.inter_ffn(self.inter_post_norm(obj_tokens))

        # Phase 3: detection heads
        pred_logits = self.class_head(obj_tokens)
        box_delta = self.bbox_head(obj_tokens)
        if self.query_prior == "anchor" and ref_logits is not None:
            pred_boxes = (box_delta + ref_logits).sigmoid().clamp(0.0, 1.0)
        elif self.query_prior == "reference" and ref_logits is not None:
            pred_xy = (box_delta[..., :2] + ref_logits[..., :2]).sigmoid()
            pred_wh = box_delta[..., 2:4].sigmoid()
            pred_boxes = torch.cat([pred_xy, pred_wh], dim=-1).clamp(0.0, 1.0)
        else:
            pred_boxes = box_delta.sigmoid().clamp(0.0, 1.0)

        if self.pred_box_space == "pixel":
            H, W = image_hw
            cx, cy, bw, bh = pred_boxes.unbind(dim=-1)
            pred_boxes = torch.stack([cx * W, cy * H, bw * W, bh * H], dim=-1)

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "aux_outputs": None,
            "num_classes": self.num_classes,
            "has_bg": self.has_bg,
            "image_hw": image_hw,
            "tokens_per_level": tokens_per_level,
            "object_tokens": obj_tokens,
            "query_prior_mode": self.query_prior,
            "query_reference_points": ref_points,
        }


__all__ = ["DualStageBackboneDetector"]
