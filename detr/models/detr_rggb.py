# -*- coding: utf-8 -*-
"""
detr_rggb.py

Vanilla DETR baseline for 4-channel RGGB input.
Structure:
1) ResNet50 backbone (4-channel stem)
2) Transformer encoder/decoder
3) DETR class / box heads with auxiliary decoder losses
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        d = int(in_dim)
        for _ in range(max(1, int(num_layers)) - 1):
            layers += [nn.Linear(d, int(hidden_dim)), nn.ReLU(inplace=True)]
            d = int(hidden_dim)
        layers += [nn.Linear(d, int(out_dim))]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        num_pos_feats: int = 128,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.num_pos_feats = int(num_pos_feats)
        self.temperature = int(temperature)
        self.normalize = bool(normalize)
        self.scale = float(scale) if scale is not None else 2.0 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, _, h, w = x.shape
        device = x.device
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2.0 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()
        return pos


def _with_pos_embed(x: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
    return x if pos is None else (x + pos)


class TransformerEncoderLayerWithPos(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(int(d_model), int(nhead), dropout=float(dropout), batch_first=True)
        self.linear1 = nn.Linear(int(d_model), int(dim_feedforward))
        self.dropout = nn.Dropout(float(dropout))
        self.linear2 = nn.Linear(int(dim_feedforward), int(d_model))
        self.norm1 = nn.LayerNorm(int(d_model))
        self.norm2 = nn.LayerNorm(int(d_model))
        self.dropout1 = nn.Dropout(float(dropout))
        self.dropout2 = nn.Dropout(float(dropout))

    def forward(self, src: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        q = k = _with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, need_weights=False)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src), inplace=True)))
        src = self.norm2(src + self.dropout2(src2))
        return src


class TransformerEncoderWithPos(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithPos(
                    d_model=int(d_model),
                    nhead=int(nhead),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.norm = nn.LayerNorm(int(d_model))

    def forward(self, src: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        out = src
        for layer in self.layers:
            out = layer(out, pos=pos)
        return self.norm(out)


class TransformerDecoderLayerWithPos(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(int(d_model), int(nhead), dropout=float(dropout), batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), int(nhead), dropout=float(dropout), batch_first=True)
        self.linear1 = nn.Linear(int(d_model), int(dim_feedforward))
        self.dropout = nn.Dropout(float(dropout))
        self.linear2 = nn.Linear(int(dim_feedforward), int(d_model))

        self.norm1 = nn.LayerNorm(int(d_model))
        self.norm2 = nn.LayerNorm(int(d_model))
        self.norm3 = nn.LayerNorm(int(d_model))
        self.dropout1 = nn.Dropout(float(dropout))
        self.dropout2 = nn.Dropout(float(dropout))
        self.dropout3 = nn.Dropout(float(dropout))

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        pos: Optional[torch.Tensor],
        query_pos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q = k = _with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, need_weights=False)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        tgt2 = self.multihead_attn(
            query=_with_pos_embed(tgt, query_pos),
            key=_with_pos_embed(memory, pos),
            value=memory,
            need_weights=False,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt), inplace=True)))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class TransformerDecoderWithPos(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayerWithPos(
                    d_model=int(d_model),
                    nhead=int(nhead),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.norm = nn.LayerNorm(int(d_model))

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        pos: Optional[torch.Tensor],
        query_pos: Optional[torch.Tensor],
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        out = tgt
        inter: List[torch.Tensor] = []
        for layer in self.layers:
            out = layer(out, memory, pos=pos, query_pos=query_pos)
            if return_intermediate:
                inter.append(self.norm(out))
        out = self.norm(out)
        if return_intermediate:
            return torch.stack(inter, dim=0)
        return out


class DETRRGGB(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 4,
        num_queries: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        enc_layers: int = 6,
        dec_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        backbone_pretrained: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.has_bg = True

        weights = torchvision.models.ResNet50_Weights.DEFAULT if bool(backbone_pretrained) else None
        backbone = torchvision.models.resnet50(weights=weights)
        old_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            4,
            old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )
        nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
        with torch.no_grad():
            if old_conv1.weight.shape[1] == 3:
                new_conv1.weight[:, :3].copy_(old_conv1.weight)
                new_conv1.weight[:, 3].copy_(old_conv1.weight.mean(dim=1))
        backbone.conv1 = new_conv1

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.input_proj = nn.Conv2d(2048, int(d_model), kernel_size=1)
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=int(d_model) // 2, normalize=True)

        self.encoder = TransformerEncoderWithPos(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            num_layers=int(enc_layers),
        )
        self.decoder = TransformerDecoderWithPos(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            num_layers=int(dec_layers),
        )

        self.query_embed = nn.Embedding(int(num_queries), int(d_model))
        self.class_embed = nn.Linear(int(d_model), self.num_classes + 1)
        self.bbox_embed = MLP(int(d_model), int(d_model), 4, num_layers=3)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.normal_(self.query_embed.weight, mean=0.0, std=0.02)

    def forward(self, raw4: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(raw4)
        src = self.input_proj(feat)
        pos = self.pos_embed(src)

        b, _, h, w = src.shape
        src_seq = src.flatten(2).transpose(1, 2).contiguous()  # [B, HW, C]
        pos_seq = pos.flatten(2).transpose(1, 2).contiguous()  # [B, HW, C]
        memory = self.encoder(src_seq, pos=pos_seq)  # [B, HW, C]

        query = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # [B, Q, C]
        tgt = torch.zeros_like(query)
        hs_all = self.decoder(
            tgt,
            memory,
            pos=pos_seq,
            query_pos=query,
            return_intermediate=True,
        )  # [L, B, Q, C]

        pred_logits_all = self.class_embed(hs_all)  # [L, B, Q, C+1]
        pred_boxes_all = self.bbox_embed(hs_all).sigmoid()  # [L, B, Q, 4]
        pred_logits = pred_logits_all[-1]
        pred_boxes = pred_boxes_all[-1]

        aux_outputs: List[Dict[str, torch.Tensor]] = []
        if pred_logits_all.size(0) > 1:
            for i in range(pred_logits_all.size(0) - 1):
                aux_outputs.append(
                    {
                        "pred_logits": pred_logits_all[i],
                        "pred_boxes": pred_boxes_all[i],
                        "num_classes": self.num_classes,
                        "has_bg": self.has_bg,
                    }
                )

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "aux_outputs": aux_outputs,
            "num_classes": self.num_classes,
            "has_bg": self.has_bg,
            "feat_hw": (h, w),
        }


VanillaDETRRGGB = DETRRGGB

__all__ = ["DETRRGGB", "VanillaDETRRGGB"]
