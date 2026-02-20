# -*- coding: utf-8 -*-
"""
rtdetr_rggb.py

Baidu RT-DETR style detector adapted for 4-channel RGGB input.
Main structure:
1) Multi-scale CNN backbone (C3/C4/C5)
2) Hybrid Encoder (AIFI + cross-scale feature fusion)
3) IoU/score-aware top-k query selection from encoder memory
4) Deformable Transformer Decoder with iterative box refinement
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(min=float(eps), max=1.0 - float(eps))
    return torch.log(x / (1.0 - x))


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
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = int(num_pos_feats)
        self.temperature = int(temperature)
        self.normalize = bool(normalize)
        self.scale = 2.0 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
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


class ConvNormAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = int(k) // 2
        self.conv = nn.Conv2d(int(c_in), int(c_out), kernel_size=int(k), stride=int(s), padding=int(p), bias=False)
        self.bn = nn.BatchNorm2d(int(c_out))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class RepBlock(nn.Module):
    """Simple conv block used for CCFM-like fusion."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.cv1 = ConvNormAct(c_in, c_out, k=1, s=1, p=0)
        self.cv2 = ConvNormAct(c_out, c_out, k=3, s=1, p=1)
        self.cv3 = ConvNormAct(c_out, c_out, k=3, s=1, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(self.cv2(self.cv1(x)))


class TransformerEncoderLayerWithPos(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(int(d_model), int(nhead), dropout=float(dropout), batch_first=True)
        self.linear1 = nn.Linear(int(d_model), int(dim_feedforward))
        self.linear2 = nn.Linear(int(dim_feedforward), int(d_model))
        self.norm1 = nn.LayerNorm(int(d_model))
        self.norm2 = nn.LayerNorm(int(d_model))
        self.drop1 = nn.Dropout(float(dropout))
        self.drop2 = nn.Dropout(float(dropout))
        self.drop_ffn = nn.Dropout(float(dropout))

    def forward(self, src: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        q = src if pos is None else (src + pos)
        src2 = self.self_attn(q, q, value=src, need_weights=False)[0]
        src = self.norm1(src + self.drop1(src2))
        src2 = self.linear2(self.drop_ffn(F.relu(self.linear1(src), inplace=True)))
        src = self.norm2(src + self.drop2(src2))
        return src


class AIFIBlock(nn.Module):
    """
    Attention-based Intra-scale Feature Interaction on the highest-level feature map.
    """

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
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=int(d_model) // 2, normalize=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pos = self.pos_embed(x).flatten(2).transpose(1, 2).contiguous()
        src = x.flatten(2).transpose(1, 2).contiguous()
        for layer in self.layers:
            src = layer(src, pos=pos)
        return src.transpose(1, 2).reshape(b, c, h, w).contiguous()


class ResNetRGGBBackbone(nn.Module):
    _SUPPORTED = {"resnet18", "resnet34", "resnet50"}

    def __init__(self, name: str = "resnet50", pretrained: bool = False):
        super().__init__()
        name = str(name).lower()
        if name not in self._SUPPORTED:
            raise ValueError(f"Unsupported backbone: {name}. Supported: {sorted(self._SUPPORTED)}")

        if name == "resnet18":
            weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            net = torchvision.models.resnet18(weights=weights)
            self.out_channels = (128, 256, 512)
        elif name == "resnet34":
            weights = torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
            net = torchvision.models.resnet34(weights=weights)
            self.out_channels = (128, 256, 512)
        else:
            weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
            net = torchvision.models.resnet50(weights=weights)
            self.out_channels = (512, 1024, 2048)

        old_conv1 = net.conv1
        new_conv1 = nn.Conv2d(
            4,
            old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )
        with torch.no_grad():
            if old_conv1.weight.shape[1] == 3:
                new_conv1.weight[:, :3].copy_(old_conv1.weight)
                new_conv1.weight[:, 3].copy_(old_conv1.weight.mean(dim=1))
            else:
                nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
        net.conv1 = new_conv1

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)  # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32
        return [c3, c4, c5]


class HybridEncoder(nn.Module):
    """
    RT-DETR style Hybrid Encoder:
    - AIFI on top-level feature
    - Cross-scale fusion (FPN top-down + PAN bottom-up)
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        hidden_dim: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        aifi_layers: int,
    ):
        super().__init__()
        in_channels = [int(c) for c in in_channels]
        self.in_proj = nn.ModuleList([nn.Conv2d(c, int(hidden_dim), kernel_size=1, bias=False) for c in in_channels])

        self.aifi = AIFIBlock(
            d_model=int(hidden_dim),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            num_layers=int(aifi_layers),
        )

        self.reduce_p5 = ConvNormAct(hidden_dim, hidden_dim, k=1, s=1, p=0)
        self.reduce_p4 = ConvNormAct(hidden_dim, hidden_dim, k=1, s=1, p=0)
        self.fpn4 = RepBlock(hidden_dim * 2, hidden_dim)
        self.fpn3 = RepBlock(hidden_dim * 2, hidden_dim)

        self.down_p3 = ConvNormAct(hidden_dim, hidden_dim, k=3, s=2, p=1)
        self.down_p4 = ConvNormAct(hidden_dim, hidden_dim, k=3, s=2, p=1)
        self.pan4 = RepBlock(hidden_dim * 2, hidden_dim)
        self.pan5 = RepBlock(hidden_dim * 2, hidden_dim)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        x3, x4, x5 = [proj(f) for proj, f in zip(self.in_proj, feats)]
        x5 = self.aifi(x5)

        p5 = x5
        up_p5 = F.interpolate(self.reduce_p5(p5), size=x4.shape[-2:], mode="nearest")
        p4 = self.fpn4(torch.cat([x4, up_p5], dim=1))

        up_p4 = F.interpolate(self.reduce_p4(p4), size=x3.shape[-2:], mode="nearest")
        p3 = self.fpn3(torch.cat([x3, up_p4], dim=1))

        n4 = self.pan4(torch.cat([self.down_p3(p3), p4], dim=1))
        n5 = self.pan5(torch.cat([self.down_p4(n4), p5], dim=1))
        return [p3, n4, n5]


class MSDeformableAttention(nn.Module):
    """
    Pure-PyTorch multi-scale deformable attention (no custom CUDA op).
    """

    def __init__(self, embed_dim: int, num_heads: int, num_levels: int, num_points: int):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_levels = int(num_levels)
        self.num_points = int(num_points)
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim({embed_dim}) must be divisible by num_heads({num_heads})")
        self.head_dim = self.embed_dim // self.num_heads

        self.sampling_offsets = nn.Linear(self.embed_dim, self.num_heads * self.num_levels * self.num_points * 2)
        self.attention_weights = nn.Linear(self.embed_dim, self.num_heads * self.num_levels * self.num_points)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.reshape(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # query: [B, Q, C]
        # reference_points: [B, Q, L, 2] or [B, Q, L, 4]
        # value: [B, S, C]
        bsz, num_query, _ = query.shape
        _, num_value, _ = value.shape
        if int(spatial_shapes[:, 0].mul(spatial_shapes[:, 1]).sum().item()) != int(num_value):
            raise ValueError("sum(H_l*W_l) must equal value sequence length")

        value = self.value_proj(value)
        if padding_mask is not None:
            value = value.masked_fill(padding_mask[..., None], 0.0)
        value = value.view(bsz, num_value, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            bsz, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bsz, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            bsz, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.size(-1) == 2:
            offset_normalizer = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[
                None, None, None, :, None, :
            ]
        elif reference_points.size(-1) == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / float(self.num_points) * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError("reference_points last dim must be 2 or 4")

        output = query.new_zeros((bsz, num_query, self.num_heads, self.head_dim))
        for lvl, (h_l, w_l) in enumerate(spatial_shapes.tolist()):
            h_l, w_l = int(h_l), int(w_l)
            start = int(level_start_index[lvl].item())
            end = start + h_l * w_l
            value_l = value[:, start:end, :, :]  # [B, Hl*Wl, H, D]
            value_l = value_l.permute(0, 2, 3, 1).reshape(bsz * self.num_heads, self.head_dim, h_l, w_l)

            sampling_grid = sampling_locations[:, :, :, lvl, :, :]  # [B,Q,H,P,2]
            sampling_grid = sampling_grid.permute(0, 2, 1, 3, 4).reshape(
                bsz * self.num_heads, num_query, self.num_points, 2
            )
            sampling_grid = sampling_grid * 2.0 - 1.0

            sampled = F.grid_sample(
                value_l,
                sampling_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # [B*H, D, Q, P]
            sampled = sampled.view(bsz, self.num_heads, self.head_dim, num_query, self.num_points)

            attn = attention_weights[:, :, :, lvl, :]  # [B,Q,H,P]
            attn = attn.permute(0, 2, 1, 3).unsqueeze(2)  # [B,H,1,Q,P]
            out_l = (sampled * attn).sum(dim=-1)  # [B,H,D,Q]
            output = output + out_l.permute(0, 3, 1, 2)

        output = output.reshape(bsz, num_query, self.embed_dim)
        return self.output_proj(output)


class RTDETRDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_levels: int,
        num_points: int,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(int(d_model), int(nhead), dropout=float(dropout), batch_first=True)
        self.cross_attn = MSDeformableAttention(
            embed_dim=int(d_model),
            num_heads=int(nhead),
            num_levels=int(num_levels),
            num_points=int(num_points),
        )
        self.linear1 = nn.Linear(int(d_model), int(dim_feedforward))
        self.linear2 = nn.Linear(int(dim_feedforward), int(d_model))

        self.norm1 = nn.LayerNorm(int(d_model))
        self.norm2 = nn.LayerNorm(int(d_model))
        self.norm3 = nn.LayerNorm(int(d_model))
        self.drop1 = nn.Dropout(float(dropout))
        self.drop2 = nn.Dropout(float(dropout))
        self.drop3 = nn.Dropout(float(dropout))
        self.drop_ffn = nn.Dropout(float(dropout))

    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt, need_weights=False)[0]
        tgt = self.norm1(tgt + self.drop1(tgt2))

        tgt2 = self.cross_attn(
            query=tgt + query_pos,
            reference_points=reference_points,
            value=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        tgt = self.norm2(tgt + self.drop2(tgt2))

        tgt2 = self.linear2(self.drop_ffn(F.relu(self.linear1(tgt), inplace=True)))
        tgt = self.norm3(tgt + self.drop3(tgt2))
        return tgt


class RTDETRRGGB(nn.Module):
    """
    RT-DETR style detector for RGGB inputs.
    """

    def __init__(
        self,
        *,
        num_classes: int = 4,
        num_queries: int = 300,
        d_model: int = 256,
        nhead: int = 8,
        dec_layers: int = 6,
        aifi_layers: int = 1,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        backbone: str = "resnet50",
        backbone_pretrained: bool = False,
        num_feature_levels: int = 3,
        num_points: int = 4,
        anchor_scale: float = 0.05,
    ):
        super().__init__()
        if int(d_model) % int(nhead) != 0:
            raise ValueError(f"d_model({d_model}) must be divisible by nhead({nhead})")
        if int(num_feature_levels) != 3:
            raise ValueError("This implementation expects 3 feature levels (P3/P4/P5).")

        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.d_model = int(d_model)
        self.has_bg = True
        self.num_feature_levels = int(num_feature_levels)
        self.anchor_scale = float(anchor_scale)

        self.backbone = ResNetRGGBBackbone(name=backbone, pretrained=bool(backbone_pretrained))
        self.encoder = HybridEncoder(
            in_channels=self.backbone.out_channels,
            hidden_dim=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            aifi_layers=int(aifi_layers),
        )

        self.level_embed = nn.Embedding(self.num_feature_levels, self.d_model)
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=self.d_model // 2, normalize=True)

        self.enc_output = nn.Linear(self.d_model, self.d_model)
        self.enc_output_norm = nn.LayerNorm(self.d_model)
        self.enc_score_head = nn.Linear(self.d_model, self.num_classes + 1)
        self.enc_bbox_head = MLP(self.d_model, self.d_model, 4, num_layers=3)

        self.query_pos_head = MLP(4, self.d_model, self.d_model, num_layers=2)

        self.decoder_layers = nn.ModuleList(
            [
                RTDETRDecoderLayer(
                    d_model=self.d_model,
                    nhead=int(nhead),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                    num_levels=self.num_feature_levels,
                    num_points=int(num_points),
                )
                for _ in range(max(1, int(dec_layers)))
            ]
        )
        self.dec_score_heads = nn.ModuleList([nn.Linear(self.d_model, self.num_classes + 1) for _ in self.decoder_layers])
        self.dec_bbox_heads = nn.ModuleList([MLP(self.d_model, self.d_model, 4, num_layers=3) for _ in self.decoder_layers])

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.level_embed.weight, mean=0.0, std=0.02)

    @staticmethod
    def _batch_gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # x: [B,N,C], idx: [B,K]
        b, _, c = x.shape
        gather_idx = idx.unsqueeze(-1).expand(b, idx.size(1), c)
        return torch.gather(x, dim=1, index=gather_idx)

    def _build_memory(
        self, features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_list: List[torch.Tensor] = []
        spatial_shapes: List[Tuple[int, int]] = []
        for lvl, feat in enumerate(features):
            b, _, h, w = feat.shape
            pos = self.pos_embed(feat).flatten(2).transpose(1, 2).contiguous()
            src = feat.flatten(2).transpose(1, 2).contiguous()
            src = src + pos + self.level_embed.weight[lvl].view(1, 1, -1).to(dtype=src.dtype, device=src.device)
            src_list.append(src)
            spatial_shapes.append((h, w))

        memory = torch.cat(src_list, dim=1)
        spatial_shapes_t = torch.tensor(spatial_shapes, dtype=torch.long, device=memory.device)
        level_start_index = torch.cat(
            [
                spatial_shapes_t.new_zeros((1,)),
                spatial_shapes_t[:, 0].mul(spatial_shapes_t[:, 1]).cumsum(dim=0)[:-1],
            ]
        )
        return memory, spatial_shapes_t, level_start_index

    def _generate_anchors(
        self, spatial_shapes: torch.Tensor, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # Return anchors in normalized cxcywh: [1, sum(HW), 4]
        anchors: List[torch.Tensor] = []
        for lvl, (h, w) in enumerate(spatial_shapes.tolist()):
            h, w = int(h), int(w)
            y, x = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype),
                indexing="ij",
            )
            cx = (x + 0.5) / float(max(1, w))
            cy = (y + 0.5) / float(max(1, h))
            wh = torch.full_like(cx, self.anchor_scale * (2.0**lvl))
            anchor_l = torch.stack([cx, cy, wh, wh], dim=-1).view(-1, 4)
            anchors.append(anchor_l)
        return torch.cat(anchors, dim=0).unsqueeze(0)

    def _select_queries(
        self, memory: torch.Tensor, spatial_shapes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors = self._generate_anchors(spatial_shapes, device=memory.device, dtype=memory.dtype)
        enc_feat = self.enc_output_norm(self.enc_output(memory))
        enc_feat = torch.nan_to_num(enc_feat, nan=0.0, posinf=1e4, neginf=-1e4)
        enc_logits = self.enc_score_head(enc_feat)
        enc_logits = torch.nan_to_num(enc_logits, nan=0.0, posinf=50.0, neginf=-50.0)
        enc_box_unact = self.enc_bbox_head(enc_feat) + _inverse_sigmoid(anchors)
        enc_boxes = enc_box_unact.sigmoid()
        enc_boxes = torch.nan_to_num(enc_boxes, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-4, 1.0 - 1e-4)

        fg_scores = enc_logits[..., : self.num_classes].max(dim=-1).values
        fg_scores = torch.nan_to_num(fg_scores, nan=-1e6, posinf=1e6, neginf=-1e6)
        topk_idx = torch.topk(fg_scores, k=min(self.num_queries, fg_scores.size(1)), dim=1).indices

        tgt = self._batch_gather(enc_feat, topk_idx)
        ref_boxes = self._batch_gather(enc_boxes, topk_idx).detach()
        enc_logits_q = self._batch_gather(enc_logits, topk_idx)
        enc_boxes_q = self._batch_gather(enc_boxes, topk_idx)
        return tgt, ref_boxes, enc_logits_q, enc_boxes_q

    def forward(self, x: torch.Tensor | Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]]:
        if isinstance(x, dict):
            raw4 = x.get("raw4", None)
            if raw4 is None:
                raise KeyError("RTDETRRGGB expects tensor input or dict containing key 'raw4'")
        else:
            raw4 = x

        feats = self.backbone(raw4)
        feats = self.encoder(feats)  # [P3,P4,P5], each [B,C,H,W]
        memory, spatial_shapes, level_start_index = self._build_memory(feats)

        tgt, ref_boxes, enc_logits, enc_boxes = self._select_queries(memory, spatial_shapes)

        outputs_logits: List[torch.Tensor] = []
        outputs_boxes: List[torch.Tensor] = []

        for i, layer in enumerate(self.decoder_layers):
            query_pos = self.query_pos_head(ref_boxes)
            ref_points = ref_boxes[:, :, None, :].repeat(1, 1, self.num_feature_levels, 1)
            tgt = layer(
                tgt=tgt,
                query_pos=query_pos,
                reference_points=ref_points,
                memory=memory,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
            logits_i = self.dec_score_heads[i](tgt)
            logits_i = torch.nan_to_num(logits_i, nan=0.0, posinf=50.0, neginf=-50.0)
            box_delta = self.dec_bbox_heads[i](tgt)
            box_i = (box_delta + _inverse_sigmoid(ref_boxes)).sigmoid()
            box_i = torch.nan_to_num(box_i, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-4, 1.0 - 1e-4)
            outputs_logits.append(logits_i)
            outputs_boxes.append(box_i)
            ref_boxes = box_i.detach()

        pred_logits = torch.nan_to_num(outputs_logits[-1], nan=0.0, posinf=50.0, neginf=-50.0)
        pred_boxes = torch.nan_to_num(outputs_boxes[-1], nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-4, 1.0 - 1e-4)
        aux_outputs: List[Dict[str, torch.Tensor]] = []
        if len(outputs_logits) > 1:
            for i in range(len(outputs_logits) - 1):
                aux_outputs.append(
                    {
                        "pred_logits": outputs_logits[i],
                        "pred_boxes": outputs_boxes[i],
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
            "enc_aux_outputs": {
                "pred_logits": enc_logits,
                "pred_boxes": enc_boxes,
                "num_classes": self.num_classes,
                "has_bg": self.has_bg,
            },
        }


__all__ = ["RTDETRRGGB"]
