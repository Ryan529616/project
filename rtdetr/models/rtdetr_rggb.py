# -*- coding: utf-8 -*-
"""
rtdetr_rggb.py

RT-DETR RGGB model with official-like module topology:
- backbone.conv1 / backbone.res_layers (PResNet-VD style)
- encoder.input_proj / encoder.encoder / encoder.lateral_convs / encoder.fpn_blocks /
  encoder.downsample_convs / encoder.pan_blocks
- decoder.input_proj / decoder.decoder.layers / decoder.{enc,dec}_* heads

Only the first conv input channel is changed from 3 -> 4 for RGGB.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(min=float(eps), max=1.0 - float(eps))
    return torch.log(x / (1.0 - x))


def _get_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name in ("identity", "none"):
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        dims = [int(in_dim)]
        for _ in range(max(1, int(num_layers)) - 1):
            dims.append(int(hidden_dim))
        dims.append(int(out_dim))
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x, inplace=True)
        return x


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = int(num_pos_feats)
        self.temperature = int(temperature)
        self.normalize = bool(normalize)
        self.scale = 2.0 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class ConvNormLayer(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int,
        s: int,
        p: Optional[int] = None,
        *,
        act: str = "silu",
        bias: bool = False,
    ):
        super().__init__()
        if p is None:
            p = int(k) // 2
        self.conv = nn.Conv2d(int(c_in), int(c_out), kernel_size=int(k), stride=int(s), padding=int(p), bias=bool(bias))
        self.norm = nn.BatchNorm2d(int(c_out))
        self.act = _get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvNormLayer(channels, channels, k=3, s=1, p=1, act="silu")
        self.conv2 = ConvNormLayer(channels, channels, k=1, s=1, p=0, act="identity")
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + x)


class CSPRepLayer(nn.Module):
    def __init__(self, c_in: int, c_out: int, num_blocks: int = 3):
        super().__init__()
        self.conv1 = ConvNormLayer(c_in, c_out, k=1, s=1, p=0, act="silu")
        self.conv2 = ConvNormLayer(c_in, c_out, k=1, s=1, p=0, act="silu")
        self.bottlenecks = nn.ModuleList([RepVggBlock(c_out) for _ in range(max(1, int(num_blocks)))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        for block in self.bottlenecks:
            x1 = block(x1)
        return x1 + x2


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


class StemBlock(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.conv1_1 = ConvNormLayer(int(in_channels), 32, k=3, s=2, p=1, act="relu")
        self.conv1_2 = ConvNormLayer(32, 32, k=3, s=1, p=1, act="relu")
        self.conv1_3 = ConvNormLayer(32, 64, k=3, s=1, p=1, act="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        return x


class ShortCut(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        stride = int(stride)
        if stride > 1:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        else:
            self.pool = None
        self.conv = ConvNormLayer(c_in, c_out, k=1, s=1, p=0, act="identity")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            x = self.pool(x)
        return self.conv(x)


class ResBottleNeckBlock(nn.Module):
    expansion = 4

    def __init__(self, c_in: int, c_mid: int, stride: int, shortcut: bool):
        super().__init__()
        stride = int(stride)
        self.stride = stride
        self.shortcut = bool(shortcut)

        self.branch2a = ConvNormLayer(c_in, c_mid, k=1, s=1, p=0, act="relu")
        self.branch2b = ConvNormLayer(c_mid, c_mid, k=3, s=stride, p=1, act="relu")
        self.branch2c = ConvNormLayer(c_mid, c_mid * self.expansion, k=1, s=1, p=0, act="identity")

        if self.shortcut:
            if stride > 1:
                self.short = ShortCut(c_in=c_in, c_out=c_mid * self.expansion, stride=stride)
            else:
                # Keep stage-0 key shape compatible with official checkpoint:
                # backbone.res_layers.0.blocks.0.short.conv.weight
                self.short = ConvNormLayer(c_in, c_mid * self.expansion, k=1, s=1, p=0, act="identity")
        else:
            self.short = None

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branch2c(self.branch2b(self.branch2a(x)))

        if self.short is None:
            identity = x
        else:
            identity = self.short(x)

        out = out + identity
        return self.act(out)


class ResStage(nn.Module):
    def __init__(self, blocks: Sequence[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(list(blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class PResNetBackbone(nn.Module):
    _SUPPORTED = {"resnet50", "r50", "r50vd"}

    def __init__(self, name: str = "resnet50", in_channels: int = 4):
        super().__init__()
        name = str(name).lower()
        if name not in self._SUPPORTED:
            raise ValueError(f"Unsupported backbone for official RT-DETR topology: {name}. Supported: {sorted(self._SUPPORTED)}")

        self.conv1 = StemBlock(in_channels=int(in_channels))
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        depths = (3, 4, 6, 3)
        stage_channels = (64, 128, 256, 512)

        in_c = 64
        stages: List[nn.Module] = []
        for stage_idx, (depth, c_mid) in enumerate(zip(depths, stage_channels)):
            blocks: List[nn.Module] = []
            for block_idx in range(int(depth)):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                shortcut = block_idx == 0
                block = ResBottleNeckBlock(c_in=in_c, c_mid=int(c_mid), stride=int(stride), shortcut=bool(shortcut))
                blocks.append(block)
                in_c = int(c_mid) * ResBottleNeckBlock.expansion
            stages.append(ResStage(blocks))
        self.res_layers = nn.ModuleList(stages)

        # RT-DETR consumes C3/C4/C5 equivalent (512/1024/2048 for R50).
        self.out_indices = (1, 2, 3)
        self.out_channels = (512, 1024, 2048)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.pool2d_max(x)

        outs: List[torch.Tensor] = []
        for i, layer in enumerate(self.res_layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class HybridEncoder(nn.Module):
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
        self.hidden_dim = int(hidden_dim)

        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, self.hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.hidden_dim),
                )
                for c in in_channels
            ]
        )

        self.encoder = nn.ModuleList(
            [
                AIFIBlock(
                    d_model=self.hidden_dim,
                    nhead=int(nhead),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                    num_layers=int(aifi_layers),
                )
            ]
        )

        self.lateral_convs = nn.ModuleList(
            [ConvNormLayer(self.hidden_dim, self.hidden_dim, k=1, s=1, p=0, act="silu") for _ in range(2)]
        )
        self.fpn_blocks = nn.ModuleList(
            [CSPRepLayer(self.hidden_dim * 2, self.hidden_dim, num_blocks=3) for _ in range(2)]
        )

        self.downsample_convs = nn.ModuleList(
            [ConvNormLayer(self.hidden_dim, self.hidden_dim, k=3, s=2, p=1, act="silu") for _ in range(2)]
        )
        self.pan_blocks = nn.ModuleList(
            [CSPRepLayer(self.hidden_dim * 2, self.hidden_dim, num_blocks=3) for _ in range(2)]
        )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        x3, x4, x5 = [proj(f) for proj, f in zip(self.input_proj, feats)]

        for enc in self.encoder:
            x5 = enc(x5)

        # top-down FPN
        inner_outs: List[torch.Tensor] = [x5]
        for i in range(2):
            high = inner_outs[0]
            low = x4 if i == 0 else x3
            up = F.interpolate(self.lateral_convs[i](high), size=low.shape[-2:], mode="nearest")
            inner_outs.insert(0, self.fpn_blocks[i](torch.cat([up, low], dim=1)))

        # bottom-up PAN
        outs: List[torch.Tensor] = [inner_outs[0]]
        for i in range(2):
            down = self.downsample_convs[i](outs[-1])
            outs.append(self.pan_blocks[i](torch.cat([down, inner_outs[i + 1]], dim=1)))
        return outs


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
            value_l = value[:, start:end, :, :]
            value_l = value_l.permute(0, 2, 3, 1).reshape(bsz * self.num_heads, self.head_dim, h_l, w_l)

            sampling_grid = sampling_locations[:, :, :, lvl, :, :]
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
            )
            sampled = sampled.view(bsz, self.num_heads, self.head_dim, num_query, self.num_points)

            attn = attention_weights[:, :, :, lvl, :]
            attn = attn.permute(0, 2, 1, 3).unsqueeze(2)
            out_l = (sampled * attn).sum(dim=-1)
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


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        num_levels: int,
        num_points: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RTDETRDecoderLayer(
                    d_model=int(d_model),
                    nhead=int(nhead),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                    num_levels=int(num_levels),
                    num_points=int(num_points),
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )


class RTDETRDecoder(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        num_queries: int,
        hidden_dim: int,
        nhead: int,
        dec_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_feature_levels: int,
        num_points: int,
        anchor_scale: float,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.hidden_dim = int(hidden_dim)
        self.num_feature_levels = int(num_feature_levels)
        self.anchor_scale = float(anchor_scale)

        self.input_proj = nn.ModuleList(
            [ConvNormLayer(self.hidden_dim, self.hidden_dim, k=1, s=1, p=0, act="identity") for _ in range(self.num_feature_levels)]
        )

        self.decoder = TransformerDecoder(
            d_model=self.hidden_dim,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            num_layers=int(dec_layers),
            num_levels=self.num_feature_levels,
            num_points=int(num_points),
        )

        self.enc_output = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim))
        self.enc_score_head = nn.Linear(self.hidden_dim, self.num_classes)
        self.enc_bbox_head = MLP(self.hidden_dim, self.hidden_dim, 4, num_layers=3)
        self.query_pos_head = MLP(4, self.hidden_dim * 2, self.hidden_dim, num_layers=2)

        self.dec_score_head = nn.ModuleList([nn.Linear(self.hidden_dim, self.num_classes) for _ in self.decoder.layers])
        self.dec_bbox_head = nn.ModuleList([MLP(self.hidden_dim, self.hidden_dim, 4, num_layers=3) for _ in self.decoder.layers])

        # Keep official key for checkpoint compatibility (not used in this training loop).
        self.denoising_class_embed = nn.Embedding(self.num_classes + 1, self.hidden_dim)

        self.pos_embed = PositionEmbeddingSine(num_pos_feats=self.hidden_dim // 2, normalize=True)

    @staticmethod
    def _batch_gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        b, _, c = x.shape
        gather_idx = idx.unsqueeze(-1).expand(b, idx.size(1), c)
        return torch.gather(x, dim=1, index=gather_idx)

    def _build_memory(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_list: List[torch.Tensor] = []
        spatial_shapes: List[Tuple[int, int]] = []
        for feat in features:
            b, _, h, w = feat.shape
            pos = self.pos_embed(feat).flatten(2).transpose(1, 2).contiguous()
            src = feat.flatten(2).transpose(1, 2).contiguous()
            src_list.append(src + pos)
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
        self,
        spatial_shapes: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
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
            wh = torch.full_like(cx, self.anchor_scale * (2.0 ** lvl))
            anchors.append(torch.stack([cx, cy, wh, wh], dim=-1).view(-1, 4))
        return torch.cat(anchors, dim=0).unsqueeze(0)

    def _select_queries(
        self,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors = self._generate_anchors(spatial_shapes, device=memory.device, dtype=memory.dtype)
        enc_feat = self.enc_output(torch.nan_to_num(memory, nan=0.0, posinf=1e4, neginf=-1e4))
        enc_logits = torch.nan_to_num(self.enc_score_head(enc_feat), nan=0.0, posinf=50.0, neginf=-50.0)
        enc_box_unact = self.enc_bbox_head(enc_feat) + _inverse_sigmoid(anchors)
        enc_boxes = torch.nan_to_num(enc_box_unact.sigmoid(), nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-4, 1.0 - 1e-4)

        scores = torch.nan_to_num(enc_logits.max(dim=-1).values, nan=-1e6, posinf=1e6, neginf=-1e6)
        topk_idx = torch.topk(scores, k=min(self.num_queries, scores.size(1)), dim=1).indices

        tgt = self._batch_gather(enc_feat, topk_idx)
        ref_boxes = self._batch_gather(enc_boxes, topk_idx).detach()
        enc_logits_q = self._batch_gather(enc_logits, topk_idx)
        enc_boxes_q = self._batch_gather(enc_boxes, topk_idx)
        return tgt, ref_boxes, enc_logits_q, enc_boxes_q

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]]:
        feats = [proj(f) for proj, f in zip(self.input_proj, features)]
        memory, spatial_shapes, level_start_index = self._build_memory(feats)

        tgt, ref_boxes, enc_logits, enc_boxes = self._select_queries(memory, spatial_shapes)

        outputs_logits: List[torch.Tensor] = []
        outputs_boxes: List[torch.Tensor] = []

        for i, layer in enumerate(self.decoder.layers):
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
            logits_i = torch.nan_to_num(self.dec_score_head[i](tgt), nan=0.0, posinf=50.0, neginf=-50.0)
            box_i = self.dec_bbox_head[i](tgt)
            box_i = (box_i + _inverse_sigmoid(ref_boxes)).sigmoid()
            box_i = torch.nan_to_num(box_i, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-4, 1.0 - 1e-4)
            outputs_logits.append(logits_i)
            outputs_boxes.append(box_i)
            ref_boxes = box_i.detach()

        pred_logits = outputs_logits[-1]
        pred_boxes = outputs_boxes[-1]

        aux_outputs: List[Dict[str, torch.Tensor]] = []
        if len(outputs_logits) > 1:
            for i in range(len(outputs_logits) - 1):
                aux_outputs.append({"pred_logits": outputs_logits[i], "pred_boxes": outputs_boxes[i]})

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "aux_outputs": aux_outputs,
            "enc_aux_outputs": {
                "pred_logits": enc_logits,
                "pred_boxes": enc_boxes,
            },
        }


class RTDETRRGGB(nn.Module):
    """
    RT-DETR RGGB detector with official-like module topology.
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
        del backbone_pretrained  # Not used for official-structure backbone.

        if int(d_model) % int(nhead) != 0:
            raise ValueError(f"d_model({d_model}) must be divisible by nhead({nhead})")
        if int(num_feature_levels) != 3:
            raise ValueError("This implementation expects 3 feature levels (P3/P4/P5).")

        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.d_model = int(d_model)
        self.has_bg = False

        self.backbone = PResNetBackbone(name=backbone, in_channels=4)
        self.encoder = HybridEncoder(
            in_channels=self.backbone.out_channels,
            hidden_dim=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            aifi_layers=int(aifi_layers),
        )
        self.decoder = RTDETRDecoder(
            num_classes=self.num_classes,
            num_queries=self.num_queries,
            hidden_dim=self.d_model,
            nhead=int(nhead),
            dec_layers=int(dec_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            num_feature_levels=int(num_feature_levels),
            num_points=int(num_points),
            anchor_scale=float(anchor_scale),
        )

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

    def forward(self, x: torch.Tensor | Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]]:
        if isinstance(x, dict):
            raw4 = x.get("raw4", None)
            if raw4 is None:
                raise KeyError("RTDETRRGGB expects tensor input or dict containing key 'raw4'")
        else:
            raw4 = x

        feats = self.backbone(raw4)
        feats = self.encoder(feats)
        out = self.decoder(feats)

        out["num_classes"] = self.num_classes
        out["has_bg"] = self.has_bg

        aux = out.get("aux_outputs", None)
        if isinstance(aux, list):
            for item in aux:
                item["num_classes"] = self.num_classes
                item["has_bg"] = self.has_bg

        enc_aux = out.get("enc_aux_outputs", None)
        if isinstance(enc_aux, dict):
            enc_aux["num_classes"] = self.num_classes
            enc_aux["has_bg"] = self.has_bg

        return out


__all__ = ["RTDETRRGGB"]
