#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_vanilla_detr_rggb.py

Pure vanilla DETR baseline for RGGB (4-channel) input:
RGGB -> ResNet50 backbone -> Transformer encoder/decoder -> DETR heads.

This script intentionally does NOT use the hardware-aware detector path.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Ensure project root is importable when launched from tools/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import Cfg
from data.dataloaders import build_dataloaders
from engine.losses.detr_losses import build_detr_detection_loss
from engine.metrics.detr_metrics import AveragePrecisionMeter, CocoAveragePrecisionMeter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True, scale: Optional[float] = None):
        super().__init__()
        self.num_pos_feats = int(num_pos_feats)
        self.temperature = int(temperature)
        self.normalize = bool(normalize)
        self.scale = float(scale) if scale is not None else 2.0 * np.pi

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
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

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
            [TransformerEncoderLayerWithPos(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        out = src
        for layer in self.layers:
            out = layer(out, pos=pos)
        return self.norm(out)


class TransformerDecoderLayerWithPos(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

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
            [TransformerDecoderLayerWithPos(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

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


class VanillaDETRRGGB(nn.Module):
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
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.has_bg = True

        backbone = torchvision.models.resnet50(weights=None)
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

        self.input_proj = nn.Conv2d(2048, d_model, kernel_size=1)
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)

        self.encoder = TransformerEncoderWithPos(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=enc_layers,
        )
        self.decoder = TransformerDecoderWithPos(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=dec_layers,
        )

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_embed = nn.Linear(d_model, self.num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, num_layers=3)

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

        b, c, h, w = src.shape
        src_seq = src.flatten(2).transpose(1, 2).contiguous()  # [B, HW, C]
        pos_seq = pos.flatten(2).transpose(1, 2).contiguous()  # [B, HW, C]
        memory = self.encoder(src_seq, pos=pos_seq)           # [B, HW, C]

        query = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # [B, Q, C]
        tgt = torch.zeros_like(query)
        hs_all = self.decoder(
            tgt, memory, pos=pos_seq, query_pos=query, return_intermediate=True
        )  # [L, B, Q, C]

        pred_logits_all = self.class_embed(hs_all)         # [L, B, Q, C+1]
        pred_boxes_all = self.bbox_embed(hs_all).sigmoid() # [L, B, Q, 4]
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


def autocast_ctx(device: torch.device, amp_mode: str):
    mode = str(amp_mode).lower()
    if device.type != "cuda" or mode in ("none", "off", "fp32"):
        return contextlib.nullcontext()
    if mode in ("fp16", "float16", "half"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def batch_to_targets(batch_targets: List[Dict[str, Any]], device: torch.device) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in batch_targets:
        o: Dict[str, Any] = {
            "boxes": t["boxes"].to(device, non_blocking=True).float(),
            "labels": t["labels"].to(device, non_blocking=True).long(),
            "size": t["size"].to(device, non_blocking=True).long(),
        }
        if "id" in t:
            o["id"] = t["id"]
        out.append(o)
    return out


def make_data_cfg(args: argparse.Namespace) -> Cfg:
    cfg = Cfg()
    cfg.seed = int(args.seed)
    cfg.data.train_merged_json = args.train_merged_json
    cfg.data.val_merged_json = args.val_merged_json
    cfg.data.raw_train = args.raw_train
    cfg.data.raw_val = args.raw_val
    cfg.data.in_h = int(args.in_h)
    cfg.data.in_w = int(args.in_w)
    cfg.data.pad_multiple = int(args.pad_multiple)
    cfg.data.batch_size = int(args.batch_size)
    cfg.data.val_batch_size = int(args.val_batch_size)
    cfg.data.num_workers = int(args.num_workers)
    cfg.data.pin_memory = bool(args.pin_memory)
    cfg.data.persistent_workers = bool(args.persistent_workers)
    cfg.data.prefetch_factor = int(args.prefetch_factor)
    cfg.data.drop_last = bool(args.drop_last)
    cfg.data.shuffle_train = True
    cfg.data.shuffle_val = False
    cfg.data.use_dali = False
    cfg.data.augment = bool(args.augment)
    cfg.data.allow_empty_val = False
    cfg.data.ann_src_size = [1280, 720]
    cfg.data.ann_src_is_hw = False
    return cfg


def write_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_official_detr_init(model: nn.Module, ckpt_path: Optional[str]) -> Dict[str, int]:
    if not ckpt_path or (not os.path.isfile(ckpt_path)):
        return {"loaded": 0, "matched": 0, "total_src": 0}

    blob = torch.load(ckpt_path, map_location="cpu")
    src = blob.get("model", blob) if isinstance(blob, dict) else blob
    if not isinstance(src, dict):
        return {"loaded": 0, "matched": 0, "total_src": 0}

    dst = model.state_dict()
    mapped: Dict[str, torch.Tensor] = {}
    matched = 0

    for k, v in src.items():
        if not torch.is_tensor(v):
            continue
        nk: Optional[str] = None

        if k.startswith("backbone.0.body."):
            tail = k[len("backbone.0.body.") :]
            if tail.startswith("conv1."):
                nk = "backbone.0." + tail[len("conv1.") :]
            elif tail.startswith("bn1."):
                nk = "backbone.1." + tail[len("bn1.") :]
            elif tail.startswith("layer1."):
                nk = "backbone.4." + tail[len("layer1.") :]
            elif tail.startswith("layer2."):
                nk = "backbone.5." + tail[len("layer2.") :]
            elif tail.startswith("layer3."):
                nk = "backbone.6." + tail[len("layer3.") :]
            elif tail.startswith("layer4."):
                nk = "backbone.7." + tail[len("layer4.") :]
        elif k.startswith("transformer.encoder."):
            nk = "encoder." + k[len("transformer.encoder.") :]
        elif k.startswith("transformer.decoder."):
            nk = "decoder." + k[len("transformer.decoder.") :]
        elif k.startswith("bbox_embed.layers."):
            tail = k[len("bbox_embed.layers.") :]  # e.g. "0.weight"
            parts = tail.split(".", 1)
            if len(parts) == 2:
                old_i, rest = parts
                idx_map = {"0": "0", "1": "2", "2": "4"}
                if old_i in idx_map:
                    nk = f"bbox_embed.net.{idx_map[old_i]}.{rest}"
        elif k in ("query_embed.weight", "input_proj.weight", "input_proj.bias"):
            nk = k

        if nk is None or nk not in dst:
            continue

        if nk == "backbone.0.weight" and v.shape[1] == 3 and dst[nk].shape[1] == 4:
            w = dst[nk].clone()
            w[:, :3].copy_(v)
            w[:, 3].copy_(v.mean(dim=1))
            mapped[nk] = w
            matched += 1
            continue

        if tuple(v.shape) != tuple(dst[nk].shape):
            continue

        mapped[nk] = v
        matched += 1

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    _ = missing, unexpected
    return {"loaded": len(mapped), "matched": matched, "total_src": len(src)}


def set_backbone_trainable(model: VanillaDETRRGGB, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = bool(trainable)


def train_one_epoch(
    *,
    epoch: int,
    model: nn.Module,
    criterion: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    amp_mode: str,
    grad_clip: float,
    log_interval: int,
    channels_last: bool,
    max_train_iters: int,
    freeze_backbone: bool,
) -> Dict[str, float]:
    model.train()
    if freeze_backbone:
        model.backbone.eval()
    criterion.train()
    t0 = time.time()
    running = 0.0
    n = 0

    for it, batch in enumerate(loader):
        if max_train_iters > 0 and it >= max_train_iters:
            break

        raw4 = batch["raw4"].to(device, non_blocking=True)
        if channels_last and raw4.device.type == "cuda":
            raw4 = raw4.to(memory_format=torch.channels_last)
        targets = batch_to_targets(batch["targets"], device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, amp_mode):
            out = model(raw4)
            loss_dict = criterion(out, targets)
            loss = sum(v for v in loss_dict.values())

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected at epoch={epoch} iter={it}: {float(loss.item())}")

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running += float(loss.item())
        n += 1
        if (it + 1) % max(1, int(log_interval)) == 0:
            lr_main = float(optimizer.param_groups[0]["lr"])
            lr_backbone = float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else lr_main
            print(
                f"[train] epoch={epoch:03d} iter={it+1:05d} loss={running/max(1,n):.4f} "
                f"lr={lr_main:.3e} lr_backbone={lr_backbone:.3e}",
                flush=True,
            )

    return {
        "loss": running / max(1, n),
        "time_s": time.time() - t0,
        "iters": float(n),
    }


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    criterion: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    amp_mode: str,
    score_thr: float,
    max_dets: int,
    nms_iou: Optional[float],
    channels_last: bool,
    max_val_iters: int,
) -> Dict[str, float]:
    model.eval()
    criterion.eval()
    t0 = time.time()

    meter50 = AveragePrecisionMeter(num_classes=num_classes, has_background=True)
    meter_coco = CocoAveragePrecisionMeter(num_classes=num_classes, has_background=True)

    loss_sum = 0.0
    n = 0
    for it, batch in enumerate(loader):
        if max_val_iters > 0 and it >= max_val_iters:
            break

        raw4 = batch["raw4"].to(device, non_blocking=True)
        if channels_last and raw4.device.type == "cuda":
            raw4 = raw4.to(memory_format=torch.channels_last)
        targets = batch_to_targets(batch["targets"], device)

        with autocast_ctx(device, amp_mode):
            out = model(raw4)
            loss_dict = criterion(out, targets)
            loss = sum(v for v in loss_dict.values())
        loss_sum += float(loss.item())
        n += 1

        meter50.update(out, targets, score_thr=score_thr, max_dets=max_dets, nms_iou=nms_iou)
        meter_coco.update(out, targets, score_thr=score_thr, max_dets=max_dets, nms_iou=nms_iou)

    res50 = meter50.compute(iou_thrs=0.5)
    rescoco = meter_coco.compute()
    iou_thrs = rescoco["iou_thrs"]

    map50 = float(res50["map"][0].item()) if isinstance(res50["map"], torch.Tensor) else float(res50["map"])
    map5095 = float(rescoco["map"].mean().item()) if isinstance(rescoco["map"], torch.Tensor) else float(rescoco["map"])
    map75 = 0.0
    if isinstance(iou_thrs, torch.Tensor):
        idx75 = int(torch.argmin(torch.abs(iou_thrs - 0.75)).item())
        map75 = float(rescoco["map"][idx75].item())

    return {
        "loss": loss_sum / max(1, n),
        "mAP50": map50,
        "mAP": map5095,
        "mAP75": map75,
        "time_s": time.time() - t0,
        "iters": float(n),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Pure vanilla DETR baseline for RGGB")
    ap.add_argument("--train-merged-json", type=str, default="/home/ryan529/project/transformer/configs/train_merged.json")
    ap.add_argument("--val-merged-json", type=str, default="/home/ryan529/project/transformer/configs/val_merged.json")
    ap.add_argument("--raw-train", type=str, default="/home/ryan529/project/bdd100k/raw10_npy/train")
    ap.add_argument("--raw-val", type=str, default="/home/ryan529/project/bdd100k/raw10_npy/val")
    ap.add_argument("--out", type=str, default="/home/ryan529/project/transformer/runs/vanilla_detr_rggb_official_like")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--val-batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--persistent-workers", action="store_true")
    ap.add_argument("--drop-last", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--in-h", type=int, default=360)
    ap.add_argument("--in-w", type=int, default=640)
    ap.add_argument("--pad-multiple", type=int, default=8)
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--num-queries", type=int, default=100)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--enc-layers", type=int, default=6)
    ap.add_argument("--dec-layers", type=int, default=6)
    ap.add_argument("--dim-feedforward", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-drop", type=int, default=100)
    ap.add_argument("--lr-drop-gamma", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=0.1)
    ap.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16", "fp16"])
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--score-thr", type=float, default=0.0)
    ap.add_argument("--max-dets", type=int, default=100)
    ap.add_argument("--nms-iou", type=float, default=None)
    ap.add_argument("--aux-loss-weight", type=float, default=1.0)
    ap.add_argument("--aux-loss-decay", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-train-iters", type=int, default=0, help="0 means full epoch")
    ap.add_argument("--max-val-iters", type=int, default=0, help="0 means full validation")
    ap.add_argument(
        "--rggb-freeze-backbone-epochs",
        type=int,
        default=5,
        help="Freeze ResNet backbone in early epochs to stabilize RGGB 4-channel stem adaptation.",
    )
    ap.add_argument(
        "--init-official-detr",
        type=str,
        default="/home/ryan529/.cache/torch/hub/checkpoints/detr-r50-e632da11.pth",
        help="Official DETR-R50 checkpoint path (optional)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    train_log_path = os.path.join(args.out, "logs", "train.jsonl")
    val_log_path = os.path.join(args.out, "logs", "val.jsonl")

    set_seed(int(args.seed))
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device} amp={args.amp} out={args.out}", flush=True)

    cfg = make_data_cfg(args)
    loaders = build_dataloaders(cfg)
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]

    ds_num_classes = getattr(train_loader.dataset, "num_classes", None)
    num_classes = int(args.num_classes)
    if isinstance(ds_num_classes, int) and ds_num_classes > 0 and ds_num_classes != num_classes:
        print(f"[warn] dataset.num_classes={ds_num_classes}, overriding num_classes from {num_classes} to {ds_num_classes}", flush=True)
        num_classes = int(ds_num_classes)

    model = VanillaDETRRGGB(
        num_classes=num_classes,
        num_queries=int(args.num_queries),
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        enc_layers=int(args.enc_layers),
        dec_layers=int(args.dec_layers),
        dim_feedforward=int(args.dim_feedforward),
        dropout=float(args.dropout),
    ).to(device)

    if bool(args.channels_last) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    init_report = load_official_detr_init(model, args.init_official_detr)
    if init_report["loaded"] > 0:
        print(
            f"[init] loaded official DETR weights: {init_report['loaded']} tensors "
            f"(matched={init_report['matched']}, src_total={init_report['total_src']})",
            flush=True,
        )
    else:
        print("[init] official DETR init not loaded (path missing or no matched keys)", flush=True)

    criterion = build_detr_detection_loss(
        num_classes,
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        iou_type="giou",
        eos_coef=0.1,
        cls_loss_weight=1.0,
        bbox_loss_weight=5.0,
        giou_loss_weight=2.0,
        aux_loss_weight=float(args.aux_loss_weight),
        aux_loss_decay=float(args.aux_loss_decay),
        qfl_align=False,
        qfl_weight=0.0,
        itc_weight=0.0,
        saliency_consistency_weight=0.0,
        frr_gamma=0.0,
        k_one2many=0,
        lambda_one2many=0.0,
        aux_apply_one2many=False,
        cls_normalize="num_boxes",
        matcher_batch_cpu_copy_max_numel=8_000_000,
    ).to(device)

    lr_main = float(args.lr)
    lr_backbone = float(args.lr_backbone)
    main_params: List[nn.Parameter] = []
    backbone_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            main_params.append(p)

    param_groups = [{"params": main_params, "lr": lr_main}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr_main,
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(args.lr_drop)),
        gamma=float(args.lr_drop_gamma),
    )

    print(
        f"[optim] lr={lr_main:.3e} lr_backbone={lr_backbone:.3e} "
        f"lr_drop={int(args.lr_drop)} gamma={float(args.lr_drop_gamma):.3f} "
        f"main_params={len(main_params)} backbone_params={len(backbone_params)}",
        flush=True,
    )

    nms_iou_eval: Optional[float] = None
    if args.nms_iou is not None:
        nms_iou_eval = float(args.nms_iou)
        if nms_iou_eval < 0:
            nms_iou_eval = None

    scaler = None
    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler(enabled=str(args.amp).lower() == "fp16")

    best_map50 = -1.0
    for epoch in range(int(args.epochs)):
        if "set_epoch" in loaders and callable(loaders["set_epoch"]):
            loaders["set_epoch"](epoch)

        freeze_backbone = epoch < int(args.rggb_freeze_backbone_epochs)
        set_backbone_trainable(model, trainable=(not freeze_backbone))

        tr = train_one_epoch(
            epoch=epoch,
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_mode=args.amp,
            grad_clip=float(args.grad_clip),
            log_interval=int(args.log_interval),
            channels_last=bool(args.channels_last),
            max_train_iters=int(args.max_train_iters),
            freeze_backbone=freeze_backbone,
        )
        va = evaluate(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            num_classes=num_classes,
            amp_mode=args.amp,
            score_thr=float(args.score_thr),
            max_dets=int(args.max_dets),
            nms_iou=nms_iou_eval,
            channels_last=bool(args.channels_last),
            max_val_iters=int(args.max_val_iters),
        )

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        train_row = {
            "ts": now,
            "epoch": epoch,
            "loss_total": tr["loss"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "lr_backbone": float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else float(optimizer.param_groups[0]["lr"]),
            "freeze_backbone": bool(freeze_backbone),
            "time_s": tr["time_s"],
            "iters": tr["iters"],
        }
        val_row = {
            "ts": now,
            "epoch": epoch,
            "loss_total": va["loss"],
            "det/mAP50": va["mAP50"],
            "det/mAP": va["mAP"],
            "det/mAP75": va["mAP75"],
            "time_s": va["time_s"],
            "iters": va["iters"],
        }
        write_jsonl(train_log_path, train_row)
        write_jsonl(val_log_path, val_row)

        print(
            f"[eval] epoch={epoch:03d} train_loss={tr['loss']:.4f} "
            f"val_loss={va['loss']:.4f} mAP50={va['mAP50']:.4f} mAP={va['mAP']:.4f} mAP75={va['mAP75']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}",
            flush=True,
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_map50": max(best_map50, va["mAP50"]),
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.out, "last.pt"))

        if va["mAP50"] > best_map50:
            best_map50 = va["mAP50"]
            torch.save(ckpt, os.path.join(args.out, "best_mAP50.pt"))
            print(f"[best] epoch={epoch:03d} mAP50={best_map50:.4f}", flush=True)

        scheduler.step()

    print(f"[done] best_mAP50={best_map50:.4f} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
