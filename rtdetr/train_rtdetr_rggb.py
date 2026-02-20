#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rtdetr_rggb.py

RT-DETR training entry for RGGB (.npy) detection data.
This script intentionally reuses the original `transformer/data` pipeline so
the RGGB loading/resize/padding/target generation behavior stays consistent.
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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
TRANSFORMER_ROOT = os.path.join(PROJECT_ROOT, "transformer")

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if TRANSFORMER_ROOT not in sys.path:
    sys.path.insert(0, TRANSFORMER_ROOT)

from models.rtdetr_rggb import RTDETRRGGB
from configs.config import Cfg
from data.dataloaders import build_dataloaders
from engine.losses.detr_losses import build_detr_detection_loss
from engine.metrics.detr_metrics import AveragePrecisionMeter, CocoAveragePrecisionMeter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        item: Dict[str, Any] = {
            "boxes": t["boxes"].to(device, non_blocking=True).float(),
            "labels": t["labels"].to(device, non_blocking=True).long(),
            "size": t["size"].to(device, non_blocking=True).long(),
        }
        if "id" in t:
            item["id"] = t["id"]
        out.append(item)
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


def save_checkpoint(
    path: str,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    best_map50: float,
    args: argparse.Namespace,
) -> None:
    ckpt = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_map50": float(best_map50),
        "args": vars(args),
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)


def maybe_resume(
    resume_path: Optional[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> Dict[str, float]:
    if not resume_path:
        return {"start_epoch": 0.0, "best_map50": -1.0}
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

    blob = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(blob["model"], strict=True)
    if "optimizer" in blob:
        optimizer.load_state_dict(blob["optimizer"])
    if "scheduler" in blob:
        scheduler.load_state_dict(blob["scheduler"])
    if scaler is not None and isinstance(blob, dict) and "scaler" in blob:
        scaler.load_state_dict(blob["scaler"])

    start_epoch = int(blob.get("epoch", -1)) + 1
    best_map50 = float(blob.get("best_map50", -1.0))
    return {"start_epoch": float(start_epoch), "best_map50": float(best_map50)}


def set_backbone_trainable(model: RTDETRRGGB, trainable: bool) -> None:
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
    skipped_nonfinite = 0

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
            bad_terms = [
                k for k, v in loss_dict.items()
                if isinstance(v, torch.Tensor) and (not torch.isfinite(v).all())
            ]
            print(
                f"[warn] skip non-finite loss at epoch={epoch} iter={it} "
                f"loss={float(loss.item())} bad_terms={bad_terms[:8]}",
                flush=True,
            )
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = None
            if grad_clip > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if (total_norm is not None) and (not torch.isfinite(total_norm)):
                print(f"[warn] skip non-finite grad norm at epoch={epoch} iter={it}", flush=True)
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            total_norm = None
            if grad_clip > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if (total_norm is not None) and (not torch.isfinite(total_norm)):
                print(f"[warn] skip non-finite grad norm at epoch={epoch} iter={it}", flush=True)
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

        running += float(loss.item())
        n += 1

        if (it + 1) % max(1, int(log_interval)) == 0:
            lr_main = float(optimizer.param_groups[0]["lr"])
            lr_backbone = float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else lr_main
            print(
                f"[train] epoch={epoch:03d} iter={it+1:05d} "
                f"loss={running/max(1,n):.4f} lr={lr_main:.3e} lr_backbone={lr_backbone:.3e}",
                flush=True,
            )

    return {
        "loss": running / max(1, n),
        "time_s": time.time() - t0,
        "iters": float(n),
        "skipped_nonfinite": float(skipped_nonfinite),
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


def _base_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--cfg", type=str, default=None, help="Path to JSON with argparse key-values.")
    return ap


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("RT-DETR RGGB trainer")
    ap.add_argument("--cfg", type=str, default=None, help="Path to JSON with argparse key-values.")

    ap.add_argument("--train-merged-json", type=str, default="/home/ryan529/project/transformer/configs/train_merged.json")
    ap.add_argument("--val-merged-json", type=str, default="/home/ryan529/project/transformer/configs/val_merged.json")
    ap.add_argument("--raw-train", type=str, default="/home/ryan529/project/bdd100k/raw10_npy/train")
    ap.add_argument("--raw-val", type=str, default="/home/ryan529/project/bdd100k/raw10_npy/val")
    ap.add_argument("--out", type=str, default="/home/ryan529/project/rtdetr_rggb/runs/rtdetr_rggb")

    ap.add_argument("--epochs", type=int, default=120)
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
    ap.add_argument("--num-queries", type=int, default=300)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--aifi-layers", type=int, default=1)
    ap.add_argument("--enc-layers", type=int, default=1, help="Backward-compat alias for --aifi-layers")
    ap.add_argument("--dec-layers", type=int, default=6)
    ap.add_argument("--dim-feedforward", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--backbone-pretrained", action="store_true")

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-drop", type=int, default=90)
    ap.add_argument("--lr-drop-gamma", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=0.1)
    ap.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16", "fp16"])
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--score-thr", type=float, default=0.0)
    ap.add_argument("--max-dets", type=int, default=100)
    ap.add_argument("--nms-iou", type=float, default=None)
    ap.add_argument("--aux-loss-weight", type=float, default=1.0)
    ap.add_argument("--aux-loss-decay", type=float, default=1.0)

    ap.add_argument("--max-train-iters", type=int, default=0, help="0 means full epoch")
    ap.add_argument("--max-val-iters", type=int, default=0, help="0 means full validation")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=1)
    ap.add_argument("--resume", type=str, default=None)
    return ap


def parse_args() -> argparse.Namespace:
    base_ap = _base_arg_parser()
    base_args, _ = base_ap.parse_known_args()
    defaults: Dict[str, Any] = {}
    if base_args.cfg:
        with open(base_args.cfg, "r", encoding="utf-8") as f:
            defaults = json.load(f)
        if not isinstance(defaults, dict):
            raise ValueError(f"--cfg must be a JSON object, got: {type(defaults).__name__}")

    ap = build_arg_parser()
    if defaults:
        ap.set_defaults(**defaults)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    train_log_path = os.path.join(args.out, "logs", "train.jsonl")
    val_log_path = os.path.join(args.out, "logs", "val.jsonl")

    with open(os.path.join(args.out, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

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
        print(
            f"[warn] dataset.num_classes={ds_num_classes}, override num_classes from {num_classes} to {ds_num_classes}",
            flush=True,
        )
        num_classes = int(ds_num_classes)

    aifi_layers = int(getattr(args, "aifi_layers", getattr(args, "enc_layers", 1)))
    model = RTDETRRGGB(
        num_classes=num_classes,
        num_queries=int(args.num_queries),
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        aifi_layers=int(aifi_layers),
        dec_layers=int(args.dec_layers),
        dim_feedforward=int(args.dim_feedforward),
        dropout=float(args.dropout),
        backbone=str(args.backbone),
        backbone_pretrained=bool(args.backbone_pretrained),
    ).to(device)

    if bool(args.channels_last) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

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

    main_params: List[nn.Parameter] = []
    backbone_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            main_params.append(p)

    param_groups = [{"params": main_params, "lr": float(args.lr)}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": float(args.lr_backbone)})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(args.lr_drop)),
        gamma=float(args.lr_drop_gamma),
    )
    scaler = None
    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler(enabled=str(args.amp).lower() == "fp16")

    resume_info = maybe_resume(
        args.resume,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    start_epoch = int(resume_info["start_epoch"])
    best_map50 = float(resume_info["best_map50"])
    if args.resume:
        print(f"[resume] start_epoch={start_epoch} best_map50={best_map50:.4f} from={args.resume}", flush=True)

    print(
        f"[optim] lr={float(args.lr):.3e} lr_backbone={float(args.lr_backbone):.3e} "
        f"lr_drop={int(args.lr_drop)} gamma={float(args.lr_drop_gamma):.3f} "
        f"main_params={len(main_params)} backbone_params={len(backbone_params)}",
        flush=True,
    )

    nms_iou_eval: Optional[float] = None
    if args.nms_iou is not None:
        nms_iou_eval = float(args.nms_iou)
        if nms_iou_eval < 0:
            nms_iou_eval = None

    for epoch in range(start_epoch, int(args.epochs)):
        if "set_epoch" in loaders and callable(loaders["set_epoch"]):
            loaders["set_epoch"](epoch)

        freeze_backbone = epoch < int(args.freeze_backbone_epochs)
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
            "skipped_nonfinite": tr.get("skipped_nonfinite", 0.0),
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
            f"lr={optimizer.param_groups[0]['lr']:.3e} skipped_nonfinite={int(tr.get('skipped_nonfinite', 0.0))}",
            flush=True,
        )

        save_checkpoint(
            os.path.join(args.out, "last.pt"),
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_map50=max(best_map50, va["mAP50"]),
            args=args,
        )

        if va["mAP50"] > best_map50:
            best_map50 = va["mAP50"]
            save_checkpoint(
                os.path.join(args.out, "best_mAP50.pt"),
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_map50=best_map50,
                args=args,
            )
            print(f"[best] epoch={epoch:03d} mAP50={best_map50:.4f}", flush=True)

        scheduler.step()

    print(f"[done] best_mAP50={best_map50:.4f} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
