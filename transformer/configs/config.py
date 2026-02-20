# -*- coding: utf-8 -*-
"""
config.py

Detector-only configuration schema for the hardware-aware object detector.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return ""


def _to_primitive(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_primitive(v) for v in obj]
    return obj


def _set_by_path(root: Any, path: str, value: Any) -> None:
    cur = root
    parts = [p for p in str(path).split(".") if p]
    if not parts:
        return
    for p in parts[:-1]:
        if isinstance(cur, dict):
            if p not in cur or cur[p] is None:
                cur[p] = {}
            cur = cur[p]
        else:
            if not hasattr(cur, p) or getattr(cur, p) is None:
                setattr(cur, p, {})
            cur = getattr(cur, p)
    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        cur_val = getattr(cur, last, None) if hasattr(cur, last) else None
        if is_dataclass(cur_val) and isinstance(value, dict):
            _apply_dict_overrides(cur_val, value)
        else:
            setattr(cur, last, value)


def _apply_dict_overrides(root: Any, overrides: Dict[str, Any]) -> None:
    for k, v in (overrides or {}).items():
        if "." in str(k):
            _set_by_path(root, str(k), v)
            continue

        if isinstance(root, dict):
            if isinstance(v, dict) and isinstance(root.get(k), dict):
                _apply_dict_overrides(root[k], v)
            else:
                root[k] = v
            continue

        if hasattr(root, k):
            cur = getattr(root, k)
            if is_dataclass(cur) and isinstance(v, dict):
                _apply_dict_overrides(cur, v)
            else:
                setattr(root, k, v)
        else:
            # Destructive mode: allow unknown keys to be attached dynamically.
            setattr(root, k, v)


@dataclass
class DataCfg:
    train_merged_json: Optional[str] = None
    val_merged_json: Optional[str] = None
    merged_json: Optional[str] = None

    train_det_json: Optional[str] = None
    val_det_json: Optional[str] = None

    raw_root: Optional[str] = None
    raw_train: Optional[str] = None
    raw_val: Optional[str] = None
    mask_root: Optional[str] = None

    in_h: int = 360
    in_w: int = 640
    pad_multiple: int = 8
    keep_aspect: bool = False
    ann_src_size: Sequence[int] = field(default_factory=lambda: [1280, 720])
    ann_src_is_hw: bool = False

    batch_size: int = 8
    val_batch_size: int = 8
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = True
    timeout: float = 0.0

    augment: bool = True
    shuffle_train: bool = True
    shuffle_val: bool = False
    allow_empty_val: bool = False
    use_dali: bool = False
    dali_device_id: int = 0
    human_aux_weight: float = 0.3
    shuffle_seed: Optional[int] = None


@dataclass
class DetectorCfg:
    arch: str = "dual_stage_backbone"
    num_classes: int = 4
    num_queries: int = 100
    d_model: int = 256
    nhead: int = 8
    dropout: float = 0.1
    attn_dtype: str = "bf16"
    pred_box_space: str = "normalized"

    dual_backbone_base_c: int = 64
    dual_token_counts: Sequence[int] = field(default_factory=lambda: [32, 64, 128, 256])
    dual_pe_max_h: int = 360
    dual_pe_max_w: int = 640
    dual_bbox_hidden_dim: int = 256
    dual_token_ffn_mult: float = 2.0
    dual_query_prior: str = "none"
    dual_query_prior_init: str = "grid"
    dual_anchor_wh_prior: float = 0.2

    dual_backbone_stem_kind: str = "cfa_grouped"
    dual_backbone_act: str = "relu"
    dual_backbone_bn_eps: float = 1e-5
    dual_backbone_bottleneck_local_blocks: int = 2
    dual_backbone_gc_reduction: int = 8
    dual_backbone_gc_gate: str = "hard_sigmoid"
    dual_backbone_gc_clamp_alpha: float = 0.2
    dual_backbone_gc_clamp_beta: float = 0.5


@dataclass
class DetectorLossCfg:
    cost_class: float = 2.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0

    cls_loss_weight: float = 1.0
    bbox_loss_weight: float = 5.0
    giou_loss_weight: float = 2.0

    eos_coef: float = 0.1
    focal_ce: bool = False
    qfl_align: bool = False
    qfl_weight: float = 0.0

    box_encoding: str = "sigmoid"
    iou_type: str = "giou"
    alpha_iou: float = 2.0
    cls_normalize: str = "num_boxes"

    min_box_wh: float = 0.001
    min_box_penalty: float = 0.01
    wh_log_l1_weight: float = 0.0

    object_token_l2_weight: float = 0.0
    object_token_diversity_weight: float = 0.0
    object_token_diversity_margin: float = 0.0
    object_token_var_weight: float = 0.0
    object_token_var_target: float = 0.25
    scale_token_balance_weight: float = 0.0
    query_bg_balance_weight: float = 0.0
    query_bg_target: float = 0.7

    k_one2many: int = 0
    lambda_one2many: float = 1.0
    aux_apply_one2many: bool = False
    matcher_batch_cpu_copy_max_numel: int = 8_000_000

    debug: bool = False
    debug_every: int = 0
    debug_verbosity: int = 0
    debug_assert_finite: bool = False


@dataclass
class OptimCfg:
    name: str = "adamw"
    lr: float = 2e-4
    lr_detector: float = 2e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    clip_norm_detector: float = 0.6
    fused: Optional[bool] = None
    foreach: Optional[bool] = None

    weight_decay_norm: float = 0.0
    weight_decay_bias: float = 0.0
    skip_names: Optional[List[str]] = None
    skip_regex: Optional[List[str]] = None
    norm_keywords: Tuple[str, ...] = (
        "norm",
        "bn",
        "gn",
        "ln",
        "layernorm",
        "batchnorm",
        "groupnorm",
    )
    bias_keyword: str = ".bias"
    layer_decay: Optional[float] = None
    layer_map: Optional[Dict[str, int]] = None
    layer_id_fn: Optional[str] = None


@dataclass
class SingleSchedCfg:
    type: str = "cosine"
    warmup_steps: int = 0
    warmup_init_lr: Optional[float] = None
    total_steps: int = 100
    min_lr: float = 1e-6

    hold_steps: int = 0
    hold_lr_scale: float = 0.5

    max_lr: Optional[float] = None
    pct_start: float = 0.3
    start_lr: float = 0.0
    final_lr: float = 0.0


@dataclass
class SchedGroupCfg:
    type: str = "cosine"
    name: str = "cosine"
    detector: SingleSchedCfg = field(default_factory=SingleSchedCfg)


@dataclass
class TrainOptCfg:
    train_mode: str = "detector_only"
    enable_detector: bool = True
    freeze_detector: bool = False

    mixed_precision: str = "bf16"
    mixed_precision_detector: str = "bf16"
    grad_accum_steps: int = 1

    channels_last: bool = False
    compile_detector: bool = False
    compile_backend: str = "inductor"
    compile_mode: str = "default"

    allow_tf32: bool = True
    force_math_sdp: bool = False
    skip_finite_checks: bool = True
    loss_guard_policy: str = "fail"

    calc_val_loss: bool = False
    calc_large_ap: bool = False
    val_max_batches: int = 0
    val_log_every: int = 50

    check_grads_after_backward: bool = False
    dump_model_params: bool = True
    grad_firewall: bool = False
    deterministic: bool = False
    debug: bool = False


@dataclass
class DetectorMetricsCfg:
    score_thr: float = 0.0
    max_dets: int = 100
    num_classes: int = 4
    nms_iou: Optional[float] = None


@dataclass
class EvalCfg:
    enable: bool = True
    interval_epochs: int = 1
    save_best_metric: str = "det/mAP50"
    best_metric_mode: str = "max"
    save_best_topk: int = 3
    early_stop_patience: Optional[int] = None
    early_stop_min_delta: float = 0.0


@dataclass
class EMACfg:
    enable_detector: bool = False
    decay: float = 0.9995
    device: Optional[str] = "cpu"
    include_buffers: bool = True


@dataclass
class StepCkptCfg:
    every_n_steps: int = 1000
    out_dir: str = ""
    tag_prefix: str = "g"
    keep_last_k: int = 5
    save_optim: bool = True
    save_sched: bool = True
    save_scaler: bool = True
    save_ema: bool = True


@dataclass
class CkptCfg:
    resume: Optional[str] = None
    init_dual: Optional[str] = None
    init_detector: Optional[str] = None
    init_cast: str = "none"
    init_strip: str = "module.,model.,_orig_mod."
    init_prefer_ema: bool = False

    ckpt_dir: str = ""
    keep_last: int = 6
    interval_steps: int = 1000
    keep_last_steps: int = 5
    save_dual: bool = True
    step: StepCkptCfg = field(default_factory=StepCkptCfg)


@dataclass
class LogDetailCfg:
    grad_norms: bool = True


@dataclass
class LogCfg:
    level: str = "INFO"
    file_name: str = "train.log"

    def setup(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        level = getattr(logging, str(self.level).upper(), logging.INFO)
        handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
        try:
            handlers.append(logging.FileHandler(os.path.join(out_dir, self.file_name), encoding="utf-8"))
        except Exception:
            pass
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=handlers,
            force=True,
        )


@dataclass
class MetaCfg:
    argv: List[str] = field(default_factory=list)
    git_commit: str = ""
    hostname: str = ""
    username: str = ""
    cli_overrides: Dict[str, Any] = field(default_factory=dict)
    cfg_json_overrides: Dict[str, Any] = field(default_factory=dict)
    env_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_primitive(self)


@dataclass
class Cfg:
    seed: int = 1337
    output_dir: str = "runs/exp"
    epochs: int = 120
    log_interval: int = 20

    data: DataCfg = field(default_factory=DataCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    sched: SchedGroupCfg = field(default_factory=SchedGroupCfg)
    ema: EMACfg = field(default_factory=EMACfg)

    loss_detector: DetectorLossCfg = field(default_factory=DetectorLossCfg)

    trainopt: TrainOptCfg = field(default_factory=TrainOptCfg)
    ckpt: CkptCfg = field(default_factory=CkptCfg)

    metrics_detector: DetectorMetricsCfg = field(default_factory=DetectorMetricsCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    log: LogCfg = field(default_factory=LogCfg)
    log_detail: LogDetailCfg = field(default_factory=LogDetailCfg)
    meta: MetaCfg = field(default_factory=MetaCfg)

    def __post_init__(self) -> None:
        self.meta.hostname = socket.gethostname()
        try:
            self.meta.username = getpass.getuser()
        except Exception:
            self.meta.username = ""

    def set_meta_run_context(self, argv: Optional[List[str]] = None) -> None:
        self.meta.argv = list(sys.argv[:] if argv is None else argv)
        self.meta.git_commit = _get_git_commit()

    def apply_overrides(self, overrides: Dict[str, Any], source: str = "cli") -> None:
        if not overrides:
            return
        if source == "cli":
            self.meta.cli_overrides.update(overrides)
        else:
            self.meta.cfg_json_overrides.update(overrides)
        _apply_dict_overrides(self, overrides)

    def setup_logging(self, out_dir: str) -> None:
        self.log.setup(out_dir)

    def validate(self, strict: bool = True) -> None:
        errors: List[str] = []
        if int(self.epochs) <= 0:
            errors.append("epochs must be > 0")
        if int(self.data.batch_size) <= 0:
            errors.append("data.batch_size must be > 0")
        if int(self.data.val_batch_size) <= 0:
            errors.append("data.val_batch_size must be > 0")
        if float(self.optim.lr_detector) <= 0 and float(self.optim.lr) <= 0:
            errors.append("optim.lr_detector or optim.lr must be > 0")
        if int(self.sched.detector.total_steps) <= 0:
            errors.append("sched.detector.total_steps must be > 0")
        if int(self.metrics_detector.max_dets) <= 0:
            errors.append("metrics_detector.max_dets must be > 0")

        if errors and strict:
            raise ValueError("Config validation failed:\n- " + "\n- ".join(errors))

    def gather_env_info(self) -> None:
        env = self.meta.env_info
        env["python"] = sys.version
        env["platform"] = platform.platform()
        try:
            import torch

            env["torch"] = torch.__version__
            env["cuda_available"] = bool(torch.cuda.is_available())
            env["cuda"] = getattr(torch.version, "cuda", None)
            if torch.cuda.is_available():
                try:
                    env["gpu0"] = torch.cuda.get_device_name(0)
                except Exception:
                    pass
        except Exception as e:
            env["torch_error"] = f"{type(e).__name__}: {e}"

    def dump_all(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config_dump.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

        self.gather_env_info()
        with open(os.path.join(out_dir, "env_info.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta.env_info, f, indent=2, ensure_ascii=False)

        with open(os.path.join(out_dir, "overrides_trace.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "argv": self.meta.argv,
                    "cli_overrides": self.meta.cli_overrides,
                    "cfg_json_overrides": self.meta.cfg_json_overrides,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


__all__ = [
    "Cfg",
    "DataCfg",
    "DetectorCfg",
    "DetectorLossCfg",
    "OptimCfg",
    "SingleSchedCfg",
    "SchedGroupCfg",
    "TrainOptCfg",
    "DetectorMetricsCfg",
    "EvalCfg",
    "EMACfg",
    "StepCkptCfg",
    "CkptCfg",
    "LogCfg",
    "LogDetailCfg",
    "MetaCfg",
]
