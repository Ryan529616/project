# optim_sched.py — Near-Recall V4 (detector-aware, safer LLRD, multi-optim support)
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Iterable, Tuple, Dict, Optional, List, Sequence, Callable, Any, Union

import math
import inspect
import re

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

# ------------------------------
# Parameter grouping
# ------------------------------

_NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
)
_NO_WD_TYPES = _NORM_TYPES + (nn.Embedding,)


def _has_kwarg(fn: Callable[..., Any], name: str) -> bool:
    try:
        sig = inspect.signature(fn)
        return name in sig.parameters
    except Exception:
        # 老環境/封裝器，退回原版粗糙判斷
        return name in getattr(getattr(fn, "__code__", None), "co_varnames", ())


def _build_param_to_module(model: nn.Module) -> Dict[int, nn.Module]:
    """
    把每個 Parameter 映射回它所屬 module，方便用 module 型別判斷是否該 weight decay。
    只遍歷一層（recurse=False），避免重複綁定名導致混亂。
    """
    p2m: Dict[int, nn.Module] = {}
    for mod in model.modules():
        for _, p in mod.named_parameters(recurse=False):
            p2m[id(p)] = mod
    return p2m


def param_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    *,
    weight_decay_norm: float = 0.0,
    weight_decay_bias: float = 0.0,
    skip_names: Optional[Iterable[str]] = None,          # 完整匹配或子字串皆可
    skip_regex: Optional[Iterable[str]] = None,          # 進階：regex
    norm_keywords: Sequence[str] = ("norm", "bn", "gn", "ln", "layernorm", "batchnorm", "groupnorm"),
    bias_keyword: str = ".bias",
    layer_decay: Optional[float] = None,                 # 例如 0.9 表示越深 lr 越小
    layer_map: Optional[Dict[str, int]] = None,          # 前綴 → layer_id（0=最前面）
    layer_id_fn: Optional[Callable[[str, nn.Parameter], int]] = None,  # 自定義 name→layer_id
    include_fn: Optional[Callable[[str, nn.Parameter], bool]] = None,  # 若提供，僅對 True 的參數分組
) -> List[Dict]:
    """
    產生 optimizer 參數組：
      - 權重衰減：一般參數用 weight_decay，Norm/Embeddings 用 weight_decay_norm，bias 用 weight_decay_bias
      - 支援 skip_names / skip_regex
      - 支援 Layer-wise LR Decay（LLRD）：layer_map 或 layer_id_fn，否則用「名稱深度」當預設 id
      - 支援 include_fn：可以只對模型的部分參數做 grouping（例如只處理 backbone 部分）
    """
    p2m = _build_param_to_module(model)
    wd_params: List[nn.Parameter] = []
    no_wd_norm: List[nn.Parameter] = []
    no_wd_bias: List[nn.Parameter] = []

    skip_set = set(skip_names or [])
    skip_patterns = [re.compile(rx) for rx in (skip_regex or [])]

    def _should_skip(name: str) -> bool:
        if name in skip_set:
            return True
        for pat in skip_patterns:
            if pat.search(name) is not None:
                return True
        return False

    for n, p in model.named_parameters():
        if include_fn is not None and not include_fn(n, p):
            continue
        if (not p.requires_grad) or _should_skip(n):
            continue

        lname = n.lower()
        mod = p2m.get(id(p), None)

        # 1) bias 一律放 bias 組
        if lname.endswith(bias_keyword) or lname.rsplit(".", 1)[-1] == "bias":
            no_wd_bias.append(p)
            continue

        # 2) 明確的 no-decay 類別（Norm/Embedding 等）
        if isinstance(mod, _NO_WD_TYPES):
            no_wd_norm.append(p)
            continue

        # 3) 退路：名字判斷 norm（避免某些自定義模塊沒在 _NO_WD_TYPES）
        if any(kw in lname for kw in norm_keywords):
            no_wd_norm.append(p)
            continue

        # 4) 維度退路：1D 參數（例如 gamma/beta）通常不 decay
        if p.ndim <= 1:
            no_wd_norm.append(p)
            continue

        # 其餘丟到帶 weight decay 的 bucket
        wd_params.append(p)

    groups: List[Dict] = []
    if wd_params:
        groups.append({"params": wd_params, "weight_decay": float(weight_decay)})
    if no_wd_norm:
        groups.append({"params": no_wd_norm, "weight_decay": float(weight_decay_norm)})
    if no_wd_bias:
        groups.append({"params": no_wd_bias, "weight_decay": float(weight_decay_bias)})

    # -------- Layer-wise LR decay --------
    if layer_decay is not None:
        # 1) 準備前綴表（長的優先）
        prefixes: List[Tuple[str, int]] = []
        if layer_map:
            prefixes = sorted(
                [(k.lower(), int(v)) for k, v in layer_map.items()],
                key=lambda kv: len(kv[0]),
                reverse=True,
            )

        def _lid_from_prefix(name: str) -> int:
            nm = name.lower()
            for pref, lid in prefixes:
                if nm.startswith(pref):
                    return lid
            return -1  # -1 代表沒命中

        def _default_depth_id(name: str) -> int:
            """
            預設退路：用名稱層級深度當 layer id。越深數字越大。
            例如 "backbone.enc3.block.0.conv.weight" → 5
            對於沒有提供 layer_map / layer_id_fn 的情況仍比完全不 decay 好。
            """
            return max(0, name.count("."))

        def _get_lid(name: str, p: nn.Parameter) -> int:
            # 使用者自訂優先
            if callable(layer_id_fn):
                try:
                    lid = int(layer_id_fn(name, p))
                    if lid >= 0:
                        return lid
                except Exception:
                    pass

            # 再看前綴表
            lid = _lid_from_prefix(name)
            if lid >= 0:
                return lid

            # 參數本身可帶 _layer_id（例如在 builder 中先標好）
            lid_attr = getattr(p, "_layer_id", None)
            if isinstance(lid_attr, int) and lid_attr >= 0:
                return lid_attr

            # 最後退路：名稱深度
            return _default_depth_id(name)

        # 建立 param(id) -> name 對應，避免 O(N^2) 掃描
        param_to_name: Dict[int, str] = {id(p): n for n, p in model.named_parameters()}

        new_groups: List[Dict] = []
        for g in groups:
            bucket: Dict[int, List[nn.Parameter]] = {}

            for p in g["params"]:
                nm = param_to_name.get(id(p), "unknown")
                lid = _get_lid(nm, p)
                bucket.setdefault(int(lid), []).append(p)

            # 產生分組，加上 lr_scale
            for lid, plist in bucket.items():
                scale = float(layer_decay) ** int(lid)
                ng = {k: v for k, v in g.items() if k != "params"}
                ng["params"] = plist
                ng["lr_scale"] = scale
                new_groups.append(ng)
        groups = new_groups

    return groups


# ------------------------------
# Optimizer builders
# ------------------------------

def _make_optimizer_from_groups(
    groups: List[Dict],
    *,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    optim_name: str = "adamw",
    **kwargs,
) -> torch.optim.Optimizer:
    """
    給單一 param_group 集合建 optimizer 的內部工具。
    注意：weight_decay 應寫在 groups 裡；這裡只處理演算法/實作選擇。
    """
    optim_name = (optim_name or "adamw").lower()

    # 把 lr_scale 寫入 group lr，讓 scheduler 的 base_lrs 正確
    for g in groups:
        if "lr_scale" in g:
            g["lr"] = lr * float(g.pop("lr_scale"))
        else:
            g.setdefault("lr", lr)

    # 1) Lion
    if optim_name == "lion":
        try:
            from lion_pytorch import Lion  # type: ignore
            return Lion(groups, lr=lr, betas=betas, weight_decay=weight_decay)
        except Exception:
            print("[optim] Lion 不可用，退回 AdamW。")
            optim_name = "adamw"

    # 2) bitsandbytes 8-bit AdamW（可選）
    if optim_name in ("adamw_8bit", "adamw8bit", "bnb_adamw"):
        try:
            import bitsandbytes as bnb  # type: ignore
            return bnb.optim.AdamW8bit(groups, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
        except Exception:
            print("[optim] bitsandbytes 不可用，退回 AdamW。")
            optim_name = "adamw"

    # 3) 原生 AdamW（優先；PyTorch 2.x 支援 fused/foreach）
    if optim_name in ("adamw", "adamw_torch", "adamw_fused", "adamw_foreach"):
        fused = kwargs.get("fused", None)
        foreach = kwargs.get("foreach", None)
        adamw_kwargs: Dict[str, Any] = dict(betas=betas, eps=eps)
        if foreach is not None:
            adamw_kwargs["foreach"] = bool(foreach)

        if hasattr(torch.optim, "AdamW"):
            if fused is None:
                fused = torch.cuda.is_available() and _has_kwarg(torch.optim.AdamW.__init__, "fused")
            if fused and _has_kwarg(torch.optim.AdamW.__init__, "fused"):
                adamw_kwargs["fused"] = True
            return torch.optim.AdamW(groups, **adamw_kwargs)

    # 4) Adam（保底）
    return torch.optim.Adam(groups, lr=lr, betas=betas, eps=eps)


def build_optimizer(
    model: nn.Module,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    傳統單一 optimizer 建構（維持相容性）。
      optim_name: {"adamw","adamw_8bit","lion","adam"}
      其它 kwargs：
        weight_decay_norm / weight_decay_bias / skip_names / skip_regex / norm_keywords
        layer_decay / layer_map / layer_id_fn
        fused: Optional[bool], foreach: Optional[bool]   # 只對原生 AdamW 有效
    """
    optim_name = (kwargs.pop("optim_name", "adamw") or "adamw").lower()

    groups = param_groups(
        model,
        weight_decay=weight_decay,
        weight_decay_norm=kwargs.get("weight_decay_norm", 0.0),
        weight_decay_bias=kwargs.get("weight_decay_bias", 0.0),
        skip_names=kwargs.get("skip_names"),
        skip_regex=kwargs.get("skip_regex"),
        norm_keywords=kwargs.get(
            "norm_keywords",
            ("norm", "bn", "gn", "ln", "layernorm", "batchnorm", "groupnorm"),
        ),
        layer_decay=kwargs.get("layer_decay"),
        layer_map=kwargs.get("layer_map"),
        layer_id_fn=kwargs.get("layer_id_fn"),
    )

    return _make_optimizer_from_groups(
        groups,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        optim_name=optim_name,
        **kwargs,
    )


def available_optimizers() -> List[str]:
    out = ["adam", "adamw"]
    try:
        from lion_pytorch import Lion  # noqa: F401
        out.append("lion")
    except Exception:
        pass
    try:
        import bitsandbytes as bnb  # noqa: F401
        out.append("adamw_8bit")
    except Exception:
        pass
    return out


# -------- Multi-optimizer helpers (Backbone / Detector 等混合架構用) --------

def _build_name_filter(keywords: Sequence[str]) -> Callable[[str, nn.Parameter], bool]:
    kws = tuple(kw.lower() for kw in keywords)

    def _f(name: str, p: nn.Parameter) -> bool:  # noqa: ARG001
        ln = name.lower()
        return any(kw in ln for kw in kws)

    return _f


def build_backbone_detector_optimizers(
    model: nn.Module,
    *,
    lr_backbone: float,
    lr_detector: float,
    weight_decay_backbone: float = 0.01,
    weight_decay_detector: float = 0.01,
    betas_backbone: Tuple[float, float] = (0.9, 0.999),
    betas_detector: Tuple[float, float] = (0.9, 0.999),
    eps_backbone: float = 1e-8,
    eps_detector: float = 1e-8,
    optim_name_backbone: str = "adamw",
    optim_name_detector: str = "adamw",
    backbone_keywords: Sequence[str] = ("backbone",),
    detector_keywords: Sequence[str] = ("detector", "transformer"),
    shared_to: str = "backbone",
    # 其餘 kwargs 將傳給 param_groups / _make_optimizer_from_groups（兩邊共用）
    **kwargs,
) -> Dict[str, torch.optim.Optimizer]:
    """
    建立 backbone / detector 分離的兩個 optimizer（可選擇性再加 shared）。
    通常用法：
      - Backbone：Cosine / 緩和 lr
      - Detector：OneCycle / aggressive lr

    參數：
      - backbone_keywords: 用於 name.contains 判斷屬於 backbone 的參數
      - detector_keywords: 同理，匹配 detector / transformer
      - shared_to: 其他沒匹配到的參數丟給哪一邊（"backbone" 或 "detector"）
    """
    backbone_filter = _build_name_filter(backbone_keywords)
    detector_filter = _build_name_filter(detector_keywords)

    def only_backbone(name: str, p: nn.Parameter) -> bool:
        return backbone_filter(name, p) and not detector_filter(name, p)

    def only_detector(name: str, p: nn.Parameter) -> bool:
        return detector_filter(name, p) and not backbone_filter(name, p)

    def only_shared(name: str, p: nn.Parameter) -> bool:
        return (not backbone_filter(name, p)) and (not detector_filter(name, p))

    # 先分別做 grouping
    backbone_groups = param_groups(
        model,
        weight_decay=weight_decay_backbone,
        include_fn=only_backbone,
        weight_decay_norm=kwargs.get("weight_decay_norm", 0.0),
        weight_decay_bias=kwargs.get("weight_decay_bias", 0.0),
        skip_names=kwargs.get("skip_names"),
        skip_regex=kwargs.get("skip_regex"),
        norm_keywords=kwargs.get(
            "norm_keywords",
            ("norm", "bn", "gn", "ln", "layernorm", "batchnorm", "groupnorm"),
        ),
        layer_decay=kwargs.get("layer_decay_backbone", kwargs.get("layer_decay")),
        layer_map=kwargs.get("layer_map_backbone", kwargs.get("layer_map")),
        layer_id_fn=kwargs.get("layer_id_fn_backbone", kwargs.get("layer_id_fn")),
    )

    detector_groups = param_groups(
        model,
        weight_decay=weight_decay_detector,
        include_fn=only_detector,
        weight_decay_norm=kwargs.get("weight_decay_norm", 0.0),
        weight_decay_bias=kwargs.get("weight_decay_bias", 0.0),
        skip_names=kwargs.get("skip_names"),
        skip_regex=kwargs.get("skip_regex"),
        norm_keywords=kwargs.get(
            "norm_keywords",
            ("norm", "bn", "gn", "ln", "layernorm", "batchnorm", "groupnorm"),
        ),
        layer_decay=kwargs.get("layer_decay_detector", kwargs.get("layer_decay")),
        layer_map=kwargs.get("layer_map_detector", kwargs.get("layer_map")),
        layer_id_fn=kwargs.get("layer_id_fn_detector", kwargs.get("layer_id_fn")),
    )

    shared_groups = param_groups(
        model,
        weight_decay=weight_decay_backbone if shared_to == "backbone" else weight_decay_detector,
        include_fn=only_shared,
        weight_decay_norm=kwargs.get("weight_decay_norm", 0.0),
        weight_decay_bias=kwargs.get("weight_decay_bias", 0.0),
        skip_names=kwargs.get("skip_names"),
        skip_regex=kwargs.get("skip_regex"),
        norm_keywords=kwargs.get(
            "norm_keywords",
            ("norm", "bn", "gn", "ln", "layernorm", "batchnorm", "groupnorm"),
        ),
        layer_decay=kwargs.get("layer_decay"),
        layer_map=kwargs.get("layer_map"),
        layer_id_fn=kwargs.get("layer_id_fn"),
    )

    if shared_groups:
        if shared_to == "backbone":
            backbone_groups.extend(shared_groups)
        elif shared_to == "detector":
            detector_groups.extend(shared_groups)
        else:
            raise ValueError(f"shared_to must be 'backbone' or 'detector', got: {shared_to!r}")

    optim_backbone = _make_optimizer_from_groups(
        backbone_groups,
        lr=lr_backbone,
        weight_decay=weight_decay_backbone,
        betas=betas_backbone,
        eps=eps_backbone,
        optim_name=optim_name_backbone,
        **kwargs,
    )

    optim_detector = _make_optimizer_from_groups(
        detector_groups,
        lr=lr_detector,
        weight_decay=weight_decay_detector,
        betas=betas_detector,
        eps=eps_detector,
        optim_name=optim_name_detector,
        **kwargs,
    )

    return {"backbone": optim_backbone, "detector": optim_detector}


# ------------------------------
# Schedulers
# ------------------------------

class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """
    線性 warmup → 餘弦退火到 min_lr。
    - 支援 total_steps 動態更新
    - 與 PyTorch LRScheduler 相容（.state_dict/.load_state_dict）
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
        warmup_init_lr: Optional[float] = None,   # None: 從 0 線性升到 base_lr
    ):
        assert total_steps >= 0 and warmup_steps >= 0, "steps must be non-negative"
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        self.warmup_init_lr = warmup_init_lr
        super().__init__(optimizer, last_epoch)

    def set_total_steps(self, total_steps: int):
        self.total_steps = int(max(0, total_steps))

    def get_lr(self) -> List[float]:  # type: ignore[override]
        s = self.last_epoch + 1
        base_lrs = self.base_lrs

        # warmup
        if self.warmup_steps > 0 and s <= self.warmup_steps:
            lrs: List[float] = []
            for base in base_lrs:
                start = self.warmup_init_lr if (self.warmup_init_lr is not None) else 0.0
                inc = (base - start) / max(1, self.warmup_steps)
                lrs.append(float(start + inc * s))
            return lrs

        # 沒有退火長度時，直接夾到 min_lr
        decay_len = max(0, self.total_steps - self.warmup_steps)
        if decay_len <= 0:
            return [max(self.min_lr, base) for base in base_lrs]

        # cosine decay
        t = min(max(0, s - self.warmup_steps), decay_len)
        cos = 0.5 * (1.0 + math.cos(math.pi * t / max(1, decay_len)))
        lrs = [self.min_lr + (base - self.min_lr) * cos for base in base_lrs]
        return [max(self.min_lr, lr) for lr in lrs]


class WarmupHoldCosine(torch.optim.lr_scheduler._LRScheduler):
    """
    線性 warmup → 固定小學習率 hold_steps → 餘弦退火。
    用於蒸餾/遷移前期想先小心翼翼摸路。
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        hold_steps: int,
        total_steps: int,
        *,
        hold_lr_scale: float = 0.5,  # 相對 base_lr 的比例
        min_lr: float = 1e-6,
        warmup_init_lr: Optional[float] = None,
        last_epoch: int = -1,
    ):
        assert total_steps >= 0 and warmup_steps >= 0 and hold_steps >= 0
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.hold_steps = int(hold_steps)
        self.total_steps = int(total_steps)
        self.hold_lr_scale = float(hold_lr_scale)
        self.min_lr = float(min_lr)
        self.warmup_init_lr = warmup_init_lr
        super().__init__(optimizer, last_epoch)

    def set_total_steps(self, total_steps: int):
        self.total_steps = int(max(0, total_steps))

    def get_lr(self) -> List[float]:  # type: ignore[override]
        s = self.last_epoch + 1
        base_lrs = self.base_lrs

        # warmup
        if self.warmup_steps > 0 and s <= self.warmup_steps:
            lrs: List[float] = []
            for base in base_lrs:
                start = self.warmup_init_lr if (self.warmup_init_lr is not None) else 0.0
                inc = (base - start) / max(1, self.warmup_steps)
                lrs.append(float(start + inc * s))
            return lrs

        # hold
        if s <= self.warmup_steps + self.hold_steps:
            return [max(self.min_lr, base * self.hold_lr_scale) for base in base_lrs]

        # cosine decay
        decay_len = max(0, self.total_steps - self.warmup_steps - self.hold_steps)
        if decay_len <= 0:
            return [max(self.min_lr, base * self.hold_lr_scale) for base in base_lrs]
        t = min(max(0, s - self.warmup_steps - self.hold_steps), decay_len)
        cos = 0.5 * (1.0 + math.cos(math.pi * t / max(1, decay_len)))
        lrs = [self.min_lr + (base - self.min_lr) * cos for base in base_lrs]
        return [max(self.min_lr, lr) for lr in lrs]


# -------- OneCycle builder --------

def _infer_max_lrs_from_optimizer(optimizer: torch.optim.Optimizer) -> Union[float, List[float]]:
    """
    如果未提供 max_lr，預設用每個 param_group 當前 lr 作為 max_lr。
    這樣可以配合 LLRD（不同 group lr）。
    """
    lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
    return lrs if len(lrs) > 1 else (lrs[0] if lrs else 0.0)


def build_onecycle(
    optimizer: torch.optim.Optimizer,
    *,
    max_lr: Optional[Union[float, List[float]]] = None,
    total_steps: Optional[int] = None,
    pct_start: float = 0.3,
    div_factor: Optional[float] = None,
    final_div_factor: Optional[float] = None,
    start_lr: Optional[float] = None,
    final_lr: Optional[float] = None,
    anneal_strategy: str = "cos",
) -> OneCycleLR:
    """
    OneCycleLR 包裝（支援兩種用法）：

    1) 舊式：指定 div_factor / final_div_factor
       build_onecycle(..., max_lr=..., total_steps=..., div_factor=25, final_div_factor=100)

    2) 新式：指定 start_lr / final_lr（你現在的 cfg 用這種）
       build_onecycle(..., max_lr=4e-4, start_lr=2e-5, final_lr=2e-5, total_steps=2000)

       這時：
         div_factor        = max_lr / start_lr
         final_div_factor  = max_lr / final_lr

    max_lr 可為 float 或 List[float]。若為 None，從 optimizer 當前 group lr 推。
    """
    if total_steps is None:
        raise ValueError("OneCycleLR 需要 total_steps（或自行用 epochs+steps_per_epoch 版本建構）。")

    if max_lr is None:
        max_lr = _infer_max_lrs_from_optimizer(optimizer)

    # 把 max_lr 變成「代表值」算比例；目前假設 cfg 傳進來的是 float（你的 DETR 用法）
    if isinstance(max_lr, (list, tuple)):
        max_lr_scalar = float(max(max_lr)) if max_lr else 0.0
    else:
        max_lr_scalar = float(max_lr)

    # 從 start_lr / final_lr 推 div_factor / final_div_factor（若呼叫端沒顯式給）
    if div_factor is None:
        if start_lr is not None and start_lr > 0.0 and max_lr_scalar > 0.0:
            div_factor = max_lr_scalar / float(start_lr)
        else:
            div_factor = 25.0

    if final_div_factor is None:
        if final_lr is not None and final_lr > 0.0 and max_lr_scalar > 0.0:
            final_div_factor = max_lr_scalar / float(final_lr)
        else:
            final_div_factor = 100.0

    return OneCycleLR(
        optimizer=optimizer,
        max_lr=max_lr,                # float 或 List[float]
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=float(div_factor),
        final_div_factor=float(final_div_factor),
        anneal_strategy=anneal_strategy,
    )


# ------------------------------
# Utilities
# ------------------------------

class GradClipper:
    """輕量級梯度裁剪器（只裁有 grad 的參數，避免無意義遍歷）。"""
    def __init__(self, max_norm: Optional[float] = None, clip_value: Optional[float] = None):
        self.max_norm = max_norm
        self.clip_value = clip_value

    def step(self, model: nn.Module):
        params = [p for p in model.parameters() if p.grad is not None]
        if not params:
            return
        if self.clip_value is not None:
            torch.nn.utils.clip_grad_value_(params, self.clip_value)
        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, self.max_norm)


def build_cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    *,
    min_lr: float = 1e-6,
    warmup_init_lr: Optional[float] = None,
) -> CosineWithWarmup:
    return CosineWithWarmup(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=min_lr,
        last_epoch=-1,
        warmup_init_lr=warmup_init_lr,
    )


def build_warmup_hold_cosine(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    hold_steps: int,
    total_steps: int,
    *,
    hold_lr_scale: float = 0.5,
    min_lr: float = 1e-6,
    warmup_init_lr: Optional[float] = None,
) -> WarmupHoldCosine:
    return WarmupHoldCosine(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        hold_steps=hold_steps,
        total_steps=total_steps,
        hold_lr_scale=hold_lr_scale,
        min_lr=min_lr,
        warmup_init_lr=warmup_init_lr,
        last_epoch=-1,
    )


def summarize_param_groups(optim: torch.optim.Optimizer) -> Dict[str, int]:
    """
    回傳每個 group 的參數量統計，還有是否帶 decay。
    幫你肉眼確認 grouping 有沒有長歪。
    """
    out: Dict[str, int] = {}
    for i, g in enumerate(optim.param_groups):
        key = f"g{i}_wd{g.get('weight_decay', 0.0):.6f}"
        out[key] = sum(p.numel() for p in g["params"])
    return out


# ------------------------------
# Unified scheduler factory
# ------------------------------

def available_schedulers() -> List[str]:
    return ["cosine", "warmup_hold_cosine", "onecycle"]


def build_scheduler_from_cfg(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    統一入口：依 cfg["type"] 建 scheduler。
    支援：
      - "cosine":         CosineWithWarmup
         參數: warmup_steps, total_steps, min_lr, warmup_init_lr
      - "warmup_hold_cosine": WarmupHoldCosine
         參數: warmup_steps, hold_steps, total_steps, hold_lr_scale, min_lr, warmup_init_lr
      - "onecycle":       OneCycleLR
         參數: max_lr(可選, float或list), total_steps, pct_start,
              div_factor/final_div_factor 或 start_lr/final_lr（二擇一概念上）
    """
    stype = (cfg.get("type", "cosine") or "cosine").lower()

    if stype in ("cos", "cosine"):
        return build_cosine_with_warmup(
            optimizer=optimizer,
            warmup_steps=int(cfg.get("warmup_steps", 0)),
            total_steps=int(cfg.get("total_steps", 0)),
            min_lr=float(cfg.get("min_lr", 1e-6)),
            warmup_init_lr=cfg.get("warmup_init_lr", None),
        )

    if stype in ("warmup_hold_cosine", "cosine_hold", "holdcos"):
        return build_warmup_hold_cosine(
            optimizer=optimizer,
            warmup_steps=int(cfg.get("warmup_steps", 0)),
            hold_steps=int(cfg.get("hold_steps", 0)),
            total_steps=int(cfg.get("total_steps", 0)),
            hold_lr_scale=float(cfg.get("hold_lr_scale", 0.5)),
            min_lr=float(cfg.get("min_lr", 1e-6)),
            warmup_init_lr=cfg.get("warmup_init_lr", None),
        )

    if stype in ("onecycle", "one_cycle", "one-cycle"):
        max_lr = cfg.get("max_lr", None)
        return build_onecycle(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=int(cfg.get("total_steps", 0)),
            pct_start=float(cfg.get("pct_start", 0.3)),
            div_factor=float(cfg.get("div_factor", 25.0)) if "div_factor" in cfg else None,
            final_div_factor=float(cfg.get("final_div_factor", 100.0)) if "final_div_factor" in cfg else None,
            start_lr=cfg.get("start_lr", None),
            final_lr=cfg.get("final_lr", None),
            anneal_strategy=str(cfg.get("anneal_strategy", "cos")),
        )

    raise ValueError(f"Unknown scheduler type: {stype}. Supported: {available_schedulers()}")


# ------------------------------
# Multi-optimizer scheduler factory（例如 backbone → cosine, detector → onecycle）
# ------------------------------

def build_multi_schedulers_from_cfg(
    optimizers: Dict[str, torch.optim.Optimizer],
    cfg_map: Dict[str, Dict[str, Any]],
) -> Dict[str, torch.optim.lr_scheduler._LRScheduler]:
    """
    針對多個 optimizer（例如 {"backbone": opt_b, "detector": opt_d}），
    依照 cfg_map 中的子配置建立對應 scheduler。
    用法示意：
        optims = build_backbone_detector_optimizers(...)
        sched_cfg = {
            "backbone": {"type": "cosine", "warmup_steps": 500, "total_steps": total_steps},
            "detector": {"type": "onecycle", "max_lr": 1e-3, "total_steps": total_steps},
        }
        scheds = build_multi_schedulers_from_cfg(optims, sched_cfg)
    """
    scheds: Dict[str, torch.optim.lr_scheduler._LRScheduler] = {}
    for key, optim in optimizers.items():
        sub_cfg = cfg_map.get(key)
        if not sub_cfg:
            continue
        scheds[key] = build_scheduler_from_cfg(optim, sub_cfg)
    return scheds
