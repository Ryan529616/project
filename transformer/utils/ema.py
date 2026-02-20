# ema.py — Exponential Moving Average (EMA) V2
# -*- coding: utf-8 -*-
"""
Exponential Moving Average (EMA) for PyTorch models

特點：
- AMP / dtype / device 安全（shadow 會跟隨參數 dtype、可選常駐 CPU）
- DDP 友好：自動去掉 'module.' 前綴做鍵名對齊
- Warmup-aware decay（timm 風格）：d = min(base_decay, (1+u)/(10+u))
- Buffer 同步策略可選：'copy'（常見做法）或 'ema'（對 BN running stats 做指數移動平均）
- 支援 store()/restore() 與 apply_to(context manager) 做臨時 EMA 推論
- synchronize()：若模型結構變更，可把 shadow 的鍵補齊或移除過期鍵
"""

from __future__ import annotations
import contextlib
from typing import Dict, Optional, Literal
import torch
import torch.nn as nn


def _norm_name(n: str) -> str:
    """去掉 DDP 'module.' 前綴，統一鍵名。"""
    return n[7:] if n.startswith("module.") else n


class EMA:
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        *,
        device: Optional[str] = None,                  # None: 跟隨各參數裝置；'cpu' 可省顯存
        include_buffers: bool = False,                 # 是否處理 buffers（如 BN running stats）
        buffer_strategy: Literal["copy", "ema"] = "copy",
    ):
        """
        Args:
            model: 來源模型
            decay: 基礎衰減因子（越接近 1 越平滑）
            device: 影子權重的存放裝置；預設跟隨參數，或指定 'cpu'
            include_buffers: 是否同步 buffers
            buffer_strategy: 'copy' 直接覆寫最新值；'ema' 也做 EMA
        """
        if not (0.0 < decay < 1.0):
            raise ValueError("decay must be in (0,1)")
        if buffer_strategy not in ("copy", "ema"):
            raise ValueError("buffer_strategy must be 'copy' or 'ema'")

        self.decay = float(decay)
        self.device = device
        self.include_buffers = bool(include_buffers)
        self.buffer_strategy = buffer_strategy

        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        self.num_updates: int = 0

        self._register_from(model)

    # ---------------- internal iters ----------------
    def _iter_params(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                yield _norm_name(n), p

    def _iter_buffers(self, model: nn.Module):
        if not self.include_buffers:
            return
        for n, b in model.named_buffers():
            # BN running stats、位置編碼 cache 等
            yield f"buffer::{_norm_name(n)}", b

    @torch.no_grad()
    def _register_from(self, model: nn.Module):
        """從模型初始化 shadow 字典。"""
        self.shadow.clear()
        for n, p in self._iter_params(model):
            self.shadow[n] = p.detach().clone().to(
                device=self.device if self.device is not None else p.device,
                dtype=p.dtype
            )
        if self.include_buffers:
            for n, b in self._iter_buffers(model):
                self.shadow[n] = b.detach().clone().to(
                    device=self.device if self.device is not None else b.device,
                    dtype=b.dtype
                )

    # ---------------- public API ----------------
    @torch.no_grad()
    def update(self, model: nn.Module, num_updates: Optional[int] = None):
        """
        用當前模型權重更新 EMA。
        Warmup-aware：d = min(base_decay, (1+u)/(10+u))，u 為更新次數。
        """
        if num_updates is None:
            self.num_updates += 1
        else:
            self.num_updates = int(num_updates)

        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))  # keep ratio for shadow
        one_minus_d = 1.0 - d

        # 參數做 true EMA
        for n, p in self._iter_params(model):
            s = self.shadow.get(n, None)
            if s is None:
                # 新增參數（模型改版），直接註冊
                self.shadow[n] = p.detach().clone().to(
                    device=self.device if self.device is not None else p.device,
                    dtype=p.dtype
                )
                continue
            if s.device != (self.device or p.device) or s.dtype != p.dtype:
                s = s.to(device=(self.device or p.device), dtype=p.dtype)
                self.shadow[n] = s
            p_src = p.detach()
            if p_src.device != s.device or p_src.dtype != s.dtype:
                p_src = p_src.to(device=s.device, dtype=s.dtype)
            s.mul_(d).add_(p_src, alpha=one_minus_d)

        # buffers：copy 或 ema
        if self.include_buffers:
            for n, b in self._iter_buffers(model):
                s = self.shadow.get(n, None)
                if s is None:
                    self.shadow[n] = b.detach().clone().to(
                        device=self.device if self.device is not None else b.device,
                        dtype=b.dtype
                    )
                    continue
                if s.device != (self.device or b.device) or s.dtype != b.dtype:
                    s = s.to(device=(self.device or b.device), dtype=b.dtype)
                    self.shadow[n] = s
                b_src = b.detach()
                if b_src.device != s.device or b_src.dtype != s.dtype:
                    b_src = b_src.to(device=s.device, dtype=s.dtype)
                if self.buffer_strategy == "copy":
                    s.copy_(b_src)
                else:  # "ema"
                    s.mul_(d).add_(b_src, alpha=one_minus_d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """
        用 shadow 覆寫模型參數（與可選 buffers）。
        如需之後還原，先呼叫 store()。
        """
        for n, p in self._iter_params(model):
            s = self.shadow.get(n, None)
            if s is not None:
                p.data.copy_(s.to(device=p.device, dtype=p.dtype))
        if self.include_buffers:
            for n, b in self._iter_buffers(model):
                s = self.shadow.get(n, None)
                if s is not None:
                    b.data.copy_(s.to(device=b.device, dtype=b.dtype))

    @torch.no_grad()
    def store(self, model: nn.Module):
        """儲存當前模型權重（與可選 buffers），以便之後 restore。"""
        self._backup = {}
        for n, p in self._iter_params(model):
            self._backup[n] = p.detach().clone()
        if self.include_buffers:
            for n, b in self._iter_buffers(model):
                self._backup[n] = b.detach().clone()

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """還原先前 store() 的權重。"""
        if not self._backup:
            return
        for n, p in self._iter_params(model):
            if n in self._backup:
                p.data.copy_(self._backup[n].to(device=p.device, dtype=p.dtype))
        if self.include_buffers:
            for n, b in self._iter_buffers(model):
                if n in self._backup:
                    b.data.copy_(self._backup[n].to(device=b.device, dtype=b.dtype))
        self._backup.clear()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """導出可序列化狀態（shadow 強制搬到 CPU，避免跨裝置炸鍋）。"""
        shadow_cpu = {k: v.detach().to("cpu") for k, v in self.shadow.items()}
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "include_buffers": self.include_buffers,
            "buffer_strategy": self.buffer_strategy,
            "shadow": shadow_cpu,
        }

    def load_state_dict(self, sd: Dict[str, torch.Tensor]):
        """載入先前保存的 EMA 狀態。"""
        self.decay = float(sd.get("decay", self.decay))
        self.num_updates = int(sd.get("num_updates", 0))
        self.include_buffers = bool(sd.get("include_buffers", self.include_buffers))
        self.buffer_strategy = str(sd.get("buffer_strategy", self.buffer_strategy))
        self.shadow = {k: v.clone() for k, v in sd["shadow"].items()}

    @torch.no_grad()
    def to(self, device: str):
        """把 shadow tensors 全部搬到指定裝置（例如移到 'cpu' 省顯存）。"""
        self.device = device
        for k, v in list(self.shadow.items()):
            self.shadow[k] = v.to(device=device)
        return self

    @torch.no_grad()
    def synchronize(self, model: nn.Module):
        """
        若你替換/擴增了模型模組，呼叫此函式讓 shadow 的鍵與當前模型對齊：
        - 新增缺的鍵（從模型複製）
        - 刪除已不存在的鍵
        """
        param_keys = {_norm_name(n) for n, p in model.named_parameters() if p.requires_grad}
        buf_keys = set()
        if self.include_buffers:
            buf_keys = {f"buffer::{_norm_name(n)}" for n, _ in model.named_buffers()}

        wanted = param_keys | buf_keys
        have = set(self.shadow.keys())

        # add missing
        for n, p in self._iter_params(model):
            if n not in have:
                self.shadow[n] = p.detach().clone().to(
                    device=self.device if self.device is not None else p.device,
                    dtype=p.dtype
                )
        if self.include_buffers:
            for n, b in self._iter_buffers(model):
                if n not in have:
                    self.shadow[n] = b.detach().clone().to(
                        device=self.device if self.device is not None else b.device,
                        dtype=b.dtype
                    )
        # drop stale
        for k in have - wanted:
            self.shadow.pop(k, None)

    @contextlib.contextmanager
    def apply_to(self, model: nn.Module):
        """
        with ema.apply_to(model):
            evaluate(...)
        """
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)
