# -*- coding: utf-8 -*-
"""
trainlog.py — Ultra-detailed, step-aligned logging for your training loop.

Features
- JSONL per-phase logs (train.jsonl, val.jsonl) with strict step alignment
- Optional TensorBoard scalars/histograms (auto-detect; disabled if unavailable)
- Lightweight event/ckpt index logs (events.jsonl, ckpt_index.jsonl)
- Robust tensor→float conversion; avoids NaN/Inf in outputs
- Flat keys preserved (e.g., "det/f1", "det/mAP50"); minimal ceremony
- Safe, crash-resilient (flush on every write)

Typical use
-----------
from trainlog import StepLogger

logger = StepLogger(output_dir="runs/1107", run_name="exp1", enable_tb=True)
logger.log_event("run_start", git_rev="...", note="...")

# each step
logger.log_metrics(
    phase="train",
    global_step=gs, epoch=ep, it=it,
    metrics={**stats, "lr_detector": lr_det},
)

# on checkpoint
logger.log_checkpoint(path, global_step=gs, epoch=ep, it=it)

# on close
logger.close()
"""

from __future__ import annotations
import os, io, json, time, math
from typing import Any, Dict, Optional

# Optional TB
try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter  # type: ignore
except Exception:
    _TBWriter = None  # type: ignore


# ---------------------------
# Utilities
# ---------------------------

def _now_ts() -> float:
    return float(time.time())

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _to_float(v: Any) -> Optional[float]:
    """Convert tensors/numbers/strings to float when reasonable; return None if impossible."""
    try:
        # Short-circuit common numeric types
        if isinstance(v, (int, float)):
            if math.isfinite(float(v)):
                return float(v)
            return None
        # Torch tensor lazy import avoidance: treat by duck typing
        if hasattr(v, "detach") and hasattr(v, "cpu") and hasattr(v, "item"):
            try:
                x = float(v.detach().cpu().item())
                if math.isfinite(x):
                    return x
                return None
            except Exception:
                return None
        # Single-item containers
        if isinstance(v, (list, tuple)) and len(v) == 1:
            return _to_float(v[0])
        # Strings that look like numbers
        if isinstance(v, str):
            try:
                x = float(v)
                if math.isfinite(x):
                    return x
            except Exception:
                pass
            return None
        # Fallback: try float(..)
        x = float(v)
        if math.isfinite(x):
            return x
        return None
    except Exception:
        return None

def _sanitize_metrics(d: Dict[str, Any]) -> Dict[str, float]:
    """Keep only finite scalars; drop non-numerics to keep logs lean and safe."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        f = _to_float(v)
        if f is not None and math.isfinite(f):
            out[str(k)] = float(f)
    return out


# ---------------------------
# Logger
# ---------------------------

class StepLogger:
    """
    Step-aligned multi-sink logger:
      - JSONL per phase (train/val/…)
      - TB scalars (optional)
      - ckpt index
      - events JSONL
    """

    def __init__(
        self,
        output_dir: str,
        run_name: Optional[str] = None,
        *,
        enable_console: bool = True,
        enable_jsonl: bool = True,
        enable_tb: bool = True,
        tb_dirname: str = "tb",
        json_dirname: str = "logs",
        flush_every_write: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.run_name = run_name or "run"
        self.enable_console = bool(enable_console)
        self.enable_jsonl = bool(enable_jsonl)
        self.enable_tb = bool(enable_tb and (_TBWriter is not None))
        self.flush_every_write = bool(flush_every_write)

        # dirs
        _ensure_dir(self.output_dir)
        self.json_dir = os.path.join(self.output_dir, json_dirname)
        _ensure_dir(self.json_dir)
        self.tb_dir = os.path.join(self.output_dir, tb_dirname)
        if self.enable_tb:
            _ensure_dir(self.tb_dir)

        # file handles
        self._fh_train = self._open_jsonl("train.jsonl") if self.enable_jsonl else None
        self._fh_val   = self._open_jsonl("val.jsonl") if self.enable_jsonl else None
        self._fh_ev    = self._open_jsonl("events.jsonl") if self.enable_jsonl else None
        self._fh_ckpt  = self._open_jsonl("ckpt_index.jsonl") if self.enable_jsonl else None

        # TB
        self.tb = _TBWriter(log_dir=os.path.join(self.tb_dir, self.run_name)) if self.enable_tb else None

        # write run header
        if metadata is None:
            metadata = {}
        self.log_event("run_init", **metadata)

    # ---------- low-level ----------

    def _open_jsonl(self, name: str):
        p = os.path.join(self.json_dir, name)
        # newline='' to avoid doubling newlines on Windows; UTF-8 always
        return open(p, "a", buffering=1, encoding="utf-8", newline="")

    def _write_jsonl(self, fh, payload: Dict[str, Any]) -> None:
        if fh is None:
            return
        try:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if self.flush_every_write:
                fh.flush()
        except Exception:
            # last-ditch: try best-effort minified
            s = "{"+",".join(f"\"{k}\":\"{str(v)}\"" for k,v in payload.items())+"}\n"
            try:
                fh.write(s)
                if self.flush_every_write:
                    fh.flush()
            except Exception:
                pass

    # ---------- public API ----------

    def log_event(self, evt: str, **kwargs: Any) -> None:
        """Small structured event line into events.jsonl."""
        payload = {"ts": _now_ts(), "run": self.run_name, "event": str(evt)}
        payload.update({k: v for k, v in kwargs.items()})
        if self.enable_console:
            print(f"[event] {evt} | {kwargs}")
        self._write_jsonl(self._fh_ev, payload)

    def log_checkpoint(self, path: str, *, global_step: int, epoch: int, it: int, **meta: Any) -> None:
        """Index a saved checkpoint for later resume/analysis."""
        payload = {
            "ts": _now_ts(),
            "run": self.run_name,
            "ckpt_path": os.path.abspath(path),
            "global_step": int(global_step),
            "epoch": int(epoch),
            "iter": int(it),
        }
        payload.update(meta)
        if self.enable_console:
            print(f"[ckpt] step={global_step} epoch={epoch} it={it} path={path}")
        self._write_jsonl(self._fh_ckpt, payload)

    def log_metrics(
        self,
        *,
        phase: str,                 # "train" | "val" | ...
        global_step: int,
        epoch: int,
        it: int,
        metrics: Dict[str, Any],
        wall_time: Optional[float] = None,
    ) -> None:
        """Core per-step metrics logging (JSONL + TB)."""
        phase = str(phase)
        wall_time = float(wall_time if wall_time is not None else _now_ts())

        m = _sanitize_metrics(metrics)
        payload = {
            "ts": wall_time,
            "run": self.run_name,
            "phase": phase,
            "global_step": int(global_step),
            "epoch": int(epoch),
            "iter": int(it),
        }
        payload.update(m)

        # Console (compact)
        if self.enable_console:
            # pick a few headline keys if present
            head = []
            for k in (
                "loss_total",
                "loss_detector",
                "det/mAP50",
                "det/mAP",
                "det/f1",
                "det/mean_iou",
                "hw/object_token_offdiag_cos2",
            ):
                if k in m:
                    head.append(f"{k}={m[k]:.4f}")
            print(f"[{phase} step {global_step:07d}] " + " ".join(head))

        # JSONL
        if self.enable_jsonl:
            if phase == "train":
                self._write_jsonl(self._fh_train, payload)
            elif phase == "val":
                self._write_jsonl(self._fh_val, payload)
            else:
                # dump unrecognized phases to events
                self._write_jsonl(self._fh_ev, {"event":"metrics", **payload})

        # TB
        if self.tb is not None:
            for k, v in m.items():
                try:
                    self.tb.add_scalar(f"{phase}/{k}", float(v), global_step)
                except Exception:
                    pass

    def add_histogram(self, tag: str, values: Any, global_step: int) -> None:
        """Optional TB histogram; ignored if TB disabled."""
        if self.tb is None:
            return
        try:
            # torch.Tensor or numpy arrays expected; we try add_histogram directly
            self.tb.add_histogram(tag, values, global_step=global_step)
        except Exception:
            pass

    def add_text(self, tag: str, text: str, global_step: int) -> None:
        if self.tb is None:
            return
        try:
            self.tb.add_text(tag, text, global_step=global_step)
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self.tb is not None:
                self.tb.flush()
                self.tb.close()
        except Exception:
            pass
        for fh in (self._fh_train, self._fh_val, self._fh_ev, self._fh_ckpt):
            try:
                if fh is not None:
                    fh.flush()
                    fh.close()
            except Exception:
                pass
