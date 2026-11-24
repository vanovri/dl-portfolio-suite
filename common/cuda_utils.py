"""Shared CUDA, reproducibility, telemetry, and plotting helpers for 4 GB VRAM GTX 1650 Ti."""
from __future__ import annotations

import gc
import json
import os
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA device if available and requested, else CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def get_vram_stats() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"device": "cpu", "peak_vram_mb": 0.0, "allocated_vram_mb": 0.0}
    torch.cuda.synchronize()
    return {
        "device": torch.cuda.get_device_name(0),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "allocated_vram_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
    }


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def amp_context(device: torch.device, enabled: bool = True):
    """Turing SM 7.5 safe autocast: uses FP16 only (never BF16)."""
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@contextmanager
def timed_block():
    """Simple wall-clock timer context manager."""
    info = {"elapsed_s": 0.0}
    t0 = time.perf_counter()
    try:
        yield info
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        info["elapsed_s"] = round(time.perf_counter() - t0, 3)


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def write_json_report(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a clean JSON report with UTF-8 encoding and no em-dashes."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    p.write_text(text + "\n", encoding="utf-8")
    return p


def configure_plot_style() -> None:
    """Apply clean publication-grade matplotlib styling."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#0f172a",
            "axes.facecolor": "#1e293b",
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#e2e8f0",
            "text.color": "#f8fafc",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "grid.color": "#334155",
            "grid.alpha": 0.5,
            "legend.facecolor": "#0f172a",
            "legend.edgecolor": "#475569",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.facecolor": "#0f172a",
        }
    )
