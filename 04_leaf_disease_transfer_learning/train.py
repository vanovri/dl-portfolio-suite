"""Training & Pareto benchmark for Project 04: PlantVillage Leaf Disease Transfer Learning."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
for _p in (str(REPO_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.cuda_utils import (
    amp_context,
    cleanup_cuda,
    count_parameters,
    get_device,
    get_vram_stats,
    reset_peak_vram,
    seed_everything,
    timed_block,
    write_json_report,
)
from src.data import PLANTVILLAGE_REPO_ID, CachedLeafDataset, prepare_plantvillage_cache
from src.model import build_leaf_model, checkpoint_size_mb
from src.viz import plot_model_comparison


def top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 3) -> float:
    k_eff = min(k, y_prob.shape[1])
    top_k = np.argsort(-y_prob, axis=1)[:, :k_eff]
    hits = [int(y_true[i] in top_k[i]) for i in range(len(y_true))]
    return float(np.mean(hits)) if hits else 0.0


@torch.no_grad()
def evaluate_leaf_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate classification accuracy, Macro-F1, Top-3 accuracy, and inference latency."""
    model.eval()
    probs_list: list[np.ndarray] = []
    preds_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    total = 0

    t0 = time.perf_counter()
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        with amp_context(device, enabled=use_amp):
            logits = model(imgs)
            probs = F.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)
        probs_list.append(probs.cpu().numpy())
        preds_list.append(preds.cpu().numpy())
        targets_list.append(targets.numpy())
        total += len(targets)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    y_prob = np.concatenate(probs_list, axis=0)
    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(targets_list, axis=0)

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "top3_accuracy": round(top_k_accuracy(y_true, y_prob, k=3), 4),
        "inference_ms_per_image": round(elapsed_ms / max(total, 1), 3),
    }
    return metrics, y_true, y_pred, y_prob


def train_and_benchmark_arch(
    arch: str,
    num_classes: int,
    class_names: list[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    use_amp: bool,
    pretrained: bool,
    ckpt_path: Path,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Train one transfer-learning backbone and measure accuracy + efficiency telemetry."""
    seed_everything(seed)
    cleanup_cuda()
    reset_peak_vram()

    model = build_leaf_model(arch, num_classes=num_classes, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_acc = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []

    with timed_block() as timer:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            for imgs, targets in train_loader:
                imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
                targets = targets.to(device, non_blocking=(device.type == "cuda"))
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, enabled=use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.item()) * len(targets)
                seen += len(targets)
            scheduler.step()

            val_metrics, _, _, _ = evaluate_leaf_model(model, val_loader, device, use_amp=use_amp)
            train_loss = running_loss / max(seen, 1)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                }
            )
            print(
                f"[{arch}] Epoch {epoch:02d}/{epochs} | loss={train_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['macro_f1']:.4f}"
            )
            if val_metrics["accuracy"] >= best_acc:
                best_acc = val_metrics["accuracy"]
                best_epoch = epoch
                torch.save(
                    {
                        "arch": arch,
                        "model_state_dict": model.state_dict(),
                        "class_names": class_names,
                        "epoch": epoch,
                    },
                    ckpt_path,
                )

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    test_metrics, y_true, y_pred, y_prob = evaluate_leaf_model(model, test_loader, device, use_amp=use_amp)
    vram = get_vram_stats()
    params = count_parameters(model)

    summary = {
        "architecture": arch,
        "pretrained": pretrained,
        **params,
        "weight_size_mb": checkpoint_size_mb(ckpt_path),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_acc, 4),
        "elapsed_s": timer["elapsed_s"],
        "peak_vram_mb": vram["peak_vram_mb"],
        "test_metrics": test_metrics,
        "checkpoint": ckpt_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    del model
    cleanup_cuda()
    return summary, history, y_true, y_pred, y_prob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EfficientNet-B0 vs MobileNetV3-Small on PlantVillage.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--train-per-class", type=int, default=160)
    parser.add_argument("--val-per-class", type=int, default=40)
    parser.add_argument("--test-per-class", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    use_amp = (not args.no_amp) and device.type == "cuda"

    cache_dir = PROJECT_ROOT / "data" / "plantvillage_cache"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    reports_dir = PROJECT_ROOT / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[04_leaf_disease_transfer_learning] Device: {device} | AMP(FP16): {use_amp}")
    manifest = prepare_plantvillage_cache(
        cache_dir=cache_dir,
        num_classes=args.num_classes,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        img_size=args.img_size,
        seed=args.seed,
    )
    class_names = manifest["class_names"]
    train_ds = CachedLeafDataset(cache_dir, manifest["splits"]["train"], img_size=args.img_size, is_train=True)
    val_ds = CachedLeafDataset(cache_dir, manifest["splits"]["val"], img_size=args.img_size, is_train=False)
    test_ds = CachedLeafDataset(cache_dir, manifest["splits"]["test"], img_size=args.img_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    eff_summary, eff_hist, y_true, eff_pred, eff_prob = train_and_benchmark_arch(
        "efficientnet_b0",
        len(class_names),
        class_names,
        train_loader,
        val_loader,
        test_loader,
        device,
        args.epochs,
        args.lr,
        use_amp,
        not args.no_pretrained,
        artifacts_dir / "efficientnet_b0_leaf.pt",
        args.seed,
    )
    mob_summary, mob_hist, _, mob_pred, mob_prob = train_and_benchmark_arch(
        "mobilenet_v3_small",
        len(class_names),
        class_names,
        train_loader,
        val_loader,
        test_loader,
        device,
        args.epochs,
        args.lr,
        use_amp,
        not args.no_pretrained,
        artifacts_dir / "mobilenet_v3_small_leaf.pt",
        args.seed,
    )

    # Pick 4 distinct classes from held-out test set for visual verification
    sample_preds: list[dict[str, Any]] = []
    seen_cls: set[int] = set()
    for idx in range(len(test_ds)):
        c = int(y_true[idx])
        if c not in seen_cls:
            seen_cls.add(c)
            sample_preds.append(
                {
                    "rgb": test_ds.load_raw_rgb(idx),
                    "true_label": class_names[c],
                    "eff_pred": class_names[int(eff_pred[idx])],
                    "eff_conf": float(eff_prob[idx, int(eff_pred[idx])]),
                    "mob_pred": class_names[int(mob_pred[idx])],
                    "mob_conf": float(mob_prob[idx, int(mob_pred[idx])]),
                }
            )
        if len(sample_preds) >= 4:
            break

    fig_path = plot_model_comparison(
        eff_hist, mob_hist, eff_summary, mob_summary, sample_preds, reports_dir / "model_comparison.png"
    )

    report = {
        "project": "04_leaf_disease_transfer_learning",
        "dataset_type": "real_public",
        "dataset_name": "PlantVillage Stratified 10-Class Subset (128x128 RGB)",
        "dataset_provenance": f"https://huggingface.co/datasets/{PLANTVILLAGE_REPO_ID}",
        "class_names": class_names,
        "splits": {
            "train": int(len(train_ds)),
            "val": int(len(val_ds)),
            "test": int(len(test_ds)),
            "img_size": args.img_size,
        },
        "models": {
            "efficientnet_b0": eff_summary,
            "mobilenet_v3_small": mob_summary,
        },
        "comparison": {
            "accuracy_delta_eff_minus_mob": round(
                eff_summary["test_metrics"]["accuracy"] - mob_summary["test_metrics"]["accuracy"], 4
            ),
            "speedup_mob_vs_eff": round(
                eff_summary["test_metrics"]["inference_ms_per_image"]
                / max(mob_summary["test_metrics"]["inference_ms_per_image"], 1e-6),
                2,
            ),
            "size_ratio_eff_vs_mob": round(
                eff_summary["weight_size_mb"] / max(mob_summary["weight_size_mb"], 0.01), 2
            ),
        },
        "hardware": {
            "device": get_vram_stats()["device"],
            "max_peak_vram_mb": max(eff_summary["peak_vram_mb"], mob_summary["peak_vram_mb"]),
        },
        "artifacts": {
            "model_comparison_plot": fig_path.relative_to(PROJECT_ROOT).as_posix(),
            "efficientnet_b0_checkpoint": eff_summary["checkpoint"],
            "mobilenet_v3_small_checkpoint": mob_summary["checkpoint"],
        },
    }
    write_json_report(reports_dir / "metrics.json", report)
    print(
        f"[04_leaf_disease_transfer_learning] Done! "
        f"EffNet-B0 Acc={eff_summary['test_metrics']['accuracy']:.4f} | "
        f"MobileNetV3 Acc={mob_summary['test_metrics']['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
