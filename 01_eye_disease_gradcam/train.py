"""Training and evaluation entry point for Project 01: Eye Disease Grading + Grad-CAM."""
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
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
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
from src.data import (
    RETINAMNIST_CLASSES,
    RETINAMNIST_URL,
    RetinaMNISTDataset,
    compute_class_weights,
    denormalize_image,
    load_retinamnist_splits,
)
from src.gradcam import GradCAM, overlay_gradcam
from src.model import build_eye_resnet18, get_gradcam_target_layer
from src.viz import plot_confusion_and_roc, plot_gradcam_gallery


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Run deterministic evaluation and return clinical DR grading metrics."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    total_images = 0

    t0 = time.perf_counter()
    for images, targets in loader:
        images = images.to(device, non_blocking=(device.type == "cuda"))
        with amp_context(device, enabled=use_amp):
            logits = model(images)
            probs = F.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())
        total_images += len(targets)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        roc_auc = 0.0

    metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "ovr_macro_roc_auc": round(roc_auc, 4),
        "inference_ms_per_image": round(elapsed_ms / max(total_images, 1), 3),
    }
    return metrics, y_true, y_pred, y_prob


def select_diverse_gradcam_indices(y_true: np.ndarray, y_pred: np.ndarray, count: int = 4) -> list[int]:
    """Select diverse test indices across DR grades for the Grad-CAM gallery."""
    chosen: list[int] = []
    for cls in range(len(RETINAMNIST_CLASSES)):
        matches = np.where((y_true == cls) & (y_pred == cls))[0]
        if len(matches) == 0:
            matches = np.where(y_true == cls)[0]
        if len(matches) > 0:
            chosen.append(int(matches[0]))
        if len(chosen) >= count:
            break
    for idx in range(len(y_true)):
        if len(chosen) >= count:
            break
        if idx not in chosen:
            chosen.append(idx)
    return chosen[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-18 on RetinaMNIST with Grad-CAM.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    use_amp = (not args.no_amp) and device.type == "cuda"

    data_dir = PROJECT_ROOT / "data"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    reports_dir = PROJECT_ROOT / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[01_eye_disease_gradcam] Device: {device} | AMP(FP16): {use_amp}")
    splits = load_retinamnist_splits(data_dir)
    train_imgs, train_lbls = splits["train"]
    val_imgs, val_lbls = splits["val"]
    test_imgs, test_lbls = splits["test"]

    train_ds = RetinaMNISTDataset(train_imgs, train_lbls, img_size=args.img_size, is_train=True)
    val_ds = RetinaMNISTDataset(val_imgs, val_lbls, img_size=args.img_size, is_train=False)
    test_ds = RetinaMNISTDataset(test_imgs, test_lbls, img_size=args.img_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = len(RETINAMNIST_CLASSES)
    model = build_eye_resnet18(num_classes=num_classes, pretrained=not args.no_pretrained, dropout=0.3).to(device)
    class_weights = compute_class_weights(train_lbls, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    reset_peak_vram()
    best_score = -1.0
    best_epoch = 0
    ckpt_path = artifacts_dir / "resnet18_retinamnist.pt"
    history: list[dict[str, Any]] = []

    with timed_block() as train_timer:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            for images, targets in train_loader:
                images = images.to(device, non_blocking=(device.type == "cuda"))
                targets = targets.to(device, non_blocking=(device.type == "cuda"))
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.item()) * len(targets)
                seen += len(targets)
            scheduler.step()

            val_metrics, _, _, _ = evaluate_model(model, val_loader, device, use_amp=use_amp)
            train_loss = running_loss / max(seen, 1)
            selection_score = 0.5 * val_metrics["quadratic_weighted_kappa"] + 0.5 * val_metrics["macro_f1"]
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_qwk": val_metrics["quadratic_weighted_kappa"],
                }
            )
            print(
                f"Epoch {epoch:02d}/{args.epochs} | loss={train_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['macro_f1']:.4f} | "
                f"val_qwk={val_metrics['quadratic_weighted_kappa']:.4f}"
            )
            if selection_score >= best_score:
                best_score = selection_score
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_names": RETINAMNIST_CLASSES,
                        "img_size": args.img_size,
                        "epoch": epoch,
                    },
                    ckpt_path,
                )

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    test_metrics, y_true, y_pred, y_prob = evaluate_model(model, test_loader, device, use_amp=use_amp)

    # Generate Grad-CAM visual proof gallery on real held-out test fundus images
    gallery_indices = select_diverse_gradcam_indices(y_true, y_pred, count=4)
    gallery_samples: list[dict[str, Any]] = []
    with GradCAM(model, get_gradcam_target_layer(model)) as cam:
        for idx in gallery_indices:
            img_tensor, true_idx = test_ds[idx]
            heatmap, pred_idx, probs = cam.generate(img_tensor.to(device))
            rgb = denormalize_image(img_tensor)
            overlay = overlay_gradcam(rgb, heatmap, alpha=0.45)
            gallery_samples.append(
                {
                    "rgb": rgb,
                    "overlay": overlay,
                    "true_label": RETINAMNIST_CLASSES[true_idx],
                    "pred_label": RETINAMNIST_CLASSES[pred_idx],
                    "confidence": float(probs[pred_idx]),
                }
            )

    gallery_path = plot_gradcam_gallery(gallery_samples, reports_dir / "gradcam_gallery.png")
    conf_roc_path = plot_confusion_and_roc(y_true, y_pred, y_prob, RETINAMNIST_CLASSES, reports_dir / "confusion_roc.png")

    vram = get_vram_stats()
    params = count_parameters(model)
    report = {
        "project": "01_eye_disease_gradcam",
        "dataset_type": "real_public",
        "dataset_name": "MedMNIST v2 RetinaMNIST (128x128 RGB)",
        "dataset_provenance": RETINAMNIST_URL,
        "compatible_imagefolder_reference": "dlzcods/eye-disease-classification (cataract, diabetic_retinopathy, glaucoma, normal)",
        "splits": {
            "train": int(len(train_ds)),
            "val": int(len(val_ds)),
            "test": int(len(test_ds)),
        },
        "model": {
            "architecture": "ResNet-18 (dropout=0.3)",
            "pretrained": not args.no_pretrained,
            "img_size": args.img_size,
            **params,
        },
        "training": {
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "amp_fp16": use_amp,
            "elapsed_s": train_timer["elapsed_s"],
            "history_tail": history[-5:],
        },
        "test_metrics": test_metrics,
        "hardware": vram,
        "artifacts": {
            "checkpoint": ckpt_path.relative_to(PROJECT_ROOT).as_posix(),
            "gradcam_gallery": gallery_path.relative_to(PROJECT_ROOT).as_posix(),
            "confusion_roc": conf_roc_path.relative_to(PROJECT_ROOT).as_posix(),
        },
    }
    write_json_report(reports_dir / "metrics.json", report)
    print(f"[01_eye_disease_gradcam] Done! Test metrics: {test_metrics} | Peak VRAM: {vram['peak_vram_mb']} MB")
    cleanup_cuda()


if __name__ == "__main__":
    main()
