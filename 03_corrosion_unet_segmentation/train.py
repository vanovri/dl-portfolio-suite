"""Training and evaluation entry point for Project 03: Corrosion U-Net Segmentation."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
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
from src.data import RUST_REPO_ID, CorrosionDataset, download_and_load_rust_coco
from src.model import (
    FocalDiceLoss,
    ResNet18UNet,
    compute_corrosion_area_pct,
    grade_corrosion_severity,
)
from src.viz import plot_corrosion_predictions


@torch.no_grad()
def evaluate_corrosion(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    threshold: float = 0.5,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate corrosion segmentation quality, area MAE, severity accuracy, and FPS."""
    model.eval()
    probs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    total_imgs = 0

    t0 = time.perf_counter()
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        with amp_context(device, enabled=use_amp):
            logits = model(imgs)
            probs = torch.sigmoid(logits.float())
        probs_list.append(probs.cpu().numpy())
        targets_list.append(masks.numpy())
        total_imgs += len(imgs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_s = max(time.perf_counter() - t0, 1e-6)

    y_prob = np.concatenate(probs_list, axis=0)
    y_true = np.concatenate(targets_list, axis=0)
    preds = (y_prob >= threshold).astype(np.uint8)
    gts = (y_true > 0.5).astype(np.uint8)

    tp = float(np.logical_and(preds == 1, gts == 1).sum())
    fp = float(np.logical_and(preds == 1, gts == 0).sum())
    fn = float(np.logical_and(preds == 0, gts == 1).sum())
    eps = 1e-7

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    area_errors: list[float] = []
    grade_hits = 0
    for i in range(len(preds)):
        gt_pct = compute_corrosion_area_pct(gts[i, 0])
        pr_pct = compute_corrosion_area_pct(preds[i, 0])
        area_errors.append(abs(pr_pct - gt_pct))
        if grade_corrosion_severity(pr_pct) == grade_corrosion_severity(gt_pct):
            grade_hits += 1

    metrics = {
        "dice": round(float(dice), 4),
        "iou": round(float(iou), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "corrosion_area_mae_pct": round(float(np.mean(area_errors)) if area_errors else 0.0, 3),
        "severity_grade_accuracy": round(float(grade_hits / max(len(preds), 1)), 4),
        "inference_fps": round(float(total_imgs / elapsed_s), 2),
    }
    return metrics, y_prob, y_true


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-18 U-Net on Rust COCO Segmentation.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    # Default FP32 on GTX 1650 Ti (Turing TU117) to avoid FP16 cuDNN NaN in U-Net decoder ConvTranspose/BN.
    use_amp = False

    data_dir = PROJECT_ROOT / "data"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    reports_dir = PROJECT_ROOT / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[03_corrosion_unet_segmentation] Device: {device} | AMP(FP16): {use_amp}")
    splits = download_and_load_rust_coco(data_dir, img_size=args.img_size)
    tr_imgs, tr_masks = splits["train"]
    va_imgs, va_masks = splits["valid"]
    te_imgs, te_masks = splits["test"]

    eval_split_name = "test"
    if len(te_imgs) == 0 or int(te_masks.sum()) == 0:
        eval_split_name = "valid"
        eval_imgs, eval_masks = va_imgs, va_masks
    else:
        eval_imgs, eval_masks = te_imgs, te_masks

    train_ds = CorrosionDataset(tr_imgs, tr_masks, is_train=True)
    val_ds = CorrosionDataset(va_imgs, va_masks, is_train=False)
    eval_ds = CorrosionDataset(eval_imgs, eval_masks, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ResNet18UNet(out_channels=1, pretrained=not args.no_pretrained).to(device)
    criterion = FocalDiceLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    reset_peak_vram()
    ckpt_path = artifacts_dir / "resnet18_unet_corrosion.pt"
    best_dice = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []

    with timed_block() as timer:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            for imgs, masks in train_loader:
                imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
                masks = masks.to(device, non_blocking=(device.type == "cuda"))
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, enabled=use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.item()) * len(imgs)
                seen += len(imgs)
            scheduler.step()

            val_metrics, _, _ = evaluate_corrosion(model, val_loader, device, use_amp=use_amp)
            train_loss = running_loss / max(seen, 1)
            history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_dice": val_metrics["dice"]})
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                print(
                    f"Epoch {epoch:02d}/{args.epochs} | loss={train_loss:.4f} | "
                    f"val_dice={val_metrics['dice']:.4f} | val_iou={val_metrics['iou']:.4f}"
                )
            if val_metrics["dice"] >= best_dice:
                best_dice = val_metrics["dice"]
                best_epoch = epoch
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    eval_metrics, eval_probs, _ = evaluate_corrosion(model, eval_loader, device, use_amp=use_amp)

    # Choose 4 diverse samples across corrosion area spectrum
    gt_areas = np.array([compute_corrosion_area_pct(m) for m in eval_masks])
    order = np.argsort(gt_areas)
    if len(order) >= 4:
        pick = [int(order[0]), int(order[len(order) // 3]), int(order[(2 * len(order)) // 3]), int(order[-1])]
    else:
        pick = [int(i) for i in order]

    vis_samples: list[dict[str, Any]] = []
    for idx in pick:
        gt_m = (eval_masks[idx] > 0).astype(np.uint8)
        pr_m = (eval_probs[idx, 0] >= 0.5).astype(np.uint8)
        gt_pct = compute_corrosion_area_pct(gt_m)
        pr_pct = compute_corrosion_area_pct(pr_m)
        vis_samples.append(
            {
                "rgb": eval_imgs[idx],
                "gt_mask": gt_m,
                "pred_mask": pr_m,
                "gt_area_pct": gt_pct,
                "pred_area_pct": pr_pct,
                "gt_grade": grade_corrosion_severity(gt_pct),
                "pred_grade": grade_corrosion_severity(pr_pct),
            }
        )

    pred_fig = plot_corrosion_predictions(vis_samples, reports_dir / "corrosion_predictions.png")
    vram = get_vram_stats()

    report = {
        "project": "03_corrosion_unet_segmentation",
        "dataset_type": "real_public",
        "dataset_name": "Dodon Corrosion LabelMe Polygon Segmentation",
        "dataset_provenance": f"https://huggingface.co/datasets/{RUST_REPO_ID}",
        "eval_split_used": eval_split_name,
        "splits": {
            "train": int(len(tr_imgs)),
            "valid": int(len(va_imgs)),
            "test": int(len(te_imgs)),
            "img_size": args.img_size,
        },
        "model": {
            "architecture": "ResNet18UNet",
            "pretrained": not args.no_pretrained,
            **count_parameters(model),
        },
        "training": {
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_val_dice": round(best_dice, 4),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "amp_fp16": use_amp,
            "elapsed_s": timer["elapsed_s"],
            "history_tail": history[-5:],
        },
        "test_metrics": eval_metrics,
        "hardware": vram,
        "artifacts": {
            "checkpoint": ckpt_path.relative_to(PROJECT_ROOT).as_posix(),
            "corrosion_predictions": pred_fig.relative_to(PROJECT_ROOT).as_posix(),
        },
    }
    write_json_report(reports_dir / "metrics.json", report)
    print(f"[03_corrosion_unet_segmentation] Done! Eval ({eval_split_name}): {eval_metrics}")
    cleanup_cuda()


if __name__ == "__main__":
    main()
