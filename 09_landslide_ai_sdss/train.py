"""Train SpatialResUNet on real Landslide4Sense tiles and generate AI-SDSS GeoJSON reports."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
for _p in (str(REPO_ROOT), str(PROJECT_DIR)):
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
from src.dataset import LandslideTileDataset, prepare_landslide_dataset
from src.model import FocalDiceLoss, SpatialResUNet
from src.sdss_engine import assess_susceptibility_tile
from src.viz import plot_landslide_sdss_map


@torch.no_grad()
def evaluate_segmentation(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate pixel-level and tile-level landslide segmentation metrics."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    # Keep FP32 on Turing GTX 1650 Ti to prevent cuDNN FP16 NaN in 14-band U-Net skips.
    use_amp = False

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with amp_context(device, enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, masks)
        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
        probs = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(masks.squeeze(1).cpu().numpy().astype(np.uint8))

    avg_loss = total_loss / max(total_n, 1)
    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    y_pred = (y_prob >= threshold).astype(np.uint8)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * tp) / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)

    y_flat = y_true.reshape(-1)
    p_flat = y_prob.reshape(-1)
    has_both = len(np.unique(y_flat)) == 2
    roc_auc = float(roc_auc_score(y_flat, p_flat)) if has_both else float("nan")
    pr_auc = float(average_precision_score(y_flat, p_flat)) if has_both else float("nan")

    # Positive tile detection accuracy (tile has landslide pixels vs predicted high-risk pixels)
    true_pos_tile = y_true.reshape(len(y_true), -1).sum(axis=1) > 0
    pred_pos_tile = y_pred.reshape(len(y_pred), -1).sum(axis=1) > 0
    tile_det_acc = float(np.mean(true_pos_tile == pred_pos_tile))

    metrics = {
        "loss": round(avg_loss, 4),
        "pixel_roc_auc": round(roc_auc, 4) if np.isfinite(roc_auc) else None,
        "pixel_pr_auc": round(pr_auc, 4) if np.isfinite(pr_auc) else None,
        "pixel_f1": round(float(f1), 4),
        "pixel_iou": round(float(iou), 4),
        "landslide_recall": round(float(recall), 4),
        "landslide_precision": round(float(precision), 4),
        "positive_tile_detection_accuracy": round(tile_det_acc, 4),
    }
    return avg_loss, y_true, y_prob, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SpatialResUNet on Landslide4Sense.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-train", type=int, default=320)
    parser.add_argument("--n-val", type=int, default=80)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(prefer_cuda=not args.no_cuda)
    reset_peak_vram()

    data_dir = PROJECT_DIR / "data"
    artifacts_dir = PROJECT_DIR / "artifacts"
    reports_dir = PROJECT_DIR / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    bundle = prepare_landslide_dataset(
        data_dir=data_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        tile_size=args.tile_size,
        seed=args.seed,
    )
    in_channels = int(bundle["X_train"].shape[1])

    train_ds = LandslideTileDataset(
        bundle["X_train"], bundle["y_train"], bundle["channel_mean"], bundle["channel_std"], augment=True
    )
    val_ds = LandslideTileDataset(
        bundle["X_val"], bundle["y_val"], bundle["channel_mean"], bundle["channel_std"], augment=False
    )
    test_ds = LandslideTileDataset(
        bundle["X_test"], bundle["y_test"], bundle["channel_mean"], bundle["channel_std"], augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Adaptive positive pixel weight clamped to a stable range
    pos_rate = max(float(bundle["y_train"].mean()), 1e-3)
    pos_weight = float(np.clip((1.0 - pos_rate) / pos_rate, 2.0, 12.0))

    model = SpatialResUNet(in_channels=in_channels, base_channels=32, dropout=0.1).to(device)
    param_info = count_parameters(model)
    criterion = FocalDiceLoss(pos_weight=pos_weight, gamma=2.0, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-5)

    # Use FP32 on Turing GTX 1650 Ti to avoid cuDNN FP16 NaNs on 14-band U-Net skip connections.
    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": []}
    best_val_iou = -1.0
    best_ckpt = artifacts_dir / "resunet_landslide_best.pt"

    with timed_block() as train_timer:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            for imgs, masks in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
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

                bs = imgs.size(0)
                running_loss += float(loss.item()) * bs
                seen += bs

            scheduler.step()
            train_loss = running_loss / max(seen, 1)
            val_loss, _, _, val_m = evaluate_segmentation(model, val_loader, criterion, device)
            history["train_loss"].append(round(train_loss, 5))
            history["val_loss"].append(round(val_loss, 5))
            history["val_iou"].append(val_m["pixel_iou"])
            history["val_f1"].append(val_m["pixel_f1"])

            if val_m["pixel_iou"] >= best_val_iou:
                best_val_iou = val_m["pixel_iou"]
                torch.save(
                    {
                        "epoch": epoch,
                        "in_channels": in_channels,
                        "val_iou": best_val_iou,
                        "channel_mean": bundle["channel_mean"].tolist(),
                        "channel_std": bundle["channel_std"].tolist(),
                        "model_state_dict": model.state_dict(),
                    },
                    best_ckpt,
                )

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    _, y_test_true, y_test_prob, test_metrics = evaluate_segmentation(model, test_loader, criterion, device)
    vram_stats = get_vram_stats()

    # Pick most illustrative positive landslide test tile (strong overlap & clear ground truth)
    tile_pos_counts = y_test_true.reshape(len(y_test_true), -1).sum(axis=1)
    pos_idxs = np.where(tile_pos_counts > 25)[0]
    if pos_idxs.size == 0:
        pos_idxs = np.where(tile_pos_counts > 0)[0]
    if pos_idxs.size > 0:
        best_idx = int(pos_idxs[0])
        best_score = -1.0
        for idx in pos_idxs:
            pred_b = (y_test_prob[idx] >= 0.5).astype(np.uint8)
            gt_b = y_test_true[idx]
            inter = float(np.sum((pred_b == 1) & (gt_b == 1)))
            union = float(np.sum((pred_b == 1) | (gt_b == 1))) + 1e-6
            sc = inter / union
            if sc > best_score:
                best_score = sc
                best_idx = int(idx)
    else:
        best_idx = int(np.argmax(y_test_prob.mean(axis=(1, 2))))

    sdss_assessment = assess_susceptibility_tile(y_test_prob[best_idx])
    plot_path = reports_dir / "landslide_sdss_map.png"
    plot_landslide_sdss_map(
        out_path=plot_path,
        raw_tile=bundle["X_test"][best_idx],
        gt_mask=y_test_true[best_idx],
        prob_map=y_test_prob[best_idx],
        history=history,
        all_y_true=y_test_true,
        all_y_prob=y_test_prob,
        tile_summary=sdss_assessment,
    )

    report: dict[str, Any] = {
        "project": "09_landslide_ai_sdss",
        "dataset_type": "real_public",
        "dataset_provenance": bundle["meta"],
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "tile_size": args.tile_size,
            "pos_weight": round(pos_weight, 3),
            "fp16_amp": use_amp,
            "seed": args.seed,
        },
        "model": {
            "architecture": "SpatialResUNet",
            "in_channels": in_channels,
            **param_info,
            "best_val_iou": round(best_val_iou, 4),
            "history": history,
        },
        "test_metrics": test_metrics,
        "sdss_sample_tile_assessment": {
            "tile_index": best_idx,
            "overall_risk_tier": sdss_assessment["overall_risk_tier"],
            "advisory": sdss_assessment["advisory"],
            "max_probability": sdss_assessment["max_probability"],
            "tier_coverage_pct": sdss_assessment["tier_coverage_pct"],
            "high_or_critical_area_ha": sdss_assessment["high_or_critical_area_ha"],
            "hazard_zone_count": sdss_assessment["hazard_zone_count"],
            "geojson_preview_features": sdss_assessment["geojson"]["features"][:3],
        },
        "runtime": {
            "train_elapsed_s": train_timer["elapsed_s"],
            **vram_stats,
        },
        "artifacts": {
            "checkpoint": "artifacts/resunet_landslide_best.pt",
            "figure": "reports/landslide_sdss_map.png",
        },
    }
    write_json_report(reports_dir / "metrics.json", report)
    cleanup_cuda()
    print(
        f"[09_landslide_ai_sdss] Done in {train_timer['elapsed_s']}s | "
        f"Pixel ROC-AUC={test_metrics['pixel_roc_auc']} | F1={test_metrics['pixel_f1']:.4f} | "
        f"IoU={test_metrics['pixel_iou']:.4f} | Peak VRAM={vram_stats['peak_vram_mb']} MB"
    )


if __name__ == "__main__":
    main()
