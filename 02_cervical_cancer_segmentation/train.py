"""Training, head-to-head ablation, and cervical biopsy morphometry for Project 02."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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
from src.data import (
    CERVICAL_BIOPSY_REPO_ID,
    DSB2018_REPO_ID,
    NucleiSegmentationDataset,
    download_and_load_dsb2018,
    stream_cervical_biopsy_samples,
)
from src.metrics import analyze_biopsy_morphometry, compute_segmentation_metrics
from src.model import AttentionUNet, DiceBCEBoundaryLoss, PlainUNet
from src.viz import plot_segmentation_overlay


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a segmentation DataLoader and return probabilities and targets."""
    model.eval()
    probs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        with amp_context(device, enabled=use_amp):
            logits = model(imgs)
            probs = torch.sigmoid(logits.float())
        probs_list.append(probs.cpu().numpy())
        targets_list.append(masks.numpy())
    return np.concatenate(probs_list, axis=0), np.concatenate(targets_list, axis=0)


def train_single_unet(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    use_amp: bool,
    ckpt_path: Path,
) -> dict[str, Any]:
    """Train a segmentation model and restore its best validation Dice checkpoint."""
    model = model.to(device)
    criterion = DiceBCEBoundaryLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_dice = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []

    with timed_block() as timer:
        for epoch in range(1, epochs + 1):
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

            val_probs, val_targets = predict_loader(model, val_loader, device, use_amp=use_amp)
            val_metrics = compute_segmentation_metrics(val_probs, val_targets)
            train_loss = running_loss / max(seen, 1)
            history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_dice": val_metrics["dice"]})
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(
                    f"[{name}] Epoch {epoch:02d}/{epochs} | loss={train_loss:.4f} | "
                    f"val_dice={val_metrics['dice']:.4f} | val_iou={val_metrics['iou']:.4f}"
                )
            if val_metrics["dice"] >= best_dice:
                best_dice = val_metrics["dice"]
                best_epoch = epoch
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    return {
        "best_epoch": best_epoch,
        "best_val_dice": round(best_dice, 4),
        "elapsed_s": timer["elapsed_s"],
        "history_tail": history[-5:],
        **count_parameters(model),
    }


@torch.no_grad()
def run_biopsy_morphometry(
    model: nn.Module,
    biopsy_samples: list[dict[str, Any]],
    device: torch.device,
    use_amp: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply trained Attention U-Net to real cervical cancer biopsy images for N/C analysis."""
    model.eval()
    per_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    visual_samples: list[dict[str, Any]] = []

    for item in biopsy_samples:
        rgb = item["rgb"]
        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with amp_context(device, enabled=use_amp):
            prob = torch.sigmoid(model(x).float())[0, 0].cpu().numpy()
        mask = (prob >= 0.5).astype(np.uint8)
        morph = analyze_biopsy_morphometry(rgb, mask)
        per_class[item["label"]].append(morph)
        if len(visual_samples) < 4:
            visual_samples.append(
                {
                    "rgb": rgb,
                    "mask": mask,
                    "label": item["label"],
                    "nc_ratio": morph["nc_ratio"],
                    "nuclei_count": morph["nuclei_count"],
                }
            )

    summary_by_class: dict[str, Any] = {}
    all_nc: list[float] = []
    all_cnt: list[int] = []
    for cls_name, rows in sorted(per_class.items()):
        nc_vals = [r["nc_ratio"] for r in rows]
        cnt_vals = [r["nuclei_count"] for r in rows]
        all_nc.extend(nc_vals)
        all_cnt.extend(cnt_vals)
        summary_by_class[cls_name] = {
            "count": len(rows),
            "mean_nc_ratio": round(float(np.mean(nc_vals)), 4),
            "std_nc_ratio": round(float(np.std(nc_vals)), 4),
            "mean_nuclei_count": round(float(np.mean(cnt_vals)), 2),
        }

    overall = {
        "total_biopsy_images": len(biopsy_samples),
        "overall_mean_nc_ratio": round(float(np.mean(all_nc)), 4) if all_nc else 0.0,
        "overall_mean_nuclei_count": round(float(np.mean(all_cnt)), 2) if all_cnt else 0.0,
        "by_class": summary_by_class,
    }
    return overall, visual_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plain vs Attention U-Net nuclei & cervical biopsy segmentation.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--biopsy-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    # Default FP32 on GTX 1650 Ti (Turing TU117) to avoid FP16 cuDNN NaN in BatchNorm2d(1) Attention Gates.
    use_amp = False

    data_dir = PROJECT_ROOT / "data"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    reports_dir = PROJECT_ROOT / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[02_cervical_cancer_segmentation] Device: {device} | AMP(FP16): {use_amp}")
    splits = download_and_load_dsb2018(data_dir, img_size=args.img_size, seed=args.seed)
    train_ds = NucleiSegmentationDataset(*splits["train"], is_train=True)
    val_ds = NucleiSegmentationDataset(*splits["val"], is_train=False)
    test_ds = NucleiSegmentationDataset(*splits["test"], is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    reset_peak_vram()

    # 1. Plain U-Net baseline
    seed_everything(args.seed)
    plain_ckpt = artifacts_dir / "plain_unet_nuclei.pt"
    plain_model = PlainUNet(in_channels=3, out_channels=1, base_channels=args.base_channels)
    plain_train_info = train_single_unet(
        "PlainUNet", plain_model, train_loader, val_loader, device, args.epochs, args.lr, use_amp, plain_ckpt
    )
    plain_probs, test_targets = predict_loader(plain_model, test_loader, device, use_amp=use_amp)
    plain_test_metrics = compute_segmentation_metrics(plain_probs, test_targets)

    # 2. Additive Attention U-Net
    seed_everything(args.seed)
    att_ckpt = artifacts_dir / "attention_unet_nuclei.pt"
    att_model = AttentionUNet(in_channels=3, out_channels=1, base_channels=args.base_channels)
    att_train_info = train_single_unet(
        "AttentionUNet", att_model, train_loader, val_loader, device, args.epochs, args.lr, use_amp, att_ckpt
    )
    att_probs, _ = predict_loader(att_model, test_loader, device, use_amp=use_amp)
    att_test_metrics = compute_segmentation_metrics(att_probs, test_targets)

    # 3. Downstream Cervical Cancer Biopsy Morphometry
    biopsy_samples = stream_cervical_biopsy_samples(n_samples=args.biopsy_samples, img_size=args.img_size)
    biopsy_stats, biopsy_vis = run_biopsy_morphometry(att_model, biopsy_samples, device, use_amp=use_amp)

    # 4. Build visual proof overlay (top 3 test images by nuclei content + 4 biopsy images)
    test_imgs, test_masks = splits["test"]
    fg_areas = test_masks.reshape(len(test_masks), -1).sum(axis=1)
    top_idx = np.argsort(-fg_areas)[:3]
    dsb_vis: list[dict[str, Any]] = []
    for idx in top_idx:
        p_m = (plain_probs[idx, 0] >= 0.5).astype(np.uint8)
        a_m = (att_probs[idx, 0] >= 0.5).astype(np.uint8)
        gt_m = (test_masks[idx] > 0).astype(np.uint8)
        dsb_vis.append(
            {
                "rgb": test_imgs[idx],
                "gt_mask": gt_m,
                "plain_pred": p_m,
                "att_pred": a_m,
                "plain_dice": compute_segmentation_metrics(plain_probs[idx : idx + 1], test_targets[idx : idx + 1])["dice"],
                "att_dice": compute_segmentation_metrics(att_probs[idx : idx + 1], test_targets[idx : idx + 1])["dice"],
            }
        )

    overlay_path = plot_segmentation_overlay(dsb_vis, biopsy_vis, reports_dir / "segmentation_overlay.png")
    vram = get_vram_stats()

    report = {
        "project": "02_cervical_cancer_segmentation",
        "dataset_type": "real_public",
        "datasets": {
            "primary_segmentation": f"https://huggingface.co/datasets/{DSB2018_REPO_ID}",
            "downstream_biopsy": f"https://huggingface.co/datasets/{CERVICAL_BIOPSY_REPO_ID}",
        },
        "splits": {
            "train": int(len(train_ds)),
            "val": int(len(val_ds)),
            "test": int(len(test_ds)),
            "img_size": args.img_size,
        },
        "ablation": {
            "plain_unet": {"training": plain_train_info, "test_metrics": plain_test_metrics},
            "attention_unet": {"training": att_train_info, "test_metrics": att_test_metrics},
            "delta_attention_minus_plain": {
                k: round(att_test_metrics[k] - plain_test_metrics[k], 4)
                for k in ("dice", "iou", "precision", "recall", "boundary_f1")
            },
        },
        "cervical_biopsy_morphometry": biopsy_stats,
        "hardware": vram,
        "artifacts": {
            "plain_unet_checkpoint": plain_ckpt.relative_to(PROJECT_ROOT).as_posix(),
            "attention_unet_checkpoint": att_ckpt.relative_to(PROJECT_ROOT).as_posix(),
            "segmentation_overlay": overlay_path.relative_to(PROJECT_ROOT).as_posix(),
        },
    }
    write_json_report(reports_dir / "metrics.json", report)
    print(
        f"[02_cervical_cancer_segmentation] Done! Plain Dice={plain_test_metrics['dice']:.4f} | "
        f"Attn Dice={att_test_metrics['dice']:.4f} | Biopsy N/C={biopsy_stats['overall_mean_nc_ratio']:.4f}"
    )
    cleanup_cuda()


if __name__ == "__main__":
    main()
