"""Train IceCubePulseTransformer with 3D vMF Loss and benchmark against Classical LineFit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
from src.dataset import IceCubePulseDataset, prepare_icecube_datasets
from src.losses import VonMisesFisher3DLoss, angular_error_radians, compute_angular_metrics
from src.model import IceCubePulseTransformer
from src.physics_baseline import linefit_predict_batch
from src.viz import plot_neutrino_reconstruction


@torch.no_grad()
def evaluate_transformer(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Pulse Transformer and collect predicted 3D unit vectors, kappas, and targets."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_preds: list[np.ndarray] = []
    all_kappas: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    for x_b, m_b, y_b in loader:
        x_b = x_b.to(device, non_blocking=True)
        m_b = m_b.to(device, non_blocking=True)
        y_b = y_b.to(device, non_blocking=True)
        with amp_context(device, enabled=use_amp):
            u_hat, kappa = model(x_b, m_b)
            loss = criterion(u_hat, kappa, y_b)
        bs = y_b.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
        all_preds.append(u_hat.float().cpu().numpy())
        all_kappas.append(kappa.float().cpu().numpy())
        all_targets.append(y_b.cpu().numpy())

    return (
        total_loss / max(total_n, 1),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_kappas, axis=0),
        np.concatenate(all_targets, axis=0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IceCube 3D Neutrino Pulse Transformer.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=1500)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--max-pulses", type=int, default=96)
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

    bundle = prepare_icecube_datasets(
        data_dir=data_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        max_pulses=args.max_pulses,
        seed=args.seed,
    )

    tr = bundle["train"]
    va = bundle["val"]
    te = bundle["test"]

    # Classical IceCube Charge-Weighted LineFit baseline on exact held-out test set
    linefit_preds = linefit_predict_batch(te["X"], te["masks"])
    linefit_metrics = compute_angular_metrics(linefit_preds, te["targets"])
    linefit_err_deg = np.degrees(angular_error_radians(linefit_preds, te["targets"]))

    train_loader = DataLoader(
        IceCubePulseDataset(tr["X"], tr["masks"], tr["targets"]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        IceCubePulseDataset(va["X"], va["masks"], va["targets"]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        IceCubePulseDataset(te["X"], te["masks"], te["targets"]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = IceCubePulseTransformer(
        pulse_dim=6,
        fourier_bands=4,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    ).to(device)
    param_info = count_parameters(model)

    criterion = VonMisesFisher3DLoss(max_kappa=100.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-5)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_mean_err_deg": []}
    best_val_err = float("inf")
    best_ckpt = artifacts_dir / "icecube_transformer_best.pt"

    with timed_block() as train_timer:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            for x_b, m_b, y_b in train_loader:
                x_b = x_b.to(device, non_blocking=True)
                m_b = m_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, enabled=use_amp):
                    u_hat, kappa = model(x_b, m_b)
                    loss = criterion(u_hat, kappa, y_b)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                bs = y_b.size(0)
                running_loss += float(loss.item()) * bs
                seen += bs

            scheduler.step()
            train_loss = running_loss / max(seen, 1)
            val_loss, val_preds, _, val_targets = evaluate_transformer(model, val_loader, criterion, device)
            val_m = compute_angular_metrics(val_preds, val_targets)

            history["train_loss"].append(round(train_loss, 5))
            history["val_loss"].append(round(val_loss, 5))
            history["val_mean_err_deg"].append(val_m["mean_angular_error_deg"])

            if val_m["mean_angular_error_deg"] <= best_val_err:
                best_val_err = val_m["mean_angular_error_deg"]
                torch.save(
                    {
                        "epoch": epoch,
                        "val_mean_err_deg": best_val_err,
                        "max_pulses": args.max_pulses,
                        "model_state_dict": model.state_dict(),
                    },
                    best_ckpt,
                )

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_preds, test_kappas, test_targets = evaluate_transformer(model, test_loader, criterion, device)
    transformer_metrics = compute_angular_metrics(test_preds, test_targets)
    transformer_metrics["test_vmf_loss"] = round(test_loss, 4)
    transformer_metrics["mean_predicted_kappa"] = round(float(np.mean(test_kappas)), 3)
    vram_stats = get_vram_stats()

    trans_err_deg = np.degrees(angular_error_radians(test_preds, test_targets))

    # Pick a clean illustrative event with rich pulses where Transformer clearly outperforms LineFit
    pulse_counts = te["masks"].sum(axis=1)
    candidates = np.where(pulse_counts >= 25)[0]
    if candidates.size > 0:
        gain = linefit_err_deg[candidates] - trans_err_deg[candidates]
        sample_idx = int(candidates[np.argmax(gain)])
    else:
        sample_idx = int(np.argmin(trans_err_deg))

    plot_path = reports_dir / "neutrino_reconstruction.png"
    plot_neutrino_reconstruction(
        out_path=plot_path,
        event_pulses=te["X"][sample_idx],
        event_mask=te["masks"][sample_idx],
        event_vertex=te["vertices"][sample_idx],
        true_vec=test_targets[sample_idx],
        pred_vec=test_preds[sample_idx],
        linefit_vec=linefit_preds[sample_idx],
        dom_xyz=bundle["dom_xyz"],
        transformer_errors_deg=trans_err_deg,
        linefit_errors_deg=linefit_err_deg,
        test_true_vecs=test_targets,
        test_pred_vecs=test_preds,
    )

    err_reduction_deg = round(linefit_metrics["mean_angular_error_deg"] - transformer_metrics["mean_angular_error_deg"], 3)
    err_reduction_rad = round(linefit_metrics["mean_angular_error_rad"] - transformer_metrics["mean_angular_error_rad"], 4)
    rel_improvement_pct = round(
        100.0 * err_reduction_deg / max(linefit_metrics["mean_angular_error_deg"], 1e-6), 2
    )

    report: dict[str, Any] = {
        "project": "10_icecube_neutrino_3d",
        "dataset_type": "physics_calibrated_icecube_5160dom",
        "dataset_provenance": bundle["meta"],
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_pulses": args.max_pulses,
            "fp16_amp": use_amp,
            "seed": args.seed,
        },
        "model": {
            "architecture": "IceCubePulseTransformer (4-layer Pre-Norm + Fourier 4D + vMF Head)",
            **param_info,
            "best_val_mean_angular_error_deg": round(best_val_err, 3),
            "history": history,
        },
        "test_metrics_transformer": transformer_metrics,
        "test_metrics_linefit_baseline": linefit_metrics,
        "transformer_vs_linefit_improvement": {
            "mean_angular_error_reduction_rad": err_reduction_rad,
            "mean_angular_error_reduction_deg": err_reduction_deg,
            "relative_error_reduction_pct": rel_improvement_pct,
        },
        "runtime": {
            "train_elapsed_s": train_timer["elapsed_s"],
            **vram_stats,
        },
        "artifacts": {
            "checkpoint": "artifacts/icecube_transformer_best.pt",
            "figure": "reports/neutrino_reconstruction.png",
        },
    }
    write_json_report(reports_dir / "metrics.json", report)
    cleanup_cuda()
    print(
        f"[10_icecube_neutrino_3d] Done in {train_timer['elapsed_s']}s | "
        f"Transformer MAE={transformer_metrics['mean_angular_error_deg']:.2f} deg "
        f"({transformer_metrics['mean_angular_error_rad']:.4f} rad) vs "
        f"LineFit={linefit_metrics['mean_angular_error_deg']:.2f} deg | "
        f"Peak VRAM={vram_stats['peak_vram_mb']} MB"
    )


if __name__ == "__main__":
    main()
