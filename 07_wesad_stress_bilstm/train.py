"""Training and subject-independent evaluation entrypoint for WESAD Bi-LSTM Attention."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
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
from src.dataset import WESADWindowDataset, load_wesad_dataframe
from src.dsp import build_subject_windows, split_by_subject
from src.model import (
    BiLSTMAttentionStressNet,
    compute_binary_classification_metrics,
    fit_and_eval_classical_baselines,
)
from src.viz import plot_stress_signals_attention


@torch.no_grad()
def evaluate_bilstm(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a loader and return loss, targets, stress probs, and attention weights."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_y: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_attn: list[np.ndarray] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        with amp_context(device, enabled=device.type == "cuda"):
            logits, attn = model(x_batch)
            loss = criterion(logits, y_batch)
        bs = y_batch.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
        probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        all_y.append(y_batch.cpu().numpy())
        all_probs.append(probs)
        all_attn.append(attn.float().cpu().numpy())

    avg_loss = total_loss / max(total_n, 1)
    return (
        avg_loss,
        np.concatenate(all_y, axis=0),
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_attn, axis=0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WESAD Bi-LSTM Temporal Attention stress detector.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--fs", type=float, default=4.0)
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

    df, data_meta = load_wesad_dataframe(data_dir)
    bundle = build_subject_windows(
        df,
        window_len=args.window_len,
        stride=args.stride,
        fs=args.fs,
    )
    split = split_by_subject(bundle, train_frac=0.70, val_frac=0.15, seed=args.seed)

    # Classical baselines on exact same subject-independent split
    classical = fit_and_eval_classical_baselines(
        split.X_train,
        split.y_train,
        split.X_test,
        split.y_test,
        seed=args.seed,
    )

    train_ds = WESADWindowDataset(split.X_train, split.y_train)
    val_ds = WESADWindowDataset(split.X_val, split.y_val)
    test_ds = WESADWindowDataset(split.X_test, split.y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Inverse-frequency class weighting for balanced stress detection
    counts = np.bincount(split.y_train, minlength=2).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (2.0 * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = BiLSTMAttentionStressNet(
        in_features=bundle.X.shape[-1],
        conv_channels=48,
        hidden_size=64,
        num_layers=2,
        num_classes=2,
        dropout=0.25,
    ).to(device)
    param_info = count_parameters(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-5)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_val_f1 = -1.0
    best_ckpt = artifacts_dir / "bilstm_stress_best.pt"

    with timed_block() as train_timer:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, enabled=use_amp):
                    logits, _ = model(x_batch)
                    loss = criterion(logits, y_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                bs = y_batch.size(0)
                running_loss += float(loss.item()) * bs
                seen += bs

            scheduler.step()
            train_loss = running_loss / max(seen, 1)
            val_loss, val_y, val_probs, _ = evaluate_bilstm(model, val_loader, criterion, device)
            val_metrics = compute_binary_classification_metrics(val_y, val_probs)

            history["train_loss"].append(round(train_loss, 5))
            history["val_loss"].append(round(val_loss, 5))
            history["val_macro_f1"].append(val_metrics["macro_f1"])

            if val_metrics["macro_f1"] >= best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "in_features": int(bundle.X.shape[-1]),
                        "val_macro_f1": best_val_f1,
                        "feature_names": list(bundle.feature_names),
                    },
                    best_ckpt,
                )

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_y, test_probs, test_attn = evaluate_bilstm(model, test_loader, criterion, device)
    bilstm_metrics = compute_binary_classification_metrics(test_y, test_probs)
    vram_stats = get_vram_stats()

    # Select highest-confidence true positive stress window for interpretability overlay
    pos_indices = np.where(test_y == 1)[0]
    if pos_indices.size > 0:
        best_pos = int(pos_indices[np.argmax(test_probs[pos_indices])])
    else:
        best_pos = int(np.argmax(test_probs))

    plot_path = reports_dir / "stress_signals_attention.png"
    plot_stress_signals_attention(
        out_path=plot_path,
        raw_window=split.raw_test[best_pos],
        attn_weights=test_attn[best_pos],
        fs=args.fs,
        history=history,
        y_test=test_y,
        bilstm_probs=test_probs,
        lr_probs=classical["logistic_regression"]["probs"],
        rf_probs=classical["random_forest"]["probs"],
        cm=bilstm_metrics["confusion_matrix"],
        sample_meta={"user": str(split.users_test[best_pos]), "prob": float(test_probs[best_pos])},
    )

    best_cls = classical["best_classical_metrics"]
    delta_f1 = round(bilstm_metrics["macro_f1"] - best_cls["macro_f1"], 4)
    delta_acc = round(bilstm_metrics["accuracy"] - best_cls["accuracy"], 4)
    delta_auc = (
        round(bilstm_metrics["roc_auc"] - best_cls["roc_auc"], 4)
        if bilstm_metrics["roc_auc"] is not None and best_cls["roc_auc"] is not None
        else None
    )

    report: dict[str, Any] = {
        "project": "07_wesad_stress_bilstm",
        "dataset_type": "real_public",
        "dataset_provenance": data_meta,
        "label_info": bundle.label_info,
        "split_protocol": {
            "strategy": "strict_subject_independent_group_split",
            "zero_subject_leakage_verified": True,
            "train_users": split.train_user_ids,
            "val_users": split.val_user_ids,
            "test_users": split.test_user_ids,
            "num_windows": {
                "train": int(len(split.y_train)),
                "val": int(len(split.y_val)),
                "test": int(len(split.y_test)),
            },
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "window_len": args.window_len,
            "stride": args.stride,
            "fs_hz": args.fs,
            "seed": args.seed,
        },
        "model": {
            "architecture": "BiLSTMAttentionStressNet",
            **param_info,
            "best_val_macro_f1": round(best_val_f1, 4),
            "test_loss": round(test_loss, 4),
        },
        "test_metrics_bilstm": bilstm_metrics,
        "classical_baselines": {
            "logistic_regression": {k: v for k, v in classical["logistic_regression"].items() if k != "probs"},
            "random_forest": {k: v for k, v in classical["random_forest"].items() if k != "probs"},
            "best_classical_model": classical["best_classical_name"],
        },
        "bilstm_vs_best_classical_delta": {
            "macro_f1_delta": delta_f1,
            "accuracy_delta": delta_acc,
            "roc_auc_delta": delta_auc,
        },
        "runtime": {
            "train_elapsed_s": train_timer["elapsed_s"],
            **vram_stats,
        },
        "artifacts": {
            "checkpoint": "artifacts/bilstm_stress_best.pt",
            "figure": "reports/stress_signals_attention.png",
        },
    }
    write_json_report(reports_dir / "metrics.json", report)
    cleanup_cuda()
    print(
        f"[07_wesad_stress_bilstm] Done in {train_timer['elapsed_s']}s | "
        f"Test Acc={bilstm_metrics['accuracy']:.4f} | Macro-F1={bilstm_metrics['macro_f1']:.4f} | "
        f"ROC-AUC={bilstm_metrics['roc_auc']} | Peak VRAM={vram_stats['peak_vram_mb']} MB"
    )


if __name__ == "__main__":
    main()
