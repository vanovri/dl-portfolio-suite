"""Train & ablate Hybrid Quantum VQC vs Parameter-Matched Classical MLP on BBBP & German Credit."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
for _p in (str(PROJECT_DIR), str(REPO_ROOT), str(Path(__file__).resolve().parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from common.cuda_utils import (
    cleanup_cuda,
    configure_plot_style,
    count_parameters,
    get_device,
    get_vram_stats,
    reset_peak_vram,
    seed_everything,
    timed_block,
    write_json_report,
)
from src.chemistry import load_bbbp_dataset
from src.finance_risk import (
    calibrate_probabilities_platt,
    compute_portfolio_credit_risk,
    load_german_credit_dataset,
)
from src.model import ClassicalMatchedMLP, HybridQuantumClassifier
from src.viz import generate_quantum_drug_finance_figure


def _compute_binary_metrics(
    y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    has_two = len(np.unique(y_true)) == 2
    roc_auc = float(roc_auc_score(y_true, probs)) if has_two else 0.5
    pr_auc = float(average_precision_score(y_true, probs)) if has_two else float(y_true.mean())
    f1 = float(f1_score(y_true, preds, zero_division=0))
    acc = float(accuracy_score(y_true, preds))
    brier = float(brier_score_loss(y_true, probs))
    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
    }


def _find_best_val_threshold(y_val: np.ndarray, val_probs: np.ndarray) -> float:
    best_th = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.25, 0.75, 21):
        score = float(f1_score(y_val, (val_probs >= th).astype(np.int64), zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_th = float(th)
    return best_th


@torch.no_grad()
def _predict_logits(
    model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 128
) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    chunks: list[np.ndarray] = []
    for (bx,) in loader:
        bx = bx.to(device)
        logits = model(bx)
        chunks.append(logits.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_and_evaluate_model(
    model: nn.Module,
    dataset: dict[str, Any],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    """Train a binary classifier with class-weighted BCEWithLogitsLoss, best val ROC-AUC selection."""
    seed_everything(seed)
    model = model.to(device)

    X_train, y_train = dataset["X_train"], dataset["y_train"]
    X_val, y_val = dataset["X_val"], dataset["y_val"]
    X_test, y_test = dataset["X_test"], dataset["y_test"]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )

    pos_count = float(max(1, int((y_train == 1).sum())))
    neg_count = float(max(1, int((y_train == 0).sum())))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_val_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    with timed_block() as timer:
        for _ in range(1, epochs + 1):
            model.train()
            for bx, by in train_loader:
                bx = bx.to(device)
                by = by.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
            scheduler.step()

            val_logits = _predict_logits(model, X_val, device)
            val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -30.0, 30.0)))
            val_auc = (
                float(roc_auc_score(y_val, val_probs)) if len(np.unique(y_val)) == 2 else 0.5
            )
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    val_logits = _predict_logits(model, X_val, device)
    test_logits = _predict_logits(model, X_test, device)

    # Platt logistic calibration fit strictly on validation logits -> calibrated probabilities & Brier score.
    val_cal_probs = calibrate_probabilities_platt(val_logits, y_val, val_logits)
    test_cal_probs = calibrate_probabilities_platt(val_logits, y_val, test_logits)
    best_th = _find_best_val_threshold(y_val, val_cal_probs)

    metrics = _compute_binary_metrics(y_test, test_cal_probs, threshold=best_th)
    metrics["val_best_roc_auc"] = round(best_val_auc, 4)
    metrics["calibrated_threshold"] = round(best_th, 3)
    metrics["train_seconds"] = timer["elapsed_s"]
    metrics["params"] = count_parameters(model)

    fpr, tpr, _ = roc_curve(y_test, test_cal_probs)
    extras = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": metrics["roc_auc"],
        "test_cal_probs": test_cal_probs,
    }
    return model, metrics, extras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project 12: Hybrid Quantum-Classical DL for Drug Discovery (BBBP) & Finance Risk (German Credit)"
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--mc-sims", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_DIR / "reports")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_DIR / "artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    configure_plot_style()

    prefer_cuda = args.device != "cpu"
    device = get_device(prefer_cuda=prefer_cuda)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested via --device cuda, but no GPU is available.")

    reset_peak_vram()
    print(f"[Project 12] Device={device} | Qubits={args.n_qubits} | Layers={args.n_layers} | Epochs={args.epochs}")

    # Load BOTH real datasets
    bbbp_ds = load_bbbp_dataset(args.data_dir, seed=args.seed)
    credit_ds = load_german_credit_dataset(args.data_dir, seed=args.seed)
    print(
        f"  Track A BBBP: {bbbp_ds['n_total']} molecules (pos={bbbp_ds['positive_rate']:.3f}, dim={bbbp_ds['X_train'].shape[1]})"
    )
    print(
        f"  Track B German Credit: {credit_ds['n_total']} loans (default={credit_ds['default_rate']:.3f}, dim={credit_ds['in_dim']})"
    )

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    with timed_block() as total_timer:
        # ---- Track A: Drug Discovery (MoleculeNet BBBP) ----
        print("\n--- Track A: Drug Discovery (MoleculeNet BBBP) ---")
        q_bbbp = HybridQuantumClassifier(
            in_dim=bbbp_ds["X_train"].shape[1], n_qubits=args.n_qubits, n_layers=args.n_layers
        )
        c_bbbp = ClassicalMatchedMLP(
            in_dim=bbbp_ds["X_train"].shape[1], bottleneck_dim=args.n_qubits
        )

        q_bbbp, q_bbbp_metrics, q_bbbp_extra = train_and_evaluate_model(
            q_bbbp, bbbp_ds, device, args.epochs, args.batch_size, args.lr, seed=args.seed
        )
        c_bbbp, c_bbbp_metrics, c_bbbp_extra = train_and_evaluate_model(
            c_bbbp, bbbp_ds, device, args.epochs, args.batch_size, args.lr, seed=args.seed + 1
        )
        print(f"  BBBP Hybrid Quantum : ROC-AUC={q_bbbp_metrics['roc_auc']:.4f} | PR-AUC={q_bbbp_metrics['pr_auc']:.4f} | F1={q_bbbp_metrics['f1']:.4f} | Brier={q_bbbp_metrics['brier_score']:.4f}")
        print(f"  BBBP Classical MLP  : ROC-AUC={c_bbbp_metrics['roc_auc']:.4f} | PR-AUC={c_bbbp_metrics['pr_auc']:.4f} | F1={c_bbbp_metrics['f1']:.4f} | Brier={c_bbbp_metrics['brier_score']:.4f}")

        # Extract Pauli-Z expectations and verify exact circuit unitarity & state norm!
        q_bbbp.eval()
        with torch.no_grad():
            x_te_t = torch.from_numpy(bbbp_ds["X_test"]).to(device)
            _, q_exp_te, q_angles_te = q_bbbp(x_te_t, return_quantum=True)
            unitary_check = q_bbbp.quantum.verify_unitary_and_norm(q_angles_te[:16])
            q_exp_np = q_exp_te.detach().cpu().numpy()
            pauli_pos = q_exp_np[bbbp_ds["y_test"] == 1]
            pauli_neg = q_exp_np[bbbp_ds["y_test"] == 0]

        torch.save(
            {"state_dict": q_bbbp.state_dict(), "metrics": q_bbbp_metrics, "in_dim": bbbp_ds["X_train"].shape[1]},
            args.artifacts_dir / "hybrid_quantum_bbbp.pt",
        )

        # ---- Track B: Finance & Credit Risk Modeling (UCI German Credit) ----
        print("\n--- Track B: Finance & Risk Modeling (UCI German Credit) ---")
        q_cred = HybridQuantumClassifier(
            in_dim=credit_ds["in_dim"], n_qubits=args.n_qubits, n_layers=args.n_layers
        )
        c_cred = ClassicalMatchedMLP(
            in_dim=credit_ds["in_dim"], bottleneck_dim=args.n_qubits
        )

        q_cred, q_cred_metrics, q_cred_extra = train_and_evaluate_model(
            q_cred, credit_ds, device, args.epochs, args.batch_size, args.lr, seed=args.seed + 10
        )
        c_cred, c_cred_metrics, c_cred_extra = train_and_evaluate_model(
            c_cred, credit_ds, device, args.epochs, args.batch_size, args.lr, seed=args.seed + 11
        )
        print(f"  Credit Hybrid Quantum: ROC-AUC={q_cred_metrics['roc_auc']:.4f} | PR-AUC={q_cred_metrics['pr_auc']:.4f} | F1={q_cred_metrics['f1']:.4f} | Brier={q_cred_metrics['brier_score']:.4f}")
        print(f"  Credit Classical MLP : ROC-AUC={c_cred_metrics['roc_auc']:.4f} | PR-AUC={c_cred_metrics['pr_auc']:.4f} | F1={c_cred_metrics['f1']:.4f} | Brier={c_cred_metrics['brier_score']:.4f}")

        # Portfolio Credit Risk Analytics (Monte Carlo Vasicek VaR95 / VaR99 / CVaR99)
        q_risk = compute_portfolio_credit_risk(
            pd_estimates=q_cred_extra["test_cal_probs"],
            exposures=credit_ds["exposures_test"],
            lgd=0.45,
            asset_correlation=0.15,
            n_simulations=args.mc_sims,
            seed=args.seed,
        )
        c_risk = compute_portfolio_credit_risk(
            pd_estimates=c_cred_extra["test_cal_probs"],
            exposures=credit_ds["exposures_test"],
            lgd=0.45,
            asset_correlation=0.15,
            n_simulations=args.mc_sims,
            seed=args.seed,
        )
        print(
            f"  Portfolio Risk (Quantum PD): EL={q_risk['expected_loss']:.1f} | "
            f"VaR95={q_risk['var_95']:.1f} | VaR99={q_risk['var_99']:.1f} | CVaR99={q_risk['cvar_99']:.1f}"
        )

        torch.save(
            {"state_dict": q_cred.state_dict(), "metrics": q_cred_metrics, "in_dim": credit_ds["in_dim"]},
            args.artifacts_dir / "hybrid_quantum_german_credit.pt",
        )

    fig_path = generate_quantum_drug_finance_figure(
        output_path=args.reports_dir / "quantum_drug_finance.png",
        pauli_z_pos=pauli_pos,
        pauli_z_neg=pauli_neg,
        unitary_error=unitary_check["unitary_max_error"],
        bbbp_roc={"hybrid_quantum": q_bbbp_extra, "classical_mlp": c_bbbp_extra},
        credit_roc={"hybrid_quantum": q_cred_extra, "classical_mlp": c_cred_extra},
        portfolio_risk=q_risk,
    )

    q_risk_json = {k: v for k, v in q_risk.items() if k != "simulated_losses"}
    c_risk_json = {k: v for k, v in c_risk.items() if k != "simulated_losses"}
    vram_stats = get_vram_stats()
    cleanup_cuda()

    report = {
        "project": "12_hybrid_quantum_dl",
        "dataset_type": "real_public",
        "quantum_verification": unitary_check,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_qubits": args.n_qubits,
            "n_layers": args.n_layers,
            "mc_simulations": args.mc_sims,
            "seed": args.seed,
        },
        "hardware": {
            **vram_stats,
            "total_wall_seconds": total_timer["elapsed_s"],
        },
        "track_a_drug_discovery_bbbp": {
            "dataset": bbbp_ds["name"],
            "source_url": bbbp_ds["source_url"],
            "n_total": bbbp_ds["n_total"],
            "n_features": int(bbbp_ds["X_train"].shape[1]),
            "positive_rate": bbbp_ds["positive_rate"],
            "hybrid_quantum": q_bbbp_metrics,
            "classical_matched_mlp": c_bbbp_metrics,
        },
        "track_b_finance_credit_risk": {
            "dataset": credit_ds["name"],
            "source_url": credit_ds["source_url"],
            "n_total": credit_ds["n_total"],
            "n_features": credit_ds["in_dim"],
            "default_rate": credit_ds["default_rate"],
            "hybrid_quantum": q_cred_metrics,
            "classical_matched_mlp": c_cred_metrics,
            "portfolio_risk_quantum": q_risk_json,
            "portfolio_risk_classical": c_risk_json,
        },
        "artifacts": {
            "bbbp_checkpoint": "artifacts/hybrid_quantum_bbbp.pt",
            "credit_checkpoint": "artifacts/hybrid_quantum_german_credit.pt",
            "figure": f"reports/{fig_path.name}",
        },
    }
    report_path = write_json_report(args.reports_dir / "metrics.json", report)
    print(f"\n[Project 12] Saved report -> {report_path}")
    print(f"[Project 12] Saved figure -> {fig_path}")
    print(f"[Project 12] Unitary max error -> {unitary_check['unitary_max_error']:.2e} | Peak VRAM -> {vram_stats['peak_vram_mb']} MB")


if __name__ == "__main__":
    main()
