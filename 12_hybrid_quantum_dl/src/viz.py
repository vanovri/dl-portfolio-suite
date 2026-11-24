"""Publication-grade 4-panel visualization for Project 12: Hybrid Quantum DL (Drug + Finance)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _draw_vqc_and_pauli_panel(
    ax: plt.Axes,
    pauli_z_pos: np.ndarray,
    pauli_z_neg: np.ndarray,
    unitary_error: float,
    n_qubits: int = 4,
    n_layers: int = 3,
) -> None:
    """Draw class-separated Pauli-Z expectation values <Z_k> alongside circuit fidelity badge."""
    qubits = np.arange(n_qubits)
    width = 0.34
    pos_mean = pauli_z_pos.mean(axis=0) if len(pauli_z_pos) else np.zeros(n_qubits)
    pos_std = pauli_z_pos.std(axis=0) if len(pauli_z_pos) else np.zeros(n_qubits)
    neg_mean = pauli_z_neg.mean(axis=0) if len(pauli_z_neg) else np.zeros(n_qubits)
    neg_std = pauli_z_neg.std(axis=0) if len(pauli_z_neg) else np.zeros(n_qubits)

    ax.bar(
        qubits - width / 2,
        pos_mean,
        yerr=pos_std,
        width=width,
        capsize=4,
        color="#38bdf8",
        alpha=0.9,
        label="BBB+ Permeable (y=1)",
    )
    ax.bar(
        qubits + width / 2,
        neg_mean,
        yerr=neg_std,
        width=width,
        capsize=4,
        color="#f43f5e",
        alpha=0.9,
        label="BBB- Non-Permeable (y=0)",
    )
    ax.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1.0)
    ax.set_ylim(-1.15, 1.18)
    ax.set_xticks(qubits)
    ax.set_xticklabels([f"Qubit {q}\n<Z_{q}>" for q in range(n_qubits)], fontsize=9)
    ax.set_ylabel("Exact Pauli-Z Expectation [-1, 1]")
    ax.set_title(
        f"A. {n_qubits}-Qubit VQC ({n_layers}L RY-RZ-CNOT Ring) Pauli-Z Readout",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8.3)

    # Compact circuit topology & unitary proof callout box
    badge = (
        f"Ansatz: RY(x) -> [RY(th)-RZ(ph)-CNOT_ring]x{n_layers}\n"
        f"Hilbert dim = 2^{n_qubits} = {1 << n_qubits} | max|U^H U - I| = {unitary_error:.1e}"
    )
    ax.text(
        0.02,
        0.96,
        badge,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", edgecolor="#38bdf8", alpha=0.9),
    )


def generate_quantum_drug_finance_figure(
    output_path: Path,
    pauli_z_pos: np.ndarray,
    pauli_z_neg: np.ndarray,
    unitary_error: float,
    bbbp_roc: dict[str, dict[str, Any]],
    credit_roc: dict[str, dict[str, Any]],
    portfolio_risk: dict[str, Any],
) -> Path:
    """Render 4-panel visual proof: VQC Pauli-Z | BBBP ROC | German Credit ROC | Portfolio VaR/CVaR."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.5))
    plt.subplots_adjust(hspace=0.32, wspace=0.24)

    # Panel A: VQC Pauli-Z Expectation Map + Unitary Verification
    _draw_vqc_and_pauli_panel(
        axes[0, 0],
        pauli_z_pos=pauli_z_pos,
        pauli_z_neg=pauli_z_neg,
        unitary_error=unitary_error,
    )

    # Panel B: Track A Drug Discovery (MoleculeNet BBBP) ROC Curves
    ax_b = axes[0, 1]
    for key, label, color, ls in [
        ("hybrid_quantum", "Hybrid Quantum VQC", "#38bdf8", "-"),
        ("classical_mlp", "Classical Matched MLP", "#f59e0b", "--"),
    ]:
        info = bbbp_roc[key]
        ax_b.plot(
            info["fpr"],
            info["tpr"],
            label=f"{label} (AUC={info['auc']:.3f})",
            color=color,
            linestyle=ls,
            linewidth=2.2,
        )
    ax_b.plot([0, 1], [0, 1], color="#64748b", linestyle=":", linewidth=1.2)
    ax_b.set_xlabel("False Positive Rate")
    ax_b.set_ylabel("True Positive Rate")
    ax_b.set_title("B. Track A: Drug Discovery (MoleculeNet BBBP ROC)", fontsize=10.5, fontweight="bold")
    ax_b.grid(alpha=0.35)
    ax_b.legend(loc="lower right", fontsize=8.8)

    # Panel C: Track B Finance Risk (UCI Statlog German Credit) ROC Curves
    ax_c = axes[1, 0]
    for key, label, color, ls in [
        ("hybrid_quantum", "Hybrid Quantum VQC", "#10b981", "-"),
        ("classical_mlp", "Classical Matched MLP", "#a855f7", "--"),
    ]:
        info = credit_roc[key]
        ax_c.plot(
            info["fpr"],
            info["tpr"],
            label=f"{label} (AUC={info['auc']:.3f})",
            color=color,
            linestyle=ls,
            linewidth=2.2,
        )
    ax_c.plot([0, 1], [0, 1], color="#64748b", linestyle=":", linewidth=1.2)
    ax_c.set_xlabel("False Positive Rate")
    ax_c.set_ylabel("True Positive Rate")
    ax_c.set_title("C. Track B: Credit Default Risk (UCI German Credit ROC)", fontsize=10.5, fontweight="bold")
    ax_c.grid(alpha=0.35)
    ax_c.legend(loc="lower right", fontsize=8.8)

    # Panel D: Portfolio Credit Loss Distribution with EL, VaR95, VaR99, CVaR99
    ax_d = axes[1, 1]
    losses = np.asarray(portfolio_risk["simulated_losses"], dtype=np.float64) / 1000.0  # in k-DM
    el_k = portfolio_risk["expected_loss"] / 1000.0
    var95_k = portfolio_risk["var_95"] / 1000.0
    var99_k = portfolio_risk["var_99"] / 1000.0
    cvar99_k = portfolio_risk["cvar_99"] / 1000.0

    ax_d.hist(losses, bins=45, density=True, color="#38bdf8", alpha=0.55, edgecolor="#0f172a")
    tail_losses = losses[losses >= var99_k]
    if tail_losses.size > 0:
        ax_d.hist(tail_losses, bins=12, density=True, color="#ef4444", alpha=0.80, label="99% Tail Region")

    ax_d.axvline(el_k, color="#22c55e", linestyle="-", linewidth=2.0, label=f"Expected Loss = {el_k:.1f}k")
    ax_d.axvline(var95_k, color="#f59e0b", linestyle="--", linewidth=2.0, label=f"95% VaR = {var95_k:.1f}k")
    ax_d.axvline(var99_k, color="#f43f5e", linestyle="-.", linewidth=2.0, label=f"99% VaR = {var99_k:.1f}k")
    ax_d.axvline(cvar99_k, color="#e879f9", linestyle=":", linewidth=2.2, label=f"99% CVaR = {cvar99_k:.1f}k")

    ax_d.set_xlabel("Simulated Held-Out Portfolio Credit Loss (Thousands)")
    ax_d.set_ylabel("Density")
    ax_d.set_title(
        "D. Calibrated Quantum PD Portfolio Loss Distribution (Vasicek MC)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_d.grid(alpha=0.35)
    ax_d.legend(loc="upper right", fontsize=8.0)

    fig.suptitle(
        "Project 12: Differentiable Hybrid Quantum-Classical DL (Drug Discovery & Credit Risk)",
        fontsize=13.2,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
