"""Deterministic offline CPU unit tests for Project 12: Hybrid Quantum-Classical DL."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.cuda_utils import count_parameters
from src.chemistry import DESCRIPTOR_NAMES, extract_smiles_descriptors, parse_bbbp_csv
from src.finance_risk import (
    calibrate_probabilities_platt,
    compute_portfolio_credit_risk,
    parse_german_credit_lines,
)
from src.model import ClassicalMatchedMLP, HybridQuantumClassifier
from src.quantum_circuit import VariationalQuantumLayer
from src.viz import generate_quantum_drug_finance_figure


def test_smiles_extractor_returns_24_valid_descriptors() -> None:
    assert len(DESCRIPTOR_NAMES) == 24
    # Aspirin SMILES: CC(=O)Oc1ccccc1C(=O)O
    aspirin = extract_smiles_descriptors("CC(=O)Oc1ccccc1C(=O)O")
    assert aspirin.shape == (24,)
    assert np.isfinite(aspirin).all()
    idx = {name: i for i, name in enumerate(DESCRIPTOR_NAMES)}
    assert aspirin[idx["count_C"]] == 9.0
    assert aspirin[idx["count_O"]] == 4.0
    assert aspirin[idx["aromatic_atom_count"]] == 6.0
    assert aspirin[idx["ring_closure_count"]] == 1.0
    assert aspirin[idx["double_bond_count"]] == 2.0
    assert aspirin[idx["estimated_mw"]] > 140.0


def test_smiles_handles_halogens_chirality_and_charges() -> None:
    # Chlorinated chiral charged molecule
    vec = extract_smiles_descriptors("Clc1ccc([C@H]([NH3+])C#N)cc1Br")
    idx = {name: i for i, name in enumerate(DESCRIPTOR_NAMES)}
    assert vec[idx["count_Cl"]] == 1.0
    assert vec[idx["count_Br"]] == 1.0
    assert vec[idx["count_C"]] == 8.0  # Does NOT miscount Cl as C!
    assert vec[idx["chiral_center_count"]] == 1.0
    assert vec[idx["formal_charge_abs_sum"]] == 1.0
    assert vec[idx["triple_bond_count"]] == 1.0
    assert vec[idx["max_branch_depth"]] == 2.0


def test_parse_bbbp_csv_offline(tmp_path: Path) -> None:
    csv_file = tmp_path / "mini_bbbp.csv"
    csv_file.write_text(
        "num,name,p_np,smiles\n"
        "1,Caffeine,1,Cn1cnc2c1c(=O)n(C)c(=O)n2C\n"
        "2,VancomycinLike,0,CC(O)CC(=O)O[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O\n"
        "3,BadRow,invalid,CCO\n",
        encoding="utf-8",
    )
    X, y, smiles = parse_bbbp_csv(csv_file)
    assert X.shape == (2, 24)
    assert y.tolist() == [1, 0]
    assert len(smiles) == 2


def test_parse_german_credit_lines_offline() -> None:
    sample_lines = [
        "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1",
        "A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2",
    ]
    X_num, X_cat, exposures, y = parse_german_credit_lines(sample_lines)
    assert X_num.shape == (2, 7)
    assert X_cat.shape == (2, 13)
    assert exposures.tolist() == [1169.0, 5951.0]
    assert y.tolist() == [0, 1]


def test_quantum_layer_unitarity_and_normalization() -> None:
    torch.manual_seed(0)
    vqc = VariationalQuantumLayer(n_qubits=4, n_layers=3)
    angles = torch.randn(8, 4)
    report = vqc.verify_unitary_and_norm(angles)
    assert report["hilbert_dim"] == 16
    assert report["is_unitary"] is True
    assert report["is_normalized"] is True
    assert report["unitary_max_error"] < 1e-5
    assert report["norm_max_error"] < 1e-5


def test_quantum_layer_exact_pauli_z_bounds_and_autograd() -> None:
    torch.manual_seed(3)
    vqc = VariationalQuantumLayer(n_qubits=4, n_layers=3)
    angles = torch.randn(5, 4, requires_grad=True)
    exp_z = vqc(angles)
    assert exp_z.shape == (5, 4)
    assert torch.all((exp_z >= -1.0) & (exp_z <= 1.0))

    loss = exp_z.pow(2).sum()
    loss.backward()
    assert angles.grad is not None and torch.isfinite(angles.grad).all()
    assert vqc.theta.grad is not None and float(vqc.theta.grad.abs().sum().item()) > 0.0
    assert vqc.phi.grad is not None and float(vqc.phi.grad.abs().sum().item()) > 0.0


def test_quantum_zero_angles_and_zero_params_gives_plus_one_z() -> None:
    vqc = VariationalQuantumLayer(n_qubits=4, n_layers=2)
    with torch.no_grad():
        vqc.theta.zero_()
        vqc.phi.zero_()
    angles = torch.zeros(2, 4)
    exp_z = vqc(angles)
    # |0000> stays |0000> under identity rotations and CNOTs -> <Z_k> == +1.0 for all qubits.
    assert torch.allclose(exp_z, torch.ones(2, 4), atol=1e-5)


def test_hybrid_and_classical_models_parameter_matched_and_trainable() -> None:
    torch.manual_seed(7)
    hq = HybridQuantumClassifier(in_dim=24, n_qubits=4, n_layers=3)
    cm = ClassicalMatchedMLP(in_dim=24, bottleneck_dim=4)
    hq_p = count_parameters(hq)["trainable_params"]
    cm_p = count_parameters(cm)["trainable_params"]
    assert abs(hq_p - cm_p) / max(hq_p, cm_p) < 0.05

    x = torch.randn(6, 24)
    logits_q, q_exp, angles = hq(x, return_quantum=True)
    logits_c = cm(x)
    assert logits_q.shape == (6,)
    assert logits_c.shape == (6,)
    assert q_exp.shape == (6, 4)
    assert torch.all(angles.abs() <= np.pi + 1e-5)


def test_platt_calibration_bounds() -> None:
    rng = np.random.default_rng(1)
    val_y = np.array([0] * 30 + [1] * 30)
    val_logits = np.where(val_y == 1, 1.5, -1.5) + rng.normal(0, 0.5, size=60)
    test_logits = np.array([-3.0, 0.0, 3.0])
    cal = calibrate_probabilities_platt(val_logits, val_y, test_logits)
    assert cal.shape == (3,)
    assert cal[0] < cal[1] < cal[2]
    assert np.all((cal > 0.0) & (cal < 1.0))


def test_portfolio_credit_risk_ordering_var_cvar() -> None:
    pds = np.full(50, 0.20, dtype=np.float64)
    exposures = np.linspace(1000.0, 5000.0, 50)
    risk = compute_portfolio_credit_risk(
        pd_estimates=pds,
        exposures=exposures,
        lgd=0.45,
        asset_correlation=0.15,
        n_simulations=5000,
        seed=42,
    )
    assert risk["expected_loss"] > 0.0
    assert risk["expected_loss"] < risk["var_95"] <= risk["var_99"] <= risk["cvar_99"]
    assert abs(risk["expected_loss"] - risk["mc_expected_loss"]) / risk["expected_loss"] < 0.12


def test_quantum_drug_finance_visualization(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    fake_roc = {
        "hybrid_quantum": {"fpr": np.array([0.0, 0.2, 1.0]), "tpr": np.array([0.0, 0.82, 1.0]), "auc": 0.81},
        "classical_mlp": {"fpr": np.array([0.0, 0.25, 1.0]), "tpr": np.array([0.0, 0.78, 1.0]), "auc": 0.77},
    }
    risk = compute_portfolio_credit_risk(
        pd_estimates=rng.uniform(0.05, 0.45, size=30),
        exposures=rng.uniform(1000.0, 8000.0, size=30),
        n_simulations=2000,
        seed=9,
    )
    out = generate_quantum_drug_finance_figure(
        output_path=tmp_path / "quantum_drug_finance.png",
        pauli_z_pos=rng.uniform(-0.8, 0.8, size=(20, 4)),
        pauli_z_neg=rng.uniform(-0.8, 0.8, size=(20, 4)),
        unitary_error=1.2e-7,
        bbbp_roc=fake_roc,
        credit_roc=fake_roc,
        portfolio_risk=risk,
    )
    assert out.exists() and out.stat().st_size > 5000
