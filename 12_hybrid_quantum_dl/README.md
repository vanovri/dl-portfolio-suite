# Project 12: Differentiable Hybrid Quantum-Classical Deep Learning (Drug Discovery & Finance Risk)

## Overview
This project implements a true complex-valued differentiable 4-qubit Variational Quantum Circuit (`torch.complex64`, 16-dimensional Hilbert statevector) with native PyTorch autograd, benchmarked head-to-head against a parameter-matched purely classical MLP across two real-world high-stakes domains: molecular blood-brain barrier drug permeability and institutional credit portfolio risk.

## Dual Real-Dataset Tracks
1. **Track A -- Drug Discovery (MoleculeNet BBBP)**:
   - Source: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv` (~2,050 real drug compounds, binary `p_np` permeability target).
   - Pure-Python RDKit-free **SMILES 24-Descriptor Cheminformatics Extractor** (`src/chemistry.py`): computes atom counts (C, N, O, S, F, Cl, Br, P), aromatic atom count & ratio, ring closures, double/triple bonds, branch depth, chiral centers, formal charges, estimated molecular weight, HBD/HBA/TPSA proxies, C:heteroatom ratio, Lipinski Rule-of-5 violation count, SMILES length, and token Shannon entropy.
   - Leakage-free stratified 70/15/15 split (`StandardScaler` fitted exclusively on training data).

2. **Track B -- Finance & Credit Risk Modeling (UCI Statlog German Credit)**:
   - Source: `https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data` (1,000 real loan credit risk records, 20 categorical + numerical attributes, binary default label).
   - **Portfolio Credit Risk Engine** (`src/finance_risk.py`): Platt logistic probability calibration (fitted strictly on validation logits) -> Expected Loss (`EL = sum(PD * Exposure * LGD)` at 45% LGD) + 20,000-trial Vasicek one-factor Monte Carlo simulation (`rho = 0.15`) yielding **95% VaR**, **99% VaR**, and **99% CVaR / Expected Shortfall**.

## Differentiable Quantum Circuit (`src/quantum_circuit.py`)
- Exact complex-valued statevector evolution (`n_qubits=4`, `dim=16`, `n_layers=3`).
- Gate sequence: Angle Encoding `RY(x_i)` -> 3 variational layers of parameterized `RY(theta)` + `RZ(phi)` + entangling ring `CNOT(0->1, 1->2, 2->3, 3->0)` -> exact Pauli-Z expectation measurement `<psi | Z_k | psi> in [-1, 1]^4`.
- Explicit mathematical verification via `verify_unitary_and_norm()` proving `max|U^H U - I| < 1e-5` and `sum_j |alpha_j|^2 = 1.0`.
- Runs cleanly in `float32`/`complex64` on both CPU and GTX 1650 Ti CUDA 12.4 (explicitly guarding complex operations outside FP16 autocast).

## Fair Scientific Ablation (`src/model.py`)
- `HybridQuantumClassifier`: Classical compressor (`in_dim -> 16 -> 4`, `pi * tanh`) -> `VariationalQuantumLayer(4, 3)` -> residual skip + logit classifier.
- `ClassicalMatchedMLP`: Parameter-matched classical baseline within ~2-3% of the quantum hybrid parameter budget.

## Reproduction
```bash
# Offline CPU unit tests (includes unitary & statevector norm proof)
pytest tests/test_quantum_hybrid.py -v

# Train and evaluate both tracks on GPU/CPU
python train.py --epochs 25 --batch-size 32 --seed 42
```

## Outputs
- `reports/metrics.json`: ROC-AUC, PR-AUC, F1, Accuracy, Brier Score, Portfolio VaR95/VaR99/CVaR99, and circuit unitary fidelity audit.
- `reports/quantum_drug_finance.png`: 4-panel visual proof (VQC Pauli-Z expectation readout, BBBP ROC, German Credit ROC, Portfolio Credit Loss distribution with VaR/CVaR).
- `artifacts/hybrid_quantum_bbbp.pt` & `artifacts/hybrid_quantum_german_credit.pt`: Saved hybrid quantum checkpoints.
