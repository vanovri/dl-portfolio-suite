"""Project 12: Hybrid Quantum-Classical Deep Learning for Drug Discovery & Finance Risk."""
from .chemistry import (
    DESCRIPTOR_NAMES,
    extract_smiles_descriptors,
    load_bbbp_dataset,
    parse_bbbp_csv,
)
from .finance_risk import (
    calibrate_probabilities_platt,
    compute_portfolio_credit_risk,
    load_german_credit_dataset,
    parse_german_credit_lines,
)
from .model import ClassicalMatchedMLP, HybridQuantumClassifier
from .quantum_circuit import VariationalQuantumLayer

__all__ = [
    "DESCRIPTOR_NAMES",
    "extract_smiles_descriptors",
    "load_bbbp_dataset",
    "parse_bbbp_csv",
    "calibrate_probabilities_platt",
    "compute_portfolio_credit_risk",
    "load_german_credit_dataset",
    "parse_german_credit_lines",
    "VariationalQuantumLayer",
    "HybridQuantumClassifier",
    "ClassicalMatchedMLP",
]
