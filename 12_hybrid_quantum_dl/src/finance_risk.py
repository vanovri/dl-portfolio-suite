"""UCI Statlog German Credit loader, Platt PD calibration, and Monte Carlo VaR/CVaR risk analytics."""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

GERMAN_CREDIT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
)

# 20 UCI German Credit attributes (0-based indices):
# Numerical: 1 (duration_mo), 4 (credit_amount), 7 (installment_rate),
#            10 (residence_since), 12 (age), 15 (existing_credits), 17 (dependents)
NUMERIC_COLS = (1, 4, 7, 10, 12, 15, 17)
CATEGORICAL_COLS = tuple(i for i in range(20) if i not in NUMERIC_COLS)
CREDIT_AMOUNT_COL = 4


def parse_german_credit_lines(lines: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse raw german.data whitespace rows into numeric, categorical, exposures, and binary default labels.

    UCI German Credit label encoding: 1 = Good credit (y=0), 2 = Bad/Default credit (y=1).
    """
    num_rows: list[list[float]] = []
    cat_rows: list[list[str]] = []
    exposures: list[float] = []
    labels: list[int] = []

    for raw in lines:
        parts = raw.strip().split()
        if len(parts) != 21:
            continue
        raw_lbl = int(parts[20])
        if raw_lbl not in (1, 2):
            continue
        num_vals = [float(parts[idx]) for idx in NUMERIC_COLS]
        cat_vals = [str(parts[idx]) for idx in CATEGORICAL_COLS]
        num_rows.append(num_vals)
        cat_rows.append(cat_vals)
        exposures.append(float(parts[CREDIT_AMOUNT_COL]))
        labels.append(1 if raw_lbl == 2 else 0)

    if not labels:
        raise ValueError("No valid German Credit rows parsed.")

    return (
        np.asarray(num_rows, dtype=np.float32),
        np.asarray(cat_rows, dtype=object),
        np.asarray(exposures, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
    )


def download_german_credit(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "german.data"
    if not path.exists() or path.stat().st_size < 1024:
        urllib.request.urlretrieve(GERMAN_CREDIT_URL, path)
    return path


def load_german_credit_dataset(data_dir: Path, seed: int = 42) -> dict[str, Any]:
    """Load real UCI Statlog German Credit dataset with leakage-free stratified 70/15/15 split."""
    path = download_german_credit(data_dir)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    X_num, X_cat, exposures, y = parse_german_credit_lines(lines)

    indices = np.arange(len(y))
    idx_train, idx_tmp, y_train, y_tmp = train_test_split(
        indices, y, test_size=0.30, stratify=y, random_state=seed
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )

    # Fit preprocessing strictly on training split.
    scaler = StandardScaler()
    num_tr = scaler.fit_transform(X_num[idx_train])
    num_va = scaler.transform(X_num[idx_val])
    num_te = scaler.transform(X_num[idx_test])

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_tr = ohe.fit_transform(X_cat[idx_train])
    cat_va = ohe.transform(X_cat[idx_val])
    cat_te = ohe.transform(X_cat[idx_test])

    X_train = np.hstack([num_tr, cat_tr]).astype(np.float32)
    X_val = np.hstack([num_va, cat_va]).astype(np.float32)
    X_test = np.hstack([num_te, cat_te]).astype(np.float32)

    return {
        "name": "UCI Statlog German Credit (Default Risk)",
        "source_url": GERMAN_CREDIT_URL,
        "X_train": X_train,
        "y_train": y_train.astype(np.int64),
        "X_val": X_val,
        "y_val": y_val.astype(np.int64),
        "X_test": X_test,
        "y_test": y_test.astype(np.int64),
        "exposures_test": exposures[idx_test],
        "n_total": int(len(y)),
        "in_dim": int(X_train.shape[1]),
        "default_rate": round(float(y.mean()), 4),
    }


def calibrate_probabilities_platt(
    val_logits: np.ndarray, val_labels: np.ndarray, test_logits: np.ndarray
) -> np.ndarray:
    """Apply Platt logistic calibration fitted strictly on validation logits -> calibrated test PD."""
    v_logits = np.asarray(val_logits, dtype=np.float64).reshape(-1, 1)
    v_labels = np.asarray(val_labels, dtype=np.int64)
    t_logits = np.asarray(test_logits, dtype=np.float64).reshape(-1, 1)

    if len(np.unique(v_labels)) < 2:
        return 1.0 / (1.0 + np.exp(-np.clip(t_logits.ravel(), -30.0, 30.0)))

    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    calibrator.fit(v_logits, v_labels)
    probs = calibrator.predict_proba(t_logits)[:, 1]
    return np.clip(probs, 1e-5, 1.0 - 1e-5)


def compute_portfolio_credit_risk(
    pd_estimates: np.ndarray,
    exposures: np.ndarray,
    lgd: float = 0.45,
    asset_correlation: float = 0.15,
    n_simulations: int = 20_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute Basel-style Portfolio Expected Loss, VaR95/VaR99, and CVaR95/CVaR99 via Vasicek Monte Carlo.

    Each obligor i defaults when latent factor Z_i = sqrt(rho)*M + sqrt(1-rho)*eps_i < Phi^{-1}(PD_i).
    """
    from scipy.stats import norm

    pds = np.clip(np.asarray(pd_estimates, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    ead = np.maximum(np.asarray(exposures, dtype=np.float64), 0.0)
    if pds.shape != ead.shape:
        raise ValueError("pd_estimates and exposures must have identical shape.")

    total_exposure = float(ead.sum())
    analytic_expected_loss = float(np.sum(pds * ead * lgd))

    rng = np.random.default_rng(seed)
    thresholds = norm.ppf(pds)  # (N,)
    rho = float(np.clip(asset_correlation, 0.0, 0.95))
    sqrt_rho = np.sqrt(rho)
    sqrt_one_minus_rho = np.sqrt(1.0 - rho)

    # Vectorized Monte Carlo simulation in chunks to keep RAM tiny (<30 MB).
    chunk_size = 4000
    losses = np.empty(n_simulations, dtype=np.float64)
    loss_given_default_ead = ead * lgd

    start = 0
    while start < n_simulations:
        end = min(n_simulations, start + chunk_size)
        m_sim = end - start
        systematic = rng.standard_normal((m_sim, 1))
        idiosyncratic = rng.standard_normal((m_sim, len(pds)))
        latent = sqrt_rho * systematic + sqrt_one_minus_rho * idiosyncratic
        defaults = (latent < thresholds[None, :]).astype(np.float64)
        losses[start:end] = defaults @ loss_given_default_ead
        start = end

    var_95 = float(np.quantile(losses, 0.95))
    var_99 = float(np.quantile(losses, 0.99))
    tail_95 = losses[losses >= var_95]
    tail_99 = losses[losses >= var_99]
    cvar_95 = float(tail_95.mean()) if tail_95.size > 0 else var_95
    cvar_99 = float(tail_99.mean()) if tail_99.size > 0 else var_99
    mc_expected_loss = float(losses.mean())

    return {
        "n_loans": int(len(pds)),
        "lgd": float(lgd),
        "asset_correlation_rho": rho,
        "n_simulations": int(n_simulations),
        "total_exposure": round(total_exposure, 2),
        "mean_calibrated_pd": round(float(pds.mean()), 4),
        "expected_loss": round(analytic_expected_loss, 2),
        "mc_expected_loss": round(mc_expected_loss, 2),
        "var_95": round(var_95, 2),
        "var_99": round(var_99, 2),
        "cvar_95": round(cvar_95, 2),
        "cvar_99": round(cvar_99, 2),
        "expected_loss_pct_exposure": round(100.0 * analytic_expected_loss / max(1.0, total_exposure), 3),
        "var_99_pct_exposure": round(100.0 * var_99 / max(1.0, total_exposure), 3),
        "simulated_losses": losses,
    }
