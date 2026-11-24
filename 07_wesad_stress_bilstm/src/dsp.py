"""Digital signal processing, per-subject normalization, and zero-leakage windowing for WESAD."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

RAW_SENSOR_COLS: tuple[str, ...] = ("acc_x", "acc_y", "acc_z", "bvp", "eda", "temp")
FEATURE_NAMES: tuple[str, ...] = (
    "acc_x_z",
    "acc_y_z",
    "acc_z_z",
    "acc_mag_z",
    "bvp_bp_z",
    "eda_tonic_z",
    "eda_phasic_z",
    "eda_diff_z",
    "temp_lp_z",
)


@dataclass
class WindowBundle:
    """Subject-grouped physiological windows ready for sequence modeling."""

    X: np.ndarray  # (N, window_len, num_features) float32
    y: np.ndarray  # (N,) int64
    users: np.ndarray  # (N,) object/str
    raw_channels: np.ndarray  # (N, window_len, 4) -> [bvp_bp, eda_tonic, eda_phasic, temp_lp]
    feature_names: tuple[str, ...]
    label_info: dict[str, Any]


@dataclass
class SubjectSplit:
    """Strict subject-independent train/val/test split with zero cross-subject leakage."""

    X_train: np.ndarray
    y_train: np.ndarray
    users_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    users_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    users_test: np.ndarray
    raw_test: np.ndarray
    train_user_ids: list[str]
    val_user_ids: list[str]
    test_user_ids: list[str]


def _safe_sosfiltfilt(sos: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Apply zero-phase SOS Butterworth filter safely across arbitrary signal lengths."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.size <= 3 or np.allclose(arr, arr[0]):
        return arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0)
    default_padlen = 3 * (2 * sos.shape[0] + 1)
    padlen = min(default_padlen, arr.size - 1)
    filtered = sosfiltfilt(sos, arr, padlen=padlen)
    return filtered.astype(np.float32)


def butter_bandpass_filter(
    signal: np.ndarray,
    fs: float = 4.0,
    low_hz: float = 0.5,
    high_hz: float = 1.8,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth bandpass filter for BVP cardiac waveforms."""
    nyq = 0.5 * float(fs)
    if nyq <= 0.0:
        raise ValueError(f"Sampling rate fs must be positive, got {fs}")
    hi = min(high_hz, 0.90 * nyq)
    lo = min(low_hz, 0.45 * hi)
    lo = max(lo, 1e-3 * nyq)
    sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return _safe_sosfiltfilt(sos, signal)


def butter_lowpass_filter(
    signal: np.ndarray,
    fs: float = 4.0,
    cutoff_hz: float = 0.05,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth low-pass filter for tonic EDA or skin temperature."""
    nyq = 0.5 * float(fs)
    if nyq <= 0.0:
        raise ValueError(f"Sampling rate fs must be positive, got {fs}")
    wn = min(cutoff_hz / nyq, 0.90)
    wn = max(wn, 1e-4)
    sos = butter(order, wn, btype="lowpass", output="sos")
    return _safe_sosfiltfilt(sos, signal)


def decompose_eda(
    eda: np.ndarray,
    fs: float = 4.0,
    tonic_cutoff_hz: float = 0.05,
    order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose EDA into slow tonic baseline (SCL) and fast phasic response (SCR)."""
    eda_clean = np.nan_to_num(np.asarray(eda, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    tonic = butter_lowpass_filter(eda_clean, fs=fs, cutoff_hz=tonic_cutoff_hz, order=order)
    phasic = (eda_clean - tonic).astype(np.float32)
    return tonic, phasic


def compute_acc_magnitude(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> np.ndarray:
    """Compute Euclidean 3-axis accelerometer magnitude sqrt(x^2 + y^2 + z^2)."""
    x = np.asarray(acc_x, dtype=np.float32)
    y = np.asarray(acc_y, dtype=np.float32)
    z = np.asarray(acc_z, dtype=np.float32)
    return np.sqrt(x * x + y * y + z * z).astype(np.float32)


def zscore_per_channel(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Standardize a single subject's (T, C) feature matrix to zero mean and unit variance."""
    arr = np.asarray(matrix, dtype=np.float32)
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, keepdims=True)
    normed = (arr - mean) / (std + eps)
    return np.clip(normed, -10.0, 10.0).astype(np.float32)


def normalize_stress_labels(series: pd.Series) -> tuple[np.ndarray, dict[str, Any]]:
    """Inspect label column dynamically and map cleanly to binary stress (0=non_stress, 1=stress)."""
    raw_non_null = series.dropna()
    raw_unique = sorted({str(v) for v in raw_non_null.unique()})

    if pd.api.types.is_bool_dtype(series):
        y = series.fillna(False).astype(np.int64).to_numpy()
        rule = "boolean_direct"
    elif pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        uniq = np.unique(vals)
        if set(np.round(uniq, 6).tolist()).issubset({0.0, 1.0}):
            y = np.round(vals).astype(np.int64)
            rule = "binary_0_1_direct"
        elif uniq.min() >= 0.0 and uniq.max() <= 1.0:
            y = (vals >= 0.5).astype(np.int64)
            rule = "continuous_threshold_0.5"
        elif 2 in np.round(uniq).astype(int) and set(np.round(uniq).astype(int)).issubset({0, 1, 2, 3, 4, 5, 6, 7}):
            # Standard WESAD protocol code: 2 = TSST stress, others = non-stress.
            y = (np.round(vals).astype(np.int64) == 2).astype(np.int64)
            rule = "wesad_protocol_class_2_as_stress"
        else:
            y = (vals > np.median(vals)).astype(np.int64)
            rule = "median_threshold_fallback"
    else:
        lower = series.astype(str).str.strip().str.lower()
        y = lower.isin({"1", "true", "stress", "stressed", "yes", "2"}).astype(np.int64).to_numpy()
        rule = "string_stress_mapping"

    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    info = {
        "raw_unique_values": raw_unique[:20],
        "mapping_rule": rule,
        "num_classes": 2,
        "class_names": ["non_stress", "stress"],
        "row_counts": {"non_stress": neg_count, "stress": pos_count},
    }
    return y, info


def process_subject_signals(
    subject_df: pd.DataFrame,
    fs: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Butterworth DSP filtering, EDA decomposition, and per-subject z-score normalization.

    Returns:
        features: (T, 9) float32 per-subject z-scored DSP features
        raw_filtered: (T, 4) float32 physical-scale [bvp_bp, eda_tonic, eda_phasic, temp_lp]
    """
    acc_x = pd.to_numeric(subject_df["acc_x"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    acc_y = pd.to_numeric(subject_df["acc_y"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    acc_z = pd.to_numeric(subject_df["acc_z"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    bvp = pd.to_numeric(subject_df["bvp"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    eda = pd.to_numeric(subject_df["eda"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    temp = pd.to_numeric(subject_df["temp"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    acc_mag = compute_acc_magnitude(acc_x, acc_y, acc_z)
    bvp_bp = butter_bandpass_filter(bvp, fs=fs, low_hz=0.5, high_hz=1.8, order=2)
    eda_tonic, eda_phasic = decompose_eda(eda, fs=fs, tonic_cutoff_hz=0.05, order=2)
    eda_diff = np.gradient(eda_tonic + eda_phasic).astype(np.float32)
    temp_lp = butter_lowpass_filter(temp, fs=fs, cutoff_hz=0.1, order=2)

    stacked = np.stack(
        [acc_x, acc_y, acc_z, acc_mag, bvp_bp, eda_tonic, eda_phasic, eda_diff, temp_lp],
        axis=-1,
    ).astype(np.float32)
    features = zscore_per_channel(stacked)
    raw_filtered = np.stack([bvp_bp, eda_tonic, eda_phasic, temp_lp], axis=-1).astype(np.float32)
    return features, raw_filtered


def _is_array_window_dataframe(df: pd.DataFrame) -> bool:
    """Detect whether sensor columns store pre-windowed 1D signal arrays per row (Empatica E4 parquet)."""
    if len(df) == 0 or "bvp" not in df.columns:
        return False
    sample = df["bvp"].iloc[0]
    return isinstance(sample, (list, tuple, np.ndarray))


def _resample_1d(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a 1D biosignal array to target_len samples."""
    x = np.asarray(arr, dtype=np.float32).ravel()
    if x.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if x.size == target_len:
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    src_t = np.linspace(0.0, 1.0, x.size, dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(dst_t, src_t, x).astype(np.float32)


def _build_array_row_windows(
    df: pd.DataFrame,
    window_len: int = 64,
    label_col: str = "stress",
    user_col: str = "user",
    window_duration_s: float = 4.0,
) -> WindowBundle:
    """Process pre-windowed Empatica E4 array rows with Butterworth DSP and per-subject z-scoring."""
    y_all, label_info = normalize_stress_labels(df[label_col])
    fs_common = float(window_len) / max(window_duration_s, 1e-3)

    work_df = df[[*RAW_SENSOR_COLS, user_col]].copy()
    work_df["_binary_stress"] = y_all

    all_X: list[np.ndarray] = []
    all_raw: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_users: list[np.ndarray] = []

    for user_id, grp in work_df.groupby(user_col, sort=False):
        n_win = len(grp)
        if n_win == 0:
            continue
        subj_feats = np.empty((n_win, window_len, len(FEATURE_NAMES)), dtype=np.float32)
        subj_raw = np.empty((n_win, window_len, 4), dtype=np.float32)

        for i, row in enumerate(grp.itertuples(index=False)):
            ax = _resample_1d(row.acc_x, window_len)
            ay = _resample_1d(row.acc_y, window_len)
            az = _resample_1d(row.acc_z, window_len)
            bvp = _resample_1d(row.bvp, window_len)
            eda = _resample_1d(row.eda, window_len)
            temp = _resample_1d(row.temp, window_len)

            acc_mag = compute_acc_magnitude(ax, ay, az)
            bvp_bp = butter_bandpass_filter(bvp, fs=fs_common, low_hz=0.5, high_hz=3.5, order=2)
            eda_tonic, eda_phasic = decompose_eda(eda, fs=fs_common, tonic_cutoff_hz=0.25, order=2)
            eda_diff = np.gradient(eda).astype(np.float32)
            temp_lp = butter_lowpass_filter(temp, fs=fs_common, cutoff_hz=0.5, order=2)

            subj_feats[i] = np.stack(
                [ax, ay, az, acc_mag, bvp_bp, eda_tonic, eda_phasic, eda_diff, temp_lp],
                axis=-1,
            )
            subj_raw[i] = np.stack([bvp_bp, eda_tonic, eda_phasic, temp_lp], axis=-1)

        # Per-subject z-score normalization across all windows of this subject only
        flat = subj_feats.reshape(-1, len(FEATURE_NAMES))
        mean = flat.mean(axis=0, keepdims=True)
        std = flat.std(axis=0, keepdims=True)
        subj_feats = np.clip((subj_feats - mean[None, :, :]) / (std[None, :, :] + 1e-6), -10.0, 10.0)

        all_X.append(subj_feats.astype(np.float32))
        all_raw.append(subj_raw.astype(np.float32))
        all_y.append(grp["_binary_stress"].to_numpy(dtype=np.int64))
        all_users.append(np.full(n_win, str(user_id), dtype=object))

    if not all_X:
        raise ValueError("No valid Empatica E4 array windows could be constructed.")

    X_arr = np.concatenate(all_X, axis=0)
    raw_arr = np.concatenate(all_raw, axis=0)
    y_arr = np.concatenate(all_y, axis=0)
    u_arr = np.concatenate(all_users, axis=0)

    label_info["window_counts"] = {
        "non_stress": int(np.sum(y_arr == 0)),
        "stress": int(np.sum(y_arr == 1)),
    }
    label_info["input_format"] = "empatica_e4_array_windows_resampled_16hz"
    return WindowBundle(
        X=X_arr,
        y=y_arr,
        users=u_arr,
        raw_channels=raw_arr,
        feature_names=FEATURE_NAMES,
        label_info=label_info,
    )


def build_subject_windows(
    df: pd.DataFrame,
    window_len: int = 64,
    stride: int = 32,
    fs: float = 4.0,
    label_col: str = "stress",
    user_col: str = "user",
) -> WindowBundle:
    """Apply DSP strictly within each subject and slice temporal sliding windows."""
    if window_len <= 1 or stride <= 0:
        raise ValueError(f"Invalid window_len={window_len}, stride={stride}")

    missing = [c for c in (*RAW_SENSOR_COLS, label_col, user_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required WESAD columns: {missing}")

    if _is_array_window_dataframe(df):
        return _build_array_row_windows(df, window_len=window_len, label_col=label_col, user_col=user_col)

    y_all, label_info = normalize_stress_labels(df[label_col])
    work_df = df.copy()
    work_df["_binary_stress"] = y_all

    all_X: list[np.ndarray] = []
    all_raw: list[np.ndarray] = []
    all_y: list[int] = []
    all_users: list[str] = []

    for user_id, grp in work_df.groupby(user_col, sort=False):
        n_rows = len(grp)
        if n_rows < window_len:
            continue
        feats, raw_filt = process_subject_signals(grp, fs=fs)
        labels = grp["_binary_stress"].to_numpy(dtype=np.int64)
        uid_str = str(user_id)

        for start in range(0, n_rows - window_len + 1, stride):
            end = start + window_len
            all_X.append(feats[start:end])
            all_raw.append(raw_filt[start:end])
            win_label = 1 if float(np.mean(labels[start:end])) >= 0.5 else 0
            all_y.append(win_label)
            all_users.append(uid_str)

    if not all_X:
        raise ValueError("No valid subject windows could be constructed (check window_len vs subject lengths).")

    X_arr = np.stack(all_X, axis=0).astype(np.float32)
    raw_arr = np.stack(all_raw, axis=0).astype(np.float32)
    y_arr = np.asarray(all_y, dtype=np.int64)
    u_arr = np.asarray(all_users, dtype=object)

    label_info["window_counts"] = {
        "non_stress": int(np.sum(y_arr == 0)),
        "stress": int(np.sum(y_arr == 1)),
    }
    return WindowBundle(
        X=X_arr,
        y=y_arr,
        users=u_arr,
        raw_channels=raw_arr,
        feature_names=FEATURE_NAMES,
        label_info=label_info,
    )


def split_by_subject(
    bundle: WindowBundle,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> SubjectSplit:
    """Strictly subject-independent split: every user appears in exactly one split."""
    unique_users = sorted({str(u) for u in bundle.users.tolist()})
    n_users = len(unique_users)
    if n_users < 3:
        raise ValueError(f"Need at least 3 distinct subjects for train/val/test split, got {n_users}")

    n_val = max(1, int(round(n_users * val_frac)))
    n_test = max(1, int(round(n_users * (1.0 - train_frac - val_frac))))
    n_train = n_users - n_val - n_test
    if n_train < 1:
        n_train = 1
        rem = n_users - 1
        n_val = max(1, rem // 2)
        n_test = max(1, rem - n_val)

    rng = np.random.default_rng(seed)
    best_perm = list(unique_users)
    best_score = -1

    # Search deterministic permutations to ensure all splits contain both classes when feasible.
    for _ in range(32):
        perm = list(rng.permutation(unique_users))
        tr_set = set(perm[:n_train])
        va_set = set(perm[n_train : n_train + n_val])
        te_set = set(perm[n_train + n_val :])

        score = 0
        for grp_set in (tr_set, va_set, te_set):
            mask = np.isin(bundle.users, list(grp_set))
            classes = np.unique(bundle.y[mask])
            if len(classes) == 2:
                score += 1
        if score > best_score:
            best_score = score
            best_perm = perm
            if score == 3:
                break

    train_users = sorted(best_perm[:n_train])
    val_users = sorted(best_perm[n_train : n_train + n_val])
    test_users = sorted(best_perm[n_train + n_val :])

    assert set(train_users).isdisjoint(val_users), "Leakage detected between train and val users"
    assert set(train_users).isdisjoint(test_users), "Leakage detected between train and test users"
    assert set(val_users).isdisjoint(test_users), "Leakage detected between val and test users"

    tr_mask = np.isin(bundle.users, train_users)
    va_mask = np.isin(bundle.users, val_users)
    te_mask = np.isin(bundle.users, test_users)

    return SubjectSplit(
        X_train=bundle.X[tr_mask],
        y_train=bundle.y[tr_mask],
        users_train=bundle.users[tr_mask],
        X_val=bundle.X[va_mask],
        y_val=bundle.y[va_mask],
        users_val=bundle.users[va_mask],
        X_test=bundle.X[te_mask],
        y_test=bundle.y[te_mask],
        users_test=bundle.users[te_mask],
        raw_test=bundle.raw_channels[te_mask],
        train_user_ids=train_users,
        val_user_ids=val_users,
        test_user_ids=test_users,
    )


def extract_window_summary_features(X: np.ndarray) -> np.ndarray:
    """Compute per-channel summary statistics (mean, std, min, max, linear slope) for classical ML.

    Args:
        X: (N, T, C) float array.
    Returns:
        (N, 5 * C) float32 tabular matrix.
    """
    arr = np.asarray(X, dtype=np.float32)
    n, t, c = arr.shape
    means = np.mean(arr, axis=1)
    stds = np.std(arr, axis=1)
    mins = np.min(arr, axis=1)
    maxs = np.max(arr, axis=1)

    t_axis = np.linspace(-1.0, 1.0, t, dtype=np.float32)
    t_centered = t_axis - np.mean(t_axis)
    denom = float(np.sum(t_centered * t_centered)) + 1e-8
    arr_centered = arr - means[:, None, :]
    slopes = np.sum(arr_centered * t_centered[None, :, None], axis=1) / denom

    feats = np.concatenate([means, stds, mins, maxs, slopes], axis=1)
    assert feats.shape == (n, 5 * c)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
