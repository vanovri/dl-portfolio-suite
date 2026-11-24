"""Pulse packing, deterministic physics dataset cache, and drop-in Kaggle IceCube parquet loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.detector import (
    azimuth_zenith_to_vec,
    build_icecube_5160_geometry,
    simulate_single_event,
)


def pack_event_pulses(
    sensor_ids: np.ndarray,
    times: np.ndarray,
    charges: np.ndarray,
    auxiliary: np.ndarray,
    dom_xyz: np.ndarray,
    max_pulses: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Select top-K informative pulses and pack normalized 6D features + validity mask.

    Feature channels per pulse:
        0: x / 500.0
        1: y / 500.0
        2: z / 500.0
        3: clip((time - t_ref) / 1000.0, -5.0, 5.0)
        4: log1p(charge)
        5: auxiliary (0.0 or 1.0)
    """
    n = int(len(sensor_ids))
    feats = np.zeros((max_pulses, 6), dtype=np.float32)
    mask = np.zeros((max_pulses,), dtype=bool)
    if n == 0:
        return feats, mask

    sids = np.asarray(sensor_ids, dtype=np.int64)
    ts = np.asarray(times, dtype=np.float32)
    qs = np.maximum(np.asarray(charges, dtype=np.float32), 0.0)
    aux = np.asarray(auxiliary, dtype=bool)

    if n > max_pulses:
        # Prioritize non-auxiliary pulses first, then highest photoelectron charge
        priority = np.lexsort((-qs, aux.astype(np.int8)))
        keep = priority[:max_pulses]
    else:
        keep = np.arange(n, dtype=np.int64)

    # Sort selected pulses chronologically
    keep = keep[np.argsort(ts[keep], kind="mergesort")]
    k = int(keep.size)

    sel_sids = np.clip(sids[keep], 0, dom_xyz.shape[0] - 1)
    sel_xyz = dom_xyz[sel_sids]
    sel_t = ts[keep]
    sel_q = qs[keep]
    sel_aux = aux[keep]

    non_aux_t = sel_t[~sel_aux]
    t_ref = float(np.median(non_aux_t)) if non_aux_t.size > 0 else float(np.median(sel_t))

    feats[:k, 0] = sel_xyz[:, 0] / 500.0
    feats[:k, 1] = sel_xyz[:, 1] / 500.0
    feats[:k, 2] = sel_xyz[:, 2] / 500.0
    feats[:k, 3] = np.clip((sel_t - t_ref) / 1000.0, -5.0, 5.0)
    feats[:k, 4] = np.log1p(sel_q)
    feats[:k, 5] = sel_aux.astype(np.float32)
    mask[:k] = True
    return feats, mask


def load_kaggle_icecube_batch(
    batch_parquet: str | Path,
    meta_parquet: str | Path,
    geometry_csv: str | Path,
    max_pulses: int = 96,
    max_events: int | None = None,
) -> dict[str, np.ndarray]:
    """Drop-in Kaggle IceCube competition loader reading official batch/meta parquet + geometry CSV.

    Supports official Kaggle schema:
        - geometry_csv: columns (sensor_id, x, y, z) or index=sensor_id with (x, y, z)
        - meta_parquet: columns (batch_id, event_id, first_pulse_index, last_pulse_index, azimuth, zenith)
        - batch_parquet: columns (sensor_id, time, charge, auxiliary) with index or column event_id
    """
    geom_df = pd.read_csv(geometry_csv)
    if "sensor_id" in geom_df.columns:
        geom_df = geom_df.sort_values("sensor_id")
    dom_xyz = geom_df[["x", "y", "z"]].to_numpy(dtype=np.float32)

    meta_df = pd.read_parquet(meta_parquet)
    batch_df = pd.read_parquet(batch_parquet)
    if max_events is not None:
        meta_df = meta_df.iloc[:max_events]

    sids_all = batch_df["sensor_id"].to_numpy(dtype=np.int32)
    times_all = batch_df["time"].to_numpy(dtype=np.float32)
    charges_all = batch_df["charge"].to_numpy(dtype=np.float32)
    aux_all = batch_df["auxiliary"].to_numpy(dtype=bool)

    has_slice_indices = "first_pulse_index" in meta_df.columns and "last_pulse_index" in meta_df.columns
    if not has_slice_indices and "event_id" not in batch_df.columns:
        batch_df = batch_df.reset_index()

    n_events = len(meta_df)
    X = np.zeros((n_events, max_pulses, 6), dtype=np.float32)
    masks = np.zeros((n_events, max_pulses), dtype=bool)
    azimuths = meta_df["azimuth"].to_numpy(dtype=np.float32)
    zeniths = meta_df["zenith"].to_numpy(dtype=np.float32)
    targets = azimuth_zenith_to_vec(azimuths, zeniths)

    if has_slice_indices:
        first_idx = meta_df["first_pulse_index"].to_numpy(dtype=np.int64)
        last_idx = meta_df["last_pulse_index"].to_numpy(dtype=np.int64)
        for i in range(n_events):
            lo, hi = int(first_idx[i]), int(last_idx[i]) + 1
            X[i], masks[i] = pack_event_pulses(
                sids_all[lo:hi], times_all[lo:hi], charges_all[lo:hi], aux_all[lo:hi], dom_xyz, max_pulses
            )
    else:
        grouped = {int(eid): grp for eid, grp in batch_df.groupby("event_id", sort=False)}
        for i, eid in enumerate(meta_df["event_id"].to_numpy(dtype=np.int64)):
            grp = grouped.get(int(eid))
            if grp is None:
                continue
            X[i], masks[i] = pack_event_pulses(
                grp["sensor_id"].to_numpy(),
                grp["time"].to_numpy(),
                grp["charge"].to_numpy(),
                grp["auxiliary"].to_numpy(),
                dom_xyz,
                max_pulses,
            )

    return {
        "X": X,
        "masks": masks,
        "targets": targets,
        "azimuth": azimuths,
        "zenith": zeniths,
    }


def _simulate_split(
    n_events: int,
    dom_xyz: np.ndarray,
    max_pulses: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.zeros((n_events, max_pulses, 6), dtype=np.float32)
    masks = np.zeros((n_events, max_pulses), dtype=bool)
    targets = np.zeros((n_events, 3), dtype=np.float32)
    azimuths = np.zeros((n_events,), dtype=np.float32)
    zeniths = np.zeros((n_events,), dtype=np.float32)
    vertices = np.zeros((n_events, 3), dtype=np.float32)

    for i in range(n_events):
        ev = simulate_single_event(dom_xyz, rng, add_noise=True, scattering_scale=1.0)
        X[i], masks[i] = pack_event_pulses(
            ev["sensor_id"], ev["time"], ev["charge"], ev["auxiliary"], dom_xyz, max_pulses=max_pulses
        )
        targets[i] = ev["direction_vec"]
        azimuths[i] = ev["azimuth"]
        zeniths[i] = ev["zenith"]
        vertices[i] = ev["vertex"]

    return {
        "X": X,
        "masks": masks,
        "targets": targets,
        "azimuth": azimuths,
        "zenith": zeniths,
        "vertices": vertices,
    }


def prepare_icecube_datasets(
    data_dir: Path,
    n_train: int = 10000,
    n_val: int = 1500,
    n_test: int = 2000,
    max_pulses: int = 96,
    seed: int = 42,
    force_regenerate: bool = False,
) -> dict[str, Any]:
    """Generate and cache deterministic physics-calibrated IceCube 5,160-DOM train/val/test sets."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    geom_path = data_dir / "sensor_geometry.csv"
    cache_path = data_dir / f"icecube_sim_tr{n_train}_va{n_val}_te{n_test}_k{max_pulses}_s{seed}.npz"

    geom_df = build_icecube_5160_geometry()
    if not geom_path.exists() or force_regenerate:
        geom_df.to_csv(geom_path, index=False)
    dom_xyz = geom_df[["x", "y", "z"]].to_numpy(dtype=np.float32)

    if cache_path.exists() and not force_regenerate:
        npz = np.load(cache_path, allow_pickle=False)
        meta = json.loads(str(npz["meta_json"]))
        meta["source"] = "local_npz_cache"
        return {
            "geometry_df": geom_df,
            "dom_xyz": dom_xyz,
            "train": {k: npz[f"train_{k}"] for k in ("X", "masks", "targets", "azimuth", "zenith")},
            "val": {k: npz[f"val_{k}"] for k in ("X", "masks", "targets", "azimuth", "zenith")},
            "test": {k: npz[f"test_{k}"] for k in ("X", "masks", "targets", "azimuth", "zenith", "vertices")},
            "meta": meta,
        }

    train_split = _simulate_split(n_train, dom_xyz, max_pulses, seed=seed)
    val_split = _simulate_split(n_val, dom_xyz, max_pulses, seed=seed + 1000)
    test_split = _simulate_split(n_test, dom_xyz, max_pulses, seed=seed + 2000)

    meta = {
        "dataset_type": "physics_calibrated_icecube_5160dom",
        "source": "relativistic_cherenkov_simulator",
        "num_doms": int(len(geom_df)),
        "standard_strings": 78,
        "deepcore_strings": 8,
        "max_pulses": int(max_pulses),
        "split_sizes": {"train": n_train, "val": n_val, "test": n_test},
        "kaggle_dropin_loader": "src.dataset.load_kaggle_icecube_batch",
    }

    np.savez_compressed(
        cache_path,
        train_X=train_split["X"],
        train_masks=train_split["masks"],
        train_targets=train_split["targets"],
        train_azimuth=train_split["azimuth"],
        train_zenith=train_split["zenith"],
        val_X=val_split["X"],
        val_masks=val_split["masks"],
        val_targets=val_split["targets"],
        val_azimuth=val_split["azimuth"],
        val_zenith=val_split["zenith"],
        test_X=test_split["X"],
        test_masks=test_split["masks"],
        test_targets=test_split["targets"],
        test_azimuth=test_split["azimuth"],
        test_zenith=test_split["zenith"],
        test_vertices=test_split["vertices"],
        meta_json=json.dumps(meta),
    )

    return {
        "geometry_df": geom_df,
        "dom_xyz": dom_xyz,
        "train": train_split,
        "val": val_split,
        "test": test_split,
        "meta": meta,
    }


class IceCubePulseDataset(Dataset):
    """PyTorch Dataset yielding (pulse_features, valid_mask, target_unit_vector)."""

    def __init__(self, X: np.ndarray, masks: np.ndarray, targets: np.ndarray) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.masks = torch.as_tensor(masks, dtype=torch.bool)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.masks[idx], self.targets[idx]
