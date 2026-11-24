"""Deterministic offline CPU unit tests for 10_icecube_neutrino_3d."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(PROJECT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.dataset import IceCubePulseDataset, load_kaggle_icecube_batch, pack_event_pulses
from src.detector import (
    azimuth_zenith_to_vec,
    build_icecube_5160_geometry,
    simulate_single_event,
    vec_to_azimuth_zenith,
)
from src.losses import VonMisesFisher3DLoss, angular_error_radians, compute_angular_metrics
from src.model import IceCubePulseTransformer
from src.physics_baseline import linefit_predict_batch, linefit_reconstruct_event
from src.viz import plot_neutrino_reconstruction


def test_exact_5160_dom_hexagonal_and_deepcore_geometry():
    geom = build_icecube_5160_geometry()
    assert len(geom) == 5160
    assert geom["sensor_id"].nunique() == 5160
    assert int(geom["is_deepcore"].sum()) == 8 * 60
    assert int((~geom["is_deepcore"]).sum()) == 78 * 60
    assert geom["string_id"].nunique() == 86


def test_spherical_coordinate_roundtrip():
    rng = np.random.default_rng(0)
    az = rng.uniform(0.0, 2.0 * np.pi, size=50).astype(np.float32)
    ze = rng.uniform(0.05, np.pi - 0.05, size=50).astype(np.float32)
    vecs = azimuth_zenith_to_vec(az, ze)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=-1), np.ones(50), atol=1e-5)
    az2, ze2 = vec_to_azimuth_zenith(vecs)
    np.testing.assert_allclose(az2, az, atol=1e-4)
    np.testing.assert_allclose(ze2, ze, atol=1e-4)


def test_cherenkov_simulator_produces_exact_kaggle_schema():
    geom = build_icecube_5160_geometry()
    dom_xyz = geom[["x", "y", "z"]].to_numpy(dtype=np.float32)
    ev = simulate_single_event(dom_xyz, np.random.default_rng(12), add_noise=True)
    for key in ("sensor_id", "time", "charge", "auxiliary", "azimuth", "zenith", "direction_vec"):
        assert key in ev
    assert len(ev["sensor_id"]) >= 12
    assert ev["auxiliary"].dtype == bool
    assert bool(ev["auxiliary"].any()) and bool((~ev["auxiliary"]).any())
    assert np.all(np.diff(ev["time"]) >= 0.0)


def test_pack_event_pulses_prioritizes_non_aux_and_shapes():
    geom = build_icecube_5160_geometry()
    dom_xyz = geom[["x", "y", "z"]].to_numpy(dtype=np.float32)
    ev = simulate_single_event(dom_xyz, np.random.default_rng(3), add_noise=True)
    feats, mask = pack_event_pulses(
        ev["sensor_id"], ev["time"], ev["charge"], ev["auxiliary"], dom_xyz, max_pulses=32
    )
    assert feats.shape == (32, 6)
    assert mask.shape == (32,)
    assert int(mask.sum()) == min(len(ev["sensor_id"]), 32)
    assert np.isfinite(feats).all()


def test_dropin_kaggle_icecube_batch_loader(tmp_path: Path):
    geom = build_icecube_5160_geometry()
    dom_xyz = geom[["x", "y", "z"]].to_numpy(dtype=np.float32)
    geom_csv = tmp_path / "sensor_geometry.csv"
    geom[["sensor_id", "x", "y", "z"]].to_csv(geom_csv, index=False)

    rng = np.random.default_rng(9)
    pulse_frames = []
    meta_rows = []
    cursor = 0
    for eid in range(5):
        ev = simulate_single_event(dom_xyz, rng, add_noise=True)
        n_p = len(ev["sensor_id"])
        pulse_frames.append(
            pd.DataFrame(
                {
                    "event_id": np.full(n_p, eid, dtype=np.int64),
                    "sensor_id": ev["sensor_id"],
                    "time": ev["time"],
                    "charge": ev["charge"],
                    "auxiliary": ev["auxiliary"],
                }
            )
        )
        meta_rows.append(
            {
                "batch_id": 1,
                "event_id": eid,
                "first_pulse_index": cursor,
                "last_pulse_index": cursor + n_p - 1,
                "azimuth": ev["azimuth"],
                "zenith": ev["zenith"],
            }
        )
        cursor += n_p

    batch_pq = tmp_path / "batch_1.parquet"
    meta_pq = tmp_path / "train_meta.parquet"
    pd.concat(pulse_frames, ignore_index=True).to_parquet(batch_pq, index=False)
    pd.DataFrame(meta_rows).to_parquet(meta_pq, index=False)

    loaded = load_kaggle_icecube_batch(batch_pq, meta_pq, geom_csv, max_pulses=64)
    assert loaded["X"].shape == (5, 64, 6)
    assert loaded["masks"].shape == (5, 64)
    assert loaded["targets"].shape == (5, 3)


def test_classical_linefit_recovers_clean_cherenkov_tracks():
    geom = build_icecube_5160_geometry()
    dom_xyz = geom[["x", "y", "z"]].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(42)

    X_list, m_list, t_list = [], [], []
    for _ in range(25):
        ev = simulate_single_event(dom_xyz, rng, add_noise=False, scattering_scale=0.05)
        f, m = pack_event_pulses(ev["sensor_id"], ev["time"], ev["charge"], ev["auxiliary"], dom_xyz, max_pulses=96)
        X_list.append(f)
        m_list.append(m)
        t_list.append(ev["direction_vec"])

    preds = linefit_predict_batch(np.stack(X_list), np.stack(m_list))
    metrics = compute_angular_metrics(preds, np.stack(t_list))
    assert metrics["median_angular_error_deg"] < 20.0


def test_linefit_handles_degenerate_single_pulse():
    f = np.zeros((16, 6), dtype=np.float32)
    m = np.zeros((16,), dtype=bool)
    m[0] = True
    u = linefit_reconstruct_event(f, m)
    assert u.shape == (3,)
    assert abs(float(np.linalg.norm(u)) - 1.0) < 1e-5


def test_vmf_3d_loss_stability_and_ordering():
    loss_fn = VonMisesFisher3DLoss(max_kappa=100.0)
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    aligned = target.clone()
    opposite = -target.clone()

    for k_val in (1e-5, 1e-3, 1.0, 20.0, 100.0):
        k = torch.tensor([k_val, k_val], requires_grad=True)
        l_good = loss_fn(aligned, k, target)
        l_bad = loss_fn(opposite, k, target)
        assert torch.isfinite(l_good) and torch.isfinite(l_bad)
        assert float(l_good) < float(l_bad)
        l_good.backward()
        assert k.grad is not None and torch.isfinite(k.grad).all()


def test_angular_error_exact_values():
    u1 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    u2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    errs = angular_error_radians(u1, u2)
    np.testing.assert_allclose(errs, np.array([0.0, np.pi / 2.0, np.pi]), atol=1e-6)


def test_pulse_transformer_forward_backward_and_padding_invariance():
    torch.manual_seed(0)
    model = IceCubePulseTransformer(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    model.eval()

    x1 = torch.randn(3, 32, 6)
    mask = torch.zeros(3, 32, dtype=torch.bool)
    mask[:, :12] = True
    x2 = x1.clone()
    x2[:, 12:, :] = 999.0  # garbage in padded tail must not affect outputs

    u1, k1 = model(x1, mask)
    u2, k2 = model(x2, mask)
    torch.testing.assert_close(u1, u2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k1, k2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.norm(u1, dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5)
    assert (k1 > 0).all()

    model.train()
    u_tr, k_tr = model(x1, mask)
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    loss = VonMisesFisher3DLoss()(u_tr, k_tr, target)
    loss.backward()
    grad_sum = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert grad_sum > 0.0


def test_icecube_pulse_dataset_shapes():
    ds = IceCubePulseDataset(
        np.zeros((8, 32, 6), dtype=np.float32),
        np.ones((8, 32), dtype=bool),
        np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (8, 1)),
    )
    assert len(ds) == 8
    x, m, y = ds[0]
    assert x.shape == (32, 6) and m.dtype == torch.bool and y.shape == (3,)


def test_plot_neutrino_reconstruction_writes_png(tmp_path: Path):
    geom = build_icecube_5160_geometry()
    dom_xyz = geom[["x", "y", "z"]].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(7)
    ev = simulate_single_event(dom_xyz, rng, add_noise=True)
    feats, mask = pack_event_pulses(ev["sensor_id"], ev["time"], ev["charge"], ev["auxiliary"], dom_xyz, max_pulses=64)

    true_vecs = azimuth_zenith_to_vec(rng.uniform(0, 2 * np.pi, 30), np.arccos(rng.uniform(-1, 1, 30)))
    pred_vecs = true_vecs.copy()
    out = plot_neutrino_reconstruction(
        out_path=tmp_path / "neutrino_reconstruction.png",
        event_pulses=feats,
        event_mask=mask,
        event_vertex=ev["vertex"],
        true_vec=ev["direction_vec"],
        pred_vec=ev["direction_vec"],
        linefit_vec=linefit_reconstruct_event(feats, mask),
        dom_xyz=dom_xyz,
        transformer_errors_deg=rng.exponential(8.0, size=100),
        linefit_errors_deg=rng.exponential(22.0, size=100),
        test_true_vecs=true_vecs,
        test_pred_vecs=pred_vecs,
    )
    assert out.exists() and out.stat().st_size > 1000
