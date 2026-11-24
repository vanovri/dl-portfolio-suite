"""3D IceCube DOM scatter, angular error histogram, and Mollweide sky map visualizations."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.cuda_utils import configure_plot_style
from src.detector import vec_to_azimuth_zenith


def plot_neutrino_reconstruction(
    out_path: Path,
    event_pulses: np.ndarray,
    event_mask: np.ndarray,
    event_vertex: np.ndarray,
    true_vec: np.ndarray,
    pred_vec: np.ndarray,
    linefit_vec: np.ndarray,
    dom_xyz: np.ndarray,
    transformer_errors_deg: np.ndarray,
    linefit_errors_deg: np.ndarray,
    test_true_vecs: np.ndarray,
    test_pred_vecs: np.ndarray,
) -> Path:
    """Render 3-panel IceCube figure: 3D DOM event + vectors, Error histogram, and Mollweide sky map."""
    configure_plot_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(17, 5.6))

    # Panel 1: 3D IceCube DOM scatter + True vs Transformer vs LineFit track vectors
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax3d.set_facecolor("#0f172a")

    # Subsample full 5,160 DOM lattice lightly to frame the cubic-kilometer volume
    bg = dom_xyz[::12]
    ax3d.scatter(bg[:, 0], bg[:, 1], bg[:, 2], s=1.5, c="#334155", alpha=0.22)

    valid = event_pulses[event_mask]
    non_aux = valid[valid[:, 5] < 0.5]
    aux = valid[valid[:, 5] >= 0.5]

    if aux.shape[0] > 0:
        ax3d.scatter(
            aux[:, 0] * 500.0,
            aux[:, 1] * 500.0,
            aux[:, 2] * 500.0,
            s=10,
            c="#64748b",
            alpha=0.35,
            marker="x",
            label="Aux noise",
        )
    if non_aux.shape[0] > 0:
        xyz = non_aux[:, 0:3] * 500.0
        rel_t = non_aux[:, 3] * 1000.0
        sizes = 18.0 + 35.0 * non_aux[:, 4]
        sc = ax3d.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=rel_t,
            s=sizes,
            cmap="plasma",
            alpha=0.92,
            edgecolors="#0f172a",
            linewidths=0.3,
        )
        cbar = fig.colorbar(sc, ax=ax3d, shrink=0.55, pad=0.08)
        cbar.set_label("Rel. Time (ns)", fontsize=8)

    origin = np.asarray(event_vertex, dtype=np.float32)
    vec_len = 360.0
    ax3d.quiver(
        origin[0], origin[1], origin[2],
        true_vec[0] * vec_len, true_vec[1] * vec_len, true_vec[2] * vec_len,
        color="#38bdf8", linewidth=2.6, arrow_length_ratio=0.18, label="True Track",
    )
    ax3d.quiver(
        origin[0], origin[1], origin[2],
        pred_vec[0] * vec_len, pred_vec[1] * vec_len, pred_vec[2] * vec_len,
        color="#34d399", linewidth=2.2, arrow_length_ratio=0.18, label="Transformer",
    )
    ax3d.quiver(
        origin[0], origin[1], origin[2],
        linefit_vec[0] * vec_len, linefit_vec[1] * vec_len, linefit_vec[2] * vec_len,
        color="#f59e0b", linewidth=1.8, linestyle="--", arrow_length_ratio=0.18, label="LineFit",
    )
    ax3d.set_xlim(-550, 550)
    ax3d.set_ylim(-550, 550)
    ax3d.set_zlim(-550, 550)
    ax3d.set_xlabel("X (m)", fontsize=8)
    ax3d.set_ylabel("Y (m)", fontsize=8)
    ax3d.set_zlabel("Z (m)", fontsize=8)
    ax3d.set_title("3D IceCube 5,160-DOM Event & Tracks")
    ax3d.legend(loc="upper left", fontsize=7.5)

    # Panel 2: Angular Error Distribution Transformer vs LineFit
    ax_hist = fig.add_subplot(1, 3, 2)
    bins = np.linspace(0.0, 90.0, 36)
    ax_hist.hist(
        linefit_errors_deg,
        bins=bins,
        alpha=0.50,
        color="#f59e0b",
        label=f"LineFit (med={np.median(linefit_errors_deg):.1f} deg)",
        density=True,
    )
    ax_hist.hist(
        transformer_errors_deg,
        bins=bins,
        alpha=0.72,
        color="#38bdf8",
        label=f"PulseTransformer (med={np.median(transformer_errors_deg):.1f} deg)",
        density=True,
    )
    ax_hist.axvline(float(np.mean(transformer_errors_deg)), color="#38bdf8", ls="--", lw=1.8, label=f"Trans. Mean={np.mean(transformer_errors_deg):.1f} deg")
    ax_hist.axvline(float(np.mean(linefit_errors_deg)), color="#f59e0b", ls=":", lw=1.8, label=f"LineFit Mean={np.mean(linefit_errors_deg):.1f} deg")
    ax_hist.set_xlabel("Angular Reconstruction Error (deg)")
    ax_hist.set_ylabel("Probability Density")
    ax_hist.set_title("Angular Error: Pulse Transformer vs LineFit")
    ax_hist.legend(loc="upper right", fontsize=8)
    ax_hist.grid(True, alpha=0.35)

    # Panel 3: Mollweide Sky Map
    ax_sky = fig.add_subplot(1, 3, 3, projection="mollweide")
    n_show = min(60, len(test_true_vecs))
    t_az, t_ze = vec_to_azimuth_zenith(test_true_vecs[:n_show])
    p_az, p_ze = vec_to_azimuth_zenith(test_pred_vecs[:n_show])

    # Mollweide expects longitude in [-pi, pi] and latitude in [-pi/2, pi/2]
    t_lon = ((t_az + np.pi) % (2.0 * np.pi)) - np.pi
    t_lat = (np.pi / 2.0) - t_ze
    p_lon = ((p_az + np.pi) % (2.0 * np.pi)) - np.pi
    p_lat = (np.pi / 2.0) - p_ze

    for i in range(n_show):
        if abs(t_lon[i] - p_lon[i]) < np.pi:
            ax_sky.plot([t_lon[i], p_lon[i]], [t_lat[i], p_lat[i]], color="#94a3b8", lw=0.7, alpha=0.55)
    ax_sky.scatter(t_lon, t_lat, s=22, color="#38bdf8", alpha=0.85, label="True Direction")
    ax_sky.scatter(p_lon, p_lat, s=18, color="#34d399", marker="^", alpha=0.85, label="Predicted")
    ax_sky.set_title("Neutrino Sky Map (Mollweide Projection)")
    ax_sky.grid(True, alpha=0.35)
    ax_sky.legend(loc="lower right", fontsize=8)

    fig.suptitle("IceCube 3D Neutrino Direction Reconstruction (vMF Pulse Transformer)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
