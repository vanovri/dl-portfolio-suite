"""Build flagship 2x3 visual proof collage and print exact benchmark metrics from MASTER_BENCHMARK_LEDGER.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


PANELS = [
    ("10 | IceCube 3D Neutrino Transformer", "10_icecube_neutrino_3d/reports/neutrino_reconstruction.png"),
    ("09 | Landslide4Sense 14-Band AI-SDSS", "09_landslide_ai_sdss/reports/landslide_sdss_map.png"),
    ("12 | Hybrid 4-Qubit Quantum-DL (BBBP + Credit)", "12_hybrid_quantum_dl/reports/quantum_drug_finance.png"),
    ("07 | Subject-Independent WESAD Bi-LSTM", "07_wesad_stress_bilstm/reports/stress_signals_attention.png"),
    ("03 | Industrial Corrosion ResNet18 U-Net", "03_corrosion_unet_segmentation/reports/corrosion_predictions.png"),
    ("01 | Eye Disease RetinaMNIST + Grad-CAM", "01_eye_disease_gradcam/reports/gradcam_gallery.png"),
]


def build_collage(root: Path) -> Path:
    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "flagship_visual_collage.png"

    fig, axes = plt.subplots(2, 3, figsize=(21, 11.5), facecolor="#0f172a")
    fig.suptitle(
        "13-Project Cross-Domain Deep Learning Portfolio Suite -- Visual Proof Collage (GTX 1650 Ti 4 GB VRAM)",
        color="#f8fafc",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for ax, (title, rel_img) in zip(axes.ravel(), PANELS):
        ax.set_facecolor("#1e293b")
        img_path = root / rel_img
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        ax.set_title(title, color="#38bdf8", fontsize=11.5, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
            spine.set_linewidth(1.2)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(out_path, dpi=140, facecolor="#0f172a")
    plt.close(fig)
    return out_path


def print_ledger_summary(root: Path) -> None:
    ledger = json.loads((root / "MASTER_BENCHMARK_LEDGER.json").read_text(encoding="utf-8"))
    for entry in ledger["projects"]:
        proj = entry["project"]
        m = entry["metrics"]
        print(f"=== {proj} ===")
        print("  visuals:", entry["visual_proofs"])
        for k in (
            "dataset_type", "dataset_name", "dataset_provenance", "datasets",
            "hardware", "runtime", "training", "test_metrics", "test_metrics_bilstm",
            "test_metrics_indobert", "indobert_vs_tfidf_gain", "ablation",
            "models", "detection", "segmentation", "tracks", "tasks",
            "transformer_test", "linefit_baseline", "variants", "summary",
            "cervical_biopsy_morphometry", "api_benchmark", "quantum_verification",
            "bilstm_vs_best_classical_delta", "splits",
        ):
            if k in m:
                val = json.dumps(m[k], ensure_ascii=True)
                print(f"  {k}: {val[:280]}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    collage = build_collage(repo_root)
    print(f"Saved collage -> {collage.relative_to(repo_root).as_posix()}")
    print_ledger_summary(repo_root)
