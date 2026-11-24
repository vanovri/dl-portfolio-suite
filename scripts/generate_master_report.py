"""Aggregate all 13 project reports/metrics.json files into MASTER_BENCHMARK_LEDGER.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECTS = [
    "01_eye_disease_gradcam",
    "02_cervical_cancer_segmentation",
    "03_corrosion_unet_segmentation",
    "04_leaf_disease_transfer_learning",
    "05_indo_car_classifier_marketplace",
    "06_yolov8_detection_segmentation",
    "07_wesad_stress_bilstm",
    "08_indobert_sentiment_nlp",
    "09_landslide_ai_sdss",
    "10_icecube_neutrino_3d",
    "11_multimodal_image_audio_fusion",
    "12_hybrid_quantum_dl",
    "13_humanities_nusantara_dl",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    entries: list[dict[str, Any]] = []
    missing: list[str] = []

    for proj in PROJECTS:
        report_path = root / proj / "reports" / "metrics.json"
        if not report_path.exists():
            missing.append(proj)
            continue
        data = json.loads(report_path.read_text(encoding="utf-8"))
        pngs = sorted(p.name for p in (root / proj / "reports").glob("*.png"))
        entries.append(
            {
                "project": proj,
                "report_path": f"{proj}/reports/metrics.json",
                "visual_proofs": [f"{proj}/reports/{name}" for name in pngs],
                "metrics": data,
            }
        )

    ledger = {
        "suite": "13-Project Deep Learning Portfolio Suite (4 GB VRAM CUDA)",
        "total_projects": len(PROJECTS),
        "completed_projects": len(entries),
        "missing_projects": missing,
        "projects": entries,
    }

    out_path = root / "MASTER_BENCHMARK_LEDGER.json"
    out_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.name}: {len(entries)}/{len(PROJECTS)} completed, {len(missing)} missing.")


if __name__ == "__main__":
    main()
