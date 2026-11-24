# 13-Project Cross-Domain Deep Learning Portfolio Suite (4 GB VRAM CUDA)

[![CI - Hygiene & 13-Domain Test Suite](https://github.com/vanovri/dl-portfolio-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/vanovri/dl-portfolio-suite/actions/workflows/ci.yml)
![Projects 13/13 Verified](https://img.shields.io/badge/projects-13%2F13%20VERIFIED-0b6b46)
![Pytest 149/149 PASS](https://img.shields.io/badge/pytest-149%2F149%20PASS-0b6b46)
![Hardware GTX 1650 Ti 4GB](https://img.shields.io/badge/GPU-GTX%201650%20Ti%204GB%20VRAM-0b5fc4)
![Real Public Datasets 12/13](https://img.shields.io/badge/datasets-12%2F13%20Real%20Public%20%2B%201%20Physics--Calibrated-1d4ed8)
![License MIT](https://img.shields.io/badge/license-MIT-blue)

End-to-end verified implementation of **13 cross-domain Deep Learning systems** covering Medical Computer Vision, Industrial Segmentation, Edge Transfer Learning, Real-Time Object Detection & Instance Segmentation, Physiological Biosignal DSP, Indonesian Transformer NLP, 14-Band Geospatial AI-SDSS, 3D Astroparticle Physics, Multimodal Audio-Visual Fusion, Hybrid Quantum-Classical ML, and Digital Humanities / Nusantara Literary Stylometry.

Every model was trained, evaluated on strictly held-out test splits, profiled for peak VRAM, visually verified, and regression-tested locally on consumer laptop hardware (**Intel Core i5-10300H, 15.8 GB RAM, NVIDIA GeForce GTX 1650 Ti 4 GB VRAM, Turing SM 7.5, CUDA 12.4**). Every metric in this README comes directly from checked-in reproducible JSON reports aggregated in [`MASTER_BENCHMARK_LEDGER.json`](./MASTER_BENCHMARK_LEDGER.json).

![13-Project Flagship Visual Proof Collage](./reports/flagship_visual_collage.png)

---

## 4 Flagship Anchor Highlights (Beyond Generic Image Classification)

Rather than treating these 13 systems as isolated classroom notebooks, the suite is engineered as a **constrained-VRAM cross-domain tensor study** anchored by four non-standard deep learning architectures:

1. **IceCube 3D Neutrino Direction Reconstruction ([`10_icecube_neutrino_3d/`](./10_icecube_neutrino_3d/))**
   - **Tensor & Physics Problem:** Irregular spatio-temporal 3D sensor pulse sequences `(x, y, z, t, log_charge, auxiliary)` over the **5,160-DOM South Pole hexagonal array** (78 standard strings + 8 DeepCore strings), avoiding `0/2pi` azimuth wrap-around discontinuities.
   - **Architecture & Result:** 4-layer Pre-Norm Pulse Transformer (`552k` params) with 4D Fourier coordinate encoding and **3D von Mises-Fisher (vMF) spherical loss** achieves **`5.65 deg` (`0.0987 rad`) Mean Angular Error** (`4.54 deg` median, `6.37 deg` 68% containment), slashing error by **`-48.64%`** against the analytical **Charge-Weighted LineFit** physics baseline (`11.00 deg`) on **`148.0 MB` peak VRAM**. Includes a verified drop-in Kaggle Parquet batch loader (`load_kaggle_icecube_batch`).

2. **Landslide4Sense 14-Band Multispectral AI-SDSS ([`09_landslide_ai_sdss/`](./09_landslide_ai_sdss/))**
   - **Tensor & Disaster Problem:** 14-channel Sentinel-2 optical/SWIR + ALOS PALSAR DEM & slope HDF5 satellite tiles (`128x128x14`) with severe foreground class imbalance.
   - **Architecture & Result:** `SpatialResUNet` (`FocalDiceLoss`, FP32 Turing-safe, `batch_size=8`) achieves **`0.9771` Pixel ROC-AUC**, **`0.6371` Pixel F1**, **`0.4675` IoU**, and **`67.83%` Landslide Recall** on official held-out `Landslide4sense` test shards in **`68.5s`** and **`592.3 MB` peak VRAM**, paired with a 4-tier operational AI-SDSS FastAPI service exporting GeoJSON evacuation hazard polygons.

3. **Differentiable 4-Qubit Hybrid Quantum-Classical DL ([`12_hybrid_quantum_dl/`](./12_hybrid_quantum_dl/))**
   - **Tensor & Quantum Problem:** Exact complex64 statevector evolution (`dim = 2^4 = 16`, 3 entangling `RY-RZ-CNOT` ring layers, native PyTorch autograd backprop, verified unitary max error `2.39e-07`, probability norm error `2.98e-07`) benchmarked honestly against parameter-matched classical MLPs (`~2%` param delta) on **two real domains**:
     - **Track A (Drug Discovery -- MoleculeNet `BBBP.csv`, 2,050 molecules):** Pure-Python 24-descriptor SMILES cheminformatics extractor -> Hybrid QNN **`0.9350` ROC-AUC / `0.9804` PR-AUC / `0.9114` F1** vs Classical MLP `0.9453` ROC-AUC.
     - **Track B (Finance & Credit Risk -- UCI Statlog `german.data`, 1,000 loans):** Hybrid QNN **`0.7848` ROC-AUC** (vs Classical `0.8114`) feeding a 20,000-trial Vasicek Monte Carlo portfolio credit-risk engine (`Expected Loss 14.12%`, `99% VaR 28.70%`, `99% CVaR 147.2k`) on only **`17.4 MB` peak VRAM**.

4. **Subject-Independent WESAD Physiological Stress Bi-LSTM ([`07_wesad_stress_bilstm/`](./07_wesad_stress_bilstm/))**
   - **Tensor & DSP Problem:** Multi-rate Empatica E4 wearable streams (`ACC` 32 Hz, `BVP` 64 Hz, `EDA` 4 Hz, `Temp` 4 Hz across 15 real subjects in `LouisSimon/wesad-parquet`) processed with **zero-phase 2nd-order SOS Butterworth digital filters** (BVP bandpass, EDA tonic/phasic decomposition, 3-axis ACC magnitude) and evaluated under a **strict zero-leakage subject-independent group split**.
   - **Architecture & Result:** `BiLSTMAttentionStressNet` hits **`89.33%` Test Accuracy**, **`0.8676` Macro-F1** (`+0.0186` over best classical RandomForest/LogReg window-stat baseline), and **`0.8530` ROC-AUC** on **`85.7 MB` peak VRAM** in **`41.8s`**, with interpretable temporal attention heatmaps pinpointing physiological stress onset.

---

## Complete 13-Project Master Benchmark Matrix (`13/13` CUDA Verified, `149/149` Pytest PASS)

| # | Domain | Project | Architecture | Real Dataset & Provenance | Primary Held-Out Test Metrics | Peak VRAM | Train Time | Pytest |
|---|---|---|---|---|---|---|---|---|
| **01** | CV -- Medical | [`01_eye_disease_gradcam/`](./01_eye_disease_gradcam/) | ResNet-18 + Grad-CAM (`layer4`) | MedMNIST v2 `RetinaMNIST` (128x128 RGB, Zenodo, 1,600 scans) + `dlzcods` ImageFolder adapter | **OvR ROC-AUC `0.8414`**, **QWK `0.7387`**, Acc `62.25%`, Macro-F1 `0.5445` (`2.48 ms/img`) | `314.5 MB` | `103.1s` | `12/12` |
| **02** | CV -- Medical | [`02_cervical_cancer_segmentation/`](./02_cervical_cancer_segmentation/) | Plain U-Net vs Additive Attention U-Net (`Dice+BCE+Sobel`) | HF `DataTorchio/nuclei-segmentation-dsb2018` (670 masks) + `MakAIHealthLab/cervical-cancer-biopsy-dataset` | **Plain Dice `0.8797` / Boundary F1 `0.9661`**, **Attn Dice `0.8739` / IoU `0.7760`**; Biopsy mean N/C `0.2468` (`23.1` nuclei/tile) | `1,153.2 MB` | `151.5s` | `12/12` |
| **03** | CV -- Industrial | [`03_corrosion_unet_segmentation/`](./03_corrosion_unet_segmentation/) | ResNet18-Encoder U-Net (`FocalDice alpha=0.75`) + ISO 4628-3 Grading | HF `Dodon/Corrosion` (LabelMe polygons, 440 metal images, 192x192) | **Test Dice `0.7720`**, **IoU `0.6286`**, Recall `0.8155`, Area MAE `5.85%`, **Severity Grade Acc `77.27%`**, **`197.0 FPS`** | `1,123.4 MB` | `104.9s` | `10/10` |
| **04** | CV -- Agriculture | [`04_leaf_disease_transfer_learning/`](./04_leaf_disease_transfer_learning/) | EfficientNet-B0 vs MobileNetV3-Small Transfer Learning | HF `BrandonFors/Plant-Diseases-PlantVillage-Dataset` (10-class stratified, 2,500 real RGB leaves) | **EffNet-B0 `97.80%` Acc / `0.9780` F1 / `99.6%` Top-3** (`12.0 ms`) vs **MobileNetV3 `97.00%` Acc** (`2.59 ms`, **`4.64x` faster**, `2.62x` smaller) | `544.9 MB` | `197.2s` | `10/10` |
| **05** | CV -- Automotive | [`05_indo_car_classifier_marketplace/`](./05_indo_car_classifier_marketplace/) | EfficientNet-B0 + FastAPI Used-Car IDR Valuation Engine | HF `tanganke/stanford_cars` (10 Indonesian-popular brand classes, 160x160 RGB) | **Top-1 `86.00%`**, **Top-3 `96.50%`**, Macro-F1 `0.8573`; FastAPI `/predict` **`p50 54.5 ms` / `p95 73.5 ms`** | `617.7 MB` | `40.1s` | `11/11` |
| **06** | CV -- Det & Seg | [`06_yolov8_detection_segmentation/`](./06_yolov8_detection_segmentation/) | Ultralytics `YOLOv8n` + `YOLOv8n-seg` + Roboflow Adapter | Real MS-COCO `coco8` & `coco8-seg` (320px) | **Det Box mAP@50 `0.6193`** (`54.5 FPS`); **Seg Mask mAP@50 `0.4762`** / Box `0.5494` (`51.1 FPS`) | `611.1 MB` | `36.3s` | `11/11` |
| **07** | Time-Series / DSP | [`07_wesad_stress_bilstm/`](./07_wesad_stress_bilstm/) | 2-Layer Bi-LSTM + Temporal Attention + SOS Butterworth DSP | HF `LouisSimon/wesad-parquet` (`wesad_4_3_120.parquet`, 15 real subjects, zero-leakage split) | **Test Acc `89.33%`**, **Macro-F1 `0.8676`** (`+1.86%` vs RF/LogReg), Stress-F1 `0.8093`, **ROC-AUC `0.8530`** | `85.7 MB` | `41.8s` | `12/12` |
| **08** | NLP (Indonesian) | [`08_indobert_sentiment_nlp/`](./08_indobert_sentiment_nlp/) | `indobenchmark/indobert-base-p1` (124.4M) vs TF-IDF + LogReg | Official **IndoNLU SmSA** (`train/valid/test_preprocess.tsv`, 3,600 stratified train / 500 official test) | **Test Acc `90.00%`**, **Macro-F1 `0.8629`** (**`+23.34%` F1 over TF-IDF** `0.6295`; Neg `0.937`, Neu `0.730`, Pos `0.922`) | `2,635.5 MB` | `1,409.7s` | `11/11` |
| **09** | Geospatial AI-SDSS | [`09_landslide_ai_sdss/`](./09_landslide_ai_sdss/) | 14-Band `SpatialResUNet` + 4-Tier GeoJSON FastAPI SDSS | HF `ibm-nasa-geospatial/Landslide4sense` (14-band Sentinel-2 + ALOS DEM/Slope HDF5 Parquet shards) | **Pixel ROC-AUC `0.9771`**, **PR-AUC `0.6441`**, **F1 `0.6371`**, IoU `0.4675`, Recall `0.6783`, Tile Det Acc `73.0%` | `592.3 MB` | `68.5s` | `12/12` |
| **10** | 3D Physics | [`10_icecube_neutrino_3d/`](./10_icecube_neutrino_3d/) | 4-Layer 3D Pulse Transformer + Fourier 4D + vMF Spherical Loss | Exact 5,160-DOM South Pole Cherenkov Physics Simulator (`13.5k` events) + Kaggle Parquet Loader | **Mean Angular Error `5.65 deg` (`0.0987 rad`)**, Median `4.54 deg` (**`-48.64%` error vs LineFit `11.00 deg`**) | `148.0 MB` | `298.2s` | `12/12` |
| **11** | Multimodal (Img+Aud) | [`11_multimodal_image_audio_fusion/`](./11_multimodal_image_audio_fusion/) | Dual CNN + Bidirectional Cross-Attention + Learned Gate `alpha` | Real Paired HF `ashraq/esc50` (Log-Mel 64x96) + `CIFAR-10` (7 overlapping classes, 0 fold/image leak) | Clean: Early `75.7%` / Gated `74.3%`; **Mixed Corrupt Winner: Gated `49.52%`** vs Img `39.3%` / Aud `44.1%` / Early `42.1%` (gate `alpha` adapts `0.115 -> 0.207` on audio noise!) | `75.0 MB` | `154.5s` | `14/14` |
| **12** | Hybrid Quantum-DL | [`12_hybrid_quantum_dl/`](./12_hybrid_quantum_dl/) | Complex64 4-Qubit 3-Layer VQC (`RY-RZ-CNOT`) + Classical MLP Ablation | Real DeepChem `BBBP.csv` (2,050 drugs) + Real UCI `german.data` (1,000 loans) | **Drug BBBP ROC-AUC `0.9350` / PR-AUC `0.9804`**; **Finance Credit ROC-AUC `0.7848`** + Vasicek `99% VaR 28.7%` (`unitary err 2.39e-07`) | `17.4 MB` | `117.0s` | `11/11` |
| **13** | Humanities & Nusantara | [`13_humanities_nusantara_dl/`](./13_humanities_nusantara_dl/) | Dual-Scale Char-Phonetic CNN + Word BiGRU Attention | Real HF `merve/poetry` (504 poems) + `maercaestro/pantun` & `LorthGyu/indonesian-pantun` (2,949 stanzas) | **Poetry Era AUC `0.9542`** (`88.2%` Acc), Theme AUC `0.7625`; **Nusantara Pantun Coherence AUC `0.8058`** (**`+26.32%` AUC vs TF-IDF `0.5426`**) | `83.6 MB` | `78.8s` | `11/11` |

---

## 4 Real Hardware & Ecosystem Engineering Bugs Caught & Fixed on 4 GB VRAM

1. **`transformers 5.x` + `indobert-lite-base-p1` Silent 100% `[UNK]` Collapse (`08_indobert_sentiment_nlp`):**
   - *Symptom:* Fine-tuning `indobenchmark/indobert-lite-base-p1` plateaued at `37.60%` test accuracy (`0.2720` Macro-F1, `0.00` neutral F1).
   - *Root Cause:* `AutoTokenizer` in `transformers 5.24` bound `AlbertTokenizer` (SentencePiece) onto the repo's WordPiece `vocab.txt`, turning every Indonesian word into `[UNK]` (`id=1`).
   - *Fix:* Switched default to canonical `indobenchmark/indobert-base-p1` (`BertTokenizer`, `0` `[UNK]`) and added an explicit `BertTokenizer` guard for `indobert-lite`. Test Accuracy jumped from `37.60% -> 90.00%` (`Macro-F1 0.2720 -> 0.8629`, `+23.34%` over TF-IDF).

2. **Turing TU117 (GTX 1650 Ti) cuDNN FP16 NaN + Inverted Focal Loss Alpha (`02`, `03`, `09`):**
   - *Symptom:* `AttentionUNet` (`02`) hit `train_loss=NaN` (`0.0000` Dice), `ResNet18UNet` (`03`) collapsed to all-background (`0.0000` Dice), and `SpatialResUNet` (`09`) emitted `NaN` during FP16 evaluation.
   - *Root Cause:* Turing SM 7.5 cuDNN FP16 overflows inside `BatchNorm2d(1)` Attention Gates and deep skip decoders; additionally, `03`'s `FocalDiceLoss(alpha=0.25)` weighted background pixels `3x` higher (`0.75`) than minority corrosion pixels (`0.25`).
   - *Fix:* Enforced FP32 + `clip_grad_norm_(1.0)` + finite-loss step guards on U-Net segmentation pipelines and inverted `alpha=0.75` for minority corrosion foreground. Results jumped immediately: `02` Attention U-Net `0.0000 -> 0.8739` Dice, `03` Corrosion U-Net `0.0000 -> 0.7720` Dice (`77.27%` ISO grade accuracy at `197 FPS`), and `09` Landslide4Sense `NaN -> 0.9771` Pixel ROC-AUC.

3. **WESAD 6.6 GB Nested-Array Parquet RAM Blowup (`07_wesad_stress_bilstm`):**
   - *Symptom:* `datasets.load_dataset("LouisSimon/wesad-parquet")` attempted to expand a 6.58 GB dense-overlap parquet (`wesad_60_0.3_120.parquet`) into 8+ GB RAM on a 15.8 GB laptop.
   - *Fix:* Targeted the exact `66.9 MB` `wesad_4_3_120.parquet` shard via `hf_hub_download` (`11,044` pre-windowed multi-rate Empatica E4 records across all `15` subjects), resampling 32/64/4 Hz streams to a unified 16 Hz Butterworth DSP grid with strict per-subject z-scoring (`85.7 MB` peak VRAM, `89.33%` Acc).

4. **Landslide4Sense HDF5 Packager Skip & HF 429 Rate-Limit (`09_landslide_ai_sdss`):**
   - *Symptom:* `datasets`'s generic `hdf5` packager skipped all 6,500 paired `.h5` files as empty, while fetching 1,000 individual `.h5` files via concurrent API calls triggered unauthenticated HTTP 429 backoff.
   - *Fix:* Consumed HuggingFace's official auto-converted Parquet shards (`https://huggingface.co/api/datasets/ibm-nasa-geospatial/Landslide4sense/parquet`), decoding embedded raw `image_<id>.h5` (`128x128x14` float64) and `mask_<id>.h5` payloads directly from memory via `h5py.File(io.BytesIO(...))` in only ~6 total HTTP requests.

---

## Additional Visual Proofs

| Cervical Nuclei & Biopsy Morphometry (`02`) | PlantVillage Transfer Learning Pareto (`04`) |
|---|---|
| ![Cervical Segmentation Overlay](./02_cervical_cancer_segmentation/reports/segmentation_overlay.png) | ![Leaf Disease Model Comparison](./04_leaf_disease_transfer_learning/reports/model_comparison.png) |

| YOLOv8n Det vs YOLOv8n-seg (`06`) | Multimodal Image+Audio Gate Ablation (`11`) |
|---|---|
| ![YOLOv8 Det vs Seg](./06_yolov8_detection_segmentation/reports/yolo_det_vs_seg.png) | ![Multimodal Ablation](./11_multimodal_image_audio_fusion/reports/multimodal_ablation.png) |

| IndoBERT vs TF-IDF SmSA (`08`) | Humanities & Nusantara Pantun Stylometry (`13`) |
|---|---|
| ![IndoBERT Eval](./08_indobert_sentiment_nlp/reports/indobert_eval.png) | ![Humanities Stylometry](./13_humanities_nusantara_dl/reports/humanities_stylometry.png) |

---

## Reproducibility & Verification Commands

Large trained checkpoints (`*.pt`, `*.pth`) and raw cached datasets (`data/`, `datasets/`, `artifacts/`) are excluded via `.gitignore` so the repository stays lightweight and clean. Every project ships with deterministic offline CPU unit tests and one-command CUDA reproduction scripts.

```powershell
# 1. Verify all 13 projects (149 offline CPU unit tests + required metrics.json checks)
.\verify_all.ps1

# 2. Reproduce sequential CUDA training on 4 GB VRAM (all 13, or pass -Only @("09_", "10_"))
.\scripts\run_all_train.ps1

# 3. Rebuild MASTER_BENCHMARK_LEDGER.json and flagship visual collage
.\.venv-dl\Scripts\python.exe .\scripts\generate_master_report.py
.\.venv-dl\Scripts\python.exe .\scripts\build_collage.py

# 4. Run pre-push security, bloat, local-path & strict-docs hygiene gate
node .\scripts\pre-push-hygiene.mjs . --strict-docs
```

### Hardware & Software Baseline
- **CPU / RAM:** Intel Core i5-10300H (4C/8T), 15.8 GB RAM
- **GPU:** NVIDIA GeForce GTX 1650 Ti (4 GB VRAM, Turing SM 7.5, CUDA 12.4 -- FP16 AMP / FP32 only, zero BF16)
- **Runtime:** Python `3.12.13`, PyTorch `2.6.0+cu124`, TorchVision `0.21.0+cu124`, Transformers `5.24.0`, Ultralytics `8.4.218`, Datasets `5.2.0`, h5py `3.16.0`, FastAPI `0.165.0`, Pytest `9.2.0`

## License

MIT License. See [LICENSE](./LICENSE).
