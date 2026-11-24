# 10 - IceCube 3D Neutrino Direction Reconstruction (vMF Pulse Transformer)

3D astroparticle neutrino direction reconstruction on the exact **IceCube 5,160-DOM South Pole hexagonal detector array** using a **4-layer Pre-Norm Pulse Transformer** trained with the **3D von Mises-Fisher (vMF) spherical loss**, benchmarked against the classical **IceCube Charge-Weighted LineFit** analytical velocity-vector estimator.

## Pipeline & Physics

1. **Exact 5,160-DOM Detector & Relativistic Cherenkov Simulator (`src/detector.py`, `src/dataset.py`)**:
   - Models 78 standard strings on a 125 m triangular lattice (60 DOMs at 17 m vertical spacing = 4,680 DOMs) plus 8 central DeepCore infill strings (60 DOMs each = 480 DOMs) for exact **5,160 sensors**.
   - Simulates isotropic neutrinos, relativistic muon propagation at $c$, Cherenkov light-cone photon emission in glacial ice ($n_{\text{ice}} = 1.31$, $\theta_C \approx 40.2^\circ$), exponential ice-scattering delays, Poisson distance-attenuated photoelectron charges, and random auxiliary dark-noise hits in exact Kaggle IceCube schema (`sensor_id, time, charge, auxiliary`, targets `azimuth, zenith`).
   - Dataset provenance is labeled transparently as `physics_calibrated_icecube_5160dom`.
   - Includes verified drop-in Kaggle competition loader `src.dataset.load_kaggle_icecube_batch(batch_parquet, meta_parquet, geometry_csv)`.
2. **Model & Spherical Loss (`src/model.py`, `src/losses.py`)**:
   - `IceCubePulseTransformer`: Top-K (`K=96`) pulse sequence `(x/500, y/500, z/500, rel_t/1000, log1p(charge), auxiliary)` -> 4D sinusoidal spacetime Fourier encoding -> 4-layer Pre-Norm Transformer Encoder (`d_model=128, nhead=4`) -> Masked Attention Pooling -> 3D unit vector $\hat{u}$ and concentration $\kappa$.
   - Trained with numerically stable **3D von Mises-Fisher Loss** (`-kappa * cos_sim + log(sinh(kappa) / kappa)`) and evaluated with exact Kaggle **Mean Angular Error** $\arccos(\text{clip}(\hat{u} \cdot u, -1, 1))$.
3. **Classical Physics Baseline (`src/physics_baseline.py`)**:
   - Analytical **IceCube Charge-Weighted LineFit** ($\vec{r}(t) = \vec{r}_0 + \vec{v}t$) evaluated on the identical held-out test events.

## Reproduction

```bash
# Run offline CPU unit tests
pytest tests/

# Generate physics-calibrated dataset, train Pulse Transformer, and benchmark vs LineFit
python train.py --epochs 20 --batch-size 64 --seed 42
```

## Outputs

- `reports/metrics.json` -- Mean Angular Error (rad & deg), Median Angular Error (deg), 68%/90% containment angles, Transformer vs LineFit improvement, and peak VRAM telemetry.
- `reports/neutrino_reconstruction.png` -- 3D IceCube DOM event scatter with true vs predicted track vectors, angular error distribution histogram, and Mollweide neutrino sky map.
- `artifacts/icecube_transformer_best.pt` -- Best validation checkpoint (gitignored).
