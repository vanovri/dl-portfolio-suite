"""Deterministic offline CPU unit tests for Project 11: Multimodal Image + Audio Fusion."""
from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.dataset import (
    CLASS_NAMES,
    CLASS_SPEC,
    LogMelSpectrogramExtractor,
    PairMetadata,
    PairedAudioVisualDataset,
    apply_modality_corruption,
    build_mel_filterbank,
    decode_audio_entry,
    verify_zero_leakage,
)
from src.model import GatedCrossAttentionFusionNet, build_model
from src.viz import generate_multimodal_ablation_figure
from train import VARIANTS, compute_training_loss


def _make_synthetic_wav_bytes(sr: int = 16000, freq: float = 440.0, duration: float = 1.0) -> bytes:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    sig = (0.5 * np.sin(2.0 * np.pi * freq * t) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(sig.tobytes())
    return buf.getvalue()


def test_class_alignment_spec_has_seven_classes() -> None:
    assert len(CLASS_SPEC) == 7
    assert CLASS_NAMES == ["dog", "cat", "bird", "frog", "airplane", "vehicle", "sea_ship"]


def test_mel_filterbank_non_negative_and_nonzero_rows() -> None:
    fb = build_mel_filterbank(sample_rate=16000, n_fft=1024, n_mels=64)
    assert fb.shape == (64, 513)
    assert np.all(fb >= 0.0)
    assert np.all(fb.sum(axis=1) > 0.0)


def test_wav_bytes_decode_and_logmel_shape() -> None:
    raw_wav = _make_synthetic_wav_bytes(sr=44100, freq=520.0, duration=1.5)
    wav, sr = decode_audio_entry({"bytes": raw_wav})
    assert sr == 44100
    assert wav.ndim == 1 and wav.size > 1000

    extractor = LogMelSpectrogramExtractor(target_sr=16000, duration_sec=5.0, n_mels=64, n_frames=96)
    spec = extractor(wav, orig_sr=sr)
    assert spec.shape == (1, 64, 96)
    assert np.isfinite(spec).all()
    assert abs(float(spec.mean())) < 1e-4


def test_decode_audio_entry_predecoded_array() -> None:
    arr = np.sin(np.linspace(0, 20, 4000, dtype=np.float32))
    wav, sr = decode_audio_entry({"array": arr, "sampling_rate": 16000})
    assert sr == 16000
    assert wav.shape == (4000,)


def test_paired_dataset_shapes_and_augmentation() -> None:
    rng = np.random.default_rng(7)
    imgs = rng.integers(0, 256, size=(14, 32, 32, 3), dtype=np.uint8)
    specs = rng.normal(size=(14, 1, 64, 96)).astype(np.float32)
    labels = np.arange(14, dtype=np.int64) % 7

    ds = PairedAudioVisualDataset(imgs, specs, labels, augment=True, seed=7)
    img, spec, lbl = ds[0]
    assert img.shape == (3, 32, 32)
    assert spec.shape == (1, 64, 96)
    assert lbl.dtype == torch.long


def test_zero_leakage_verifier_clean_and_leaky() -> None:
    train_meta = [
        PairMetadata(0, "a1.wav", 1, "cifar10_train", 10),
        PairMetadata(1, "a2.wav", 4, "cifar10_train", 20),
    ]
    test_meta_clean = [
        PairMetadata(0, "a3.wav", 5, "cifar10_test", 10),
        PairMetadata(1, "a4.wav", 5, "cifar10_test", 20),
    ]
    clean_rep = verify_zero_leakage(train_meta, test_meta_clean)
    assert clean_rep["zero_leakage_verified"] is True

    test_meta_leaky = [PairMetadata(0, "a1.wav", 1, "cifar10_train", 10)]
    leaky_rep = verify_zero_leakage(train_meta, test_meta_leaky)
    assert leaky_rep["zero_leakage_verified"] is False
    assert leaky_rep["audio_id_overlap"] == 1


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_four_variants_forward_and_backward(variant: str) -> None:
    torch.manual_seed(0)
    model = build_model(variant, num_classes=7, embed_dim=64)
    imgs = torch.randn(4, 3, 32, 32)
    specs = torch.randn(4, 1, 64, 96)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    details = model(imgs, specs, return_details=True)
    assert details["logits"].shape == (4, 7)
    assert details["alpha"].shape == (4, 1)
    assert torch.all((details["alpha"] >= 0.0) & (details["alpha"] <= 1.0))

    loss = compute_training_loss(model, imgs, specs, labels, torch.nn.CrossEntropyLoss())
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0


def test_gated_cross_attention_eval_gate_strictly_interior() -> None:
    torch.manual_seed(11)
    model = GatedCrossAttentionFusionNet(num_classes=7, embed_dim=64, num_heads=4)
    model.eval()
    imgs = torch.randn(6, 3, 32, 32)
    specs = torch.randn(6, 1, 64, 96)
    out = model(imgs, specs, return_details=True)
    alpha = out["alpha"]
    assert torch.all((alpha > 0.0) & (alpha < 1.0))
    assert out["img_to_aud_attn"].shape == (6, 16, 24)
    assert out["aud_to_img_attn"].shape == (6, 24, 16)


def test_apply_modality_corruption_modes() -> None:
    imgs = torch.ones(6, 3, 32, 32)
    specs = torch.ones(6, 1, 64, 96)

    ci, cs, flags = apply_modality_corruption(imgs, specs, mode="clean", seed=1)
    assert torch.allclose(ci, imgs) and torch.allclose(cs, specs)
    assert torch.all(flags == 0)

    ci_v, cs_v, flags_v = apply_modality_corruption(imgs, specs, mode="visual_corrupt", seed=2)
    assert not torch.allclose(ci_v, imgs)
    assert torch.allclose(cs_v, specs)
    assert torch.all(flags_v == 1)

    ci_a, cs_a, flags_a = apply_modality_corruption(imgs, specs, mode="audio_corrupt", seed=3)
    assert torch.allclose(ci_a, imgs)
    assert not torch.allclose(cs_a, specs)
    assert torch.all(flags_a == 2)

    _, _, flags_m = apply_modality_corruption(imgs, specs, mode="mixed_corrupt", seed=4)
    assert (flags_m == 1).sum().item() == 3
    assert (flags_m == 2).sum().item() == 3


def test_mini_training_step_reduces_or_stays_finite() -> None:
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    imgs = rng.integers(0, 256, size=(16, 32, 32, 3), dtype=np.uint8)
    specs = rng.normal(size=(16, 1, 64, 96)).astype(np.float32)
    labels = np.arange(16, dtype=np.int64) % 7
    ds = PairedAudioVisualDataset(imgs, specs, labels, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    model = GatedCrossAttentionFusionNet(num_classes=7, embed_dim=64, num_heads=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()

    losses = []
    for batch_imgs, batch_specs, batch_lbls in loader:
        opt.zero_grad()
        loss = compute_training_loss(model, batch_imgs, batch_specs, batch_lbls, crit)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert all(np.isfinite(losses))


def test_ablation_figure_generation(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    imgs = rng.integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)
    specs = rng.normal(size=(4, 1, 64, 96)).astype(np.float32)
    ablation = {
        v: {
            m: {"accuracy": 0.72, "macro_f1": 0.70, "mean_alpha": 0.5}
            for m in ("clean", "visual_corrupt", "audio_corrupt", "mixed_corrupt")
        }
        for v in VARIANTS
    }
    out_png = generate_multimodal_ablation_figure(
        output_path=tmp_path / "multimodal_ablation.png",
        sample_images=imgs,
        sample_specs=specs,
        sample_labels=["dog", "cat", "bird", "frog"],
        sample_alphas=[0.52, 0.48, 0.61, 0.44],
        gate_distributions={
            "clean": [0.51, 0.49],
            "visual_corrupt": [0.18, 0.22],
            "audio_corrupt": [0.84, 0.81],
        },
        ablation_results=ablation,
    )
    assert out_png.exists() and out_png.stat().st_size > 5000
