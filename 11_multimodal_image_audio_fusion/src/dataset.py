"""Real paired Audio (ESC-50) + Image (CIFAR-10) dataset and pure-NumPy/SciPy Log-Mel extractor."""
from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
from torch.utils.data import Dataset

# 7-class semantic alignment between ESC-50 environmental audio and CIFAR-10 natural RGB images.
CLASS_SPEC: list[dict[str, Any]] = [
    {
        "id": 0,
        "name": "dog",
        "esc50_categories": ("dog",),
        "cifar10_labels": (5,),  # dog
    },
    {
        "id": 1,
        "name": "cat",
        "esc50_categories": ("cat",),
        "cifar10_labels": (3,),  # cat
    },
    {
        "id": 2,
        "name": "bird",
        "esc50_categories": ("rooster", "crow", "chirping_birds"),
        "cifar10_labels": (2,),  # bird
    },
    {
        "id": 3,
        "name": "frog",
        "esc50_categories": ("frog",),
        "cifar10_labels": (6,),  # frog
    },
    {
        "id": 4,
        "name": "airplane",
        "esc50_categories": ("airplane", "helicopter"),
        "cifar10_labels": (0,),  # airplane
    },
    {
        "id": 5,
        "name": "vehicle",
        "esc50_categories": ("car_horn", "engine"),
        "cifar10_labels": (1, 9),  # automobile, truck
    },
    {
        "id": 6,
        "name": "sea_ship",
        "esc50_categories": ("sea_waves",),
        "cifar10_labels": (8,),  # ship
    },
]

CLASS_NAMES: list[str] = [spec["name"] for spec in CLASS_SPEC]
ESC50_TO_CLASS: dict[str, int] = {
    cat: spec["id"] for spec in CLASS_SPEC for cat in spec["esc50_categories"]
}
CIFAR10_TO_CLASS: dict[int, int] = {
    lbl: spec["id"] for spec in CLASS_SPEC for lbl in spec["cifar10_labels"]
}

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32).view(3, 1, 1)


def _hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 1024,
    n_mels: int = 64,
    f_min: float = 20.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Create a (n_mels, n_fft // 2 + 1) triangular Mel filterbank using pure NumPy."""
    nyquist = sample_rate / 2.0
    upper = nyquist if f_max is None else min(float(f_max), nyquist)
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, nyquist, n_freqs, dtype=np.float64)

    mel_points = np.linspace(_hz_to_mel(f_min), _hz_to_mel(upper), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        f_left, f_center, f_right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        denom_left = max(f_center - f_left, 1e-8)
        denom_right = max(f_right - f_center, 1e-8)
        up_slope = (fft_freqs - f_left) / denom_left
        down_slope = (f_right - fft_freqs) / denom_right
        tri = np.maximum(0.0, np.minimum(up_slope, down_slope))
        # Ensure narrow low-frequency mel bands still map to the closest FFT bin.
        if tri.sum() < 1e-8:
            nearest_bin = int(np.argmin(np.abs(fft_freqs - f_center)))
            tri[nearest_bin] = 1.0
        # Slaney-style area normalization keeps energy balanced across mel bands.
        enorm = 2.0 / max(f_right - f_left, 1e-8)
        fb[m] = (tri * enorm).astype(np.float32)
    return fb


def decode_audio_entry(audio_entry: Any, fallback_sr: int = 44100) -> tuple[np.ndarray, int]:
    """Decode a HuggingFace Audio dict/bytes/path robustly without torchcodec or librosa."""
    if isinstance(audio_entry, dict):
        if audio_entry.get("array") is not None and audio_entry.get("sampling_rate") is not None:
            arr = np.asarray(audio_entry["array"], dtype=np.float32)
            sr = int(audio_entry["sampling_rate"])
            return _to_mono_float(arr), sr

        raw_bytes = audio_entry.get("bytes")
        if raw_bytes is not None:
            return decode_wav_bytes(bytes(raw_bytes))

        path_str = audio_entry.get("path")
        if path_str and Path(path_str).exists():
            return decode_wav_bytes(Path(path_str).read_bytes())

    if isinstance(audio_entry, (bytes, bytearray)):
        return decode_wav_bytes(bytes(audio_entry))

    if isinstance(audio_entry, (str, Path)) and Path(audio_entry).exists():
        return decode_wav_bytes(Path(audio_entry).read_bytes())

    if isinstance(audio_entry, np.ndarray):
        return _to_mono_float(audio_entry.astype(np.float32)), fallback_sr

    raise ValueError(f"Unsupported audio entry type: {type(audio_entry).__name__}")


def _to_mono_float(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    if np.issubdtype(arr.dtype, np.integer):
        max_val = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / max(max_val, 1.0)
    else:
        arr = arr.astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def decode_wav_bytes(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV/PCM bytes using SciPy, with stdlib wave and soundfile fallbacks."""
    try:
        sr, data = wavfile.read(io.BytesIO(raw_bytes))
        return _to_mono_float(data), int(sr)
    except Exception:
        pass

    try:
        with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
        if sampwidth in dtype_map:
            raw = np.frombuffer(frames, dtype=dtype_map[sampwidth])
            if sampwidth == 1:
                raw = (raw.astype(np.float32) - 128.0) / 128.0
            if n_channels > 1:
                raw = raw.reshape(-1, n_channels)
            return _to_mono_float(raw), int(sr)
    except Exception:
        pass

    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        return _to_mono_float(data), int(sr)
    except Exception as exc:
        raise RuntimeError("Failed to decode audio bytes via scipy/wave/soundfile.") from exc


class LogMelSpectrogramExtractor:
    """Pure NumPy/SciPy Log-Mel Spectrogram extractor producing exact (1, 64, 96) tensors."""

    def __init__(
        self,
        target_sr: int = 16000,
        duration_sec: float = 5.0,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
        n_frames: int = 96,
        f_min: float = 20.0,
        f_max: float = 8000.0,
    ) -> None:
        self.target_sr = int(target_sr)
        self.duration_sec = float(duration_sec)
        self.target_samples = int(round(self.target_sr * self.duration_sec))
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.n_frames = int(n_frames)
        self.mel_fb = build_mel_filterbank(
            sample_rate=self.target_sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            f_min=f_min,
            f_max=f_max,
        )

    def resample_and_pad(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        wav = _to_mono_float(waveform)
        if wav.size == 0:
            return np.zeros(self.target_samples, dtype=np.float32)

        if orig_sr != self.target_sr and orig_sr > 0:
            g = math.gcd(int(orig_sr), self.target_sr)
            up = self.target_sr // g
            down = int(orig_sr) // g
            wav = resample_poly(wav, up, down).astype(np.float32)

        # Peak-normalize only if signal is non-silent to keep dynamic range stable.
        peak = float(np.max(np.abs(wav)))
        if peak > 1e-6:
            wav = wav / peak

        if wav.shape[0] < self.target_samples:
            pad_total = self.target_samples - wav.shape[0]
            wav = np.pad(wav, (0, pad_total), mode="constant")
        elif wav.shape[0] > self.target_samples:
            wav = wav[: self.target_samples]
        return wav.astype(np.float32)

    def __call__(self, waveform: np.ndarray, orig_sr: int = 16000) -> np.ndarray:
        """Compute standardized Log-Mel spectrogram of shape (1, n_mels, n_frames)."""
        wav = self.resample_and_pad(waveform, orig_sr)
        noverlap = max(0, self.n_fft - self.hop_length)
        _, _, zxx = stft(
            wav,
            fs=self.target_sr,
            window="hann",
            nperseg=self.n_fft,
            noverlap=noverlap,
            boundary="zeros",
            padded=True,
        )
        power_spec = (np.abs(zxx) ** 2).astype(np.float32)  # (n_freqs, T)
        mel_spec = self.mel_fb @ power_spec  # (n_mels, T)
        log_mel = np.log(np.maximum(mel_spec, 1e-8)).astype(np.float32)

        # Resize time axis deterministically to exact (n_mels, n_frames).
        t_tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            t_tensor,
            size=(self.n_mels, self.n_frames),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (1, n_mels, n_frames)
        arr = resized.numpy()

        mean = float(arr.mean())
        std = float(arr.std())
        if std > 1e-6:
            arr = (arr - mean) / std
        else:
            arr = arr - mean
        return arr.astype(np.float32)


@dataclass(frozen=True)
class PairMetadata:
    class_id: int
    audio_id: str
    audio_fold: int
    image_split: str
    image_index: int


class PairedAudioVisualDataset(Dataset):
    """Paired (RGB image, Log-Mel spectrogram, class label) dataset with optional augmentation."""

    def __init__(
        self,
        images: np.ndarray,
        spectrograms: np.ndarray,
        labels: np.ndarray,
        pair_meta: list[PairMetadata] | None = None,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        if len(images) != len(spectrograms) or len(images) != len(labels):
            raise ValueError("images, spectrograms, and labels must share identical length.")
        self.images = np.asarray(images)  # (N, 32, 32, 3) uint8 or float32 [0,1]
        self.spectrograms = np.asarray(spectrograms, dtype=np.float32)  # (N, 1, 64, 96)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.pair_meta = pair_meta or []
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def _spec_augment(self, spec: torch.Tensor) -> torch.Tensor:
        out = spec.clone()
        _, n_mels, n_frames = out.shape
        # Mild SpecAugment: 1 frequency mask + 1 time mask.
        if self.rng.random() < 0.7:
            f_width = int(self.rng.integers(2, 10))
            f0 = int(self.rng.integers(0, max(1, n_mels - f_width)))
            out[:, f0 : f0 + f_width, :] = 0.0
        if self.rng.random() < 0.7:
            t_width = int(self.rng.integers(4, 16))
            t0 = int(self.rng.integers(0, max(1, n_frames - t_width)))
            out[:, :, t0 : t0 + t_width] = 0.0
        if self.rng.random() < 0.5:
            noise = torch.from_numpy(
                self.rng.normal(0.0, 0.05, size=out.shape).astype(np.float32)
            )
            out = out + noise
        return out

    def _image_augment(self, img: torch.Tensor) -> torch.Tensor:
        out = img
        if self.rng.random() < 0.5:
            out = torch.flip(out, dims=[2])
        if self.rng.random() < 0.5:
            # Random 2-pixel pad + crop preserves 32x32 resolution while improving generalization.
            padded = F.pad(out.unsqueeze(0), (2, 2, 2, 2), mode="reflect").squeeze(0)
            top = int(self.rng.integers(0, 5))
            left = int(self.rng.integers(0, 5))
            out = padded[:, top : top + 32, left : left + 32]
        return out

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_raw = self.images[idx]
        if img_raw.dtype == np.uint8:
            img = torch.from_numpy(img_raw.astype(np.float32) / 255.0).permute(2, 0, 1)
        elif img_raw.ndim == 3 and img_raw.shape[-1] == 3:
            img = torch.from_numpy(img_raw.astype(np.float32)).permute(2, 0, 1)
        else:
            img = torch.from_numpy(img_raw.astype(np.float32))

        if self.augment:
            img = self._image_augment(img)
        img = (img - CIFAR10_MEAN) / CIFAR10_STD

        spec = torch.from_numpy(self.spectrograms[idx].copy())
        if self.augment:
            spec = self._spec_augment(spec)

        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return img, spec, label


def verify_zero_leakage(
    train_meta: list[PairMetadata], test_meta: list[PairMetadata]
) -> dict[str, Any]:
    """Verify strict zero overlap across audio clip IDs, ESC-50 folds, and CIFAR-10 splits."""
    train_audio_ids = {m.audio_id for m in train_meta}
    test_audio_ids = {m.audio_id for m in test_meta}
    audio_overlap = train_audio_ids & test_audio_ids

    train_folds = sorted({m.audio_fold for m in train_meta})
    test_folds = sorted({m.audio_fold for m in test_meta})
    fold_overlap = set(train_folds) & set(test_folds)

    train_img_keys = {(m.image_split, m.image_index) for m in train_meta}
    test_img_keys = {(m.image_split, m.image_index) for m in test_meta}
    image_overlap = train_img_keys & test_img_keys

    is_clean = (len(audio_overlap) == 0) and (len(fold_overlap) == 0) and (len(image_overlap) == 0)
    return {
        "zero_leakage_verified": bool(is_clean),
        "train_unique_audio_clips": len(train_audio_ids),
        "test_unique_audio_clips": len(test_audio_ids),
        "audio_id_overlap": len(audio_overlap),
        "train_folds": train_folds,
        "test_folds": test_folds,
        "fold_overlap": len(fold_overlap),
        "image_key_overlap": len(image_overlap),
    }


def apply_modality_corruption(
    images: torch.Tensor,
    specs: torch.Tensor,
    mode: str,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply deterministic corruption for robustness stress tests.

    Returns:
        corrupted_images: Tensor (B, 3, 32, 32)
        corrupted_specs: Tensor (B, 1, 64, 96)
        corrupted_modality: LongTensor (B,) where 0=clean, 1=visual_corrupt, 2=audio_corrupt
    """
    bsz = images.size(0)
    device = images.device
    img_out = images.clone()
    spec_out = specs.clone()
    corrupt_flag = torch.zeros(bsz, dtype=torch.long, device=device)

    if mode == "clean":
        return img_out, spec_out, corrupt_flag

    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def _corrupt_image_sample(x: torch.Tensor) -> torch.Tensor:
        # Heavy blur + 50%+ cutout + additive noise destroys visual semantics.
        blurred = F.avg_pool2d(x.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
        noise = torch.randn(x.shape, generator=gen, dtype=x.dtype).to(device) * 1.4
        corrupted = blurred * 0.25 + noise
        # 50%+ random cutout block (24x24 inside 32x32 = 56.25% area).
        top = int(torch.randint(0, 9, (1,), generator=gen).item())
        left = int(torch.randint(0, 9, (1,), generator=gen).item())
        corrupted[:, top : top + 24, left : left + 24] = 0.0
        return corrupted

    def _corrupt_audio_sample(s: torch.Tensor) -> torch.Tensor:
        # High Gaussian noise + 60% time-frequency masking obliterates acoustic patterns.
        noise = torch.randn(s.shape, generator=gen, dtype=s.dtype).to(device) * 2.2
        corrupted = s * 0.15 + noise
        _, h, w = corrupted.shape
        t0 = int(torch.randint(0, max(1, w // 3), (1,), generator=gen).item())
        corrupted[:, :, t0 : min(w, t0 + (w * 3) // 5)] = 0.0
        f0 = int(torch.randint(0, max(1, h // 2), (1,), generator=gen).item())
        corrupted[:, f0 : min(h, f0 + h // 2), :] = 0.0
        return corrupted

    if mode == "visual_corrupt":
        for i in range(bsz):
            img_out[i] = _corrupt_image_sample(img_out[i])
            corrupt_flag[i] = 1
    elif mode == "audio_corrupt":
        for i in range(bsz):
            spec_out[i] = _corrupt_audio_sample(spec_out[i])
            corrupt_flag[i] = 2
    elif mode == "mixed_corrupt":
        for i in range(bsz):
            if i % 2 == 0:
                img_out[i] = _corrupt_image_sample(img_out[i])
                corrupt_flag[i] = 1
            else:
                spec_out[i] = _corrupt_audio_sample(spec_out[i])
                corrupt_flag[i] = 2
    else:
        raise ValueError(f"Unknown corruption mode: {mode}")

    return img_out, spec_out, corrupt_flag


def _load_or_extract_esc50_cache(
    data_dir: Path, extractor: LogMelSpectrogramExtractor
) -> dict[str, Any]:
    """Download ashraq/esc50 from HuggingFace and cache extracted Log-Mel spectrograms."""
    cache_path = data_dir / "esc50_logmel_7class.npz"
    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=False)
        return {
            "specs": loaded["specs"],
            "class_ids": loaded["class_ids"],
            "folds": loaded["folds"],
            "filenames": loaded["filenames"].tolist(),
        }

    from datasets import Audio, load_dataset

    data_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = data_dir / "hf_cache"
    ds = load_dataset("ashraq/esc50", split="train", cache_dir=str(hf_cache))
    # Avoid torchcodec by disabling automatic audio decoding before row iteration.
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))

    specs: list[np.ndarray] = []
    class_ids: list[int] = []
    folds: list[int] = []
    filenames: list[str] = []

    for idx, row in enumerate(ds):
        cat = str(row["category"]).strip()
        if cat not in ESC50_TO_CLASS:
            continue
        wav, sr = decode_audio_entry(row["audio"])
        log_mel = extractor(wav, orig_sr=sr)
        specs.append(log_mel)
        class_ids.append(ESC50_TO_CLASS[cat])
        folds.append(int(row["fold"]))
        fname = str(row.get("filename") or f"esc50_{idx}_{cat}.wav")
        filenames.append(fname)

    payload = {
        "specs": np.stack(specs, axis=0).astype(np.float32),
        "class_ids": np.asarray(class_ids, dtype=np.int64),
        "folds": np.asarray(folds, dtype=np.int64),
        "filenames": np.asarray(filenames, dtype="<U128"),
    }
    np.savez_compressed(cache_path, **payload)
    payload["filenames"] = filenames
    return payload


def _build_balanced_pairs(
    esc_specs: np.ndarray,
    esc_classes: np.ndarray,
    esc_folds: np.ndarray,
    esc_filenames: list[str],
    allowed_folds: set[int],
    cifar_images: np.ndarray,
    cifar_targets: np.ndarray,
    image_split: str,
    pairs_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[PairMetadata]]:
    rng = np.random.default_rng(seed)
    pair_imgs: list[np.ndarray] = []
    pair_specs: list[np.ndarray] = []
    pair_labels: list[int] = []
    pair_meta: list[PairMetadata] = []

    for cls_id in range(len(CLASS_SPEC)):
        audio_indices = [
            i
            for i in range(len(esc_classes))
            if int(esc_classes[i]) == cls_id and int(esc_folds[i]) in allowed_folds
        ]
        img_indices = [
            i
            for i in range(len(cifar_targets))
            if CIFAR10_TO_CLASS.get(int(cifar_targets[i]), -1) == cls_id
        ]
        if not audio_indices or not img_indices:
            raise RuntimeError(
                f"Missing audio ({len(audio_indices)}) or image ({len(img_indices)}) samples for class {cls_id}"
            )

        # Sample images without replacement when pool is large enough (CIFAR-10 has 5000/1000 per class).
        replace_img = len(img_indices) < pairs_per_class
        chosen_imgs = rng.choice(img_indices, size=pairs_per_class, replace=replace_img)
        # Cycle shuffled audio clips uniformly across pairs_per_class so every clip is balanced.
        audio_pool: list[int] = []
        while len(audio_pool) < pairs_per_class:
            perm = rng.permutation(audio_indices).tolist()
            audio_pool.extend(perm)
        chosen_auds = audio_pool[:pairs_per_class]

        for img_idx, aud_idx in zip(chosen_imgs, chosen_auds):
            pair_imgs.append(cifar_images[int(img_idx)])
            pair_specs.append(esc_specs[int(aud_idx)])
            pair_labels.append(cls_id)
            pair_meta.append(
                PairMetadata(
                    class_id=cls_id,
                    audio_id=esc_filenames[int(aud_idx)],
                    audio_fold=int(esc_folds[int(aud_idx)]),
                    image_split=image_split,
                    image_index=int(img_idx),
                )
            )

    order = rng.permutation(len(pair_labels))
    imgs_arr = np.stack([pair_imgs[i] for i in order], axis=0)
    specs_arr = np.stack([pair_specs[i] for i in order], axis=0)
    labels_arr = np.asarray([pair_labels[i] for i in order], dtype=np.int64)
    meta_ordered = [pair_meta[i] for i in order]
    return imgs_arr, specs_arr, labels_arr, meta_ordered


def build_real_paired_datasets(
    data_dir: Path,
    train_pairs_per_class: int = 240,
    test_pairs_per_class: int = 60,
    seed: int = 42,
) -> tuple[PairedAudioVisualDataset, PairedAudioVisualDataset, dict[str, Any]]:
    """Build zero-leakage real ESC-50 + CIFAR-10 paired train (~1,680) and test (~420) datasets."""
    from torchvision.datasets import CIFAR10

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    extractor = LogMelSpectrogramExtractor()
    esc = _load_or_extract_esc50_cache(data_dir, extractor)

    cifar_root = data_dir / "cifar10"
    cifar_train = CIFAR10(root=str(cifar_root), train=True, download=True)
    cifar_test = CIFAR10(root=str(cifar_root), train=False, download=True)

    tr_imgs, tr_specs, tr_labels, tr_meta = _build_balanced_pairs(
        esc_specs=esc["specs"],
        esc_classes=esc["class_ids"],
        esc_folds=esc["folds"],
        esc_filenames=esc["filenames"],
        allowed_folds={1, 2, 3, 4},
        cifar_images=np.asarray(cifar_train.data, dtype=np.uint8),
        cifar_targets=np.asarray(cifar_train.targets, dtype=np.int64),
        image_split="cifar10_train",
        pairs_per_class=train_pairs_per_class,
        seed=seed,
    )
    te_imgs, te_specs, te_labels, te_meta = _build_balanced_pairs(
        esc_specs=esc["specs"],
        esc_classes=esc["class_ids"],
        esc_folds=esc["folds"],
        esc_filenames=esc["filenames"],
        allowed_folds={5},
        cifar_images=np.asarray(cifar_test.data, dtype=np.uint8),
        cifar_targets=np.asarray(cifar_test.targets, dtype=np.int64),
        image_split="cifar10_test",
        pairs_per_class=test_pairs_per_class,
        seed=seed + 1000,
    )

    leakage_report = verify_zero_leakage(tr_meta, te_meta)
    if not leakage_report["zero_leakage_verified"]:
        raise RuntimeError(f"Data leakage detected: {leakage_report}")

    train_ds = PairedAudioVisualDataset(
        images=tr_imgs,
        spectrograms=tr_specs,
        labels=tr_labels,
        pair_meta=tr_meta,
        augment=True,
        seed=seed,
    )
    test_ds = PairedAudioVisualDataset(
        images=te_imgs,
        spectrograms=te_specs,
        labels=te_labels,
        pair_meta=te_meta,
        augment=False,
        seed=seed + 1,
    )

    info = {
        "dataset_type": "real_public_paired",
        "audio_source": "HuggingFace ashraq/esc50 (44.1kHz -> 16kHz Log-Mel 64x96)",
        "image_source": "torchvision CIFAR-10 (32x32 RGB)",
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "train_pairs": len(train_ds),
        "test_pairs": len(test_ds),
        "leakage_audit": leakage_report,
    }
    return train_ds, test_ds, info
