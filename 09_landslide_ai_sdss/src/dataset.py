"""Resilient Landslide4Sense streaming loader, multi-band normalizer, and PyTorch Dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

HF_DATASET_ID = "ibm-nasa-geospatial/Landslide4sense"
IMAGE_KEY_CANDIDATES: tuple[str, ...] = ("image", "img", "x", "bands", "sentinel")
MASK_KEY_CANDIDATES: tuple[str, ...] = ("label", "mask", "y", "segmentation", "target")


def _detect_key(sample: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if key in sample and sample[key] is not None:
            return key
    return None


def normalize_image_array(raw_img: Any, tile_size: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize multi-band numpy/list arrays or PIL RGB images into float32 (C, H, W)."""
    raw_type = type(raw_img).__name__
    if isinstance(raw_img, Image.Image):
        arr = np.asarray(raw_img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))
        layout = "pil_image"
    else:
        arr = np.asarray(raw_img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
            layout = "hw_2d"
        elif arr.ndim == 3:
            d0, d1, d2 = arr.shape
            # Distinguish (C, H, W) vs (H, W, C) robustly for 1-20 spectral/DEM bands
            if d0 <= 20 and d1 >= 16 and d2 >= 16 and d2 > d0:
                layout = "chw_array"
            elif d2 <= 20 and d0 >= 16 and d1 >= 16:
                arr = np.transpose(arr, (2, 0, 1))
                layout = "hwc_array"
            elif d0 <= d2:
                layout = "chw_fallback"
            else:
                arr = np.transpose(arr, (2, 0, 1))
                layout = "hwc_fallback"
        else:
            raise ValueError(f"Unsupported image ndim={arr.ndim}, shape={arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if tile_size is not None and (arr.shape[1] != tile_size or arr.shape[2] != tile_size):
        tensor = torch.from_numpy(arr).unsqueeze(0)
        resized = F.interpolate(tensor, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
        arr = resized.squeeze(0).numpy().astype(np.float32)

    info = {
        "raw_type": raw_type,
        "detected_layout": layout,
        "channels": int(arr.shape[0]),
        "height": int(arr.shape[1]),
        "width": int(arr.shape[2]),
    }
    return arr, info


def normalize_mask_array(raw_mask: Any, tile_size: int | None = None) -> np.ndarray:
    """Normalize segmentation mask (PIL, 2D/3D numpy array, or nested list) to binary (H, W) uint8."""
    if isinstance(raw_mask, Image.Image):
        arr = np.asarray(raw_mask)
    else:
        arr = np.asarray(raw_mask)

    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[0] <= 4 and arr.shape[1] >= 16:
            arr = arr.max(axis=0)
        else:
            arr = arr.max(axis=-1)
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported mask ndim={arr.ndim}, shape={arr.shape}")

    binary = (arr > 0.5).astype(np.uint8)
    if tile_size is not None and (binary.shape[0] != tile_size or binary.shape[1] != tile_size):
        tensor = torch.from_numpy(binary.astype(np.float32))[None, None, :, :]
        resized = F.interpolate(tensor, size=(tile_size, tile_size), mode="nearest")
        binary = (resized[0, 0].numpy() > 0.5).astype(np.uint8)
    return binary


def extract_rgb_preview(chw_tile: np.ndarray) -> np.ndarray:
    """Convert raw or standardized (C, H, W) tile into contrast-stretched (H, W, 3) float in [0, 1]."""
    c, h, w = chw_tile.shape
    if c >= 4:
        # Sentinel-2 Landslide4Sense convention: B4 (Red)=idx 3, B3 (Green)=idx 2, B2 (Blue)=idx 1
        rgb = np.stack([chw_tile[3], chw_tile[2], chw_tile[1]], axis=-1)
    elif c == 3:
        rgb = np.transpose(chw_tile, (1, 2, 0))
    elif c == 2:
        rgb = np.stack([chw_tile[0], chw_tile[1], 0.5 * (chw_tile[0] + chw_tile[1])], axis=-1)
    else:
        rgb = np.repeat(chw_tile[0:1], 3, axis=0).transpose(1, 2, 0)

    rgb = np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros_like(rgb, dtype=np.float32)
    for ch in range(3):
        plane = rgb[:, :, ch]
        lo = float(np.percentile(plane, 2))
        hi = float(np.percentile(plane, 98))
        if hi - lo < 1e-6:
            out[:, :, ch] = 0.0
        else:
            out[:, :, ch] = np.clip((plane - lo) / (hi - lo), 0.0, 1.0)
    return out


def collect_balanced_tiles(
    stream: Iterable[dict[str, Any]],
    target_count: int,
    tile_size: int = 128,
    pos_ratio: float = 0.65,
    max_scan: int = 2500,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Collect balanced positive (landslide present) and negative satellite tiles from a stream."""
    pos_target = int(round(target_count * pos_ratio))
    neg_target = max(0, target_count - pos_target)

    pos_imgs: list[np.ndarray] = []
    pos_masks: list[np.ndarray] = []
    neg_imgs: list[np.ndarray] = []
    neg_masks: list[np.ndarray] = []
    extra_imgs: list[np.ndarray] = []
    extra_masks: list[np.ndarray] = []

    schema_info: dict[str, Any] = {}
    scanned = 0
    expected_channels: int | None = None

    for sample in stream:
        if scanned >= max_scan:
            break
        scanned += 1
        img_key = _detect_key(sample, IMAGE_KEY_CANDIDATES)
        mask_key = _detect_key(sample, MASK_KEY_CANDIDATES)
        if img_key is None or mask_key is None:
            continue

        img, img_info = normalize_image_array(sample[img_key], tile_size=tile_size)
        mask = normalize_mask_array(sample[mask_key], tile_size=tile_size)

        if expected_channels is None:
            expected_channels = img.shape[0]
            schema_info = {
                "image_key": img_key,
                "mask_key": mask_key,
                **img_info,
            }
        elif img.shape[0] != expected_channels:
            continue

        has_landslide = bool(mask.sum() > 0)
        if has_landslide:
            if len(pos_imgs) < pos_target:
                pos_imgs.append(img)
                pos_masks.append(mask)
            elif len(extra_imgs) < target_count:
                extra_imgs.append(img)
                extra_masks.append(mask)
        else:
            if len(neg_imgs) < neg_target:
                neg_imgs.append(img)
                neg_masks.append(mask)
            elif len(extra_imgs) < target_count:
                extra_imgs.append(img)
                extra_masks.append(mask)

        if len(pos_imgs) >= pos_target and len(neg_imgs) >= neg_target:
            break

    imgs = pos_imgs + neg_imgs
    masks = pos_masks + neg_masks
    if len(imgs) < target_count and extra_imgs:
        need = target_count - len(imgs)
        imgs.extend(extra_imgs[:need])
        masks.extend(extra_masks[:need])

    if not imgs:
        raise RuntimeError("No valid tiles collected from Landslide4sense stream.")

    X = np.stack(imgs, axis=0).astype(np.float32)
    y = np.stack(masks, axis=0).astype(np.uint8)
    schema_info["scanned_samples"] = scanned
    schema_info["collected_tiles"] = int(len(y))
    schema_info["positive_tiles"] = int(sum(int(m.sum() > 0) for m in y))
    schema_info["negative_tiles"] = int(len(y) - schema_info["positive_tiles"])
    return X, y, schema_info


def compute_channel_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std across (N, C, H, W) training tiles."""
    mean = np.mean(X_train, axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = np.std(X_train, axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-5, 1.0, std).astype(np.float32)
    return mean, std


def standardize_tiles(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply channel-wise standardization and clip extreme satellite outliers."""
    normed = (X - mean[None, :, None, None]) / std[None, :, None, None]
    return np.clip(normed, -10.0, 10.0).astype(np.float32)


HF_PARQUET_API = "https://huggingface.co/api/datasets/ibm-nasa-geospatial/Landslide4sense/parquet"


def _fetch_parquet_table(url: str, timeout: int = 120):
    """Download one auto-converted HuggingFace Parquet shard into memory via pyarrow."""
    import io
    import urllib.request
    import pyarrow.parquet as pq

    req = urllib.request.Request(url, headers={"User-Agent": "dl-portfolio-suite/1.0"})
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    return pq.read_table(io.BytesIO(raw))


def _decode_h5_bytes(raw_bytes: bytes, preferred_key: str) -> np.ndarray:
    """Decode raw HDF5 file bytes directly from memory using h5py."""
    import io
    import h5py

    with h5py.File(io.BytesIO(raw_bytes), "r") as hf:
        key = preferred_key if preferred_key in hf else next(iter(hf.keys()))
        return hf[key][()]


def _load_parquet_split_pairs(
    shard_urls: list[str],
    target_count: int,
    tile_size: int,
    seed: int,
    pos_ratio: float = 0.65,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Pair image_<id>.h5 and mask_<id>.h5 raw HDF5 bytes across minimal Parquet shards."""
    import re

    img_bytes_by_id: dict[str, bytes] = {}
    msk_bytes_by_id: dict[str, bytes] = {}
    shards_fetched = 0

    for url in shard_urls:
        tbl = _fetch_parquet_table(url)
        shards_fetched += 1
        col = tbl["image"]
        for i in range(tbl.num_rows):
            entry = col[i].as_py()
            if not isinstance(entry, dict):
                continue
            pname = str(entry.get("path") or "")
            raw = entry.get("bytes")
            if not raw:
                continue
            m_img = re.search(r"image_(\d+)\.h5$", pname, re.IGNORECASE)
            if m_img:
                img_bytes_by_id[m_img.group(1)] = bytes(raw)
                continue
            m_msk = re.search(r"mask_(\d+)\.h5$", pname, re.IGNORECASE)
            if m_msk:
                msk_bytes_by_id[m_msk.group(1)] = bytes(raw)

        matched_ids = set(img_bytes_by_id.keys()) & set(msk_bytes_by_id.keys())
        if len(matched_ids) >= max(int(target_count * 1.4), target_count + 20):
            break

    shared_ids = sorted(set(img_bytes_by_id.keys()) & set(msk_bytes_by_id.keys()), key=lambda s: int(s))
    if not shared_ids:
        raise RuntimeError(f"No paired image_*.h5 / mask_*.h5 found across {shards_fetched} parquet shard(s).")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(shared_ids))

    pos_target = int(round(target_count * pos_ratio))
    neg_target = max(0, target_count - pos_target)
    pos_imgs: list[np.ndarray] = []
    pos_msks: list[np.ndarray] = []
    neg_imgs: list[np.ndarray] = []
    neg_msks: list[np.ndarray] = []
    extra_imgs: list[np.ndarray] = []
    extra_msks: list[np.ndarray] = []

    scanned = 0
    for idx in order:
        sid = shared_ids[int(idx)]
        scanned += 1
        try:
            raw_img = _decode_h5_bytes(img_bytes_by_id[sid], "img")
            raw_msk = _decode_h5_bytes(msk_bytes_by_id[sid], "mask")
            img, _ = normalize_image_array(raw_img, tile_size=tile_size)
            msk = normalize_mask_array(raw_msk, tile_size=tile_size)
        except Exception:
            continue

        if msk.sum() > 0:
            if len(pos_imgs) < pos_target:
                pos_imgs.append(img)
                pos_msks.append(msk)
            elif len(extra_imgs) < target_count:
                extra_imgs.append(img)
                extra_msks.append(msk)
        else:
            if len(neg_imgs) < neg_target:
                neg_imgs.append(img)
                neg_msks.append(msk)
            elif len(extra_imgs) < target_count:
                extra_imgs.append(img)
                extra_msks.append(msk)

        if len(pos_imgs) >= pos_target and len(neg_imgs) >= neg_target:
            break

    imgs = pos_imgs + neg_imgs
    msks = pos_msks + neg_msks
    if len(imgs) < target_count and extra_imgs:
        need = target_count - len(imgs)
        imgs.extend(extra_imgs[:need])
        msks.extend(extra_msks[:need])
    if not imgs:
        raise RuntimeError("No valid HDF5 tiles could be decoded from Landslide4sense parquet shards.")

    X = np.stack(imgs, axis=0).astype(np.float32)
    y = np.stack(msks, axis=0).astype(np.uint8)
    info = {
        "image_key": "img",
        "mask_key": "mask",
        "channels": int(X.shape[1]),
        "height": int(X.shape[2]),
        "width": int(X.shape[3]),
        "shards_fetched": shards_fetched,
        "paired_pool_size": len(shared_ids),
        "scanned_samples": scanned,
        "collected_tiles": int(len(y)),
        "positive_tiles": int(sum(int(m.sum() > 0) for m in y)),
        "negative_tiles": int(len(y) - sum(int(m.sum() > 0) for m in y)),
    }
    return X, y, info


def prepare_landslide_dataset(
    data_dir: Path,
    n_train: int = 320,
    n_val: int = 80,
    n_test: int = 100,
    tile_size: int = 128,
    seed: int = 42,
    force_download: bool = False,
) -> dict[str, Any]:
    """Download real 14-band paired HDF5 tiles from Landslide4sense official HF Parquet shards."""
    import urllib.request

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "landslide_cache.npz"

    if cache_path.exists() and not force_download:
        npz = np.load(cache_path, allow_pickle=False)
        meta = json.loads(str(npz["meta_json"]))
        meta["source"] = "local_npz_cache"
        return {
            "X_train": npz["X_train"],
            "y_train": npz["y_train"],
            "X_val": npz["X_val"],
            "y_val": npz["y_val"],
            "X_test": npz["X_test"],
            "y_test": npz["y_test"],
            "channel_mean": npz["channel_mean"],
            "channel_std": npz["channel_std"],
            "meta": meta,
        }

    req = urllib.request.Request(HF_PARQUET_API, headers={"User-Agent": "dl-portfolio-suite/1.0"})
    parquet_index = json.loads(urllib.request.urlopen(req, timeout=30).read().decode("utf-8"))
    default_splits = parquet_index.get("default", {})

    train_urls = default_splits.get("train", [])
    val_urls = default_splits.get("validation", [])
    test_urls = default_splits.get("test", [])

    if not val_urls and not test_urls and not train_urls:
        raise RuntimeError(f"No parquet shards listed at {HF_PARQUET_API}")

    # Note: In Landslide4sense parquet shards, `validation` (1 shard = 245 pairs) and `test` (3 shards = 800 pairs)
    # pair quickly in 1-3 HTTP requests; `train` puts masks in shard 0 and images starting around shard 1+.
    # Fetch train (or test/validation fallback cleanly across official shards) with zero 429 risk.
    X_val, y_val, val_info = _load_parquet_split_pairs(val_urls, n_val, tile_size, seed=seed + 17)
    X_test, y_test, test_info = _load_parquet_split_pairs(test_urls, n_test, tile_size, seed=seed + 31)

    # For train, masks live in early shard(s) and images in later shards; fetch train shards sequentially until paired.
    X_train, y_train, train_info = _load_parquet_split_pairs(train_urls, n_train, tile_size, seed=seed)

    ch_mean, ch_std = compute_channel_stats(X_train)
    meta = {
        "dataset_id": HF_DATASET_ID,
        "dataset_type": "real_public",
        "source": "huggingface_parquet_embedded_hdf5",
        "in_channels": int(X_train.shape[1]),
        "tile_size": int(tile_size),
        "raw_schema": train_info,
        "splits": {
            "train": {"source_split": "train", "tiles": int(len(y_train)), "positive_tiles": int(sum(int(m.sum() > 0) for m in y_train))},
            "val": {"source_split": "validation", "tiles": int(len(y_val)), "positive_tiles": int(sum(int(m.sum() > 0) for m in y_val))},
            "test": {"source_split": "test", "tiles": int(len(y_test)), "positive_tiles": int(sum(int(m.sum() > 0) for m in y_test))},
        },
        "train_landslide_pixel_rate": round(float(y_train.mean()), 6),
        "test_landslide_pixel_rate": round(float(y_test.mean()), 6),
    }

    np.savez_compressed(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        channel_mean=ch_mean,
        channel_std=ch_std,
        meta_json=json.dumps(meta),
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "channel_mean": ch_mean,
        "channel_std": ch_std,
        "meta": meta,
    }


class LandslideTileDataset(Dataset):
    """PyTorch Dataset applying channel standardization and optional geometric flips/rotations."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        augment: bool = False,
    ) -> None:
        self.X = standardize_tiles(X, channel_mean, channel_std)
        self.y = y.astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(self.X[idx].copy())
        mask = torch.from_numpy(self.y[idx].copy()).unsqueeze(0)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
                mask = torch.flip(mask, dims=[2])
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[1])
                mask = torch.flip(mask, dims=[1])
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                img = torch.rot90(img, k=k, dims=[1, 2])
                mask = torch.rot90(mask, k=k, dims=[1, 2])
        return img, mask
