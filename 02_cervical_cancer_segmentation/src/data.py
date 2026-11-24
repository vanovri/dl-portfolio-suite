"""Dataset loaders for DSB2018 nuclei segmentation and cervical cancer biopsy morphometry."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DSB2018_REPO_ID = "DataTorchio/nuclei-segmentation-dsb2018"
CERVICAL_BIOPSY_REPO_ID = "MakAIHealthLab/cervical-cancer-biopsy-dataset"


def decode_coco_rle(rle: dict[str, Any]) -> np.ndarray:
    """Decode COCO uncompressed or LEB128-compressed RLE without external pycocotools."""
    size = rle.get("size")
    counts = rle.get("counts")
    if not size or counts is None:
        raise ValueError(f"Malformed RLE dictionary: keys={list(rle.keys())}")
    h, w = int(size[0]), int(size[1])

    if isinstance(counts, (bytes, bytearray)):
        counts = counts.decode("ascii")

    if isinstance(counts, str):
        # Standard COCO compressed LEB128 RLE decoding
        cnts: list[int] = []
        p = 0
        n = len(counts)
        while p < n:
            x = 0
            k = 0
            more = 1
            while more and p < n:
                c = ord(counts[p]) - 48
                x |= (c & 0x1F) << (5 * k)
                more = c & 0x20
                p += 1
                k += 1
                if not more and (c & 0x10):
                    x |= -1 << (5 * k)
            if len(cnts) > 2:
                x += cnts[-2]
            cnts.append(x)
        runs = cnts
    else:
        runs = [int(c) for c in counts]

    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for run in runs:
        if run > 0:
            end = min(pos + run, flat.size)
            if val == 1 and pos < end:
                flat[pos:end] = 1
            pos = end
        val = 1 - val
    return flat.reshape((h, w), order="F")


def rasterize_coco_annotations(
    annotations: list[dict[str, Any]],
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize a list of COCO annotations (polygons, RLE, or bbox fallback) into a binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        seg = ann.get("segmentation")
        drawn = False
        if isinstance(seg, list) and len(seg) > 0:
            # Either list of polygon lists [[x1, y1, ...]] or single flat polygon [x1, y1, ...]
            polys = seg if isinstance(seg[0], list) else [seg]
            for poly in polys:
                if isinstance(poly, list) and len(poly) >= 6:
                    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                    pts_int = np.round(pts).astype(np.int32)
                    cv2.fillPoly(mask, [pts_int], 1)
                    drawn = True
        elif isinstance(seg, dict) and "counts" in seg and "size" in seg:
            rle_mask = decode_coco_rle(seg)
            if rle_mask.shape != (height, width):
                rle_mask = cv2.resize(rle_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, (rle_mask > 0).astype(np.uint8))
            drawn = True

        if not drawn and "bbox" in ann and isinstance(ann["bbox"], list) and len(ann["bbox"]) == 4:
            bx, by, bw, bh = [float(v) for v in ann["bbox"]]
            x0 = max(0, int(round(bx)))
            y0 = max(0, int(round(by)))
            x1 = min(width, int(round(bx + bw)))
            y1 = min(height, int(round(by + bh)))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 1
    return mask


def _build_image_lookup(root_dir: Path) -> dict[str, Path]:
    """Index all image files inside root_dir by relative POSIX path and basename."""
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    lookup: dict[str, Path] = {}
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in valid_exts:
            rel = p.relative_to(root_dir).as_posix()
            lookup[rel] = p
            lookup.setdefault(p.name, p)
    return lookup


def _resolve_image_path(root_dir: Path, file_name: str, lookup: dict[str, Path]) -> Path | None:
    clean = file_name.replace("\\", "/").lstrip("./")
    candidates = [
        root_dir / clean,
        root_dir / "images" / clean,
        root_dir / "images" / Path(clean).name,
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    if clean in lookup:
        return lookup[clean]
    return lookup.get(Path(clean).name)


def parse_dsb2018_directory(
    root_dir: str | Path,
    img_size: int = 128,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse DSB2018 nuclei images and masks from a local snapshot directory.

    Supports COCO `annotations.json` (polygon, RLE, bbox fallback), custom list-of-records
    JSON, or paired `images/` + `masks/` folders.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"DSB2018 directory not found: {root}")

    ann_candidates = sorted(root.rglob("*annotations*.json")) + sorted(root.rglob("*.json"))
    ann_file = next((p for p in ann_candidates if p.is_file()), None)
    lookup = _build_image_lookup(root)

    images_out: list[np.ndarray] = []
    masks_out: list[np.ndarray] = []

    if ann_file is not None:
        raw = json.loads(ann_file.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "images" in raw and "annotations" in raw:
            anns_by_img: dict[Any, list[dict[str, Any]]] = {}
            for ann in raw["annotations"]:
                anns_by_img.setdefault(ann.get("image_id"), []).append(ann)

            for img_meta in raw["images"]:
                if max_samples is not None and len(images_out) >= max_samples:
                    break
                fname = str(img_meta.get("file_name") or img_meta.get("filename") or "")
                img_path = _resolve_image_path(root, fname, lookup)
                if img_path is None:
                    continue
                with Image.open(img_path) as pil:
                    rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)
                h, w = rgb.shape[:2]
                ann_h = int(img_meta.get("height") or h)
                ann_w = int(img_meta.get("width") or w)
                mask = rasterize_coco_annotations(anns_by_img.get(img_meta.get("id"), []), ann_h, ann_w)
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                rgb_res = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
                mask_res = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                images_out.append(rgb_res)
                masks_out.append((mask_res > 0).astype(np.uint8))
        elif isinstance(raw, list):
            for rec in raw:
                if max_samples is not None and len(images_out) >= max_samples:
                    break
                if not isinstance(rec, dict):
                    continue
                fname = str(rec.get("file_name") or rec.get("image") or rec.get("image_path") or "")
                img_path = _resolve_image_path(root, fname, lookup)
                if img_path is None:
                    continue
                with Image.open(img_path) as pil:
                    rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)
                h, w = rgb.shape[:2]
                anns = rec.get("annotations") or rec.get("objects") or []
                if isinstance(anns, list) and anns:
                    mask = rasterize_coco_annotations(anns, h, w)
                elif "segmentation" in rec or "bbox" in rec:
                    mask = rasterize_coco_annotations([rec], h, w)
                else:
                    continue
                images_out.append(cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA))
                masks_out.append((cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))
        else:
            sample_keys = list(raw.keys())[:10] if isinstance(raw, dict) else type(raw).__name__
            raise RuntimeError(
                f"Unsupported annotations schema in {ann_file.name}: top-level={sample_keys}"
            )

    if len(images_out) == 0:
        # Fallback: paired images/ and masks/ folders
        img_dir = root / "images"
        mask_dir = root / "masks"
        if img_dir.exists() and mask_dir.exists():
            for img_path in sorted(img_dir.iterdir()):
                if max_samples is not None and len(images_out) >= max_samples:
                    break
                if not img_path.is_file():
                    continue
                mask_path = mask_dir / img_path.name
                if not mask_path.exists():
                    mask_path = mask_dir / f"{img_path.stem}.png"
                if not mask_path.exists():
                    continue
                with Image.open(img_path) as pil:
                    rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)
                with Image.open(mask_path) as mpil:
                    m = np.asarray(mpil.convert("L"), dtype=np.uint8)
                images_out.append(cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA))
                masks_out.append((cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8))

    if len(images_out) == 0:
        raise RuntimeError(
            f"No valid image/mask pairs parsed from {root}. "
            f"Found {len(lookup)} image keys and annotation file={ann_file.name if ann_file else None}."
        )

    return np.stack(images_out, axis=0), np.stack(masks_out, axis=0)


def download_and_load_dsb2018(
    data_dir: str | Path,
    img_size: int = 128,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Download DataTorchio/nuclei-segmentation-dsb2018 via huggingface_hub and split 80/10/10."""
    from huggingface_hub import snapshot_download

    target_dir = Path(data_dir) / "dsb2018"
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = Path(data_dir) / f"dsb2018_cache_{img_size}.npz"

    if cache_npz.exists():
        with np.load(cache_npz) as npz:
            images = npz["images"]
            masks = npz["masks"]
    else:
        snapshot_download(
            repo_id=DSB2018_REPO_ID,
            repo_type="dataset",
            local_dir=str(target_dir),
        )
        images, masks = parse_dsb2018_directory(target_dir, img_size=img_size)
        np.savez_compressed(cache_npz, images=images, masks=masks)

    return deterministic_split(images, masks, seed=seed)


def deterministic_split(
    images: np.ndarray,
    masks: np.ndarray,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Deterministically split images and masks into train/val/test."""
    n = len(images)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n and n >= 3:
        n_train = n - 2
        n_val = 1
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    if len(test_idx) == 0:
        test_idx = val_idx
    return {
        "train": (images[train_idx], masks[train_idx]),
        "val": (images[val_idx], masks[val_idx]),
        "test": (images[test_idx], masks[test_idx]),
    }


class NucleiSegmentationDataset(Dataset):
    """PyTorch Dataset for 128x128 RGB microscopy images and binary nuclear masks."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, is_train: bool = False) -> None:
        if len(images) != len(masks):
            raise ValueError(f"Images ({len(images)}) and masks ({len(masks)}) length mismatch")
        self.images = np.asarray(images, dtype=np.uint8)
        self.masks = (np.asarray(masks) > 0).astype(np.float32)
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx].astype(np.float32) / 255.0
        mask = self.masks[idx].copy()

        if self.is_train:
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[::-1, :, :])
                mask = np.ascontiguousarray(mask[::-1, :])
            if np.random.rand() < 0.5:
                k = int(np.random.randint(1, 4))
                img = np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))
                mask = np.ascontiguousarray(np.rot90(mask, k, axes=(0, 1)))
            gain = float(np.random.uniform(0.9, 1.1))
            bias = float(np.random.uniform(-0.05, 0.05))
            img = np.clip(img * gain + bias, 0.0, 1.0)

        img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        mask_t = torch.from_numpy(mask).unsqueeze(0).contiguous()
        return img_t, mask_t


def _to_rgb_array(obj: Any) -> np.ndarray:
    """Convert HuggingFace image payload (PIL, dict, bytes, or ndarray) to RGB uint8 ndarray."""
    if isinstance(obj, Image.Image):
        return np.asarray(obj.convert("RGB"), dtype=np.uint8)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 2:
            return np.stack([obj, obj, obj], axis=-1).astype(np.uint8)
        return obj[:, :, :3].astype(np.uint8)
    if isinstance(obj, dict):
        if obj.get("bytes") is not None:
            with Image.open(io.BytesIO(obj["bytes"])) as pil:
                return np.asarray(pil.convert("RGB"), dtype=np.uint8)
        if obj.get("path") is not None:
            with Image.open(obj["path"]) as pil:
                return np.asarray(pil.convert("RGB"), dtype=np.uint8)
    raise TypeError(f"Unsupported image payload type: {type(obj).__name__}")


def stream_cervical_biopsy_samples(
    n_samples: int = 40,
    img_size: int = 128,
) -> list[dict[str, Any]]:
    """Stream real cervical cancer biopsy images from MakAIHealthLab/cervical-cancer-biopsy-dataset."""
    from datasets import load_dataset

    ds = load_dataset(CERVICAL_BIOPSY_REPO_ID, split="train", streaming=True)
    features = getattr(ds, "features", None)

    samples: list[dict[str, Any]] = []
    for row in ds:
        if len(samples) >= n_samples:
            break
        # Locate image field dynamically
        img_val = None
        for key in ("image", "img", "jpg", "png"):
            if key in row and row[key] is not None:
                img_val = row[key]
                break
        if img_val is None:
            for val in row.values():
                if isinstance(val, Image.Image) or (isinstance(val, dict) and ("bytes" in val or "path" in val)):
                    img_val = val
                    break
        if img_val is None:
            continue

        rgb = _to_rgb_array(img_val)
        rgb_res = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

        label_name = "biopsy"
        for lkey in ("label", "class", "diagnosis", "category", "dx", "subtype"):
            if lkey in row and row[lkey] is not None:
                raw_lbl = row[lkey]
                if isinstance(raw_lbl, int) and features and lkey in features and hasattr(features[lkey], "int2str"):
                    label_name = str(features[lkey].int2str(raw_lbl))
                else:
                    label_name = str(raw_lbl)
                break

        samples.append({"rgb": rgb_res, "label": label_name})

    return samples
