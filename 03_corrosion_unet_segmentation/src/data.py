"""COCO polygon rust segmentation loader for AiForFly/Rust-segmentation."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

RUST_REPO_ID = "Dodon/Corrosion"
DODON_ZIP_FILENAME = "original-20231018T064149Z-001.zip"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def rasterize_labelme_shapes(shapes: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    """Rasterize LabelMe polygon/rectangle corrosion annotations into a binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for shp in shapes:
        pts_raw = shp.get("points")
        if not isinstance(pts_raw, list) or len(pts_raw) < 2:
            continue
        shape_type = str(shp.get("shape_type", "polygon")).lower()
        pts = np.asarray(pts_raw, dtype=np.float32).reshape(-1, 2)
        if shape_type == "rectangle" or len(pts) == 2:
            x0, y0 = np.floor(pts.min(axis=0)).astype(int)
            x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(width, x1), min(height, y1)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 1
        elif len(pts) >= 3:
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def parse_labelme_split_dir(
    images_dir: str | Path,
    json_dir: str | Path,
    img_size: int = 192,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse paired LabelMe images/ and json/ directories into resized (RGB, binary_mask) arrays."""
    idir = Path(images_dir)
    jdir = Path(json_dir)
    if not idir.exists() or not jdir.exists():
        return np.empty((0, img_size, img_size, 3), dtype=np.uint8), np.empty((0, img_size, img_size), dtype=np.uint8)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_by_stem: dict[str, Path] = {}
    for p in sorted(idir.iterdir()):
        if p.is_file() and p.suffix.lower() in valid_exts:
            img_by_stem[p.stem.lower()] = p

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for jpath in sorted(jdir.glob("*.json")):
        img_path = img_by_stem.get(jpath.stem.lower())
        if img_path is None:
            continue
        try:
            payload = json.loads(jpath.read_text(encoding="utf-8", errors="ignore"))
            with Image.open(img_path) as pil:
                rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)
        except Exception:
            continue
        h, w = rgb.shape[:2]
        ann_h = int(payload.get("imageHeight") or h)
        ann_w = int(payload.get("imageWidth") or w)
        mask = rasterize_labelme_shapes(payload.get("shapes", []), ann_h, ann_w)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        images.append(cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA))
        masks.append((cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))

    if not images:
        return np.empty((0, img_size, img_size, 3), dtype=np.uint8), np.empty((0, img_size, img_size), dtype=np.uint8)
    return np.stack(images, axis=0), np.stack(masks, axis=0)


def rasterize_rust_polygons(annotations: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    """Rasterize COCO segmentation polygons (with bbox fallback) via cv2.fillPoly."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        seg = ann.get("segmentation")
        drawn = False
        if isinstance(seg, list) and len(seg) > 0:
            polys = seg if isinstance(seg[0], list) else [seg]
            for poly in polys:
                if isinstance(poly, list) and len(poly) >= 6:
                    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
                    drawn = True
        if not drawn and isinstance(ann.get("bbox"), list) and len(ann["bbox"]) == 4:
            bx, by, bw, bh = [float(v) for v in ann["bbox"]]
            x0, y0 = max(0, int(round(bx))), max(0, int(round(by)))
            x1, y1 = min(width, int(round(bx + bw))), min(height, int(round(by + bh)))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 1
    return mask


def parse_coco_split_dir(split_dir: str | Path, img_size: int = 192) -> tuple[np.ndarray, np.ndarray]:
    """Parse a single COCO split directory containing `_annotations.coco.json` and images."""
    sdir = Path(split_dir)
    if not sdir.exists():
        return np.empty((0, img_size, img_size, 3), dtype=np.uint8), np.empty((0, img_size, img_size), dtype=np.uint8)

    ann_file = sdir / "_annotations.coco.json"
    if not ann_file.exists():
        json_files = sorted(sdir.glob("*.json"))
        ann_file = json_files[0] if json_files else ann_file
    if not ann_file.exists():
        return np.empty((0, img_size, img_size, 3), dtype=np.uint8), np.empty((0, img_size, img_size), dtype=np.uint8)

    coco = json.loads(ann_file.read_text(encoding="utf-8"))
    anns_by_img: dict[Any, list[dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann.get("image_id"), []).append(ann)

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for meta in coco.get("images", []):
        fname = str(meta.get("file_name", ""))
        img_path = sdir / fname
        if not img_path.exists():
            img_path = sdir / Path(fname).name
        if not img_path.exists():
            continue
        with Image.open(img_path) as pil:
            rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        ann_h = int(meta.get("height") or h)
        ann_w = int(meta.get("width") or w)
        mask = rasterize_rust_polygons(anns_by_img.get(meta.get("id"), []), ann_h, ann_w)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        images.append(cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA))
        masks.append((cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))

    if not images:
        return np.empty((0, img_size, img_size, 3), dtype=np.uint8), np.empty((0, img_size, img_size), dtype=np.uint8)
    return np.stack(images, axis=0), np.stack(masks, axis=0)


def locate_coco_root(root_dir: str | Path) -> Path:
    """Locate the directory containing train/valid/test COCO folders (auto-extracting zip if needed)."""
    root = Path(root_dir)
    for zpath in sorted(root.rglob("*.zip")):
        extract_dir = zpath.parent / zpath.stem
        if not extract_dir.exists():
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(extract_dir)

    for ann in sorted(root.rglob("_annotations.coco.json")):
        if ann.parent.name.lower() in {"train", "valid", "val", "test"}:
            return ann.parent.parent

    preferred = root / "rust.v3i.coco-segmentation"
    if preferred.exists():
        return preferred
    return root


def download_and_load_rust_coco(
    data_dir: str | Path,
    img_size: int = 192,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Download ungated Dodon/Corrosion LabelMe polygon dataset (or COCO root) and return resized splits."""
    from huggingface_hub import hf_hub_download

    data_path = Path(data_dir)
    rust_dir = data_path / "dodon_corrosion"
    rust_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = data_path / f"dodon_corrosion_cache_{img_size}.npz"

    if cache_npz.exists():
        with np.load(cache_npz) as npz:
            return {
                "train": (npz["train_images"], npz["train_masks"]),
                "valid": (npz["valid_images"], npz["valid_masks"]),
                "test": (npz["test_images"], npz["test_masks"]),
            }

    zip_path = hf_hub_download(
        repo_id=RUST_REPO_ID,
        filename=DODON_ZIP_FILENAME,
        repo_type="dataset",
        local_dir=str(rust_dir),
    )
    extract_dir = rust_dir / "extracted"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    # Locate Train/{images,json} and Test/{images,json} robustly inside extracted archive
    train_root = None
    test_root = None
    for cand in sorted(extract_dir.rglob("json")):
        parent_name = cand.parent.name.lower()
        if parent_name == "train" and (cand.parent / "images").exists():
            train_root = cand.parent
        elif parent_name == "test" and (cand.parent / "images").exists():
            test_root = cand.parent

    if train_root is None:
        raise RuntimeError(f"Could not locate LabelMe Train/images + Train/json under {extract_dir}")

    all_tr_imgs, all_tr_masks = parse_labelme_split_dir(train_root / "images", train_root / "json", img_size=img_size)
    if test_root is not None:
        te_imgs, te_masks = parse_labelme_split_dir(test_root / "images", test_root / "json", img_size=img_size)
    else:
        te_imgs = np.empty((0, img_size, img_size, 3), dtype=np.uint8)
        te_masks = np.empty((0, img_size, img_size), dtype=np.uint8)

    if len(all_tr_imgs) == 0:
        raise RuntimeError(f"Failed to parse any corrosion training images from {train_root}")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(all_tr_imgs))
    val_count = max(1, int(round(0.15 * len(all_tr_imgs))))
    va_idx = perm[:val_count]
    tr_idx = perm[val_count:]
    tr_imgs, tr_masks = all_tr_imgs[tr_idx], all_tr_masks[tr_idx]
    va_imgs, va_masks = all_tr_imgs[va_idx], all_tr_masks[va_idx]

    np.savez_compressed(
        cache_npz,
        train_images=tr_imgs,
        train_masks=tr_masks,
        valid_images=va_imgs,
        valid_masks=va_masks,
        test_images=te_imgs,
        test_masks=te_masks,
    )
    return {
        "train": (tr_imgs, tr_masks),
        "valid": (va_imgs, va_masks),
        "test": (te_imgs, te_masks),
    }


class CorrosionDataset(Dataset):
    """PyTorch Dataset applying ImageNet normalization and optional geometric/photometric augs."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, is_train: bool = False) -> None:
        if len(images) != len(masks):
            raise ValueError(f"Images ({len(images)}) and masks ({len(masks)}) count mismatch")
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
            brightness = float(np.random.uniform(0.85, 1.15))
            contrast = float(np.random.uniform(0.85, 1.15))
            mean_val = img.mean(axis=(0, 1), keepdims=True)
            img = np.clip((img - mean_val) * contrast + mean_val * brightness, 0.0, 1.0)

        chw = img.transpose(2, 0, 1)
        chw = (chw - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(chw.astype(np.float32)), torch.from_numpy(mask).unsqueeze(0)
