"""Stratified streaming cache and PyTorch Dataset for PlantVillage leaf diseases."""
from __future__ import annotations

import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PLANTVILLAGE_REPO_ID = "BrandonFors/Plant-Diseases-PlantVillage-Dataset"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_leaf_transforms(img_size: int = 128, is_train: bool = False) -> transforms.Compose:
    """Build standard RGB transforms for 128x128 leaf images."""
    ops: list[Any] = [transforms.Resize((img_size, img_size))]
    if is_train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ]
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)


def _decode_pil(obj: Any) -> Image.Image:
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    if isinstance(obj, np.ndarray):
        return Image.fromarray(obj.astype(np.uint8)).convert("RGB")
    if isinstance(obj, dict):
        if obj.get("bytes") is not None:
            return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
        if obj.get("path") is not None:
            return Image.open(obj["path"]).convert("RGB")
    raise TypeError(f"Unsupported image object type: {type(obj).__name__}")


def clean_class_name(raw_name: str) -> str:
    """Format PlantVillage class names cleanly for plots and reports."""
    return raw_name.replace("___", " - ").replace("__", " ").replace("_", " ").strip()


def prepare_plantvillage_cache(
    cache_dir: str | Path,
    num_classes: int = 10,
    train_per_class: int = 160,
    val_per_class: int = 40,
    test_per_class: int = 50,
    img_size: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    """Stream a balanced 10-class subset from PlantVillage into local PNG cache + manifest.json."""
    cdir = Path(cache_dir)
    manifest_path = cdir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            len(manifest.get("class_names", [])) == num_classes
            and len(manifest.get("splits", {}).get("train", [])) >= num_classes * 10
        ):
            return manifest

    from datasets import load_dataset

    for sub in ("train", "val", "test"):
        (cdir / sub).mkdir(parents=True, exist_ok=True)

    train_stream = load_dataset(PLANTVILLAGE_REPO_ID, split="train", streaming=True)
    features = getattr(train_stream, "features", None)
    label_feat = features["label"] if features and "label" in features else None
    all_names: list[str] | None = list(label_feat.names) if label_feat and hasattr(label_feat, "names") else None

    needed_trainval = train_per_class + val_per_class
    selected_raw_labels: list[Any] = []
    selected_set: set[Any] = set()
    trainval_buckets: dict[Any, list[Image.Image]] = defaultdict(list)

    for row in train_stream:
        lbl = row.get("label")
        img_obj = row.get("image")
        if lbl is None or img_obj is None:
            continue
        if lbl not in selected_set:
            if len(selected_raw_labels) < num_classes:
                selected_raw_labels.append(lbl)
                selected_set.add(lbl)
            else:
                continue
        if len(trainval_buckets[lbl]) < needed_trainval:
            pil_img = _decode_pil(img_obj).resize((img_size, img_size), Image.Resampling.BILINEAR)
            trainval_buckets[lbl].append(pil_img)

        if len(selected_raw_labels) == num_classes and all(
            len(trainval_buckets[k]) >= needed_trainval for k in selected_raw_labels
        ):
            break

    if len(selected_raw_labels) < num_classes:
        raise RuntimeError(
            f"Expected {num_classes} PlantVillage classes, found only {len(selected_raw_labels)}."
        )

    raw_to_idx = {raw: i for i, raw in enumerate(selected_raw_labels)}
    class_names: list[str] = []
    for raw in selected_raw_labels:
        if isinstance(raw, int) and all_names and 0 <= raw < len(all_names):
            class_names.append(clean_class_name(all_names[raw]))
        else:
            class_names.append(clean_class_name(str(raw)))

    # Stream held-out test split for the exact same selected classes
    test_buckets: dict[Any, list[Image.Image]] = defaultdict(list)
    try:
        test_stream = load_dataset(PLANTVILLAGE_REPO_ID, split="test", streaming=True)
        for row in test_stream:
            lbl = row.get("label")
            img_obj = row.get("image")
            if lbl not in selected_set or img_obj is None:
                continue
            if len(test_buckets[lbl]) < test_per_class:
                pil_img = _decode_pil(img_obj).resize((img_size, img_size), Image.Resampling.BILINEAR)
                test_buckets[lbl].append(pil_img)
            if all(len(test_buckets[k]) >= test_per_class for k in selected_raw_labels):
                break
    except Exception:
        test_buckets = defaultdict(list)

    rng = np.random.default_rng(seed)
    splits_manifest: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for raw in selected_raw_labels:
        cidx = raw_to_idx[raw]
        tv_imgs = trainval_buckets[raw]
        perm = rng.permutation(len(tv_imgs))
        tv_imgs = [tv_imgs[int(i)] for i in perm]

        te_imgs = test_buckets.get(raw, [])
        if len(te_imgs) == 0:
            # Fallback if HF test split unavailable: split train/val/test cleanly
            n_te = max(1, min(test_per_class, len(tv_imgs) // 5))
            te_imgs = tv_imgs[-n_te:]
            tv_imgs = tv_imgs[:-n_te]

        n_val = min(val_per_class, max(1, len(tv_imgs) // 5))
        n_tr = min(train_per_class, len(tv_imgs) - n_val)
        tr_imgs = tv_imgs[:n_tr]
        va_imgs = tv_imgs[n_tr : n_tr + n_val]

        for split_name, imgs in (("train", tr_imgs), ("val", va_imgs), ("test", te_imgs)):
            for j, pil_img in enumerate(imgs):
                rel = f"{split_name}/{cidx:02d}_{j:04d}.png"
                pil_img.save(cdir / rel, format="PNG")
                splits_manifest[split_name].append({"path": rel, "label": cidx})

    manifest = {
        "repo_id": PLANTVILLAGE_REPO_ID,
        "img_size": img_size,
        "class_names": class_names,
        "splits": splits_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class CachedLeafDataset(Dataset):
    """Fast local PNG dataset backed by manifest.json."""

    def __init__(
        self,
        cache_dir: str | Path,
        records: list[dict[str, Any]],
        img_size: int = 128,
        is_train: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.records = list(records)
        self.transform = build_leaf_transforms(img_size=img_size, is_train=is_train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.records[idx]
        with Image.open(self.cache_dir / rec["path"]) as pil:
            img = pil.convert("RGB")
        return self.transform(img), int(rec["label"])

    def load_raw_rgb(self, idx: int) -> np.ndarray:
        rec = self.records[idx]
        with Image.open(self.cache_dir / rec["path"]) as pil:
            return np.asarray(pil.convert("RGB"), dtype=np.uint8)
