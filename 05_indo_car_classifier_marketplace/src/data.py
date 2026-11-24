"""Streaming dataset builder for 10 Stanford Cars classes favoring Indonesian-popular brands."""
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

from src.model import build_car_transforms

STANFORD_CARS_REPO_ID = "tanganke/stanford_cars"

PREFERRED_INDO_BRANDS = (
    "Toyota",
    "Honda",
    "Suzuki",
    "Hyundai",
    "BMW",
    "Mercedes",
    "Audi",
    "Nissan",
    "Mitsubishi",
    "Ford",
    "Chevrolet",
)


def rank_candidate_class_ids(
    all_class_names: list[str],
    max_candidates: int = 24,
) -> list[int]:
    """Select candidate Stanford Cars class IDs prioritizing brands popular in Indonesia."""
    by_brand: dict[str, list[int]] = defaultdict(list)
    other_ids: list[int] = []

    for idx, name in enumerate(all_class_names):
        lower = name.lower()
        matched_brand = None
        for brand in PREFERRED_INDO_BRANDS:
            if brand.lower() in lower:
                matched_brand = brand
                break
        if matched_brand is not None:
            by_brand[matched_brand].append(idx)
        else:
            other_ids.append(idx)

    ordered: list[int] = []
    # Round-robin across preferred brands first for rich brand diversity
    max_depth = max((len(v) for v in by_brand.values()), default=0)
    for depth in range(max_depth):
        for brand in PREFERRED_INDO_BRANDS:
            bucket = by_brand.get(brand, [])
            if depth < len(bucket):
                ordered.append(bucket[depth])
                if len(ordered) >= max_candidates:
                    return ordered

    for idx in other_ids:
        ordered.append(idx)
        if len(ordered) >= max_candidates:
            break
    return ordered


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
    raise TypeError(f"Unsupported image payload: {type(obj).__name__}")


def prepare_stanford_cars_cache(
    cache_dir: str | Path,
    num_classes: int = 10,
    min_train_per_class: int = 35,
    max_train_per_class: int = 45,
    test_per_class: int = 20,
    img_size: int = 160,
    seed: int = 42,
) -> dict[str, Any]:
    """Stream 10 well-represented car classes from tanganke/stanford_cars and cache locally."""
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

    train_stream = load_dataset(STANFORD_CARS_REPO_ID, split="train", streaming=True)
    features = getattr(train_stream, "features", None)
    label_feat = features["label"] if features and "label" in features else None
    all_names: list[str] = list(label_feat.names) if label_feat and hasattr(label_feat, "names") else []

    if all_names:
        candidate_ids = rank_candidate_class_ids(all_names, max_candidates=max(num_classes * 3, 24))
    else:
        candidate_ids = list(range(max(num_classes * 3, 24)))
    candidate_set = set(candidate_ids)
    rank_map = {cid: r for r, cid in enumerate(candidate_ids)}

    buckets: dict[int, list[Image.Image]] = defaultdict(list)
    fallback_buckets: dict[int, list[Image.Image]] = defaultdict(list)

    for row in train_stream:
        raw_lbl = row.get("label")
        img_obj = row.get("image")
        if raw_lbl is None or img_obj is None:
            continue
        lbl = int(raw_lbl)
        if lbl in candidate_set:
            if len(buckets[lbl]) < max_train_per_class:
                buckets[lbl].append(_decode_pil(img_obj).resize((img_size, img_size), Image.Resampling.BILINEAR))
        elif len(fallback_buckets) < 20 or lbl in fallback_buckets:
            if len(fallback_buckets[lbl]) < max_train_per_class:
                fallback_buckets[lbl].append(_decode_pil(img_obj).resize((img_size, img_size), Image.Resampling.BILINEAR))

        ready = [cid for cid in candidate_ids if len(buckets[cid]) >= max_train_per_class]
        if len(ready) >= num_classes:
            break

    # Pick top `num_classes` candidates meeting min_train_per_class first, then highest counts
    combined: dict[int, list[Image.Image]] = {**fallback_buckets, **buckets}
    scored = sorted(
        combined.keys(),
        key=lambda cid: (
            len(combined[cid]) >= min_train_per_class,
            cid in candidate_set,
            -rank_map.get(cid, 9999),
            len(combined[cid]),
        ),
        reverse=True,
    )
    selected_raw_ids = scored[:num_classes]
    if len(selected_raw_ids) < num_classes:
        raise RuntimeError(f"Expected {num_classes} car classes, found {len(selected_raw_ids)}.")

    selected_set = set(selected_raw_ids)
    raw_to_idx = {raw: i for i, raw in enumerate(selected_raw_ids)}
    class_names = [
        all_names[raw] if all_names and 0 <= raw < len(all_names) else f"Car Class {raw}"
        for raw in selected_raw_ids
    ]

    # Stream held-out test split for selected classes
    test_buckets: dict[int, list[Image.Image]] = defaultdict(list)
    try:
        test_stream = load_dataset(STANFORD_CARS_REPO_ID, split="test", streaming=True)
        for row in test_stream:
            raw_lbl = row.get("label")
            img_obj = row.get("image")
            if raw_lbl is None or img_obj is None:
                continue
            lbl = int(raw_lbl)
            if lbl not in selected_set:
                continue
            if len(test_buckets[lbl]) < test_per_class:
                test_buckets[lbl].append(_decode_pil(img_obj).resize((img_size, img_size), Image.Resampling.BILINEAR))
            if all(len(test_buckets[cid]) >= test_per_class for cid in selected_raw_ids):
                break
    except Exception:
        test_buckets = defaultdict(list)

    rng = np.random.default_rng(seed)
    splits_manifest: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for raw in selected_raw_ids:
        cidx = raw_to_idx[raw]
        tv_imgs = combined[raw]
        perm = rng.permutation(len(tv_imgs))
        tv_imgs = [tv_imgs[int(i)] for i in perm]

        te_imgs = test_buckets.get(raw, [])
        if len(te_imgs) == 0:
            n_te = max(1, len(tv_imgs) // 5)
            te_imgs = tv_imgs[-n_te:]
            tv_imgs = tv_imgs[:-n_te]

        n_val = max(4, len(tv_imgs) // 5)
        tr_imgs = tv_imgs[:-n_val]
        va_imgs = tv_imgs[-n_val:]

        for split_name, imgs in (("train", tr_imgs), ("val", va_imgs), ("test", te_imgs)):
            for j, pil_img in enumerate(imgs):
                rel = f"{split_name}/{cidx:02d}_{j:04d}.png"
                pil_img.save(cdir / rel, format="PNG")
                splits_manifest[split_name].append({"path": rel, "label": cidx})

    manifest = {
        "repo_id": STANFORD_CARS_REPO_ID,
        "img_size": img_size,
        "class_names": class_names,
        "splits": splits_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class CachedCarDataset(Dataset):
    """Local PNG dataset for Stanford Cars subset."""

    def __init__(
        self,
        cache_dir: str | Path,
        records: list[dict[str, Any]],
        img_size: int = 160,
        is_train: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.records = list(records)
        self.transform = build_car_transforms(img_size=img_size, is_train=is_train)

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
