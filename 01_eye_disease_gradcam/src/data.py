"""Dataset downloaders and PyTorch dataset wrappers for retinal fundus classification."""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

RETINAMNIST_URL = "https://zenodo.org/records/10519652/files/retinamnist_128.npz"
RETINAMNIST_FILENAME = "retinamnist_128.npz"

RETINAMNIST_CLASSES = [
    "No DR (0)",
    "Mild NPDR (1)",
    "Moderate NPDR (2)",
    "Severe NPDR (3)",
    "Proliferative DR (4)",
]

EYE_DISEASE_CLASSES = [
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int = 128, is_train: bool = False) -> transforms.Compose:
    """Build deterministic or augmented RGB fundus transforms."""
    ops: list[Any] = [transforms.Resize((img_size, img_size))]
    if is_train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def download_retinamnist(data_dir: str | Path, url: str = RETINAMNIST_URL) -> Path:
    """Download MedMNIST v2 RetinaMNIST 128x128 NPZ archive into data_dir if missing."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    target = data_path / RETINAMNIST_FILENAME
    if target.exists() and target.stat().st_size > 100_000:
        return target

    tmp_path = data_path / f"{RETINAMNIST_FILENAME}.tmp"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "dl-portfolio-suite/1.0 (MedMNIST downloader)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(resp, dst)
        tmp_path.replace(target)
        return target
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        # Fallback to medmnist package if direct download is transiently unavailable
        try:
            import medmnist  # type: ignore

            ds = medmnist.RetinaMNIST(split="train", size=128, download=True, root=str(data_path))
            _ = len(ds)
            if target.exists():
                return target
        except Exception:
            pass
        raise


def load_retinamnist_splits(data_dir: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load train, val, and test numpy arrays from retinamnist_128.npz."""
    npz_path = download_retinamnist(data_dir)
    return load_retinamnist_from_npz(npz_path)


def load_retinamnist_from_npz(npz_path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read train/val/test image and label arrays from a local NPZ file."""
    with np.load(npz_path) as data:
        splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for split in ("train", "val", "test"):
            imgs = np.asarray(data[f"{split}_images"], dtype=np.uint8)
            labels = np.asarray(data[f"{split}_labels"]).reshape(-1).astype(np.int64)
            splits[split] = (imgs, labels)
    return splits


class RetinaMNISTDataset(Dataset):
    """PyTorch Dataset backed by uint8 NHWC RGB images and integer DR grade labels."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        img_size: int = 128,
        is_train: bool = False,
        transform: transforms.Compose | None = None,
    ) -> None:
        if len(images) != len(labels):
            raise ValueError(f"Image count ({len(images)}) != label count ({len(labels)})")
        self.images = np.asarray(images, dtype=np.uint8)
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        self.transform = transform if transform is not None else build_transforms(img_size, is_train)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_arr = self.images[idx]
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr, img_arr, img_arr], axis=-1)
        pil_img = Image.fromarray(img_arr, mode="RGB")
        tensor = self.transform(pil_img)
        return tensor, int(self.labels[idx])


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute smoothed inverse-frequency class weights for imbalanced DR grading."""
    arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(arr, minlength=num_classes).astype(np.float64)
    total = float(counts.sum())
    weights = np.ones(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
        else:
            weights[c] = 1.0
    # Dampen extreme weights slightly with sqrt to keep gradients stable on small minority classes
    weights = np.sqrt(weights)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def load_eye_disease_imagefolder(
    root_dir: str | Path,
    img_size: int = 128,
    is_train: bool = False,
) -> tuple[datasets.ImageFolder, list[str]]:
    """Drop-in loader for 4-class Eye Disease ImageFolder datasets.

    Compatible with Kaggle / GitHub eye disease datasets structured as:
      root_dir/
        cataract/
        diabetic_retinopathy/
        glaucoma/
        normal/
    Reference: dlzcods/eye-disease-classification.
    """
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"ImageFolder directory does not exist: {root}")
    tfm = build_transforms(img_size=img_size, is_train=is_train)
    dataset = datasets.ImageFolder(root=str(root), transform=tfm)
    return dataset, list(dataset.classes)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert an ImageNet-normalized CHW float tensor back to HWC uint8 RGB."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    img = (tensor.detach().cpu() * std.cpu() + mean.cpu()).clamp(0.0, 1.0)
    return (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
