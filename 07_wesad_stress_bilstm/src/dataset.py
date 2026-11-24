"""Dataset loader and PyTorch Dataset for HuggingFace LouisSimon/wesad-parquet."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

HF_DATASET_ID = "LouisSimon/wesad-parquet"
HF_PARQUET_FILENAME = "wesad_4_3_120.parquet"


def load_wesad_dataframe(
    data_dir: Path,
    force_download: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load real WESAD Empatica E4 records (66.9 MB parquet) from cache or HuggingFace Hub."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / HF_PARQUET_FILENAME

    if cache_path.exists() and not force_download:
        df = pd.read_parquet(cache_path)
        source = "local_parquet_cache"
    else:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename=HF_PARQUET_FILENAME,
            repo_type="dataset",
            local_dir=str(data_dir),
        )
        df = pd.read_parquet(downloaded)
        source = f"huggingface_hub:{HF_PARQUET_FILENAME}"

    meta = {
        "dataset_id": HF_DATASET_ID,
        "dataset_file": HF_PARQUET_FILENAME,
        "dataset_type": "real_public",
        "source": source,
        "num_rows": int(len(df)),
        "num_subjects": int(df["user"].nunique()) if "user" in df.columns else 0,
        "columns": list(df.columns),
    }
    return df, meta


class WESADWindowDataset(Dataset):
    """PyTorch Dataset yielding (sequence_tensor, label_tensor)."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
