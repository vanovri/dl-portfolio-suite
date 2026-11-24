"""IndoNLU SmSA dataset downloader, TSV parser, and dynamic-padding PyTorch Dataset."""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

INDONLU_SMSA_URLS: dict[str, str] = {
    "train": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv",
    "valid": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/valid_preprocess.tsv",
    "test": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/test_preprocess.tsv",
}

LABEL2ID: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_ALIASES: dict[str, str] = {
    "negative": "negative",
    "neg": "negative",
    "0": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "1": "neutral",
    "positive": "positive",
    "pos": "positive",
    "2": "positive",
}
HEADER_TOKENS: set[str] = {"label", "sentiment", "sentimen", "class", "target"}


def _download_with_retry(
    url: str,
    dest: Path,
    retries: int = 4,
    timeout: int = 30,
) -> None:
    """Download a file via HTTPS with exponential backoff retry."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "dl-portfolio-suite/1.0 (+https://github.com/IndoNLP/indonlu)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data or len(data) < 32:
                raise RuntimeError(f"Downloaded payload too small ({len(data)} bytes) from {url}")
            tmp_path = dest.with_suffix(dest.suffix + ".tmp")
            tmp_path.write_bytes(data)
            tmp_path.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(min(2.0 ** (attempt - 1), 8.0))
    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_err}") from last_err


def download_smsa_dataset(
    data_dir: Path,
    retries: int = 4,
    timeout: int = 30,
    force_download: bool = False,
) -> dict[str, Path]:
    """Ensure train, valid, and test TSV files from official IndoNLU SmSA are cached locally."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, url in INDONLU_SMSA_URLS.items():
        fname = f"{split}_preprocess.tsv"
        dest = data_dir / fname
        if force_download or not dest.exists() or dest.stat().st_size == 0:
            _download_with_retry(url, dest, retries=retries, timeout=timeout)
        paths[split] = dest
    return paths


def load_smsa_tsv(path: Path) -> tuple[list[str], list[int]]:
    """Parse an IndoNLU SmSA TSV file (text<TAB>label), skipping blank lines and optional header."""
    path = Path(path)
    texts: list[str] = []
    labels: list[int] = []
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    for idx, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line:
            continue
        parts = raw_line.rstrip("\r\n").split("\t")
        if len(parts) < 2:
            continue
        text = "\t".join(parts[:-1]).strip()
        label_str = parts[-1].strip().lower()

        if idx == 0 and label_str in HEADER_TOKENS:
            continue
        if not text:
            continue

        canonical = LABEL_ALIASES.get(label_str)
        if canonical is None:
            continue
        texts.append(text)
        labels.append(LABEL2ID[canonical])

    if not texts:
        raise ValueError(f"No valid text/label rows parsed from {path}")
    return texts, labels


class SmsaDataset(Dataset):
    """Dataset holding raw Indonesian review strings and integer sentiment labels."""

    def __init__(self, texts: list[str], labels: list[int]) -> None:
        if len(texts) != len(labels):
            raise ValueError(f"Mismatched lengths: {len(texts)} texts vs {len(labels)} labels")
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], self.labels[idx]


def make_dynamic_collate_fn(tokenizer: Any, max_length: int = 128):
    """Build a DataLoader collate_fn applying dynamic batch padding up to max_length."""

    def _collate(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        batch_texts = [item[0] for item in batch]
        batch_labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out: dict[str, torch.Tensor] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": batch_labels,
        }
        if "token_type_ids" in encoded:
            out["token_type_ids"] = encoded["token_type_ids"]
        return out

    return _collate
