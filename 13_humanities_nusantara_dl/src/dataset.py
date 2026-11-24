"""Humanities literary datasets: World Classical Poetry (merve/poetry) & Nusantara Pantun."""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

ERA_NAMES: list[str] = ["Renaissance", "Modern"]
ERA_TO_ID: dict[str, int] = {name: i for i, name in enumerate(ERA_NAMES)}

THEME_NAMES: list[str] = ["Love", "Nature", "Mythology & Folklore"]
THEME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(THEME_NAMES)}

_WORD_RE = re.compile(r"[a-zA-Z0-9'-]+")
_CLEAN_SUFFIX_RE = re.compile(r"[^a-z0-9]+")


def tokenize_words(text: str) -> list[str]:
    """Lowercase literary word tokenizer preserving contractions and archaic forms (thou, o'er, 'tis)."""
    return [tok.strip("'-") for tok in _WORD_RE.findall(text.lower()) if tok.strip("'-")]


def extract_line_suffixes(text: str, suffix_len: int = 10) -> list[str]:
    """Extract the last `suffix_len` alphanumeric characters of 4 stanza lines (or 4 poem quarters)."""
    raw_lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]
    if not raw_lines:
        return ["", "", "", ""]

    if len(raw_lines) == 4:
        selected = raw_lines
    elif len(raw_lines) > 4:
        # Take the final line of each of the 4 equal stanza quarters.
        selected = [
            raw_lines[min(len(raw_lines) - 1, ((q + 1) * len(raw_lines)) // 4 - 1)]
            for q in range(4)
        ]
    else:
        selected = raw_lines + [raw_lines[-1]] * (4 - len(raw_lines))

    suffixes: list[str] = []
    for ln in selected:
        cleaned = _CLEAN_SUFFIX_RE.sub("", ln.lower())
        suffixes.append(cleaned[-suffix_len:] if cleaned else "")
    return suffixes


class WordCharTokenizer:
    """Dual word-level + character-level tokenizer fitted strictly on training texts."""

    def __init__(self, max_word_vocab: int = 8000, min_word_freq: int = 1) -> None:
        self.max_word_vocab = int(max_word_vocab)
        self.min_word_freq = int(min_word_freq)
        self.word2idx: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: list[str] = ["<PAD>", "<UNK>"]

        # Fixed character alphabet covering English classical poetry & Indonesian Nusantara pantun.
        base_chars = (
            "abcdefghijklmnopqrstuvwxyz0123456789 \n.,;:!?'\"-()[]/&"
        )
        self.char2idx: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        for ch in base_chars:
            if ch not in self.char2idx:
                self.char2idx[ch] = len(self.char2idx)

    @property
    def word_vocab_size(self) -> int:
        return len(self.word2idx)

    @property
    def char_vocab_size(self) -> int:
        return len(self.char2idx)

    def fit(self, texts: list[str]) -> WordCharTokenizer:
        counts: Counter[str] = Counter()
        for txt in texts:
            counts.update(tokenize_words(txt))
            for ch in txt.lower():
                if ch not in self.char2idx and len(self.char2idx) < 128:
                    self.char2idx[ch] = len(self.char2idx)

        for word, freq in counts.most_common(self.max_word_vocab - 2):
            if freq < self.min_word_freq:
                continue
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word.append(word)
        return self

    def encode(
        self,
        text: str,
        max_words: int = 128,
        max_chars: int = 480,
        suffix_len: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        tokens = tokenize_words(text)[:max_words]
        word_ids = np.zeros(max_words, dtype=np.int64)
        for i, tok in enumerate(tokens):
            word_ids[i] = self.word2idx.get(tok, 1)

        norm_text = text.replace("\r\n", "\n").lower()[:max_chars]
        char_ids = np.zeros(max_chars, dtype=np.int64)
        for i, ch in enumerate(norm_text):
            char_ids[i] = self.char2idx.get(ch, 1)

        suffixes = extract_line_suffixes(text, suffix_len=suffix_len)
        suffix_arr = np.zeros((4, suffix_len), dtype=np.int64)
        for line_idx, suf in enumerate(suffixes):
            # Right-align line endings so final rhyme phonemes align at identical positions!
            pad_left = suffix_len - len(suf)
            for j, ch in enumerate(suf):
                suffix_arr[line_idx, pad_left + j] = self.char2idx.get(ch, 1)

        return word_ids, char_ids, suffix_arr, tokens


class HumanitiesTextDataset(Dataset):
    """Pre-encoded dual-scale literary dataset returning (word_ids, char_ids, suffix_chars, label)."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: WordCharTokenizer,
        max_words: int = 128,
        max_chars: int = 480,
        suffix_len: int = 10,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have equal length.")
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.word_ids: list[np.ndarray] = []
        self.char_ids: list[np.ndarray] = []
        self.suffix_chars: list[np.ndarray] = []
        self.tokens_list: list[list[str]] = []

        for txt in self.texts:
            w, c, s, toks = tokenizer.encode(
                txt, max_words=max_words, max_chars=max_chars, suffix_len=suffix_len
            )
            self.word_ids.append(w)
            self.char_ids.append(c)
            self.suffix_chars.append(s)
            self.tokens_list.append(toks)

    def __len__(self) -> int:
        return int(len(self.labels))

    def get_tokens(self, idx: int) -> list[str]:
        return self.tokens_list[idx]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.word_ids[idx]),
            torch.from_numpy(self.char_ids[idx]),
            torch.from_numpy(self.suffix_chars[idx]),
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
        )


def _stratified_75_10_15(
    texts: list[str], labels: list[int], seed: int = 42
) -> dict[str, Any]:
    indices = np.arange(len(texts))
    y = np.asarray(labels, dtype=np.int64)
    _, counts = np.unique(y, return_counts=True)
    strat_first = y if (counts.min() >= 2 and int(0.25 * len(y)) >= len(counts)) else None
    idx_tr, idx_tmp, _, y_tmp = train_test_split(
        indices, y, test_size=0.25, stratify=strat_first, random_state=seed
    )
    # 10% val / 15% test from the remaining 25% => test_size = 15/25 = 0.60
    _, tmp_counts = np.unique(y_tmp, return_counts=True)
    n_val_est = len(idx_tmp) - int(round(0.60 * len(idx_tmp)))
    strat_second = (
        y_tmp
        if (tmp_counts.min() >= 2 and min(n_val_est, len(idx_tmp) - n_val_est) >= len(tmp_counts))
        else None
    )
    idx_va, idx_te, _, _ = train_test_split(
        idx_tmp, y_tmp, test_size=0.60, stratify=strat_second, random_state=seed
    )
    return {
        "train_texts": [texts[i] for i in idx_tr],
        "train_labels": [int(labels[i]) for i in idx_tr],
        "val_texts": [texts[i] for i in idx_va],
        "val_labels": [int(labels[i]) for i in idx_va],
        "test_texts": [texts[i] for i in idx_te],
        "test_labels": [int(labels[i]) for i in idx_te],
    }


def build_poetry_splits(records: list[dict[str, Any]], seed: int = 42) -> dict[str, Any]:
    """Deduplicate merve/poetry rows and produce leakage-free stratified 75/10/15 Era & Theme splits."""
    seen: set[str] = set()
    era_texts: list[str] = []
    era_labels: list[int] = []
    theme_texts: list[str] = []
    theme_labels: list[int] = []

    for row in records:
        content = str(row.get("content") or "").replace("\r\n", "\n").strip()
        if len(content) < 25:
            continue
        dedup_key = re.sub(r"\s+", " ", content.lower())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        age = str(row.get("age") or "").strip()
        genre = str(row.get("type") or "").strip()

        if age in ERA_TO_ID:
            era_texts.append(content)
            era_labels.append(ERA_TO_ID[age])
        if genre in THEME_TO_ID:
            theme_texts.append(content)
            theme_labels.append(THEME_TO_ID[genre])

    if not era_texts or not theme_texts:
        raise ValueError("Insufficient valid classical poetry records for Era/Theme classification.")

    return {
        "n_poems": len(seen),
        "era": _stratified_75_10_15(era_texts, era_labels, seed=seed),
        "theme": _stratified_75_10_15(theme_texts, theme_labels, seed=seed + 1),
    }


def extract_pantun_four_lines(row: dict[str, Any]) -> list[str] | None:
    """Normalize heterogeneous pantun dataset schemas into 4 clean lines."""
    # Schema 1: maercaestro/pantun structured_pantun2.csv -> Line_1..Line_4
    if all(k in row for k in ("Line_1", "Line_2", "Line_3", "Line_4")):
        lines = [str(row[f"Line_{i}"] or "").strip() for i in range(1, 5)]
    # Schema 2: LorthGyu/indonesian-pantun -> lines (list or stringified list)
    elif "lines" in row and row["lines"] is not None:
        raw_lines = row["lines"]
        if isinstance(raw_lines, str):
            try:
                parsed = ast.literal_eval(raw_lines)
                lines = [str(x).strip() for x in parsed] if isinstance(parsed, list) else raw_lines.splitlines()
            except Exception:
                lines = [ln.strip() for ln in raw_lines.splitlines() if ln.strip()]
        elif isinstance(raw_lines, (list, tuple)):
            lines = [str(x).strip() for x in raw_lines]
        else:
            return None
    else:
        # Generic fallback: single multi-line text field
        txt = str(row.get("pantun") or row.get("text") or row.get("content") or "").strip()
        lines = [ln.strip() for ln in txt.replace("\r\n", "\n").splitlines() if ln.strip()]

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in lines if ln.strip()]
    if len(lines) != 4:
        return None
    # Filter degenerate single-word rows (e.g. "Lemang / lontong / rendang / ketupat")
    if not all(len(ln.split()) >= 2 and len(ln) >= 10 for ln in lines):
        return None
    return lines


def _rhyme_tail(line: str) -> str:
    cleaned = _CLEAN_SUFFIX_RE.sub("", line.lower())
    return cleaned[-2:] if len(cleaned) >= 2 else cleaned


def build_pantun_coherence_pairs(
    stanzas: list[list[str]], seed: int = 42
) -> tuple[list[str], list[int]]:
    """Build balanced Authentic (y=1) vs Hard-Negative Permuted/Broken-Rhyme (y=0) pantun examples.

    Hard negatives (50/50 mix):
      1. Cross-stanza Sampiran-Isi Swap: [L1, L2] paired with [L3', L4'] from another stanza whose
         rhyme endings differ -> fluent Indonesian lines, broken a-b-a-b cadence & sampiran-isi link.
      2. Intra-stanza Cadence Permutation: [L1, L2, L4, L3] -> identical word multiset (exposes
         bag-of-words TF-IDF blindness) while breaking classical a-b-a-b into a-b-b-a!
    """
    n = len(stanzas)
    if n < 2:
        raise ValueError("Need at least 2 authentic pantun stanzas to build coherence pairs.")

    rng = np.random.default_rng(seed)
    texts: list[str] = []
    labels: list[int] = []

    for i, st in enumerate(stanzas):
        texts.append("\n".join(st))
        labels.append(1)

        if i % 2 == 0:
            # Cross-stanza sampiran-isi swap selecting a partner with non-matching a-rhyme tail
            t0 = _rhyme_tail(st[0])
            partner_idx = (i + int(rng.integers(1, n))) % n
            for _ in range(12):
                cand = int(rng.integers(0, n))
                if cand != i and _rhyme_tail(stanzas[cand][2]) != t0:
                    partner_idx = cand
                    break
            neg_lines = [st[0], st[1], stanzas[partner_idx][2], stanzas[partner_idx][3]]
        else:
            # Intra-stanza a-b-b-a rhyme perturbation (same words, broken rhyme alternation)
            neg_lines = [st[0], st[1], st[3], st[2]]

        texts.append("\n".join(neg_lines))
        labels.append(0)

    order = rng.permutation(len(labels))
    return [texts[i] for i in order], [int(labels[i]) for i in order]


def build_pantun_splits(stanzas: list[list[str]], seed: int = 42) -> dict[str, Any]:
    """Split authentic pantun stanzas 75/10/15 FIRST, then build hard negatives strictly inside each split."""
    # Deduplicate authentic stanzas before splitting to ensure zero leakage.
    seen: set[str] = set()
    unique_stanzas: list[list[str]] = []
    for st in stanzas:
        key = "\n".join(ln.lower() for ln in st)
        if key not in seen:
            seen.add(key)
            unique_stanzas.append(st)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_stanzas))
    n_total = len(unique_stanzas)
    n_train = max(2, int(round(0.75 * n_total)))
    n_val = max(2, int(round(0.10 * n_total)))
    if n_train + n_val >= n_total - 1:
        n_train = max(2, n_total - 4)
        n_val = max(2, (n_total - n_train) // 2)

    tr_stanzas = [unique_stanzas[i] for i in perm[:n_train]]
    va_stanzas = [unique_stanzas[i] for i in perm[n_train : n_train + n_val]]
    te_stanzas = [unique_stanzas[i] for i in perm[n_train + n_val :]]

    tr_texts, tr_labels = build_pantun_coherence_pairs(tr_stanzas, seed=seed + 10)
    va_texts, va_labels = build_pantun_coherence_pairs(va_stanzas, seed=seed + 20)
    te_texts, te_labels = build_pantun_coherence_pairs(te_stanzas, seed=seed + 30)

    return {
        "n_authentic_stanzas": n_total,
        "coherence": {
            "train_texts": tr_texts,
            "train_labels": tr_labels,
            "val_texts": va_texts,
            "val_labels": va_labels,
            "test_texts": te_texts,
            "test_labels": te_labels,
        },
    }


def load_world_poetry_dataset(data_dir: Path, seed: int = 42) -> dict[str, Any]:
    """Download/cache HuggingFace merve/poetry and return leakage-free Era + Theme splits."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "merve_poetry_cache.json"

    if cache_path.exists():
        records = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        from datasets import load_dataset

        ds = load_dataset("merve/poetry", split="train", cache_dir=str(data_dir / "hf_cache"))
        records = [
            {
                "author": row.get("author"),
                "content": row.get("content"),
                "poem name": row.get("poem name"),
                "age": row.get("age"),
                "type": row.get("type"),
            }
            for row in ds
        ]
        cache_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    return build_poetry_splits(records, seed=seed)


def load_nusantara_pantun_dataset(
    data_dir: Path, max_maercaestro: int = 3000, seed: int = 42
) -> dict[str, Any]:
    """Stream maercaestro/pantun + LorthGyu/indonesian-pantun, cache locally, and build zero-leakage splits."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / f"nusantara_pantun_cache_{max_maercaestro}.json"

    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        stanzas = cached["stanzas"]
        source_counts = cached["source_counts"]
    else:
        from datasets import load_dataset

        stanzas = []
        count_maer = 0
        count_lorth = 0

        # 1. Stream maercaestro/pantun structured_pantun2.csv (avoids mixed-schema jsonl CastError)
        ds_maer = load_dataset(
            "maercaestro/pantun",
            data_files="structured_pantun2.csv",
            split="train",
            streaming=True,
        )
        for row in ds_maer:
            lines = extract_pantun_four_lines(row)
            if lines is not None:
                stanzas.append(lines)
                count_maer += 1
            if count_maer >= max_maercaestro:
                break

        # 2. Load LorthGyu/indonesian-pantun annotated cultural pantun stanzas
        try:
            ds_lorth = load_dataset(
                "LorthGyu/indonesian-pantun",
                split="train",
                cache_dir=str(data_dir / "hf_cache"),
            )
            for row in ds_lorth:
                lines = extract_pantun_four_lines(row)
                if lines is not None:
                    stanzas.append(lines)
                    count_lorth += 1
        except Exception:
            # Graceful continuation if optional secondary repo is temporarily rate-limited
            pass

        source_counts = {"maercaestro": count_maer, "lorthgyu": count_lorth}
        cache_path.write_text(
            json.dumps({"stanzas": stanzas, "source_counts": source_counts}, ensure_ascii=False),
            encoding="utf-8",
        )

    splits = build_pantun_splits(stanzas, seed=seed)
    splits["source_counts"] = source_counts
    return splits
