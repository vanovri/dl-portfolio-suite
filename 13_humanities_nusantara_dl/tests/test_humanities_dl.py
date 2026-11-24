"""Deterministic offline CPU unit tests for Project 13: Classical Literature & Nusantara Pantun DL."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.dataset import (
    ERA_NAMES,
    THEME_NAMES,
    HumanitiesTextDataset,
    WordCharTokenizer,
    build_pantun_coherence_pairs,
    build_pantun_splits,
    build_poetry_splits,
    extract_line_suffixes,
    extract_pantun_four_lines,
)
from src.model import DualScaleHumanitiesNet
from src.viz import generate_humanities_stylometry_figure
from train import evaluate_neural_model, evaluate_tfidf_baseline


def _sample_pantun_stanzas() -> list[list[str]]:
    return [
        [
            "Berakit-rakit ke hulu",
            "Berenang-renang ke tepian",
            "Bersakit-sakit dahulu",
            "Bersenang-senang kemudian",
        ],
        [
            "Air surut memungut bayam",
            "Sayur diisi ke dalam kantung",
            "Kalau tuan tahu jawaban malam",
            "Tentu hati merasa beruntung",
        ],
        [
            "Pergi ke pasar membeli ketan",
            "Jangan lupa membawa ragi",
            "Kalau ingin hidup berkecukupan",
            "Rajin belajar sejak pagi",
        ],
        [
            "Burung merpati terbang tinggi",
            "Hinggap sebentar di dahan jati",
            "Ilmu dicari sepenuh hati",
            "Menjadi bekal sepanjang hari",
        ],
        [
            "Kayu cendana di atas batu",
            "Sudah diikat dibawa pulang",
            "Adat dunia memang begitu",
            "Benda yang buruk memang terbuang",
        ],
        [
            "Dari mana hendak ke mana",
            "Tinggi rumput dari padi",
            "Hari mana bulan mana",
            "Dapat kita berjumpa lagi",
        ],
        [
            "Asam kandis asam gelugur",
            "Ketiga asam riang-riang",
            "Menangis mayat di pintu kubur",
            "Teringat badan tidak sembahyang",
        ],
        [
            "Pisang emas dibawa berlayar",
            "Masak sebiji di atas peti",
            "Hutang emas boleh dibayar",
            "Hutang budi dibawa mati",
        ],
        [
            "Kalau ada sumur di ladang",
            "Boleh kita menumpang mandi",
            "Kalau ada umur yang panjang",
            "Boleh kita berjumpa lagi",
        ],
        [
            "Buah cempedak di luar pagar",
            "Ambil galah tolong jolokkan",
            "Saya budak baru belajar",
            "Kalau salah tolong tunjukkan",
        ],
        [
            "Pulau Pandan jauh ke tengah",
            "Gunung Daik bercabang tiga",
            "Hancur badan dikandung tanah",
            "Budi yang baik dikenang juga",
        ],
        [
            "Anak ayam turun sepuluh",
            "Mati satu tinggal sembilan",
            "Tuntutlah ilmu dengan sungguh",
            "Supaya hidup tidak sesalan",
        ],
    ]


def test_tokenizer_encodes_word_char_and_right_aligned_suffixes() -> None:
    tok = WordCharTokenizer(max_word_vocab=500)
    stanza = "\n".join(_sample_pantun_stanzas()[0])
    tok.fit([stanza])
    w, c, s, words = tok.encode(stanza, max_words=32, max_chars=180, suffix_len=8)
    assert w.shape == (32,)
    assert c.shape == (180,)
    assert s.shape == (4, 8)
    assert len(words) > 4
    # Final character of line 1 ('u' in hulu) matches final character of line 3 ('u' in dahulu)
    assert s[0, -1] == s[2, -1]
    assert s[1, -1] == s[3, -1]


def test_extract_line_suffixes_handles_multistanzas() -> None:
    poem = "Thou art more lovely\nAnd more temperate\nRough winds do shake\nThe darling buds\nOf May\nAnd summer lease"
    sufs = extract_line_suffixes(poem, suffix_len=6)
    assert len(sufs) == 4
    assert all(len(x) <= 6 for x in sufs)


def test_extract_pantun_four_lines_both_hf_schemas() -> None:
    row_maer = {
        "Line_1": "Rayuan datang demi rayuan",
        "Line_2": "Jangan mudah undi diberi",
        "Line_3": "Rumahtangga dibina dengan kesucian",
        "Line_4": "Keberkatan hidup itulah yang dicari.",
    }
    row_lorth = {
        "lines": [
            "Air surut memungut bayam",
            "Sayur diisi ke dalam kantung",
            "Kalau tuan bijak menanam",
            "Tentu panen menjadi untung",
        ]
    }
    bad_row = {"Line_1": "Lemang", "Line_2": "lontong", "Line_3": "rendang", "Line_4": "ketupat"}
    assert len(extract_pantun_four_lines(row_maer) or []) == 4
    assert len(extract_pantun_four_lines(row_lorth) or []) == 4
    assert extract_pantun_four_lines(bad_row) is None


def test_pantun_coherence_pairs_balanced_and_zero_split_leakage() -> None:
    stanzas = _sample_pantun_stanzas()
    texts, labels = build_pantun_coherence_pairs(stanzas, seed=42)
    assert len(texts) == len(stanzas) * 2
    assert sum(labels) == len(stanzas)

    splits = build_pantun_splits(stanzas, seed=42)
    coh = splits["coherence"]
    tr_set = set(coh["train_texts"])
    te_set = set(coh["test_texts"])
    assert len(tr_set & te_set) == 0


def test_poetry_splits_deduplication_and_stratification() -> None:
    records = []
    for i in range(48):
        age = ERA_NAMES[i % 2]
        theme = THEME_NAMES[i % 3]
        text = f"Thou shalt behold the {theme.lower()} verse number {i}\nWith golden metre bright and clear\n"
        records.append({"content": text, "age": age, "type": theme})
    # Add an exact duplicate that must be removed before splitting
    records.append(dict(records[0]))

    out = build_poetry_splits(records, seed=42)
    assert out["n_poems"] == 48
    tr = set(out["era"]["train_texts"])
    te = set(out["era"]["test_texts"])
    assert len(tr & te) == 0
    assert len(set(out["theme"]["test_labels"])) == 3


def test_dual_scale_net_forward_attention_and_gradients() -> None:
    torch.manual_seed(0)
    model = DualScaleHumanitiesNet(
        word_vocab_size=200, char_vocab_size=64, num_classes=3, word_embed_dim=32, gru_hidden=32
    )
    w = torch.randint(1, 200, (4, 24))
    w[:, -4:] = 0  # padding
    c = torch.randint(1, 64, (4, 120))
    s = torch.randint(1, 64, (4, 4, 10))
    y = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    details = model(w, c, line_suffix_chars=s, return_details=True)
    logits = details["logits"]
    attn = details["token_attention"]
    rhyme = details["rhyme_similarities"]

    assert logits.shape == (4, 3)
    assert attn.shape == (4, 24)
    assert rhyme.shape == (4, 6)
    # Attention sums to 1.0 and assigns 0.0 mass to padded tokens!
    assert torch.allclose(attn.sum(dim=-1), torch.ones(4), atol=1e-4)
    assert torch.allclose(attn[:, -4:], torch.zeros(4, 4), atol=1e-5)

    loss = torch.nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def test_dual_scale_net_fallback_without_explicit_suffix_tensor() -> None:
    model = DualScaleHumanitiesNet(word_vocab_size=100, char_vocab_size=50, num_classes=2)
    w = torch.randint(1, 100, (3, 16))
    c = torch.randint(1, 50, (3, 80))
    logits = model(w, c, line_suffix_chars=None, return_details=False)
    assert logits.shape == (3, 2)


def test_tfidf_baseline_runs_cleanly_offline() -> None:
    tr_texts = [
        "Thou art fairest maiden thy sweet love doth shine",
        "Thy noble crown doth gleam o'er ancient hall",
        "Steel trains rumble across concrete neon subway streets",
        "Modern glass skyscraper reflects electric city dusk",
    ] * 4
    tr_labels = [0, 0, 1, 1] * 4
    te_texts = [
        "Thou doth sing thy ancient renaissance sonnet",
        "Subway concrete neon lights glow across modern streets",
    ]
    te_labels = [0, 1]
    metrics, extras = evaluate_tfidf_baseline(tr_texts, tr_labels, te_texts, te_labels, num_classes=2)
    assert metrics["accuracy"] >= 0.5
    assert "fpr" in extras and "tpr" in extras


def test_evaluate_neural_model_extracts_salient_tokens() -> None:
    torch.manual_seed(4)
    stanzas = _sample_pantun_stanzas()
    texts, labels = build_pantun_coherence_pairs(stanzas, seed=4)
    tok = WordCharTokenizer(max_word_vocab=500).fit(texts)
    ds = HumanitiesTextDataset(texts, labels, tok, max_words=40, max_chars=200)
    model = DualScaleHumanitiesNet(tok.word_vocab_size, tok.char_vocab_size, num_classes=2)

    metrics, extras = evaluate_neural_model(
        model, ds, batch_size=8, device=torch.device("cpu"), class_names=["Permuted", "Authentic"]
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert isinstance(extras["top_tokens"], dict)


def test_mini_training_step_on_pantun_dataset() -> None:
    torch.manual_seed(12)
    texts, labels = build_pantun_coherence_pairs(_sample_pantun_stanzas()[:6], seed=12)
    tok = WordCharTokenizer(max_word_vocab=400).fit(texts)
    ds = HumanitiesTextDataset(texts, labels, tok, max_words=32, max_chars=180)
    loader = DataLoader(ds, batch_size=6, shuffle=True)

    model = DualScaleHumanitiesNet(tok.word_vocab_size, tok.char_vocab_size, num_classes=2)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    crit = torch.nn.CrossEntropyLoss()

    for w, c, s, y in loader:
        opt.zero_grad()
        loss = crit(model(w, c, line_suffix_chars=s), y)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)


def test_humanities_stylometry_figure_generation(tmp_path: Path) -> None:
    comp = {
        "era": {"neural_macro_f1": 0.88, "tfidf_macro_f1": 0.84},
        "theme": {"neural_macro_f1": 0.74, "tfidf_macro_f1": 0.70},
        "pantun": {"neural_macro_f1": 0.91, "tfidf_macro_f1": 0.56},
    }
    roc = {
        "neural": {"fpr": np.array([0.0, 0.1, 1.0]), "tpr": np.array([0.0, 0.92, 1.0]), "auc": 0.94},
        "tfidf": {"fpr": np.array([0.0, 0.4, 1.0]), "tpr": np.array([0.0, 0.55, 1.0]), "auc": 0.58},
    }
    out = generate_humanities_stylometry_figure(
        output_path=tmp_path / "humanities_stylometry.png",
        comparison_scores=comp,
        pantun_roc=roc,
        era_top_tokens={
            "Renaissance": [("thou", 0.24), ("thy", 0.19), ("doth", 0.16)],
            "Modern": [("street", 0.21), ("city", 0.17), ("light", 0.14)],
        },
        pantun_top_tokens=[("hati", 0.18), ("budi", 0.16), ("hidup", 0.14), ("ilmu", 0.12)],
    )
    assert out.exists() and out.stat().st_size > 5000
