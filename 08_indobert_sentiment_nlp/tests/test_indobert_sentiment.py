"""Deterministic offline CPU unit tests for 08_indobert_sentiment_nlp."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AlbertConfig, AlbertForSequenceClassification

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(PROJECT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.baseline import compute_multiclass_metrics, train_and_eval_tfidf_baseline
from src.dataset import ID2LABEL, LABEL2ID, SmsaDataset, load_smsa_tsv, make_dynamic_collate_fn
from src.model import build_linear_warmup_scheduler, predict_texts
from src.viz import plot_indobert_evaluation


class _DummyIndonesianTokenizer:
    """Fast offline whitespace tokenizer matching HuggingFace __call__ contract for CPU tests."""

    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 32,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        seqs: list[list[int]] = []
        for t in texts:
            toks = [2] + [(abs(hash(w)) % (self.vocab_size - 4)) + 4 for w in t.lower().split()] + [3]
            if truncation:
                toks = toks[:max_length]
            seqs.append(toks)
        pad_len = max(len(s) for s in seqs) if padding else max_length
        input_ids = torch.zeros((len(seqs), pad_len), dtype=torch.long)
        attn_mask = torch.zeros((len(seqs), pad_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn_mask[i, : len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attn_mask}


def _make_tiny_albert() -> AlbertForSequenceClassification:
    cfg = AlbertConfig(
        vocab_size=128,
        embedding_size=16,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    torch.manual_seed(0)
    return AlbertForSequenceClassification(cfg)


def test_load_smsa_tsv_without_header(tmp_path: Path):
    tsv = tmp_path / "sample.tsv"
    tsv.write_text(
        "makanan enak sekali dan pelayanan ramah\tpositive\n"
        "biasa saja tidak terlalu spesial\tneutral\n"
        "kecewa berat pengiriman sangat lambat\tnegative\n",
        encoding="utf-8",
    )
    texts, labels = load_smsa_tsv(tsv)
    assert len(texts) == 3
    assert labels == [2, 1, 0]


def test_load_smsa_tsv_skips_optional_header_and_blank_lines(tmp_path: Path):
    tsv = tmp_path / "with_header.tsv"
    tsv.write_text(
        "text\tlabel\n\n"
        "produk bagus sekali\tPOSITIVE\n"
        "kurang memuaskan\tNegative\n",
        encoding="utf-8",
    )
    texts, labels = load_smsa_tsv(tsv)
    assert texts == ["produk bagus sekali", "kurang memuaskan"]
    assert labels == [2, 0]


def test_smsa_dataset_and_dynamic_collate_fn():
    ds = SmsaDataset(["halo dunia", "review restoran bandung yang sangat lezat sekali"], [1, 2])
    assert len(ds) == 2
    tok = _DummyIndonesianTokenizer()
    collate = make_dynamic_collate_fn(tok, max_length=16)
    batch = collate([ds[0], ds[1]])
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] <= 16
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].tolist() == [1, 2]


def test_tfidf_baseline_runs_and_scores_cleanly():
    train_texts = [
        "makanan enak sekali dan harga terjangkau",
        "pelayanan ramah tempat bersih sangat puas",
        "produk bagus kualitas mantap suka sekali",
        "biasa saja standar seperti umumnya",
        "lokasi di jalan utama buka setiap hari",
        "informasi jadwal kereta hari ini",
        "kecewa berat makanan dingin dan kotor",
        "pelayanan buruk sangat lambat tidak ramah",
        "rusak parah sangat menyesal beli disini",
    ] * 4
    train_labels = [2, 2, 2, 1, 1, 1, 0, 0, 0] * 4
    test_texts = [
        "makanan enak sekali sangat puas",
        "informasi jadwal hari ini standar",
        "pelayanan buruk sangat kecewa rusak",
    ]
    test_labels = [2, 1, 0]

    res = train_and_eval_tfidf_baseline(train_texts, train_labels, test_texts, test_labels, seed=42)
    assert res["accuracy"] >= 0.66
    assert set(res["per_class_f1"].keys()) == {"negative", "neutral", "positive"}
    assert len(res["confusion_matrix"]) == 3


def test_compute_multiclass_metrics_perfect():
    m = compute_multiclass_metrics([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_linear_warmup_scheduler_profile():
    param = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([param], lr=1e-3)
    sched = build_linear_warmup_scheduler(opt, num_warmup_steps=2, num_training_steps=10)
    lrs = []
    for _ in range(10):
        opt.step()
        lrs.append(sched.get_last_lr()[0])
        sched.step()
    assert lrs[0] < lrs[1]
    assert lrs[-1] < lrs[2]


def test_offline_albert_training_step_and_grad_clip():
    model = _make_tiny_albert()
    tok = _DummyIndonesianTokenizer()
    collate = make_dynamic_collate_fn(tok, max_length=16)
    batch = collate([("sangat enak mantap", 2), ("biasa saja", 1), ("kecewa buruk", 0)])

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    assert torch.isfinite(out.loss)
    out.loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert float(total_norm) > 0.0
    opt.step()


def test_predict_texts_output_schema_and_probabilities():
    model = _make_tiny_albert()
    tok = _DummyIndonesianTokenizer()
    preds = predict_texts(
        model,
        tok,
        ["warung kopi nyaman banget", "pengiriman telat parah"],
        device="cpu",
        max_length=16,
    )
    assert len(preds) == 2
    for p in preds:
        assert p["predicted_label"] in {"negative", "neutral", "positive"}
        prob_sum = sum(p["probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-2
        assert 0.0 <= p["confidence"] <= 1.0


def test_predict_texts_empty_returns_empty():
    model = _make_tiny_albert()
    tok = _DummyIndonesianTokenizer()
    assert predict_texts(model, tok, [], device="cpu") == []


def test_label_bijection_consistency():
    for label, idx in LABEL2ID.items():
        assert ID2LABEL[idx] == label


def test_plot_indobert_evaluation_writes_png(tmp_path: Path):
    out = tmp_path / "indobert_eval.png"
    m_bert = {
        "accuracy": 0.91,
        "macro_f1": 0.89,
        "per_class_f1": {"negative": 0.90, "neutral": 0.84, "positive": 0.93},
        "confusion_matrix": [[180, 10, 14], [12, 68, 8], [6, 5, 197]],
    }
    m_tfidf = {
        "accuracy": 0.82,
        "macro_f1": 0.78,
        "per_class_f1": {"negative": 0.81, "neutral": 0.67, "positive": 0.86},
        "confusion_matrix": [[160, 20, 24], [20, 50, 18], [15, 12, 181]],
    }
    samples = [
        {
            "text": "makanan di restoran ini enak banget dan pelayanan super ramah",
            "true_label": "positive",
            "predicted_label": "positive",
            "probabilities": {"negative": 0.01, "neutral": 0.04, "positive": 0.95},
        },
        {
            "text": "kecewa banget barang datang rusak dan penjual tidak responsif",
            "true_label": "negative",
            "predicted_label": "negative",
            "probabilities": {"negative": 0.93, "neutral": 0.05, "positive": 0.02},
        },
    ]
    saved = plot_indobert_evaluation(out, m_bert, m_tfidf, samples)
    assert saved.exists() and saved.stat().st_size > 1000
