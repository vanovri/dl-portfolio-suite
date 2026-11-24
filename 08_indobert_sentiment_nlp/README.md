# 08 - Indonesian Sentiment Analysis (IndoBERT-Lite on IndoNLU SmSA)

End-to-end fine-tuning of pretrained **`indobenchmark/indobert-lite-base-p1`** (ALBERT-base Indonesian language model) on the official **IndoNLU SmSA (`smsa_doc-sentiment-prosa`)** benchmark with a direct classical **TF-IDF + LogisticRegression** comparison on the identical official test set.

## Pipeline & Architecture

1. **Official Dataset (`src/dataset.py`)**:
   - Downloads and caches `train_preprocess.tsv`, `valid_preprocess.tsv`, and `test_preprocess.tsv` directly from `IndoNLP/indonlu` with exponential retry logic.
   - 3-class sentiment schema: `negative`, `neutral`, `positive`, dynamically padded per batch up to `max_length=128`.
2. **Pretrained Transformer (`src/model.py`)**:
   - `indobenchmark/indobert-lite-base-p1` fine-tuned with AdamW (`lr=3e-5`), 10% linear warmup + decay, gradient norm clipping (`1.0`), and Turing-safe FP16 AMP (`GradScaler`).
   - Includes reusable `predict_texts(model, tokenizer, texts)` helper returning predicted class, confidence, and full class probability distribution.
3. **Classical Baseline (`src/baseline.py`)**:
   - `TfidfVectorizer(ngram_range=(1, 2), max_features=20000)` + `LogisticRegression(max_iter=500, class_weight="balanced")` evaluated on the exact same official 500-row `test_preprocess.tsv`.

## Reproduction

```bash
# Run offline CPU unit tests
pytest tests/

# Fine-tune IndoBERT-Lite and benchmark on official SmSA test split
python train.py --epochs 3 --batch-size 32 --seed 42
```

## Outputs

- `reports/metrics.json` -- Test Accuracy, Macro-F1, Per-Class F1, IndoBERT vs TF-IDF gain, inference latency (`ms/text`), and peak VRAM telemetry.
- `reports/indobert_eval.png` -- Confusion matrix, per-class F1 bar comparison, and sample Indonesian review predictions with probabilities.
- `artifacts/indobert_smsa_best.pt` -- Best validation checkpoint (gitignored).
