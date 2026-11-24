"""Pretrained IndoBERT-Lite sequence classification wrapper and inference helper."""
from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer

from src.dataset import ID2LABEL, LABEL2ID

DEFAULT_MODEL_NAME = "indobenchmark/indobert-base-p1"


def load_indobert_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    num_labels: int = 3,
) -> tuple[Any, Any]:
    """Load pretrained IndoBERT tokenizer (WordPiece-safe) and sequence classifier head."""
    if "indobert-lite" in model_name:
        # Guard against transformers 5.x AlbertTokenizer loading vocab.txt as 100% [UNK].
        tokenizer = BertTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model, tokenizer


def build_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Version-independent linear warmup + linear decay LR scheduler."""
    total = max(int(num_training_steps), 1)
    warmup = max(0, min(int(num_warmup_steps), total))

    def _lr_lambda(current_step: int) -> float:
        if warmup > 0 and current_step < warmup:
            return float(current_step + 1) / float(warmup)
        denom = max(total - warmup, 1)
        return max(0.0, float(total - current_step) / float(denom))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


@torch.no_grad()
def predict_texts(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    device: torch.device | str | None = None,
    max_length: int = 128,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Run batched sentiment inference on raw Indonesian texts and return calibrated probabilities."""
    if not texts:
        return []
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()
    results: list[dict[str, Any]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        model_inputs = {
            k: v.to(device)
            for k, v in encoded.items()
            if k in ("input_ids", "attention_mask", "token_type_ids")
        }
        outputs = model(**model_inputs)
        logits = outputs.logits.float()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for text, prob_vec in zip(chunk, probs):
            pred_id = int(prob_vec.argmax())
            results.append(
                {
                    "text": text,
                    "predicted_id": pred_id,
                    "predicted_label": ID2LABEL[pred_id],
                    "confidence": round(float(prob_vec[pred_id]), 4),
                    "probabilities": {
                        ID2LABEL[i]: round(float(prob_vec[i]), 4) for i in range(3)
                    },
                }
            )
    return results
