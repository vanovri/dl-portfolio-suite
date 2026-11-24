"""Fine-tune IndoBERT-Lite on official IndoNLU SmSA and benchmark against TF-IDF + LogReg."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
for _p in (str(REPO_ROOT), str(PROJECT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.cuda_utils import (
    amp_context,
    cleanup_cuda,
    count_parameters,
    get_device,
    get_vram_stats,
    reset_peak_vram,
    seed_everything,
    timed_block,
    write_json_report,
)
from src.baseline import compute_multiclass_metrics, train_and_eval_tfidf_baseline
from src.dataset import (
    ID2LABEL,
    INDONLU_SMSA_URLS,
    SmsaDataset,
    download_smsa_dataset,
    load_smsa_tsv,
    make_dynamic_collate_fn,
)
from src.model import (
    DEFAULT_MODEL_NAME,
    build_linear_warmup_scheduler,
    load_indobert_model_and_tokenizer,
    predict_texts,
)
from src.viz import plot_indobert_evaluation


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[int], list[int], float]:
    """Evaluate sequence classifier over a DataLoader and measure average latency ms/text."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    use_amp = device.type == "cuda"

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True)
        model_inputs = {
            k: v.to(device, non_blocking=True)
            for k, v in batch.items()
            if k in ("input_ids", "attention_mask", "token_type_ids")
        }
        with amp_context(device, enabled=use_amp):
            outputs = model(**model_inputs, labels=labels)
        bs = labels.size(0)
        total_loss += float(outputs.loss.item()) * bs
        total_n += bs
        preds = outputs.logits.float().argmax(dim=-1).cpu().tolist()
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    latency_ms = elapsed_ms / max(total_n, 1)
    return total_loss / max(total_n, 1), y_true, y_pred, latency_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune IndoBERT on IndoNLU SmSA.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-train-samples", type=int, default=3600, help="0 uses full 11k train split; default 3600 stratified for 4GB VRAM speed.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(prefer_cuda=not args.no_cuda)
    reset_peak_vram()

    data_dir = PROJECT_DIR / "data"
    artifacts_dir = PROJECT_DIR / "artifacts"
    reports_dir = PROJECT_DIR / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    tsv_paths = download_smsa_dataset(data_dir)
    train_texts, train_labels = load_smsa_tsv(tsv_paths["train"])
    val_texts, val_labels = load_smsa_tsv(tsv_paths["valid"])
    test_texts, test_labels = load_smsa_tsv(tsv_paths["test"])

    if args.max_train_samples > 0 and args.max_train_samples < len(train_texts):
        from sklearn.model_selection import train_test_split
        train_texts, _, train_labels, _ = train_test_split(
            train_texts,
            train_labels,
            train_size=args.max_train_samples,
            stratify=train_labels,
            random_state=args.seed,
        )

    # Classical TF-IDF + LogisticRegression baseline on exact same train & test splits
    tfidf_metrics = train_and_eval_tfidf_baseline(
        train_texts,
        train_labels,
        test_texts,
        test_labels,
        seed=args.seed,
    )

    model, tokenizer = load_indobert_model_and_tokenizer(args.model_name, num_labels=3)
    model = model.to(device)
    param_info = count_parameters(model)

    collate_fn = make_dynamic_collate_fn(tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        SmsaDataset(train_texts, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        SmsaDataset(val_texts, val_labels),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        SmsaDataset(test_texts, test_labels),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * max(args.epochs, 1)
    warmup_steps = int(0.10 * total_steps)
    scheduler = build_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_val_f1 = -1.0
    best_ckpt = artifacts_dir / "indobert_smsa_best.pt"

    with timed_block() as train_timer:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            for batch in train_loader:
                labels = batch["labels"].to(device, non_blocking=True)
                model_inputs = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch.items()
                    if k in ("input_ids", "attention_mask", "token_type_ids")
                }
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, enabled=use_amp):
                    outputs = model(**model_inputs, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                bs = labels.size(0)
                running_loss += float(loss.item()) * bs
                seen += bs

            train_loss = running_loss / max(seen, 1)
            val_loss, v_true, v_pred, _ = evaluate_loader(model, val_loader, device)
            val_metrics = compute_multiclass_metrics(v_true, v_pred)
            history["train_loss"].append(round(train_loss, 5))
            history["val_loss"].append(round(val_loss, 5))
            history["val_macro_f1"].append(val_metrics["macro_f1"])
            print(
                f"  Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f}",
                flush=True,
            )

            if val_metrics["macro_f1"] >= best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_name": args.model_name,
                        "val_macro_f1": best_val_f1,
                        "model_state_dict": model.state_dict(),
                    },
                    best_ckpt,
                )

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, t_true, t_pred, test_latency_ms = evaluate_loader(model, test_loader, device)
    indobert_metrics = compute_multiclass_metrics(t_true, t_pred)
    indobert_metrics["test_loss"] = round(test_loss, 4)
    indobert_metrics["latency_ms_per_text"] = round(float(test_latency_ms), 4)
    vram_stats = get_vram_stats()

    # Representative sample predictions covering all 3 classes from official test_preprocess.tsv
    sample_indices: list[int] = []
    for target_cls in (2, 1, 0):
        for i, lbl in enumerate(test_labels):
            if lbl == target_cls and i not in sample_indices:
                sample_indices.append(i)
                if len(sample_indices) >= 2 and target_cls == 2:
                    break
                if len(sample_indices) >= 3 and target_cls == 1:
                    break
                if len(sample_indices) >= 5 and target_cls == 0:
                    break
    sample_texts = [test_texts[i] for i in sample_indices[:5]]
    sample_preds = predict_texts(model, tokenizer, sample_texts, device=device, max_length=args.max_length)
    for item, idx in zip(sample_preds, sample_indices[:5]):
        item["true_label"] = ID2LABEL[test_labels[idx]]

    plot_path = reports_dir / "indobert_eval.png"
    plot_indobert_evaluation(
        out_path=plot_path,
        indobert_metrics=indobert_metrics,
        tfidf_metrics=tfidf_metrics,
        sample_predictions=sample_preds,
    )

    report: dict[str, Any] = {
        "project": "08_indobert_sentiment_nlp",
        "dataset_type": "real_public",
        "dataset_provenance": {
            "benchmark": "IndoNLU SmSA (smsa_doc-sentiment-prosa)",
            "urls": INDONLU_SMSA_URLS,
            "split_sizes": {
                "train": len(train_texts),
                "valid": len(val_texts),
                "test": len(test_texts),
            },
            "labels": list(ID2LABEL.values()),
        },
        "hyperparameters": {
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "warmup_ratio": 0.10,
            "max_length": args.max_length,
            "gradient_clip": 1.0,
            "fp16_amp": use_amp,
            "seed": args.seed,
        },
        "model": {
            "architecture": args.model_name,
            **param_info,
            "best_val_macro_f1": round(best_val_f1, 4),
            "history": history,
        },
        "test_metrics_indobert": indobert_metrics,
        "test_metrics_tfidf_baseline": tfidf_metrics,
        "indobert_vs_tfidf_gain": {
            "accuracy_delta": round(indobert_metrics["accuracy"] - tfidf_metrics["accuracy"], 4),
            "macro_f1_delta": round(indobert_metrics["macro_f1"] - tfidf_metrics["macro_f1"], 4),
        },
        "sample_predictions": sample_preds,
        "runtime": {
            "train_elapsed_s": train_timer["elapsed_s"],
            **vram_stats,
        },
        "artifacts": {
            "checkpoint": "artifacts/indobert_smsa_best.pt",
            "figure": "reports/indobert_eval.png",
        },
    }
    write_json_report(reports_dir / "metrics.json", report)
    cleanup_cuda()
    print(
        f"[08_indobert_sentiment_nlp] Done in {train_timer['elapsed_s']}s | "
        f"Test Acc={indobert_metrics['accuracy']:.4f} | Macro-F1={indobert_metrics['macro_f1']:.4f} | "
        f"Gain vs TF-IDF={report['indobert_vs_tfidf_gain']['macro_f1_delta']:+.4f} | "
        f"Peak VRAM={vram_stats['peak_vram_mb']} MB"
    )


if __name__ == "__main__":
    main()
