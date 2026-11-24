"""Train & evaluate DualScaleHumanitiesNet vs TF-IDF baseline on Classical Poetry & Nusantara Pantun."""
from __future__ import annotations

import argparse
import copy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
for _p in (str(PROJECT_DIR), str(REPO_ROOT), str(Path(__file__).resolve().parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from common.cuda_utils import (
    amp_context,
    cleanup_cuda,
    configure_plot_style,
    count_parameters,
    get_device,
    get_vram_stats,
    reset_peak_vram,
    seed_everything,
    timed_block,
    write_json_report,
)
from src.dataset import (
    ERA_NAMES,
    THEME_NAMES,
    HumanitiesTextDataset,
    WordCharTokenizer,
    load_nusantara_pantun_dataset,
    load_world_poetry_dataset,
)
from src.model import DualScaleHumanitiesNet
from src.viz import generate_humanities_stylometry_figure


def evaluate_tfidf_baseline(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    num_classes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Word (1-2) + Character (3-5) TF-IDF Linear Baseline on identical train/test splits."""
    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=6000, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=8000, sublinear_tf=True)

    X_tr = hstack([word_vec.fit_transform(train_texts), char_vec.fit_transform(train_texts)])
    X_te = hstack([word_vec.transform(test_texts), char_vec.transform(test_texts)])

    clf = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, train_labels)

    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te)

    acc = float(accuracy_score(test_labels, preds))
    macro_f1 = float(f1_score(test_labels, preds, average="macro", zero_division=0))

    extras: dict[str, Any] = {"probs": probs}
    if num_classes == 2 and len(np.unique(test_labels)) == 2:
        pos_probs = probs[:, 1]
        roc_auc = float(roc_auc_score(test_labels, pos_probs))
        fpr, tpr, _ = roc_curve(test_labels, pos_probs)
        extras.update({"fpr": fpr, "tpr": tpr, "auc": round(roc_auc, 4)})
    elif len(np.unique(test_labels)) > 2:
        roc_auc = float(roc_auc_score(test_labels, probs, multi_class="ovr", average="macro"))
    else:
        roc_auc = 0.5

    metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "roc_auc": round(roc_auc, 4),
    }
    return metrics, extras


def _build_class_weights(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    inv = counts.sum() / (num_classes * counts)
    return torch.tensor(inv, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate_neural_model(
    model: DualScaleHumanitiesNet,
    dataset: HumanitiesTextDataset,
    batch_size: int,
    device: torch.device,
    class_names: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate DualScaleHumanitiesNet and aggregate close-reading token attention salience per class."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits: list[np.ndarray] = []
    all_labels: list[int] = []
    token_scores_by_class: dict[str, dict[str, list[float]]] = {
        cname: defaultdict(list) for cname in class_names
    }

    offset = 0
    for word_ids, char_ids, suffix_chars, labels in loader:
        word_ids = word_ids.to(device)
        char_ids = char_ids.to(device)
        suffix_chars = suffix_chars.to(device)
        details = model(
            word_ids, char_ids, line_suffix_chars=suffix_chars, return_details=True
        )
        logits = details["logits"]
        attn = details["token_attention"].detach().cpu().numpy()  # (B, L_w)

        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.extend(labels.numpy().tolist())

        bsz = labels.size(0)
        for i in range(bsz):
            sample_idx = offset + i
            tokens = dataset.get_tokens(sample_idx)
            cname = class_names[int(preds[i])]
            n_tok = min(len(tokens), attn.shape[1])
            for t_idx in range(n_tok):
                tok = tokens[t_idx]
                if len(tok) >= 3 and tok.isalpha():
                    token_scores_by_class[cname][tok].append(float(attn[i, t_idx]))
        offset += bsz

    logits_np = np.concatenate(all_logits, axis=0)
    # Numerically stable softmax
    shifted = logits_np - logits_np.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds_all = probs.argmax(axis=1)
    y_true = np.asarray(all_labels, dtype=np.int64)

    acc = float(accuracy_score(y_true, preds_all))
    macro_f1 = float(f1_score(y_true, preds_all, average="macro", zero_division=0))

    num_classes = len(class_names)
    extras: dict[str, Any] = {"probs": probs}
    if num_classes == 2 and len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, probs[:, 1]))
        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
        extras.update({"fpr": fpr, "tpr": tpr, "auc": round(roc_auc, 4)})
    elif len(np.unique(y_true)) > 2:
        roc_auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    else:
        roc_auc = 0.5

    # Extract top salient tokens per class (require >=2 occurrences when corpus permits for stability).
    top_tokens: dict[str, list[tuple[str, float]]] = {}
    for cname, tok_map in token_scores_by_class.items():
        ranked = [
            (tok, float(np.mean(scores)))
            for tok, scores in tok_map.items()
            if len(scores) >= 2
        ]
        if len(ranked) < 6:
            ranked = [(tok, float(np.mean(scores))) for tok, scores in tok_map.items()]
        ranked.sort(key=lambda kv: kv[1], reverse=True)
        top_tokens[cname] = [(t, round(s, 4)) for t, s in ranked[:12]]

    extras["top_tokens"] = top_tokens
    metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "roc_auc": round(roc_auc, 4),
    }
    return metrics, extras


def train_humanities_task(
    task_name: str,
    splits: dict[str, Any],
    class_names: list[str],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    max_words: int,
    max_chars: int,
    use_amp: bool,
    seed: int,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train DualScaleHumanitiesNet + TF-IDF baseline on one humanities task."""
    seed_everything(seed)
    tr_texts, tr_labels = splits["train_texts"], splits["train_labels"]
    va_texts, va_labels = splits["val_texts"], splits["val_labels"]
    te_texts, te_labels = splits["test_texts"], splits["test_labels"]

    # 1. TF-IDF Char+Word Linear Baseline
    tfidf_metrics, tfidf_extras = evaluate_tfidf_baseline(
        tr_texts, tr_labels, te_texts, te_labels, num_classes=len(class_names)
    )

    # 2. Fit tokenizer strictly on training split (zero vocab leakage!)
    tokenizer = WordCharTokenizer(max_word_vocab=8000, min_word_freq=1)
    tokenizer.fit(tr_texts)

    train_ds = HumanitiesTextDataset(tr_texts, tr_labels, tokenizer, max_words=max_words, max_chars=max_chars)
    val_ds = HumanitiesTextDataset(va_texts, va_labels, tokenizer, max_words=max_words, max_chars=max_chars)
    test_ds = HumanitiesTextDataset(te_texts, te_labels, tokenizer, max_words=max_words, max_chars=max_chars)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = DualScaleHumanitiesNet(
        word_vocab_size=tokenizer.word_vocab_size,
        char_vocab_size=tokenizer.char_vocab_size,
        num_classes=len(class_names),
    ).to(device)

    class_weights = _build_class_weights(tr_labels, len(class_names), device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    best_val_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())

    with timed_block() as timer:
        for _ in range(1, epochs + 1):
            model.train()
            for word_ids, char_ids, suffix_chars, labels in train_loader:
                word_ids = word_ids.to(device)
                char_ids = char_ids.to(device)
                suffix_chars = suffix_chars.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, enabled=use_amp):
                    logits = model(word_ids, char_ids, line_suffix_chars=suffix_chars)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()
            val_metrics, _ = evaluate_neural_model(model, val_ds, batch_size, device, class_names)
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    neural_metrics, neural_extras = evaluate_neural_model(model, test_ds, batch_size, device, class_names)
    neural_metrics["val_best_macro_f1"] = round(best_val_f1, 4)
    neural_metrics["train_seconds"] = timer["elapsed_s"]
    neural_metrics["params"] = count_parameters(model)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": task_name,
            "class_names": class_names,
            "state_dict": model.state_dict(),
            "word_vocab_size": tokenizer.word_vocab_size,
            "char_vocab_size": tokenizer.char_vocab_size,
            "metrics": neural_metrics,
        },
        checkpoint_path,
    )

    print(
        f"  [{task_name:14s}] Neural: acc={neural_metrics['accuracy']:.4f} f1={neural_metrics['macro_f1']:.4f} "
        f"auc={neural_metrics['roc_auc']:.4f} | TF-IDF: acc={tfidf_metrics['accuracy']:.4f} "
        f"f1={tfidf_metrics['macro_f1']:.4f} auc={tfidf_metrics['roc_auc']:.4f}"
    )

    result = {
        "n_train": len(tr_texts),
        "n_val": len(va_texts),
        "n_test": len(te_texts),
        "class_names": class_names,
        "dual_scale_neural": neural_metrics,
        "tfidf_char_word_baseline": tfidf_metrics,
        "top_salient_tokens": {
            k: [{"token": t, "attention": s} for t, s in v[:8]]
            for k, v in neural_extras["top_tokens"].items()
        },
    }
    extras = {
        "neural": neural_extras,
        "tfidf": tfidf_extras,
    }
    del model
    cleanup_cuda()
    return result, extras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project 13: Classical Literature Stylometry & Nusantara Pantun Coherence DL"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-pantun-stanzas", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_DIR / "reports")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_DIR / "artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    configure_plot_style()

    prefer_cuda = args.device != "cpu"
    device = get_device(prefer_cuda=prefer_cuda)
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested via --device cuda, but no GPU is available.")
    use_amp = (not args.no_amp) and (device.type == "cuda")

    reset_peak_vram()
    print(f"[Project 13] Device={device} | AMP={use_amp} | Epochs={args.epochs} | Batch={args.batch_size}")

    poetry_data = load_world_poetry_dataset(args.data_dir, seed=args.seed)
    pantun_data = load_nusantara_pantun_dataset(
        args.data_dir, max_maercaestro=args.max_pantun_stanzas, seed=args.seed
    )
    print(
        f"  Track A Poetry: {poetry_data['n_poems']} deduplicated poems | "
        f"Track B Pantun: {pantun_data['n_authentic_stanzas']} authentic stanzas "
        f"(maercaestro={pantun_data['source_counts']['maercaestro']}, lorthgyu={pantun_data['source_counts']['lorthgyu']})"
    )

    with timed_block() as total_timer:
        print("\n--- Track A1: Classical Poetry Literary Era (Renaissance vs Modern) ---")
        era_res, era_extras = train_humanities_task(
            task_name="poetry_era",
            splits=poetry_data["era"],
            class_names=ERA_NAMES,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_words=160,
            max_chars=600,
            use_amp=use_amp,
            seed=args.seed,
            checkpoint_path=args.artifacts_dir / "dual_scale_era.pt",
        )

        print("\n--- Track A2: Classical Poetry Thematic Genre (Love / Nature / Mythology) ---")
        theme_res, _ = train_humanities_task(
            task_name="poetry_theme",
            splits=poetry_data["theme"],
            class_names=THEME_NAMES,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_words=160,
            max_chars=600,
            use_amp=use_amp,
            seed=args.seed + 10,
            checkpoint_path=args.artifacts_dir / "dual_scale_theme.pt",
        )

        print("\n--- Track B: Nusantara Pantun Sampiran-Isi & Rhyme Coherence ---")
        pantun_classes = ["Permuted_Incoherent", "Authentic_Pantun"]
        pantun_res, pantun_extras = train_humanities_task(
            task_name="pantun_coherence",
            splits=pantun_data["coherence"],
            class_names=pantun_classes,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_words=64,
            max_chars=280,
            use_amp=use_amp,
            seed=args.seed + 20,
            checkpoint_path=args.artifacts_dir / "dual_scale_pantun.pt",
        )

    comparison_scores = {
        "era": {
            "neural_macro_f1": era_res["dual_scale_neural"]["macro_f1"],
            "tfidf_macro_f1": era_res["tfidf_char_word_baseline"]["macro_f1"],
        },
        "theme": {
            "neural_macro_f1": theme_res["dual_scale_neural"]["macro_f1"],
            "tfidf_macro_f1": theme_res["tfidf_char_word_baseline"]["macro_f1"],
        },
        "pantun": {
            "neural_macro_f1": pantun_res["dual_scale_neural"]["macro_f1"],
            "tfidf_macro_f1": pantun_res["tfidf_char_word_baseline"]["macro_f1"],
        },
    }
    pantun_roc = {
        "neural": pantun_extras["neural"],
        "tfidf": pantun_extras["tfidf"],
    }
    fig_path = generate_humanities_stylometry_figure(
        output_path=args.reports_dir / "humanities_stylometry.png",
        comparison_scores=comparison_scores,
        pantun_roc=pantun_roc,
        era_top_tokens=era_extras["neural"]["top_tokens"],
        pantun_top_tokens=pantun_extras["neural"]["top_tokens"].get("Authentic_Pantun", []),
    )

    vram_stats = get_vram_stats()
    report = {
        "project": "13_humanities_nusantara_dl",
        "dataset_type": "real_public",
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_pantun_stanzas": args.max_pantun_stanzas,
            "seed": args.seed,
            "amp_fp16": use_amp,
        },
        "hardware": {
            **vram_stats,
            "total_wall_seconds": total_timer["elapsed_s"],
        },
        "track_a_classical_literature": {
            "source": "HuggingFace merve/poetry",
            "n_deduplicated_poems": poetry_data["n_poems"],
            "era_classification": era_res,
            "theme_classification": theme_res,
        },
        "track_b_nusantara_pantun": {
            "sources": ["HuggingFace maercaestro/pantun", "HuggingFace LorthGyu/indonesian-pantun"],
            "n_authentic_stanzas": pantun_data["n_authentic_stanzas"],
            "source_counts": pantun_data["source_counts"],
            "coherence_rhyme_benchmark": pantun_res,
        },
        "artifacts": {
            "era_checkpoint": "artifacts/dual_scale_era.pt",
            "theme_checkpoint": "artifacts/dual_scale_theme.pt",
            "pantun_checkpoint": "artifacts/dual_scale_pantun.pt",
            "figure": f"reports/{fig_path.name}",
        },
    }
    report_path = write_json_report(args.reports_dir / "metrics.json", report)
    print(f"\n[Project 13] Saved report -> {report_path}")
    print(f"[Project 13] Saved figure -> {fig_path}")
    print(f"[Project 13] Peak VRAM -> {vram_stats['peak_vram_mb']} MB")


if __name__ == "__main__":
    main()
