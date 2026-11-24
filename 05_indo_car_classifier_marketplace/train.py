"""Training, FastAPI latency benchmarking, and gallery generation for Project 05."""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi.testclient import TestClient
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
for _p in (str(REPO_ROOT), str(PROJECT_ROOT)):
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
from src.api import create_app
from src.data import STANFORD_CARS_REPO_ID, CachedCarDataset, prepare_stanford_cars_cache
from src.marketplace import estimate_idr_valuation
from src.model import CarPredictor, build_car_classifier
from src.viz import plot_car_predictions_gallery


def top_k_acc(y_true: np.ndarray, y_prob: np.ndarray, k: int = 3) -> float:
    k_eff = min(k, y_prob.shape[1])
    topk = np.argsort(-y_prob, axis=1)[:, :k_eff]
    return float(np.mean([int(y_true[i] in topk[i]) for i in range(len(y_true))])) if len(y_true) else 0.0


@torch.no_grad()
def evaluate_cars(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Top-1, Top-3 accuracy, and Macro-F1."""
    model.eval()
    probs_list: list[np.ndarray] = []
    preds_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        with amp_context(device, enabled=use_amp):
            logits = model(imgs)
            probs = F.softmax(logits.float(), dim=1)
        probs_list.append(probs.cpu().numpy())
        preds_list.append(probs.argmax(dim=1).cpu().numpy())
        targets_list.append(targets.numpy())

    y_prob = np.concatenate(probs_list, axis=0)
    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(targets_list, axis=0)

    metrics = {
        "top1_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "top3_accuracy": round(top_k_acc(y_true, y_prob, k=3), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }
    return metrics, y_true, y_pred, y_prob


def benchmark_fastapi_latency(
    predictor: CarPredictor,
    test_ds: CachedCarDataset,
    n_requests: int = 25,
) -> dict[str, float]:
    """Benchmark end-to-end FastAPI POST /predict latency (p50 and p95 ms)."""
    app = create_app(predictor=predictor)
    client = TestClient(app)

    # Encode sample test images once so measurements reflect realistic HTTP + decode + inference
    encoded_imgs: list[bytes] = []
    count = min(n_requests, len(test_ds))
    for i in range(count):
        buf = io.BytesIO()
        Image.fromarray(test_ds.load_raw_rgb(i)).save(buf, format="JPEG", quality=88)
        encoded_imgs.append(buf.getvalue())

    # Warmup
    if encoded_imgs:
        resp = client.post("/predict", content=encoded_imgs[0], headers={"content-type": "image/jpeg"})
        resp.raise_for_status()

    latencies_ms: list[float] = []
    for payload in encoded_imgs:
        t0 = time.perf_counter()
        resp = client.post("/predict", content=payload, headers={"content-type": "image/jpeg"})
        resp.raise_for_status()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "api_p50_ms": round(float(np.percentile(arr, 50)), 2) if len(arr) else 0.0,
        "api_p95_ms": round(float(np.percentile(arr, 95)), 2) if len(arr) else 0.0,
        "api_requests_benchmarked": int(len(arr)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Indonesian Car Classifier & Benchmark Marketplace API.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    use_amp = (not args.no_amp) and device.type == "cuda"

    cache_dir = PROJECT_ROOT / "data" / "cars_cache"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    reports_dir = PROJECT_ROOT / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[05_indo_car_classifier_marketplace] Device: {device} | AMP(FP16): {use_amp}")
    manifest = prepare_stanford_cars_cache(
        cache_dir=cache_dir,
        num_classes=args.num_classes,
        img_size=args.img_size,
        seed=args.seed,
    )
    class_names = manifest["class_names"]
    train_ds = CachedCarDataset(cache_dir, manifest["splits"]["train"], img_size=args.img_size, is_train=True)
    val_ds = CachedCarDataset(cache_dir, manifest["splits"]["val"], img_size=args.img_size, is_train=False)
    test_ds = CachedCarDataset(cache_dir, manifest["splits"]["test"], img_size=args.img_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_car_classifier(num_classes=len(class_names), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    reset_peak_vram()
    ckpt_path = artifacts_dir / "efficientnet_b0_indo_cars.pt"
    best_acc = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []

    with timed_block() as timer:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            for imgs, targets in train_loader:
                imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
                targets = targets.to(device, non_blocking=(device.type == "cuda"))
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, enabled=use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.item()) * len(targets)
                seen += len(targets)
            scheduler.step()

            val_metrics, _, _, _ = evaluate_cars(model, val_loader, device, use_amp=use_amp)
            train_loss = running_loss / max(seen, 1)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "val_top1_acc": val_metrics["top1_accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                }
            )
            print(
                f"Epoch {epoch:02d}/{args.epochs} | loss={train_loss:.4f} | "
                f"val_top1={val_metrics['top1_accuracy']:.4f} | val_top3={val_metrics['top3_accuracy']:.4f}"
            )
            if val_metrics["top1_accuracy"] >= best_acc:
                best_acc = val_metrics["top1_accuracy"]
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_names": class_names,
                        "img_size": args.img_size,
                        "epoch": epoch,
                    },
                    ckpt_path,
                )

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    test_metrics, y_true, y_pred, y_prob = evaluate_cars(model, test_loader, device, use_amp=use_amp)

    predictor = CarPredictor(model=model, class_names=class_names, img_size=args.img_size, device=device)
    api_bench = benchmark_fastapi_latency(predictor, test_ds, n_requests=25)

    # Pick 8 diverse held-out car photos across classes
    gallery_samples: list[dict[str, Any]] = []
    seen_cls: set[int] = set()
    for idx in range(len(test_ds)):
        c = int(y_true[idx])
        if c not in seen_cls:
            seen_cls.add(c)
            pred_c = int(y_pred[idx])
            pred_name = class_names[pred_c]
            val_info = estimate_idr_valuation(pred_name, odometer_km=52_000, condition="good")
            gallery_samples.append(
                {
                    "rgb": test_ds.load_raw_rgb(idx),
                    "true_label": class_names[c],
                    "pred_label": pred_name,
                    "confidence": float(y_prob[idx, pred_c]),
                    "idr_range": val_info["formatted_range"],
                }
            )
        if len(gallery_samples) >= 8:
            break

    gallery_path = plot_car_predictions_gallery(gallery_samples, reports_dir / "car_predictions_gallery.png")
    vram = get_vram_stats()

    report = {
        "project": "05_indo_car_classifier_marketplace",
        "dataset_type": "real_public",
        "dataset_name": "Stanford Cars 10-Class Indonesian-Popular Brand Subset (160x160 RGB)",
        "dataset_provenance": f"https://huggingface.co/datasets/{STANFORD_CARS_REPO_ID}",
        "class_names": class_names,
        "splits": {
            "train": int(len(train_ds)),
            "val": int(len(val_ds)),
            "test": int(len(test_ds)),
            "img_size": args.img_size,
        },
        "model": {
            "architecture": "EfficientNet-B0",
            "pretrained": not args.no_pretrained,
            **count_parameters(model),
        },
        "training": {
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_val_top1_acc": round(best_acc, 4),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "amp_fp16": use_amp,
            "elapsed_s": timer["elapsed_s"],
            "history_tail": history[-5:],
        },
        "test_metrics": {**test_metrics, **api_bench},
        "hardware": vram,
        "artifacts": {
            "checkpoint": ckpt_path.relative_to(PROJECT_ROOT).as_posix(),
            "car_predictions_gallery": gallery_path.relative_to(PROJECT_ROOT).as_posix(),
        },
    }
    write_json_report(reports_dir / "metrics.json", report)
    print(
        f"[05_indo_car_classifier_marketplace] Done! Top-1={test_metrics['top1_accuracy']:.4f} | "
        f"Top-3={test_metrics['top3_accuracy']:.4f} | API p50={api_bench['api_p50_ms']} ms"
    )
    cleanup_cuda()


if __name__ == "__main__":
    main()
