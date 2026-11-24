"""Train and ablate 4 multimodal architectures on real paired ESC-50 + CIFAR-10 data."""
from __future__ import annotations

import argparse
import sys
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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
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
    CLASS_NAMES,
    apply_modality_corruption,
    build_real_paired_datasets,
)
from src.model import GatedCrossAttentionFusionNet, build_model
from src.viz import generate_multimodal_ablation_figure

VARIANTS = ("image_only", "audio_only", "early_concat", "gated_cross_attention")
EVAL_MODES = ("clean", "visual_corrupt", "audio_corrupt", "mixed_corrupt")


def compute_training_loss(
    model: nn.Module,
    images: torch.Tensor,
    specs: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute primary CE loss plus auxiliary unimodal & gate reliability calibration for GatedCrossAttention."""
    if isinstance(model, GatedCrossAttentionFusionNet):
        out = model(images, specs, return_details=True)
        main_loss = criterion(out["logits"], labels)
        drop_state = out["drop_state"]

        # Train auxiliary heads on uncorrupted samples of each modality so neither branch collapses.
        img_clean = drop_state != 1
        aud_clean = drop_state != 2
        aux_loss = torch.tensor(0.0, device=images.device, dtype=main_loss.dtype)
        if img_clean.any():
            aux_loss = aux_loss + 0.30 * criterion(out["img_aux_logits"][img_clean], labels[img_clean])
        if aud_clean.any():
            aux_loss = aux_loss + 0.30 * criterion(out["aud_aux_logits"][aud_clean], labels[aud_clean])

        # Soft reliability calibration for stochastically corrupted training samples.
        alpha = out["alpha"].squeeze(-1).float()
        gate_loss = torch.tensor(0.0, device=images.device, dtype=torch.float32)
        vis_bad = drop_state == 1
        aud_bad = drop_state == 2
        if vis_bad.any():
            gate_loss = gate_loss + F.mse_loss(alpha[vis_bad], torch.full_like(alpha[vis_bad], 0.12))
        if aud_bad.any():
            gate_loss = gate_loss + F.mse_loss(alpha[aud_bad], torch.full_like(alpha[aud_bad], 0.88))

        return main_loss + aux_loss + 0.35 * gate_loss.to(main_loss.dtype)

    logits = model(images, specs, return_details=False)
    return criterion(logits, labels)


def train_single_variant(
    variant: str,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    use_amp: bool,
    seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train one architecture variant and return model + training telemetry."""
    seed_everything(seed)
    model = build_model(variant, num_classes=len(CLASS_NAMES), embed_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    history: list[dict[str, float]] = []
    with timed_block() as timer:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            n_seen = 0
            for images, specs, labels in train_loader:
                images = images.to(device, non_blocking=True)
                specs = specs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, enabled=use_amp):
                    loss = compute_training_loss(model, images, specs, labels, criterion)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()

                bsz = labels.size(0)
                running_loss += float(loss.item()) * bsz
                n_seen += bsz

            scheduler.step()
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": round(running_loss / max(1, n_seen), 4),
                    "lr": round(float(scheduler.get_last_lr()[0]), 6),
                }
            )

    telemetry = {
        "params": count_parameters(model),
        "train_seconds": timer["elapsed_s"],
        "final_train_loss": history[-1]["train_loss"] if history else 0.0,
        "history": history,
    }
    return model, telemetry


@torch.no_grad()
def evaluate_variant(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    mode: str,
    seed: int,
) -> tuple[dict[str, Any], list[float]]:
    """Evaluate model on held-out test set under a specified modality corruption mode."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_alphas: list[float] = []
    vis_bad_alphas: list[float] = []
    aud_bad_alphas: list[float] = []

    for batch_idx, (images, specs, labels) in enumerate(test_loader):
        images = images.to(device, non_blocking=True)
        specs = specs.to(device, non_blocking=True)
        c_imgs, c_specs, c_flags = apply_modality_corruption(
            images, specs, mode=mode, seed=seed + batch_idx * 97
        )
        details = model(c_imgs, c_specs, return_details=True)
        logits = details["logits"]
        alphas = details["alpha"].squeeze(-1).detach().cpu().numpy()
        preds = logits.argmax(dim=-1).detach().cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_alphas.extend(alphas.tolist())

        flags_np = c_flags.detach().cpu().numpy()
        for a_val, fl in zip(alphas.tolist(), flags_np.tolist()):
            if fl == 1:
                vis_bad_alphas.append(a_val)
            elif fl == 2:
                aud_bad_alphas.append(a_val)

    acc = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    alpha_arr = np.asarray(all_alphas, dtype=np.float32)

    result: dict[str, Any] = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "mean_alpha": round(float(alpha_arr.mean()), 4),
        "std_alpha": round(float(alpha_arr.std()), 4),
    }
    if vis_bad_alphas:
        result["mean_alpha_when_visual_corrupt"] = round(float(np.mean(vis_bad_alphas)), 4)
    if aud_bad_alphas:
        result["mean_alpha_when_audio_corrupt"] = round(float(np.mean(aud_bad_alphas)), 4)
    return result, all_alphas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project 11: Multimodal Image (CIFAR-10) + Audio (ESC-50) Fusion Ablation"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-pairs-per-class", type=int, default=240)
    parser.add_argument("--test-pairs-per-class", type=int, default=60)
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
    print(f"[Project 11] Device={device} | AMP={use_amp} | Epochs={args.epochs} | Batch={args.batch_size}")

    train_ds, test_ds, data_info = build_real_paired_datasets(
        data_dir=args.data_dir,
        train_pairs_per_class=args.train_pairs_per_class,
        test_pairs_per_class=args.test_pairs_per_class,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")
    )

    ablation_metrics: dict[str, dict[str, Any]] = {}
    training_telemetry: dict[str, dict[str, Any]] = {}
    gated_alpha_distributions: dict[str, list[float]] = {}
    gated_model: nn.Module | None = None

    with timed_block() as total_timer:
        for idx, variant in enumerate(VARIANTS):
            print(f"\n--- Training variant [{idx + 1}/{len(VARIANTS)}]: {variant} ---")
            model, telemetry = train_single_variant(
                variant=variant,
                train_loader=train_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                use_amp=use_amp,
                seed=args.seed + idx * 17,
            )
            training_telemetry[variant] = {
                "params": telemetry["params"],
                "train_seconds": telemetry["train_seconds"],
                "final_train_loss": telemetry["final_train_loss"],
            }
            ablation_metrics[variant] = {}
            for mode in EVAL_MODES:
                m_metrics, alphas = evaluate_variant(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    mode=mode,
                    seed=args.seed + 500,
                )
                ablation_metrics[variant][mode] = m_metrics
                if variant == "gated_cross_attention":
                    gated_alpha_distributions[mode] = alphas
                print(
                    f"  {variant:22s} | {mode:14s} | acc={m_metrics['accuracy']:.4f} "
                    f"| macro_f1={m_metrics['macro_f1']:.4f} | alpha={m_metrics['mean_alpha']:.3f}"
                )

            if variant == "gated_cross_attention":
                gated_model = model
                args.artifacts_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = args.artifacts_dir / "gated_cross_attention_best.pt"
                torch.save(
                    {
                        "variant": variant,
                        "state_dict": model.state_dict(),
                        "class_names": CLASS_NAMES,
                        "metrics": ablation_metrics[variant],
                    },
                    ckpt_path,
                )
            else:
                del model
                cleanup_cuda()

    # Collect 4 distinct held-out class samples + learned gate alphas for visual proof.
    sample_indices: list[int] = []
    seen_classes: set[int] = set()
    for i in range(len(test_ds)):
        lbl = int(test_ds.labels[i])
        if lbl not in seen_classes:
            seen_classes.add(lbl)
            sample_indices.append(i)
        if len(sample_indices) == 4:
            break

    sample_imgs = np.stack([test_ds.images[i] for i in sample_indices], axis=0)
    sample_specs = np.stack([test_ds.spectrograms[i] for i in sample_indices], axis=0)
    sample_labels = [CLASS_NAMES[int(test_ds.labels[i])] for i in sample_indices]
    sample_alphas: list[float] = []
    if gated_model is not None:
        gated_model.eval()
        with torch.no_grad():
            for idx_s in sample_indices:
                img_t, spec_t, _ = test_ds[idx_s]
                out = gated_model(
                    img_t.unsqueeze(0).to(device),
                    spec_t.unsqueeze(0).to(device),
                    return_details=True,
                )
                sample_alphas.append(float(out["alpha"].item()))
        del gated_model
        cleanup_cuda()

    fig_path = generate_multimodal_ablation_figure(
        output_path=args.reports_dir / "multimodal_ablation.png",
        sample_images=sample_imgs,
        sample_specs=sample_specs,
        sample_labels=sample_labels,
        sample_alphas=sample_alphas,
        gate_distributions=gated_alpha_distributions,
        ablation_results=ablation_metrics,
    )

    vram_stats = get_vram_stats()
    gated = ablation_metrics["gated_cross_attention"]
    report = {
        "project": "11_multimodal_image_audio_fusion",
        "dataset_type": "real_public_paired",
        "dataset": data_info,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "amp_fp16": use_amp,
        },
        "hardware": {
            **vram_stats,
            "total_wall_seconds": total_timer["elapsed_s"],
        },
        "ablation_results": ablation_metrics,
        "training_telemetry": training_telemetry,
        "key_findings": {
            "gated_clean_accuracy": gated["clean"]["accuracy"],
            "gated_mixed_corrupt_accuracy": gated["mixed_corrupt"]["accuracy"],
            "image_only_mixed_corrupt_accuracy": ablation_metrics["image_only"]["mixed_corrupt"]["accuracy"],
            "audio_only_mixed_corrupt_accuracy": ablation_metrics["audio_only"]["mixed_corrupt"]["accuracy"],
            "early_concat_mixed_corrupt_accuracy": ablation_metrics["early_concat"]["mixed_corrupt"]["accuracy"],
            "gate_alpha_clean": gated["clean"]["mean_alpha"],
            "gate_alpha_visual_corrupt": gated["visual_corrupt"]["mean_alpha"],
            "gate_alpha_audio_corrupt": gated["audio_corrupt"]["mean_alpha"],
            "gate_adaptation_delta": round(
                gated["audio_corrupt"]["mean_alpha"] - gated["visual_corrupt"]["mean_alpha"], 4
            ),
        },
        "artifacts": {
            "checkpoint": "artifacts/gated_cross_attention_best.pt",
            "figure": f"reports/{fig_path.name}",
        },
    }
    report_path = write_json_report(args.reports_dir / "metrics.json", report)
    print(f"\n[Project 11] Saved report -> {report_path}")
    print(f"[Project 11] Saved figure -> {fig_path}")
    print(f"[Project 11] Peak VRAM -> {vram_stats['peak_vram_mb']} MB")


if __name__ == "__main__":
    main()
