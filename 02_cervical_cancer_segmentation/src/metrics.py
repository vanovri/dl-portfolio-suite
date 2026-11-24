"""Segmentation metrics (Dice, IoU, Precision, Recall, Boundary F1) and biopsy morphometry."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _extract_boundary(binary_mask: np.ndarray) -> np.ndarray:
    mask_u8 = (binary_mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return (mask_u8 - eroded).astype(np.uint8)


def boundary_f1_score(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance: int = 2) -> float:
    """Compute morphological Boundary F1 score with a pixel tolerance band."""
    pred_b = _extract_boundary(pred_mask)
    gt_b = _extract_boundary(gt_mask)

    pred_cnt = int(pred_b.sum())
    gt_cnt = int(gt_b.sum())
    if pred_cnt == 0 and gt_cnt == 0:
        return 1.0
    if pred_cnt == 0 or gt_cnt == 0:
        return 0.0

    k = 2 * tolerance + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    pred_dil = cv2.dilate(pred_b, kernel, iterations=1)
    gt_dil = cv2.dilate(gt_b, kernel, iterations=1)

    prec = float((pred_b & gt_dil).sum()) / float(pred_cnt)
    rec = float((gt_b & pred_dil).sum()) / float(gt_cnt)
    if prec + rec == 0.0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def compute_segmentation_metrics(
    pred_probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute Dice, IoU, Precision, Recall, and Boundary F1 across a batch of masks."""
    preds = (np.asarray(pred_probs) >= threshold).astype(np.uint8)
    gts = (np.asarray(targets) > 0.5).astype(np.uint8)

    if preds.ndim == 4:
        preds = preds[:, 0]
    if gts.ndim == 4:
        gts = gts[:, 0]

    tp = float(np.logical_and(preds == 1, gts == 1).sum())
    fp = float(np.logical_and(preds == 1, gts == 0).sum())
    fn = float(np.logical_and(preds == 0, gts == 1).sum())

    eps = 1e-7
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    bf1_scores = [boundary_f1_score(preds[i], gts[i], tolerance=2) for i in range(len(preds))]
    bf1 = float(np.mean(bf1_scores)) if bf1_scores else 0.0

    return {
        "dice": round(float(dice), 4),
        "iou": round(float(iou), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "boundary_f1": round(float(bf1), 4),
    }


def analyze_biopsy_morphometry(
    rgb_image: np.ndarray,
    nuclear_mask: np.ndarray,
    min_nucleus_area_px: int = 4,
) -> dict[str, Any]:
    """Estimate Nucleus-to-Cytoplasm (N/C) area ratio and nuclei count from biopsy RGB + mask."""
    mask_u8 = (np.asarray(nuclear_mask) > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    valid_nuclei = 0
    nuclear_area_px = 0
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area >= min_nucleus_area_px:
            valid_nuclei += 1
            nuclear_area_px += area

    # Estimate cellular tissue area (non-glass background in H&E stained microscopy)
    hsv = cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2HSV)
    tissue_mask = ((hsv[:, :, 1] > 18) | (hsv[:, :, 2] < 235)).astype(np.uint8)
    tissue_area_px = int(tissue_mask.sum())
    total_px = int(mask_u8.size)
    if tissue_area_px < int(0.15 * total_px):
        tissue_area_px = total_px

    cytoplasm_area_px = max(tissue_area_px - nuclear_area_px, 1)
    nc_ratio = float(nuclear_area_px) / float(cytoplasm_area_px)
    nuclear_area_pct = 100.0 * float(nuclear_area_px) / float(max(tissue_area_px, 1))

    return {
        "nuclei_count": valid_nuclei,
        "nuclear_area_px": nuclear_area_px,
        "cytoplasm_area_px": cytoplasm_area_px,
        "nc_ratio": round(nc_ratio, 4),
        "nuclear_area_pct": round(nuclear_area_pct, 2),
    }
