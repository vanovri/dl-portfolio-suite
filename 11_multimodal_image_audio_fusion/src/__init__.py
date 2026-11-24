"""Project 11: Multimodal Image + Audio Fusion with Gated Cross-Attention."""
from .dataset import (
    CLASS_NAMES,
    CLASS_SPEC,
    LogMelSpectrogramExtractor,
    PairedAudioVisualDataset,
    apply_modality_corruption,
    build_real_paired_datasets,
    verify_zero_leakage,
)
from .model import (
    AudioEncoder,
    AudioOnlyNet,
    EarlyConcatFusionNet,
    GatedCrossAttentionFusionNet,
    ImageEncoder,
    ImageOnlyNet,
    build_model,
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_SPEC",
    "LogMelSpectrogramExtractor",
    "PairedAudioVisualDataset",
    "apply_modality_corruption",
    "build_real_paired_datasets",
    "verify_zero_leakage",
    "ImageEncoder",
    "AudioEncoder",
    "ImageOnlyNet",
    "AudioOnlyNet",
    "EarlyConcatFusionNet",
    "GatedCrossAttentionFusionNet",
    "build_model",
]
