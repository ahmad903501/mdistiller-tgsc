"""Sample weighting shared by TGSC-enabled distillers."""
import torch


def quality_gate(scores, threshold=0.75):
    if threshold <= 0:
        raise ValueError("TGSC gate threshold must be positive")
    scores = scores.detach().float()
    if not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Oracle metrics must be finite and nonnegative")
    return (scores / threshold).clamp(max=1.0)


def weighted_mean(losses, metadata=None, threshold=0.75, component="q"):
    if losses.ndim != 1:
        raise ValueError("Distillation losses must have one value per sample")
    if metadata is None:
        return losses.mean()
    values = metadata[component].to(losses.device)
    if values.shape != losses.shape:
        raise ValueError("Gate/loss sample counts differ")
    return (losses * quality_gate(values, threshold)).mean()
