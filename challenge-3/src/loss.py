"""Training losses matching the competition metric.

The metric is a weighted Pearson correlation computed independently for each
sequence and each target, with weights abs(target). The loss below is the same
quantity on a truncated-BPTT chunk: every sequence in the batch contributes its
own correlation over the chunk's scored rows, so the network is optimised for
within-sequence ranking rather than a pooled correlation across sequences.
"""
from __future__ import annotations

import torch
import torch.nn as nn

CLIP = 2.0


def weighted_pearson(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                     eps: float = 1e-6) -> torch.Tensor:
    """Per-sequence, per-target weighted Pearson. Returns (B, 2)."""
    y = targets.clamp(-CLIP, CLIP)
    m = mask.to(preds.dtype).unsqueeze(-1)
    w = y.abs() * m
    total = w.sum(dim=1).clamp_min(eps)                 # (B, 2)
    mean_y = (w * y).sum(dim=1) / total
    mean_p = (w * preds).sum(dim=1) / total
    yc = (y - mean_y.unsqueeze(1)) * m
    pc = (preds - mean_p.unsqueeze(1)) * m
    cov = (w * yc * pc).sum(dim=1) / total
    var_y = (w * yc * yc).sum(dim=1) / total
    var_p = (w * pc * pc).sum(dim=1) / total
    return cov / torch.sqrt(var_y * var_p + eps)


class PearsonLoss(nn.Module):
    """1 - mean weighted Pearson over sequences and targets."""

    def forward(self, preds, targets, mask):
        return 1.0 - weighted_pearson(preds, targets, mask).mean()


class WeightedMSELoss(nn.Module):
    """abs(target)-weighted MSE on scored rows; gives the outputs a scale."""

    def forward(self, preds, targets, mask, eps: float = 1e-6):
        y = targets.clamp(-CLIP, CLIP)
        w = y.abs() * mask.to(preds.dtype).unsqueeze(-1)
        return (w * (preds - y) ** 2).sum() / w.sum().clamp_min(eps)


class CombinedLoss(nn.Module):
    def __init__(self, pearson_weight: float = 0.9):
        super().__init__()
        self.alpha = pearson_weight
        self.pearson = PearsonLoss()
        self.mse = WeightedMSELoss()

    def forward(self, preds, targets, mask):
        return self.alpha * self.pearson(preds, targets, mask) + (1 - self.alpha) * self.mse(preds, targets, mask)


class SequencePearsonLoss(nn.Module):
    """Sequence-level weighted Pearson under truncated BPTT.

    The metric removes one weighted mean per *sequence*; a per-chunk Pearson
    removes one per chunk and so ignores slow drift of the prediction level
    across the sequence. This loss keeps the running sufficient statistics
    (sum w, w*y, w*p, w*y^2, w*p^2, w*y*p) of the sequence so far, detached,
    and evaluates the correlation of the whole prefix with gradients flowing
    only through the current chunk. Call reset() at the start of each batch of
    sequences and step() once per chunk in order.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.stats = None

    def reset(self):
        self.stats = None

    def step(self, preds, targets, mask):
        y = targets.clamp(-CLIP, CLIP)
        m = mask.to(preds.dtype).unsqueeze(-1)
        w = y.abs() * m
        cur = torch.stack([w.sum(1), (w * y).sum(1), (w * preds).sum(1),
                           (w * y * y).sum(1), (w * preds * preds).sum(1), (w * y * preds).sum(1)])  # (6, B, 2)
        tot = cur if self.stats is None else self.stats + cur
        sw, sy, sp, syy, spp, syp = tot
        sw = sw.clamp_min(self.eps)
        my, mp = sy / sw, sp / sw
        cov = syp / sw - my * mp
        var_y = (syy / sw - my * my).clamp_min(0)
        var_p = (spp / sw - mp * mp).clamp_min(0)
        corr = cov / torch.sqrt(var_y * var_p + self.eps)
        self.stats = tot.detach()
        return 1.0 - corr.mean()
