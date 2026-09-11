import torch
import torch.nn as nn


class WeightedPearsonLoss(nn.Module):
    """
    Differentiable Weighted Pearson Correlation Loss.
    Loss = 1 - WeightedPearsonCorrelation

    V4: Added prediction clipping to match competition metric.
    """
    def __init__(self, clip_pred: float = 6.0, eps: float = 1e-8):
        super().__init__()
        self.clip = clip_pred
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: (batch, seq_len, 2)
            targets: (batch, seq_len, 2)
            mask: (batch, seq_len) - 1 if prediction needed, 0 otherwise
        """
        # Clip predictions like competition metric does
        preds = torch.clamp(preds, -self.clip, self.clip)

        mask_bool = mask > 0
        if not mask_bool.any():
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        loss = 0.0
        num_targets = preds.shape[-1]

        for i in range(num_targets):
            p = preds[..., i][mask_bool]
            t = targets[..., i][mask_bool]

            # Weights are absolute value of targets
            w = torch.abs(t)
            w = torch.clamp(w, min=self.eps)

            # Weighted means
            sum_w = torch.sum(w)
            mean_p = torch.sum(p * w) / sum_w
            mean_t = torch.sum(t * w) / sum_w

            # Weighted deviations
            dev_p = p - mean_p
            dev_t = t - mean_t

            # Weighted covariance
            cov = torch.sum(w * dev_p * dev_t)

            # Weighted variances
            var_p = torch.sum(w * dev_p ** 2)
            var_t = torch.sum(w * dev_t ** 2)

            # Correlation
            corr = cov / (torch.sqrt(var_p * var_t) + self.eps)

            loss += (1.0 - corr)

        return loss / num_targets


class WeightedMSELoss(nn.Module):
    """MSE loss weighted by abs(target), only on scored steps."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        weights = torch.abs(targets) * mask
        weights = torch.clamp(weights, min=self.eps)
        se = (preds - targets) ** 2
        weighted_se = se * weights
        return weighted_se.sum() / weights.sum()


class CombinedLoss(nn.Module):
    """
    Mixed loss: alpha * WeightedPearsonLoss + (1-alpha) * WeightedMSELoss.
    Pearson drives correlation, MSE provides stable gradients.
    """
    def __init__(self, alpha: float = 0.85, clip_pred: float = 6.0):
        super().__init__()
        self.alpha = alpha
        self.pearson_loss = WeightedPearsonLoss(clip_pred=clip_pred)
        self.mse_loss = WeightedMSELoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        l_pearson = self.pearson_loss(preds, targets, mask)
        l_mse = self.mse_loss(preds, targets, mask)
        return self.alpha * l_pearson + (1 - self.alpha) * l_mse
