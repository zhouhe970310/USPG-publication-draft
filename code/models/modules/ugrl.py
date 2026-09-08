import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyGuidedRegionLearning(nn.Module):
    """Bounded uncertainty-aware reliability loss without extra training passes."""

    def __init__(self, boundary_weight=0.35, uncertainty_strength=0.75):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.uncertainty_strength = uncertainty_strength
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_x.t().view(1, 1, 3, 3))

    @staticmethod
    def training_uncertainty(pred_orig, pred_perturbed=None):
        p = pred_orig.detach().clamp(1e-6, 1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2.0)
        if pred_perturbed is None:
            return entropy
        disagreement = (p - pred_perturbed.detach()).abs()
        return (0.65 * entropy + 0.35 * disagreement).clamp(0.0, 1.0)

    def boundary(self, target):
        gx = F.conv2d(target, self.sobel_x, padding=1)
        gy = F.conv2d(target, self.sobel_y, padding=1)
        return (torch.sqrt(gx.square() + gy.square() + 1e-6) > 0.1).float()

    def forward(self, pred, target, uncertainty):
        bce = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), target, reduction='none')
        # Ambiguous pixels receive less direct BCE pressure; reliable boundaries
        # remain emphasized. Normalization preserves the global loss scale.
        weights = 1.0 / (1.0 + self.uncertainty_strength * uncertainty)
        weights = weights * (1.0 + self.boundary_weight * self.boundary(target))
        weights = weights / weights.mean().clamp_min(1e-6)
        return (bce * weights).mean()

    @staticmethod
    def _enable_dropout(model):
        types = (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)
        found = 0
        for module in model.modules():
            if isinstance(module, types):
                module.train()
                found += 1
        if found == 0:
            raise RuntimeError('MC-dropout requested, but the model contains no dropout layer.')

    def estimate_uncertainty(self, model, x, n_samples=8):
        was_training = model.training
        model.eval()
        self._enable_dropout(model)
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(torch.sigmoid(model(x)))
        model.train(was_training)
        samples = torch.stack(samples)
        return samples.mean(0), samples.var(0, unbiased=False)
