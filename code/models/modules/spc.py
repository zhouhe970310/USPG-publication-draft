import torch
import torch.nn as nn
import torch.nn.functional as F


class StructurePreservingConsistency(nn.Module):
    """Teacher-to-perturbed consistency with confidence-aware supervision."""

    def __init__(self, pred_weight=1.0, boundary_weight=0.2, confidence_margin=0.15):
        super().__init__()
        self.pred_weight = pred_weight
        self.boundary_weight = boundary_weight
        self.confidence_margin = confidence_margin
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_x.t().view(1, 1, 3, 3))

    def boundary(self, probability):
        gx = F.conv2d(probability, self.sobel_x, padding=1)
        gy = F.conv2d(probability, self.sobel_y, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)

    def forward(self, pred_orig, pred_perturbed):
        teacher = pred_orig.detach()
        confidence = (teacher - 0.5).abs() * 2.0
        mask = (confidence >= self.confidence_margin).float()
        pred_error = F.smooth_l1_loss(pred_perturbed, teacher, reduction='none')
        pred_loss = (pred_error * mask).sum() / mask.sum().clamp_min(1.0)
        boundary_loss = F.smooth_l1_loss(
            self.boundary(pred_perturbed), self.boundary(teacher)
        )
        total = self.pred_weight * pred_loss + self.boundary_weight * boundary_loss
        return total, {
            'spc_total': total.detach().item(),
            'spc_pred': pred_loss.detach().item(),
            'spc_boundary': boundary_loss.detach().item(),
        }
