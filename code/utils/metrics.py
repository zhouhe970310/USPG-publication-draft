import math

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_coefficient(pred, target, smooth=1e-7):
    pred, target = pred.reshape(-1), target.reshape(-1)
    return (2.0 * (pred * target).sum() + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-7):
    pred, target = pred.reshape(-1), target.reshape(-1)
    intersection = (pred * target).sum()
    return (intersection + smooth) / (pred.sum() + target.sum() - intersection + smooth)


def hausdorff_distance_95(pred, target, spacing=(1.0, 1.0)):
    """Symmetric surface HD95 in resized-image pixels (or supplied spacing units)."""
    pred = np.asarray(pred > 0.5, dtype=bool)
    target = np.asarray(target > 0.5, dtype=bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float(math.sqrt(sum((s * n) ** 2 for s, n in zip(spacing, pred.shape))))
    pred_surface = pred ^ binary_erosion(pred, border_value=0)
    target_surface = target ^ binary_erosion(target, border_value=0)
    target_distance = distance_transform_edt(~target_surface, sampling=spacing)
    pred_distance = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([
        target_distance[pred_surface],
        pred_distance[target_surface],
    ])
    return float(np.percentile(distances, 95))


def per_image_metrics(probability, target, threshold=0.5):
    probability = probability.detach().float().cpu()
    target = target.detach().float().cpu()
    binary = (probability > threshold).float()
    return {
        'dice': float(dice_coefficient(binary, target)),
        'iou': float(iou_score(binary, target)),
        'hd95': hausdorff_distance_95(binary.squeeze().numpy(), target.squeeze().numpy()),
        'mae': float((probability - target).abs().mean()),
    }


class Metrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.rows = []

    def update(self, pred, target, names=None):
        names = names or [str(len(self.rows) + i) for i in range(pred.shape[0])]
        for i in range(pred.shape[0]):
            row = {'image': names[i]}
            row.update(per_image_metrics(pred[i], target[i], self.threshold))
            self.rows.append(row)

    def get_results(self):
        if not self.rows:
            raise RuntimeError('No samples were evaluated.')
        result = {'n_images': len(self.rows)}
        for metric in ('dice', 'iou', 'hd95', 'mae'):
            values = np.asarray([row[metric] for row in self.rows], dtype=float)
            result[metric] = float(values.mean())
            result[f'{metric}_std_images'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        return result

    def print_results(self, prefix=''):
        result = self.get_results()
        for name in ('dice', 'iou', 'hd95', 'mae'):
            print(f"{prefix}{name.upper()}: {result[name]:.4f} "
                  f"± {result[f'{name}_std_images']:.4f} (across images)")
        return result
