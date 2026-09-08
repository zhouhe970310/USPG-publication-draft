import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainStylePerturbation(nn.Module):
    """Mild per-image style perturbations applied in RGB [0, 1] space."""

    def __init__(self, prob=0.7):
        super().__init__()
        self.prob = prob
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _to_rgb(self, image):
        return (image * self.std + self.mean).clamp(0.0, 1.0)

    def _to_normalized(self, image):
        return (image.clamp(0.0, 1.0) - self.mean) / self.std

    @staticmethod
    def _rand(shape, ref, low, high):
        return torch.empty(shape, device=ref.device, dtype=ref.dtype).uniform_(low, high)

    def brightness(self, x):
        factor = self._rand((x.shape[0], 1, 1, 1), x, 0.85, 1.15)
        offset = self._rand((x.shape[0], 1, 1, 1), x, -0.04, 0.04)
        return (x * factor + offset).clamp(0.0, 1.0)

    def contrast(self, x):
        factor = self._rand((x.shape[0], 1, 1, 1), x, 0.80, 1.20)
        mean = x.mean(dim=(2, 3), keepdim=True)
        return ((x - mean) * factor + mean).clamp(0.0, 1.0)

    def color(self, x):
        factor = self._rand((x.shape[0], 3, 1, 1), x, 0.90, 1.10)
        return (x * factor).clamp(0.0, 1.0)

    def frequency(self, x):
        # A differentiable low/high-frequency decomposition is substantially
        # faster than a full complex FFT on CPU while preserving the intended
        # mild, phase-safe band-amplitude perturbation.
        low = F.avg_pool2d(x, kernel_size=17, stride=1, padding=8)
        high = x - low
        low_scale = self._rand((x.shape[0], x.shape[1], 1, 1), x, 0.88, 1.12)
        high_scale = self._rand((x.shape[0], x.shape[1], 1, 1), x, 0.95, 1.05)
        return (low * low_scale + high * high_scale).clamp(0.0, 1.0)

    def forward(self, image, apply_dsp=True):
        if not apply_dsp or not self.training:
            return image
        rgb = self._to_rgb(image)
        original = rgb.clone()
        for method in (self.brightness, self.contrast, self.color, self.frequency):
            selected = torch.rand(rgb.shape[0], device=rgb.device) < self.prob / 2
            if selected.any():
                updated = rgb.clone()
                updated[selected] = method(rgb[selected])
                rgb = updated
        unchanged = (rgb - original).abs().flatten(1).sum(1) < 1e-8
        if unchanged.any():
            rgb[unchanged] = self.contrast(rgb[unchanged])
        return self._to_normalized(rgb)
