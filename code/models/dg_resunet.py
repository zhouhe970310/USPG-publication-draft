"""Strong pretrained ResNet18-U-Net backbone with DSP/SPC ablations."""

import os
from pathlib import Path

import torch
import torch.nn as nn

from models.modules.dsp import DomainStylePerturbation
from models.modules.spc import StructurePreservingConsistency


class DGResUNet(nn.Module):
    def __init__(self, use_dsp=False, use_spc=False, encoder_weights='imagenet',
                 dsp_prob=0.5, spc_weight=0.15, spc_boundary_weight=0.2,
                 spc_confidence_margin=0.15, perturbed_seg_weight=1.0, **_):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.backbone = smp.Unet(
            encoder_name='resnet18',
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        if encoder_weights == 'imagenet':
            default = (Path(os.environ.get('TORCH_HOME', Path.home() / '.cache' / 'torch')) /
                       'hub' / 'checkpoints' / 'resnet18-5c106cde.pth')
            weight_path = Path(os.environ.get('USPG_RESNET18_WEIGHTS', default))
            if not weight_path.is_file():
                raise FileNotFoundError(
                    f'Local ImageNet weights not found: {weight_path}. Set '
                    'USPG_RESNET18_WEIGHTS to resnet18-5c106cde.pth.'
                )
            # Official torchvision ResNet-18 weights use the legacy tar serializer.
            state = torch.load(weight_path, map_location='cpu', weights_only=False)
            self.backbone.encoder.load_state_dict(state, strict=False)
        self.use_dsp = use_dsp
        self.use_spc = use_spc
        self.spc_weight = spc_weight
        self.perturbed_seg_weight = perturbed_seg_weight
        if use_dsp:
            self.dsp = DomainStylePerturbation(prob=dsp_prob)
        if use_spc:
            self.spc = StructurePreservingConsistency(
                boundary_weight=spc_boundary_weight,
                confidence_margin=spc_confidence_margin,
            )

    def forward(self, x, return_all=False):
        if not self.training or not return_all:
            return self.backbone(x)
        pred_orig = self.backbone(x)
        outputs = {'pred_orig': pred_orig}
        if self.use_dsp:
            perturbed = self.dsp(x, apply_dsp=True)
            pred_perturbed = self.backbone(perturbed)
            outputs['pred_perturbed'] = pred_perturbed
            if self.use_spc:
                consistency, parts = self.spc(
                    torch.sigmoid(pred_orig), torch.sigmoid(pred_perturbed)
                )
                outputs['spc_loss'] = consistency
                outputs['spc_loss_dict'] = parts
        return outputs

    def compute_loss(self, outputs, target, seg_criterion, auxiliary_scale=1.0):
        original_loss = seg_criterion(outputs['pred_orig'], target)
        loss = original_loss
        parts = {'seg_loss': original_loss.detach().item()}
        if 'pred_perturbed' in outputs:
            perturbed_loss = seg_criterion(outputs['pred_perturbed'], target)
            alpha = self.perturbed_seg_weight
            loss = (original_loss + alpha * perturbed_loss) / (1.0 + alpha)
            parts['seg_loss_perturbed'] = perturbed_loss.detach().item()
        if self.use_spc and 'spc_loss' in outputs:
            loss = loss + auxiliary_scale * self.spc_weight * outputs['spc_loss']
            parts.update(outputs['spc_loss_dict'])
        parts['total_loss'] = loss.detach().item()
        return loss, parts


def build_dg_resunet(model_type, **kwargs):
    if model_type == 'resunet':
        return DGResUNet(use_dsp=False, use_spc=False, **kwargs)
    if model_type == 'resunet_dsp':
        return DGResUNet(use_dsp=True, use_spc=False, **kwargs)
    if model_type == 'resunet_spc_region':
        return DGResUNet(use_dsp=True, use_spc=True, spc_boundary_weight=0.0,
                         **{k: v for k, v in kwargs.items()
                            if k != 'spc_boundary_weight'})
    if model_type == 'resunet_spc':
        return DGResUNet(use_dsp=True, use_spc=True, **kwargs)
    raise ValueError(f'Unknown DG ResUNet type: {model_type}')
