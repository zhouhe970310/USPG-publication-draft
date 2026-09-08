import torch
import torch.nn as nn
import os
import sys

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.backbone.hardnet_mseg import HarDNetMSEG
from models.modules.dsp import DomainStylePerturbation
from models.modules.spc import StructurePreservingConsistency
from models.modules.ugrl import UncertaintyGuidedRegionLearning


class USPGNet(nn.Module):
    """
    USPG-Net: Uncertainty-Guided Structure-Preserving Generalization Network

    完整模型包含：
    1. HarDNet-MSEG Backbone
    2. DSP (Domain Style Perturbation)
    3. SPC (Structure-Preserving Consistency)
    4. UGRL (Uncertainty-Guided Region Learning)
    """

    def __init__(self, in_channels=3, use_dsp=True, use_spc=True, use_ugrl=True,
                 spc_weight=0.15, ugrl_weight=0.03, dsp_prob=0.5,
                 spc_boundary_weight=0.2, spc_confidence_margin=0.15,
                 perturbed_seg_weight=1.0):
        """
        Args:
            in_channels: 输入通道数
            use_dsp: 是否使用DSP模块
            use_spc: 是否使用SPC模块
            use_ugrl: 是否使用UGRL模块
            spc_weight: SPC损失权重
            ugrl_weight: UGRL损失权重
        """
        super(USPGNet, self).__init__()

        # Backbone
        self.backbone = HarDNetMSEG(in_channels=in_channels)

        # 创新模块
        self.use_dsp = use_dsp
        self.use_spc = use_spc
        self.use_ugrl = use_ugrl
        self.perturbed_seg_weight = perturbed_seg_weight

        if use_dsp:
            self.dsp = DomainStylePerturbation(prob=dsp_prob)

        if use_spc:
            self.spc = StructurePreservingConsistency(
                boundary_weight=spc_boundary_weight,
                confidence_margin=spc_confidence_margin,
            )
            self.spc_weight = spc_weight

        if use_ugrl:
            self.ugrl = UncertaintyGuidedRegionLearning()
            self.ugrl_weight = ugrl_weight

    def forward(self, x, return_all=False):
        """
        前向传播

        Args:
            x: 输入图像 (B, C, H, W)
            return_all: 是否返回所有中间结果（训练时用）

        Returns:
            如果return_all=False (测试模式):
                pred: 预测 logits (B, 1, H, W)
            如果return_all=True (训练模式):
                outputs: 包含所有预测和损失的字典
        """
        if not self.training or not return_all:
            # 测试模式：简单前向传播
            pred = self.backbone(x)
            return pred

        # 训练模式：计算所有组件
        outputs = {}

        # 1. 原始图像预测
        pred_orig = self.backbone(x)
        outputs['pred_orig'] = pred_orig

        # 2. 如果使用DSP，生成扰动图像并预测
        if self.use_dsp:
            x_perturbed = self.dsp(x, apply_dsp=True)
            pred_perturbed = self.backbone(x_perturbed)
            outputs['pred_perturbed'] = pred_perturbed
            outputs['x_perturbed'] = x_perturbed
        else:
            pred_perturbed = None

        # 3. 如果使用SPC，计算结构一致性损失
        if self.use_spc and pred_perturbed is not None:
            pred_orig_prob = torch.sigmoid(pred_orig)
            pred_perturbed_prob = torch.sigmoid(pred_perturbed)
            spc_loss, spc_loss_dict = self.spc(pred_orig_prob, pred_perturbed_prob)
            outputs['spc_loss'] = spc_loss
            outputs['spc_loss_dict'] = spc_loss_dict

        # 4. Fast training uncertainty: entropy + cross-style disagreement.
        if self.use_ugrl:
            pred_orig_prob = torch.sigmoid(pred_orig)
            pred_perturbed_prob = (
                torch.sigmoid(pred_perturbed) if pred_perturbed is not None else None
            )
            uncertainty = self.ugrl.training_uncertainty(
                pred_orig_prob, pred_perturbed_prob
            )
            outputs['uncertainty'] = uncertainty
        else:
            uncertainty = None

        outputs['uncertainty_map'] = uncertainty

        return outputs

    def compute_loss(self, outputs, target, seg_criterion, auxiliary_scale=1.0):
        """
        计算总损失

        Args:
            outputs: forward返回的outputs字典
            target: 真实mask (B, 1, H, W)
            seg_criterion: 分割损失函数（如BCEDiceLoss）

        Returns:
            total_loss: 总损失
            loss_dict: 各个损失的字典
        """
        loss_dict = {}

        # 1. 基础分割损失
        pred_orig = outputs['pred_orig']
        seg_loss = seg_criterion(pred_orig, target)
        loss_dict['seg_loss'] = seg_loss.item()
        total_loss = seg_loss

        # 2. 如果有扰动预测，也计算其分割损失
        if 'pred_perturbed' in outputs:
            pred_perturbed = outputs['pred_perturbed']
            seg_loss_perturbed = seg_criterion(pred_perturbed, target)
            loss_dict['seg_loss_perturbed'] = seg_loss_perturbed.item()
            # Average the two supervised views. Summing them changes the effective
            # learning rate relative to the baseline and invalidates the ablation.
            alpha = self.perturbed_seg_weight
            total_loss = (seg_loss + alpha * seg_loss_perturbed) / (1.0 + alpha)

        # 3. SPC损失
        if self.use_spc and 'spc_loss' in outputs:
            spc_loss = outputs['spc_loss']
            total_loss = total_loss + auxiliary_scale * self.spc_weight * spc_loss
            loss_dict.update(outputs['spc_loss_dict'])

        # 4. UGRL损失
        if self.use_ugrl and 'uncertainty_map' in outputs:
            pred_orig_prob = torch.sigmoid(pred_orig)
            uncertainty = outputs['uncertainty_map']
            ugrl_loss = self.ugrl(pred_orig_prob, target, uncertainty)
            total_loss = total_loss + auxiliary_scale * self.ugrl_weight * ugrl_loss
            loss_dict['ugrl_loss'] = ugrl_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


def build_model(model_type='baseline', **kwargs):
    """
    构建不同配置的模型

    Args:
        model_type: 'baseline', 'dsp', 'spc', 'uspg'
        **kwargs: 其他参数

    Returns:
        model: 构建的模型
    """
    if model_type in {'resunet', 'resunet_dsp', 'resunet_spc_region',
                      'resunet_spc'}:
        from models.dg_resunet import build_dg_resunet
        return build_dg_resunet(model_type, **kwargs)
    if model_type in {'unet', 'unetpp', 'deeplabv3p'}:
        from models.comparison import build_comparison_model
        return build_comparison_model(model_type)
    if model_type == 'baseline':
        # 只有HarDNet-MSEG
        model = USPGNet(use_dsp=False, use_spc=False, use_ugrl=False, **kwargs)
    elif model_type == 'dsp':
        # HarDNet-MSEG + DSP
        model = USPGNet(use_dsp=True, use_spc=False, use_ugrl=False, **kwargs)
    elif model_type == 'spc':
        # HarDNet-MSEG + DSP + SPC
        model = USPGNet(use_dsp=True, use_spc=True, use_ugrl=False, **kwargs)
    elif model_type == 'uspg':
        # 完整USPG-Net
        model = USPGNet(use_dsp=True, use_spc=True, use_ugrl=True, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def test_uspg_net():
    """测试USPG-Net模型"""
    print("=" * 80)
    print("Testing USPG-Net Models")
    print("=" * 80)

    batch_size = 2
    x = torch.randn(batch_size, 3, 352, 352)
    target = torch.randint(0, 2, (batch_size, 1, 352, 352)).float()

    models = {
        'baseline': build_model('baseline'),
        'dsp': build_model('dsp'),
        'spc': build_model('spc'),
        'uspg': build_model('uspg'),
    }

    for name, model in models.items():
        print(f"\n{'=' * 40}")
        print(f"Testing {name.upper()} model")
        print(f"{'=' * 40}")

        # 统计参数
        total, trainable = count_parameters(model)
        print(f"Total parameters: {total / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable / 1e6:.2f}M")

        # 测试前向传播（测试模式）
        model.eval()
        with torch.no_grad():
            pred = model(x)
            print(f"Test mode output shape: {pred.shape}")

        # 测试前向传播（训练模式）
        model.train()
        outputs = model(x, return_all=True)
        print(f"Train mode outputs: {list(outputs.keys())}")

        # 测试损失计算
        from sys import path
        path.insert(0, '../losses')
        from losses import BCEDiceLoss
        criterion = BCEDiceLoss()

        total_loss, loss_dict = model.compute_loss(outputs, target, criterion)
        print(f"Total loss: {total_loss.item():.4f}")
        print(f"Loss breakdown: {loss_dict}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_uspg_net()
