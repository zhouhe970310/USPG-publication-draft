import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, probability, target):
        probability = probability.flatten(1)
        target = target.flatten(1)
        intersection = (probability * target).sum(1)
        denominator = probability.sum(1) + target.sum(1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        dice = self.dice(torch.sigmoid(logits), target)
        return self.bce_weight * bce + self.dice_weight * dice
