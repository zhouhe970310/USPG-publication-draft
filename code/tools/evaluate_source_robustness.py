"""Select augmentation settings using source validation data only.

The fixed Kvasir validation subset is evaluated under deterministic photometric
shifts.  No target-domain (CVC-ColonDB) sample is used by this script, which
keeps hyper-parameter selection independent from the final test set.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import get_dataloaders
from test import checkpoint_sha256, load_model, set_seed


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def apply_shift(images, shift):
    mean = images.new_tensor(MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(STD).view(1, 3, 1, 1)
    rgb = (images * std + mean).clamp(0.0, 1.0)
    if shift == 'clean':
        changed = rgb
    elif shift == 'dark':
        changed = rgb * 0.70
    elif shift == 'bright':
        changed = rgb * 1.20 + 0.03
    elif shift == 'high_contrast':
        center = rgb.mean(dim=(2, 3), keepdim=True)
        changed = (rgb - center) * 1.40 + center
    elif shift == 'gamma_dark':
        changed = rgb.pow(1.40)
    elif shift == 'gamma_bright':
        changed = rgb.pow(0.70)
    elif shift == 'warm':
        changed = rgb * rgb.new_tensor([1.15, 0.95, 0.82]).view(1, 3, 1, 1)
    elif shift == 'cool':
        changed = rgb * rgb.new_tensor([0.84, 0.98, 1.15]).view(1, 3, 1, 1)
    else:
        raise ValueError(f'Unknown shift: {shift}')
    return (changed.clamp(0.0, 1.0) - mean) / std


@torch.no_grad()
def evaluate_shift(model, loader, device, shift):
    dice, iou, mae = [], [], []
    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        probability = torch.sigmoid(model(apply_shift(images, shift)))
        binary = probability > 0.5
        target = masks > 0.5
        axes = (1, 2, 3)
        intersection = (binary & target).sum(dim=axes).float()
        pred_sum = binary.sum(dim=axes).float()
        target_sum = target.sum(dim=axes).float()
        union = (binary | target).sum(dim=axes).float()
        dice.extend(((2 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7)).cpu())
        iou.extend(((intersection + 1e-7) / (union + 1e-7)).cpu())
        mae.extend((probability - masks).abs().mean(dim=axes).cpu())
    return {
        'dice': float(torch.stack(dice).mean()),
        'iou': float(torch.stack(iou).mean()),
        'mae': float(torch.stack(mae).mean()),
    }


def evaluate(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint = load_model(args.checkpoint, args.model_type, device)
    _, loader, _ = get_dataloaders(
        'kvasir', args.batch_size, args.num_workers, args.image_size,
        seed=args.seed, split_seed=args.split_seed, split_file=args.split_file,
        dataset_root=args.dataset_root,
    )
    shifts = ['clean', 'dark', 'bright', 'high_contrast', 'gamma_dark',
              'gamma_bright', 'warm', 'cool']
    rows = []
    for shift in shifts:
        result = evaluate_shift(model, loader, device, shift)
        rows.append({'shift': shift, **result})
        print(f"{shift}: Dice={result['dice']:.4f}")
    frame = pd.DataFrame(rows)
    corrupt = frame[frame['shift'] != 'clean']
    summary = {
        'selection_dataset': 'Kvasir-SEG fixed validation subset only',
        'model_type': args.model_type,
        'seed': args.seed,
        'image_size': args.image_size,
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': checkpoint_sha256(args.checkpoint),
        'best_epoch': checkpoint.get('epoch'),
        'best_val_dice': checkpoint.get('best_dice'),
        'clean_dice': float(frame.loc[frame['shift'] == 'clean', 'dice'].iloc[0]),
        'mean_corrupt_dice': float(corrupt['dice'].mean()),
        'worst_corrupt_dice': float(corrupt['dice'].min()),
        # Clean accuracy remains important, while shifted-source accuracy is the
        # target-free proxy for robustness used to rank configurations.
        'selection_score': float(
            0.35 * frame.loc[frame['shift'] == 'clean', 'dice'].iloc[0]
            + 0.65 * corrupt['dice'].mean()
        ),
        'shifts': rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    frame.to_csv(output.with_suffix('.csv'), index=False)
    print(json.dumps({key: value for key, value in summary.items() if key != 'shifts'}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_type', required=True,
        choices=['baseline', 'dsp', 'spc', 'uspg',
                 'resunet', 'resunet_dsp', 'resunet_spc_region',
                 'resunet_spc', 'unet', 'unetpp', 'deeplabv3p']
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--image_size', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_seed', type=int, default=20260822)
    parser.add_argument('--split_file', default='splits/kvasir_fixed_split.json')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
