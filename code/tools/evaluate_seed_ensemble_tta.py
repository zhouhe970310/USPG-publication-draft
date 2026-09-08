"""Evaluate an equal-weight three-seed ensemble with source-only calibration.

For each model family, the threshold and optional horizontal-flip TTA are
selected on the fixed Kvasir validation subset.  CVC is not instantiated until
the selection has been written to disk.
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
from utils.metrics import Metrics


@torch.no_grad()
def predict_ensemble(models, images, mode):
    probability = torch.stack([torch.sigmoid(model(images)) for model in models]).mean(0)
    if mode == 'hflip':
        flipped = torch.flip(images, dims=(-1,))
        probability_flip = torch.stack([
            torch.flip(torch.sigmoid(model(flipped)), dims=(-1,)) for model in models
        ]).mean(0)
        probability = 0.5 * (probability + probability_flip)
    elif mode != 'plain':
        raise ValueError(mode)
    return probability


@torch.no_grad()
def threshold_curve(models, loader, device, mode, thresholds):
    totals = torch.zeros(len(thresholds), dtype=torch.float64)
    count = 0
    threshold_tensor = torch.tensor(thresholds, device=device).view(-1, 1, 1, 1, 1)
    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        probability = predict_ensemble(models, images, mode)
        binary = probability.unsqueeze(0) > threshold_tensor
        target = (masks > 0.5).unsqueeze(0)
        axes = (2, 3, 4)
        intersection = (binary & target).sum(dim=axes).float()
        denominator = binary.sum(dim=axes).float() + target.sum(dim=axes).float()
        dice = (2 * intersection + 1e-7) / (denominator + 1e-7)
        totals += dice.sum(dim=1).double().cpu()
        count += masks.shape[0]
    return (totals / count).numpy()


def load_models(args, model_type, device):
    models, checkpoints, checkpoint_paths = [], [], []
    for seed in args.seeds:
        path = Path(args.runs_root) / model_type / f'seed{seed}' / 'best_model.pth'
        model, checkpoint = load_model(path, model_type, device)
        models.append(model)
        checkpoints.append(checkpoint)
        checkpoint_paths.append(path)
    return models, checkpoints, checkpoint_paths


@torch.no_grad()
def evaluate_loader(models, loader, device, mode, threshold):
    metrics = Metrics(threshold=threshold)
    for images, masks, names in loader:
        images, masks = images.to(device), masks.to(device)
        metrics.update(predict_ensemble(models, images, mode), masks, names)
    return metrics


def run_model_family(args, model_type, device):
    models, checkpoints, checkpoint_paths = load_models(args, model_type, device)
    _, val_loader, kvasir_test = get_dataloaders(
        'kvasir', args.batch_size, args.num_workers, args.image_size,
        seed=args.seeds[0], split_seed=args.split_seed, split_file=args.split_file,
        dataset_root=args.kvasir_root,
    )
    thresholds = np.round(np.arange(args.threshold_min, args.threshold_max + 1e-9,
                                    args.threshold_step), 4)
    rows = []
    for mode in args.modes:
        scores = threshold_curve(models, val_loader, device, mode, thresholds)
        rows.extend({'model': model_type, 'mode': mode, 'threshold': float(t),
                     'val_dice': float(s)} for t, s in zip(thresholds, scores))
    curve = pd.DataFrame(rows)
    best = curve.sort_values(['val_dice', 'mode', 'threshold'],
                             ascending=[False, True, True]).iloc[0]
    selection = {
        'model': model_type,
        'ensemble_seeds': args.seeds,
        'ensemble_rule': 'Equal probability average across three independently trained seeds',
        'mode': str(best['mode']),
        'threshold': float(best['threshold']),
        'kvasir_val_dice': float(best['val_dice']),
        'selection_dataset': 'Kvasir-SEG fixed validation subset only',
    }
    calibration_dir = Path(args.output_root) / 'calibration'
    calibration_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(calibration_dir / f'{model_type}_ensemble_curve.csv', index=False)
    (calibration_dir / f'{model_type}_locked_selection.json').write_text(
        json.dumps(selection, indent=2), encoding='utf-8')
    print('LOCKED_SELECTION', json.dumps(selection), flush=True)

    # CVC loaders are intentionally created only after the source-only choice is locked.
    cvc = get_dataloaders('cvc', args.batch_size, args.num_workers, args.image_size,
                          seed=args.seeds[0], dataset_root=args.cvc_root)
    cvc_dedup = get_dataloaders(
        'cvc', args.batch_size, args.num_workers, args.image_size,
        seed=args.seeds[0], dataset_root=args.cvc_root, deduplicate_exact_pairs=True)
    for dataset, loader in [('kvasir', kvasir_test), ('cvc', cvc),
                            ('cvc_dedup', cvc_dedup)]:
        metrics = evaluate_loader(models, loader, device, selection['mode'],
                                  selection['threshold'])
        aggregate = metrics.get_results()
        output = Path(args.output_root) / model_type / dataset
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metrics.rows).to_csv(output / 'per_image_metrics.csv', index=False)
        summary = {
            'model': model_type,
            'dataset': dataset,
            'ensemble_seeds': args.seeds,
            'ensemble_rule': selection['ensemble_rule'],
            'split_seed': args.split_seed,
            'checkpoint_paths': [str(path.resolve()) for path in checkpoint_paths],
            'checkpoint_sha256': [checkpoint_sha256(path) for path in checkpoint_paths],
            'best_epochs': [checkpoint.get('epoch') for checkpoint in checkpoints],
            'deduplicate_exact_pairs': dataset == 'cvc_dedup',
            'calibration_dataset': selection['selection_dataset'],
            'inference_mode': selection['mode'],
            'threshold': selection['threshold'],
            **aggregate,
        }
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        pd.DataFrame([summary]).to_csv(output / 'summary.csv', index=False)
        print(json.dumps({'model': model_type, 'dataset': dataset,
                          'mode': selection['mode'], 'threshold': selection['threshold'],
                          **aggregate}), flush=True)
    return selection


def main(args):
    set_seed(args.seeds[0])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    selections = {}
    for model_type in ['baseline', 'spc']:
        selections[model_type] = run_model_family(args, model_type, device)
    calibration_dir = Path(args.output_root) / 'calibration'
    (calibration_dir / 'locked_selections.json').write_text(
        json.dumps(selections, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_root', required=True)
    parser.add_argument('--kvasir_root', required=True)
    parser.add_argument('--cvc_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--image_size', type=int, default=352)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--split_seed', type=int, default=20260822)
    parser.add_argument('--split_file', default='splits/kvasir_fixed_split.json')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--modes', nargs='+', choices=['plain', 'hflip'], default=['plain', 'hflip'])
    parser.add_argument('--threshold_min', type=float, default=0.20)
    parser.add_argument('--threshold_max', type=float, default=0.65)
    parser.add_argument('--threshold_step', type=float, default=0.025)
    main(parser.parse_args())
