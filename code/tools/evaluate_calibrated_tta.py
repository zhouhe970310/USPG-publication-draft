"""Source-only threshold/TTA calibration followed by locked final evaluation.

A single inference mode and probability threshold are chosen for each model
family by maximizing mean Dice across seeds on the fixed Kvasir validation
subset.  The selected configuration is then locked before any CVC prediction.
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
def predict(model, images, mode):
    probability = torch.sigmoid(model(images))
    if mode == 'hflip':
        flipped = torch.flip(images, dims=(-1,))
        probability_flip = torch.flip(torch.sigmoid(model(flipped)), dims=(-1,))
        probability = 0.5 * (probability + probability_flip)
    elif mode != 'plain':
        raise ValueError(mode)
    return probability


@torch.no_grad()
def threshold_curve(model, loader, device, mode, thresholds):
    totals = torch.zeros(len(thresholds), dtype=torch.float64)
    count = 0
    threshold_tensor = torch.tensor(thresholds, device=device).view(-1, 1, 1, 1, 1)
    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        probability = predict(model, images, mode)
        binary = probability.unsqueeze(0) > threshold_tensor
        target = (masks > 0.5).unsqueeze(0)
        axes = (2, 3, 4)
        intersection = (binary & target).sum(dim=axes).float()
        denominator = binary.sum(dim=axes).float() + target.sum(dim=axes).float()
        dice = (2 * intersection + 1e-7) / (denominator + 1e-7)
        totals += dice.sum(dim=1).double().cpu()
        count += masks.shape[0]
    return (totals / count).numpy()


def choose_configuration(args, model_type, device):
    thresholds = np.round(np.arange(args.threshold_min, args.threshold_max + 1e-9,
                                    args.threshold_step), 4)
    rows = []
    for seed in args.seeds:
        checkpoint = Path(args.runs_root) / model_type / f'seed{seed}' / 'best_model.pth'
        model, _ = load_model(checkpoint, model_type, device)
        _, val_loader, _ = get_dataloaders(
            'kvasir', args.batch_size, args.num_workers, args.image_size,
            seed=seed, split_seed=args.split_seed, split_file=args.split_file,
            dataset_root=args.kvasir_root,
        )
        for mode in args.modes:
            scores = threshold_curve(model, val_loader, device, mode, thresholds)
            for threshold, score in zip(thresholds, scores):
                rows.append({
                    'model': model_type, 'seed': seed, 'mode': mode,
                    'threshold': threshold, 'val_dice': score,
                })
        del model
    frame = pd.DataFrame(rows)
    mean_curve = frame.groupby(['model', 'mode', 'threshold'], as_index=False).val_dice.mean()
    best = mean_curve.sort_values(
        ['val_dice', 'mode', 'threshold'], ascending=[False, True, True]
    ).iloc[0]
    return {
        'model': model_type,
        'mode': str(best['mode']),
        'threshold': float(best['threshold']),
        'mean_kvasir_val_dice': float(best['val_dice']),
        'selection_dataset': 'Kvasir-SEG fixed validation subset only',
        'selection_rule': (
            'One common mode and threshold per model across seeds '
            + ','.join(map(str, args.seeds))
        ),
    }, frame, mean_curve


@torch.no_grad()
def evaluate_loader(model, loader, device, mode, threshold):
    metrics = Metrics(threshold=threshold)
    for images, masks, names in loader:
        images, masks = images.to(device), masks.to(device)
        metrics.update(predict(model, images, mode), masks, names)
    return metrics


def final_evaluate(args, model_type, selection, device):
    for seed in args.seeds:
        set_seed(seed)
        checkpoint_path = Path(args.runs_root) / model_type / f'seed{seed}' / 'best_model.pth'
        model, checkpoint = load_model(checkpoint_path, model_type, device)
        _, _, kvasir_test = get_dataloaders(
            'kvasir', args.batch_size, args.num_workers, args.image_size,
            seed=seed, split_seed=args.split_seed, split_file=args.split_file,
            dataset_root=args.kvasir_root,
        )
        cvc = get_dataloaders(
            'cvc', args.batch_size, args.num_workers, args.image_size,
            seed=seed, dataset_root=args.cvc_root,
        )
        cvc_dedup = get_dataloaders(
            'cvc', args.batch_size, args.num_workers, args.image_size,
            seed=seed, dataset_root=args.cvc_root, deduplicate_exact_pairs=True,
        )
        for dataset, loader in [
            ('kvasir', kvasir_test), ('cvc', cvc), ('cvc_dedup', cvc_dedup)
        ]:
            metrics = evaluate_loader(
                model, loader, device, selection['mode'], selection['threshold']
            )
            aggregate = metrics.get_results()
            output = Path(args.output_root) / model_type / f'seed{seed}' / dataset
            output.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(metrics.rows).to_csv(output / 'per_image_metrics.csv', index=False)
            summary = {
                'model': model_type,
                'dataset': dataset,
                'seed': seed,
                'split_seed': args.split_seed,
                'checkpoint': str(checkpoint_path.resolve()),
                'checkpoint_sha256': checkpoint_sha256(checkpoint_path),
                'best_epoch': checkpoint.get('epoch'),
                'best_val_dice': checkpoint.get('best_dice'),
                'init_checkpoint': checkpoint.get('init_checkpoint'),
                'deduplicate_exact_pairs': dataset == 'cvc_dedup',
                'mc_samples': 1,
                'calibration_dataset': selection['selection_dataset'],
                'inference_mode': selection['mode'],
                'threshold': selection['threshold'],
                **aggregate,
            }
            (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            pd.DataFrame([summary]).to_csv(output / 'summary.csv', index=False)
            print(json.dumps({
                'model': model_type, 'seed': seed, 'dataset': dataset,
                'mode': selection['mode'], 'threshold': selection['threshold'],
                **aggregate,
            }))


def main(args):
    set_seed(args.seeds[0])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    calibration_dir = Path(args.output_root) / 'calibration'
    calibration_dir.mkdir(parents=True, exist_ok=True)
    selections = {}
    for model_type in args.model_types:
        selection, per_seed, mean_curve = choose_configuration(args, model_type, device)
        selections[model_type] = selection
        per_seed.to_csv(calibration_dir / f'{model_type}_per_seed_curve.csv', index=False)
        mean_curve.to_csv(calibration_dir / f'{model_type}_mean_curve.csv', index=False)
        print('LOCKED_SELECTION', json.dumps(selection))
    (calibration_dir / 'locked_selections.json').write_text(
        json.dumps(selections, indent=2), encoding='utf-8'
    )
    # Only after both configurations are locked do we instantiate CVC loaders.
    for model_type in args.model_types:
        final_evaluate(args, model_type, selections[model_type], device)


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
    parser.add_argument('--model_types', nargs='+',
                        default=['baseline', 'spc'])
    main(parser.parse_args())
