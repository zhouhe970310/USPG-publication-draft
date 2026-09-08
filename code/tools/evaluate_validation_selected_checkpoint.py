"""Evaluate one checkpoint per family selected strictly on source validation."""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import get_dataloaders
from test import checkpoint_sha256, load_model, set_seed
from tools.evaluate_calibrated_tta import evaluate_loader


def select_from_source(calibration_root, model_type):
    curve_path = Path(calibration_root) / f'{model_type}_per_seed_curve.csv'
    curve = pd.read_csv(curve_path)
    best = curve.sort_values(
        ['val_dice', 'seed', 'mode', 'threshold'],
        ascending=[False, True, True, True],
    ).iloc[0]
    return {
        'model': model_type,
        'seed': int(best['seed']),
        'mode': str(best['mode']),
        'threshold': float(best['threshold']),
        'kvasir_val_dice': float(best['val_dice']),
        'selection_dataset': 'Kvasir-SEG fixed validation subset only',
        'selection_rule': 'Highest calibrated validation Dice among seeds 42-44',
    }


def evaluate_selection(args, selection, device):
    model_type, seed = selection['model'], selection['seed']
    set_seed(seed)
    checkpoint_path = Path(args.runs_root) / model_type / f'seed{seed}' / 'best_model.pth'
    model, checkpoint = load_model(checkpoint_path, model_type, device)
    _, _, kvasir_test = get_dataloaders(
        'kvasir', args.batch_size, args.num_workers, args.image_size,
        seed=seed, split_seed=args.split_seed, split_file=args.split_file,
        dataset_root=args.kvasir_root)
    # CVC is instantiated only after the source-selected configuration is locked.
    cvc = get_dataloaders('cvc', args.batch_size, args.num_workers, args.image_size,
                          seed=seed, dataset_root=args.cvc_root)
    cvc_dedup = get_dataloaders(
        'cvc', args.batch_size, args.num_workers, args.image_size,
        seed=seed, dataset_root=args.cvc_root, deduplicate_exact_pairs=True)
    for dataset, loader in [('kvasir', kvasir_test), ('cvc', cvc),
                            ('cvc_dedup', cvc_dedup)]:
        metrics = evaluate_loader(model, loader, device, selection['mode'],
                                  selection['threshold'])
        aggregate = metrics.get_results()
        output = Path(args.output_root) / model_type / dataset
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metrics.rows).to_csv(output / 'per_image_metrics.csv', index=False)
        summary = {
            **selection,
            'dataset': dataset,
            'split_seed': args.split_seed,
            'checkpoint': str(checkpoint_path.resolve()),
            'checkpoint_sha256': checkpoint_sha256(checkpoint_path),
            'best_epoch': checkpoint.get('epoch'),
            'deduplicate_exact_pairs': dataset == 'cvc_dedup',
            **aggregate,
        }
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        pd.DataFrame([summary]).to_csv(output / 'summary.csv', index=False)
        print(json.dumps({'model': model_type, 'dataset': dataset,
                          'seed': seed, 'mode': selection['mode'],
                          'threshold': selection['threshold'], **aggregate}), flush=True)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    selections = {
        model: select_from_source(args.calibration_root, model)
        for model in ['baseline', 'spc']
    }
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'locked_selections.json').write_text(
        json.dumps(selections, indent=2), encoding='utf-8')
    for selection in selections.values():
        print('LOCKED_SELECTION', json.dumps(selection), flush=True)
    for selection in selections.values():
        evaluate_selection(args, selection, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_root', required=True)
    parser.add_argument('--calibration_root', required=True)
    parser.add_argument('--kvasir_root', required=True)
    parser.add_argument('--cvc_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--image_size', type=int, default=352)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--split_seed', type=int, default=20260822)
    parser.add_argument('--split_file', default='splits/kvasir_fixed_split.json')
    main(parser.parse_args())
