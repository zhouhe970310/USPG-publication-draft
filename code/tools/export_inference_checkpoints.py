"""Strip optimizer state from final checkpoints for portable inference archives."""

import argparse
from pathlib import Path

import torch


KEEP = {
    'epoch', 'best_dice', 'model_type', 'seed', 'split_seed', 'image_size',
    'spc_weight', 'ugrl_weight', 'dsp_prob', 'spc_boundary_weight',
    'spc_confidence_margin', 'perturbed_seg_weight', 'init_checkpoint',
    'model_state_dict',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.runs_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for model in ('resunet', 'resunet_spc'):
        for seed in (42, 43, 44):
            source = root / model / f'seed{seed}' / 'best_model.pth'
            checkpoint = torch.load(source, map_location='cpu', weights_only=False)
            compact = {key: value for key, value in checkpoint.items() if key in KEEP}
            target = output / f'{model}_seed{seed}_inference.pth'
            torch.save(compact, target)
            print(target, target.stat().st_size)


if __name__ == '__main__':
    main()
