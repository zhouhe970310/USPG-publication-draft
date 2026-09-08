#!/usr/bin/env python3
"""Three-seed ensemble qualitative comparison selected by explicit rules."""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import binary_erosion


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
COLORS = {'gt': np.array([.20, .70, .43]), 'base': np.array([.30, .48, .70]),
          'ours': np.array([.89, .33, .32]), 'fixed': np.array([.15, .68, .38]),
          'new': np.array([.63, .34, .76]), 'shared': np.array([.65, .65, .65])}


def load_model(project, checkpoint, model_type):
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from test import load_model as loader
    return loader(checkpoint, model_type, torch.device('cpu'))[0]


def preprocess(path, size):
    image = Image.open(path).convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, np.float32) / 255
    tensor = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1))
    return rgb, tensor


@torch.inference_mode()
def ensemble(project, runs, model_type, batch, chunk_size=16):
    probabilities = []
    for seed in (42, 43, 44):
        checkpoint = runs / model_type / f'seed{seed}' / 'best_model.pth'
        model = load_model(project, checkpoint, model_type)
        chunks = []
        for start in range(0, len(batch), chunk_size):
            current = batch[start:start + chunk_size]
            plain = torch.sigmoid(model(current))
            flip = torch.flip(torch.sigmoid(model(torch.flip(current, (-1,)))), (-1,))
            chunks.append(((plain + flip) / 2).cpu())
        probabilities.append(torch.cat(chunks))
        del model
    return torch.stack(probabilities).mean(0)[:, 0].numpy()


def dice(pred, target):
    return (2 * np.logical_and(pred, target).sum() + 1e-7) / (pred.sum() + target.sum() + 1e-7)


def overlay(image, mask, color):
    result = image.copy()
    result[mask] = .56 * result[mask] + .44 * color
    edge = np.logical_xor(mask, binary_erosion(mask, iterations=2, border_value=0))
    result[edge] = color
    return result


def change_map(base, ours, target):
    be, oe = np.logical_xor(base, target), np.logical_xor(ours, target)
    canvas = np.ones((*target.shape, 3)) * .965
    canvas[target] = np.array([.90, .94, .91])
    canvas[be & oe] = COLORS['shared']
    canvas[be & ~oe] = COLORS['fixed']
    canvas[~be & oe] = COLORS['new']
    return canvas


def main(args):
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 7.2, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    names = sorted(path.name for path in (args.dataset_root / 'images').iterdir()
                   if path.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    all_rgb, tensors, all_masks = [], [], []
    for name in names:
        image, tensor = preprocess(args.dataset_root / 'images' / name, args.image_size)
        mask = Image.open(args.dataset_root / 'masks' / name).convert('L').resize(
            (args.image_size, args.image_size), Image.Resampling.NEAREST)
        all_rgb.append(image); tensors.append(tensor); all_masks.append(np.asarray(mask) > 127)
    batch = torch.stack(tensors)
    base_prob = ensemble(args.project_root, args.runs_root, 'resunet', batch)
    ours_prob = ensemble(args.project_root, args.runs_root, 'resunet_spc', batch)
    all_base, all_ours = base_prob >= .5, ours_prob >= .5
    deltas = np.array([dice(op, target) - dice(bp, target)
                       for bp, op, target in zip(all_base, all_ours, all_masks)])
    positive = np.argsort(deltas)[-2:][::-1]
    typical = np.argsort(np.abs(deltas - np.median(deltas)))[:2]
    negative = np.argsort(deltas)[:2]
    indices = np.r_[positive, typical, negative]
    labels = ['largest gain', 'second-largest gain', 'typical effect',
              'typical effect', 'largest regression', 'second-largest regression']
    cases = [(names[index], label) for index, label in zip(indices, labels)]
    rgb = [all_rgb[index] for index in indices]
    masks = [all_masks[index] for index in indices]
    base = all_base[indices]
    ours = all_ours[indices]

    fig, axes = plt.subplots(6, 5, figsize=(7.16, 8.35))
    for col, title in enumerate(['Image', 'Ground truth', 'Baseline', 'DSP+SPC', 'Changed errors']):
        axes[0, col].set_title(title, fontweight='bold', pad=5)
    records = []
    for row, ((name, rule), image, target, bp, op) in enumerate(zip(cases, rgb, masks, base, ours)):
        bd, od = dice(bp, target), dice(op, target)
        records.append({'panel': chr(97 + row), 'image': name, 'selection_rule': rule,
                        'baseline_ensemble_dice': bd, 'ours_ensemble_dice': od,
                        'delta': od - bd})
        views = [image, overlay(image, target, COLORS['gt']),
                 overlay(image, bp, COLORS['base']), overlay(image, op, COLORS['ours']),
                 change_map(bp, op, target)]
        for col, view in enumerate(views):
            axes[row, col].imshow(view); axes[row, col].axis('off')
        axes[row, 0].text(-.07, .98, chr(97 + row), transform=axes[row, 0].transAxes,
                          va='top', ha='right', fontsize=9, fontweight='bold')
        axes[row, 0].text(.02, .03, f'{Path(name).stem} · {rule}',
                          transform=axes[row, 0].transAxes, color='white', fontsize=5.7,
                          bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': .62})
        axes[row, 2].text(.5, .025, f'Dice {bd:.3f}', transform=axes[row, 2].transAxes,
                          ha='center', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .82})
        axes[row, 3].text(.5, .025, f'Dice {od:.3f} ({od-bd:+.3f})',
                          transform=axes[row, 3].transAxes, ha='center',
                          bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .82})
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COLORS['fixed'], label='Corrected error'),
                        Patch(color=COLORS['new'], label='Introduced error'),
                        Patch(color=COLORS['shared'], label='Shared error')],
               loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .008))
    fig.subplots_adjust(left=.035, right=.995, top=.965, bottom=.045, wspace=.025, hspace=.055)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / 'Fig_4_qualitative.pdf', bbox_inches='tight')
    # 300 dpi remains publication-ready for this raster-heavy panel and avoids
    # excessive encoder memory on CPU-only environments.
    fig.savefig(args.output / 'Fig_4_qualitative.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(records).to_csv(args.output / 'Fig_4_qualitative_selection.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_root', type=Path, required=True)
    parser.add_argument('--dataset_root', type=Path, required=True)
    parser.add_argument('--runs_root', type=Path, required=True)
    parser.add_argument('--eval_root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image_size', type=int, default=256)
    main(parser.parse_args())
