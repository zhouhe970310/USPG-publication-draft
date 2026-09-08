import argparse
import hashlib
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from data.dataset import get_dataloaders
from models.uspg_net import build_model
from utils.metrics import Metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path, requested_type, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_type = checkpoint.get('model_type', requested_type)
    if saved_type != requested_type:
        raise ValueError(f'Checkpoint is {saved_type}, not {requested_type}.')
    model = build_model(
        saved_type,
        spc_weight=checkpoint.get('spc_weight', 0.15),
        ugrl_weight=checkpoint.get('ugrl_weight', 0.03),
        dsp_prob=checkpoint.get('dsp_prob', 0.5),
        spc_boundary_weight=checkpoint.get('spc_boundary_weight', 0.2),
        spc_confidence_margin=checkpoint.get('spc_confidence_margin', 0.15),
        perturbed_seg_weight=checkpoint.get('perturbed_seg_weight', 1.0),
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.to(device).eval(), checkpoint


def enable_mc_dropout(model):
    types = (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d, torch.nn.AlphaDropout)
    count = 0
    for module in model.modules():
        if isinstance(module, types):
            module.train()
            count += 1
    if count == 0:
        raise RuntimeError('Uncertainty evaluation requested but no dropout layer exists.')


@torch.no_grad()
def infer(model, images, mc_samples):
    model.eval()
    probability = torch.sigmoid(model(images))
    if mc_samples <= 1:
        return probability, None
    enable_mc_dropout(model)
    samples = torch.stack([torch.sigmoid(model(images)) for _ in range(mc_samples)])
    model.eval()
    # Accuracy is always computed from the same deterministic prediction for all
    # models. MC samples are used only to estimate uncertainty.
    return probability, samples.var(0, unbiased=False)


def unnormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def save_visualization(path, image, target, probability, uncertainty=None):
    target = target.squeeze().cpu().numpy()
    probability = probability.squeeze().cpu().numpy()
    prediction = probability > 0.5
    error = np.logical_xor(prediction, target > 0.5)
    panels = 5 if uncertainty is not None else 4
    fig, axes = plt.subplots(1, panels, figsize=(3.3 * panels, 3.4))
    axes[0].imshow(unnormalize(image)); axes[0].set_title('Image')
    axes[1].imshow(target, cmap='gray', vmin=0, vmax=1); axes[1].set_title('Ground truth')
    axes[2].imshow(probability, cmap='gray', vmin=0, vmax=1); axes[2].set_title('Probability')
    axes[3].imshow(error, cmap='magma', vmin=0, vmax=1); axes[3].set_title('Error')
    if uncertainty is not None:
        axes[4].imshow(uncertainty.squeeze().cpu().numpy(), cmap='viridis')
        axes[4].set_title('MC uncertainty')
    for axis in axes:
        axis.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def evaluate(args):
    set_seed(args.seed)
    if args.dataset == 'external' and not args.dataset_label:
        raise ValueError('--dataset_label is required with --dataset external.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint = load_model(args.checkpoint, args.model_type, device)
    if args.dataset == 'kvasir':
        _, _, loader = get_dataloaders(
            'kvasir', args.batch_size, args.num_workers, args.image_size,
            seed=args.seed, split_seed=args.split_seed, split_file=args.split_file,
            dataset_root=args.dataset_root,
        )
    else:
        loader = get_dataloaders(
            args.dataset, args.batch_size, args.num_workers, args.image_size,
            seed=args.seed, dataset_root=args.dataset_root,
            deduplicate_exact_pairs=args.deduplicate_exact_pairs,
        )
    dataset_tag = args.dataset_label or args.dataset
    output = Path(args.output_root) / args.model_type / f'seed{args.seed}' / dataset_tag
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = output / 'predictions'
    visualization_dir = output / 'visualizations'
    if args.save_predictions:
        prediction_dir.mkdir(exist_ok=True)
    if args.visualize > 0:
        visualization_dir.mkdir(exist_ok=True)

    metrics, uncertainty_means, visualized = Metrics(), {}, 0
    for images, masks, names in tqdm(loader, desc=f'Test {args.dataset}'):
        images, masks = images.to(device), masks.to(device)
        probability, uncertainty = infer(model, images, args.mc_samples)
        metrics.update(probability, masks, names)
        for index, name in enumerate(names):
            if uncertainty is not None:
                uncertainty_means[name] = float(uncertainty[index].mean().cpu())
            if args.save_predictions:
                array = (probability[index].squeeze().cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(array).save(prediction_dir / f'{Path(name).stem}.png')
            if visualized < args.visualize:
                save_visualization(
                    visualization_dir / f'{Path(name).stem}.png', images[index], masks[index],
                    probability[index], None if uncertainty is None else uncertainty[index]
                )
                visualized += 1

    aggregate = metrics.get_results()
    per_image = pd.DataFrame(metrics.rows)
    if uncertainty_means:
        per_image['uncertainty_mean'] = per_image['image'].map(uncertainty_means)
        if per_image['uncertainty_mean'].std() > 0:
            aggregate['uncertainty_error_pearson'] = float(
                per_image['uncertainty_mean'].corr(1.0 - per_image['dice'])
            )
    per_image.to_csv(output / 'per_image_metrics.csv', index=False)
    summary = {
        'model': args.model_type,
        'dataset': dataset_tag,
        'seed': args.seed,
        'split_seed': args.split_seed,
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': checkpoint_sha256(args.checkpoint),
        'best_epoch': checkpoint.get('epoch'),
        'best_val_dice': checkpoint.get('best_dice'),
        'init_checkpoint': checkpoint.get('init_checkpoint'),
        'deduplicate_exact_pairs': args.deduplicate_exact_pairs,
        'mc_samples': args.mc_samples,
        **aggregate,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    pd.DataFrame([summary]).to_csv(output / 'summary.csv', index=False)
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True,
                        choices=['baseline', 'dsp', 'spc', 'uspg',
                                 'resunet', 'resunet_dsp',
                                 'resunet_spc_region', 'resunet_spc',
                                 'unet', 'unetpp', 'deeplabv3p'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', choices=['kvasir', 'cvc', 'external'], default='cvc')
    parser.add_argument('--dataset_label', default=None,
                        help='Required manuscript label when --dataset external is used.')
    parser.add_argument('--dataset_root', default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=352)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_seed', type=int, default=20260822)
    parser.add_argument('--split_file', default='splits/kvasir_fixed_split.json')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mc_samples', type=int, default=1)
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--visualize', type=int, default=12)
    parser.add_argument('--deduplicate_exact_pairs', action='store_true')
    parser.add_argument('--output_root', default='results_clean')
    return parser.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
