import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.dataset import get_dataloaders
from losses.losses import BCEDiceLoss
from models.uspg_net import build_model
from utils.metrics import Metrics


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, args):
    model.train()
    losses, components = [], {}
    auxiliary_scale = min(1.0, max(0.0, (epoch - args.aux_warmup) / max(1, args.aux_ramp)))
    for images, masks, _ in tqdm(loader, desc=f'Train {epoch}/{args.epochs}',
                                 disable=args.quiet):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp and device.type == 'cuda'):
            if args.model_type in {'baseline', 'resunet', 'unet', 'unetpp',
                                   'deeplabv3p'}:
                logits = model(images)
                loss = criterion(logits, masks)
                loss_dict = {'seg_loss': loss.detach().item()}
            else:
                outputs = model(images, return_all=True)
                loss, loss_dict = model.compute_loss(
                    outputs, masks, criterion, auxiliary_scale=auxiliary_scale
                )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.detach().item())
        for key, value in loss_dict.items():
            components.setdefault(key, []).append(value)
    summary = {key: float(np.mean(value)) for key, value in components.items()}
    summary['auxiliary_scale'] = auxiliary_scale
    return float(np.mean(losses)), summary


@torch.no_grad()
def validate(model, loader, criterion, device, quiet=False):
    model.eval()
    losses, metrics = [], Metrics()
    for images, masks, names in tqdm(loader, desc='Validate', disable=quiet):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        losses.append(criterion(logits, masks).item())
        metrics.update(torch.sigmoid(logits), masks, names)
    return float(np.mean(losses)), metrics.get_results()


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_dice, args):
    payload = {
        'epoch': epoch,
        'best_dice': best_dice,
        'model_type': args.model_type,
        'seed': args.seed,
        'split_seed': args.split_seed,
        'image_size': args.image_size,
        'spc_weight': args.spc_weight,
        'ugrl_weight': args.ugrl_weight,
        'dsp_prob': args.dsp_prob,
        'spc_boundary_weight': args.spc_boundary_weight,
        'spc_confidence_margin': args.spc_confidence_margin,
        'perturbed_seg_weight': args.perturbed_seg_weight,
        'init_checkpoint': args.init_checkpoint,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(payload, path)


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('WARNING: CUDA not available; training will be slow.')
    train_loader, val_loader, _ = get_dataloaders(
        'kvasir', args.batch_size, args.num_workers, args.image_size,
        seed=args.seed, split_seed=args.split_seed, split_file=args.split_file,
        dataset_root=args.kvasir_root,
    )
    model = build_model(
        args.model_type, spc_weight=args.spc_weight, ugrl_weight=args.ugrl_weight,
        dsp_prob=args.dsp_prob, spc_boundary_weight=args.spc_boundary_weight,
        spc_confidence_margin=args.spc_confidence_margin,
        perturbed_seg_weight=args.perturbed_seg_weight,
    ).to(device)
    initial = None
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(initial['model_state_dict'], strict=False)
        invalid_missing = [key for key in missing if key.startswith('backbone.')]
        if invalid_missing or unexpected:
            raise RuntimeError(
                f'Unsafe initialization. Missing backbone keys={invalid_missing}, '
                f'unexpected keys={unexpected}'
            )
        print(f'Initialized from shared backbone checkpoint: {args.init_checkpoint}')
    criterion = BCEDiceLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler(enabled=args.amp and device.type == 'cuda')

    run_dir = Path(args.output_root) / args.model_type / f'seed{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding='utf-8'
    )
    history, best_dice, best_epoch, stale = [], -1.0, 0, 0
    best_path = run_dir / 'best_model.pth'
    if args.preserve_init_if_better and initial is not None:
        best_dice = float(initial.get('best_dice', -1.0))
        save_checkpoint(best_path, model, optimizer, scheduler, 0, best_dice, args)
        print(f'Preserving initial checkpoint unless validation Dice exceeds {best_dice:.6f}.')
    for epoch in range(1, args.epochs + 1):
        train_loss, parts = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args
        )
        val_loss, val = validate(model, val_loader, criterion, device, args.quiet)
        scheduler.step()
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val['dice'],
            'val_iou': val['iou'],
            'val_hd95': val['hd95'],
            'val_mae': val['mae'],
            'lr': optimizer.param_groups[0]['lr'],
            **parts,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / 'training_history.csv', index=False)
        print(f"Epoch {epoch}: val Dice={val['dice']:.4f}, best={best_dice:.4f}")
        if val['dice'] > best_dice + args.min_delta:
            best_dice, best_epoch, stale = val['dice'], epoch, 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_dice, args)
        else:
            stale += 1
        if epoch >= args.min_epochs and stale >= args.patience:
            print(f'Early stopping at epoch {epoch}; best epoch={best_epoch}.')
            break
    print(f'BEST_CHECKPOINT={best_path}')
    print(f'BEST_VAL_DICE={best_dice:.6f}')
    return best_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True,
                        choices=['baseline', 'dsp', 'spc', 'uspg',
                                 'resunet', 'resunet_dsp',
                                 'resunet_spc_region', 'resunet_spc',
                                 'unet', 'unetpp', 'deeplabv3p'])
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--min_epochs', type=int, default=22)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=352)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--spc_weight', type=float, default=0.15)
    parser.add_argument('--ugrl_weight', type=float, default=0.03)
    parser.add_argument('--dsp_prob', type=float, default=0.5)
    parser.add_argument('--spc_boundary_weight', type=float, default=0.2)
    parser.add_argument('--spc_confidence_margin', type=float, default=0.15)
    parser.add_argument('--perturbed_seg_weight', type=float, default=1.0)
    parser.add_argument('--aux_warmup', type=int, default=3)
    parser.add_argument('--aux_ramp', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_seed', type=int, default=20260822)
    parser.add_argument('--split_file', default='splits/kvasir_fixed_split.json')
    parser.add_argument('--kvasir_root', default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_root', default='runs')
    parser.add_argument('--init_checkpoint', default=None,
                        help='Shared baseline checkpoint for fair two-stage fine-tuning.')
    parser.add_argument('--preserve_init_if_better', action='store_true',
                        help='Keep the initialized checkpoint unless source validation improves.')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
