import hashlib
import json
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


def _resolve_dataset_root(name, explicit_root=None):
    candidates = []
    if explicit_root:
        candidates.append(Path(explicit_root))
    env_root = os.environ.get('USPG_DATA_ROOT')
    if env_root:
        candidates.append(Path(env_root) / name)
    project_root = Path(__file__).resolve().parents[1]
    candidates.extend([
        project_root / 'datasets' / name,
        project_root.parent / name,
        Path.cwd() / 'datasets' / name,
    ])
    for path in candidates:
        if (path / 'images').is_dir() and (path / 'masks').is_dir():
            return path
    searched = '\n'.join(f'  - {p}' for p in candidates)
    raise FileNotFoundError(f'Cannot locate {name}. Searched:\n{searched}')


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    worker = torch.utils.data.get_worker_info()
    transform = getattr(worker.dataset, 'transform', None) if worker else None
    if transform is not None and hasattr(transform, 'set_random_seed'):
        transform.set_random_seed(worker_seed)


class PolypDataset(Dataset):
    def __init__(self, root_dir, image_size=352, mode='train', image_names=None,
                 deduplicate_exact_pairs=False):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / 'images'
        self.mask_dir = self.root_dir / 'masks'
        self.image_size = image_size
        self.mode = mode
        names = image_names or sorted(
            p.name for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        missing = [name for name in names if not (self.mask_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} masks are missing; first: {missing[0]}')
        if deduplicate_exact_pairs:
            names = self._deduplicate_pairs(names)
        self.images = list(names)
        self.transform = self._build_transform()

    def _deduplicate_pairs(self, names):
        retained, seen = [], set()
        for name in names:
            digest = hashlib.sha256(
                (self.image_dir / name).read_bytes() + b'\0' +
                (self.mask_dir / name).read_bytes()
            ).digest()
            if digest not in seen:
                retained.append(name)
                seen.add(digest)
        return retained

    def _build_transform(self):
        transforms = [A.Resize(self.image_size, self.image_size)]
        if self.mode == 'train':
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.08, 0.08),
                         rotate=(-25, 25), shear=(-5, 5), p=0.5),
            ])
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        return A.Compose(transforms)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        image = cv2.imread(str(self.image_dir / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.mask_dir / name), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f'Failed to read image/mask pair: {name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype(np.float32)
        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return image, mask.float(), name


def _load_or_create_split(root_dir, split_file, split_seed):
    root_dir = Path(root_dir)
    all_images = sorted(
        p.name for p in (root_dir / 'images').iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    split_file = Path(split_file)
    if split_file.is_file():
        payload = json.loads(split_file.read_text(encoding='utf-8'))
        assigned = payload['train'] + payload['val'] + payload['test']
        if sorted(assigned) != all_images:
            raise RuntimeError(f'Split manifest {split_file} does not match the current dataset.')
        return payload
    rng = np.random.default_rng(split_seed)
    shuffled = np.asarray(all_images)[rng.permutation(len(all_images))].tolist()
    n_train = int(0.8 * len(shuffled))
    n_val = int(0.1 * len(shuffled))
    payload = {
        'dataset': 'Kvasir-SEG',
        'split_seed': int(split_seed),
        'train': shuffled[:n_train],
        'val': shuffled[n_train:n_train + n_val],
        'test': shuffled[n_train + n_val:],
    }
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload


def get_dataloaders(dataset_name='kvasir', batch_size=16, num_workers=4,
                    image_size=352, seed=42, split_seed=20260822,
                    split_file='splits/kvasir_fixed_split.json',
                    dataset_root=None, deduplicate_exact_pairs=False):
    generator = torch.Generator().manual_seed(seed)
    if dataset_name == 'kvasir':
        root = _resolve_dataset_root('Kvasir-SEG', dataset_root)
        split = _load_or_create_split(root, split_file, split_seed)
        train_set = PolypDataset(root, image_size, 'train', split['train'])
        val_set = PolypDataset(root, image_size, 'test', split['val'])
        test_set = PolypDataset(root, image_size, 'test', split['test'])
        common = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_seed_worker,
            persistent_workers=num_workers > 0,
        )
        return (
            DataLoader(train_set, shuffle=True, generator=generator, **common),
            DataLoader(val_set, shuffle=False, **common),
            DataLoader(test_set, shuffle=False, **common),
        )
    if dataset_name in {'cvc', 'external'}:
        if dataset_name == 'external':
            if not dataset_root:
                raise ValueError('dataset_root is required for an external dataset.')
            root = Path(dataset_root)
        else:
            root = _resolve_dataset_root('CVC-ColonDB', dataset_root)
        dataset = PolypDataset(
            root, image_size, 'test', deduplicate_exact_pairs=deduplicate_exact_pairs
        )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), worker_init_fn=_seed_worker,
            persistent_workers=num_workers > 0,
        )
    raise ValueError(f'Unknown dataset: {dataset_name}')
