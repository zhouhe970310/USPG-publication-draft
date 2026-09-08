import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    root = Path(root)
    image_dir, mask_dir = root / 'images', root / 'masks'
    images = {p.name: p for p in image_dir.iterdir() if p.suffix.lower() in SUFFIXES}
    masks = {p.name: p for p in mask_dir.iterdir() if p.suffix.lower() in SUFFIXES}
    image_groups, pair_groups = defaultdict(list), defaultdict(list)
    for name, path in images.items():
        image_hash = digest(path)
        image_groups[image_hash].append(name)
        if name in masks:
            pair_hash = hashlib.sha256(
                bytes.fromhex(image_hash) + bytes.fromhex(digest(masks[name]))
            ).hexdigest()
            pair_groups[pair_hash].append(name)
    duplicate_images = [sorted(v) for v in image_groups.values() if len(v) > 1]
    duplicate_pairs = [sorted(v) for v in pair_groups.values() if len(v) > 1]
    inconsistent = []
    for names in duplicate_images:
        mask_hashes = {digest(masks[name]) for name in names if name in masks}
        if len(mask_hashes) > 1:
            inconsistent.append(names)
    return {
        'root': str(root.resolve()),
        'n_images': len(images),
        'n_masks': len(masks),
        'missing_masks': sorted(set(images) - set(masks)),
        'orphan_masks': sorted(set(masks) - set(images)),
        'duplicate_image_groups': duplicate_images,
        'duplicate_image_extras': sum(len(g) - 1 for g in duplicate_images),
        'duplicate_pair_groups': duplicate_pairs,
        'duplicate_pair_extras': sum(len(g) - 1 for g in duplicate_pairs),
        'same_image_different_mask_groups': inconsistent,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--output', default=None)
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    report = audit(args.root)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    fatal = bool(report['missing_masks'] or report['orphan_masks'])
    suspicious = bool(report['duplicate_pair_extras'] or report['same_image_different_mask_groups'])
    if args.strict and (fatal or suspicious):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
