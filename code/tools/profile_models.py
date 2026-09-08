"""Record parameter counts and reproducible CPU inference latency."""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.uspg_net import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(8)
    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    rows = []
    for model_type in ['resunet', 'resunet_spc', 'unetpp', 'deeplabv3p']:
        model = build_model(model_type).eval()
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(x)
            timings = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                model(x)
                timings.append((time.perf_counter() - start) * 1000 / args.batch_size)
        rows.append({
            'model': model_type,
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'cpu_latency_ms_per_image_mean': sum(timings) / len(timings),
            'cpu_latency_ms_per_image_sd': float(torch.tensor(timings).std().item()),
            'batch_size': args.batch_size,
            'image_size': args.image_size,
            'threads': torch.get_num_threads(),
            'repeats': args.repeats,
        })
        del model
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    output.with_suffix('.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
