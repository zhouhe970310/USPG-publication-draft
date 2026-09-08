import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy.stats import t


METRICS = ('dice', 'iou', 'hd95', 'mae')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results_clean')
    parser.add_argument('--dataset', default='cvc')
    parser.add_argument('--output', default='results_clean/summary_submission.csv')
    parser.add_argument('--require_uspg_improvement', action='store_true')
    args = parser.parse_args()
    files = sorted(Path(args.root).glob(f'*/seed*/{args.dataset}/summary.json'))
    if not files:
        raise SystemExit(f'No summary.json files found under {args.root}.')
    records, keys = [], set()
    for path in files:
        record = json.loads(path.read_text(encoding='utf-8'))
        key = (record['model'], record['dataset'], int(record['seed']))
        if key in keys:
            raise RuntimeError(f'Duplicate run identity: {key}')
        keys.add(key)
        records.append(record)
    grouped, rows = defaultdict(list), []
    for record in records:
        grouped[(record['model'], record['dataset'])].append(record)
    for (model, dataset), group in sorted(grouped.items()):
        row = {
            'model': model,
            'dataset': dataset,
            'n_seeds': len(group),
            'seeds': ','.join(str(r['seed']) for r in sorted(group, key=lambda x: x['seed'])),
        }
        for metric in METRICS:
            values = pd.Series([float(r[metric]) for r in group])
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else float('nan')
            half = float(t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values))) \
                if len(values) > 1 else float('nan')
            row[f'{metric}_mean'] = mean
            row[f'{metric}_std_seeds'] = std
            row[f'{metric}_ci95_low'] = mean - half if len(values) > 1 else float('nan')
            row[f'{metric}_ci95_high'] = mean + half if len(values) > 1 else float('nan')
        rows.append(row)
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    printable = frame.fillna('')
    header = '| ' + ' | '.join(printable.columns) + ' |'
    divider = '| ' + ' | '.join(['---'] * len(printable.columns)) + ' |'
    body = [
        '| ' + ' | '.join(str(value) for value in row) + ' |'
        for row in printable.itertuples(index=False, name=None)
    ]
    output.with_suffix('.md').write_text(
        '\n'.join([header, divider, *body]) + '\n', encoding='utf-8'
    )
    print(frame.to_string(index=False))
    if args.require_uspg_improvement:
        base = frame.loc[frame.model == 'baseline', 'dice_mean']
        full = frame.loc[frame.model == 'uspg', 'dice_mean']
        if base.empty or full.empty or float(full.iloc[0]) <= float(base.iloc[0]):
            print('GATE FAILED: USPG-Net does not exceed the baseline Dice.')
            raise SystemExit(3)
        print('GATE PASSED: USPG-Net exceeds the baseline Dice.')


if __name__ == '__main__':
    main()
