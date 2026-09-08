"""Aggregate the optimized BSPC experiments and build manuscript-ready outputs."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = ('dice', 'iou', 'hd95', 'mae')
LABELS = {
    'resunet': 'U-Net (ResNet-18)',
    'resunet_dsp': '+ DSP',
    'resunet_spc_region': '+ DSP + region consistency',
    'resunet_spc': '+ DSP + SPC (ours)',
    'unetpp': 'U-Net++',
    'deeplabv3p': 'DeepLabV3+',
}
COLORS = {
    'resunet': '#4C78A8', 'resunet_dsp': '#72B7B2',
    'resunet_spc_region': '#F2CF5B', 'resunet_spc': '#E45756',
    'unetpp': '#B279A2', 'deeplabv3p': '#59A14F',
}


def style():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
        'legend.fontsize': 7.5, 'xtick.labelsize': 7.3,
        'ytick.labelsize': 7.3, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.linewidth': 0.75, 'savefig.dpi': 600,
    })


def polish(ax, axis='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis=axis, color='#E8EBEF', linewidth=0.65)
    ax.set_axisbelow(True)


def save(fig, output, stem):
    fig.savefig(output / f'{stem}.pdf', bbox_inches='tight', pad_inches=0.03)
    fig.savefig(output / f'{stem}.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def collect_summaries(results_root):
    records = []
    for path in sorted(results_root.glob('*/seed*/*/summary.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        row['source_file'] = str(path.resolve())
        records.append(row)
    if not records:
        raise RuntimeError(f'No evaluation summaries under {results_root}')
    return pd.DataFrame(records)


def aggregate(frame):
    rows = []
    for (model, dataset), group in frame.groupby(['model', 'dataset']):
        row = {'model': model, 'method': LABELS.get(model, model),
               'dataset': dataset, 'n_seeds': len(group)}
        for metric in METRICS:
            values = group[metric].astype(float)
            row[f'{metric}_mean'] = values.mean()
            row[f'{metric}_sd_seeds'] = values.std(ddof=1) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_statistics(results_root, output, datasets=('kvasir', 'cvc', 'cvc_dedup')):
    rows = []
    rng = np.random.default_rng(20260823)
    for dataset in datasets:
        by_model = {}
        for model in ('resunet', 'resunet_spc'):
            seed_frames = []
            for seed in (42, 43, 44):
                path = results_root / model / f'seed{seed}' / dataset / 'per_image_metrics.csv'
                if not path.is_file():
                    break
                seed_frames.append(pd.read_csv(path).set_index('image')['dice'].rename(str(seed)))
            if len(seed_frames) != 3:
                by_model = {}
                break
            by_model[model] = pd.concat(seed_frames, axis=1).mean(axis=1)
        if not by_model:
            continue
        paired = pd.concat(by_model, axis=1).dropna()
        delta = (paired['resunet_spc'] - paired['resunet']).to_numpy()
        boot = np.empty(10000)
        for index in range(len(boot)):
            boot[index] = rng.choice(delta, len(delta), replace=True).mean()
        try:
            test = wilcoxon(delta, alternative='greater', zero_method='pratt')
            statistic, pvalue = float(test.statistic), float(test.pvalue)
        except ValueError:
            statistic, pvalue = np.nan, 1.0
        rows.append({
            'dataset': dataset, 'n_images': len(delta),
            'mean_dice_baseline': paired['resunet'].mean(),
            'mean_dice_ours': paired['resunet_spc'].mean(),
            'mean_paired_delta': delta.mean(),
            'bootstrap_ci95_low': np.quantile(boot, 0.025),
            'bootstrap_ci95_high': np.quantile(boot, 0.975),
            'positive_image_fraction': (delta > 0).mean(),
            'wilcoxon_greater_statistic': statistic,
            'wilcoxon_greater_p': pvalue,
        })
    result = pd.DataFrame(rows)
    result.to_csv(output / 'paired_image_statistics.csv', index=False)
    return result


def plot_main(agg, output):
    datasets = ['kvasir', 'cvc', 'cvc_dedup']
    names = ['Kvasir-SEG', 'CVC-ColonDB', 'CVC-Dedup']
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15))
    for ax, metric, title in zip(axes, ('dice', 'iou'), ('Dice', 'IoU')):
        for model, offset, marker in [('resunet', -0.09, 'o'), ('resunet_spc', 0.09, 's')]:
            sub = agg[agg.model == model].set_index('dataset').reindex(datasets)
            mean = sub[f'{metric}_mean'].to_numpy(float)
            sd = sub[f'{metric}_sd_seeds'].fillna(0).to_numpy(float)
            x = np.arange(3) + offset
            ax.errorbar(x, mean, yerr=sd, color=COLORS[model], marker=marker,
                        markerfacecolor='white', capsize=3, linewidth=1.5,
                        label=LABELS[model])
            for xi, yi in zip(x, mean):
                horizontal = -8 if model == 'resunet' else 8
                ax.annotate(f'{yi:.3f}', (xi, yi), xytext=(horizontal, 7),
                            textcoords='offset points', ha='center', va='bottom',
                            color=COLORS[model], fontsize=6.8)
        ax.set_xticks(range(3), names)
        ax.set_ylabel(title)
        values = agg[agg.model.isin(['resunet', 'resunet_spc'])][f'{metric}_mean']
        ax.set_ylim(max(0, values.min() - .08), min(1, values.max() + .08))
        polish(ax)
    axes[0].legend(frameon=False, loc='lower left')
    fig.tight_layout(w_pad=2.2)
    save(fig, output, 'Fig_1_main_performance')


def plot_seed_pairs(frame, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
    seed_colors = {42: '#4E79A7', 43: '#59A14F', 44: '#F28E2B'}
    for ax, dataset, title in zip(axes, ('cvc', 'cvc_dedup'),
                                  ('CVC-ColonDB', 'CVC-Dedup')):
        sub = frame[frame.dataset == dataset]
        for seed in (42, 43, 44):
            a = sub[(sub.model == 'resunet') & (sub.seed == seed)].dice
            b = sub[(sub.model == 'resunet_spc') & (sub.seed == seed)].dice
            if a.empty or b.empty:
                continue
            ax.plot([0, 1], [a.iloc[0], b.iloc[0]], marker='o',
                    color=seed_colors[seed], label=f'Seed {seed}')
        ax.set_xticks([0, 1], ['Baseline', 'DSP+SPC'])
        ax.set_ylabel('Dice')
        ax.set_title(title)
        polish(ax)
    axes[1].legend(frameon=False, loc='best')
    fig.tight_layout(w_pad=2.5)
    save(fig, output, 'Fig_2_seed_consistency')


def plot_ablation(frame, output):
    models = ['resunet', 'resunet_dsp', 'resunet_spc_region', 'resunet_spc']
    sub = frame[(frame.dataset == 'cvc') & (frame.seed == 42)].set_index('model').reindex(models)
    if sub.dice.isna().any():
        return
    fig, ax = plt.subplots(figsize=(7.16, 3.25))
    bars = ax.bar(range(4), sub.dice, color=[COLORS[m] for m in models], width=.65)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=7.5)
    ax.set_xticks(range(4), [LABELS[m] for m in models])
    ax.set_ylabel('CVC-ColonDB Dice (seed 42)')
    ax.set_ylim(max(0, sub.dice.min() - .08), min(1, sub.dice.max() + .08))
    polish(ax)
    fig.tight_layout()
    save(fig, output, 'Fig_3_ablation')


def write_report(frame, agg, stats, output):
    def markdown_table(table):
        rendered = table.copy()
        for column in rendered.select_dtypes(include=[np.number]).columns:
            rendered[column] = rendered[column].map(
                lambda value: '' if pd.isna(value) else f'{value:.4f}'
            )
        header = '| ' + ' | '.join(rendered.columns) + ' |'
        divider = '| ' + ' | '.join(['---'] * len(rendered.columns)) + ' |'
        body = ['| ' + ' | '.join(map(str, row)) + ' |'
                for row in rendered.itertuples(index=False, name=None)]
        return '\n'.join([header, divider, *body])

    def metric(model, dataset, name='dice_mean'):
        row = agg[(agg.model == model) & (agg.dataset == dataset)]
        return float(row[name].iloc[0]) if not row.empty else np.nan
    b = metric('resunet', 'cvc')
    o = metric('resunet_spc', 'cvc')
    lines = [
        '# Optimized BSPC experiment report', '',
        '## Protocol', '',
        '- Source dataset: Kvasir-SEG, fixed 80/10/10 split.',
        '- External test: CVC-ColonDB; all inference settings locked on Kvasir validation.',
        '- Image size: 256 × 256; ImageNet-pretrained ResNet-18 encoder.',
        '- Main results: three seeds (42, 43, 44), reported as mean ± SD across seeds.',
        '- Statistical test: one-sided paired Wilcoxon on per-image Dice after averaging each image across seeds; 95% bootstrap CI for mean paired change.',
        '', '## Main finding', '',
        f'- CVC Dice: baseline {b:.4f}; DSP+SPC {o:.4f}; absolute change {o-b:+.4f}.',
        '', '## Aggregate results', '',
        markdown_table(agg), '',
        '## Paired statistics', '',
        markdown_table(stats) if not stats.empty else 'Not available.',
        '', '## Interpretation', '',
        '- A method claim should be retained only if the three-seed external Dice improves and the paired confidence interval is compatible with a positive effect.',
        '- CVC-ColonDB is the only available external dataset in the supplied files; this remains a limitation for a Q2 medical-imaging submission.',
        '- The seed-42 ablation and standard-model comparisons are supporting evidence, not substitutes for three-seed main results.',
    ]
    (output / 'BSPC_RESULTS_FOR_MANUSCRIPT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    results_root, output = Path(args.results_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    style()
    frame = collect_summaries(results_root)
    frame.to_csv(output / 'all_run_summaries.csv', index=False)
    agg = aggregate(frame)
    agg.to_csv(output / 'aggregate_mean_sd.csv', index=False)
    stats = paired_statistics(results_root, output)
    plot_main(agg, output)
    plot_seed_pairs(frame, output)
    plot_ablation(frame, output)
    write_report(frame, agg, stats, output)
    print(agg.to_string(index=False))


if __name__ == '__main__':
    main()
