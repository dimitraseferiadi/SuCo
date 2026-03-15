"""
gen_figures.py
==============
Generate all benchmark figures for the SuCo FAISS implementation,
mirroring the paper's Fig 6, 7, 8, 11, 12 structure.

Usage
-----
    python gen_figures.py \
        --log1  bench_all_20260313_142528.log \
        --log2  bench_all_20260314_164625.log \
        --fig6  fig6_sift10m.tsv \
        --fig7  fig7_sift10m.tsv \
        --fig8  fig8_sift10m.tsv \
        --out   ./figures/

All arguments default to files in the same directory as this script.
Requires: matplotlib, numpy  (pip install matplotlib numpy)
"""

import argparse
import csv
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── colour / style palette ──────────────────────────────────────────────────
COLORS = {
    'SuCo':    '#e6194b',
    'HNSW':    '#3cb44b',
    'IVFFlat': '#4363d8',
    'IVFPQ':   '#f58231',
    'OPQ':     '#911eb4',
    'LSH':     '#42d4f4',
    'DynAct':  '#e6194b',
    'MultiSeq':'#4363d8',
}
MARKERS = {
    'SuCo': 'o', 'HNSW': 's', 'IVFFlat': '^', 'IVFPQ': 'D', 'OPQ': 'v',
    'LSH': 'x',
    'DynAct': 'o', 'MultiSeq': 's',
}
LSTYLE = {
    'SuCo':    '-',
    'HNSW':    '--',
    'IVFFlat': '-.',
    'IVFPQ':   ':',
    'OPQ':     (0, (3, 1, 1, 1)),
    'LSH':     (0, (1, 1)),       # densely dotted
}
METHODS_ORDER = ['SuCo', 'HNSW', 'IVFFlat', 'IVFPQ', 'OPQ', 'LSH']

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})


# ═══════════════════════════════════════════════════════════════════════════
# LOG PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_log_full(filepath):
    """Parse a SuCo benchmark log.  Returns dict[dataset] → dict of rows."""
    results = {}
    current_dataset = None
    current_method = None
    in_table = False

    with open(filepath) as f:
        raw = f.read()

    # Strip carriage-return-only progress lines (terminal overwrite sequences)
    raw = re.sub(r'\r[^\n]', '', raw)
    lines = raw.splitlines()

    HARD_STOP_PREFIXES = ('>>> python', '## ', 'All benchmarks')
    SKIP_SUBSTRINGS = (
        '======', '------', 'efSearch', 'nprobe', 'Ns   hd',
        'Building ', 'Training ', 'Adding ', 'Loading ',
        'Build done', 'IndexSuCo::', 'IndexLSH::', '(done in',
        'ms/query', 'ms / query', 'nsubspaces', 'ntotal', '%)',
        'nbits   size',
    )

    for line in lines:
        # Dataset header
        m = re.match(
            r'##\s+(SIFT1M|GIST1M|Deep1M|Deep10M|SIFT10M|SpaceV10M)\s+', line)
        if m:
            current_dataset = m.group(1)
            results.setdefault(current_dataset, {})

        if not current_dataset:
            continue

        # Build metadata (only outside tables, take first non-zero value)
        if not in_table:
            m2 = re.search(r'build time\s*=\s*([\d.]+)s', line)
            if m2 and float(m2.group(1)) > 0:
                results[current_dataset].setdefault('SuCo_build', float(m2.group(1)))
            m3 = re.search(r'index size\s*=\s*([\d.]+)\s*(MiB|GiB)', line)
            if m3:
                v = float(m3.group(1)) * (1024 if m3.group(2) == 'GiB' else 1)
                results[current_dataset]['SuCo_size_mb'] = v

        # Sweep-table header detection
        if 'Ns   hd    nc   alpha' in line:
            current_method = 'SuCo'; in_table = True
            results[current_dataset].setdefault('SuCo', []); continue
        if 'HNSW efSearch sweep' in line:
            current_method = 'HNSW'; in_table = True
            results[current_dataset].setdefault('HNSW', []); continue
        if re.search(r'IVFFlat nprobe sweep', line) and 'OPQ' not in line:
            current_method = 'IVFFlat'; in_table = True
            results[current_dataset].setdefault('IVFFlat', []); continue
        if re.search(r'\bIVFPQ nprobe sweep', line) and 'OPQ' not in line:
            current_method = 'IVFPQ'; in_table = True
            results[current_dataset].setdefault('IVFPQ', []); continue
        if 'OPQ+IVFPQ nprobe sweep' in line:
            current_method = 'OPQ'; in_table = True
            results[current_dataset].setdefault('OPQ', []); continue
        if 'IndexLSH nbits sweep' in line:
            current_method = 'LSH'; in_table = True
            results[current_dataset].setdefault('LSH', []); continue

        # Table rows / noise
        if in_table and current_method:
            s = line.strip()
            if any(line.lstrip().startswith(tok) for tok in HARD_STOP_PREFIXES):
                in_table = False; current_method = None; continue
            if s == '' or any(tok in s for tok in SKIP_SUBSTRINGS):
                continue
            parts = s.split()
            if parts and re.match(r'^\d+$', parts[0]) and len(parts) >= 5:
                try:
                    if current_method == 'SuCo':
                        results[current_dataset]['SuCo'].append({
                            'Ns': int(parts[0]), 'hd': int(parts[1]),
                            'nc': int(parts[2]), 'alpha': float(parts[3]),
                            'beta': float(parts[4]), 'ms_q': float(parts[5]),
                            'qps': float(parts[6]), 'r1': float(parts[7]),
                            'r10': float(parts[8]),
                        })
                    elif current_method == 'HNSW':
                        results[current_dataset]['HNSW'].append({
                            'ef': int(parts[0]), 'ms_q': float(parts[1]),
                            'qps': float(parts[2]), 'r1': float(parts[3]),
                            'r10': float(parts[4]),
                        })
                    elif current_method == 'LSH':
                        # LSH row: nbits  size_mb  build_s  ms_q  qps  r1  r10
                        results[current_dataset]['LSH'].append({
                            'nbits':    int(parts[0]),
                            'size_mb':  float(parts[1]),
                            'build_s':  float(parts[2]),
                            'ms_q':     float(parts[3]),
                            'qps':      float(parts[4]),
                            'r1':       float(parts[5]),
                            'r10':      float(parts[6]),
                        })
                    else:
                        results[current_dataset][current_method].append({
                            'nprobe': int(parts[0]), 'ms_q': float(parts[1]),
                            'qps': float(parts[2]), 'r1': float(parts[3]),
                            'r10': float(parts[4]),
                        })
                except Exception:
                    pass
    return results


def load_tsv(path):
    with open(path, newline='') as f:
        return [{k: v.strip() for k, v in row.items()}
                for row in csv.DictReader(f, delimiter='\t')]


def merge_parsed_results(dst, src):
    for dataset, src_info in src.items():
        dst_info = dst.setdefault(dataset, {})
        for key, value in src_info.items():
            if isinstance(value, list):
                dst_info.setdefault(key, []).extend(value)
            elif key not in dst_info or dst_info[key] in (None, 0, 0.0):
                dst_info[key] = value
    return dst


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved {name}')


def _dual_axis(ax, x, y_left, y_right, xlabel, ylabel_l, ylabel_r, title,
               ylim_r=None):
    ax2 = ax.twinx()
    l1, = ax.plot(x, y_left,  '-o',  color='#e6194b', ms=5, lw=1.6,
                  label=ylabel_l)
    l2, = ax2.plot(x, y_right, '--^', color='#4363d8', ms=5, lw=1.6,
                   label=ylabel_r)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel_l,  color='#e6194b')
    ax2.set_ylabel(ylabel_r, color='#4363d8')
    if ylim_r:
        ax2.set_ylim(*ylim_r)
    ax.set_title(title)
    ax.legend(handles=[l1, l2], loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Dynamic Activation vs Multi-sequence  (paper Fig 6)
# ═══════════════════════════════════════════════════════════════════════════

def fig6_dynamic_activation(fig6_rows, out):
    da = sorted([(float(r['alpha']), float(r['QPS']))
                 for r in fig6_rows if r['algorithm'] == 'DynamicActivation'])
    ms = sorted([(float(r['alpha']), float(r['QPS']))
                 for r in fig6_rows if r['algorithm'] == 'MultiSequence'])
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot([p[0] for p in da], [p[1] for p in da],
            '-o', color=COLORS['DynAct'],  ms=5, lw=1.8,
            label='Dynamic Activation')
    ax.plot([p[0] for p in ms], [p[1] for p in ms],
            '--s', color=COLORS['MultiSeq'], ms=5, lw=1.8,
            label='Multi-sequence')
    ax.set_xlabel('Collision ratio α')
    ax.set_ylabel('Queries per second (QPS)')
    ax.set_title('Fig 6 — Dynamic Activation vs Multi-sequence  (SIFT10M)')
    ax.legend()
    fig.tight_layout()
    _save(fig, out, 'fig6_dynamic_activation')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Parameter sweep: Ns and nc  (paper Fig 7)
# ═══════════════════════════════════════════════════════════════════════════

def fig7_param_sweep(fig7_rows, out):
    def extract(var):
        rows = [r for r in fig7_rows if r['sweep_var'] == var]
        return dict(
            x   = [int(r['x_val'])      for r in rows],
            qps = [float(r['QPS'])      for r in rows],
            r10 = [float(r['R@10'])     for r in rows],
            bt  = [float(r['build_s'])  for r in rows],
            mb  = [float(r['index_mb']) for r in rows],
        )

    ns = extract('Ns');  nc = extract('nc')
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5))
    _dual_axis(axes[0,0], ns['x'], ns['qps'], ns['r10'],
               'Number of subspaces Ns', 'QPS', 'Recall@10',
               '(a) Query perf vs Ns  (SIFT10M)', ylim_r=(0.90, 1.01))
    _dual_axis(axes[0,1], ns['x'], ns['bt'],  [m/1024 for m in ns['mb']],
               'Number of subspaces Ns', 'Build time (s)', 'Index size (GB)',
               '(b) Indexing perf vs Ns  (SIFT10M)')
    _dual_axis(axes[1,0], nc['x'], nc['qps'], nc['r10'],
               'Centroids per half-subspace nc', 'QPS', 'Recall@10',
               '(c) Query perf vs nc  (SIFT10M)', ylim_r=(0.90, 1.01))
    _dual_axis(axes[1,1], nc['x'], nc['bt'],  [m/1024 for m in nc['mb']],
               'Centroids per half-subspace nc', 'Build time (s)', 'Index size (GB)',
               '(d) Indexing perf vs nc  (SIFT10M)')
    fig.suptitle('Fig 7 — SuCo parameter study: Ns and nc  (SIFT10M)',
                 fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, out, 'fig7_param_Ns_nc')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8 – Parameter sweep: α and β  (paper Fig 8)
# ═══════════════════════════════════════════════════════════════════════════

def fig8_param_alphabeta(fig8_rows, out):
    def extract(var):
        rows = [r for r in fig8_rows if r['sweep_var'] == var]
        return dict(
            x   = [float(r['x_val']) for r in rows],
            qps = [float(r['QPS'])   for r in rows],
            r10 = [float(r['R@10'])  for r in rows],
        )
    alpha = extract('alpha');  beta = extract('beta')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, d, xlabel, title in [
        (axes[0], alpha, 'Collision ratio α',
         '(a) Varying α  (β=0.005, SIFT10M)'),
        (axes[1], beta,  'Re-rank ratio β',
         '(b) Varying β  (α=0.05, SIFT10M)'),
    ]:
        _dual_axis(ax, d['x'], d['qps'], d['r10'],
                   xlabel, 'QPS', 'Recall@10', title, ylim_r=(0.90, 1.01))
    fig.suptitle('Fig 8 — Query performance vs α and β  (SIFT10M)',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    _save(fig, out, 'fig8_param_alpha_beta')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES 11 & 12 – Recall-QPS  (paper Fig 11 / 12)
# ═══════════════════════════════════════════════════════════════════════════

def _recall_qps_panel(ax, data, ds_name):
    ds = data.get(ds_name, {})
    for m in METHODS_ORDER:
        rows = ds.get(m, [])
        if not rows:
            continue
        pts = sorted([(r['r10'], r['qps']) for r in rows])
        ax.semilogy([p[0] for p in pts], [p[1] for p in pts],
                    linestyle=LSTYLE.get(m, '-'), marker=MARKERS.get(m, 'o'),
                    color=COLORS.get(m, 'grey'), ms=4, lw=1.6, label=m)
    ax.set_xlabel('Recall@10')
    ax.set_ylabel('Queries per second')
    ax.set_xlim(0.10, 1.02)
    ax.legend(fontsize=6.5)
    ax.grid(True, which='both', alpha=0.3)


def fig11_recall_qps_1M(data, out):
    datasets = [('SIFT1M', 'SIFT1M  (easy, d=128)'),
                ('GIST1M', 'GIST1M  (hard, d=960)'),
                ('Deep1M', 'Deep1M  (easy, d=96)')]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, (ds, label) in zip(axes, datasets):
        _recall_qps_panel(ax, data, ds); ax.set_title(label)
    fig.suptitle('Fig 11 — Recall@10 vs QPS: 1M-scale datasets',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    _save(fig, out, 'fig11_recall_qps_1M')


def fig12_recall_qps_10M(data, out):
    datasets = [('SIFT10M',   'SIFT10M  (easy, d=128)'),
                ('Deep10M',   'Deep10M  (easy, d=96)'),
                ('SpaceV10M', 'SpaceV10M  (hard, d=100)')]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, (ds, label) in zip(axes, datasets):
        _recall_qps_panel(ax, data, ds); ax.set_title(label)
    fig.suptitle('Fig 12 — Recall@10 vs QPS: 10M-scale datasets',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    _save(fig, out, 'fig12_recall_qps_10M')


# ═══════════════════════════════════════════════════════════════════════════
# BONUS – SuCo across all datasets
# ═══════════════════════════════════════════════════════════════════════════

def fig_suco_all_datasets(data, out):
    ds_list = ['SIFT1M','GIST1M','Deep1M','SIFT10M','Deep10M','SpaceV10M']
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(ds_list)))
    fig, ax = plt.subplots(figsize=(6, 4))
    has_points = False
    for ds, col in zip(ds_list, palette):
        rows = data.get(ds, {}).get('SuCo', [])
        if not rows: continue
        pts = sorted([(r['r10'], r['qps']) for r in rows])
        ax.semilogy([p[0] for p in pts], [p[1] for p in pts],
                    '-o', color=col, ms=5, lw=1.6, label=ds)
        has_points = True
    ax.set_xlabel('Recall@10'); ax.set_ylabel('QPS (log scale)')
    ax.set_title('SuCo — Recall@10 vs QPS across all datasets')
    if has_points:
        ax.legend(fontsize=7.5)
    else:
        ax.text(0.5, 0.5, 'No SuCo rows found in parsed logs',
                transform=ax.transAxes, ha='center', va='center', fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(0.85, 1.01); fig.tight_layout()
    _save(fig, out, 'fig_suco_all_datasets')


# ═══════════════════════════════════════════════════════════════════════════
# BONUS – Indexing build time & memory
# ═══════════════════════════════════════════════════════════════════════════

def fig_indexing(data, out):
    ds_order = ['SIFT1M','GIST1M','Deep1M','SIFT10M','Deep10M','SpaceV10M']
    labels, build_t, size_gb = [], [], []
    for ds in ds_order:
        bt = data.get(ds, {}).get('SuCo_build')
        sz = data.get(ds, {}).get('SuCo_size_mb')
        if bt and sz:
            labels.append(ds); build_t.append(bt); size_gb.append(sz/1024)

    if not labels:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis('off')
        ax.text(0.5, 0.5,
                'No SuCo build/index-size metadata found in parsed logs',
                ha='center', va='center', fontsize=9)
        fig.tight_layout()
        _save(fig, out, 'fig_indexing')
        return

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    b1 = ax1.bar(x, build_t, color='#e6194b', alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha='right')
    ax1.set_ylabel('Build time (s)'); ax1.set_title('(a) Index build time  [SuCo]')
    ax1.bar_label(b1, fmt='%.2fs', fontsize=7, padding=2)
    if any(v > 0 for v in build_t):
        ax1.set_yscale('log')
    ax1.grid(True, axis='y', alpha=0.3)
    b2 = ax2.bar(x, size_gb, color='#3cb44b', alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha='right')
    ax2.set_ylabel('Index size (GB)'); ax2.set_title('(b) Index memory footprint  [SuCo]')
    ax2.bar_label(b2, fmt='%.1fGB', fontsize=7, padding=2)
    ax2.grid(True, axis='y', alpha=0.3)
    fig.suptitle('SuCo Indexing Performance', fontsize=10, fontweight='bold')
    fig.tight_layout()
    _save(fig, out, 'fig_indexing')


# ═══════════════════════════════════════════════════════════════════════════
# BONUS – Recall@10 heatmap
# ═══════════════════════════════════════════════════════════════════════════

def fig_recall_heatmap(data, out):
    methods = ['SuCo','HNSW','IVFFlat','IVFPQ','OPQ','LSH']
    ds_list = ['SIFT1M','GIST1M','Deep1M','SIFT10M','Deep10M','SpaceV10M']
    matrix  = np.full((len(ds_list), len(methods)), np.nan)
    for i, ds in enumerate(ds_list):
        for j, m in enumerate(methods):
            rows = data.get(ds, {}).get(m, [])
            if rows:
                matrix[i, j] = max(r['r10'] for r in rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, vmin=0.4, vmax=1.0, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(methods)));  ax.set_xticklabels(methods)
    ax.set_yticks(range(len(ds_list))); ax.set_yticklabels(ds_list)
    plt.colorbar(im, ax=ax, label='Best Recall@10')
    for i in range(len(ds_list)):
        for j in range(len(methods)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        fontsize=7.5, color='black' if v > 0.6 else 'white')
    ax.set_title('Best Recall@10 per dataset × method', fontweight='bold')
    fig.tight_layout()
    _save(fig, out, 'fig_recall_heatmap')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    here = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--log1', default=os.path.join(here, 'bench_all_20260313_142528.log'))
    p.add_argument('--log2', default=os.path.join(here, 'bench_all_20260314_164625.log'))
    p.add_argument('--log3', default=os.path.join(here, 'bench_all_20260315_142830.log'))
    p.add_argument('--log4', default=os.path.join(here, 'bench_all_20260315_155134.log'))
    p.add_argument('--fig6', default=os.path.join(here, 'fig6_sift10m.tsv'))
    p.add_argument('--fig7', default=os.path.join(here, 'fig7_sift10m.tsv'))
    p.add_argument('--fig8', default=os.path.join(here, 'fig8_sift10m.tsv'))
    p.add_argument('--out',  default=os.path.join(here, 'figures'))
    args = p.parse_args()

    print('Parsing log files …')
    data = parse_log_full(args.log1)
    for log_path in (args.log2, args.log3, args.log4):
        if log_path and os.path.isfile(log_path):
            merge_parsed_results(data, parse_log_full(log_path))
        elif log_path:
            print(f'  Warning: log file not found, skipping: {log_path}')

    for ds, info in sorted(data.items()):
        for m, rows in info.items():
            if isinstance(rows, list):
                print(f'  {ds}/{m}: {len(rows)} rows')

    print(f'\nLoading TSV data …')
    fig6_rows = load_tsv(args.fig6)
    fig7_rows = load_tsv(args.fig7)
    fig8_rows = load_tsv(args.fig8)

    print(f'\nGenerating figures → {args.out}')
    fig6_dynamic_activation(fig6_rows, args.out)
    fig7_param_sweep(fig7_rows, args.out)
    fig8_param_alphabeta(fig8_rows, args.out)
    fig11_recall_qps_1M(data, args.out)
    fig12_recall_qps_10M(data, args.out)
    fig_suco_all_datasets(data, args.out)
    fig_indexing(data, args.out)
    fig_recall_heatmap(data, args.out)
    print('\nAll done.')


if __name__ == '__main__':
    main()