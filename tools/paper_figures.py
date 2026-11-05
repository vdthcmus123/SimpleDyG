"""Paper-style tables and figures for the SimpleDyG reproduction (CPU only, no torch).

Mirrors what the paper reports where our runs allow it:
  Table 1  dataset statistics                -> table1_datasets
  Table 2  link prediction vs baselines      -> table2_performance + fig_table2_bars.png
  Table 4  temporal alignment ablation       -> table4_temporal + fig_table4_ablation.png
  Table 5  training time per epoch           -> table5_time
  Fig. A.1 temporal pattern of each dataset  -> fig_A1_temporal_pattern.png
  Fig. 4c  hidden dimension (baselines' validation search) -> fig_baseline_dims.png
Extra diagnostics: per-seed paired plot, validation curves, prediction-collapse plot.
Figure 3 (multi-step) and the layer/head sweeps of Figure 4 were not run and are not drawn.

Entry point: render_all(result_roots, author_root, out_dir) -> dict of DataFrames/paths.
"""
import collections
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ['UCI_13', 'ML_10M_13']
LABEL = {'UCI_13': 'UCI', 'ML_10M_13': 'ML-10M'}
RAW = {'UCI_13': ('uci', 12), 'ML_10M_13': ('movie', 12)}
DOMAIN = {'UCI_13': 'Social', 'ML_10M_13': 'Rating'}
PAPER_T1 = {'UCI_13': (1781, 16743), 'ML_10M_13': (15841, 48561)}
# Table 2 (NDCG@5 mean, std, Jaccard mean, std) for UCI and ML-10M.
PAPER_T2 = {
    'DySAT': {'UCI_13': (.010, .003, .010, .001), 'ML_10M_13': (.058, .073, .050, .068)},
    'EvolveGCN': {'UCI_13': (.064, .045, .032, .026), 'ML_10M_13': (.097, .071, .092, .067)},
    'DyRep': {'UCI_13': (.011, .018, .010, .005), 'ML_10M_13': (.064, .036, .038, .001)},
    'JODIE': {'UCI_13': (.022, .023, .012, .009), 'ML_10M_13': (.059, .016, .020, .004)},
    'TGAT': {'UCI_13': (.061, .007, .020, .002), 'ML_10M_13': (.066, .035, .021, .007)},
    'TGN': {'UCI_13': (.041, .017, .011, .003), 'ML_10M_13': (.071, .029, .023, .001)},
    'TREND': {'UCI_13': (.067, .010, .039, .020), 'ML_10M_13': (.079, .028, .024, .003)},
    'GraphMixer': {'UCI_13': (.104, .013, .042, .005), 'ML_10M_13': (.081, .033, .043, .022)},
    'SimpleDyG': {'UCI_13': (.104, .010, .092, .014), 'ML_10M_13': (.138, .009, .131, .008)},
}
PAPER_T4 = {
    'full': PAPER_T2['SimpleDyG'],
    'same_time': {'UCI_13': (.090, .013, .083, .012), 'ML_10M_13': (.147, .001, .139, .001)},
    'no_time': {'UCI_13': (.111, .015, .091, .014), 'ML_10M_13': (.117, .062, .111, .059)},
}
PAPER_T5 = {'DySAT': 12.24, 'EvolveGCN': 11.89, 'DyRep': 6.4, 'JODIE': 6.25, 'TGAT': 18.54,
            'TGN': 8.05, 'TREND': 7.45, 'GraphMixer': 6.89, 'SimpleDyG': 6.21}
VARIANT_ORDER = ['full', 'same_time', 'no_time', 'per_ego_permute']
VARIANT_LABEL = {'full': 'SimpleDyG (full)', 'same_time': 'same time', 'no_time': 'no time',
                 'per_ego_permute': 'per-ego permuted time'}
# Validated categorical slots (blue, orange, aqua, violet); color follows the variant.
COLOR = {'full': '#2a78d6', 'same_time': '#eb6834', 'no_time': '#1baf7a', 'per_ego_permute': '#6250d6',
         'TGAT': '#eb6834', 'GraphMixer': '#1baf7a', 'SimpleDyG': '#2a78d6'}
INK, INK2, GRID, NEUTRAL = '#0b0b0b', '#52514e', '#e6e5e1', '#8f8d88'
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145}


# ----------------------------------------------------------------- loading
def load_results(result_roots):
    frames, runs = [], {}
    for root in map(Path, result_roots):
        table = root / 'results_per_seed.csv'
        if table.exists() and table.stat().st_size > 1:
            frames.append(pd.read_csv(table))
        if (root / 'runs').is_dir():
            for d in (root / 'runs').iterdir():
                if d.is_dir():
                    runs.setdefault(d.name, d)
    if not frames:
        raise FileNotFoundError('No results_per_seed.csv in ' + str(list(result_roots)))
    df = pd.concat(frames, ignore_index=True).drop_duplicates('run_id', keep='last')
    df['seed'] = df.seed.astype(int)
    return df, runs


def load_audit(result_roots):
    audit = {}
    for root in map(Path, result_roots):
        p = root / 'data_audit.json'
        if p.exists():
            for k, v in json.loads(p.read_text(encoding='utf-8')).items():
                audit.setdefault(k, v)
    return audit


def _fmt(m, s=None, digits=3):
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return '–'
    text = f'{m:.{digits}f}'.replace('0.', '.', 1) if abs(m) < 1 else f'{m:.{digits}f}'
    if s is not None and not (isinstance(s, float) and math.isnan(s)):
        text += ' ± ' + (f'{s:.{digits}f}'.replace('0.', '.', 1))
    return text


def _style_axes(ax):
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
                         'axes.labelcolor': INK2, 'figure.facecolor': 'white'})
    return plt


# ----------------------------------------------------------------- Table 1 / Fig A.1
def table1_datasets(df, audit, author_root=None):
    rows = []
    for ds in DATASETS:
        if ds not in set(df.dataset) and ds not in audit:
            continue
        a = audit.get(ds, {})
        sp = a.get('splits', {})
        raw_steps = '–'
        if author_root:
            raw, t = RAW[ds]
            p = Path(author_root) / 'all_data' / raw / str(t) / f'ml_{raw}.csv'
            if p.exists():
                raw_steps = int(pd.read_csv(p, usecols=['timestamp']).timestamp.nunique())
        rows.append({'Dataset': LABEL[ds], 'Domain': DOMAIN[ds],
                     '# Nodes': a.get('nodes'), '# Edges': a.get('edges'),
                     'Paper # Nodes / # Edges': '{:,} / {:,}'.format(*PAPER_T1[ds]),
                     '# Time steps': raw_steps,
                     'Train / Val / Test queries': '{} / {} / {}'.format(
                         sp.get('train', {}).get('queries', '–'), sp.get('val', {}).get('queries', '–'),
                         sp.get('test', {}).get('queries', '–')),
                     'Train sequences > 512 tokens': sp.get('train', {}).get('truncated', '–'),
                     'Test inputs > 1024 tokens': sp.get('test', {}).get('above_context', '–')})
    return pd.DataFrame(rows)


def fig_temporal_pattern(author_root, out):
    plt = _plt()
    present = []
    for ds in DATASETS:
        raw, t = RAW[ds]
        p = Path(author_root) / 'all_data' / raw / str(t) / f'ml_{raw}.csv'
        if p.exists():
            present.append((ds, pd.read_csv(p, usecols=['timestamp']).timestamp.value_counts().sort_index()))
    if not present:
        return None
    fig, axes = plt.subplots(1, len(present), figsize=(6.2 * len(present), 3.6), squeeze=False)
    for ax, (ds, counts) in zip(axes[0], present):
        last = counts.index.max()
        colors = ['#2a78d6' if i < last - 1 else ('#eda100' if i == last - 1 else '#e34948') for i in counts.index]
        ax.bar(counts.index, counts.values, color=colors, width=.8, edgecolor='white', linewidth=1)
        ax.set_title(f'{LABEL[ds]}: interactions per time step')
        ax.set_xlabel('time step')
        ax.set_ylabel('# interactions')
        ax.set_xticks(counts.index)
        _style_axes(ax)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='#2a78d6', label='train'), Patch(color='#eda100', label='validation (T−1)'),
                           Patch(color='#e34948', label='test (T)')], frameon=False, fontsize=8)
    fig.suptitle('Figure A.1 (reproduced): temporal pattern of each dataset', color=INK, fontsize=12)
    fig.tight_layout()
    path = Path(out) / 'fig_A1_temporal_pattern.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ----------------------------------------------------------------- trivial baselines
def _read(path):
    return [x.split() for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def _targets(line):
    i = line.index('<|pre|>')
    j = line.index('<|endofpre|>') if '<|endofpre|>' in line else len(line)
    return [x for x in line[i + 1:j] if not x.startswith('<|')]


def _ndcg(pred, truth):
    p, y = list(dict.fromkeys(pred)), list(dict.fromkeys(truth))
    den = sum(1 / math.log2(i + 2) for i in range(min(5, len(y))))
    return sum(1 / math.log2(i + 2) for i, x in enumerate(p[:5]) if x in y) / den


def _jac(pred, truth):
    a, b = set(pred), set(truth)
    return len(a & b) / len(a | b)


def trivial_scores(author_root):
    rows = []
    for ds in DATASETS:
        folder = Path(author_root) / 'resources' / ds / '12'
        if not (folder / 'test.link_prediction').exists():
            continue
        pop = collections.Counter(x for line in _read(folder / 'train.link_prediction') for x in _targets(line))
        ranked = [x for x, _ in pop.most_common(10)]
        acc = collections.defaultdict(list)
        for inp, gt in zip(_read(folder / 'test.link_prediction'), _read(folder / 'test_gt.link_prediction')):
            ego = inp[2]
            truth = [x for x in _targets(gt) if x != ego]
            if not truth:
                continue
            end = inp.index('<|endofhistory|>')
            slots, cur = [], None
            for x in inp[3:end]:
                if x.startswith('<|time'):
                    if cur:
                        slots.append(cur)
                    cur = []
                elif not x.startswith('<|'):
                    cur = (cur or []) + [x]
            if cur:
                slots.append(cur)
            for name, pred in [('Popular top-1 (train)', [x for x in ranked if x != ego][:1]),
                               ('Repeat last active slot', slots[-1] if slots else [])]:
                acc[name].append((_ndcg(pred, truth), _jac(pred, truth)))
        for name, v in acc.items():
            a = np.array(v)
            rows.append(dict(dataset=ds, method=name, ndcg5=a[:, 0].mean(), jaccard=a[:, 1].mean()))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- Table 2
def _agg(g, col):
    return g[col].mean(), g[col].std(ddof=1)


def table2_performance(df, trivial, ndcg_col='ndcg5'):
    """Rows: paper methods + trivial; columns per dataset: ours / paper for NDCG@5 and Jaccard.
    Baseline Jaccard uses max over k=1,5,10,20 on test (paper protocol, oracle); SimpleDyG uses the generated set."""
    full = df[df.variant == 'full']
    rows, numeric = [], []
    methods = list(PAPER_T2) + (list(trivial.method.unique()) if trivial is not None and len(trivial) else [])
    for method in methods:
        row, num = {'Method': method}, {'Method': method}
        for ds in DATASETS:
            if ds not in set(df.dataset):
                continue
            g = full[(full.dataset == ds) & (full.model == method)]
            jcol = 'jaccard' if method == 'SimpleDyG' else 'jaccard_oracle_max_k'
            if len(g):
                (nm, ns), (jm, js) = _agg(g, ndcg_col), _agg(g, jcol)
                n = len(g)
            elif trivial is not None and method in set(trivial.method):
                t = trivial[(trivial.dataset == ds) & (trivial.method == method)]
                nm, ns, jm, js, n = (float(t.ndcg5.iloc[0]), None, float(t.jaccard.iloc[0]), None, 0) if len(t) else (None,) * 5
            else:
                nm = ns = jm = js = n = None
            p = PAPER_T2.get(method, {}).get(ds)
            row[f'{LABEL[ds]} NDCG@5 (ours)'] = _fmt(nm, ns)
            row[f'{LABEL[ds]} NDCG@5 (paper)'] = _fmt(*p[:2]) if p else '–'
            row[f'{LABEL[ds]} Jaccard (ours)'] = _fmt(jm, js)
            row[f'{LABEL[ds]} Jaccard (paper)'] = _fmt(*p[2:]) if p else '–'
            num[f'{LABEL[ds]} NDCG@5 (ours)'], num[f'{LABEL[ds]} Jaccard (ours)'] = nm, jm
            num[f'{LABEL[ds]} NDCG@5 (paper)'] = p[0] if p else None
            num[f'{LABEL[ds]} Jaccard (paper)'] = p[2] if p else None
        rows.append(row)
        numeric.append(num)
    return pd.DataFrame(rows), pd.DataFrame(numeric)


def html_table(text_df, numeric_df=None, caption='', note=''):
    """HTML table; in every numeric column the best value is bold and the runner-up underlined (paper style)."""
    marks = {}
    if numeric_df is not None:
        for col in numeric_df.columns:
            if col == 'Method':
                continue
            values = pd.to_numeric(numeric_df[col], errors='coerce')
            order = values.dropna().sort_values(ascending=False)
            if len(order) >= 1:
                marks[(order.index[0], col)] = 'b'
            if len(order) >= 2:
                marks[(order.index[1], col)] = 'u'
    head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in text_df.columns)
    body = []
    for i, r in text_df.iterrows():
        cells = []
        for c in text_df.columns:
            v = html.escape(str(r[c]))
            m = marks.get((i, c))
            v = f'<b>{v}</b>' if m == 'b' else (f'<u>{v}</u>' if m == 'u' else v)
            cells.append(f'<td>{v}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    style = ('<style>.pt{border-collapse:collapse;font:12px/1.4 system-ui,sans-serif;margin:6px 0 2px}'
             '.pt th,.pt td{padding:4px 8px;border-bottom:1px solid #e6e5e1;text-align:right;white-space:nowrap}'
             '.pt th{border-bottom:2px solid #0b0b0b;color:#0b0b0b}.pt td:first-child,.pt th:first-child{text-align:left}'
             '.pt caption{caption-side:top;text-align:left;font-weight:600;padding-bottom:4px}'
             '.ptn{font:11px system-ui,sans-serif;color:#52514e;margin:0 0 14px}</style>')
    return (style + f'<table class="pt"><caption>{html.escape(caption)}</caption><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table><p class="ptn">{html.escape(note)}</p>')


def fig_table2_bars(df, trivial, out, ndcg_col='ndcg5'):
    plt = _plt()
    full = df[df.variant == 'full']
    present = [d for d in DATASETS if d in set(df.dataset)]
    fig, axes = plt.subplots(len(present), 2, figsize=(11, 3.4 * len(present)), squeeze=False)
    models = [m for m in ['TGAT', 'GraphMixer', 'SimpleDyG'] if m in set(full.model)]
    for r, ds in enumerate(present):
        for c, (metric, title) in enumerate([(ndcg_col, 'NDCG@5'), ('jaccard', 'Jaccard')]):
            ax = axes[r][c]
            means, stds, seeds = [], [], []
            for m in models:
                g = full[(full.dataset == ds) & (full.model == m)]
                col = metric if (metric != 'jaccard' or m == 'SimpleDyG') else 'jaccard_oracle_max_k'
                means.append(g[col].mean()); stds.append(g[col].std(ddof=1)); seeds.append(g[col].to_numpy())
            x = np.arange(len(models))
            ax.bar(x, means, width=.6, color=[COLOR[m] for m in models], edgecolor='white', linewidth=2,
                   yerr=stds, capsize=3, error_kw=dict(ecolor=INK2, lw=1))
            for i, s in enumerate(seeds):
                ax.scatter(np.full(len(s), x[i]) + np.linspace(-.18, .18, len(s)), s, s=10, color=INK, alpha=.35, zorder=3)
                top = max(means[i] + (0 if np.isnan(stds[i]) else stds[i]), s.max() if len(s) else means[i])
                ax.text(x[i], top, _fmt(means[i]), ha='center', va='bottom', fontsize=8, color=INK, fontweight='bold')
            paper = [PAPER_T2[m][ds][0 if c == 0 else 2] for m in models]
            ax.scatter(x + .33, paper, marker='D', s=36, color='white', edgecolor=INK, linewidth=1.4, zorder=4,
                       label='paper (Table 2)')
            if trivial is not None and len(trivial):
                for name, style in [('Repeat last active slot', '--'), ('Popular top-1 (train)', ':')]:
                    t = trivial[(trivial.dataset == ds) & (trivial.method == name)]
                    if len(t):
                        ax.axhline(float(t[('ndcg5' if c == 0 else 'jaccard')].iloc[0]), ls=style, color=NEUTRAL,
                                   lw=1.5, label=name + ' (no training)')
            ax.set_xticks(x, models)
            ax.set_title(f'{LABEL[ds]} – {title}')
            _style_axes(ax)
            if r == 0 and c == 1:
                ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('Table 2 (reproduced): mean ± std over seeds, dots = seeds, ◇ = paper', color=INK, fontsize=12)
    fig.tight_layout()
    path = Path(out) / 'fig_table2_bars.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ----------------------------------------------------------------- Table 4
def table4_temporal(df, ndcg_col='ndcg5'):
    s = df[df.model == 'SimpleDyG']
    rows = []
    for ds in [d for d in DATASETS if d in set(s.dataset)]:
        for protocol in sorted(set(s[s.dataset == ds].protocol)):
            g = s[(s.dataset == ds) & (s.protocol == protocol)]
            base = g[g.variant == 'full'].set_index('seed')
            for v in [x for x in VARIANT_ORDER if x in set(g.variant)]:
                gv = g[g.variant == v]
                row = {'Dataset': LABEL[ds], 'Protocol': protocol, 'Variant': VARIANT_LABEL[v], 'n seeds': gv.seed.nunique(),
                       'NDCG@5 (ours)': _fmt(*_agg(gv, ndcg_col)), 'Jaccard (ours)': _fmt(*_agg(gv, 'jaccard'))}
                p = PAPER_T4.get(v, {}).get(ds)
                row['NDCG@5 (paper)'] = _fmt(*p[:2]) if p else '–'
                row['Jaccard (paper)'] = _fmt(*p[2:]) if p else '–'
                if v != 'full' and len(base):
                    other = gv.set_index('seed')
                    common = base.index.intersection(other.index)
                    d = (other.loc[common, ndcg_col] - base.loc[common, ndcg_col]).to_numpy()
                    if len(d) >= 2:
                        h = T975.get(len(d) - 1, 1.96) * d.std(ddof=1) / math.sqrt(len(d))
                        row['ΔNDCG vs full [95% CI]'] = f'{d.mean():+.4f} [{d.mean() - h:+.4f}, {d.mean() + h:+.4f}]'
                        row['Significant?'] = 'yes' if (d.mean() - h > 0 or d.mean() + h < 0) else 'no'
                rows.append(row)
    return pd.DataFrame(rows).fillna('–')


def fig_table4_ablation(df, out, ndcg_col='ndcg5'):
    plt = _plt()
    s = df[df.model == 'SimpleDyG']
    groups = [(ds, p) for ds in DATASETS for p in sorted(set(s[s.dataset == ds].protocol))]
    fig, axes = plt.subplots(len(groups), 2, figsize=(11, 3.4 * len(groups)), squeeze=False)
    for r, (ds, protocol) in enumerate(groups):
        g = s[(s.dataset == ds) & (s.protocol == protocol)]
        variants = [v for v in VARIANT_ORDER if v in set(g.variant)]
        for c, (metric, title) in enumerate([(ndcg_col, 'NDCG@5'), ('jaccard', 'Jaccard')]):
            ax = axes[r][c]
            x = np.arange(len(variants))
            for i, v in enumerate(variants):
                vals = g[g.variant == v][metric].to_numpy()
                m, sd = vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0)
                ax.bar(i, m, width=.6, color=COLOR[v], edgecolor='white', linewidth=2, yerr=sd, capsize=3,
                       error_kw=dict(ecolor=INK2, lw=1))
                ax.scatter(np.full(len(vals), i) + np.linspace(-.18, .18, len(vals)), vals, s=10, color=INK, alpha=.35, zorder=3)
                ax.text(i, max(m + sd, vals.max()), _fmt(m), ha='center', va='bottom', fontsize=8, color=INK,
                        fontweight='bold')
                p = PAPER_T4.get(v, {}).get(ds)
                if p and protocol == 'author_compat':
                    ax.scatter(i + .33, p[0 if c == 0 else 2], marker='D', s=36, color='white', edgecolor=INK,
                               linewidth=1.4, zorder=4, label='paper (Table 4)' if i == 0 else None)
            ax.set_xticks(x, [VARIANT_LABEL[v] for v in variants], rotation=12)
            ax.set_title(f'{LABEL[ds]} ({protocol}) – {title}')
            _style_axes(ax)
            if c == 1 and protocol == 'author_compat':
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Table 4 (reproduced): temporal alignment designs, mean ± std, dots = seeds', color=INK, fontsize=12)
    fig.tight_layout()
    path = Path(out) / 'fig_table4_ablation.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_paired_seeds(df, out, ndcg_col='ndcg5'):
    plt = _plt()
    s = df[df.model == 'SimpleDyG']
    panels = []
    for ds in DATASETS:
        for protocol in sorted(set(s[s.dataset == ds].protocol)):
            g = s[(s.dataset == ds) & (s.protocol == protocol)]
            if 'full' in set(g.variant):
                panels += [(ds, protocol, v) for v in VARIANT_ORDER if v in set(g.variant) and v != 'full']
    if not panels:
        return None
    cols = min(3, len(panels))
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (ds, protocol, v) in zip(axes.flat, panels):
        g = s[(s.dataset == ds) & (s.protocol == protocol)]
        base = g[g.variant == 'full'].set_index('seed')[ndcg_col]
        o = g[g.variant == v].set_index('seed')[ndcg_col]
        common = base.index.intersection(o.index)
        up = int((o[common] > base[common] + 1e-12).sum()); down = int((o[common] < base[common] - 1e-12).sum())
        for seed in common:
            better = o[seed] > base[seed]
            ax.plot([0, 1], [base[seed], o[seed]], color=COLOR[v] if better else NEUTRAL, lw=1.3, alpha=.8)
        ax.scatter(np.zeros(len(common)), base[common], s=22, color=COLOR['full'], edgecolor='white', zorder=3)
        ax.scatter(np.ones(len(common)), o[common], s=22, color=COLOR[v], edgecolor='white', zorder=3)
        ax.set_xticks([0, 1], ['full', VARIANT_LABEL[v]])
        ax.set_xlim(-.3, 1.3)
        ax.set_ylabel('test NDCG@5')
        ax.set_title(f'{LABEL[ds]} ({protocol})\n{up} seeds up, {down} down (colored = up)', fontsize=10)
        _style_axes(ax)
    fig.suptitle('Same seed, full → variant', color=INK, fontsize=12)
    fig.tight_layout()
    path = Path(out) / 'fig_paired_seeds.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ----------------------------------------------------------------- curves / dims / time / collapse
def fig_training_curves(df, runs, out):
    plt = _plt()
    s = df[df.model == 'SimpleDyG']
    groups = [(ds, p) for ds in DATASETS for p in sorted(set(s[s.dataset == ds].protocol))]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.8 * len(groups), 3.8), squeeze=False)
    drawn = False
    for ax, (ds, protocol) in zip(axes[0], groups):
        g = s[(s.dataset == ds) & (s.protocol == protocol)]
        key = 'ndcg5_author' if protocol == 'author_compat' else 'ndcg5'
        for v in [x for x in VARIANT_ORDER if x in set(g.variant)]:
            curves = []
            for rid in g[g.variant == v].run_id:
                p = runs.get(rid, Path('_')) / 'history.json'
                if p.exists():
                    h = json.loads(p.read_text(encoding='utf-8'))
                    curves.append(pd.Series([e[key] for e in h], index=[e['epoch'] for e in h]))
            if not curves:
                continue
            for cu in curves:
                ax.plot(cu.index, cu.values, color=COLOR[v], lw=.6, alpha=.25)
            table = pd.concat(curves, axis=1)
            alive = table.notna().sum(axis=1)
            mean = table.mean(axis=1)[alive >= min(3, len(curves))]
            ax.plot(mean.index, mean.values, color=COLOR[v], lw=2, label=VARIANT_LABEL[v])
            drawn = True
        ax.set_xlabel('epoch')
        ax.set_ylabel(f'validation {key}')
        ax.set_title(f'{LABEL[ds]} ({protocol}): validation curves\n(thin = seeds, thick = mean while ≥3 seeds still training)', fontsize=10)
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = Path(out) / 'fig_validation_curves.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path if drawn else None


def fig_baseline_dims(df, runs, out):
    plt = _plt()
    rows = []
    for _, r in df[df.model.isin(['TGAT', 'GraphMixer'])].iterrows():
        p = runs.get(r.run_id, Path('_')) / 'validation_search.json'
        if p.exists():
            for e in json.loads(p.read_text(encoding='utf-8')):
                rows.append(dict(dataset=r.dataset, model=r.model, seed=r.seed, dim=e['dim'],
                                 ndcg5=e['validation']['ndcg5'], seconds=e.get('training_seconds'),
                                 epochs=e.get('epochs_run')))
    if not rows:
        return None, pd.DataFrame()
    d = pd.DataFrame(rows)
    present = [x for x in DATASETS if x in set(d.dataset)]
    fig, axes = plt.subplots(1, len(present), figsize=(5.8 * len(present), 3.6), squeeze=False)
    for ax, ds in zip(axes[0], present):
        for m in ['TGAT', 'GraphMixer']:
            g = d[(d.dataset == ds) & (d.model == m)].groupby('dim').ndcg5.agg(['mean', 'std'])
            if len(g):
                ax.errorbar(g.index, g['mean'], yerr=g['std'], color=COLOR[m], lw=2, marker='o', ms=6, capsize=3, label=m)
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(d.dim.unique()), [str(x) for x in sorted(d.dim.unique())])
        ax.set_xlabel('hidden dimension')
        ax.set_ylabel('validation NDCG@5')
        ax.set_title(f'{LABEL[ds]}: baselines vs hidden dimension')
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Figure 4(c)-style: hidden dimension (baselines’ validation search; SimpleDyG not swept)', color=INK, fontsize=11)
    fig.tight_layout()
    path = Path(out) / 'fig_baseline_dims.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, d


def table5_time(df, dims):
    rows = []
    for (ds, model, variant, protocol), g in df.groupby(['dataset', 'model', 'variant', 'protocol']):
        if model == 'SimpleDyG':
            per_epoch = (g.training_seconds / g.epochs_run).mean()
            epochs = g.epochs_run.mean()
        else:
            dd = dims[(dims.dataset == ds) & (dims.model == model)] if len(dims) else pd.DataFrame()
            if len(dd) and dd.seconds.notna().all():
                per_epoch, epochs = dd.seconds.sum() / dd.epochs.sum(), dd.epochs.sum() / g.seed.nunique()
            else:
                per_epoch, epochs = (g.training_seconds / g.epochs_run).mean(), g.epochs_run.mean()
        rows.append({'Dataset': LABEL.get(ds, ds), 'Method': model if model != 'SimpleDyG' else f'SimpleDyG ({variant})',
                     'Protocol': protocol, 'Seconds / epoch (ours, T4)': round(per_epoch, 1),
                     'Epochs / run': round(epochs, 1), 'Total train hours / run': round(g.training_seconds.mean() / 3600, 2),
                     'Test seconds / run': round(g.testing_seconds.mean(), 1),
                     'Parameters': f'{int(g.parameters.mean()):,}', 'Peak GPU MB': round(g.peak_gpu_mb.mean()),
                     'Paper s / epoch (UCI, L40)': PAPER_T5.get(model, '–') if ds == 'UCI_13' and variant == 'full' else '–'})
    note = ('Ours includes validation decoding after every epoch; baselines sum all hidden-dimension candidates. '
            'Paper Table 5 measures training only, on one NVIDIA L40.')
    return pd.DataFrame(rows), note


def collapse_table(df, runs, threshold=.7):
    rows = []
    for _, r in df[df.model == 'SimpleDyG'].iterrows():
        p = runs.get(r.run_id, Path('_')) / 'test_predictions.jsonl'
        if not p.exists():
            continue
        recs = [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
        recs = [x for x in recs if x.get('truth')]
        keys = collections.Counter(json.dumps(sorted(set(map(str, x['pred'])))) for x in recs)
        top, cnt = keys.most_common(1)[0]
        rows.append(dict(dataset=r.dataset, protocol=r.protocol, variant=r.variant, seed=r.seed,
                         top_share=cnt / len(recs), top_prediction=top, distinct=len(keys), ndcg5=r.ndcg5,
                         collapsed=cnt / len(recs) >= threshold))
    return pd.DataFrame(rows)


def fig_collapse(col, out, threshold=.7):
    if col.empty:
        return None
    plt = _plt()
    groups = [(ds, p) for ds in DATASETS for p in sorted(set(col[col.dataset == ds].protocol))]
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * sum(col[(col.dataset == d) & (col.protocol == p)].variant.nunique() for d, p in groups) + 2), 4))
    ticks = []
    pos = 0
    for ds, protocol in groups:
        g = col[(col.dataset == ds) & (col.protocol == protocol)]
        for v in [x for x in VARIANT_ORDER if x in set(g.variant)]:
            vals = g[g.variant == v].top_share.to_numpy()
            ax.scatter(np.full(len(vals), pos) + np.linspace(-.2, .2, len(vals)), vals, s=26, color=COLOR[v],
                       edgecolor='white', zorder=3)
            short = {'author_compat': 'compat', 'corrected': 'corrected'}.get(protocol, protocol)
            ticks.append((pos, f'{LABEL[ds]}\n{short}\n{v.replace("_", " ")}'))
            pos += 1
        pos += .6
    ax.axhline(threshold, color='#e34948', ls='--', lw=1.2, label=f'collapse threshold ({threshold:.0%})')
    ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('share of test queries with\nthe single most common prediction')
    ax.set_title('Prediction collapse check (each dot = one run)')
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc='center right')
    fig.tight_layout()
    path = Path(out) / 'fig_collapse.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ----------------------------------------------------------------- driver
def render_all(result_roots, author_root=None, out_dir='paper_figures', ndcg_col='ndcg5'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, runs = load_results(result_roots)
    audit = load_audit(result_roots)
    trivial = trivial_scores(author_root) if author_root else None
    res = {'figures': {}, 'tables': {}, 'html': []}
    metric_note = ('NDCG@5 = standard (deduplicated) metric' if ndcg_col == 'ndcg5'
                   else 'NDCG@5 = author-compatible metric (keeps duplicates)')

    t1 = table1_datasets(df, audit, author_root)
    res['tables']['table1_datasets'] = t1
    res['html'].append(html_table(t1, caption='Table 1 (reproduced): dataset statistics',
                                  note='Node/edge counts are taken from the data audit of the released preprocessed files.'))
    t2, t2n = table2_performance(df, trivial, ndcg_col)
    res['tables']['table2_performance'] = t2
    res['html'].append(html_table(t2, t2n.drop(columns=[c for c in t2n.columns if 'paper' in c]),
        caption='Table 2 (reproduced): dynamic link prediction, mean ± std over seeds (bold = best, underline = runner-up among our runs)',
        note=metric_note + '. Baseline Jaccard = max over k∈{1,5,10,20} on test as in the paper (oracle); '
             'SimpleDyG Jaccard uses its generated set. Six paper baselines were not run (–). '
             'Trivial rows use no training.'))
    t4 = table4_temporal(df, ndcg_col)
    res['tables']['table4_temporal'] = t4
    res['html'].append(html_table(t4, caption='Table 4 (reproduced): impact of temporal alignment designs',
        note=metric_note + '. Δ is paired by seed against full of the same dataset and protocol; '
             'CI is t-based. Paper values apply to the author_compat protocol only.'))
    dims_path, dims = fig_baseline_dims(df, runs, out)
    t5, t5note = table5_time(df, dims)
    res['tables']['table5_time'] = t5
    res['html'].append(html_table(t5, caption='Table 5 (reproduced): time efficiency', note=t5note))
    col = collapse_table(df, runs)
    if len(col):
        summary = (col.groupby(['dataset', 'protocol', 'variant'])
                   .agg(runs=('seed', 'count'), collapsed=('collapsed', 'sum'), top_share_min=('top_share', 'min'),
                        top_share_max=('top_share', 'max'), distinct_sets_max=('distinct', 'max')).reset_index())
        summary['dataset'] = summary.dataset.map(LABEL)
        for c in ['top_share_min', 'top_share_max']:
            summary[c] = summary[c].map(lambda v: f'{v:.1%}')
        res['tables']['collapse_summary'] = summary
        res['html'].append(html_table(summary, caption='Prediction collapse summary (SimpleDyG, test)',
            note='A run is collapsed when one prediction set is given to ≥70% of test queries.'))

    for name, fn in [('fig_A1_temporal_pattern', lambda: fig_temporal_pattern(author_root, out) if author_root else None),
                     ('fig_table2_bars', lambda: fig_table2_bars(df, trivial, out, ndcg_col)),
                     ('fig_table4_ablation', lambda: fig_table4_ablation(df, out, ndcg_col)),
                     ('fig_paired_seeds', lambda: fig_paired_seeds(df, out, ndcg_col)),
                     ('fig_validation_curves', lambda: fig_training_curves(df, runs, out)),
                     ('fig_baseline_dims', lambda: dims_path),
                     ('fig_collapse', lambda: fig_collapse(col, out))]:
        try:
            path = fn()
        except Exception as exc:  # one failing figure must not hide the others
            print('Figure skipped:', name, '-', exc)
            path = None
        if path:
            res['figures'][name] = Path(path)
    for name, table in res['tables'].items():
        table.to_csv(out / f'{name}.csv', index=False)
    page = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>SimpleDyG – paper-style tables</title></head>'
            '<body style="font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px">',
            '<h1 style="font-size:20px">SimpleDyG reproduction – paper-style tables and figures</h1>']
    page += res['html']
    for name, path in res['figures'].items():
        page.append(f'<h3 style="font-size:14px">{name}</h3><img src="{path.name}" style="max-width:100%">')
    (out / 'paper_tables.html').write_text('\n'.join(page) + '</body></html>', encoding='utf-8')
    res['df'], res['out'] = df, out
    return res


def show(res):
    """Display everything inline in Jupyter."""
    from IPython.display import HTML, Image, display
    for h in res['html']:
        display(HTML(h))
    for name, path in res['figures'].items():
        print(name)
        display(Image(filename=str(path)))
