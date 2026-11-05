"""Charts for the result slides, styled after MidTermSlideSimpleDyG (slate/sky palette, Inter)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

NAVY, INK2, MUTED, GRID, BG = '#0f172a', '#334155', '#64748b', '#e2e8f0', '#ffffff'
SKY, ORANGE, TEAL, VIOLET, SLATE = '#0ea5e9', '#f97316', '#0d9488', '#8b5cf6', '#94a3b8'
VCOL = {'full': SKY, 'same_time': ORANGE, 'no_time': TEAL, 'per_ego_permute': VIOLET}
VLAB = {'full': 'full', 'same_time': 'same time', 'no_time': 'no time', 'per_ego_permute': 'per-ego permute'}
MCOL = {'TGAT': ORANGE, 'GraphMixer': TEAL, 'SimpleDyG': SKY}

plt.rcParams.update({'font.family': 'Inter', 'font.size': 13, 'svg.fonttype': 'none',
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': MUTED,
                     'axes.titlesize': 15, 'axes.titleweight': 'bold', 'axes.titlecolor': NAVY,
                     'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG})

runs = pd.read_csv('data/all_runs.csv')
TRIV = {('UCI_13', 'repeat'): .2928, ('UCI_13', 'popular'): .0000, ('ML_10M_13', 'repeat'): .1193, ('ML_10M_13', 'popular'): .1475,
        ('UCI_13', 'own1'): .0955, ('UCI_13', 'own5'): .1460, ('ML_10M_13', 'own1'): .0904, ('ML_10M_13', 'own5'): .1324}
PAPER = {('UCI_13', 'TGAT'): .061, ('UCI_13', 'GraphMixer'): .104, ('UCI_13', 'SimpleDyG'): .104,
         ('ML_10M_13', 'TGAT'): .066, ('ML_10M_13', 'GraphMixer'): .081, ('ML_10M_13', 'SimpleDyG'): .138}
PAPER_T4 = {('UCI_13', 'full'): .104, ('UCI_13', 'same_time'): .090, ('UCI_13', 'no_time'): .111,
            ('ML_10M_13', 'full'): .138, ('ML_10M_13', 'same_time'): .147, ('ML_10M_13', 'no_time'): .117}


def f3(v):
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(str(round(float(v), 6))).quantize(Decimal('0.001'), ROUND_HALF_UP)).replace('0.', '.', 1)


def axes_style(ax, ygrid=True):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(length=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)


def seeds(ax, x, vals, width=.34):
    jitter = np.linspace(-width / 2, width / 2, len(vals)) if len(vals) > 1 else [0]
    ax.scatter(x + np.array(jitter), vals, s=16, color=NAVY, alpha=.45, zorder=4, linewidths=0)


def bar_label(ax, x, top, text, bold=True):
    ax.text(x, top, text, ha='center', va='bottom', fontsize=12.5, color=NAVY, fontweight='bold' if bold else 'normal')


def sel(ds, model='SimpleDyG', variant='full', protocol='author_compat'):
    g = runs[(runs.dataset == ds) & (runs.model == model) & (runs.variant == variant) & (runs.protocol == protocol)]
    return g.sort_values('seed').ndcg5.to_numpy()


def table2_chart(path):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.85))
    for ax, ds, title in zip(axes, ['UCI_13', 'ML_10M_13'], ['UCI', 'ML-10M']):
        models = ['TGAT', 'GraphMixer', 'SimpleDyG']
        for i, m in enumerate(models):
            v = sel(ds, m)
            mean, sd = v.mean(), v.std(ddof=1)
            ax.bar(i, mean, width=.56, color=MCOL[m], edgecolor='white', linewidth=2, zorder=2)
            ax.errorbar(i, mean, yerr=sd, color=INK2, lw=1.2, capsize=4, zorder=3)
            seeds(ax, i, v)
            bar_label(ax, i, max(mean + sd, v.max()) + .004, f3(mean))
            ax.scatter(i + .36, PAPER[(ds, m)], marker='D', s=58, facecolor='white', edgecolor=NAVY, lw=1.8, zorder=5)
        ax.axhline(TRIV[(ds, 'repeat')], ls=(0, (5, 3)), color=MUTED, lw=1.6, zorder=1)
        ax.text(2.75, TRIV[(ds, 'repeat')] + .003, f'Lặp lại slot gần nhất  {f3(TRIV[(ds, "repeat")])}', ha='right', va='bottom', fontsize=11, color=MUTED,
                bbox=dict(fc='white', ec='none', pad=1.5, alpha=.9), zorder=6)
        if TRIV[(ds, 'popular')] > 0:
            ax.axhline(TRIV[(ds, 'popular')], ls=(0, (1.5, 2.5)), color=MUTED, lw=1.8, zorder=1)
            ax.text(-.5, TRIV[(ds, 'popular')] + .003, f'Luôn đoán phim phổ biến nhất  {f3(TRIV[(ds, "popular")])}', ha='left', va='bottom', fontsize=11, color=MUTED,
                    bbox=dict(fc='white', ec='none', pad=1.5, alpha=.9), zorder=6)
        ax.set_xticks(range(3), models, fontsize=13)
        ax.set_xlim(-.6, 2.8)
        ax.set_ylim(0, .33 if ds == 'UCI_13' else .25)
        ax.set_title(title + ' · NDCG@5 (test)', loc='left', pad=10)
        axes_style(ax)
    handles = [Line2D([], [], marker='D', ls='', markerfacecolor='white', markeredgecolor=NAVY, markersize=8, label='Paper (Table 2)'),
               Line2D([], [], marker='o', ls='', color=NAVY, alpha=.45, markersize=5, label='Từng seed (10)'),
               Line2D([], [], color=MUTED, ls=(0, (5, 3)), label='Baseline không huấn luyện')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(.5, -.02))
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def trivial_chart(path):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.2))
    for ax, ds, title in zip(axes, ['UCI_13', 'ML_10M_13'], ['UCI', 'ML-10M']):
        items = [('Lặp lại slot gần nhất', TRIV[(ds, 'repeat')], 'h'),
                 ('Neighbour quen nhất (top-5)', TRIV[(ds, 'own5')], 'h'),
                 ('Neighbour quen nhất (top-1)', TRIV[(ds, 'own1')], 'h'),
                 ('Node phổ biến nhất (top-1)', TRIV[(ds, 'popular')], 'h'),
                 ('SimpleDyG', sel(ds).mean(), 'm'), ('GraphMixer', sel(ds, 'GraphMixer').mean(), 'm'),
                 ('TGAT', sel(ds, 'TGAT').mean(), 'm')]
        items.sort(key=lambda t: t[1])
        y = np.arange(len(items))
        colors = [SKY if k == 'm' else SLATE for _, _, k in items]
        ax.barh(y, [v for _, v, _ in items], height=.62, color=colors, edgecolor='white', linewidth=2)
        for yi, (name, v, k) in zip(y, items):
            ax.text(v + .004, yi, f3(v), va='center', fontsize=12, color=NAVY, fontweight='bold')
        ax.set_yticks(y, [n for n, _, _ in items], fontsize=12.5)
        for lab, (_, _, k) in zip(ax.get_yticklabels(), items):
            lab.set_color(NAVY if k == 'm' else INK2)
            lab.set_fontweight('bold' if k == 'm' else 'normal')
        ax.set_xlim(0, .34)
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        for s in ['top', 'right', 'left']:
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.tick_params(length=0)
        ax.set_title(title + ' · NDCG@5 (test)', loc='left', pad=10)
    fig.legend(handles=[Patch(color=SKY, label='Mô hình có huấn luyện'), Patch(color=SLATE, label='Heuristic không huấn luyện')],
               loc='lower center', ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(.5, -.02))
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def collapse_chart(path):
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    groups = [('UCI', 'UCI_13', 'author_compat', ['full', 'same_time', 'no_time', 'per_ego_permute']),
              ('ML-10M · mã gốc', 'ML_10M_13', 'author_compat', ['full', 'same_time', 'no_time']),
              ('ML-10M · mask padding', 'ML_10M_13', 'corrected', ['full', 'same_time', 'no_time'])]
    pos, ticks, centers = 0, [], []
    for gname, ds, proto, variants in groups:
        start = pos
        for v in variants:
            g = runs[(runs.dataset == ds) & (runs.model == 'SimpleDyG') & (runs.protocol == proto) & (runs.variant == v)]
            vals = g.top_share.to_numpy() * 100
            ax.scatter(pos + np.linspace(-.22, .22, len(vals)), vals, s=34, color=VCOL[v], edgecolor='white', lw=.8, zorder=3)
            ticks.append((pos, VLAB[v].replace(' ', '\n', 1) if v == 'per_ego_permute' else VLAB[v]))
            pos += 1
        centers.append(((start + pos - 1) / 2, gname))
        pos += .8
    ax.axhline(70, color=NAVY, ls=(0, (4, 3)), lw=1.2)
    ax.text(-.4, 72.5, 'ngưỡng suy biến 70%', ha='left', fontsize=11, color=NAVY)
    ax.set_xticks([c for c, _ in centers], [name for _, name in centers], fontsize=12.5, fontweight='bold')
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=VCOL[v], markersize=8, label=VLAB[v])
                       for v in ['full', 'same_time', 'no_time', 'per_ego_permute']],
              loc='upper center', bbox_to_anchor=(.5, -.1), ncol=4, frameon=False, fontsize=11.5, handletextpad=.3)
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
    ax.set_ylabel('% truy vấn test nhận cùng\nmột dự đoán', fontsize=12)
    axes_style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def mlfix_chart(path):
    fig, ax = plt.subplots(figsize=(6.6, 4.15))
    variants = ['full', 'same_time', 'no_time']
    for i, v in enumerate(variants):
        for k, (proto, off) in enumerate([('author_compat', -.2), ('corrected', .2)]):
            vals = sel('ML_10M_13', 'SimpleDyG', v, proto)
            m, sd = vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0)
            x = i + off
            if proto == 'author_compat':
                ax.bar(x, m, width=.36, facecolor='white', edgecolor=VCOL[v], hatch='///', linewidth=1.8, zorder=2)
            else:
                ax.bar(x, m, width=.36, color=VCOL[v], edgecolor='white', linewidth=2, zorder=2)
            ax.errorbar(x, m, yerr=sd, color=INK2, lw=1.1, capsize=3, zorder=3)
            seeds(ax, x, vals, .2)
            bar_label(ax, x, max(m + sd, vals.max()) + .004, f3(m))
    ax.axhline(.1475, ls=(0, (1.5, 2.5)), color=MUTED, lw=1.8)
    ax.text(2.55, .1475 + .003, 'luôn đoán phim 1419 = .148', ha='right', va='bottom', fontsize=11, color=MUTED)
    ax.set_xticks(range(3), [VLAB[v] for v in variants], fontsize=13)
    ax.set_ylim(0, .24)
    ax.set_title('ML-10M · NDCG@5 (test)', loc='left', pad=10)
    axes_style(ax)
    ax.legend(handles=[Patch(facecolor='white', edgecolor=INK2, hatch='///', label='Mã gốc (author_compat, 10 seed)'),
                       Patch(facecolor=INK2, label='Mask padding (corrected, 5 seed)')],
              loc='upper left', frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def temporal_chart(path, deltas):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.6, 3.55), gridspec_kw=dict(width_ratios=[1.15, 1]))
    variants = ['full', 'same_time', 'no_time', 'per_ego_permute']
    for i, v in enumerate(variants):
        vals = sel('UCI_13', 'SimpleDyG', v)
        m, sd = vals.mean(), vals.std(ddof=1)
        ax.bar(i, m, width=.58, color=VCOL[v], edgecolor='white', linewidth=2, zorder=2)
        ax.errorbar(i, m, yerr=sd, color=INK2, lw=1.2, capsize=4, zorder=3)
        seeds(ax, i, vals)
        bar_label(ax, i, max(m + sd, vals.max()) + .003, f3(m))
        if ('UCI_13', v) in PAPER_T4:
            ax.scatter(i + .37, PAPER_T4[('UCI_13', v)], marker='D', s=58, facecolor='white', edgecolor=NAVY, lw=1.8, zorder=5)
    ax.set_xticks(range(4), ['full', 'same time', 'no time', 'per-ego\npermute'], fontsize=12.5)
    ax.set_ylim(0, .15)
    ax.set_title('UCI · NDCG@5 theo biến thể (10 seed)', loc='left', pad=10)
    axes_style(ax)
    ax.legend(handles=[Line2D([], [], marker='D', ls='', markerfacecolor='white', markeredgecolor=NAVY, markersize=8, label='Paper (Table 4)'),
                       Line2D([], [], marker='o', ls='', color=NAVY, alpha=.45, markersize=5, label='Từng seed')],
              loc='upper center', bbox_to_anchor=(.5, -.14), frameon=False, fontsize=11, ncol=2)
    # forest plot
    rows = [('same_time', 'same time − full'), ('no_time', 'no time − full'), ('per_ego_permute', 'per-ego permute − full')]
    for k, (v, lab) in enumerate(rows):
        d = deltas[(deltas.dataset == 'UCI_13') & (deltas.protocol == 'author_compat') &
                   (deltas.comparison == f'{v} - full') & (deltas.metric == 'ndcg5')].iloc[0]
        y = len(rows) - 1 - k
        bx.plot([d.ci95_low, d.ci95_high], [y, y], color=VCOL[v], lw=4, solid_capstyle='round')
        bx.scatter(d.delta_mean, y, s=90, color='white', edgecolor=VCOL[v], lw=2.6, zorder=3)
        bx.text(.0235, y, f'{d.delta_mean:+.4f}\n[{d.ci95_low:+.3f}, {d.ci95_high:+.3f}]', va='center', fontsize=11, color=NAVY)
        bx.text(-.0305, y + .3, f'{int(d.wins)} seed tăng · {int(d.losses)} giảm', fontsize=10.5, color=MUTED)
    bx.axvline(0, color=NAVY, lw=1.2)
    bx.set_yticks(range(len(rows)), [r[1] for r in rows][::-1], fontsize=12.5)
    bx.set_xlim(-.031, .036)
    bx.set_ylim(-.6, len(rows) - .3)
    bx.set_xticks([-.03, -.02, -.01, 0, .01, .02])
    bx.xaxis.grid(True, color=GRID)
    bx.set_axisbelow(True)
    for s in ['top', 'right', 'left']:
        bx.spines[s].set_visible(False)
    bx.spines['bottom'].set_color('#cbd5e1')
    bx.tick_params(length=0)
    bx.set_title('ΔNDCG@5 so với full · cặp cùng seed, CI 95%', loc='left', pad=10, fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)




def cost_chart(path):
    items = [('GraphMixer · ML-10M', 423.8, 'GraphMixer'), ('TGAT · ML-10M', 108.1, 'TGAT'),
             ('GraphMixer · UCI', 75.6, 'GraphMixer'), ('SimpleDyG · ML-10M', 44.1, 'SimpleDyG'),
             ('SimpleDyG · UCI', 21.7, 'SimpleDyG'), ('TGAT · UCI', 16.1, 'TGAT')]
    items = items[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.25))
    y = np.arange(len(items))
    ax.barh(y, [v for _, v, _ in items], height=.62, color=[MCOL[m] for _, _, m in items], edgecolor='white', linewidth=2)
    for yi, (_, v, _) in zip(y, items):
        ax.text(v + 6, yi, f'{v:.0f} h'.replace('.', ','), va='center', fontsize=12, color=NAVY, fontweight='bold')
    ax.set_yticks(y, [n for n, _, _ in items], fontsize=12)
    ax.set_xlim(0, 500)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(length=0)
    ax.set_xlabel('GPU-giờ (T4), cộng mọi seed và mọi dimension', fontsize=11.5)
    ax.set_title('Chi phí theo mô hình · dataset', loc='left', pad=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def temporal_pattern_chart(path):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.05))
    for ax, ds, title in zip(axes, ['UCI', 'ML'], ['UCI · số tương tác mỗi time step', 'ML-10M · số tương tác mỗi time step']):
        c = pd.read_csv(f'data/steps_{ds}.csv', index_col=0).iloc[:, 0]
        last = c.index.max()
        cols = [SKY if i < last - 1 else (ORANGE if i == last - 1 else NAVY) for i in c.index]
        ax.bar(c.index, c.values, width=.72, color=cols, edgecolor='white', linewidth=2)
        for i in [last - 1, last]:
            ax.text(i, c[i] + c.max() * .02, f'{int(c[i]):,}'.replace(',', '.'), ha='center', va='bottom', fontsize=11, color=NAVY, fontweight='bold')
        top = c.idxmax()
        ax.text(top, c[top] + c.max() * .02, f'{int(c[top]):,}'.replace(',', '.'), ha='center', va='bottom', fontsize=11, color=NAVY, fontweight='bold')
        ax.set_xticks(c.index)
        ax.set_xlabel('time step', fontsize=12)
        ax.set_ylim(0, c.max() * 1.15)
        ax.set_title(title, loc='left', pad=10)
        axes_style(ax)
    fig.legend(handles=[Patch(color=SKY, label='train (step 0–10)'), Patch(color=ORANGE, label='validation (T−1)'), Patch(color=NAVY, label='test (T)')],
               loc='lower center', ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(.5, -.02))
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def _histories(root, ds, proto):
    import glob, json, re
    out = {}
    for d in glob.glob(f'{root}/{ds}_SimpleDyG_*'):
        m = re.match(rf'{ds}_SimpleDyG_(.+)_s(\d+)_', Path(d).name)
        if not m or not Path(d, 'history.json').exists():
            continue
        h = json.loads(Path(d, 'history.json').read_text())
        out.setdefault(m.group(1), []).append(pd.Series([e['ndcg5'] for e in h], index=[e['epoch'] for e in h]))
    return out


from pathlib import Path


def _draw_curves(ax, root, ds, pop, title, big=False):
    hs = _histories(root, ds, None)
    for v in ['full', 'same_time', 'no_time']:
        curves = hs.get(v, [])
        for cu in curves:
            ax.plot(cu.index, cu.values, color=VCOL[v], lw=1.0, alpha=.28)
        if curves:
            t = pd.concat(curves, axis=1)
            med = t.median(axis=1)[t.notna().sum(axis=1) >= min(3, len(curves))]
            ax.plot(med.index, med.values, color=VCOL[v], lw=3 if big else 2.6, label=VLAB[v])
    if pop:
        ax.axhline(pop, color=NAVY, ls=(0, (4, 3)), lw=1.4)
        ax.text(99, pop + .003, 'luôn đoán 1419 = .0955', ha='right', va='bottom', fontsize=12, color=NAVY,
                bbox=dict(fc='white', ec='none', pad=1.5, alpha=.9))
        ax.set_ylim(0, .12)
        ax.set_xlim(0, 101)
    else:
        ax.set_ylim(0, .25)
        ax.set_xlim(0, 60)
        ax.set_xticks(range(0, 61, 5))
    ax.set_xlabel('epoch', fontsize=13.5)
    ax.set_ylabel('NDCG@5 validation', fontsize=13.5)
    ax.tick_params(labelsize=12.5)
    ax.set_title(title, loc='left', pad=8, fontsize=16)
    axes_style(ax)


def curves_chart(path, which='uci'):
    P_OLD = 'data/previous_results/runs'
    if which == 'uci':
        fig, ax = plt.subplots(figsize=(7.3, 4.9))
        _draw_curves(ax, P_OLD, 'UCI_13', None, '', big=True)
        ax.set_ylim(0, .25)
        ax.set_yticks([0, .05, .10, .15, .20, .25], ['0', '.05', '.10', '.15', '.20', '.25'])
        ax.set_xticks(range(0, 61, 10))
        ax.legend(handles=[Line2D([], [], color=VCOL[v], lw=3, label=VLAB[v]) for v in ['full', 'same_time', 'no_time']] +
                  [Line2D([], [], color=MUTED, lw=1, alpha=.6, label='từng seed')],
                  loc='lower right', ncol=2, frameon=False, fontsize=12.5)
        fig.tight_layout()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.55), sharey=True)
        _draw_curves(axes[0], P_OLD, 'ML_10M_13', .0955, 'Mã gốc · 10 seed')
        _draw_curves(axes[1], 'data/hist_ml/runs', 'ML_10M_13', .0955, 'Mask padding · 5 seed')
        axes[1].set_ylabel('')
        for ax in axes:
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_yticks([0, .02, .04, .06, .08, .10, .12], ['0', '.02', '.04', '.06', '.08', '.10', '.12'])
        fig.legend(handles=[Line2D([], [], color=VCOL[v], lw=2.6, label=VLAB[v]) for v in ['full', 'same_time', 'no_time']] +
                   [Line2D([], [], color=MUTED, lw=1, alpha=.6, label='từng seed')],
                   loc='lower center', ncol=4, frameon=False, fontsize=12.5, bbox_to_anchor=(.5, -.02))
        fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def paired_chart(path):
    panels = [('UCI_13', 'author_compat', 'same_time'), ('UCI_13', 'author_compat', 'no_time'), ('UCI_13', 'author_compat', 'per_ego_permute'),
              ('ML_10M_13', 'author_compat', 'same_time'), ('ML_10M_13', 'author_compat', 'no_time'), ('ML_10M_13', 'corrected', 'no_time')]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 4.15))
    for ax, (ds, proto, v) in zip(axes.flat, panels):
        g = runs[(runs.dataset == ds) & (runs.model == 'SimpleDyG') & (runs.protocol == proto)]
        base = g[g.variant == 'full'].set_index('seed').ndcg5
        o = g[g.variant == v].set_index('seed').ndcg5
        common = base.index.intersection(o.index)
        up = int((o[common] > base[common] + 1e-9).sum()); down = int((o[common] < base[common] - 1e-9).sum())
        for s_ in common:
            ax.plot([0, 1], [base[s_], o[s_]], color=VCOL[v] if o[s_] > base[s_] else SLATE, lw=1.4, alpha=.85)
        ax.scatter(np.zeros(len(common)), base[common], s=22, color=SKY, edgecolor='white', zorder=3)
        ax.scatter(np.ones(len(common)), o[common], s=22, color=VCOL[v], edgecolor='white', zorder=3)
        ax.set_xticks([0, 1], ['full', VLAB[v]], fontsize=11.5)
        ax.set_xlim(-.25, 1.25)
        name = 'UCI' if ds == 'UCI_13' else ('ML-10M' if proto == 'author_compat' else 'ML-10M · corrected')
        ax.set_title(f'{name}: {up} tăng · {down} giảm', loc='left', fontsize=12.5, pad=6)
        axes_style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def dims_chart(path):
    import glob, json
    rows = []
    for d in glob.glob('data/previous_results/runs/*'):
        p = Path(d, 'validation_search.json')
        if p.exists():
            name = Path(d).name
            ds = 'UCI_13' if name.startswith('UCI') else 'ML_10M_13'
            model = 'TGAT' if '_TGAT_' in name else 'GraphMixer'
            for e in json.loads(p.read_text()):
                rows.append(dict(ds=ds, model=model, dim=e['dim'], ndcg=e['validation']['ndcg5'], hours=e['training_seconds'] / 3600))
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.35))
    for ax, ds, title in zip(axes, ['UCI_13', 'ML_10M_13'], ['UCI', 'ML-10M']):
        for m in ['TGAT', 'GraphMixer']:
            g = d[(d.ds == ds) & (d.model == m)].groupby('dim').ndcg.agg(['mean', 'std'])
            ax.errorbar(range(len(g)), g['mean'], yerr=g['std'], color=MCOL[m], lw=2.2, marker='o', ms=7, capsize=3, label=m)
        ax.set_xticks(range(6), ['16', '32', '64', '128', '256', '512'])
        ax.set_xlabel('hidden dimension', fontsize=12)
        ax.set_title(title + ' · NDCG@5 validation theo dim (10 seed)', loc='left', pad=8)
        axes_style(ax)
    fig.legend(handles=[Line2D([], [], color=MCOL[m], lw=2.2, marker='o', ms=7, label=m) for m in ['TGAT', 'GraphMixer']],
               loc='lower center', ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(.5, -.02))
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    t = d.groupby(['ds', 'model', 'dim']).hours.mean().unstack('dim').round(2)
    return t


if __name__ == '__main__':
    import os
    os.makedirs('build', exist_ok=True)
    os.makedirs('../figures', exist_ok=True)
    deltas = pd.concat([pd.read_csv('data/results_uci_permute/simpledyg_completion_analysis/paired_temporal_deltas.csv'),
                        pd.read_csv('data/result_ml/simpledyg_completion_analysis/paired_temporal_deltas.csv')]).drop_duplicates()
    jobs = {'table2': table2_chart, 'trivial': trivial_chart, 'collapse': collapse_chart, 'mlfix': mlfix_chart,
            'temporal': lambda p: temporal_chart(p, deltas), 'cost': cost_chart, 'steps': temporal_pattern_chart,
            'curves1': lambda p: curves_chart(p, 'uci'), 'curves2': lambda p: curves_chart(p, 'ml'),
            'paired': paired_chart, 'dims': dims_chart}
    png = {'table2': 'fig_final_table2', 'trivial': 'fig_final_trivial', 'collapse': 'fig_final_collapse',
           'mlfix': 'fig_final_ml_corrected', 'temporal': 'fig_final_temporal_uci', 'cost': 'fig_final_cost',
           'steps': 'fig_final_steps', 'curves1': 'fig_final_curves_uci_ml', 'curves2': 'fig_final_curves_ml_fix',
           'paired': 'fig_final_paired', 'dims': 'fig_final_dims'}
    for name, fn in jobs.items():
        fn(f'build/ch_{"mlfix" if name == "mlfix" else name}.svg')
        plt.rcParams['savefig.dpi'] = 150
        fn(f'../figures/{png[name]}.png')
        plt.rcParams['savefig.dpi'] = 100
        print('chart', name)
