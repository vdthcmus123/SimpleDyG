"""Post-hoc analysis for the SimpleDyG course project (CPU only, no torch).

1. Trivial baselines computed from the released sequences (no training):
   popularity top-k, repeat-last-active-slot, own-most-frequent neighbours.
2. Collapse census: share of test queries that receive the single most common
   prediction in each run (detects "always predict the most popular node").
3. Paired temporal-alignment statistics (variant - full, same seed), t-based 95% CI,
   exact sign test, optionally merging previous and new result tables.

Metric definitions copy kaggle_pipeline_dual.metrics: deduplicated binary NDCG@5
and set Jaccard; truth excludes the ego node and empty targets are skipped.
"""
import collections
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_DIRS = {'UCI_13': 'UCI_13/12', 'ML_10M_13': 'ML_10M_13/12'}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 19: 2.093, 29: 2.045}


def _unique(xs):
    return list(dict.fromkeys(xs))


def ndcg5(pred, truth):
    p, y = _unique(pred), _unique(truth)
    denom = sum(1 / math.log2(i + 2) for i in range(min(5, len(y))))
    return sum(1 / math.log2(i + 2) for i, x in enumerate(p[:5]) if x in y) / denom


def jaccard(pred, truth):
    a, b = set(pred), set(truth)
    return len(a & b) / len(a | b)


def _read(path):
    return [x.split() for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def _targets(line):
    i = line.index('<|pre|>')
    j = line.index('<|endofpre|>') if '<|endofpre|>' in line else len(line)
    return [x for x in line[i + 1:j] if not x.startswith('<|')]


def _history_slots(line):
    """Return (ego, list of per-time-slot node lists) from an input sequence."""
    ego = line[2]
    end = line.index('<|endofhistory|>') if '<|endofhistory|>' in line else len(line)
    slots, current = [], None
    for x in line[3:end]:
        if x.startswith('<|time'):
            if current is not None:
                slots.append(current)
            current = []
        elif not x.startswith('<|'):
            if current is None:
                current = []
            current.append(x)
    if current is not None:
        slots.append(current)
    return ego, slots


def trivial_baselines(author_root, datasets=('UCI_13', 'ML_10M_13'), splits=('val', 'test')):
    rows = []
    for name in datasets:
        folder = Path(author_root) / 'resources' / SPLIT_DIRS[name]
        train = _read(folder / 'train.link_prediction')
        popularity = collections.Counter(x for line in train for x in _targets(line))
        ranked = [x for x, _ in popularity.most_common(50)]
        for split in splits:
            inputs, gts = _read(folder / f'{split}.link_prediction'), _read(folder / f'{split}_gt.link_prediction')
            scores = collections.defaultdict(list)
            for inp, gt in zip(inputs, gts):
                ego, slots = _history_slots(inp)
                truth = [x for x in _targets(gt) if x != ego]
                if not truth:
                    continue
                flat = [x for s in slots for x in s]
                nonempty = [s for s in slots if s]
                own = [x for x, _ in collections.Counter(flat).most_common()]
                methods = {
                    'Popular top-1 (train)': [x for x in ranked if x != ego][:1],
                    'Popular top-5 (train)': [x for x in ranked if x != ego][:5],
                    'Repeat last active slot': nonempty[-1] if nonempty else [],
                    'Own most frequent top-1': own[:1],
                    'Own most frequent top-5': own[:5],
                }
                for method, pred in methods.items():
                    scores[method].append((ndcg5(pred, truth), jaccard(pred, truth)))
            for method, values in scores.items():
                arr = np.array(values)
                rows.append(dict(dataset=name, split=split, method=method, n_queries=len(arr),
                                 ndcg5=arr[:, 0].mean(), jaccard=arr[:, 1].mean()))
    return pd.DataFrame(rows)


def collapse_census(runs_root, split='test'):
    rows = []
    for path in sorted(Path(runs_root).glob(f'*/{split}_predictions.jsonl')):
        records = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
        records = [r for r in records if r.get('truth')]
        if not records:
            continue
        keys = collections.Counter(json.dumps(sorted(set(map(str, r['pred'])))) for r in records)
        top, count = keys.most_common(1)[0]
        rows.append(dict(run_id=path.parent.name, n=len(records), distinct_prediction_sets=len(keys),
                         top_prediction=top, top_share=count / len(records),
                         empty_share=sum(not r['pred'] for r in records) / len(records),
                         ended_fraction=float(np.mean([r.get('ended', True) for r in records]))))
    return pd.DataFrame(rows)


def label_collapse(df, threshold=0.7):
    """A run is degenerate when one prediction set (or empty output) covers >= threshold of queries."""
    def status(r):
        if r.empty_share >= threshold:
            return 'collapsed: empty/no end token'
        if r.top_share >= threshold:
            return 'collapsed: constant prediction ' + r.top_prediction
        return 'personalised'
    df = df.copy()
    df['status'] = df.apply(status, axis=1)
    return df


def _sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_deltas(results, reference='full', metrics=('ndcg5', 'ndcg5_author', 'jaccard')):
    """results: rows of SimpleDyG runs with dataset, protocol, variant, seed and metric columns."""
    rows = []
    s = results[results.model == 'SimpleDyG']
    for (dataset, protocol), g in s.groupby(['dataset', 'protocol']):
        base = g[g.variant == reference].set_index('seed')
        for variant in sorted(set(g.variant) - {reference}):
            other = g[g.variant == variant].set_index('seed')
            common = base.index.intersection(other.index)
            for metric in metrics:
                d = (other.loc[common, metric] - base.loc[common, metric]).to_numpy()
                n = len(d)
                if n < 2:
                    continue
                se = d.std(ddof=1) / math.sqrt(n)
                t = T975.get(n - 1, 1.96)
                wins, losses = int((d > 1e-12).sum()), int((d < -1e-12).sum())
                rows.append(dict(dataset=dataset, protocol=protocol, comparison=f'{variant} - {reference}',
                                 metric=metric, n_pairs=n, delta_mean=d.mean(), ci95_low=d.mean() - t * se,
                                 ci95_high=d.mean() + t * se, wins=wins, losses=losses,
                                 sign_test_p=_sign_test(wins, losses)))
    return pd.DataFrame(rows)


def variant_summary(results):
    s = results[results.model == 'SimpleDyG']
    return (s.groupby(['dataset', 'protocol', 'variant'])
             .agg(n_seeds=('seed', 'nunique'), ndcg5_mean=('ndcg5', 'mean'), ndcg5_std=('ndcg5', 'std'),
                  ndcg5_author_mean=('ndcg5_author', 'mean'), jaccard_mean=('jaccard', 'mean'),
                  jaccard_std=('jaccard', 'std'), best_epoch_mean=('best_epoch', 'mean'))
             .reset_index())


def run_all(author_root, result_tables, runs_roots, out_dir):
    """result_tables: list of results_per_seed.csv paths; runs_roots: folders with runs/<id>/."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = pd.concat([pd.read_csv(p) for p in result_tables], ignore_index=True)
    results = results.drop_duplicates('run_id')
    baselines = trivial_baselines(author_root)
    census = pd.concat([collapse_census(r) for r in runs_roots], ignore_index=True)
    census = label_collapse(census).merge(
        results[['run_id', 'dataset', 'model', 'variant', 'protocol', 'seed', 'ndcg5', 'jaccard']], on='run_id', how='left')
    summary, paired = variant_summary(results), paired_deltas(results)
    baselines.to_csv(out / 'trivial_baselines.csv', index=False)
    census.to_csv(out / 'collapse_census.csv', index=False)
    summary.to_csv(out / 'variant_summary.csv', index=False)
    paired.to_csv(out / 'paired_temporal_deltas.csv', index=False)
    counts = (census[census.model == 'SimpleDyG']
              .assign(kind=lambda d: d.status.str.split(':').str[0])
              .groupby(['dataset', 'protocol', 'variant', 'kind']).size().unstack(fill_value=0))
    counts.to_csv(out / 'collapse_counts.csv')
    return dict(baselines=baselines, census=census, summary=summary, paired=paired, counts=counts)


if __name__ == '__main__':
    import sys
    author, out = sys.argv[1], sys.argv[2]
    tables = sys.argv[3].split(',')
    runs = sys.argv[4].split(',')
    res = run_all(author, tables, runs, out)
    pd.set_option('display.width', 200)
    for key in ['baselines', 'summary', 'paired', 'counts']:
        print('==', key)
        print(res[key].round(4).to_string())
