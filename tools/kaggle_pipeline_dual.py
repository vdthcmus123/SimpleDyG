"""Source embedded verbatim in the standalone Kaggle notebook. No CLI side effects."""
from pathlib import Path
from collections import Counter, defaultdict
import contextlib
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
import types
import zipfile

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.utils.data import DataLoader


def gelu_new(x):
    """GPT-2 tanh GELU, same arithmetic as transformers.activations.NewGELUActivation."""
    return 0.5*x*(1.0+torch.tanh(math.sqrt(2.0/math.pi)*(x+0.044715*torch.pow(x, 3.0))))


def get_activation(name):
    activations = {'gelu_new': gelu_new, 'gelu': torch.nn.functional.gelu,
                   'relu': torch.nn.functional.relu, 'tanh': torch.tanh,
                   'sigmoid': torch.sigmoid, 'silu': torch.nn.functional.silu,
                   'swish': torch.nn.functional.silu}
    if name not in activations:
        raise ValueError(f'Activation outside the released GPT-2 path: {name}')
    return activations[name]


class AuthorAdamW(torch.optim.Optimizer):
    """Legacy HF AdamW arithmetic, including epsilon before bias correction.

    Reference: Hugging Face transformers v4.44.2 optimization.py (Apache-2.0).
    Embedded to keep author_compat independent of the optional HF/Rust packages.
    """
    def __init__(self, params, lr=1e-3, betas=(.9, .999), eps=1e-8,
                 weight_decay=0., correct_bias=True):
        if lr < 0 or eps < 0 or weight_decay < 0 or any(not 0 <= b < 1 for b in betas):
            raise ValueError('Invalid AdamW parameters')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, correct_bias=correct_bias))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError('AuthorAdamW requires dense gradients')
                state = self.state[parameter]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
                state['step'] += 1
                m, v = state['exp_avg'], state['exp_avg_sq']
                m.mul_(beta1).add_(gradient, alpha=1.-beta1)
                v.mul_(beta2).addcmul_(gradient, gradient, value=1.-beta2)
                denominator = v.sqrt().add_(group['eps'])
                step_size = group['lr']
                if group['correct_bias']:
                    step_size = step_size*math.sqrt(1.-beta2**state['step'])/(1.-beta1**state['step'])
                parameter.addcdiv_(m, denominator, value=-step_size)
                if group['weight_decay']:
                    parameter.add_(parameter, alpha=-group['lr']*group['weight_decay'])
        return loss


def linear_warmup_decay(optimizer, warmup_steps, total_steps):
    def scale(step):
        if step < warmup_steps:
            return float(step)/max(1, warmup_steps)
        return max(0., float(total_steps-step)/max(1, total_steps-warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)

AUTHOR_COMMIT = 'fa0dc82587d8ce787345a0379f65a0dcf5d9d9ff'
DYGLIB_COMMIT = '3aacc36b94b8d2d8293d70a74fdf6d39089b4163'
SPECS = {
    'UCI_13': dict(raw='uci', timestamp=12, layers=6, heads=8, dim=768, batch=32, lr=1e-4),
    'ML_10M_13': dict(raw='movie', timestamp=12, layers=2, heads=4, dim=128, batch=64, lr=1e-5),
}
PAPER = {
    'UCI_13': {'full': (.104, .010, .092, .014), 'same_time': (.090, .013, .083, .012), 'no_time': (.111, .015, .091, .014)},
    'ML_10M_13': {'full': (.138, .009, .131, .008), 'same_time': (.147, .001, .139, .001), 'no_time': (.117, .062, .111, .059)},
}


def make_config(profile='pilot', **overrides):
    if profile not in {'smoke', 'pilot', 'paper'}:
        raise ValueError(profile)
    cfg = dict(profile=profile, datasets=['UCI_13', 'ML_10M_13'], seeds=[0, 1, 2],
               protocol='corrected', variants=['full', 'same_time', 'no_time'],
               baseline_models=['TGAT', 'GraphMixer'], baseline_dims=[64, 128],
               epochs=100, baseline_epochs=50, patience=10, train_block=512,
               context=1024, max_new_tokens=64, micro_batch=8, amp=True,
               baseline_batch=128, score_batch=64, history_cap=128,
               mixer_tokens=64, tgat_neighbors=20, time_unit_seconds=86400.,
               train_limit=None, eval_limit=None, edge_limit=None,
               export_checkpoints=False, device='auto', baseline_micro_batch=8, execution_format='dual_resume_v1')
    if profile == 'smoke':
        cfg.update(datasets=['UCI_13'], seeds=[0], epochs=1, baseline_epochs=1,
                   patience=1, train_block=64, context=128, max_new_tokens=8,
                   micro_batch=4, baseline_batch=16, score_batch=64,
                   baseline_dims=[16], baseline_micro_batch=16, history_cap=8, mixer_tokens=8,
                   tgat_neighbors=4, train_limit=32, eval_limit=4, edge_limit=64, amp=False)
    if profile == 'paper':
        cfg.update(seeds=list(range(10)), protocol='author_compat', amp=False,
                   baseline_epochs=100, baseline_dims=[16, 32, 64, 128, 256, 512],
                   history_cap=1024, mixer_tokens=1024)
    cfg.update(overrides)
    if not cfg['seeds'] or not cfg['datasets'] or not (cfg['variants'] or cfg['baseline_models']):
        raise ValueError('At least one dataset, seed and model/variant is required')
    if any(key not in {'TGAT', 'GraphMixer', 'DyGFormer'} for key in cfg['baseline_models']):
        raise ValueError('Unsupported baseline name')
    if any(v not in {'full', 'same_time', 'no_time', 'per_ego_permute'} for v in cfg['variants']):
        raise ValueError('Unsupported temporal variant')
    if any(d < 2 or d % 2 for d in cfg['baseline_dims']):
        raise ValueError('Baseline dimensions must be positive even integers')
    if cfg['time_unit_seconds'] <= 0 or cfg['history_cap'] < 2 or cfg['mixer_tokens'] < 2:
        raise ValueError('Invalid history/time configuration')
    if not isinstance(cfg['baseline_micro_batch'], int) or cfg['baseline_micro_batch'] < 1:
        raise ValueError('baseline_micro_batch must be a positive integer')
    if cfg['protocol'] not in {'corrected', 'author_compat'}:
        raise ValueError('Unknown protocol')
    if cfg['protocol'] == 'author_compat' and cfg['context'] != 1024:
        raise ValueError('author_compat requires the released 1024-position architecture')
    if cfg['train_block'] > cfg['context'] or cfg['max_new_tokens'] >= cfg['context']:
        raise ValueError('Invalid sequence limits')
    if any(ds not in SPECS for ds in cfg['datasets']):
        raise ValueError('This pipeline supports the complete UCI and ML-10M releases only')
    return cfg


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_lines(path):
    return [x.split() for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def unique(xs):
    return list(dict.fromkeys(xs))


def metrics(pred, truth, author=False):
    """Standard binary NDCG deduplicates both lists; author metric preserves duplicates."""
    p, y = (list(pred), list(truth)) if author else (unique(pred), unique(truth))
    if not y:
        raise ValueError('Empty ground truth must be explicitly excluded before scoring')
    denom = sum(1 / math.log2(i + 2) for i in range(min(5, len(y))))
    ndcg = sum(1 / math.log2(i + 2) for i, x in enumerate(p[:5]) if x in y) / denom
    a, b = set(pred), set(truth)
    jac = len(a & b) / len(a | b)
    return {'ndcg5': ndcg, 'jaccard': jac}


def summarize_predictions(rows, out_path=None):
    eligible = [r for r in rows if r['truth']]
    if not eligible:
        raise ValueError('No eligible evaluation queries')
    result = {}
    for key in ['ndcg5', 'ndcg5_author', 'jaccard', 'jaccard_at5']:
        result[key] = float(np.mean([r[key] for r in eligible]))
    result.update(n_total=len(rows), n_evaluated=len(eligible), n_empty_target=len(rows)-len(eligible),
                  ended_fraction=float(np.mean([r.get('ended', True) for r in eligible])))
    if out_path is not None:
        with Path(out_path).open('w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return result


def load_dyglib(root):
    """Namespace-only import rewriting prevents clashes with the author's models/utils."""
    base = 'dyglib_vendor'
    for suffix, folder in [('', root), ('.models', Path(root)/'models'), ('.utils', Path(root)/'utils')]:
        name = base+suffix
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(folder)]
            sys.modules[name] = mod
    for rel in ['utils/DataLoader.py', 'utils/utils.py', 'models/modules.py',
                'models/TGAT.py', 'models/GraphMixer.py', 'models/DyGFormer.py']:
        name = base+'.'+rel[:-3].replace('/', '.')
        if name in sys.modules:
            continue
        file = Path(root)/rel
        source = file.read_text(encoding='utf-8').replace('from models.', f'from {base}.models.').replace('from utils.', f'from {base}.utils.')
        mod = types.ModuleType(name)
        mod.__file__ = str(file)
        sys.modules[name] = mod
        exec(compile(source, str(file), 'exec'), mod.__dict__)
    return {key: getattr(sys.modules[base+'.models.'+key], key) for key in ['TGAT', 'GraphMixer', 'DyGFormer']}, sys.modules[base+'.utils.utils'].NeighborSampler


def build_sampler(ds, cfg, Sampler):
    # Only events strictly before the test bin may enter the neighbor store.
    # Validation and training further filter by exact query time using searchsorted.
    maximum = ds['spec']['timestamp']
    observed = ds['raw'][ds['raw'].timestamp < maximum]
    adjacency = [[] for _ in range(len(ds['vocab'])+1)]
    for row in observed.itertuples():
        u, v = ds['vocab'][str(row.u)]+1, ds['vocab'][str(row.i)]+1
        t = float(row.ts)/cfg['time_unit_seconds']
        adjacency[u].append((v, int(row.idx), t))
        adjacency[v].append((u, int(row.idx), t))
    class CappedSampler(Sampler):
        def find_neighbors_before(self, node_id, interact_time, return_sampled_probabilities=False):
            values = super().find_neighbors_before(node_id, interact_time, return_sampled_probabilities)
            return tuple(x[-cfg['history_cap']:] if x is not None else None for x in values)
    sampler = CappedSampler(adjacency, sample_neighbor_strategy='recent', seed=0)
    # Executable no-future-history check at both frozen evaluation cutoffs.
    for bin_id in [maximum-1, maximum]:
        cutoff = float(ds['raw'].loc[ds['raw'].timestamp == bin_id, 'ts'].min())/cfg['time_unit_seconds']
        for u in range(1, len(adjacency)):
            _, _, times, _ = sampler.find_neighbors_before(u, cutoff)
            assert np.all(times < cutoff)
    return sampler


class TemporalRanker(nn.Module):
    def __init__(self, key, dim, ds, cfg, device, sampler, classes):
        super().__init__()
        self.key, self.cfg = key, cfg
        # UCI release contains all-zero features; ML-10M has no aligned .npy.
        # Both baselines use explicit zero features. No invented semantic features.
        nodes = np.zeros((len(ds['vocab'])+1, dim), dtype=np.float32)
        edges = np.zeros((int(ds['raw'].idx.max())+1, dim), dtype=np.float32)
        common = dict(node_raw_features=nodes, edge_raw_features=edges, neighbor_sampler=sampler,
                      time_feat_dim=dim, dropout=.1, device=str(device))
        if key == 'TGAT':
            self.backbone = classes[key](**common, num_layers=1, num_heads=2)
        elif key == 'GraphMixer':
            self.backbone = classes[key](**common, num_tokens=cfg['mixer_tokens'],
                                         num_layers=1 if ds['name']=='UCI_13' else 2)
        elif key == 'DyGFormer':
            self.backbone = classes[key](**common, channel_embedding_dim=dim, patch_size=2,
                                         num_layers=2, num_heads=2,
                                         max_input_sequence_length=cfg['history_cap'])
        else:
            raise ValueError(key)
        self.head = nn.Sequential(nn.Linear(2*dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.to(device)

    def forward(self, src, dst, times):
        kw = dict(src_node_ids=src, dst_node_ids=dst, node_interact_times=times)
        if self.key == 'TGAT':
            kw['num_neighbors'] = self.cfg['tgat_neighbors']
        if self.key == 'GraphMixer':
            kw.update(num_neighbors=self.cfg['mixer_tokens'], time_gap=self.cfg['history_cap'])
        a, b = self.backbone.compute_src_dst_node_temporal_embeddings(**kw)
        return self.head(torch.cat([a, b], dim=-1)).squeeze(-1)

    def embed(self, node_ids, times):
        if self.key == 'TGAT':
            return self.backbone.compute_node_temporal_embeddings(node_ids, times,
                current_layer_num=self.backbone.num_layers, num_neighbors=self.cfg['tgat_neighbors'])
        if self.key == 'GraphMixer':
            return self.backbone.compute_node_temporal_embeddings(node_ids, times,
                num_neighbors=self.cfg['mixer_tokens'], time_gap=self.cfg['history_cap'])
        raise ValueError('DyGFormer embeddings depend on the node pair and cannot be cached separately')


@torch.no_grad()
def evaluate_ranker(model, ds, split, cfg, device, k_choice=None, output=None):
    model.eval()
    target_bin = ds['spec']['timestamp'] - (split == 'val')
    cutoff = float(ds['raw'].loc[ds['raw'].timestamp == target_bin, 'ts'].min())/cfg['time_unit_seconds']
    candidates = np.arange(1, len(ds['vocab'])+1, dtype=np.int64)
    inverse = {v+1: k for k, v in ds['vocab'].items()}
    # TGAT/GraphMixer have independent node embeddings, so full-candidate evaluation
    # can cache them at one frozen cutoff. DyGFormer needs full pairwise scoring.
    cached = None
    if model.key != 'DyGFormer':
        cached = torch.cat([model.embed(ids, np.full(len(ids), cutoff))
                            for ids in np.array_split(candidates, math.ceil(len(candidates)/cfg['score_batch']))])
    rows, per_k = [], {k: [] for k in [1, 5, 10, 20]}
    for inp, target in zip(ds[split], ds[split+'_gt']):
        check_deadline()
        ego = inp[2]
        u = ds['vocab'][ego]+1
        truth = [x for x in target if x in ds['vocab'] and x != ego]
        parts = []
        for start in range(0, len(candidates), cfg['score_batch']):
            check_deadline()
            ids = candidates[start:start+cfg['score_batch']]
            if cached is None:
                scores = model(np.full(len(ids), u), ids, np.full(len(ids), cutoff))
            else:
                src_embedding = cached[u-1].expand(len(ids), -1)
                scores = model.head(torch.cat([src_embedding, cached[start:start+len(ids)]], dim=-1)).squeeze(-1)
            parts.append(scores.cpu().numpy())
        scores = np.concatenate(parts)
        if not np.isfinite(scores).all():
            raise FloatingPointError('Non-finite candidate scores')
        scores[u-1] = -np.inf
        # Stable ties are broken by canonical vocabulary IDs, equally for all baselines.
        ranked = [inverse[int(i+1)] for i in np.argsort(-scores, kind='stable')[:20]]
        if truth:
            for k in per_k:
                check_deadline()
                per_k[k].append(metrics(ranked[:k], truth)['jaccard'])
        rows.append(dict(ego=ego, ranking=ranked, truth=truth))
    if not any(per_k.values()):
        raise ValueError('No eligible ranking queries')
    macro_k = {k: float(np.mean(v)) for k, v in per_k.items()}
    # Select output cardinality on validation only. The paper's test max-over-k
    # is reported separately as an oracle reference, never as the primary score.
    selected = max(macro_k, key=lambda k: (macro_k[k], -k)) if k_choice is None else k_choice
    predictions = [prediction_row(r['ego'], r['ranking'][:selected], r['truth'], ranking=r['ranking']) for r in rows]
    for r, original in zip(predictions, rows):
        check_deadline()
        if r['truth']:
            r['ndcg5'] = metrics(original['ranking'], r['truth'])['ndcg5']
            r['ndcg5_author'] = metrics(original['ranking'], r['truth'], author=True)['ndcg5']
            r['jaccard_at5'] = metrics(original['ranking'][:5], r['truth'])['jaccard']
    result = summarize_predictions(predictions, output)
    result.update(selected_k=selected, jaccard_oracle_max_k=max(macro_k.values()),
                  jaccard_by_k=macro_k, candidate_count=len(candidates)-1)
    return result


def baseline_training_edges(ds, cfg):
    train = ds['raw'][ds['raw'].timestamp <= ds['spec']['timestamp']-2].copy()
    forbidden = defaultdict(set)
    # All positives in each TRAIN bin are labels, never validation/test labels.
    for r in train.itertuples():
        u, v = ds['vocab'][str(r.u)]+1, ds['vocab'][str(r.i)]+1
        forbidden[(u, int(r.timestamp))].update([u, v])
        forbidden[(v, int(r.timestamp))].update([v, u])
    pairs = [(ds['vocab'][str(r.u)]+1, ds['vocab'][str(r.i)]+1,
              float(r.ts)/cfg['time_unit_seconds'], int(r.timestamp)) for r in train.itertuples()]
    pairs += [(v, u, t, b) for u, v, t, b in pairs]
    if cfg['edge_limit']:
        pairs = pairs[:cfg['edge_limit']]
    return pairs, forbidden


def train_ranker_candidate(key, dim, seed, ds, cfg, device, sampler, classes, run_dir):
    seed_all(seed)
    model = TemporalRanker(key, dim, ds, cfg, device, sampler, classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pairs, forbidden = baseline_training_edges(ds, cfg)
    rng = np.random.default_rng(seed)
    best, stale, best_epoch, history = -float('inf'), 0, 0, []
    checkpoint = run_dir/f'candidate_dim{dim}.pt'
    for epoch in range(1, cfg['baseline_epochs']+1):
        model.train()
        losses = []
        for start in range(0, len(pairs), cfg['baseline_batch']):
            # Chronological batches; models here have no stateful node memory.
            batch = pairs[start:start+cfg['baseline_batch']]
            src = np.array([r[0] for r in batch], dtype=np.int64)
            dst = np.array([r[1] for r in batch], dtype=np.int64)
            times = np.array([r[2] for r in batch], dtype=np.float64)
            neg = []
            for u, _, _, bin_id in batch:
                blocked = forbidden[(u, bin_id)]
                if len(blocked) >= len(ds['vocab']):
                    raise ValueError('No negative candidate')
                item = int(rng.integers(1, len(ds['vocab'])+1))
                while item in blocked:
                    item = int(rng.integers(1, len(ds['vocab'])+1))
                neg.append(item)
            optimizer.zero_grad(set_to_none=True)
            # One joint call uses a common realization of dropout for the BPR batch.
            scores = model(np.concatenate([src, src]), np.concatenate([dst, np.array(neg)]), np.concatenate([times, times]))
            pos, negative = scores.chunk(2)
            loss = torch.nn.functional.softplus(negative-pos).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite BPR loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate_ranker(model, ds, 'val', cfg, device)
        score = validation['ndcg5']
        if score > best + 1e-12:
            best, stale, best_epoch = score, 0, epoch
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), **validation))
        dump_json(run_dir/f'history_dim{dim}.json', history)
        print(f"{ds['name']} {key} d={dim} seed={seed} epoch={epoch} val_NDCG5={score:.4f} stale={stale}", flush=True)
        if stale >= cfg['patience']:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    validation = evaluate_ranker(model, ds, 'val', cfg, device)
    return model, validation, best_epoch, len(history), checkpoint


def train_ranker(key, seed, ds, cfg, device, dyglib, run_dir):
    classes, Sampler = load_dyglib(dyglib)
    sampler = build_sampler(ds, cfg, Sampler)
    began = time.perf_counter()
    choices = []
    for dim in cfg['baseline_dims']:
        model, validation, epoch, epochs_run, checkpoint = train_ranker_candidate(key, dim, seed, ds, cfg, device, sampler, classes, run_dir)
        choices.append(dict(dim=dim, validation=validation, best_epoch=epoch,
                            epochs_run=epochs_run, checkpoint=str(checkpoint)))
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    winner = max(choices, key=lambda c: (c['validation']['ndcg5'], -c['dim']))
    dump_json(run_dir/'validation_search.json', choices)
    model = TemporalRanker(key, winner['dim'], ds, cfg, device, sampler, classes)
    model.load_state_dict(torch.load(winner['checkpoint'], map_location=device, weights_only=True))
    validation = evaluate_ranker(model, ds, 'val', cfg, device, output=run_dir/'val_predictions.jsonl')
    training_seconds = time.perf_counter()-began
    testing = time.perf_counter()
    test = evaluate_ranker(model, ds, 'test', cfg, device, k_choice=validation['selected_k'], output=run_dir/'test_predictions.jsonl')
    torch.save(model.state_dict(), run_dir/'best.pt')
    # Candidate checkpoints are local intermediates; retain only the chosen checkpoint.
    for c in choices:
        Path(c['checkpoint']).unlink()
    result = dict(model=key, variant='full', seed=seed, architecture={'dim': winner['dim']},
                  best_epoch=winner['best_epoch'], epochs_run=winner['epochs_run'],
                  parameters=sum(p.numel() for p in model.parameters()), validation=validation, test=test,
                  training_seconds=training_seconds, testing_seconds=time.perf_counter()-testing,
                  feature_policy='all-zero node/edge features; dimensions selected on validation',
                  training_task='chronological event BPR, both edge directions; frozen next-bin ranking at evaluation')
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def run_id(ds, model, variant, seed, cfg):
    payload = dict(dataset=ds['name'], model=model, variant=variant, seed=seed, config=cfg,
                   data=ds['audit']['files'], author_commit=AUTHOR_COMMIT,
                   dyglib_commit=DYGLIB_COMMIT, implementation='dual_resume_v1')
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"{ds['name']}_{model}_{variant}_s{seed}_{digest}"


def execute_run(ds, model, variant, seed, cfg, device, output_root, author_root, dyglib_root):
    identifier = run_id(ds, model, variant, seed, cfg)
    directory = Path(output_root)/'runs'/identifier
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory/'result.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding='utf-8'))
        if previous['status'] == 'complete':
            print('Reuse completed run:', identifier, flush=True)
            return previous
    dump_json(directory/'config.json', cfg)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        if model == 'SimpleDyG':
            result = train_lm(author_root, ds, variant, seed, cfg, device, directory)
        else:
            result = train_ranker(model, seed, ds, cfg, device, dyglib_root, directory)
        result.update(status='complete', run_id=identifier, dataset=ds['name'],
                      protocol=cfg['protocol'], profile=cfg['profile'],
                      truncated_data=bool(cfg['train_limit'] or cfg['eval_limit'] or cfg['edge_limit']),
                      peak_gpu_mb=float(torch.cuda.max_memory_allocated(device)/1e6) if device.type=='cuda' else None)
    except Exception as exc:
        import traceback
        (directory/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        result = dict(status='failed', run_id=identifier, dataset=ds['name'], model=model,
                      variant=variant, seed=seed, error=str(exc), protocol=cfg['protocol'], profile=cfg['profile'])
        print('FAILED:', identifier, str(exc), flush=True)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    dump_json(result_path, result)
    return result


def export_results(records, cfg, output_root, make_archive=True):
    root = Path(output_root)
    dump_json(root/'run_manifest.json', records)
    complete = [r for r in records if r['status'] == 'complete']
    table = []
    for r in complete:
        table.append({k: r[k] for k in ['run_id', 'dataset', 'model', 'variant', 'seed', 'protocol', 'profile',
                                      'best_epoch', 'epochs_run', 'parameters', 'training_seconds', 'testing_seconds', 'peak_gpu_mb', 'worker_id', 'physical_gpu']} |
                     {k: r['test'][k] for k in ['ndcg5', 'ndcg5_author', 'jaccard', 'jaccard_at5', 'n_evaluated', 'ended_fraction']} |
                     {'val_ndcg5': r['validation']['ndcg5'], 'selected_k': r['test'].get('selected_k'),
                      'jaccard_oracle_max_k': r['test'].get('jaccard_oracle_max_k'), 'truncated_data': r['truncated_data']})
    df = pd.DataFrame(table)
    df.to_csv(root/'results_per_seed.csv', index=False)
    summaries = []
    if not df.empty:
        for keys, g in df.groupby(['dataset', 'model', 'variant', 'protocol', 'profile']):
            row = dict(zip(['dataset', 'model', 'variant', 'protocol', 'profile'], keys))
            row['n_seeds'] = g.seed.nunique()
            for metric in ['ndcg5', 'ndcg5_author', 'jaccard', 'jaccard_at5', 'training_seconds', 'parameters']:
                row[metric+'_mean'] = g[metric].mean()
                row[metric+'_std'] = g[metric].std(ddof=1)  # NaN for one seed: uncertainty is not zero.
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(root/'summary_mean_std.csv', index=False)
    paper_rows = []
    for dataset, variants in PAPER.items():
        for variant, (n, ns, j, js) in variants.items():
            paper_rows.append(dict(dataset=dataset, variant=variant, source='paper Table 2/4; not measured here',
                                   paper_ndcg5=n, paper_ndcg5_std=ns, paper_jaccard=j, paper_jaccard_std=js))
    pd.DataFrame(paper_rows).to_csv(root/'paper_reference.csv', index=False)
    comparisons = []
    for row in summaries:
        if row['model'] != 'SimpleDyG' or row['variant'] not in PAPER[row['dataset']]:
            continue
        reference = PAPER[row['dataset']][row['variant']]
        measured_ndcg = row['ndcg5_author_mean'] if cfg['protocol']=='author_compat' else row['ndcg5_mean']
        comparisons.append(dict(dataset=row['dataset'], variant=row['variant'], protocol=cfg['protocol'],
            profile=cfg['profile'], n_seeds=row['n_seeds'], measured_ndcg5=measured_ndcg,
            paper_ndcg5=reference[0], delta_ndcg5=measured_ndcg-reference[0],
            measured_jaccard=row['jaccard_mean'], paper_jaccard=reference[2],
            delta_jaccard=row['jaccard_mean']-reference[2],
            interpretation='descriptive gap only; profile/protocol may differ; no reproduction success threshold'))
    pd.DataFrame(comparisons, columns=['dataset', 'variant', 'protocol', 'profile', 'n_seeds',
        'measured_ndcg5', 'paper_ndcg5', 'delta_ndcg5', 'measured_jaccard', 'paper_jaccard',
        'delta_jaccard', 'interpretation']).to_csv(root/'paper_comparison.csv', index=False)
    paired = []
    if not df.empty:
        for dataset, g in df[df.model=='SimpleDyG'].groupby('dataset'):
            full = g[g.variant=='full'].set_index('seed')
            for variant in cfg['variants']:
                if variant == 'full':
                    continue
                ablated = g[g.variant==variant].set_index('seed')
                common = full.index.intersection(ablated.index)
                for metric in ['ndcg5', 'jaccard', 'jaccard_at5']:
                    delta = ablated.loc[common, metric]-full.loc[common, metric]
                    paired.append(dict(dataset=dataset, variant=variant, metric=metric, n_pairs=len(common),
                                       delta_mean=delta.mean(), delta_std=delta.std(ddof=1)))
    pd.DataFrame(paired, columns=['dataset', 'variant', 'metric', 'n_pairs', 'delta_mean', 'delta_std']).to_csv(root/'ablation_paired_deltas.csv', index=False)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if not summary.empty:
            for metric in ['ndcg5', 'jaccard_at5']:
                datasets = list(summary.dataset.unique())
                fig, axes = plt.subplots(1, len(datasets), figsize=(7*len(datasets), 4), squeeze=False)
                for ax, dataset in zip(axes[0], datasets):
                    sub = summary[summary.dataset==dataset]
                    labels = sub.model+' / '+sub.variant
                    ax.bar(range(len(sub)), sub[metric+'_mean'], yerr=sub[metric+'_std'].fillna(0), capsize=4)
                    ax.set_xticks(range(len(sub)), labels, rotation=35, ha='right')
                    ax.set_title(dataset+' | '+cfg['profile'])
                    ax.set_ylabel(metric)
                    ax.grid(axis='y', alpha=.2)
                fig.tight_layout()
                fig.savefig(root/f'{metric}.png', dpi=160)
                plt.close(fig)
    except ImportError:
        print('Charts skipped: matplotlib unavailable')
    notes = [f"Profile: {cfg['profile']}; protocol: {cfg['protocol']}.",
             f"Completed: {len(complete)}/{len(records)} runs. Expected seeds: {cfg['seeds']}.",
             'Smoke/subset runs only validate execution and cannot support scientific claims.',
             'NDCG5 and Jaccard@5 are the common comparison metrics; both deduplicate predictions/labels.',
             'SimpleDyG Jaccard uses generated set; baseline Jaccard uses k selected on validation.',
             'Baseline test max-over-k is a separate oracle reference; it is not used for model selection.',
             'Author NDCG preserves duplicates and may exceed 1. Values are explicitly marked ndcg5_author.',
             'Baseline history uses undirected raw edges; released sequence targets remain source-only.',
             'Baselines are BPR adaptations from DyGLib, not a claim to exactly reproduce Table 2 baseline scores.',
             'Pilot history limits and corrected padding/decoding differ from the original release.',
             'Ten seeds do not guarantee agreement with the paper. Check data scope and protocol before interpreting gaps.']
    (root/'READ_RESULTS.txt').write_text('\n'.join(notes), encoding='utf-8')
    archive = root.parent/(root.name+'.zip')
    if make_archive:
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in root.rglob('*'):
                if p.is_file() and (cfg['export_checkpoints'] or p.suffix != '.pt'):
                    z.write(p, p.relative_to(root))
    return df, summary, archive


def run_pipeline(author_root, dyglib_root, cfg, output_root):
    cfg = copy.deepcopy(cfg)
    source_files = {}
    for label, directory in [('author', Path(author_root)/'models'), ('dyglib', Path(dyglib_root))]:
        for file in sorted(directory.rglob('*.py')):
            source_files[label+'/'+str(file.relative_to(directory)).replace('\\', '/')] = sha256(file)
    cfg['_source_fingerprint'] = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    cfg['_pipeline_fingerprint'] = globals().get('PIPELINE_SOURCE_SHA256', sha256(__file__) if '__file__' in globals() else 'notebook')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg)
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('USE_TORCH', '1')
    os.environ.setdefault('MPLCONFIGDIR', str(root/'mpl_cache'))
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    datasets = {name: load_dataset(Path(author_root), name, cfg) for name in cfg['datasets']}
    dump_json(root/'data_audit.json', {n: d['audit'] for n, d in datasets.items()})
    dump_json(root/'config.json', cfg)
    dump_json(root/'source_hashes.json', source_files)
    environment = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                       cuda=torch.version.cuda, device=str(device),
                       gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                       author_expected_commit=AUTHOR_COMMIT, dyglib_expected_commit=DYGLIB_COMMIT)
    import boto3
    environment.update(boto3=boto3.__version__,
                       tokenizer_runtime='embedded NodeTokenizer; no Hugging Face/Rust dependency',
                       author_optimizer='embedded legacy AdamW arithmetic')
    dump_json(root/'environment.json', environment)
    (root/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    records = []
    for ds in datasets.values():
        for variant in cfg['variants']:
            for seed in cfg['seeds']:
                records.append(execute_run(ds, 'SimpleDyG', variant, seed, cfg, device, root, author_root, dyglib_root))
                export_results(records, cfg, root)
        for key in cfg['baseline_models']:
            for seed in cfg['seeds']:
                records.append(execute_run(ds, key, 'full', seed, cfg, device, root, author_root, dyglib_root))
                export_results(records, cfg, root)
    df, summary, archive = export_results(records, cfg, root)
    print('Outputs:', root, '\nArchive:', archive, flush=True)
    failed = [r for r in records if r['status']=='failed']
    if failed:
        raise RuntimeError(f"{len(failed)} runs failed; see run_manifest.json and each error.txt. No success claim was generated.")
    return df, summary, archive


def prediction_row(ego, pred, truth, **extra):
    if not truth:
        return dict(ego=ego, pred=pred, truth=[], **extra)
    m = metrics(pred, truth)
    return dict(ego=ego, pred=pred, truth=truth, **m,
                ndcg5_author=metrics(pred, truth, author=True)['ndcg5'],
                jaccard_at5=metrics(unique(pred)[:5], truth)['jaccard'], **extra)


def transform_time(tokens, variant, ego, seed, maximum=12):
    """Transforms labels only, preserving node chronology and split/target membership."""
    times = {f'<|time{i}|>' for i in range(maximum + 1)}
    if variant == 'full':
        return list(tokens)
    if variant == 'no_time':
        return [x for x in tokens if x not in times]
    if variant == 'same_time':
        return ['<|time0|>' if x in times else x for x in tokens]
    if variant == 'per_ego_permute':
        key = int(hashlib.sha256(f'{seed}:{ego}'.encode()).hexdigest()[:16], 16)
        labels = list(range(maximum + 1))
        random.Random(key).shuffle(labels)
        mapping = {f'<|time{i}|>': f'<|time{j}|>' for i, j in enumerate(labels)}
        return [mapping.get(x, x) for x in tokens]
    raise ValueError(variant)


class NodeTokenizer:
    """Matches the author's WordLevel node IDs and order of added special tokens."""
    def __init__(self, vocab, maximum):
        self.vocab = dict(vocab)
        self.nodes = set(vocab)
        if sorted(vocab.values()) != list(range(len(vocab))):
            raise ValueError('Node vocabulary is not contiguous')
        self.special = ['<|endoftext|>', '<|history|>', '<|endofhistory|>', '<|pre|>', '<|endofpre|>']
        self.special += [f'<|time{i}|>' for i in range(maximum + 1)] + ['[PAD]']
        for token in self.special:
            self.vocab[token] = len(self.vocab)
        self.inverse = {i: x for x, i in self.vocab.items()}
        self.pad = self.vocab['[PAD]']
        self.eos = self.vocab['<|endoftext|>']
        self.endpred = self.vocab['<|endofpre|>']

    def encode(self, tokens):
        return [self.vocab[x] for x in tokens]


def load_dataset(root, name, cfg):
    spec = SPECS[name]
    directory = Path(root) / 'resources' / name / str(spec['timestamp'])
    vpath = Path(root) / 'vocabs' / name / str(spec['timestamp']) / 'vocab.json'
    files = {f: directory / f'{f}.link_prediction' for f in ['train', 'val', 'val_gt', 'test', 'test_gt']}
    for p in [*files.values(), vpath]:
        if not p.is_file():
            raise FileNotFoundError(p)
    vocab = json.loads(vpath.read_text())
    data = {k: read_lines(p) for k, p in files.items()}
    raw_path = Path(root) / 'all_data' / spec['raw'] / str(spec['timestamp']) / f"ml_{spec['raw']}.csv"
    raw = pd.read_csv(raw_path).sort_values('ts', kind='stable').reset_index(drop=True)
    assert {'u', 'i', 'ts', 'timestamp', 'idx'} <= set(raw.columns)
    assert raw.timestamp.max() == spec['timestamp']
    assert raw[['u', 'i', 'ts', 'timestamp']].notna().all().all()
    assert set(raw.u.astype(str)) | set(raw.i.astype(str)) == set(vocab)
    bins = raw.groupby('timestamp').ts.agg(['min', 'max'])
    assert all(bins.iloc[i]['max'] < bins.iloc[i+1]['min'] for i in range(len(bins)-1))
    audit = {'dataset': name, 'nodes': len(vocab), 'edges': len(raw), 'splits': {}, 'files': {}}
    for k, p in files.items():
        audit['files'][str(p.relative_to(root)).replace('\\', '/')] = sha256(p)
        for tokens in data[k]:
            assert all(x in vocab or x.startswith('<|') for x in tokens)
    audit['files'][str(vpath.relative_to(root)).replace('\\', '/')] = sha256(vpath)
    audit['files'][str(raw_path.relative_to(root)).replace('\\', '/')] = sha256(raw_path)
    for split in ['val', 'test']:
        assert len(data[split]) == len(data[split + '_gt'])
        expected = spec['timestamp'] - (split == 'val')
        for inp, target in zip(data[split], data[split + '_gt']):
            assert inp[:2] == ['<|endoftext|>', '<|history|>']
            assert inp[-1] == '<|endofhistory|>'
            assert target[:2] == ['<|pre|>', f'<|time{expected}|>']
            hist_times = [int(x[6:-2]) for x in inp if x.startswith('<|time')]
            assert hist_times == sorted(hist_times) and max(hist_times, default=-1) < expected
            actual = raw[(raw.timestamp == expected) & (raw.u == int(inp[2]))].i.astype(str).tolist()
            assert Counter(actual) == Counter(x for x in target if x in vocab)
        lens = [len(x) for x in data[split]]
        audit['splits'][split] = dict(queries=len(lens), max_length=max(lens),
            above_context=sum(x > cfg['context'] for x in lens),
            duplicate_targets=sum(len([x for x in y if x in vocab]) != len(set(x for x in y if x in vocab)) for y in data[split+'_gt']))
    for line in data['train']:
        before, after = line[:line.index('<|pre|>')], line[line.index('<|pre|>'):]
        output_time = [int(x[6:-2]) for x in after if x.startswith('<|time')]
        assert len(output_time) == 1 and output_time[0] <= spec['timestamp'] - 2
        assert all(int(x[6:-2]) < output_time[0] for x in before if x.startswith('<|time'))
    train_lens = [len(x) for x in data['train']]
    audit['splits']['train'] = dict(queries=len(train_lens), max_length=max(train_lens),
                                  truncated=sum(x > cfg['train_block'] for x in train_lens))
    # The released queries score original source nodes only; do not silently symmetrize targets.
    audit['query_scope'] = 'released source-node queries; raw graph used as undirected history by baselines'
    audit['closed_world_vocabulary'] = 'all released node identities, including future identities; no future edges as history'
    if cfg['train_limit']:
        data['train'] = data['train'][:cfg['train_limit']]
    if cfg['eval_limit']:
        for split in ['val', 'test']:
            data[split] = data[split][:cfg['eval_limit']]
            data[split+'_gt'] = data[split+'_gt'][:cfg['eval_limit']]
    data.update(name=name, spec=spec, vocab=vocab, raw=raw, audit=audit)
    return data


def load_author_model(root):
    """Load released architecture; route its two HF activation imports to identical math."""
    namespace = 'simpledyg_author_models'
    if namespace not in sys.modules:
        path = Path(root) / 'models'
        module = types.ModuleType(namespace)
        module.__path__ = [str(path)]
        module.__package__ = namespace
        module.__version__ = '2.5.1'
        sys.modules[namespace] = module
        activation_module = types.ModuleType(namespace+'.runtime_activations')
        activation_module.gelu_new = gelu_new
        activation_module.get_activation = get_activation
        sys.modules[activation_module.__name__] = activation_module
        try:
            for stem in ['file_utils', 'configuration_utils', 'configuration_gpt2', 'modeling_utils', 'modeling_gpt2']:
                file = path/(stem+'.py')
                source = file.read_text(encoding='utf-8')
                source = source.replace('from transformers.activations import get_activation',
                                        'from .runtime_activations import get_activation')
                source = source.replace('from transformers.activations import gelu_new',
                                        'from .runtime_activations import gelu_new')
                name = namespace+'.'+stem
                child = types.ModuleType(name)
                child.__file__ = str(file)
                child.__package__ = namespace
                sys.modules[name] = child
                exec(compile(source, str(file), 'exec'), child.__dict__)
                setattr(module, stem, child)
            module.GPT2Config = sys.modules[namespace+'.configuration_gpt2'].GPT2Config
            module.GPT2LMHeadModel = sys.modules[namespace+'.modeling_gpt2'].GPT2LMHeadModel
        except Exception:
            for name in list(sys.modules):
                if name == namespace or name.startswith(namespace+'.'):
                    del sys.modules[name]
            raise
    module = sys.modules[namespace]
    return module.GPT2Config, module.GPT2LMHeadModel


def choose_device(cfg):
    requested = cfg['device']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else torch.device(requested)
    if device.type == 'cpu' and cfg['profile'] != 'smoke':
        raise RuntimeError('Full training needs Kaggle GPU; select smoke for a CPU check')
    return device


def amp_context(cfg, device):
    return torch.autocast(device_type='cuda', dtype=torch.float16) if cfg['amp'] and device.type == 'cuda' else contextlib.nullcontext()


def make_lm(root, ds, cfg):
    Config, Model = load_author_model(root)
    params = {k: ds['spec'][k] for k in ['layers', 'heads', 'dim']}
    if cfg['profile'] == 'smoke':
        params = dict(layers=2, heads=2, dim=64)
    tok = NodeTokenizer(ds['vocab'], ds['spec']['timestamp'])
    # Match original initialize-50257-then-resize RNG behavior for author_compat.
    c = Config(vocab_size=50257 if cfg['protocol'] == 'author_compat' else len(tok.vocab),
               n_positions=cfg['context'], n_ctx=cfg['context'], n_embd=params['dim'],
               n_layer=params['layers'], n_head=params['heads'])
    model = Model(c)
    model.resize_token_embeddings(len(tok.vocab))
    return model, tok, params


@torch.no_grad()
def evaluate_lm(model, tok, ds, split, variant, seed, cfg, device, output=None):
    model.eval()
    rows = []
    for inp, target in zip(ds[split], ds[split+'_gt']):
        check_deadline()
        ego = inp[2]
        truth = [x for x in target if x in tok.nodes and x != ego]
        tokens = transform_time(inp, variant, ego, seed, ds['spec']['timestamp'])
        ids = tok.encode(tokens)
        compat = cfg['protocol'] == 'author_compat'
        if compat:
            if len(ids) > cfg['context']:
                ids = ids[-1000:]
            limit = 11 if split == 'val' else max(1, cfg['context'] - (len(tok.special)-1) - len(ids))
            ids = ids[-(cfg['context']-1):]
        else:
            limit = cfg['max_new_tokens']
            ids = ids[-(cfg['context']-limit):]
        limit = min(limit, cfg['context']-len(ids))
        generated, past, ended = [], None, False
        tensor = torch.tensor([ids], device=device)
        for _ in range(limit):
            check_deadline()
            with amp_context(cfg, device):
                outputs = model(tensor, past=past)
            idx = int(outputs[0][0, -1].argmax().item())
            generated.append(idx)
            past = outputs[1]
            if idx == tok.eos or (not compat and idx == tok.endpred):
                ended = True
                break
            tensor = torch.tensor([[idx]], device=device)
        pred = [tok.inverse[x] for x in generated if tok.inverse[x] in tok.nodes and tok.inverse[x] != ego]
        rows.append(prediction_row(ego, pred, truth, ended=ended, generated_tokens=len(generated)))
    return summarize_predictions(rows, output)


def train_lm(root, ds, variant, seed, cfg, device, run_dir):
    seed_all(seed)
    model, tok, architecture = make_lm(root, ds, cfg)
    model.to(device)
    sequences = [tok.encode(transform_time(x, variant, x[2], seed, ds['spec']['timestamp']))[-cfg['train_block']:] for x in ds['train']]
    compat = cfg['protocol'] == 'author_compat'
    def collate(rows):
        padded = torch.full((len(rows), max(map(len, rows))), tok.pad, dtype=torch.long)
        for i, row in enumerate(rows):
            padded[i, :len(row)] = torch.tensor(row)
        return padded
    logical_batch = ds['spec']['batch'] if cfg['profile'] != 'smoke' else cfg['micro_batch']
    micro = min(logical_batch, cfg['micro_batch'])
    if logical_batch % micro:
        raise ValueError('micro_batch must divide the dataset logical batch size')
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(sequences, batch_size=logical_batch, shuffle=True, generator=generator,
                        collate_fn=collate, drop_last=compat)
    if len(loader) == 0:
        raise ValueError('No training batches')
    if compat:
        optimizer = AuthorAdamW(model.parameters(), lr=ds['spec']['lr'], eps=1e-8, weight_decay=0.)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=ds['spec']['lr'], eps=1e-8, weight_decay=0.)
    scheduler = linear_warmup_decay(optimizer, 0, len(loader)*cfg['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'] and device.type == 'cuda')
    best, best_epoch, stale, history = -float('inf'), 0, 0, []
    checkpoint = run_dir / 'best.pt'
    began = time.perf_counter()
    for epoch in range(1, cfg['epochs']+1):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            valid_total = (batch[:, 1:] != tok.pad).sum().item() if not compat else batch[:, 1:].numel()
            for part in batch.split(micro):
                inputs = part.to(device)
                labels = inputs.clone()
                if not compat:
                    labels[labels == tok.pad] = -100
                weight = ((labels[:, 1:] != -100).sum().item()) / valid_total
                with amp_context(cfg, device):
                    outputs = model(inputs, labels=labels, attention_mask=None if compat else inputs.ne(tok.pad))
                    loss = outputs[0]
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite LM loss')
                scaler.scale(loss * weight).backward()
                losses.append(float(loss.detach().cpu()))
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        validation = evaluate_lm(model, tok, ds, 'val', variant, seed, cfg, device)
        score = validation['ndcg5_author' if compat else 'ndcg5']
        improved = score >= best if compat else score > best + 1e-12
        if improved:
            best, best_epoch, stale = score, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), **validation))
        dump_json(run_dir/'history.json', history)
        print(f"{ds['name']} SimpleDyG/{variant} seed={seed} epoch={epoch} val_NDCG5={score:.4f} stale={stale}", flush=True)
        if stale >= cfg['patience']:
            break
    training_seconds = time.perf_counter()-began
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    validation = evaluate_lm(model, tok, ds, 'val', variant, seed, cfg, device, run_dir/'val_predictions.jsonl')
    test_began = time.perf_counter()
    test = evaluate_lm(model, tok, ds, 'test', variant, seed, cfg, device, run_dir/'test_predictions.jsonl')
    result = dict(model='SimpleDyG', variant=variant, seed=seed, architecture=architecture,
                  parameters=sum(p.numel() for p in model.parameters()), best_epoch=best_epoch,
                  epochs_run=len(history), training_seconds=training_seconds,
                  testing_seconds=time.perf_counter()-test_began, validation=validation, test=test,
                  tokenizer=tok.vocab)
    del model, optimizer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


"""Appended to the scientific core to support isolated workers and full resume."""
import argparse
import signal
import shutil
import sqlite3
import stat

_RUNTIME = dict(deadline=float('inf'), checkpoint_seconds=300., last_save=0.,
                pause_after_steps=None, worker_id=None, stop_file=None)


class TrainingPaused(Exception):
    pass


def runtime_expired():
    return (time.time() >= _RUNTIME['deadline'] or
            (_RUNTIME.get('stop_file') is not None and Path(_RUNTIME['stop_file']).exists()))


def check_deadline():
    if runtime_expired():
        raise TrainingPaused('Session budget reached; restore latest.pt next session.')


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(name=state[0],
                keys=torch.from_numpy(state[1].astype(np.int64)), position=state[2],
                gaussian=state[3], cached=state[4]), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['name'], n['keys'].numpy().astype(np.uint32),
                         n['position'], n['gaussian'], n['cached']))
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise RuntimeError('Checkpoint visible GPU count differs from worker.')
        torch.cuda.set_rng_state_all(state['cuda'])


def checkpoint_environment(device):
    return dict(python=sys.version.split()[0], torch=str(torch.__version__), numpy=np.__version__,
                accelerator=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu')


def new_training_state(identity, total_epochs):
    return dict(identity=identity, format_version=1, epoch=1, next_batch=0, phase='train',
                global_step=0, best=-float('inf'), best_epoch=0, stale=0, best_model=None,
                history=[], losses=[], total_epochs=total_epochs, elapsed_seconds=0.)


def load_training_state(path, identity, model, optimizer, device, scheduler=None, scaler=None):
    if not Path(path).exists():
        return None
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['format_version'] != 1 or saved['identity'] != identity:
        raise RuntimeError('Checkpoint identity/config mismatch: ' + str(path))
    if saved['environment'] != checkpoint_environment(device):
        raise RuntimeError('Checkpoint environment differs. Keep the previous Kaggle image/runtime: ' + str(path))
    model.load_state_dict(saved.pop('model'))
    optimizer.load_state_dict(saved.pop('optimizer'))
    if scheduler is not None:
        scheduler.load_state_dict(saved.pop('scheduler'))
    if scaler is not None:
        scaler.load_state_dict(saved.pop('scaler'))
    restore_rng(saved.pop('rng'))
    print(f"RESUME {identity}: epoch={saved['epoch']} next_batch={saved['next_batch']} phase={saved['phase']}", flush=True)
    return saved


def save_training_state(path, state, model, optimizer, device, scheduler=None, scaler=None,
                        force=False, carry=0., began=None):
    now = time.monotonic()
    if not force and now - _RUNTIME['last_save'] < _RUNTIME['checkpoint_seconds']:
        return
    saved = dict(state, model=model.state_dict(), optimizer=optimizer.state_dict(),
                 rng=rng_state(), environment=checkpoint_environment(device))
    if began is not None:
        saved['elapsed_seconds'] = carry + time.perf_counter() - began
    if scheduler is not None:
        saved['scheduler'] = scheduler.state_dict()
    if scaler is not None:
        saved['scaler'] = scaler.state_dict()
    atomic_torch_save(saved, path)
    dump_json(Path(path).with_suffix('.json'),
              dict(identity=state['identity'], epoch=state['epoch'], next_batch=state['next_batch'],
                   phase=state['phase'], global_step=state['global_step'], saved_at=time.time(),
                   worker_id=_RUNTIME['worker_id']))
    _RUNTIME['last_save'] = now


def maybe_pause(state, save):
    if runtime_expired() or (_RUNTIME['pause_after_steps'] is not None and
                             state['global_step'] >= _RUNTIME['pause_after_steps']):
        save(force=True)
        raise TrainingPaused('Checkpoint saved at completed optimizer step.')


def train_lm(root, ds, variant, seed, cfg, device, run_dir):
    seed_all(seed)
    model, tok, architecture = make_lm(root, ds, cfg)
    model.to(device)
    sequences = [tok.encode(transform_time(x, variant, x[2], seed, ds['spec']['timestamp']))[-cfg['train_block']:]
                 for x in ds['train']]
    compat = cfg['protocol'] == 'author_compat'

    def collate(rows):
        padded = torch.full((len(rows), max(map(len, rows))), tok.pad, dtype=torch.long)
        for i, row in enumerate(rows):
            padded[i, :len(row)] = torch.tensor(row)
        return padded

    logical_batch = ds['spec']['batch'] if cfg['profile'] != 'smoke' else cfg['micro_batch']
    micro = min(logical_batch, cfg['micro_batch'])
    if logical_batch % micro:
        raise ValueError('micro_batch must divide logical batch')
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(sequences, batch_size=logical_batch, shuffle=True, generator=generator,
                        collate_fn=collate, drop_last=compat)
    if not len(loader):
        raise ValueError('No training batches')
    optimizer = (AuthorAdamW if compat else torch.optim.AdamW)(model.parameters(),
                    lr=ds['spec']['lr'], eps=1e-8, weight_decay=0.)
    scheduler = linear_warmup_decay(optimizer, 0, len(loader) * cfg['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'] and device.type == 'cuda')
    identity = run_id(ds, 'SimpleDyG', variant, seed, cfg)
    latest = run_dir / 'latest.pt'
    state = load_training_state(latest, identity, model, optimizer, device, scheduler, scaler)
    if state is None:
        state = new_training_state(identity, cfg['epochs'])
        state['epoch_generator'] = generator.get_state()
    carry, began = state['elapsed_seconds'], time.perf_counter()

    def save(force=False):
        save_training_state(latest, state, model, optimizer, device, scheduler, scaler,
                            force=force, carry=carry, began=began)

    # A first checkpoint exists even if the session stops before the first batch.
    save(force=True)
    while state['epoch'] <= cfg['epochs'] and state['phase'] != 'done':
        if state['phase'] == 'train':
            model.train()
            # Replay only CPU collation up to the saved cursor. This recreates
            # the original DataLoader shuffle without repeating gradient steps.
            generator.set_state(state['epoch_generator'])
            for index, batch in enumerate(loader):
                if index < state['next_batch']:
                    continue
                maybe_pause(state, save)
                optimizer.zero_grad(set_to_none=True)
                valid_total = (batch[:, 1:] != tok.pad).sum().item() if not compat else batch[:, 1:].numel()
                for part in batch.split(micro):
                    inputs = part.to(device)
                    labels = inputs.clone()
                    if not compat:
                        labels[labels == tok.pad] = -100
                    weight = (labels[:, 1:] != -100).sum().item() / valid_total
                    with amp_context(cfg, device):
                        outputs = model(inputs, labels=labels,
                                        attention_mask=None if compat else inputs.ne(tok.pad))
                        loss = outputs[0]
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite LM loss')
                    scaler.scale(loss * weight).backward()
                    state['losses'].append(float(loss.detach().cpu()))
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                state['global_step'] += 1
                state['next_batch'] = index + 1
                save()
                maybe_pause(state, save)
            state['phase'] = 'validate'
            state['next_generator'] = generator.get_state()
            save(force=True)
        if state['phase'] == 'validate':
            validation = evaluate_lm(model, tok, ds, 'val', variant, seed, cfg, device)
            score = validation['ndcg5_author' if compat else 'ndcg5']
            improved = score >= state['best'] if compat else score > state['best'] + 1e-12
            if improved:
                state['best'], state['best_epoch'], state['stale'] = score, state['epoch'], 0
                state['best_model'] = cpu_state(model)
                atomic_torch_save(state['best_model'], run_dir / 'best.pt')
            else:
                state['stale'] += 1
            state['history'].append(dict(epoch=state['epoch'], train_loss=float(np.mean(state['losses'])), **validation))
            dump_json(run_dir / 'history.json', state['history'])
            print(f"{ds['name']} SimpleDyG/{variant} seed={seed} epoch={state['epoch']} "
                  f"val_NDCG5={score:.4f} stale={state['stale']}", flush=True)
            state['epoch'] += 1
            state['next_batch'], state['losses'] = 0, []
            state['epoch_generator'] = state.pop('next_generator')
            state['phase'] = 'done' if state['stale'] >= cfg['patience'] or state['epoch'] > cfg['epochs'] else 'train'
            save(force=True)
    model.load_state_dict(state['best_model'])
    atomic_torch_save(state['best_model'], run_dir / 'best.pt')
    training_seconds = carry + time.perf_counter() - began
    validation = evaluate_lm(model, tok, ds, 'val', variant, seed, cfg, device, run_dir / 'val_predictions.jsonl')
    test_began = time.perf_counter()
    test = evaluate_lm(model, tok, ds, 'test', variant, seed, cfg, device, run_dir / 'test_predictions.jsonl')
    return dict(model='SimpleDyG', variant=variant, seed=seed, architecture=architecture,
                parameters=sum(p.numel() for p in model.parameters()), best_epoch=state['best_epoch'],
                epochs_run=len(state['history']), training_seconds=training_seconds,
                testing_seconds=time.perf_counter() - test_began, validation=validation, test=test,
                tokenizer=tok.vocab, optimizer_steps=state['global_step'])


def train_ranker_candidate(key, dim, seed, ds, cfg, device, sampler, classes, run_dir):
    seed_all(seed)
    model = TemporalRanker(key, dim, ds, cfg, device, sampler, classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pairs, forbidden = baseline_training_edges(ds, cfg)
    rng = np.random.default_rng(seed)
    identity = run_id(ds, key, 'full', seed, cfg) + f':dim{dim}'
    latest = run_dir / f'latest_dim{dim}.pt'
    checkpoint = run_dir / f'candidate_dim{dim}.pt'
    done = run_dir / f'candidate_dim{dim}.json'
    if done.exists() and checkpoint.exists():
        receipt = json.loads(done.read_text())
        if receipt['identity'] != identity:
            raise RuntimeError('Completed baseline candidate identity differs')
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        return model, receipt['validation'], receipt['best_epoch'], receipt['epochs_run'], checkpoint, receipt['training_seconds']
    state = load_training_state(latest, identity, model, optimizer, device)
    if state is None:
        state = new_training_state(identity, cfg['baseline_epochs'])
    else:
        rng.bit_generator.state = state['negative_rng']
    carry, began = state['elapsed_seconds'], time.perf_counter()

    def save(force=False):
        state['negative_rng'] = copy.deepcopy(rng.bit_generator.state)
        save_training_state(latest, state, model, optimizer, device, force=force, carry=carry, began=began)

    save(force=True)
    while state['epoch'] <= cfg['baseline_epochs'] and state['phase'] != 'done':
        if state['phase'] == 'train':
            model.train()
            for index, start in enumerate(range(0, len(pairs), cfg['baseline_batch'])):
                if index < state['next_batch']:
                    continue
                maybe_pause(state, save)
                batch = pairs[start:start + cfg['baseline_batch']]
                src = np.array([r[0] for r in batch], dtype=np.int64)
                dst = np.array([r[1] for r in batch], dtype=np.int64)
                times = np.array([r[2] for r in batch], dtype=np.float64)
                negative = []
                for u, _, _, bin_id in batch:
                    blocked = forbidden[(u, bin_id)]
                    if len(blocked) >= len(ds['vocab']):
                        raise ValueError('No negative candidate')
                    item = int(rng.integers(1, len(ds['vocab']) + 1))
                    while item in blocked:
                        item = int(rng.integers(1, len(ds['vocab']) + 1))
                    negative.append(item)
                optimizer.zero_grad(set_to_none=True)
                negatives = np.array(negative)
                batch_loss = 0.
                # Keep the logical event batch and one optimizer step. Smaller
                # forward batches make GraphMixer's 1024-token history fit T4.
                for sub_start in range(0, len(batch), cfg['baseline_micro_batch']):
                    part = slice(sub_start, sub_start + cfg['baseline_micro_batch'])
                    a, b, n, t = src[part], dst[part], negatives[part], times[part]
                    scores = model(np.concatenate([a, a]), np.concatenate([b, n]), np.concatenate([t, t]))
                    positive, negative_score = scores.chunk(2)
                    loss = torch.nn.functional.softplus(negative_score - positive).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite BPR loss')
                    weight = len(a) / len(batch)
                    (loss * weight).backward()
                    batch_loss += float(loss.detach().cpu()) * weight
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                state['losses'].append(batch_loss)
                state['global_step'] += 1
                state['next_batch'] = index + 1
                save()
                maybe_pause(state, save)
            state['phase'] = 'validate'
            save(force=True)
        if state['phase'] == 'validate':
            validation = evaluate_ranker(model, ds, 'val', cfg, device)
            score = validation['ndcg5']
            if score > state['best'] + 1e-12:
                state['best'], state['best_epoch'], state['stale'] = score, state['epoch'], 0
                state['best_model'] = cpu_state(model)
                atomic_torch_save(state['best_model'], checkpoint)
            else:
                state['stale'] += 1
            state['history'].append(dict(epoch=state['epoch'], train_loss=float(np.mean(state['losses'])), **validation))
            dump_json(run_dir / f'history_dim{dim}.json', state['history'])
            print(f"{ds['name']} {key} d={dim} seed={seed} epoch={state['epoch']} "
                  f"val_NDCG5={score:.4f} stale={state['stale']}", flush=True)
            state['epoch'] += 1
            state['next_batch'], state['losses'] = 0, []
            state['phase'] = 'done' if state['stale'] >= cfg['patience'] or state['epoch'] > cfg['baseline_epochs'] else 'train'
            save(force=True)
    model.load_state_dict(state['best_model'])
    atomic_torch_save(state['best_model'], checkpoint)
    validation = evaluate_ranker(model, ds, 'val', cfg, device)
    seconds = carry + time.perf_counter() - began
    receipt = dict(identity=identity, validation=validation, best_epoch=state['best_epoch'],
                   epochs_run=len(state['history']), training_seconds=seconds)
    dump_json(done, receipt)
    # Candidate completion is committed; a later interruption can skip its optimizer work.
    latest.unlink(missing_ok=True)
    latest.with_suffix('.json').unlink(missing_ok=True)
    return model, validation, state['best_epoch'], len(state['history']), checkpoint, seconds


def train_ranker(key, seed, ds, cfg, device, dyglib, run_dir):
    classes, Sampler = load_dyglib(dyglib)
    sampler = build_sampler(ds, cfg, Sampler)
    choices = []
    for dim in cfg['baseline_dims']:
        check_deadline()
        model, validation, epoch, epochs_run, checkpoint, seconds = train_ranker_candidate(
            key, dim, seed, ds, cfg, device, sampler, classes, run_dir)
        choices.append(dict(dim=dim, validation=validation, best_epoch=epoch,
                            epochs_run=epochs_run, checkpoint=str(checkpoint), training_seconds=seconds))
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    winner = max(choices, key=lambda c: (c['validation']['ndcg5'], -c['dim']))
    dump_json(run_dir / 'validation_search.json', choices)
    model = TemporalRanker(key, winner['dim'], ds, cfg, device, sampler, classes)
    model.load_state_dict(torch.load(winner['checkpoint'], map_location=device, weights_only=True))
    validation = evaluate_ranker(model, ds, 'val', cfg, device, output=run_dir / 'val_predictions.jsonl')
    testing = time.perf_counter()
    test = evaluate_ranker(model, ds, 'test', cfg, device, k_choice=validation['selected_k'],
                           output=run_dir / 'test_predictions.jsonl')
    atomic_torch_save(cpu_state(model), run_dir / 'best.pt')
    return dict(model=key, variant='full', seed=seed, architecture={'dim': winner['dim']},
                best_epoch=winner['best_epoch'], epochs_run=winner['epochs_run'],
                parameters=sum(p.numel() for p in model.parameters()), validation=validation, test=test,
                training_seconds=sum(c['training_seconds'] for c in choices),
                testing_seconds=time.perf_counter() - testing,
                feature_policy='all-zero node/edge features; dimensions selected on validation',
                training_task='chronological event BPR, both edge directions; frozen next-bin ranking at evaluation')


def execute_run(ds, model, variant, seed, cfg, device, output_root, author_root, dyglib_root):
    identifier = run_id(ds, model, variant, seed, cfg)
    directory = Path(output_root) / 'runs' / identifier
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'result.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous['status'] == 'complete':
            return previous
    dump_json(directory / 'config.json', cfg)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        if model == 'SimpleDyG':
            result = train_lm(author_root, ds, variant, seed, cfg, device, directory)
        else:
            result = train_ranker(model, seed, ds, cfg, device, dyglib_root, directory)
        result.update(status='complete', run_id=identifier, dataset=ds['name'],
                      protocol=cfg['protocol'], profile=cfg['profile'],
                      truncated_data=bool(cfg['train_limit'] or cfg['eval_limit'] or cfg['edge_limit']),
                      peak_gpu_mb=float(torch.cuda.max_memory_allocated(device) / 1e6) if device.type == 'cuda' else None,
                      worker_id=_RUNTIME['worker_id'], physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      checkpoint_format=1)
    except TrainingPaused as exc:
        result = dict(status='paused', run_id=identifier, dataset=ds['name'], model=model,
                      variant=variant, seed=seed, reason=str(exc), protocol=cfg['protocol'], profile=cfg['profile'])
    except Exception as exc:
        import traceback
        (directory / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        result = dict(status='failed', run_id=identifier, dataset=ds['name'], model=model,
                      variant=variant, seed=seed, error=str(exc), protocol=cfg['protocol'], profile=cfg['profile'])
        print('FAILED:', identifier, str(exc), flush=True)
    dump_json(result_path, result)
    if result['status'] == 'complete':
        # Delete only this run's temporary training states, after result commit.
        for pattern in ['latest*.pt', 'latest*.json', 'candidate_dim*.pt']:
            for path in directory.glob(pattern):
                if path.resolve().is_relative_to(directory.resolve()):
                    path.unlink()
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def prepare_config(author_root, dyglib_root, cfg):
    cfg = copy.deepcopy(cfg)
    files = {}
    for label, folder in [('author', Path(author_root) / 'models'), ('dyglib', Path(dyglib_root))]:
        for path in sorted(folder.rglob('*.py')):
            files[label + '/' + path.relative_to(folder).as_posix()] = sha256(path)
    cfg['_source_fingerprint'] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    cfg['_pipeline_fingerprint'] = hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest()
    return cfg, files


def dual_plan(datasets, cfg):
    rows = []
    for ds in datasets.values():
        tasks = [('SimpleDyG', variant, seed) for variant in cfg['variants'] for seed in cfg['seeds']]
        tasks += [(model, 'full', seed) for model in cfg['baseline_models'] for seed in cfg['seeds']]
        for model, variant, seed in tasks:
            rows.append(dict(dataset=ds['name'], model=model, variant=variant, seed=seed,
                             run_id=run_id(ds, model, variant, seed, cfg)))
    return rows


def read_records(root, plan):
    records = []
    for row in plan:
        path = Path(root) / 'runs' / row['run_id'] / 'result.json'
        if path.exists():
            record = json.loads(path.read_text())
            if record['run_id'] != row['run_id']:
                raise RuntimeError('Result run ID mismatch: ' + str(path))
            records.append(record)
        else:
            records.append(dict(row, status='pending'))
    return records


def initialize_queue(database, root, plan):
    with contextlib.closing(sqlite3.connect(database, timeout=60)) as connection, connection:
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('CREATE TABLE IF NOT EXISTS jobs '
                           '(run_id TEXT PRIMARY KEY, ordinal INTEGER, payload TEXT, status TEXT, worker INTEGER, priority INTEGER)')
        connection.execute('DELETE FROM jobs')
        for ordinal, row in enumerate(plan):
            path = Path(root) / 'runs' / row['run_id']
            result = json.loads((path / 'result.json').read_text()) if (path / 'result.json').exists() else {}
            complete = result.get('status') == 'complete'
            priority = int(any(path.glob('latest*.pt')) or any(path.glob('candidate_dim*.json')))
            connection.execute('INSERT INTO jobs VALUES (?, ?, ?, ?, NULL, ?)',
                               (row['run_id'], ordinal, json.dumps(row), 'complete' if complete else 'queued', priority))


def claim_job(database, worker_id):
    with contextlib.closing(sqlite3.connect(database, timeout=60, isolation_level=None)) as connection:
        connection.execute('BEGIN IMMEDIATE')
        row = connection.execute("SELECT run_id, payload FROM jobs WHERE status='queued' "
                                 'ORDER BY priority DESC, ordinal LIMIT 1').fetchone()
        if row is not None:
            connection.execute("UPDATE jobs SET status='running', worker=? WHERE run_id=?", (worker_id, row[0]))
        connection.execute('COMMIT')
        return json.loads(row[1]) if row else None


def worker_environment(gpu_id=None):
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', USE_TF='0',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    if gpu_id is None:
        env['CUDA_VISIBLE_DEVICES'] = ''
    else:
        existing = os.environ.get('CUDA_VISIBLE_DEVICES')
        devices = existing.split(',') if existing else None
        env['CUDA_VISIBLE_DEVICES'] = devices[gpu_id] if devices else str(gpu_id)
    return env


def dual_worker(job_file, worker_id):
    job = json.loads(Path(job_file).read_text())
    root, cfg = Path(job['output']), job['cfg']
    torch.set_num_threads(job['worker_threads'])
    torch.use_deterministic_algorithms(True)
    if job['cpu_workers']:
        if cfg['profile'] != 'smoke':
            raise RuntimeError('CPU workers only validate smoke')
        device = torch.device('cpu')
    else:
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Each GPU worker must see exactly one GPU')
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    _RUNTIME.update(deadline=job['deadline'], checkpoint_seconds=job['checkpoint_seconds'],
                    last_save=0., pause_after_steps=job.get('pause_after_steps'), worker_id=worker_id,
                    stop_file=job['stop_file'])
    env = checkpoint_environment(device)
    dump_json(root / f'worker_{worker_id}.json', dict(env, pid=os.getpid(), session_id=job['session_id'], state='ready',
                    physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), ready_at=time.time()))
    while not Path(job['gate']).exists():
        if runtime_expired():
            return
        time.sleep(.2)
    datasets = {}
    while not runtime_expired():
        row = claim_job(job['database'], worker_id)
        if row is None:
            break
        print('CLAIM', row['run_id'], 'device=', str(device), 'physical_gpu=', os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True)
        if row['dataset'] not in datasets:
            datasets[row['dataset']] = load_dataset(Path(job['author']), row['dataset'], cfg)
        result = execute_run(datasets[row['dataset']], row['model'], row['variant'], row['seed'],
                             cfg, device, root, Path(job['author']), Path(job['dyglib']))
        with contextlib.closing(sqlite3.connect(job['database'], timeout=60)) as connection, connection:
            connection.execute('UPDATE jobs SET status=? WHERE run_id=?', (result['status'], row['run_id']))
        if result['status'] == 'paused':
            break
    dump_json(root / f'worker_{worker_id}.json', dict(env, pid=os.getpid(), session_id=job['session_id'], state='stopped',
                    physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), stopped_at=time.time()))


def create_resume_snapshot(root, destination, include_completed_best=False):
    """Store checkpoint bytes without expensive recompression; commit atomically."""
    root, destination = Path(root), Path(destination)
    temporary = destination.with_name(destination.name + '.writing')
    statuses = {p.parent.name: json.loads(p.read_text()).get('status')
                for p in (root / 'runs').glob('*/result.json')}
    members = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        if path.resolve() == destination.resolve():
            continue
        if path.name == 'SNAPSHOT_MANIFEST.json':
            continue
        relative = path.relative_to(root)
        if path.name.endswith(('.tmp', '.writing', '.lock', '.sqlite', '.sqlite-wal', '.sqlite-shm')):
            continue
        if (path.name.startswith('session_') and not path.name.startswith('session_history_')) or path.name.startswith(('gate_', 'stop_')):
            continue
        if path.suffix == '.pt' and len(relative.parts) >= 3 and relative.parts[0] == 'runs':
            if statuses.get(relative.parts[1]) == 'complete' and not include_completed_best:
                continue
        members.append((path, relative.as_posix()))
    needed = sum(path.stat().st_size for path, _ in members)
    if needed > shutil.disk_usage(destination.parent).free:
        raise RuntimeError('Not enough space for resume ZIP; preserve the full Output directory instead.')
    inventory = []
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path, relative in members:
            archive.write(path, relative)
            inventory.append(dict(path=relative, bytes=path.stat().st_size))
        archive.writestr('SNAPSHOT_MANIFEST.json', json.dumps(dict(format_version=1,
            purpose='resume full training states and reuse completed metrics',
            include_completed_best=include_completed_best, members=inventory), indent=2))
    temporary.replace(destination)
    print('Resume ZIP:', destination, '| bytes:', destination.stat().st_size, flush=True)
    return destination


def restore_dual_output(source, destination):
    source, destination = Path(source), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    def safe_target(name):
        if '\\' in name or ':' in name or '..' in name.split('/') or name.startswith('/'):
            raise ValueError('Unsafe snapshot path: ' + name)
        target = (destination / name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError('Snapshot path outside output')
        return target

    def copy_file(target, stream):
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.restoring')
        with temporary.open('wb') as handle:
            shutil.copyfileobj(stream, handle, 1024 * 1024)
        temporary.replace(target)

    def verify_marker(config):
        if config.get('execution_format') != 'dual_resume_v1':
            raise RuntimeError('This is an older one-GPU output. Start fresh with the new runtime.')
        existing = destination / 'config.json'
        if existing.exists() and json.loads(existing.read_text()) != config:
            raise RuntimeError('Output folder has a different experiment configuration')

    if source.is_dir():
        verify_marker(json.loads((source / 'config.json').read_text()))
        if source.resolve() == destination:
            return destination
        if destination.is_relative_to(source.resolve()):
            raise ValueError('Output destination is inside source')
        for path in source.rglob('*'):
            if path.is_symlink():
                raise ValueError('Snapshot source contains a symlink')
            if path.is_file() and path.name != 'resume_snapshot.zip' and not path.name.endswith(('.tmp', '.writing')):
                with path.open('rb') as handle:
                    copy_file(safe_target(path.relative_to(source).as_posix()), handle)
    else:
        with zipfile.ZipFile(source) as archive:
            names = set(archive.namelist())
            configs = [n for n in names if n.endswith('config.json') and
                       n[:-len('config.json')] + 'dual_runtime.json' in names]
            if len(configs) != 1:
                raise RuntimeError('Choose the new resume_snapshot.zip or its extracted output folder.')
            prefix = configs[0][:-len('config.json')]
            verify_marker(json.loads(archive.read(configs[0])))
            for info in archive.infolist():
                if not info.filename.startswith(prefix) or info.is_dir():
                    continue
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('Snapshot contains a symlink')
                target = safe_target(info.filename[len(prefix):])
                with archive.open(info) as handle:
                    copy_file(target, handle)
    return destination


def find_dual_output(search_root):
    folders = [p.parent for p in Path(search_root).rglob('dual_runtime.json')
               if (p.parent / 'config.json').is_file()]
    if len(folders) == 1:
        return folders[0]
    if len(folders) > 1:
        raise RuntimeError('Multiple output versions: set RESUME_FROM explicitly.')
    archives = list(Path(search_root).rglob('resume_snapshot.zip'))
    if len(archives) != 1:
        raise RuntimeError('Add the latest dual-GPU Output or resume_snapshot.zip as Input.')
    return archives[0]


def run_dual_pipeline(author_root, dyglib_root, cfg, output_root, gpu_ids=(0, 1),
                      session_hours=10.5, checkpoint_seconds=300., worker_threads=2,
                      include_completed_best=False, cpu_workers=0, pause_after_steps=None):
    from filelock import FileLock
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / 'coordinator.lock'), timeout=0):
        return _run_dual_locked(author_root, dyglib_root, cfg, root, gpu_ids, session_hours,
                                checkpoint_seconds, worker_threads, include_completed_best,
                                cpu_workers, pause_after_steps)


def _run_dual_locked(author_root, dyglib_root, cfg, root, gpu_ids, session_hours,
                     checkpoint_seconds, worker_threads, include_completed_best,
                     cpu_workers, pause_after_steps):
    if session_hours <= 0 or checkpoint_seconds <= 0 or worker_threads < 1:
        raise ValueError('Invalid session/checkpoint limits')
    if cpu_workers:
        if cfg['profile'] != 'smoke' or cpu_workers != 2:
            raise ValueError('CPU validation requires exactly two smoke workers')
        devices = [None] * cpu_workers
    else:
        if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or any(i < 0 or i >= torch.cuda.device_count() for i in gpu_ids):
            raise RuntimeError('Select Kaggle GPU T4 x2; two distinct visible GPUs are required.')
        devices = list(gpu_ids)
    start = time.time()
    os.environ.setdefault('MPLCONFIGDIR', str(root / 'mpl_cache'))
    cfg, sources = prepare_config(Path(author_root), Path(dyglib_root), cfg)
    existing = root / 'config.json'
    if existing.exists() and json.loads(existing.read_text()) != cfg:
        raise RuntimeError('Existing output has different config/core/source. Do not mix experiments.')
    environment = dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
                       pandas=pd.__version__, cuda=torch.version.cuda,
                       gpu_names=[torch.cuda.get_device_name(i) for i in devices] if not cpu_workers else ['cpu', 'cpu'])
    if (root / 'environment.json').exists():
        previous_environment = json.loads((root / 'environment.json').read_text())
        if previous_environment != environment:
            raise RuntimeError('Resume environment differs. Keep the previous PyTorch image and GPU type.')
    datasets = {n: load_dataset(author_root, n, cfg) for n in cfg['datasets']}
    audits = {n: ds['audit'] for n, ds in datasets.items()}
    if (root / 'data_audit.json').exists():
        previous = json.loads((root / 'data_audit.json').read_text())
        if any(previous[n]['files'] != audits[n]['files'] for n in audits):
            raise RuntimeError('Resume data hash differs')
    dump_json(root / 'config.json', cfg)
    dump_json(root / 'source_hashes.json', sources)
    dump_json(root / 'data_audit.json', audits)
    dump_json(root / 'dual_runtime.json', dict(format_version=1, parallelism='independent runs',
                workers=2, physical_devices=devices, session_hours=session_hours,
                checkpoint_seconds=checkpoint_seconds, started_at=start,
                include_completed_best=include_completed_best))
    dump_json(root / 'environment.json', environment)
    (root / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    plan = dual_plan(datasets, cfg)
    dump_json(root / 'experiment_plan.json', plan)
    database = root / 'jobs.sqlite'
    initialize_queue(database, root, plan)
    session = str(time.time_ns())
    gate, stop_file = root / f'gate_{session}', root / f'stop_{session}'
    job_file = root / f'session_{session}.json'
    job = dict(output=str(root), cfg=cfg, author=str(Path(author_root).resolve()),
               dyglib=str(Path(dyglib_root).resolve()), database=str(database), gate=str(gate),
               stop_file=str(stop_file), deadline=start + session_hours * 3600,
               checkpoint_seconds=checkpoint_seconds, worker_threads=worker_threads,
               cpu_workers=cpu_workers, pause_after_steps=pause_after_steps, session_id=session)
    dump_json(job_file, job)
    workers, handles, offsets = [], [], [0, 0]
    last_export, previous_status = 0., None
    session_history = dict(started_at=start, devices=devices, worker_pids=[], gate_opened_at=None,
                           stopped_at=None, status='running', cfg_fingerprint=cfg['_pipeline_fingerprint'])

    def show_logs():
        for index in range(2):
            path = root / f'worker_{index}.log'
            if path.exists():
                with path.open('r', encoding='utf-8', errors='replace') as handle:
                    handle.seek(offsets[index])
                    lines = handle.read()
                    offsets[index] = handle.tell()
                for line in lines.splitlines():
                    print(f'[worker {index} / device {devices[index]}] {line}', flush=True)

    try:
        for index, device in enumerate(devices):
            log = (root / f'worker_{index}.log').open('w', encoding='utf-8')
            handles.append(log)
            flags = dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name == 'nt' else {}
            worker = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                        '--worker-job', str(job_file), '--worker-id', str(index)],
                                       env=worker_environment(device), stdout=log, stderr=subprocess.STDOUT, **flags)
            workers.append(worker)
            session_history['worker_pids'].append(worker.pid)
        ready_deadline = min(job['deadline'], time.time() + 180)
        while True:
            show_logs()
            ready = [json.loads((root / f'worker_{i}.json').read_text())
                     if (root / f'worker_{i}.json').exists() else {} for i in range(2)]
            if all(r.get('session_id') == session and r.get('state') == 'ready' for r in ready):
                gate.touch()
                session_history['actual_worker_pids'] = [r['pid'] for r in ready]
                session_history['gate_opened_at'] = time.time()
                print('Two workers READY:', ready, flush=True)
                break
            if any(w.poll() is not None for w in workers) or time.time() >= ready_deadline:
                raise RuntimeError('Worker startup failed; see worker_0.log and worker_1.log')
            time.sleep(.5)
        while any(w.poll() is None for w in workers):
            show_logs()
            records = read_records(root, plan)
            statuses = tuple((r['run_id'], r['status']) for r in records)
            if statuses != previous_status and time.time() - last_export > 10:
                export_results(records, cfg, root, make_archive=False)
                previous_status, last_export = statuses, time.time()
                counts = dict(Counter(r['status'] for r in records))
                print('Progress:', counts, '| total:', len(plan), flush=True)
            time.sleep(.5 if cfg['profile'] == 'smoke' else 3)
    except BaseException:
        stop_file.touch()
        # Workers save on the next batch/evaluation boundary. Preserve existing
        # states if a worker hangs; never wait until the external 12h deadline.
        for worker in workers:
            try:
                worker.wait(timeout=60)
            except subprocess.TimeoutExpired:
                worker.terminate()
                worker.wait(timeout=15)
        session_history['status'] = 'interrupted'
        raise
    finally:
        for handle in handles:
            handle.close()
        show_logs()
        session_history['stopped_at'] = time.time()
        if session_history['status'] == 'running':
            session_history['status'] = 'finished'
        dump_json(root / f'session_history_{session}.json', session_history)
        records = read_records(root, plan)
        export_results(records, cfg, root, make_archive=False)
        snapshot = create_resume_snapshot(root, root / 'resume_snapshot.zip', include_completed_best)
    failed_workers = [w.returncode for w in workers if w.returncode != 0]
    if failed_workers or any(r['status'] == 'failed' for r in records):
        raise RuntimeError('Some runs/workers failed; checkpoints and ZIP retained. Inspect worker logs/error.txt.')
    df, summary, _ = export_results(records, cfg, root, make_archive=False)
    print('SESSION SAVED:', dict(Counter(r['status'] for r in records)), '| resume ZIP:', snapshot, flush=True)
    return df, summary, snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-job', required=True)
    parser.add_argument('--worker-id', type=int, required=True)
    arguments = parser.parse_args()
    dual_worker(arguments.worker_job, arguments.worker_id)
