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
