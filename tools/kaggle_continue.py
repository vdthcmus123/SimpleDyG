"""Restore completed SimpleDyG runs without changing training config or run IDs.

This is deliberately separate from kaggle_pipeline.py: its embedded fingerprint
must stay identical to the notebook that produced the previous results.
"""
import hashlib
import json
import shutil
import stat
import time
import zipfile
from pathlib import Path, PurePosixPath


def _recovery_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _recovery_root(path):
    path = Path(path)
    return all((path / x).is_file() for x in
               ['config.json', 'data_audit.json', 'source_hashes.json']) and (path / 'runs').is_dir()


def _recovery_zip_prefixes(archive):
    names = set(archive.namelist())
    prefixes = []
    for name in names:
        if PurePosixPath(name).name != 'config.json':
            continue
        prefix = name[:-len('config.json')]
        if all(prefix + x in names for x in ['data_audit.json', 'source_hashes.json']):
            if any(x.startswith(prefix + 'runs/') for x in names):
                prefixes.append(prefix)
    return sorted(prefixes)


def find_previous_output(search_root):
    """Prefer a saved output directory over a ZIP snapshot of that directory."""
    root = Path(search_root)
    folders = ([root] if _recovery_root(root) else [])
    folders += [p.parent for p in root.rglob('source_hashes.json') if _recovery_root(p.parent)]
    folders = sorted(set(folders))
    if len(folders) == 1:
        return folders[0]
    if len(folders) > 1:
        raise RuntimeError('Có nhiều Output cũ. Điền PREVIOUS_OUTPUT_PATH: ' + str(folders))
    archives = []
    for path in root.rglob('*.zip'):
        try:
            with zipfile.ZipFile(path) as archive:
                if _recovery_zip_prefixes(archive):
                    archives.append(path)
        except zipfile.BadZipFile:
            continue
    if len(archives) != 1:
        raise RuntimeError('Cần Add Input Output cũ hoặc ZIP kết quả, rồi điền '
                           'PREVIOUS_OUTPUT_PATH. ZIP phù hợp tìm thấy: ' + str(archives))
    return archives[0]


def restore_previous_output(source, destination):
    """Copy data only, with safe ZIP paths and atomic writes; preserve newer files."""
    source, destination = Path(source), Path(destination).resolve()
    if source.is_dir() and source.resolve() == destination:
        if not _recovery_root(destination):
            raise RuntimeError('Output hiện tại thiếu metadata hoặc thư mục runs.')
        return destination
    if not source.exists():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0

    def target_for(name):
        if '\\' in name or ':' in name or name.startswith('/') or '..' in name.split('/'):
            raise ValueError('ZIP path không hợp lệ: ' + name)
        target = (destination / Path(*PurePosixPath(name).parts)).resolve()
        if not target.is_relative_to(destination):
            raise ValueError('Output path nằm ngoài thư mục đích: ' + name)
        return target

    def check_config(saved):
        target = destination / 'config.json'
        if target.exists() and _recovery_json(target) != saved:
            raise RuntimeError('Thư mục đích đã có cấu hình khác; chọn OUTPUT mới.')

    def copy_stream(target, stream):
        nonlocal copied
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.restoring')
        with temporary.open('wb') as handle:
            shutil.copyfileobj(stream, handle, length=1024 * 1024)
        temporary.replace(target)
        copied += 1

    if source.is_dir():
        if not _recovery_root(source):
            raise RuntimeError('Chọn thư mục chứa config.json, data_audit.json và runs/.')
        if destination.is_relative_to(source.resolve()):
            raise ValueError('Thư mục đích không được nằm bên trong Output nguồn.')
        check_config(_recovery_json(source / 'config.json'))
        for path in source.rglob('*'):
            if path.is_symlink():
                raise ValueError('Không sao chép symlink từ Output.')
            if path.is_file():
                with path.open('rb') as stream:
                    copy_stream(target_for(path.relative_to(source).as_posix()), stream)
    else:
        with zipfile.ZipFile(source) as archive:
            prefixes = _recovery_zip_prefixes(archive)
            if len(prefixes) != 1:
                raise RuntimeError('ZIP phải chứa đúng một Output SimpleDyG. Nếu ZIP '
                                   'chỉ chứa ZIP con, hãy giải nén ZIP kết quả bên trong trước.')
            prefix = prefixes[0]
            saved = json.loads(archive.read(prefix + 'config.json'))
            check_config(saved)
            selected = [i for i in archive.infolist() if i.filename.startswith(prefix)
                        and not i.is_dir()]
            targets = [target_for(i.filename[len(prefix):]) for i in selected]
            if len(set(targets)) != len(targets):
                raise ValueError('ZIP chứa file trùng đường dẫn.')
            needed = sum(i.file_size for i, target in zip(selected, targets) if not target.exists())
            if needed > shutil.disk_usage(destination).free:
                raise RuntimeError(f'Không đủ dung lượng giải nén: cần thêm {needed / 2**30:.2f} GiB.')
            for info, target in zip(selected, targets):
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('ZIP chứa symlink: ' + info.filename)
                with archive.open(info) as stream:
                    copy_stream(target, stream)
    if not _recovery_root(destination):
        raise RuntimeError('Output khôi phục chưa đầy đủ.')
    print(f'Đã khôi phục {copied} file vào {destination}; file đang có được giữ lại.')
    return destination


def verify_previous_output(pipeline, author_root, dyglib_root, output):
    """Require identical core, source, data and full config before reusing results."""
    output = Path(output)
    cfg = _recovery_json(output / 'config.json')
    if cfg.get('_pipeline_fingerprint') != pipeline['PIPELINE_SOURCE_SHA256']:
        raise RuntimeError('Core khác notebook cũ. Dùng đúng notebook/core đã tạo Output.')
    source_files = {}
    for label, directory in [('author', Path(author_root) / 'models'), ('dyglib', Path(dyglib_root))]:
        for file in sorted(directory.rglob('*.py')):
            source_files[label + '/' + file.relative_to(directory).as_posix()] = pipeline['sha256'](file)
    expected_sources = _recovery_json(output / 'source_hashes.json')
    if source_files != expected_sources:
        changed = sorted(k for k in set(source_files) | set(expected_sources)
                         if source_files.get(k) != expected_sources.get(k))
        raise RuntimeError('Source khác phiên cũ: ' + str(changed[:12]) +
                           '. Giữ đúng input repo cũ; chưa bắt đầu training.')
    fingerprint = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    if fingerprint != cfg.get('_source_fingerprint'):
        raise RuntimeError('Fingerprint source không khớp metadata cũ.')
    datasets = {name: pipeline['load_dataset'](author_root, name, cfg) for name in cfg['datasets']}
    audits = _recovery_json(output / 'data_audit.json')
    for name, ds in datasets.items():
        if ds['audit']['files'] != audits[name]['files']:
            raise RuntimeError('Dữ liệu khác phiên cũ: ' + name)
    plan = []
    for ds in datasets.values():
        tasks = [('SimpleDyG', v, s) for v in cfg['variants'] for s in cfg['seeds']]
        tasks += [(m, 'full', s) for m in cfg['baseline_models'] for s in cfg['seeds']]
        for model, variant, seed in tasks:
            identifier = pipeline['run_id'](ds, model, variant, seed, cfg)
            directory = output / 'runs' / identifier
            previous = _recovery_json(directory / 'result.json') if (directory / 'result.json').exists() else None
            status = previous['status'] if previous else ('interrupted' if directory.exists() else 'pending')
            if previous and previous.get('run_id') != identifier:
                raise RuntimeError('result.json có run_id không khớp: ' + identifier)
            plan.append(dict(dataset=ds['name'], model=model, variant=variant,
                             seed=seed, run_id=identifier, status=status))
    expected = {row['run_id'] for row in plan}
    unknown = [p.parent.name for p in (output / 'runs').glob('*/result.json')
               if p.parent.name not in expected]
    if unknown:
        raise RuntimeError('Có run ngoài cấu hình hiện tại; chưa tự gộp: ' + str(unknown[:8]))
    return cfg, datasets, plan


def continue_previous_output(pipeline, author_root, dyglib_root, cfg, datasets, plan, output,
                             session_datasets=None, session_models=None, session_seeds=None,
                             max_new_runs=2, max_session_hours=10):
    """Batch the schedule outside cfg so prior run IDs remain unchanged.

    The time guard only applies before starting a run. An individual run may
    still exceed the budget. Partial old runs restart; no optimizer resume.
    """
    if max_new_runs < 1 or max_session_hours <= 0:
        raise ValueError('Giới hạn phiên phải dương.')
    for values, allowed, label in [(session_datasets, cfg['datasets'], 'dataset'),
                                   (session_models, ['SimpleDyG'] + cfg['baseline_models'], 'model'),
                                   (session_seeds, cfg['seeds'], 'seed')]:
        if values is not None and (not values or not set(values).issubset(allowed)):
            raise ValueError('Bộ lọc phiên không hợp lệ: ' + label)
    output = Path(output)
    records = [_recovery_json(output / 'runs' / r['run_id'] / 'result.json')
               for r in plan if r['status'] == 'complete']
    began, launched = time.perf_counter(), 0
    eligible = [r for r in plan if r['status'] != 'complete'
                and (session_datasets is None or r['dataset'] in session_datasets)
                and (session_models is None or r['model'] in session_models)
                and (session_seeds is None or r['seed'] in session_seeds)]
    pipeline['torch'].set_num_threads(min(4, pipeline['os'].cpu_count() or 1))
    device = pipeline['choose_device'](cfg)
    pipeline['dump_json'](output / 'continuation_environment.json',
                          dict(python=pipeline['sys'].version, torch=pipeline['torch'].__version__,
                               device=str(device), scheduling_only=True,
                               max_new_runs=max_new_runs, max_session_hours=max_session_hours))
    print(f'Tái sử dụng {len(records)} lượt hoàn thành; còn {len(eligible)} lượt trong bộ lọc phiên.')
    for row in eligible:
        if launched >= max_new_runs or time.perf_counter() - began >= max_session_hours * 3600:
            break
        print('Bắt đầu:', row['run_id'], '| Trạng thái cũ:', row['status'], flush=True)
        result = pipeline['execute_run'](datasets[row['dataset']], row['model'], row['variant'],
                                         row['seed'], cfg, device, output, author_root, dyglib_root)
        records.append(result)
        launched += 1
        export_continuation_results(pipeline, records, cfg, output)
        if result['status'] != 'complete':
            raise RuntimeError('Lượt chạy thất bại; xem error.txt: ' + row['run_id'])
    result = export_continuation_results(pipeline, records, cfg, output)
    complete = sum(r['status'] == 'complete' for r in records)
    print(f'Hoàn thành {complete}/{len(plan)} lượt của kế hoạch gốc. '
          'Lưu Output phiên này để tiếp tục các lượt còn lại.')
    return result


def export_continuation_results(pipeline, records, cfg, output):
    """Keep the previous complete ZIP until its replacement is fully closed."""
    final = Path(output).parent / (Path(output).name + '.zip')
    original_zip = pipeline['zipfile'].ZipFile

    class AtomicOutputZip(original_zip):
        def __init__(self, file, mode='r', *args, **kwargs):
            self.recovery_final = final if mode == 'w' and Path(file) == final else None
            actual = final.with_name(final.name + '.writing') if self.recovery_final else file
            super().__init__(actual, mode, *args, **kwargs)

        def __exit__(self, exc_type, exc_value, traceback):
            result = super().__exit__(exc_type, exc_value, traceback)
            if self.recovery_final is not None and exc_type is None:
                Path(self.filename).replace(self.recovery_final)
            return result

    pipeline['zipfile'].ZipFile = AtomicOutputZip
    try:
        return pipeline['export_results'](records, cfg, output)
    finally:
        pipeline['zipfile'].ZipFile = original_zip
