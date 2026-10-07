"""Prepare a source tree with the exact saved inventory before resuming.

Kept outside the training runtime so existing run/checkpoint identities stay valid.
Extra Python files in a full clone are excluded only after every saved file has
passed its original checksum. No saved fingerprint or checkpoint is rewritten.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def align_resume_sources(runtime, author_root, dyglib_root, saved_config, output_root, work_root):
    output_root, work_root = Path(output_root), Path(work_root)
    expected = json.loads((output_root / 'source_hashes.json').read_text(encoding='utf-8'))
    fingerprint = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    if fingerprint != saved_config.get('_source_fingerprint'):
        raise RuntimeError('Saved source manifest does not match config; choose the intact previous Output.')
    science_config = {k: v for k, v in saved_config.items() if not k.startswith('_')}
    current_config, actual = runtime.prepare_config(author_root, dyglib_root, science_config)
    if current_config['_pipeline_fingerprint'] != saved_config.get('_pipeline_fingerprint'):
        raise RuntimeError('Training runtime differs from the saved run. Use the notebook with its original runtime.')

    roots = {'author': Path(author_root) / 'models', 'dyglib': Path(dyglib_root)}

    def checked_path(root, relative):
        if not relative or '\\' in relative or ':' in relative or '..' in relative.split('/') or relative.startswith('/'):
            raise ValueError('Unsafe saved source/data path: ' + relative)
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Saved source/data path leaves its root: ' + relative)
        if not path.is_file():
            raise FileNotFoundError('Saved source/data file is missing: ' + str(path))
        return path

    source_paths = {}
    labels = set()
    for name, wanted in expected.items():
        label, separator, relative = name.partition('/')
        if not separator or label not in roots or not relative.endswith('.py'):
            raise ValueError('Unsupported saved source entry: ' + name)
        path = checked_path(roots[label], relative)
        if runtime.sha256(path) != wanted:
            raise RuntimeError('Saved source checksum differs: ' + name + '. Attach the original source bundle.')
        source_paths[name] = path
        labels.add(label)
    if labels != {'author', 'dyglib'}:
        raise RuntimeError('Saved source manifest must include author and DyGLib files.')
    extras = sorted(set(actual) - set(expected))
    report = dict(saved_source_files=len(expected), verified_source_files=len(source_paths),
                  extra_files_excluded=extras, source_fingerprint=fingerprint,
                  runtime_fingerprint=current_config['_pipeline_fingerprint'], aligned=False)
    if actual == expected:
        if current_config != saved_config:
            raise RuntimeError('Saved experiment config differs even with matching source.')
        print('Resume source checksums verified:', len(expected), 'files; original inventory already matches.')
        return Path(author_root), Path(dyglib_root), report

    # Copy only the original source inventory into a fresh workspace directory.
    # Fresh staging avoids changing or deleting the full clone or the Input.
    work_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='resume_verified_sources_', dir=work_root))
    author_view, dyglib_view = stage / 'SimpleDyG', stage / 'DyGLib'
    targets = {'author': author_view / 'models', 'dyglib': dyglib_view}
    for name, path in source_paths.items():
        label, relative = name.split('/', 1)
        target = targets[label] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    # The author root also holds dataset/vocab/raw graph files used by the core.
    # Reproduce that tree from the saved data audit and check its original bytes.
    audits = json.loads((output_root / 'data_audit.json').read_text(encoding='utf-8'))
    data_files = {}
    for dataset in science_config['datasets']:
        for relative, wanted in audits[dataset]['files'].items():
            if relative in data_files and data_files[relative] != wanted:
                raise RuntimeError('Conflicting saved dataset hashes: ' + relative)
            data_files[relative] = wanted
    for relative, wanted in data_files.items():
        if relative.split('/')[0] not in {'resources', 'vocabs', 'all_data'}:
            raise ValueError('Unsupported saved dataset entry: ' + relative)
        path = checked_path(Path(author_root), relative)
        if runtime.sha256(path) != wanted:
            raise RuntimeError('Saved dataset checksum differs: ' + relative + '. Attach the original dataset bundle.')
        target = author_view / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    verified_config, verified_files = runtime.prepare_config(author_view, dyglib_view, science_config)
    if verified_files != expected or verified_config != saved_config:
        raise RuntimeError('Verified source view could not reproduce the original experiment identity.')
    report.update(aligned=True, verified_data_files=len(data_files),
                  author_view=str(author_view), dyglib_view=str(dyglib_view))
    print('RESUME source aligned:', len(expected), 'verified Python files;',
          len(extras), 'extra files excluded; original runtime/config identity preserved.')
    return author_view, dyglib_view, report
