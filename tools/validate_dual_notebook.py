"""Execute the fixed notebook, then resume from ZIP and directory with full clones."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import nbformat

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--notebook', default='SimpleDyG_Kaggle_2GPU_Continue_Fixed.ipynb')
args = parser.parse_args()
NOTEBOOK = ROOT / 'notebooks' / args.notebook
nb = nbformat.read(NOTEBOOK, as_version=4)
nbformat.validate(nb)
original_cwd = Path.cwd()
fixture = Path(tempfile.mkdtemp(prefix='resume_source_notebook_', dir=ROOT / 'tmp'))
assert fixture.resolve().is_relative_to((ROOT / 'tmp').resolve())
folders = [fixture / name for name in ['fresh', 'restored_zip_full_clone', 'restored_directory_full_clone']]
for folder in folders:
    if (folder / 'kaggle_working/simpledyg_2gpu_results/config.json').exists():
        raise RuntimeError('Validation folder already contains a previous run; choose fresh fixture paths.')
    folder.mkdir(parents=True, exist_ok=True)
evidence, previous, fresh_output, began = [], None, None, time.perf_counter()
full_author, full_dyglib = fixture / 'full_source/SimpleDyG', fixture / 'full_source/DyGLib'
try:
    for index, folder in enumerate(folders):
        os.chdir(folder)
        namespace = {'__name__': '__notebook_validation__', 'display': lambda *args: None}
        executed = 0
        for cell_index, cell in enumerate(nb.cells):
            if cell.cell_type != 'code':
                continue
            exec(compile(cell.source, f'{NOTEBOOK.name}:cell{cell_index}', 'exec'), namespace)
            if cell_index == 1:
                namespace.update(PROFILE='smoke', OFFLINE=True, INSTALL_DEPENDENCIES=False,
                                 INPUT_BUNDLE_OVERRIDE=str(ROOT / 'notebooks/SimpleDyG_Kaggle_2GPU_Inputs.zip') if index == 0 else '',
                                 SESSION_HOURS=.1, CHECKPOINT_SECONDS=1., RESUME=index == 1,
                                 RESUME_FROM=str(previous) if index == 1 else str(fresh_output) if index == 2 else '')
                namespace['RESUME'] = index > 0
                if index > 0:
                    namespace.update(AUTHOR_ROOT_OVERRIDE=str(full_author), DYGLIB_ROOT_OVERRIDE=str(full_dyglib))
            executed += 1
        output = namespace['OUTPUT']
        previous = namespace['resume_archive']
        if index == 0:
            fresh_output = output
            shutil.copytree(namespace['AUTHOR_ROOT'], full_author)
            shutil.copytree(namespace['DYGLIB_ROOT'], full_dyglib)
            (full_author / 'models/unused_extra.py').write_text('# full-clone extra\n', encoding='utf-8')
            (full_dyglib / 'models/UnusedBackbone.py').write_text('# full-clone extra\n', encoding='utf-8')
        records = json.loads((output / 'run_manifest.json').read_text())
        assert len(records) == 5 and all(r['status'] == 'complete' for r in records)
        assert len({r['run_id'] for r in records}) == 5
        assert set(r['worker_id'] for r in records) == {0, 1}
        assert all((output / 'runs' / r['run_id'] / 'best.pt').exists() for r in records)
        if index > 0:
            assert not any('CLAIM' in path.read_text() for path in output.glob('worker_*.log'))
            alignment = json.loads((output / 'resume_source_alignment.json').read_text())
            assert alignment['aligned'] and len(alignment['extra_files_excluded']) == 2
        evidence.append(dict(phase=folders[index].name, code_cells_executed=executed,
                             complete_runs=5, worker_ids=[0, 1], reused_without_training=index > 0,
                             all_best_checkpoints_retained=True,
                             source_alignment=namespace.get('source_alignment_report')))
finally:
    os.chdir(original_cwd)
report = dict(date='2026-10-07', passed=True, notebook=str(NOTEBOOK), phases=evidence,
              elapsed_seconds=round(time.perf_counter() - began, 2),
              scope='Python 3.13 local CPU; all delivered cells; actual two-process pipeline; ZIP and directory resume with additional full-clone files; no live dual-T4 test')
(ROOT / 'reports/resume_source_fix_notebook_validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2))
