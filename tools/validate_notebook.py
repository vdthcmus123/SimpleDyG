"""Execute every delivered notebook code cell in a fresh CPU process, offline."""
from pathlib import Path
import json
import os
import time
import sys
import importlib.abc

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('SIMPLEDYG_BLOCK_HF_IMPORTS') == '1':
    class BlockHF(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'transformers', 'tokenizers'}:
                raise AssertionError('Unexpected dependency import: '+fullname)
    sys.meta_path.insert(0, BlockHF())
nb = json.loads((ROOT/'notebooks/SimpleDyG_Kaggle_Full_Pipeline.ipynb').read_text(encoding='utf-8'))
working = ROOT/('tmp/notebook_validation_'+str(sys.version_info.major)+str(sys.version_info.minor))
working.mkdir(parents=True, exist_ok=True)
os.chdir(working)
namespace = {'__name__': '__main__', 'display': lambda obj: print(type(obj).__name__)}
started = time.perf_counter()
code_cells = 0
for index, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    print(f'Execute notebook cell {index}', flush=True)
    source = ''.join(cell['source'])
    exec(compile(source, f'notebook:cell{index}', 'exec'), namespace)
    if index == 1:
        namespace.update(PROFILE='smoke', OFFLINE=True,
                         INPUT_BUNDLE_OVERRIDE=str(ROOT/'notebooks/SimpleDyG_Kaggle_Inputs.zip'))
    code_cells += 1
records = json.loads((namespace['OUTPUT']/'run_manifest.json').read_text())
assert len(records) == 5 and all(r['status']=='complete' for r in records)
# Completed-run reuse must not mix duplicates or restart training.
reused, _, _ = namespace['run_pipeline'](namespace['AUTHOR_ROOT'], namespace['DYGLIB_ROOT'], namespace['CFG'], namespace['OUTPUT'])
assert len(reused) == 5 and reused.run_id.nunique() == 5
evidence = dict(code_cells_executed=code_cells, all_cells_passed=True, offline_bundle_passed=True,
                completed_run_reuse_passed=True, smoke_runs=len(records),
                elapsed_seconds=round(time.perf_counter()-started, 2),
                environment=json.loads((namespace['OUTPUT']/'environment.json').read_text()),
                hf_imports_blocked=os.environ.get('SIMPLEDYG_BLOCK_HF_IMPORTS')=='1',
                scope='fresh local Python execution of all notebook cells; CPU smoke; not a Kaggle GPU full run')
version_suffix=str(sys.version_info.major)+str(sys.version_info.minor)
(ROOT/f'reports/notebook_execution_validation_py{version_suffix}.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
print(json.dumps(evidence, indent=2))
