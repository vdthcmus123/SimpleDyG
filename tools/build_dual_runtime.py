"""Keep the original scientific core, generate its dedicated two-worker version."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = (ROOT / 'tools/kaggle_pipeline.py').read_text(encoding='utf-8')
    source = source.replace('import torch\n', 'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")\nimport torch\n', 1)
    source = source.replace("export_checkpoints=False, device='auto')", "export_checkpoints=False, device='auto', baseline_micro_batch=8, execution_format='dual_resume_v1')", 1)
    source = source.replace('baseline_dims=[16], history_cap=8,', 'baseline_dims=[16], baseline_micro_batch=16, history_cap=8,', 1)
    source = source.replace("    if cfg['protocol'] not in {'corrected', 'author_compat'}:",
                            "    if not isinstance(cfg['baseline_micro_batch'], int) or cfg['baseline_micro_batch'] < 1:\n        raise ValueError('baseline_micro_batch must be a positive integer')\n    if cfg['protocol'] not in {'corrected', 'author_compat'}:", 1)
    source = source.replace("implementation='1')", "implementation='dual_resume_v1')", 1)
    source = source.replace('def export_results(records, cfg, output_root):',
                            'def export_results(records, cfg, output_root, make_archive=True):', 1)
    source = source.replace("'testing_seconds', 'peak_gpu_mb']} |", "'testing_seconds', 'peak_gpu_mb', 'worker_id', 'physical_gpu']} |", 1)
    begin = source.index("    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:", source.index('def export_results'))
    end = source.index('    return df, summary, archive', begin)
    source = source[:begin] + '    if make_archive:\n' + ''.join('    ' + line if line.strip() else line for line in source[begin:end].splitlines(keepends=True)) + source[end:]
    # All inference is deterministic/replayable. Deadline checks do not consume
    # RNG; phase checkpoints were committed before entering these functions.
    parsed = ast.parse(source)
    lines = source.splitlines(keepends=True)
    insertions = []
    for node in parsed.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ['evaluate_lm', 'evaluate_ranker']:
            for child in ast.walk(node):
                if isinstance(child, (ast.For, ast.While)):
                    insertions.append((child.body[0].lineno - 1, ' ' * child.body[0].col_offset + 'check_deadline()\n'))
    for index, line in sorted(insertions, reverse=True):
        lines.insert(index, line)
    source = ''.join(lines) + '\n\n' + (ROOT / 'tools/dual_runtime_extensions.py').read_text(encoding='utf-8')
    ast.parse(source)
    path = ROOT / 'tools/kaggle_pipeline_dual.py'
    path.write_text(source, encoding='utf-8', newline='\n')
    print(path)


if __name__ == '__main__':
    build()
