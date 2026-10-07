"""Check result reuse, unchanged run IDs, ZIP recovery and export interruption."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import kaggle_pipeline as p
import kaggle_continue as c
from recover_incomplete_zip import recover_zip

p.PIPELINE_SOURCE_SHA256 = hashlib.sha256((ROOT / 'tools/kaggle_pipeline.py').read_bytes()).hexdigest()
AUTHOR = ROOT / 'repo_clone'
DYGLIB = ROOT / 'tmp/inspection/DyGLib'
SMOKE = ROOT / 'tmp/notebook_validation_313/kaggle_working/simpledyg_smoke_corrected'


class ContinuationChecks(unittest.TestCase):
    def test_actual_paper_recovery_plan(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            output = c.restore_previous_output(ROOT / 'recovery/SimpleDyG_Continue_Input.zip', Path(temp) / 'output')
            cfg, datasets, plan = c.verify_previous_output(vars(p), AUTHOR, DYGLIB, output)
            self.assertEqual(len(plan), 100)
            self.assertEqual(sum(r['status'] == 'complete' for r in plan), 35)
            self.assertEqual(sum(r['status'] != 'complete' for r in plan), 65)
            self.assertEqual(sorted(r['seed'] for r in plan if r['model'] == 'TGAT'
                                    and r['dataset'] == 'UCI_13' and r['status'] != 'complete'), list(range(5, 10)))
            saved = json.loads((output / 'config.json').read_text())
            self.assertEqual(cfg, saved)
            self.assertEqual(len(datasets), 2)
            bad = dict(vars(p), PIPELINE_SOURCE_SHA256='wrong')
            with self.assertRaisesRegex(RuntimeError, 'Core khác'):
                c.verify_previous_output(bad, AUTHOR, DYGLIB, output)

    def test_two_smoke_sessions_preserve_ids_and_train_only_pending(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            output = c.restore_previous_output(SMOKE, Path(temp) / 'output')
            saved_config = (output / 'config.json').read_bytes()
            expected = {}
            for path in (output / 'runs').glob('*/result.json'):
                result = json.loads(path.read_text())
                expected[result['run_id']] = result
                if result['model'] in ['TGAT', 'GraphMixer']:
                    path.unlink()  # Only remove copied test metadata inside this temporary workspace.
            cfg, datasets, plan = c.verify_previous_output(vars(p), AUTHOR, DYGLIB, output)
            self.assertEqual(sum(r['status'] == 'complete' for r in plan), 3)
            calls = []
            original_execute = p.execute_run

            def counted(*args, **kwargs):
                calls.append((args[1], args[3]))
                return original_execute(*args, **kwargs)

            namespace = dict(vars(p), execute_run=counted)
            for completed in [4, 5]:
                c.continue_previous_output(namespace, AUTHOR, DYGLIB, cfg, datasets, plan, output,
                                            session_datasets=['UCI_13'], max_new_runs=1)
                cfg, datasets, plan = c.verify_previous_output(vars(p), AUTHOR, DYGLIB, output)
                self.assertEqual(sum(r['status'] == 'complete' for r in plan), completed)
            self.assertEqual(calls, [('TGAT', 0), ('GraphMixer', 0)])
            self.assertEqual((output / 'config.json').read_bytes(), saved_config)
            for identifier, previous in expected.items():
                current = json.loads((output / 'runs' / identifier / 'result.json').read_text())
                self.assertEqual(current['run_id'], previous['run_id'])
                self.assertEqual(current['test'], previous['test'])
                self.assertEqual(current['validation'], previous['validation'])
            with zipfile.ZipFile(output.parent / 'output.zip') as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(len(json.loads(archive.read('run_manifest.json'))), 5)

    def test_interrupted_export_preserves_previous_zip(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            output = Path(temp) / 'output'
            output.mkdir()
            final = output.parent / 'output.zip'
            with zipfile.ZipFile(final, 'w') as archive:
                archive.writestr('receipt.txt', 'previous complete ZIP')
            before = final.read_bytes()

            def interrupted(records, cfg, directory):
                with zipfile.ZipFile(final, 'w') as archive:
                    archive.writestr('partial.txt', 'new snapshot')
                    raise RuntimeError('simulated interruption')

            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                c.export_continuation_results(dict(vars(p), export_results=interrupted), [], {}, output)
            self.assertEqual(final.read_bytes(), before)
            self.assertIs(zipfile.ZipFile, p.zipfile.ZipFile)

    def test_zip_paths_do_not_escape_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            archive = Path(temp) / 'unsafe.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for name in ['config.json', 'data_audit.json', 'source_hashes.json', 'runs/a/result.json']:
                    z.writestr(name, '{}')
                z.writestr('../escape.txt', 'unsafe')
            with self.assertRaises(ValueError):
                c.restore_previous_output(archive, Path(temp) / 'output')
            self.assertFalse((Path(temp) / 'escape.txt').exists())

    def test_recover_truncated_local_records(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            source, target = Path(temp) / 'bad.zip', Path(temp) / 'fixed.zip'
            with zipfile.ZipFile(source, 'w', zipfile.ZIP_DEFLATED) as z:
                z.writestr('result.json', '{"status":"complete"}')
                z.writestr('best.pt', bytes(range(256)) * 300)
            raw = source.read_bytes()
            with zipfile.ZipFile(source) as z:
                last = z.infolist()[-1]
                limit = last.header_offset + 30 + len(last.filename) + last.compress_size // 2
            source.write_bytes(raw[:limit])
            before = source.read_bytes()
            report = recover_zip(source, target)
            self.assertEqual(report['complete_records'], 1)
            self.assertEqual(source.read_bytes(), before)
            with zipfile.ZipFile(target) as z:
                self.assertIsNone(z.testzip())
                self.assertEqual(json.loads(z.read('result.json'))['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
