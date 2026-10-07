"""Regression: resume a bundle run from a full clone without changing identity."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import kaggle_pipeline_dual as d
from resume_source_alignment import align_resume_sources

AUTHOR, DYGLIB = ROOT / 'repo_clone', ROOT / 'tmp/inspection/DyGLib'


class ResumeSourceAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='source_resume_test_', dir=ROOT / 'tmp')
        self.root = Path(self.temp.name)
        assert self.root.resolve().is_relative_to((ROOT / 'tmp').resolve())
        self.output = self.root / 'saved'
        self.cfg = d.make_config('smoke', epochs=2, baseline_epochs=2, train_limit=8,
                                 eval_limit=2, edge_limit=32, baseline_micro_batch=4)
        self.saved, self.sources = d.prepare_config(AUTHOR, DYGLIB, self.cfg)
        self.audits = {n: d.load_dataset(AUTHOR, n, self.cfg)['audit'] for n in self.cfg['datasets']}
        d.dump_json(self.output / 'config.json', self.saved)
        d.dump_json(self.output / 'source_hashes.json', self.sources)
        d.dump_json(self.output / 'data_audit.json', self.audits)

    def tearDown(self):
        assert self.root.resolve().is_relative_to((ROOT / 'tmp').resolve())
        self.temp.cleanup()

    def full_clone_fixture(self):
        author, dyglib = self.root / 'full/SimpleDyG', self.root / 'full/DyGLib'
        for name in self.sources:
            label, relative = name.split('/', 1)
            original = (AUTHOR / 'models' if label == 'author' else DYGLIB) / relative
            target = (author / 'models' if label == 'author' else dyglib) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        for audit in self.audits.values():
            for relative in audit['files']:
                target = author / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(AUTHOR / relative, target)
        (author / 'models/unused_extra.py').write_text('# unused full-repo file\n', encoding='utf-8')
        (dyglib / 'models/UnusedBackbone.py').write_text('# unused full-repo model\n', encoding='utf-8')
        return author, dyglib

    def align(self, author=AUTHOR, dyglib=DYGLIB):
        return align_resume_sources(d, author, dyglib, self.saved, self.output, self.root)

    def test_matching_inventory_is_reused(self):
        author, dyglib, report = self.align()
        self.assertEqual((author, dyglib), (AUTHOR, DYGLIB))
        self.assertFalse(report['aligned'])

    def test_extra_files_reproduce_error_then_verified_view_preserves_identity(self):
        author, dyglib = self.full_clone_fixture()
        original_config = (self.output / 'config.json').read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'different config/core/source'):
            d.run_dual_pipeline(author, dyglib, self.cfg, self.output, cpu_workers=2)
        aligned_author, aligned_dyglib, report = self.align(author, dyglib)
        self.assertTrue(report['aligned'])
        self.assertEqual(len(report['extra_files_excluded']), 2)
        cfg, files = d.prepare_config(aligned_author, aligned_dyglib, self.cfg)
        self.assertEqual(cfg, self.saved)
        self.assertEqual(files, self.sources)
        self.assertEqual((self.output / 'config.json').read_bytes(), original_config)
        for name in self.cfg['datasets']:
            audit = d.load_dataset(aligned_author, name, self.cfg)['audit']
            self.assertEqual(audit, self.audits[name])
        self.assertTrue((author / 'models/unused_extra.py').exists())

    def test_changed_tracked_source_is_rejected(self):
        author, dyglib = self.full_clone_fixture()
        with (dyglib / 'models/TGAT.py').open('a', encoding='utf-8') as f:
            f.write('\n# changed source\n')
        with self.assertRaisesRegex(RuntimeError, 'Saved source checksum differs: dyglib/models/TGAT.py'):
            self.align(author, dyglib)

    def test_two_datasets_preserve_audits_after_alignment(self):
        self.cfg = dict(self.cfg, datasets=['UCI_13', 'ML_10M_13'])
        self.saved, self.sources = d.prepare_config(AUTHOR, DYGLIB, self.cfg)
        self.audits = {n: d.load_dataset(AUTHOR, n, self.cfg)['audit'] for n in self.cfg['datasets']}
        d.dump_json(self.output / 'config.json', self.saved)
        d.dump_json(self.output / 'data_audit.json', self.audits)
        author, dyglib = self.full_clone_fixture()
        author, dyglib, report = self.align(author, dyglib)
        self.assertEqual(report['verified_data_files'], 14)
        self.assertEqual(d.prepare_config(author, dyglib, self.cfg)[0], self.saved)
        for name in self.cfg['datasets']:
            self.assertEqual(d.load_dataset(author, name, self.cfg)['audit'], self.audits[name])

    def test_changed_dataset_is_rejected(self):
        author, dyglib = self.full_clone_fixture()
        relative = next(iter(self.audits['UCI_13']['files']))
        with (author / relative).open('ab') as f:
            f.write(b'\n')
        with self.assertRaisesRegex(RuntimeError, 'Saved dataset checksum differs'):
            self.align(author, dyglib)

    def test_changed_training_runtime_is_rejected(self):
        changed = dict(self.saved, _pipeline_fingerprint='different-runtime')
        with self.assertRaisesRegex(RuntimeError, 'Training runtime differs'):
            align_resume_sources(d, AUTHOR, DYGLIB, changed, self.output, self.root)

    def test_two_baseline_workers_resume_full_states_with_identical_run_ids(self):
        cfg = dict(self.cfg, variants=[], baseline_models=['TGAT', 'GraphMixer'])
        first = self.root / 'paused'
        _, _, snapshot = d.run_dual_pipeline(AUTHOR, DYGLIB, cfg, first, cpu_workers=2,
                           session_hours=.1, checkpoint_seconds=1, pause_after_steps=1)
        plan = json.loads((first / 'experiment_plan.json').read_text())
        states = list((first / 'runs').glob('*/latest_dim16.pt'))
        self.assertEqual(len(states), 2)
        old_hashes = {p.relative_to(first).as_posix(): d.sha256(p) for p in states}
        restored = self.root / 'restored'
        d.restore_dual_output(snapshot, restored)
        saved = json.loads((restored / 'config.json').read_text())
        author, dyglib = self.full_clone_fixture()
        author, dyglib, report = align_resume_sources(d, author, dyglib, saved, restored, self.root)
        self.assertTrue(report['aligned'])
        self.assertEqual(old_hashes, {n: d.sha256(restored / n) for n in old_hashes})
        results, _, _ = d.run_dual_pipeline(author, dyglib, cfg, restored, cpu_workers=2,
                           session_hours=.1, checkpoint_seconds=1, include_completed_best=True)
        self.assertEqual(len(results), 2)
        manifest = json.loads((restored / 'run_manifest.json').read_text())
        self.assertTrue(all(r['status'] == 'complete' for r in manifest))
        self.assertEqual({r['run_id'] for r in manifest}, {r['run_id'] for r in plan})
        self.assertEqual({r['worker_id'] for r in manifest}, {0, 1})
        self.assertTrue(all('RESUME' in p.read_text() for p in restored.glob('worker_*.log')))
        self.assertTrue(all((restored / 'runs' / r['run_id'] / 'best.pt').exists() for r in manifest))


if __name__ == '__main__':
    unittest.main(verbosity=2)
