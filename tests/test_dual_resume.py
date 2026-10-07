"""Real interruption/replay checks and two-process integration on CPU smoke."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import kaggle_pipeline_dual as d
import kaggle_pipeline as old

AUTHOR, DYGLIB = ROOT / 'repo_clone', ROOT / 'tmp/inspection/DyGLib'


def exact_tree(test, first, second):
    if isinstance(first, d.torch.Tensor):
        d.torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        test.assertEqual(set(first), set(second))
        for key in first:
            exact_tree(test, first[key], second[key])
    elif isinstance(first, (tuple, list)):
        test.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            exact_tree(test, a, b)
    else:
        test.assertEqual(first, second)


class DualResumeChecks(unittest.TestCase):
    def setUp(self):
        d.torch.set_num_threads(2)
        d._RUNTIME.update(deadline=float('inf'), checkpoint_seconds=100000., last_save=0.,
                          pause_after_steps=None, worker_id=None, stop_file=None)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tmp')
        self.root = Path(self.temp.name)
        self.cfg = d.make_config('smoke', epochs=3, baseline_epochs=3, patience=10,
                                 train_limit=8, eval_limit=2, edge_limit=32, baseline_micro_batch=4)
        self.ds = d.load_dataset(AUTHOR, 'UCI_13', self.cfg)

    def tearDown(self):
        d._RUNTIME.update(deadline=float('inf'), pause_after_steps=None, stop_file=None)
        self.temp.cleanup()

    def test_lm_mid_epoch_resume_preserves_model_optimizer_scheduler_rng(self):
        a, b = self.root / 'reference', self.root / 'interrupted'
        a.mkdir(); b.mkdir()
        first = d.train_lm(AUTHOR, self.ds, 'full', 7, self.cfg, d.torch.device('cpu'), a)
        d._RUNTIME['pause_after_steps'] = 1
        with self.assertRaises(d.TrainingPaused):
            d.train_lm(AUTHOR, self.ds, 'full', 7, self.cfg, d.torch.device('cpu'), b)
        partial = d.torch.load(b / 'latest.pt', weights_only=True)
        self.assertEqual(partial['next_batch'], 1)
        self.assertIn('optimizer', partial)
        self.assertIn('scheduler', partial)
        self.assertIn('scaler', partial)
        self.assertIn('rng', partial)
        d._RUNTIME['pause_after_steps'] = None
        resumed = d.train_lm(AUTHOR, self.ds, 'full', 7, self.cfg, d.torch.device('cpu'), b)
        x, y = [d.torch.load(path / 'latest.pt', weights_only=True) for path in [a, b]]
        for key in ['model', 'optimizer', 'scheduler', 'scaler', 'rng', 'history', 'epoch_generator', 'global_step']:
            exact_tree(self, x[key], y[key])
        for key in ['test', 'validation', 'best_epoch', 'epochs_run', 'optimizer_steps']:
            self.assertEqual(first[key], resumed[key])

    def test_lm_validation_phase_resume(self):
        a, b = self.root / 'reference', self.root / 'interrupted'
        a.mkdir(); b.mkdir()
        reference = d.train_lm(AUTHOR, self.ds, 'same_time', 3, self.cfg, d.torch.device('cpu'), a)
        real = d.evaluate_lm
        with patch.object(d, 'evaluate_lm', side_effect=d.TrainingPaused('validation interrupted')):
            with self.assertRaises(d.TrainingPaused):
                d.train_lm(AUTHOR, self.ds, 'same_time', 3, self.cfg, d.torch.device('cpu'), b)
        self.assertEqual(d.torch.load(b / 'latest.pt', weights_only=True)['phase'], 'validate')
        result = d.train_lm(AUTHOR, self.ds, 'same_time', 3, self.cfg, d.torch.device('cpu'), b)
        self.assertEqual(reference['test'], result['test'])
        exact_tree(self, d.torch.load(a / 'latest.pt', weights_only=True)['model'],
                   d.torch.load(b / 'latest.pt', weights_only=True)['model'])
        self.assertIs(d.evaluate_lm, real)

    def test_author_compat_legacy_optimizer_resume(self):
        cfg = d.make_config('smoke', protocol='author_compat', context=1024,
                            epochs=3, train_limit=8, eval_limit=1, patience=10)
        ds = d.load_dataset(AUTHOR, 'UCI_13', cfg)
        a, b = self.root / 'author_reference', self.root / 'author_resume'
        a.mkdir(); b.mkdir()
        # Exercise the actual author model/padding/legacy optimizer. A fixed
        # validation score keeps the checkpoint policy predictable and avoids
        # spending this unit check on up to 1024 greedy generation steps.
        score = dict(ndcg5=1., ndcg5_author=1., jaccard=1., jaccard_at5=1.,
                     n_evaluated=1, ended_fraction=1.)
        with patch.object(d, 'evaluate_lm', return_value=score):
            d.train_lm(AUTHOR, ds, 'full', 6, cfg, d.torch.device('cpu'), a)
            d._RUNTIME['pause_after_steps'] = 1
            with self.assertRaises(d.TrainingPaused):
                d.train_lm(AUTHOR, ds, 'full', 6, cfg, d.torch.device('cpu'), b)
            d._RUNTIME['pause_after_steps'] = None
            d.train_lm(AUTHOR, ds, 'full', 6, cfg, d.torch.device('cpu'), b)
        x, y = [d.torch.load(path / 'latest.pt', weights_only=True) for path in [a, b]]
        for key in ['model', 'optimizer', 'scheduler', 'rng', 'global_step']:
            exact_tree(self, x[key], y[key])

    def test_baseline_microbatch_mid_epoch_resume(self):
        classes, Sampler = d.load_dyglib(DYGLIB)
        for key in ['TGAT', 'GraphMixer']:
            with self.subTest(model=key):
                a, b = self.root / (key + '_reference'), self.root / (key + '_interrupted')
                a.mkdir(); b.mkdir()
                sampler = d.build_sampler(self.ds, self.cfg, Sampler)
                first = d.train_ranker_candidate(key, 16, 4, self.ds, self.cfg,
                           d.torch.device('cpu'), sampler, classes, a)
                d._RUNTIME['pause_after_steps'] = 1
                with self.assertRaises(d.TrainingPaused):
                    d.train_ranker_candidate(key, 16, 4, self.ds, self.cfg,
                            d.torch.device('cpu'), sampler, classes, b)
                partial = d.torch.load(b / 'latest_dim16.pt', weights_only=True)
                self.assertEqual(partial['next_batch'], 1)
                self.assertIn('negative_rng', partial)
                d._RUNTIME['pause_after_steps'] = None
                resumed = d.train_ranker_candidate(key, 16, 4, self.ds, self.cfg,
                            d.torch.device('cpu'), sampler, classes, b)
                self.assertEqual(first[1:4], resumed[1:4])
                exact_tree(self, d.torch.load(a / 'candidate_dim16.pt', weights_only=True),
                           d.torch.load(b / 'candidate_dim16.pt', weights_only=True))
                self.assertEqual(json.loads((a / 'history_dim16.json').read_text()),
                                 json.loads((b / 'history_dim16.json').read_text()))

    def test_old_and_new_lm_training_match_without_interruption(self):
        a, b = self.root / 'old', self.root / 'new'
        a.mkdir(); b.mkdir()
        reference = old.train_lm(AUTHOR, self.ds, 'no_time', 2, self.cfg, d.torch.device('cpu'), a)
        result = d.train_lm(AUTHOR, self.ds, 'no_time', 2, self.cfg, d.torch.device('cpu'), b)
        self.assertEqual(reference['test'], result['test'])
        self.assertEqual(reference['validation'], result['validation'])
        exact_tree(self, d.torch.load(a / 'best.pt', weights_only=True),
                   d.torch.load(b / 'best.pt', weights_only=True))

    def test_checkpoint_rejects_changed_identity_or_environment(self):
        path = self.root / 'run'; path.mkdir()
        d._RUNTIME['pause_after_steps'] = 1
        with self.assertRaises(d.TrainingPaused):
            d.train_lm(AUTHOR, self.ds, 'full', 0, self.cfg, d.torch.device('cpu'), path)
        d._RUNTIME['pause_after_steps'] = None
        with self.assertRaisesRegex(RuntimeError, 'identity/config mismatch'):
            d.train_lm(AUTHOR, self.ds, 'full', 1, self.cfg, d.torch.device('cpu'), path)
        saved = d.torch.load(path / 'latest.pt', weights_only=True)
        saved['environment']['torch'] = 'different-version'
        d.atomic_torch_save(saved, path / 'latest.pt')
        with self.assertRaisesRegex(RuntimeError, 'environment differs'):
            d.train_lm(AUTHOR, self.ds, 'full', 0, self.cfg, d.torch.device('cpu'), path)

    def test_queue_claims_are_unique_and_gpu_masks_are_distinct(self):
        database = self.root / 'jobs.sqlite'
        plan = [dict(run_id=f'job_{i}', dataset='UCI_13', model='TGAT', variant='full', seed=i) for i in range(20)]
        d.initialize_queue(database, self.root, plan)
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(lambda worker: d.claim_job(database, worker), [0, 1] * 10))
        self.assertEqual(len({r['run_id'] for r in rows}), 20)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '4,6'}):
            self.assertEqual(d.worker_environment(0)['CUDA_VISIBLE_DEVICES'], '4')
            self.assertEqual(d.worker_environment(1)['CUDA_VISIBLE_DEVICES'], '6')

    def test_snapshot_interruption_keeps_previous_archive(self):
        output = self.root / 'output'; output.mkdir()
        (output / 'record.json').write_text('{}')
        archive = output / 'resume_snapshot.zip'
        d.create_resume_snapshot(output, archive)
        before = archive.read_bytes()
        with patch.object(d.zipfile.ZipFile, 'write', side_effect=RuntimeError('simulated ZIP interruption')):
            with self.assertRaises(RuntimeError):
                d.create_resume_snapshot(output, archive)
        self.assertEqual(archive.read_bytes(), before)
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip())
            self.assertNotIn('resume_snapshot.zip', z.namelist())

    def test_two_workers_pause_snapshot_restore_and_finish_without_duplicates(self):
        cfg = d.make_config('smoke', epochs=2, baseline_epochs=2, train_limit=8,
                             eval_limit=2, edge_limit=32, baseline_micro_batch=4)
        first = self.root / 'first'
        result = d.run_dual_pipeline(AUTHOR, DYGLIB, cfg, first, cpu_workers=2,
                     session_hours=.1, checkpoint_seconds=1, pause_after_steps=1)
        records = json.loads((first / 'run_manifest.json').read_text())
        self.assertEqual(sum(r['status'] == 'paused' for r in records), 2)
        self.assertEqual(len(list((first / 'runs').glob('*/latest.pt'))), 2)
        with zipfile.ZipFile(result[2]) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(sum(n.endswith('/latest.pt') for n in z.namelist()), 2)
        second = self.root / 'restored'
        d.restore_dual_output(result[2], second)
        results, summary, snapshot = d.run_dual_pipeline(AUTHOR, DYGLIB, cfg, second,
                      cpu_workers=2, session_hours=.1, checkpoint_seconds=1)
        self.assertEqual(len(results), 5)
        self.assertEqual(len(set(results.run_id)), 5)
        self.assertEqual(set(results.worker_id), {0, 1})
        records = json.loads((second / 'run_manifest.json').read_text())
        self.assertTrue(all(r['status'] == 'complete' for r in records))
        self.assertFalse(list((second / 'runs').glob('*/latest*.pt')))
        self.assertTrue(all((second / 'runs' / r['run_id'] / 'best.pt').exists() for r in records))
        self.assertTrue(any('RESUME' in p.read_text() for p in second.glob('worker_*.log')))
        with zipfile.ZipFile(snapshot) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(len(json.loads(z.read('run_manifest.json'))), 5)


if __name__ == '__main__':
    unittest.main()
