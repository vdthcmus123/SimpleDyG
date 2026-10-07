"""Semantic checks for scientific evaluation, split integrity and notebook packaging."""
import ast
import json
from pathlib import Path
import sys
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import kaggle_pipeline as p


class PipelineChecks(unittest.TestCase):
    def test_standard_ndcg_does_not_reward_duplicates(self):
        self.assertAlmostEqual(p.metrics(['a', 'a', 'a'], ['a'])['ndcg5'], 1.)
        self.assertGreater(p.metrics(['a', 'a', 'a'], ['a'], author=True)['ndcg5'], 1.)
        self.assertAlmostEqual(p.metrics(['x', 'a'], ['a'])['ndcg5'], 1/p.math.log2(3))

    def test_set_metrics_and_empty_target(self):
        self.assertAlmostEqual(p.metrics(['a', 'b', 'b'], ['b', 'c'])['jaccard'], 1/3)
        with self.assertRaises(ValueError):
            p.metrics(['a'], [])
        row = p.prediction_row('0', [], [])
        with self.assertRaises(ValueError):
            p.summarize_predictions([row])

    def test_ablation_preserves_events_and_stable_labels(self):
        line = ['<|history|>', '0', '<|time1|>', '4', '4', '<|time2|>', '6', '<|endofhistory|>']
        expected = [x for x in line if not x.startswith('<|time')]
        for variant in ['full', 'same_time', 'no_time', 'per_ego_permute']:
            transformed = p.transform_time(line, variant, '0', 13)
            self.assertEqual([x for x in transformed if not x.startswith('<|time')], expected)
        first = p.transform_time(['<|time1|>'], 'per_ego_permute', '0', 13)
        later = p.transform_time(['<|time1|>', '<|time12|>'], 'per_ego_permute', '0', 13)
        self.assertEqual(first[0], later[0])

    def test_tokenizer_matches_released_ids(self):
        for name in p.SPECS:
            root = ROOT/'repo_clone'
            tok = p.NodeTokenizer(json.loads((root/'vocabs'/name/'12/vocab.json').read_text()), 12)
            released_json = json.loads((root/'tokenizers'/name/'12/tokenizer.json').read_text())
            released = dict(released_json['model']['vocab'])
            released.update({x['content']: x['id'] for x in released_json['added_tokens']})
            text = (root/'resources'/name/'12/train.link_prediction').read_text().splitlines()[0]
            self.assertEqual(tok.encode(text.split()), [released[x] for x in text.split()])
            self.assertEqual(tok.vocab, released)

    def test_setup_does_not_install_rust_or_hf_packages(self):
        nb = json.loads((ROOT/'notebooks/SimpleDyG_Kaggle_Full_Pipeline.ipynb').read_text(encoding='utf-8'))
        parsed = ast.parse(''.join(nb['cells'][3]['source']))
        requirements = next(ast.literal_eval(x.value) for x in parsed.body if isinstance(x, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id=='requirements' for t in x.targets))
        self.assertNotIn('transformers', requirements)
        self.assertNotIn('tokenizers', requirements)
        self.assertIn('--only-binary=:all:', ''.join(nb['cells'][3]['source']))

    def test_embedded_gelu_matches_original_math(self):
        import importlib.util
        if importlib.util.find_spec('transformers') is None:
            self.skipTest('Reference HF package intentionally absent; compared in Python 3.12 environment')
        from transformers.activations import gelu_new
        x = p.torch.linspace(-10, 10, 101, dtype=p.torch.float64, requires_grad=True)
        p.torch.testing.assert_close(p.gelu_new(x), gelu_new(x), rtol=0, atol=0)

    def test_legacy_adamw_and_scheduler_match_reference(self):
        import importlib.util
        if importlib.util.find_spec('transformers') is None:
            self.skipTest('Reference HF package intentionally absent; compared in Python 3.12 environment')
        from transformers import AdamW, get_linear_schedule_with_warmup
        for correct_bias in [True, False]:
            a = p.nn.Parameter(p.torch.tensor([.2, -.3, .4], dtype=p.torch.float64))
            b = p.nn.Parameter(a.detach().clone())
            reference = AdamW([a], lr=.01, eps=1e-8, weight_decay=.03, correct_bias=correct_bias, no_deprecation_warning=True)
            embedded = p.AuthorAdamW([b], lr=.01, eps=1e-8, weight_decay=.03, correct_bias=correct_bias)
            ra = get_linear_schedule_with_warmup(reference, 2, 8)
            rb = p.linear_warmup_decay(embedded, 2, 8)
            for i in range(10):
                a.grad = p.torch.tensor([1e-10, .13*(i+1), -.23], dtype=p.torch.float64)
                b.grad = a.grad.clone()
                reference.step()
                embedded.step()
                ra.step()
                rb.step()
                p.torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertEqual(ra.get_last_lr(), rb.get_last_lr())

    def test_author_model_matches_original_forward_and_gradients(self):
        import importlib.util
        if importlib.util.find_spec('transformers') is None:
            self.skipTest('Reference HF package intentionally absent; compared in Python 3.12 environment')
        namespace = 'simpledyg_original_reference'
        path = ROOT/'repo_clone/models'
        spec = importlib.util.spec_from_file_location(namespace, path/'__init__.py', submodule_search_locations=[str(path)])
        reference = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = reference
        spec.loader.exec_module(reference)
        Config, Model = p.load_author_model(ROOT/'repo_clone')
        kwargs = dict(vocab_size=64, n_positions=32, n_ctx=32, n_embd=16, n_layer=2, n_head=2)
        p.seed_all(5)
        a = reference.GPT2LMHeadModel(reference.GPT2Config(**kwargs)).eval()
        p.seed_all(5)
        b = Model(Config(**kwargs)).eval()
        for key in a.state_dict():
            p.torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
        inputs = p.torch.tensor([[1, 4, 5, 8, 2], [3, 6, 9, 5, 2]])
        out_a = a(inputs, labels=inputs)
        out_b = b(inputs, labels=inputs)
        p.torch.testing.assert_close(out_a[0], out_b[0], rtol=0, atol=0)
        p.torch.testing.assert_close(out_a[1], out_b[1], rtol=0, atol=0)
        out_a[0].backward()
        out_b[0].backward()
        for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
            p.torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)

    def test_full_data_scope_and_strict_history_cutoff(self):
        cfg = p.make_config('pilot')
        _, Sampler = p.load_dyglib(ROOT/'tmp/inspection/DyGLib')
        for name, counts in [('UCI_13', (1708, 146, 110)), ('ML_10M_13', (13701, 738, 396))]:
            ds = p.load_dataset(ROOT/'repo_clone', name, cfg)
            self.assertEqual(tuple(len(ds[k]) for k in ['train', 'val', 'test']), counts)
            sampler = p.build_sampler(ds, cfg, Sampler)
            cutoff = float(ds['raw'].loc[ds['raw'].timestamp==11, 'ts'].min())/cfg['time_unit_seconds']
            for node in range(1, len(ds['vocab'])+1):
                _, _, times, _ = sampler.find_neighbors_before(node, cutoff)
                self.assertTrue((times < cutoff).all())

    def test_training_negatives_do_not_consult_test_labels(self):
        cfg = p.make_config('smoke')
        ds = p.load_dataset(ROOT/'repo_clone', 'UCI_13', cfg)
        pairs, blocked = p.baseline_training_edges(ds, cfg)
        self.assertTrue(all(bin_id <= 10 for _, bin_id in blocked))
        for u, v, _, b in pairs:
            self.assertIn(u, blocked[(u, b)])
            self.assertIn(v, blocked[(u, b)])

    def test_run_fingerprint_separates_data_and_configuration(self):
        cfg = p.make_config('smoke')
        ds = p.load_dataset(ROOT/'repo_clone', 'UCI_13', cfg)
        original = p.run_id(ds, 'SimpleDyG', 'full', 0, cfg)
        changed = dict(cfg, max_new_tokens=9)
        self.assertNotEqual(original, p.run_id(ds, 'SimpleDyG', 'full', 0, changed))
        self.assertNotEqual(original, p.run_id(ds, 'SimpleDyG', 'no_time', 0, cfg))

    def test_notebook_code_and_clean_outputs(self):
        import nbformat
        notebook = nbformat.read(ROOT/'notebooks/SimpleDyG_Kaggle_Full_Pipeline.ipynb', as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)

    def test_offline_bundle_checksums(self):
        import hashlib
        with zipfile.ZipFile(ROOT/'notebooks/SimpleDyG_Kaggle_Inputs.zip') as archive:
            manifest = json.loads(archive.read('SOURCE_MANIFEST.json'))
            self.assertEqual(manifest['author_commit'], p.AUTHOR_COMMIT)
            self.assertEqual(manifest['dyglib_commit'], p.DYGLIB_COMMIT)
            for name, expected in manifest['sha256'].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
