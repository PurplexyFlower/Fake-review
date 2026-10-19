"""CPU-only checks. Synthetic fixtures are not scientific experiments."""
import csv
import json
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runs'))
import rebuild_common as common
import rebuild_baseline as baseline
import rebuild_worker as worker


def write_fixture(path, *, leaking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, part in enumerate(('train', 'validation', 'test')):
        for label in ('OR', 'CG'):
            rows.append({'source_id': str(group), 'source_group': '0' if leaking else str(group),
                'split': part, 'label': label, 'text_': f'{part} {label} fixture',
                'category': 'Books', 'rating': '5.0', 'generator': 'or' if label == 'OR' else 'deepseek'})
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


class RebuildTests(unittest.TestCase):
    def test_exact_24_gpu_and_24_cpu_scope(self):
        plan = common.make_plan()
        self.assertEqual(len(plan['jobs']), 48)
        self.assertEqual(Counter(j['kind'] for j in plan['jobs']),
                         {'adapter': 15, 'transformer': 9, 'tfidf': 24})
        for shard, seed in enumerate(common.SEEDS, 1):
            jobs = worker.select_jobs(plan, shard, 'gpu')
            self.assertEqual(len(jobs), 8)
            self.assertEqual({j['seed'] for j in jobs}, {seed})
            paired = [j for j in jobs if j['id'].startswith('pair_')]
            self.assertEqual({j['scaling'] for j in paired}, {'standard', 'rs'})
            self.assertEqual(len(worker.select_jobs(plan, shard, 'cpu')), 8)

    def test_no_historical_data_and_pinned_models(self):
        for job in common.make_plan()['jobs']:
            self.assertTrue(job['data'].startswith('dataset/source_grouped/'))
            if job['kind'] == 'transformer':
                self.assertRegex(job['revision'], r'^[0-9a-f]{40}$')
                self.assertEqual(job['learning_rate'], 2e-5)
                self.assertEqual(job['effective_batch'], 32)
            elif job['kind'] == 'adapter':
                args = common.adapter_args(job)
                self.assertIn('--predefined-splits', args)
                self.assertTrue(job['tag'].startswith('sgreb'))

    def test_grouped_split_and_reject_source_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fixture.csv'
            write_fixture(path)
            self.assertEqual({k: len(v) for k, v in common.partitions(path).items()},
                             {'train': 2, 'validation': 2, 'test': 2})
            write_fixture(path, leaking=True)
            with self.assertRaisesRegex(RuntimeError, 'crosses partitions'):
                common.partitions(path)

    def test_metric_counts_and_tied_roc_ap(self):
        labels = [{'label': 'OR'}, {'label': 'CG'}, {'label': 'OR'}, {'label': 'CG'}]
        perfect = common.metrics(labels, [.1, .9, .2, .8])
        self.assertEqual(perfect['TN'], 2)
        self.assertEqual(perfect['TP'], 2)
        self.assertEqual(perfect['accuracy'], 1.)
        self.assertEqual(perfect['roc_auc'], 1.)
        self.assertEqual(perfect['average_precision'], 1.)
        tied = common.metrics(labels, [.5] * 4)
        self.assertEqual(tied['roc_auc'], .5)
        self.assertEqual(tied['average_precision'], .5)
        with self.assertRaises(RuntimeError):
            common.metrics(labels, [.1, .2, .3, math.nan])

    def test_completion_requires_weights_provenance_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'dataset/source_grouped/modern_reviews_deepseek.csv'
            write_fixture(data)
            manifest = root / 'dataset/source_grouped/manifest.json'
            manifest.write_text('{}')
            plan_path = root / 'plan.json'
            plan_path.write_text('{}')
            job = {'id': 'fixture_s1998', 'data': data.relative_to(root).as_posix(),
                   'target': 'deepseek', 'kind': 'transformer'}
            plan = {'protocol': 'source-grouped-followup-v1',
                    'source_manifest_sha256': common.sha256(manifest),
                    'data_sha256': {job['data']: common.sha256(data)}}
            out = root / 'result'
            directory = out / job['id']
            directory.mkdir(parents=True)
            with patch.object(common, 'ROOT', root), patch.object(common, 'OUT', out), \
                 patch.object(common, 'PLAN', plan_path):
                self.assertFalse(common.complete(plan, job))
                common.save_json(directory / 'config.json', common.config_for(plan, job))
                common.save_json(directory / 'runtime_environment.json', {'synthetic': True})
                partitions = common.partitions(data)
                result = {}
                for name, part in (('val', 'validation'), ('test', 'test')):
                    rows = partitions[part]
                    scores = [.1 if r['label'] == 'OR' else .9 for r in rows]
                    common.write_predictions(directory / f'{name}_probs.csv', rows, scores)
                    result[name] = common.metrics(rows, scores)
                common.save_json(directory / 'metrics.json', result)
                (directory / 'model').mkdir()
                (directory / 'model/config.json').write_text('{}')
                (directory / 'model/model.safetensors').write_bytes(b'synthetic test bytes, not a model')
                common.finish(plan, job)
                self.assertTrue(common.complete(plan, job))
                (directory / 'test_probs.csv').write_text('tampered')
                self.assertFalse(common.complete(plan, job))

    def test_logo_validation_excludes_held_out_generator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pooled = root / 'dataset/source_grouped/pooled_wo_deepseek.csv'
            write_fixture(pooled)
            target = root / 'dataset/source_grouped/modern_reviews_deepseek.csv'
            write_fixture(target)
            job = {'data': pooled.relative_to(root).as_posix(), 'target': 'deepseek', 'evaluation': 'logo'}
            with patch.object(baseline, 'ROOT', root):
                with self.assertRaisesRegex(RuntimeError, 'held-out generator'):
                    baseline.job_partitions(job)


if __name__ == '__main__':
    unittest.main()
