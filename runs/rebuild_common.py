"""New-data-only follow-up protocol. Never consumes the historical ledgers."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'grids/source_grouped_rebuild/plan.json'
OUT = ROOT / 'results/source_grouped_rebuild'
SEEDS = (1998, 7, 42)
DATA = 'dataset/source_grouped/modern_reviews_deepseek.csv'
MODELS = (
    ('roberta', 'FacebookAI/roberta-base', 'e2da8e2f811d1448a5b465c236feacd80ffbac7b', 32),
    ('deberta', 'microsoft/deberta-v3-base', '8ccc9b6f36199bec6961081d44eb72fb3f7353f3', 16),
    ('modernbert', 'answerdotai/ModernBERT-large', '45bb4654a4d5aaff24dd11d4781fa46d39bf8c13', 16),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def save_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def partitions(path):
    rows = read_rows(path)
    required = {'split', 'source_id', 'source_group', 'text_', 'label', 'category', 'rating', 'generator'}
    if not rows or not required <= rows[0].keys():
        raise RuntimeError(f'missing source-grouped columns: {path}')
    groups, sources, unique = {}, {}, set()
    result = {key: [] for key in ('train', 'validation', 'test')}
    for row in rows:
        part = row['split']
        if part not in result or row['label'] not in {'OR', 'CG'} or not row['text_'].strip():
            raise RuntimeError('invalid row or split')
        for mapping, key in ((groups, 'source_group'), (sources, 'source_id')):
            if mapping.setdefault(row[key], part) != part:
                raise RuntimeError(f'{key} crosses partitions')
        identity = (row['source_id'], row['label'])
        if identity in unique:
            raise RuntimeError('duplicate source/label row')
        unique.add(identity)
        result[part].append(row)
    for part, subset in result.items():
        counts = Counter(row['label'] for row in subset)
        if set(counts) != {'OR', 'CG'} or counts['OR'] != counts['CG']:
            raise RuntimeError(f'unbalanced or empty {part}')
    return result


def make_plan():
    jobs = []
    for shard, seed in enumerate(SEEDS, 1):
        base = {'shard': shard, 'seed': seed, 'data': DATA, 'target': 'deepseek'}
        pair = ['rs', 'standard'] if shard != 2 else ['standard', 'rs']
        for scaling in pair:
            jobs.append(base | {'id': f'pair_{scaling}_s{seed}', 'kind': 'adapter',
                'tag': 'sgrebpair', 'scaling': scaling, 'rank': 64, 'alpha': 128,
                'scheduler': 'cosine_with_restarts', 'label_smoothing': 0.01})
        for name, scheduler, smoothing in (
            ('cosine', 'cosine', 0.01), ('linear', 'linear', 0.01),
            ('no_smoothing', 'cosine_with_restarts', 0.0),
        ):
            jobs.append(base | {'id': f'{name}_s{seed}', 'kind': 'adapter',
                'tag': 'sgreb' + name.replace('_', ''), 'scaling': 'rs', 'rank': 64,
                'alpha': 128, 'scheduler': scheduler, 'label_smoothing': smoothing})
        for short, model, revision, batch in MODELS:
            jobs.append(base | {'id': f'{short}_s{seed}', 'kind': 'transformer',
                'model': model, 'revision': revision, 'batch': batch,
                'learning_rate': 2e-5, 'epochs': 3, 'max_seq': 512,
                'effective_batch': 32, 'weight_decay': 0.01, 'warmup_ratio': 0.06,
                'eval_steps': 200, 'patience': 3})
        for target in ('deepseek', 'lfm', 'deepseek_holdout', 'glm'):
            matched = target == 'deepseek'
            generator = 'deepseek' if target == 'deepseek_holdout' else target
            data = DATA if matched else f'dataset/source_grouped/pooled_wo_{generator}.csv'
            for estimator in ('logreg', 'linsvm'):
                jobs.append(base | {'id': f'tfidf_{target}_{estimator}_s{seed}',
                    'kind': 'tfidf', 'data': data, 'target': generator,
                    'evaluation': 'matched' if matched else 'logo',
                    'estimator': estimator, 'word_features': 200000,
                    'char_features': 200000, 'min_df': 2,
                    'C': 4.0 if estimator == 'logreg' else 0.5,
                    'max_iter': 2000 if estimator == 'logreg' else 5000})
    manifest = ROOT / 'dataset/source_grouped/manifest.json'
    data_files = sorted({job['data'] for job in jobs} | {
        f'dataset/source_grouped/modern_reviews_{name}.csv'
        for name in ('lfm', 'deepseek', 'glm')})
    return {'schema_version': 1, 'protocol': 'source-grouped-followup-v1',
        'training_seeds': list(SEEDS), 'split_seed': 1998,
        'gpu_training_runs': 24, 'cpu_fits': 24, 'shards': 3,
        'source_manifest_sha256': sha256(manifest),
        'data_sha256': {name: sha256(ROOT / name) for name in data_files},
        'original_train_code_sha256': sha256(ROOT / 'runs/train_run.py'),
        'baseline_learning_rate_selection': 'fixed a priori, no new LR sweep or test selection',
        'resource_pairing': 'both scaling methods rerun on the same GPU/host for each seed',
        'jobs': jobs}


def load_plan(path=PLAN):
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    if plan['protocol'] != 'source-grouped-followup-v1':
        raise RuntimeError('wrong follow-up protocol')
    if len(plan['jobs']) != 48 or len({job['id'] for job in plan['jobs']}) != 48:
        raise RuntimeError('expected exactly 24 GPU runs and 24 CPU fits')
    for shard, seed in enumerate(SEEDS, 1):
        jobs = [j for j in plan['jobs'] if j['shard'] == shard]
        if len(jobs) != 16 or any(j['seed'] != seed for j in jobs):
            raise RuntimeError('shard/seed mismatch')
        if Counter(j['kind'] for j in jobs) != {'adapter': 5, 'transformer': 3, 'tfidf': 8}:
            raise RuntimeError('incorrect shard composition')
    if sha256(ROOT / 'dataset/source_grouped/manifest.json') != plan['source_manifest_sha256']:
        raise RuntimeError('source manifest changed')
    if sha256(ROOT / 'runs/train_run.py') != plan['original_train_code_sha256']:
        raise RuntimeError('frozen adapter training code changed')
    for name, expected in plan['data_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f'new-data hash changed: {name}')
        partitions(ROOT / name)
    return plan


def get_job(plan, job_id):
    return next(job for job in plan['jobs'] if job['id'] == job_id)


def adapter_args(job):
    return ['--tag', job['tag'], '--scaling', job['scaling'], '--rank', str(job['rank']),
        '--alpha', str(job['alpha']), '--seed', str(job['seed']),
        '--scheduler', job['scheduler'], '--label-smoothing', str(job['label_smoothing']),
        '--data', job['data'], '--predefined-splits']


def config_for(plan, job):
    return {'protocol': plan['protocol'], 'job': job, 'plan_sha256': sha256(PLAN),
        'source_manifest_sha256': plan['source_manifest_sha256'],
        'training_data_sha256': plan['data_sha256'][job['data']],
        'target_data_sha256': plan['data_sha256'][
            f"dataset/source_grouped/modern_reviews_{job['target']}.csv"]}


def complete(plan, job):
    directory = OUT / job['id']
    marker = directory / 'COMPLETE.json'
    if not marker.is_file():
        return False
    receipt = json.loads(marker.read_text())
    if receipt['config'] != config_for(plan, job):
        return False
    if json.loads((directory / 'config.json').read_text()) != receipt['config']:
        return False
    required = {'config.json', 'metrics.json', 'test_probs.csv', 'val_probs.csv',
                'runtime_environment.json'}
    files = receipt['files']
    if not required <= files.keys():
        return False
    if job['kind'] == 'adapter' and not {'adapter/adapter_config.json',
            'adapter/adapter_model.safetensors'} <= files.keys():
        return False
    if job['kind'] == 'transformer' and not {'model/config.json', 'model/model.safetensors'} <= files.keys():
        return False
    if job['kind'] == 'tfidf' and 'model.joblib' not in files:
        return False
    for name, entry in files.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            return False
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            return False
    training = partitions(ROOT / job['data'])
    target = partitions(ROOT / f"dataset/source_grouped/modern_reviews_{job['target']}.csv")
    result = json.loads((directory / 'metrics.json').read_text())
    for name, expected in (('val', training['validation']), ('test', target['test'])):
        predictions = read_rows(directory / f'{name}_probs.csv')
        expected_keys = {(r['source_id'], r['source_group'], r['label'], r['text_'],
                          r['category'], r['rating'], r['generator'], r['split']) for r in expected}
        actual_keys = {(r['source_id'], r['source_group'], r['true'], r['text'],
                        r['category'], r['rating'], r['generator'], r['split']) for r in predictions}
        if len(predictions) != len(expected) or len(actual_keys) != len(predictions) or actual_keys != expected_keys:
            return False
        values = [float(r['p_CG']) for r in predictions]
        calculated = metrics([{'label': r['true']} for r in predictions], values)
        for metric, value in calculated.items():
            tolerance = 1e-5 if metric in {'roc_auc', 'average_precision'} else 1e-12
            if abs(result[name][metric] - value) > tolerance:
                return False
    return True


def finish(plan, job):
    directory = OUT / job['id']
    files = {path.relative_to(directory).as_posix(): {'bytes': path.stat().st_size,
                'sha256': sha256(path)}
        for path in sorted(directory.rglob('*')) if path.is_file() and path.name != 'COMPLETE.json'}
    save_json(directory / 'COMPLETE.json', {'config': config_for(plan, job), 'files': files}, exclusive=True)
    if not complete(plan, job):
        raise RuntimeError('completion verification failed')


def metrics(rows, probabilities):
    # Independent tied-score ROC/AP; no dependence on old ledger metrics.
    from collections import defaultdict
    if len(rows) != len(probabilities) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise RuntimeError('invalid prediction export')
    labels = [r['label'] == 'CG' for r in rows]
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        raise RuntimeError('both classes required')
    tp = sum(y and p >= .5 for y, p in zip(labels, probabilities))
    fp = sum(not y and p >= .5 for y, p in zip(labels, probabilities))
    fn, tn = positives - tp, negatives - fp
    scores = defaultdict(lambda: [0, 0])
    for y, p in zip(labels, probabilities):
        scores[p][int(y)] += 1
    lower, wins = 0, 0.
    for _, (neg, pos) in sorted(scores.items()):
        wins += pos * (lower + .5 * neg)
        lower += neg
    seen_pos = seen_total = 0
    ap = 0.
    for _, (neg, pos) in sorted(scores.items(), reverse=True):
        seen_pos += pos
        seen_total += pos + neg
        ap += (pos / positives) * seen_pos / seen_total
    return {'n': len(labels), 'n_OR': negatives, 'n_CG': positives,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'accuracy': (tp + tn) / len(labels),
        'precision_CG': tp / (tp + fp) if tp + fp else 0., 'recall_CG': tp / positives,
        'fpr_OR': fp / negatives, 'f1_OR': 2 * tn / (2 * tn + fn + fp),
        'f1_CG': 2 * tp / (2 * tp + fn + fp),
        'roc_auc': wins / (positives * negatives), 'average_precision': ap}


def write_predictions(path, rows, probabilities):
    columns = ['row', 'true', 'p_CG', 'category', 'rating', 'text',
               'source_id', 'source_group', 'generator', 'split']
    with Path(path).open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i, (row, probability) in enumerate(zip(rows, probabilities, strict=True)):
            record = {key: row[key] for key in columns if key in row}
            record.update(row=i, true=row['label'], text=row['text_'], p_CG=f'{probability:.12f}')
            writer.writerow(record)


def assert_isolated(training_rows, evaluation_rows):
    if {r['source_group'] for r in training_rows} & {r['source_group'] for r in evaluation_rows}:
        raise RuntimeError('source group leakage into evaluation')
