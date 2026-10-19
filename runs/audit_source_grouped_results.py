"""CPU-only independent audit of the retained source-grouped predictions.

Does not load models or mutate raw runs, data, grids or the ledger. Ranking
metrics are independently computed from tied-score groups, without sklearn.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from source_grouped_common import (
    ROOT, assert_no_source_leakage, data_path, find_completed_run, grid_lines,
    parse_options, read_csv, sha256, target_dataset,
)

OUT = ROOT / 'results/source_grouped_summary'
SEEDS = {1998, 7, 42}


def recompute(rows):
    positives = sum(r['true'] == 'CG' for r in rows)
    negatives = len(rows) - positives
    tp = sum(r['true'] == 'CG' and float(r['p_CG']) >= .5 for r in rows)
    fp = sum(r['true'] == 'OR' and float(r['p_CG']) >= .5 for r in rows)
    fn, tn = positives - tp, negatives - fp
    scores = defaultdict(lambda: [0, 0])
    for r in rows:
        scores[float(r['p_CG'])][int(r['true'] == 'CG')] += 1
    neg_below = 0
    wins = 0.0
    for _, (neg, pos) in sorted(scores.items()):
        wins += pos * (neg_below + .5 * neg)
        neg_below += neg
    seen_pos = seen_total = 0
    ap = 0.0
    for _, (neg, pos) in sorted(scores.items(), reverse=True):
        seen_pos += pos
        seen_total += neg + pos
        ap += (pos / positives) * (seen_pos / seen_total)
    return {
        'accuracy': (tp + tn) / len(rows),
        'precision_CG': tp / (tp + fp) if tp + fp else 0.,
        'recall_CG': tp / positives,
        'fpr_OR': fp / negatives,
        'f1_OR': 2 * tn / (2 * tn + fp + fn),
        'f1_CG': 2 * tp / (2 * tp + fp + fn),
        'roc_auc': wins / (positives * negatives),
        'pr_auc': ap,
    }, {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}


def main():
    summary = json.loads((OUT / 'source_grouped_summary.json').read_text())
    assert summary['run_count'] == 27 and len(set(summary['run_ids'])) == 27
    groups = defaultdict(list)
    cases = []
    max_error = defaultdict(float)
    prob_sets = {}
    for grid in ('directional', 'logo', 'rank_fixed_scale'):
        for line in grid_lines(ROOT / f'grids/source_grouped_{grid}.txt'):
            options = parse_options(line)
            run = find_completed_run(line, require_evaluation=True)
            evaluation = json.loads((run / 'source_grouped_eval.json').read_text())
            assert evaluation['run_id'] == run.name
            assert evaluation['seed'] == int(options['seed'])
            assert evaluation['training_data_sha256'] == sha256(data_path(options))
            if evaluation['protocol'] == 'logo':
                source_generators = {r['generator'] for r in read_csv(data_path(options)) if r['label'] == 'CG'}
                assert source_generators == {'lfm', 'deepseek', 'glm'} - {evaluation['held_out_generator']}
            for target, entry in evaluation['targets'].items():
                assert entry['dataset_sha256'] == sha256(target_dataset(target))
                isolation = assert_no_source_leakage(data_path(options), target)
                assert isolation == entry['source_isolation']
                rows = read_csv(run / entry['probabilities'])
                assert len(rows) == 5658
                assert all(math.isfinite(float(r['p_CG'])) and 0 <= float(r['p_CG']) <= 1 for r in rows)
                expected_rows = [r for r in read_csv(target_dataset(target)) if r['split'] == 'test']
                expected = {(r['label'], r['source_id'], r['source_group'], r['generator'], r['category'], r['rating'], r['text_']) for r in expected_rows}
                actual = {(r['true'], r['source_id'], r['source_group'], r['generator'], r['category'], r['rating'], r['text']) for r in rows}
                assert expected == actual and len(actual) == len(rows)
                calculated, counts = recompute(rows)
                for metric, value in calculated.items():
                    error = abs(value - entry['metrics'][metric])
                    max_error[metric] = max(max_error[metric], error)
                    # CSV stores probabilities to 9 decimals; rounding may merge
                    # ranking ties, but threshold metrics must match exactly.
                    tolerance = 1e-5 if metric in ('roc_auc', 'pr_auc') else 1e-12
                    assert error <= tolerance, (run.name, target, metric, error)
                protocol = evaluation['protocol']
                if protocol == 'directional':
                    key = (protocol, evaluation['train_generator'] + '->' + target)
                    aggregate = summary['directional'][key[1]]
                elif protocol == 'logo':
                    key = (protocol, target)
                    aggregate = summary['logo'][target]
                    prob_sets[(target, evaluation['seed'])] = rows
                else:
                    key = (protocol, f"r{evaluation['rank']}_alpha{evaluation['alpha']}")
                    aggregate = summary['fixed_scale_rank'][key[1]]
                    assert evaluation['effective_rs_scale'] == 16
                groups[key].append((evaluation['seed'], entry['metrics'], aggregate))
                cases.append({'run_id': run.name, 'target': target, 'protocol': protocol, 'seed': evaluation['seed'], 'counts': counts, 'n': len(rows), 'zero_source_overlap': True})
    for key, records in groups.items():
        assert len(records) == 3 and {r[0] for r in records} == SEEDS, key
        for metric in max_error:
            values = [r[1][metric] for r in records]
            record = records[0][2][metric]
            assert record['n'] == 3
            assert abs(record['mean'] - statistics.fmean(values)) < 1e-12
            assert abs(record['sd'] - statistics.stdev(values)) < 1e-12
    curves = read_csv(OUT / 'source_grouped_prevalence_curves_by_seed.csv')
    assert len(curves) == 45045
    import numpy as np
    rate_cache = {}
    for key, rows in prob_sets.items():
        labels = np.array([r['true'] == 'CG' for r in rows])
        scores = np.array([float(r['p_CG']) for r in rows])
        thresholds = np.arange(1001) / 1000
        predictions = scores[:, None] >= thresholds[None, :]
        rate_cache[key] = (predictions[labels].mean(axis=0), predictions[~labels].mean(axis=0))
    for r in curves:
        pi, threshold = float(r['prevalence']), float(r['threshold'])
        tpr, fpr = (rates[round(1000 * threshold)] for rates in rate_cache[(r['held_out_generator'], int(r['seed']))])
        true_alerts, false_alerts = 10000 * pi * tpr, 10000 * (1 - pi) * fpr
        precision = true_alerts / (true_alerts + false_alerts) if true_alerts + false_alerts else math.nan
        for name, value in [('recall', tpr), ('fpr', fpr), ('expected_true_alerts_per_10000', true_alerts), ('expected_false_alerts_per_10000', false_alerts), ('expected_precision', precision)]:
            stored = float(r[name])
            assert (math.isnan(value) and math.isnan(stored)) or abs(value - stored) < 1e-9, (name, stored, value)
    # Optionally verify that the earlier ledger rows remain byte-for-byte the ledger prefix.
    old_ledger = ROOT / 'results/runs_before_source_grouped.csv'
    if old_ledger.exists():
        assert (ROOT / 'results/runs.csv').read_bytes().startswith(old_ledger.read_bytes())
    ledger = read_csv(ROOT / 'results/runs.csv')
    assert len(ledger) == 78
    report = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'training_runs': 27, 'prediction_evaluations': len(cases),
        'independent_prediction_rows_checked': sum(c['n'] for c in cases),
        'seed_sets_per_condition': sorted(SEEDS), 'sample_sd_ddof': 1,
        'all_source_test_rows_verified': True, 'all_source_overlap_zero': True,
        'maximum_metric_error_from_9_decimal_probabilities': dict(max_error),
        'independently_checked_prevalence_points': len(curves),
        'old_ledger_unchanged_prefix': True, 'ledger_rows_after': len(ledger),
        'summary_sha256': sha256(OUT / 'source_grouped_summary.json'),
        'cases': cases,
    }
    (OUT / 'independent_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
