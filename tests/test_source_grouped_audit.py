"""Independent metric audit sanity checks, using only synthetic examples."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runs'))
from audit_source_grouped_results import recompute


class IndependentMetricTests(unittest.TestCase):
    def test_tied_scores_and_threshold(self):
        rows = [
            {'true': 'OR', 'p_CG': '.1'}, {'true': 'OR', 'p_CG': '.4'},
            {'true': 'CG', 'p_CG': '.4'}, {'true': 'CG', 'p_CG': '.8'},
        ]
        metrics, counts = recompute(rows)
        self.assertEqual(counts, {'TN': 2, 'FP': 0, 'FN': 1, 'TP': 1})
        self.assertEqual(metrics['accuracy'], .75)
        self.assertEqual(metrics['roc_auc'], .875)
        self.assertAlmostEqual(metrics['pr_auc'], 5 / 6)

    def test_half_threshold_is_positive(self):
        metrics, counts = recompute([
            {'true': 'OR', 'p_CG': '.499999999'},
            {'true': 'CG', 'p_CG': '.5'},
        ])
        self.assertEqual(metrics['accuracy'], 1.)
        self.assertEqual(counts['TP'], 1)

    def test_no_predicted_positives(self):
        metrics, counts = recompute([
            {'true': 'OR', 'p_CG': '.1'}, {'true': 'CG', 'p_CG': '.2'},
        ])
        self.assertEqual(metrics['precision_CG'], 0.)
        self.assertEqual(metrics['recall_CG'], 0.)
        self.assertEqual(metrics['roc_auc'], 1.)
        self.assertEqual(counts['FN'], 1)


if __name__ == '__main__':
    unittest.main()
