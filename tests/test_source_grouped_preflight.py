"""CPU-only contract checks. Synthetic fixtures never enter real results."""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs"))

import aggregate_source_grouped_results as aggregate
import run_queue as queue
import source_grouped_common as common
import validate_source_grouped_plan as plan


class SourceGroupedPreflightTests(unittest.TestCase):
    def test_full_plan_has_three_seeds_per_condition(self):
        grids = [ROOT / f"grids/source_grouped_{kind}.txt"
                 for kind in ("directional", "logo", "rank_fixed_scale")]
        lines = [line for grid in grids for line in common.grid_lines(grid)]
        self.assertEqual(len(lines), 27)
        conditions = {}
        for line in lines:
            options = common.parse_options(line)
            plan.validate_line(line)
            key = (options["tag"], options["rank"], options.get("alpha"))
            conditions.setdefault(key, []).append(int(str(options["seed"])))
        self.assertEqual(len(conditions), 9)
        for seeds in conditions.values():
            self.assertEqual(sorted(seeds), [7, 42, 1998])
        shards = [common.grid_lines(ROOT / f"grids/source_grouped/instance_{n}_full.txt")
                  for n in (1, 2, 3)]
        self.assertEqual([len(shard) for shard in shards], [9, 9, 9])
        self.assertCountEqual(lines, [line for shard in shards for line in shard])

    def test_controlled_rank_holds_effective_scale_fixed(self):
        lines = common.grid_lines(ROOT / "grids/source_grouped_rank_fixed_scale.txt")
        cells = set()
        for line in lines:
            options = common.parse_options(line)
            rank, alpha = int(str(options["rank"])), int(str(options["alpha"]))
            self.assertEqual(alpha / math.sqrt(rank), 16.0)
            cells.add((rank, alpha))
        self.assertEqual(cells, {(16, 64), (36, 96), (64, 128)})

    def test_sample_standard_deviation(self):
        result = aggregate.mean_sd([0.8, 0.9, 1.0])
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mean"], 0.9)
        self.assertAlmostEqual(result["sd"], 0.1)

    def test_rare_prevalence_expected_alert_precision(self):
        point = aggregate.curve_point([0.6, 0.9], [0.1, 0.8], 0.5, 0.01)
        self.assertEqual(point["recall"], 1.0)
        self.assertEqual(point["fpr"], 0.5)
        self.assertAlmostEqual(point["expected_precision"], 0.01 / 0.505)
        self.assertAlmostEqual(point["expected_true_alerts_per_10000"], 100)
        self.assertAlmostEqual(point["expected_false_alerts_per_10000"], 4950)

    def test_no_alerts_do_not_claim_perfect_precision(self):
        point = aggregate.curve_point([0.2], [0.1], 0.9, 0.01)
        self.assertTrue(math.isnan(point["expected_precision"]))
        self.assertEqual(point["expected_alerts_per_10000"], 0)

    def test_strict_completion_checks_and_legacy_flag_default(self):
        with tempfile.TemporaryDirectory(prefix="source_grouped_test_") as directory:
            root = Path(directory)
            data = root / "fixture.csv"
            data.write_bytes(b"fixture\n")
            line = "--tag fixture --scaling rs --rank 64 --seed 7 --data fixture.csv"
            run_id = "fixture_rs_r64_s7_10031200"
            run_dir = root / "results" / run_id
            (run_dir / "adapter").mkdir(parents=True)
            for relative in ("metrics.json", "test_probs.csv", "adapter/adapter_config.json",
                             "adapter/adapter_model.safetensors"):
                (run_dir / relative).write_bytes(b"synthetic test fixture")
            config = {"tag": "fixture", "scaling": "rs", "rank": 64, "alpha": 128,
                      "seed": 7, "scheduler": "cosine_with_restarts",
                      "label_smoothing": 0.01, "holdout_category": None,
                      "data_sha256": common.sha256(data)}
            config_path = run_dir / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(queue, "ROOT", root):
                self.assertTrue(queue.is_completed(line, [{"run_id": run_id}], True))
                self.assertFalse(queue.is_completed(
                    line + " --predefined-splits", [{"run_id": run_id}], True))
                config["predefined_splits"] = True
                config_path.write_text(json.dumps(config), encoding="utf-8")
                self.assertTrue(queue.is_completed(
                    line + " --predefined-splits", [{"run_id": run_id}], True))
                (run_dir / "adapter/adapter_model.safetensors").unlink()
                self.assertFalse(queue.is_completed(
                    line + " --predefined-splits", [{"run_id": run_id}], True))

    def test_leaked_source_group_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="source_grouped_test_") as directory:
            training = Path(directory) / "train.csv"
            with training.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_group", "split"])
                writer.writeheader()
                writer.writerow({"source_group": "source-a", "split": "train"})
            target_rows = [{"source_group": "source-a", "label": "OR"},
                           {"source_group": "source-a", "label": "CG"}]
            with patch.object(common, "test_rows", return_value=target_rows):
                with self.assertRaisesRegex(RuntimeError, "source leakage"):
                    common.assert_no_source_leakage(training, "lfm")


if __name__ == "__main__":
    unittest.main()
