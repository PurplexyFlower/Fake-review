"""Aggregate the 27 source-grouped directional, LOGO, and rank runs."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from source_grouped_common import ROOT, grid_lines, parse_options, find_completed_run, read_csv


RESULTS = ROOT / "results"
EXPERIMENT_GRIDS = (
    ROOT / "grids/source_grouped_directional.txt",
    ROOT / "grids/source_grouped_logo.txt",
    ROOT / "grids/source_grouped_rank_fixed_scale.txt",
)
PREVALENCES = (0.001, 0.005, 0.01, 0.05, 0.10)
THRESHOLDS = tuple(index / 1000 for index in range(1001))


def mean_sd(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def probability_rows(run_dir: Path, target: str) -> list[dict[str, str]]:
    path = run_dir / f"source_grouped_eval_{target}.csv"
    if not path.exists():
        raise RuntimeError(f"missing retained held-out probabilities: {path}")
    return read_csv(path)


def score_arrays(rows: list[dict[str, str]]) -> tuple[list[float], list[float]]:
    positives = sorted(float(row["p_CG"]) for row in rows if row["true"] == "CG")
    negatives = sorted(float(row["p_CG"]) for row in rows if row["true"] == "OR")
    if not positives or not negatives:
        raise RuntimeError("prevalence curve requires both OR and CG probabilities")
    return positives, negatives


def curve_point(
    positives: list[float],
    negatives: list[float],
    threshold: float,
    prevalence: float,
) -> dict[str, float]:
    tpr = (len(positives) - bisect.bisect_left(positives, threshold)) / len(positives)
    fpr = (len(negatives) - bisect.bisect_left(negatives, threshold)) / len(negatives)
    true_alerts = 10000 * prevalence * tpr
    false_alerts = 10000 * (1 - prevalence) * fpr
    alerts = true_alerts + false_alerts
    precision = true_alerts / alerts if alerts else math.nan
    return {
        "recall": tpr,
        "fpr": fpr,
        "expected_precision": precision,
        "expected_alerts_per_10000": alerts,
        "expected_true_alerts_per_10000": true_alerts,
        "expected_false_alerts_per_10000": false_alerts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(RESULTS))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [line for grid in EXPERIMENT_GRIDS for line in grid_lines(grid)]
    if len(lines) != 27:
        raise RuntimeError(f"expected 27 source-grouped runs, found {len(lines)}")

    directional_rows: list[dict[str, object]] = []
    logo_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    logo_probability_sets: list[tuple[str, int, Path, list[dict[str, str]]]] = []
    run_ids: list[str] = []
    for line in lines:
        options = parse_options(line)
        run_dir = find_completed_run(line, require_evaluation=True)
        summary = json.loads(
            (run_dir / "source_grouped_eval.json").read_text(encoding="utf-8")
        )
        run_ids.append(run_dir.name)
        for target, evaluation in summary["targets"].items():
            metric = evaluation["metrics"]
            record = {
                "run_id": run_dir.name,
                "seed": int(str(options["seed"])),
                "target_generator": target,
                **metric,
            }
            if summary["protocol"] == "directional":
                record["train_generator"] = summary["train_generator"]
                directional_rows.append(record)
            elif summary["protocol"] == "logo":
                record["held_out_generator"] = summary["held_out_generator"]
                logo_rows.append(record)
                logo_probability_sets.append(
                    (
                        summary["held_out_generator"],
                        int(str(options["seed"])),
                        run_dir,
                        probability_rows(run_dir, target),
                    )
                )
            elif summary["protocol"] == "fixed_scale_rank":
                record["rank"] = int(summary["rank"])
                record["alpha"] = int(summary["alpha"])
                record["effective_rs_scale"] = float(summary["effective_rs_scale"])
                rank_rows.append(record)
            else:
                raise RuntimeError(f"unknown protocol in {run_dir}")

    if len(directional_rows) != 27 or len(logo_rows) != 9 or len(rank_rows) != 9:
        raise RuntimeError(
            "expected 27 directional cells, 9 LOGO cells, and 9 rank cells; got "
            f"{len(directional_rows)}, {len(logo_rows)}, and {len(rank_rows)}"
        )

    metric_names = [
        "accuracy",
        "precision_CG",
        "recall_CG",
        "f1_OR",
        "f1_CG",
        "fpr_OR",
        "roc_auc",
        "pr_auc",
    ]
    directional_aggregate: dict[str, object] = {}
    directional_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in directional_rows:
        directional_groups[(str(row["train_generator"]), str(row["target_generator"]))].append(row)
    for (train, target), group in sorted(directional_groups.items()):
        directional_aggregate[f"{train}->{target}"] = {
            name: mean_sd([float(row[name]) for row in group]) for name in metric_names
        }

    logo_aggregate: dict[str, object] = {}
    logo_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in logo_rows:
        logo_groups[str(row["held_out_generator"])].append(row)
    for target, group in sorted(logo_groups.items()):
        logo_aggregate[target] = {
            name: mean_sd([float(row[name]) for row in group]) for name in metric_names
        }

    rank_aggregate: dict[str, object] = {}
    rank_groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rank_rows:
        rank_groups[(int(row["rank"]), int(row["alpha"]))].append(row)
    for (rank, alpha), group in sorted(rank_groups.items()):
        rank_aggregate[f"r{rank}_alpha{alpha}"] = {
            "effective_rs_scale": alpha / (rank**0.5),
            **{
                name: mean_sd([float(row[name]) for row in group])
                for name in metric_names
            },
        }

    write_csv(
        output_dir / "source_grouped_directional_by_seed.csv",
        directional_rows,
        ["run_id", "train_generator", "target_generator", "seed", "n", "n_OR", "n_CG"]
        + metric_names,
    )
    write_csv(
        output_dir / "source_grouped_logo_by_seed.csv",
        logo_rows,
        ["run_id", "held_out_generator", "target_generator", "seed", "n", "n_OR", "n_CG"]
        + metric_names,
    )
    write_csv(
        output_dir / "source_grouped_fixed_scale_rank_by_seed.csv",
        rank_rows,
        [
            "run_id",
            "rank",
            "alpha",
            "effective_rs_scale",
            "target_generator",
            "seed",
            "n",
            "n_OR",
            "n_CG",
        ]
        + metric_names,
    )

    curve_by_seed: list[dict[str, object]] = []
    curve_groups: dict[tuple[str, float, float], list[dict[str, object]]] = defaultdict(list)
    for held_out, seed, run_dir, rows in logo_probability_sets:
        positives, negatives = score_arrays(rows)
        for prevalence in PREVALENCES:
            for threshold in THRESHOLDS:
                point = curve_point(positives, negatives, threshold, prevalence)
                record: dict[str, object] = {
                    "run_id": run_dir.name,
                    "held_out_generator": held_out,
                    "seed": seed,
                    "prevalence": prevalence,
                    "threshold": threshold,
                    **point,
                }
                curve_by_seed.append(record)
                curve_groups[(held_out, prevalence, threshold)].append(record)

    curve_fields = [
        "run_id",
        "held_out_generator",
        "seed",
        "prevalence",
        "threshold",
        "recall",
        "fpr",
        "expected_precision",
        "expected_alerts_per_10000",
        "expected_true_alerts_per_10000",
        "expected_false_alerts_per_10000",
    ]
    write_csv(output_dir / "source_grouped_prevalence_curves_by_seed.csv", curve_by_seed, curve_fields)

    curve_aggregate: list[dict[str, object]] = []
    curve_metrics = curve_fields[5:]
    for (held_out, prevalence, threshold), group in sorted(curve_groups.items()):
        record: dict[str, object] = {
            "held_out_generator": held_out,
            "prevalence": prevalence,
            "threshold": threshold,
            "n_seeds": len(group),
        }
        for name in curve_metrics:
            values = [float(row[name]) for row in group if not math.isnan(float(row[name]))]
            record[f"{name}_mean"] = statistics.fmean(values) if values else math.nan
            record[f"{name}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        curve_aggregate.append(record)
    aggregate_fields = ["held_out_generator", "prevalence", "threshold", "n_seeds"]
    for name in curve_metrics:
        aggregate_fields.extend([f"{name}_mean", f"{name}_sd"])
    write_csv(output_dir / "source_grouped_prevalence_curves.csv", curve_aggregate, aggregate_fields)

    threshold_points = [
        row for row in curve_aggregate if abs(float(row["threshold"]) - 0.5) < 1e-12
    ]
    write_csv(
        output_dir / "source_grouped_prevalence_at_threshold_0p5.csv",
        threshold_points,
        aggregate_fields,
    )

    summary = {
        "schema_version": 1,
        "protocol": "source-grouped-contemporary-v1",
        "run_count": len(run_ids),
        "run_ids": sorted(run_ids),
        "directional": directional_aggregate,
        "logo": logo_aggregate,
        "fixed_scale_rank": rank_aggregate,
        "prevalence_assumptions": list(PREVALENCES),
        "prevalence_curve_threshold_step": 0.001,
        "claim_boundary": (
            "strict source-independent internal evaluation across three contemporary "
            "generators; not independent external-platform validation"
        ),
    }
    (output_dir / "source_grouped_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "aggregated 27 runs: 27 directional cells, 9 LOGO cells, "
        "9 controlled-rank cells"
    )
    print(f"outputs -> {output_dir}")


if __name__ == "__main__":
    main()
