#!/usr/bin/env python3
"""Paired image-cluster bootstrap analysis for the CV801 3 x 3 grid.

The input contains per-image sufficient statistics so every bootstrap resample
recomputes micro object-link F1 and unsupported-claim share using the same
denominators as the frozen aggregate evaluator. Required columns are:

sample_id, writer, reviewer_tier, schema_valid, link_matched,
link_predicted, link_reference, visual_total, supported_claims,
unsupported_claims
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


WRITERS = ("2B", "4B", "9B")
TIERS = ("low", "mid", "high")
CELL_ORDER = tuple((writer, tier) for writer in WRITERS for tier in TIERS)
VALUE_COLUMNS = (
    "schema_valid",
    "link_matched",
    "link_predicted",
    "link_reference",
    "visual_total",
    "supported_claims",
    "unsupported_claims",
)
REQUIRED_COLUMNS = {"sample_id", "writer", "reviewer_tier", *VALUE_COLUMNS}
METRICS = (
    "independent_visual_total",
    "object_id_link_f1",
    "schema_valid_rate",
    "unsupported_claim_share",
)
METRIC_DIRECTIONS = {
    "independent_visual_total": "higher-is-better",
    "object_id_link_f1": "higher-is-better",
    "schema_valid_rate": "higher-is-better",
    "unsupported_claim_share": "lower-is-better",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_261_104)
    parser.add_argument("--expected-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=500)
    return parser.parse_args()


def read_input(
    path: Path,
) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    data: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {sorted(REQUIRED_COLUMNS)}")
        for line_number, row in enumerate(reader, 2):
            writer = row["writer"].strip()
            tier = row["reviewer_tier"].strip().lower()
            sample_id = row["sample_id"].strip()
            if writer not in WRITERS or tier not in TIERS or not sample_id:
                raise ValueError(f"Invalid grid key at line {line_number}: {row}")
            cell = (writer, tier)
            if sample_id in data[cell]:
                raise ValueError(
                    f"Duplicate cell/sample at line {line_number}: "
                    f"{writer}/{tier}/{sample_id}"
                )
            values = {name: float(row[name]) for name in VALUE_COLUMNS}
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite value at line {line_number}")
            if values["schema_valid"] not in {0.0, 1.0}:
                raise ValueError(f"schema_valid must be 0 or 1 at line {line_number}")
            if any(values[name] < 0 for name in VALUE_COLUMNS if name != "visual_total"):
                raise ValueError(f"Count/indicator values must be non-negative at line {line_number}")
            if not 0 <= values["visual_total"] <= 6:
                raise ValueError(f"visual_total must be in [0, 6] at line {line_number}")
            data[cell][sample_id] = values
    if not data:
        raise ValueError("Input CSV is empty")
    return data


def aligned_arrays(
    cells: dict[tuple[str, str], dict[str, dict[str, float]]],
    expected_samples: int,
) -> tuple[list[str], dict[tuple[str, str], dict[str, np.ndarray]]]:
    missing = [cell for cell in CELL_ORDER if cell not in cells]
    if missing:
        raise ValueError(f"Input is missing grid cells: {missing}")
    sample_sets = [set(cells[cell]) for cell in CELL_ORDER]
    common = set.intersection(*sample_sets)
    union = set.union(*sample_sets)
    if common != union:
        details = {f"{w}/{t}": len(cells[(w, t)]) for w, t in CELL_ORDER}
        raise ValueError(f"Input is not paired across cells: {details}")
    if len(common) != expected_samples:
        raise ValueError(
            f"Input has {len(common)} paired samples; expected {expected_samples}"
        )
    sample_ids = sorted(common)
    arrays = {
        cell: {
            column: np.asarray(
                [cells[cell][sample_id][column] for sample_id in sample_ids], dtype=float
            )
            for column in VALUE_COLUMNS
        }
        for cell in CELL_ORDER
    }
    return sample_ids, arrays


def safe_ratio(numerator: np.ndarray | float, denominator: np.ndarray | float):
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    result = np.full(np.broadcast_shapes(numerator_array.shape, denominator_array.shape), np.nan)
    return np.divide(numerator_array, denominator_array, out=result, where=denominator_array > 0)


def aggregate(
    metric: str,
    values: dict[str, np.ndarray],
    indices: np.ndarray | None = None,
) -> np.ndarray | float:
    def selected(column: str) -> np.ndarray:
        return values[column] if indices is None else values[column][indices]

    axis = None if indices is None else 1
    if metric == "independent_visual_total":
        return selected("visual_total").mean(axis=axis)
    if metric == "schema_valid_rate":
        return selected("schema_valid").mean(axis=axis)
    if metric == "object_id_link_f1":
        matched = selected("link_matched").sum(axis=axis)
        precision = safe_ratio(matched, selected("link_predicted").sum(axis=axis))
        recall = safe_ratio(matched, selected("link_reference").sum(axis=axis))
        denominator = precision + recall
        result = safe_ratio(2 * precision * recall, denominator)
        # Match the frozen evaluator contract: when precision and recall are
        # both defined but both zero, F1 is zero. It remains undefined only
        # when a component denominator (for example, predicted objects) is zero.
        both_defined_zero = np.isfinite(precision) & np.isfinite(recall) & (denominator == 0)
        return np.where(both_defined_zero, 0.0, result)
    if metric == "unsupported_claim_share":
        unsupported = selected("unsupported_claims").sum(axis=axis)
        total = unsupported + selected("supported_claims").sum(axis=axis)
        return safe_ratio(unsupported, total)
    raise ValueError(f"Unknown metric: {metric}")


def contrast_definitions() -> list[dict]:
    definitions: list[dict] = []
    for tier in TIERS:
        definitions.extend(
            [
                {
                    "name": f"writer_4B_minus_2B_at_{tier}",
                    "family": "exploratory_writer_effect",
                    "coefficients": {("4B", tier): 1, ("2B", tier): -1},
                },
                {
                    "name": f"writer_9B_minus_4B_at_{tier}",
                    "family": "exploratory_writer_effect",
                    "coefficients": {("9B", tier): 1, ("4B", tier): -1},
                },
            ]
        )
    for writer in WRITERS:
        definitions.extend(
            [
                {
                    "name": f"reviewer_mid_minus_low_at_{writer}",
                    "family": "exploratory_reviewer_effect",
                    "coefficients": {(writer, "mid"): 1, (writer, "low"): -1},
                },
                {
                    "name": f"reviewer_high_minus_mid_at_{writer}",
                    "family": "exploratory_reviewer_effect",
                    "coefficients": {(writer, "high"): 1, (writer, "mid"): -1},
                },
            ]
        )
    for writer in ("2B", "4B"):
        definitions.append(
            {
                "name": f"gap_closure_{writer}_vs_9B_high_minus_low",
                "family": "primary_interaction",
                "coefficients": {
                    (writer, "high"): 1,
                    ("9B", "high"): -1,
                    (writer, "low"): -1,
                    ("9B", "low"): 1,
                },
            }
        )
    return definitions


def observed_contrast(metric: str, arrays: dict, coefficients: dict) -> float | None:
    value = sum(
        coefficient * float(aggregate(metric, arrays[cell]))
        for cell, coefficient in coefficients.items()
    )
    if not np.isfinite(value):
        return None
    return value


def bootstrap_contrast(
    metric: str,
    arrays: dict,
    coefficients: dict,
    sample_count: int,
    iterations: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None, int]:
    estimates = np.empty(iterations, dtype=float)
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        indices = rng.integers(0, sample_count, size=(stop - start, sample_count))
        batch = np.zeros(stop - start, dtype=float)
        for cell, coefficient in coefficients.items():
            batch += coefficient * aggregate(metric, arrays[cell], indices)
        estimates[start:stop] = batch
    finite = estimates[np.isfinite(estimates)]
    if not finite.size:
        return None, None, 0
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high), int(finite.size)


def analyze(args: argparse.Namespace) -> dict:
    if args.iterations <= 0 or args.expected_samples <= 0 or args.batch_size <= 0:
        raise ValueError("iterations, expected-samples, and batch-size must be positive")
    sample_ids, arrays = aligned_arrays(read_input(args.input_csv), args.expected_samples)
    rng = np.random.default_rng(args.seed)
    result: dict = {
        "protocol": "cv801_writer_reviewer_paired_bootstrap_v2_micro_metrics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input_csv),
        "independent_unit": "Golden Test image",
        "paired_samples": len(sample_ids),
        "bootstrap_iterations": args.iterations,
        "seed": args.seed,
        "interval": "percentile 95% paired image-cluster bootstrap CI",
        "metric_note": (
            "Object-link F1 and unsupported-claim share are recomputed from "
            "resampled per-image sufficient statistics, preserving frozen micro denominators."
        ),
        "multiplicity_note": (
            "Two gap-closure interactions per endpoint are prespecified primary contrasts. "
            "Adjacent writer and reviewer contrasts are exploratory effect-size families."
        ),
        "metrics": {},
    }
    definitions = contrast_definitions()
    for metric in METRICS:
        cell_values = {}
        undefined_cells = {}
        for writer, tier in CELL_ORDER:
            key = f"{writer}/{tier}"
            value = float(aggregate(metric, arrays[(writer, tier)]))
            if np.isfinite(value):
                cell_values[key] = value
            else:
                cell_values[key] = None
                undefined_cells[key] = (
                    "The frozen micro metric is undefined because its required "
                    "denominator is zero."
                )
        contrasts = []
        for definition in definitions:
            estimate = observed_contrast(metric, arrays, definition["coefficients"])
            low, high, valid_iterations = bootstrap_contrast(
                metric,
                arrays,
                definition["coefficients"],
                len(sample_ids),
                args.iterations,
                args.batch_size,
                rng,
            )
            contrast = {
                "name": definition["name"],
                "family": definition["family"],
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "valid_bootstrap_iterations": valid_iterations,
            }
            if estimate is None or not valid_iterations:
                contrast["undefined_reason"] = (
                    "At least one required cell has an undefined frozen metric "
                    "denominator."
                )
            contrasts.append(contrast)
        result["metrics"][metric] = {
            "direction": METRIC_DIRECTIONS[metric],
            "cell_values": cell_values,
            "undefined_cells": undefined_cells,
            "contrasts": contrasts,
        }
    return result


def render_markdown(payload: dict) -> str:
    def shown(value: float | None) -> str:
        return "NA" if value is None else f"{value:.6f}"

    lines = [
        "# CV801 Writer x Reviewer paired analysis",
        "",
        f"- Independent unit: {payload['independent_unit']}",
        f"- Paired samples per cell: {payload['paired_samples']}",
        f"- Bootstrap iterations: {payload['bootstrap_iterations']:,}",
        f"- Interval: {payload['interval']}",
        f"- Metric calculation: {payload['metric_note']}",
        f"- Multiplicity: {payload['multiplicity_note']}",
        "",
    ]
    for metric, values in payload["metrics"].items():
        lines.extend(
            [
                f"## {metric}",
                "",
                f"Direction: {values['direction']}.",
                "",
                "| Writer | R-low | R-mid | R-high |",
                "|---|---:|---:|---:|",
            ]
        )
        for writer in WRITERS:
            cells = values["cell_values"]
            lines.append(
                f"| {writer} | {shown(cells[f'{writer}/low'])} | "
                f"{shown(cells[f'{writer}/mid'])} | {shown(cells[f'{writer}/high'])} |"
            )
        lines.extend(
            [
                "",
                "| Contrast | Family | Estimate | 95% CI |",
                "|---|---|---:|---:|",
            ]
        )
        for contrast in values["contrasts"]:
            estimate = shown(contrast["estimate"])
            interval = (
                "NA"
                if contrast["ci95_low"] is None or contrast["ci95_high"] is None
                else f"[{contrast['ci95_low']:.6f}, {contrast['ci95_high']:.6f}]"
            )
            lines.append(
                f"| {contrast['name']} | {contrast['family']} | "
                f"{estimate} | {interval} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    payload = analyze(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"metrics": list(payload["metrics"]), "status": "PASS"}))


if __name__ == "__main__":
    main()
