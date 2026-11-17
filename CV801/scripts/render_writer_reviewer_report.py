#!/usr/bin/env python3
"""Render audited Writer x Reviewer statistics for the report and release.

The renderer accepts only a completed paired-bootstrap payload.  It deliberately
produces tables rather than prose conclusions so that an incomplete or changed
analysis cannot silently create an unsupported manuscript claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WRITERS = ("2B", "4B", "9B")
TIERS = ("low", "mid", "high")
METRICS = (
    "independent_visual_total",
    "object_id_link_f1",
    "schema_valid_rate",
    "unsupported_claim_share",
)
METRIC_LABELS = {
    "independent_visual_total": "Independent visual total",
    "object_id_link_f1": "Object-ID link F1",
    "schema_valid_rate": "Schema-valid rate",
    "unsupported_claim_share": "Unsupported-claim share",
}
PRIMARY_CONTRASTS = (
    "gap_closure_2B_vs_9B_high_minus_low",
    "gap_closure_4B_vs_9B_high_minus_low",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def require_optional_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return require_number(value, field)


def validate(payload: dict[str, Any]) -> None:
    if payload.get("protocol") != "cv801_writer_reviewer_paired_bootstrap_v2_micro_metrics":
        raise ValueError("Unexpected or missing paired-bootstrap protocol")
    if payload.get("paired_samples") != 1_000:
        raise ValueError("Report tables require exactly 1,000 paired samples per cell")
    if payload.get("bootstrap_iterations") != 10_000:
        raise ValueError("Report tables require the frozen 10,000 bootstrap iterations")
    if payload.get("seed") != 20_261_104:
        raise ValueError("Report tables require frozen seed 20261104")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
        raise ValueError(f"Expected exactly these metrics: {METRICS}")
    expected_cells = {f"{writer}/{tier}" for writer in WRITERS for tier in TIERS}
    for metric in METRICS:
        values = metrics[metric]
        if not isinstance(values, dict):
            raise ValueError(f"Invalid metric payload: {metric}")
        cells = values.get("cell_values")
        if not isinstance(cells, dict) or set(cells) != expected_cells:
            raise ValueError(f"{metric} does not contain the complete 3 x 3 grid")
        undefined_cells = values.get("undefined_cells", {})
        if not isinstance(undefined_cells, dict):
            raise ValueError(f"{metric} undefined_cells must be an object")
        for cell, value in cells.items():
            if require_optional_number(value, f"{metric}/{cell}") is None:
                if metric != "object_id_link_f1" or not isinstance(
                    undefined_cells.get(cell), str
                ):
                    raise ValueError(f"{metric}/{cell} is unexpectedly undefined")
            elif cell in undefined_cells:
                raise ValueError(f"{metric}/{cell} is numeric but marked undefined")
        contrasts = values.get("contrasts")
        if not isinstance(contrasts, list):
            raise ValueError(f"{metric} contrasts are missing")
        by_name = {
            item.get("name"): item
            for item in contrasts
            if isinstance(item, dict) and item.get("family") == "primary_interaction"
        }
        if set(by_name) != set(PRIMARY_CONTRASTS):
            raise ValueError(f"{metric} primary interactions are incomplete")
        for name, item in by_name.items():
            values_or_none = [
                require_optional_number(item.get(field), f"{metric}/{name}/{field}")
                for field in ("estimate", "ci95_low", "ci95_high")
            ]
            if all(value is not None for value in values_or_none):
                if item.get("valid_bootstrap_iterations") != 10_000:
                    raise ValueError(f"{metric}/{name} has incomplete bootstrap iterations")
            elif (
                metric != "object_id_link_f1"
                or any(value is not None for value in values_or_none)
                or item.get("valid_bootstrap_iterations") != 0
                or not isinstance(item.get("undefined_reason"), str)
            ):
                raise ValueError(f"{metric}/{name} has an invalid undefined contrast")


def display_value(metric: str, value: float | None) -> str:
    if value is None:
        return "--"
    if metric == "independent_visual_total":
        return f"{value:.3f}"
    return f"{100.0 * value:.1f}\\%"


def display_effect(metric: str, value: float | None) -> str:
    if value is None:
        return "--"
    if metric == "independent_visual_total":
        return f"{value:+.3f}"
    return f"{100.0 * value:+.1f} pp"


def render_tex(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "% Auto-generated from paired_analysis.json; do not edit by hand.",
        "% Frozen protocol: 1,000 paired images, 10,000 bootstrap resamples, seed 20261104.",
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Endpoint & T2 Writer & R-low & R-mid & R-high \\\\",
        "\\midrule",
    ]
    for metric_index, metric in enumerate(METRICS):
        cells = metrics[metric]["cell_values"]
        for writer_index, writer in enumerate(WRITERS):
            label = METRIC_LABELS[metric] if writer_index == 0 else ""
            lines.append(
                f"{label} & {writer} & "
                f"{display_value(metric, cells[f'{writer}/low'])} & "
                f"{display_value(metric, cells[f'{writer}/mid'])} & "
                f"{display_value(metric, cells[f'{writer}/high'])} \\\\"
            )
        if metric_index != len(METRICS) - 1:
            lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
            "\\caption{Primary endpoints for the controlled Writer $\\times$ reviewer grid. "
            "All nine cells contain the same 1,000 Golden Test images. Rates and F1 are "
            "percentages; independent visual total is reported on its 0--6 scale.}",
            "\\label{tab:writer-reviewer-grid}",
            "\\end{table}",
            "",
            "\\begin{table}[H]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3.5pt}",
            "\\resizebox{\\columnwidth}{!}{%",
            "\\begin{tabular}{llrr}",
            "\\toprule",
            "Endpoint & Gap closure & Estimate & Paired-bootstrap 95\\% CI \\\\",
            "\\midrule",
        ]
    )
    for metric_index, metric in enumerate(METRICS):
        primary = {
            item["name"]: item
            for item in metrics[metric]["contrasts"]
            if item["family"] == "primary_interaction"
        }
        for contrast_index, name in enumerate(PRIMARY_CONTRASTS):
            item = primary[name]
            label = METRIC_LABELS[metric] if contrast_index == 0 else ""
            writer = "2B vs. 9B" if name.startswith("gap_closure_2B") else "4B vs. 9B"
            low = display_effect(metric, item["ci95_low"])
            high = display_effect(metric, item["ci95_high"])
            interval = "--" if item["ci95_low"] is None else f"[{low}, {high}]"
            lines.append(
                f"{label} & {writer} & {display_effect(metric, item['estimate'])} "
                f"& {interval} \\\\"
            )
        if metric_index != len(METRICS) - 1:
            lines.append("\\addlinespace")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
            "\\caption{Prespecified high-minus-low reviewer gap-closure contrasts. "
            "Positive values use the raw endpoint direction. For unsupported-claim share, "
            "which is lower-is-better, a negative estimate is favorable. Intervals use "
            "10,000 paired image-cluster bootstrap resamples. A dash means that the frozen "
            "metric is undefined because a required denominator is zero.}",
            "\\label{tab:writer-reviewer-interaction}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# Writer x Reviewer report tables",
        "",
        "Generated only after the complete 1,000-image paired audit. Values are "
        "derived from 10,000 paired image-cluster bootstrap resamples with seed "
        "`20261104`.",
        "",
    ]
    for metric in METRICS:
        cells = metrics[metric]["cell_values"]
        lines.extend(
            [
                f"## {METRIC_LABELS[metric]}",
                "",
                "| T2 Writer | R-low | R-mid | R-high |",
                "|---|---:|---:|---:|",
            ]
        )
        for writer in WRITERS:
            values = [cells[f"{writer}/{tier}"] for tier in TIERS]
            shown = [display_value(metric, value).replace("\\%", "%") for value in values]
            lines.append(f"| {writer} | {shown[0]} | {shown[1]} | {shown[2]} |")
        lines.extend(["", "| Gap closure | Estimate | 95% CI |", "|---|---:|---:|"])
        primary = {
            item["name"]: item
            for item in metrics[metric]["contrasts"]
            if item["family"] == "primary_interaction"
        }
        for name in PRIMARY_CONTRASTS:
            item = primary[name]
            writer = "2B vs. 9B" if name.startswith("gap_closure_2B") else "4B vs. 9B"
            estimate = display_effect(metric, item["estimate"])
            if item["ci95_low"] is None:
                interval = "--"
            else:
                low = display_effect(metric, item["ci95_low"])
                high = display_effect(metric, item["ci95_high"])
                interval = f"[{low}, {high}]"
            lines.append(f"| {writer}, R-high minus R-low | {estimate} | {interval} |")
        lines.append("")
    lines.extend(
        [
            "The tables intentionally contain no automated causal or superiority claim. "
            "Interpretation must combine all four endpoints and the stated direction of each metric.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload = json.loads(args.analysis_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Analysis JSON must contain an object")
    validate(payload)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(render_tex(payload), encoding="utf-8")
    args.output_markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_tex": str(args.output_tex),
                "output_markdown": str(args.output_markdown),
            }
        )
    )


if __name__ == "__main__":
    main()
