#!/usr/bin/env python3
"""Build an auditable manifest for the nine-cell evaluator repair bundle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


WRITERS = ("2B", "4B", "9B")
TIERS = ("low", "mid", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        action="append",
        required=True,
        metavar="WRITER,TIER,REPAIR_ENV",
    )
    parser.add_argument("--bundle-job-id", required=True)
    parser.add_argument("--parent-bundle-job-id")
    parser.add_argument("--residual-bundle-line", action="append", type=int, default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.lstrip().startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid environment row at {path}:{line_number}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def relative(path_text: str, project_root: Path) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        raise ValueError(f"Lineage path is outside the project root: {path}") from None


def parse_lines(text: str) -> list[int]:
    normalized = text.replace("\\ ", ":").replace(" ", ":")
    values = [int(part) for part in normalized.split(":") if part]
    if not values or len(values) != len(set(values)) or any(value <= 0 for value in values):
        raise ValueError(f"Invalid repair line list: {text!r}")
    return values


def build(args: argparse.Namespace) -> dict:
    project_root = args.project_root.resolve()
    cells = []
    keys = set()
    for specification in args.cell:
        parts = specification.split(",", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid --cell specification: {specification}")
        writer, tier, env_text = (part.strip() for part in parts)
        tier = tier.lower()
        if writer not in WRITERS or tier not in TIERS or (writer, tier) in keys:
            raise ValueError(f"Invalid or duplicate Writer/tier: {writer}/{tier}")
        keys.add((writer, tier))
        env_path = Path(env_text)
        if not env_path.is_absolute():
            env_path = project_root / env_path
        if not env_path.is_file():
            raise FileNotFoundError(env_path)
        values = read_env(env_path)
        for required in ("output_dir", "source_predictions", "invalid_lines"):
            if not values.get(required):
                raise ValueError(f"{env_path} is missing {required}")
        line_numbers = parse_lines(values["invalid_lines"])
        replacement = env_path.parent / f"bundled-predictions-{args.bundle_job_id}.jsonl"
        if not replacement.is_file():
            raise FileNotFoundError(replacement)
        replacement_rows = sum(1 for line in replacement.open(encoding="utf-8") if line.strip())
        if replacement_rows != len(line_numbers):
            raise ValueError(
                f"{writer}/{tier} replacement rows={replacement_rows}; "
                f"expected {len(line_numbers)}"
            )
        cells.append(
            {
                "writer": writer,
                "reviewer_tier": tier,
                "invalid_rows": len(line_numbers),
                "line_numbers_1_based": line_numbers,
                "repair_environment": relative(str(env_path), project_root),
                "source_predictions": relative(values["source_predictions"], project_root),
                "replacement_predictions": relative(str(replacement), project_root),
                "canonical_output_directory": relative(values["output_dir"], project_root),
            }
        )
    expected = {(writer, tier) for writer in WRITERS for tier in TIERS}
    if keys != expected:
        raise ValueError("Provide each of the nine Writer/reviewer cells exactly once")
    cells.sort(key=lambda item: (WRITERS.index(item["writer"]), TIERS.index(item["reviewer_tier"])))
    total = sum(item["invalid_rows"] for item in cells)
    if total != 111:
        raise ValueError(f"Frozen repair bundle must contain 111 rows, found {total}")
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_job_id": str(args.bundle_job_id),
        "policy": (
            "Only invalid independent-evaluator rows were rerun with the same frozen "
            "Pixtral-12B evaluator protocol; valid rows were retained."
        ),
        "cells": cells,
        "total_repaired_rows": total,
        "expected_rows_per_cell": 1_000,
    }
    if bool(args.parent_bundle_job_id) != bool(args.residual_bundle_line):
        raise ValueError(
            "parent-bundle-job-id and residual-bundle-line must be provided together"
        )
    if args.residual_bundle_line:
        residual_lines = args.residual_bundle_line
        if (
            len(residual_lines) != len(set(residual_lines))
            or any(value <= 0 or value > total for value in residual_lines)
        ):
            raise ValueError("Residual bundle lines must be unique positions in the repair bundle")
        result["parent_bundle_job_id"] = str(args.parent_bundle_job_id)
        result["residual_bundle_line_numbers_1_based"] = residual_lines
        result["residual_repaired_rows"] = len(residual_lines)
    return result


def render_markdown(payload: dict) -> str:
    lines = [
        "# Writer x Reviewer evaluator-repair lineage",
        "",
        payload["policy"],
        "",
        f"- Bundle Slurm job: `{payload['bundle_job_id']}`",
        f"- Repaired rows: {payload['total_repaired_rows']} of 9,000 evaluator rows",
        "- Line numbers are one-based positions in each cell's canonical prediction file.",
        "",
        "| Writer | Reviewer | Repaired rows | One-based line numbers |",
        "|---|---|---:|---|",
    ]
    if payload.get("parent_bundle_job_id"):
        residual = ", ".join(
            str(value) for value in payload["residual_bundle_line_numbers_1_based"]
        )
        lines[7:7] = [
            f"- Parent 111-row bundle job: `{payload['parent_bundle_job_id']}`",
            f"- Residual responses rerun by the finalizer: {payload['residual_repaired_rows']} "
            f"(one-based bundle rows: {residual})",
        ]
    for cell in payload["cells"]:
        line_numbers = ", ".join(str(value) for value in cell["line_numbers_1_based"])
        lines.append(
            f"| {cell['writer']} | R-{cell['reviewer_tier']} | "
            f"{cell['invalid_rows']} | {line_numbers} |"
        )
    lines.extend(
        [
            "",
            "The complete machine-readable paths and replacement files are recorded in "
            "`repair_lineage.json`. Repair does not remove invalid outputs from the denominator.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload = build(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "PASS", "repaired_rows": payload["total_repaired_rows"]}))


if __name__ == "__main__":
    main()
