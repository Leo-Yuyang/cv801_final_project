#!/usr/bin/env python3
"""Refresh CPU-side status and result material for CV801 Writer scaling.

This script is deliberately read-only with respect to experiment outputs. It
queries Slurm and copies available aggregate metrics into snapshot CSV/Markdown
files under the requested output directory. It never submits, cancels, holds,
or releases a job and never edits the main report.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
JOB_KEY = re.compile(r"(?:^|_)(?:job|jobs)$")
JOB_ID = re.compile(r"^\d+(?:_\d+)?$")

WRITER_RUNS = [
    {
        "writer": "2B",
        "variant": "T0",
        "job_id": "443953",
        "predictions": "wkdrs/cv801-golden-test/writer-w2b-t0-443953/predictions.jsonl",
        "structured": "wkdrs/cv801-writer-scale/qwen35-2b/eval-t0/metrics.golden.json",
        "independent": "wkdrs/cv801-independent-eval/writer-w2b-t0/metrics.json",
    },
    {
        "writer": "2B",
        "variant": "T1",
        "job_id": "448130",
        "predictions": "wkdrs/cv801-golden-test/writer-w2b-t1-448130/predictions.jsonl",
        "structured": "wkdrs/cv801-writer-scale/qwen35-2b/eval-t1/metrics.golden.json",
        "independent": "wkdrs/cv801-independent-eval/writer-w2b-t1/metrics.json",
    },
    {
        "writer": "2B",
        "variant": "T2",
        "job_id": "448138",
        "predictions": "wkdrs/cv801-golden-test/writer-w2b-t2-448138/predictions.jsonl",
        "structured": "wkdrs/cv801-writer-scale/qwen35-2b/eval-t2/metrics.golden.json",
        "independent": "wkdrs/cv801-independent-eval/writer-w2b-t2/metrics.json",
    },
    {
        "writer": "4B",
        "variant": "T0",
        "job_id": "443961",
        "predictions": "wkdrs/cv801-golden-test/writer-w4b-t0-443961/predictions.jsonl",
        "structured": "wkdrs/cv801-writer-scale/qwen35-4b/eval-t0/metrics.golden.json",
        "independent": "wkdrs/cv801-independent-eval/writer-w4b-t0/metrics.json",
    },
    {
        "writer": "4B",
        "variant": "T1",
        "job_id": "447989",
        "predictions": "wkdrs/cv801-golden-test/writer-w4b-t1-447989/predictions.jsonl",
        "structured": "wkdrs/cv801-writer-scale/qwen35-4b/eval-t1/metrics.golden.json",
        "independent": "wkdrs/cv801-independent-eval/writer-w4b-t1/metrics.json",
    },
    {
        "writer": "4B",
        "variant": "T2",
        "job_id": "449714",
        "predictions": "wkdrs/cv801-golden-test/writer-w4b-t2-r1-449714/predictions.jsonl",
        "structured": "wkdrs/cv801-writer-scale/qwen35-4b/eval-t2-r1/metrics.golden.json",
        "independent": "wkdrs/cv801-independent-eval/writer-w4b-t2-r1/metrics.json",
    },
    {
        "writer": "9B",
        "variant": "T0",
        "job_id": "403229",
        "predictions": "wkdrs/cv801-golden-test/writer-t0-403229/predictions.jsonl",
        "structured": "wkdrs/cv801-golden-test/writer-t0-403229/metrics.golden.json",
        "independent": "wkdrs/cv801-independent-eval/golden-t0/metrics.json",
    },
    {
        "writer": "9B",
        "variant": "T1",
        "job_id": "403230",
        "predictions": "wkdrs/cv801-golden-test/writer-t1-403230/predictions.jsonl",
        "structured": "wkdrs/cv801-golden-test/writer-t1-403230/metrics.golden.json",
        "independent": "wkdrs/cv801-independent-eval/golden-t1/metrics.json",
    },
    {
        "writer": "9B",
        "variant": "T2",
        "job_id": "403232",
        "predictions": "wkdrs/cv801-golden-test/writer-t2-403232/predictions.jsonl",
        "structured": "wkdrs/cv801-golden-test/writer-t2-403232/metrics.golden.json",
        "independent": "wkdrs/cv801-independent-eval/golden-t2/metrics.json",
    },
]

GRID_RUNS = [
    ("2B", "low", "wkdrs/cv801-writer-reviewer-grid/grid-w2b-low"),
    ("2B", "mid", "wkdrs/cv801-writer-reviewer-grid/grid-w2b-mid"),
    ("2B", "high", "wkdrs/cv801-writer-reviewer-grid/grid-w2b-high"),
    ("4B", "low", "wkdrs/cv801-writer-reviewer-grid-r1/grid-w4b-low"),
    ("4B", "mid", "wkdrs/cv801-writer-reviewer-grid-r1/grid-w4b-mid"),
    ("4B", "high", "wkdrs/cv801-writer-reviewer-grid-r1/grid-w4b-high"),
    ("9B", "low", "wkdrs/cv801-writer-reviewer-grid/grid-w9b-low"),
    ("9B", "mid", "wkdrs/cv801-reviewer-scale/gemma12-internvl14-435502"),
    ("9B", "high", "wkdrs/cv801-reviewer-scale/gemma27-internvl38-435547"),
]
FINAL_GRID_ANALYSIS = ROOT / "wkdrs/cv801-writer-reviewer-statistics/paired_analysis.json"
FINAL_GRID_SAMPLES = ROOT / "wkdrs/cv801-writer-reviewer-statistics/grid_samples.csv"
FINAL_GRID_LINEAGE = ROOT / "wkdrs/cv801-writer-reviewer-statistics/repair_lineage.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "wkdrs/cv801-writer-scale-analysis",
    )
    return parser.parse_args()


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def env_job_ids(*paths: Path) -> list[str]:
    ids: list[str] = []
    for path in paths:
        for key, value in read_env(path).items():
            if not JOB_KEY.search(key):
                continue
            for candidate in value.split(":"):
                if JOB_ID.fullmatch(candidate) and candidate not in ids:
                    ids.append(candidate)
    return ids


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def query_jobs(job_ids: list[str]) -> dict[str, dict[str, str]]:
    unique = list(dict.fromkeys(job_ids))
    if not unique:
        return {}
    rows: dict[str, dict[str, str]] = {}
    accounting = run(
        [
            "sacct",
            "-X",
            "-j",
            ",".join(unique),
            "--format=JobIDRaw,JobName,State,ElapsedRaw,Start,End,ReqTRES",
            "-P",
            "-n",
        ]
    )
    for line in accounting.splitlines():
        fields = line.split("|")
        if len(fields) != 7 or fields[0] not in unique:
            continue
        rows[fields[0]] = dict(
            zip(
                ("job_id", "job_name", "state", "elapsed_seconds", "start", "end", "req_tres"),
                fields,
                strict=True,
            )
        )
    live = run(
        [
            "squeue",
            "-h",
            "-j",
            ",".join(unique),
            "-o",
            "%i|%T|%R|%M|%l|%b",
        ]
    )
    for line in live.splitlines():
        fields = line.split("|")
        if len(fields) != 6:
            continue
        row = rows.setdefault(fields[0], {"job_id": fields[0]})
        row.update(
            live_state=fields[1],
            reason_or_node=fields[2],
            live_elapsed=fields[3],
            time_limit=fields[4],
            gres=fields[5],
        )
    return rows


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def unsupported_share(metrics: dict) -> float | None:
    supported = metrics.get("supported_claims")
    unsupported = metrics.get("unsupported_claims")
    if not isinstance(supported, int) or not isinstance(unsupported, int):
        return None
    total = supported + unsupported
    return unsupported / total if total else None


def writer_rows(jobs: dict[str, dict[str, str]]) -> list[dict]:
    rows: list[dict] = []
    for spec in WRITER_RUNS:
        structured = load_json(ROOT / spec["structured"])
        independent = load_json(ROOT / spec["independent"])
        job = jobs.get(spec["job_id"], {})
        state = job.get("live_state") or job.get("state") or "UNKNOWN"
        prediction_count = count_jsonl(ROOT / spec["predictions"])
        if independent.get("samples") == 1000 and structured.get("samples") == 1000:
            material_status = "metrics-ready"
        elif prediction_count == 1000:
            material_status = "awaiting-evaluation"
        elif state == "RUNNING":
            material_status = "running"
        elif state == "PENDING":
            material_status = "queued"
        else:
            material_status = "incomplete"
        mean_scores = independent.get("mean_scores", {})
        rows.append(
            {
                "writer": spec["writer"],
                "variant": spec["variant"],
                "job_id": spec["job_id"],
                "slurm_state": state,
                "reason_or_node": job.get("reason_or_node", ""),
                "predictions": prediction_count,
                "json_valid_rate": structured.get("json_valid_rate"),
                "schema_valid_rate": structured.get("schema_valid_rate"),
                "object_count_exact_rate": structured.get("object_count_exact_rate"),
                "object_id_link_f1": structured.get("object_id_link_f1"),
                "caption_token_f1": structured.get("caption_token_f1"),
                "independent_visual_total": mean_scores.get("total"),
                "unsupported_claim_share": unsupported_share(independent),
                "material_status": material_status,
                "structured_source": spec["structured"],
                "independent_source": spec["independent"],
            }
        )
    return rows


def aggregate_grid_state(ids: list[str], jobs: dict[str, dict[str, str]]) -> tuple[str, str]:
    states = Counter(
        (jobs.get(job_id, {}).get("live_state") or jobs.get(job_id, {}).get("state") or "UNKNOWN")
        for job_id in ids
    )
    if states.get("RUNNING"):
        overall = "running"
    elif states.get("PENDING"):
        overall = "queued"
    elif ids and states.get("COMPLETED") == len(ids):
        overall = "complete"
    elif states.get("FAILED") or states.get("CANCELLED") or states.get("TIMEOUT"):
        overall = "repair-lineage"
    else:
        overall = "incomplete"
    summary = ";".join(f"{state}:{count}" for state, count in sorted(states.items()))
    return overall, summary


def final_grid_complete() -> bool:
    analysis = load_json(FINAL_GRID_ANALYSIS)
    lineage = load_json(FINAL_GRID_LINEAGE)
    return (
        analysis.get("protocol") == "cv801_writer_reviewer_paired_bootstrap_v2_micro_metrics"
        and analysis.get("paired_samples") == 1_000
        and analysis.get("bootstrap_iterations") == 10_000
        and set(analysis.get("metrics", {}))
        == {
            "independent_visual_total",
            "object_id_link_f1",
            "schema_valid_rate",
            "unsupported_claim_share",
        }
        and lineage.get("total_repaired_rows") == 111
        and count_jsonl(FINAL_GRID_SAMPLES) == 9_001
    )


def grid_rows(jobs: dict[str, dict[str, str]], analysis_complete: bool) -> list[dict]:
    rows: list[dict] = []
    for writer, tier, relative in GRID_RUNS:
        root = ROOT / relative
        ids = env_job_ids(root / "submission.env", root / "downstream/downstream.env")
        overall, summary = aggregate_grid_state(ids, jobs)
        if analysis_complete:
            overall = "paired-analysis-complete"
        rows.append(
            {
                "writer": writer,
                "reviewer_tier": tier,
                "status": overall,
                "job_count": len(ids),
                "state_counts": summary,
                "source": relative,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, scale: float = 1.0, digits: int = 3) -> str:
    return "--" if value is None else f"{float(value) * scale:.{digits}f}"


def render_markdown(
    writer: list[dict], grid: list[dict], generated: str, analysis_complete: bool
) -> str:
    lines = [
        "# CV801 Writer-scale CPU material snapshot",
        "",
        f"Generated: {generated}",
        "",
        "## Writer-only material readiness",
        "",
        "| Writer | Variant | Slurm | Predictions | JSON % | Schema % | ID-F1 % | Visual | Material |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in writer:
        lines.append(
            f"| {row['writer']} | {row['variant']} | {row['slurm_state']} | "
            f"{row['predictions'] if row['predictions'] is not None else '--'} | "
            f"{fmt(row['json_valid_rate'], 100, 1)} | "
            f"{fmt(row['schema_valid_rate'], 100, 1)} | "
            f"{fmt(row['object_id_link_f1'], 100, 1)} | "
            f"{fmt(row['independent_visual_total'])} | {row['material_status']} |"
        )
    lines.extend(
        [
            "",
            "## Writer x Reviewer grid execution and analysis readiness",
            "",
            "| Writer | Tier | Status | Jobs | State counts | Source |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for row in grid:
        lines.append(
            f"| {row['writer']} | {row['reviewer_tier']} | {row['status']} | "
            f"{row['job_count']} | {row['state_counts']} | `{row['source']}` |"
        )
    final_boundary = (
        "- Final paired statistics are complete: all nine cells share the same "
        "1,000 sample IDs and the frozen 10,000-resample analysis is present."
        if analysis_complete
        else "- Final paired statistics require identical 1,000-sample ID sets across all nine grid cells."
    )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This is a live CPU-side snapshot, not a final result table.",
            "- A row is `metrics-ready` only when both structured and independent aggregate files contain 1,000 samples.",
            "- Low validity remains visible and is not silently converted into success.",
            final_boundary,
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    grid_specs = []
    for _, _, relative in GRID_RUNS:
        root = ROOT / relative
        grid_specs.extend(env_job_ids(root / "submission.env", root / "downstream/downstream.env"))
    all_ids = [spec["job_id"] for spec in WRITER_RUNS] + grid_specs
    jobs = query_jobs(all_ids)
    writers = writer_rows(jobs)
    analysis_complete = final_grid_complete()
    grid = grid_rows(jobs, analysis_complete)
    generated = datetime.now(ZoneInfo("Asia/Dubai")).isoformat(timespec="seconds")

    write_csv(output / "writer_results.current.csv", writers)
    write_csv(output / "reviewer_grid_status.current.csv", grid)
    job_rows = [jobs[job_id] for job_id in sorted(jobs, key=lambda value: int(value.split("_")[0]))]
    if job_rows:
        all_keys = list(dict.fromkeys(key for row in job_rows for key in row))
        normalized = [{key: row.get(key, "") for key in all_keys} for row in job_rows]
        write_csv(output / "slurm_jobs.current.csv", normalized)
    (output / "status.current.md").write_text(
        render_markdown(writers, grid, generated, analysis_complete), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "generated_uae": generated,
                "writer_rows": len(writers),
                "grid_rows": len(grid),
                "grid_analysis_complete": analysis_complete,
                "slurm_rows": len(jobs),
                "output_dir": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
