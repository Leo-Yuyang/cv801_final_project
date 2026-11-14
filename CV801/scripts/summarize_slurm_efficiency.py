#!/usr/bin/env python3
"""Summarize completed Slurm inference jobs as GPU-hours and wall time."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SYSTEM_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
JOB_PATTERN = re.compile(r"^[0-9]+$")
GPU_TRES_PATTERN = re.compile(r"(?:^|,)gres/gpu(?::[^=,]+)?=([0-9]+)(?:,|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        metavar="NAME=JOB_ID[,JOB_ID...]",
    )
    parser.add_argument("--images", type=int, default=1000)
    parser.add_argument(
        "--accept-failed",
        action="append",
        default=[],
        metavar="JOB_ID[,JOB_ID...]",
        help="Accept jobs that failed only in a separately certified post-inference audit.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_system_specs(specs: list[str]) -> dict[str, list[str]]:
    systems: dict[str, list[str]] = {}
    for spec in specs:
        name, separator, raw_ids = spec.partition("=")
        job_ids = [item for item in raw_ids.split(",") if item]
        if not separator or not SYSTEM_PATTERN.fullmatch(name) or not job_ids:
            raise ValueError(f"Invalid system specification: {spec}")
        if name in systems:
            raise ValueError(f"Duplicate system name: {name}")
        if any(not JOB_PATTERN.fullmatch(job_id) for job_id in job_ids):
            raise ValueError(f"Invalid Slurm job ID in: {spec}")
        if len(set(job_ids)) != len(job_ids):
            raise ValueError(f"Duplicate Slurm job ID in: {spec}")
        systems[name] = job_ids
    return systems


def requested_gpus(req_tres: str) -> int:
    values = [int(value) for value in GPU_TRES_PATTERN.findall(req_tres)]
    if not values:
        raise ValueError(f"No GPU request found in ReqTRES: {req_tres}")
    # Slurm commonly reports both gres/gpu:TYPE=N and gres/gpu=N. They are
    # aliases for the same allocation, so take the maximum rather than sum.
    return max(values)


def parse_timestamp(value: str) -> datetime:
    if not value or value == "Unknown":
        raise ValueError(f"Missing Slurm timestamp: {value}")
    return datetime.fromisoformat(value)


def query_jobs(job_ids: list[str]) -> dict[str, dict[str, Any]]:
    command = [
        "sacct",
        "-X",
        "-j",
        ",".join(job_ids),
        "--format=JobIDRaw,State,ElapsedRaw,Start,End,ReqTRES",
        "-P",
        "-n",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    jobs: dict[str, dict[str, Any]] = {}
    requested = set(job_ids)
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", 5)
        if len(fields) != 6:
            raise ValueError(f"Unexpected sacct output row: {line}")
        job_id, state, elapsed, start, end, req_tres = fields
        if job_id not in requested:
            continue
        jobs[job_id] = {
            "job_id": job_id,
            "state": state,
            "elapsed_seconds": int(elapsed),
            "start": start,
            "end": end,
            "requested_gpus": requested_gpus(req_tres),
        }
    missing = sorted(requested - set(jobs))
    if missing:
        raise ValueError(f"sacct did not return jobs: {', '.join(missing)}")
    return jobs


def summarize_system(
    job_ids: list[str],
    jobs: dict[str, dict[str, Any]],
    images: int,
    accepted_failed: set[str],
) -> dict[str, Any]:
    records = [jobs[job_id] for job_id in job_ids]
    incomplete = [
        record["job_id"]
        for record in records
        if not record["state"].startswith("COMPLETED")
        and not (
            record["job_id"] in accepted_failed and record["state"].startswith("FAILED")
        )
    ]
    if incomplete:
        raise ValueError(f"Jobs are not complete: {', '.join(incomplete)}")
    starts = [parse_timestamp(record["start"]) for record in records]
    ends = [parse_timestamp(record["end"]) for record in records]
    gpu_seconds = sum(record["elapsed_seconds"] * record["requested_gpus"] for record in records)
    wall_seconds = (max(ends) - min(starts)).total_seconds()
    return {
        "job_ids": job_ids,
        "job_count": len(job_ids),
        "images": images,
        "gpu_seconds": gpu_seconds,
        "gpu_hours": gpu_seconds / 3600,
        "gpu_hours_per_image": gpu_seconds / 3600 / images,
        "execution_span_seconds": wall_seconds,
        "execution_span_seconds_per_image": wall_seconds / images,
        "post_inference_audit_failures_accepted": sorted(
            record["job_id"] for record in records if record["job_id"] in accepted_failed
        ),
        "jobs": records,
    }


def build_summary(
    systems: dict[str, list[str]],
    jobs: dict[str, dict[str, Any]],
    images: int,
    accepted_failed: set[str] | None = None,
) -> dict[str, Any]:
    if images <= 0:
        raise ValueError("--images must be positive")
    accepted_failed = accepted_failed or set()
    return {
        "protocol": "cv801_slurm_efficiency_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "accounting_note": (
            "GPU-hours sum elapsed allocation time; execution span is max(end)-min(start) "
            "for each logical system and excludes queue wait. Shared stages are charged to "
            "each system that requires them. Jobs listed as accepted post-inference failures "
            "produced complete outputs that passed a corrected independent audit."
        ),
        "systems": {
            name: summarize_system(job_ids, jobs, images, accepted_failed)
            for name, job_ids in systems.items()
        },
    }


def main() -> None:
    args = parse_args()
    systems = parse_system_specs(args.system)
    all_job_ids = list(dict.fromkeys(job_id for job_ids in systems.values() for job_id in job_ids))
    accepted_failed = {
        job_id
        for group in args.accept_failed
        for job_id in group.split(",")
        if job_id
    }
    if any(not JOB_PATTERN.fullmatch(job_id) for job_id in accepted_failed):
        raise ValueError("--accept-failed contains an invalid Slurm job ID")
    unknown_accepted = accepted_failed - set(all_job_ids)
    if unknown_accepted:
        raise ValueError(f"--accept-failed jobs are not used by a system: {sorted(unknown_accepted)}")
    payload = build_summary(
        systems, query_jobs(all_job_ids), args.images, accepted_failed=accepted_failed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
