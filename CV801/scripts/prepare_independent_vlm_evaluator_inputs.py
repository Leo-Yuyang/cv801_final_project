#!/usr/bin/env python3
"""Prepare reference-free inputs for an independent visual quality evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .evaluate_vlm_predictions import clean_response
except ImportError:
    from evaluate_vlm_predictions import clean_response


SYSTEM_PROMPT = (
    "You are an independent agricultural visual evaluator. Score a candidate response only against "
    "the visible image. You are not a Writer, reviewer, or Judge in the revision loop. Do not use or "
    "invent hidden reference captions. Return strict JSON without markdown."
)
OUTPUT_SCHEMA = (
    '{"sample_id":"...","object_reviews":[{"object_id":"object_001",'
    '"supported_claims":[],"unsupported_claims":[],"object_linkage":"correct|incorrect|uncertain"}],'
    '"scores":{"visual_evidence":0,"conclusion_support":0,"confidence_appropriateness":0},'
    '"total":0,"rationale":[]}'
)
RUBRIC = (
    "Use integer scores from 0 to 2. visual_evidence: 0=multiple unsupported visible claims or major "
    "missed objects, 1=mixed support or minor omissions, 2=visible claims and object coverage are "
    "supported. conclusion_support: 0=assessment/action conflicts with or does not follow from visible "
    "evidence, 1=partly supported or too strong, 2=assessment/action follows conservatively from visible "
    "evidence. confidence_appropriateness: 0=materially miscalibrated, 1=minor calibration issue, "
    "2=confidence matches visible ambiguity. total must equal the three scores and range from 0 to 6."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def image_path(record: dict[str, Any]) -> str:
    images = record.get("images") or []
    if not images:
        raise ValueError("Prediction has no image")
    first = images[0]
    path = first.get("path") if isinstance(first, dict) else first
    if not path:
        raise ValueError("Prediction image has no path")
    return str(path)


def prepare_inputs(predictions: Path, output: Path, max_samples: int | None = None) -> int:
    records = read_jsonl(predictions)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        records = records[:max_samples]
    prepared: list[dict[str, Any]] = []
    for line_number, record in enumerate(records, 1):
        path = image_path(record)
        candidate, _ = clean_response(str(record.get("response", "")))
        identifier = hashlib.sha256(f"{line_number}:{path}".encode()).hexdigest()[:16]
        prompt = (
            f"<image>\nEvaluate sample {identifier}; sample_id in the output must equal {identifier}. "
            f"{RUBRIC} Review every candidate object and list "
            f"concrete supported and unsupported claims. Use this exact schema: {OUTPUT_SCHEMA}\n"
            f"CANDIDATE_RESPONSE:\n{candidate}"
        )
        prepared.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "images": [path],
            "sample_id": identifier,
            "evaluator_protocol": "cv801_independent_visual_rubric_v1",
            "references_exposed_to_evaluator": False,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in prepared:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(prepared)


def main() -> None:
    args = parse_args()
    samples = prepare_inputs(args.predictions, args.output, args.max_samples)
    print(json.dumps({
        "samples": samples,
        "protocol": "cv801_independent_visual_rubric_v1",
        "references_exposed_to_evaluator": False,
    }))


if __name__ == "__main__":
    main()
