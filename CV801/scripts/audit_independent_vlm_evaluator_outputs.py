#!/usr/bin/env python3
"""Validate and aggregate the independent visual evaluator's 0-6 rubric outputs."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

try:
    from .evaluate_vlm_predictions import clean_response
except ImportError:
    from evaluate_vlm_predictions import clean_response


SCORE_KEYS = ("visual_evidence", "conclusion_support", "confidence_appropriateness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all-valid", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def parse_valid_payload(response: str, expected_sample_id: str) -> dict[str, Any] | None:
    cleaned, _ = clean_response(response)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("sample_id") != expected_sample_id:
        return None
    reviews = payload.get("object_reviews")
    if not isinstance(reviews, list):
        return None
    for review in reviews:
        if not isinstance(review, dict) or not isinstance(review.get("object_id"), str):
            return None
        if not string_list(review.get("supported_claims")) or not string_list(review.get("unsupported_claims")):
            return None
        if review.get("object_linkage") not in {"correct", "incorrect", "uncertain"}:
            return None
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        return None
    values: list[int] = []
    for key in SCORE_KEYS:
        value = scores.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1, 2}:
            return None
        values.append(value)
    if payload.get("total") != sum(values) or not string_list(payload.get("rationale")):
        return None
    return payload


def audit(predictions_path: Path, inputs_path: Path) -> dict[str, Any]:
    predictions = read_jsonl(predictions_path)
    inputs = read_jsonl(inputs_path)
    if len(predictions) != len(inputs):
        raise ValueError("Evaluator prediction and input counts differ")
    invalid_lines: list[int] = []
    score_sums: collections.Counter[str] = collections.Counter()
    linkage: collections.Counter[str] = collections.Counter()
    supported_claims = 0
    unsupported_claims = 0
    for line_number, (prediction, source) in enumerate(zip(predictions, inputs, strict=True), 1):
        expected_sample_id = str(source.get("sample_id", ""))
        payload = parse_valid_payload(str(prediction.get("response", "")), expected_sample_id)
        if payload is None:
            invalid_lines.append(line_number)
            continue
        scores = payload["scores"]
        for key in SCORE_KEYS:
            score_sums[key] += int(scores[key])
        score_sums["total"] += int(payload["total"])
        for review in payload["object_reviews"]:
            linkage[str(review["object_linkage"])] += 1
            supported_claims += len(review["supported_claims"])
            unsupported_claims += len(review["unsupported_claims"])
    valid = len(predictions) - len(invalid_lines)
    means = {
        key: score_sums[key] / valid if valid else 0.0
        for key in (*SCORE_KEYS, "total")
    }
    return {
        "protocol": "cv801_independent_visual_rubric_v1",
        "samples": len(predictions),
        "valid": valid,
        "valid_rate": valid / len(predictions) if predictions else 0.0,
        "mean_scores": means,
        "object_linkage": dict(sorted(linkage.items())),
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "invalid_lines": invalid_lines,
    }


def main() -> None:
    args = parse_args()
    result = audit(args.predictions, args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_all_valid and result["valid"] != result["samples"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
