#!/usr/bin/env python3
"""Audit strict JSON contracts for Rc, Rd, Judge J, and J recheck outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_agent_review_inputs import final_response, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("reviewer", "judge", "recheck"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-all-valid", action="store_true")
    return parser.parse_args()


def is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def valid_reviewer(payload: object) -> bool:
    if not isinstance(payload, dict) or payload.get("verdict") not in {"pass", "revise"}:
        return False
    if not all(is_string_list(payload.get(key)) for key in ("global_issues", "revision_instructions")):
        return False
    reviews = payload.get("object_reviews")
    if not isinstance(reviews, list):
        return False
    for review in reviews:
        if not isinstance(review, dict) or not isinstance(review.get("object_id"), str):
            return False
        if not all(
            is_string_list(review.get(key))
            for key in ("supported_claims", "unsupported_claims", "missed_evidence")
        ):
            return False
        if review.get("confidence_issue") not in {"none", "overconfident", "underconfident"}:
            return False
    return True


def valid_judge(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("decision") in {"accept", "revise"}
        and all(
            is_string_list(payload.get(key))
            for key in ("agreements", "resolved_disagreements", "revision_feedback", "recheck_requirements")
        )
    )


def valid_recheck(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("decision") in {"accept", "revise_again"}
        and all(
            is_string_list(payload.get(key))
            for key in ("resolved_issues", "remaining_issues", "revision_feedback")
        )
    )


def audit(kind: str, predictions: Path) -> dict[str, object]:
    validators = {"reviewer": valid_reviewer, "judge": valid_judge, "recheck": valid_recheck}
    records = read_jsonl(predictions)
    json_valid = 0
    schema_valid = 0
    decisions: Counter[str] = Counter()
    invalid_lines: list[int] = []
    for line_number, record in enumerate(records, 1):
        try:
            payload = json.loads(final_response(record))
        except json.JSONDecodeError:
            invalid_lines.append(line_number)
            continue
        json_valid += 1
        if not validators[kind](payload):
            invalid_lines.append(line_number)
            continue
        schema_valid += 1
        decisions[str(payload.get("verdict", payload.get("decision")))] += 1
    total = len(records)
    return {
        "kind": kind,
        "samples": total,
        "json_valid": json_valid,
        "json_valid_rate": json_valid / total if total else 0.0,
        "schema_valid": schema_valid,
        "schema_valid_rate": schema_valid / total if total else 0.0,
        "decisions": dict(sorted(decisions.items())),
        "invalid_lines": invalid_lines,
    }


def main() -> None:
    args = parse_args()
    result = audit(args.kind, args.predictions)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.require_all_valid and result["schema_valid"] != result["samples"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
