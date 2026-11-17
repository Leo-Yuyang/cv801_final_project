#!/usr/bin/env python3
"""Recover complete object reviews from a truncated reviewer response.

The recovery is deliberately conservative: it keeps only complete JSON object
reviews emitted by the model, retains the first review for each object ID, and
uses empty top-level feedback arrays when truncation removed those fields.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def complete_json_objects(text: str, start: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    depth = 0
    object_start: int | None = None
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                object_start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and object_start is not None:
                candidate = json.loads(text[object_start : index + 1])
                if isinstance(candidate, dict):
                    objects.append(candidate)
                object_start = None
        elif character == "]" and depth == 0:
            break
    return objects


def recover(response: str) -> dict[str, object]:
    verdict_match = re.search(r'"verdict"\s*:\s*"(pass|revise)"', response)
    reviews_match = re.search(r'"object_reviews"\s*:\s*\[', response)
    if not verdict_match or not reviews_match:
        raise ValueError("Response lacks a recoverable verdict or object_reviews array")

    reviews = complete_json_objects(response, reviews_match.end())
    unique_reviews: list[dict[str, object]] = []
    seen: set[str] = set()
    for review in reviews:
        object_id = review.get("object_id")
        if not isinstance(object_id, str) or object_id in seen:
            continue
        seen.add(object_id)
        unique_reviews.append(review)
    if not unique_reviews:
        raise ValueError("No complete object reviews could be recovered")

    return {
        "verdict": verdict_match.group(1),
        "object_reviews": unique_reviews,
        "global_issues": [],
        "revision_instructions": [],
    }


def main() -> None:
    args = parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    recovered_reviews = 0
    for record in records:
        response = record.get("response")
        if not isinstance(response, str):
            raise SystemExit("Input record has no string response field")
        payload = recover(response)
        canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        record["response"] = canonical
        messages = record.get("messages") or []
        if messages and messages[-1].get("role") == "assistant":
            messages[-1]["content"] = canonical
        recovered_reviews += len(payload["object_reviews"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "status": "RECOVERED",
                "records": len(records),
                "object_reviews": recovered_reviews,
                "deduplication": "first-complete-review-per-object-id",
                "top_level_feedback": "empty-because-truncated",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
