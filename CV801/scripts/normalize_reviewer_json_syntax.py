#!/usr/bin/env python3
"""Normalize model-emitted key/value pairs inside reviewer string arrays."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from audit_agent_outputs import valid_reviewer
from evaluate_vlm_predictions import clean_response


ARRAY_KEYS = (
    "supported_claims",
    "unsupported_claims",
    "missed_evidence",
    "global_issues",
    "revision_instructions",
)
PAIR = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:\s*"((?:[^"\\]|\\.)*)"')
NUMBER_PAIR = re.compile(
    r'"((?:[^"\\]|\\.)*)"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def array_end(text: str, start: int) -> int:
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Reviewer response contains an unterminated target array")


def merge_pair(match: re.Match[str]) -> str:
    left = json.loads(f'"{match.group(1)}"')
    right = json.loads(f'"{match.group(2)}"')
    return json.dumps(f"{left}: {right}", ensure_ascii=False)


def merge_number_pair(match: re.Match[str]) -> str:
    left = json.loads(f'"{match.group(1)}"')
    return json.dumps(f"{left}: {match.group(2)}", ensure_ascii=False)


def normalize(text: str) -> dict[str, object]:
    cleaned, _ = clean_response(text)
    for key in ARRAY_KEYS:
        cursor = 0
        marker = f'"{key}"'
        while True:
            key_start = cleaned.find(marker, cursor)
            if key_start < 0:
                break
            start = cleaned.find("[", key_start + len(marker))
            if start < 0:
                raise ValueError(f"Missing array for {key}")
            end = array_end(cleaned, start)
            body = NUMBER_PAIR.sub(merge_number_pair, cleaned[start + 1:end])
            body = PAIR.sub(merge_pair, body)
            cleaned = cleaned[:start + 1] + body + cleaned[end:]
            cursor = start + len(body) + 2
    payload = json.loads(cleaned)
    if not valid_reviewer(payload):
        raise ValueError("Normalized response still violates the reviewer schema")
    return payload


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        response = record.get("response")
        if not isinstance(response, str):
            raise ValueError("Prediction record is missing response")
        canonical = json.dumps(normalize(response), ensure_ascii=False, separators=(",", ":"))
        record["response"] = canonical
        messages = record.get("messages") or []
        if messages and messages[-1].get("role") == "assistant":
            messages[-1]["content"] = canonical
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"samples": len(records), "status": "PASS"}))


if __name__ == "__main__":
    main()
