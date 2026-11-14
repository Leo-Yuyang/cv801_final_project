#!/usr/bin/env python3
"""Canonicalize JSON inference responses without changing their payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .evaluate_vlm_predictions import clean_response
except ImportError:
    from evaluate_vlm_predictions import clean_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def canonical_payload(response: str) -> dict[str, Any]:
    cleaned, _ = clean_response(response)
    decoder = json.JSONDecoder()
    payload, end = decoder.raw_decode(cleaned)
    trailing = cleaned[end:].strip()
    if trailing and set(trailing) != {"}"}:
        raise ValueError(f"Unsafe non-brace trailing response content: {trailing[:80]!r}")
    if not isinstance(payload, dict):
        raise ValueError("Response JSON root must be an object")
    return payload


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        response = record.get("response")
        if not isinstance(response, str):
            raise ValueError("Prediction record has no response string")
        canonical = json.dumps(canonical_payload(response), ensure_ascii=False, separators=(",", ":"))
        record["response"] = canonical
        messages = record.get("messages") or []
        if messages and messages[-1].get("role") == "assistant":
            messages[-1]["content"] = canonical
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"samples": len(records), "canonicalized": True}))


if __name__ == "__main__":
    main()
