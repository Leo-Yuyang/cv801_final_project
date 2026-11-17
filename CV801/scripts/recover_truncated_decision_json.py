#!/usr/bin/env python3
"""Conservatively recover complete list items from truncated Judge/Recheck JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CONTRACTS = {
    "judge": {
        "decisions": ("accept", "revise"),
        "lists": {
            "agreements": ("agreements", "review_agreements"),
            "resolved_disagreements": ("resolved_disagreements",),
            "revision_feedback": ("revision_feedback",),
            "recheck_requirements": ("recheck_requirements",),
        },
    },
    "recheck": {
        "decisions": ("accept", "revise_again"),
        "lists": {
            "resolved_issues": ("resolved_issues",),
            "remaining_issues": ("remaining_issues",),
            "revision_feedback": ("revision_feedback",),
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=tuple(CONTRACTS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def array_start(text: str, aliases: tuple[str, ...]) -> int | None:
    matches = []
    for alias in aliases:
        match = re.search(rf'"{re.escape(alias)}"\s*:\s*\[', text)
        if match:
            matches.append(match)
    return min(matches, key=lambda match: match.start()).end() if matches else None


def complete_string_items(text: str, start: int | None) -> list[str]:
    if start is None:
        return []
    items: list[str] = []
    seen: set[str] = set()
    index = start
    nested = 0
    while index < len(text):
        character = text[index]
        if character == "]" and nested == 0:
            break
        if character in "[{":
            nested += 1
            index += 1
            continue
        if character in "]}" and nested:
            nested -= 1
            index += 1
            continue
        if character != '"' or nested:
            index += 1
            continue
        try:
            item, consumed = json.JSONDecoder().raw_decode(text[index:])
        except json.JSONDecodeError:
            break
        if not isinstance(item, str):
            index += max(consumed, 1)
            continue
        if item not in seen:
            seen.add(item)
            items.append(item)
        index += consumed
    return items


def recover(kind: str, response: str) -> dict[str, object]:
    contract = CONTRACTS[kind]
    allowed = "|".join(contract["decisions"])
    decision_match = re.search(rf'"decision"\s*:\s*"({allowed})"', response)
    if not decision_match:
        raise ValueError("Response lacks a recoverable decision")
    payload: dict[str, object] = {"decision": decision_match.group(1)}
    for canonical, aliases in contract["lists"].items():
        payload[canonical] = complete_string_items(response, array_start(response, aliases))
    return payload


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    recovered_items = 0
    for record in records:
        response = record.get("response")
        if not isinstance(response, str):
            raise ValueError("Prediction record is missing response")
        payload = recover(args.kind, response)
        recovered_items += sum(len(value) for value in payload.values() if isinstance(value, list))
        canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        record["response"] = canonical
        messages = record.get("messages") or []
        if messages and messages[-1].get("role") == "assistant":
            messages[-1]["content"] = canonical
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"status": "RECOVERED", "kind": args.kind, "records": len(records), "complete_items": recovered_items}))


if __name__ == "__main__":
    main()
