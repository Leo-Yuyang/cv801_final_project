#!/usr/bin/env python3
"""Repair only invalid Judge/Recheck rows while preserving valid responses verbatim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_agent_outputs import valid_judge, valid_recheck
from prepare_agent_review_inputs import final_response
from recover_truncated_decision_json import recover


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("judge", "recheck"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    validators = {"judge": valid_judge, "recheck": valid_recheck}
    validator = validators[args.kind]
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    repaired_lines: list[int] = []
    for line_number, record in enumerate(records, 1):
        response = final_response(record)
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            payload = None
        if validator(payload):
            continue
        recovered = recover(args.kind, response)
        if not validator(recovered):
            raise ValueError(f"Recovery did not satisfy {args.kind} schema at line {line_number}")
        canonical = json.dumps(recovered, ensure_ascii=False, separators=(",", ":"))
        record["response"] = canonical
        messages = record.get("messages") or []
        if messages and messages[-1].get("role") == "assistant":
            messages[-1]["content"] = canonical
        repaired_lines.append(line_number)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"status": "PASS", "kind": args.kind, "samples": len(records), "repaired_lines": repaired_lines}))


if __name__ == "__main__":
    main()
