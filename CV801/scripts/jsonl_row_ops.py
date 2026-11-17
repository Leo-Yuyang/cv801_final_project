#!/usr/bin/env python3
"""Select, replace, and merge JSONL rows with strict input alignment checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_image(record: dict[str, Any]) -> str:
    images = record.get("images") or []
    if not images:
        raise ValueError("Record has no image")
    image = images[0]
    return str(image.get("path") if isinstance(image, dict) else image)


def write_jsonl(rows: list[dict[str, Any]], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)


def select_rows(source: Path, output: Path, start: int, end: int) -> int:
    rows = read_jsonl(source)
    if start < 1 or end < start or end > len(rows):
        raise ValueError(f"Invalid inclusive range {start}:{end} for {len(rows)} rows")
    selected = rows[start - 1 : end]
    write_jsonl(selected, output)
    return len(selected)


def select_line_numbers(source: Path, output: Path, line_numbers: list[int]) -> int:
    rows = read_jsonl(source)
    if len(set(line_numbers)) != len(line_numbers):
        raise ValueError("Selected line numbers must be unique")
    if any(line_number < 1 or line_number > len(rows) for line_number in line_numbers):
        raise ValueError(f"Selected line number is out of range for {len(rows)} rows")
    selected = [rows[line_number - 1] for line_number in line_numbers]
    write_jsonl(selected, output)
    return len(selected)


def replace_rows(source: Path, replacements: Path, output: Path, line_numbers: list[int]) -> int:
    rows = read_jsonl(source)
    replacement_rows = read_jsonl(replacements)
    if len(set(line_numbers)) != len(line_numbers):
        raise ValueError("Replacement line numbers must be unique")
    if len(line_numbers) != len(replacement_rows):
        raise ValueError("Replacement count does not match line-number count")
    for line_number, replacement in zip(line_numbers, replacement_rows, strict=True):
        if line_number < 1 or line_number > len(rows):
            raise ValueError(f"Replacement line {line_number} is out of range")
        original = rows[line_number - 1]
        if first_image(original) != first_image(replacement):
            raise ValueError(f"Image mismatch at replacement line {line_number}")
        rows[line_number - 1] = replacement
    write_jsonl(rows, output)
    return len(rows)


def last_message_content(record: dict[str, Any], role: str) -> str:
    messages = record.get("messages") or []
    matching = [message for message in messages if message.get("role") == role]
    if not matching or not isinstance(matching[-1].get("content"), str):
        raise ValueError(f"Record has no {role} message with text content")
    return str(matching[-1]["content"])


def replace_responses(
    source: Path, replacements: Path, output: Path, line_numbers: list[int]
) -> int:
    """Replace text-only inference responses after checking prompt ancestry."""
    rows = read_jsonl(source)
    replacement_rows = read_jsonl(replacements)
    if len(set(line_numbers)) != len(line_numbers):
        raise ValueError("Replacement line numbers must be unique")
    if len(line_numbers) != len(replacement_rows):
        raise ValueError("Replacement count does not match line-number count")
    for line_number, replacement in zip(line_numbers, replacement_rows, strict=True):
        if line_number < 1 or line_number > len(rows):
            raise ValueError(f"Replacement line {line_number} is out of range")
        original = rows[line_number - 1]
        original_user = last_message_content(original, "user")
        replacement_user = last_message_content(replacement, "user")
        if not replacement_user.startswith(original_user):
            raise ValueError(f"Prompt mismatch at replacement line {line_number}")
        response = replacement.get("response")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"Replacement line {line_number} has no response text")
        original["response"] = response
        messages = original.get("messages") or []
        if messages and messages[-1].get("role") == "assistant":
            messages[-1]["content"] = response
    write_jsonl(rows, output)
    return len(rows)


def merge_rows(sources: list[Path], reference_input: Path, output: Path) -> int:
    merged: list[dict[str, Any]] = []
    for source in sources:
        merged.extend(read_jsonl(source))
    references = read_jsonl(reference_input)
    if len(merged) != len(references):
        raise ValueError(f"Merged count {len(merged)} does not match reference count {len(references)}")
    for index, (prediction, reference) in enumerate(zip(merged, references, strict=True), 1):
        if prediction.get("images"):
            if first_image(prediction) != first_image(reference):
                raise ValueError(f"Image mismatch at merged line {index}")
        elif last_message_content(prediction, "user") != last_message_content(reference, "user"):
            raise ValueError(f"Prompt mismatch at merged line {index}")
    write_jsonl(merged, output)
    return len(merged)


def concatenate_rows(sources: list[Path], output: Path) -> int:
    rows = [row for source in sources for row in read_jsonl(source)]
    write_jsonl(rows, output)
    return len(rows)


def append_user_instruction(source: Path, output: Path, instruction: str) -> int:
    rows = read_jsonl(source)
    if not instruction.strip():
        raise ValueError("Instruction must not be empty")
    for index, row in enumerate(rows, 1):
        messages = row.get("messages") or []
        user_messages = [message for message in messages if message.get("role") == "user"]
        if not user_messages:
            raise ValueError(f"Row {index} has no user message")
        user = user_messages[-1]
        content = user.get("content")
        if not isinstance(content, str):
            raise ValueError(f"Row {index} user content is not text")
        user["content"] = content.rstrip() + "\n\n" + instruction.strip()
    write_jsonl(rows, output)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--source", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--start", type=int, required=True, help="One-based inclusive start line")
    select.add_argument("--end", type=int, required=True, help="One-based inclusive end line")

    select_lines = subparsers.add_parser("select-lines")
    select_lines.add_argument("--source", type=Path, required=True)
    select_lines.add_argument("--output", type=Path, required=True)
    select_lines.add_argument("--line-numbers", type=int, nargs="+", required=True)

    replace = subparsers.add_parser("replace")
    replace.add_argument("--source", type=Path, required=True)
    replace.add_argument("--replacements", type=Path, required=True)
    replace.add_argument("--output", type=Path, required=True)
    replace.add_argument("--line-numbers", type=int, nargs="+", required=True)

    replace_responses_parser = subparsers.add_parser("replace-responses")
    replace_responses_parser.add_argument("--source", type=Path, required=True)
    replace_responses_parser.add_argument("--replacements", type=Path, required=True)
    replace_responses_parser.add_argument("--output", type=Path, required=True)
    replace_responses_parser.add_argument("--line-numbers", type=int, nargs="+", required=True)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--sources", type=Path, nargs="+", required=True)
    merge.add_argument("--reference-input", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)

    concatenate = subparsers.add_parser("concat")
    concatenate.add_argument("--sources", type=Path, nargs="+", required=True)
    concatenate.add_argument("--output", type=Path, required=True)

    append = subparsers.add_parser("append-instruction")
    append.add_argument("--source", type=Path, required=True)
    append.add_argument("--output", type=Path, required=True)
    append.add_argument("--instruction", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "select":
        count = select_rows(args.source, args.output, args.start, args.end)
    elif args.command == "select-lines":
        count = select_line_numbers(args.source, args.output, args.line_numbers)
    elif args.command == "replace":
        count = replace_rows(args.source, args.replacements, args.output, args.line_numbers)
    elif args.command == "replace-responses":
        count = replace_responses(args.source, args.replacements, args.output, args.line_numbers)
    elif args.command == "merge":
        count = merge_rows(args.sources, args.reference_input, args.output)
    elif args.command == "concat":
        count = concatenate_rows(args.sources, args.output)
    else:
        count = append_user_instruction(args.source, args.output, args.instruction)
    print(json.dumps({"command": args.command, "samples": count}))


if __name__ == "__main__":
    main()
