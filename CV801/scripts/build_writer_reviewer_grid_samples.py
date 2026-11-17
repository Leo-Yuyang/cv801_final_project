#!/usr/bin/env python3
"""Build per-image sufficient statistics for the CV801 paired grid analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluate_vlm_predictions import clean_response, is_sample_schema


WRITERS = {"2B", "4B", "9B"}
TIERS = {"low", "mid", "high"}
FIELDS = (
    "sample_id",
    "writer",
    "reviewer_tier",
    "schema_valid",
    "link_matched",
    "link_predicted",
    "link_reference",
    "visual_total",
    "supported_claims",
    "unsupported_claims",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell",
        action="append",
        required=True,
        metavar="WRITER,TIER,STRUCTURED_JSONL,INDEPENDENT_JSONL",
        help="Repeat exactly once for each of the nine grid cells.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=1_000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def image_path(row: dict[str, Any]) -> str:
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("Every row must contain exactly one image")
    image = images[0]
    if isinstance(image, str):
        return image
    if isinstance(image, dict) and isinstance(image.get("path"), str):
        return image["path"]
    raise ValueError("Image row has no usable path")


def parse_response(row: dict[str, Any]) -> dict[str, Any]:
    text, _ = clean_response(row.get("response", ""))
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Invalid final evaluator JSON for image {image_path(row)}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Evaluator output is not an object for image {image_path(row)}")
    return value


def structured_statistics(row: dict[str, Any]) -> tuple[int, int, int, int]:
    reference = json.loads(row["labels"])
    response, _ = clean_response(row.get("response", ""))
    try:
        prediction = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        prediction = None
    reference_objects = reference.get("objects", [])
    if not isinstance(reference_objects, list):
        reference_objects = []
    prediction_objects = prediction.get("objects", []) if isinstance(prediction, dict) else []
    if not isinstance(prediction_objects, list):
        prediction_objects = []
    reference_ids = {
        item["object_id"]
        for item in reference_objects
        if isinstance(item, dict) and isinstance(item.get("object_id"), str)
    }
    predicted_ids = {
        item["object_id"]
        for item in prediction_objects
        if isinstance(item, dict) and isinstance(item.get("object_id"), str)
    }
    return (
        int(is_sample_schema(prediction)),
        len(reference_ids & predicted_ids),
        len(prediction_objects),
        len(reference_objects),
    )


def independent_statistics(value: dict[str, Any]) -> tuple[str, int, int, int]:
    sample_id = value.get("sample_id")
    total = value.get("total")
    reviews = value.get("object_reviews")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Independent evaluator output has no sample_id")
    if not isinstance(total, int) or isinstance(total, bool) or not 0 <= total <= 6:
        raise ValueError(f"Invalid visual total for {sample_id}")
    if not isinstance(reviews, list):
        raise ValueError(f"Invalid object_reviews for {sample_id}")
    supported = 0
    unsupported = 0
    for review in reviews:
        if not isinstance(review, dict):
            raise ValueError(f"Invalid object review for {sample_id}")
        supported_values = review.get("supported_claims")
        unsupported_values = review.get("unsupported_claims")
        if not isinstance(supported_values, list) or not isinstance(unsupported_values, list):
            raise ValueError(f"Invalid claim lists for {sample_id}")
        supported += len(supported_values)
        unsupported += len(unsupported_values)
    return sample_id, total, supported, unsupported


def build_cell(
    writer: str,
    tier: str,
    structured_path: Path,
    independent_path: Path,
    expected_samples: int,
) -> list[dict[str, Any]]:
    structured_rows = read_jsonl(structured_path)
    independent_rows = read_jsonl(independent_path)
    if len(structured_rows) != expected_samples or len(independent_rows) != expected_samples:
        raise ValueError(
            f"{writer}/{tier} has structured={len(structured_rows)} and "
            f"independent={len(independent_rows)}; expected {expected_samples} each"
        )
    structured_by_image = {image_path(row): row for row in structured_rows}
    independent_by_image = {image_path(row): row for row in independent_rows}
    if len(structured_by_image) != expected_samples or len(independent_by_image) != expected_samples:
        raise ValueError(f"{writer}/{tier} contains duplicate image paths")
    if set(structured_by_image) != set(independent_by_image):
        raise ValueError(f"{writer}/{tier} structured/independent image sets differ")
    output = []
    seen_ids: set[str] = set()
    for path in sorted(structured_by_image):
        schema, matched, predicted, reference = structured_statistics(structured_by_image[path])
        sample_id, total, supported, unsupported = independent_statistics(
            parse_response(independent_by_image[path])
        )
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id in {writer}/{tier}: {sample_id}")
        seen_ids.add(sample_id)
        output.append(
            {
                "sample_id": sample_id,
                "writer": writer,
                "reviewer_tier": tier,
                "schema_valid": schema,
                "link_matched": matched,
                "link_predicted": predicted,
                "link_reference": reference,
                "visual_total": total,
                "supported_claims": supported,
                "unsupported_claims": unsupported,
            }
        )
    return output


def parse_cell(specification: str) -> tuple[str, str, Path, Path]:
    parts = specification.split(",", 3)
    if len(parts) != 4:
        raise ValueError(f"Invalid --cell: {specification}")
    writer, tier, structured, independent = (part.strip() for part in parts)
    tier = tier.lower()
    if writer not in WRITERS or tier not in TIERS:
        raise ValueError(f"Invalid Writer/tier in --cell: {specification}")
    return writer, tier, Path(structured), Path(independent)


def main() -> None:
    args = parse_args()
    cells = [parse_cell(specification) for specification in args.cell]
    keys = [(writer, tier) for writer, tier, _, _ in cells]
    expected_keys = {(writer, tier) for writer in WRITERS for tier in TIERS}
    if len(cells) != 9 or set(keys) != expected_keys or len(keys) != len(set(keys)):
        raise ValueError("Provide each of the nine Writer/tier cells exactly once")
    rows = [
        row
        for writer, tier, structured, independent in cells
        for row in build_cell(writer, tier, structured, independent, args.expected_samples)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "PASS", "rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
