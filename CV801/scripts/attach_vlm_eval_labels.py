#!/usr/bin/env python3
"""Attach hidden references to frozen predictions only after inference completes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Use exactly the first N references; the prediction file must also contain N rows.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_image(record: dict[str, Any]) -> str:
    images = record.get("images") or []
    if not images:
        raise ValueError("Record has no image")
    image = images[0]
    return str(image.get("path") if isinstance(image, dict) else image)


def attach_labels(
    predictions: Path,
    reference_predictions: Path,
    output: Path,
    max_samples: int | None = None,
) -> int:
    generated = read_jsonl(predictions)
    references = read_jsonl(reference_predictions)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        if len(generated) != max_samples:
            raise ValueError("Prediction count does not match --max-samples")
        references = references[:max_samples]
    if len(generated) != len(references):
        raise ValueError("Prediction and reference counts differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, (prediction, reference) in enumerate(zip(generated, references, strict=True), 1):
            if prediction.get("labels") is not None:
                raise ValueError(f"Prediction line {index} already contains non-empty labels")
            if first_image(prediction) != first_image(reference):
                raise ValueError(f"Image mismatch at line {index}")
            labels = reference.get("labels")
            if not isinstance(labels, str):
                raise ValueError(f"Reference line {index} has no serialized label")
            frozen = dict(prediction)
            frozen["labels"] = labels
            frozen["labels_attached_post_inference"] = True
            handle.write(json.dumps(frozen, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(generated)


def main() -> None:
    args = parse_args()
    count = attach_labels(
        args.predictions,
        args.reference_predictions,
        args.output,
        max_samples=args.max_samples,
    )
    print(json.dumps({"samples": count, "labels_attached_post_inference": True}))


if __name__ == "__main__":
    main()
