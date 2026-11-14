#!/usr/bin/env python3
"""Export hidden Golden VLM labels only after every inference branch is frozen."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

try:
    from .prepare_vlm_dataset import build_sample, is_training_caption, read_jsonl, atomic_write_jsonl
except ImportError:
    from prepare_vlm_dataset import build_sample, is_training_caption, read_jsonl, atomic_write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "CV801/agro_workspace/annotations.jsonl",
    )
    parser.add_argument(
        "--image-manifest",
        type=Path,
        default=PROJECT_ROOT / "datas/cv801_detection_v1/manifests/images.jsonl",
    )
    parser.add_argument(
        "--safe-manifest",
        type=Path,
        default=PROJECT_ROOT / "datas/cv801_golden_test_inputs_v1/manifest.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=1000)
    parser.add_argument("--expected-objects", type=int, default=1747)
    return parser.parse_args()


def export_references(
    annotations_path: Path,
    image_manifest_path: Path,
    safe_manifest_path: Path,
    output: Path,
    expected_images: int,
    expected_objects: int,
) -> dict[str, int]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen Golden references: {output}")
    images = {record["image_id"]: record for record in read_jsonl(image_manifest_path)}
    safe_records = list(read_jsonl(safe_manifest_path))
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in read_jsonl(annotations_path):
        image = images.get(record["image_id"])
        if image is not None and image["split"] == "test" and is_training_caption(record):
            grouped[record["image_id"]].append(record)

    exported: list[dict[str, Any]] = []
    object_count = 0
    for safe in safe_records:
        image_id = safe["image_id"]
        image = images.get(image_id)
        records = grouped.get(image_id, [])
        if image is None or image["split"] != "test" or not records:
            raise ValueError(f"Golden reference is missing for {image_id}")
        sample, audit = build_sample(image, records)
        if sample["images"][0] != safe["image_path"]:
            raise ValueError(f"Golden image mismatch for {image_id}")
        label = sample["messages"][-1]["content"]
        exported.append({
            "sample_id": safe["sample_id"],
            "images": sample["images"],
            "labels": label,
            "object_count": audit["object_count"],
            "label_source": audit["label_source"],
        })
        object_count += int(audit["object_count"])
    if len(exported) != expected_images or object_count != expected_objects:
        raise ValueError(
            f"Expected {expected_images} images/{expected_objects} objects, "
            f"found {len(exported)} images/{object_count} objects"
        )
    atomic_write_jsonl(output, exported)
    return {"images": len(exported), "objects": object_count}


def main() -> None:
    args = parse_args()
    result = export_references(
        args.annotations,
        args.image_manifest,
        args.safe_manifest,
        args.output,
        args.expected_images,
        args.expected_objects,
    )
    print(json.dumps({**result, "references_exported_post_inference": True}))


if __name__ == "__main__":
    main()
