#!/usr/bin/env python3
"""Validate CV801 detection artifacts, split isolation, and box formats."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOX_PATTERN = re.compile(
    r"<ref>[^<]+</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "datas/cv801_detection_v1",
    )
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    root = parse_args().dataset_root.resolve()
    manifest = list(read_jsonl(root / "manifests/images.jsonl"))
    if len(manifest) != 8097:
        raise ValueError(f"Expected 8,097 included images, found {len(manifest)}")
    if len({item["image_id"] for item in manifest}) != len(manifest):
        raise ValueError("Duplicate image_id in manifest")

    groups: dict[str, set[str]] = collections.defaultdict(set)
    counts = collections.Counter()
    object_counts = collections.Counter()
    for item in manifest:
        groups[item["split_group"]].add(item["split"])
        counts[item["split"]] += 1
        object_counts[item["split"]] += item["object_count"]
        image_path = root / item["image_path"]
        label_path = root / item["label_path"]
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing derived pair for {item['image_id']}")
        with Image.open(image_path) as image:
            if image.size != (item["width"], item["height"]) or image.mode != "RGB":
                raise ValueError(f"Bad derived image metadata: {image_path}")
            image.verify()
        labels = [line.split() for line in label_path.read_text().splitlines() if line]
        if len(labels) != item["object_count"]:
            raise ValueError(f"Label count mismatch: {label_path}")
        for label in labels:
            if len(label) != 5 or not 0 <= int(label[0]) < 11:
                raise ValueError(f"Invalid YOLO label: {label_path}")
            if any(not 0 < float(value) <= 1 for value in label[3:]):
                raise ValueError(f"Invalid YOLO box extent: {label_path}")
            if any(not 0 <= float(value) <= 1 for value in label[1:3]):
                raise ValueError(f"Invalid YOLO box center: {label_path}")

    crossing = {group: splits for group, splits in groups.items() if len(splits) != 1}
    if crossing:
        raise ValueError(f"Split leakage in {len(crossing)} duplicate groups")
    expected_images = {"train": 6387, "val": 710, "test": 1000}
    expected_objects = {"train": 11555, "val": 1299, "test": 1747}
    if dict(counts) != expected_images or dict(object_counts) != expected_objects:
        raise ValueError(f"Unexpected split counts: {counts}, {object_counts}")

    for split in ("train", "val", "test"):
        coco = json.loads((root / "coco" / f"instances_{split}.json").read_text())
        if len(coco["images"]) != expected_images[split]:
            raise ValueError(f"COCO image count mismatch for {split}")
        if len(coco["annotations"]) != expected_objects[split]:
            raise ValueError(f"COCO object count mismatch for {split}")
        samples = list(read_jsonl(root / "locateanything" / f"{split}.jsonl"))
        if len(samples) != expected_images[split]:
            raise ValueError(f"LocateAnything count mismatch for {split}")
        for sample in samples:
            answer = sample["conversations"][1]["value"]
            boxes = BOX_PATTERN.findall(answer)
            if not boxes or any(any(not 0 <= int(value) <= 1000 for value in box) for box in boxes):
                raise ValueError(f"Invalid LocateAnything target: {sample['image']}")

    excluded = list(read_jsonl(root / "manifests/excluded_images.jsonl"))
    if len(excluded) != 93:
        raise ValueError(f"Expected 93 excluded images, found {len(excluded)}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "images": dict(counts),
                "objects": dict(object_counts),
                "split_groups": len(groups),
                "crossing_groups": 0,
                "excluded_images": len(excluded),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
