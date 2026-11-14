#!/usr/bin/env python3
"""Build leakage-safe T1 data for agricultural vision-language SFT."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT = (
    "You are an agricultural vision assistant. Describe only visible evidence, "
    "keep every object separate, and return valid JSON without markdown."
)


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
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "datas/cv801_vlm_v1",
    )
    parser.add_argument("--smoke-train-images", type=int, default=32)
    parser.add_argument("--smoke-val-images", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def is_training_caption(record: dict[str, Any]) -> bool:
    return bool(
        record.get("localization_decision") in {"accept", "correct"}
        and record.get("ai_analysis_status") == "complete"
        and record.get("ai_target_valid") is True
        and isinstance(record.get("ai_caption"), dict)
    )


def spatial_key(record: dict[str, Any]) -> tuple[float, float, str]:
    x1, y1, x2, y2 = record["bbox"]
    return ((y1 + y2) / 2, (x1 + x2) / 2, record["annotation_id"])


def assessment_from(record: dict[str, Any]) -> dict[str, str]:
    caption = record["ai_caption"]
    if record["image_type"] == "fruit":
        return {
            "stage": caption["stage"],
            "disease": caption["fruit_disease"],
            "severity": caption["severity"],
        }
    return {
        "disease": caption["disease"],
        "severity": caption["severity"],
    }


def target_object(record: dict[str, Any], index: int) -> dict[str, Any]:
    caption = record["ai_caption"]
    return {
        "object_id": f"object_{index:03d}",
        "caption": caption["evidence"],
        "assessment": assessment_from(record),
        "image_quality": caption["image_quality"],
        "confidence": caption["confidence"],
        "action": caption["action"],
    }


def user_prompt(crop: str, image_type: str) -> str:
    return (
        f"<image>\nAnalyze every visible target object in this {crop} {image_type} image. "
        "Return one JSON object with keys crop, plant_part, and objects. Assign object IDs "
        "object_001, object_002, ... in top-to-bottom then left-to-right order. Each object "
        "must contain object_id, caption, assessment, image_quality, confidence, and action. "
        "Do not output coordinates or bounding boxes. Do not infer details that are not visible."
    )


def build_sample(image: dict[str, Any], records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    image_types = {record["image_type"] for record in records}
    crops = {record["crop"] for record in records}
    if image_types != {image["image_type"]} or crops != {image["crop"]}:
        raise ValueError(f"Inconsistent metadata for {image['image_id']}")
    ordered = sorted(records, key=spatial_key)
    target = {
        "crop": image["crop"],
        "plant_part": image["image_type"],
        "objects": [target_object(record, index) for index, record in enumerate(ordered, 1)],
    }
    sample = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(image["crop"], image["image_type"])},
            {
                "role": "assistant",
                "content": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "images": [str(Path("datas/cv801_detection_v1") / image["image_path"])],
    }
    audit = {
        "sample_id": hashlib.sha256(image["image_id"].encode()).hexdigest()[:16],
        "image_id": image["image_id"],
        "image_path": sample["images"][0],
        "split": image["split"],
        "crop": image["crop"],
        "plant_part": image["image_type"],
        "object_count": len(ordered),
        "annotation_ids": [record["annotation_id"] for record in ordered],
        "reference_boxes": [record["bbox"] for record in ordered],
        "label_source": "reviewed_gpt_caption_not_independent_reference",
        "boxes_exposed_to_t1": False,
    }
    return sample, audit


def stable_subset(records: list[tuple[dict[str, Any], dict[str, Any]]], size: int):
    return sorted(records, key=lambda item: item[1]["sample_id"])[:size]


def longest_target(records: list[tuple[dict[str, Any], dict[str, Any]]]):
    """Select a deterministic long-tail sample for sequence-length stress tests."""
    return max(
        records,
        key=lambda item: (len(item[0]["messages"][-1]["content"]), item[1]["sample_id"]),
    )


def build_dataset(
    annotations_path: Path,
    manifest_path: Path,
    output_root: Path,
    smoke_train_images: int,
    smoke_val_images: int,
) -> dict[str, Any]:
    images = {record["image_id"]: record for record in read_jsonl(manifest_path)}
    expected_splits = {"train", "val", "test"}
    observed_splits = {record["split"] for record in images.values()}
    if observed_splits != expected_splits:
        raise ValueError(f"Expected frozen train/val/test splits, found {sorted(observed_splits)}")

    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    exclusion_counts: collections.Counter[str] = collections.Counter()
    for record in read_jsonl(annotations_path):
        image = images.get(record["image_id"])
        if image is None:
            exclusion_counts["outside_frozen_manifest"] += 1
        elif not is_training_caption(record):
            exclusion_counts[f"ineligible_{image['split']}"] += 1
        else:
            grouped[record["image_id"]].append(record)

    built: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {"train": [], "val": []}
    eligible_test_images = 0
    eligible_test_objects = 0
    for image_id, records in grouped.items():
        image = images[image_id]
        if image["split"] == "test":
            eligible_test_images += 1
            eligible_test_objects += len(records)
            continue
        sample, audit = build_sample(image, records)
        built[image["split"]].append((sample, audit))

    for split in built:
        built[split].sort(key=lambda item: item[1]["image_id"])
        atomic_write_jsonl(output_root / f"t1_{split}.jsonl", (item[0] for item in built[split]))
        atomic_write_jsonl(output_root / "manifests" / f"t1_{split}.jsonl", (item[1] for item in built[split]))

    smoke_train = stable_subset(built["train"], smoke_train_images)
    smoke_val = stable_subset(built["val"], smoke_val_images)
    atomic_write_jsonl(output_root / "t1_train_smoke.jsonl", (item[0] for item in smoke_train))
    atomic_write_jsonl(output_root / "t1_val_smoke.jsonl", (item[0] for item in smoke_val))
    stress_train = longest_target(built["train"])
    stress_val = longest_target(built["val"])
    atomic_write_jsonl(output_root / "t1_train_stress.jsonl", [stress_train[0]])
    atomic_write_jsonl(output_root / "t1_val_stress.jsonl", [stress_val[0]])

    summary = {
        "schema_version": "cv801_vlm_v1",
        "task": "T1 full-image agricultural object captioning without input boxes",
        "label_source": "reviewed GPT captions; not independent engineer reference labels",
        "split_policy": "reuse cv801_detection_v1 exactly",
        "t1": {
            split: {
                "images": len(items),
                "objects": sum(item[1]["object_count"] for item in items),
                "multi_object_images": sum(item[1]["object_count"] > 1 for item in items),
            }
            for split, items in built.items()
        },
        "smoke": {
            "train_images": len(smoke_train),
            "val_images": len(smoke_val),
            "stress_train_objects": stress_train[1]["object_count"],
            "stress_val_objects": stress_val[1]["object_count"],
        },
        "golden_test": {
            "eligible_images_present_but_not_exported": eligible_test_images,
            "eligible_objects_present_but_not_exported": eligible_test_objects,
        },
        "excluded_annotations": dict(sorted(exclusion_counts.items())),
        "t2_status": "blocked until one localization output is frozen and matched to reference objects",
    }
    atomic_write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        args.annotations,
        args.image_manifest,
        args.output_root,
        args.smoke_train_images,
        args.smoke_val_images,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
