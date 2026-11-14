#!/usr/bin/env python3
"""Build T2 SFT data by adding frozen detector proposals to the T1 inputs."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTION_PREFIX = "datas/cv801_detection_v1/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1-root", type=Path, default=PROJECT_ROOT / "datas/cv801_vlm_v1")
    parser.add_argument(
        "--detection-root", type=Path, default=PROJECT_ROOT / "datas/cv801_detection_v1"
    )
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "datas/cv801_vlm_t2_yolo26x_v1"
    )
    parser.add_argument("--locator-model", type=Path, required=True)
    parser.add_argument("--locator-metrics", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-proposals", type=int, default=20)
    parser.add_argument("--coverage-iou", type=float, default=0.5)
    parser.add_argument("--smoke-train-images", type=int, default=32)
    parser.add_argument("--smoke-val-images", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = box
    return [x, y, x + width, y + height]


def iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def greedy_match_count(
    references: list[list[float]], proposals: list[list[float]], threshold: float
) -> int:
    candidates = sorted(
        (
            (iou(reference, proposal), reference_index, proposal_index)
            for reference_index, reference in enumerate(references)
            for proposal_index, proposal in enumerate(proposals)
        ),
        reverse=True,
    )
    used_references: set[int] = set()
    used_proposals: set[int] = set()
    for overlap, reference_index, proposal_index in candidates:
        if overlap < threshold:
            break
        if reference_index in used_references or proposal_index in used_proposals:
            continue
        used_references.add(reference_index)
        used_proposals.add(proposal_index)
    return len(used_references)


def normalized_box(box: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = xywh_to_xyxy(box)
    values = [x1 / width, y1 / height, x2 / width, y2 / height]
    return [max(0, min(1000, round(value * 1000))) for value in values]


def detection_file_name(image_path: str) -> str:
    if not image_path.startswith(DETECTION_PREFIX):
        raise ValueError(f"Unexpected image path outside detection dataset: {image_path}")
    return image_path[len(DETECTION_PREFIX):]


def proposal_prompt(proposals: list[dict[str, Any]]) -> str:
    compact = json.dumps(proposals, ensure_ascii=False, separators=(",", ":"))
    return (
        "\nFrozen YOLO26x localization hints are provided below. Coordinates are "
        "[x1,y1,x2,y2] normalized to 0-1000 over the full image. These predictions may "
        "contain false positives or miss objects: use them to focus visual inspection, but "
        "still analyze every visible target object, keep the required spatial object order, "
        "and do not include coordinates in the JSON response.\n"
        f"Predicted localization hints: {compact}"
    )


def build_split(
    split: str,
    t1_root: Path,
    detection_root: Path,
    predictions_path: Path,
    score_threshold: float,
    max_proposals: int,
    coverage_iou: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    samples = read_jsonl(t1_root / f"t1_{split}.jsonl")
    manifests = read_jsonl(t1_root / "manifests" / f"t1_{split}.jsonl")
    if len(samples) != len(manifests):
        raise ValueError(f"T1 {split} samples/manifests length mismatch")

    coco = json.loads(
        (detection_root / "coco" / f"instances_{split}.json").read_text(encoding="utf-8")
    )
    image_by_file = {image["file_name"]: image for image in coco["images"]}
    category_name = {int(category["id"]): category["name"] for category in coco["categories"]}
    category_id = {category["name"]: int(category["id"]) for category in coco["categories"]}
    predictions_by_image: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for prediction in json.loads(predictions_path.read_text(encoding="utf-8")):
        predictions_by_image[int(prediction["image_id"])].append(prediction)

    built_samples: list[dict[str, Any]] = []
    built_manifests: list[dict[str, Any]] = []
    total_references = 0
    covered_references = 0
    proposal_counts: list[int] = []
    target_mismatches = 0
    for source_sample, source_manifest in zip(samples, manifests, strict=True):
        if source_sample["images"][0] != source_manifest["image_path"]:
            raise ValueError(f"T1 image mismatch for {source_manifest['image_id']}")
        file_name = detection_file_name(source_manifest["image_path"])
        image = image_by_file.get(file_name)
        if image is None:
            raise ValueError(f"No COCO image for {file_name}")
        candidates = [
            prediction
            for prediction in predictions_by_image.get(int(image["id"]), [])
            if float(prediction["score"]) >= score_threshold
        ]
        candidates.sort(key=lambda prediction: float(prediction["score"]), reverse=True)
        candidates = candidates[:max_proposals]
        candidates.sort(
            key=lambda prediction: (
                prediction["bbox"][1] + prediction["bbox"][3] / 2,
                prediction["bbox"][0] + prediction["bbox"][2] / 2,
                -float(prediction["score"]),
            )
        )
        proposals = [
            {
                "proposal_id": f"box_{index:03d}",
                "label": category_name[int(prediction["category_id"])],
                "box": normalized_box(prediction["bbox"], int(image["width"]), int(image["height"])),
                "score": round(float(prediction["score"]), 4),
            }
            for index, prediction in enumerate(candidates, 1)
        ]

        sample = copy.deepcopy(source_sample)
        original_target = sample["messages"][-1]["content"]
        sample["messages"][1]["content"] += proposal_prompt(proposals)
        if sample["messages"][-1]["content"] != original_target:
            target_mismatches += 1
        built_samples.append(sample)

        expected_category = category_id[source_manifest["crop"]]
        relevant_boxes = [
            xywh_to_xyxy(prediction["bbox"])
            for prediction in candidates
            if int(prediction["category_id"]) == expected_category
        ]
        matched = greedy_match_count(
            source_manifest["reference_boxes"], relevant_boxes, coverage_iou
        )
        total_references += int(source_manifest["object_count"])
        covered_references += matched
        proposal_counts.append(len(proposals))
        manifest = copy.deepcopy(source_manifest)
        manifest.update({
            "boxes_exposed_to_t2": True,
            "locator": "YOLO26x fine-tuned train-357714 best.pt",
            "locator_score_threshold": score_threshold,
            "locator_max_proposals": max_proposals,
            "predicted_proposals": proposals,
            "reference_objects_covered_at_iou": matched,
            "coverage_iou_threshold": coverage_iou,
        })
        built_manifests.append(manifest)

    if target_mismatches:
        raise AssertionError(f"T2 changed {target_mismatches} assistant targets")
    ordered_counts = sorted(proposal_counts)
    split_summary = {
        "images": len(built_samples),
        "reference_objects": total_references,
        "reference_objects_covered_at_iou": covered_references,
        "reference_coverage": covered_references / total_references if total_references else None,
        "empty_proposal_images": sum(count == 0 for count in proposal_counts),
        "mean_proposals": sum(proposal_counts) / len(proposal_counts),
        "p95_proposals": ordered_counts[min(len(ordered_counts) - 1, round(0.95 * len(ordered_counts)))],
        "assistant_targets_changed": target_mismatches,
    }
    return built_samples, built_manifests, split_summary


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    built: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = {}
    for split, predictions in (
        ("train", args.train_predictions),
        ("val", args.val_predictions),
    ):
        built[split] = build_split(
            split,
            args.t1_root.resolve(),
            args.detection_root.resolve(),
            predictions.resolve(),
            args.score_threshold,
            args.max_proposals,
            args.coverage_iou,
        )
        samples, manifests, _ = built[split]
        atomic_write_jsonl(output_root / f"t2_{split}.jsonl", samples)
        atomic_write_jsonl(output_root / "manifests" / f"t2_{split}.jsonl", manifests)

    train_samples = built["train"][0]
    val_samples = built["val"][0]
    atomic_write_jsonl(output_root / "t2_train_smoke.jsonl", train_samples[: args.smoke_train_images])
    atomic_write_jsonl(output_root / "t2_val_smoke.jsonl", val_samples[: args.smoke_val_images])
    atomic_write_jsonl(
        output_root / "t2_train_stress.jsonl",
        [max(train_samples, key=lambda sample: sum(len(message["content"]) for message in sample["messages"]))],
    )
    atomic_write_jsonl(
        output_root / "t2_val_stress.jsonl",
        [max(val_samples, key=lambda sample: sum(len(message["content"]) for message in sample["messages"]))],
    )

    locator_metrics = json.loads(args.locator_metrics.read_text(encoding="utf-8"))
    summary = {
        "schema_version": "cv801_vlm_t2_yolo26x_v1",
        "task": "T2 full-image captioning with frozen predicted localization hints",
        "comparison_contract": "same images and byte-identical assistant targets as T1; only user input adds predictions",
        "label_source": "reviewed GPT captions; not independent engineer reference labels",
        "locator": {
            "name": "YOLO26x fine-tuned",
            "model": str(args.locator_model.resolve()),
            "model_sha256": sha256_file(args.locator_model),
            "validation_metrics": str(args.locator_metrics.resolve()),
            "validation_AP_50_95": locator_metrics["metrics"]["AP_50_95"],
            "prediction_score_threshold": args.score_threshold,
            "max_proposals_per_image": args.max_proposals,
            "coordinates": "xyxy normalized integers in [0,1000]",
        },
        "splits": {split: built[split][2] for split in ("train", "val")},
        "golden_test": "sealed; no T2 test dataset generated",
    }
    atomic_write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = build_dataset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
