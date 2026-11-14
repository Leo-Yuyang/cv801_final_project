#!/usr/bin/env python3
"""Prepare reference-free CV801 golden-test inputs before protocol unsealing."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.prepare_t2_dataset import normalized_box, proposal_prompt
    from scripts.prepare_vlm_dataset import SYSTEM_PROMPT, user_prompt
except ModuleNotFoundError:
    from prepare_t2_dataset import normalized_box, proposal_prompt
    from prepare_vlm_dataset import SYSTEM_PROMPT, user_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTION_PREFIX = "datas/cv801_detection_v1/"
SAFE_MANIFEST_KEYS = {
    "sample_id",
    "image_id",
    "image_path",
    "coco_image_id",
    "crop",
    "plant_part",
    "width",
    "height",
    "file_sha256",
}
FORBIDDEN_MODEL_KEYS = {
    "annotations",
    "assistant",
    "bbox",
    "ground_truth",
    "label",
    "labels",
    "object_count",
    "reference",
    "reference_boxes",
    "response",
    "target",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    base = subparsers.add_parser("base", help="Build reference-free T0/T1 and detector inputs.")
    base.add_argument(
        "--image-manifest",
        type=Path,
        default=PROJECT_ROOT / "datas/cv801_detection_v1/manifests/images.jsonl",
    )
    base.add_argument(
        "--classes",
        type=Path,
        default=PROJECT_ROOT / "datas/cv801_detection_v1/manifests/classes.json",
    )
    base.add_argument(
        "--detection-root",
        type=Path,
        default=PROJECT_ROOT / "datas/cv801_detection_v1",
    )
    base.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "datas/cv801_golden_test_inputs_v1",
    )
    base.add_argument("--expected-images", type=int, default=1000)

    t2 = subparsers.add_parser("t2", help="Add frozen YOLO predictions to golden Writer inputs.")
    t2.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "datas/cv801_golden_test_inputs_v1",
    )
    t2.add_argument("--predictions", type=Path, required=True)
    t2.add_argument("--locator-model", type=Path, required=True)
    t2.add_argument("--score-threshold", type=float, default=0.25)
    t2.add_argument("--max-proposals", type=int, default=20)
    t2.add_argument("--expected-images", type=int, default=1000)

    audit = subparsers.add_parser("audit", help="Verify golden model inputs contain no references.")
    audit.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "datas/cv801_golden_test_inputs_v1",
    )
    audit.add_argument("--require-t2", action="store_true")
    audit.add_argument("--expected-images", type=int, default=1000)

    predictions = subparsers.add_parser(
        "predictions", help="Verify reference-free Writer prediction alignment."
    )
    predictions.add_argument("--predictions", type=Path, required=True)
    predictions.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "datas/cv801_golden_test_inputs_v1",
    )
    predictions.add_argument("--variant", choices=("t0_t1", "t2"), required=True)
    predictions.add_argument("--expected-images", type=int, default=1000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, record: Any) -> None:
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


def _assert_count(records: list[Any], expected: int, name: str) -> None:
    if len(records) != expected:
        raise ValueError(f"Expected {expected} {name}, found {len(records)}")


def build_base(args: argparse.Namespace) -> dict[str, Any]:
    source_records = read_jsonl(args.image_manifest.resolve())
    test_records = sorted(
        (record for record in source_records if record.get("split") == "test"),
        key=lambda record: record["image_id"],
    )
    _assert_count(test_records, args.expected_images, "golden test images")
    classes_payload = json.loads(args.classes.read_text(encoding="utf-8"))
    classes = classes_payload["names"]
    if len(classes) != len(set(classes)):
        raise ValueError("Detection class names are not unique")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    writer_records: list[dict[str, Any]] = []
    safe_manifests: list[dict[str, Any]] = []
    detector_images: list[dict[str, Any]] = []
    detection_root = args.detection_root.resolve()
    for coco_image_id, record in enumerate(test_records, 1):
        image_id = str(record["image_id"])
        relative_path = str(record["image_path"])
        crop = str(record["crop"])
        plant_part = str(record["image_type"])
        width, height = int(record["width"]), int(record["height"])
        if image_id in seen_ids or relative_path in seen_paths:
            raise ValueError(f"Duplicate golden image identity: {image_id} / {relative_path}")
        if crop not in classes or plant_part not in {"fruit", "leaf"}:
            raise ValueError(f"Invalid golden metadata for {image_id}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions for {image_id}")
        image_file = detection_root / relative_path
        if not image_file.is_file():
            raise FileNotFoundError(image_file)
        seen_ids.add(image_id)
        seen_paths.add(relative_path)

        model_path = f"{DETECTION_PREFIX}{relative_path}"
        writer_records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(crop, plant_part)},
            ],
            "images": [model_path],
        })
        safe_manifests.append({
            "sample_id": hashlib.sha256(image_id.encode()).hexdigest()[:16],
            "image_id": image_id,
            "image_path": model_path,
            "coco_image_id": coco_image_id,
            "crop": crop,
            "plant_part": plant_part,
            "width": width,
            "height": height,
            "file_sha256": str(record["file_sha256"]),
        })
        detector_images.append({
            "id": coco_image_id,
            "file_name": relative_path,
            "width": width,
            "height": height,
        })

    detector_payload = {
        "info": {
            "description": "CV801 golden test detector input without annotations",
            "version": "1.0",
            "references_included": False,
        },
        "images": detector_images,
        "categories": [
            {"id": index, "name": name, "supercategory": "crop"}
            for index, name in enumerate(classes, 1)
        ],
    }
    output_root = args.output_root.resolve()
    atomic_write_jsonl(output_root / "writer_t0_t1.jsonl", writer_records)
    atomic_write_jsonl(output_root / "manifest.jsonl", safe_manifests)
    atomic_write_json(output_root / "detector_input.coco.json", detector_payload)
    summary = {
        "schema_version": "cv801_golden_test_inputs_v1",
        "images": len(writer_records),
        "classes": len(classes),
        "source": str(args.image_manifest.resolve()),
        "reference_policy": "no captions, boxes, object counts, labels, or assistant targets exported",
        "writer_input": "writer_t0_t1.jsonl",
        "detector_input": "detector_input.coco.json",
        "t2_status": "not generated until frozen YOLO test predictions exist",
    }
    atomic_write_json(output_root / "summary.json", summary)
    audit_inputs(output_root, args.expected_images, require_t2=False)
    return summary


def build_t2(args: argparse.Namespace) -> dict[str, Any]:
    input_root = args.input_root.resolve()
    base_records = read_jsonl(input_root / "writer_t0_t1.jsonl")
    manifests = read_jsonl(input_root / "manifest.jsonl")
    detector_payload = json.loads(
        (input_root / "detector_input.coco.json").read_text(encoding="utf-8")
    )
    _assert_count(base_records, args.expected_images, "base Writer inputs")
    _assert_count(manifests, args.expected_images, "safe manifest rows")
    image_by_id = {int(record["id"]): record for record in detector_payload["images"]}
    category_by_id = {
        int(record["id"]): str(record["name"])
        for record in detector_payload["categories"]
    }
    predictions_by_image: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for prediction in json.loads(args.predictions.read_text(encoding="utf-8")):
        image_id = int(prediction["image_id"])
        category_id = int(prediction["category_id"])
        score = float(prediction["score"])
        box = prediction["bbox"]
        if image_id not in image_by_id or category_id not in category_by_id:
            raise ValueError(f"Prediction uses unknown image/category: {image_id}/{category_id}")
        if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
            raise ValueError(f"Invalid prediction box for image {image_id}")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid prediction score for image {image_id}")
        predictions_by_image[image_id].append(prediction)

    t2_records: list[dict[str, Any]] = []
    t2_manifests: list[dict[str, Any]] = []
    proposal_counts: list[int] = []
    for base, manifest in zip(base_records, manifests, strict=True):
        coco_image_id = int(manifest["coco_image_id"])
        candidates = [
            prediction
            for prediction in predictions_by_image.get(coco_image_id, [])
            if float(prediction["score"]) >= args.score_threshold
        ]
        candidates.sort(key=lambda prediction: float(prediction["score"]), reverse=True)
        candidates = candidates[: args.max_proposals]
        candidates.sort(
            key=lambda prediction: (
                float(prediction["bbox"][1]) + float(prediction["bbox"][3]) / 2,
                float(prediction["bbox"][0]) + float(prediction["bbox"][2]) / 2,
                -float(prediction["score"]),
            )
        )
        proposals = [
            {
                "proposal_id": f"box_{index:03d}",
                "label": category_by_id[int(prediction["category_id"])],
                "box": normalized_box(
                    [float(value) for value in prediction["bbox"]],
                    int(manifest["width"]),
                    int(manifest["height"]),
                ),
                "score": round(float(prediction["score"]), 4),
            }
            for index, prediction in enumerate(candidates, 1)
        ]
        record = copy.deepcopy(base)
        record["messages"][-1]["content"] += proposal_prompt(proposals)
        t2_records.append(record)
        t2_manifest = copy.deepcopy(manifest)
        t2_manifest.update({
            "boxes_exposed_to_t2": True,
            "locator": "YOLO26x fine-tuned train-357714 best.pt",
            "locator_score_threshold": args.score_threshold,
            "locator_max_proposals": args.max_proposals,
            "predicted_proposals": proposals,
        })
        t2_manifests.append(t2_manifest)
        proposal_counts.append(len(proposals))

    atomic_write_jsonl(input_root / "writer_t2.jsonl", t2_records)
    atomic_write_jsonl(input_root / "manifest_t2.jsonl", t2_manifests)
    summary = json.loads((input_root / "summary.json").read_text(encoding="utf-8"))
    summary["t2_status"] = "generated from frozen YOLO predictions without references"
    summary["t2"] = {
        "writer_input": "writer_t2.jsonl",
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions.resolve()),
        "locator_model": str(args.locator_model.resolve()),
        "locator_model_sha256": sha256_file(args.locator_model.resolve()),
        "score_threshold": args.score_threshold,
        "max_proposals": args.max_proposals,
        "empty_proposal_images": sum(count == 0 for count in proposal_counts),
        "mean_proposals": sum(proposal_counts) / len(proposal_counts),
    }
    atomic_write_json(input_root / "summary.json", summary)
    audit_inputs(input_root, args.expected_images, require_t2=True)
    return summary["t2"]


def _assert_reference_free_writer(record: dict[str, Any], expected: dict[str, Any]) -> None:
    if set(record) != {"messages", "images"}:
        raise ValueError(f"Unexpected Writer input keys: {sorted(record)}")
    if record["images"] != [expected["image_path"]]:
        raise ValueError(f"Writer image mismatch for {expected['image_id']}")
    messages = record["messages"]
    if [message.get("role") for message in messages] != ["system", "user"]:
        raise ValueError(f"Writer input is not reference-free for {expected['image_id']}")
    if any(set(message) != {"role", "content"} for message in messages):
        raise ValueError(f"Unexpected message fields for {expected['image_id']}")
    forbidden = FORBIDDEN_MODEL_KEYS.intersection(record)
    if forbidden:
        raise ValueError(f"Forbidden Writer fields for {expected['image_id']}: {sorted(forbidden)}")


def audit_inputs(input_root: Path, expected_images: int, require_t2: bool) -> dict[str, Any]:
    manifests = read_jsonl(input_root / "manifest.jsonl")
    writers = read_jsonl(input_root / "writer_t0_t1.jsonl")
    detector = json.loads((input_root / "detector_input.coco.json").read_text(encoding="utf-8"))
    _assert_count(manifests, expected_images, "safe manifest rows")
    _assert_count(writers, expected_images, "T0/T1 Writer inputs")
    _assert_count(detector.get("images", []), expected_images, "detector input images")
    if "annotations" in detector:
        raise ValueError("Detector input contains annotations")
    if set(detector) != {"info", "images", "categories"}:
        raise ValueError(f"Unexpected detector input keys: {sorted(detector)}")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for position, (manifest, writer, detector_image) in enumerate(
        zip(manifests, writers, detector["images"], strict=True), 1
    ):
        if set(manifest) != SAFE_MANIFEST_KEYS:
            raise ValueError(f"Unsafe manifest keys at row {position}: {sorted(manifest)}")
        if manifest["image_id"] in seen_ids or manifest["image_path"] in seen_paths:
            raise ValueError(f"Duplicate golden row at position {position}")
        seen_ids.add(manifest["image_id"])
        seen_paths.add(manifest["image_path"])
        if int(manifest["coco_image_id"]) != position or int(detector_image["id"]) != position:
            raise ValueError(f"COCO ID/order mismatch at position {position}")
        if manifest["image_path"] != f"{DETECTION_PREFIX}{detector_image['file_name']}":
            raise ValueError(f"Image path mismatch at position {position}")
        _assert_reference_free_writer(writer, manifest)
        if writer["messages"] != [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_prompt(manifest["crop"], manifest["plant_part"]),
            },
        ]:
            raise ValueError(f"Unexpected T0/T1 prompt for {manifest['image_id']}")

    result: dict[str, Any] = {
        "status": "PASS",
        "images": len(manifests),
        "t0_t1_reference_free": True,
        "detector_annotations_present": False,
        "t2_checked": False,
    }
    if require_t2:
        t2_records = read_jsonl(input_root / "writer_t2.jsonl")
        t2_manifests = read_jsonl(input_root / "manifest_t2.jsonl")
        _assert_count(t2_records, expected_images, "T2 Writer inputs")
        _assert_count(t2_manifests, expected_images, "T2 safe manifest rows")
        for base, t2, base_manifest, t2_manifest in zip(
            writers, t2_records, manifests, t2_manifests, strict=True
        ):
            _assert_reference_free_writer(t2, base_manifest)
            if set(t2_manifest) != SAFE_MANIFEST_KEYS | {
                "boxes_exposed_to_t2",
                "locator",
                "locator_score_threshold",
                "locator_max_proposals",
                "predicted_proposals",
            }:
                raise ValueError(f"Unsafe T2 manifest keys for {base_manifest['image_id']}")
            if any(t2_manifest[key] != base_manifest[key] for key in SAFE_MANIFEST_KEYS):
                raise ValueError(f"T2 manifest identity mismatch for {base_manifest['image_id']}")
            expected_messages = copy.deepcopy(base["messages"])
            expected_messages[-1]["content"] += proposal_prompt(t2_manifest["predicted_proposals"])
            if t2["messages"] != expected_messages:
                raise ValueError(f"Unexpected T2 prompt for {base_manifest['image_id']}")
        result["t2_checked"] = True
        result["t2_reference_free"] = True
    return result


def _prediction_image_path(record: dict[str, Any]) -> str:
    images = record.get("images") or []
    if len(images) != 1:
        raise ValueError("Writer prediction must contain exactly one image")
    image = images[0]
    path = image.get("path") if isinstance(image, dict) else image
    if not path:
        raise ValueError("Writer prediction image has no path")
    return str(path)


def audit_writer_predictions(
    predictions_path: Path,
    input_root: Path,
    variant: str,
    expected_images: int,
) -> dict[str, Any]:
    input_name = "writer_t2.jsonl" if variant == "t2" else "writer_t0_t1.jsonl"
    source_records = read_jsonl(input_root / input_name)
    predictions = read_jsonl(predictions_path)
    _assert_count(source_records, expected_images, f"{variant} source inputs")
    _assert_count(predictions, expected_images, f"{variant} Writer predictions")
    for position, (source, prediction) in enumerate(zip(source_records, predictions, strict=True), 1):
        if _prediction_image_path(prediction) != source["images"][0]:
            raise ValueError(f"Writer prediction image mismatch at row {position}")
        if prediction.get("labels") not in {None, ""}:
            raise ValueError(f"Writer prediction contains a reference label at row {position}")
        response = prediction.get("response")
        if not isinstance(response, str):
            raise ValueError(f"Writer prediction has no text response at row {position}")
        prediction_messages = prediction.get("messages")
        source_messages = source["messages"]
        # ms-swift preserves the source conversation and appends the generated
        # assistant turn. Older result formats kept only the source turns. Both
        # are safe if the immutable source prefix is exact and the appended turn
        # is byte-identical to the separately recorded response.
        source_only = prediction_messages == source_messages
        source_plus_response = (
            isinstance(prediction_messages, list)
            and len(prediction_messages) == len(source_messages) + 1
            and prediction_messages[:-1] == source_messages
            and prediction_messages[-1] == {"role": "assistant", "content": response}
        )
        if not (source_only or source_plus_response):
            raise ValueError(f"Writer prediction source messages changed at row {position}")
    return {
        "status": "PASS",
        "variant": variant,
        "images": len(predictions),
        "aligned": True,
        "nonempty_reference_labels": 0,
    }


def main() -> None:
    args = parse_args()
    if args.command == "base":
        result = build_base(args)
    elif args.command == "t2":
        result = build_t2(args)
    elif args.command == "audit":
        result = audit_inputs(args.input_root.resolve(), args.expected_images, args.require_t2)
    else:
        result = audit_writer_predictions(
            args.predictions.resolve(),
            args.input_root.resolve(),
            args.variant,
            args.expected_images,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
