#!/usr/bin/env python3
"""Build one leakage-safe detection dataset for all three model families."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
import threading
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSES = [
    "bitter_gourd",
    "cauliflower",
    "cherry_tomato",
    "chili",
    "date_palm",
    "eggplant",
    "fig",
    "lemon",
    "luffa",
    "mulberry",
    "tomato",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT / "OneDrive_1_2026-11-4",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "CV801/exports/annotations_after_verification.jsonl",
    )
    parser.add_argument(
        "--workspace-annotations",
        type=Path,
        default=PROJECT_ROOT / "CV801/agro_workspace/annotations.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "datas/cv801_detection_v1",
    )
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--skip-images", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_records(annotation_path: Path, workspace_path: Path):
    objects: dict[str, list[dict]] = collections.defaultdict(list)
    for record in read_jsonl(annotation_path):
        if record["localization_decision"] not in {"accept", "correct"}:
            raise ValueError(f"Unexpected exported decision: {record['localization_decision']}")
        objects[record["image_id"]].append(record)

    metadata: dict[str, dict] = {}
    decisions: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for record in read_jsonl(workspace_path):
        image_id = record["image_id"]
        decisions[image_id][record.get("localization_decision")] += 1
        candidate = {
            "image_id": image_id,
            "image_type": record["image_type"],
            "crop": record["crop"],
            "width": record["image_width"],
            "height": record["image_height"],
            "file_sha256": record["file_sha256"],
            "perceptual_hash": record.get("perceptual_hash"),
            "duplicate_group": record.get("duplicate_group"),
        }
        previous = metadata.setdefault(image_id, candidate)
        if previous != candidate:
            raise ValueError(f"Inconsistent image metadata: {image_id}")

    missing = sorted(set(objects) - set(metadata))
    if missing:
        raise ValueError(f"Exported images missing workspace metadata: {missing[:5]}")
    return objects, metadata, decisions


def stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def allocate_targets(counts: collections.Counter, fraction: float) -> dict[tuple[str, str], int]:
    total_target = round(sum(counts.values()) * fraction)
    raw = {key: value * fraction for key, value in counts.items()}
    targets = {key: math.floor(value) for key, value in raw.items()}
    remaining = total_target - sum(targets.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - targets[key]), key))
    for key in order[:remaining]:
        targets[key] += 1
    return targets


def make_splits(objects: dict[str, list[dict]], metadata: dict[str, dict], fraction: float, seed: int):
    original_split = {}
    for image_id, records in objects.items():
        values = {record["split"] for record in records}
        if len(values) != 1 or values - {"train", "golden"}:
            raise ValueError(f"Invalid original split for {image_id}: {values}")
        original_split[image_id] = values.pop()

    groups: dict[str, list[str]] = collections.defaultdict(list)
    for image_id in objects:
        duplicate_group = metadata[image_id]["duplicate_group"]
        key = f"duplicate:{duplicate_group}" if duplicate_group else f"image:{image_id}"
        groups[key].append(image_id)

    for key, image_ids in groups.items():
        values = {original_split[image_id] for image_id in image_ids}
        if len(values) != 1:
            raise ValueError(f"Duplicate group crosses the fixed golden split: {key}")

    development_groups = {
        key: image_ids
        for key, image_ids in groups.items()
        if original_split[image_ids[0]] == "train"
    }
    strata = {
        image_id: (metadata[image_id]["image_type"], metadata[image_id]["crop"])
        for image_id in objects
    }
    development_counts = collections.Counter(
        strata[image_id]
        for image_id, split in original_split.items()
        if split == "train"
    )
    targets = allocate_targets(development_counts, fraction)
    target_total = sum(targets.values())

    group_counts = {
        key: collections.Counter(strata[image_id] for image_id in image_ids)
        for key, image_ids in development_groups.items()
    }
    selected: set[str] = set()
    current = collections.Counter()
    current_total = 0
    remaining_groups = set(development_groups)
    while current_total < target_total:
        capacity = target_total - current_total
        candidates = [key for key in remaining_groups if len(development_groups[key]) <= capacity]
        if not candidates:
            raise RuntimeError("Duplicate grouping prevents the requested validation size")

        def delta(key: str):
            change = 0.0
            for stratum, added in group_counts[key].items():
                target = max(targets[stratum], 1)
                before = current[stratum] - targets[stratum]
                after = current[stratum] + added - targets[stratum]
                change += (after * after - before * before) / target
            return change, stable_rank(seed, key)

        best = min(candidates, key=delta)
        selected.add(best)
        remaining_groups.remove(best)
        current.update(group_counts[best])
        current_total += len(development_groups[best])

    split_by_image = {}
    group_by_image = {}
    for key, image_ids in groups.items():
        for image_id in image_ids:
            group_by_image[image_id] = key
            if original_split[image_id] == "golden":
                split_by_image[image_id] = "test"
            else:
                split_by_image[image_id] = "val" if key in selected else "train"
    return split_by_image, group_by_image, targets


def derived_paths(output_root: Path, split: str, image_id: str) -> tuple[Path, Path]:
    source_relative = Path(image_id)
    image = output_root / "images" / split / source_relative.parent / f"{source_relative.name}.jpg"
    label = output_root / "labels" / split / source_relative.parent / f"{source_relative.name}.txt"
    return image, label


def convert_image(job: tuple[Path, Path, int, int, int]) -> str:
    source, target, expected_width, expected_height, quality = job
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            with Image.open(target) as existing:
                if existing.size == (expected_width, expected_height) and existing.mode == "RGB":
                    existing.verify()
                    return "reused"
        except OSError:
            pass
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if image.size != (expected_width, expected_height):
        raise ValueError(
            f"Oriented dimensions differ for {source}: {image.size} != "
            f"{(expected_width, expected_height)}"
        )
    temporary = target.with_name(
        f".{target.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    image.save(temporary, format="JPEG", quality=quality, subsampling=0, optimize=False)
    os.replace(temporary, target)
    return "converted"


def write_outputs(
    output_root: Path,
    source_root: Path,
    objects: dict[str, list[dict]],
    metadata: dict[str, dict],
    decisions: dict[str, collections.Counter],
    split_by_image: dict[str, str],
    group_by_image: dict[str, str],
    targets: dict[tuple[str, str], int],
    seed: int,
    val_fraction: float,
) -> tuple[dict, list[tuple[Path, Path, int, int]]]:
    class_ids = {name: index for index, name in enumerate(CLASSES)}
    unknown_classes = sorted({item["crop"] for item in metadata.values()} - set(CLASSES))
    if unknown_classes:
        raise ValueError(f"Unknown crop classes: {unknown_classes}")

    manifests = []
    coco = {
        split: {
            "info": {"description": "CV801 detection v1", "version": "1.0"},
            "images": [],
            "annotations": [],
            "categories": [
                {"id": index + 1, "name": name, "supercategory": "crop"}
                for index, name in enumerate(CLASSES)
            ],
        }
        for split in ("train", "val", "test")
    }
    locateanything = {split: [] for split in ("train", "val", "test")}
    detection_prompt = (
        "Locate all the instances that match the following description: "
        + "</c>".join(name.replace("_", " ") for name in CLASSES)
        + "."
    )
    annotation_ids = collections.Counter()
    split_image_ids = collections.Counter()
    split_object_counts = collections.Counter()
    jobs = []

    for image_id in sorted(objects):
        meta = metadata[image_id]
        split = split_by_image[image_id]
        source = source_root / image_id
        if not source.is_file():
            raise FileNotFoundError(source)
        image_path, label_path = derived_paths(output_root, split, image_id)
        image_relative = image_path.relative_to(output_root).as_posix()
        label_relative = label_path.relative_to(output_root).as_posix()
        width, height = meta["width"], meta["height"]
        split_image_ids[split] += 1
        coco_image_id = split_image_ids[split]
        label_lines = []
        locateanything_labels = []

        for record in objects[image_id]:
            x1, y1, x2, y2 = map(float, record["bbox"])
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid bounding box in {record['annotation_id']}: {record['bbox']}")
            crop = record["crop"]
            class_id = class_ids[crop]
            box_width, box_height = x2 - x1, y2 - y1
            center_x, center_y = x1 + box_width / 2, y1 + box_height / 2
            label_lines.append(
                f"{class_id} {center_x / width:.8f} {center_y / height:.8f} "
                f"{box_width / width:.8f} {box_height / height:.8f}"
            )
            normalized = [
                min(1000, max(0, round(value)))
                for value in (
                    x1 / width * 1000,
                    y1 / height * 1000,
                    x2 / width * 1000,
                    y2 / height * 1000,
                )
            ]
            locateanything_labels.append(
                f"<ref>{crop.replace('_', ' ')}</ref><box>"
                + "".join(f"<{value}>" for value in normalized)
                + "</box>"
            )
            annotation_ids[split] += 1
            split_object_counts[split] += 1
            coco[split]["annotations"].append(
                {
                    "id": annotation_ids[split],
                    "image_id": coco_image_id,
                    "category_id": class_id + 1,
                    "bbox": [x1, y1, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
        atomic_write(label_path, "\n".join(label_lines) + "\n")
        coco[split]["images"].append(
            {
                "id": coco_image_id,
                "file_name": image_relative,
                "width": width,
                "height": height,
            }
        )
        locateanything[split].append(
            {
                "conversations": [
                    {"from": "human", "value": detection_prompt},
                    {"from": "gpt", "value": "".join(locateanything_labels)},
                ],
                "image": image_relative,
            }
        )
        manifests.append(
            {
                "image_id": image_id,
                "source_path": source.relative_to(PROJECT_ROOT).as_posix(),
                "image_path": image_relative,
                "label_path": label_relative,
                "split": split,
                "original_split": objects[image_id][0]["split"],
                "image_type": meta["image_type"],
                "crop": meta["crop"],
                "width": width,
                "height": height,
                "object_count": len(objects[image_id]),
                "duplicate_group": meta["duplicate_group"],
                "split_group": group_by_image[image_id],
                "file_sha256": meta["file_sha256"],
            }
        )
        jobs.append((source, image_path, width, height))

    for split, payload in coco.items():
        atomic_write(
            output_root / "coco" / f"instances_{split}.json",
            json.dumps(payload, indent=2) + "\n",
        )
    for split, samples in locateanything.items():
        atomic_write(
            output_root / "locateanything" / f"{split}.jsonl",
            "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples),
        )
    smoke_samples = locateanything["train"][:32]
    atomic_write(
        output_root / "locateanything/train_smoke.jsonl",
        "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in smoke_samples),
    )
    recipe_root = output_root.resolve()
    for recipe_name, annotation_name in (
        ("recipe_train.json", "train.jsonl"),
        ("recipe_smoke.json", "train_smoke.jsonl"),
    ):
        recipe = {
            "cv801_detection": {
                "annotation": str(recipe_root / "locateanything" / annotation_name),
                "root": str(recipe_root),
                "repeat_time": 1.0,
                "data_augment": True,
            }
        }
        atomic_write(
            output_root / "locateanything" / recipe_name,
            json.dumps(recipe, indent=2) + "\n",
        )
    atomic_write(
        output_root / "manifests/images.jsonl",
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in manifests),
    )
    excluded = []
    for image_id in sorted(set(metadata) - set(objects)):
        meta = metadata[image_id]
        excluded.append(
            {
                "image_id": image_id,
                "reason": "no_accepted_or_corrected_localization",
                "decisions": dict(decisions[image_id]),
                "image_type": meta["image_type"],
                "crop": meta["crop"],
            }
        )
    atomic_write(
        output_root / "manifests/excluded_images.jsonl",
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in excluded),
    )
    atomic_write(
        output_root / "manifests/classes.json",
        json.dumps({"names": CLASSES}, indent=2) + "\n",
    )
    yaml_names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    atomic_write(
        output_root / "dataset.yaml",
        f"path: {output_root.resolve()}\ntrain: images/train\nval: images/val\n"
        f"test: images/test\nnames:\n{yaml_names}\n",
    )

    stratum_counts = collections.Counter(
        (item["split"], item["image_type"], item["crop"]) for item in manifests
    )
    summary = {
        "version": "cv801_detection_v1",
        "seed": seed,
        "val_fraction_of_development_pool": val_fraction,
        "images": dict(sorted(split_image_ids.items())),
        "objects": dict(sorted(split_object_counts.items())),
        "excluded_images": len(excluded),
        "classes": CLASSES,
        "validation_targets": {
            f"{image_type}/{crop}": count
            for (image_type, crop), count in sorted(targets.items())
        },
        "strata": {
            f"{split}/{image_type}/{crop}": count
            for (split, image_type, crop), count in sorted(stratum_counts.items())
        },
    }
    atomic_write(
        output_root / "manifests/split_summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    return summary, jobs


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between zero and one")
    register_heif_opener()
    objects, metadata, decisions = load_records(args.annotations, args.workspace_annotations)
    split_by_image, group_by_image, targets = make_splits(
        objects, metadata, args.val_fraction, args.seed
    )
    summary, jobs = write_outputs(
        args.output_root,
        args.source_root,
        objects,
        metadata,
        decisions,
        split_by_image,
        group_by_image,
        targets,
        args.seed,
        args.val_fraction,
    )
    if not args.skip_images:
        conversion_jobs = [(*job, args.jpeg_quality) for job in jobs]
        progress = collections.Counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, result in enumerate(executor.map(convert_image, conversion_jobs), start=1):
                progress[result] += 1
                if index % 250 == 0 or index == len(conversion_jobs):
                    print(f"images={index}/{len(conversion_jobs)} {dict(progress)}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
