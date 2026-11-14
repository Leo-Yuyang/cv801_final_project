#!/usr/bin/env python3
"""Distributed YOLO inference that exports standard COCO detections."""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def normalize_label(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    return rank, world_size, local_rank


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = distributed_context()
    payload = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    images = payload["images"][: args.max_images]
    local_images = images[rank::world_size]
    category_by_name = {
        normalize_label(category["name"]): int(category["id"])
        for category in payload["categories"]
    }

    model = YOLO(str(args.model))
    model_category = {
        int(class_id): category_by_name[normalize_label(name)]
        for class_id, name in model.names.items()
    }
    local_predictions: list[dict] = []
    processed = 0
    # Ultralytics treats a complete Python list of paths as one in-memory
    # batch.  Explicitly chunk the list so --batch-size is an actual memory
    # bound rather than only a predictor hint.
    for start in range(0, len(local_images), args.batch_size):
        batch_images = local_images[start : start + args.batch_size]
        batch_paths = [str(args.dataset_root / item["file_name"]) for item in batch_images]
        results = model.predict(
            source=batch_paths,
            stream=True,
            batch=len(batch_paths),
            imgsz=args.image_size,
            conf=args.confidence,
            iou=args.nms_iou,
            max_det=args.max_detections,
            device=local_rank,
            verbose=False,
        )
        batch_processed = 0
        for item, result in zip(batch_images, results, strict=True):
            batch_processed += 1
            processed += 1
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.detach().float().cpu()
            scores = result.boxes.conf.detach().float().cpu()
            labels = result.boxes.cls.detach().long().cpu()
            for box, score, label in zip(xyxy, scores, labels, strict=True):
                x1, y1, x2, y2 = box.tolist()
                if x2 <= x1 or y2 <= y1:
                    continue
                local_predictions.append({
                    "image_id": int(item["id"]),
                    "category_id": model_category[int(label)],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                })
        if batch_processed != len(batch_images):
            raise RuntimeError(
                f"Processed {batch_processed} of {len(batch_images)} images in batch {start}"
            )
    if processed != len(local_images):
        raise RuntimeError(f"Processed {processed} of {len(local_images)} assigned images")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rank_path = args.output.with_name(f"{args.output.stem}.rank{rank}.json")
    rank_path.write_text(json.dumps(local_predictions) + "\n", encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        merged: list[dict] = []
        for worker_rank in range(world_size):
            worker_path = args.output.with_name(f"{args.output.stem}.rank{worker_rank}.json")
            merged.extend(json.loads(worker_path.read_text(encoding="utf-8")))
            worker_path.unlink()
        merged.sort(key=lambda record: (record["image_id"], -record["score"]))
        args.output.write_text(json.dumps(merged) + "\n", encoding="utf-8")
        print(json.dumps({
            "images": len(images),
            "predictions": len(merged),
            "confidence_floor": args.confidence,
            "output": str(args.output),
        }))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
