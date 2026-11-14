#!/usr/bin/env python3
"""Distributed Grounding DINO inference that exports standard COCO detections."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from torchvision.ops import batched_nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box-threshold", type=float, default=0.01)
    parser.add_argument("--text-threshold", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def normalize_label(value: object) -> str:
    return str(value).strip().lower().replace("_", " ").rstrip(".")


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return rank, world_size, local_rank


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = distributed_context()
    device = torch.device(f"cuda:{local_rank}")
    payload = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    images = payload["images"][: args.max_images]
    categories = payload["categories"]
    category_queries = [
        (int(category["id"]), category["name"].replace("_", " "))
        for category in categories
    ]

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    local_predictions: list[dict] = []
    for position in range(rank, len(images), world_size):
        item = images[position]
        with Image.open(args.dataset_root / item["file_name"]) as opened:
            image = opened.convert("RGB")
        mapped_boxes = []
        mapped_scores = []
        mapped_categories = []
        # Grounding DINO predicts token spans rather than closed-set class IDs.
        # Querying each class independently gives every returned box an
        # unambiguous COCO category while preserving the model's native score.
        for category_id, prompt in category_queries:
            inputs = processor(images=image, text=[[prompt]], return_tensors="pt")
            device_inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**device_inputs)
            result = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=device_inputs.get("input_ids"),
                threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[(image.height, image.width)],
            )[0]
            boxes = result["boxes"].detach().float().cpu()
            scores = result["scores"].detach().float().cpu()
            mapped_boxes.extend(boxes)
            mapped_scores.extend(scores)
            mapped_categories.extend([category_id] * len(boxes))
        if mapped_boxes:
            box_tensor = torch.stack(mapped_boxes)
            score_tensor = torch.stack(mapped_scores)
            category_tensor = torch.tensor(mapped_categories, dtype=torch.long)
            keep = batched_nms(box_tensor, score_tensor, category_tensor, args.nms_iou)
            keep = keep[: args.max_detections].tolist()
            for index in keep:
                x1, y1, x2, y2 = box_tensor[index].tolist()
                x1 = max(0.0, min(x1, image.width))
                y1 = max(0.0, min(y1, image.height))
                x2 = max(x1, min(x2, image.width))
                y2 = max(y1, min(y2, image.height))
                if x2 <= x1 or y2 <= y1:
                    continue
                local_predictions.append({
                    "image_id": item["id"],
                    "category_id": mapped_categories[index],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score_tensor[index]),
                })

    rank_path = args.output.with_name(f"{args.output.stem}.rank{rank}.json")
    rank_path.parent.mkdir(parents=True, exist_ok=True)
    rank_path.write_text(json.dumps(local_predictions) + "\n", encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        merged = []
        for worker_rank in range(world_size):
            worker_path = args.output.with_name(f"{args.output.stem}.rank{worker_rank}.json")
            merged.extend(json.loads(worker_path.read_text(encoding="utf-8")))
            worker_path.unlink()
        args.output.write_text(json.dumps(merged) + "\n", encoding="utf-8")
        print(json.dumps({
            "images": len(images), "predictions": len(merged),
            "queries_per_image": len(category_queries), "output": str(args.output),
        }))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
