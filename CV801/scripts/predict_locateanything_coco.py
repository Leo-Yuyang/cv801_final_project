#!/usr/bin/env python3
"""Distributed LocateAnything inference that exports standard COCO detections."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoModel, AutoProcessor

from evaluation.inference_compat import (
    apply_chat_template,
    build_generate_kwargs,
    decode_generation_output,
    prepare_generation_inputs,
    process_vision_info,
)


REF_PATTERN = re.compile(r"<ref>([^<]+)</ref>((?:<box>.*?</box>)+)")
BOX_PATTERN = re.compile(
    r"<box>\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
    r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
    r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
    r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*</box>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--generation-mode", choices=("fast", "slow", "hybrid"), default="hybrid")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--short-side-size", type=int, default=1024)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=20261104)
    return parser.parse_args()


def normalize_label(value: object) -> str:
    return str(value).strip().lower().replace("_", " ").rstrip(".")


def parse_boxes(text: str) -> list[tuple[str, list[float]]]:
    detections: list[tuple[str, list[float]]] = []
    for label, boxes_text in REF_PATTERN.findall(text):
        for values in BOX_PATTERN.findall(boxes_text):
            coordinates = [float(value) for value in values]
            if all(0.0 <= value <= 10000.0 for value in coordinates):
                detections.append((normalize_label(label), coordinates))
    return detections


def normalized_to_xywh(coordinates: list[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = coordinates
    x1 = max(0.0, min(x1 * width / 1000.0, float(width)))
    y1 = max(0.0, min(y1 * height / 1000.0, float(height)))
    x2 = max(0.0, min(x2 * width / 1000.0, float(width)))
    y2 = max(0.0, min(y2 * height / 1000.0, float(height)))
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right <= left or bottom <= top:
        return None
    return [left, top, right - left, bottom - top]


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    return rank, world_size, local_rank


def generate(model, processor, image: Image.Image, categories: list[str], args: argparse.Namespace, device):
    category_text = "</c>".join(categories)
    prompt = f"Locate all the instances that matches the following description: {category_text}."
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    text = [apply_chat_template(processor, messages)]
    image_inputs, video_inputs = process_vision_info(processor, messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    prepared = prepare_generation_inputs(inputs, device)
    kwargs = build_generate_kwargs(
        prepared,
        processor,
        generation_mode=args.generation_mode,
        max_new_tokens=args.max_new_tokens,
    )
    with torch.inference_mode():
        output = model.generate(**kwargs)
    return prompt, decode_generation_output(output, prepared["input_ids"], processor)


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = distributed_context()
    device = torch.device(f"cuda:{local_rank}")
    payload = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    images = payload["images"][: args.max_images]
    categories = [category["name"].replace("_", " ") for category in payload["categories"]]
    category_by_name = {
        normalize_label(category["name"]): int(category["id"])
        for category in payload["categories"]
    }

    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True, use_fast=True
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    local_predictions: list[dict] = []
    local_raw: list[dict] = []
    for position in range(rank, len(images), world_size):
        item = images[position]
        with Image.open(args.dataset_root / item["file_name"]) as opened:
            image = opened.convert("RGB")
        original_width, original_height = image.size
        if args.short_side_size > 0 and min(image.size) != args.short_side_size:
            scale = args.short_side_size / min(image.size)
            resized_size = tuple(round(dimension * scale) for dimension in image.size)
            image = image.resize(resized_size, Image.Resampling.BILINEAR)
        torch.manual_seed(args.seed + int(item["id"]))
        torch.cuda.manual_seed_all(args.seed + int(item["id"]))
        prompt, response = generate(model, processor, image, categories, args, device)
        parsed = parse_boxes(response)
        accepted = 0
        unknown_labels: set[str] = set()
        for label, coordinates in parsed:
            if accepted >= args.max_detections:
                break
            category_id = category_by_name.get(label)
            if category_id is None:
                unknown_labels.add(label)
                continue
            # Generated coordinates are normalized, so map them directly to
            # the original image even when inference uses an official-style
            # short-side resize to stay within the fine-tuning token budget.
            bbox = normalized_to_xywh(coordinates, original_width, original_height)
            if bbox is None:
                continue
            local_predictions.append({
                "image_id": int(item["id"]),
                "category_id": category_id,
                "bbox": bbox,
                # LocateAnything emits no calibrated confidence. A constant
                # preserves the model's generation order for COCO evaluation.
                "score": 0.2,
            })
            accepted += 1
        local_raw.append({
            "image_id": int(item["id"]),
            "file_name": item["file_name"],
            "prompt": prompt,
            "response": response,
            "parsed_detections": len(parsed),
            "accepted_detections": accepted,
            "unknown_labels": sorted(unknown_labels),
            "original_size": [original_width, original_height],
            "inference_size": [image.width, image.height],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    prediction_rank_path = args.output.with_name(f"{args.output.stem}.rank{rank}.json")
    raw_rank_path = args.raw_output.with_name(f"{args.raw_output.stem}.rank{rank}.jsonl")
    prediction_rank_path.write_text(json.dumps(local_predictions) + "\n", encoding="utf-8")
    with raw_rank_path.open("w", encoding="utf-8") as handle:
        for record in local_raw:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        predictions: list[dict] = []
        raw_records: list[dict] = []
        for worker_rank in range(world_size):
            worker_predictions = args.output.with_name(
                f"{args.output.stem}.rank{worker_rank}.json"
            )
            predictions.extend(json.loads(worker_predictions.read_text(encoding="utf-8")))
            worker_predictions.unlink()
            worker_raw = args.raw_output.with_name(
                f"{args.raw_output.stem}.rank{worker_rank}.jsonl"
            )
            raw_records.extend(
                json.loads(line)
                for line in worker_raw.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            worker_raw.unlink()
        predictions.sort(key=lambda record: (record["image_id"], -record["score"]))
        raw_records.sort(key=lambda record: record["image_id"])
        args.output.write_text(json.dumps(predictions) + "\n", encoding="utf-8")
        with args.raw_output.open("w", encoding="utf-8") as handle:
            for record in raw_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({
            "images": len(raw_records),
            "predictions": len(predictions),
            "empty_images": sum(record["accepted_detections"] == 0 for record in raw_records),
            "output": str(args.output),
            "raw_output": str(args.raw_output),
        }))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
