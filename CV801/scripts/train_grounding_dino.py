#!/usr/bin/env python3
"""Fine-tune Grounding DINO Tiny on the shared CV801 COCO split."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)


CLASSES = [
    "bitter gourd",
    "cauliflower",
    "cherry tomato",
    "chili",
    "date palm",
    "eggplant",
    "fig",
    "lemon",
    "luffa",
    "mulberry",
    "tomato",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--epochs", type=float, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    return parser.parse_args()


class CocoDetectionDataset(Dataset):
    def __init__(self, root: Path, split: str, limit: int | None = None):
        payload = json.loads((root / "coco" / f"instances_{split}.json").read_text())
        self.root = root
        self.images = payload["images"][:limit]
        annotations: dict[int, list[dict]] = {item["id"]: [] for item in self.images}
        valid_ids = set(annotations)
        for annotation in payload["annotations"]:
            if annotation["image_id"] in valid_ids:
                converted = dict(annotation)
                converted["category_id"] -= 1
                annotations[annotation["image_id"]].append(converted)
        self.annotations = annotations

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        item = self.images[index]
        with Image.open(self.root / item["file_name"]) as opened:
            image = opened.convert("RGB")
        target = {
            "image_id": item["id"],
            "annotations": self.annotations[item["id"]],
        }
        return {"image": image, "target": target}


@dataclass
class GroundingDinoCollator:
    processor: object

    def __call__(self, samples: list[dict]):
        return self.processor(
            images=[sample["image"] for sample in samples],
            text=[CLASSES for _ in samples],
            annotations=[sample["target"] for sample in samples],
            return_tensors="pt",
        )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model,
        id2label={index: label for index, label in enumerate(CLASSES)},
        label2id={label: index for index, label in enumerate(CLASSES)},
    )
    train_dataset = CocoDetectionDataset(dataset_root, "train", args.max_train_samples)
    val_dataset = CocoDetectionDataset(dataset_root, "val", args.max_val_samples)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_do_concat_batches=False,
        logging_steps=10,
        dataloader_num_workers=args.workers,
        remove_unused_columns=False,
        report_to="tensorboard",
        # Grounding DINO keeps auxiliary prediction heads whose parameters are
        # not used by every loss path. DDP must explicitly tolerate them.
        ddp_find_unused_parameters=True,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=GroundingDinoCollator(processor),
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(output_dir / "final")
    processor.save_pretrained(output_dir / "final")


if __name__ == "__main__":
    main()
