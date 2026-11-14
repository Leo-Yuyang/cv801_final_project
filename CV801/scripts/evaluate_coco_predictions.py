#!/usr/bin/env python3
"""Evaluate COCO detections with COCO AP and a frozen operating point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


STAT_NAMES = (
    "AP_50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_max1", "AR_max10", "AR_max100", "AR_small", "AR_medium", "AR_large",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--score-policy", default="native_model_score")
    parser.add_argument(
        "--operating-score-threshold",
        type=float,
        help=(
            "Frozen score threshold used for class-aware micro precision, recall, "
            "F1, and matched-box IoU at IoU >= 0.5. The threshold must be selected "
            "without using the evaluated split."
        ),
    )
    parser.add_argument("--only-predicted-images", action="store_true")
    return parser.parse_args()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list):
        raise ValueError("COCO predictions must be a JSON array or JSONL records")
    return records


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) and value >= 0 else None


def bbox_iou_xywh(left: list[float], right: list[float]) -> float:
    """Return IoU for two COCO-format ``xywh`` boxes."""
    left_x1, left_y1, left_w, left_h = (float(value) for value in left)
    right_x1, right_y1, right_w, right_h = (float(value) for value in right)
    left_x2, left_y2 = left_x1 + max(0.0, left_w), left_y1 + max(0.0, left_h)
    right_x2, right_y2 = right_x1 + max(0.0, right_w), right_y1 + max(0.0, right_h)
    intersection_w = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_h = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_w * intersection_h
    union = max(0.0, left_w) * max(0.0, left_h) + max(0.0, right_w) * max(0.0, right_h) - intersection
    return intersection / union if union > 0 else 0.0


def greedy_match_outcomes(
    annotations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> tuple[list[dict[str, float | bool]], int]:
    """Greedily match class-aware predictions in descending score order."""
    ground_truth: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for annotation in annotations:
        if int(annotation.get("iscrowd", 0)):
            continue
        key = (int(annotation["image_id"]), int(annotation["category_id"]))
        ground_truth.setdefault(key, []).append(annotation)

    matched: dict[tuple[int, int], set[int]] = {}
    indexed_predictions = list(enumerate(predictions))
    indexed_predictions.sort(key=lambda item: -float(item[1].get("score", 0.0)))
    outcomes: list[dict[str, float | bool]] = []
    for _, prediction in indexed_predictions:
        key = (int(prediction["image_id"]), int(prediction["category_id"]))
        candidates = ground_truth.get(key, [])
        already_matched = matched.setdefault(key, set())
        best_index: int | None = None
        best_iou = 0.0
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in already_matched:
                continue
            iou = bbox_iou_xywh(prediction["bbox"], candidate["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_index = candidate_index
        is_true_positive = best_index is not None and best_iou >= iou_threshold
        if is_true_positive:
            already_matched.add(best_index)
        outcomes.append({
            "score": float(prediction.get("score", 0.0)),
            "true_positive": is_true_positive,
            "iou": best_iou if is_true_positive else 0.0,
        })
    return outcomes, sum(len(records) for records in ground_truth.values())


def summarize_operating_point(
    outcomes: list[dict[str, float | bool]],
    ground_truth_count: int,
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Summarize one frozen score threshold using micro-averaged counts."""
    included = [record for record in outcomes if float(record["score"]) >= score_threshold]
    true_positives = sum(bool(record["true_positive"]) for record in included)
    false_positives = len(included) - true_positives
    false_negatives = ground_truth_count - true_positives
    precision = true_positives / len(included) if included else 0.0
    recall = true_positives / ground_truth_count if ground_truth_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched_ious = [float(record["iou"]) for record in included if bool(record["true_positive"])]
    return {
        "score_threshold": float(score_threshold),
        "iou_threshold": float(iou_threshold),
        "averaging": "micro",
        "matching": "class-aware greedy one-to-one in descending score order",
        "ground_truth_count": ground_truth_count,
        "included_predictions": len(included),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else None,
    }


def best_f1_threshold(
    annotations: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Select the highest-scoring threshold attaining maximum micro F1."""
    outcomes, ground_truth_count = greedy_match_outcomes(annotations, predictions, iou_threshold)
    if not outcomes:
        return summarize_operating_point(outcomes, ground_truth_count, float("inf"), iou_threshold)
    thresholds = sorted({float(record["score"]) for record in outcomes}, reverse=True)
    summaries = [
        summarize_operating_point(outcomes, ground_truth_count, threshold, iou_threshold)
        for threshold in thresholds
    ]
    return max(summaries, key=lambda record: (float(record["f1"]), float(record["score_threshold"])))


def per_category_metrics(evaluator: COCOeval, coco: COCO) -> dict[str, dict[str, float | None]]:
    precision = evaluator.eval["precision"]  # IoU, recall, category, area, maxDets
    iou_thresholds = evaluator.params.iouThrs
    iou50_index = int(np.argmin(np.abs(iou_thresholds - 0.5)))
    results: dict[str, dict[str, float | None]] = {}
    for category_index, category_id in enumerate(evaluator.params.catIds):
        category = coco.loadCats([category_id])[0]
        all_iou = precision[:, :, category_index, 0, -1]
        at_50 = precision[iou50_index, :, category_index, 0, -1]
        all_iou = all_iou[all_iou > -1]
        at_50 = at_50[at_50 > -1]
        results[category["name"]] = {
            "AP_50_95": float(np.mean(all_iou)) if all_iou.size else None,
            "AP50": float(np.mean(at_50)) if at_50.size else None,
        }
    return results


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.predictions)
    coco = COCO(str(args.ground_truth))
    if predictions:
        coco_results = coco.loadRes(predictions)
    else:
        # COCO.loadRes indexes the first result and therefore cannot accept an
        # empty list. An empty detector is still a valid result and must score
        # zero recall rather than disappearing from the evaluation denominator.
        coco_results = COCO()
        coco_results.dataset = {
            "images": list(coco.dataset["images"]),
            "categories": list(coco.dataset["categories"]),
            "annotations": [],
        }
        coco_results.createIndex()
    evaluator = COCOeval(coco, coco_results, "bbox")
    if args.only_predicted_images:
        if not predictions:
            raise ValueError("--only-predicted-images requires at least one prediction")
        evaluator.params.imgIds = sorted({int(record["image_id"]) for record in predictions})
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {
        name: finite_or_none(value) for name, value in zip(STAT_NAMES, evaluator.stats)
    }
    operating_point = None
    if args.operating_score_threshold is not None:
        outcomes, ground_truth_count = greedy_match_outcomes(
            list(coco.dataset.get("annotations", [])), predictions, iou_threshold=0.5
        )
        operating_point = summarize_operating_point(
            outcomes,
            ground_truth_count,
            score_threshold=args.operating_score_threshold,
            iou_threshold=0.5,
        )
    payload = {
        "metric_contract": "pycocotools_cocoeval_bbox_v2_operating_point",
        "model": args.model_name,
        "split": "validation",
        "ground_truth": str(args.ground_truth),
        "predictions": str(args.predictions),
        "score_policy": args.score_policy,
        "evaluated_images": len(evaluator.params.imgIds),
        "prediction_count": len(predictions),
        "metrics": metrics,
        "operating_point": operating_point,
        "per_category": per_category_metrics(evaluator, coco),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
