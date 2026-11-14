#!/usr/bin/env python3
"""Evaluate CV801 T0/T1/T2 structured VLM predictions with one metric contract."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


THINK_PREFIX = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
WORD = re.compile(r"[a-z0-9]+")
OBJECT_FIELDS = {"object_id", "caption", "assessment", "image_quality", "confidence", "action"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid result row at {path}:{line_number}") from error


def clean_response(text: str) -> tuple[str, bool]:
    stripped, count = THINK_PREFIX.subn("", text, count=1)
    match = CODE_FENCE.match(stripped)
    if match:
        stripped = match.group(1)
    return stripped.strip(), bool(count)


def is_object_schema(value: Any, plant_part: str) -> bool:
    if not isinstance(value, dict) or not OBJECT_FIELDS.issubset(value):
        return False
    if not all(isinstance(value[key], str) for key in ("object_id", "caption", "image_quality", "action")):
        return False
    if not isinstance(value["confidence"], (int, float)) or isinstance(value["confidence"], bool):
        return False
    assessment = value["assessment"]
    required = {"disease", "severity"} | ({"stage"} if plant_part == "fruit" else set())
    return isinstance(assessment, dict) and required.issubset(assessment) and all(
        isinstance(assessment[key], str) for key in required
    )


def is_sample_schema(value: Any) -> bool:
    if not isinstance(value, dict) or not {"crop", "plant_part", "objects"}.issubset(value):
        return False
    if not isinstance(value["crop"], str) or value["plant_part"] not in {"fruit", "leaf"}:
        return False
    objects = value["objects"]
    return isinstance(objects, list) and bool(objects) and all(
        is_object_schema(item, value["plant_part"]) for item in objects
    )


def token_f1(prediction: str, reference: str) -> float:
    pred = Counter(WORD.findall(prediction.lower()))
    ref = Counter(WORD.findall(reference.lower()))
    common = sum((pred & ref).values())
    if not pred and not ref:
        return 1.0
    if not pred or not ref or not common:
        return 0.0
    precision = common / sum(pred.values())
    recall = common / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def ngrams(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index:index + order]) for index in range(len(tokens) - order + 1))


def corpus_bleu4(pairs: list[tuple[str, str]]) -> float | None:
    """Compute unsmoothed corpus BLEU-4 over single-reference text pairs."""
    if not pairs:
        return None
    clipped = [0, 0, 0, 0]
    totals = [0, 0, 0, 0]
    prediction_length = 0
    reference_length = 0
    for prediction, reference in pairs:
        pred_tokens = WORD.findall(prediction.lower())
        ref_tokens = WORD.findall(reference.lower())
        prediction_length += len(pred_tokens)
        reference_length += len(ref_tokens)
        for index, order in enumerate(range(1, 5)):
            pred_ngrams = ngrams(pred_tokens, order)
            ref_ngrams = ngrams(ref_tokens, order)
            clipped[index] += sum((pred_ngrams & ref_ngrams).values())
            totals[index] += sum(pred_ngrams.values())
    if prediction_length == 0:
        return 0.0
    precisions = [ratio(matches, total) for matches, total in zip(clipped, totals, strict=True)]
    if any(precision in {None, 0.0} for precision in precisions):
        return 0.0
    brevity_penalty = (
        1.0
        if prediction_length > reference_length
        else math.exp(1.0 - reference_length / prediction_length)
    )
    return brevity_penalty * math.exp(sum(math.log(float(value)) for value in precisions) / 4)


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = WORD.findall(prediction.lower())
    ref_tokens = WORD.findall(reference.lower())
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    previous = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        current = [0]
        for index, ref_token in enumerate(ref_tokens, 1):
            if pred_token == ref_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    common = previous[-1]
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall) if common else 0.0


def ratio(numerator: int | float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def harmonic_mean(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    count_errors: list[float] = []
    caption_f1: list[float] = []
    caption_id_pairs: list[tuple[str, str]] = []
    caption_rouge_l: list[float] = []
    action_f1: list[float] = []
    confidence_errors: list[float] = []
    categorical: Counter[str] = Counter()

    for row in records:
        counts["samples"] += 1
        reference = json.loads(row["labels"])
        ref_objects = reference.get("objects", [])
        if not isinstance(ref_objects, list):
            ref_objects = []
        counts["reference_objects"] += len(ref_objects)
        response, stripped_think = clean_response(row["response"])
        counts["think_prefix_stripped"] += stripped_think
        counts["forbidden_box_output"] += any(
            key in response.lower() for key in ('"bbox"', '"box"', '"coordinates"')
        )
        try:
            prediction = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            prediction = None

        if isinstance(prediction, dict):
            counts["json_valid"] += 1
            counts["schema_valid"] += is_sample_schema(prediction)

            if prediction.get("crop") == reference.get("crop"):
                counts["crop_exact"] += 1
            if prediction.get("plant_part") == reference.get("plant_part"):
                counts["plant_part_exact"] += 1

            pred_objects = prediction.get("objects")
        else:
            pred_objects = []
        if not isinstance(pred_objects, list):
            pred_objects = []
        counts["predicted_objects"] += len(pred_objects)
        extra_objects = max(len(pred_objects) - len(ref_objects), 0)
        counts["extra_predicted_objects"] += extra_objects
        counts["samples_with_extra_objects"] += extra_objects > 0
        matched = min(len(pred_objects), len(ref_objects))
        counts["index_matched_objects"] += matched
        counts["object_count_exact"] += len(pred_objects) == len(ref_objects)
        count_errors.append(abs(len(pred_objects) - len(ref_objects)))

        for prediction_object, reference_object in zip(pred_objects, ref_objects):
            if not isinstance(prediction_object, dict):
                continue
            categorical["object_id_total"] += 1
            categorical["object_id_exact"] += prediction_object.get("object_id") == reference_object.get("object_id")
            if isinstance(prediction_object.get("caption"), str):
                caption_f1.append(token_f1(prediction_object["caption"], reference_object["caption"]))
            if isinstance(prediction_object.get("action"), str):
                action_f1.append(token_f1(prediction_object["action"], reference_object["action"]))
            if isinstance(prediction_object.get("confidence"), (int, float)) and not isinstance(
                prediction_object.get("confidence"), bool
            ):
                confidence_errors.append(abs(prediction_object["confidence"] - reference_object["confidence"]))
            for field in ("image_quality",):
                categorical[f"{field}_total"] += 1
                categorical[f"{field}_exact"] += prediction_object.get(field) == reference_object.get(field)
            pred_assessment = prediction_object.get("assessment")
            ref_assessment = reference_object.get("assessment", {})
            if isinstance(pred_assessment, dict):
                for field in ("stage", "disease", "severity"):
                    if field in ref_assessment:
                        categorical[f"{field}_total"] += 1
                        categorical[f"{field}_exact"] += pred_assessment.get(field) == ref_assessment[field]

        ref_by_id = {
            item["object_id"]: item
            for item in ref_objects
            if isinstance(item, dict) and isinstance(item.get("object_id"), str)
        }
        pred_ids = [
            item.get("object_id")
            for item in pred_objects
            if isinstance(item, dict) and isinstance(item.get("object_id"), str)
        ]
        counts["duplicate_predicted_object_ids"] += sum(
            amount - 1 for amount in Counter(pred_ids).values() if amount > 1
        )
        unique_pred_ids = set(pred_ids)
        counts["id_matched_objects"] += len(unique_pred_ids & set(ref_by_id))
        counts["object_id_set_exact"] += unique_pred_ids == set(ref_by_id) and len(pred_ids) == len(unique_pred_ids)
        seen_ids: set[str] = set()
        for prediction_object in pred_objects:
            if not isinstance(prediction_object, dict):
                continue
            object_id = prediction_object.get("object_id")
            if not isinstance(object_id, str) or object_id in seen_ids or object_id not in ref_by_id:
                continue
            seen_ids.add(object_id)
            reference_object = ref_by_id[object_id]
            prediction_caption = prediction_object.get("caption")
            reference_caption = reference_object.get("caption")
            if isinstance(prediction_caption, str) and isinstance(reference_caption, str):
                caption_id_pairs.append((prediction_caption, reference_caption))
                caption_rouge_l.append(rouge_l_f1(prediction_caption, reference_caption))

    samples = counts["samples"]
    object_link_precision = ratio(counts["id_matched_objects"], counts["predicted_objects"])
    object_link_recall = ratio(counts["id_matched_objects"], counts["reference_objects"])
    result = {
        "metric_contract": "cv801_vlm_structured_v3_all_samples",
        "label_source": "reviewed_gpt_caption_not_independent_reference",
        "samples": samples,
        "json_valid_rate": ratio(counts["json_valid"], samples),
        "schema_valid_rate": ratio(counts["schema_valid"], samples),
        "think_prefix_rate": ratio(counts["think_prefix_stripped"], samples),
        "forbidden_box_output_rate": ratio(counts["forbidden_box_output"], samples),
        "crop_exact_rate": ratio(counts["crop_exact"], samples),
        "plant_part_exact_rate": ratio(counts["plant_part_exact"], samples),
        "object_count_exact_rate": ratio(counts["object_count_exact"], samples),
        "object_count_mae": mean(count_errors),
        "object_count_precision": ratio(counts["index_matched_objects"], counts["predicted_objects"]),
        "object_count_recall": ratio(counts["index_matched_objects"], counts["reference_objects"]),
        "object_id_link_precision": object_link_precision,
        "object_id_link_recall": object_link_recall,
        "object_id_link_f1": harmonic_mean(object_link_precision, object_link_recall),
        "object_id_set_exact_rate": ratio(counts["object_id_set_exact"], samples),
        "duplicate_object_id_rate": ratio(
            counts["duplicate_predicted_object_ids"], counts["predicted_objects"]
        ),
        "extra_object_hallucination_proxy_rate": ratio(
            counts["extra_predicted_objects"], counts["predicted_objects"]
        ),
        "samples_with_extra_object_rate": ratio(counts["samples_with_extra_objects"], samples),
        "caption_token_f1": mean(caption_f1),
        "caption_bleu4_corpus": corpus_bleu4(caption_id_pairs),
        "caption_rouge_l_f1": mean(caption_rouge_l),
        "action_token_f1": mean(action_f1),
        "confidence_mae": mean(confidence_errors),
        "categorical_exact": {
            field: ratio(categorical[f"{field}_exact"], categorical[f"{field}_total"])
            for field in ("object_id", "image_quality", "stage", "disease", "severity")
        },
        "metric_notes": {
            "caption_matching": "unique object_id matches only",
            "caption_bleu4_corpus": "lowercase alphanumeric tokens; single reference; unsmoothed corpus BLEU-4",
            "caption_rouge_l_f1": "mean object-level ROUGE-L F1 over unique object_id matches",
            "extra_object_hallucination_proxy_rate": (
                "surplus predicted objects divided by all predicted objects; this is not a visual claim-level hallucination score"
            ),
            "object_id_link_metrics": (
                "reference-ID association only; visual linkage correctness requires an independent image evaluator"
            ),
        },
        "counts": dict(sorted((counts | categorical).items())),
    }
    return result


def main() -> None:
    args = parse_args()
    result = evaluate(read_jsonl(args.predictions))
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
