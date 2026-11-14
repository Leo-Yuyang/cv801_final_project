#!/usr/bin/env python3
"""Aggregate the frozen Golden Test metrics into one JSON and Markdown report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for kind in ("detection", "vlm", "independent", "review"):
        parser.add_argument(f"--{kind}", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--efficiency", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def load_named(specs: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for spec in specs:
        name, separator, raw_path = spec.partition("=")
        if not separator or not name or not raw_path or name in result:
            raise ValueError(f"Invalid or duplicate named input: {spec}")
        path = Path(raw_path)
        result[name] = json.loads(path.read_text(encoding="utf-8"))
    return result


def number(value: object, digits: int = 4) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{value:.{digits}f}"


def ratio(numerator: object, denominator: object) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)) or denominator == 0:
        return None
    return numerator / denominator


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CV801 Golden Test 最终结果",
        "",
        f"生成时间（UTC）：{payload['generated_at_utc']}",
        "",
        "## Detection",
        "",
        "| 配置 | COCO mAP 50–95 | AP50 | 阈值 | Precision | Recall | F1 | TP mean IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, record in payload["detection"].items():
        metrics = record.get("metrics", {})
        operating = record.get("operating_point") or {}
        lines.append(
            f"| {name} | {number(metrics.get('AP_50_95'))} | {number(metrics.get('AP50'))} "
            f"| {number(operating.get('score_threshold'), 6)} | {number(operating.get('precision'))} "
            f"| {number(operating.get('recall'))} | {number(operating.get('f1'))} "
            f"| {number(operating.get('mean_matched_iou'))} |"
        )
    lines.extend([
        "",
        "## Structured VLM",
        "",
        "| 配置 | JSON valid | Object count exact | Object-ID P | Object-ID R | Object-ID F1 | Caption token F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, record in payload["vlm"].items():
        lines.append(
            f"| {name} | {number(record.get('json_valid_rate'))} | "
            f"{number(record.get('object_count_exact_rate'))} | "
            f"{number(record.get('object_id_link_precision'))} | "
            f"{number(record.get('object_id_link_recall'))} | "
            f"{number(record.get('object_id_link_f1'))} | "
            f"{number(record.get('caption_token_f1'))} |"
        )
    lines.extend([
        "",
        "### Structured attribute details",
        "",
        "| 配置 | Crop | Plant part | Image quality | Stage | Disease | Severity | Action F1 | Confidence MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, record in payload["vlm"].items():
        categorical = record.get("categorical_exact", {})
        lines.append(
            f"| {name} | {number(record.get('crop_exact_rate'))} | "
            f"{number(record.get('plant_part_exact_rate'))} | "
            f"{number(categorical.get('image_quality'))} | {number(categorical.get('stage'))} | "
            f"{number(categorical.get('disease'))} | {number(categorical.get('severity'))} | "
            f"{number(record.get('action_token_f1'))} | {number(record.get('confidence_mae'))} |"
        )
    lines.extend([
        "",
        "## Independent visual evaluator (0–6)",
        "",
        "| 配置 | Valid | Visual evidence | Conclusion support | Confidence | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, record in payload["independent"].items():
        scores = record.get("mean_scores", {})
        lines.append(
            f"| {name} | {number(record.get('valid_rate'))} | "
            f"{number(scores.get('visual_evidence'))} | {number(scores.get('conclusion_support'))} | "
            f"{number(scores.get('confidence_appropriateness'))} | {number(scores.get('total'))} |"
        )
    lines.extend([
        "",
        "### Independent claim and linkage audit",
        "",
        "| 配置 | Supported claims | Unsupported claims | Unsupported share | Correct links | Incorrect links | Uncertain links |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, record in payload["independent"].items():
        supported = record.get("supported_claims")
        unsupported = record.get("unsupported_claims")
        linkage = record.get("object_linkage", {})
        lines.append(
            f"| {name} | {supported if isinstance(supported, int) else '—'} | "
            f"{unsupported if isinstance(unsupported, int) else '—'} | "
            f"{number(ratio(unsupported, supported + unsupported) if isinstance(supported, int) and isinstance(unsupported, int) else None)} | "
            f"{linkage.get('correct', '—')} | {linkage.get('incorrect', '—')} | {linkage.get('uncertain', '—')} |"
        )
    lines.extend([
        "",
        "## Review decisions",
        "",
        "| 阶段 | 类型 | 样本数 | 决策计数 |",
        "|---|---|---:|---|",
    ])
    for name, record in payload["review"].items():
        decisions = json.dumps(record.get("decisions", {}), ensure_ascii=False, sort_keys=True)
        lines.append(f"| {name} | {record.get('kind', '—')} | {record.get('samples', '—')} | `{decisions}` |")
    lines.extend([
        "",
        "## Efficiency",
        "",
        "| 配置 | GPU-hours | GPU-hours/image | Execution seconds/image | Execution span (min) |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, record in payload["efficiency"].get("systems", {}).items():
        span = record.get("execution_span_seconds")
        lines.append(
            f"| {name} | {number(record.get('gpu_hours'))} | "
            f"{number(record.get('gpu_hours_per_image'), 6)} | "
            f"{number(record.get('execution_span_seconds_per_image'), 3)} | "
            f"{number(span / 60 if isinstance(span, (int, float)) else None, 2)} |"
        )
    lines.extend([
        "",
        "说明：reference-based caption 指标采用 reviewed GPT caption；独立 0–6 视觉评估单列，避免将二者混为同一真值口径。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload = {
        "protocol": "cv801_golden_final_summary_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"images": 8097, "objects": 14601, "golden_images": 1000},
        "detection": load_named(args.detection),
        "vlm": load_named(args.vlm),
        "independent": load_named(args.independent),
        "review": load_named(args.review),
        "efficiency": json.loads(args.efficiency.read_text(encoding="utf-8")),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output_json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
