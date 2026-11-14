#!/usr/bin/env python3
"""Prepare leakage-safe Rc/Rd/J inputs from frozen Writer predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


THINK_PREFIX = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)

VISUAL_SYSTEM = (
    "You are visual reviewer Rc. Audit a draft only against visible image evidence. "
    "Do not use hidden references, invent facts, or rewrite the answer. Return strict JSON."
)
DOMAIN_SYSTEM = (
    "You are agricultural domain reviewer Rd. Audit crop, plant-part, stage, disease, "
    "severity, confidence, and action claims conservatively against the image and standard "
    "domain knowledge. Do not use hidden dataset references or rewrite the answer. Return strict JSON."
)
JUDGE_SYSTEM = (
    "You are text-only Judge J. Resolve two independent reviews of a frozen Writer draft. "
    "You cannot see the image and must not add visual facts. Return strict JSON revision feedback."
)

REVIEW_SCHEMA = (
    '{"verdict":"pass|revise","object_reviews":[{"object_id":"object_001",'
    '"supported_claims":[],"unsupported_claims":[],"missed_evidence":[],'
    '"confidence_issue":"none|overconfident|underconfident"}],'
    '"global_issues":[],"revision_instructions":[]}'
)
JUDGE_SCHEMA = (
    '{"decision":"accept|revise","agreements":[],"resolved_disagreements":[],'
    '"revision_feedback":[],"recheck_requirements":[]}'
)
RECHECK_SCHEMA = (
    '{"decision":"accept|revise_again","resolved_issues":[],"remaining_issues":[],'
    '"revision_feedback":[]}'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    reviewers = subparsers.add_parser("reviewers")
    reviewers.add_argument("--writer-predictions", type=Path, required=True)
    reviewers.add_argument("--output-dir", type=Path, required=True)
    reviewers.add_argument("--max-samples", type=int)
    judge = subparsers.add_parser("judge")
    judge.add_argument("--manifest", type=Path, required=True)
    judge.add_argument("--visual-predictions", type=Path, required=True)
    judge.add_argument("--domain-predictions", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--max-samples", type=int)
    revision = subparsers.add_parser("revision")
    revision.add_argument("--writer-predictions", type=Path, required=True)
    revision.add_argument("--feedback-source", choices=("rc", "rd", "rc_rd", "judge"), required=True)
    revision.add_argument("--visual-predictions", type=Path)
    revision.add_argument("--domain-predictions", type=Path)
    revision.add_argument("--judge-predictions", type=Path)
    revision.add_argument("--output", type=Path, required=True)
    revision.add_argument("--max-samples", type=int)
    revision_round2 = subparsers.add_parser("revision-round2")
    revision_round2.add_argument("--revised-predictions", type=Path, required=True)
    revision_round2.add_argument("--recheck-predictions", type=Path, required=True)
    revision_round2.add_argument("--output", type=Path, required=True)
    revision_round2.add_argument("--max-samples", type=int)
    recheck = subparsers.add_parser("recheck")
    recheck.add_argument("--manifest", type=Path, required=True)
    recheck.add_argument("--visual-predictions", type=Path, required=True)
    recheck.add_argument("--domain-predictions", type=Path, required=True)
    recheck.add_argument("--judge-predictions", type=Path, required=True)
    recheck.add_argument("--revised-predictions", type=Path, required=True)
    recheck.add_argument("--output", type=Path, required=True)
    recheck.add_argument("--max-samples", type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def final_response(record: dict[str, Any]) -> str:
    return THINK_PREFIX.sub("", str(record.get("response", ""))).strip()


def image_path(record: dict[str, Any]) -> str:
    images = record.get("images") or []
    if not images:
        raise ValueError("Writer prediction has no image")
    first = images[0]
    path = first.get("path") if isinstance(first, dict) else first
    if not path:
        raise ValueError("Writer prediction image has no path")
    return str(path)


def sample_id(path: str, line_number: int) -> str:
    return hashlib.sha256(f"{line_number}:{path}".encode()).hexdigest()[:16]


def reviewer_prompt(role: str, draft: str) -> str:
    focus = (
        "Check object coverage, object linkage, visible attributes, and whether every claimed visual "
        "detail is supported."
        if role == "visual"
        else "Check agricultural terminology and whether stage, disease, severity, confidence, and action are justified."
    )
    return (
        f"<image>\n{focus} Review each object separately. Do not score wording similarity and do not "
        f"consult reference answers. Required schema: {REVIEW_SCHEMA}\nWRITER_DRAFT:\n{draft}"
    )


def prepare_reviewers(
    writer_predictions: Path, output_dir: Path, max_samples: int | None = None
) -> dict[str, int]:
    writers = read_jsonl(writer_predictions)
    if max_samples is not None:
        writers = writers[:max_samples]
    visual_records: list[dict[str, Any]] = []
    domain_records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for index, writer in enumerate(writers, 1):
        path = image_path(writer)
        draft = final_response(writer)
        identifier = sample_id(path, index)
        common = {"images": [path]}
        visual_records.append({
            "messages": [
                {"role": "system", "content": VISUAL_SYSTEM},
                {"role": "user", "content": reviewer_prompt("visual", draft)},
            ],
            **common,
        })
        domain_records.append({
            "messages": [
                {"role": "system", "content": DOMAIN_SYSTEM},
                {"role": "user", "content": reviewer_prompt("domain", draft)},
            ],
            **common,
        })
        manifests.append({
            "sample_id": identifier,
            "image_path": path,
            "draft": draft,
            "draft_sha256": hashlib.sha256(draft.encode()).hexdigest(),
            "writer_prediction_line": index,
            "references_exposed_to_reviewers": False,
        })
    write_jsonl(output_dir / "visual_reviewer_inputs.jsonl", visual_records)
    write_jsonl(output_dir / "domain_reviewer_inputs.jsonl", domain_records)
    write_jsonl(output_dir / "review_manifest.jsonl", manifests)
    return {"samples": len(manifests)}


def prepare_judge(
    manifest_path: Path,
    visual_predictions: Path,
    domain_predictions: Path,
    output: Path,
    max_samples: int | None = None,
) -> dict[str, int]:
    manifests = read_jsonl(manifest_path)
    visual = read_jsonl(visual_predictions)
    domain = read_jsonl(domain_predictions)
    if max_samples is not None:
        manifests = manifests[:max_samples]
        visual = visual[:max_samples]
        domain = domain[:max_samples]
    if not (len(manifests) == len(visual) == len(domain)):
        raise ValueError("Manifest and reviewer prediction counts differ")
    records: list[dict[str, Any]] = []
    for manifest, visual_record, domain_record in zip(manifests, visual, domain, strict=True):
        prompt = (
            f"Resolve the reviews without seeing the image. If a visual claim is disputed, request a "
            f"conservative revision rather than deciding the image content yourself. Required schema: "
            f"{JUDGE_SCHEMA}\nWRITER_DRAFT:\n{manifest['draft']}\nVISUAL_REVIEW:\n"
            f"{final_response(visual_record)}\nDOMAIN_REVIEW:\n{final_response(domain_record)}"
        )
        records.append({
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        })
    write_jsonl(output, records)
    return {"samples": len(records)}


def _conversation_prefix(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages") or []
    prefix = [message for message in messages if message.get("role") in {"system", "user"}]
    if not prefix or prefix[-1].get("role") != "user":
        raise ValueError("Writer prediction has no source user prompt")
    return [{"role": str(message["role"]), "content": str(message["content"])} for message in prefix]


def prepare_revision(
    writer_predictions: Path,
    feedback_source: str,
    output: Path,
    visual_predictions: Path | None = None,
    domain_predictions: Path | None = None,
    judge_predictions: Path | None = None,
    max_samples: int | None = None,
) -> dict[str, int]:
    writers = read_jsonl(writer_predictions)
    feedback_paths = {
        "rc": [visual_predictions],
        "rd": [domain_predictions],
        "rc_rd": [visual_predictions, domain_predictions],
        "judge": [judge_predictions],
    }[feedback_source]
    if any(path is None for path in feedback_paths):
        raise ValueError(f"Missing predictions for feedback source {feedback_source}")
    feedback_sets = [read_jsonl(path) for path in feedback_paths if path is not None]
    if max_samples is not None:
        writers = writers[:max_samples]
        feedback_sets = [records[:max_samples] for records in feedback_sets]
    if any(len(records) != len(writers) for records in feedback_sets):
        raise ValueError("Writer and feedback prediction counts differ")

    records: list[dict[str, Any]] = []
    for index, writer in enumerate(writers):
        feedback = [final_response(records_[index]) for records_ in feedback_sets]
        feedback_names = {
            "rc": ["VISUAL_REVIEW"],
            "rd": ["DOMAIN_REVIEW"],
            "rc_rd": ["VISUAL_REVIEW", "DOMAIN_REVIEW"],
            "judge": ["JUDGE_FEEDBACK"],
        }[feedback_source]
        feedback_text = "\n".join(
            f"{name}:\n{text}" for name, text in zip(feedback_names, feedback, strict=True)
        )
        prompt = (
            "Revise the previous draft using the review feedback below. Treat feedback as advisory: "
            "check it against the image and localization hints, remove unsupported claims, and do not add "
            "facts that are not visible. Return the complete revised JSON object in the original schema, "
            "without markdown, coordinates, or commentary.\n"
            f"{feedback_text}"
        )
        messages = _conversation_prefix(writer)
        messages.append({"role": "assistant", "content": final_response(writer)})
        messages.append({"role": "user", "content": prompt})
        records.append({"messages": messages, "images": [image_path(writer)]})
    write_jsonl(output, records)
    return {"samples": len(records)}


def prepare_recheck(
    manifest_path: Path,
    visual_predictions: Path,
    domain_predictions: Path,
    judge_predictions: Path,
    revised_predictions: Path,
    output: Path,
    max_samples: int | None = None,
) -> dict[str, int]:
    datasets = [
        read_jsonl(manifest_path),
        read_jsonl(visual_predictions),
        read_jsonl(domain_predictions),
        read_jsonl(judge_predictions),
        read_jsonl(revised_predictions),
    ]
    if max_samples is not None:
        datasets = [records[:max_samples] for records in datasets]
    if len({len(records) for records in datasets}) != 1:
        raise ValueError("Recheck input counts differ")
    manifests, visual, domain, judges, revised = datasets
    records: list[dict[str, Any]] = []
    for manifest, visual_record, domain_record, judge_record, revised_record in zip(
        manifests, visual, domain, judges, revised, strict=True
    ):
        prompt = (
            "Recheck the revised response against the two original reviews and your previous feedback. "
            "You still cannot see the image: do not add or adjudicate visual facts. Return strict JSON using "
            f"this schema: {RECHECK_SCHEMA}\nORIGINAL_DRAFT:\n{manifest['draft']}\nVISUAL_REVIEW:\n"
            f"{final_response(visual_record)}\nDOMAIN_REVIEW:\n{final_response(domain_record)}\n"
            f"PREVIOUS_JUDGE_FEEDBACK:\n{final_response(judge_record)}\nREVISED_RESPONSE:\n"
            f"{final_response(revised_record)}"
        )
        records.append({
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        })
    write_jsonl(output, records)
    return {"samples": len(records)}


def prepare_revision_round2(
    revised_predictions: Path,
    recheck_predictions: Path,
    output: Path,
    max_samples: int | None = None,
) -> dict[str, int]:
    revised = read_jsonl(revised_predictions)
    rechecks = read_jsonl(recheck_predictions)
    if max_samples is not None:
        revised = revised[:max_samples]
        rechecks = rechecks[:max_samples]
    if len(revised) != len(rechecks):
        raise ValueError("Revised and recheck prediction counts differ")

    records: list[dict[str, Any]] = []
    for revised_record, recheck_record in zip(revised, rechecks, strict=True):
        messages = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in revised_record.get("messages", [])
            if message.get("role") in {"system", "user", "assistant"}
        ]
        if not messages:
            raise ValueError("Revised prediction has no source conversation")
        current_response = final_response(revised_record)
        if messages[-1]["role"] == "assistant":
            messages[-1]["content"] = current_response
        else:
            messages.append({"role": "assistant", "content": current_response})
        messages.append({
            "role": "user",
            "content": (
                "Revise the current response one final time using the text-only Judge recheck below. "
                "Treat it as advisory, verify every requested change against the image and localization "
                "hints, remove unsupported claims, and return the complete JSON object in the original "
                "schema without markdown or commentary.\nRECHECK_FEEDBACK:\n"
                f"{final_response(recheck_record)}"
            ),
        })
        records.append({"messages": messages, "images": [image_path(revised_record)]})
    write_jsonl(output, records)
    return {"samples": len(records)}


def main() -> None:
    args = parse_args()
    if args.command == "reviewers":
        result = prepare_reviewers(args.writer_predictions, args.output_dir, args.max_samples)
    elif args.command == "judge":
        result = prepare_judge(
            args.manifest,
            args.visual_predictions,
            args.domain_predictions,
            args.output,
            args.max_samples,
        )
    elif args.command == "revision":
        result = prepare_revision(
            args.writer_predictions,
            args.feedback_source,
            args.output,
            args.visual_predictions,
            args.domain_predictions,
            args.judge_predictions,
            args.max_samples,
        )
    elif args.command == "revision-round2":
        result = prepare_revision_round2(
            args.revised_predictions,
            args.recheck_predictions,
            args.output,
            args.max_samples,
        )
    else:
        result = prepare_recheck(
            args.manifest,
            args.visual_predictions,
            args.domain_predictions,
            args.judge_predictions,
            args.revised_predictions,
            args.output,
            args.max_samples,
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
