from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_writer_reviewer_report.py"
WRITERS = ("2B", "4B", "9B")
TIERS = ("low", "mid", "high")
METRICS = (
    "independent_visual_total",
    "object_id_link_f1",
    "schema_valid_rate",
    "unsupported_claim_share",
)


def payload() -> dict:
    metrics = {}
    for metric_index, metric in enumerate(METRICS):
        cells = {
            f"{writer}/{tier}": 0.1 * (metric_index + 1) + 0.01 * writer_index + 0.001 * tier_index
            for writer_index, writer in enumerate(WRITERS)
            for tier_index, tier in enumerate(TIERS)
        }
        if metric == "independent_visual_total":
            cells = {key: 3.0 + value for key, value in cells.items()}
        contrasts = []
        for writer in ("2B", "4B"):
            contrasts.append(
                {
                    "name": f"gap_closure_{writer}_vs_9B_high_minus_low",
                    "family": "primary_interaction",
                    "estimate": 0.02,
                    "ci95_low": -0.01,
                    "ci95_high": 0.05,
                    "valid_bootstrap_iterations": 10_000,
                }
            )
        metrics[metric] = {
            "direction": "higher-is-better",
            "cell_values": cells,
            "contrasts": contrasts,
        }
    return {
        "protocol": "cv801_writer_reviewer_paired_bootstrap_v2_micro_metrics",
        "paired_samples": 1_000,
        "bootstrap_iterations": 10_000,
        "seed": 20_261_104,
        "metrics": metrics,
    }


def run_renderer(tmp_path: Path, value: dict) -> subprocess.CompletedProcess[str]:
    analysis = tmp_path / "analysis.json"
    output_tex = tmp_path / "tables.tex"
    output_markdown = tmp_path / "tables.md"
    analysis.write_text(json.dumps(value), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(analysis),
            "--output-tex",
            str(output_tex),
            "--output-markdown",
            str(output_markdown),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_renderer_writes_complete_tables(tmp_path: Path) -> None:
    result = run_renderer(tmp_path, payload())
    assert result.returncode == 0, result.stderr
    tex = (tmp_path / "tables.tex").read_text(encoding="utf-8")
    markdown = (tmp_path / "tables.md").read_text(encoding="utf-8")
    assert "\\label{tab:writer-reviewer-grid}" in tex
    assert "\\label{tab:writer-reviewer-interaction}" in tex
    assert "Unsupported-claim share" in tex
    assert "2B vs. 9B" in markdown
    assert "no automated causal or superiority claim" in markdown


def test_renderer_rejects_incomplete_bootstrap(tmp_path: Path) -> None:
    value = payload()
    value["bootstrap_iterations"] = 999
    result = run_renderer(tmp_path, value)
    assert result.returncode != 0
    assert "10,000 bootstrap iterations" in result.stderr


def test_renderer_marks_audited_undefined_object_link_metric(tmp_path: Path) -> None:
    value = payload()
    metric = value["metrics"]["object_id_link_f1"]
    metric["cell_values"]["4B/low"] = None
    metric["undefined_cells"] = {"4B/low": "zero predicted-object denominator"}
    contrast = next(
        item
        for item in metric["contrasts"]
        if item["name"] == "gap_closure_4B_vs_9B_high_minus_low"
    )
    contrast.update(
        {
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "valid_bootstrap_iterations": 0,
            "undefined_reason": "required cell metric is undefined",
        }
    )
    result = run_renderer(tmp_path, value)
    assert result.returncode == 0, result.stderr
    assert "--" in (tmp_path / "tables.tex").read_text(encoding="utf-8")
