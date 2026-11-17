from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_writer_reviewer_repair_lineage.py"
CELLS = (
    ("2B", "low", 12),
    ("2B", "mid", 11),
    ("2B", "high", 17),
    ("4B", "low", 5),
    ("4B", "mid", 7),
    ("4B", "high", 8),
    ("9B", "low", 12),
    ("9B", "mid", 19),
    ("9B", "high", 20),
)


def test_lineage_manifest_records_all_111_rows(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(SCRIPT),
        "--bundle-job-id",
        "12345",
        "--parent-bundle-job-id",
        "12222",
        "--project-root",
        str(tmp_path),
        "--output-json",
        str(tmp_path / "lineage.json"),
        "--output-markdown",
        str(tmp_path / "lineage.md"),
    ]
    for line_number in (7, 8, 22):
        command.extend(["--residual-bundle-line", str(line_number)])
    for writer, tier, count in CELLS:
        repair_dir = tmp_path / "repairs" / f"{writer}-{tier}"
        repair_dir.mkdir(parents=True)
        env_path = repair_dir / "repair.env"
        env_path.write_text(
            "\n".join(
                [
                    f"output_dir=outputs/{writer}-{tier}",
                    f"source_predictions=outputs/{writer}-{tier}/predictions.jsonl",
                    "invalid_lines=" + ":".join(str(value) for value in range(1, count + 1)),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        replacement = repair_dir / "bundled-predictions-12345.jsonl"
        replacement.write_text("".join("{}\n" for _ in range(count)), encoding="utf-8")
        command.extend(["--cell", f"{writer},{tier},{env_path}"])
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "lineage.json").read_text(encoding="utf-8"))
    assert payload["total_repaired_rows"] == 111
    assert payload["parent_bundle_job_id"] == "12222"
    assert payload["residual_bundle_line_numbers_1_based"] == [7, 8, 22]
    assert len(payload["cells"]) == 9
    assert "111 of 9,000" in (tmp_path / "lineage.md").read_text(encoding="utf-8")
    assert "Residual responses rerun" in (tmp_path / "lineage.md").read_text(encoding="utf-8")
