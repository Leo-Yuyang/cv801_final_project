import csv
import tempfile
import unittest
from pathlib import Path

from CV801.scripts.analyze_writer_reviewer_grid import analyze


def write_grid(
    path: Path,
    samples: int,
    omit: tuple[str, str] | None = None,
    zero_predictions: tuple[str, str] | None = None,
    zero_matches: tuple[str, str] | None = None,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "writer",
                "reviewer_tier",
                "schema_valid",
                "link_matched",
                "link_predicted",
                "link_reference",
                "visual_total",
                "supported_claims",
                "unsupported_claims",
            ),
        )
        writer.writeheader()
        for writer_size, base in (("2B", 0.0), ("4B", 1.0), ("9B", 2.0)):
            for tier, gain in (("low", 0.0), ("mid", 0.5), ("high", 1.0)):
                if omit == (writer_size, tier):
                    continue
                for sample in range(samples):
                    predicted = 0 if zero_predictions == (writer_size, tier) else 4
                    matched = (
                        0
                        if predicted == 0 or zero_matches == (writer_size, tier)
                        else 1 + base + gain
                    )
                    writer.writerow(
                        {
                            "sample_id": f"sample-{sample}",
                            "writer": writer_size,
                            "reviewer_tier": tier,
                            "schema_valid": 1,
                            "link_matched": matched,
                            "link_predicted": predicted,
                            "link_reference": 4,
                            "visual_total": base + gain + sample / 100,
                            "supported_claims": 3 + base + gain,
                            "unsupported_claims": 1,
                        }
                    )


def args_for(path: Path, samples: int):
    class Args:
        input_csv = path
        output_json = path.with_suffix(".json")
        output_markdown = path.with_suffix(".md")
        iterations = 200
        seed = 20261104
        expected_samples = samples
        batch_size = 50

    return Args()


class AnalyzeWriterReviewerGridTest(unittest.TestCase):
    def test_complete_grid_is_paired_and_interactions_are_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "grid.csv"
            write_grid(source, samples=8)
            result = analyze(args_for(source, samples=8))
        contrasts = {
            item["name"]: item
            for item in result["metrics"]["independent_visual_total"]["contrasts"]
        }
        self.assertAlmostEqual(
            contrasts["writer_4B_minus_2B_at_low"]["estimate"], 1.0
        )
        self.assertAlmostEqual(
            contrasts["gap_closure_2B_vs_9B_high_minus_low"]["estimate"], 0.0
        )

    def test_missing_cell_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "grid.csv"
            write_grid(source, samples=4, omit=("4B", "mid"))
            with self.assertRaisesRegex(ValueError, "missing grid cells"):
                analyze(args_for(source, samples=4))

    def test_zero_prediction_denominator_is_reported_as_undefined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "grid.csv"
            write_grid(source, samples=8, zero_predictions=("4B", "low"))
            result = analyze(args_for(source, samples=8))
        metric = result["metrics"]["object_id_link_f1"]
        self.assertIsNone(metric["cell_values"]["4B/low"])
        contrast = next(
            item
            for item in metric["contrasts"]
            if item["name"] == "writer_4B_minus_2B_at_low"
        )
        self.assertIsNone(contrast["estimate"])
        self.assertEqual(contrast["valid_bootstrap_iterations"], 0)

    def test_defined_zero_f1_keeps_all_bootstrap_iterations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "grid.csv"
            write_grid(source, samples=8, zero_matches=("2B", "low"))
            result = analyze(args_for(source, samples=8))
        metric = result["metrics"]["object_id_link_f1"]
        self.assertEqual(metric["cell_values"]["2B/low"], 0.0)
        contrast = next(
            item
            for item in metric["contrasts"]
            if item["name"] == "gap_closure_2B_vs_9B_high_minus_low"
        )
        self.assertEqual(contrast["valid_bootstrap_iterations"], 200)


if __name__ == "__main__":
    unittest.main()
