import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_independent_vlm_evaluator_outputs import audit


class AuditIndependentVlmEvaluatorOutputsTest(unittest.TestCase):
    def test_validates_scores_and_aggregates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs.jsonl"
            predictions = root / "predictions.jsonl"
            inputs.write_text(json.dumps({"sample_id": "sample-a"}) + "\n")
            payload = {
                "sample_id": "sample-a",
                "object_reviews": [{
                    "object_id": "object_001",
                    "supported_claims": ["Visible."],
                    "unsupported_claims": [],
                    "object_linkage": "correct",
                }],
                "scores": {
                    "visual_evidence": 2,
                    "conclusion_support": 1,
                    "confidence_appropriateness": 2,
                },
                "total": 5,
                "rationale": ["Mostly supported."],
            }
            predictions.write_text(json.dumps({"response": json.dumps(payload)}) + "\n")
            result = audit(predictions, inputs)
            self.assertEqual(result["valid"], 1)
            self.assertEqual(result["mean_scores"]["total"], 5.0)
            self.assertEqual(result["object_linkage"], {"correct": 1})

    def test_rejects_bad_total(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs.jsonl"
            predictions = root / "predictions.jsonl"
            inputs.write_text(json.dumps({"sample_id": "sample-a"}) + "\n")
            predictions.write_text(json.dumps({"response": json.dumps({
                "sample_id": "sample-a",
                "object_reviews": [],
                "scores": {
                    "visual_evidence": 2,
                    "conclusion_support": 2,
                    "confidence_appropriateness": 2,
                },
                "total": 5,
                "rationale": [],
            })}) + "\n")
            result = audit(predictions, inputs)
            self.assertEqual(result["valid"], 0)
            self.assertEqual(result["invalid_lines"], [1])


if __name__ == "__main__":
    unittest.main()
