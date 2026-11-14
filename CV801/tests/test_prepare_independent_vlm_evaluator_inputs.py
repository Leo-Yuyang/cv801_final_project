import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_independent_vlm_evaluator_inputs import prepare_inputs


class PrepareIndependentVlmEvaluatorInputsTest(unittest.TestCase):
    def test_references_and_thinking_are_not_exposed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            output = root / "evaluator_inputs.jsonl"
            predictions.write_text(json.dumps({
                "response": '<think>PRIVATE_REASONING</think>{"objects":[]}',
                "labels": "HIDDEN_REFERENCE",
                "images": [{"path": "image.jpg"}],
            }) + "\n", encoding="utf-8")

            self.assertEqual(prepare_inputs(predictions, output), 1)
            text = output.read_text(encoding="utf-8")
            record = json.loads(text)
            self.assertNotIn("HIDDEN_REFERENCE", text)
            self.assertNotIn("PRIVATE_REASONING", text)
            self.assertIn('{"objects":[]}', record["messages"][1]["content"])
            self.assertEqual(record["images"], ["image.jpg"])
            self.assertFalse(record["references_exposed_to_evaluator"])
            self.assertEqual(len(record["sample_id"]), 16)

    def test_max_samples_must_be_positive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            predictions.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_inputs(predictions, root / "output.jsonl", max_samples=0)


if __name__ == "__main__":
    unittest.main()
