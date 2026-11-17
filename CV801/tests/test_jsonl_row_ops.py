import json
import tempfile
import unittest
from pathlib import Path

from scripts.jsonl_row_ops import (
    append_user_instruction,
    concatenate_rows,
    merge_rows,
    read_jsonl,
    replace_responses,
    replace_rows,
    select_line_numbers,
    select_rows,
)


class JsonlRowOpsTest(unittest.TestCase):
    def test_select_replace_and_merge_preserve_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            rows = [{"images": [f"{index}.jpg"], "value": index} for index in range(1, 5)]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            first = root / "first.jsonl"
            second = root / "second.jsonl"
            self.assertEqual(select_rows(source, first, 1, 2), 2)
            self.assertEqual(select_rows(source, second, 3, 4), 2)

            selected = root / "selected.jsonl"
            self.assertEqual(select_line_numbers(source, selected, [1, 3]), 2)
            self.assertEqual([row["value"] for row in read_jsonl(selected)], [1, 3])

            merged = root / "merged.jsonl"
            self.assertEqual(merge_rows([first, second], source, merged), 4)
            self.assertEqual(read_jsonl(merged), rows)

            concatenated = root / "concatenated.jsonl"
            self.assertEqual(concatenate_rows([first, second], concatenated), 4)
            self.assertEqual(read_jsonl(concatenated), rows)

            replacement = root / "replacement.jsonl"
            replacement.write_text(
                json.dumps({"images": ["2.jpg"], "value": 20}) + "\n",
                encoding="utf-8",
            )
            repaired = root / "repaired.jsonl"
            self.assertEqual(replace_rows(source, replacement, repaired, [2]), 4)
            self.assertEqual(read_jsonl(repaired)[1]["value"], 20)

    def test_append_user_instruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            output = root / "output.jsonl"
            source.write_text(
                json.dumps({"messages": [{"role": "user", "content": "Review."}]}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(append_user_instruction(source, output, "Be concise."), 1)
            self.assertTrue(read_jsonl(output)[0]["messages"][0]["content"].endswith("Be concise."))

    def test_merge_text_only_predictions_checks_prompt_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            references = root / "references.jsonl"
            predictions = root / "predictions.jsonl"
            output = root / "output.jsonl"
            references.write_text(json.dumps({
                "images": ["1.jpg"],
                "messages": [{"role": "user", "content": "Review sample one."}],
            }) + "\n", encoding="utf-8")
            prediction = {
                "messages": [{"role": "user", "content": "Review sample one."}],
                "response": "{}",
            }
            predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")

            self.assertEqual(merge_rows([predictions], references, output), 1)
            self.assertEqual(read_jsonl(output), [prediction])

            mismatched = root / "mismatched.jsonl"
            mismatched.write_text(json.dumps({
                "messages": [{"role": "user", "content": "Wrong sample."}],
                "response": "{}",
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Prompt mismatch"):
                merge_rows([mismatched], references, root / "mismatch-output.jsonl")

    def test_replace_text_response_checks_prompt_ancestry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            replacement = root / "replacement.jsonl"
            output = root / "output.jsonl"
            source.write_text(json.dumps({
                "messages": [
                    {"role": "user", "content": "Resolve reviews."},
                    {"role": "assistant", "content": "{truncated"},
                ],
                "response": "{truncated",
            }) + "\n", encoding="utf-8")
            replacement.write_text(json.dumps({
                "messages": [
                    {"role": "user", "content": "Resolve reviews.\nBe concise."},
                    {"role": "assistant", "content": "{}"},
                ],
                "response": "{}",
            }) + "\n", encoding="utf-8")

            self.assertEqual(replace_responses(source, replacement, output, [1]), 1)
            repaired = read_jsonl(output)[0]
            self.assertEqual(repaired["response"], "{}")
            self.assertEqual(repaired["messages"][-1]["content"], "{}")


if __name__ == "__main__":
    unittest.main()
