from __future__ import annotations

import unittest
from datetime import datetime

from CV801.scripts.summarize_slurm_efficiency import (
    build_summary,
    parse_system_specs,
    requested_gpus,
)


class SummarizeSlurmEfficiencyTest(unittest.TestCase):
    def test_requested_gpus_does_not_double_count_typed_and_generic_tres(self) -> None:
        tres = "billing=64,cpu=64,gres/gpu:mi210=8,gres/gpu=8,mem=512G,node=1"
        self.assertEqual(requested_gpus(tres), 8)

    def test_parse_system_specs_rejects_duplicate_job_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate Slurm job ID"):
            parse_system_specs(["t2=10,10"])

    def test_build_summary_charges_elapsed_gpu_time(self) -> None:
        jobs = {
            "10": {
                "job_id": "10",
                "state": "FAILED",
                "elapsed_seconds": 100,
                "start": "2026-11-11T10:00:00",
                "end": "2026-11-11T10:01:40",
                "requested_gpus": 2,
            },
            "11": {
                "job_id": "11",
                "state": "COMPLETED",
                "elapsed_seconds": 60,
                "start": "2026-11-11T10:02:00",
                "end": "2026-11-11T10:03:00",
                "requested_gpus": 1,
            },
        }
        payload = build_summary(
            {"system": ["10", "11"]}, jobs, images=10, accepted_failed={"10"}
        )
        result = payload["systems"]["system"]
        self.assertEqual(result["gpu_seconds"], 260)
        self.assertEqual(result["post_inference_audit_failures_accepted"], ["10"])
        self.assertEqual(
            result["execution_span_seconds"],
            (
                datetime.fromisoformat("2026-11-11T10:03:00")
                - datetime.fromisoformat("2026-11-11T10:00:00")
            ).total_seconds(),
        )


if __name__ == "__main__":
    unittest.main()
