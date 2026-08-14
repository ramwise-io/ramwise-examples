from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.custom_cuda.benchmark import numpy_score, one_shot_input_layout
from experiments.custom_cuda.common import atomic_json, read_json, stable_id
from experiments.custom_cuda.kernels import SCORE_SOURCE
from experiments.custom_cuda.matrix_runner import expand_jobs, job_id, validate_config
from experiments.custom_cuda.summarize import aggregate_fusion
from experiments.custom_cuda.summarize_profiles import duration_seconds, parse_export


class CustomCudaTests(unittest.TestCase):
    def test_numpy_reference(self) -> None:
        values = np.array([[1.0, 3.0], [-10.0, 2.0]], dtype=np.float32)
        means = np.array([1.0, 1.0], dtype=np.float32)
        inverse = np.array([2.0, 0.5], dtype=np.float32)
        weights = np.array([3.0, -2.0], dtype=np.float32)
        scores, flags = numpy_score(values, means, inverse, weights, 0.0)
        np.testing.assert_allclose(scores, [-2.0, -16.0])
        np.testing.assert_array_equal(flags, [0, 0])

    def test_one_shot_uploads_only_the_selected_layout(self) -> None:
        self.assertEqual(one_shot_input_layout("raw_soa_b256"), "soa")
        self.assertEqual(one_shot_input_layout("raw_aos_b256"), "aos")
        self.assertEqual(one_shot_input_layout("cupy_composed"), "aos")

    def test_source_contains_good_and_bad_layouts(self) -> None:
        self.assertIn("values[base + feature]", SCORE_SOURCE)
        self.assertIn("values[feature * rows + row]", SCORE_SOURCE)
        self.assertIn("row_stride", SCORE_SOURCE)

    def test_profile_target_is_importable(self) -> None:
        from experiments.custom_cuda import profile_target

        self.assertTrue(callable(profile_target.main))

    def test_nsight_csv_parser(self) -> None:
        text = """notice\n\"ID\",\"Process ID\",\"Metric Name\"\n\"0\",\"1\",\"Duration\"\n"""
        rows = parse_export(text)
        self.assertEqual(rows[0]["Metric Name"], "Duration")
        self.assertEqual(duration_seconds(2.5, "ms"), 0.0025)

    def test_matrix_expansion(self) -> None:
        config = {
            "schema_version": 1,
            "name": "test",
            "benchmark": {"warmups": 1, "trials": 2, "replications": 2, "seed": 1},
            "launch": [10],
            "transfer": {
                "bytes": [4096],
                "directions": ["h2d", "d2h"],
                "memory": ["pageable", "pinned"],
            },
            "fusion": [{"rows": 100, "features": 4, "blocks": [128]}],
        }
        validate_config(config)
        jobs = expand_jobs(config)
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({job_id(job) for job in jobs}), 12)

    def test_invalid_block_size(self) -> None:
        config = {
            "schema_version": 1,
            "name": "test",
            "benchmark": {"warmups": 1, "trials": 1, "replications": 1, "seed": 1},
            "fusion": [{"rows": 100, "features": 4, "blocks": [48]}],
        }
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_atomic_json_and_stable_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            atomic_json(path, {"b": 2, "a": 1})
            self.assertEqual(read_json(path), {"a": 1, "b": 2})
        self.assertEqual(stable_id("x", {"a": 1}), stable_id("x", {"a": 1}))

    def test_fusion_aggregation_speedup(self) -> None:
        environment = {"gpu_name": "test"}
        base = {
            "mode": "fusion",
            "condition": {"rows": 100, "features": 4},
            "compile_seconds": {"raw_module": 0.2, "cupy_partial_fusion": 0.1},
            "layout_conversion": {"aos_to_soa_wall_seconds": 0.01},
            "environment": environment,
            "records": [
                {
                    "implementation": "numpy",
                    "layout": "aos",
                    "block_size": None,
                    "resident_wall_seconds": 2.0,
                    "resident_device_seconds": None,
                    "one_shot_seconds": 2.0,
                    "quality": {"status": "reference"},
                    "kernel": None,
                },
                {
                    "implementation": "raw_soa_b256",
                    "layout": "soa",
                    "block_size": 256,
                    "resident_wall_seconds": 0.5,
                    "resident_device_seconds": 0.4,
                    "one_shot_seconds": 1.0,
                    "quality": {
                        "status": "pass",
                        "max_abs_score_error": 0.0,
                        "flag_mismatches": 0,
                    },
                    "kernel": {
                        "registers_per_thread": 10,
                        "static_shared_memory_bytes": 0,
                        "active_blocks_per_multiprocessor": 4,
                        "theoretical_occupancy": 0.5,
                    },
                },
            ],
        }
        rows = aggregate_fusion([base])
        gpu = next(row for row in rows if row["implementation"] == "raw_soa_b256")
        self.assertEqual(gpu["resident_speedup_vs_numpy"], 4.0)
        self.assertEqual(gpu["one_shot_speedup_vs_numpy"], 2.0)


if __name__ == "__main__":
    unittest.main()
