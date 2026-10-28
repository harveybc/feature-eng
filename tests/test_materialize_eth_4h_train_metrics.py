"""Mechanical evidence that ETH metrics are isolated to declared TRAIN rows."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location("eth_metrics", Path(__file__).resolve().parents[1] / "tools/materialize_eth_4h_train_metrics.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def contract(train_rows=96):
    features = [f"feature_{index:02d}" for index in range(83)]
    return {"schema": "selected_feature_manifest.v1", "feature_count": 83, "features": features,
            "resource": {"asset": "ETHUSDT spot", "bar": "4h", "timestamp_column": "DATE_TIME",
                         "availability_class": "DEVELOPMENT", "point_in_time": {"meaning": "uncertified"}},
            "task": {"decision_time": "bar-close convention, uncertified"},
            "split": {"train": {"rows": [0, train_rows]}}}


def write_source(path, heldout_value):
    rng = np.random.default_rng(4)
    c = contract()
    header = ["DATE_TIME", "CLOSE", *c["features"]]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in range(96):
            values = [f"2020-01-{1 + row // 6:02d} {4 * (row % 6):02d}:00:00", f"{100 + row:.8f}"]
            values += [f"{np.sin(row / 7 + col):.8f}" for col in range(83)]
            writer.writerow(values)
        for row in range(12):
            writer.writerow(["2099-01-01 00:00:00", heldout_value, *([heldout_value] * 83)])


class MaterializationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.contract_path = self.root / "contract.json"
        self.contract_path.write_text(json.dumps(contract()))

    def tearDown(self):
        self.tmp.cleanup()

    def test_validation_and_test_mutations_do_not_change_rows_or_digest(self):
        first, second = self.root / "first.csv", self.root / "second.csv"
        write_source(first, "999999")
        write_source(second, "POISON_HELDOUT")
        one = M.run(self.contract_path, first, Path(__file__).resolve().parents[1] / "agent_out_test_one", trailing_window=48)
        two = M.run(self.contract_path, second, Path(__file__).resolve().parents[1] / "agent_out_test_two", trailing_window=48)
        try:
            self.assertEqual(one["rows"], two["rows"])
            self.assertEqual(one["receipt"]["data_digest"], two["receipt"]["data_digest"])
            self.assertEqual(one["rows"][0]["fit_scope"]["row_range"], [0, 96])
        finally:
            import shutil
            shutil.rmtree(Path(__file__).resolve().parents[1] / "agent_out_test_one")
            shutil.rmtree(Path(__file__).resolve().parents[1] / "agent_out_test_two")

    def test_rows_are_typed_complete_and_contract_bound(self):
        source = self.root / "source.csv"
        write_source(source, "0")
        output = Path(__file__).resolve().parents[1] / "agent_out_test_complete"
        result = M.run(self.contract_path, source, output, trailing_window=48)
        try:
            self.assertEqual(len(result["rows"]), 83)
            row = result["rows"][0]
            self.assertEqual(row["type"], "feature_train_metric_row.v1")
            self.assertTrue(row["row_id"].startswith("eth_4h:feature:"))
            self.assertEqual(set(row["acf"]), {"1", "6", "24"})
            self.assertEqual(set(row["target_association"]), {"6", "12", "18", "24", "30", "36"})
            self.assertEqual(row["availability"]["resource_availability_class"], "DEVELOPMENT")
            self.assertTrue(row["data_digest"])
            self.assertTrue(row["parameter_digest"])
            self.assertTrue((output / "metrics.jsonl").is_file())
            self.assertTrue((output / "metrics.csv").is_file())
        finally:
            import shutil
            shutil.rmtree(output)

    def test_contract_refusal_is_before_input_open(self):
        bad = contract()
        bad["feature_count"] = 82
        self.contract_path.write_text(json.dumps(bad))
        with self.assertRaises(ValueError):
            M.run(self.contract_path, self.root / "not-opened.csv", Path(__file__).resolve().parents[1] / "agent_out_test_bad")

    def test_committed_inventory_is_the_frozen_83_feature_contract(self):
        frozen = Path(__file__).resolve().parents[1] / "docs/feature_metrics/laneB/SELECTED_FEATURE_MANIFEST.eth_4h.v1.FROZEN_DEVELOPMENT.json"
        loaded = json.loads(frozen.read_text())
        M.validate_contract(loaded)
        self.assertEqual(loaded["feature_count"], 83)
        self.assertEqual(loaded["split"]["train"]["rows"], [0, 13699])


if __name__ == "__main__":
    unittest.main()
