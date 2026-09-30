import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("inventory_audit", ROOT / "tools/audit_feature_inventory.py")
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


class InventoryTests(unittest.TestCase):
    def test_exact_identity_only(self):
        self.assertEqual(A.matching_id({"kind": "physical_appearance", "id": "app_1"}),
                         "financial_data.census_appearance.app_1")
        self.assertEqual(A.matching_id({"kind": "dataset", "id": "weather"}), "weather")

    def test_completed_does_not_mean_covered(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            index = {"common_rows": [{"bank": "financial", "kind": "physical_appearance", "id": "app_1"},
                                      {"bank": "public", "kind": "dataset", "id": "absent"}]}
            inventory = {"datasets": [{"dataset_id": "basic", "variables": [{"name": "x"}]}]}
            raw = json.dumps({"row": {"dataset_id": "financial_data.census_appearance.app_1",
                                       "partition": "train", "variable_id": "x", "metric": "adf_pvalue",
                                       "status": "FAILED"}}).encode() + b"\n"
            (root / "profile.jsonl").write_bytes(raw)
            receipt = {"datasets": [{"dataset_id": "financial_data.census_appearance.app_1",
                                      "bank": "FINANCIAL", "file": "profile.jsonl", "status": "COMPLETED",
                                      "sha256": hashlib.sha256(raw).hexdigest(), "numeric_variables": 1},
                                     {"dataset_id": "missing", "bank": "PUBLIC", "file": "missing.jsonl",
                                      "status": "COMPLETED", "numeric_variables": 2}]}
            for name, doc in (("index", index), ("inventory", inventory), ("receipt", receipt)):
                (root / (name + ".json")).write_text(json.dumps(doc))
            result = A.audit(root / "index.json", root / "inventory.json", [root / "receipt.json"], root / "out")
            self.assertEqual(result["artifact_verification"], {"HASH_VERIFIED": 1, "MISSING_ARTIFACT": 1})
            self.assertEqual(result["inventory_detailed_train_certified_datasets"], 0)
            table = (root / "out/existing_train_variable_coverage.csv").read_text()
            self.assertIn("FAILED", table)
            self.assertIn("False", table)
            capped = A.audit(root / "index.json", root / "inventory.json", [root / "receipt.json"],
                             root / "capped", artifact_byte_budget=0)
            self.assertEqual(capped["artifact_bytes_inspected"], 0)
            self.assertEqual(capped["artifact_verification"]["NOT_INSPECTED_RESOURCE_CAP"], 1)


if __name__ == "__main__":
    unittest.main()
