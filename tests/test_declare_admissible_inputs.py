"""Declaration digest and exclusion carry-over, on a synthetic wide profile. Local only."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from tests.test_profile_train_wide import W, manifest, write_fixture

SPEC = importlib.util.spec_from_file_location("decl", Path(__file__).resolve().parents[1] / "tools/declare_admissible_inputs.py")
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class DeclarationTests(unittest.TestCase):
    def test_declaration_carries_exclusions_and_self_digest(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            path = write_fixture(root)
            mp = root / "m.json"
            mp.write_text(json.dumps(manifest(path)))
            W.run(mp, root, root / "out")
            doc = D.declare(root / "out/profile.json")
            self.assertEqual(doc["all_admissible_control"], ["a", "b", "d", "OT"])
            self.assertEqual([b["feature"] for b in doc["branches_one_feature_each"]], doc["all_admissible_control"])
            ex = {e["column"]: e["reason"] for e in doc["exclusions"]}
            self.assertEqual(ex["c"], "CONSTANT_IN_TRAIN")
            self.assertIn("date", ex)
            self.assertEqual(doc["declaration_sha256"], D.canonical_sha(doc))
            doc["branches_one_feature_each"].pop()
            self.assertNotEqual(doc["declaration_sha256"], D.canonical_sha(doc))


if __name__ == "__main__":
    unittest.main()
