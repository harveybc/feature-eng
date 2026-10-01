"""SRC-3: equivalent cached recipes reuse; different vintages, assets or feeds never do."""
import tempfile
import unittest
from pathlib import Path

from tests._src_fixtures import L

BASE = dict(parent_sha256="a" * 64, transform="fracdiff", version="1", params={"d": 0.4},
            fold_state="inner_1:[0,7474)", availability="WINDOW_END+1h", units="log_price")


class RecipeIdentity(unittest.TestCase):
    def test_equivalent_recipe_reuses_and_param_order_does_not_matter(self):
        with tempfile.TemporaryDirectory() as t:
            c = L.RecipeCache(Path(t))
            c.put(L.recipe_key(**BASE), {"channels": ["fd_0.4"]})
            same = dict(BASE, params={"d": 0.4})
            self.assertEqual(c.get(L.recipe_key(**same)), {"channels": ["fd_0.4"]})

    def test_vintage_asset_feed_fold_or_units_change_misses(self):
        with tempfile.TemporaryDirectory() as t:
            c = L.RecipeCache(Path(t))
            c.put(L.recipe_key(**BASE), {"channels": ["fd_0.4"]})
            for change in ({"parent_sha256": "b" * 64}, {"fold_state": "inner_2:[0,9529)"},
                           {"availability": "UNKNOWN"}, {"units": "price"}, {"version": "2"}, {"params": {"d": 0.5}}):
                self.assertIsNone(c.get(L.recipe_key(**dict(BASE, **change))), change)

    def test_name_match_is_not_identity(self):
        a = L.recipe_key(**dict(BASE, parent_sha256="c" * 64))
        b = L.recipe_key(**dict(BASE, parent_sha256="d" * 64))
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
