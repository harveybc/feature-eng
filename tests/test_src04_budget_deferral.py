"""SRC-4: feature-budget overflow yields a named deferred candidate, never truncation."""
import unittest

from tests._src_fixtures import L


def cand(name, channels, family="technical"):
    return {"candidate_id": name, "family": family, "channels": channels, "bytes": 1000 * channels}


class BudgetDeferral(unittest.TestCase):
    def test_overflow_defers_whole_candidates_by_name(self):
        cands = [cand("a", 30), cand("b", 50), cand("c", 40)]
        plan = L.plan_active_set(cands, budget={"channels": 80})
        sel = {c["candidate_id"]: c for c in plan["active"]}
        dfr = {c["candidate_id"]: c for c in plan["deferred"]}
        self.assertEqual(set(sel) | set(dfr), {"a", "b", "c"})        # nothing leaves the ledger
        for c in plan["active"]:
            self.assertEqual(c["channels"], next(x for x in cands if x["candidate_id"] == c["candidate_id"])["channels"])
        self.assertLessEqual(sum(c["channels"] for c in plan["active"]), 80)
        for c in plan["deferred"]:
            self.assertTrue(c["reason"].startswith("BUDGET_OVERFLOW"))
            self.assertIn("channels", c["reason"])

    def test_candidate_larger_than_budget_is_deferred_not_cut(self):
        plan = L.plan_active_set([cand("huge", 500)], budget={"channels": 100})
        self.assertEqual(plan["active"], [])
        self.assertEqual(plan["deferred"][0]["candidate_id"], "huge")
        self.assertEqual(plan["deferred"][0]["channels"], 500)


if __name__ == "__main__":
    unittest.main()
