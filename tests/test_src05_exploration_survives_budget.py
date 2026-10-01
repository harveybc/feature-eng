"""SRC-5: family exploration and group reintroduction survive a resource-limited batch."""
import unittest

from tests._src_fixtures import L


def cands():
    out = []
    for fam in ("raw", "returns", "technical", "wavelet_proxy", "hilbert", "multitaper", "regime", "calendar"):
        for k in range(3):
            out.append({"candidate_id": f"{fam}_{k}", "family": fam, "channels": 10, "score": 10 - k if fam == "technical" else 1})
    return out


class ExplorationSurvivesBudget(unittest.TestCase):
    def test_every_family_is_explored_across_limited_batches(self):
        state = L.ExplorationState(seed=3)
        seen = set()
        for _ in range(4):
            batch = L.plan_batch(cands(), state, budget={"channels": 30}, exploration_quota=2)
            self.assertLessEqual(sum(c["channels"] for c in batch["active"]), 30)
            seen |= {c["family"] for c in batch["active"]}
            state = batch["state"]
        self.assertEqual(seen, {"raw", "returns", "technical", "wavelet_proxy", "hilbert", "multitaper", "regime", "calendar"})

    def test_untested_families_are_reported_each_batch(self):
        batch = L.plan_batch(cands(), L.ExplorationState(seed=3), budget={"channels": 30}, exploration_quota=2)
        tested = {c["family"] for c in batch["active"]}
        self.assertEqual(set(batch["families_untested"]), {f for f in {c["family"] for c in cands()}} - tested)

    def test_deferred_group_is_reintroduced(self):
        state = L.ExplorationState(seed=3)
        state.reintroduce.append({"candidate_id": "group:x1+x2", "family": "synergy_group", "channels": 10,
                                  "reason": "SYNERGY_PAIR deferred by budget"})
        batch = L.plan_batch(cands(), state, budget={"channels": 30}, exploration_quota=1)
        self.assertIn("group:x1+x2", {c["candidate_id"] for c in batch["active"]})


if __name__ == "__main__":
    unittest.main()
