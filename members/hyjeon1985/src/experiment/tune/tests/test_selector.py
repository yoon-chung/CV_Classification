from __future__ import annotations

import unittest

from experiment.tune.selector import rank_candidates


class SelectorTests(unittest.TestCase):
    def _cfg(self) -> dict:
        return {
            "tune": {
                "selector": {
                    "std_weight": 0.2,
                    "overfit_gap_threshold": 0.03,
                    "overfit_weight": 1.0,
                    "fail_weight": 0.5,
                }
            }
        }

    def test_deterministic_order(self) -> None:
        rows = [
            {"candidate_hash": "aaa", "status": "success", "macro_f1": 0.9, "val_loss": 0.3, "overfit_gap": 0.02},
            {"candidate_hash": "bbb", "status": "success", "macro_f1": 0.91, "val_loss": 0.32, "overfit_gap": 0.02},
        ]
        out1 = rank_candidates(rows, self._cfg())
        out2 = rank_candidates(rows, self._cfg())
        self.assertEqual(out1, out2)

    def test_fail_rate_penalty(self) -> None:
        rows = [
            {"candidate_hash": "good", "status": "success", "macro_f1": 0.9, "val_loss": 0.3, "overfit_gap": 0.01},
            {"candidate_hash": "good", "status": "success", "macro_f1": 0.9, "val_loss": 0.31, "overfit_gap": 0.01},
            {"candidate_hash": "bad", "status": "success", "macro_f1": 0.9, "val_loss": 0.3, "overfit_gap": 0.01},
            {"candidate_hash": "bad", "status": "failed", "macro_f1": "", "val_loss": "", "overfit_gap": ""},
        ]
        ranked = rank_candidates(rows, self._cfg())
        self.assertEqual(ranked[0]["candidate_hash"], "good")
        self.assertEqual(ranked[1]["candidate_hash"], "bad")


if __name__ == "__main__":
    unittest.main()
