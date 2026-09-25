"""WP29's driver: one corpus, one contest, and every refusal written down instead of filled in.

The engine here is a fake on purpose. What these tests check is not that Laya answers — the real run does that, on the
worker, and the whole point of WP29 is that whatever it answers is recorded rather than arranged. What they check is
the driver's arithmetic of honesty:

* a choice above the declared threshold is fitted, evaluated on the corpus's own holdout, ranked against the person's
  alternative, and linked — and the outcome names the contest it was ranked in;
* a choice below it is an abstention: nothing is fitted under it, no second question is asked, and the link is refused
  `ABSTENTION_HAS_NO_OUTCOME` rather than quietly skipped;
* the person's alternative is recorded as a decision with `chosen_by: HUMAN`, from the declared option set, and is the
  same policy on every corpus so it cannot have been tuned to a holdout.
"""

import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

import numpy as np
import pytest

from feature_eng_m5phet import calibration_run, choose_regimes, corpora, regime_space

M5PHET_ROOT = Path(os.environ.get("M5PHET_EVALUATION_ROOT", "")) if os.environ.get("M5PHET_EVALUATION_ROOT") else None
REPORT = Path.home() / ".local" / "state" / "m5phet" / "wp09-quality-20260925" / "report_laya_zero_shot.json"

pytestmark = pytest.mark.skipif(
    M5PHET_ROOT is None or not (M5PHET_ROOT / "evaluation" / "compare_stages.py").is_file() or not REPORT.is_file(),
    reason="needs M5PHET_EVALUATION_ROOT pointing at an M5PHET checkout and the WP09 report the threshold is cited from")

THRESHOLD = {"min_confidence": 0.8, "abstention_source": str(REPORT)}


def write_corpus(path, *, rows=900, seed=11):
    """Three separated blobs on a regular clock: synthetic, so no index computed here reads as a finding."""
    rng = np.random.default_rng(seed)
    start = datetime(2019, 1, 1, tzinfo=timezone.utc)
    centres = ((0.0, 0.0), (9.0, 9.0), (18.0, 0.0))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([corpora.TIME_COLUMN, "alpha", "beta"])
        for index in range(rows):
            centre = centres[index % len(centres)]
            writer.writerow([(start + timedelta(minutes=index)).strftime(corpora.TIME_FORMAT),
                             f"{centre[0] + rng.normal(0, 0.4):.6f}", f"{centre[1] + rng.normal(0, 0.4):.6f}"])
    return {"id": "blobs", "path": str(path), "sha256": corpora.file_sha256(path), "rows": rows,
            "features": ["alpha", "beta"], "time_column": corpora.TIME_COLUMN}


class Engine:
    """Answers every declared question with one option at a declared confidence. Records what it was asked."""

    def __init__(self, answers, confidence):
        self.answers, self.confidence, self.asked = answers, confidence, []

    def execute_task(self, _prompt, envelope, _attachments, language="en"):
        self.asked.append(envelope)
        out = {}
        for name, question in envelope["questions"].items():
            options = [key for key, _label in question["options"]]
            chosen = self.answers.get(name, options[0])
            assert chosen in options, f"the fake engine may only answer inside the declared options {options}"
            rest = round((1.0 - self.confidence) / (len(options) - 1), 6)
            out[name] = {"status": "OK", "backend": "laya", "label": chosen, "probability_decimals": 4,
                         "uncalibrated_probabilities": {key: (self.confidence if key == chosen else rest)
                                                        for key in options}}
        return {"response": {"answers": out, "state_ref": "laya-checkpoint:test"}}


def run_one(tmp_path, engine, corpus):
    return calibration_run.run_corpus(corpus, engine, threshold=THRESHOLD, artifacts=tmp_path / "artifacts",
                                      record_dir=tmp_path / "decisions", outcome_dir=tmp_path / "outcomes",
                                      as_of="2026-09-25T00:00:00Z", evaluation_root=str(M5PHET_ROOT))


# --------------------------------------------------------------------------------------------------------------------
# a confident choice: fitted, ranked against the person's alternative, linked, and told which contest it was ranked in
# --------------------------------------------------------------------------------------------------------------------

def test_a_choice_above_the_threshold_is_fitted_ranked_and_linked_with_its_contest(tmp_path):
    corpus = write_corpus(tmp_path / "blobs.csv")
    # a different point from the person's k = 3, so the two stages really are two configurations and the ranks differ
    engine = Engine({choose_regimes.METHOD_DECISION: "gaussian_mixture",
                     choose_regimes.PARAMETER_DECISION: "k6"}, 0.94)
    result = run_one(tmp_path, engine, corpus)

    assert set(result["stages"]) == {calibration_run.LAYA_STAGE, calibration_run.HUMAN_STAGE}
    assert result["stages"][calibration_run.LAYA_STAGE]["method"] == "gaussian_mixture"
    assert result["stages"][calibration_run.HUMAN_STAGE]["method"] == calibration_run.HUMAN_METHOD

    rows = result["table"]["rows"]
    assert {row["metric"] for row in rows.values()} == {"silhouette"}
    assert {row["comparability"] for row in rows.values()} == {"COMPARABLE"}
    assert sorted(row["rank"] for row in rows.values()) == [1, 2]

    # four outcomes: two questions for each of the two stages, all of one contest, because it is one holdout
    assert len(result["outcomes"]) == 4
    assert len({entry["contest"] for entry in result["outcomes"]}) == 1
    assert {entry["chosen_by"] for entry in result["outcomes"]} == {"LAYA", "HUMAN"}
    assert not result["refusals"], result["refusals"]


def test_the_persons_alternative_is_the_same_declared_policy_whatever_laya_says(tmp_path):
    corpus = write_corpus(tmp_path / "blobs.csv")
    result = run_one(tmp_path, Engine({choose_regimes.METHOD_DECISION: "dbscan"}, 0.94), corpus)
    human = result["decisions"][calibration_run.HUMAN_STAGE]["questions"]

    assert human[choose_regimes.METHOD_DECISION]["chosen"] == calibration_run.HUMAN_METHOD == "kmeans"
    assert human[choose_regimes.METHOD_DECISION]["chosen_by"] == "HUMAN"
    assert human[choose_regimes.METHOD_DECISION]["probabilities"] == {}
    assert result["decisions"][calibration_run.HUMAN_STAGE]["parameter_point"] == \
        regime_space.point_key("kmeans", calibration_run.HUMAN_PARAMETERS)


# --------------------------------------------------------------------------------------------------------------------
# an abstention: recorded, never fitted, never linked, never asked around
# --------------------------------------------------------------------------------------------------------------------

def test_a_choice_below_the_threshold_abstains_asks_nothing_further_and_cannot_be_linked(tmp_path):
    corpus = write_corpus(tmp_path / "blobs.csv")
    engine = Engine({choose_regimes.METHOD_DECISION: "kmeans"}, 0.41)
    result = run_one(tmp_path, engine, corpus)

    method = result["decisions"][calibration_run.LAYA_STAGE]["questions"][choose_regimes.METHOD_DECISION]
    assert method["status"] == "REFUSED" and method["refusal"] == calibration_run.ABSTAINED
    assert method["abstention"]["top_probability"] == 0.41
    # the parameter question is never asked: a point under a method nobody chose is a configuration nobody chose
    assert len(engine.asked) == 1
    assert set(engine.asked[0]["questions"]) == {choose_regimes.METHOD_DECISION}

    # the person's stage still enters the table and is still ranked; Laya's stage does not exist
    assert set(result["stages"]) == {calibration_run.HUMAN_STAGE}
    assert result["table"]["rows"][calibration_run.HUMAN_STAGE]["rank"] == 1
    assert {entry["chosen_by"] for entry in result["outcomes"]} == {"HUMAN"}
    reasons = " ".join(item["why"] for item in result["refusals"])
    assert "ABSTENTION_HAS_NO_OUTCOME" in reasons and calibration_run.ABSTAINED in reasons


def test_a_finished_corpus_is_skipped_rather_than_asked_again(tmp_path):
    corpus = write_corpus(tmp_path / "blobs.csv")
    engine = Engine({choose_regimes.METHOD_DECISION: "kmeans"}, 0.41)
    run_one(tmp_path, engine, corpus)
    asked_once = len(engine.asked)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": corpora.SCHEMA, "corpora": [corpus]}), encoding="utf-8")
    summary = calibration_run.run(manifest, artifacts=tmp_path / "artifacts", record_dir=tmp_path / "decisions",
                                  outcome_dir=tmp_path / "outcomes", threshold=THRESHOLD, engine=engine,
                                  as_of="2026-09-25T00:00:00Z", evaluation_root=str(M5PHET_ROOT))
    assert summary["corpora_skipped_already_done"] == ["blobs"] and summary["corpora_run"] == []
    assert len(engine.asked) == asked_once
