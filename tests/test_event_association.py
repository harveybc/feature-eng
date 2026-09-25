"""Rung 1 on planted data: the naive picture must show the sign that was planted, and claim nothing more.

The same synthetic market as the builder's tests, kept deliberately small here: event A moves the LEVEL in proportion
to its surprise, event B moves the VOLATILITY in proportion to its surprise and leaves the level where it was. So the
association job must find a positive `log_return` relationship for A, a positive `realized_vol` relationship for B,
and an `n` that equals the rows it was given -- no more, because a correlation computed over more rows than exist is
the commonest way a table lies.

The other half of the file is about the table's honesty rather than its arithmetic: every cell carries its own `n`,
the tercile cut points are reported so anybody can recompute the bins, a group too small for a correlation says
`TOO_FEW_EVENTS` instead of showing a number, and the document states in its own words that nothing in it is causal.
"""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from feature_eng_m5phet import association, events

START = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
LEVEL = 0.002                       # event A's planted level response per unit of raw surprise, over 30 minutes
BURST = 0.0004                      # event B's planted zigzag amplitude is BURST * (3 + s): rising with the surprise
RAMP = 30
FIRST = 600
SPACING = 480
N_EVENTS = 40
TAIL = 300
BASE = float(np.log(1.1))


def plan(n=N_EVENTS):
    values = np.round(np.random.default_rng(20260925).uniform(-2.0, 2.0, n), 4)
    return [("A" if i % 2 == 0 else "B", FIRST + i * SPACING, float(s)) for i, s in enumerate(values)]


def write_inputs(tmp_path, releases, *, name=""):
    minutes = FIRST + (len(releases) - 1) * SPACING + TAIL
    path = np.full(minutes, BASE, dtype=np.float64)
    for kind, minute, s in releases:
        if kind == "A":
            for offset in range(1, minutes - minute):
                path[minute + offset] += LEVEL * s * min(1.0, offset / RAMP)
        else:
            amplitude = BURST * (3.0 + s)
            for offset in range(1, RAMP + 1):
                if minute + offset < minutes and offset % 2 == 1:
                    path[minute + offset] += amplitude
    bars = tmp_path / f"bars{name}.csv"
    bars.write_text("timestamp,close\n" + "\n".join(
        f"{(START + timedelta(minutes=i)).isoformat()},{float(np.exp(path[i]))!r}" for i in range(minutes)) + "\n",
        encoding="utf-8")
    header = ("event_type,event_time,published_at,consensus_published_at,actual,consensus,previous,"
              "historical_availability")
    lines = [header]
    for kind, minute, s in releases:
        moment = START + timedelta(minutes=minute)
        lines.append(",".join([kind, moment.isoformat(), moment.isoformat(),
                               (moment - timedelta(days=1)).isoformat(),
                               repr(100.0 + s), "100.0", "100.0", "KNOWN"]))
    calendar = tmp_path / f"calendar{name}.csv"
    calendar.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return bars, calendar


def rows_document(tmp_path, releases=None, *, name="", **kwargs):
    bars, calendar = write_inputs(tmp_path, releases or plan(), name=name)
    return events.build(str(bars), str(calendar), events.CalendarMapping(), **kwargs)


def group_of(summary, event_type, horizon):
    return next(g for g in summary["groups"]
                if g["event_type"] == event_type and g["horizon_minutes"] == horizon)


# --------------------------------------------------------------------------------- the planted sign is recovered

def test_the_planted_level_response_shows_as_a_positive_correlation_with_the_right_n(tmp_path):
    document = rows_document(tmp_path)
    summary = association.summarise(document)
    assert summary["schema"] == association.SCHEMA
    block = group_of(summary, "A", 30)["outcomes"]["log_return"]
    expected_n = len([r for r in document["rows"] if r["event_type"] == "A" and r["horizon_minutes"] == 30])
    assert block["n"] == expected_n == block["pearson"]["n"] == block["spearman"]["n"]
    assert block["pearson"]["value"] > 0.8 and block["spearman"]["value"] > 0.8
    assert block["pearson"]["reason"] is None


def test_the_planted_volatility_burst_shows_as_a_positive_rank_correlation(tmp_path):
    summary = association.summarise(rows_document(tmp_path))
    block = group_of(summary, "B", 30)["outcomes"]["realized_vol"]
    assert block["spearman"]["value"] > 0.8, "the planted burst rising with the surprise is not visible at rung 1"
    assert block["pearson"]["value"] > 0.5


def test_the_level_event_leaves_the_volatility_outcome_without_the_level_sign(tmp_path):
    """Event B moves volatility and not the level: its log_return correlation must not carry the level story."""
    summary = association.summarise(rows_document(tmp_path))
    block = group_of(summary, "B", 30)["outcomes"]["log_return"]
    assert block["pearson"]["value"] is None or abs(block["pearson"]["value"]) < 0.5


def test_the_sign_table_separates_the_negative_and_positive_surprises_and_counts_both(tmp_path):
    summary = association.summarise(rows_document(tmp_path))
    block = group_of(summary, "A", 30)["outcomes"]["log_return"]
    table = block["by_surprise_sign"]
    assert sum(cell["n"] for cell in table.values()) == block["n"]
    assert table["negative"]["n"] > 0 and table["positive"]["n"] > 0
    assert table["negative"]["mean"] < 0 < table["positive"]["mean"]
    assert table["negative"]["median"] < 0 < table["positive"]["median"]


def test_the_tercile_table_reports_its_cuts_and_its_bins_add_up(tmp_path):
    summary = association.summarise(rows_document(tmp_path))
    block = group_of(summary, "A", 30)["outcomes"]["log_return"]
    table = block["by_surprise_tercile"]
    assert table["cuts"]["lower"] < table["cuts"]["upper"]
    assert sum(cell["n"] for cell in table["bins"].values()) == block["n"]
    assert all(cell["n"] > 0 for cell in table["bins"].values())
    assert table["bins"]["T1"]["mean"] < table["bins"]["T3"]["mean"]
    # the rule must be enough to rebuild the bins by hand: the two cut points, the tie convention, the method
    assert repr(table["cuts"]["lower"]) in table["rule"] and repr(table["cuts"]["upper"]) in table["rule"]
    assert "'linear'" in table["rule"] and "T1 is value <=" in table["rule"]


def test_the_pre_event_volatility_table_is_built_and_counted_separately(tmp_path):
    summary = association.summarise(rows_document(tmp_path))
    block = group_of(summary, "A", 30)["outcomes"]["log_return"]
    assert block["pre_event_volatility_n"] == block["n"]
    table = block["by_pre_event_volatility_tercile"]
    # the planted base is flat, so every pre-event hour is identical and the cuts collapse onto one value: the table
    # must show that as bins with n, not as three evenly filled bins invented by a rebalancing rule
    assert table["cuts"] is not None
    assert sum(cell["n"] for cell in table["bins"].values()) == block["n"]


# ------------------------------------------------------------------------------------------ what it will not claim

def test_a_group_with_too_few_events_refuses_the_correlation_by_name(tmp_path):
    document = rows_document(tmp_path)
    document["rows"] = [row for row in document["rows"]
                        if row["event_type"] == "A" and row["horizon_minutes"] == 30][:2]
    summary = association.summarise(document)
    block = group_of(summary, "A", 30)["outcomes"]["log_return"]
    assert block["pearson"]["value"] is None and "TOO_FEW_EVENTS" in block["pearson"]["reason"]
    assert block["spearman"]["value"] is None and "TOO_FEW_EVENTS" in block["spearman"]["reason"]
    assert block["by_surprise_tercile"]["cuts"] is None
    assert "TOO_FEW_EVENTS" in block["by_surprise_tercile"]["reason"]


def test_a_column_that_does_not_vary_refuses_the_correlation_by_name(tmp_path):
    document = rows_document(tmp_path)
    for row in document["rows"]:
        row["log_return"] = 0.5
    block = association.summarise(document)["groups"][0]["outcomes"]["log_return"]
    assert block["pearson"]["value"] is None and "ZERO_VARIANCE" in block["pearson"]["reason"]


def test_the_document_says_in_its_own_words_that_nothing_in_it_is_causal(tmp_path):
    summary = association.summarise(rows_document(tmp_path))
    assert "RUNG 1, ASSOCIATION ONLY" in summary["reading"]
    assert "is an effect" in summary["reading"] and "NAIVE" in summary["reading"]
    assert summary["fitted"].startswith("NOTHING")
    assert summary["minimum_events_for_a_correlation"] == association.MIN_CORRELATION_N


def test_it_reads_only_the_rows_schema_it_declares(tmp_path):
    with pytest.raises(association.AssociationRefusal) as refusal:
        association.summarise({"schema": "something.else.v1", "rows": []})
    assert refusal.value.code == "WRONG_SCHEMA"


def test_the_summary_carries_the_identity_of_the_inputs_the_rows_came_from(tmp_path):
    document = rows_document(tmp_path)
    summary = association.summarise(document)
    assert summary["rows_document"]["bars_sha256"] == document["bars"]["sha256"]
    assert summary["rows_document"]["calendar_sha256"] == document["calendar"]["sha256"]
    assert summary["rows_document"]["parameters"] == document["parameters"]
    assert summary["counts"]["event_types"] == ["A", "B"]


def test_the_same_rows_summarise_to_the_same_bytes(tmp_path):
    document = rows_document(tmp_path, plan(20))
    first = json.dumps(association.summarise(document), sort_keys=False)
    second = json.dumps(association.summarise(document), sort_keys=False)
    assert first == second


def test_the_cli_writes_the_summary_and_refuses_a_missing_file(tmp_path, capsys):
    document = rows_document(tmp_path, plan(20))
    rows_path = tmp_path / "event_rows.json"
    rows_path.write_text(json.dumps(document), encoding="utf-8")
    out = tmp_path / "association.json"
    assert association.main(["--rows", str(rows_path), "--out", str(out)]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["schema"] == association.SCHEMA and summary["groups"]
    assert association.main(["--rows", str(tmp_path / "absent.json"), "--out", str(out)]) == 2
    assert "REFUSED NO_SUCH_FILE" in capsys.readouterr().err


def test_the_summary_carries_the_declared_clock_and_repeats_its_caveat(tmp_path):
    """An association table read on its own must say what the instants under it were, or it will be read as measured."""
    releases = plan(20)
    bars, _ = write_inputs(tmp_path, releases)
    header = "event_type,event_time,actual,consensus,previous,historical_availability"
    lines = [header]
    for kind, minute, s in releases:
        moment = START + timedelta(minutes=minute)
        lines.append(",".join([kind, moment.isoformat(), repr(100.0 + s), "100.0", "100.0", "KNOWN"]))
    calendar = tmp_path / "scheduled.csv"
    calendar.write_text("\n".join(lines) + "\n", encoding="utf-8")
    document = events.build(str(bars), str(calendar), events.CalendarMapping(), publication_clock="scheduled")
    summary = association.summarise(document)
    assert summary["provenance"] == "DEVELOPMENT_ASSUMED_CLOCK"
    assert summary["publication_clock"] == document["publication_clock"]
    assert summary["reading"].startswith("PROVENANCE DEVELOPMENT_ASSUMED_CLOCK")
    assert "not identified until a publication clock exists" in summary["reading"]
    assert "RUNG 1, ASSOCIATION ONLY" in summary["reading"]


def test_a_summary_of_observed_clock_rows_says_so_too(tmp_path):
    summary = association.summarise(rows_document(tmp_path, plan(20)))
    assert summary["provenance"] == "DEVELOPMENT_OBSERVED_CLOCK"
    assert summary["publication_clock"]["mode"] == "OBSERVED_PUBLICATION_CLOCK"
