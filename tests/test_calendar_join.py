"""The join that gives a consensus row an OBSERVED publication instant, and everything it must refuse to join.

A join is the one place in this chain where a number can be attached to an instant nobody measured for it. Every
test here is about that: a match happens only when the economy, the release name and the calendar date all agree;
a synonym is declared and listed; an ambiguity on either side is dropped rather than resolved by position; a
release that sits on the other side of a date boundary is not reached for; and every row that did not join is
counted by name, exactly.

Nothing here reads a file outside tmp_path, nothing is fitted, and no clock is invented.
"""

import csv
import json

import pytest

from feature_eng_m5phet import calendar_join


ARCHIVE_COLUMNS = list(calendar_join.DEFAULT_ARCHIVE_COLUMNS)


def archive_row(date, time, country, description, actual, forecast, previous="1.0"):
    return [date, time, country, "Moderate Volatility Expected", description, "", "% ", actual, forecast, previous]


def write_archive(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return path


ANNOUNCEMENT_HEADER = ["currency", "indicator", "date", "val", "announcement_datetime_utc"]


def write_announcements(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(ANNOUNCEMENT_HEADER)
        writer.writerows(rows)
    return path


def run(tmp_path, archive_rows, announcement_rows, **kwargs):
    archive = write_archive(tmp_path / "archive.csv", archive_rows)
    announcements = write_announcements(tmp_path / "announcements.csv", announcement_rows)
    return calendar_join.join(str(archive), str(announcements), **kwargs)


# ------------------------------------------------------------------------------------------------ what does match

def test_a_release_both_archives_spell_the_same_way_matches_exactly_and_takes_the_observed_instant(tmp_path):
    document = run(
        tmp_path,
        [archive_row("2025/03/06", "8:30:00", "United States", "Initial Jobless Claims", "220", "215")],
        [["USD", "initial_jobless_claims", "2025-03-01", "220", "2025-03-06 13:30:00+00:00"]])
    assert document["counts"]["joined"] == 1
    assert document["counts"]["join_rate"] == 1.0
    row = document["rows"][0]
    assert row["published_at"] == "2025-03-06T13:30:00+00:00"       # the OBSERVED instant, not the archive's 8:30
    assert row["event_time"] == "2025-03-06T08:30:00"               # the scheduled one travels beside it
    assert row["match_method"] == "EXACT"
    assert row["synonym"] == ""
    assert (row["actual"], row["consensus"]) == (220.0, 215.0)      # the consensus archive is the source of both
    assert document["publication_clock"]["mode"] == "OBSERVED_ACTUAL_PUBLICATION"
    assert document["publication_clock"]["consensus_clock"] == "ASSUMED_BEFORE_RELEASE"
    assert document["provenance"] == "DEVELOPMENT_OBSERVED_CLOCK"


def test_a_release_the_two_archives_spell_differently_matches_only_through_a_synonym_that_is_listed(tmp_path):
    rows = [archive_row("2025/04/04", "8:30:00", "United States", "Nonfarm Payrolls", "228", "140")]
    announcements = [["USD", "non_farm_payrolls", "2025-03-31", "228", "2025-04-04 12:30:00+00:00"]]
    document = run(tmp_path, rows, announcements)
    assert document["counts"]["joined"] == 1
    assert document["rows"][0]["match_method"] == "SYNONYM"
    assert document["rows"][0]["synonym"] == "nonfarm payrolls"
    assert document["matching"]["synonyms_used"] == {"nonfarm payrolls": 1}
    # every declared synonym is in the output, used or not, so the table can be audited from the artifact alone
    assert "nonfarm payrolls" in document["matching"]["synonyms_declared"]
    assert "cpi" in document["matching"]["synonyms_declared_but_unused"]

    # and with the synonym table emptied, the same pair does not match: the mapping is declared, never inferred
    bare = run(tmp_path, rows, announcements, synonyms={})
    assert bare["counts"]["joined"] == 0
    assert bare["excluded"]["counts"]["NO_OBSERVED_ANNOUNCEMENT"] == 1


def test_the_normalisation_is_case_and_punctuation_only(tmp_path):
    document = run(
        tmp_path,
        [archive_row("2025/03/06", "8:30:00", "United States", "  INITIAL-JOBLESS, CLAIMS ", "220", "215")],
        [["USD", "Initial Jobless Claims", "2025-03-01", "220", "2025-03-06 13:30:00+00:00"]])
    assert document["counts"]["joined"] == 1
    assert calendar_join.normalise("  INITIAL-JOBLESS, CLAIMS ") == "initial jobless claims"


# ---------------------------------------------------------------------------------------------- what is refused

def test_an_announcement_that_two_consensus_rows_answer_to_is_dropped_and_counted(tmp_path):
    document = run(
        tmp_path,
        [archive_row("2025/03/06", "8:30:00", "United States", "Initial Jobless Claims", "220", "215"),
         archive_row("2025/03/06", "8:30:00", "United States", "Initial Jobless Claims", "221", "216")],
        [["USD", "initial_jobless_claims", "2025-03-01", "220", "2025-03-06 13:30:00+00:00"]])
    assert document["counts"]["joined"] == 0
    assert document["excluded"]["counts"]["AMBIGUOUS_MATCH"] == 2       # both sides dropped, neither chosen
    assert document["excluded"]["counts"]["NO_OBSERVED_ANNOUNCEMENT"] == 0


def test_a_consensus_row_that_two_announcements_answer_to_is_dropped_and_counted(tmp_path):
    document = run(
        tmp_path,
        [archive_row("2025/03/06", "8:30:00", "United States", "Initial Jobless Claims", "220", "215")],
        [["USD", "initial_jobless_claims", "2025-03-01", "220", "2025-03-06 13:30:00+00:00"],
         ["USD", "initial_jobless_claims", "2025-02-22", "218", "2025-03-06 18:00:00+00:00"]])
    assert document["counts"]["joined"] == 0
    assert document["excluded"]["counts"]["AMBIGUOUS_MATCH"] == 1


def test_a_match_across_a_date_boundary_is_refused_rather_than_reached_for(tmp_path):
    # the announcement is one day later than the consensus row's date: a real release, but not this row's release
    document = run(
        tmp_path,
        [archive_row("2025/03/06", "23:30:00", "United States", "Initial Jobless Claims", "220", "215")],
        [["USD", "initial_jobless_claims", "2025-03-01", "220", "2025-03-07 02:30:00+00:00"]])
    assert document["counts"]["joined"] == 0
    assert document["excluded"]["counts"]["NO_OBSERVED_ANNOUNCEMENT"] == 1
    assert document["excluded"]["no_observed_announcement_by_reason"]["NOTHING_ANNOUNCED_ON_THAT_DATE"] == 1

    # declaring the zone the archive's wall clock is in moves its instant across the boundary, and THEN it matches:
    # the boundary was a fact about the clocks, and it is fixed by declaring one, never by widening the key
    with_zone = run(
        tmp_path,
        [archive_row("2025/03/06", "23:30:00", "United States", "Initial Jobless Claims", "220", "215")],
        [["USD", "initial_jobless_claims", "2025-03-01", "220", "2025-03-07 02:30:00+00:00"]],
        archive_timezone="Etc/GMT+3")
    assert with_zone["counts"]["joined"] == 1
    assert with_zone["matching"]["archive_timezone"] == "Etc/GMT+3"


def test_a_country_with_no_exact_currency_is_dropped_rather_than_joined_to_a_neighbour(tmp_path):
    document = run(
        tmp_path,
        [archive_row("2025/03/06", "8:30:00", "Germany", "Industrial Production", "1.0", "0.5")],
        [["EUR", "industrial_production", "2025-02-28", "1.0", "2025-03-06 07:00:00+00:00"]])
    assert document["counts"]["joined"] == 0
    assert document["excluded"]["no_observed_announcement_by_reason"]["COUNTRY_NOT_IN_TABLE"] == 1


def test_a_consensus_row_with_no_announcement_of_that_release_name_is_counted_by_its_own_reason(tmp_path):
    document = run(
        tmp_path,
        [archive_row("2025/03/06", "10:30:00", "United States", "Crude Oil Inventories", "3.6", "1.2")],
        [["USD", "initial_jobless_claims", "2025-03-01", "220", "2025-03-06 13:30:00+00:00"]])
    assert document["counts"]["joined"] == 0
    reasons = document["excluded"]["no_observed_announcement_by_reason"]
    assert reasons["RELEASE_NAME_NOT_IN_THE_ANNOUNCEMENT_ARCHIVE"] == 1
    assert reasons["NOTHING_ANNOUNCED_ON_THAT_DATE"] == 0


def test_a_row_without_a_consensus_is_not_a_join_failure_and_is_counted_apart(tmp_path):
    document = run(
        tmp_path,
        [archive_row("2025/03/06", "8:30:00", "United States", "Initial Jobless Claims", "220", ""),
         archive_row("2025/03/06", "8:30:00", "United States", "PPI", "", "")],
        [["USD", "initial_jobless_claims", "2025-03-01", "220", "2025-03-06 13:30:00+00:00"]])
    assert document["excluded"]["counts"]["NO_CONSENSUS"] == 1       # has an actual, no consensus
    assert document["excluded"]["counts"]["NO_OBSERVED_ANNOUNCEMENT"] == 0
    assert document["counts"]["consensus_rows_with_an_actual_and_a_consensus"] == 0
    assert document["counts"]["joined"] == 0


def test_an_announcement_whose_instant_carries_no_zone_is_refused_by_name(tmp_path):
    with pytest.raises(calendar_join.JoinRefusal) as refusal:
        run(tmp_path,
            [archive_row("2025/03/06", "8:30:00", "United States", "Initial Jobless Claims", "220", "215")],
            [["USD", "initial_jobless_claims", "2025-03-01", "220", "2025-03-06 13:30:00"]])
    assert refusal.value.code == "ANNOUNCEMENT_INSTANT_WITHOUT_A_ZONE"


# --------------------------------------------------------------------------------------------- the counts are exact

def test_every_consensus_row_is_either_joined_or_counted_under_exactly_one_code(tmp_path):
    rows = [
        archive_row("2025/03/06", "8:30:00", "United States", "Initial Jobless Claims", "220", "215"),   # joins
        archive_row("2025/03/13", "8:30:00", "United States", "Initial Jobless Claims", "222", "218"),   # no day
        archive_row("2025/03/06", "10:30:00", "United States", "Crude Oil Inventories", "3.6", "1.2"),   # no name
        archive_row("2025/03/06", "8:30:00", "Germany", "Industrial Production", "1.0", "0.5"),          # no country
        archive_row("2025/03/07", "8:30:00", "United States", "PPI", "0.3", "0.2"),                      # ambiguous
        archive_row("2025/03/07", "8:30:00", "United States", "PPI", "0.4", "0.2"),                      # ambiguous
        archive_row("2025/03/06", "8:30:00", "United States", "Retail Sales", "0.2", ""),                # no consensus
    ]
    announcements = [["USD", "initial_jobless_claims", "2025-03-01", "220", "2025-03-06 13:30:00+00:00"],
                     ["USD", "ppi", "2025-02-28", "0.3", "2025-03-07 13:30:00+00:00"],
                     ["JPY", "gdp", "2025-01-31", "1.0", "2025-03-09 23:50:00+00:00"]]
    document = run(tmp_path, rows, announcements)
    counts = document["counts"]
    excluded = document["excluded"]["counts"]
    assert counts["consensus_rows_with_an_actual_and_a_consensus"] == 6
    assert counts["joined"] == 1
    assert excluded["NO_OBSERVED_ANNOUNCEMENT"] == 3
    assert excluded["AMBIGUOUS_MATCH"] == 2
    assert excluded["NO_CONSENSUS"] == 1
    assert counts["joined"] + excluded["NO_OBSERVED_ANNOUNCEMENT"] + excluded["AMBIGUOUS_MATCH"] == 6
    assert document["excluded"]["no_observed_announcement_by_reason"] == {
        "COUNTRY_NOT_IN_TABLE": 1, "RELEASE_NAME_NOT_IN_THE_ANNOUNCEMENT_ARCHIVE": 1,
        "NOTHING_ANNOUNCED_ON_THAT_DATE": 1}
    # the announcements nobody claimed are counted too: a join rate is about both archives
    assert counts["announcements_matched"] == 1
    assert counts["announcements_no_consensus_row_claimed"] == 2


def test_the_join_rate_is_reported_per_event_type_over_the_rows_that_could_have_joined(tmp_path):
    rows = [archive_row("2025/03/06", "8:30:00", "United States", "Initial Jobless Claims", "220", "215"),
            archive_row("2025/03/13", "8:30:00", "United States", "Initial Jobless Claims", "222", "218"),
            archive_row("2025/03/06", "10:30:00", "United States", "Crude Oil Inventories", "3.6", "1.2")]
    document = run(tmp_path, rows,
                   [["USD", "initial_jobless_claims", "2025-03-01", "220", "2025-03-06 13:30:00+00:00"]])
    rates = document["counts"]["by_event_type"]
    assert rates["United States | Initial Jobless Claims"] == {
        "consensus_rows_with_an_actual": 2, "joined": 1, "join_rate": 0.5}
    assert rates["United States | Crude Oil Inventories"] == {
        "consensus_rows_with_an_actual": 1, "joined": 0, "join_rate": 0.0}


def test_the_written_csv_carries_the_names_the_event_builder_reads(tmp_path):
    document = run(
        tmp_path,
        [archive_row("2025/03/06", "8:30:00", "United States", "Initial Jobless Claims", "220", "215")],
        [["USD", "initial_jobless_claims", "2025-03-01", "220", "2025-03-06 13:30:00+00:00"]])
    out = calendar_join.write_csv(document, tmp_path / "joined.csv")
    header, row = list(csv.reader(out.open(encoding="utf-8")))[:2]
    assert header[:5] == ["event_type", "country", "currency", "event_time", "published_at"]
    assert "published_at" in header and "consensus" in header and "historical_availability" in header
    assert dict(zip(header, row))["published_at"] == "2025-03-06T13:30:00+00:00"


def test_the_cli_writes_both_the_csv_and_the_report(tmp_path, capsys):
    archive = write_archive(tmp_path / "a.csv",
                            [archive_row("2025/03/06", "8:30:00", "United States",
                                         "Initial Jobless Claims", "220", "215")])
    announcements = write_announcements(
        tmp_path / "n.csv", [["USD", "initial_jobless_claims", "2025-03-01", "220", "2025-03-06 13:30:00+00:00"]])
    code = calendar_join.main(["--archive", str(archive), "--announcements", str(announcements),
                               "--out", str(tmp_path / "joined.csv"), "--report", str(tmp_path / "report.json")])
    assert code == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["schema"] == calendar_join.SCHEMA
    assert report["counts"]["joined"] == 1
    assert (tmp_path / "joined.csv").is_file()
