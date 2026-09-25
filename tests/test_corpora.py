"""WP29's corpora: many problems with declared identities, assembled from data already on disk and from nothing else.

The failure these tests exist to close is not a crash. It is a run of thirty outcomes over one dataset presented as a
measurement of a chooser, and the two ways it happens quietly: one source entered twice under two names, and a corpus
whose values came from somewhere other than its declared source. So what is checked here is the *inventory*, not the
reader: that every declared corpus names a distinct source and derivation, that assembly is byte-reproducible, that
the manifest binds each file and its source by digest, and that a corpus too small to hold anything out is refused by
name rather than written and reported as underpowered.
"""

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from feature_eng_m5phet import corpora, design


def write_source(path, *, rows=900, columns=("alpha", "beta")):
    """A small dated CSV in the shape the feature-store reader reads; its values are this file's own."""
    start = datetime(2015, 1, 1, tzinfo=timezone.utc)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["Date", *columns])
        for index in range(rows):
            writer.writerow([(start + timedelta(days=index)).strftime("%Y-%m-%d"),
                             f"{index % 17:.6f}", f"{(index % 5) * 2.5:.6f}"])
    return Path(path)


def entry(source, **overrides):
    declared = {"id": "probe", "reader": "feature_store_csv", "source": str(source),
                "features": ["alpha", "beta"], "derivation": "the two columns of the source, copied"}
    declared.update(overrides)
    return declared


# --------------------------------------------------------------------------------------------------------------------
# the inventory: distinct problems, not one dataset under many names
# --------------------------------------------------------------------------------------------------------------------

def test_every_declared_corpus_has_a_distinct_id_and_declares_its_source_and_its_derivation():
    declared = corpora.declared_corpora()
    assert len(declared) >= 15, "WP29 asks for at least fifteen distinct corpora"
    assert len({item["id"] for item in declared}) == len(declared)
    for item in declared:
        assert item["reader"] in corpora.READERS
        assert item["features"] and len(set(item["features"])) == len(item["features"])
        assert item["derivation"].strip()


def test_two_corpora_may_share_a_source_file_only_when_they_declare_disjoint_slices_of_it():
    by_source = {}
    for item in corpora.declared_corpora():
        by_source.setdefault(item["source"], []).append(item)
    for source, items in by_source.items():
        if len(items) == 1:
            continue
        years = [item.get("year") for item in items]
        assert all(year is not None for year in years), f"{source} is declared twice without a disjoint slice"
        assert len(set(years)) == len(years), f"{source} declares the same slice twice"


# --------------------------------------------------------------------------------------------------------------------
# assembly: reproducible, identified by content, and honest about what it dropped
# --------------------------------------------------------------------------------------------------------------------

def test_assembly_is_byte_reproducible_and_the_manifest_binds_the_file_and_its_source(tmp_path):
    source = write_source(tmp_path / "source.csv")
    first = corpora.assemble_one(entry(source), tmp_path / "one")
    second = corpora.assemble_one(entry(source), tmp_path / "two")

    assert first["sha256"] == second["sha256"]
    assert (tmp_path / "one" / "probe.csv").read_bytes() == (tmp_path / "two" / "probe.csv").read_bytes()
    assert first["sha256"] == corpora.file_sha256(first["path"])
    assert first["source"]["sha256"] == corpora.file_sha256(source)
    assert first["rows"] == 900 and first["selection"]["stride"] == 1


def test_an_assembled_corpus_is_a_table_the_rest_of_the_package_reads(tmp_path):
    source = write_source(tmp_path / "source.csv")
    assembled = corpora.assemble_one(entry(source), tmp_path / "out")
    table = design.read_table(assembled["path"])
    assert table["time_column"] == corpora.TIME_COLUMN
    assert sorted(table["cells"]) == sorted(assembled["features"])
    assert table["rows_read"] == assembled["rows"]


def test_a_corpus_larger_than_the_cap_is_strided_and_says_so(tmp_path):
    source = write_source(tmp_path / "source.csv", rows=2000)
    assembled = corpora.assemble_one(entry(source), tmp_path / "out", max_rows=500)
    assert assembled["rows"] == 500
    assert assembled["selection"]["stride"] == 4 and assembled["selection"]["cap"] == 500
    assert assembled["selection"]["finite_rows"] == 2000


def test_a_row_whose_feature_is_missing_is_dropped_and_counted_never_filled(tmp_path):
    source = tmp_path / "gappy.csv"
    lines = write_source(source).read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].rsplit(",", 1)[0] + ","                       # the first row loses its last feature
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assembled = corpora.assemble_one(entry(source), tmp_path / "out")
    assert assembled["rows"] == 899 and assembled["selection"]["dropped_nonfinite"] == 1


def test_a_corpus_too_small_to_hold_anything_out_is_refused_by_name(tmp_path):
    source = write_source(tmp_path / "small.csv", rows=40)
    with pytest.raises(corpora.CorpusRefusal) as raised:
        corpora.assemble_one(entry(source), tmp_path / "out")
    assert str(raised.value).startswith(corpora.TOO_FEW_ROWS)
    assert not (tmp_path / "out" / "probe.csv").exists()


def test_a_missing_source_is_reported_in_the_manifest_and_never_silently_skipped(tmp_path):
    source = write_source(tmp_path / "source.csv")
    inventory = corpora.assemble(tmp_path / "out",
                                 corpora=[entry(source, id="present"),
                                          entry(tmp_path / "absent.csv", id="missing")])
    assert inventory["distinct_corpora"] == 1
    assert [item["id"] for item in inventory["refused"]] == ["missing"]
    assert inventory["refused"][0]["refusal"].startswith(corpora.SOURCE_NOT_FOUND)


def test_the_manifest_counts_the_corpora_and_the_source_files_they_rest_on(tmp_path):
    first = write_source(tmp_path / "a.csv")
    second = write_source(tmp_path / "b.csv", rows=800)
    inventory = corpora.assemble(tmp_path / "out",
                                 corpora=[entry(first, id="a"), entry(second, id="b")])
    assert inventory["schema"] == corpora.SCHEMA
    assert inventory["distinct_corpora"] == 2 and inventory["distinct_source_files"] == 2
    assert json.loads(json.dumps(inventory)) == inventory        # the manifest is plain JSON, written as it is read
