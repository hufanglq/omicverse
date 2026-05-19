"""Tests for `omicverse.utils.preflight_alignment` and friends.

The cases exercise the three motivating bug classes:

1. Pandas auto-rename of duplicate column labels (most common silent
   failure mode — counts CSVs with two columns sharing a sample ID).
2. Samples present in the matrix but absent from metadata, and vice
   versa.
3. Mixed in-memory DataFrame / file-path / AnnData inputs.

No network calls; everything uses tmp_path fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_pair(tmp_path: Path):
    """Counts (genes × samples) + meta (samples × phenotype), all aligned."""
    samples = ["s1", "s2", "s3", "s4"]
    counts = pd.DataFrame(
        {s: [10 * i + j for j in range(3)] for i, s in enumerate(samples)},
        index=["GENE_A", "GENE_B", "GENE_C"],
    )
    counts.index.name = "gene"
    counts_path = tmp_path / "counts.csv"
    counts.to_csv(counts_path)

    meta = pd.DataFrame(
        {"sample_id": samples, "condition": ["A", "A", "B", "B"], "batch": [1, 2, 1, 2]}
    )
    meta_path = tmp_path / "meta.csv"
    meta.to_csv(meta_path, index=False)
    return counts_path, meta_path


@pytest.fixture
def dup_pair(tmp_path: Path):
    """Counts CSV where two columns share the same sample ID — the
    pandas-auto-rename case. Naive `pd.read_csv` plus `.duplicated()`
    would report 0 here; the helper must catch it via raw-header read."""
    counts_path = tmp_path / "counts_dup.csv"
    counts_path.write_text(
        "gene,s1,s2,s2,s3\n"
        "GENE_A,1,2,3,4\n"
        "GENE_B,5,6,7,8\n"
    )

    meta = pd.DataFrame({"sample_id": ["s1", "s2", "s3"], "condition": ["A", "B", "C"]})
    meta_path = tmp_path / "meta_dup.csv"
    meta.to_csv(meta_path, index=False)
    return counts_path, meta_path


@pytest.fixture
def missing_meta_pair(tmp_path: Path):
    """One matrix sample is absent from metadata."""
    counts_path = tmp_path / "counts_miss.csv"
    counts_path.write_text(
        "gene,s1,s2,s3,s4\n"
        "GENE_A,1,2,3,4\n"
    )
    meta = pd.DataFrame({"sample_id": ["s1", "s2", "s3"], "condition": ["A", "B", "C"]})
    meta_path = tmp_path / "meta_miss.csv"
    meta.to_csv(meta_path, index=False)
    return counts_path, meta_path


# ---------------------------------------------------------------------------
# preflight_alignment
# ---------------------------------------------------------------------------


def test_preflight_clean(clean_pair):
    from omicverse.utils import preflight_alignment

    counts_path, meta_path = clean_pair
    r = preflight_alignment(counts_path, meta_path)
    assert r.is_clean
    assert not r.needs_alignment
    assert r.n_dup_in_matrix == 0
    assert r.n_dup_in_meta == 0
    assert r.n_missing_from_meta == 0
    assert r.n_missing_from_matrix == 0
    assert r.sample_col_used == "sample_id"
    assert r.matrix_sample_axis == "columns"


def test_preflight_catches_pandas_renamed_duplicates(dup_pair):
    """The canonical bug: pandas reads the CSV with two `s2` columns as
    `s2` + `s2.1`. A naive `df.columns.duplicated().sum()` returns 0.
    The helper reads the raw header and reports the real count."""
    from omicverse.utils import preflight_alignment

    counts_path, meta_path = dup_pair
    r = preflight_alignment(counts_path, meta_path)
    assert r.needs_alignment, r
    # Two `s2` columns → 1 ID appears more than once → counter > 1 once.
    assert r.n_dup_in_matrix == 1, r

    # Sanity: naive pandas check would NOT have caught this.
    naive = pd.read_csv(counts_path).columns.duplicated().sum()
    assert naive == 0, "this test asserts pandas's silent rename behavior"


def test_preflight_catches_missing_from_meta(missing_meta_pair):
    from omicverse.utils import preflight_alignment

    counts_path, meta_path = missing_meta_pair
    r = preflight_alignment(counts_path, meta_path)
    assert r.needs_alignment
    assert r.n_dup_in_matrix == 0
    assert r.n_missing_from_meta == 1  # `s4` in matrix, not in meta
    assert r.n_missing_from_matrix == 0


def test_preflight_accepts_dataframe_inputs(clean_pair):
    """In-memory DataFrames go through the same code path."""
    from omicverse.utils import preflight_alignment

    counts_path, meta_path = clean_pair
    counts = pd.read_csv(counts_path, index_col=0)
    meta = pd.read_csv(meta_path)
    r = preflight_alignment(counts, meta)
    assert r.is_clean


def test_preflight_explicit_sample_col(missing_meta_pair):
    """User-supplied `sample_col` overrides auto-detect."""
    from omicverse.utils import preflight_alignment

    counts_path, meta_path = missing_meta_pair
    r = preflight_alignment(counts_path, meta_path, sample_col="sample_id")
    assert r.sample_col_used == "sample_id"


def test_preflight_no_overlap_raises(tmp_path: Path):
    """When no metadata column overlaps the matrix sample axis, fail
    clearly rather than producing a garbage diff."""
    from omicverse.utils import preflight_alignment

    counts = tmp_path / "counts.csv"
    counts.write_text("gene,s1,s2\nA,1,2\n")
    meta = tmp_path / "meta.csv"
    meta.write_text("subject,age\nfoo,10\nbar,20\n")
    with pytest.raises(ValueError, match="overlaps"):
        preflight_alignment(counts, meta)


def test_preflight_str_summary_includes_status(dup_pair):
    from omicverse.utils import preflight_alignment

    counts_path, meta_path = dup_pair
    r = preflight_alignment(counts_path, meta_path)
    s = str(r)
    assert "needs alignment" in s
    assert "dup_matrix=1" in s


def test_preflight_summary_dict_is_json_serializable(dup_pair):
    from omicverse.utils import preflight_alignment

    counts_path, meta_path = dup_pair
    r = preflight_alignment(counts_path, meta_path)
    d = r.summary_dict()
    assert "n_dup_in_matrix" in d
    assert "matrix_sample_ids" not in d  # the heavy lists are kept off
    json.dumps(d)  # raises on non-serializable values


# ---------------------------------------------------------------------------
# align_to_common
# ---------------------------------------------------------------------------


def test_align_drops_pandas_renamed_duplicates(dup_pair):
    """After alignment, the matrix has no duplicates and matches meta."""
    from omicverse.utils import align_to_common, preflight_alignment

    counts_path, meta_path = dup_pair
    r = preflight_alignment(counts_path, meta_path)
    mat, meta = align_to_common(counts_path, meta_path, r)
    # Both `s2` columns dropped (default `keep=False`); only s1 and s3 remain.
    assert list(mat.columns) == ["s1", "s3"]
    assert list(meta.index) == ["s1", "s3"]


def test_align_drops_missing_from_meta(missing_meta_pair):
    from omicverse.utils import align_to_common, preflight_alignment

    counts_path, meta_path = missing_meta_pair
    r = preflight_alignment(counts_path, meta_path)
    mat, meta = align_to_common(counts_path, meta_path, r)
    # `s4` (matrix-only) dropped.
    assert list(mat.columns) == ["s1", "s2", "s3"]
    assert list(meta.index) == ["s1", "s2", "s3"]


def test_align_samples_one_shot(missing_meta_pair):
    from omicverse.utils import align_samples

    counts_path, meta_path = missing_meta_pair
    mat, meta, r = align_samples(counts_path, meta_path)
    assert list(mat.columns) == ["s1", "s2", "s3"]
    assert list(meta.index) == ["s1", "s2", "s3"]
    assert r.n_missing_from_meta == 1


def test_align_clean_pair_is_passthrough(clean_pair):
    from omicverse.utils import align_samples

    counts_path, meta_path = clean_pair
    mat, meta, r = align_samples(counts_path, meta_path)
    assert r.is_clean
    assert sorted(mat.columns) == ["s1", "s2", "s3", "s4"]


# ---------------------------------------------------------------------------
# Misc edge cases
# ---------------------------------------------------------------------------


def test_preflight_handles_int_sample_ids(tmp_path: Path):
    """CSV readers can infer one side as int64; the helper must coerce
    both to str before comparing."""
    from omicverse.utils import preflight_alignment

    counts = tmp_path / "counts.csv"
    counts.write_text("gene,1,2,3\nA,1,2,3\n")
    meta = tmp_path / "meta.csv"
    meta.write_text("sample_id,condition\n1,A\n2,B\n3,C\n")  # int-looking
    r = preflight_alignment(counts, meta)
    assert r.is_clean


def test_preflight_tsv(tmp_path: Path):
    """Tab-separated text works without explicit `sep=`."""
    from omicverse.utils import preflight_alignment

    counts = tmp_path / "counts.tsv"
    counts.write_text("gene\ts1\ts2\nA\t1\t2\n")
    meta = tmp_path / "meta.tsv"
    meta.write_text("sample_id\tcondition\ns1\tA\ns2\tB\n")
    r = preflight_alignment(counts, meta)
    assert r.is_clean
