"""Sample × metadata alignment pre-flight (generic, format-agnostic).

A pre-flight check + alignment helper for any joint analysis that
touches a sample axis (expression × metadata, counts × phenotype,
ASV × sample sheet, peak × treatment, methylation × age, AnnData ×
phenotype). Specifically robust against the most common silent failure
mode: **pandas `read_csv` / `read_excel` auto-renames duplicate column
labels** (`id154` → `id154` + `id154.1`), which masks the duplicate
count and shifts downstream statistics (PCA / DEG / clustering) by the
unaligned-sample delta. The check itself is `O(n)` and adds ~1 s wall
clock; the actual alignment runs only when the pre-flight flags an
issue.

Supported inputs
----------------

For both `matrix` and `meta`, accept either:

  * file paths (`str` / `pathlib.Path`) — CSV / TSV / TXT / TAB
    (any text format with a stable delimiter), Excel `.xlsx` / `.xls`,
    HDF5 `.h5`, AnnData `.h5ad`, Parquet `.parquet`
  * in-memory `pd.DataFrame`
  * (matrix only) in-memory `anndata.AnnData`

Format detection is by extension. Text-file delimiter is inferred from
extension (`.csv` → ',', `.tsv`/`.txt`/`.tab` → tab) but may be
overridden with `sep=`.

Examples
--------

Basic usage::

    import omicverse as ov

    result = ov.utils.preflight_alignment("counts.csv", "meta.csv")
    print(result)
    # PreflightResult(needs alignment; dup_matrix=4, dup_meta=21,
    # missing_from_meta=23, missing_from_matrix=0, ...)

    if result.needs_alignment:
        matrix, meta = ov.utils.align_to_common(
            "counts.csv", "meta.csv", result
        )
        # `matrix` / `meta` are now sorted to the same sample list,
        # duplicates dropped, missing-on-either-side dropped.

One-shot::

    matrix, meta, result = ov.utils.align_samples("counts.csv", "meta.csv")

AnnData::

    result = ov.utils.preflight_alignment(adata, "meta.csv")
    if not result.is_clean:
        adata, meta = ov.utils.align_to_common(adata, "meta.csv", result)

Why this exists
---------------

The single most common reason a methodologically correct analysis
disagrees with a reference notebook by a small numerical delta is
**sample-set mismatch**: duplicate IDs, samples missing from one
side, ID-dtype mismatches. None of these raise exceptions. They just
shift PCA percentages, DEG counts, clustering structure by the size
of the unaligned subset. This helper makes the check trivial to
invoke and impossible to silently bypass.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .registry import register_function

try:  # optional, only needed for h5ad / AnnData inputs
    import anndata as _ad
except Exception:  # pragma: no cover
    _ad = None


_TEXT_SEPS = {".csv": ",", ".tsv": "\t", ".txt": "\t", ".tab": "\t"}
_EXCEL_SUFFIXES = {".xlsx", ".xls"}
_H5_SUFFIXES = {".h5", ".h5ad", ".loom"}
_PARQUET_SUFFIXES = {".parquet", ".pq"}


@dataclass(frozen=True)
class PreflightResult:
    """Outcome of :func:`preflight_alignment`.

    All four `n_*` counts are integers. `is_clean` is True iff all four
    are zero. `needs_alignment` is the inverse of `is_clean` and is
    provided for readability.
    """

    n_dup_in_matrix: int
    n_dup_in_meta: int
    n_missing_from_meta: int
    n_missing_from_matrix: int
    sample_col_used: str | None
    matrix_sample_axis: str  # "columns" or "rows"
    matrix_kind: str
    meta_kind: str
    matrix_sample_ids: list[str]
    meta_sample_ids: list[str]

    @property
    def is_clean(self) -> bool:
        return not any([
            self.n_dup_in_matrix,
            self.n_dup_in_meta,
            self.n_missing_from_meta,
            self.n_missing_from_matrix,
        ])

    @property
    def needs_alignment(self) -> bool:
        return not self.is_clean

    def summary_dict(self) -> dict[str, Any]:
        """Lightweight, JSON-serializable view (drops the full ID lists)."""
        return {
            "n_dup_in_matrix": self.n_dup_in_matrix,
            "n_dup_in_meta": self.n_dup_in_meta,
            "n_missing_from_meta": self.n_missing_from_meta,
            "n_missing_from_matrix": self.n_missing_from_matrix,
            "sample_col_used": self.sample_col_used,
            "matrix_sample_axis": self.matrix_sample_axis,
            "matrix_kind": self.matrix_kind,
            "meta_kind": self.meta_kind,
            "n_matrix_samples": len(self.matrix_sample_ids),
            "n_meta_samples": len(self.meta_sample_ids),
            "is_clean": self.is_clean,
        }

    def __str__(self) -> str:  # human-friendly for `print(result)`
        status = "clean" if self.is_clean else "needs alignment"
        return (
            f"PreflightResult({status}; "
            f"dup_matrix={self.n_dup_in_matrix}, "
            f"dup_meta={self.n_dup_in_meta}, "
            f"missing_from_meta={self.n_missing_from_meta}, "
            f"missing_from_matrix={self.n_missing_from_matrix}, "
            f"sample_col={self.sample_col_used!r}, "
            f"axis={self.matrix_sample_axis}, "
            f"n_matrix={len(self.matrix_sample_ids)}, "
            f"n_meta={len(self.meta_sample_ids)})"
        )


# ---------------------------------------------------------------------------
# Format-aware label readers — preserve duplicates that pandas would rename
# ---------------------------------------------------------------------------


def _infer_kind(obj) -> str:
    """Classify an input into one of: anndata, dataframe, csv, tsv, xlsx,
    h5ad, parquet. Defaults to "csv" for unknown text extensions."""
    if hasattr(obj, "obs_names") and hasattr(obj, "var_names"):
        return "anndata"
    if isinstance(obj, pd.DataFrame):
        return "dataframe"
    suffix = Path(str(obj)).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".tsv", ".txt", ".tab"}:
        return "tsv"
    if suffix in _EXCEL_SUFFIXES:
        return "xlsx"
    if suffix in _H5_SUFFIXES:
        return "h5ad"
    if suffix in _PARQUET_SUFFIXES:
        return "parquet"
    return "csv"  # generic fallback


def _read_raw_header(path: Path, sep: str) -> list[str]:
    """Read the first non-comment line of a text file and split on `sep`.

    Uses raw file IO + `str.split` so duplicate labels are preserved.
    `pd.read_csv` would silently auto-rename duplicates to `name.1`,
    `name.2`, ... — defeating the duplicate count.
    """
    with open(path) as f:
        for raw_line in f:
            if raw_line.strip().startswith("#"):
                continue
            return raw_line.rstrip("\r\n").split(sep)
    raise ValueError(f"empty file: {path}")


def _read_xlsx_first_row(path: Path) -> list[str]:
    """First row of an Excel sheet via openpyxl's low-level cell iterator
    (bypasses pandas's column de-dup). Requires `openpyxl`."""
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "openpyxl is required for Excel pre-flight. "
            "Install with `pip install openpyxl`."
        ) from exc
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    for row in ws.iter_rows(max_row=1, values_only=True):
        return [str(c) if c is not None else "" for c in row]
    raise ValueError(f"empty xlsx: {path}")


def _load_matrix_axis_ids(
    matrix, sep: str | None = None
) -> tuple[dict[str, list[str]], str]:
    """Return ({"columns": [...], "rows": [...]}, kind) for the matrix.

    For text/Excel formats the "columns" axis is read raw-byte to bypass
    pandas auto-rename. The "rows" axis is read via pandas — duplicates
    on the index are preserved by pandas and don't need the workaround.
    """
    kind = _infer_kind(matrix)
    if kind == "dataframe":
        return (
            {
                "columns": list(matrix.columns.astype(str)),
                "rows": list(matrix.index.astype(str)),
            },
            kind,
        )
    if kind == "anndata":
        return (
            {
                "columns": list(map(str, matrix.var_names)),
                "rows": list(map(str, matrix.obs_names)),
            },
            kind,
        )
    p = Path(str(matrix))
    if kind in ("csv", "tsv"):
        sep_ = sep or _TEXT_SEPS[p.suffix.lower() or ".csv"]
        header = _read_raw_header(p, sep_)
        cols = header[1:] if header else []
        try:
            rows_index = pd.read_csv(p, sep=sep_, usecols=[0], comment="#").iloc[:, 0]
            rows = list(rows_index.astype(str))
        except Exception:
            rows = []
        return {"columns": cols, "rows": rows}, kind
    if kind == "xlsx":
        header = _read_xlsx_first_row(p)
        cols = header[1:] if header else []
        try:
            rows_index = pd.read_excel(p, usecols=[0]).iloc[:, 0]
            rows = list(rows_index.astype(str))
        except Exception:
            rows = []
        return {"columns": cols, "rows": rows}, kind
    if kind == "h5ad":
        if _ad is None:
            raise ImportError("anndata is required for h5ad pre-flight")
        adata = _ad.read_h5ad(p)
        return (
            {
                "columns": list(map(str, adata.var_names)),
                "rows": list(map(str, adata.obs_names)),
            },
            "anndata",
        )
    if kind == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pyarrow is required for parquet pre-flight") from exc
        names = list(pq.read_schema(str(p)).names)
        cols = names[1:] if names else []
        return {"columns": cols, "rows": []}, kind
    raise ValueError(f"unsupported matrix kind: {matrix!r}")


def _load_meta(meta, sep: str | None = None) -> tuple[pd.DataFrame, str]:
    """Load metadata as a DataFrame. Pandas does not silently rename row-
    index duplicates, so plain readers are safe on the metadata side."""
    kind = _infer_kind(meta)
    if kind == "dataframe":
        return meta.copy(), kind
    p = Path(str(meta))
    if kind in ("csv", "tsv"):
        sep_ = sep or _TEXT_SEPS[p.suffix.lower() or ".csv"]
        return pd.read_csv(p, sep=sep_, comment="#"), kind
    if kind == "xlsx":
        return pd.read_excel(p), kind
    if kind == "parquet":
        return pd.read_parquet(p), kind
    raise ValueError(f"unsupported metadata kind: {meta!r}")


def _pick_sample_col(
    meta: pd.DataFrame, matrix_axis_ids: dict[str, list[str]]
) -> tuple[str | None, str | None]:
    """Pick the (sample_col, matrix_axis) pair with the highest overlap.

    Iterates every metadata column × matrix axis and chooses whichever
    pair has the most overlapping IDs. Ties broken by column order in
    `meta`. Returns `(None, None)` if no column has any overlap.
    """
    best_col: str | None = None
    best_axis: str | None = None
    best_overlap = 0
    for col in meta.columns:
        try:
            vals = set(meta[col].dropna().astype(str))
        except Exception:
            continue
        for axis_name, axis_ids in matrix_axis_ids.items():
            if not axis_ids:
                continue
            overlap = len(vals & set(map(str, axis_ids)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_col = col
                best_axis = axis_name
    return best_col, best_axis


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@register_function(
    aliases=[
        "preflight_alignment", "sample_metadata_alignment", "sample_alignment",
        "alignment_preflight", "preflight", "sample axis check",
        "样本对齐预检", "样本-元数据对齐", "样本对齐", "对齐预检", "联表预检",
        "metadata join check", "sample sheet alignment",
    ],
    category="utils",
    description=(
        "Diagnose sample-axis alignment between a sample × feature matrix "
        "and its metadata table. Counts duplicate-on-each-side and missing-"
        "on-each-side in a single ~O(n) pass. Robust against pandas's "
        "silent auto-rename of duplicate column labels (`id154` → "
        "`id154.1`), which is the canonical false-negative source. Returns "
        "a `PreflightResult` whose `is_clean` / `needs_alignment` booleans "
        "drive the IF-branch into `align_to_common`. Run this BEFORE any "
        "joint PCA / DEG / clustering / batch-correction / survival."
    ),
    examples=[
        "# Diagnose alignment before downstream analysis",
        "result = ov.utils.preflight_alignment('counts.csv', 'meta.csv')",
        "print(result)  # PreflightResult(needs alignment; dup_matrix=4, ...)",
        "if result.needs_alignment:",
        "    matrix, meta = ov.utils.align_to_common('counts.csv', 'meta.csv', result)",
        "",
        "# AnnData input — sample axis auto-detected from .obs_names",
        "r = ov.utils.preflight_alignment(adata, 'phenotype.csv')",
        "",
        "# Override auto-detect when ambiguous",
        "r = ov.utils.preflight_alignment(",
        "    'counts.tsv', 'samples.xlsx', sample_col='subject_id',",
        "    matrix_sample_axis='columns', sep='\\t')",
    ],
    related=[
        "utils.align_to_common",
        "utils.align_samples",
        "utils.read",
    ],
    auto_fix="escalate",
)
def preflight_alignment(
    matrix,
    meta,
    *,
    sample_col: str | None = None,
    sep: str | None = None,
    matrix_sample_axis: str | None = None,
) -> PreflightResult:
    """Diagnose sample-axis alignment between `matrix` and `meta`.

    Parameters
    ----------
    matrix
        Sample × feature matrix. Accepts a path (CSV / TSV / TXT / TAB /
        XLSX / H5AD / parquet), a `pd.DataFrame`, or an
        `anndata.AnnData`.
    meta
        Metadata / sample-sheet table. Accepts a path (same formats as
        matrix except AnnData) or a `pd.DataFrame`.
    sample_col
        Name of the column in `meta` that holds sample IDs. When
        `None`, the column whose values overlap most with the matrix's
        sample axis is auto-detected.
    sep
        Delimiter override for text files. Inferred from extension when
        `None`.
    matrix_sample_axis
        `"columns"` or `"rows"`. When `None`, inferred from which axis
        of the matrix overlaps with `meta`'s sample column.

    Returns
    -------
    :class:`PreflightResult`
    """
    matrix_axis_ids, matrix_kind = _load_matrix_axis_ids(matrix, sep=sep)
    meta_df, meta_kind = _load_meta(meta, sep=sep)

    if sample_col is None or matrix_sample_axis is None:
        guessed_col, guessed_axis = _pick_sample_col(meta_df, matrix_axis_ids)
        sample_col = sample_col or guessed_col
        matrix_sample_axis = matrix_sample_axis or guessed_axis

    if sample_col is None or matrix_sample_axis is None:
        raise ValueError(
            "No metadata column overlaps the matrix's sample axis. "
            "Pass `sample_col=` and/or `matrix_sample_axis=` explicitly."
        )
    if matrix_sample_axis not in {"columns", "rows"}:
        raise ValueError(
            f"matrix_sample_axis must be 'columns' or 'rows', got {matrix_sample_axis!r}"
        )

    matrix_sample_ids = list(map(str, matrix_axis_ids[matrix_sample_axis]))
    meta_sample_ids = list(meta_df[sample_col].dropna().astype(str))

    mc = collections.Counter(matrix_sample_ids)
    me = collections.Counter(meta_sample_ids)
    return PreflightResult(
        n_dup_in_matrix=sum(c > 1 for c in mc.values()),
        n_dup_in_meta=sum(c > 1 for c in me.values()),
        n_missing_from_meta=len(set(mc) - set(me)),
        n_missing_from_matrix=len(set(me) - set(mc)),
        sample_col_used=sample_col,
        matrix_sample_axis=matrix_sample_axis,
        matrix_kind=matrix_kind,
        meta_kind=meta_kind,
        matrix_sample_ids=matrix_sample_ids,
        meta_sample_ids=meta_sample_ids,
    )


@register_function(
    aliases=[
        "align_to_common", "align_samples_to_common", "sample_alignment_apply",
        "样本对齐", "样本对齐应用", "联表对齐", "去重对齐", "样本交集",
        "metadata join", "sample set intersection", "drop unmatched samples",
    ],
    category="utils",
    description=(
        "Apply the alignment fix flagged by `preflight_alignment`: drop "
        "duplicate sample IDs on each side, intersect to the common "
        "sample set, reorder both tables to the same sample list. "
        "Returns `(aligned_matrix, aligned_meta)` with types matching the "
        "inputs (DataFrame / AnnData on the matrix side; DataFrame "
        "indexed by the resolved sample column on the meta side)."
    ),
    examples=[
        "result = ov.utils.preflight_alignment('counts.csv', 'meta.csv')",
        "if result.needs_alignment:",
        "    matrix, meta = ov.utils.align_to_common(",
        "        'counts.csv', 'meta.csv', result)",
        "",
        "# Standalone (runs the pre-flight internally)",
        "matrix, meta = ov.utils.align_to_common('counts.csv', 'meta.csv')",
        "",
        "# Keep first duplicate instead of dropping all copies",
        "# (useful when the dataset README names a canonical replicate)",
        "matrix, meta = ov.utils.align_to_common(",
        "    'counts.csv', 'meta.csv', drop_dups=True)",
    ],
    related=[
        "utils.preflight_alignment",
        "utils.align_samples",
    ],
    prerequisites={"optional_functions": ["preflight_alignment"]},
    auto_fix="auto",
)
def align_to_common(
    matrix,
    meta,
    result: PreflightResult | None = None,
    *,
    sample_col: str | None = None,
    sep: str | None = None,
    matrix_sample_axis: str | None = None,
    drop_dups: bool = True,
):
    """Drop duplicates + intersect to the common sample set.

    Returns ``(aligned_matrix, aligned_meta)`` where types match the
    inputs (DataFrame / AnnData on the matrix side; DataFrame on the
    metadata side, indexed by the resolved sample column).
    """
    if result is None:
        result = preflight_alignment(
            matrix,
            meta,
            sample_col=sample_col,
            sep=sep,
            matrix_sample_axis=matrix_sample_axis,
        )

    meta_df, _ = _load_meta(meta, sep=sep)
    if drop_dups and result.n_dup_in_meta:
        meta_df = meta_df[
            ~meta_df[result.sample_col_used].astype(str).duplicated(keep=False)
        ]

    if result.matrix_kind == "anndata":
        if result.matrix_sample_axis != "rows":
            raise ValueError(
                "AnnData expected to carry samples on `rows` (.obs); "
                f"got matrix_sample_axis={result.matrix_sample_axis}"
            )
        if hasattr(matrix, "obs_names"):
            adata = matrix
        else:
            if _ad is None:
                raise ImportError("anndata is required to align an AnnData input")
            adata = _ad.read_h5ad(str(matrix))
        if drop_dups and result.n_dup_in_matrix:
            adata = adata[~adata.obs_names.duplicated(keep=False)].copy()
        common = sorted(
            set(adata.obs_names.astype(str))
            & set(meta_df[result.sample_col_used].astype(str))
        )
        adata = adata[common].copy()
        aligned_meta = meta_df.set_index(result.sample_col_used).loc[common]
        return adata, aligned_meta

    # DataFrame-shaped matrix (in-memory or any non-AnnData path).
    if result.matrix_kind == "dataframe":
        mat = matrix.copy()
    else:
        p = Path(str(matrix))
        if result.matrix_kind in ("csv", "tsv"):
            sep_ = sep or _TEXT_SEPS[p.suffix.lower() or ".csv"]
            mat = pd.read_csv(p, sep=sep_, index_col=0, comment="#")
            # Restore raw column labels so pandas's auto-rename doesn't
            # destroy duplicate columns we explicitly want to dedup.
            mat.columns = _read_raw_header(p, sep_)[1:]
        elif result.matrix_kind == "xlsx":
            mat = pd.read_excel(p, index_col=0)
            mat.columns = _read_xlsx_first_row(p)[1:]
        elif result.matrix_kind == "parquet":
            mat = pd.read_parquet(p)
        else:
            raise ValueError(
                f"alignment not supported for matrix_kind={result.matrix_kind!r}"
            )

    if result.matrix_sample_axis == "columns":
        if drop_dups and result.n_dup_in_matrix:
            mat = mat.loc[:, ~mat.columns.astype(str).duplicated(keep=False)]
        common = sorted(
            set(mat.columns.astype(str))
            & set(meta_df[result.sample_col_used].astype(str))
        )
        mat = mat[common]
    else:  # rows
        if drop_dups and result.n_dup_in_matrix:
            mat = mat[~mat.index.astype(str).duplicated(keep=False)]
        common = sorted(
            set(mat.index.astype(str))
            & set(meta_df[result.sample_col_used].astype(str))
        )
        mat = mat.loc[common]

    aligned_meta = meta_df.set_index(result.sample_col_used).loc[common]
    return mat, aligned_meta


@register_function(
    aliases=[
        "align_samples", "preflight_and_align", "sample_alignment_oneshot",
        "一键样本对齐", "对齐+预检", "样本对齐(一步)",
        "align matrix and metadata", "ensure samples aligned",
    ],
    category="utils",
    description=(
        "One-shot wrapper: returns `(aligned_matrix, aligned_meta, "
        "PreflightResult)` for any pair of inputs. Always returns "
        "aligned tables (passthrough when the pre-flight is clean). "
        "Prefer over `preflight_alignment` + `align_to_common` when the "
        "caller wants both the cleaned tables and the diff audit in a "
        "single call."
    ),
    examples=[
        "matrix, meta, result = ov.utils.align_samples(",
        "    'counts.csv', 'meta.csv')",
        "print(result)  # the diff audit",
        "# matrix and meta are now safe for joint PCA / DEG / clustering",
        "",
        "# Save the audit to disk for the run report",
        "import json",
        "json.dump(result.summary_dict(),",
        "          open('outputs/sample_alignment.json', 'w'), indent=2)",
    ],
    related=[
        "utils.preflight_alignment",
        "utils.align_to_common",
    ],
    auto_fix="auto",
)
def align_samples(
    matrix,
    meta,
    *,
    sample_col: str | None = None,
    sep: str | None = None,
    matrix_sample_axis: str | None = None,
    drop_dups: bool = True,
):
    """One-shot ``(aligned_matrix, aligned_meta, result)`` convenience.

    Always returns aligned tables (re-uses the raw inputs when the
    pre-flight is clean). The third element of the tuple is the
    `PreflightResult` so callers can audit what was dropped.
    """
    result = preflight_alignment(
        matrix,
        meta,
        sample_col=sample_col,
        sep=sep,
        matrix_sample_axis=matrix_sample_axis,
    )
    aligned_matrix, aligned_meta = align_to_common(
        matrix,
        meta,
        result,
        sample_col=sample_col,
        sep=sep,
        matrix_sample_axis=matrix_sample_axis,
        drop_dups=drop_dups,
    )
    return aligned_matrix, aligned_meta, result


__all__ = [
    "PreflightResult",
    "preflight_alignment",
    "align_to_common",
    "align_samples",
]
