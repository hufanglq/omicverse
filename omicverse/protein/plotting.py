"""Minimal plotting for ``ov.protein``.

Three convenience plots:

* :func:`volcano` — log2FC × -log10(adj.P) scatter for a DE result.
* :func:`missing_pattern_plot` — heatmap of per-protein × per-sample
  missingness (white = observed, dark = missing).
* :func:`abundance_rank_plot` — per-sample rank-vs-intensity diagnostic
  (mirrors the MaxQuant ``QC_plot``).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .._registry import register_function


@register_function(
    aliases=["protein_volcano", "volcano", "蛋白火山图"],
    category="visualization",
    description=(
        "Volcano plot for a DE result table from ``ov.protein.de``. "
        "Highlights proteins passing ``adj_p_threshold`` (FDR) and "
        "``logfc_threshold`` (absolute log2FC). Returns the matplotlib "
        "axes for further annotation."
    ),
    examples=[
        "ov.protein.volcano(res, fc_col='logFC', p_col='adj.P.Val')",
    ],
)
def volcano(
    de_table: pd.DataFrame,
    *,
    fc_col: str = "logFC",
    p_col: str = "adj.P.Val",
    raw_p_col: str = "P.Value",
    logfc_threshold: float = 1.0,
    adj_p_threshold: float = 0.05,
    label_top: int = 10,
    gene_col: str = "gene",
    ax: Optional["matplotlib.axes.Axes"] = None,
    figsize: tuple[float, float] = (5.0, 4.5),
    s: float = 8.0,
    up_color: str = "#d62728",
    down_color: str = "#1f77b4",
    nochange_color: str = "#cccccc",
    title: Optional[str] = None,
):
    """Standard volcano: x = logFC, y = -log10(p). Highlights significant proteins."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    df = de_table.copy()
    pcol_used = p_col if p_col in df.columns else raw_p_col
    p = df[pcol_used].to_numpy(dtype=float)
    fc = df[fc_col].to_numpy(dtype=float)
    valid = np.isfinite(p) & np.isfinite(fc)
    df = df.loc[valid].reset_index(drop=True)
    p = p[valid]; fc = fc[valid]
    nlogp = -np.log10(np.clip(p, 1e-300, 1.0))

    up_mask = (fc >= logfc_threshold) & (p <= adj_p_threshold)
    down_mask = (fc <= -logfc_threshold) & (p <= adj_p_threshold)
    other_mask = ~(up_mask | down_mask)

    ax.scatter(fc[other_mask], nlogp[other_mask], s=s, c=nochange_color,
               edgecolor="none", alpha=0.7, label=None)
    ax.scatter(fc[up_mask], nlogp[up_mask], s=s, c=up_color,
               edgecolor="none", alpha=0.9,
               label=f"Up ({int(up_mask.sum())})")
    ax.scatter(fc[down_mask], nlogp[down_mask], s=s, c=down_color,
               edgecolor="none", alpha=0.9,
               label=f"Down ({int(down_mask.sum())})")

    ax.axhline(-np.log10(adj_p_threshold), color="black", lw=0.6, ls="--")
    ax.axvline(logfc_threshold, color="black", lw=0.6, ls="--")
    ax.axvline(-logfc_threshold, color="black", lw=0.6, ls="--")

    ax.set_xlabel(fc_col)
    ax.set_ylabel(f"-log10({pcol_used})")
    ax.legend(loc="best", fontsize=8, frameon=False)
    if title:
        ax.set_title(title)

    if label_top and gene_col in df.columns:
        # Label top-N most-significant proteins among the highlighted set.
        sig_idx = np.where(up_mask | down_mask)[0]
        if sig_idx.size:
            top = sig_idx[np.argsort(p[sig_idx])][:label_top]
            for i in top:
                ax.annotate(
                    str(df[gene_col].iloc[i]),
                    (fc[i], nlogp[i]),
                    fontsize=7, alpha=0.85,
                    xytext=(3, 3), textcoords="offset points",
                )

    return ax


@register_function(
    aliases=["protein_missing_pattern_plot", "missing_pattern_plot"],
    category="visualization",
    description=(
        "Heatmap of the proteomics missingness pattern. Rows are "
        "proteins (sorted by overall missingness), columns are samples. "
        "Useful for diagnosing MNAR vs MCAR before imputation."
    ),
    examples=["ov.protein.missing_pattern_plot(adata)"],
)
def missing_pattern_plot(
    adata,
    *,
    ax: Optional["matplotlib.axes.Axes"] = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    cmap: str = "Greys",
    max_proteins: int = 1000,
):
    """Heatmap of per-protein × per-sample missingness — diagnose MNAR vs MCAR."""
    import matplotlib.pyplot as plt
    miss = np.isnan(adata.X).T.astype(float)  # proteins × samples
    order = np.argsort(-miss.mean(axis=1))
    miss = miss[order][:max_proteins]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(miss, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xlabel("samples"); ax.set_ylabel(f"proteins (top {max_proteins} missing)")
    ax.set_xticks(np.arange(adata.n_obs))
    ax.set_xticklabels(adata.obs_names, rotation=90, fontsize=6)
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, label="missing")
    return ax


@register_function(
    aliases=["protein_abundance_rank_plot", "abundance_rank_plot"],
    category="visualization",
    description=(
        "Per-sample rank-vs-log-intensity diagnostic — one line per "
        "sample, sorted by descending abundance. Use to spot under-/"
        "over-loaded samples before normalization."
    ),
    examples=["ov.protein.abundance_rank_plot(adata)"],
)
def abundance_rank_plot(
    adata,
    *,
    ax: Optional["matplotlib.axes.Axes"] = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    log: bool = True,
    color_by: Optional[str] = None,
):
    """Per-sample rank-vs-log-intensity diagnostic; spots under/over-loaded samples."""
    import matplotlib.pyplot as plt
    X = adata.X.astype(float)
    if log:
        X = np.log2(np.where(X > 0, X, np.nan))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    palette = None
    if color_by and color_by in adata.obs.columns:
        groups = adata.obs[color_by].astype(str)
        unique = pd.unique(groups)
        palette = {g: c for g, c in zip(
            unique, plt.cm.tab10.colors[: len(unique)]  # type: ignore[attr-defined]
        )}
    for i, sample in enumerate(adata.obs_names):
        row = X[i]
        row = row[~np.isnan(row)]
        row = np.sort(row)[::-1]
        if row.size == 0:
            continue
        color = palette[str(adata.obs[color_by].iloc[i])] if palette else None
        ax.plot(np.arange(row.size), row, lw=0.6, alpha=0.7,
                color=color, label=sample if not palette else None)
    ax.set_xlabel("rank (sorted high → low)")
    ax.set_ylabel("log2 intensity" if log else "intensity")
    if not palette and adata.n_obs <= 20:
        ax.legend(fontsize=6, ncol=2)
    return ax
