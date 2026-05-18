r"""Plotting helpers for in-silico gene-perturbation results.

Targets the dictionary returned by :meth:`omicverse.llm.SCLLMManager.perturb_genes`
(see also :meth:`omicverse.llm.geneformer_model.GeneformerModel.perturb_genes`),
which has the schema:

    {
        "cosine_similarities": DataFrame (cell × gene),
        "stats":                DataFrame (gene × {mean, std, n_cells_perturbed, ...}),
        "original_embeddings":  ndarray (n_cells × d),
        "perturbed_embeddings": dict[gene → ndarray (n_cells × d)],
        ...
    }

Two helpers:

* :func:`perturbation_shift_violin` — per-gene violin of per-cell
  cosine-similarity drops. Lower bars = stronger perturbation effect.
* :func:`perturbation_embedding_shift` — UMAP / PCA scatter overlaid with
  arrows from each cell's original embedding position to its post-perturbation
  position, projected onto an existing 2-D basis (``adata.obsm[basis]``).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._registry import register_function


# --------------------------------------------------------------------------- #
# violin: per-gene cosine-similarity distribution                              #
# --------------------------------------------------------------------------- #


@register_function(
    aliases=["扰动小提琴图", "perturbation_shift_violin"],
    category="pl",
    description=(
        "Violin plot of per-cell cosine similarity (original ↔ perturbed) "
        "for each target gene returned by ov.llm.SCLLMManager.perturb_genes. "
        "Lower values = bigger embedding shift = stronger perturbation effect."
    ),
    examples=[
        "result = manager.perturb_genes(adata, ['CD3D', 'PTPRC'], perturb_type='delete')",
        "ov.pl.perturbation_shift_violin(result)",
    ],
    related=["llm.SCLLMManager.perturb_genes", "pl.perturbation_embedding_shift"],
)
def perturbation_shift_violin(
    result: Dict,
    *,
    adata: Optional[AnnData] = None,
    groupby: Optional[str] = None,
    figsize: tuple[float, float] = (6.0, 3.5),
    color: str = "#5B8FF9",
    order: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
):
    r"""Violin of per-cell cosine similarity for each perturbed gene.

    Parameters
    ----------
    result : dict
        The dictionary returned by ``manager.perturb_genes(...)``. Must
        contain ``'cosine_similarities'`` — a ``DataFrame`` with one column
        per perturbed gene (cells × genes).
    adata, groupby : AnnData and str
        Optional — when both are passed and ``groupby`` is a column in
        ``adata.obs``, the violin for each gene is split into one violin
        per category (e.g. ``groupby='cell_type'`` reveals lineage-
        specific perturbation effects).
    figsize : tuple
        Figure size in inches.
    color : str
        Default violin fill colour when ``groupby`` is not provided.
    order : sequence of str or None
        Gene plot order. ``None`` = sort by mean cosine ascending (strongest
        effect leftmost).
    title : str or None
        Optional title.
    ax : matplotlib Axes or None
        Pre-allocated axes; ``None`` creates a new figure.

    Returns
    -------
    fig, ax
    """
    cos = result.get("cosine_similarities")
    if cos is None or not hasattr(cos, "columns"):
        raise KeyError(
            "result['cosine_similarities'] is missing or not a DataFrame — "
            "this helper expects ov.llm.SCLLMManager.perturb_genes() output."
        )
    df = pd.DataFrame(cos).dropna(how="all")
    if df.empty:
        raise ValueError("All per-cell cosine similarities are NaN — no cell carried any of the requested genes.")

    if order is None:
        order = df.mean(axis=0, skipna=True).sort_values().index.tolist()
    df = df[order]
    # Drop genes that ended up entirely NaN (no cell carried the gene — typical
    # for narrowly-expressed TFs when max_ncells is small).
    nonempty = [c for c in df.columns if df[c].notna().any()]
    skipped = [c for c in df.columns if c not in nonempty]
    if skipped:
        import warnings as _warnings
        _warnings.warn(
            f"Skipping genes with no perturbed cells: {skipped} "
            f"(increase max_ncells or pick more broadly-expressed targets)."
        )
    df = df[nonempty]
    order = nonempty
    if df.empty:
        raise ValueError("No genes have any perturbed cells to plot.")

    # If a groupby is requested, render one violin per category per gene.
    if groupby is not None and adata is not None:
        if groupby not in adata.obs.columns:
            raise KeyError(f"adata.obs[{groupby!r}] not found")
        from ._palette import palette_28
        group = adata.obs[groupby].astype(str)
        # Align to the cells scored by perturb_genes (which used obs_names index)
        cell_idx = result.get("cell_indices")
        if cell_idx is not None:
            group = group.iloc[cell_idx].reset_index(drop=True)
            df_aligned = df.reset_index(drop=True)
        else:
            df_aligned = df.copy()
            group = group.loc[df_aligned.index]
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        cats = sorted(group.unique())
        pal = list(palette_28)
        cat_to_color = {c: pal[i % len(pal)] for i, c in enumerate(cats)}
        n_g = len(order)
        n_c = len(cats)
        width = 0.8 / max(n_c, 1)
        # Cap each gene's group set to top-K most-perturbed categories (by mean
        # |1-cos| over that gene), so the plot stays readable with many categories.
        max_groups_per_gene = 6
        for gi, g in enumerate(order):
            shifts = (1 - df_aligned[g]).abs()
            by_cat = shifts.groupby(group).mean().dropna()
            top_cats = by_cat.sort_values(ascending=False).head(max_groups_per_gene).index.tolist()
            for ci, c in enumerate(top_cats):
                vals = df_aligned[g][(group == c) & df_aligned[g].notna()].values
                if vals.size < 2:
                    continue
                pos = gi + (ci - (len(top_cats) - 1) / 2) * width
                parts = ax.violinplot(
                    [vals], positions=[pos], widths=width * 0.9,
                    showmeans=True, showextrema=False,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(cat_to_color[c])
                    body.set_edgecolor("#333333")
                    body.set_alpha(0.75)
                if "cmeans" in parts:
                    parts["cmeans"].set_color("#222222")
            # legend (once)
            if gi == 0:
                for c in top_cats:
                    ax.scatter([], [], s=30, c=cat_to_color[c], label=str(c))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_xlabel("Perturbed gene")
        ax.set_ylabel("Cosine similarity\n(original ↔ perturbed)")
        ax.axhline(1.0, color="#888888", linewidth=0.5, linestyle="--")
        for spine_name in ("top", "right"):
            ax.spines[spine_name].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(
                loc="center left", bbox_to_anchor=(1.02, 0.5),
                fontsize=7, frameon=False, title=groupby,
            )
        if title:
            ax.set_title(title, fontsize=11)
        elif result.get("perturb_type"):
            ax.set_title(
                f"Geneformer in-silico {result['perturb_type']} — by {groupby}",
                fontsize=11,
            )
        return fig, ax

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    data_per_gene = [df[c].dropna().values for c in order]
    parts = ax.violinplot(data_per_gene, showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.7)
    if "cmeans" in parts:
        parts["cmeans"].set_color("#222222")

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylabel("Cosine similarity\n(original ↔ perturbed)")
    ax.set_xlabel("Perturbed gene")
    ax.axhline(1.0, color="#888888", linewidth=0.5, linestyle="--")
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11)
    elif result.get("perturb_type"):
        ax.set_title(
            f"Geneformer in-silico {result['perturb_type']} — per-cell cosine similarity",
            fontsize=11,
        )

    return fig, ax


# --------------------------------------------------------------------------- #
# downstream genes: bar plot of top-shifted genes                              #
# --------------------------------------------------------------------------- #


@register_function(
    aliases=["扰动下游基因", "perturbation_top_downstream_genes", "perturbation_top_genes"],
    category="pl",
    description=(
        "Bar plot of the genes whose contextual embedding shifted most after "
        "in-silico knockout of a target transcription factor — i.e. candidate "
        "downstream targets ranked by 1 - cos(orig_emb, perturbed_emb)."
    ),
    examples=[
        "result = manager.perturb_genes(adata, ['PAX5'], perturb_type='delete')",
        "ov.pl.perturbation_top_downstream_genes(result, gene='PAX5', top_n=20)",
    ],
    related=["llm.SCLLMManager.perturb_genes", "pl.perturbation_shift_violin"],
)
def perturbation_top_downstream_genes(
    result: Dict,
    *,
    gene: str,
    top_n: int = 20,
    figsize: tuple[float, float] = (6.5, 5.0),
    bar_color: str = "#cd3a3a",
    min_cells: int = 5,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
):
    r"""Plot the top-N most-shifted downstream genes after a TF knockout.

    Requires ``manager.perturb_genes(..., compute_gene_shifts=True)`` (the
    default). The shift metric per downstream gene is
    ``mean_cells(1 - cos(emb_orig, emb_perturbed))`` — averaged over the
    cells where both the target and the downstream gene were present in
    the rank-encoded input. Higher = larger context shift = stronger
    model-predicted downstream effect.

    Parameters
    ----------
    result : dict
        Output of ``manager.perturb_genes(...)``. Must contain
        ``'gene_shifts'`` — a dict keyed by target-gene label.
    gene : str
        Which target gene's perturbation to summarise.
    top_n : int
        Number of downstream genes to display.
    figsize, bar_color, title, ax
        Standard matplotlib styling.
    min_cells : int
        Minimum number of cells in which a downstream gene must be present
        for it to enter the ranking. Filters out noise from rare genes.

    Returns
    -------
    fig, ax
    """
    shifts_by_target = result.get("gene_shifts") or {}
    if gene not in shifts_by_target:
        raise KeyError(
            f"result['gene_shifts'] has no entry for {gene!r} (keys: "
            f"{list(shifts_by_target.keys())}). Re-run perturb_genes with "
            f"compute_gene_shifts=True (default)."
        )
    df = pd.DataFrame(shifts_by_target[gene])
    if df.empty:
        raise ValueError(f"No downstream-gene shifts recorded for {gene!r}.")
    df = df[df["n_cells"] >= min_cells].sort_values("mean_shift", ascending=False).head(top_n)
    if df.empty:
        raise ValueError(f"No downstream genes pass min_cells={min_cells} filter.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    y_pos = np.arange(len(df))[::-1]  # tallest bar at the top
    labels = df["gene"].astype(str).tolist()
    ax.barh(y_pos, df["mean_shift"].values, color=bar_color, alpha=0.8, edgecolor="#333333")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean 1 − cos(orig, perturbed)\n(higher = more shifted)")
    ax.set_ylabel("Downstream gene")
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    ax.set_title(title or f"Top {len(df)} downstream genes shifted by {result.get('perturb_type', 'perturbation')} of {gene}", fontsize=11)
    return fig, ax


# --------------------------------------------------------------------------- #
# embedding shift: arrows on a 2-D basis                                       #
# --------------------------------------------------------------------------- #


@register_function(
    aliases=["扰动嵌入位移", "perturbation_embedding_shift"],
    category="pl",
    description=(
        "Project each cell's original and perturbed Geneformer embedding onto "
        "an existing 2-D basis (e.g. adata.obsm['X_umap']) and draw an arrow "
        "from the original to the perturbed position. Shows the *direction* of "
        "cell-state shift in response to the in-silico knockout."
    ),
    examples=[
        "ov.pl.perturbation_embedding_shift(adata, result, gene='CD3D', basis='X_umap')",
    ],
    related=["llm.SCLLMManager.perturb_genes", "pl.perturbation_shift_violin"],
)
def perturbation_embedding_shift(
    adata: AnnData,
    result: Dict,
    *,
    gene: str,
    basis: str = "X_umap",
    color: Optional[str] = None,
    figsize: tuple[float, float] = (5.0, 5.0),
    arrow_color: str = "#cd3a3a",
    arrow_alpha: float = 0.5,
    arrow_lw: float = 0.6,
    point_size: float = 6.0,
    max_arrows: int = 300,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
):
    r"""Plot per-cell embedding shift arrows on an existing 2-D basis.

    Projects the (n_cells × d) Geneformer original / perturbed embeddings onto
    ``adata.obsm[basis]`` by least-squares (one linear map fit on the original
    pair so both endpoints share the same projection). Then draws an arrow
    from each cell's original position to its post-perturbation position.

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.obsm[basis]`` (e.g. precomputed UMAP).
    result : dict
        Output of ``manager.perturb_genes(...)`` — needs both
        ``original_embeddings`` and ``perturbed_embeddings[gene]``.
    gene : str
        Which target gene's perturbation to visualise (key in
        ``result['perturbed_embeddings']``).
    basis : str
        ``adata.obsm`` key for the 2-D layout. Default ``'X_umap'``.
    color : str or None
        Optional ``adata.obs`` column to colour the scatter background.
    figsize : tuple
        Figure size in inches.
    arrow_color / arrow_alpha / arrow_lw : float
        Arrow styling.
    point_size : float
        Scatter marker size for the cell positions.
    max_arrows : int
        Cap on number of arrows drawn (random subsample to keep plot readable).
    ax : matplotlib Axes or None
        Pre-allocated axes.
    title : str or None
        Optional title.

    Returns
    -------
    fig, ax
    """
    if basis not in adata.obsm:
        raise KeyError(f"adata.obsm[{basis!r}] not found")
    perturbed = result.get("perturbed_embeddings", {})
    if gene not in perturbed:
        raise KeyError(f"result['perturbed_embeddings'] has no {gene!r} (keys: {list(perturbed.keys())})")
    orig = np.asarray(result["original_embeddings"])
    pert = np.asarray(perturbed[gene])
    cell_idx = np.asarray(result.get("cell_indices", np.arange(orig.shape[0])))

    layout = np.asarray(adata.obsm[basis])
    if layout.shape[1] < 2:
        raise ValueError(f"adata.obsm[{basis!r}] must be ≥ 2-D")
    Y = layout[cell_idx, :2]

    # Solve X · W = Y on original embeddings; reuse W to project perturbed.
    # Centre both sides so the projection has no bias term.
    Xc = orig - orig.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    W, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)
    proj_orig = (orig - orig.mean(axis=0)) @ W + Y.mean(axis=0)
    proj_pert = (pert - orig.mean(axis=0)) @ W + Y.mean(axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Background scatter — either uniform grey or coloured by adata.obs[color].
    if color is not None and color in adata.obs:
        from ._palette import palette_28

        cats = pd.Categorical(adata.obs[color].iloc[cell_idx])
        pal = adata.uns.get(f"{color}_colors")
        if pal is None or len(pal) < len(cats.categories):
            pal = list(palette_28)
        c_to_color = {c: pal[i % len(pal)] for i, c in enumerate(cats.categories)}
        colors = [c_to_color.get(c, "#bdbdbd") for c in cats]
        ax.scatter(Y[:, 0], Y[:, 1], s=point_size, c=colors, alpha=0.6, linewidths=0)
        # legend
        for c, col in c_to_color.items():
            ax.scatter([], [], s=20, c=col, label=str(c))
        ax.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False
        )
    else:
        ax.scatter(Y[:, 0], Y[:, 1], s=point_size, c="#bdbdbd", alpha=0.6, linewidths=0)

    # Arrows — sub-sample if too many.
    rng = np.random.default_rng(0)
    n = proj_orig.shape[0]
    if n > max_arrows:
        sel = rng.choice(n, max_arrows, replace=False)
    else:
        sel = np.arange(n)
    # Only draw arrows for cells that actually had the gene (cosine != NaN).
    if "cosine_similarities" in result and gene in result["cosine_similarities"].columns:
        cs = np.asarray(result["cosine_similarities"][gene].values)
        sel = sel[~np.isnan(cs[sel])]

    dx = proj_pert[sel, 0] - proj_orig[sel, 0]
    dy = proj_pert[sel, 1] - proj_orig[sel, 1]
    ax.quiver(
        proj_orig[sel, 0], proj_orig[sel, 1], dx, dy,
        angles="xy", scale_units="xy", scale=1,
        color=arrow_color, alpha=arrow_alpha, width=0.003,
        linewidth=arrow_lw, headwidth=4, headlength=5,
    )

    ax.set_xlabel(f"{basis} 1")
    ax.set_ylabel(f"{basis} 2")
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    ax.set_title(title or f"In-silico {result.get('perturb_type', 'perturbation')} of {gene}", fontsize=11)
    return fig, ax
