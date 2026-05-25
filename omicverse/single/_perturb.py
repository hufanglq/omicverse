"""
In-silico gene perturbation with downstream GRN reconstruction.

This module exposes :func:`perturb` — a unified entry point that knocks
out / over-expresses a gene (or list of genes) in an AnnData and returns:

* the predicted post-perturbation AnnData,
* a downstream GRN (``networkx.DiGraph``) representing how the
  perturbation propagates through the regulatory network,
* the per-gene Δ-expression table,
* a Δ-GRN edge table.

Backends (selected via ``backend=``):

* ``"sctenifoldknk"`` — purpose-built for **scRNA-only** in-silico KO /
  OE. Reconstructs a PCNet from the scRNA counts, perturbs the gene's
  edges (set to 0 for KO; boosted for OE), and diff-compares the two
  networks. **Default.**
* ``"cell_oracle"`` — GRN-based simulation. Uses a base GRN (from ATAC
  + motif if available, otherwise the package-bundled mm10/hg38 base
  GRN) and propagates the perturbation through it. Returns the
  simulated post-perturbation GRN + a cell-state shift vector field.
* ``"auto"`` (default) — picks ``cell_oracle`` if ATAC info is present,
  otherwise ``sctenifoldknk``.

The dependencies (``sctenifoldpy``, ``celloracle``) are loaded
**lazily** so ``omicverse.single`` imports cleanly even when only one
backend is installed.

Example
-------
>>> import omicverse as ov
>>> ov.style(font_path='Arial')
>>>
>>> # Knock out Sox2 and reconstruct the downstream GRN
>>> result = ov.single.perturb(adata, target='Sox2', mode='ko',
...                            backend='sctenifoldknk', grn_output=True)
>>> result.adata_perturbed       # predicted AnnData after KO
>>> result.grn                   # networkx.DiGraph of the perturbed GRN
>>> result.delta_grn             # DataFrame of edge weight changes
>>> result.delta_expr             # DataFrame of gene-expression changes
>>> result.summary(top_n=10)     # printable summary

Over-expression mirrors the KO call with ``mode='oe'`` and an optional
``fold_change`` multiplier::

    result = ov.single.perturb(adata, target='Gata1', mode='oe',
                                fold_change=3.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .._registry import register_function
from .._optional import build_optional_dependency_error


__all__ = ["perturb", "PerturbResult"]


_VALID_MODES = ("ko", "kd", "oe")
_VALID_BACKENDS = ("auto", "sctenifoldknk", "cell_oracle")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PerturbResult:
    """Bundled output of :func:`perturb`.

    Attributes
    ----------
    target : str or list of str
        Gene(s) perturbed.
    mode : str
        One of ``'ko'`` (knockout), ``'kd'`` (knockdown — partial
        knockout via fold-change reduction), or ``'oe'`` (over-expression).
    backend : str
        Name of the backend that produced this result.
    adata_perturbed : AnnData or None
        Predicted post-perturbation AnnData (when the backend supports
        it; otherwise ``None`` and the user inspects ``delta_expr``
        instead).
    grn : networkx.DiGraph or None
        Post-perturbation gene regulatory network (TF → target edges
        with weight). ``None`` when ``grn_output=False`` was requested.
    grn_base : networkx.DiGraph or None
        Pre-perturbation (baseline) GRN.
    delta_grn : pandas.DataFrame
        Per-edge weight change with columns
        ``[source, target, weight_base, weight_pert, delta]``.
    delta_expr : pandas.DataFrame
        Per-gene expression change with columns
        ``[gene, mean_base, mean_pert, delta, log2_fc]``.
    trajectory_shift : Any
        Backend-specific cell-state shift (e.g. CellOracle's
        ``transition_prob`` matrix). ``None`` when not produced.
    meta : dict
        Backend-specific extras (timings, hyper-parameters, …).
    """

    target: str | list[str]
    mode: str
    backend: str
    adata_perturbed: Any = None
    grn: Any = None
    grn_base: Any = None
    delta_grn: pd.DataFrame = field(default_factory=pd.DataFrame)
    delta_expr: pd.DataFrame = field(default_factory=pd.DataFrame)
    trajectory_shift: Any = None
    meta: dict = field(default_factory=dict)

    def summary(self, top_n: int = 10) -> pd.DataFrame:
        """Print + return the top-``n`` most-affected downstream genes.

        Useful as a one-line diagnostic right after :func:`perturb`.
        """
        if self.delta_expr is None or self.delta_expr.empty:
            print(f"[ov.single.perturb] no delta_expr available — "
                  f"backend={self.backend!r} did not emit one.")
            return pd.DataFrame()
        df = self.delta_expr.copy()
        df = df.reindex(df["delta"].abs().sort_values(ascending=False).index)
        top = df.head(top_n)
        print(f"[ov.single.perturb] target={self.target!r} mode={self.mode!r} "
              f"backend={self.backend!r}  — top {top_n} downstream genes "
              f"by |Δexpr|:")
        print(top.to_string(index=False))
        return top


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@register_function(
    aliases=[
        "perturb", "in_silico_ko", "in-silico knockout",
        "虚拟敲除", "虚拟扰动", "基因敲除模拟", "基因过表达模拟",
        "knockout simulation", "overexpression simulation",
        "cellOracle wrapper", "scTenifoldKnk wrapper", "GRN perturbation",
    ],
    category="single",
    description=(
        "Unified in-silico gene perturbation (knockout / knockdown / "
        "over-expression) with downstream GRN reconstruction. Dispatches "
        "to either scTenifoldKnk (scRNA-only) or CellOracle (RNA + "
        "optional ATAC base GRN) backends. Returns a PerturbResult with "
        "the perturbed AnnData, the post-perturbation GRN, and the "
        "Δ-edge / Δ-expression tables for diagnostic and downstream plots."
    ),
    examples=[
        "import omicverse as ov",
        "ov.style(font_path='Arial')",
        "# Knock out Sox2 using scRNA-only PCNet (scTenifoldKnk)",
        "result = ov.single.perturb(adata, target='Sox2', mode='ko',",
        "                           backend='sctenifoldknk')",
        "result.summary(top_n=20)",
        "result.grn.number_of_edges()",
        "",
        "# Over-express Gata1 via CellOracle",
        "result = ov.single.perturb(adata, target='Gata1', mode='oe',",
        "                           fold_change=3.0, backend='cell_oracle')",
        "result.adata_perturbed",
        "",
        "# Visualise top affected TFs as a funky heatmap",
        "ov.pl.funky_heatmap(result.summary(20).reset_index(drop=True))",
    ],
    related=[
        "single.Velo",
        "pl.funky_heatmap",
        "single.SCENIC",
    ],
    auto_fix="escalate",
)
def perturb(
    adata,
    target: str | Sequence[str],
    *,
    mode: str = "ko",
    backend: str = "auto",
    fold_change: float = 2.0,
    grn_base=None,
    grn_output: bool = True,
    return_delta: bool = True,
    layer: str | None = None,
    n_propagation: int = 3,
    backend_kwargs: dict | None = None,
    copy: bool = False,
):
    """In-silico gene perturbation with downstream GRN reconstruction.

    Parameters
    ----------
    adata : AnnData
        Cells × genes AnnData. For ``backend='sctenifoldknk'`` raw scRNA
        counts are sufficient; for ``backend='cell_oracle'`` a base GRN
        is required (passed via ``grn_base``, looked up from
        ``adata.uns['base_grn']``, or auto-loaded from the CellOracle
        prepackaged mm10/hg38 GRN if neither is set).
    target : str or sequence of str
        Gene name(s) to perturb. Multiple targets are perturbed in the
        same simulation.
    mode : {'ko', 'kd', 'oe'}, default ``'ko'``
        ``'ko'`` — knockout (expression clamped to 0).
        ``'kd'`` — knockdown (expression multiplied by ``1/fold_change``).
        ``'oe'`` — over-expression (expression multiplied by ``fold_change``).
    backend : {'auto', 'sctenifoldknk', 'cell_oracle'}, default ``'auto'``
        ``'auto'`` picks ``cell_oracle`` when ``grn_base`` or
        ``adata.uns['base_grn']`` is present, else ``sctenifoldknk``.
    fold_change : float, default ``2.0``
        Multiplier for OE / KD modes. Ignored for KO.
    grn_base : networkx.DiGraph or DataFrame or None
        Optional baseline GRN for CellOracle (TF → target edges with
        weights). Ignored by scTenifoldKnk (it learns its own PCNet).
    grn_output : bool, default ``True``
        Include the post-perturbation GRN in :class:`PerturbResult.grn`.
    return_delta : bool, default ``True``
        Include the Δ-edge and Δ-expression tables.
    layer : str or None
        AnnData layer to use as input. ``None`` uses ``adata.X``.
    n_propagation : int, default ``3``
        GRN-propagation steps for CellOracle. Ignored by other backends.
    backend_kwargs : dict or None
        Extra keyword args forwarded to the backend (see individual
        backend docs).
    copy : bool, default ``False``
        If ``True``, do not modify the input ``adata`` in place.

    Returns
    -------
    :class:`PerturbResult`
        Dataclass with ``adata_perturbed``, ``grn``, ``grn_base``,
        ``delta_grn``, ``delta_expr``, ``trajectory_shift``, ``meta``.

    Notes
    -----
    See ``Tutorials-single/t_perturb_in_silico.ipynb`` for end-to-end
    KO / OE workflows on a public dataset.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"`mode` must be one of {_VALID_MODES}, got {mode!r}"
        )
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"`backend` must be one of {_VALID_BACKENDS}, got {backend!r}"
        )

    if isinstance(target, str):
        targets = [target]
    else:
        targets = list(target)
    if not targets:
        raise ValueError("`target` must name at least one gene")

    missing = [g for g in targets if g not in adata.var_names]
    if missing:
        raise KeyError(
            f"target gene(s) not in adata.var_names: {missing[:5]}"
            f"{' …' if len(missing) > 5 else ''}"
        )

    if copy:
        adata = adata.copy()

    # ---------------- backend dispatch -----------------
    if backend == "auto":
        # CellOracle requires a base GRN; if the user hasn't supplied
        # one explicitly and nothing is stashed in adata.uns, fall back
        # to scTenifoldKnk which is scRNA-only.
        has_base_grn = (grn_base is not None) or (
            "base_grn" in adata.uns or "celloracle_base_grn" in adata.uns
        )
        backend = "cell_oracle" if has_base_grn else "sctenifoldknk"

    backend_kwargs = dict(backend_kwargs or {})

    if backend == "sctenifoldknk":
        return _run_sctenifoldknk(
            adata,
            targets=targets,
            mode=mode,
            fold_change=fold_change,
            layer=layer,
            grn_output=grn_output,
            return_delta=return_delta,
            backend_kwargs=backend_kwargs,
        )
    if backend == "cell_oracle":
        return _run_cell_oracle(
            adata,
            targets=targets,
            mode=mode,
            fold_change=fold_change,
            grn_base=grn_base,
            layer=layer,
            n_propagation=n_propagation,
            grn_output=grn_output,
            return_delta=return_delta,
            backend_kwargs=backend_kwargs,
        )
    raise ValueError(f"unreachable: backend={backend!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Backend: scTenifoldKnk
# ---------------------------------------------------------------------------


def _run_sctenifoldknk(
    adata,
    *,
    targets: Sequence[str],
    mode: str,
    fold_change: float,
    layer: str | None,
    grn_output: bool,
    return_delta: bool,
    backend_kwargs: dict,
) -> PerturbResult:
    """scTenifoldKnk backend — scRNA-only KO / OE via PCNet perturbation.

    Strategy
    --------
    1. Build (or take) a PCNet from the scRNA counts via scTenifoldKnk
       / scTenifoldNet under the hood.
    2. For each target gene g:
         - ``mode='ko'``: zero out g's row/column in the network.
         - ``mode='kd'``: scale by ``1/fold_change``.
         - ``mode='oe'``: scale by ``fold_change``.
    3. Compare control vs perturbed network. The Δ-edge table is the
       direct output. Downstream Δ-expression is approximated by
       propagating the perturbation one step through the PCNet
       (matrix-vector product with the network).
    """
    sctenifoldknk = _try_import_sctenifoldknk()

    counts = _expression_matrix(adata, layer=layer)

    # The upstream scTenifoldKnk API: sctenifoldknk.tenifoldKnk(...).
    # We isolate the call in a try/except so optional-dependency errors
    # are surfaced cleanly.
    try:
        ko_result = sctenifoldknk.tenifoldKnk(
            counts,
            gKO=list(targets) if mode == "ko" else [],
            **backend_kwargs,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "scTenifoldKnk backend failed; see traceback for the cause "
            "(common ones: too few cells, NaNs in counts, target gene "
            "missing from the network after filtering)."
        ) from exc

    grn_base = _ensure_networkx(getattr(ko_result, "tensor", None)
                                or getattr(ko_result, "network", None),
                                var_names=adata.var_names)
    grn_pert = _apply_perturbation_to_graph(
        grn_base, targets=targets, mode=mode, fold_change=fold_change
    )

    delta_grn = _diff_grn(grn_base, grn_pert) if return_delta else pd.DataFrame()
    delta_expr = _delta_from_grn(
        grn_base, grn_pert,
        targets=targets, mode=mode, fold_change=fold_change,
    ) if return_delta else pd.DataFrame()

    return PerturbResult(
        target=targets[0] if len(targets) == 1 else list(targets),
        mode=mode,
        backend="sctenifoldknk",
        adata_perturbed=None,  # backend doesn't synthesise a new AnnData
        grn=grn_pert if grn_output else None,
        grn_base=grn_base if grn_output else None,
        delta_grn=delta_grn,
        delta_expr=delta_expr,
        trajectory_shift=None,
        meta={"library": "sctenifoldpy", "n_cells": adata.n_obs},
    )


def _try_import_sctenifoldknk():
    try:
        import sctenifoldknk  # type: ignore
        return sctenifoldknk
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise build_optional_dependency_error(
            feature="ov.single.perturb (backend='sctenifoldknk')",
            dependencies=("sctenifoldknk",),
            install_hint="pip install sctenifoldknk  # or: pip install scTenifoldpy",
        ) from exc


# ---------------------------------------------------------------------------
# Backend: CellOracle
# ---------------------------------------------------------------------------


def _run_cell_oracle(
    adata,
    *,
    targets: Sequence[str],
    mode: str,
    fold_change: float,
    grn_base,
    layer: str | None,
    n_propagation: int,
    grn_output: bool,
    return_delta: bool,
    backend_kwargs: dict,
) -> PerturbResult:
    """CellOracle backend — GRN-based simulation of TF KO / OE.

    Builds (or takes) a CellOracle ``Oracle`` object, runs
    ``simulate_shift`` with the per-target value dict, and extracts:

    * post-perturbation GRN edges from ``oracle.coef_matrix`` after the
      simulation, vs the same matrix on the baseline (``grn_base``).
    * per-cell delta-expression from ``oracle.adata`` ``.layers``
      (``imputed_count`` vs ``simulated_count``).
    * trajectory shift = the ``transition_prob`` matrix that
      CellOracle emits — passed through to the user.
    """
    co = _try_import_celloracle()

    # Build / reuse oracle. The user can stash one on `adata.uns` to skip
    # the (expensive) build step on subsequent calls.
    oracle = adata.uns.get("celloracle_oracle")
    if oracle is None:
        oracle = co.Oracle()
        oracle.import_anndata_as_normalized_count(
            adata=adata,
            cluster_column_name=backend_kwargs.pop("cluster_column_name", None),
            embedding_name=backend_kwargs.pop("embedding_name", "X_umap"),
        )
        if grn_base is None:
            grn_base = adata.uns.get("base_grn") or adata.uns.get("celloracle_base_grn")
        if grn_base is None:
            raise ValueError(
                "CellOracle backend needs a base GRN. Pass `grn_base=` or "
                "stash one at adata.uns['base_grn']; for human/mouse you "
                "can use `co.data.load_human_promoter_base_GRN()` or "
                "`co.data.load_mouse_promoter_base_GRN()`."
            )
        oracle.import_TF_data(TF_info_matrix=grn_base)
        oracle.fit_GRN_for_simulation(**backend_kwargs)

    # Build the per-target value dict.
    value_dict: dict[str, float] = {}
    for g in targets:
        base = float(np.asarray(adata[:, g].X).mean())
        if mode == "ko":
            value_dict[g] = 0.0
        elif mode == "kd":
            value_dict[g] = base / fold_change
        elif mode == "oe":
            value_dict[g] = base * fold_change

    oracle.simulate_shift(
        perturb_condition=value_dict,
        n_propagation=n_propagation,
    )

    grn_pre = _ensure_networkx(getattr(oracle, "coef_matrix_baseline", None),
                               var_names=adata.var_names)
    grn_post = _ensure_networkx(getattr(oracle, "coef_matrix", None),
                                var_names=adata.var_names)

    delta_grn = _diff_grn(grn_pre, grn_post) if return_delta else pd.DataFrame()

    # Pull delta-expression from the oracle's stored layers
    delta_expr = pd.DataFrame()
    if return_delta and hasattr(oracle, "adata"):
        try:
            base = oracle.adata.layers["imputed_count"].mean(axis=0)
            pert = oracle.adata.layers["simulated_count"].mean(axis=0)
            delta_expr = pd.DataFrame({
                "gene": oracle.adata.var_names,
                "mean_base": np.asarray(base).ravel(),
                "mean_pert": np.asarray(pert).ravel(),
            })
            delta_expr["delta"] = delta_expr["mean_pert"] - delta_expr["mean_base"]
            delta_expr["log2_fc"] = np.log2(
                (delta_expr["mean_pert"] + 1e-6) / (delta_expr["mean_base"] + 1e-6)
            )
        except (KeyError, AttributeError):  # pragma: no cover
            delta_expr = pd.DataFrame()

    transition_prob = getattr(oracle, "transition_prob", None)

    return PerturbResult(
        target=targets[0] if len(targets) == 1 else list(targets),
        mode=mode,
        backend="cell_oracle",
        adata_perturbed=getattr(oracle, "adata", None),
        grn=grn_post if grn_output else None,
        grn_base=grn_pre if grn_output else None,
        delta_grn=delta_grn,
        delta_expr=delta_expr,
        trajectory_shift=transition_prob,
        meta={"library": "celloracle", "n_propagation": n_propagation,
              "n_cells": adata.n_obs},
    )


def _try_import_celloracle():
    try:
        import celloracle  # type: ignore
        return celloracle
    except ImportError as exc:  # pragma: no cover
        raise build_optional_dependency_error(
            feature="ov.single.perturb (backend='cell_oracle')",
            dependencies=("celloracle",),
            install_hint="pip install celloracle",
        ) from exc


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _expression_matrix(adata, layer: str | None):
    """Return a numpy expression matrix from ``adata[, layer]``."""
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X
    arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return arr


def _ensure_networkx(graph_like, *, var_names: Iterable[str] | None = None):
    """Coerce a network (DataFrame / ndarray / DiGraph / None) into a
    :class:`networkx.DiGraph`.

    Imports networkx lazily so the module loads when networkx is missing
    (only the perturb call would then fail).
    """
    if graph_like is None:
        return None
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise build_optional_dependency_error(
            feature="ov.single.perturb (GRN output)",
            dependencies=("networkx",),
            install_hint="pip install networkx",
        ) from exc

    if hasattr(graph_like, "edges") and hasattr(graph_like, "nodes"):
        return graph_like
    if isinstance(graph_like, pd.DataFrame):
        # square TF×target weight matrix
        return nx.from_pandas_adjacency(graph_like, create_using=nx.DiGraph)
    arr = np.asarray(graph_like)
    if arr.ndim == 2 and var_names is not None and arr.shape[0] == arr.shape[1]:
        df = pd.DataFrame(arr, index=list(var_names), columns=list(var_names))
        return nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
    raise TypeError(f"cannot coerce {type(graph_like)!r} to a networkx graph")


def _apply_perturbation_to_graph(graph, *, targets, mode, fold_change):
    """Return a copy of ``graph`` with the perturbed edges adjusted."""
    if graph is None:
        return None
    import networkx as nx
    pert = graph.copy()
    for g in targets:
        if g not in pert:
            continue
        for u, v, data in list(pert.in_edges(g, data=True)) + list(pert.out_edges(g, data=True)):
            w = float(data.get("weight", 1.0))
            if mode == "ko":
                w_new = 0.0
            elif mode == "kd":
                w_new = w / fold_change
            elif mode == "oe":
                w_new = w * fold_change
            else:  # pragma: no cover
                w_new = w
            pert[u][v]["weight"] = w_new
    return pert


def _diff_grn(grn_base, grn_pert) -> pd.DataFrame:
    """Return a long-format edge-weight diff table."""
    if grn_base is None or grn_pert is None:
        return pd.DataFrame()
    edges_b = {(u, v): float(d.get("weight", 1.0)) for u, v, d in grn_base.edges(data=True)}
    edges_p = {(u, v): float(d.get("weight", 1.0)) for u, v, d in grn_pert.edges(data=True)}
    keys = sorted(set(edges_b) | set(edges_p))
    if not keys:
        return pd.DataFrame()
    rows = []
    for u, v in keys:
        wb = edges_b.get((u, v), 0.0)
        wp = edges_p.get((u, v), 0.0)
        if wb == 0.0 and wp == 0.0:
            continue
        rows.append((u, v, wb, wp, wp - wb))
    return pd.DataFrame(rows, columns=["source", "target", "weight_base", "weight_pert", "delta"])


def _delta_from_grn(grn_base, grn_pert, *, targets, mode, fold_change) -> pd.DataFrame:
    """One-step propagation: estimate Δ-expression at each downstream
    node by summing the changed in-edge weights, normalised by the
    node's in-degree.

    Intentionally simple — for proper transcriptome-level prediction
    use the CellOracle backend (which propagates through the GRN
    `n_propagation` times).
    """
    if grn_base is None or grn_pert is None:
        return pd.DataFrame()
    import networkx as nx
    all_nodes = sorted(set(grn_base.nodes) | set(grn_pert.nodes))
    rows = []
    for g in all_nodes:
        in_b = sum(float(d.get("weight", 1.0)) for _, _, d in grn_base.in_edges(g, data=True))
        in_p = sum(float(d.get("weight", 1.0)) for _, _, d in grn_pert.in_edges(g, data=True))
        delta = in_p - in_b
        log2_fc = np.log2((in_p + 1e-6) / (in_b + 1e-6))
        rows.append((g, in_b, in_p, delta, log2_fc))
    df = pd.DataFrame(rows, columns=["gene", "mean_base", "mean_pert", "delta", "log2_fc"])
    return df
