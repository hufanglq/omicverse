r"""``ov.es`` — enrichment / gene-set scoring.

Vendored fork of `decoupler.mt
<https://decoupler-py.readthedocs.io/en/latest/api/mt.html>`_'s scoring
kernels (`aucell` / `gsea` / `gsva` / `mdt` / `mlm` / `ora` / `udt` /
`ulm` / `viper` / `waggr` / `zscore`), plus `decouple` / `consensus` /
`query_set`. The decoupler source was copied in tree (under
``omicverse.es._*``) so each kernel can be modified independently — the
plan is to ship per-method GPU accelerations without depending on
upstream decoupler releases. Imports of ``decoupler.*`` were rewritten
to ``omicverse.es.*``; original copyright remains with the decoupler
authors (GPL-3.0).

The public surface accepts **dict-style signatures** (omicverse's
convention) and converts them to decoupler's ``net`` DataFrame
internally — you don't see the ``source/target/weight`` long format
unless you want to.

Example
-------
>>> import omicverse as ov
>>> sigs = {
...     'HALLMARK_INTERFERON_ALPHA': ['IFI6', 'ISG15', 'MX1'],
...     'HALLMARK_INFLAMMATORY':     ['IL6', 'TNF', 'CXCL8'],
... }
>>> ov.es.aucell(adata, signatures=sigs)
>>> adata.obsm['score_aucell']        # cells × signatures DataFrame
>>>
>>> # Weighted / signed signatures (viper / mlm / zscore care about sign):
>>> regulons = {
...     'NFKB': {'TNF': 1.0, 'IL6': 1.0, 'IL10': -1.0},
... }
>>> ov.es.viper(adata, signatures=regulons)

Power users can still pass a raw ``net`` DataFrame via the ``net=``
keyword — the dict path is just the default, more ergonomic option.

Notes
-----
``ov.single.aucell`` (the SCENIC/ctxcore-based legacy path) is retained
for back-compat with pySCENIC workflows that depend on its exact
numerical output or its weighted-regulon / leading-edge side products.
New code should prefer ``ov.es.aucell`` — it is ~15-20× faster
single-threaded and shares preprocessing with the other scoring
methods.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence, Union

import pandas as pd

# omicverse's coloured pre/post-call summary box ("Duration / Shape /
# CHANGES DETECTED" — same one `ov.pp.qc` etc. emit). Wraps each scoring
# call so users see what got added to ``adata.obsm`` for free.
from .._monitor import monitor as _monitor

# Vendored Method instances (decoupler kernels behind a `Method` facade).
from ._aucell    import aucell    as _aucell_m
from ._gsea      import gsea      as _gsea_m
from ._gsva      import gsva      as _gsva_m
from ._mdt       import mdt       as _mdt_m
from ._mlm       import mlm       as _mlm_m
from ._ora       import ora       as _ora_m
from ._udt       import udt       as _udt_m
from ._ulm       import ulm       as _ulm_m
from ._viper     import viper     as _viper_m
from ._waggr     import waggr     as _waggr_m
from ._zscore    import zscore    as _zscore_m
from ._decouple  import decouple  as _decouple_fn
from ._consensus import consensus as _consensus_fn
from ._query_set import query_set as _query_set_fn

SignatureValue = Union[Sequence[str], Mapping[str, float]]
Signatures = Mapping[str, SignatureValue]


def signatures_to_net(
    signatures: Signatures,
    default_weight: float = 1.0,
) -> pd.DataFrame:
    r"""Convert dict-of-genes to decoupler's long ``net`` DataFrame.

    Accepts two value shapes per signature:

    * ``list``/``tuple``/``set`` of gene names — binary set, all genes
      get ``weight = default_weight``.
    * ``dict`` mapping ``gene → weight`` (float, sign-aware) — passed
      through unchanged. Required for signed regulons (viper / mlm /
      zscore use the sign to distinguish activators from repressors).

    Parameters
    ----------
    signatures
        Mapping ``name → list[str] | dict[str, float]``.
    default_weight
        Weight applied when the value is an unweighted iterable.

    Returns
    -------
    pandas.DataFrame
        Long-format ``net`` with columns ``source``, ``target``,
        ``weight`` — the shape every vendored kernel consumes.
    """
    rows = []
    for name, item in signatures.items():
        if isinstance(item, Mapping):
            for g, w in item.items():
                rows.append({'source': name, 'target': str(g), 'weight': float(w)})
        elif isinstance(item, (list, tuple, set, frozenset)):
            for g in item:
                rows.append(
                    {'source': name, 'target': str(g), 'weight': float(default_weight)}
                )
        else:
            raise TypeError(
                f"signatures[{name!r}] must be list / tuple / set / dict, "
                f"got {type(item).__name__}"
            )
    if not rows:
        raise ValueError("`signatures` is empty.")
    return pd.DataFrame(rows)


def _resolve_net(signatures, net):
    """Either ``signatures`` (dict) or ``net`` (DataFrame), not both."""
    if signatures is not None and net is not None:
        raise ValueError("pass either `signatures` or `net`, not both")
    if signatures is not None:
        return signatures_to_net(signatures)
    if net is None:
        raise ValueError("must pass `signatures` (dict) or `net` (DataFrame)")
    return net


def _bind_method(method):
    """Wrap a vendored ``Method`` so it accepts ``signatures=`` (dict)."""

    def wrapped(
        data,
        signatures=None,
        *,
        net=None,
        tmin=5,
        raw=False,
        empty=True,
        bsize=250_000,
        verbose=False,
        engine: str = 'auto',
        **kwargs,
    ):
        resolved_net = _resolve_net(signatures, net)
        return method(
            data,
            net=resolved_net,
            tmin=tmin,
            raw=raw,
            empty=empty,
            bsize=bsize,
            verbose=verbose,
            engine=engine,
            **kwargs,
        )

    wrapped.__name__ = method.name
    wrapped.__qualname__ = f'ov.es.{method.name}'
    wrapped.__doc__ = (
        f"Run the ``{method.name}`` scoring kernel "
        f"with omicverse-style dict input.\n"
        f"\n"
        f"Parameters\n"
        f"----------\n"
        f"data : AnnData | DataFrame\n"
        f"    Expression matrix the kernel scores.\n"
        f"signatures : dict, optional\n"
        f"    Mapping ``{{name → list[gene]}}`` (binary) or\n"
        f"    ``{{name → dict[gene, weight]}}`` (weighted/signed).\n"
        f"    Mutually exclusive with ``net``.\n"
        f"net : pandas.DataFrame, optional\n"
        f"    Long-format DataFrame with ``source / target / weight``\n"
        f"    columns. Power-user escape hatch; ``signatures`` is the\n"
        f"    default.\n"
        f"tmin : int, default 5\n"
        f"    Minimum number of targets per source — sets below this are\n"
        f"    silently dropped.\n"
        f"raw : bool, default False\n"
        f"    Use ``adata.raw.X`` instead of ``adata.X``.\n"
        f"empty : bool, default True\n"
        f"    Whether to write all-zero results for filtered-out sources.\n"
        f"bsize : int, default 250000\n"
        f"    Cells per processing chunk (controls peak memory).\n"
        f"verbose : bool, default False\n"
        f"    Stream progress bars / info logs.\n"
        f"**kwargs\n"
        f"    Method-specific options forwarded to the kernel.\n"
        f"\n"
        f"Returns\n"
        f"-------\n"
        f"None\n"
        f"    Writes scores to ``adata.obsm['score_{method.name}']`` "
        f"(and p-values to ``adata.obsm['padj_{method.name}']`` for "
        f"methods that produce them).\n"
        f"\n"
        f"Original kernel docstring\n"
        f"-------------------------\n"
        f"{method.__doc__ or ''}"
    )
    return wrapped


aucell = _monitor(_bind_method(_aucell_m))
gsea = _monitor(_bind_method(_gsea_m))
gsva = _monitor(_bind_method(_gsva_m))
mdt = _monitor(_bind_method(_mdt_m))
mlm = _monitor(_bind_method(_mlm_m))
ora = _monitor(_bind_method(_ora_m))
udt = _monitor(_bind_method(_udt_m))
ulm = _monitor(_bind_method(_ulm_m))
viper = _monitor(_bind_method(_viper_m))
waggr = _monitor(_bind_method(_waggr_m))
zscore = _monitor(_bind_method(_zscore_m))


@_monitor
def decouple(
    data,
    signatures=None,
    *,
    net=None,
    methods=None,
    args=None,
    cons: bool = True,
    **kwargs,
):
    """Run multiple scoring kernels in one pass; optional consensus.

    Equivalent of ``decoupler.mt.decouple`` with dict signature input.
    See the original ``decouple`` docstring for ``methods`` / ``args``
    semantics.
    """
    resolved_net = _resolve_net(signatures, net)
    return _decouple_fn(
        data, net=resolved_net, methods=methods, args=args, cons=cons, **kwargs,
    )


def consensus(result, verbose: bool = False):
    """Build a consensus score across per-method outputs (Stouffer-like).

    Pass the dict returned by ``ov.es.decouple(..., cons=False)``.
    """
    return _consensus_fn(result, verbose=verbose)


def query_set(
    features,
    signatures=None,
    *,
    net=None,
    alternative: str = 'two-sided',
    n_bg: int = 1000,
    ha_corr: str = 'BH',
    tmin: int = 5,
    verbose: bool = False,
):
    """Hypergeometric-style enrichment test of ``features`` in each signature."""
    resolved_net = _resolve_net(signatures, net)
    return _query_set_fn(
        features,
        net=resolved_net,
        alternative=alternative,
        n_bg=n_bg,
        ha_corr=ha_corr,
        tmin=tmin,
        verbose=verbose,
    )


__all__ = [
    'aucell', 'gsea', 'gsva', 'mdt', 'mlm', 'ora',
    'udt', 'ulm', 'viper', 'waggr', 'zscore',
    'decouple', 'consensus', 'query_set',
    'signatures_to_net',
]
