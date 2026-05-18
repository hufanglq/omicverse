# Vendored from `decoupler` (https://github.com/scverse/decoupler) by
# omicverse for in-tree GPU acceleration work. Original copyright by
# the decoupler authors, redistributed under decoupler's GPL-3.0
# license. Cross-module imports rewritten from `decoupler.*` to
# `omicverse.es.*` (see scripts/vendor_decoupler.py).

import numba as nb
import numpy as np
import scipy.sparse as sps
import scipy.stats as sts
from tqdm.auto import tqdm

from omicverse.es._docs import docs
from omicverse.es._log import _log
from omicverse.es._method import Method, MethodMeta
from omicverse.es._net import _getset


@nb.njit(parallel=True, cache=True)
def _auc(
    row: np.ndarray,
    cnct: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    n_up: int,
    nsrc: int,
) -> np.ndarray:
    # Empty acts
    es = np.zeros(nsrc)
    # For each feature set
    for j in nb.prange(nsrc):
        # Extract feature set
        fset = _getset(cnct, starts, offsets, j)
        # Compute max AUC for fset
        x_th = np.arange(1, stop=fset.shape[0] + 1)
        x_th = x_th[x_th < n_up]
        max_auc: float = np.sum(np.diff(np.append(x_th, n_up)) * x_th)
        # Compute AUC
        x = row[fset]
        x = np.sort(x[x <= n_up])
        y = np.arange(x.shape[0]) + 1
        x = np.append(x, n_up)
        # Update acts matrix
        es[j] = np.sum(np.diff(x) * y) / max_auc
    return es


def _validate_n_up(
    nvar: int,
    n_up: int | float | None = None,
) -> int:
    assert isinstance(n_up, int | float) or n_up is None, "n_up must be numerical or None"
    if n_up is None:
        n_up = np.ceil(0.05 * nvar)
        n_up = int(np.clip(n_up, a_min=2, a_max=nvar))
    else:
        n_up = int(np.ceil(n_up))
    assert nvar >= n_up > 1, f"For nvar={nvar}, n_up={n_up} must be between 1 and {nvar}"
    return n_up


@docs.dedent
def _func_aucell(
    mat: np.ndarray,
    cnct: np.ndarray,
    starts: np.ndarray,
    offsets: np.ndarray,
    n_up: int | float | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, None]:
    r"""
    Area Under the Curve for set enrichment within single cells (AUCell) :cite:`aucell`.

    Given a ranked list of features per observation, AUCell calculates the AUC by measuring how early the features in
    the set appear in this ranking. Specifically, the enrichment score :math:`ES` is:

    .. math::

       {ES}_{i, F} = \int_0^1 {RecoveryCurve}_{i, F}(r_i) \, dr

    Where:

    - :math:`i` is the obervation
    - :math:`F` is the feature set
    - :math:`{RecoveryCurve}_{i, F}(r_i)` is the proportion of features from :math:`F` recovered in the top :math:`r_i`-fraction of the ranked list for observation :math:`i`

    %(notest)s

    %(params)s
    n_up
        Number of features to include in the AUC calculation.
        If ``None``, the top 5% of features based on their magnitude are selected.

    %(returns)s

    Example
    -------
    .. code-block:: python

        import omicverse as ov

        adata, net = ov.es.toy_data()  # or your own (adata, net)
        ov.es.aucell(adata, net, tmin=3)
    """
    nobs, nvar = mat.shape
    nsrc = starts.size
    n_up = _validate_n_up(nvar, n_up)
    m = f"aucell - calculating {nsrc} AUCs for {nvar} targets across {nobs} observations, categorizing features at rank={n_up}"
    _log(m, level="info", verbose=verbose)
    es = np.zeros(shape=(nobs, nsrc))
    for i in tqdm(range(mat.shape[0]), disable=not verbose):
        if isinstance(mat, sps.csr_matrix):
            row = mat[i].toarray()[0]
        else:
            row = mat[i]
        row = sts.rankdata(a=-row, method="ordinal")
        es[i] = _auc(row=row, cnct=cnct, starts=starts, offsets=offsets, n_up=n_up, nsrc=nsrc)
    return es, None


def _func_aucell_torch(
    mat,
    cnct,
    starts,
    offsets,
    n_up=None,
    verbose: bool = False,
):
    r"""Torch (GPU) port of :func:`_func_aucell` — bit-for-bit
    equivalent on fp64.

    Vectorisation strategy
    ----------------------
    1. **Rank** all genes per cell once via ``argsort(stable=True) +
       scatter``. Matches ``scipy.stats.rankdata(..., method='ordinal')``
       because both break ties by appearance order in the array.
    2. For each signature, gather the ranks of that signature's genes
       across **all cells** at once → shape ``(nobs, k_j)``. Mask
       ranks above ``n_up`` by clamping them to ``n_up`` so they
       contribute zero to the AUC integral after sorting + diff.
    3. The integral matches the CPU formula exactly:
       ``AUC = sum(diff([sorted_ranks, n_up]) * y) / max_auc`` where
       ``max_auc`` is the per-signature scalar identical to the numba
       version.

    No p-values are produced (aucell is non-statistical), so ``pv``
    returns ``None`` to match :func:`_func_aucell`.
    """
    import torch
    from omicverse.es._engine import torch_device

    device = torch_device()
    nobs, nvar = mat.shape
    nsrc = starts.size
    n_up = _validate_n_up(nvar, n_up)

    # Densify if sparse, then move to GPU.
    if sps.issparse(mat):
        mat = mat.toarray()
    M = torch.as_tensor(np.asarray(mat), dtype=torch.float64, device=device)

    # Per-cell rank, stable to match scipy 'ordinal'.
    sort_idx = torch.argsort(-M, dim=1, stable=True)            # (nobs, nvar)
    ranks = torch.empty_like(sort_idx, dtype=torch.long)
    rank_vals = torch.arange(
        1, nvar + 1, dtype=torch.long, device=device,
    ).unsqueeze(0).expand(nobs, -1)
    ranks.scatter_(1, sort_idx, rank_vals)

    es = torch.zeros(nobs, nsrc, dtype=torch.float64, device=device)
    n_up_int = int(n_up)

    for j in range(nsrc):
        # Numpy indices for the j-th feature set
        fset_np = cnct[starts[j]:starts[j] + offsets[j]]
        k = int(fset_np.shape[0])
        # Per-signature scalar max_auc — identical to CPU code path.
        x_th = np.arange(1, k + 1)
        x_th = x_th[x_th < n_up_int]
        max_auc = float(np.sum(np.diff(np.append(x_th, n_up_int)) * x_th))
        if max_auc == 0:
            continue

        fset = torch.as_tensor(fset_np, dtype=torch.long, device=device)
        x = ranks[:, fset]                                       # (nobs, k)
        # Saturate at n_up so values above the threshold contribute 0 to
        # the integral once sorted (`diff` between consecutive n_up's = 0).
        x = torch.where(
            x <= n_up_int,
            x,
            torch.full_like(x, n_up_int, dtype=x.dtype),
        )
        x_sorted, _ = torch.sort(x, dim=1)                       # (nobs, k)
        last = torch.full(
            (nobs, 1), n_up_int, dtype=x_sorted.dtype, device=device,
        )
        x_full = torch.cat([x_sorted, last], dim=1)              # (nobs, k+1)
        dx = torch.diff(x_full, dim=1).to(torch.float64)         # (nobs, k)
        y_seq = torch.arange(
            1, k + 1, dtype=torch.float64, device=device,
        ).unsqueeze(0)                                           # (1, k)
        es[:, j] = (dx * y_seq).sum(dim=1) / max_auc

    return es.cpu().numpy(), None


_aucell = MethodMeta(
    name="aucell",
    desc="AUCell",
    func=_func_aucell,
    func_torch=_func_aucell_torch,
    stype="categorical",
    adj=False,
    weight=False,
    test=False,
    limits=(0, 1),
    reference="https://doi.org/10.1038/nmeth.4463",
)
aucell = Method(_method=_aucell)
