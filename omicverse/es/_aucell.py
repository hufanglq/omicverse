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

    Parameters
    ----------
    n_up
        Number of features to include in the AUC calculation.
        If ``None``, the top 5% of features based on their magnitude are selected.

    Returns
    -------
    es : np.ndarray
        Enrichment score matrix (observations × signatures).
    pv : np.ndarray or None
        P-value matrix; ``None`` for kernels without a statistical test.

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
    equivalent on fp64 via a fully batched recovery-curve formulation.

    Algorithmic redesign
    --------------------
    The original CPU kernel computes, per (cell, signature):

    .. math::
        AUC = \sum_{t=1}^{k} (x_{t+1} - x_t) \cdot t

    where :math:`x_1 \le x_2 \le \ldots \le x_k \le x_{k+1} = n\_up`
    are the sorted ranks of the signature's genes that fall within
    the top-:math:`n\_up`. Equivalently, this is the area under the
    recovery curve :math:`R(r)` integrated on :math:`[x_1, n\_up]`.

    Rewriting via the discrete recovery sum:

    .. math::
        AUC = \sum_{r=1}^{n\_up} R(r) \;-\; k_\text{valid}

    where :math:`R(r)` is the number of signature genes with rank
    :math:`\le r` and :math:`k_\text{valid}` is the signature's gene
    count within the top-:math:`n\_up`. The correction
    :math:`-k_\text{valid}` accounts for the half-open integration
    boundary in the CPU formulation.

    This rewrite **eliminates the per-signature sort + Python loop**.
    We can compute every (cell, signature) AUC simultaneously by:

    1. Pre-computing rank-order indices once: ``sort_idx[i, r]`` gives
       which gene sits at rank ``r+1`` in cell ``i`` (top-N only).
    2. Building a binary membership matrix ``M[gene, sig]``.
    3. Gathering ``M[sort_idx]`` → ``(nobs, n_up, nsrc)`` tensor whose
       ``cumsum`` along axis 1 *is* the recovery curve.
    4. ``sum(cumsum) - sum(membership)`` gives the unnormalised AUC.

    Memory model
    ------------
    The ``(nobs, n_up, nsrc)`` membership tensor is processed in cell
    batches sized to fit roughly 1 GB of fp32 working memory, so the
    routine scales to ~100k cells × thousands of signatures without
    running the GPU out of memory.
    """
    import torch
    from omicverse.es._engine import torch_device, chunk_size_for, to_gpu_dense

    device = torch_device()
    nobs, nvar = mat.shape
    nsrc = starts.size
    n_up = _validate_n_up(nvar, n_up)
    n_up_int = int(n_up)

    M = to_gpu_dense(mat, device, dtype=torch.float64)

    # Per-cell argsort, stable → matches scipy 'ordinal'. Then keep only
    # the indices that land in the top `n_up`; that's all we need for
    # the recovery curve.
    sort_idx = torch.argsort(-M, dim=1, stable=True)            # (nobs, nvar)
    top_idx = sort_idx[:, :n_up_int].contiguous()               # (nobs, n_up)

    # Build the dense (gene → signature) binary membership matrix once.
    # Vectorised: ``np.repeat(arange(nsrc), offsets)`` replaces the
    # per-signature Python loop that filled ``sig_id_per_target``
    # element-by-element. Use float32 because entries are 0/1 and the
    # final division to fp64 reinstates precision.
    membership = torch.zeros(nvar, nsrc, dtype=torch.float32, device=device)
    cnct_t = torch.as_tensor(cnct, dtype=torch.long, device=device)
    sig_id_per_target = torch.as_tensor(
        np.repeat(np.arange(nsrc, dtype=np.int64), offsets),
        device=device,
    )
    membership[cnct_t, sig_id_per_target] = 1.0

    # Per-signature scalar max_auc — closed form, vectorised:
    #
    #   if k <  n_up:  max_auc = k(k-1)/2 + k(n_up - k)
    #   if k >= n_up:  max_auc = (n_up - 1) * n_up / 2
    #
    # Derived by working through the CPU formula
    # ``sum(diff(append(arange(1,k+1)[<n_up], n_up)) * arange(1,k+1)[<n_up])``
    # in both regimes; identical to the numba kernel.
    sig_sizes = offsets.astype(np.int64)
    max_aucs = np.where(
        sig_sizes < n_up_int,
        sig_sizes * (sig_sizes - 1) / 2.0 + sig_sizes * (n_up_int - sig_sizes),
        (n_up_int - 1) * n_up_int / 2.0,
    ).astype(np.float64)
    max_aucs_t = torch.as_tensor(max_aucs, dtype=torch.float64, device=device)
    safe_max = torch.where(
        max_aucs_t == 0,
        torch.ones_like(max_aucs_t),  # avoid 0/0; we zero those columns at end
        max_aucs_t,
    )

    # Cell-batch loop with a fixed target-element budget, mirroring
    # ``ov.pp._pca._auto_dense_chunk_size``. Keeps the per-batch
    # working tensor ``(B × n_up × nsrc)`` around ~32 MB so torch's
    # caching allocator reuses the same blocks across batches without
    # fragmentation or driver-side OOM.
    cells_per_batch = chunk_size_for(n_up_int * nsrc, max_units=nobs)

    es = torch.zeros(nobs, nsrc, dtype=torch.float64, device=device)
    for b0 in range(0, nobs, cells_per_batch):
        b1 = min(b0 + cells_per_batch, nobs)
        # M_top[i, r, j] = 1 if gene at rank r+1 in cell i is in signature j
        m_top = membership[top_idx[b0:b1]]               # (B, n_up, nsrc) fp32
        # Recovery curve cumsum + integral; k_valid = total membership.
        cs = m_top.cumsum(dim=1)                         # (B, n_up, nsrc)
        total = cs.sum(dim=1)                            # (B, nsrc)
        k_valid = m_top.sum(dim=1)                       # (B, nsrc)
        es[b0:b1] = (total - k_valid).to(torch.float64) / safe_max

    # Columns where max_auc was 0 (signatures empty after pruning to n_up)
    # were rescued by `safe_max`; restore them as 0 to match CPU.
    zero_cols = (max_aucs_t == 0).nonzero(as_tuple=False).squeeze(-1)
    if zero_cols.numel() > 0:
        es[:, zero_cols] = 0.0

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
_func_aucell_torch._accepts_sparse = True
aucell = Method(_method=_aucell)
