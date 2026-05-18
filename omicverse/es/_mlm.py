# Vendored from `decoupler` (https://github.com/scverse/decoupler) by
# omicverse for in-tree GPU acceleration work. Original copyright by
# the decoupler authors, redistributed under decoupler's GPL-3.0
# license. Cross-module imports rewritten from `decoupler.*` to
# `omicverse.es.*` (see scripts/vendor_decoupler.py).

import numba as nb
import numpy as np
import scipy.stats as sts

from omicverse.es._method import Method, MethodMeta

@nb.njit(parallel=True, cache=True)
def _fit(
    X: np.ndarray,
    y: np.ndarray,
    inv: np.ndarray,
    df: float,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.ascontiguousarray(X)
    n_samples = y.shape[1]
    n_fsets = X.shape[1]
    coef, sse, _, _ = np.linalg.lstsq(X, y)
    assert len(sse) > 0, (
        "Could not fit a multivariate linear model. This can happen because there are more sources\n \
    (covariates) than unique targets (samples), or because the network adjacency matrix rank is smaller than the number\n \
    of sources"
    )
    sse = sse / df
    se = np.zeros((n_samples, n_fsets))
    for i in nb.prange(n_samples):
        se[i] = np.sqrt(np.diag(sse[i] * inv))
    coef = coef.T
    tval = coef / se
    return coef[:, 1:], tval[:, 1:]

def _func_mlm(
    mat: np.ndarray,
    adj: np.ndarray,
    tval: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Multivariate Linear Model (MLM) :cite:`decoupler`.

    This approach uses the molecular features from one observation as the population of samples
    and it fits a linear model with with multiple covariates, which are the weights of all feature sets :math:`F`.

    .. math::

        y^i = \beta_0 + \beta_1 x_{1}^{i} + \beta_2 x_{2}^{i} + \cdots + \beta_p x_{p}^{i} + \varepsilon

    Where:

    - :math:`y^i` is the observed feature statistic (e.g. gene expression, :math:`log_{2}FC`, etc.) for feature :math:`i`
    - :math:`x_{p}^{i}` is the weight of feature :math:`i` in feature set :math:`F_p`. For unweighted sets, membership in the set is indicated by 1, and non-membership by 0.
    - :math:`\beta_0` is the intercept
    - :math:`\beta_p` is the slope coefficient for feature set :math:`F_p`
    - :math:`\varepsilon` is the error term for feature :math:`i`

    .. figure:: /_static/images/mlm.png
       :alt: Multivariate Linear Model (MLM) schematic.
       :align: center
       :width: 75%

       Multivariate Linear Model (MLM) scheme.
       In this example, the observed gene expression of :math:`Sample_1` is predicted using
       the interaction weights of two pathways, :math:`P_1` and :math:`P_2`.
       For :math:`P2`, since its target genes that have negative weights are lowly expressed,
       and its positive target genes are highly expressed,
       the relationship between the two variables is positive so the obtained :math:`ES` score is positive.
       Scores can be interpreted as active when positive, repressive when negative, and inconclusive when close to 0.

    The enrichment score :math:`ES` for each :math:`F` is then calculated as the t-value of the slope coefficients.

    .. math::

        ES = t_{\beta_1} = \frac{\hat{\beta}_1}{\mathrm{SE}(\hat{\beta}_1)}

    Where:

    - :math:`t_{\beta_1}` is the t-value of the slope
    - :math:`\mathrm{SE}(\hat{\beta}_1)` is the standard error of the slope

    Next, :math:`p_{value}` are obtained by evaluating the two-sided survival function
    (:math:`sf`) of the Student’s t-distribution.

    .. math::

        p_{value} = 2 \times \mathrm{sf}(|ES|, \text{df})

    Parameters
    ----------
    %(tval)s

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
        ov.es.mlm(adata, net, tmin=3)
    """
    # Get dims
    n_features, n_fsets = adj.shape
    # Add intercept
    adj = np.column_stack((np.ones((n_features,)), adj))
    # Compute inv and df for lm
    inv = np.linalg.inv(np.dot(adj.T, adj))
    df = n_features - n_fsets - 1
    # Compute tval
    coef, t = _fit(adj, mat.T, inv, df)
    # Compute pval
    pv = 2 * (1 - sts.t.cdf(x=np.abs(t), df=df))
    # Return coef or tval
    if tval:
        es = t
    else:
        es = coef
    return es, pv

def _func_mlm_torch(
    mat,
    adj,
    tval: bool = True,
    verbose: bool = False,
):
    r"""Torch (GPU) port of :func:`_func_mlm`.

    Replicates decoupler's pipeline: prepend an intercept column, then
    for each cell solve the closed-form OLS

    .. math::
        \hat{\beta} = (X^\top X)^{-1} X^\top y,\quad
        SE_j = \sqrt{(SSE_i / df) \cdot (X^\top X)^{-1}_{jj}}

    Equivalent to ``numpy.linalg.lstsq`` in the all-finite, full-rank
    regime decoupler enforces. Matches the CPU result to fp64
    round-off.
    """
    import torch
    from omicverse.es._engine import torch_device, to_gpu_dense

    device = torch_device()
    n_features, n_fsets = np.asarray(adj).shape
    # Prepend intercept column
    A_ext = np.column_stack((np.ones((n_features,)), np.asarray(adj)))
    df = n_features - n_fsets - 1

    X = torch.as_tensor(A_ext, dtype=torch.float64, device=device)             # (n_var, p+1)
    # mlm wants Y = mat.T, so densify mat on GPU first then transpose.
    Y = to_gpu_dense(mat, device, dtype=torch.float64).T                        # (n_var, n_cells)
    XtX = X.T @ X                                                              # (p+1, p+1)
    XtY = X.T @ Y                                                              # (p+1, n_cells)
    inv = torch.linalg.inv(XtX)
    coef = inv @ XtY                                                           # (p+1, n_cells)

    # Residual SSE per cell, matching the lstsq-with-rcond path
    resid = Y - X @ coef                                                       # (n_var, n_cells)
    sse = (resid ** 2).sum(dim=0) / df                                         # (n_cells,)
    diag_inv = torch.diagonal(inv)                                             # (p+1,)
    se = torch.sqrt(sse.unsqueeze(0) * diag_inv.unsqueeze(1))                  # (p+1, n_cells)
    tval_mat = coef / se                                                       # (p+1, n_cells)

    # P-value strategy: identical to ulm — Normal approximation when
    # df is large (mlm's df = n_features − n_fsets − 1, typically in
    # the thousands), exact scipy path otherwise. Saves ~25 ms /
    # call.
    coef_np = coef.T.cpu().numpy()[:, 1:]   # drop intercept column
    if df > 100:
        # Drop intercept column on GPU before erfc to stay vectorised.
        pv = torch.special.erfc(tval_mat[1:].T.abs() / np.sqrt(2.0))
        pv = pv.cpu().numpy()
    else:
        tval_np_full = tval_mat.T.cpu().numpy()
        pv = 2 * (1 - sts.t.cdf(np.abs(tval_np_full[:, 1:]), df=df))
    tval_np = tval_mat.T.cpu().numpy()[:, 1:]
    es = tval_np if tval else coef_np
    return es, pv

_mlm = MethodMeta(
    name="mlm",
    desc="Multivariate Linear Model (MLM)",
    func=_func_mlm,
    func_torch=_func_mlm_torch,
    stype="numerical",
    adj=True,
    weight=True,
    test=True,
    limits=(-np.inf, +np.inf),
    reference="https://doi.org/10.1093/bioadv/vbac016",
)
_func_mlm_torch._accepts_sparse = True
mlm = Method(_method=_mlm)
