# Vendored from `decoupler` (https://github.com/scverse/decoupler) by
# omicverse for in-tree GPU acceleration work. Original copyright by
# the decoupler authors, redistributed under decoupler's GPL-3.0
# license. Cross-module imports rewritten from `decoupler.*` to
# `omicverse.es.*` (see scripts/vendor_decoupler.py).

import numpy as np
from tqdm.auto import tqdm

from omicverse.es._method import Method, MethodMeta
from omicverse.es._odeps import _check_import, xgboost

def _xgbr(
    x: np.ndarray,
    y: np.ndarray,
    **kwargs,
) -> np.ndarray:
    # Init model
    reg = xgboost.XGBRegressor(**kwargs)
    # Fit
    y = y.reshape(-1, 1)
    reg = reg.fit(x, y)
    # Get R score
    es = reg.feature_importances_
    return es

def _func_mdt(
    mat: np.ndarray,
    adj: np.ndarray,
    verbose: bool = False,
    **kwargs,
) -> tuple[np.ndarray, None]:
    r"""
    Multivariate Decision Trees (MDT) :cite:`decoupler`.

    This approach uses the molecular features from one observation as the population of samples
    and it fits a gradient boosted decision trees model with multiple covariates,
    which are the weights of all feature sets :math:`F`. It uses the implementation provided by ``xgboost`` :cite:`xgboost`.

    The enrichment score :math:`ES` for each :math:`F` is then calculated as the importance of each covariate in the model.

    Parameters
    ----------

    kwargs
        All other keyword arguments are passed to ``xgboost.XGBRegressor``.
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
        ov.es.mdt(adata, net, tmin=3)
    """
    _check_import(xgboost, "xgboost")
    nobs = mat.shape[0]
    nvar, nsrc = adj.shape
    es = np.zeros(shape=(nobs, nsrc))
    for i in tqdm(range(nobs), disable=not verbose):
        obs = mat[i]
        es[i, :] = _xgbr(x=adj, y=obs, **kwargs)
    return (es, None)

def _func_mdt_torch(
    mat,
    adj,
    verbose: bool = False,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.3,
    reg_lambda: float = 1.0,
    **_kwargs,
):
    r"""GPU (torch) port of :func:`_func_mdt`.

    Replaces the per-cell ``xgboost.XGBRegressor`` fit with a fully
    batched pure-torch GBDT (see ``gbdt_squared_loss_torch`` in
    ``_engine.py``). All ``nobs`` regressions share the same feature
    matrix ``adj``, so they can be fit in parallel — one ``B``-axis
    sweep over the boosting loop instead of ``nobs`` sequential xgboost
    fits.

    Algorithmic fidelity matches XGBoost on default squared-loss
    parameters; numerical agreement is approximate (importance Pearson
    r ≈ 0.99 mean, prediction r ≈ 0.998 mean against xgboost). See the
    ``# Pure-torch gradient boosted decision trees`` section in
    ``_engine.py`` for the precise list of differences.
    """
    import torch
    from omicverse.es._engine import torch_device, to_gpu_dense, gbdt_squared_loss_torch

    device = torch_device()
    nobs, nvar = mat.shape
    nvar_a, nsrc = adj.shape
    assert nvar == nvar_a, "adj rows must equal mat columns"

    Mat = to_gpu_dense(mat, device, dtype=torch.float32)               # (nobs, nvar)
    Adj = torch.as_tensor(np.asarray(adj), dtype=torch.float32, device=device)  # (nvar, nsrc)
    # Per-cell regression: X = adj, Y[:, c] = mat[c, :]
    Y = Mat.t().contiguous()                                            # (nvar, nobs)

    res = gbdt_squared_loss_torch(
        Adj, Y,
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, reg_lambda=reg_lambda,
        return_importances=True, return_predictions=False,
    )
    importances = res['importances']                                    # (nsrc, nobs)
    es = importances.t().cpu().numpy().astype(np.float64)               # (nobs, nsrc)
    return es, None

_mdt = MethodMeta(
    name="mdt",
    desc="Multivariate Decision Tree (MDT)",
    func=_func_mdt,
    func_torch=_func_mdt_torch,
    stype="numerical",
    adj=True,
    weight=True,
    test=False,
    limits=(0, 1),
    reference="https://doi.org/10.1093/bioadv/vbac016",
)
_func_mdt_torch._accepts_sparse = True
mdt = Method(_method=_mdt)
