# Vendored from `decoupler` (https://github.com/scverse/decoupler) by
# omicverse for in-tree GPU acceleration work. Original copyright by
# the decoupler authors, redistributed under decoupler's GPL-3.0
# license. Cross-module imports rewritten from `decoupler.*` to
# `omicverse.es.*` (see scripts/vendor_decoupler.py).

import numpy as np
from tqdm.auto import tqdm

from omicverse.es._docs import docs
from omicverse.es._log import _log
from omicverse.es._method import Method, MethodMeta
from omicverse.es._odeps import _check_import, xgboost


def _xgbr(
    x: np.ndarray,
    y: np.ndarray,
    **kwargs,
) -> np.ndarray:
    kwargs.setdefault("n_estimators", 10)
    # Init model
    reg = xgboost.XGBRegressor(**kwargs)
    # Fit
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    reg = reg.fit(x, y)
    # Get R score
    es = reg.score(x, y)
    # Clip to [0, 1]
    es = np.clip(es, 0, 1)
    return es


@docs.dedent
def _func_udt(
    mat: np.ndarray,
    adj: np.ndarray,
    verbose: bool = False,
    **kwargs,
) -> tuple[np.ndarray, None]:
    """
    Univariate Decision Tree (UDT) :cite:`decoupler`.

    This approach uses the molecular features from one observation as the population of samples
    and it fits a gradient boosted decision trees model with a single covariate,
    which is the feature weights of a set :math:`F`.
    It uses the implementation provided by ``xgboost`` :cite:`xgboost`.

    The enrichment score :math:`ES` is then calculated as the coefficient of determination :math:`R^2`.

    %(notest)s

    %(params)s

    kwargs
        All other keyword arguments are passed to ``xgboost.XGBRegressor``.
    %(returns)s

    Example
    -------
    .. code-block:: python

        import omicverse as ov

        adata, net = ov.es.toy_data()  # or your own (adata, net)
        ov.es.udt(adata, net, tmin=3)
    """
    _check_import(xgboost, "xgboost")
    nobs = mat.shape[0]
    nvar, nsrc = adj.shape
    m = f"udt - fitting {nsrc} univariate decision tree models (XGBoost) of {nvar} targets across {nobs} observations"
    _log(m, level="info", verbose=verbose)
    es = np.zeros(shape=(nobs, nsrc))
    for i in tqdm(range(nobs), disable=not verbose):
        obs = mat[i]
        for j in range(adj.shape[1]):
            es[i, j] = _xgbr(x=adj[:, j], y=obs, **kwargs)
    return es, None


def _func_udt_torch(
    mat,
    adj,
    verbose: bool = False,
    n_estimators: int = 10,
    max_depth: int = 6,
    learning_rate: float = 0.3,
    reg_lambda: float = 1.0,
    **_kwargs,
):
    r"""GPU (torch) port of :func:`_func_udt`.

    UDT fits ``nobs × nsrc`` univariate gradient-boosted trees (one per
    cell-signature pair), each on the single feature ``adj[:, j]``. The
    CPU version's double Python loop dominates wall time (≈ 128 k tiny
    xgboost fits on PBMC3k). On GPU we batch across cells per signature
    — for each signature ``j`` we run a single ``gbdt_squared_loss_torch``
    call with ``X = adj[:, j:j+1]`` and ``Y = mat.T`` (all ``nobs``
    targets at once) and read off :math:`R^2` from the final
    predictions.

    Same algorithmic notes as ``_func_mdt_torch`` apply; see the
    GBDT section in ``_engine.py`` for the differences from xgboost.
    """
    import torch
    from omicverse.es._engine import torch_device, to_gpu_dense, gbdt_squared_loss_torch

    device = torch_device()
    nobs, nvar = mat.shape
    nvar_a, nsrc = adj.shape
    assert nvar == nvar_a, "adj rows must equal mat columns"

    Mat = to_gpu_dense(mat, device, dtype=torch.float32)               # (nobs, nvar)
    Adj = torch.as_tensor(np.asarray(adj), dtype=torch.float32, device=device)  # (nvar, nsrc)
    Y = Mat.t().contiguous()                                            # (nvar, nobs)

    # Per-cell totals — denominator of R² = 1 - SS_res / SS_tot is shared
    # across signatures.
    y_mean = Y.mean(dim=0, keepdim=True)                                # (1, nobs)
    ss_tot = ((Y - y_mean) ** 2).sum(dim=0)                             # (nobs,)
    ss_tot_safe = ss_tot.clamp(min=1e-30)

    es = torch.zeros(nobs, nsrc, dtype=torch.float32, device=device)
    for j in range(nsrc):
        X_j = Adj[:, j:j + 1].contiguous()                              # (nvar, 1)
        res = gbdt_squared_loss_torch(
            X_j, Y,
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, reg_lambda=reg_lambda,
            return_importances=False, return_predictions=True,
        )
        pred = res['predictions']                                       # (nvar, nobs)
        ss_res = ((Y - pred) ** 2).sum(dim=0)                           # (nobs,)
        r2 = 1.0 - ss_res / ss_tot_safe
        es[:, j] = r2.clamp(0.0, 1.0)

    return es.cpu().numpy().astype(np.float64), None


_udt = MethodMeta(
    name="udt",
    desc="Univariate Decision Tree (UDT)",
    func=_func_udt,
    func_torch=_func_udt_torch,
    stype="numerical",
    adj=True,
    weight=True,
    test=False,
    limits=(0, 1),
    reference="https://doi.org/10.1093/bioadv/vbac016",
)
_func_udt_torch._accepts_sparse = True
udt = Method(_method=_udt)
