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
    # Init model
    reg = xgboost.XGBRegressor(**kwargs)
    # Fit
    y = y.reshape(-1, 1)
    reg = reg.fit(x, y)
    # Get R score
    es = reg.feature_importances_
    return es


@docs.dedent
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
        ov.es.mdt(adata, net, tmin=3)
    """
    _check_import(xgboost, "xgboost")
    nobs = mat.shape[0]
    nvar, nsrc = adj.shape
    m = f"mdt - fitting {nsrc} multivariate decision tree models (XGBoost) of {nvar} targets across {nobs} observations"
    _log(m, level="info", verbose=verbose)
    es = np.zeros(shape=(nobs, nsrc))
    for i in tqdm(range(nobs), disable=not verbose):
        obs = mat[i]
        es[i, :] = _xgbr(x=adj, y=obs, **kwargs)
    return (es, None)


_mdt = MethodMeta(
    name="mdt",
    desc="Multivariate Decision Tree (MDT)",
    func=_func_mdt,
    stype="numerical",
    adj=True,
    weight=True,
    test=False,
    limits=(0, 1),
    reference="https://doi.org/10.1093/bioadv/vbac016",
)
mdt = Method(_method=_mdt)
