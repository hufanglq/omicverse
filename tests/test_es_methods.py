"""Smoke tests for ``ov.es`` — every public scoring method runs end-to-end.

Each of the 11 vendored kernels (``aucell / gsea / gsva / ora / ulm / mlm /
waggr / zscore / viper / mdt / udt``) is exercised on a small
20-cells × 20-genes synthetic AnnData under both ``engine='cpu'`` and
``engine='gpu'`` (the latter is skipped when CUDA is unavailable).

The point is to catch import-time / dispatcher / kernel-signature
regressions — not to validate numerics. Each call must:

* run without raising,
* write ``adata.obsm['score_<method>']`` with the right shape,
* produce no NaN/inf values.

``mdt`` and ``udt`` need ``xgboost`` for the CPU path; they are skipped
if the dependency is missing.
"""
from __future__ import annotations

import importlib
import importlib.util

import anndata as ad
import numpy as np
import pytest


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _has_cuda() -> bool:
    if not _has_module("torch"):
        return False
    try:
        import torch  # local import keeps test collection cheap

        return bool(torch.cuda.is_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_module("omicverse"), reason="omicverse not importable"
)


N_CELLS = 20
N_GENES = 20


@pytest.fixture(scope="module")
def adata():
    """20 × 20 synthetic AnnData with reproducible Poisson counts."""
    rng = np.random.default_rng(0)
    X = rng.poisson(3.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    a = ad.AnnData(X)
    a.var_names = [f"G{i:02d}" for i in range(N_GENES)]
    a.obs_names = [f"C{i:02d}" for i in range(N_CELLS)]
    return a


@pytest.fixture(scope="module")
def signatures():
    """3 binary + 1 weighted signature, sized so ``tmin=3`` keeps all four."""
    genes = [f"G{i:02d}" for i in range(N_GENES)]
    return {
        "sig_A": genes[0:8],
        "sig_B": genes[6:14],
        "sig_C": genes[12:20],
        "sig_weighted": {
            g: (1.0 if i % 2 == 0 else -1.0) for i, g in enumerate(genes[2:12])
        },
    }


# Per-method extra kwargs. ``times=50`` on gsea dodges a numba kernel bug
# at very small ``times`` values; the other methods need nothing special.
METHOD_KWARGS = {
    "aucell": {},
    "gsea": {"times": 50},
    "gsva": {},
    "ora": {},
    "ulm": {},
    "mlm": {},
    "waggr": {},
    "zscore": {},
    "viper": {},
    "mdt": {"n_estimators": 5},
    "udt": {},
}

# Methods that depend on xgboost for the CPU kernel.
NEEDS_XGBOOST = {"mdt", "udt"}


ENGINES = ["cpu"]
if _has_cuda():
    ENGINES.append("gpu")


@pytest.mark.parametrize("method", list(METHOD_KWARGS.keys()))
@pytest.mark.parametrize("engine", ENGINES)
def test_es_method_runs(adata, signatures, method, engine):
    """Each ov.es method writes a finite (cells × signatures) score matrix."""
    if method in NEEDS_XGBOOST and not _has_module("xgboost"):
        pytest.skip(f"xgboost not installed; {method} CPU path unavailable")

    import omicverse as ov

    fn = getattr(ov.es, method)
    a = adata.copy()

    fn(
        a,
        signatures=signatures,
        tmin=3,
        engine=engine,
        verbose=False,
        **METHOD_KWARGS[method],
    )

    key = f"score_{method}"
    assert key in a.obsm, f"{method!r} ({engine}) did not write adata.obsm[{key!r}]"
    score = np.asarray(a.obsm[key])
    assert score.shape == (
        N_CELLS,
        len(signatures),
    ), f"{method!r} ({engine}) wrote shape {score.shape}, expected ({N_CELLS}, {len(signatures)})"
    assert np.all(np.isfinite(score)), f"{method!r} ({engine}) produced non-finite values"


def test_decoupler_dispatcher(adata, signatures):
    """The unified ``ov.es.decoupler(method=...)`` matches the direct call."""
    import omicverse as ov

    a1 = adata.copy()
    a2 = adata.copy()
    ov.es.aucell(a1, signatures=signatures, tmin=3, engine="cpu")
    ov.es.decoupler(a2, signatures=signatures, tmin=3, engine="cpu", method="aucell")

    np.testing.assert_allclose(
        np.asarray(a1.obsm["score_aucell"]),
        np.asarray(a2.obsm["score_aucell"]),
        rtol=0,
        atol=0,
        err_msg="ov.es.decoupler(method='aucell') diverged from ov.es.aucell",
    )
