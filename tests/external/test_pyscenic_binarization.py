import numpy as np
import pandas as pd

from omicverse.external.pyscenic.binarization import binarize, derive_threshold


def test_derive_threshold_hdt_does_not_require_numpy_msort(monkeypatch):
    monkeypatch.delattr(np, "msort", raising=False)
    auc_mtx = pd.DataFrame({"regulon": [0.1, 0.2, 0.2, 0.8, 0.9, 1.0]})

    threshold = derive_threshold(auc_mtx, "regulon", seed=1, method="hdt")

    assert np.isfinite(threshold)


def test_binarize_exposes_bic_threshold_method():
    auc_mtx = pd.DataFrame(
        {
            "regulon_a": [0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
            "regulon_b": [0.05, 0.06, 0.07, 0.5, 0.6, 0.7],
        }
    )

    binary_mtx, thresholds = binarize(
        auc_mtx,
        seed=1,
        num_workers=1,
        method="bic",
        use_tqdm=False,
    )

    assert list(binary_mtx.columns) == list(auc_mtx.columns)
    assert list(thresholds.index) == list(auc_mtx.columns)
    assert np.isfinite(thresholds).all()


def test_binarize_parallel_bic_uses_picklable_worker():
    auc_mtx = pd.DataFrame(
        {
            "regulon_a": [0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
            "regulon_b": [0.05, 0.06, 0.07, 0.5, 0.6, 0.7],
        }
    )

    binary_mtx, thresholds = binarize(
        auc_mtx,
        seed=1,
        num_workers=2,
        method="bic",
        use_tqdm=False,
    )

    assert binary_mtx.shape == auc_mtx.shape
    assert list(thresholds.index) == list(auc_mtx.columns)


def test_binarize_preserves_empty_regulon_matrix():
    auc_mtx = pd.DataFrame(index=["cell_a", "cell_b"])

    binary_mtx, thresholds = binarize(
        auc_mtx,
        num_workers=1,
        method="bic",
        use_tqdm=False,
    )

    assert binary_mtx.shape == auc_mtx.shape
    assert thresholds.empty
    assert list(thresholds.index) == []
