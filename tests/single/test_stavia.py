import types
import builtins

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import omicverse.single._stavia as stavia_mod
from omicverse.single._stavia import StaVIA


def _make_adata():
    obs = pd.DataFrame(
        {
            "clusters": ["stem", "stem", "late", "late"],
            "time": [0, 1, 2, 3],
            "slice": ["s1", "s1", "s2", "s2"],
        },
        index=[f"cell{i}" for i in range(4)],
    )
    adata = AnnData(
        X=np.ones((4, 3)),
        obs=obs,
        var=pd.DataFrame(index=[f"gene{i}" for i in range(3)]),
    )
    adata.obsm["X_pca"] = np.arange(20, dtype=float).reshape(4, 5)
    adata.obsm["X_umap"] = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    adata.obsm["spatial"] = np.array(
        [[10.0, 20.0, 1.0], [11.0, 20.0, 1.0], [50.0, 60.0, 2.0], [51.0, 61.0, 2.0]]
    )
    return adata


def _fake_via_backend(captured):
    class FakeVIA:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            n_obs = kwargs["data"].shape[0]
            self.single_cell_pt_markov = np.linspace(0.0, 1.0, n_obs)
            self.labels = np.array([0, 0, 1, 1])
            self.terminal_clusters = [5, 8]
            self.single_cell_bp = np.vstack(
                [np.linspace(1.0, 0.0, n_obs), np.linspace(0.0, 1.0, n_obs)]
            )
            self.ran = False

        def run_VIA(self):
            self.ran = True

    return types.SimpleNamespace(core=types.SimpleNamespace(VIA=FakeVIA))


def test_stavia_fit_translates_anndata_keys_and_writes_results(monkeypatch):
    adata = _make_adata()
    captured = {}
    monkeypatch.setattr(
        stavia_mod,
        "_load_via_backend",
        lambda *, rw2_mode=False: _fake_via_backend(captured),
    )

    model = StaVIA(
        adata,
        use_rep="X_pca",
        n_comps=3,
        basis="X_umap",
        cluster_key="clusters",
        spatial_key="spatial",
        time_key="time",
        sample_key="slice",
        spatial_knn=7,
        root="stem",
        random_seed=11,
    ).fit()

    assert model.model.ran is True
    assert captured["data"].shape == (4, 3)
    np.testing.assert_allclose(captured["embedding"], adata.obsm["X_umap"])
    np.testing.assert_allclose(captured["spatial_coords"], adata.obsm["spatial"][:, :2])
    assert captured["do_spatial_knn"] is True
    assert captured["spatial_knn"] == 7
    assert captured["spatial_aux"] == ["s1", "s1", "s2", "s2"]
    assert captured["time_series"] is True
    assert captured["time_series_labels"] == [0, 1, 2, 3]
    assert captured["root_user"] == ["stem"]
    assert captured["random_seed"] == 11

    assert "stavia_pseudotime" in adata.obs
    np.testing.assert_allclose(adata.obs["stavia_pseudotime"], np.linspace(0.0, 1.0, 4))
    assert list(adata.obs["stavia_cluster"].astype(str)) == ["0", "0", "1", "1"]
    assert "stavia_lineage_probabilities" in adata.obsm
    lineage = adata.obsm["stavia_lineage_probabilities"]
    assert list(lineage.columns) == ["lineage_5", "lineage_8"]
    assert lineage.shape == (4, 2)
    assert adata.uns["stavia"]["spatial_key"] == "spatial"
    assert adata.uns["stavia"]["pseudotime_key"] == "stavia_pseudotime"


def test_stavia_rejects_missing_spatial_key():
    adata = _make_adata()
    model = StaVIA(adata, spatial_key="missing")

    with pytest.raises(KeyError, match="spatial_key='missing'"):
        model.fit()


def test_stavia_plot_methods_are_not_exposed_on_wrapper():
    for name in (
        "plot_stream",
        "plot_graph",
        "plot_trajectory",
        "plot_lineage_probability",
    ):
        assert not hasattr(StaVIA, name)


def test_stavia_missing_core_dependency_message(monkeypatch):
    real_import_module = stavia_mod.importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "leidenalg":
            raise ImportError("blocked import: leidenalg")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(stavia_mod.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="core dependencies"):
        stavia_mod._load_via_backend()


def test_stavia_dependency_helpers_avoid_module_level_required_lists():
    assert not hasattr(stavia_mod, "_STAVIA_REQUIRED_MODULES")
    assert not hasattr(stavia_mod, "_STAVIA_RW2_MODULES")
    assert stavia_mod._stavia_required_modules() == ("leidenalg",)
    assert stavia_mod._stavia_required_modules(rw2=True) == (
        "leidenalg",
        "pecanpy",
        "numba_progress",
    )


def test_stavia_loader_does_not_require_hnswlib_or_pygam_for_basic_mode(monkeypatch):
    checked = []

    def fake_require_modules(dependencies, *, rw2=False):
        checked.append((tuple(dependencies), rw2))

    def fake_import_module(name, *args, **kwargs):
        if name == "..external.VIA":
            return types.SimpleNamespace(core=types.SimpleNamespace(VIA=object))
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(stavia_mod, "_require_modules", fake_require_modules)
    monkeypatch.setattr(stavia_mod.importlib, "import_module", fake_import_module)

    stavia_mod._load_via_backend()

    assert checked == [(("leidenalg",), False)]


def test_stavia_loader_checks_rw2_dependencies_only_when_enabled(monkeypatch):
    checked = []

    def fake_require_modules(dependencies, *, rw2=False):
        checked.append((tuple(dependencies), rw2))

    def fake_import_module(name, *args, **kwargs):
        if name == "..external.VIA":
            return types.SimpleNamespace(core=types.SimpleNamespace(VIA=object))
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(stavia_mod, "_require_modules", fake_require_modules)
    monkeypatch.setattr(stavia_mod.importlib, "import_module", fake_import_module)

    stavia_mod._load_via_backend(rw2_mode=True)

    assert checked == [
        (("leidenalg",), False),
        (("pecanpy", "numba_progress"), True),
    ]


def test_stavia_rw2_missing_dependency_message(monkeypatch):
    def fake_require_modules(dependencies, *, rw2=False):
        if rw2:
            stavia_mod._raise_stavia_dependency_error(dependencies, rw2=rw2)

    monkeypatch.setattr(stavia_mod, "_require_modules", fake_require_modules)

    with pytest.raises(ImportError, match="pip install pecanpy numba-progress"):
        stavia_mod._load_via_backend(rw2_mode=True)


def test_via_knn_uses_sklearn_fallback_when_hnswlib_is_missing(monkeypatch):
    from omicverse.external.VIA import utils_via

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "hnswlib":
            raise ImportError("blocked import: hnswlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    data = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    index = utils_via._construct_knn(data, knn=2, distance="l2", num_threads=1)
    labels, distances = index.knn_query(data[:1], k=2)

    assert index.__class__.__name__ == "_SklearnKNNIndex"
    assert labels.shape == (1, 2)
    assert distances.shape == (1, 2)


def test_via_knn_prefers_hnswlib_when_available(monkeypatch):
    from omicverse.external.VIA import utils_via

    class FakeIndex:
        def __init__(self, space, dim):
            self.space = space
            self.dim = dim
            self.calls = []

        def set_num_threads(self, value):
            self.calls.append(("set_num_threads", value))

        def init_index(self, **kwargs):
            self.calls.append(("init_index", kwargs))

        def add_items(self, data):
            self.calls.append(("add_items", data.shape))

        def set_ef(self, value):
            self.calls.append(("set_ef", value))

    fake_hnswlib = types.SimpleNamespace(Index=FakeIndex)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "hnswlib":
            return fake_hnswlib
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)

    index = utils_via._construct_knn(data, knn=2, distance="l2", num_threads=1)

    assert isinstance(index, FakeIndex)
    assert index.space == "l2"
    assert index.dim == 2
    assert ("add_items", data.shape) in index.calls


def test_via_pygam_is_lazy_and_reports_actionable_install_hint(monkeypatch):
    from omicverse.external.VIA import utils_via

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pygam":
            raise ImportError("blocked import: pygam")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pygam.*omicverse\\[full\\]"):
        utils_via.get_gene_trend(types.SimpleNamespace())


def test_via_rw2_optional_dependency_detection():
    from omicverse.external.VIA import core as via_core

    assert via_core._is_missing_rw2_dependency(ImportError("No module named 'pecanpy'"))
    assert via_core._is_missing_rw2_dependency(
        ImportError("blocked optional import: numba_progress")
    )
    assert not via_core._is_missing_rw2_dependency(ImportError("unrelated import failure"))


def test_trajinfer_dispatches_stavia(monkeypatch):
    pytest.importorskip("torch")
    from omicverse.single._traj import TrajInfer

    adata = _make_adata()
    captured = {}
    monkeypatch.setattr(
        stavia_mod,
        "_load_via_backend",
        lambda *, rw2_mode=False: _fake_via_backend(captured),
    )

    traj = TrajInfer(adata, basis="X_umap", use_rep="X_pca", n_comps=2, groupby="clusters")
    model = traj.inference(
        method="stavia",
        spatial_key="spatial",
        time_key="time",
        sample_key="slice",
        key_added="traj_stavia",
    )

    assert model.__class__.__name__ == "StaVIA"
    assert traj.stavia is model
    assert captured["data"].shape == (4, 2)
    assert captured["true_label"].equals(adata.obs["clusters"])
    assert "traj_stavia_pseudotime" in adata.obs
