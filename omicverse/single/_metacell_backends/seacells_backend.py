"""SEACells backend (Persad et al., Nat Biotech 2023)."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy import sparse

from .base import FitResult, MetaCellBackend


class SEACellsBackend(MetaCellBackend):
    name = "seacells"
    capabilities = {"soft", "latent"}

    def __init__(
        self,
        adata,
        use_rep: str = "X_pca",
        n_metacells: Optional[int] = None,
        device: str = "cpu",
        random_state: int = 0,
        verbose: bool = False,
        n_waypoint_eigs: int = 10,
        n_neighbors: int = 15,
        convergence_epsilon: float = 1e-3,
        l2_penalty: float = 0.0,
        max_franke_wolfe_iters: int = 50,
        use_sparse: bool = False,
        **kwargs,
    ):
        self.adata = adata
        self.use_rep = use_rep
        self.n_metacells = n_metacells or (adata.n_obs // 75)
        self.device = device
        self.random_state = random_state
        self._init_kwargs = dict(
            verbose=verbose,
            n_waypoint_eigs=n_waypoint_eigs,
            n_neighbors=n_neighbors,
            convergence_epsilon=convergence_epsilon,
            l2_penalty=l2_penalty,
            max_franke_wolfe_iters=max_franke_wolfe_iters,
            use_sparse=use_sparse,
        )
        self._extra = kwargs
        self.model = None

    def fit(self, n_metacells: Optional[int] = None, min_iter: int = 10, max_iter: int = 50, **kwargs) -> FitResult:
        from ...external.SEACells.core import SEACells
        from ...external.SEACells.core import summarize_by_SEACell, summarize_by_soft_SEACell  # noqa: F401

        if n_metacells is not None:
            self.n_metacells = n_metacells

        self.model = SEACells(
            self.adata,
            build_kernel_on=self.use_rep,
            n_SEACells=self.n_metacells,
            use_gpu=(self.device != "cpu"),
            **self._init_kwargs,
        )
        self.model.construct_kernel_matrix()
        self.model.initialize_archetypes()

        t0 = time.time()
        self.model.fit(min_iter=min_iter, max_iter=max_iter, **kwargs)
        runtime = time.time() - t0

        labels = self.adata.obs["SEACell"].astype(str)
        uniq = {lab: i for i, lab in enumerate(sorted(labels.unique()))}
        assignments = labels.map(uniq).to_numpy().astype(np.int64)

        # SEACells writes A_ as (n_metacells, n_cells); we want (n_cells, n_mc).
        soft = sparse.csr_matrix(self.model.A_.T)

        latent = None
        if self.use_rep in self.adata.obsm:
            latent = np.asarray(self.adata.obsm[self.use_rep])

        return FitResult(
            assignments=assignments,
            soft=soft,
            latent=latent,
            codebook=None,
            n_iter=len(getattr(self.model, "RSS_iters", [])),
            converged=True,
            runtime_s=float(runtime),
            backend_meta={"label_map": uniq},
        )

    # capability methods -----------------------------------------------------

    def soft_membership(self) -> sparse.csr_matrix:
        if self.model is None or self.model.A_ is None:
            raise RuntimeError("Call .fit() first.")
        return sparse.csr_matrix(self.model.A_.T)

    def latent(self) -> np.ndarray:
        return np.asarray(self.adata.obsm[self.use_rep])

    # persistence -------------------------------------------------------------

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(
                {"model": self.model, "use_rep": self.use_rep, "n_metacells": self.n_metacells},
                f,
            )

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.model = state["model"]
        self.use_rep = state["use_rep"]
        self.n_metacells = state["n_metacells"]
