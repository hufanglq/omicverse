"""Apple-Silicon (MLX/metal) edge-SGD for non-parametric UMAP.

Mirrors :func:`._sgd.optimize_layout_torch` but runs the gradient
computation on Apple's MLX (metal) backend — the same Apple path
omicverse already uses for PCA (``_pca_mlx.MLXPCA``) and Harmony, rather
than torch-MPS (MLX has the float64/sparse coverage MPS lacks here).

Design: the cheap O(edges) schedule bookkeeping and the RNG stay in NumPy
on the host; only the per-epoch gather → gradient → scatter-add runs on
MLX. On Apple Silicon's unified memory the host↔device handoff of the
(small) index arrays is nearly free, so this keeps the embedding resident
on metal while sidestepping MLX's gaps around boolean-index / argwhere.

Negative sampling uses the same schedule-based per-edge count as the torch
path (``(n - epoch_of_next_negative_sample) / epochs_per_negative_sample``).
Bit-equality with umap-learn is not the goal (different RNG + parallel
updates); the embedding is validated by trustworthiness. Spectral init is
done on the CPU (scipy) by the caller.
"""
from __future__ import annotations

import numpy as np

CLIP = 4.0


def mlx_available() -> bool:
    """True when MLX is importable and a metal device is present."""
    try:
        import mlx.core as mx

        return bool(mx.metal.is_available())
    except Exception:  # noqa: BLE001
        return False


def optimize_layout_mlx(
    embedding: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    n_epochs: int,
    epochs_per_sample: np.ndarray,
    a: float,
    b: float,
    *,
    gamma: float = 1.0,
    initial_alpha: float = 1.0,
    negative_sample_rate: float = 5.0,
    seed: int = 0,
    move_other: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Optimise ``embedding`` with the MLX (metal) edge-SGD.

    Same gradient formulas / clip(±4) / linear alpha decay as the torch and
    umap-learn paths. Returns the optimised ``(n, dim)`` float32 array.
    """
    import mlx.core as mx

    head = np.asarray(head)
    tail = np.asarray(tail)
    eps_sample = np.asarray(epochs_per_sample, dtype=np.float64)
    eps_neg = eps_sample / negative_sample_rate
    next_sample = eps_sample.copy()
    next_neg = eps_neg.copy()
    sampleable = eps_sample > 0

    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    n_vertices = embedding.shape[0]
    emb = mx.array(np.ascontiguousarray(embedding, dtype=np.float32))

    rng_iter = range(n_epochs)
    if verbose:
        try:
            from tqdm.auto import tqdm

            rng_iter = tqdm(rng_iter, desc="UMAP(MLX)")
        except Exception:  # noqa: BLE001
            pass

    for n in rng_iter:
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))
        active = np.nonzero(sampleable & (next_sample <= n))[0]
        if active.size == 0:
            continue
        j = head[active]
        k = tail[active]

        # ---- attractive ----
        jx = mx.array(j)
        kx = mx.array(k)
        yj = emb[jx]
        yk = emb[kx]
        diff = yj - yk
        d2 = mx.sum(diff * diff, axis=1)
        posm = d2 > 0.0
        d2c = mx.maximum(d2, 1e-12)
        gc = (-2.0 * a * b * mx.power(d2c, b - 1.0)) / (a * mx.power(d2c, b) + 1.0)
        gc = mx.where(posm, gc, mx.zeros_like(gc))
        grad = mx.clip(mx.expand_dims(gc, 1) * diff, -CLIP, CLIP) * alpha
        emb = emb.at[jx].add(grad)
        if move_other:
            emb = emb.at[kx].add(-grad)

        # ---- negative sampling (fixed rate per active edge) ----
        n_neg = ((n - next_neg[active]) / eps_neg[active]).astype(np.int64)
        n_neg = np.clip(n_neg, 0, None)
        total = int(n_neg.sum())
        if total > 0:
            anchors = np.repeat(j, n_neg)
            neg_k = rng.integers(0, n_vertices, size=total)
            ax = mx.array(anchors)
            nx = mx.array(neg_k)
            ya = emb[ax]
            yn = emb[nx]
            dn = ya - yn
            d2n = mx.sum(dn * dn, axis=1)
            posn = d2n > 0.0
            d2nc = mx.maximum(d2n, 1e-12)
            gcn = (2.0 * gamma * b) / ((0.001 + d2nc) * (a * mx.power(d2nc, b) + 1.0))
            gcn = mx.where(posn, gcn, mx.zeros_like(gcn))
            gradn = mx.clip(mx.expand_dims(gcn, 1) * dn, -CLIP, CLIP) * alpha
            emb = emb.at[ax].add(gradn)
            next_neg[active] += n_neg * eps_neg[active]

        next_sample[active] += eps_sample[active]
        mx.eval(emb)  # materialise; keeps the lazy graph from growing per epoch

    return np.array(emb).astype(np.float32)
