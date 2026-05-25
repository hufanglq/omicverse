r"""``method=`` dispatchers for HE→ST prediction and super-resolution."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

if TYPE_CHECKING:
    from anndata import AnnData
    from wsidata import WSIData


PredictMethod = Literal["stpath", "stflow", "hest_fm", "bleep"]
SuperResMethod = Literal["istar"]


def predict_expression(
    wsi: "WSIData",
    *,
    method: PredictMethod = "stpath",
    tile_key: str = "tiles",
    key_added: str | None = None,
    genes: Sequence[str] | None = None,
    organ: str | None = None,
    tech: str | None = "Visium",
    reference: "AnnData | None" = None,
    feature_key: str | None = None,
    fm_backbone: str | None = None,
    device: str | None = None,
    **kwargs,
) -> "AnnData":
    """Predict spot-level spatial gene expression from a tiled WSI.

    Writes the predicted ``AnnData`` to
    ``wsi.tables['{key_added or method}_{tile_key}']`` and also returns it.
    The output AnnData has tile barcodes as ``obs_names``, gene symbols as
    ``var_names``, predicted log1p expression as ``X``, and tile pixel
    centroids in ``obsm['spatial']``.

    Parameters
    ----------
    method
        Prediction backend (see :mod:`ov.space.histo`).
    genes
        Gene symbols to retain. ``None`` returns the model's full vocabulary
        (37k for STPath/STFlow) or all reference genes (HEST-FM/BLEEP).
    organ, tech
        Hint tokens used by STPath/STFlow. Examples: ``organ='Breast'``,
        ``tech='Visium'``. Ignored by HEST-FM.
    reference
        Paired Visium :class:`AnnData` used to fit a per-slide head
        (HEST-FM, BLEEP). Not required by STPath/STFlow zero-shot.
    feature_key
        Name of the tile-level feature table in ``wsi.tables``. Defaults to
        ``'gigapath'`` for STPath/STFlow and ``fm_backbone`` for HEST-FM.
    fm_backbone
        Pathology FM used to extract patch features when ``feature_key`` is
        not already present. Defaults to ``'gigapath'`` (STPath/STFlow) or
        ``'ctranspath'`` (HEST-FM).
    """
    if method == "stpath":
        from ._stpath import predict_stpath
        return predict_stpath(
            wsi,
            tile_key=tile_key,
            key_added=key_added,
            genes=genes,
            organ=organ,
            tech=tech,
            feature_key=feature_key,
            fm_backbone=fm_backbone or "gigapath",
            device=device,
            **kwargs,
        )
    if method == "stflow":
        from ._stflow import predict_stflow
        return predict_stflow(
            wsi,
            tile_key=tile_key,
            key_added=key_added,
            genes=genes,
            organ=organ,
            tech=tech,
            feature_key=feature_key,
            fm_backbone=fm_backbone or "gigapath",
            device=device,
            **kwargs,
        )
    if method == "hest_fm":
        from ._hest_fm import predict_hest_fm
        if reference is None:
            raise ValueError(
                "method='hest_fm' requires `reference=` (a paired Visium AnnData "
                "with H&E in the same physical frame)."
            )
        return predict_hest_fm(
            wsi,
            reference=reference,
            tile_key=tile_key,
            key_added=key_added,
            genes=genes,
            feature_key=feature_key,
            fm_backbone=fm_backbone or "ctranspath",
            device=device,
            **kwargs,
        )
    if method == "bleep":
        from ._bleep import predict_bleep
        if reference is None:
            raise ValueError("method='bleep' requires `reference=`.")
        return predict_bleep(
            wsi,
            reference=reference,
            tile_key=tile_key,
            key_added=key_added,
            genes=genes,
            feature_key=feature_key,
            fm_backbone=fm_backbone or "ctranspath",
            device=device,
            **kwargs,
        )
    raise ValueError(
        f"Unknown method={method!r}. Pick one of: stpath, stflow, hest_fm, bleep."
    )


def super_resolve(
    adata: "AnnData",
    *,
    wsi: "WSIData | None" = None,
    he_image: str | None = None,
    method: SuperResMethod = "istar",
    factor: int = 8,
    genes: Sequence[str] | None = None,
    device: str | None = None,
    cache_dir: str | None = None,
    **kwargs,
) -> "AnnData":
    """Super-resolve a paired (Visium, H&E) sample to near-single-cell tiles.

    Parameters
    ----------
    adata
        Visium :class:`AnnData` carrying spot counts and
        ``obsm['spatial']``.
    wsi
        Optional :class:`wsidata.WSIData` wrapping the source H&E. If absent,
        ``he_image`` must point to the slide and the wrapper opens it.
    he_image
        Path to the full-resolution H&E slide.
    method
        Currently only ``'istar'`` is supported (Nature Biotechnology 2024).
    factor
        Super-resolution factor; ``8`` gives ~8 µm sub-spot tiles for
        Visium (55 µm spots).
    """
    if method == "istar":
        from ._istar import super_resolve_istar
        return super_resolve_istar(
            adata,
            wsi=wsi,
            he_image=he_image,
            factor=factor,
            genes=genes,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    raise ValueError(f"Unknown super-resolution method={method!r}.")
