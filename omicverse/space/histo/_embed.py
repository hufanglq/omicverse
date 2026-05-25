r"""Tile-level pathology foundation-model embedding.

Thin convenience wrapper around :func:`lazyslide.tl.feature_extraction`.
LazySlide already registers UNI/UNI2/CONCH/Virchow/Virchow2/GigaPath/
H-Optimus/Phikon/CTransPath/Hibou/Midnight; this layer adds:

* an ``available_backbones`` helper that filters the LazySlide registry
  down to backbones useful for HE→ST prediction;
* a default ``key_added`` convention that matches what the prediction
  backends (STPath, HEST-FM) expect.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wsidata import WSIData

# Pathology FMs that we recommend for HE→ST prediction. Models that only
# do segmentation/QC/zero-shot classification are filtered out.
_RECOMMENDED_BACKBONES = (
    "gigapath",       # STPath / STFlow default
    "uni2", "uni",    # HEST-Bench standard
    "virchow2", "virchow",
    "conch_vision", "conch_v1.5",
    "h-optimus-1", "h-optimus-0", "h0-mini",
    "phikonv2", "phikon",
    "ctranspath",
    "hibou-l", "hibou-b",
    "midnight",
    "chief",
)


def available_backbones() -> list[str]:
    """Return the curated list of pathology FMs recommended for HE→ST."""
    try:
        from lazyslide.models import list_models
        registered = set(list_models())
    except ImportError:
        return list(_RECOMMENDED_BACKBONES)
    return [name for name in _RECOMMENDED_BACKBONES if name in registered]


def embed(
    wsi: "WSIData",
    model: str = "gigapath",
    *,
    tile_key: str = "tiles",
    key_added: str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | None = None,
    amp: bool = True,
    token: str | None = None,
    pbar: bool = True,
    **kwargs,
) -> "WSIData":
    """Extract per-tile foundation-model embeddings.

    Writes the embedding AnnData into
    ``wsi.tables['{key_added}_{tile_key}']`` (default ``'<model>_tiles'``)
    where ``X`` is the (N_tiles × D) feature matrix and ``obs`` carries
    tile barcodes matching ``wsi.shapes[tile_key]``.

    Parameters
    ----------
    model
        Backbone name; one of :func:`available_backbones`.
    token
        HuggingFace token for gated models (UNI2/CONCH/Virchow2/GigaPath).
        Falls back to ``HUGGING_FACE_HUB_TOKEN`` env var or the cached
        ``~/.cache/huggingface/token``.
    """
    import lazyslide as zs

    # NOTE: LazySlide only applies its `{name}_{tile_key}` naming when
    # ``key_added`` is ``None``. If the caller hands us a short name we
    # expand it ourselves to keep the table-naming contract.
    if key_added is None:
        lazyslide_key = None
    else:
        lazyslide_key = key_added if key_added.endswith(f"_{tile_key}") else f"{key_added}_{tile_key}"

    zs.tl.feature_extraction(
        wsi,
        model=model,
        tile_key=tile_key,
        key_added=lazyslide_key,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        amp=amp,
        token=token,
        pbar=pbar,
        **kwargs,
    )
    return wsi
