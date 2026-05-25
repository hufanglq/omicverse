r"""Convenience loader for the 10x Visium Breast Cancer Block A Section 1.

This is the canonical demo dataset used across the HE-zoo tutorials so
every backend predicts on the **same H&E**. The full-resolution image
weighs ~1.7 GB; the loader caches everything under
``OV_HISTO_CACHE/he_zoo/visium_breast`` (default
``~/.cache/omicverse/histo/he_zoo/visium_breast``) and only re-downloads
missing assets.

Tutorials that need just predictions and no training reference can drop
the full-resolution image and use the Space Ranger ``tissue_hires_image``
that ships inside ``spatial.tar.gz``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData
    from wsidata import WSIData


_BASE = (
    "https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/"
    "V1_Breast_Cancer_Block_A_Section_1/"
)
_FILES = {
    "counts": "V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5",
    "spatial": "V1_Breast_Cancer_Block_A_Section_1_spatial.tar.gz",
    "image": "V1_Breast_Cancer_Block_A_Section_1_image.tif",
}


def _default_dir() -> Path:
    base = os.environ.get("OV_HISTO_CACHE", Path.home() / ".cache" / "omicverse" / "histo")
    return Path(base) / "he_zoo" / "visium_breast"


def _download(target: Path, url: str) -> None:
    import urllib.request
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size < 1024:
        print(f"  downloading {target.name} …", flush=True)
        urllib.request.urlretrieve(url, target)


def download_breast(
    cache_dir: str | Path | None = None,
    *,
    include_image: bool = True,
) -> Path:
    """Download the Visium Breast Cancer demo dataset.

    Returns the dataset directory. Skips files that already exist.
    """
    dst = Path(cache_dir) if cache_dir is not None else _default_dir()
    dst.mkdir(parents=True, exist_ok=True)
    _download(dst / _FILES["counts"], _BASE + _FILES["counts"])
    _download(dst / _FILES["spatial"], _BASE + _FILES["spatial"])
    if not (dst / "spatial").is_dir():
        import tarfile
        with tarfile.open(dst / _FILES["spatial"]) as tar:
            tar.extractall(dst)
    if include_image:
        _download(dst / _FILES["image"], _BASE + _FILES["image"])
    return dst


def load_breast(
    cache_dir: str | Path | None = None,
    *,
    include_image: bool = True,
) -> "tuple[AnnData, WSIData | None]":
    """Download the demo sample and return ``(adata, wsi)``.

    Examples
    --------
    >>> import omicverse as ov
    >>> adata, wsi = ov.space.histo.load_breast()
    >>> ov.space.histo.tile(wsi, tile_px=224, mpp=0.5)
    """
    from ._io import read_visium_with_image
    base = download_breast(cache_dir=cache_dir, include_image=include_image)
    return read_visium_with_image(
        base,
        image_path=(base / _FILES["image"]) if include_image else None,
        count_file=_FILES["counts"],
    )
