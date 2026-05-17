# Vendored from `decoupler` (https://github.com/scverse/decoupler) by
# omicverse for in-tree GPU acceleration work. Original copyright by
# the decoupler authors, redistributed under decoupler's GPL-3.0
# license. Cross-module imports rewritten from `decoupler.*` to
# `omicverse.es.*` (see scripts/vendor_decoupler.py).

from omicverse.es._aucell import aucell
from omicverse.es._gsea import gsea
from omicverse.es._gsva import gsva
from omicverse.es._mdt import mdt
from omicverse.es._mlm import mlm
from omicverse.es._ora import ora
from omicverse.es._udt import udt
from omicverse.es._ulm import ulm
from omicverse.es._viper import viper
from omicverse.es._waggr import waggr
from omicverse.es._zscore import zscore

_methods = [
    aucell,
    gsea,
    gsva,
    mdt,
    mlm,
    ora,
    udt,
    ulm,
    viper,
    waggr,
    zscore,
]
