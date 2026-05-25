"""Thin wrapper around :mod:`pyfunkyheatmap` for ``omicverse.pl``.

Re-exports the upstream :func:`pyfunkyheatmap.funky_heatmap` entry point and
helpers so users can call them via ``ov.pl.funky_heatmap(...)`` without
importing the third-party package explicitly. The dependency is loaded
lazily so that importing :mod:`omicverse.pl` doesn't fail if
``pyfunkyheatmap`` isn't installed.

Example::

    import omicverse as ov
    ov.style(font_path='Arial')

    import pandas as pd
    df = pd.DataFrame({
        'id':  ['A', 'B', 'C', 'D'],
        'x':   [0.1, 0.55, 0.8, 0.95],
        'y':   [0.5, 0.25, 0.75, 0.6],
        'tag': ['alpha', 'beta', 'gamma', 'delta'],
    })
    fh = ov.pl.funky_heatmap(df)
    fh.save('out.png', dpi=150)

Upstream: https://github.com/omicverse/py-funkyheatmap
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_MISSING_MSG = (
    "ov.pl.funky_heatmap requires the `pyfunkyheatmap` package.\n"
    "Install with: pip install pyfunkyheatmap"
)


def _load():
    try:
        return import_module("pyfunkyheatmap")
    except ImportError as exc:  # pragma: no cover - exercised at call time
        raise ImportError(_MISSING_MSG) from exc


def funky_heatmap(*args: Any, **kwargs: Any):
    """Generate a funky heatmap from a :class:`pandas.DataFrame`.

    Thin wrapper around :func:`pyfunkyheatmap.funky_heatmap` — see the
    upstream docs for the full parameter list.
    """
    return _load().funky_heatmap(*args, **kwargs)


def position_arguments(**kwargs: Any):
    """Build a layout-args container.

    Thin wrapper around :func:`pyfunkyheatmap.position_arguments`.
    """
    return _load().position_arguments(**kwargs)


def scale_minmax(x):
    """Min-max scale a vector to ``[0, 1]``.

    Thin wrapper around :func:`pyfunkyheatmap.scale_minmax`.
    """
    return _load().scale_minmax(x)


__all__ = ["funky_heatmap", "position_arguments", "scale_minmax"]
