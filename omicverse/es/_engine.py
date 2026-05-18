"""Engine selector for the ov.es scoring kernels.

Each method has a numba CPU kernel (`_func_<name>`) and optionally a
torch GPU kernel (`_func_<name>_torch`). The wrappers in
``ov.es.__init__`` expose an ``engine='auto'|'cpu'|'gpu'`` kwarg; this
module resolves the choice consistently across methods.

Resolution rules
----------------
``'auto'`` (default) — picks GPU when torch+CUDA available **and** the
method exposes a torch kernel; else CPU.

``'cpu'`` — forces the numba kernel.

``'gpu'`` — forces the torch kernel. Raises if torch is missing, CUDA
is unavailable, or the method has no torch implementation yet.
"""
from __future__ import annotations

from typing import Literal, Optional


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _cuda_available() -> bool:
    if not _torch_available():
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def torch_device(prefer: str = 'cuda'):
    """Return a torch.device handle, falling back to CPU when needed."""
    import torch
    if prefer == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def resolve_engine(
    engine: Literal['auto', 'cpu', 'gpu'] = 'auto',
    has_torch_kernel: bool = False,
) -> str:
    """Return either ``'cpu'`` or ``'gpu'`` from a tri-state input.

    Parameters
    ----------
    engine
        User request — one of ``'auto'``, ``'cpu'``, ``'gpu'``.
    has_torch_kernel
        Whether the calling method has a torch implementation. When
        ``False`` and ``engine='gpu'`` was explicitly requested we
        raise; when ``engine='auto'`` we silently fall back to CPU.

    Returns
    -------
    str
        Resolved engine identifier — ``'cpu'`` or ``'gpu'``.
    """
    if engine not in ('auto', 'cpu', 'gpu'):
        raise ValueError(
            f"engine must be 'auto' | 'cpu' | 'gpu', got {engine!r}"
        )

    if engine == 'cpu':
        return 'cpu'

    if engine == 'gpu':
        if not has_torch_kernel:
            raise RuntimeError(
                "engine='gpu' requested but this method does not yet "
                "ship a torch kernel. Use engine='cpu' (or 'auto')."
            )
        if not _torch_available():
            raise ImportError(
                "engine='gpu' requested but torch is not installed. "
                "`pip install torch` or use engine='cpu'."
            )
        if not _cuda_available():
            raise RuntimeError(
                "engine='gpu' requested but CUDA is not available. "
                "Falling back to CPU is opt-in: pass engine='auto'."
            )
        return 'gpu'

    # engine == 'auto'
    if has_torch_kernel and _cuda_available():
        return 'gpu'
    return 'cpu'
