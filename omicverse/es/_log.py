# Stubbed for omicverse — was decoupler's chatty per-call info logger.
# Public output is handled by omicverse's `@monitor` decorator on the
# `__init__.py` wrappers (coloured SUMMARY box + adata diff). The
# kernels' embedded `_log(...)` calls become no-ops so the vendored
# source stays bit-for-bit identical with upstream decoupler.
def _log(*args, **kwargs):
    return None
