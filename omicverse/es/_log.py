# Vendored from `decoupler` (https://github.com/scverse/decoupler) by
# omicverse for in-tree GPU acceleration work. Original copyright by
# the decoupler authors, redistributed under decoupler's GPL-3.0
# license. Cross-module imports rewritten from `decoupler.*` to
# `omicverse.es.*` (see scripts/vendor_decoupler.py).

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def _log(message: str, level: str = "info", verbose: bool = False) -> None:
    """
    Log a message with a specified logging level.

    Parameters
    ----------
    message
        The message to log.
    level
        The logging level.
    verbose
        Whether to emit the log.
    """
    level = level.lower()
    if verbose:
        if level == "warn":
            logging.warning(message)
        elif level == "info":
            logging.info(message)
