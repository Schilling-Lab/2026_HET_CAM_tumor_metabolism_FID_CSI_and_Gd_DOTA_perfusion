"""
Shim for backward compatibility of FEXI utilities.

Deprecated: This module has been moved to hypermri.utils.spectroscopy.fexi
"""

import warnings

warnings.warn(
    "Importing from hypermri.utils.fexi is deprecated. "
    "Please import from hypermri.utils.spectroscopy.fexi instead.",
    DeprecationWarning,
    stacklevel=2,
)
