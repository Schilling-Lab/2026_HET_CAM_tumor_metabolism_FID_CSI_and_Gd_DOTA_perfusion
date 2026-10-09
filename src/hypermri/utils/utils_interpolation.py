"""
Shim for backward compatibility of interpolation utilities.

Deprecated: This module has been moved to hypermri.utils.image.interpolation
"""

import warnings

warnings.warn(
    "Importing from hypermri.utils.utils_interpolation is deprecated. "
    "Please import from hypermri.utils.image.interpolation instead.",
    DeprecationWarning,
    stacklevel=2,
)
