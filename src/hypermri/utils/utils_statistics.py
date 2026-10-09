"""
Shim for backward compatibility of statistics utilities.

Deprecated: This module has been moved to hypermri.utils.mathematics.statistics
"""

import warnings

warnings.warn(
    "Importing from hypermri.utils.utils_statistics is deprecated. "
    "Please import from hypermri.utils.mathematics.statistics instead.",
    DeprecationWarning,
    stacklevel=2,
)
