"""
Shim for backward compatibility of logging utilities.

Deprecated: This module has been moved to hypermri.utils.system.logging
"""

import warnings

warnings.warn(
    "Importing from hypermri.utils.utils_logging is deprecated. "
    "Please import from hypermri.utils.system.logging instead.",
    DeprecationWarning,
    stacklevel=2,
)
