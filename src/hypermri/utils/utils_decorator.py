"""
Shim for backward compatibility of decorator utilities.

Deprecated: This module has been moved to hypermri.utils.system.decorators
"""

import warnings

warnings.warn(
    "Importing from hypermri.utils.utils_decorator is deprecated. "
    "Please import from hypermri.utils.system.decorators instead.",
    DeprecationWarning,
    stacklevel=2,
)
