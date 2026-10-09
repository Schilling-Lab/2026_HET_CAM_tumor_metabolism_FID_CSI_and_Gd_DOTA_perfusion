"""
Shim for backward compatibility of study utilities.

Deprecated: This module has been moved to hypermri.utils.study.manipulation
"""

import warnings

warnings.warn(
    "Importing from hypermri.utils.utils_study is deprecated. "
    "Please import from hypermri.utils.study.manipulation instead.",
    DeprecationWarning,
    stacklevel=2,
)
