"""
Shim for backward compatibility of sequence utilities.

Deprecated: This module has been moved to hypermri.utils.sequence.generation
"""

import warnings

warnings.warn(
    "Importing from hypermri.utils.utils_sequences is deprecated. "
    "Please import from hypermri.utils.sequence.generation instead.",
    DeprecationWarning,
    stacklevel=2,
)
