"""
Shim for backward compatibility of channel combination utilities.

Deprecated: This module has been moved to hypermri.utils.mri.channels
"""

import warnings

warnings.warn(
    "Importing from hypermri.utils.channel_combination is deprecated. "
    "Please import from hypermri.utils.mri.channels instead.",
    DeprecationWarning,
    stacklevel=2,
)
