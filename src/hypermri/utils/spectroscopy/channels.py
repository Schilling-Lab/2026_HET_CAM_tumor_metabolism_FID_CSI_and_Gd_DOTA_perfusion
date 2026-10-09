"""Channel combination utilities for multi-channel spectroscopy data.

Deprecated: This module has been moved to hypermri.utils.mri.channels
"""

import warnings

# Import everything from the new location
from hypermri.utils.mri.channels import (
    _combine_channels_mean,
    _combine_channels_phase_aligned,
    _combine_channels_rss,
    _combine_channels_sum,
    _combine_channels_weighted_mean,
    combine_channels,
)

warnings.warn(
    "Importing from hypermri.utils.spectroscopy.channels is deprecated. "
    "Please import from hypermri.utils.mri.channels instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "combine_channels",
    "_combine_channels_rss",
    "_combine_channels_sum",
    "_combine_channels_mean",
    "_combine_channels_weighted_mean",
    "_combine_channels_phase_aligned",
]
