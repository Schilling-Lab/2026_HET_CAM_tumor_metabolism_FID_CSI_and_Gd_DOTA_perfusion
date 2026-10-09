"""
Shim for backward compatibility of image segmentation utilities.

Deprecated: This module has been moved to hypermri.utils.image.segmentation
"""

import warnings

warnings.warn(
    "Importing from hypermri.utils.image_segmenter_3d is deprecated. "
    "Please import from hypermri.utils.image.segmentation instead.",
    DeprecationWarning,
    stacklevel=2,
)
