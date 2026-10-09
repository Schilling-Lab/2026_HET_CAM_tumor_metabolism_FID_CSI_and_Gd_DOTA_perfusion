"""
Backward compatibility module for utils_denoising.

This module maintains backward compatibility by importing all functions from the new
analysis.denoising.algorithms module. Users can continue to use:

    from hypermri.utils.utils_denoising import pca_denoise

All functions are now located in hypermri.analysis.denoising.algorithms
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from hypermri.utils.utils_denoising is deprecated. "
    "Please update your code to import from hypermri.analysis.denoising.algorithms instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import all functions from the new location for backward compatibility
try:
    pass
except ImportError as e:
    # Fallback error message if the new module isn't available
    raise ImportError(
        f"Could not import from analysis.denoising.algorithms: {e}. "
        "The analysis module migration may be incomplete."
    )
