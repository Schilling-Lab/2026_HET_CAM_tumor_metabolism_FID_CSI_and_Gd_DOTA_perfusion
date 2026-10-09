"""
Backward compatibility module for utils_spectral_fitting.

This module maintains backward compatibility by importing all functions from the new
analysis.spectroscopy.advanced module. Users can continue to use:

    from hypermri.utils.utils_spectral_fitting import fid_basefunc

All functions are now located in hypermri.analysis.spectroscopy.advanced
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from hypermri.utils.utils_spectral_fitting is deprecated. "
    "Please update your code to import from hypermri.analysis.spectroscopy.advanced instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import all functions from the new location for backward compatibility
try:
    pass
except ImportError as e:
    # Fallback error message if the new module isn't available
    raise ImportError(
        f"Could not import from analysis.spectroscopy.advanced: {e}. "
        "The analysis module migration may be incomplete."
    )
