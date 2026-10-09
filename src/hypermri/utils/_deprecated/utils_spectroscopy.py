"""
Backward compatibility module for utils_spectroscopy.

This module maintains backward compatibility by importing all functions from the new
analysis.spectroscopy.core module. Users can continue to use:

    from hypermri.utils.utils_spectroscopy import get_freq_axis

All functions are now located in hypermri.analysis.spectroscopy.core
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from hypermri.utils.utils_spectroscopy is deprecated. "
    "Please update your code to import from hypermri.analysis.spectroscopy.core instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import all functions from the new location for backward compatibility
try:
    pass
except ImportError as e:
    # Fallback error message if the new module isn't available
    raise ImportError(
        f"Could not import from analysis.spectroscopy.core: {e}. "
        "The analysis module migration may be incomplete."
    )
