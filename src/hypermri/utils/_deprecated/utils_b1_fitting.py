"""
Backward compatibility module for utils_b1_fitting.

This module maintains backward compatibility by importing all functions from the new
analysis.anatomical.b1_mapping module. Users can continue to use:

    from hypermri.utils.utils_b1_fitting import refpow_parameter_adjustment

All functions are now located in hypermri.analysis.anatomical.b1_mapping
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from hypermri.utils.utils_b1_fitting is deprecated. "
    "Please update your code to import from hypermri.analysis.anatomical.b1_mapping instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import all functions from the new location for backward compatibility
try:
    pass
except ImportError as e:
    # Fallback error message if the new module isn't available
    raise ImportError(
        f"Could not import from analysis.anatomical.b1_mapping: {e}. "
        "The analysis module migration may be incomplete."
    )
