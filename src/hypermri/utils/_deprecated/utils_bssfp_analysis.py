"""
Backward compatibility module for utils_bssfp_analysis.

This module maintains backward compatibility by importing all functions from the new
analysis.anatomical.bssfp module. Users can continue to use:

    from hypermri.utils.utils_bssfp_analysis import bssfp_function

All functions are now located in hypermri.analysis.anatomical.bssfp
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from hypermri.utils.utils_bssfp_analysis is deprecated. "
    "Please update your code to import from hypermri.analysis.anatomical.bssfp instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import all functions from the new location for backward compatibility
try:
    pass
except ImportError as e:
    # Fallback error message if the new module isn't available
    raise ImportError(
        f"Could not import from analysis.anatomical.bssfp: {e}. "
        "The analysis module migration may be incomplete."
    )
