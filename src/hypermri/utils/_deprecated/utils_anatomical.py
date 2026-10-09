"""
Backward compatibility module for utils_anatomical.

This module maintains backward compatibility by importing all functions from the new
analysis.anatomical.core module. Users can continue to use:

    from hypermri.utils.utils_anatomical import create_mask

All functions are now located in hypermri.analysis.anatomical.core
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from hypermri.utils.utils_anatomical is deprecated. "
    "Please update your code to import from hypermri.analysis.anatomical.core instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import all functions from the new location for backward compatibility
try:
    pass
except ImportError as e:
    # Fallback error message if the new module isn't available
    raise ImportError(
        f"Could not import from analysis.anatomical.core: {e}. "
        "The analysis module migration may be incomplete."
    )
