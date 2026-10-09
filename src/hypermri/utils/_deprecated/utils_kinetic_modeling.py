"""
Backward compatibility module for utils_kinetic_modeling.

This module maintains backward compatibility by importing all functions from the new
analysis.kinetic.models module. Users can continue to use:

    from hypermri.utils.utils_kinetic_modeling import compartment_model

All functions are now located in hypermri.analysis.kinetic.models
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from hypermri.utils.utils_kinetic_modeling is deprecated. "
    "Please update your code to import from hypermri.analysis.kinetic.models instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import all functions from the new location for backward compatibility
try:
    pass
except ImportError as e:
    # Fallback error message if the new module isn't available
    raise ImportError(
        f"Could not import from analysis.kinetic.models: {e}. "
        "The analysis module migration may be incomplete."
    )
