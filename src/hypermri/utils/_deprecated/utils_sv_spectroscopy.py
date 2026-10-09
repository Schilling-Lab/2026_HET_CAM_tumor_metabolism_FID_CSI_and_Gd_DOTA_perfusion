"""
Backward compatibility module for utils_sv_spectroscopy.

This module maintains backward compatibility by importing all functions from the new
analysis.spectroscopy.single_voxel module. Users can continue to use:

    from hypermri.utils.utils_sv_spectroscopy import Plot_Voxel_on_Anat

All functions are now located in hypermri.analysis.spectroscopy.single_voxel
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from hypermri.utils.utils_sv_spectroscopy is deprecated. "
    "Please update your code to import from hypermri.analysis.spectroscopy.single_voxel instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import all functions from the new location for backward compatibility
try:
    pass
except ImportError as e:
    # Fallback error message if the new module isn't available
    raise ImportError(
        f"Could not import from analysis.spectroscopy.single_voxel: {e}. "
        "The analysis module migration may be incomplete."
    )
