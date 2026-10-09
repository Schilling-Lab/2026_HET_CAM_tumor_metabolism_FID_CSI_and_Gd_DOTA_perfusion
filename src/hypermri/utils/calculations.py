"""
Shim for backward compatibility of calculation utilities.

Deprecated: These functions have been moved to:
- hypermri.utils.mathematics.calculations
- hypermri.utils.physics
- hypermri.utils.spectroscopy.calibration
"""

import warnings

warnings.warn(
    "Importing from hypermri.utils.calculations is deprecated. "
    "Please update your code to import from: \n"
    " - hypermri.utils.mathematics.calculations (Pure math)\n"
    " - hypermri.utils.physics (MRI Physics)\n"
    " - hypermri.utils.spectroscopy.calibration (Spectroscopy-specific)",
    DeprecationWarning,
    stacklevel=2,
)
