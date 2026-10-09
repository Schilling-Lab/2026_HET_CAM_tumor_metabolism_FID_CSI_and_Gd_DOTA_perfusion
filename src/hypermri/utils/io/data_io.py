"""
Shim for backward compatibility of data I/O utilities.

Deprecated: These functions have been moved to:
- hypermri.utils.io.files
- hypermri.utils.io.loaders
- hypermri.utils.io.converters
"""

import warnings

warnings.warn(
    "Importing from hypermri.utils.io.data_io is deprecated. "
    "Please update your code to import from: \n"
    " - hypermri.utils.io.loaders (load_phys_data, load_scan_timing)\n"
    " - hypermri.utils.io.converters (export_to_csv, export_to_matlab)\n"
    " - hypermri.utils.io.files (general file ops)",
    DeprecationWarning,
    stacklevel=2,
)

# Alias for backward compatibility if the names matched exactly
# In files.py I standardized on save_data/load_data, but data_io had save_data_pickle
# I've imported them as aliases above.
