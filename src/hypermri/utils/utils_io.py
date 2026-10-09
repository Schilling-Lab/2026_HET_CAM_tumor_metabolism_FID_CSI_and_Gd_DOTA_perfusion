"""
Shim for backward compatibility of I/O utilities.

Deprecated: These functions have been moved to:
- hypermri.utils.io.files
- hypermri.utils.io.converters
"""

import warnings

warnings.warn(
    "Importing from hypermri.utils.utils_io is deprecated. "
    "Please update your code to import from: \n"
    " - hypermri.utils.io.files (save_data, load_data, create_folder_structure)\n"
    " - hypermri.utils.io.converters (save_markdown_as_word, save_dict_to_excel)",
    DeprecationWarning,
    stacklevel=2,
)
