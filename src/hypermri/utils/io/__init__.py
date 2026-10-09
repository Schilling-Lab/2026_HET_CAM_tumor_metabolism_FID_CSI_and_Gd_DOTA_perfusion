"""
Input/Output utilities for hypermri.

This module contains I/O functions extracted from the monolithic utils_general.py
as part of the Phase 2 refactoring to create focused utility modules.
"""

from .converters import (
    export_to_csv,
    export_to_dicom,
    export_to_matlab,
    export_xarray_to_netcdf,
    load_from_matlab,
    load_xarray_from_netcdf,
    save_dict_to_excel,
    save_markdown_as_word,
)
from .data_io import *
from .file_utils import *

__all__ = [
    # Data I/O
    "load_pc_sam_temp_file",
    "save_data_pickle",
    "load_data_pickle",
    "export_to_csv",
    "export_to_matlab",
    "export_to_dicom",
    "load_from_matlab",
    "export_xarray_to_netcdf",
    "load_xarray_from_netcdf",
    "save_dict_to_excel",
    "save_markdown_as_word",
    # File utilities
    "ensure_directory",
    "get_file_info",
    "find_files_by_pattern",
    "backup_file",
]
