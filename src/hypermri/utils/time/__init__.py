"""
Time axis and temporal utilities for hypermri.

This module contains temporal calculations and time axis generation
extracted from the monolithic utils_general.py as part of Phase 2 refactoring.
"""

from .temporal import *

__all__ = [
    # Time axis calculations
    "get_dwell_time_s",
    "calc_sampling_time_axis",
    "calc_inversion_time_multi_slice",
    # Temporal processing
    "create_time_axis",
    "resample_time_series",
    "interpolate_temporal_data",
]
