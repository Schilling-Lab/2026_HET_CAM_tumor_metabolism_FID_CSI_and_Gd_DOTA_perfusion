"""
Mathematical utilities for hypermri.

This module contains mathematical functions and physics calculations
extracted from the monolithic utils_general.py as part of Phase 2 refactoring.

Note: Renamed from 'math' to 'mathematics' to avoid conflicts with Python's built-in math module.
"""

from .calculations import *

__all__ = [
    # Mathematical calculations
    "gaussian",
    "get_histogram",
    "compute_weighted_mean_and_std",
    "find_max_and_index",
    "moving_average",
    "polynomial_fit",
    "apply_gaussian_smoothing",
]
