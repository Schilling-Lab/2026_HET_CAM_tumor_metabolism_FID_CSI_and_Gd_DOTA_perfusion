"""
Coordinate system utilities for hypermri.

This package contains coordinate transformation and spatial grid functions
extracted from the monolithic utils_general.py as part of Phase 2 refactoring.

The coordinate system utilities are organized into:
- transformations: Scanner coordinate system transformations and matrix calculations
- grids: Spatial grid generation and management
"""

from .grids import (
    create_isotropic_grid,
    define_grid_lines,
    define_xyz_grid,
    get_rps_axes,
    get_rps_extent,
    get_rps_grid,
    get_xyz_axes,
    get_xyz_extent,
    get_xyz_extent_for_coordinate,
    get_xyz_grid,
    resample_grid,
)
from .image_shifts import (
    align_image_data,
    apply_image_alignment,
    calc_mat_origin_diff,
    determine_FOV_shift,
    estimate_image_alignment,
    get_orientation_flip_multipliers,
    plot_alignment_overview,
    shift_image,
)
from .transformations import (
    COORDINATE_PRECISION,
    convert_rps_to_xyz_grid,
    convert_rps_to_xyz_matrix_based,
    define_imagematrix_parameters,
    translate_coordinates,
)

__all__ = [
    # Transformations
    "define_imagematrix_parameters",
    "convert_rps_to_xyz_grid",
    "convert_rps_to_xyz_matrix_based",
    "translate_coordinates",
    "COORDINATE_PRECISION",
    # Grids
    "define_grid_lines",
    "define_xyz_grid",
    "get_rps_axes",
    "get_rps_grid",
    "get_xyz_grid",
    "get_xyz_axes",
    "get_rps_extent",
    "get_xyz_extent",
    "get_xyz_extent_for_coordinate",
    "create_isotropic_grid",
    "resample_grid",
    # Image shifts
    "align_image_data",
    "apply_image_alignment",
    "calc_mat_origin_diff",
    "shift_image",
    "determine_FOV_shift",
    "get_orientation_flip_multipliers",
    "estimate_image_alignment",
    "plot_alignment_overview",
]
