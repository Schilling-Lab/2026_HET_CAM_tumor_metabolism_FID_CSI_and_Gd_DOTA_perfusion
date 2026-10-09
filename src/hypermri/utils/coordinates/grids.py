"""
Spatial grid generation and management for MRI data.

This module handles creation and manipulation of coordinate grids:
- RPS (read, phase, slice) grids
- XYZ physical coordinate grids
- Grid line definitions and extents

Extracted from utils_general.py as part of Phase 2 refactoring.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from ..system.logging import init_default_logger
from .transformations import (
    convert_rps_to_xyz_grid,
    convert_rps_to_xyz_matrix_based,
    define_imagematrix_parameters,
    find_indexing_scanner_dim,
)

logger = init_default_logger(__name__)
COORDINATE_PRECISION = np.float16


def define_grid_lines(data_obj: Any = None, fov=None, mat=None, res=None, offset=0.0, slice_direction=False):
    """
    Define grid lines for a given FOV, matrix size, or resolution.

    Parameters
    ----------
    data_obj : MRIExperiment
        Data object containing FOV and matrix information
    fov : float, optional
        Field of view in mm
    mat : int, optional
        Matrix size in voxels
    res : float, optional
        Resolution in mm/voxel
    offset : float, optional
        Offset to apply to the grid lines in mm
    slice_direction: boolean, optional
        Flag indicating if the grid line is in slice direction

    Returns
    -------
    array_like
        Grid line positions, with applied offset and half-voxel correction (Bruker)

    Notes
    -----
    Apply negative offset.

    """
    if data_obj is not None:
        # Extract from data object if provided
        if hasattr(data_obj, "method"):
            if fov is None:
                fov = data_obj.get("fov_mm", [100])  # Default 100mm
            if mat is None:
                mat = data_obj.get("matrix", [64])  # Default 64x64

    if fov is not None and mat is not None:
        # Calculate resolution from FOV and matrix
        res = fov / mat
        # Calculate grid from FOV and matrix
        pixel_size = fov / mat
        half_fov = fov / 2.0
        grid_line = np.linspace(
            -half_fov + pixel_size / 2, half_fov - pixel_size / 2, mat
        )
    elif fov is not None and res is not None:
        # Calculate grid from FOV and resolution
        mat = int(fov / res)
        half_fov = fov / 2.0
        grid_line = np.linspace(-half_fov + res / 2, half_fov - res / 2, mat)
    elif mat is not None and res is not None:
        # Calculate grid from matrix and resolution
        fov = mat * res
        half_fov = fov / 2.0
        grid_line = np.linspace(-half_fov + res / 2, half_fov - res / 2, mat)
    else:
        raise ValueError("Need at least two of: fov, mat, res")

    # To make Bruker data overlay:
    if slice_direction:
        pass
    else:
        grid_line = grid_line - res / 2

    # Apply offset if provided (negative)
    grid_line = grid_line - offset

    return grid_line


# ------------------------------------------------------------
# ------------------ RPS Coordinate System -------------------
# ------------------------------------------------------------


def get_rps_axes(
    data_obj: Any = None,
    dtype=COORDINATE_PRECISION,
    with_offsets: bool = True,
    rps_grid_mm: tuple = None,
):
    """
    Get RPS (read, phase, slice) coordinate axes.

    Parameters
    ----------
    data_obj: Any
        Data object containing imaging parameters
    dtype  type, optional
        Data type for coordinate arrays

    Returns
    -------
    tuple
        (read_axis, phase_axis, slice_axis)

    """
    if data_obj is None:
        raise ValueError("Data object required for RPS axes")

    if rps_grid_mm is not None:
        get_rps_axes_from_grid(rps_grid_mm=rps_grid_mm)

    # Get FOV and matrix information
    try:
        fov = np.array(data_obj.fov_mm)
        if len(fov) == 2:
            # 2D acquisition - add slice dimension
            slice_thickness = data_obj.get("slice_thickness_mm", 1.0)
            n_slices = data_obj.get("n_slices", 1)
            fov = np.append(fov, slice_thickness * n_slices)

        matrix = np.array(data_obj.matrix)
        if len(matrix) == 2:
            # 2D acquisition - add slice dimension
            n_slices = data_obj.get("n_slices", 1)
            matrix = np.append(matrix, n_slices)

    except (KeyError, TypeError, AttributeError) as e:
        logger.warning(f"Could not extract FOV/matrix from data object: {e}")
        # Use defaults
        fov = np.array([100, 100, 10])
        matrix = np.array([64, 64, 1])

    read_offset = 0
    phase_offset = 0
    slice_offset = 0

    if with_offsets:
        # Add offset like in grid definition
        read_offset = data_obj.get("read_offset_mm", 0.0)
        phase_offset = data_obj.get("phase_offset_mm", 0.0)
        slice_offset = -data_obj.get("slice_offset_mm", 0.0)

    logger.debug("=== get_rps_axes: Generating RPS axes ===")
    logger.debug(f"  FOV: read={fov[0]:.2f}, phase={fov[1]:.2f}, slice={fov[2]:.2f} mm")
    logger.debug(f"  Matrix: read={matrix[0]}, phase={matrix[1]}, slice={matrix[2]}")
    logger.debug(
        f"  Offsets: read={read_offset:.2f}, phase={phase_offset:.2f}, slice={slice_offset:.2f} mm"
    )

    # Create axes
    read_axis = define_grid_lines(fov=fov[0], mat=matrix[0], offset=read_offset).astype(
        dtype
    )
    phase_axis = define_grid_lines(
        fov=fov[1], mat=matrix[1], offset=phase_offset
    ).astype(dtype)
    slice_axis = define_grid_lines(
        fov=fov[2], mat=matrix[2], offset=slice_offset, slice_direction=True,
    ).astype(dtype)

    logger.debug(
        f"  RPS axis ranges: read=[{read_axis.min():.2f}, {read_axis.max():.2f}] mm ({len(read_axis)} points)"
    )
    logger.debug(
        f"                   phase=[{phase_axis.min():.2f}, {phase_axis.max():.2f}] mm ({len(phase_axis)} points)"
    )
    logger.debug(
        f"                   slice=[{slice_axis.min():.2f}, {slice_axis.max():.2f}] mm ({len(slice_axis)} points)"
    )

    return read_axis, phase_axis, slice_axis


def get_rps_axes_from_grid(rps_grid_mm=None):
    """
    Get RPS axes from the passed grid.

    Parameters
    ----------
    rps_grid_mm: tuple
        The read, phase slice grid in milimeter. Can be used to determine the extent.

    Returns
    -------
    read_axis, phase_axis, slice_axis

    """

    # Extract axes from the grid tuple
    read_axis = rps_grid_mm[0][:, 0, 0]
    phase_axis = rps_grid_mm[1][0, :, 0]
    slice_axis = rps_grid_mm[2][0, 0, :]

    return read_axis, phase_axis, slice_axis


def get_rps_extent_from_grid(data_obj: Any = None, rps_grid_mm=None):
    """
    Get RPS coordinate extents.

    Parameters
    ----------
    data_obj : MRIExperiment
        Data object containing imaging parameters.
    rps_grid_mm: tuple
        The read, phase slice grid in milimeter. Can be used to determine the extent.

    Returns
    -------
    rps_extent: list
        (min_read_extent, max_read_extent, min_phase_extent, max_phase_extent, min_slice_extent, max_slice_extent)

    """
    if rps_grid_mm is None:
        return None

    read_axis_mm, phase_axis_mm, slice_axis_mm = get_rps_axes_from_grid(
        rps_grid_mm=rps_grid_mm
    )

    # Resolutions in all dimensions:
    read_resolution_mm = None
    phase_resolution_mm = None
    slice_resolution_mm = None

    # Imagign matrix. If any of these is 1, resolution cant be calcilated from the grid
    if data_obj is not None:
        matrix = data_obj.get("matrix", None)
        fov_mm = data_obj.get("fov_mm", None)
    else:
        raise Exception("Pass data_obj.")

    for i, e in enumerate(matrix):
        if e == 1:
            if i == 0:
                read_resolution_mm = fov_mm[0]
            if i == 1:
                phase_resolution_mm = fov_mm[1]
            if i == 2:
                slice_resolution_mm = fov_mm[2]

    try:
        read_resolution_mm = read_axis_mm[1] - read_axis_mm[0]
    except:
        pass
    if read_resolution_mm is None:
        raise Exception("Read resolution could not be determined.")

    try:
        phase_resolution_mm = phase_axis_mm[1] - phase_axis_mm[0]
    except:
        pass
    if phase_resolution_mm is None:
        raise Exception("Phase resolution could not be determined.")

    try:
        slice_resolution_mm = slice_axis_mm[1] - slice_axis_mm[0]
    except:
        pass
    if slice_resolution_mm is None:
        raise Exception("Slice resolution could not be determined.")

    rps_extent_mm = [
        read_axis_mm[0] - read_resolution_mm / 2,
        read_axis_mm[-1] + read_resolution_mm / 2,
        phase_axis_mm[0] - phase_resolution_mm / 2,
        phase_axis_mm[-1] + phase_resolution_mm / 2,
        slice_axis_mm[0] - slice_resolution_mm / 2,
        slice_axis_mm[-1] + slice_resolution_mm / 2,
    ]
    return rps_extent_mm


def get_rps_extent(
    data_obj: Any = None,
    dtype=COORDINATE_PRECISION,
    old: bool = False,
    with_offsets: bool = True,
    with_correction: bool = True,
    rps_grid_mm: tuple = None,
):
    """
    Get RPS coordinate extents for plotting.

    Parameters
    ----------
    data_obj : MRIExperiment
        Data object containing imaging parameters.
    dtype : type, optional
        Data type for coordinate arrays.
    with_offsets: bool, optional
        Consider the read, phase and slice offset in the extent calculation.
    with_correction: bool, optional
        Correct for the half voxel offset in read, phase, slice. Under revision.
    old: bool
        Use the old extent calculation method. Keeping for legacy reasons.
    rps_grid_mm: tuple
        The read, phase slice grid in milimeter. Can be used to determine the extent.

    Returns
    -------
    rps_extent: list
        (min_read_extent, max_read_extent, min_phase_extent, max_phase_extent, min_slice_extent, max_slice_extent)

    Notes
    -----
    Apply negative offset

    """
    if old:
        read_axis, phase_axis, slice_axis = get_rps_axes(
            data_obj, dtype, with_offsets=with_offsets
        )

        read_extent = [read_axis.min(), read_axis.max()]
        phase_extent = [phase_axis.min(), phase_axis.max()]
        slice_extent = [slice_axis.min(), slice_axis.max()]
    else:
        if rps_grid_mm is not None:
            get_rps_extent_from_grid(rps_grid_mm=rps_grid_mm, data_obj=data_obj)

        read_fov = data_obj.get("fov_mm")[0]
        phase_fov = data_obj.get("fov_mm")[1]
        slice_fov = data_obj.get("fov_mm")[2]

        if with_offsets:
            read_offset = data_obj.get("read_offset_mm", 0.0)
            phase_offset = data_obj.get("phase_offset_mm", 0.0)
            slice_offset = -data_obj.get("slice_offset_mm", 0.0)
        else:
            read_offset = 0.0
            phase_offset = 0.0
            slice_offset = 0.0

        # Add offset (negative)
        read_extent = [-read_fov / 2 - read_offset, read_fov / 2 - read_offset]
        phase_extent = [-phase_fov / 2 - phase_offset, phase_fov / 2 - phase_offset]
        slice_extent = [-slice_fov / 2 - slice_offset, slice_fov / 2 - slice_offset]

        # Fix Bruker half voxel offset
        if with_correction:
            # As the grid points are center of voxel, we need to correct the extent by half voxel
            try:
                resolution_mm = data_obj.get("resolution_mm")
                
                # Subtract half voxel from min extent and add half voxel to max extent
                read_extent[0] = read_extent[0] - resolution_mm[0] / 2
                phase_extent[0] = phase_extent[0] - resolution_mm[1] / 2
                slice_extent[0] = slice_extent[0] # + resolution_mm[2] / 2

                read_extent[1] = read_extent[1] - resolution_mm[0] / 2
                phase_extent[1] = phase_extent[1] - resolution_mm[1] / 2
                slice_extent[1] = slice_extent[1] # + resolution_mm[2] / 2

            except:
                print(f"Couldnt correct for half voxel shift in read, phase, slice")

    # Flatten the extents into a single list
    return [*read_extent, *phase_extent, *slice_extent]


def select_rps_extent_by_axis(
    data_obj: Any = None,
    dtype=COORDINATE_PRECISION,
    with_offsets: bool = True,
    with_correction: bool = True,
    rps_grid_mm: tuple = None,
    axis: str = "slice_idx",
):
    """
    Reveice the extent of two out of three dimensions (read_idx, phase_idx, slice_idx) by passing a third dimension.
    Order is always (read_idx, phase_idx, slice_idx).

    Parameters
    ----------
    data_obj: Any
        Data object containing imaging parameters
    dtype: type, optional
        Data type for coordinate arrays
    with_offset: bool
        Consider the read, phase and slice offset in the extent calculation.
    with_correction: bool
        Fix the Bruker half-voxel shift.
    rps_grid_mm: tuple
        Contains the read, phase and slice grid points. shape is [3, read, phase, slice].
    axis: str
        The name of the axis perpendicular to the extent. Can be read_idx, phase_idx, slice_idx.

    Returns
    -------
    extent: list
        List containing the 4 extent points.
    """

    rps_extent_mm = get_rps_extent(
        data_obj=data_obj,
        dtype=dtype,
        with_correction=with_correction,
        with_offsets=with_offsets,
        rps_grid_mm=rps_grid_mm,
    )

    valid_axis_names = ["read_idx", "phase_idx", "slice_idx"]
    if axis in valid_axis_names:
        if axis == "read_idx":
            return rps_extent_mm[2:]
        if axis == "phase_idx":
            return [rps_extent_mm[:2], rps_extent_mm[-2:]]
        if axis == "slice_idx":
            return rps_extent_mm[:-2]


def get_rps_grid(
    data_obj: Any = None,
    dtype=COORDINATE_PRECISION,
    with_offsets: bool = True,
    read_axis=None,
    phase_axis=None,
    slice_axis=None,
):
    """
    Generate RPS (read, phase, slice) coordinate grid.

    Parameters
    ----------
    data_obj : MRIExperiment
        Data object containing imaging parameters
    dtype : type, optional
        Data type for coordinate arrays
    with_offset: bool
        Consider the offset in read, phase and slice orientation

    Returns
    -------
    tuple
        (read_grid, phase_grid, slice_grid) as 3D meshgrids

    """
    if read_axis is None:
        read_axis, phase_axis, slice_axis = get_rps_axes(
            data_obj, dtype, with_offsets=with_offsets
        )

    # Create 3D meshgrids
    read_grid, phase_grid, slice_grid = np.meshgrid(
        read_axis, phase_axis, slice_axis, indexing="ij"
    )

    return (read_grid.astype(dtype), phase_grid.astype(dtype), slice_grid.astype(dtype))


# ------------------------------------------------------------
# ------------------ XYZ Coordinate System -------------------
# ------------------------------------------------------------


def define_xyz_grid(data_obj: Any = None, fov=None, mat=None):
    """
    Define a 3D grid for imaging data.

    Parameters
    ----------
    data_obj : MRIExperiment, optional
        Data object containing imaging parameters
    fov : array_like, optional
        Field of view [fov_x, fov_y, fov_z] in mm
    mat : array_like, optional
        Matrix size [mat_x, mat_y, mat_z] in voxels

    Returns
    -------
    tuple
        (grid_x, grid_y, grid_z) coordinate grids

    """
    if data_obj is not None:
        if fov is None:
            fov = np.array(data_obj.get("fov_mm", [100, 100, 10]))
        if mat is None:
            mat = np.array(define_imagematrix_parameters(data_obj=data_obj))

    if fov is None or mat is None:
        raise ValueError("Need FOV and matrix size information")

    fov = np.asarray(fov)
    mat = np.asarray(mat)

    if len(fov) != 3 or len(mat) != 3:
        raise ValueError("FOV and matrix must be 3-element arrays")

    # Create grid lines for each dimension
    grid_x = define_grid_lines(fov=fov[0], mat=mat[0])
    grid_y = define_grid_lines(fov=fov[1], mat=mat[1])
    grid_z = define_grid_lines(fov=fov[2], mat=mat[2])

    return grid_x, grid_y, grid_z


def get_xyz_grid(data_obj=None, dtype=COORDINATE_PRECISION, with_offsets=True):
    """
    Generate XYZ physical coordinate grid.

    Parameters
    ----------
    data_obj : object
        Data object containing imaging parameters
    dtype : type, optional
        Data type for coordinate arrays
    with_offsets : bool, optional
        Whether to apply position offsets (default True)

    Returns
    -------
    tuple
        (x_grid, y_grid, z_grid) in physical coordinates

    """
    if data_obj is None:
        raise ValueError("Data object required for XYZ grid")

    # Log orientation parameters
    logger.debug("=== get_xyz_grid: Generating XYZ coordinates ===")
    logger.debug(f"  read_orientation: {data_obj.get('read_orientation', 'N/A')}")
    logger.debug(f"  slice_orientation: {data_obj.get('slice_orientation', 'N/A')}")
    logger.debug(f"  patient_pos: {data_obj.get('patient_pos', 'N/A')}")
    logger.debug(f"  with_offsets: {with_offsets}")
    if with_offsets:
        logger.debug(
            f"  Offsets: read={data_obj.get('read_offset_mm', 0):.2f}, "
            f"phase={data_obj.get('phase_offset_mm', 0):.2f}, "
            f"slice={data_obj.get('slice_offset_mm', 0):.2f} mm"
        )

    # Get RPS grid first
    rps_grid = get_rps_grid(data_obj, dtype, with_offsets=with_offsets)

    # Log RPS grid info
    logger.debug(
        f"  RPS grid shapes: read={rps_grid[0].shape}, phase={rps_grid[1].shape}, slice={rps_grid[2].shape}"
    )
    logger.debug(
        f"  RPS ranges: read=[{rps_grid[0].min():.2f}, {rps_grid[0].max():.2f}] mm"
    )
    logger.debug(
        f"              phase=[{rps_grid[1].min():.2f}, {rps_grid[1].max():.2f}] mm"
    )
    logger.debug(
        f"              slice=[{rps_grid[2].min():.2f}, {rps_grid[2].max():.2f}] mm"
    )

    # Try to use matrix-based transformation if ACQ_grad_matrix is available
    acquisition_matrix = data_obj.get("acquisition_matrix", None)

    if acquisition_matrix is not None:
        logger.debug("  Using matrix-based transformation (ACQ_grad_matrix available)")
        try:
            x_grid, y_grid, z_grid = convert_rps_to_xyz_matrix_based(
                rps_grid, data_obj=data_obj, dtype=dtype
            )
        except Exception as e:
            logger.warning(f"  Matrix-based transformation failed: {e}")
            logger.warning("  Falling back to orientation-based transformation")
            x_grid, y_grid, z_grid = convert_rps_to_xyz_grid(
                rps_grid, data_obj=data_obj, dtype=dtype
            )
    else:
        logger.debug(
            "  Using orientation-based transformation (ACQ_grad_matrix not available)"
        )
        x_grid, y_grid, z_grid = convert_rps_to_xyz_grid(
            rps_grid, data_obj=data_obj, dtype=dtype
        )

    logger.debug(
        f"  XYZ grid shapes: x={x_grid.shape}, y={y_grid.shape}, z={z_grid.shape}"
    )
    logger.debug(f"  XYZ ranges: x=[{x_grid.min():.2f}, {x_grid.max():.2f}] mm")
    logger.debug(f"              y=[{y_grid.min():.2f}, {y_grid.max():.2f}] mm")
    logger.debug(f"              z=[{z_grid.min():.2f}, {z_grid.max():.2f}] mm")
    logger.debug("=== get_xyz_grid: Complete ===")

    return x_grid, y_grid, z_grid


def get_xyz_grid_old(data_obj=None, dtype=COORDINATE_PRECISION, db=False):
    """
    Function to get the grid points of a 3D data array.

    Parameters
    ----------
    data_obj: hypermri object
    dtype: precision of coordinates
    db: bool
        debug flag, optional

    Returns
    -------
    x_grid_mm, y_grid_mm, z_grid_mm: grid points in mm

    Notes
    -----
    The function first calls get_rps_grid to get the read, phase, and slice grid points.

    """
    read_grid_mm, phase_grid_mm, slice_grid_mm = get_rps_grid(data_obj=data_obj)
    # read_grid_mm, phase_grid_mm, slice_grid_mm = self.get_rps_grid # -->
    # Calls get_rps_grid

    # Check for None values in the grid variables
    if read_grid_mm is None or phase_grid_mm is None or slice_grid_mm is None:
        if db:
            print("One or more grid variables are None. Returning None.")
        return None, None, None

    rps_grid_mm = np.stack((read_grid_mm, phase_grid_mm, slice_grid_mm), axis=-1)
    if rps_grid_mm.size == 0 or read_grid_mm.size == 0:  # Corrected condition
        if db:
            print(f"rps_grid_mm.size={rps_grid_mm.size}\nrps_grid_mm={rps_grid_mm}")
            print(f"read_grid_mm.size={read_grid_mm.size}\nread_grid_mm={read_grid_mm}")
        return None, None, None
    rot_matrix = data_obj.acquisition_matrix
    position_matrix = data_obj.position_matrix
    logger.debug(f"rps_grid_mm.shape: {rps_grid_mm.shape}")
    xyz_grid_mm = rps_grid_mm @ rot_matrix @ position_matrix
    x_grid_mm = xyz_grid_mm[..., 0].astype(dtype)
    y_grid_mm = xyz_grid_mm[..., 1].astype(dtype)
    z_grid_mm = xyz_grid_mm[..., 2].astype(dtype)

    return x_grid_mm, y_grid_mm, z_grid_mm


def get_xyz_axes(data_obj=None, dtype=COORDINATE_PRECISION):
    """
    Get XYZ coordinate axes (1D arrays).

    Warning
    -------
    This function has limitations for tilted/oblique acquisitions where coordinates
    are interdependent (each voxel has unique x, y, z values). In such cases, the
    returned "axes" will contain all unique coordinate values across the entire volume,
    which may not be meaningful for visualization or analysis.

    For tilted acquisitions, consider using:
    - `get_xyz_grid()` to get the full 3D coordinate grids
    - `get_xyz_extent()` to get the bounding box extents

    This function works correctly for axis-aligned acquisitions where coordinates
    are independent (x varies only along one dimension, y along another, etc.).

    Parameters
    ----------
    data_obj : object
        Data object containing imaging parameters
    dtype : type, optional
        Data type for coordinate arrays

    Returns
    -------
    tuple
        (x_axis, y_axis, z_axis) as 1D arrays
        For tilted acquisitions, these contain all unique values (may be large)
    """
    # Get grid first
    x_grid, y_grid, z_grid = get_xyz_grid(data_obj, dtype, with_offsets=True)

    # Check if this is a matrix-based transformation (tilted acquisition)
    acquisition_matrix = data_obj.get("acquisition_matrix", None)
    if acquisition_matrix is not None:
        # Check if the acquisition matrix is NOT identity (i.e., tilted)
        is_identity = np.allclose(acquisition_matrix, np.eye(3), atol=1e-6)
        if not is_identity:
            logger.warning(
                "get_xyz_axes() called on tilted/oblique acquisition. "
                "The returned axes contain all unique coordinate values and may not represent "
                "independent axes. Consider using get_xyz_grid() or get_xyz_extent() instead."
            )

    # Extract unique values for each axis
    x_axis = np.unique(x_grid.ravel())
    y_axis = np.unique(y_grid.ravel())
    z_axis = np.unique(z_grid.ravel())

    return x_axis, y_axis, z_axis


def get_xyz_extent(data_obj=None, dtype=COORDINATE_PRECISION):
    """
    Get XYZ coordinate extents for plotting including voxel dimensions.

    This function computes the physical extent of the volume by transforming the
    corner points of the RPS bounding box to XYZ coordinates. This correctly
    accounts for voxel dimensions in both axis-aligned and tilted acquisitions.

    Parameters
    ----------
    data_obj : object
        Data object containing imaging parameters
    dtype : type, optional
        Data type for coordinate arrays

    Returns
    -------
    tuple
        (x_extent, y_extent, z_extent) as [min, max] pairs
        These represent the physical boundaries of the volume including voxel dimensions

    """
    # Get FOV and offsets in RPS coordinates
    fov = np.array(data_obj.get("fov_mm", [100, 100, 10]))

    read_offset = data_obj.get("read_offset_mm", 0.0)
    phase_offset = data_obj.get("phase_offset_mm", 0.0)
    slice_offset = data_obj.get("slice_offset_mm", 0.0)

    # Define the 8 corners of the RPS bounding box (including voxel dimensions)
    # These go from -FOV/2 to +FOV/2 (edge to edge, not center to center)
    read_min = -fov[0] / 2 + read_offset
    read_max = fov[0] / 2 + read_offset
    phase_min = -fov[1] / 2 + phase_offset
    phase_max = fov[1] / 2 + phase_offset
    slice_min = -fov[2] / 2 + slice_offset
    slice_max = fov[2] / 2 + slice_offset

    # Create 8 corner points in RPS space
    corners_rps = np.array(
        [
            [read_min, phase_min, slice_min],
            [read_max, phase_min, slice_min],
            [read_min, phase_max, slice_min],
            [read_max, phase_max, slice_min],
            [read_min, phase_min, slice_max],
            [read_max, phase_min, slice_max],
            [read_min, phase_max, slice_max],
            [read_max, phase_max, slice_max],
        ]
    )

    # Get transformation matrices
    acquisition_matrix = data_obj.get("acquisition_matrix", None)
    position_matrix = data_obj.get("position_matrix", None)

    if acquisition_matrix is None:
        acquisition_matrix = np.eye(3)
        logger.debug("get_xyz_extent: No acquisition_matrix, using identity")

    if position_matrix is None:
        position_matrix = np.eye(3)
        logger.debug("get_xyz_extent: No position_matrix, using identity")

    # Transform corners to XYZ: (x,y,z) = (r,p,s) @ ACQ_grad_matrix @ Position_Matrix
    combined_matrix = acquisition_matrix @ position_matrix
    corners_xyz = corners_rps @ combined_matrix

    # Find min/max across all corners
    x_extent = [float(corners_xyz[:, 0].min()), float(corners_xyz[:, 0].max())]
    y_extent = [float(corners_xyz[:, 1].min()), float(corners_xyz[:, 1].max())]
    z_extent = [float(corners_xyz[:, 2].min()), float(corners_xyz[:, 2].max())]

    logger.debug(
        f"get_xyz_extent: Computed extent from {len(corners_rps)} corner points"
    )
    logger.debug(f"  x_extent: {x_extent}")
    logger.debug(f"  y_extent: {y_extent}")
    logger.debug(f"  z_extent: {z_extent}")

    return x_extent, y_extent, z_extent


def _astype_float64_if_numeric(coord_array):
    """
    Return a float64 view of numeric coordinate data to avoid float16 overflow.

    Parameters
    ----------
    coord_array : xarray.DataArray or np.ndarray
        Coordinate array that may need promotion to float64 for reductions.

    Returns
    -------
    Same type as input
        ``coord_array`` converted to float64 when numeric, otherwise untouched.

    """
    try:
        dtype = coord_array.dtype
    except AttributeError:
        return coord_array

    try:
        if np.issubdtype(dtype, np.number):
            return coord_array.astype(np.float64, copy=False)
    except TypeError:
        return coord_array

    return coord_array


def _reduce_axis_to_object_dimension(axis_coord, object_dim):
    """
    Reduce a coordinate grid to a 1D array along the requested object dimension.

    Parameters
    ----------
    axis_coord : xarray.DataArray
        Coordinate grid (e.g., x, y, z) defined on the imaging volume.
    object_dim : str
        Name of the object dimension (read_idx, phase_idx, or slice_idx).

    Returns
    -------
    np.ndarray or None
        1D array of coordinate values along ``object_dim`` or ``None`` if the
        coordinate does not depend on ``object_dim``.

    """
    axis_coord = _astype_float64_if_numeric(axis_coord)
    if object_dim not in axis_coord.dims:
        return None

    # Average across all other dimensions so only the requested dimension remains.
    reduce_dims = [dim for dim in axis_coord.dims if dim != object_dim]
    reduced = axis_coord
    if reduce_dims:
        reduced = reduced.mean(dim=reduce_dims)

    # In case averaging still leaves extraneous singleton dimensions, squeeze them.
    reduced = reduced.squeeze()

    if object_dim not in reduced.dims:
        return None

    return np.asarray(reduced, dtype=np.float64)


def _compute_spacing_from_values(values, eps=1e-9):
    """
    Estimate the smallest non-zero spacing present in the array of values.

    Parameters
    ----------
    values : np.ndarray
        Array with coordinate values (flattened or otherwise).
    eps : float
        Tolerance below which differences are treated as zero.

    Returns
    -------
    float or None
        Estimated spacing or ``None`` if it cannot be determined.

    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size < 2:
        return None

    sorted_vals = np.unique(finite)
    if sorted_vals.size < 2:
        return None

    diffs = np.diff(sorted_vals)
    valid_diffs = np.abs(diffs[~np.isclose(diffs, 0.0, atol=eps)])

    if valid_diffs.size == 0:
        return None

    return float(valid_diffs.min())


def _compute_extent_with_padding(values, object_line, object_index, eps=1e-9):
    """
    Compute min/max extent for a coordinate plane with optional padding.

    Parameters
    ----------
    values : np.ndarray
        Coordinate values for the selected plane.
    object_line : np.ndarray or None
        1D coordinate values along the object dimension to fall back on when
        the plane itself does not provide spacing information (e.g., constant
        slice).
    object_index : int
        Index along the object dimension.
    eps : float
        Tolerance for detecting zero spacing.

    Returns
    -------
    list
        ``[min_val, max_val]`` describing the extent along the coordinate axis.

    """
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return [np.nan, np.nan]

    min_val = float(finite.min())
    max_val = float(finite.max())

    spacing = _compute_spacing_from_values(finite, eps=eps)

    lower_pad = upper_pad = None
    if spacing is not None and spacing > eps:
        half_spacing = spacing / 2.0
        lower_pad = half_spacing
        upper_pad = half_spacing
    elif object_line is not None and object_line.size > object_index:
        center_val = float(object_line[object_index])
        prev_val = float(object_line[object_index - 1]) if object_index > 0 else None
        next_val = (
            float(object_line[object_index + 1])
            if object_index < object_line.size - 1
            else None
        )

        if prev_val is not None:
            lower_pad = abs(center_val - prev_val) / 2.0
        if next_val is not None:
            upper_pad = abs(next_val - center_val) / 2.0

        if lower_pad is None and upper_pad is not None:
            lower_pad = upper_pad
        if upper_pad is None and lower_pad is not None:
            upper_pad = lower_pad

        # When the plane is effectively constant, use the center value
        # and pads derived from neighbouring slices to create thickness.
        if spacing is None or spacing <= eps:
            min_val = center_val
            max_val = center_val

    if lower_pad is not None:
        min_val -= lower_pad
    if upper_pad is not None:
        max_val += upper_pad

    return [float(min_val), float(max_val)]


def get_xyz_extent_for_coordinate(
    input_data,
    object_coord_name,
    object_index,
    scanner_axes=("x", "y", "z"),
    plane_axes=None,
):
    """
    Determine the physical XYZ extents for a specific plane in an xarray object.

    Parameters
    ----------
    input_data : xarray.DataArray or xarray.Dataset
        Data structure containing spatial coordinates (``x``, ``y``, ``z``) and
        object coordinates (``read_idx``, ``phase_idx``, ``slice_idx``).
    object_coord_name : str
        Name of the object coordinate to index along (``read_idx``,
        ``phase_idx``, or ``slice_idx``).
    object_index : int
        Zero-based index along ``object_coord_name`` identifying the plane for
        which the extents should be calculated.
    scanner_axes : tuple, optional
        Coordinate names that define the scanner space axes. Defaults to
        ``("x", "y", "z")``.
    plane_axes : sequence of str, optional
        Explicit pair of scanner axes that span the plane to be visualised (for
        example, ``("x", "y")``). When not provided, the function derives the
        in-plane axes by excluding the dominant axis associated with the
        selected object coordinate. If the dominant axis cannot be determined,
        the first two axes from ``scanner_axes`` are used as a fallback.

    Returns
    -------
    dict
        Dictionary containing:
            - ``coordinate``: selected object coordinate name
            - ``index``: integer index that was used
            - ``coordinate_value``: mean physical value of the selected plane
            - ``extents``: mapping of each scanner axis to ``[min, max]``
            - ``plane_mapping``: best-fit scanner axis associated with the
              object coordinate (if determinable), provided as a dict with
              keys ``axis``, ``axis_index``, and ``axis_value``
            - ``plane_axes``: mapping of ``horizontal`` / ``vertical`` axes used
              for imaging representations (``None`` if unavailable)
            - ``imshow_extent``: extent list ``[xmin, xmax, ymin, ymax]`` suitable
              for ``matplotlib.pyplot.imshow`` (``None`` if unavailable)
            - ``imshow_origin``: suggested ``origin`` argument for ``imshow``.

    Raises
    ------
    TypeError
        If the input object does not expose xarray-like coordinates.
    ValueError
        If required coordinates are missing.
    IndexError
        If ``object_index`` is out of bounds for ``object_coord_name``.

    """
    if not hasattr(input_data, "coords"):
        raise TypeError("input_data must be an xarray Dataset or DataArray")

    if not isinstance(object_index, (int, np.integer)):
        raise TypeError("object_index must be an integer index")

    if object_coord_name not in getattr(input_data, "sizes", {}):
        raise ValueError(
            f"Coordinate '{object_coord_name}' not present in the input data dimensions"
        )

    coord_size = int(input_data.sizes[object_coord_name])
    if object_index < 0 or object_index >= coord_size:
        raise IndexError(
            f"Index {object_index} is out of bounds for '{object_coord_name}' "
            f"with size {coord_size}"
        )

    for axis in scanner_axes:
        if axis not in input_data.coords:
            raise ValueError(f"Scanner axis '{axis}' not found in input_data.coords")

    selector = {object_coord_name: int(object_index)}

    # Determine mean coordinate value for the selected plane
    coordinate_value = np.nan
    if object_coord_name in input_data.coords:
        coord_array = _astype_float64_if_numeric(input_data.coords[object_coord_name])
        try:
            plane_coord = coord_array.isel(**selector)
        except ValueError:
            plane_coord = coord_array
            for dim, idx in selector.items():
                if dim in plane_coord.dims:
                    plane_coord = plane_coord.isel({dim: idx})
        coordinate_value = float(
            np.nanmean(np.asarray(plane_coord.values, dtype=np.float64))
        )

    # Pre-compute 1D lines along the object dimension for each scanner axis.
    axis_lines = {}
    for axis in scanner_axes:
        axis_lines[axis] = _reduce_axis_to_object_dimension(
            _astype_float64_if_numeric(input_data.coords[axis]), object_coord_name
        )

    extents = {}
    for axis in scanner_axes:
        axis_coord = _astype_float64_if_numeric(input_data.coords[axis])
        try:
            plane_values = axis_coord.isel(**selector)
        except ValueError:
            plane_values = axis_coord
            for dim, idx in selector.items():
                if dim in plane_values.dims:
                    plane_values = plane_values.isel({dim: idx})

        extents[axis] = _compute_extent_with_padding(
            plane_values.values, axis_lines[axis], int(object_index)
        )

    plane_mapping = None
    try:
        mapping_result = find_indexing_scanner_dim(
            input_data=input_data,
            object_coord_name=object_coord_name,
            object_idx=int(object_index),
        )
        if mapping_result:
            plane_mapping = {
                "axis": mapping_result[0],
                "axis_index": int(mapping_result[1]),
                "axis_value": float(mapping_result[2]),
            }
    except Exception as exc:  # pragma: no cover - diagnostic logging only
        logger.debug(
            "Failed to determine dominant scanner axis for %s[%d]: %s",
            object_coord_name,
            object_index,
            exc,
        )

    # Determine plane axes for plotting/imshow representations
    def _validate_plane_axes(candidate):
        if candidate is None:
            return None
        if len(candidate) != 2:
            raise ValueError("plane_axes must contain exactly two axis names")
        cleaned = []
        for ax in candidate:
            if ax not in scanner_axes:
                raise ValueError(
                    f"Plane axis '{ax}' not recognised. Available axes: {scanner_axes}"
                )
            cleaned.append(ax)
        if cleaned[0] == cleaned[1]:
            raise ValueError("plane_axes entries must be distinct axis names")
        return cleaned

    resolved_plane_axes = _validate_plane_axes(plane_axes)

    if resolved_plane_axes is None:
        normal_axis = plane_mapping["axis"] if plane_mapping else None
        preferred_order = ["x", "y", "z"]
        candidate_axes = []

        for axis in preferred_order:
            if axis in scanner_axes and axis != normal_axis:
                candidate_axes.append(axis)

        for axis in scanner_axes:
            if axis != normal_axis and axis not in candidate_axes:
                candidate_axes.append(axis)

        if len(candidate_axes) >= 2:
            resolved_plane_axes = candidate_axes[:2]

    if resolved_plane_axes is None:
        resolved_plane_axes = list(scanner_axes)[:2]

    imshow_extent = None
    plane_axes_info = None

    if resolved_plane_axes and len(resolved_plane_axes) == 2:
        horiz_axis, vert_axis = resolved_plane_axes
        horiz_extent = extents.get(horiz_axis)
        vert_extent = extents.get(vert_axis)

        if horiz_extent and vert_extent:
            imshow_extent = [
                float(horiz_extent[0]),
                float(horiz_extent[1]),
                float(vert_extent[0]),
                float(vert_extent[1]),
            ]
            plane_axes_info = {
                "horizontal": horiz_axis,
                "vertical": vert_axis,
                "normal": plane_mapping["axis"] if plane_mapping else None,
            }

    imshow_origin = "lower" if imshow_extent is not None else None

    return {
        "coordinate": object_coord_name,
        "index": int(object_index),
        "coordinate_value": coordinate_value,
        "extents": extents,
        "plane_mapping": plane_mapping,
        "plane_axes": plane_axes_info,
        "imshow_extent": imshow_extent,
        "imshow_origin": imshow_origin,
    }


def get_imshow_extent_from_coords(
    data_array,
    y_dim: str = "read_idx",
    x_dim: str = "phase_idx",
) -> List[float]:
    """
    Construct an ``imshow`` extent directly from xarray coordinates.

    Parameters
    ----------
    data_array : xr.DataArray
        Array containing coordinate grids (e.g., ``read_idx``/``phase_idx`` or
        ``x``/``y``) as xarray coordinates.
    y_dim : str, default ``\"read_idx\"``
        Coordinate name to map to the vertical axis of the plot.
    x_dim : str, default ``\"phase_idx\"``
        Coordinate name to map to the horizontal axis of the plot.

    Returns
    -------
    list
        Extent in the form ``[xmin, xmax, ymin, ymax]`` suitable for
        ``matplotlib.pyplot.imshow``.

    Raises
    ------
    TypeError
        If ``data_array`` is not xarray-like.
    ValueError
        If the requested coordinate names are missing.

    """
    if not hasattr(data_array, "coords"):
        raise TypeError("data_array must be an xarray DataArray with coordinates")

    if x_dim not in data_array.coords or y_dim not in data_array.coords:
        raise ValueError(
            f"Coordinates '{y_dim}' and/or '{x_dim}' not present in DataArray"
        )

    x_vals = np.asarray(data_array.coords[x_dim].values, dtype=np.float64)
    y_vals = np.asarray(data_array.coords[y_dim].values, dtype=np.float64)

    return [
        float(np.nanmin(x_vals)),
        float(np.nanmax(x_vals)),
        float(np.nanmin(y_vals)),
        float(np.nanmax(y_vals)),
    ]


def create_isotropic_grid(center, fov, resolution):
    """
    Create an isotropic grid centered at given position.

    Parameters
    ----------
    center : array_like
        Center position [x, y, z] in mm
    fov : float
        Field of view in mm (same for all dimensions)
    resolution : float
        Voxel resolution in mm

    Returns
    -------
    tuple
        (x_grid, y_grid, z_grid) coordinate grids

    """
    center = np.asarray(center)
    if len(center) != 3:
        raise ValueError("Center must be 3-element array")

    # Calculate number of voxels
    n_vox = int(fov / resolution)
    half_fov = fov / 2.0

    # Create 1D axes
    axis = np.linspace(-half_fov, half_fov, n_vox)

    # Create 3D grids centered at specified position
    x_axis = axis + center[0]
    y_axis = axis + center[1]
    z_axis = axis + center[2]

    x_grid, y_grid, z_grid = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")

    return x_grid, y_grid, z_grid


def resample_grid(original_grid, original_axes, new_axes):
    """
    Resample data from one grid to another using interpolation.

    Parameters
    ----------
    original_grid : array_like
        Original data on the source grid
    original_axes : tuple
        (x_axis, y_axis, z_axis) of original grid
    new_axes : tuple
        (x_axis, y_axis, z_axis) of target grid

    Returns
    -------
    array_like
        Resampled data on new grid

    """
    from scipy.interpolate import RegularGridInterpolator

    # Create interpolator
    interpolator = RegularGridInterpolator(
        original_axes, original_grid, method="linear", bounds_error=False, fill_value=0
    )

    # Create target grid points
    new_x, new_y, new_z = np.meshgrid(*new_axes, indexing="ij")
    target_points = np.stack([new_x.ravel(), new_y.ravel(), new_z.ravel()], axis=1)

    # Interpolate
    resampled_flat = interpolator(target_points)
    resampled_grid = resampled_flat.reshape(new_x.shape)

    return resampled_grid


def get_scanner_coords_extent(
    input_data, scanner_coord_name, object_coord_name="slice_idx", object_idx=0
):
    """

    Parameters
    ----------
    input_data : xarray DataArray
        Input data containing scanner coordinates
    scanner_coord_name : str
        Name of the scanner coordinate (e.g., 'x', 'y', 'z')
    object_coord_name : str, optional
        Name of the object coordinate (default is 'slice_idx')
    object_idx : int, optional
        Index of the object coordinate to analyze (default is 0)

    Returns
    -------
    tuple
        (min_x, max_x, min_y, max_y) coordinates for the specified object index, the outermost
        coordinates in the scanner coordinate system at the specified object index.

    """
    return (
        input_data.coords[scanner_coord_name]
        .isel(**{object_coord_name: object_idx})
        .values[0, 0],
        input_data.coords[scanner_coord_name]
        .isel(**{object_coord_name: object_idx})
        .values[-1, 0],
        input_data.coords[scanner_coord_name]
        .isel(**{object_coord_name: object_idx})
        .values[0, -1],
        input_data.coords[scanner_coord_name]
        .isel(**{object_coord_name: object_idx})
        .values[-1, -1],
    )
