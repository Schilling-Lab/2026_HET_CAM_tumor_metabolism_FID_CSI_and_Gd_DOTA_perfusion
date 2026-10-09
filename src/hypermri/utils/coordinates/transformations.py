"""
Coordinate system transformations for MRI scanner geometry.

This module handles coordinate transformations between different reference systems:
- Scanner coordinates (read, phase, slice - RPS)
- Physical coordinates (x, y, z - XYZ)
- Patient position and orientation handling

Extracted from utils_general.py as part of Phase 2 refactoring.
"""

import warnings
from typing import Optional, Tuple

import numpy as np

from ..system.logging import init_default_logger

logger = init_default_logger(__name__)
COORDINATE_PRECISION = np.float16


def define_imagematrix_parameters(
    data_obj=None,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Define the imaging matrix in voxels.

    Warning: Does not take into account the orientation and offsets of the
    object (yet).

    Parameters
    ----------
    data_obj : Sequence object
        The data object containing sequence information.

    Returns
    -------
    dim_z : int
        Number of voxels in the z dimension.
    dim_x : int
        Number of voxels in the x dimension.
    dim_y : int
        Number of voxels in the y dimension.

    Notes
    -----
    The function returns dimensions as dim_z, dim_x, dim_y, which correspond
    to z, x, y in the scanner coordinate system.
    """
    # Initialize empty return values
    dim_z = dim_x = dim_y = None

    if data_obj is None:
        return dim_z, dim_x, dim_y

    if data_obj is not None:
        read_orient = data_obj.read_orientation
        slice_orient = data_obj.slice_orientation
        patient_pos = data_obj.patient_pos

    if patient_pos == "Head_Prone":
        # All axial slice options:
        if slice_orient == "axial":
            if read_orient == "L_R":
                # x - dimension
                dim_x = data_obj.matrix[0]
                # y - dimension
                dim_y = data_obj.matrix[1]
                # z - dimension
                if data_obj.spatial_dimension == "3D":
                    dim_z = data_obj.matrix[2]
                else:
                    dim_z = data_obj.n_slices
            elif read_orient == "A_P":
                # x - dimension
                dim_x = data_obj.matrix[1]
                # y - dimension
                dim_y = data_obj.matrix[0]
                # z - dimension
                if data_obj.spatial_dimension == "3D":
                    dim_z = data_obj.matrix[2]
                else:
                    dim_z = data_obj.n_slices
            else:
                raise Exception(
                    f"define_imagematrix_parameters:\\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )

        # All sagittal slice options:
        elif slice_orient == "sagittal":
            if read_orient == "H_F":
                # y - dimension
                dim_y = data_obj.matrix[1]
                # z - dimension
                dim_z = data_obj.matrix[0]
                # x - dimension
                if data_obj.spatial_dimension == "3D":
                    dim_x = data_obj.matrix[2]
                else:
                    dim_x = data_obj.n_slices
            elif read_orient == "A_P":
                # y - dimension
                dim_y = data_obj.matrix[0]
                # z - dimension
                dim_z = data_obj.matrix[1]
                # x - dimension
                if data_obj.spatial_dimension == "3D":
                    dim_x = data_obj.matrix[2]
                else:
                    dim_x = data_obj.n_slices
            else:
                raise Exception(
                    f"define_imagematrix_parameters:\\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )

        # All coronal slice options:
        elif slice_orient == "coronal":
            if read_orient == "H_F":
                # z - dimension
                dim_z = data_obj.matrix[0]
                # x - dimension
                dim_x = data_obj.matrix[1]
                # y - dimension
                if data_obj.spatial_dimension == "3D":
                    dim_y = data_obj.matrix[2]
                else:
                    dim_y = data_obj.n_slices
            elif read_orient == "L_R":
                # z - dimension
                dim_z = data_obj.matrix[1]
                # x - dimension
                dim_x = data_obj.matrix[0]
                # y - dimension
                if data_obj.spatial_dimension == "3D":
                    dim_y = data_obj.matrix[2]
                else:
                    dim_y = data_obj.n_slices
            else:
                raise Exception(
                    f"define_imagematrix_parameters:\\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )

        elif slice_orient == "axial sagittal coronal":
            raise Exception(
                f"Slice orientation: {slice_orient} --> probably Localizer, not implemented yet!"
            )
        else:
            raise Exception("unknown slice orientation: " + slice_orient)

    elif patient_pos == "Head_Supine":
        warnings.warn("Head_Supine is not fully implemented yet!")
        # Implementation for Head_Supine would go here
        # For now, using same logic as Head_Prone
        if slice_orient == "axial":
            if read_orient == "L_R":
                dim_x = data_obj.matrix[0]
                dim_y = data_obj.matrix[1]
                if data_obj.spatial_dimension == "3D":
                    dim_z = data_obj.matrix[2]
                else:
                    dim_z = data_obj.n_slices
            elif read_orient == "A_P":
                dim_x = data_obj.matrix[1]
                dim_y = data_obj.matrix[0]
                if data_obj.spatial_dimension == "3D":
                    dim_z = data_obj.matrix[2]
                else:
                    dim_z = data_obj.n_slices
        # Add other orientations as needed
    else:
        raise Exception(f"Unknown patient position: {patient_pos}")

    return dim_z, dim_x, dim_y


def convert_rps_to_xyz_grid(
    rps_grid: Tuple[np.ndarray, np.ndarray, np.ndarray],
    data_obj=None,
    read_orient: str = None,
    slice_orient: str = None,
    patient_pos: str = None,
    dtype=COORDINATE_PRECISION,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RPS (read, phase, slice) coordinate grid to XYZ coordinate grid.

    Parameters
    ----------
    rps_grid : array_like
        RPS coordinate grid
    data_obj : object, optional
        Data object containing orientation information
    read_orient : str, optional
        Read orientation (e.g., "L_R", "A_P", "H_F")
    slice_orient : str, optional
        Slice orientation (e.g., "axial", "sagittal", "coronal")
    patient_pos : str, optional
        Patient position (e.g., "Head_Prone", "Head_Supine")
    dtype : type, optional
        Data type for output arrays

    Returns
    -------
    tuple
        (x_grid, y_grid, z_grid) in physical coordinates
    """
    if data_obj is not None:
        read_orient = data_obj.get("read_orientation", read_orient)
        slice_orient = data_obj.get("slice_orientation", slice_orient)
        patient_pos = data_obj.get("patient_pos", patient_pos)

    if not all([read_orient, slice_orient, patient_pos]):
        raise ValueError("Missing orientation information for RPS to XYZ conversion")

    if rps_grid is None and data_obj is not None:
        from .grids import get_rps_grid

        rps_grid = get_rps_grid(data_obj=data_obj, dtype=dtype)

    read_grid, phase_grid, slice_grid = rps_grid

    logger.debug("=== convert_rps_to_xyz_grid: Starting transformation ===")
    logger.debug(
        f"  Orientation: slice={slice_orient}, read={read_orient}, patient={patient_pos}"
    )
    logger.debug(f"  Input RPS ranges:")
    logger.debug(f"    read:  [{read_grid.min():.2f}, {read_grid.max():.2f}] mm")
    logger.debug(f"    phase: [{phase_grid.min():.2f}, {phase_grid.max():.2f}] mm")
    logger.debug(f"    slice: [{slice_grid.min():.2f}, {slice_grid.max():.2f}] mm")

    # Initialize output grids
    x_grid = np.zeros_like(read_grid, dtype=dtype)
    y_grid = np.zeros_like(read_grid, dtype=dtype)
    z_grid = np.zeros_like(read_grid, dtype=dtype)

    transformation_applied = False

    # Should be independent of patient_pos for now
    if slice_orient == "axial":
        if read_orient == "L_R":
            logger.debug(f"  Using transformation: axial + L_R")
            x_grid = read_grid
            y_grid = phase_grid
            z_grid = slice_grid
            transformation_applied = True
        elif read_orient == "A_P":
            logger.debug(f"  Using transformation: axial + A_P")
            x_grid = phase_grid
            y_grid = read_grid
            z_grid = slice_grid
            transformation_applied = True
    elif slice_orient == "sagittal":
        if read_orient == "H_F":
            logger.debug(f"  Using transformation: sagittal + H_F")
            x_grid = slice_grid
            y_grid = phase_grid
            z_grid = read_grid
            transformation_applied = True
        elif read_orient == "A_P":
            logger.debug(f"  Using transformation: sagittal + A_P")
            x_grid = slice_grid
            y_grid = read_grid
            z_grid = phase_grid
            transformation_applied = True
    elif slice_orient == "coronal":
        if read_orient == "H_F":
            logger.debug(f"  Using transformation: coronal + H_F")
            x_grid = phase_grid
            y_grid = slice_grid
            z_grid = read_grid
            transformation_applied = True
        elif read_orient == "L_R":
            logger.debug(f"  Using transformation: coronal + L_R")
            x_grid = read_grid
            y_grid = slice_grid
            z_grid = phase_grid
            transformation_applied = True

    if not transformation_applied:
        logger.warning(
            f"  WARNING: No transformation matched for slice={slice_orient}, read={read_orient}"
        )
        logger.warning(f"  Defaulting to identity transformation (may be incorrect!)")
        x_grid = read_grid
        y_grid = phase_grid
        z_grid = slice_grid

    logger.debug(f"  Output XYZ ranges:")
    logger.debug(f"    x: [{x_grid.min():.2f}, {x_grid.max():.2f}] mm")
    logger.debug(f"    y: [{y_grid.min():.2f}, {y_grid.max():.2f}] mm")
    logger.debug(f"    z: [{z_grid.min():.2f}, {z_grid.max():.2f}] mm")
    logger.debug("=== convert_rps_to_xyz_grid: Complete ===")

    # if patient_pos == "Head_Prone":

    # elif patient_pos == "Head_Supine":
    #     # Implement Head_Supine transformations
    #     warnings.warn("Head_Supine transformations not fully implemented")
    #     # For now, use same as Head_Prone
    #     if slice_orient == "axial":
    #         if read_orient == "L_R":
    #             x_grid = read_grid
    #             y_grid = phase_grid
    #             z_grid = slice_grid

    return x_grid.astype(dtype), y_grid.astype(dtype), z_grid.astype(dtype)


def convert_rps_to_xyz_matrix_based(
    rps_grid: Tuple[np.ndarray, np.ndarray, np.ndarray],
    data_obj=None,
    acquisition_matrix: np.ndarray = None,
    position_matrix: np.ndarray = None,
    dtype=COORDINATE_PRECISION,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RPS (read, phase, slice) coordinate grid to XYZ coordinate grid using matrix multiplication.

    This implements the Bruker transformation formula:
        (x,y,z) = (r,p,s) @ ACQ_grad_matrix @ Position_Matrix

    Where:
    - ACQ_grad_matrix: Transformation from encoding coordinates (RPS) to subject coordinates
    - Position_Matrix: Transformation from subject coordinates to scanner physical coordinates

    This approach properly handles tilted/oblique acquisitions where each voxel's XYZ position
    depends on all three RPS coordinates (interdependent coordinates).

    Parameters
    ----------
    rps_grid : tuple of np.ndarray
        (read_grid, phase_grid, slice_grid) as 3D meshgrids in mm
    data_obj : object, optional
        Data object containing acquisition_matrix and position_matrix
    acquisition_matrix : np.ndarray, optional
        3x3 ACQ_grad_matrix from Bruker (if not in data_obj)
    position_matrix : np.ndarray, optional
        3x3 Position_Matrix based on patient position (if not in data_obj)
    dtype : type, optional
        Data type for output arrays

    Returns
    -------
    tuple
        (x_grid, y_grid, z_grid) as 3D meshgrids in physical coordinates
    """
    # Extract matrices from data_obj if provided
    if data_obj is not None:
        if acquisition_matrix is None:
            acquisition_matrix = data_obj.get("acquisition_matrix", None)
        if position_matrix is None:
            position_matrix = data_obj.get("position_matrix", None)

    # Validate matrices
    if acquisition_matrix is None:
        raise ValueError(
            "acquisition_matrix (ACQ_grad_matrix) is required for matrix-based transformation. "
            "This should be loaded from the Bruker acqp file."
        )

    if position_matrix is None:
        # Default to identity matrix if not provided
        logger.warning("position_matrix not provided, using identity matrix")
        position_matrix = np.eye(3)

    # Ensure matrices are 3x3 numpy arrays
    acquisition_matrix = np.array(acquisition_matrix)
    position_matrix = np.array(position_matrix)

    if acquisition_matrix.shape != (3, 3):
        raise ValueError(
            f"acquisition_matrix must be 3x3, got shape {acquisition_matrix.shape}"
        )
    if position_matrix.shape != (3, 3):
        raise ValueError(
            f"position_matrix must be 3x3, got shape {position_matrix.shape}"
        )

    read_grid, phase_grid, slice_grid = rps_grid

    logger.debug("=== convert_rps_to_xyz_matrix_based: Starting transformation ===")
    logger.debug(f"  ACQ_grad_matrix:\n{acquisition_matrix}")
    logger.debug(f"  Position_Matrix:\n{position_matrix}")
    logger.debug(f"  Input RPS ranges:")
    logger.debug(f"    read:  [{read_grid.min():.2f}, {read_grid.max():.2f}] mm")
    logger.debug(f"    phase: [{phase_grid.min():.2f}, {phase_grid.max():.2f}] mm")
    logger.debug(f"    slice: [{slice_grid.min():.2f}, {slice_grid.max():.2f}] mm")

    # Get the shape of the grids
    grid_shape = read_grid.shape
    n_points = np.prod(grid_shape)

    # Stack RPS grids into (N, 3) array where each row is [r, p, s] for one voxel
    # Flatten to handle all voxels at once
    rps_points = np.stack(
        [read_grid.ravel(), phase_grid.ravel(), slice_grid.ravel()], axis=1
    )  # Shape: (N, 3)

    logger.debug(f"  Reshaped to {n_points} points for transformation")

    # Combined transformation matrix: ACQ_grad_matrix @ Position_Matrix
    # Note: We apply ACQ_grad_matrix first, then Position_Matrix
    combined_matrix = acquisition_matrix @ position_matrix

    logger.debug(f"  Combined transformation matrix:\n{combined_matrix}")

    # Apply transformation: xyz = rps @ combined_matrix
    # Each row: [x, y, z] = [r, p, s] @ combined_matrix
    xyz_points = rps_points @ combined_matrix  # Shape: (N, 3)

    # Reshape back to original grid shape
    x_grid = xyz_points[:, 0].reshape(grid_shape).astype(dtype)
    y_grid = xyz_points[:, 1].reshape(grid_shape).astype(dtype)
    z_grid = xyz_points[:, 2].reshape(grid_shape).astype(dtype)

    logger.debug(f"  Output XYZ ranges:")
    logger.debug(f"    x: [{x_grid.min():.2f}, {x_grid.max():.2f}] mm")
    logger.debug(f"    y: [{y_grid.min():.2f}, {y_grid.max():.2f}] mm")
    logger.debug(f"    z: [{z_grid.min():.2f}, {z_grid.max():.2f}] mm")
    logger.debug("=== convert_rps_to_xyz_matrix_based: Complete ===")

    return x_grid, y_grid, z_grid


def translate_coordinates(original_coords, fov, matrix_size):
    """
    Translate coordinates based on field of view and matrix size.

    Parameters
    ----------
    original_coords : array_like
        Original coordinate values
    fov : float
        Field of view in mm
    matrix_size : int
        Matrix size in voxels

    Returns
    -------
    array_like
        Translated coordinates
    """
    voxel_size = fov / matrix_size
    center_offset = fov / 2.0

    # Convert from voxel indices to physical coordinates
    physical_coords = (original_coords * voxel_size) - center_offset

    return physical_coords


def find_indexing_scanner_dim(input_data=None, object_coord_name=None, object_idx=None):
    """
    Find the scanner dimension with the least variation along a specified dimension.
    This helps to find the matching scanner dimension for a given object dimension.

    Parameters
    ----------
    input_data: xr.DataArray
        The input data containing coordinates.
    object_coord_name: str
        name of the object coordinate (e.g., 'read_idx', 'phase_idx', 'slice_idx')
    object_idx: int
        index of the object coordinate to analyze

    Example
    -------
    >>> from hypermri.utils.coordinates.transformations import find_indexing_scanner_dim, find_original_object_dimension
    >>> scanner_dim_1 = find_indexing_scanner_dim(input_data=t2_cor.reco_data, object_coord_name='slice_idx', object_idx=10)
    >>> print(f"scanner_dim_1={scanner_dim_1}")
    >>> print(f"Scanner dimension for read_idx: {scanner_dim_1[0]}, index: {scanner_dim_1[1]}, closest actual value: {scanner_dim_1[2]}")

    >>> result = find_original_object_dimension(
    >>>     input_data=t2_ax.reco_data,
    >>>     scanner_coord_name=scanner_dim_1[0],  # 'z'
    >>>     closest_actual_val=scanner_dim_1[2]   # -14.5
    >>> )
    >>> print(f"Corrected dimension mapping: {result}")
    >>> fig, ax = plt.subplots(1,2)
    >>> ax[0].imshow(np.squeeze(np.abs(t2_cor.reco_data.isel(slice_idx=10))), aspect='auto')
    >>> ax[1].imshow(np.squeeze(np.abs(t2_ax.reco_data.isel(**result))), aspect='auto')

    Returns
    -------
    tuple
        [min_key, coordinate, str] the scanner dimension with the least variation
    """
    coordinate_diffs = {}
    for scanner_dim in ["x", "y", "z"]:
        if scanner_dim in input_data.coords:
            coordinate_diffs[scanner_dim] = np.min(
                input_data.coords[scanner_dim]
                .isel(**{object_coord_name: object_idx})
                .values
            ) - np.max(
                input_data.coords[scanner_dim]
                .isel(**{object_coord_name: object_idx})
                .values
            )

    # find minimum mean --> means that the cooridantes along this dimension vary the least
    min_val = None
    for key, value in coordinate_diffs.items():
        if min_val is None or np.abs(value) < min_val:
            min_val = np.abs(value)
            min_key = key

    mean_val = np.mean(
        input_data.coords[min_key].isel(**{object_coord_name: object_idx}).values
    )

    mean_coords_1d = input_data.coords[object_coord_name]
    for object_dim in ["read_idx", "phase_idx", "slice_idx"]:
        if object_dim in input_data.coords and object_dim != object_coord_name:
            mean_coords_1d = mean_coords_1d.mean(dim=object_dim)
    # print(f"Mean value along {object_coord_name} = {mean_coords_1d.values}")

    index = np.argmin(np.abs(np.array(mean_coords_1d) - mean_val))
    closest_actual_val = mean_coords_1d.values[int(index)]

    return [min_key, int(index), closest_actual_val]  # Convert to Python int


def find_original_object_dimension(input_data, scanner_coord_name, closest_actual_val):
    """
    Find which object dimension corresponds to the scanner coordinate by analyzing
    the relationship across the entire coordinate space, not just at one point.

    Parameters
    ----------
    input_data: xarray DataArray
        containing the coordinates
    scanner_coord_name: str
        name of the scanner coordinate (e.g., 'x', 'y', 'z')
    closest_actual_val: float
        the actual value of the scanner coordinate at the desired index

    Example
    -------
    >>> from hypermri.utils.coordinates.transformations import find_indexing_scanner_dim, find_original_object_dimension
    >>> scanner_dim_1 = find_indexing_scanner_dim(input_data=t2_cor.reco_data, object_coord_name='slice_idx', object_idx=10)
    >>> print(f"scanner_dim_1={scanner_dim_1}")
    >>> print(f"Scanner dimension for read_idx: {scanner_dim_1[0]}, index: {scanner_dim_1[1]}, closest actual value: {scanner_dim_1[2]}")

    >>> result = find_original_object_dimension(
    >>>     input_data=t2_ax.reco_data,
    >>>     scanner_coord_name=scanner_dim_1[0],  # 'z'
    >>>     closest_actual_val=scanner_dim_1[2]   # -14.5
    >>> )
    >>> print(f"Corrected dimension mapping: {result}")
    >>> fig, ax = plt.subplots(1,2)
    >>> ax[0].imshow(np.squeeze(np.abs(t2_cor.reco_data.isel(slice_idx=10))), aspect='auto')
    >>> ax[1].imshow(np.squeeze(np.abs(t2_ax.reco_data.isel(**result))), aspect='auto')

    Returns
    -------
    dict
        mapping of the object dimension to the index in that dimension closest to the target
    """
    coord_array = input_data.coords[scanner_coord_name]
    dims = coord_array.dims  # ('read_idx', 'phase_idx', 'slice_idx')

    # For each object dimension, calculate how much the scanner coordinate
    # varies when that dimension changes (averaged across other dimensions)
    dimension_variations = {}

    for dim in dims:
        # Calculate variation of scanner coordinate along this dimension
        # by averaging over the other dimensions
        other_dims = [d for d in dims if d != dim]

        # Average the scanner coordinate over the other dimensions
        averaged_coord = coord_array.mean(dim=other_dims)

        # Calculate the variation (range) along this dimension
        variation = float(averaged_coord.max().values - averaged_coord.min().values)
        dimension_variations[dim] = variation

    # The dimension with the MOST variation is the one that corresponds
    # to our scanner coordinate
    max_variation_dim = max(dimension_variations, key=dimension_variations.get)

    # Now find the index in that dimension closest to our target
    averaged_coord = coord_array.mean(dim=[d for d in dims if d != max_variation_dim])
    index = np.argmin(np.abs(averaged_coord.values - closest_actual_val))

    return {max_variation_dim: int(index)}
