"""
Image shift and transformation utilities.

This module contains functions for shifting and transforming medical images,
extracted from the monolithic utils_general.py as part of Phase 2 refactoring.
"""

import logging
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr
from scipy.ndimage import shift
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

# Initialize logger without dependencies
logger = logging.getLogger(__name__)


def get_orientation_flip_multipliers(data_obj) -> np.ndarray:
    """
    Determine the sign multipliers for read/phase/slice axes based on patient and slice orientation.
    This function uses the patient position, slice orientation, read orientation, to determine of raw reco Bruker data
    needs to be flipped along any of the three axes to match a standard orientation (seq2d).

    Parameters
    ----------
    data_obj : object
        Data object providing patient_pos, slice_orientation, read_orientation, and spat_dim attributes.

    Returns
    -------
    np.ndarray
        Array of length 3 with values {1, -1} indicating whether each axis should be flipped.
        Order corresponds to (read, phase, slice).
    """

    try:
        patient_pos = getattr(data_obj, "patient_pos", None)
        slice_orientation = getattr(data_obj, "slice_orientation", None)
        read_orientation = getattr(data_obj, "read_orientation", None)
        spat_dim = getattr(data_obj, "spat_dim", None)
    except Exception:
        # Fallback to identity if attributes are missing
        return np.ones(3, dtype=int)

    read_mult = 1
    phase_mult = 1
    slice_mult = 1

    if patient_pos == "Head_Prone":
        if slice_orientation == "axial":
            if read_orientation == "L_R":
                read_mult = -1
                phase_mult = -1
            elif read_orientation == "A_P":
                read_mult = -1
                phase_mult = -1
            else:
                Warning(
                    f"Unexpected read_orientation '{read_orientation}' for patient_pos 'Head_Prone' and slice_orientation '{slice_orientation}'. No flips applied."
                )
        elif slice_orientation == "sagittal":
            if read_orientation == "A_P":
                read_mult = -1
                phase_mult = -1
            elif read_orientation == "H_F":
                read_mult = -1
                phase_mult = -1
            else:
                Warning(
                    f"Unexpected read_orientation '{read_orientation}' for patient_pos 'Head_Prone' and slice_orientation '{slice_orientation}'. No flips applied."
                )

        elif slice_orientation == "coronal":
            read_mult = -1
            phase_mult = -1
            if spat_dim == "3D":
                slice_mult = -1

    elif patient_pos == "Head_Supine":
        if slice_orientation == "axial":
            if read_orientation == "L_R":
                read_mult = -1
                phase_mult = -1
            elif read_orientation == "A_P":
                read_mult = -1
                phase_mult = -1
                slice_mult = -1
            else:
                Warning(
                    f"Unexpected read_orientation '{read_orientation}' for patient_pos 'Head_Supine' and slice_orientation '{slice_orientation}'. No flips applied."
                )
        elif slice_orientation == "sagittal":
            if read_orientation == "H_F":
                read_mult = -1
                phase_mult = -1
            elif read_orientation == "A_P":
                read_mult = -1
                phase_mult = -1
            else:
                Warning(
                    f"Unexpected read_orientation '{read_orientation}' for patient_pos 'Head_Supine' and slice_orientation '{slice_orientation}'. No flips applied."
                )
        elif slice_orientation == "coronal":
            if read_orientation in ("H_F", "L_R"):
                read_mult = -1
                phase_mult = -1
            else:
                Warning(
                    f"Unexpected read_orientation '{read_orientation}' for patient_pos 'Head_Supine' and slice_orientation '{slice_orientation}'. No flips applied."
                )

    elif patient_pos == "Foot_Prone":
        Warning(
            "Patient position 'Foot_Prone' not fully tested, please verify results!"
        )
        if slice_orientation == "axial":
            if read_orientation in ("L_R", "A_P"):
                read_mult = -1
                phase_mult = -1
                slice_mult = -1
        elif slice_orientation == "sagittal":
            if read_orientation == "H_F":
                read_mult = -1
                phase_mult = -1
            elif read_orientation == "A_P":
                read_mult = -1
        elif slice_orientation == "coronal":
            if read_orientation in ("H_F", "L_R"):
                read_mult = -1
                phase_mult = -1

    return np.array([read_mult, phase_mult, slice_mult], dtype=int)


def calc_mat_origin_diff(
    res_metab=None,
    fov_metab=None,
    mat_metab=None,
    res_anat=None,
    fov_anat=None,
    mat_anat=None,
):
    """Calculate voxel offsets required to align metabolite and anatomical datasets.

    Parameters mirror the legacy implementation to remain backward compatible. Either
    the explicit resolutions or the combination of field-of-view and matrix sizes must
    be provided for both datasets. Returned offsets follow the order ``[z, x, y]`` and
    are expressed in voxel units of the high-resolution anatomical acquisition.
    """

    if res_metab is None:
        if fov_metab is not None and mat_metab is not None:
            res_metab = [a / b for a, b in zip(fov_metab, mat_metab)]
        else:
            return None, None, None

    if res_anat is None:
        if fov_anat is not None and mat_anat is not None:
            res_anat = [a / b for a, b in zip(fov_anat, mat_anat)]
        else:
            return None, None, None

    if fov_anat is None or mat_anat is None:
        return None, None, None

    shift_vox_list = [
        ma * (rm - ra) / 2 / fa
        for rm, ra, fa, ma in zip(res_metab, res_anat, fov_anat, mat_anat)
    ]
    return shift_vox_list


def shift_image(
    input_data=None,
    shift_vox=None,
    shift_method="phase",
    force_shift=False,
    data_obj=None,
    domain="kspace",
):
    """
    Shift the input data using either a phase ramp or a direct spatial shift.

    This function modifies the data using either a phase ramp or a direct spatial shift via the
    scipy.ndimage.shift function, based on the specified method. The input_data is expected to be
    in the k-space domain (default) and can be changed with the domain parameter.

    Parameters
    ----------
    input_data : ndarray or xarray.DataArray, optional
        The input data array. If None and `data_obj` is provided, the k-space data from
        `data_obj.kspace_xarray` is used. If None and `data_obj` is also None, an error is raised.
    shift_vox : array_like of float, optional
        The number of voxels to shift along each axis. If None, the shift is determined by the
        `determine_necessary_shift` method (only applicable when `data_obj` is provided).
        If `input_data` is a numpy array and `data_obj` is None, this parameter is required
        and must have the same number of dimensions as `input_data`.
    shift_method : {'phase', 'scipy'}, default 'phase'
        The method used to perform the shift. 'phase' uses a linear phase ramp, while 'scipy' uses
        scipy's shift function.
    force_shift : bool, default False
        If True, forces the shift operation even if shifts have been previously applied.
    domain : {'kspace', 'image'}, default 'kspace'
        The domain in which the shift should be applied. The input data will be transformed
        to this domain if necessary and transformed back afterwards.
    data_obj : object, optional
        An object (e.g., hypemri.sequence object) that might contain k-space data or information
        needed to determine the shift.

    Returns
    -------
    ndarray or xarray.DataArray
        The shifted data array.

    Raises
    ------
    ValueError
        If an unsupported shift method is provided.
        If `input_data` is None and `data_obj` is also None.
        If `input_data` is a numpy array, `data_obj` is None, and `shift_vox` is None.
        If `input_data` is a numpy array, `data_obj` is None, and the number of dimensions of
        `shift_vox` does not match the number of dimensions of `input_data`.
    ImportError
        If `shift_method` is 'scipy' and scipy is not installed.

    Notes
    -----
    The shift is applied in the specified domain. The input_data will be transformed to this
    domain if it is not already in it, and transformed back to the original domain afterwards.
    """

    if input_data is None:
        if data_obj is None:
            raise ValueError("Either input_data or data_obj must be provided.")
        if hasattr(data_obj, "kspace_xarray"):
            kspace_xarray = data_obj.kspace_xarray
            input_is_xarray = True
            data_np = kspace_xarray.to_numpy()
        else:
            raise ValueError("data_obj does not have the attribute kspace_xarray.")
    elif isinstance(input_data, xr.DataArray):
        kspace_xarray = input_data
        input_is_xarray = True
        data_np = kspace_xarray.to_numpy()
    elif isinstance(input_data, np.ndarray):
        data_np = input_data
        input_is_xarray = False
        if data_obj is None:
            if shift_vox is None:
                raise ValueError(
                    "shift_vox must be provided when input_data is a numpy array and data_obj is None."
                )
            if len(shift_vox) != data_np.ndim:
                raise ValueError(
                    "The number of dimensions of shift_vox must match the number of dimensions of input_data when data_obj is None."
                )
    else:
        raise ValueError(
            "Unsupported type for input_data. Must be ndarray or xarray.DataArray."
        )

    if input_is_xarray and "domain" in kspace_xarray.attrs:
        domain = kspace_xarray.attrs["domain"]

    if input_is_xarray:
        read_axis = kspace_xarray.get_axis_num("read_idx")
        phase_axis = kspace_xarray.get_axis_num("phase_idx")
        if kspace_xarray.spat_dim == "2D":
            slice_axis = None
        elif kspace_xarray.spat_dim == "3D":
            slice_axis = kspace_xarray.get_axis_num("slice_idx")
        else:
            slice_axis = None

        if slice_axis is not None:
            spatial_axes = (read_axis, phase_axis, slice_axis)
        else:
            spatial_axes = (read_axis, phase_axis)
        is_complex = np.iscomplexobj(data_np)

        num_image_fov_shift = kspace_xarray.attrs.get("num_image_fov_shift", 0)
        if num_image_fov_shift > 0 and not force_shift:
            return kspace_xarray

        if shift_vox is None:
            try:
                # Determine debug mode from log_mode
                shift_rps_vox, _ = determine_FOV_shift(
                    data_obj=data_obj,
                )

            except Exception as e:
                raise ValueError(
                    "Error determining FOV shift: "
                    f"{ str(e)}. Please provide shift_vox or ensure data_obj is valid."
                )
        else:
            shift_rps_vox = shift_vox

    else:  # numpy array
        if shift_vox is None:
            raise ValueError(
                "shift_vox must be provided for numpy array input when data_obj is None."
            )
        if len(shift_vox) != data_np.ndim:
            raise ValueError(
                "The length of shift_vox must match the number of dimensions of input_data."
            )
        spatial_axes = tuple(range(data_np.ndim - len(shift_vox), data_np.ndim))
        is_complex = np.iscomplexobj(data_np)
        shift_rps_vox = shift_vox

    # Determine if this is 3D data for debugging
    is_3d = (
        input_is_xarray
        and hasattr(kspace_xarray, "spat_dim")
        and kspace_xarray.spat_dim == "3D"
    )
    logger.debug(
        f"Shifting {'3D' if is_3d else '2D'} data with shape {data_np.shape}, "
        f"shift_rps_vox: {shift_rps_vox}, shift_method: {shift_method}, domain: {domain}"
    )
    if is_3d:
        logger.debug(f"3D spatial axes: {spatial_axes} (read, phase, slice)")
    else:
        logger.debug(f"2D spatial axes: {spatial_axes} (read, phase)")
    # Transform to the specified domain
    shifted_data = data_np.copy()
    if np.all(np.iscomplex(shifted_data)):
        pass
    else:
        shifted_data = shifted_data.astype(complex)

    # Apply the shift
    if shift_method == "scipy":
        from scipy.ndimage import shift

        if domain == "kspace":
            for axis in spatial_axes:
                shifted_data = np.fft.fftshift(shifted_data, axes=axis)
                shifted_data = np.fft.fft(shifted_data, axis=axis)
                shifted_data = np.fft.ifftshift(shifted_data, axes=axis)

        # perform shift:
        real_shifted = shift(np.real(shifted_data), shift_rps_vox, mode="wrap")
        imag_shifted = shift(np.imag(shifted_data), shift_rps_vox, mode="wrap")
        shifted_data = real_shifted + 1j * imag_shifted

        if domain == "kspace":
            # Transform back to k-space
            for axis in spatial_axes:
                shifted_data = np.fft.fftshift(shifted_data, axes=axis)
                shifted_data = np.fft.ifft(shifted_data, axis=axis)
                shifted_data = np.fft.ifftshift(shifted_data, axes=axis)
        else:
            # perform shift:
            real_shifted = shift(np.real(shifted_data), shift_rps_vox, mode="wrap")
            imag_shifted = shift(np.imag(shifted_data), shift_rps_vox, mode="wrap")
            shifted_data = real_shifted + 1j * imag_shifted

    elif shift_method == "phase":
        if domain != "kspace":
            # Transform to k-space to apply phase shift
            for axis in spatial_axes:
                shifted_data = np.fft.ifftshift(shifted_data, axes=axis)
                shifted_data = np.fft.ifft(shifted_data, axis=axis)
                shifted_data = np.fft.ifftshift(shifted_data, axes=axis)
        else:
            pass

        grid = np.meshgrid(*[np.arange(s) for s in shifted_data.shape], indexing="ij")
        for i, axis in enumerate(spatial_axes):
            axis_names = ["read", "phase", "slice"]
            axis_name = axis_names[i] if i < len(axis_names) else f"axis_{i}"
            logger.debug(
                f"Applying phase shift on {axis_name} axis {axis} with shift {shift_rps_vox[i]} voxels"
            )
            shift = shift_rps_vox[i]
            if shift == 0:
                continue
            size = shifted_data.shape[axis]
            phase_ramp = np.exp(1j * 2 * np.pi * grid[axis] / size * shift)
            shifted_data *= phase_ramp

        # Transform back to image domain
        if domain != "kspace":
            for axis in spatial_axes:
                shifted_data = np.fft.fftshift(shifted_data, axes=axis)
                shifted_data = np.fft.fft(shifted_data, axis=axis)
                shifted_data = np.fft.ifftshift(shifted_data, axes=axis)
        else:
            pass
    else:
        raise ValueError(
            f"Unsupported shift method: {shift_method}, use either 'phase' or 'scipy'."
        )

    if input_is_xarray:
        if not is_complex:
            shifted_data = np.abs(shifted_data)
        kspace_xarray.data = shifted_data
        if not hasattr(kspace_xarray.attrs, "num_image_fov_shift"):
            kspace_xarray.attrs["num_image_fov_shift"] = 0
        kspace_xarray.attrs["num_image_fov_shift"] += 1
        return kspace_xarray
    else:
        if not is_complex:
            shifted_data = np.abs(shifted_data)
        return shifted_data


def determine_FOV_shift(
    data_obj=None,
):
    """
    Calculate the necessary voxel and millimeter shifts required to align the MRI image
    based on the patient's position [Phase/Read/Slice Offset] and the scan orientation.

    Parameters
    ----------
    data_obj : object, optional
        Data object containing scan parameters (read_offset_mm, phase_offset_mm, slice_offset_mm, resolution_mm)

    Returns
    -------
    tuple
        (shift_rps_vox, shift_rps_mm) - shift in voxels and millimeters for read, phase, and slice directions
    """
    # Set logger level based on log_mode if provided
    if data_obj is None:
        logger.critical("data_obj is None, returning zero shifts")
        return np.array([0, 0]), np.array([0.0, 0.0])

    # Get offsets and resolution from data object
    try:
        read_offset_mm = data_obj.read_offset_mm
        phase_offset_mm = data_obj.phase_offset_mm

        # Try to get slice offset, default to 0 if not available
        slice_offset_mm = getattr(data_obj, "slice_offset_mm", 0)
        resolution_mm = data_obj.resolution_mm
    except AttributeError as e:
        logger.warning(
            f"Missing required attribute in data_obj: {e}, returning zero shifts"
        )
        return np.array([0, 0]), np.array([0.0, 0.0])

    if data_obj.type == "bruker":
        pass

    # Calculate shifts in mm and voxels
    shift_rps_mm = [read_offset_mm, phase_offset_mm, slice_offset_mm]
    shift_rps_vox = [0, 0, 0]
    
    # Check if this is a CSI sequence to determine read direction shifting
    is_csi = False
    if hasattr(data_obj, "type") and data_obj.type:
        # Check sequence type string (case insensitive)
        sequence_type = str(data_obj.type).lower()
        is_csi = "csi" in sequence_type
        logger.debug(f"Sequence type: {data_obj.type}, is_csi: {is_csi}")
    elif hasattr(data_obj, "__class__"):
        # Check class name if type attribute not available
        class_name = data_obj.__class__.__name__.lower()
        is_csi = "csi" in class_name
        logger.debug(f"Class name: {data_obj.__class__.__name__}, is_csi: {is_csi}")
    else:
        logger.debug("Could not determine sequence type, assuming non-CSI")

    # Convert mm shifts to voxel shifts (note the negative sign for proper alignment)
    # Read direction: only apply shift for CSI sequences
    shift_rps_vox[0] = (
        shift_rps_mm[0] / resolution_mm[0] if is_csi else 0
    )  # read direction
    shift_rps_vox[1] = shift_rps_mm[1] / resolution_mm[1]  # phase direction
    shift_rps_vox[2] = (
        shift_rps_mm[2] / resolution_mm[2] if len(resolution_mm) > 2 else 0
    )  # slice direction

    if is_csi:
        logger.debug("CSI sequence detected: applying read direction shift")
    else:
        logger.debug("Non-CSI sequence: read direction shift set to 0")

    # Output debug information based on db flag and log_mode
    flip_multipliers = get_orientation_flip_multipliers(data_obj)
    # Apply orientation multipliers to both voxel and mm shifts
    shift_rps_vox = np.multiply(shift_rps_vox, flip_multipliers).astype(float)
    shift_rps_mm = np.multiply(shift_rps_mm, flip_multipliers).astype(float)

    logger.info(f"shift_rps_vox: {shift_rps_vox}")
    logger.info(f"shift_rps_mm: {shift_rps_mm}")

    # For 3D data, return all 3 components; for 2D data, return only read/phase
    if hasattr(data_obj, "spat_dim") and data_obj.spat_dim == "3D":
        return shift_rps_vox, shift_rps_mm
    else:
        return shift_rps_vox[:2], shift_rps_mm[:2]


def _ensure_dataarray(
    data: Union[xr.DataArray, np.ndarray],
    dims_hint: Optional[Sequence[str]] = None,
) -> xr.DataArray:
    """Return an xarray.DataArray, inferring dimensions when possible."""
    if isinstance(data, xr.DataArray):
        return data

    if dims_hint is None:
        raise ValueError(
            "dims_hint is required when providing numpy arrays to alignment utilities."
        )

    array = np.asarray(data)
    if len(dims_hint) != array.ndim:
        raise ValueError(
            f"dims_hint has length {len(dims_hint)}, but data has {array.ndim} dimensions."
        )

    return xr.DataArray(array, dims=tuple(dims_hint))


def _normalize_for_ssim(values: np.ndarray) -> np.ndarray:
    """Normalize values into the [0, 1] interval for SSIM computation."""
    arr = np.asarray(values, dtype=np.float32)
    arr = arr - np.nanmin(arr)
    max_val = np.nanmax(arr)
    if max_val > 0:
        arr /= max_val
    return arr


def _shift_complex_nd(
    data: np.ndarray,
    shift_amounts: Sequence[float],
    order: int = 3,
    mode: str = "constant",
) -> np.ndarray:
    """Apply a sub-pixel shift to complex data."""
    real_shifted = shift(np.real(data), shift_amounts, order=order, mode=mode)
    imag_shifted = shift(np.imag(data), shift_amounts, order=order, mode=mode)
    return real_shifted + 1j * imag_shifted


def _average_repetition_dimensions(
    data: xr.DataArray,
    plane_dims: Sequence[str],
    repetition_dim_keywords: Sequence[str],
) -> xr.DataArray:
    """Average dimensions likely representing repetitions before projection."""
    lowered_keywords = [kw.lower() for kw in repetition_dim_keywords]
    rep_dims = [
        dim
        for dim in data.dims
        if dim not in plane_dims
        and any(keyword in dim.lower() for keyword in lowered_keywords)
    ]
    if not rep_dims:
        return data
    return data.mean(dim=rep_dims)


def _prepare_projection(
    data: xr.DataArray,
    plane_dims: Sequence[str],
    reduce_dims: Union[str, Sequence[str], None],
    reduce_func: Union[
        str, Callable[[xr.DataArray, Sequence[str]], xr.DataArray]
    ] = "mean",
) -> xr.DataArray:
    """
    Reduce an N-D xarray.DataArray down to the requested imaging plane.

    Parameters
    ----------
    data
        Source array with at least the dimensions listed in ``plane_dims``.
    plane_dims
        Ordered tuple of dimensions that describe the 2D imaging plane
        (e.g., ``(\"read_idx\", \"phase_idx\")``). The returned DataArray is
        transposed to match this order.
    reduce_dims
        Dimensions to collapse before extracting the plane. ``\"auto\"`` (default)
        removes every dimension not present in ``plane_dims``. ``None`` skips the
        reduction step. A string is treated as a single dimension name.
    reduce_func
        Aggregation to apply when reducing. Supported values are ``\"mean\"``,
        ``\"median\"``, or a callable accepting ``(data, dims_to_reduce)`` and
        returning an xarray.DataArray.

    Returns
    -------
    xr.DataArray
        2D plane ordered according to ``plane_dims``. All requested reductions are
        applied and singleton dimensions squeezed.

    Raises
    ------
    ValueError
        If ``plane_dims`` contain missing dimensions or ``reduce_func`` is not one
        of the supported values/callables.
    """
    missing = [dim for dim in plane_dims if dim not in data.dims]
    if missing:
        raise ValueError(f"Data is missing required dimensions: {missing}")

    if reduce_dims == "auto" or reduce_dims is None:
        reduce_dims = [dim for dim in data.dims if dim not in plane_dims]
    else:
        reduce_dims = list(reduce_dims)

    reduced = data
    if reduce_dims:
        if callable(reduce_func):
            reduced = reduce_func(reduced, reduce_dims)
        elif reduce_func == "mean":
            reduced = reduced.mean(dim=reduce_dims)
        elif reduce_func == "median":
            reduced = reduced.median(dim=reduce_dims)
        else:
            raise ValueError(
                f"Unsupported reduce_func '{reduce_func}'. Provide 'mean', 'median', or a callable."
            )

    reduced = reduced.squeeze()
    reduced = reduced.transpose(*plane_dims)
    return reduced


def estimate_image_alignment(
    source: Union[xr.DataArray, np.ndarray],
    reference: Union[xr.DataArray, np.ndarray],
    image_dims: Sequence[str] = ("read_idx", "phase_idx"),
    reduce_dims: Union[str, Sequence[str], None] = "auto",
    reduce_func: Union[
        str, Callable[[xr.DataArray, Sequence[str]], xr.DataArray]
    ] = "mean",
    flip_mode: str = "auto",
    flip_dim: Optional[str] = "phase_idx",
    upsample_factor: int = 10,
    shift_order: int = 3,
    average_repetitions: bool = True,
    repetition_dim_keywords: Sequence[str] = (
        "rep",
        "repetition",
        "average",
        "repetition_idx",
    ),
) -> Dict[str, Union[float, Dict[str, float], bool, xr.DataArray]]:
    """
    Estimate the flip + sub-voxel translation required to align ``source`` to ``reference``.

    Parameters
    ----------
    source, reference
        Complex-valued data defined over at least the dimensions listed in ``image_dims``.
        ``numpy`` arrays must be ordered according to ``image_dims``.
    image_dims
        Size-2 iterable naming the imaging axes (typically ``(\"read_idx\", \"phase_idx\")``).
    reduce_dims
        Dimensions to average/aggregate away prior to registration. ``\"auto\"`` removes
        every dimension not present in ``image_dims``. ``None`` disables reduction.
    reduce_func
        Aggregation used when collapsing ``reduce_dims``. Accepts ``\"mean\"``, ``\"median\"``,
        or a callable ``f(data, dims)`` returning an xarray.DataArray.
    flip_mode
        Controls whether a flip along ``flip_dim`` is considered: ``\"auto\"`` tests both
        orientations, ``\"always\"`` forces a flip, and ``\"never\"`` skips it.
    flip_dim
        Dimension inside ``image_dims`` to flip when testing mirrored alignments. ``None``
        disables flipping entirely.
    upsample_factor
        Sub-pixel refinement factor passed to ``phase_cross_correlation``.
    shift_order
        Interpolation order used when previewing the shift (0 = nearest neighbor, 3 = cubic).
    average_repetitions
        When ``True`` (default), automatically average dimensions whose name contains any
        of ``repetition_dim_keywords`` before extracting the imaging plane. This reduces
        noise from repeated acquisitions while leaving the original data untouched.
    repetition_dim_keywords
        Keywords used to detect repetition-like dimensions. Matching ignores case.

    Returns
    -------
    dict
        Dictionary containing:

        - ``\"shifts\"``: mapping of each image dimension to the detected translation in voxels.
        - ``\"shift_vector\"``: tuple sorted according to ``image_dims``.
        - ``\"flip_applied\"`` / ``\"flip_dim\"``: indicates whether the best match required a flip.
        - ``\"ssim\"`` plus raw ``phase_cross_correlation`` metrics.
        - ``\"preview_source\"``, ``\"preview_reference\"``, ``\"preview_aligned\"``: 2D projections
          useful for quick visualization.

    Raises
    ------
    ValueError
        If inputs are missing required dimensions or invalid flip/reduce settings are supplied.
    RuntimeError
        If no alignment scenario can be evaluated (should only occur with empty inputs).
    """
    plane_dims = tuple(image_dims)
    source_da = _ensure_dataarray(source, dims_hint=plane_dims)
    reference_da = _ensure_dataarray(reference, dims_hint=plane_dims)

    if average_repetitions:
        source_da = _average_repetition_dimensions(
            source_da,
            plane_dims=plane_dims,
            repetition_dim_keywords=repetition_dim_keywords,
        )
        reference_da = _average_repetition_dimensions(
            reference_da,
            plane_dims=plane_dims,
            repetition_dim_keywords=repetition_dim_keywords,
        )

    source_plane = _prepare_projection(
        source_da,
        plane_dims=plane_dims,
        reduce_dims=reduce_dims,
        reduce_func=reduce_func,
    )
    reference_plane = _prepare_projection(
        reference_da,
        plane_dims=plane_dims,
        reduce_dims=reduce_dims,
        reduce_func=reduce_func,
    )

    if flip_dim and flip_dim not in plane_dims:
        raise ValueError(
            f"flip_dim '{flip_dim}' is not part of the image plane {plane_dims}."
        )

    flip_axis = (
        plane_dims.index(flip_dim) if flip_dim and flip_dim in plane_dims else None
    )
    if flip_mode not in {"auto", "never", "always"}:
        raise ValueError("flip_mode must be 'auto', 'never', or 'always'.")

    flip_options = [False]
    if flip_axis is not None:
        if flip_mode == "always":
            flip_options = [True]
        elif flip_mode == "auto":
            flip_options = [False, True]
    elif flip_mode in {"always"}:
        raise ValueError(
            "flip_mode 'always' requires flip_dim to be part of the plane."
        )

    best_result: Optional[
        Dict[str, Union[float, Dict[str, float], bool, xr.DataArray]]
    ] = None

    reference_values = reference_plane.data
    mag_reference = np.abs(reference_values)

    for flip in flip_options:
        candidate = source_plane
        if flip and flip_axis is not None:
            candidate = candidate.isel({plane_dims[flip_axis]: slice(None, None, -1)})

        candidate_values = candidate.data
        mag_source = np.abs(candidate_values)

        shifts, error, phasediff = phase_cross_correlation(
            mag_reference, mag_source, upsample_factor=upsample_factor
        )
        aligned_preview = _shift_complex_nd(
            candidate_values, shifts, order=shift_order, mode="constant"
        )

        ssim_score = float(
            structural_similarity(
                _normalize_for_ssim(mag_reference),
                _normalize_for_ssim(np.abs(aligned_preview)),
                data_range=1.0,
            )
        )

        preview_coords = {
            dim: candidate.coords[dim] for dim in plane_dims if dim in candidate.coords
        }

        result: Dict[str, Union[float, Dict[str, float], bool, xr.DataArray]] = {
            "shifts": {
                dim: float(shift_val) for dim, shift_val in zip(plane_dims, shifts)
            },
            "shift_vector": tuple(float(val) for val in shifts),
            "flip_applied": bool(flip),
            "flip_dim": flip_dim if (flip and flip_axis is not None) else None,
            "ssim": ssim_score,
            "phase_correlation_error": float(error),
            "phase_correlation_phasediff": float(phasediff),
            "preview_source": candidate,
            "preview_reference": reference_plane,
            "preview_aligned": xr.DataArray(
                aligned_preview,
                dims=plane_dims,
                coords=preview_coords,
            ),
        }

        if best_result is None or ssim_score > best_result["ssim"]:  # type: ignore[index]
            best_result = result

    if best_result is None:
        raise RuntimeError("Alignment estimation failed to evaluate any scenarios.")

    return best_result


def apply_image_alignment(
    input_data: Union[xr.DataArray, np.ndarray],
    alignment: Mapping[str, Union[float, Sequence[float], Mapping[str, float], bool]],
    *,
    image_dims: Sequence[str] = ("read_idx", "phase_idx"),
    order: int = 3,
    mode: str = "constant",
    wrap_edges: bool = True,
) -> xr.DataArray:
    """
    Apply previously estimated flip/shift parameters to an entire dataset.

    Parameters
    ----------
    input_data
        Data to transform. Accepts an xarray.DataArray (preferred) or a numpy ndarray
        ordered according to ``image_dims``.
    alignment
        Mapping produced by :func:`estimate_image_alignment`. At minimum it should
        contain ``\"shifts\"`` and ``\"flip_applied\"``. Extra metadata is preserved.
    image_dims
        Ordered imaging axes that describe the plane over which the shift is applied.
    order
        Interpolation order forwarded to ``scipy.ndimage.shift`` (0=nearest, 1=linear, ...).
    mode
        Padding mode used during interpolation. Ignored when ``wrap_edges=True``.
    wrap_edges
        When ``True``, perform circular wrapping at the image boundaries (``mode=\"wrap\"``).
        ``False`` keeps the previous zero-padding behaviour.

    Returns
    -------
    xr.DataArray
        Copy of ``input_data`` with shifts applied across the specified axes and an
        ``alignment_parameters`` attribute describing what was executed.

    Raises
    ------
    ValueError
        If ``input_data`` lacks required dimensions or mismatched hints are provided.
    """
    plane_dims = tuple(image_dims)
    dims_hint = plane_dims if isinstance(input_data, np.ndarray) else None
    data_da = _ensure_dataarray(input_data, dims_hint=dims_hint)

    for dim in plane_dims:
        if dim not in data_da.dims:
            raise ValueError(f"Data is missing required dimension '{dim}'.")

    result = data_da
    original_dim_order = tuple(result.dims)

    flip_dim = alignment.get("flip_dim")
    flip_applied = alignment.get("flip_applied", False)
    if flip_applied and isinstance(flip_dim, str) and flip_dim in result.dims:
        result = result.isel({flip_dim: slice(None, None, -1)})

    shift_mapping = alignment.get("shifts", {})
    shift_vector = tuple(
        (
            float(shift_mapping.get(dim, 0.0))
            if isinstance(shift_mapping, Mapping)
            else 0.0
        )
        for dim in plane_dims
    )

    shift_mode = "wrap" if wrap_edges else mode
    if any(abs(val) > 1e-6 for val in shift_vector):
        result = xr.apply_ufunc(
            _shift_complex_nd,
            result,
            input_core_dims=[list(plane_dims)],
            output_core_dims=[list(plane_dims)],
            kwargs={"shift_amounts": shift_vector, "order": order, "mode": shift_mode},
            vectorize=True,
            dask="parallelized",
        )
        # xarray moves the broadcast dimensions ahead of the core dims inside
        # apply_ufunc(vectorize=True); transpose back so callers see the original layout.
        result = result.transpose(*original_dim_order)

    attrs = dict(result.attrs)
    attrs["alignment_parameters"] = {
        "image_dims": plane_dims,
        "shift_vector": shift_vector,
        "flip_dim": flip_dim if flip_applied else None,
        "ssim": alignment.get("ssim"),
        "shift_mode": shift_mode,
        "wrap_edges": bool(shift_mode == "wrap"),
    }
    result = result.assign_attrs(attrs)
    return result


def align_image_data(
    source: xr.DataArray,
    reference: xr.DataArray,
    *,
    image_dims: Sequence[str] = ("read_idx", "phase_idx"),
    reduce_dims: Union[str, Sequence[str], None] = "auto",
    reduce_func: Union[
        str, Callable[[xr.DataArray, Sequence[str]], xr.DataArray]
    ] = "mean",
    flip_mode: str = "auto",
    flip_dim: Optional[str] = "phase_idx",
    upsample_factor: int = 10,
    shift_order: int = 3,
    order: int = 3,
    mode: str = "constant",
    wrap_edges: bool = False,
    average_repetitions: bool = True,
    repetition_dim_keywords: Sequence[str] = ("rep", "repetition", "average"),
) -> Tuple[xr.DataArray, Dict[str, Union[float, Dict[str, float], bool, xr.DataArray]]]:
    """
    Estimate alignment parameters and immediately apply them to ``source``.

    Parameters
    ----------
    source, reference
        xarray objects describing the moving and fixed datasets.
    image_dims, reduce_dims, reduce_func, flip_mode, flip_dim, upsample_factor, shift_order,
    average_repetitions, repetition_dim_keywords
        Forwarded verbatim to :func:`estimate_image_alignment`.
    order, mode, wrap_edges
        Forwarded to :func:`apply_image_alignment` when applying the detected shift.

    Returns
    -------
    tuple
        ``(aligned_source, alignment_info)`` where ``aligned_source`` is the shifted
        DataArray and ``alignment_info`` matches the structure produced by
        :func:`estimate_image_alignment`.

    """
    alignment = estimate_image_alignment(
        source,
        reference,
        image_dims=image_dims,
        reduce_dims=reduce_dims,
        reduce_func=reduce_func,
        flip_mode=flip_mode,
        flip_dim=flip_dim,
        upsample_factor=upsample_factor,
        shift_order=shift_order,
        average_repetitions=average_repetitions,
        repetition_dim_keywords=repetition_dim_keywords,
    )
    aligned = apply_image_alignment(
        source,
        alignment,
        image_dims=image_dims,
        order=order,
        mode=mode,
        wrap_edges=wrap_edges,
    )
    return aligned, alignment


def plot_alignment_overview(
    source_preview: xr.DataArray,
    reference_preview: xr.DataArray,
    aligned_preview: xr.DataArray,
    alignment_info: Mapping[str, Union[float, bool, Sequence[float]]],
):
    """
    Create a compact diagnostic figure summarizing the alignment result.

    Parameters
    ----------
    source_preview, reference_preview, aligned_preview
        2D magnitude images (typically produced by :func:`estimate_image_alignment`)
        that illustrate the original moving data, the fixed reference, and the
        aligned output.
    alignment_info
        Mapping containing SSIM/shift metadata used for figure annotations. Passing
        the dict returned by :func:`estimate_image_alignment` is recommended.

    Returns
    -------
    matplotlib.figure.Figure
        Handle to the constructed figure for further customization or saving.
    """
    import matplotlib.pyplot as plt  # Local import to avoid mandatory dependency at import time

    source_vals = np.abs(source_preview.values)
    reference_vals = np.abs(reference_preview.values)
    aligned_vals = np.abs(aligned_preview.values)

    vmax = max(
        np.nanmax(source_vals),
        np.nanmax(reference_vals),
        np.nanmax(aligned_vals),
    )

    fig, axes = plt.subplots(3, 3, figsize=(12, 8))

    axes[0, 0].imshow(source_vals, cmap="viridis", vmax=vmax)
    axes[0, 0].set_title("Original Source")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(reference_vals, cmap="viridis", vmax=vmax)
    axes[0, 1].set_title("Reference")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(aligned_vals, cmap="viridis", vmax=vmax)
    ssim_val = alignment_info.get("ssim")
    if isinstance(ssim_val, (float, int)):
        axes[0, 2].set_title(f"Aligned (SSIM={ssim_val:.3f})")
    else:
        axes[0, 2].set_title("Aligned")
    axes[0, 2].axis("off")

    overlay_before = np.zeros((*source_vals.shape, 3))
    overlay_before[..., 0] = reference_vals / vmax
    overlay_before[..., 1] = source_vals / vmax
    axes[1, 0].imshow(overlay_before)
    axes[1, 0].set_title("Overlay Before\n(Red=Reference, Green=Source)")
    axes[1, 0].axis("off")

    overlay_after = np.zeros((*source_vals.shape, 3))
    overlay_after[..., 0] = reference_vals / vmax
    overlay_after[..., 1] = aligned_vals / vmax
    axes[1, 1].imshow(overlay_after)
    axes[1, 1].set_title("Overlay After\n(Red=Reference, Green=Aligned)")
    axes[1, 1].axis("off")

    diff = np.abs(reference_vals - aligned_vals)
    axes[1, 2].imshow(diff, cmap="hot")
    axes[1, 2].set_title("Absolute Difference")
    axes[1, 2].axis("off")

    mid = source_vals.shape[0] // 2
    axes[2, 0].plot(np.abs(source_vals[mid, :]))
    axes[2, 0].set_title("Source Profile")

    axes[2, 1].plot(np.abs(reference_vals[mid, :]))
    axes[2, 1].set_title("Reference Profile")

    axes[2, 2].plot(np.abs(aligned_vals[mid, :]), label="Aligned")
    axes[2, 2].plot(np.abs(reference_vals[mid, :]), label="Reference")
    axes[2, 2].legend()
    axes[2, 2].set_title("Profile Comparison")

    shift_vector = alignment_info.get("shift_vector")
    axes[2, 2].set_xlabel(f"Shift: {shift_vector}")

    plt.suptitle(
        f"Alignment preview (flip={alignment_info.get('flip_applied')}, shift={shift_vector})"
    )
    plt.tight_layout()
    return fig
