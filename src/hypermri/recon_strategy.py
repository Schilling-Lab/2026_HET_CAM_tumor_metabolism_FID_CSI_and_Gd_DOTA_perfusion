# hypermri/recon_strategy.py

import time
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from hypermri.compatibility import assign_frequency_axis
from hypermri.reconstruction.aliases import resolve_sequence_alias
from hypermri.utils.coordinates.grids import get_xyz_grid
from hypermri.utils.coordinates.image_shifts import (
    get_orientation_flip_multipliers,
    shift_image,
)
from hypermri.utils.system.logging import debug_log, init_default_logger
from hypermri.utils.time.temporal import (
    calc_inversion_time_multi_slice,
)

# initialize logger
logger = init_default_logger(__name__)

DEBUG_FILE = "/tmp/recon_strategy_debug.log"
DEBUG_LOG = True


def debug_log_recon_strategy(message="", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG):
    """Debug logging system for brukerexp"""
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
    )


def _create_inversion_idx_coordinate(
    handler, n_inversions_final, n_slices_final, shape_dict
):
    """Create slice-dependent inversion time coordinate for FAIR_RARE sequences.

    In multi-slice FAIR_RARE acquisitions, different slices are acquired at different
    time points after the inversion pulse, resulting in different effective inversion times
    for each slice. This function calculates the actual inversion time for each
    (slice, inversion_idx) combination.

    Parameters
    ----------
    handler : VendorHandler
        Data handler object containing sequence parameters (slice_order, inversion_time_ms, etc.)
    n_inversions_final : int
        Number of inversion time points in the acquisition
    n_slices_final : int
        Number of slices in the acquisition
    shape_dict : dict
        Dictionary mapping dimension names to their sizes

    Returns
    -------
    tuple
        Coordinate tuple in format: (("slice_idx", "inversion_idx"), effective_ti_ms, {"unit": "ms"})
        Returns simple 1D coordinate if slice-dependent calculation fails or is not needed
    """
    try:
        # Get base inversion times
        handler.inversion_time_ms
        base_inversion_time_ms = np.atleast_1d(handler.inversion_time_ms)

        # Calculate per-slice time increment
        slice_TI_increment_ms = calc_inversion_time_multi_slice(data_obj=handler)

        # If increment is zero or negligible, return simple 1D coordinate
        if slice_TI_increment_ms == 0 or abs(slice_TI_increment_ms) < 1e-6:
            logger.debug(
                "Slice TI increment is zero, using simple 1D inversion coordinate"
            )
            ti_coords = (
                base_inversion_time_ms[:n_inversions_final]
                if len(base_inversion_time_ms) >= n_inversions_final
                else np.arange(n_inversions_final)
            )
            return ("inversion_idx", ti_coords, {"unit": "ms"})

        # Get and validate slice order
        slice_order = handler.slice_order
        if slice_order is None or len(slice_order) != n_slices_final:
            logger.warning(
                f"slice_order length ({len(slice_order) if slice_order is not None else 'None'}) "
                f"!= number of slices ({n_slices_final}). Using simple 1D inversion coordinate."
            )
            ti_coords = (
                base_inversion_time_ms[:n_inversions_final]
                if len(base_inversion_time_ms) >= n_inversions_final
                else np.arange(n_inversions_final)
            )
            return ("inversion_idx", ti_coords, {"unit": "ms"})

        if not np.all(np.sort(slice_order) == np.arange(n_slices_final)):
            logger.warning(
                "slice_order does not contain unique indices from 0 to N_slices-1. "
                "Using simple 1D inversion coordinate."
            )
            ti_coords = (
                base_inversion_time_ms[:n_inversions_final]
                if len(base_inversion_time_ms) >= n_inversions_final
                else np.arange(n_inversions_final)
            )
            return ("inversion_idx", ti_coords, {"unit": "ms"})

        # Calculate acquisition rank for each physical slice index
        # Map acquisition order back onto the physical slice index: slice_order[i]
        # stores which physical slice was acquired at position i.
        slice_order = np.asarray(slice_order, dtype=int)
        acquisition_rank = np.empty_like(slice_order)
        acquisition_rank[slice_order] = np.arange(n_slices_final)

        logger.info(
            f"[DIAGNOSTIC] Acquisition rank calculation:\n"
            f"  - slice_order: {slice_order}\n"
            f"  - acquisition_rank: {acquisition_rank}\n"
            f"  - slice_TI_increment: {slice_TI_increment_ms:.2f} ms"
        )

        # Calculate offset for each slice based on its rank
        slice_offsets_ms = acquisition_rank * slice_TI_increment_ms

        # Create the 2D effective TI array using broadcasting
        # Ensure base_inversion_time_ms covers all expected inversions
        if len(base_inversion_time_ms) < n_inversions_final:
            logger.warning(
                f"Base inversion times array length ({len(base_inversion_time_ms)}) "
                f"is less than expected inversions ({n_inversions_final}). Padding with zeros."
            )
            padded_tis = np.pad(
                base_inversion_time_ms,
                (0, n_inversions_final - len(base_inversion_time_ms)),
                mode="constant",
                constant_values=0,
            )
        else:
            padded_tis = base_inversion_time_ms[:n_inversions_final]

        # Broadcast: slice_offsets_ms (n_slices, 1) + padded_tis (1, n_inversions) -> (n_slices, n_inversions)
        effective_ti_ms = slice_offsets_ms[:, np.newaxis] + padded_tis[np.newaxis, :]

        logger.info(
            f"[DIAGNOSTIC] Created slice-dependent inversion coordinate:\n"
            f"  - Coordinate shape: {effective_ti_ms.shape} (slice_idx, inversion_idx)\n"
            f"  - n_slices: {n_slices_final}, n_inversions: {n_inversions_final}\n"
            f"  - Slice TI increment: {slice_TI_increment_ms:.2f} ms\n"
            f"  - Slice order: {slice_order}\n"
            f"  - Base inversion times (first 3): {base_inversion_time_ms[:3]} ms\n"
            f"  - Slice 0 TI values: {effective_ti_ms[0, :3]}... ms\n"
            f"  - Slice 1 TI values: {effective_ti_ms[1, :3] if n_slices_final > 1 else 'N/A'}... ms"
        )

        # Return as multidimensional coordinate
        return (("slice_idx", "inversion_idx"), effective_ti_ms, {"unit": "ms"})

    except Exception as e:
        logger.warning(
            f"Failed to create slice-dependent inversion coordinate: {e}. "
            "Falling back to simple 1D coordinate."
        )
        # Fallback to simple 1D coordinate
        try:
            ti_coords = (
                base_inversion_time_ms[:n_inversions_final]
                if len(base_inversion_time_ms) >= n_inversions_final
                else np.arange(n_inversions_final)
            )
        except:
            ti_coords = np.arange(n_inversions_final)
        return ("inversion_idx", ti_coords, {"unit": "ms"})


class BaseRecoStrategy(ABC):
    """Abstract base class for a reconstruction strategy."""

    def __init__(self, *args, **kwargs):
        """Initialize the reconstruction strategy."""
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def reconstruct(self, handler, **kwargs) -> xr.DataArray:
        """
        Perform the reconstruction.

        Parameters
        ----------
        input_data: np.ndarray
            The raw k-space data from the vendor handler.
        reco_options: dict
            A dictionary of parameters from the vendor handler.

        Returns
        -------
        A reconstructed xarray.DataArray.
        """

    def _parse_data_precision(self, data_precision_value):
        """
        Convert string or numpy type to numpy data type.

        Parameters
        ----------
        data_precision_value: str or numpy type
            The data precision specification.

        Returns
        -------
        numpy type
            The corresponding numpy data type.

        """
        if isinstance(data_precision_value, str):
            precision_map = {
                "float16": np.float16,
                "float32": np.float32,
                "float64": np.float64,
            }
            return precision_map.get(data_precision_value.lower(), np.float16)
        else:
            return data_precision_value

    def _convert_array_precision(self, array, precision):
        """
        Convert array to specified precision, handling complex data appropriately.

        Parameters
        ----------
        array: numpy array
            The array to convert.
        precision: numpy type
            The target precision type.

        Returns
        -------
        numpy array
            The array with converted precision.

        """
        if array is None or array.size == 0:
            return array

        # Check if array is complex
        try:
            is_complex = np.any(np.iscomplex(array))
        except (TypeError, ValueError):
            is_complex = False

        if is_complex:
            # For complex data, convert to appropriate complex type
            if precision == np.float16:
                # complex64 uses float32 for real/imag parts (smallest complex type)
                return array.astype(np.complex64)
            elif precision == np.float32:
                return array.astype(np.complex64)
            elif precision == np.float64:
                return array.astype(np.complex128)
        else:
            # For real data, convert directly
            return array.astype(precision)

        return array

    def _data_precision(self, handler, DATA_PRECISION):
        """
        Set the data precision for the handler's fid.

        Parameters
        ----------
        handler: VendorHandler
            The vendor handler containing the fid data.
        DATA_PRECISION: type
            The desired numpy data type for the fid (e.g., np.float16, np.float32, np.float64).

        Returns
        -------
        handler: VendorHandler
            The vendor handler with the fid converted to the specified precision.

        """

        def convert_array_precision(array, precision):
            """Helper function to convert array to specified precision."""
            # Return None if input is None
            if array is None:
                logger.debug(
                    f"convert_array_precision: returning None as input is None"
                )
                return None

            if array is not None:
                logger.debug(
                    f"convert_array_precision: array.shape: {getattr( array, 'shape', 'no shape')}"
                )
                logger.debug(
                    f"convert_array_precision: array.dtype: {getattr( array, 'dtype', 'no dtype')}"
                )

            # Check for problematic values - handle different data types safely
            try:
                has_nan = np.any(np.isnan(array))
                has_inf = np.any(np.isinf(array))
                if has_nan or has_inf:
                    logger.critical(f"Warning: Array contains NaN or inf values")
                    # Optionally clean the data:
                    # array = np.nan_to_num(array, nan=0.0, posinf=0.0,
                    # neginf=0.0)
            except (TypeError, ValueError) as e:
                # Some data types don't support isnan/isinf operations
                logger.debug(
                    f"Cannot check for NaN/inf on array dtype {getattr(array, 'dtype', 'no dtype')}: {e}"
                )

            # Check if array is complex (safely handle None)
            try:
                is_complex = np.any(np.iscomplex(array))
            except (TypeError, ValueError):
                is_complex = False

            if is_complex:
                # Check the range of your data safely
                try:
                    max_val = np.max(np.abs(array))
                    if precision == np.float16:
                        # complex64 uses float32 for real/imag parts
                        if max_val > np.finfo(np.float32).max:
                            logger.warning(
                                f"Warning: Values too large for complex64, max={max_val}"
                            )
                        return array.astype(np.complex64)
                except (TypeError, ValueError) as e:
                    logger.debug(
                        f"Cannot calculate max for array dtype {getattr( array, 'dtype', 'no dtype')}: {e}"
                    )
                    # Return original array if conversion fails
                    return array
            else:
                # Handle non-complex arrays
                if array is None:
                    return None
                if precision == np.float16:
                    return array.astype(np.float16)
                elif precision == np.float32:
                    return array.astype(np.float32)
                elif precision == np.float64:
                    return array.astype(np.float64)

            return array

        if DATA_PRECISION not in [np.float16, np.float32, np.float64]:
            raise ValueError(
                f"Unsupported data precision: {DATA_PRECISION}. Supported types are np.float16, np.float32 and np.float64."
            )

        if hasattr(handler, "fid"):
            logger.debug(f"_data_precision: handler.fid type: {type( handler.fid)}")
            logger.debug(f"_data_precision: handler.fid is None: {handler.fid is None}")
            if handler.fid is not None:
                logger.debug(
                    f"_data_precision: handler.fid.shape: {getattr( handler.fid, 'shape', 'no shape')}"
                )
            handler.fid = convert_array_precision(handler.fid, DATA_PRECISION)
            logger.debug(f"_data_precision: convert_array_precision completed for fid")
        if hasattr(handler, "fid_pp"):
            logger.debug(f"_data_precision: converting fid_pp")
            handler.fid_pp = convert_array_precision(handler.fid_pp, DATA_PRECISION)
        if hasattr(handler, "seq2d"):
            logger.debug(f"_data_precision: converting seq2d")
            handler.seq2d = convert_array_precision(handler.seq2d, DATA_PRECISION)
        logger.debug(f"_data_precision: returning handler")
        return handler

    def _append_parameters(self, array, handler, **kwargs):
        """
        Append parameters to the xarray DataArray.

        Parameters
        ----------
        array: xr.DataArray
            The xarray DataArray to which parameters will be appended.
        handler:
            The vendor handler containing the parameters.
        kwargs:
            Additional keyword arguments to append.

        """
        array.attrs["sampling_time_axis_s"] = handler.get("sampling_time_axis_s", None)
        assign_frequency_axis(
            array.attrs, getattr(handler, "freq_axis_ppm", None), "ppm"
        )
        assign_frequency_axis(array.attrs, getattr(handler, "freq_axis_hz", None), "Hz")
        array.attrs["is_spectroscopic"] = handler.is_spectroscopic
        return array


# ---------------------------------------------------------------
# ----------------       BRUKER RECO     ------------------------
# ---------------------------------------------------------------


# -----------       BRUKER RECO BASE CLASSES     ----------------
class BrukerRawRecoStrategy(BaseRecoStrategy):
    """
    Reconstruction strategy for Bruker raw data (fid).

    Parameters
    ----------
    handler: VendorHandler
        The vendor handler containing the raw data and parameters.
    reco_options: dict
        A dictionary of reconstruction options. Possible keys include:
            - no_reconstruction: bool
                If True, no reconstruction is performed.
            - postprocess_fid: bool
                If True, postprocess the fid data before reconstruction.
            - reshape_fid: bool
                If True, reshape the fid data into k-space.
            - set_kspace_xarray: bool
                If True, create an xarray DataArray for k-space data.
            - shift_xarray: bool
                If True, apply phase shifts to the k-space data.
            - set_reco_data: bool
                If True, create an xarray DataArray for spatial domain data.
            - db: bool
                If True, enable debug logging.
            - data_precision: type
                The desired numpy data type for the fid (e.g., np.float16, np.float32, np.float64).

    Returns
    -------
    xr.DataArray
        The reconstructed xarray DataArray.

    Notes
    -----
    reco_options are defined in the reconstruction/config/strategy_mapping.yaml file.

    """

    def reconstruct(self, handler, **reco_options):
        start_time = time.time()
        no_reconstruction = reco_options.get("no_reconstruction", False)
        if no_reconstruction:
            logger.info(
                "[BrukerRawRecoStrategy] No reconstruction performed as per settings."
            )
            return None

        postprocess_fid = reco_options.get("postprocess_fid", True)
        reshape_fid = reco_options.get("reshape_fid", True)
        set_kspace_xarray = reco_options.get("set_kspace_xarray", True)
        shift_xarray = reco_options.get("shift_xarray", True)
        set_reco_data = reco_options.get("set_reco_data", True)
        db = reco_options.get("db", True)
        DATA_PRECISION = self._parse_data_precision(
            reco_options.get("data_precision", np.float16)
        )

        # Ensure the handler is set to the correct data precision
        handler = self._data_precision(handler=handler, DATA_PRECISION=DATA_PRECISION)

        # Reconstruct the raw data into an xarray DataArray
        reco_array = self.reco_rawdata(
            handler=handler,
            postprocess_fid=postprocess_fid,
            reshape_fid=reshape_fid,
            set_kspace_xarray=set_kspace_xarray,
            shift_xarray=shift_xarray,
            set_reco_data=set_reco_data,
            db=db,
            data_precision=DATA_PRECISION,
        )
        logger.info(
            "[BrukerRawRecoStrategy] Reconstruction completed., took {:.3f}s".format(
                time.time() - start_time
            )
        )
        return reco_array

    def reco_rawdata(
        self,
        handler=None,
        postprocess_fid=True,
        reshape_fid=True,
        set_kspace_xarray=True,
        shift_xarray=True,
        set_reco_data=True,
        db=True,
        data_precision=np.float16,
    ):
        """
        Reconstruct raw fid data into k-space and spatial domain xarray DataArrays.

        Parameters
        ----------
        handler: VendorHandler
            The vendor handler containing the raw data and parameters.
        postprocess_fid: bool
            If True, postprocess the fid data before reconstruction.
        reshape_fid: bool
            If True, reshape the fid data into k-space.
        set_kspace_xarray: bool
            If True, create an xarray DataArray for k-space data.
        shift_xarray: bool
            If True, apply phase shifts to the k-space data to apply image shifts in phase encoding directions.
        set_reco_data: bool
            If True, create an xarray DataArray for spatial domain data.
        db: bool
            If True, enable debug logging.
        data_precision: type
            The desired numpy data type for the fid (e.g., np.float16, np.float32, np.float64).

        Returns
        -------
        xr.DataArray
            The reconstructed xarray DataArray.


        """

        reco_data = None
        if hasattr(handler, "fid") and postprocess_fid is True:
            fid = handler.fid
        elif hasattr(handler, "fid_pp") and postprocess_fid is False:
            fid = handler.fid_pp
        else:
            fid = getattr(handler, "fid", None)

        fid_pp = fid

        if fid_pp is None:
            raise ValueError("No FID data available for reconstruction")

        if postprocess_fid is True:
            try:
                fid_pp = self.postprocess_fid_file(input_data=fid, handler=handler)
            except Exception as e:
                full_traceback = traceback.format_exc()
                err_msg, err_type = str(e), e.__class__.__name__

                msg = f"{handler.ExpNum}: base_anatomy: reco_rawdata()\nThe following error occured during "
                msg += f"postprocess_fid_file:\n    {err_type}: {err_msg}\n"
                if db:
                    msg += f"Full traceback:\n{full_traceback}"
                logger.debug(msg)

        if reshape_fid is True:
            # We explicitly want this to raise an exception if it fails,
            # so we can see the root cause (e.g. dimension mismatch)
            # instead of failing later with "AttributeError: no attribute 'kspace'"
            handler.kspace = self.reshape_fid_to_kspace(
                input_data=fid_pp, handler=handler
            )
        else:
            if not hasattr(handler, "kspace"):
                handler.kspace = fid_pp

        # generate kspace_xarray:
        if set_kspace_xarray is True:
            if not hasattr(handler, "kspace"):
                raise ValueError(
                    f"{handler.ExpNum}: base_anatomy: reco_rawdata() - "
                    "Cannot set kspace xarray: 'kspace' data is missing. "
                    "This likely means reshape_fid_to_kspace failed or was skipped."
                )

            try:
                # Check for FAIR_RARE to determine if inversion_idx dimension
                # is needed
                n_inversion_time = getattr(handler, "n_inversion_time", 1)
                is_fair_rare = n_inversion_time > 1

                if is_fair_rare:
                    dims = [
                        "acquisition_idx",
                        "repetition_idx",
                        "slice_idx",
                        "phase_idx",
                        "inversion_idx",  # NEW for FAIR_RARE
                        "channel_idx",
                        "read_idx",
                    ]
                else:
                    dims = [
                        "acquisition_idx",
                        "repetition_idx",
                        "slice_idx",
                        "phase_idx",
                        "channel_idx",
                        "read_idx",
                    ]
                coords = {
                    "acquisition_idx": (
                        "acquisition_idx",  # Add dimension label as first element of tuple
                        handler.sampling_time_axis_s,
                        {"unit": "s"},  # Add unit in attrs dictionary
                    ),
                    "repetition_idx": (
                        "repetition_idx",
                        np.array(np.arange(handler.kspace.shape[1])),
                        {"unit": "index"},
                        # repetition_idx is count of repetition
                    ),
                    "slice_idx": (
                        ["read_idx", "phase_idx", "slice_idx"],
                        handler.rps_grid_mm[2],
                        {"unit": "mm"},
                    ),
                    "phase_idx": (
                        ["read_idx", "phase_idx", "slice_idx"],
                        handler.rps_grid_mm[1],
                        {"unit": "mm"},
                    ),
                    "channel_idx": (
                        "channel_idx",
                        np.array(np.arange(handler.n_receivers)),
                        {"unit": "index"},  # channel_idx is count of receiver
                    ),
                    "read_idx": (
                        ["read_idx", "phase_idx", "slice_idx"],
                        handler.rps_grid_mm[0],
                        {"unit": "mm"},
                    ),
                }

                # Add inversion_idx coordinate for FAIR_RARE
                if is_fair_rare:
                    # Get number of slices from kspace shape
                    n_slices = handler.kspace.shape[
                        2
                    ]  # slice_idx is 3rd dimension in kspace

                    # Use helper function to create slice-dependent inversion coordinate
                    coords["inversion_idx"] = _create_inversion_idx_coordinate(
                        handler,
                        n_inversion_time,
                        n_slices,
                        {"slice_idx": n_slices, "inversion_idx": n_inversion_time},
                    )

                handler.kspace_xarray = self.set_kspace_xarray(
                    handler, data=handler.kspace, dims=dims, coords=coords
                )
            except Exception as e:
                full_traceback = traceback.format_exc()
                err_msg, err_type = str(e), e.__class__.__name__

                msg = f"{handler.ExpNum}: base_anatomy: reco_rawdata()\nThe following error occured during "
                msg += f"set_kspace_xarray:\n    {err_type}: {err_msg}\n"
                if db:
                    msg += f"Full traceback:\n{full_traceback}"
                raise ValueError(msg)

        if shift_xarray is True:
            try:
                # shift images by applying phase ramp to the k-space data
                handler.kspace_xarray = shift_image(
                    input_data=handler.kspace_xarray,
                    shift_method="phase",
                    data_obj=handler,
                    domain="kspace",
                )
            except Exception as e:
                if db:
                    full_traceback = traceback.format_exc()
                    err_msg, err_type = str(e), e.__class__.__name__

                    msg = f"{handler.ExpNum}: base_anatomy: reco_rawdata()\nThe following error occured during "
                    msg += f"shift_image:\n    {err_type}: {err_msg}\n"
                    if db:
                        msg += f"Full traceback:\n{full_traceback}"
                    raise ValueError(msg)

        domain = "kspace"
        if set_reco_data is True:
            logger.debug(
                "set_reco_data is True, proceeding with spatial domain reconstruction"
            )
            # Check for FAIR_RARE to determine the correct dimension order
            n_inversion_time = getattr(handler, "n_inversion_time", 1)
            is_fair_rare = n_inversion_time > 1

            if is_fair_rare:
                new_order = (
                    "acquisition_idx",
                    "read_idx",
                    "phase_idx",
                    "slice_idx",
                    "repetition_idx",
                    "inversion_idx",  # Include inversion_idx for FAIR_RARE
                    "channel_idx",
                )
            else:
                new_order = (
                    "acquisition_idx",
                    "read_idx",
                    "phase_idx",
                    "slice_idx",
                    "repetition_idx",
                    "channel_idx",
                )

            try:
                handler.kspace_xarray = handler.kspace_xarray.transpose(*new_order)
                # get read and phase encode axis
                # read_axis = self.kspace_xarray.get_axis_num("read_idx")
                # phase_axis = self.kspace_xarray.get_axis_num("phase_idx")

                # flip along read and phase axis. This seems arbitrary but images seem to be aligned with seq2ds
                # self.kspace_xarray.data = np.flip(
                #     self.kspace_xarray.data, axis=phase_axis
                # )
                # self.kspace_xarray.data = np.flip(
                #     self.kspace_xarray.data, axis=read_axis
                # )

                if handler.spat_dim == "2D":
                    axes_names = ["read_idx", "phase_idx"]
                elif handler.spat_dim == "3D":
                    axes_names = ["read_idx", "phase_idx", "slice_idx"]
                else:
                    axes_names = []
                # set reco_data (will be written to handler.spatom_xarray):
                reco_data = self.set_reco_data(
                    kspace_xarray=handler.kspace_xarray,
                    axes_names=axes_names,
                    handler=handler,
                    write_to_object=False,
                    ift=False,
                )

                # Apply data precision conversion to the reconstructed data
                reco_data.data = self._convert_array_precision(
                    reco_data.data, data_precision
                )

                domain = "spatial"
            except Exception as e:
                full_traceback = traceback.format_exc()
                err_msg, err_type = str(e), e.__class__.__name__

                msg = f"{handler.ExpNum}: base_anatomy: reco_rawdata()\nThe following error occured during "
                msg += f"set_reco_data:\n    {err_type}: {err_msg}\n"
                if db:
                    msg += f"Full traceback:\n{full_traceback}"

                raise ValueError(msg)

        reco_data.attrs["domain"] = domain

        # Add time coordinate if frame_time_s is available
        if (
            "repetition_idx" in reco_data.dims
            and hasattr(handler, "frame_time_s")
            and handler.frame_time_s is not None
        ):
            try:
                rep_size = reco_data.sizes["repetition_idx"]
                if len(handler.frame_time_s) >= rep_size:
                    time_coords = handler.frame_time_s[:rep_size]
                    logger.debug(
                        f"[BrukerRawRecoStrategy] Adding 'time' coordinate from frame_time_s: "
                        f"shape={time_coords.shape}, range=[{time_coords[0]:.3f}, {time_coords[-1]:.3f}] s"
                    )
                    reco_data = reco_data.assign_coords(
                        time=("repetition_idx", time_coords)
                    )
                    reco_data["time"].attrs["unit"] = "s"
                    reco_data["time"].attrs["long_name"] = "Frame acquisition time"
                    reco_data["time"].attrs[
                        "description"
                    ] = "Time in seconds for each repetition/frame"
                else:
                    logger.warning(
                        f"[BrukerRawRecoStrategy] frame_time_s length ({len(handler.frame_time_s)}) < repetition_idx size ({rep_size}), "
                        "skipping 'time' coordinate creation"
                    )
            except Exception as e:
                logger.warning(
                    f"[BrukerRawRecoStrategy] Failed to add 'time' coordinate from frame_time_s: {e}"
                )

        return reco_data

    def ft(self, xarray, axes_names=["read_idx", "phase_idx"], ift=False):
        """
        Quickly computes the fftshift(fft(fftshift(input_data))) along the input axis.

        Parameters
        ----------
        xarray : (complex) xarray
            Data that should be Fourier-transformed.

        axes_names : list of strings
            Axes names along the Fourier transform should be performed.

        ift : boolean
            Use inverse Fourier transform and inverese fftshifts.

        """
        axes_names = np.atleast_1d(axes_names)
        axes_idx = [xarray.get_axis_num(axis_name) for axis_name in axes_names]

        ft_array = xarray.to_numpy().copy()

        try:
            if ift:
                for d in range(len(axes_idx)):
                    ft_array = np.fft.ifftshift(
                        np.fft.ifft(
                            np.fft.ifftshift(
                                ft_array,
                                axes=(axes_idx[d],),
                            ),
                            axis=axes_idx[d],
                        ),
                        axes=(axes_idx[d],),
                    )
                    logger.debug("ift along " + str(axes_idx[d]))

            else:
                for d in range(len(axes_idx)):
                    ft_array = np.fft.fftshift(
                        np.fft.fft(
                            np.fft.fftshift(
                                ft_array,
                                axes=(axes_idx[d],),
                            ),
                            axis=axes_idx[d],
                        ),
                        axes=(axes_idx[d],),
                    )
        except Exception as e:
            raise ValueError(
                f"Error during Fourier transform: {e}. Check the axes names and data shape."
            ) from e

        result = xr.DataArray(
            ft_array,
            dims=xarray.dims,
            coords=xarray.coords,
            attrs=dict(xarray.attrs),
            name=xarray.name,
        )

        # Log dimension order for FAIR_RARE data
        if "inversion_idx" in result.dims and "slice_idx" in result.dims:
            logger.info(
                f"[DIAGNOSTIC] xarray data dimension order after FFT:\n"
                f"  - Dimensions: {result.dims}\n"
                f"  - Shape: {result.shape}\n"
                f"  - inversion_idx position: {result.dims.index('inversion_idx')}\n"
                f"  - slice_idx position: {result.dims.index('slice_idx')}"
            )

        return result

    def set_kspace_xarray(
        self,
        handler,
        data: np.ndarray = None,
        dims: list = None,
        coords: dict = None,
        write_to_object: bool = False,
    ):
        """
        Initialize or updates the kspace_xarray with given data, dimensions, and coordinates.

        This method creates a new xarray DataArray if one does not already exist, or updates
        the existing one if present. It allows for flexible handling of k-space data
        with various shapes and dimensions.

        Parameters
        ----------
        handler : VendorHandler
            The vendor handler containing necessary attributes for coordinate assignment.
        data : np.ndarray
            The k-space data to be stored in the xarray.
        dims : list of str
            List of dimension names.
        coords : dict, optional
            Dictionary of coordinates for each dimension. If None, default coordinates
            will be created using np.arange() for each dimension based on the shape of the data.
        write_to_object : bool, optional
            If True (default), the generated kspace xarray is written to self(),
            otherwise the generated kspace xarray is returned.

        Raises
        ------
        ValueError
            If the number of dimensions does not match the data shape.

        Returns
        -------
        An xarray containing the k-space data, dimensions, and coordinates.

        Notes
        -----
        - If coords is not specified, it will be automatically generated using np.arange() based on the shape of the data.
        - This method can be used for both initial creation and subsequent updates of the xarray.

        """
        logger.debug(f"set_kspace_xarray called with data shape: {data.shape}")
        # Validate dimensions
        if len(data.shape) != len(dims):
            raise ValueError(
                f"Number of dimensions ({len(dims)}) must match the data shape ({len(data.shape)})"
            )

        # Create default coordinates if not provided
        if coords is None:
            coords = {dim: np.arange(size) for dim, size in zip(dims, data.shape)}
        # Create or update xarray DataArray
        try:
            kspace_xarray = xr.DataArray(data, dims=dims, coords=coords)
        except BaseException:
            coord_info = {name: np.array(coords[name][1]).shape for name in coords}
            raise ValueError(
                f"""\nCould not create xarray DataArray, shape of: \nData: {
                    data.shape},
                \nDimensions: {dims}, \nCoordinates: {coord_info}\n
                """
            )
        # Retrieve x, y, z coordinate data arrays
        try:
            x_grid_mm = handler.xyz_grid_mm[0]
            y_grid_mm = handler.xyz_grid_mm[1]
            z_grid_mm = handler.xyz_grid_mm[2]
        except AttributeError:
            raise ValueError(
                "Attribute 'handler.xyz_grid_mm' not found or not initialized."
            )
        except IndexError:
            raise ValueError(
                "'handler.xyz_grid_mm' must contain at least three elements (x, y, z grids)."
            )

        spatial_dims_for_coords = ["read_idx", "phase_idx", "slice_idx"]

        # Check if this is spectroscopic data and handle coordinates appropriately
        is_spectroscopic = getattr(handler, "is_spectroscopic", False)

        # For spectroscopic data, check if spatial dimensions match coordinate grids
        spatial_dims_sizes = [
            kspace_xarray.sizes[dim] for dim in spatial_dims_for_coords
        ]
        coord_grid_shapes = [x_grid_mm.shape, y_grid_mm.shape, z_grid_mm.shape]

        # Check if coordinate shapes match the spatial dimensions
        coords_compatible = all(
            coord_shape == tuple(spatial_dims_sizes)
            for coord_shape in coord_grid_shapes
        )

        # Assign new coordinates only if compatible or create appropriate coordinates for spectroscopy
        try:
            if is_spectroscopic and not coords_compatible:
                # For spectroscopy data with mismatched coordinates, create single-point grids
                # Use the first value from each coordinate grid for spectroscopy
                x_spectro = np.array([[[x_grid_mm.flat[0]]]])  # Shape (1, 1, 1)
                y_spectro = np.array([[[y_grid_mm.flat[0]]]])  # Shape (1, 1, 1)
                z_spectro = np.array([[[z_grid_mm.flat[0]]]])  # Shape (1, 1, 1)

                kspace_xarray = kspace_xarray.assign_coords(
                    {
                        "x": (spatial_dims_for_coords, x_spectro),
                        "y": (spatial_dims_for_coords, y_spectro),
                        "z": (spatial_dims_for_coords, z_spectro),
                    }
                )
            else:
                # Standard coordinate assignment for imaging data
                kspace_xarray = kspace_xarray.assign_coords(
                    {
                        "x": (spatial_dims_for_coords, x_grid_mm),
                        "y": (spatial_dims_for_coords, y_grid_mm),
                        "z": (spatial_dims_for_coords, z_grid_mm),
                    }
                )

        except ValueError as e:
            print(
                f"Exp: {handler.ExpNum}: Error assigning x, y, z - coordinates with assign_coords: {e}"
            )
            print("\n--- Debugging Information ---")
            print(f"Attempting to assign coordinates with names: 'x', 'y', 'z'")
            print(
                f"Expected dimensions for these coordinates: {spatial_dims_for_coords}"
            )
            print(f"Shape of provided x_grid_mm: {x_grid_mm.shape}")
            print(f"Shape of provided y_grid_mm: {y_grid_mm.shape}")
            print(f"Shape of provided z_grid_mm: {z_grid_mm.shape}")
            print("Sizes of corresponding dimensions in the existing kspace_xarray:")
            for dim_name in spatial_dims_for_coords:
                try:
                    print(
                        f"  - kspace_xarray.sizes['{dim_name}']: {kspace_xarray.sizes[dim_name]}"
                    )
                except KeyError:
                    print(
                        f"  - Dimension '{dim_name}' not found in kspace_xarray.sizes!"
                    )
            print("--- End of Debugging Information ---")
            raise ValueError(
                f"Could not assign x,y,z-coordinates due to shape or dimension mismatch. See details above."
            ) from e
        except Exception as e:
            print(f"An unexpected error occurred during assign_coords: {e}")

        try:
            logger.debug(f"trying to set acquisition_time_s coordinate")
            if "acquisition_idx" in dims:
                kspace_xarray = kspace_xarray.assign_coords(
                    {
                        "acquisition_time_s": (
                            "acquisition_idx",
                            handler.sampling_time_axis_s,
                            {"unit": "s"},
                        )
                    }
                )
        except BaseException:
            pass

        freq_axis_ppm = handler.get("freq_axis_ppm", None)
        freq_axis_hz = handler.get("freq_axis_hz", None)
        if freq_axis_ppm is not None and len(freq_axis_ppm) == kspace_xarray.sizes.get(
            "acquisition_idx", 0
        ):
            kspace_xarray = kspace_xarray.assign_coords(
                {
                    "freq_axis_ppm": (
                        "acquisition_idx",
                        freq_axis_ppm,
                        {"unit": "ppm"},
                    )
                }
            )
        if freq_axis_hz is not None and len(freq_axis_hz) == kspace_xarray.sizes.get(
            "acquisition_idx", 0
        ):
            kspace_xarray = kspace_xarray.assign_coords(
                {
                    "freq_axis_hz": (
                        "acquisition_idx",
                        freq_axis_hz,
                        {"unit": "Hz"},
                    )
                }
            )

        try:
            if handler.n_slices > 1:
                slice_order = np.argsort(handler.slice_order)
                original_coords = kspace_xarray.coords
                slice_dim_index = kspace_xarray.dims.index("slice_idx")

                # Create a slicing object to reorder along the slice dimension
                slices = [slice(None)] * kspace_xarray.ndim
                slices[slice_dim_index] = slice_order

                reordered_data = kspace_xarray.values[tuple(slices)]

                # Create a new xarray with the reordered data and original
                # coordinates
                kspace_xarray = xr.DataArray(
                    reordered_data,
                    coords=original_coords,
                    dims=kspace_xarray.dims,
                )

        except BaseException:
            raise ValueError(
                f"\n Exp: {handler.ExpNum}: Could not set slice order {handler.slice_order} for xarray DataArray, shape data: {data.shape}, dims: {dims}, coords: {coords}\n"
            )

        kspace_xarray.attrs["patient_pos"] = handler.patient_pos
        kspace_xarray.attrs["slice_orientation"] = handler.slice_orientation
        kspace_xarray.attrs["read_orientation"] = handler.read_orientation
        kspace_xarray.attrs["num_image_fov_shift"] = 0
        kspace_xarray.attrs["signal_domain"] = "temporal"  # or "spectral"
        kspace_xarray.attrs["domain"] = "kspace"
        kspace_xarray.attrs["read_direction"] = handler.acquisition_matrix[0]
        kspace_xarray.attrs["phase_direction"] = handler.acquisition_matrix[1]
        kspace_xarray.attrs["slice_direction"] = handler.acquisition_matrix[2]
        kspace_xarray.attrs["slice_thickness_mm"] = handler.slice_thickness_mm
        kspace_xarray.attrs["n_slices"] = handler.n_slices
        kspace_xarray.attrs["git_commit_hash"] = handler.git_commit_hash
        kspace_xarray.attrs["spat_dim"] = handler.spat_dim

        # those cases need to be tested because purley spectroscopy data does
        # not have those attributes
        try:
            kspace_xarray["read_idx"].attrs["matrix"] = handler.matrix[0]
            kspace_xarray["read_idx"].attrs["resolution_mm"] = handler.resolution_mm[
                0
            ]  # should be change to kspace unit at some point
            kspace_xarray["read_idx"].attrs["extent_mm"] = handler.rps_extent_mm[0]
            kspace_xarray["read_idx"].attrs["axis_mm"] = handler.rps_axes_mm[0]
        except BaseException:
            kspace_xarray["read_idx"].attrs["matrix"] = 1
            kspace_xarray["read_idx"].attrs["resolution_mm"] = 0.0
            kspace_xarray["read_idx"].attrs["extent_mm"] = 0.0
            kspace_xarray["read_idx"].attrs["axis_mm"] = 0.0
        try:
            kspace_xarray["phase_idx"].attrs["matrix"] = handler.matrix[1]
            kspace_xarray["phase_idx"].attrs["resolution_mm"] = handler.resolution_mm[1]
            kspace_xarray["phase_idx"].attrs["extent_mm"] = handler.rps_extent_mm[1]
            kspace_xarray["phase_idx"].attrs["axis_mm"] = handler.rps_axes_mm[1]
        except BaseException:
            kspace_xarray["phase_idx"].attrs["matrix"] = 1
            kspace_xarray["phase_idx"].attrs["resolution_mm"] = 0.0
            kspace_xarray["phase_idx"].attrs["extent_mm"] = 0.0
            kspace_xarray["phase_idx"].attrs["axis_mm"] = 0.0
        try:
            kspace_xarray["slice_idx"].attrs["matrix"] = handler.matrix[2]
            kspace_xarray["slice_idx"].attrs["resolution_mm"] = handler.resolution_mm[2]
            kspace_xarray["slice_idx"].attrs["extent_mm"] = handler.rps_extent_mm[2]
            kspace_xarray["slice_idx"].attrs["axis_mm"] = handler.rps_axes_mm[2]
        except BaseException:
            kspace_xarray["slice_idx"].attrs["matrix"] = handler.n_slices
            kspace_xarray["slice_idx"].attrs["resolution_mm"] = 0.0
            kspace_xarray["slice_idx"].attrs["extent_mm"] = 0.0
            kspace_xarray["slice_idx"].attrs["axis_mm"] = 0.0

        try:
            kspace_xarray["x"].attrs["matrix"] = handler.matrix[0]
            kspace_xarray["x"].attrs["resolution_mm"] = handler.resolution_mm[0]
            kspace_xarray["x"].attrs["extent_mm"] = handler.xyz_extent_mm[0]
            kspace_xarray["x"].attrs["axis_mm"] = handler.xyz_axes_mm[0]
        except BaseException:
            kspace_xarray["x"].attrs["matrix"] = 1
            kspace_xarray["x"].attrs["resolution_mm"] = 0.0
            kspace_xarray["x"].attrs["extent_mm"] = 0.0
            kspace_xarray["x"].attrs["axis_mm"] = 0.0
        try:
            kspace_xarray["y"].attrs["matrix"] = handler.matrix[1]
            kspace_xarray["y"].attrs["resolution_mm"] = handler.resolution_mm[1]
            kspace_xarray["y"].attrs["extent_mm"] = handler.xyz_extent_mm[1]
            kspace_xarray["y"].attrs["axis_mm"] = handler.xyz_axes_mm[1]
        except BaseException:
            kspace_xarray["y"].attrs["matrix"] = 1
            kspace_xarray["y"].attrs["resolution_mm"] = 0.0
            kspace_xarray["y"].attrs["extent_mm"] = 0.0
            kspace_xarray["y"].attrs["axis_mm"] = 0.0
        try:
            kspace_xarray["z"].attrs["matrix"] = handler.matrix[2]
            kspace_xarray["z"].attrs["resolution_mm"] = handler.resolution_mm[2]
            kspace_xarray["z"].attrs["extent_mm"] = handler.xyz_extent_mm[2]
            kspace_xarray["z"].attrs["axis_mm"] = handler.xyz_axes_mm[2]
        except BaseException:
            kspace_xarray["z"].attrs["matrix"] = handler.n_slices
            kspace_xarray["z"].attrs["resolution_mm"] = 0.0
            kspace_xarray["z"].attrs["extent_mm"] = 0.0
            kspace_xarray["z"].attrs["axis_mm"] = 0.0

        if write_to_object:
            handler.kspace_xarray = kspace_xarray
        else:
            return kspace_xarray

    def set_reco_data(
        self,
        kspace_xarray: xr.DataArray,
        axes_names: list,
        handler=None,
        ift: bool = False,
        write_to_object: bool = False,
    ):
        """
        Calculate the reco_data from given kspace_xarray.

        Parameters
        ----------
        kspace_xarray : xarray
            The k-space data to be Fourier-transformed.
        axes_names : list of str
            List of dimension names to be Fourier-transformed.
        handler : VendorHandler, optional
            The vendor handler containing necessary attributes for coordinate assignment.
        ift : bool, optional
            If True, perform inverse Fourier transform. Default is False.
        write_to_object : bool, optional
            If True, the resulting spatial domain xarray is written to the handler object.

        Returns
        -------
        xarray or None
            The spatial domain xarray if write_to_object is False, otherwise None.
        """
        # Keep track of the signal domain without cloning the array upfront
        updated_domain = kspace_xarray.attrs.get("signal_domain")
        if "acquisition_idx" in axes_names:
            if updated_domain == "temporal":
                updated_domain = "spectral"
            elif updated_domain == "spectral":
                updated_domain = "temporal"
            else:
                updated_domain = None

        # perform the Fourier transform along the specified axes
        reco_data = self.ft(xarray=kspace_xarray, axes_names=axes_names, ift=ift)
        if updated_domain is not None:
            reco_data.attrs["signal_domain"] = updated_domain
        if (
            "read_idx" in axes_names
            or "phase_idx" in axes_names
            or "slice_idx" in axes_names
        ):
            reco_data.attrs["domain"] = "spatial"
        if getattr(handler, "freq_axis_ppm", None) is not None:
            assign_frequency_axis(reco_data.attrs, handler.freq_axis_ppm, "ppm")
        if getattr(handler, "freq_axis_hz", None) is not None:
            assign_frequency_axis(reco_data.attrs, handler.freq_axis_hz, "Hz")

        # preliminary to match seq2d:
        reco_data.data = np.roll(
            reco_data.data, -1, axis=reco_data.get_axis_num("read_idx")
        )
        reco_data.data = np.roll(
            reco_data.data, -1, axis=reco_data.get_axis_num("phase_idx")
        )
        # reco_data.data = np.roll(
        #     reco_data.data, -1, axis=reco_data.get_axis_num("slice_idx")
        # )
        try:
            reco_data.attrs["git_commit_hash"] = handler.git_commit_hash
        except:
            reco_data.attrs["git_commit_hash"] = None
        reco_data = self.rearrange_reco_data_xarray(
            input_data=reco_data, handler=handler
        )

        reco_data.attrs["domain"] = "spatial"

        # write the xarray to the object or return it:
        if write_to_object:
            handler.reco_data = reco_data
        else:
            return reco_data

    def rearrange_reco_data_xarray(
        self, input_data=None, handler=None, force_orient: bool = False
    ):
        """
        Rearrange the from-raw-data-reconstructed xarray according to the patient position and slice orientation to match the seq2d.

        Parameters
        ----------
        input_data : xarray, optional
            The xarray to be rearranged. If None, uses handler.reco_data.
        handler : VendorHandler
            The vendor handler containing necessary attributes for processing.
        force_orient : bool, optional
            If True, forces rearrangement even if it has been done before. Default is False.

        Returns
        -------
        xarray
            The rearranged xarray.

        """
        if input_data is None:
            reco_data = handler.reco_data.copy()
        else:
            reco_data = input_data

        # make sure to onlt rearange data once unless it's explicitly
        # requested:
        num_rearrange = reco_data.attrs.get("num_rearrange", 0)
        if num_rearrange == 0:
            reco_data.attrs["num_rearrange"] = 0
        elif num_rearrange > 0 and force_orient is False:
            return
        elif num_rearrange > 0 and force_orient is True:
            pass

        logger.debug(f"rearranging reco_data for {handler.patient_pos}")
        flip_multipliers = get_orientation_flip_multipliers(handler)
        axis_multipliers = (
            ("read_idx", flip_multipliers[0]),
            ("phase_idx", flip_multipliers[1]),
            ("slice_idx", flip_multipliers[2]),
        )

        for axis_name, multiplier in axis_multipliers:
            if multiplier >= 0:
                continue
            if axis_name not in reco_data.dims:
                continue
            axis_idx = reco_data.get_axis_num(axis_name)
            logger.debug(
                "rearrange_reco_data_xarray: flipping %s (axis %d) for patient_pos=%s, "
                "slice_orientation=%s, read_orientation=%s",
                axis_name,
                axis_idx,
                getattr(handler, "patient_pos", None),
                getattr(handler, "slice_orientation", None),
                getattr(handler, "read_orientation", None),
            )
            reco_data.data = np.flip(reco_data.data, axis=axis_idx)
        reco_data.attrs["num_rearrange"] = reco_data.attrs["num_rearrange"] + 1
        return reco_data

    def postprocess_fid_file(
        self,
        input_data: np.ndarray = None,
        handler=None,
    ):  # taken from MATLAB pvtools - readBrukerRaw.m
        """
        Complex data provided in fid file is padded with zeros to reach 128bits in read direction.
        Function deletes padded zeros from data depending on actual number of sampling steps in read direction.

        Parameters
        ----------
        input_data : np.ndarray, optional
            Complex data from fid file. If None, uses handler.fid. Contains additional zeros.

        handler : VendorHandler
            The vendor handler containing necessary attributes for processing.

        Returns
        -------
        complex_file : np.ndarray
            Complex data from fid file without additional zeros.
            Reshaped to dimensions [ACQ_size[1]*ACQ_size[2]*NR*NI, PVM_Matrix[0]]

        """
        if input_data is None:
            complex_file = handler.fid
        else:
            complex_file = input_data

        # delete zero lines if ACQ-experiment_mode == "Single Experiment"
        if handler.acqp["ACQ_experiment_mode"] == "SingleExperiment":
            if handler.method["PVM_EncNReceivers"] == 1:
                numSelectedReceivers = 1

                if handler.acqp["GO_raw_data_format"] == "GO_32BIT_SGN_INT":
                    bits = 32
                elif handler.acqp["GO_raw_data_format"] == "GO_16BIT_SGN_INT":
                    bits = 16
                elif handler.acqp["GO_raw_data_format"] == "GO_32BIT_FLOAT":
                    bits = 32
                else:
                    bits = 32

                try:
                    acqp_size = handler.acqp["ACQ_size"][0]
                except BaseException:
                    acqp_size = handler.acqp["ACQ_size"]

                if handler.acqp["GO_block_size"] == "Standard_KBlock_Format":
                    blockSize = int(
                        (
                            (
                                np.ceil(
                                    acqp_size * numSelectedReceivers * (bits / 8) / 1024
                                )
                                * 1024
                                / (bits / 8)
                            )
                            / 2
                        )
                    )
                else:
                    blockSize = int((acqp_size * numSelectedReceivers) / 2)

                other_dimensions = int(
                    np.prod(handler.acqp["ACQ_size"][1:])
                    * handler.acqp["NI"]
                    * handler.acqp["NR"]
                )

                complex_file = np.reshape(
                    complex_file,
                    (other_dimensions, blockSize),
                )
                if (
                    blockSize != acqp_size * numSelectedReceivers / 2
                ):  # remove zero-lines
                    complex_file = complex_file[
                        :,
                        0 : int(acqp_size * numSelectedReceivers / 2),
                    ]
        else:  # Parallel Experiment
            try:
                numSelectedReceivers = handler.method["PVM_EncNReceivers"]

                if handler.acqp["GO_raw_data_format"] == "GO_32BIT_SGN_INT":
                    bits = 32
                elif handler.acqp["GO_raw_data_format"] == "GO_16BIT_SGN_INT":
                    bits = 16
                elif handler.acqp["GO_raw_data_format"] == "GO_32BIT_FLOAT":
                    bits = 32
                else:
                    bits = 32

                try:
                    acqp_size = handler.acqp["ACQ_size"][0]
                except BaseException:
                    acqp_size = handler.acqp["ACQ_size"]

                if handler.acqp["GO_block_size"] == "Standard_KBlock_Format":
                    blockSize = int(
                        (
                            (
                                np.ceil(
                                    acqp_size * numSelectedReceivers * (bits / 8) / 1024
                                )
                                * 1024
                                / (bits / 8)
                            )
                            / 2
                        )
                    )
                else:
                    blockSize = int((acqp_size * numSelectedReceivers) / 2)

                other_dimensions = int(
                    np.prod(handler.acqp["ACQ_size"][1:])
                    * handler.acqp["NI"]
                    * handler.acqp["NR"]
                )

                complex_file = np.reshape(
                    complex_file,
                    (other_dimensions, blockSize),
                )
                if (
                    blockSize != acqp_size * numSelectedReceivers / 2
                ):  # remove zero-lines
                    complex_file = complex_file[
                        :,
                        0 : int(acqp_size * numSelectedReceivers / 2),
                    ]

            except BaseException:
                Warning(
                    "Could not postprocess bssfp fid with "
                    + str(numSelectedReceivers)
                    + "Channels"
                )

        return complex_file

    def reshape_fid_to_kspace(self, input_data=None, handler=None):
        """
        Reshape fid to kspace.

        Parameters
        ----------
        input_data : np.ndarray, optional
            The fid data to be reshaped. If None, uses handler.fid_pp or handler.fid.
        handler : VendorHandler
            The vendor handler containing necessary attributes for processing.

        Returns
        -------
        kspace : np.ndarray
            The reshaped k-space data.
        """
        if input_data is None:
            try:
                fid = handler.fid_pp
            except BaseException:
                fid = handler.fid
        else:
            fid = input_data

        # use encoding matrix to reshape raw data:
        kspace = np.reshape(
            fid,
            (
                1,
                handler.n_repetitions,
                handler.enc_matrix[2],
                handler.enc_matrix[1],
                handler.n_receivers,
                handler.enc_matrix[0],
            ),
        )

        pe_order, pe_order_2 = self._get_phase_encode_order(handler)
        kspace = kspace[:, :, :, pe_order, :, :]
        if pe_order_2 is not None:
            kspace = kspace[:, :, pe_order_2, :, :, :]
        # init empty array:
        target_dtype = kspace.dtype
        kspace_zeros = np.zeros(
            (
                1,
                handler.n_repetitions,
                handler.matrix[2],
                handler.matrix[1],
                handler.n_receivers,
                handler.matrix[0],
            ),
            dtype=target_dtype,
        )
        # fill empty array:
        kspace_zeros[
            :,
            :,
            : handler.enc_matrix[2],
            : handler.enc_matrix[1],
            :,
            : handler.enc_matrix[0],
        ] = kspace
        return kspace_zeros

    def _get_phase_encode_order(self, handler):
        """
        Get the phase encoding order from the handler.

        Parameters
        ----------
        handler : VendorHandler
            The handler containing the phase encoding order.

        Returns
        -------
        list
            The phase encoding order.

        """
        order_ind = handler.method["PVM_EncSteps1"]
        order = np.argsort(order_ind)
        if handler.spat_dim == "3D":
            order_ind_2 = handler.method["PVM_EncSteps2"]
            order_2 = np.argsort(order_ind_2)
        else:
            order_2 = None

        return order, order_2


class BrukerMGERawRecoStrategy(BrukerRawRecoStrategy):
    """
    Reconstruction strategy for Bruker MGE raw data.
    """

    def reco_rawdata(
        self,
        handler=None,
        postprocess_fid=True,
        reshape_fid=True,
        set_kspace_xarray=True,
        shift_xarray=True,
        set_reco_data=True,
        db=True,
        data_precision=np.float16,
    ):
        """
        Reconstruct raw fid data into k-space and spatial domain xarray DataArrays.
        Specialized for MGE to handle echoes as acquisition_idx.
        """
        reco_data = None
        if hasattr(handler, "fid") and postprocess_fid is True:
            fid = handler.fid
        elif hasattr(handler, "fid_pp") and postprocess_fid is False:
            fid = handler.fid_pp
        else:
            fid = getattr(handler, "fid", None)

        fid_pp = fid

        if fid_pp is None:
            raise ValueError("No FID data available for reconstruction")

        if postprocess_fid is True:
            try:
                fid_pp = self.postprocess_fid_file(input_data=fid, handler=handler)
            except Exception as e:
                full_traceback = traceback.format_exc()
                err_msg, err_type = str(e), e.__class__.__name__

                msg = f"{handler.ExpNum}: base_anatomy: reco_rawdata()\nThe following error occured during "
                msg += f"postprocess_fid_file:\n    {err_type}: {err_msg}\n"
                if db:
                    msg += f"Full traceback:\n{full_traceback}"
                logger.debug(msg)

        if reshape_fid is True:
            handler.kspace = self.reshape_fid_to_kspace(
                input_data=fid_pp, handler=handler
            )
        else:
            if not hasattr(handler, "kspace"):
                handler.kspace = fid_pp

        # generate kspace_xarray:
        if set_kspace_xarray is True:
            if not hasattr(handler, "kspace"):
                raise ValueError(
                    f"{handler.ExpNum}: base_anatomy: reco_rawdata() - "
                    "Cannot set kspace xarray: 'kspace' data is missing."
                )

            try:
                # MGE dimensions: (1, rep, slice, phase, echo, channel, read)
                # We want to map echo to acquisition_idx
                dims = [
                    "unused_idx",  # The '1' dimension
                    "repetition_idx",
                    "slice_idx",
                    "phase_idx",
                    "acquisition_idx",  # Mapping echo to acquisition
                    "channel_idx",
                    "read_idx",
                ]

                # Calculate frequency axes based on echo spacing
                # Use echo_spacing_ms if available, otherwise derive from EffectiveTE
                echo_spacing_ms = getattr(handler, "echo_spacing_ms", 0.0)

                # Robustly retrieve EffectiveTE (support dict and object access)
                if isinstance(handler.method, dict) or hasattr(handler.method, "get"):
                    effective_te = handler.method.get("EffectiveTE", None)
                else:
                    effective_te = getattr(handler.method, "EffectiveTE", None)

                if effective_te is None:
                    effective_te = np.arange(handler.n_echoes)

                if echo_spacing_ms == 0.0 and len(effective_te) > 1:
                    echo_spacing_ms = float(effective_te[1] - effective_te[0])

                if echo_spacing_ms > 0:
                    echo_spacing_s = echo_spacing_ms / 1000.0
                else:
                    echo_spacing_s = 1.0  # arbitrary placeholder

                # Set spectral parameters on handler for BaseSpectroscopy compatibility
                handler.dwell_time_s = echo_spacing_s
                handler.bandwidth_hz = 1.0 / echo_spacing_s
                handler.n_acquisition = handler.n_echoes

                # Calculate sampling_time_axis_s from echo_spacing_ms
                # If effective_te is available, it provides the actual echo times (including TE1)
                if len(effective_te) == handler.n_echoes:
                    handler.sampling_time_axis_s = np.array(effective_te) / 1000.0
                else:
                    # Fallback: start at TE1 if available, otherwise at 0
                    te1 = (
                        effective_te[0]
                        if len(effective_te) > 0
                        else getattr(handler, "echo_time_ms", 0.0)
                    )
                    handler.sampling_time_axis_s = (
                        te1 + np.arange(handler.n_echoes) * echo_spacing_ms
                    ) / 1000.0

                # Calculate ppm bandwidth if reference frequency is available
                if (
                    hasattr(handler, "reference_frequency_mhz")
                    and handler.reference_frequency_mhz
                ):
                    handler.bandwidth_ppm = (
                        handler.bandwidth_hz / handler.reference_frequency_mhz
                    )
                else:
                    # Default/fallback
                    handler.bandwidth_ppm = None

                # Calculate axes using helper
                # We use npoints=handler.n_echoes and dwell_time_s=echo_spacing_s
                from hypermri.analysis.spectroscopy.core import get_freq_axis

                # Manually setting n_acquisition on handler for get_freq_axis if needed,
                # but we can pass npoints directly.
                freq_axis_hz = get_freq_axis(
                    data_obj=None,
                    unit="Hz",
                    npoints=handler.n_echoes,
                    dwell_time_s=echo_spacing_s,
                    centered_at_0=True,
                )

                # Calculate ppm axis manually or via helper if we can set up a dummy object or pass parameters
                # Since get_freq_axis with data_obj=None only returns Hz, we do ppm manually here
                # or rely on the handler attributes we just set.
                if handler.bandwidth_ppm != 0:
                    # Using the handler we just updated
                    freq_axis_ppm = get_freq_axis(
                        data_obj=handler,
                        unit="ppm",
                        npoints=handler.n_echoes,
                        centered_at_0=False,
                    )
                else:
                    freq_axis_ppm = None

                # Create coordinates
                coords = {
                    "unused_idx": np.array([0]),
                    "repetition_idx": (
                        "repetition_idx",
                        np.arange(handler.kspace.shape[1]),
                        {"unit": "index"},
                    ),
                    "slice_idx": (
                        ["read_idx", "phase_idx", "slice_idx"],
                        handler.rps_grid_mm[2],
                        {"unit": "mm"},
                    ),
                    "phase_idx": (
                        ["read_idx", "phase_idx", "slice_idx"],
                        handler.rps_grid_mm[1],
                        {"unit": "mm"},
                    ),
                    "acquisition_idx": (
                        "acquisition_idx",
                        effective_te,
                        {"unit": "ms"},
                    ),
                    "channel_idx": (
                        "channel_idx",
                        np.arange(handler.n_receivers),
                        {"unit": "index"},
                    ),
                    "read_idx": (
                        ["read_idx", "phase_idx", "slice_idx"],
                        handler.rps_grid_mm[0],
                        {"unit": "mm"},
                    ),
                }

                handler.kspace_xarray = self.set_kspace_xarray(
                    handler, data=handler.kspace, dims=dims, coords=coords
                )

                # Add frequency coordinates if calculated
                if freq_axis_hz is not None:
                    handler.kspace_xarray = handler.kspace_xarray.assign_coords(
                        Hz=("acquisition_idx", freq_axis_hz)
                    )
                    handler.kspace_xarray["Hz"].attrs["unit"] = "Hz"

                if freq_axis_ppm is not None:
                    handler.kspace_xarray = handler.kspace_xarray.assign_coords(
                        ppm=("acquisition_idx", freq_axis_ppm)
                    )
                    handler.kspace_xarray["ppm"].attrs["unit"] = "ppm"

                # Squeeze out the unused dimension
                if "unused_idx" in handler.kspace_xarray.dims:
                    handler.kspace_xarray = handler.kspace_xarray.squeeze("unused_idx")

            except Exception as e:
                full_traceback = traceback.format_exc()
                err_msg, err_type = str(e), e.__class__.__name__

                msg = f"{handler.ExpNum}: BrukerMGERawRecoStrategy: reco_rawdata()\nThe following error occured during "
                msg += f"set_kspace_xarray:\n    {err_type}: {err_msg}\n"
                if db:
                    msg += f"Full traceback:\n{full_traceback}"
                raise ValueError(msg)

        if shift_xarray is True:
            try:
                handler.kspace_xarray = shift_image(
                    input_data=handler.kspace_xarray,
                    shift_method="phase",
                    data_obj=handler,
                    domain="kspace",
                )
            except Exception as e:
                if db:
                    full_traceback = traceback.format_exc()
                    err_msg, err_type = str(e), e.__class__.__name__
                    msg = f"{handler.ExpNum}: BrukerMGERawRecoStrategy: reco_rawdata()\nError during shift_image:\n{err_type}: {err_msg}\n"
                    logger.debug(msg)
                    raise ValueError(msg)

        domain = "kspace"
        if set_reco_data is True:
            logger.debug(
                "set_reco_data is True, proceeding with spatial domain reconstruction"
            )

            # Target order: acquisition, read, phase, slice, rep, channel
            new_order = (
                "acquisition_idx",
                "read_idx",
                "phase_idx",
                "slice_idx",
                "repetition_idx",
                "channel_idx",
            )

            try:
                # Transpose to desired order
                # Filter new_order to only include dims that exist
                valid_order = [d for d in new_order if d in handler.kspace_xarray.dims]
                handler.kspace_xarray = handler.kspace_xarray.transpose(*valid_order)

                if handler.spat_dim == "2D":
                    axes_names = ["read_idx", "phase_idx"]
                elif handler.spat_dim == "3D":
                    axes_names = ["read_idx", "phase_idx", "slice_idx"]
                else:
                    axes_names = []

                reco_data = self.set_reco_data(
                    kspace_xarray=handler.kspace_xarray,
                    axes_names=axes_names,
                    handler=handler,
                    write_to_object=False,
                    ift=False,
                )

                reco_data.data = self._convert_array_precision(
                    reco_data.data, data_precision
                )

                domain = "spatial"
            except Exception as e:
                full_traceback = traceback.format_exc()
                err_msg, err_type = str(e), e.__class__.__name__
                msg = f"{handler.ExpNum}: BrukerMGERawRecoStrategy: reco_rawdata()\nError during set_reco_data:\n{err_type}: {err_msg}\n"
                if db:
                    msg += f"Full traceback:\n{full_traceback}"
                raise ValueError(msg)

        reco_data.attrs["domain"] = domain
        return reco_data

    def reshape_fid_to_kspace(self, input_data=None, handler=None):
        """
        Reshape MGE fid to kspace including echo dimension.
        """
        if input_data is None:
            try:
                fid = handler.fid_pp
            except BaseException:
                fid = handler.fid
        else:
            fid = input_data

        # Reshape including echoes
        # Assumed structure: (1, Reps, Slices, Phase, Echoes, Receivers, Read)
        # We need to verify the exact order. Based on typical Bruker filling:
        # Loop structure usually: Reps -> Slices -> Phase -> Echoes -> Read
        # Receivers are usually separate.

        # Using the standard Bruker logic but adding echoes inside the phase loop?
        # Typically echoes are acquired sequentially after excitation.
        # So for one phase encoding step, we get N echoes.

        try:
            kspace = np.reshape(
                fid,
                (
                    1,
                    handler.n_repetitions,
                    handler.enc_matrix[
                        2
                    ],  # Slices (usually 1 for 2D, but handled as matrix[2])
                    handler.enc_matrix[1],  # Phase
                    handler.n_echoes,  # Echoes
                    handler.n_receivers,
                    handler.enc_matrix[0],  # Read
                ),
            )
        except Exception as e:
            raise ValueError(
                f"Failed to reshape MGE data with n_echoes={handler.n_echoes}: {e}"
            )

        # Handle phase encoding reordering
        pe_order, pe_order_2 = self._get_phase_encode_order(handler)

        # Apply reordering to the phase dimension (index 3)
        if pe_order is not None:
            kspace = kspace[:, :, :, pe_order, :, :, :]

        # If 3D, apply reordering to slice dimension (index 2) - check if this is correct for 3D MGE
        if pe_order_2 is not None:
            kspace = kspace[:, :, pe_order_2, :, :, :, :]

        # Create zero-filled target array (for partial fourier etc)
        target_dtype = kspace.dtype
        kspace_zeros = np.zeros(
            (
                1,
                handler.n_repetitions,
                handler.matrix[2],
                handler.matrix[1],
                handler.n_echoes,
                handler.n_receivers,
                handler.matrix[0],
            ),
            dtype=target_dtype,
        )

        # Fill
        kspace_zeros[
            :,
            :,
            : handler.enc_matrix[2],
            : handler.enc_matrix[1],
            :,
            :,
            : handler.enc_matrix[0],
        ] = kspace

        return kspace_zeros


class Brukerseq2dRecoStrategy(BaseRecoStrategy):
    """
    Reconstruction strategy for Bruker seq2d data.

    Parameters
    ----------
    handler: VendorHandler
        The handler object containing the MRI data and metadata.
    reco_options: dict
        Reconstruction options, such as data precision and whether to skip reconstruction.

    Returns
    -------
    xr.DataArray
        The reconstructed data as an xarray DataArray.

    """

    def reconstruct(self, handler, **reco_options):
        try:
            return self._reconstruct_internal(handler, **reco_options)
        except Exception as e:
            import hypermri.utils.system.logging as logging_utils

            logger = logging_utils.init_default_logger(__name__)
            logger.error(
                f"Exception in Brukerseq2dRecoStrategy.reconstruct: {type(e).__name__}: {e}"
            )
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _reconstruct_internal(self, handler, **reco_options):
        """
        Creates an xarray DataArray from MRI data, handling potentially squeezed dimensions.

        This function determines the actual dimensions present in the input `seq2d`
        based on the expected input order (determined by slice_orientation) and
        expected dimension sizes. It expands squeezed dimensions, transposes the
        array to a target order, and assigns coordinates.

        Parameters
        ----------
        handler : VendorHandler
            The handler object containing the MRI data and metadata.

        db : bool, optional
            Debug flag (currently unused in the provided snippet).

        Returns
        -------
        xr.DataArray
            The fully formed DataArray with correct dimensions, order, and coordinates.

        Raises
        ------
        ValueError
            If slice_orientation is unknown, or if dimensions/shapes cannot be
            consistently matched between inputs and expectations.

        Uses Attributes from 'self':
        ----------------------------
        seq2d : np.ndarray
            The input data array, potentially with squeezed dimensions.
        slice_orientation : str
            Orientation of slices ('axial', 'sagittal', 'coronal').
        rps_grid_mm : list of np.ndarray
            List containing coordinate grids [read, phase, slice] in mm.
        n_repetitions : int
            Expected number of repetitions.
        n_receivers : int
            Expected number of receiver channels.
        sampling_time_axis_s : np.ndarray
            Time points for the acquisition axis.
        n_echoes : int
            Expected number of echoes.
        b_values : np.ndarray
            Array of b-values.
        """
        time.time()
        logger.info(
            f"[TIMING] Starting Brukerseq2dRecoStrategy reconstruction for scan {handler.ExpNum}"
        )
        no_reconstruction = reco_options.get("no_reconstruction", False)
        if no_reconstruction:
            logger.info(
                "[Brukerseq2dRecoStrategy] No reconstruction performed as per settings."
            )
            return None

        # Define the potential dimensions in a standard, unsqueezed order (used for reference and target)
        # Near the beginning of the function
        DATA_PRECISION = self._parse_data_precision(
            reco_options.get("data_precision", np.float16)
        )
        append_parameters = reco_options.get("append_parameters", True)

        # Ensure the handler is using the correct data precision
        handler = self._data_precision(handler=handler, DATA_PRECISION=DATA_PRECISION)

        recon_num = reco_options.get("recon_num", 1)
        set_reco_file = reco_options.get("set_reco_file", False)

        potential_original_dims_order_ref = [
            "read_idx",
            "phase_idx",
            "slice_idx",
            "inversion_idx",
            "echo_idx",
            "bvalue_idx",
            "channel_idx",
            "acquisition_idx",
            "repetition_idx",
        ]
        # Get data and parameters from handler
        # Assuming the raw data that needs to be processed into seq2d is available
        # as handler.fid_pp after postprocess_fid_flash in FLASH.
        data_load_start = time.time()
        logger.debug(f"seq2d reconstruction starting for scan {handler.ExpNum}")

        seq2d_raw = handler.load_2dseq_file(
            recon_num=recon_num,
            set_reco_file=set_reco_file,
            data_precision=DATA_PRECISION,
        )
        logger.debug(f"seq2d_raw type: {type(seq2d_raw)}")
        logger.debug(f"seq2d_raw shape: {getattr( seq2d_raw, 'shape', 'no shape')}")
        logger.debug(f"seq2d_raw size: {getattr( seq2d_raw, 'size', 'no size')}")

        if seq2d_raw is None or getattr(seq2d_raw, "size", 0) == 0:
            logger.debug(
                f"{getattr(handler, 'id', 'unknown')}: No 2dseq data available; skipping seq2d reconstruction."
            )
            return None

        seq2d = (
            seq2d_raw.squeeze()
        )  # Use raw_data_for_seq2d as the input for reconstruction
        logger.debug(f"seq2d after squeeze type: {type(seq2d)}")
        logger.debug(
            f"seq2d after squeeze shape: {getattr( seq2d, 'shape', 'no shape')}"
        )
        logger.debug(f"seq2d after squeeze size: {getattr( seq2d, 'size', 'no size')}")

        data_load_time = time.time() - data_load_start
        logger.debug(f"[TIMING] seq2d data loading took {data_load_time:.3f}s")

        # Check if seq2d data is available
        if seq2d.size == 0:
            logger.debug(
                f"{getattr(handler, 'id', 'unknown')}: Empty 2dseq data encountered; skipping seq2d reconstruction."
            )
            return None

        time.time()
        slice_orientation = handler.slice_orientation
        try:
            rps_grid_mm = handler.rps_grid_mm
        except Exception as exc:
            logger.debug(
                f"{getattr(handler, 'id', 'unknown')}: Failed to access rps_grid_mm ({exc}); skipping seq2d reconstruction."
            )
            return None

        if rps_grid_mm is None:
            logger.debug(
                f"{getattr(handler, 'id', 'unknown')}: rps_grid_mm not available; skipping seq2d reconstruction."
            )
            return None
        n_repetitions = handler.n_repetitions
        # handler.n_receivers # need to fix this in case indivial channels were
        # reconstructed
        n_receivers = handler.reco.get("RecoNumOutputChan", 1)
        sampling_time_axis_s = handler.sampling_time_axis_s
        n_echoes = handler.n_echoes
        b_values = handler.b_values
        inversion_time_ms = handler.inversion_time_ms
        base_inversion_time_ms = np.atleast_1d(
            handler.inversion_time_ms
        )  # Ensure it's an array

        n_inversion = np.size(inversion_time_ms)
        slice_TI_increment_ms = calc_inversion_time_multi_slice(data_obj=handler)

        # --- Determine expected full shape ---
        # !!! IMPORTANT: Logic to derive read/phase/slice sizes from rps_grid_mm
        # needs to be correct based on YOUR data structure.
        # This assumes rps_grid_mm[0] always provides the read/phase/slice grid
        # and its shape reflects those dimensions directly.
        try:
            # Assuming grid shapes are consistently (read, phase, slice) in
            # rps_grid_mm[0]
            read_size = rps_grid_mm[0].shape[0]
            phase_size = rps_grid_mm[0].shape[1]
            slice_size = rps_grid_mm[0].shape[2]

        except (IndexError, AttributeError) as e:
            logger.debug(
                f"{getattr(handler, 'id', 'unknown')}: Invalid rps_grid_mm structure ({e}); skipping seq2d reconstruction."
            )
            return None

        # Check for PVM_FwsOnOff to determine water/fat separation
        pvm_fws_on_off = handler.fat_water_separation
        if pvm_fws_on_off == "On":
            n_waterfat = 2  # Water and Fat
        else:
            n_waterfat = 1  # No water/fat separation

        # Detect canonical sequence via alias mapping to determine sequence-specific handling
        raw_sequence_name = getattr(handler, "sequence_name", None)
        canonical_sequence = resolve_sequence_alias(raw_sequence_name)
        if canonical_sequence is None and hasattr(handler, "method"):
            canonical_sequence = resolve_sequence_alias(handler.method.get("Method"))

        # Check if this is an MGE sequence - use EffectiveTE for acquisition_idx instead of echo_idx
        is_mge_sequence = canonical_sequence == "mge"
        effective_te_values = None
        if is_mge_sequence:
            method_params = getattr(handler, "method", None) or {}
            effective_te_values = method_params.get("EffectiveTE")
            if effective_te_values is None:
                logger.warning(
                    f"{handler.id}: Sequence resolved to 'mge' via alias but EffectiveTE is missing; "
                    "falling back to standard echo-based handling."
                )
                is_mge_sequence = False
            else:
                logger.info(
                    f"{handler.id}: Detected MGE sequence with EffectiveTE: {effective_te_values}"
                )

        # Check if this is an FieldMap sequence - do not use echo_idx or acquisition_idx
        is_fieldmap_sequence = canonical_sequence == "fieldmap"
        effective_te_values = None
        if is_fieldmap_sequence:
            method_params = getattr(handler, "method", None) or {}
            effective_te_values = method_params.get("EffectiveTE")
            if effective_te_values is None:
                logger.warning(
                    f"{handler.id}: Sequence resolved to 'mge' via alias but EffectiveTE is missing; "
                    "falling back to standard echo-based handling."
                )
                is_fieldmap_sequence = False
            else:
                logger.info(
                    f"{handler.id}: Detected MGE sequence with EffectiveTE: {effective_te_values}"
                )

        # Map all potential dimensions to their expected sizes
        full_shape_map = {
            "read_idx": read_size,
            "phase_idx": phase_size,
            "slice_idx": slice_size,
            "repetition_idx": n_repetitions,
            "channel_idx": n_receivers,
            "acquisition_idx": (
                n_echoes
                if is_mge_sequence
                else (
                    len(sampling_time_axis_s) if sampling_time_axis_s is not None else 0
                )
            ),
            "echo_idx": (
                0 if (is_mge_sequence or is_fieldmap_sequence) else n_echoes
            ),  # For MGE, don't use echo_idx
            "bvalue_idx": np.size(b_values),
            "inversion_idx": n_inversion,
            "waterfat_idx": n_waterfat,  # Add water/fat dimension
        }

        # --- Determine actual dimension order based on known expected input order and sizes ---

        # Define the expected input order *before* squeezing, based on orientation
        # Add 'waterfat_idx' to the potential dimensions
        if slice_orientation is None:
            # If slice_orientation is None, we assume the input order is
            # already correct
            return None

        # Determine the input order of spatial dimensions based on the transpose_read_phase flag
        if getattr(handler, "transpose_read_phase", False):
            # If transpose is needed, the input order is phase, then read
            spatial_dims_input = ("phase_idx", "read_idx", "slice_idx")
        else:
            # The default input order is read, then phase
            spatial_dims_input = ("read_idx", "phase_idx", "slice_idx")

        # Define the full expected input order by combining spatial and non-spatial dimensions
        expected_input_order_unsqueezed = list(
            spatial_dims_input
        ) + [  # Convert tuple to list to allow concatenation
            "waterfat_idx",
            "inversion_idx",
            "echo_idx",
            "bvalue_idx",
            "channel_idx",
            "acquisition_idx",
            "repetition_idx",
        ]

        # Determine the actual dimensions present in seq2d by filtering the expected input order
        # Only include dimensions whose expected size is greater than 1
        actual_dims_order = [
            dim
            for dim in expected_input_order_unsqueezed
            if full_shape_map.get(dim, 0) > 1
        ]

        # Validation check: Does the number of determined dimensions match the
        # input array's dimensions?
        if len(actual_dims_order) != seq2d.ndim:
            raise ValueError(
                f"Dimension mismatch: Input seq2d has {seq2d.ndim} dimensions "
                f"(shape {seq2d.shape}), but based on the expected input order for '{slice_orientation}' "
                f"and sizes > 1, expected {len(actual_dims_order)} dimensions ({actual_dims_order}). "
                f"Check input parameters (n_repetitions, n_receivers, etc.), derived sizes, "
                f"and the expected input order convention. Full shape map used: {full_shape_map}"
            )
        # --- End of dimension order determination ---

        # --- Define target spatial dimensions based on desired output orientation ---
        # This determines the *output* order, independent of the input order.
        if slice_orientation == "axial":
            spatial_dims_target = ["read_idx", "phase_idx", "slice_idx"]
        elif slice_orientation == "sagittal":
            # Define desired output order for sagittal
            spatial_dims_target = ["read_idx", "phase_idx", "slice_idx"]
        elif slice_orientation == "coronal":
            # Define desired output order for coronal
            spatial_dims_target = ["read_idx", "slice_idx", "phase_idx"]
        else:
            # This case was already checked, but included for completeness
            raise ValueError(
                f"Unknown slice_orientation for determining target order: {slice_orientation}"
            )

        # --- Define the full target dimension order for the output DataArray ---
        target_dims_potential = spatial_dims_target + [
            "waterfat_idx",  # Added
            # Keep desired non-spatial order here
            "inversion_idx",
            "echo_idx",
            "bvalue_idx",
            "channel_idx",
            "acquisition_idx",
            "repetition_idx",
        ]
        # Filter target_dims based on available dimensions (size >= 1)
        target_dims = [
            dim for dim in target_dims_potential if full_shape_map.get(dim, 0) >= 1
        ]

        # --- Reshape/Transpose using xarray ---

        # 1. Create a temporary DataArray with the squeezed data and its
        # *correctly ordered* dimensions
        try:
            # The dims provided here MUST match the order of axes in seq2d
            temp_da = xr.DataArray(seq2d, dims=actual_dims_order, name="mri_data")
        except ValueError as e:
            print(f"Error creating temp_da: {e}")
            print(f"seq2d shape: {seq2d.shape}")
            print(f"actual_dims_order: {actual_dims_order}")
            raise

        # 2. Identify missing dimensions (present in target_dims but not in
        # actual_dims_order)
        missing_dims = [d for d in target_dims if d not in temp_da.dims]

        # 3. Expand the temporary DataArray to include the missing dimensions
        # (size 1)
        if missing_dims:
            # Ensure we only try to expand dimensions that should actually
            # exist (size >= 1)
            dims_to_expand = {
                d: 1 for d in missing_dims if full_shape_map.get(d, 0) >= 1
            }
            if dims_to_expand:
                expanded_da = temp_da.expand_dims(dim=dims_to_expand)
            else:
                expanded_da = temp_da

        else:
            expanded_da = temp_da

        # 4. Transpose the expanded DataArray to match the final target dimension order
        # Ensure all target_dims actually exist in the expanded_da before
        # transposing
        final_transpose_dims = [d for d in target_dims if d in expanded_da.dims]
        final_da = expanded_da.transpose(*final_transpose_dims)

        # --- Define and Assign Coordinates ---
        coords_dict = {}
        shape_dict = dict(
            zip(final_da.dims, final_da.shape)
        )  # Map dim names to sizes in the *final* array

        # Spatial coordinates (using sizes derived earlier)
        # *** Replace placeholder coordinate generation with your actual 1D coordinate extraction logic ***
        if "read_idx" in shape_dict:
            # Use read_size determined earlier for the coordinate vector length
            coords_dict["read_idx"] = (
                ["read_idx", "phase_idx", "slice_idx"],
                handler.rps_grid_mm[0],
                {"unit": "mm"},
            )
        if "phase_idx" in shape_dict:
            # Use phase_size determined earlier

            coords_dict["phase_idx"] = (
                ["read_idx", "phase_idx", "slice_idx"],
                handler.rps_grid_mm[1],
                {"unit": "mm"},
            )
        if "slice_idx" in shape_dict:
            # Use slice_size determined earlier
            np.linspace(-slice_size / 2, slice_size / 2, slice_size)
            coords_dict["slice_idx"] = (
                ["read_idx", "phase_idx", "slice_idx"],
                handler.rps_grid_mm[2],
                {"unit": "mm"},
            )

        # Add water/fat coordinates if present
        if "waterfat_idx" in shape_dict:
            coords_dict["waterfat_idx"] = (
                "waterfat_idx",
                ["Water", "Fat"] if shape_dict["waterfat_idx"] == 2 else ["Unknown"],
                {"unit": "type"},
            )

        # Non-spatial coordinates (adjusting for potential squeezing reflected
        # in final_da shape)
        if "repetition_idx" in shape_dict:
            # Always use indices for repetition_idx to maintain backward compatibility
            rep_size = shape_dict["repetition_idx"]
            coords_dict["repetition_idx"] = (
                "repetition_idx",
                np.arange(rep_size),
                {"unit": "index"},
            )
        if "channel_idx" in shape_dict:
            # Use the actual size from the final array
            coords_dict["channel_idx"] = (
                "channel_idx",
                np.arange(shape_dict["channel_idx"]),
                {"unit": "index"},
            )

        if "acquisition_idx" in shape_dict:
            acq_size = shape_dict["acquisition_idx"]  # Actual size in final_da

            # For MGE sequences, use EffectiveTE values as coordinates
            if is_mge_sequence and effective_te_values is not None:
                debug_log_recon_strategy(
                    f"{handler.id}: EffectiveTE {effective_te_values} is a single integer value, expected a list/array for MGE sequence. Using indices instead."
                )

                if len(effective_te_values) >= acq_size:
                    # EffectiveTE is typically in milliseconds, convert to seconds
                    acq_coords = np.array(effective_te_values[:acq_size]) / 1000.0
                    logger.info(
                        f"Using EffectiveTE for acquisition_idx coordinates: {acq_coords}"
                    )
                else:
                    logger.warning(
                        f"EffectiveTE has {len(effective_te_values)} values but need {acq_size}, using indices"
                    )
                    acq_coords = np.arange(acq_size)
            elif (
                acq_size > 0
                and sampling_time_axis_s is not None
                and len(sampling_time_axis_s) >= acq_size
            ):
                # Select the correct number of time points
                acq_coords = sampling_time_axis_s[:acq_size]
            elif acq_size > 0:  # Fallback if time axis is unexpectedly short
                acq_coords = np.arange(acq_size)
            else:
                acq_coords = np.array([])
            if acq_coords.size > 0:
                coords_dict["acquisition_idx"] = (
                    "acquisition_idx",
                    acq_coords,
                    {"unit": "s"},
                )

        if "echo_idx" in shape_dict:
            # Use the actual size from the final array
            coords_dict["echo_idx"] = (
                "echo_idx",
                np.arange(shape_dict["echo_idx"]),
                {"unit": "index"},
            )

        if "bvalue_idx" in shape_dict:
            bval_size = shape_dict["bvalue_idx"]  # Actual size in final_da
            if bval_size == 1:
                bval_coords = np.array([b_values])
            if bval_size > 1 and np.size(b_values) >= bval_size:
                # Select the correct number of b-values
                bval_coords = b_values[:bval_size]
                # Ensure it's an array even if scalar after slicing
                if np.isscalar(bval_coords):
                    bval_coords = np.array([bval_coords])
            elif bval_size > 0:  # Fallback if b_values array is unexpectedly short
                bval_coords = np.arange(bval_size)
            else:
                bval_coords = np.array([])
            if bval_coords.size > 0:
                coords_dict["bvalue_idx"] = (
                    "bvalue_idx",
                    bval_coords,
                    {"unit": "s/mm2"},
                )

        # Add inversion_idx coordinates using shared helper function
        if "inversion_idx" in shape_dict:
            n_slices_final = shape_dict["slice_idx"]
            n_inversions_final = shape_dict["inversion_idx"]

            # Use helper function to create slice-dependent inversion coordinate
            coords_dict["inversion_idx"] = _create_inversion_idx_coordinate(
                handler, n_inversions_final, n_slices_final, shape_dict
            )

        # 5. Assign the coordinates to the final DataArray
        # --- Assign Coordinates ---
        if "inversion_idx" in coords_dict:
            required_dims_for_eff_ti = coords_dict["inversion_idx"][
                0
            ]  # ("slice_idx", "inversion_idx")
            if not all(dim in final_da.dims for dim in required_dims_for_eff_ti):
                del coords_dict["inversion_idx"]
            # You might add similar checks for other multi-dimensional coords
            # if needed

        # Assign all prepared coordinates (both dimension and non-dimension)
        # xarray will handle matching them correctly.
        try:
            # Ensure coordinates match the dimensions *present* in final_da
            # before assigning
            coords_to_assign = {}
            for name, coord_data in coords_dict.items():
                # Check if it's a dimension coordinate (name matches a dim)
                if name in final_da.dims:
                    # Check if the coordinate length matches the dimension size
                    if isinstance(
                        coord_data, tuple
                    ):  # Standard (dims, data, attrs) format
                        coord_values = coord_data[1]
                        coord_dims = coord_data[0]
                        if isinstance(coord_dims, str):  # 1D coord
                            if len(coord_values) == final_da.sizes[name]:
                                coords_to_assign[name] = coord_data
                            else:
                                pass
                        else:  # Multi-dimensional coord (like spatial grids)
                            # Basic check if specified dims exist, shape check
                            # might be more complex
                            if all(d in final_da.dims for d in coord_dims):
                                coords_to_assign[name] = coord_data
                            else:
                                pass

                    # Simple case: data array only (assume dim name matches)
                    else:
                        if len(coord_data) == final_da.sizes[name]:
                            coords_to_assign[name] = coord_data
                        else:
                            pass

                # Check if it's a non-dimension coordinate
                elif isinstance(coord_data, tuple) and isinstance(coord_data[0], tuple):
                    coord_dims = coord_data[0]
                    # Check if all dimensions it depends on exist in the final
                    # DataArray
                    if all(dim in final_da.dims for dim in coord_dims):
                        # Optional: Add shape check against final_da.sizes for
                        # those dims
                        expected_shape = tuple(final_da.sizes[d] for d in coord_dims)
                        actual_shape = np.shape(
                            coord_data[1]
                        )  # Shape of the coordinate data array
                        if actual_shape == expected_shape:
                            coords_to_assign[name] = coord_data
                        else:
                            pass
                    else:
                        pass
                else:
                    pass

            final_da = final_da.assign_coords(coords_to_assign)

        except Exception as e:
            id = handler.id if hasattr(handler, "id") else "unknown"
            print(f"================== Error Details [{id}] =================")
            print(f"Error assigning coordinates: {e}")
            print("Coords dictionary contents:")
            for k, v in coords_dict.items():
                if isinstance(v, tuple):
                    print(f"  {k}: dims={v[0]}, shape={np.shape(v[1])}")
                else:
                    print(f"  {k}: shape={np.shape(v)}")
            raise  # Re-raise the exception after printing details

        # Add attributes if needed
        final_da.attrs["slice_orientation"] = slice_orientation
        final_da.attrs["description"] = "MRI data reshaped and transposed"
        final_da.attrs["original_seq2d_shape"] = str(seq2d.shape)
        final_da.attrs["determined_actual_dims_order"] = str(actual_dims_order)
        final_da.attrs["expected_input_order_convention"] = str(
            expected_input_order_unsqueezed
        )
        data_array = self.set_data_array(
            data=final_da.data,
            dims=final_da.dims,
            coords=coords_to_assign,
            handler=handler,
            reco_options=reco_options,
        )
        if append_parameters:
            data_array = self._append_parameters(array=data_array, handler=handler)

        return data_array

    def set_data_array(
        self,
        data: np.ndarray,
        dims: list,
        coords: dict = None,
        handler=None,
        reco_options={},
    ):
        """
        Initialize or update the data_array (2dseq) with given data, dimensions, and coordinates.

        This method creates a new xarray DataArray (.data_array) if one doesn't exist, or updates
        the existing one if it does. It allows for flexible handling of seq2d data
        with different shapes and dimensions.

        Parameters
        ----------
        data : numpy.ndarray
            The 2dseq data to be stored in the xarray.
        dims : list of str
            List of dimension names.
        coords : dict, optional
            Dictionary of coordinates for each dimension. If None, default
            coordinates will be created using np.arange() for each dimension
            based on the shape of the data.
        handler : VendorHandler
            The vendor handler containing necessary attributes for processing.
        reco_options : dict, optional
            Reconstruction options, such as data precision and whether to skip reconstruction.

        Raises
        ------
        ValueError
            If the number of dimensions doesn't match the data shape.

        Notes
        -----
        - If coords is not provided, it will be automatically generated using
          np.arange() based on the shape of the data.
        - This method can be used both for initial creation and subsequent updates of the xarray.
        """
        start_time = time.time()

        # Create default coordinates if not provided
        if coords is None:
            coords = {dim: np.arange(size) for dim, size in zip(dims, data.shape)}

        # Create or update the xarray DataArray
        try:
            data_array = xr.DataArray(data, dims=dims, coords=coords)
        except BaseException:
            raise ValueError(
                f"""
                    Could not create xarray DataArray, shape of:
                    Data: {data.shape},
                    Dimensions: {dims},
                    Coordinates: {coords.sizes}

                """
            )

        x_physical, y_physical, z_physical = get_xyz_grid(data_obj=handler)

        current_spatial_dims = ("read_idx", "phase_idx", "slice_idx")

        data_array = data_array.assign_coords(
            {
                "x": (current_spatial_dims, x_physical),
                "y": (current_spatial_dims, y_physical),
                "z": (current_spatial_dims, z_physical),
            }
        )

        # 5. Optionally, add attributes (like unit or descriptions) to your new
        # coordinates.
        data_array["x"].attrs["unit"] = "mm"
        data_array["x"].attrs["long_name"] = "Physical x-coordinate"
        data_array["x"].attrs["coordinate_system"] = "scanner_physical_space"

        data_array["y"].attrs["unit"] = "mm"
        data_array["y"].attrs["long_name"] = "Physical y-coordinate"
        data_array["y"].attrs["coordinate_system"] = "scanner_physical_space"

        data_array["z"].attrs["unit"] = "mm"
        data_array["z"].attrs["long_name"] = "Physical z-coordinate"
        data_array["z"].attrs["coordinate_system"] = "scanner_physical_space"

        # Add time coordinate if frame_time_s is available
        if (
            "repetition_idx" in data_array.dims
            and hasattr(handler, "frame_time_s")
            and handler.frame_time_s is not None
        ):
            try:
                rep_size = data_array.sizes["repetition_idx"]
                if len(handler.frame_time_s) >= rep_size:
                    time_coords = handler.frame_time_s[:rep_size]
                    logger.debug(
                        f"Adding 'time' coordinate from frame_time_s: "
                        f"shape={time_coords.shape}, range=[{time_coords[0]:.3f}, {time_coords[-1]:.3f}] s"
                    )
                    data_array = data_array.assign_coords(
                        time=("repetition_idx", time_coords)
                    )
                    data_array["time"].attrs["unit"] = "s"
                    data_array["time"].attrs["long_name"] = "Frame acquisition time"
                    data_array["time"].attrs[
                        "description"
                    ] = "Time in seconds for each repetition/frame"
                else:
                    logger.warning(
                        f"frame_time_s length ({len(handler.frame_time_s)}) < repetition_idx size ({rep_size}), "
                        "skipping 'time' coordinate creation"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to add 'time' coordinate from frame_time_s: {e}"
                )

        # Add acquisition_time_s coordinate if acquisition_idx dimension exists
        if (
            "acquisition_idx" in data_array.dims
            and hasattr(handler, "sampling_time_axis_s")
            and handler.sampling_time_axis_s is not None
        ):
            try:
                acq_size = data_array.sizes["acquisition_idx"]
                if len(handler.sampling_time_axis_s) >= acq_size:
                    acq_time_coords = handler.sampling_time_axis_s[:acq_size]
                    logger.debug(
                        f"Adding 'acquisition_time_s' coordinate from sampling_time_axis_s: "
                        f"shape={acq_time_coords.shape}, range=[{acq_time_coords[0]:.6f}, {acq_time_coords[-1]:.6f}] s"
                    )
                    data_array = data_array.assign_coords(
                        acquisition_time_s=("acquisition_idx", acq_time_coords)
                    )
                    data_array["acquisition_time_s"].attrs["unit"] = "s"
                    data_array["acquisition_time_s"].attrs[
                        "long_name"
                    ] = "Acquisition time"
                    data_array["acquisition_time_s"].attrs[
                        "description"
                    ] = "Time in seconds for each acquisition point"
                else:
                    logger.warning(
                        f"sampling_time_axis_s length ({len(handler.sampling_time_axis_s)}) < acquisition_idx size ({acq_size}), "
                        "skipping 'acquisition_time_s' coordinate creation"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to add 'acquisition_time_s' coordinate from sampling_time_axis_s: {e}"
                )

        data_array.attrs["patient_pos"] = handler.patient_pos
        data_array.attrs["slice_orientation"] = handler.slice_orientation
        data_array.attrs["read_orientation"] = handler.read_orientation
        data_array.attrs["num_image_fov_shift"] = 0
        data_array.attrs["read_direction"] = handler.acquisition_matrix[0]
        data_array.attrs["phase_direction"] = handler.acquisition_matrix[1]
        data_array.attrs["slice_direction"] = handler.acquisition_matrix[2]
        data_array.attrs["domain"] = "spatial"
        data_array.attrs["n_slices"] = handler.n_slices
        data_array.attrs["git_commit_hash"] = handler.git_commit_hash

        try:
            data_array["read_idx"].attrs["matrix"] = handler.matrix[0]
            data_array["read_idx"].attrs["resolution_mm"] = handler.resolution_mm[0]
            data_array["read_idx"].attrs["extent_mm"] = handler.rps_extent_mm[0]
            data_array["read_idx"].attrs["axis_mm"] = handler.rps_axes_mm[0]

            data_array["phase_idx"].attrs["matrix"] = handler.matrix[1]
            data_array["phase_idx"].attrs["resolution_mm"] = handler.resolution_mm[1]
            data_array["phase_idx"].attrs["extent_mm"] = handler.rps_extent_mm[1]
            data_array["phase_idx"].attrs["axis_mm"] = handler.rps_axes_mm[1]

            data_array["slice_idx"].attrs["matrix"] = handler.matrix[2]
            data_array["slice_idx"].attrs["resolution_mm"] = handler.resolution_mm[2]
            data_array["slice_idx"].attrs["extent_mm"] = handler.rps_extent_mm[2]
            data_array["slice_idx"].attrs["axis_mm"] = handler.rps_axes_mm[2]

            data_array["repetition_idx"].attrs["matrix"] = handler.n_repetitions
            # Keep axis_s for backward compatibility, but note that the coordinate values
            # themselves now contain the time values in seconds (if available)
            if hasattr(handler, "frame_time_s") and handler.frame_time_s is not None:
                data_array["repetition_idx"].attrs["axis_s"] = handler.frame_time_s
        except BaseException:
            pass

        total_time = time.time() - start_time
        logger.warning(
            f"[TIMING] Brukerseq2dRecoStrategy total reconstruction took {total_time: .3f}s for scan {handler.ExpNum}"
        )

        logger.debug(
            f"Brukerseq2dRecoStrategy returning data_array: {type(data_array)}"
        )
        logger.debug(f"data_array is None: {data_array is None}")
        if data_array is not None:
            logger.debug(
                f"data_array.shape: {getattr( data_array, 'shape', 'no shape')}"
            )

        return data_array


# -----------       BRUKER RAW RECO CLASSES     -----------------
class BrukerFlashRawRecoStrategy(BrukerRawRecoStrategy):
    """Reconstruction strategy for Bruker FLASH data."""

    def reconstruct(self, handler, **reco_options) -> xr.DataArray:
        no_reconstruction = reco_options.get("no_reconstruction", False)
        if no_reconstruction:
            logger.info(
                "[BrukerFlashRawRecoStrategy] No reconstruction performed as per settings."
            )
            return None
        postprocess_fid = reco_options.get("postprocess_fid", False)
        reshape_fid = reco_options.get("reshape_fid", False)
        set_kspace_xarray = reco_options.get("set_kspace_xarray", True)
        shift_xarray = reco_options.get("shift_xarray", True)
        set_reco_data = reco_options.get("set_reco_data", True)
        db = reco_options.get("db", False)  # Changed to True for debugging
        append_parameters = reco_options.get("append_parameters", True)
        DATA_PRECISION = self._parse_data_precision(
            reco_options.get("data_precision", np.float16)
        )

        # Ensure the handler is set to the correct data precision
        handler = self._data_precision(handler=handler, DATA_PRECISION=DATA_PRECISION)

        # Postprocess the FID data for FLASH
        handler.fid_pp = self.postprocess_fid_flash(
            handler=handler,
        )

        reco_array = self.reco_rawdata(
            handler=handler,
            postprocess_fid=postprocess_fid,
            reshape_fid=reshape_fid,
            set_kspace_xarray=set_kspace_xarray,
            shift_xarray=shift_xarray,
            set_reco_data=set_reco_data,
            db=db,
        )

        if append_parameters:
            reco_array = self._append_parameters(array=reco_array, handler=handler)

        return reco_array

    def postprocess_fid_flash(self, handler=None, db=False):
        try:
            if db:
                print(f"postprocess_fid_flash: handler.fid: {handler.fid.shape}")
            # process rawdata:
            fid_pp = self.postprocess_fid_file(input_data=handler.fid, handler=handler)
            if db:
                print(f"postprocess_fid_flash:fid_pp: {fid_pp.shape}")
                print(
                    f"Trying to reshape to:\n{handler.n_repetitions},"
                    f"\nhandler.n_slices={handler.n_slices}",
                    f"\nhandler.n_receivers={handler.n_receivers}",
                    f"\nhandler.enc_matrix={handler.enc_matrix}",
                    f"\nhandler.matrix={handler.matrix}",
                )
                plt.figure()
                plt.imshow(np.log(np.abs(fid_pp.squeeze())), aspect="auto")
            fid_process = fid_pp.copy()
            fid_reordered = np.reshape(
                fid_process,
                (
                    handler.n_repetitions,
                    handler.enc_matrix[1],
                    handler.enc_matrix[2],
                    handler.n_receivers,
                    handler.enc_matrix[0],
                ),
            )
            # reorder to: repetitions x slices x phase x channels x read
            fid_reordered = np.transpose(fid_reordered, (0, 2, 1, 3, 4))
            # print(f"1. postprocess_fid_flash: fid_reordered: {fid_reordered.shape}")
            fid_reordered = np.expand_dims(
                fid_reordered, axis=0
            )  # Add acquisition_idx dimension

            fid_reordered_zeros = np.zeros(
                (
                    fid_reordered.shape[0],  # acquisitions
                    handler.n_repetitions,  # repetitions
                    handler.matrix[2],  # slices
                    handler.matrix[1],  # FULL phase matrix size
                    handler.n_receivers,  # channels
                    handler.matrix[0],  # FULL read matrix size
                ),
                dtype=fid_reordered.dtype,
            )
            # print(f"2. postprocess_fid_flash: fid_reordered: {fid_reordered.shape}")
            # print(
            #     f"2. postprocess_fid_flash: fid_reordered_zeros: {fid_reordered_zeros.shape}"
            # )
            fid_reordered_zeros[
                :,
                :,
                : handler.enc_matrix[2],
                : handler.enc_matrix[1],
                : handler.n_receivers,
                : handler.enc_matrix[0],
            ] = fid_reordered

            handler.fid_pp = fid_reordered_zeros
            handler.kspace = fid_reordered_zeros.copy()
            return fid_reordered_zeros
        except Exception as e:
            traceback.format_exc()
            raise ValueError(
                f"Error in postprocess_fid_flash for ExpNum={handler.ExpNum}: {e}. "
                "Please check the input data and parameters for correctness."
            )


class BrukerRareRawRecoStrategy(BrukerRawRecoStrategy):
    def reconstruct(self, handler, **reco_options) -> xr.DataArray:
        import hypermri.utils.system.logging as logging_utils

        logger = logging_utils.init_default_logger(__name__)

        logger.debug(
            f"BrukerRareRawRecoStrategy.reconstruct starting for scan {handler.ExpNum}"
        )
        no_reconstruction = reco_options.get("no_reconstruction", False)
        if no_reconstruction:
            logger.info(
                "[BrukerRareRawRecoStrategy] No reconstruction performed as per settings."
            )
            return None

        postprocess_fid = reco_options.get("postprocess_fid", False)
        reshape_fid = reco_options.get("reshape_fid", False)
        set_kspace_xarray = reco_options.get("set_kspace_xarray", True)
        shift_xarray = reco_options.get("shift_xarray", True)
        set_reco_data = reco_options.get("set_reco_data", True)
        db = reco_options.get("db", True)  # Changed to True for debugging
        DATA_PRECISION = self._parse_data_precision(
            reco_options.get("data_precision", np.float16)
        )
        append_parameters = reco_options.get("append_parameters", True)

        logger.debug(f"About to call _data_precision")
        # Ensure the handler is set to the correct data precision
        handler = self._data_precision(handler=handler, DATA_PRECISION=DATA_PRECISION)
        logger.debug(f"_data_precision completed")

        try:
            # Postprocess the FID data for RARE
            logger.debug(f"About to call postprocess_fid_rare")
            handler.fid_pp = self.postprocess_fid_rare(
                handler=handler,
            )
            logger.debug(
                f"postprocess_fid_rare completed, handler.fid_pp type: {type( handler.fid_pp)}"
            )

            # Reconstruct the raw data into an xarray DataArray
            logger.debug(f"About to call reco_rawdata")
            reco_array = self.reco_rawdata(
                handler=handler,
                postprocess_fid=postprocess_fid,
                reshape_fid=reshape_fid,
                set_kspace_xarray=set_kspace_xarray,
                shift_xarray=shift_xarray,
                set_reco_data=set_reco_data,
                db=db,
            )
            logger.debug(f"reco_rawdata returned: {type(reco_array)}")
            logger.debug(f"reco_array is None: {reco_array is None}")

            if append_parameters:
                logger.debug(f"About to append parameters")
                reco_array = self._append_parameters(array=reco_array, handler=handler)
                logger.debug(f"_append_parameters completed")

            logger.debug(f"BrukerRareRawRecoStrategy returning: {type(reco_array)}")
            return reco_array

        except Exception as e:
            logger.error(
                f"Exception in BrukerRareRawRecoStrategy.reconstruct: {type(e).__name__}: {e}"
            )
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def postprocess_fid_rare(self, handler=None, db=True):
        """
        Reorders and zero-pads raw FID data from a RARE sequence.

        This function handles both 2D multi-slice and 3D acquisitions, sorts
        the k-space data according to the phase-encoding order, and centers
        the acquired data within the final zero-padded matrix.

        Parameters
        ----------
        handler: VendorHandler
            An object containing metadata for the experiment.
        db: bool, optional
            A flag for debugging purposes (not used in this snippet).

        Returns
        -------
        np.ndarray:
            The reordered and centered k-space data.
        """
        import hypermri.utils.system.logging as logging_utils

        logger = logging_utils.init_default_logger(__name__)

        logger.debug(f"RARE postprocess_fid_rare starting for scan {handler.ExpNum}")
        logger.debug(f"handler type: {type(handler)}")
        logger.debug(f"handler id: {id(handler)}")
        logger.debug(
            f"handler attributes: {[attr for attr in dir(handler) if not attr.startswith('_')]}"
        )
        logger.debug(f"handler has fid attr: {hasattr( handler, 'fid')}")

        if hasattr(handler, "fid"):
            logger.debug(f"handler.fid type: {type(handler.fid)}")
            logger.debug(f"handler.fid is None: {handler.fid is None}")
            if handler.fid is not None:
                logger.debug(f"handler.fid.size: {handler.fid.size}")
                logger.debug(
                    f"handler.fid.shape: {getattr( handler.fid, 'shape', 'no shape attr')}"
                )
        else:
            logger.debug(f"handler does not have 'fid' attribute!")

        # Check if FID data is available
        if not hasattr(handler, "fid"):
            logger.info(
                f"Handler for scan {handler.ExpNum} missing FID data - loading automatically"
            )
            # Try to load the data if it's missing
            try:
                handler.load_data(source=handler.path)
                logger.info(f"Successfully loaded FID data for scan {handler.ExpNum}")
            except Exception as e:
                logger.error(f"Failed to load FID data for scan {handler.ExpNum}: {e}")
                raise ValueError(
                    f"FID data attribute missing for scan {handler.ExpNum} and could not load data. Cannot perform raw RARE reconstruction."
                )

        if handler.fid is None or handler.fid.size == 0:
            logger.error(
                f"Handler for scan {handler.ExpNum} has fid attribute but it's None or empty"
            )
            raise ValueError(
                f"FID data is not available for scan {handler.ExpNum}. Cannot perform raw RARE reconstruction."
            )

        # process rawdata:
        logger.debug(f"About to call postprocess_fid_file with handler.fid")
        fid_pp = self.postprocess_fid_file(input_data=handler.fid, handler=handler)
        logger.debug(
            f"postprocess_fid_file returned: {type(fid_pp)}, shape: {getattr( fid_pp, 'shape', 'no shape')}"
        )

        fid_process = fid_pp.copy()

        fid_process = np.reshape(
            fid_process,
            (-1, handler.rare_factor, handler.n_receivers, handler.matrix[0]),
        )

        pe_order, pe2_order = self._get_phase_encode_order(handler=handler)

        # --- 1. Get essential parameters from the handler ---
        n_reps = handler.n_repetitions
        n_channels = handler.n_receivers
        n_read_encoded = handler.enc_matrix[0]
        n_phase_encoded = handler.enc_matrix[1]
        n_slice_encoded = handler.enc_matrix[2]  # slices or 2nd phase encode

        final_read_size = handler.matrix[0]
        final_phase_size = handler.matrix[1]
        final_slice_size = handler.matrix[2]

        # --- FAIR_RARE Detection and Parameters ---
        n_inversion_time = getattr(handler, "n_inversion_time", 1)
        is_fair_rare = n_inversion_time > 1
        logger.debug(
            f"FAIR_RARE detection: is_fair_rare={is_fair_rare}, n_inversion_time={n_inversion_time}"
        )

        if is_fair_rare:
            logger.info(
                f"FAIR_RARE sequence detected with {n_inversion_time} inversion time points"
            )
            # For FAIR_RARE, the acquisition order includes inversion time loop
            # The total number of acquired data points is n_reps *
            # n_inversion_time

        all_repetitions_data = []

        # --- 2. Process based on spatial dimension (3D or 2D Multi-Slice) ---
        if handler.spat_dim == "3D":
            # --- 3D Acquisition Logic ---
            if is_fair_rare:
                # For FAIR_RARE, include inversion time dimension in the reshaping
                # Acquisition order: RARE echoes → slices → RARE segments →
                # inversion recovery increments → repetitions
                fid_process_by_rep = np.reshape(
                    fid_process,
                    (n_reps * n_inversion_time, -1, n_channels, n_read_encoded),
                )
            else:
                fid_process_by_rep = np.reshape(
                    fid_process, (n_reps, -1, n_channels, n_read_encoded)
                )

            if is_fair_rare:
                # For FAIR_RARE: process data with inversion time dimension
                total_iterations = n_reps * n_inversion_time
                for combined_idx in range(total_iterations):
                    single_combined_data = fid_process_by_rep[combined_idx, ...]
                    fid_3d_reshaped = np.reshape(
                        single_combined_data,
                        (n_slice_encoded, n_phase_encoded, n_channels, n_read_encoded),
                    )
                    fid_3d_reordered_pe2 = fid_3d_reshaped[pe2_order, ...]
                    processed_single_combined = []
                    for slice_idx in range(n_slice_encoded):
                        for channel_idx in range(n_channels):
                            temp_slice_channel_data = fid_3d_reordered_pe2[
                                slice_idx, :, channel_idx, :
                            ]
                            temp_reordered = temp_slice_channel_data[pe_order, :]
                            processed_single_combined.append(temp_reordered)
                    combined_array = np.array(processed_single_combined)
                    combined_array = np.reshape(
                        combined_array,
                        (n_slice_encoded, n_channels, n_phase_encoded, n_read_encoded),
                    )
                    all_repetitions_data.append(combined_array)
            else:
                # Regular RARE processing
                for rep_idx in range(n_reps):
                    single_rep_data = fid_process_by_rep[rep_idx, ...]
                    fid_3d_reshaped = np.reshape(
                        single_rep_data,
                        (n_slice_encoded, n_phase_encoded, n_channels, n_read_encoded),
                    )
                    fid_3d_reordered_pe2 = fid_3d_reshaped[pe2_order, ...]
                    processed_single_rep = []
                    for slice_idx in range(n_slice_encoded):
                        for channel_idx in range(n_channels):
                            temp_slice_channel_data = fid_3d_reordered_pe2[
                                slice_idx, :, channel_idx, :
                            ]
                            temp_reordered = temp_slice_channel_data[pe_order, :]
                            processed_single_rep.append(temp_reordered)
                    rep_array = np.array(processed_single_rep)
                    rep_array = np.reshape(
                        rep_array,
                        (n_slice_encoded, n_channels, n_phase_encoded, n_read_encoded),
                    )
                    all_repetitions_data.append(rep_array)

        else:
            # --- 2D Multi-Slice Acquisition Logic ---
            if is_fair_rare:
                # For FAIR_RARE 2D, adjust block calculation to include
                # inversion time
                total_iterations = n_reps * n_inversion_time
                num_blocks_per_combined = fid_process.shape[0] // total_iterations
                for combined_idx in range(total_iterations):
                    combined_start_idx = combined_idx * num_blocks_per_combined
                    combined_end_idx = (combined_idx + 1) * num_blocks_per_combined
                    single_combined_data = fid_process[
                        combined_start_idx:combined_end_idx, ...
                    ]
                    processed_single_combined = []
                    for slice_idx in range(n_slice_encoded):
                        for channel_idx in range(n_channels):
                            temp_slice_channel_data = single_combined_data[
                                slice_idx::n_slice_encoded, :, channel_idx, :
                            ]
                            temp_reshaped = np.reshape(
                                temp_slice_channel_data, (-1, final_read_size)
                            )
                            temp_reordered = temp_reshaped[pe_order, ...]
                            processed_single_combined.append(temp_reordered)
                    combined_array = np.array(processed_single_combined)
                    combined_array = np.reshape(
                        combined_array,
                        (n_slice_encoded, n_channels, n_phase_encoded, n_read_encoded),
                    )
                    all_repetitions_data.append(combined_array)
            else:
                # Regular 2D Multi-Slice processing
                num_blocks_per_rep = fid_process.shape[0] // n_reps
                for rep_idx in range(n_reps):
                    rep_start_idx = rep_idx * num_blocks_per_rep
                    rep_end_idx = (rep_idx + 1) * num_blocks_per_rep
                    single_rep_data = fid_process[rep_start_idx:rep_end_idx, ...]
                    processed_single_rep = []
                    for slice_idx in range(n_slice_encoded):
                        for channel_idx in range(n_channels):
                            temp_slice_channel_data = single_rep_data[
                                slice_idx::n_slice_encoded, :, channel_idx, :
                            ]
                            temp_reshaped = np.reshape(
                                temp_slice_channel_data, (-1, final_read_size)
                            )
                            temp_reordered = temp_reshaped[pe_order, ...]
                            processed_single_rep.append(temp_reordered)
                    rep_array = np.array(processed_single_rep)
                    rep_array = np.reshape(
                        rep_array,
                        (n_slice_encoded, n_channels, n_phase_encoded, n_read_encoded),
                    )
                    all_repetitions_data.append(rep_array)

        # --- 3. Finalize Data Structure ---
        logger.debug(f"all_repetitions_data length: {len(all_repetitions_data)}")
        if len(all_repetitions_data) > 0:
            logger.debug(
                f"all_repetitions_data[0] type: {type( all_repetitions_data[0])}"
            )
            logger.debug(
                f"all_repetitions_data[0] shape: {getattr( all_repetitions_data[0], 'shape', 'no shape')}"
            )

        fid_reordered = np.stack(all_repetitions_data, axis=0)
        logger.debug(
            f"fid_reordered after stack: type={type(fid_reordered)}, shape={getattr( fid_reordered, 'shape', 'no shape')}"
        )

        if is_fair_rare:
            # For FAIR_RARE: reshape to separate inversion time from repetitions
            # Current shape: (n_reps * n_inversion_time, n_slice_encoded, n_channels, n_phase_encoded, n_read_encoded)
            # Target shape:  (n_reps, n_inversion_time, n_slice_encoded,
            # n_channels, n_phase_encoded, n_read_encoded)
            fid_reordered = np.reshape(
                fid_reordered,
                (
                    n_reps,
                    n_inversion_time,
                    n_slice_encoded,
                    n_channels,
                    n_phase_encoded,
                    n_read_encoded,
                ),
            )
            logger.debug(
                f"FAIR_RARE fid_reordered after reshape: shape={fid_reordered.shape}"
            )

            # Transpose to match expected dimension order: (1, n_reps,
            # n_slice_encoded, n_phase_encoded, n_inversion_time, n_channels,
            # n_read_encoded)
            fid_reordered = np.transpose(fid_reordered, (0, 2, 4, 1, 3, 5))
            fid_reordered = np.expand_dims(fid_reordered, axis=0)
            logger.debug(f"FAIR_RARE fid_reordered final shape: {fid_reordered.shape}")
        else:
            # Regular RARE processing
            fid_reordered = np.transpose(fid_reordered, (0, 1, 3, 2, 4))
            logger.debug(
                f"fid_reordered after transpose: type={type(fid_reordered)}, shape={getattr( fid_reordered, 'shape', 'no shape')}"
            )

            fid_reordered = np.expand_dims(fid_reordered, axis=0)
            logger.debug(
                f"fid_reordered after expand_dims: type={type(fid_reordered)}, shape={getattr( fid_reordered, 'shape', 'no shape')}"
            )

        logger.debug(f"fid_reordered is None: {fid_reordered is None}")
        logger.debug(
            f"fid_reordered.dtype: {getattr( fid_reordered, 'dtype', 'no dtype attr')}"
        )

        # --- 4. Zero-Padding to Final Matrix Size with Centering ---
        if is_fair_rare:
            # For FAIR_RARE: include inversion_idx dimension
            final_shape = (
                1,  # acquisition_idx
                n_reps,  # repetition_idx
                final_slice_size,  # slice_idx
                final_phase_size,  # phase_idx
                n_inversion_time,  # inversion_idx - NEW (key for FAIR_RARE)
                n_channels,  # channel_idx
                final_read_size,  # read_idx
            )
        else:
            # Regular RARE: original shape
            final_shape = (
                1,  # acquisition_idx
                n_reps,  # repetition_idx
                final_slice_size,  # slice_idx
                final_phase_size,  # phase_idx
                n_channels,  # channel_idx
                final_read_size,  # read_idx
            )
        logger.debug(f"About to create zeros with final_shape: {final_shape}")
        logger.debug(f"About to access fid_reordered.dtype: {fid_reordered.dtype}")
        fid_reordered_zeros = np.zeros(final_shape, dtype=fid_reordered.dtype)

        # Calculate padding and starting indices for centering the data
        slice_pad = final_slice_size - n_slice_encoded
        slice_start = slice_pad // 2

        phase_pad = final_phase_size - n_phase_encoded
        phase_start = phase_pad // 2

        read_pad = final_read_size - n_read_encoded
        read_start = read_pad // 2

        # Calculate ending indices
        slice_end = slice_start + n_slice_encoded
        phase_end = phase_start + n_phase_encoded
        read_end = read_start + n_read_encoded

        # Fill the zero-padded array with the reordered data, now centered
        if is_fair_rare:
            # For FAIR_RARE: (1, n_reps, final_slice_size, final_phase_size,
            # n_inversion_time, n_channels, final_read_size)
            fid_reordered_zeros[
                :,  # acquisition_idx
                :,  # repetition_idx
                slice_start:slice_end,  # slice_idx
                phase_start:phase_end,  # phase_idx
                :,  # inversion_idx (full range)
                :,  # channel_idx
                read_start:read_end,  # read_idx
            ] = fid_reordered
        else:
            # Regular RARE: (1, n_reps, final_slice_size, final_phase_size,
            # n_channels, final_read_size)
            fid_reordered_zeros[
                :,  # acquisition_idx
                :,  # repetition_idx
                slice_start:slice_end,  # slice_idx
                phase_start:phase_end,  # phase_idx
                :,  # channel_idx
                read_start:read_end,  # read_idx
            ] = fid_reordered

        handler.fid_pp = fid_reordered_zeros

        return fid_reordered_zeros


class BrukerSpectroRecoStrategy(BrukerRawRecoStrategy):
    """Reconstruction strategy for Bruker Spectroscopy data."""

    def reconstruct(self, handler, **reco_options) -> xr.DataArray:
        # check if no reconstruction is requested
        no_reconstruction = reco_options.get("no_reconstruction", False)
        if no_reconstruction:
            logger.info(
                "[BrukerSpectroRecoStrategy] No reconstruction performed as per settings."
            )
            return None
        postprocess_fid = reco_options.get("postprocess_fid", False)
        reshape_fid = reco_options.get("reshape_fid", False)
        set_kspace_xarray = reco_options.get("set_kspace_xarray", True)
        shift_xarray = reco_options.get("shift_xarray", True)
        set_reco_data = reco_options.get("set_reco_data", True)
        db = reco_options.get("db", True)
        DATA_PRECISION = self._parse_data_precision(
            reco_options.get("data_precision", np.float16)
        )
        append_parameters = reco_options.get("append_parameters", True)

        # Ensure the handler is set to the correct data precision
        handler = self._data_precision(handler=handler, DATA_PRECISION=DATA_PRECISION)

        # Reorder and reshape the FID data for spectroscopy
        handler.fid_pp = self.reorder_reshape_fid(handler)

        # Remove trailing zeros from the FID data (Bruker Default 70 zeros for
        # 13c-spectroscopy)
        handler.fid_pp = self.remove_trailing_zeros(handler)

        # Adjust n_receivers based on the actual shape of fid_pp
        # The channel dimension is at index 4 after reorder_reshape_fid and
        # transpose
        if handler.fid_pp.shape[4] == 1 and handler.n_receivers != 1:
            print(
                f"scan  # {handler.id}: Warning: handler.n_receivers({handler.n_receivers}) does not match actual data channels(1). Adjusting handler.n_receivers to 1."
            )
            handler.n_receivers = 1

        # For spectroscopy data, create kspace_xarray manually with simple coordinates
        # instead of using spatial grids that cause coordinate validation errors
        logger.debug(f"set_kspace_xarray: {set_kspace_xarray}")
        if set_kspace_xarray:
            try:
                # Use handler.fid_pp as kspace data for spectroscopy
                handler.kspace = handler.fid_pp

                # Create simple dimensions for spectroscopy data
                dims = [
                    "acquisition_idx",
                    "repetition_idx",
                    "slice_idx",
                    "phase_idx",
                    "channel_idx",
                    "read_idx",
                ]
                logger.debug(f"handler.kspace.shape: {handler.kspace.shape}")
                # Create simple index-based coordinates instead of spatial grids
                coords = {
                    "acquisition_idx": (
                        "acquisition_idx",
                        np.arange(handler.kspace.shape[0]),
                        {"unit": "index"},
                    ),
                    "repetition_idx": (
                        "repetition_idx",
                        np.arange(handler.kspace.shape[1]),
                        {"unit": "index"},
                    ),
                    "slice_idx": (
                        "slice_idx",
                        np.arange(handler.kspace.shape[2]),
                        {"unit": "index"},
                    ),
                    "phase_idx": (
                        "phase_idx",
                        np.arange(handler.kspace.shape[3]),
                        {"unit": "index"},
                    ),
                    "channel_idx": (
                        "channel_idx",
                        np.arange(handler.kspace.shape[4]),
                        {"unit": "index"},
                    ),
                    "read_idx": (
                        "read_idx",
                        np.arange(handler.kspace.shape[5]),
                        {"unit": "index"},
                    ),
                }

                # Create the kspace xarray with simple coordinates
                kspace_xarray = xr.DataArray(
                    handler.kspace,
                    dims=dims,
                    coords=coords,
                    attrs={"domain": "kspace", "signal_domain": "temporal"},
                )

                try:
                    logger.debug(f"trying to set acquisition_time_s coordinate")
                    if "acquisition_idx" in dims:
                        kspace_xarray = kspace_xarray.assign_coords(
                            {
                                "acquisition_time_s": (
                                    "acquisition_idx",
                                    handler.sampling_time_axis_s,
                                    {"unit": "s"},
                                )
                            }
                        )
                except Exception as e:
                    logger.debug(f"Failed to set acquisition_time_s coordinate: {e}")
                    logger.debug(
                        f"handler.sampling_time_axis_s={len(handler.sampling_time_axis_s)}"
                        f"kspace_xarray.acquisition_idx={len(kspace_xarray.acquisition_idx)}"
                    )
                handler.kspace_xarray = kspace_xarray

            except Exception as e:
                raise ValueError(
                    f"Failed to create kspace_xarray for spectroscopy data: {e}"
                )

        # Call parent's reco_rawdata with kspace_xarray creation disabled
        # since we handled it above
        reco_array = self.reco_rawdata(
            handler=handler,
            postprocess_fid=postprocess_fid,
            reshape_fid=reshape_fid,
            set_kspace_xarray=False,  # Disabled since we created it above
            shift_xarray=shift_xarray,
            set_reco_data=set_reco_data,
            db=db,
        )
        if append_parameters:
            reco_array = self._append_parameters(array=reco_array, handler=handler)

        try:
            axes_idx = reco_array.get_axis_num("acquisition_idx")
            reco_array.data = np.fft.fftshift(
                np.fft.fft(reco_array.data, axis=axes_idx), axes=axes_idx
            )
            reco_array.attrs["signal_domain"] = "spectral"
        except Exception as e:
            raise ValueError(
                f"Error during Fourier transform in BrukerSpectroRecoStrategy: {e}"
            )

        return reco_array

    def remove_trailing_zeros(self, handler):
        """
        Remove trailing zeros from the FID data.
        This is useful for spectroscopy data where trailing zeros may not be meaningful.
        """
        fid_pp = handler.fid_pp
        fid_pp = fid_pp[handler.cut_off : :, ...]
        return fid_pp

    def reorder_reshape_fid(self, handler):
        # Calculate expected size for different channel configurations
        logger.debug(f"handler.fid size: {handler.fid.size}")
        logger.debug(
            f"handler.n_repetitions: {handler.n_repetitions}, handler.n_averages: {handler.n_averages}, handler.n_slices: {handler.n_slices}, handler.n_receivers: {handler.n_receivers}, handler.n_acquisition: {handler.n_acquisition}"
        )
        expected_size_single_channel = (
            handler.n_repetitions
            * handler.n_averages
            * handler.n_slices
            * 1  # Assuming 1 for the two '1' dimensions
            * 1
            * handler.n_acquisition
        )

        expected_size_multi_channel = (
            handler.n_repetitions
            * handler.n_averages
            * handler.n_slices
            * 1
            * 1
            * handler.n_receivers
            * handler.n_acquisition
        )

        original_n_receivers = handler.n_receivers

        # Determine the correct n_receivers based on actual FID size
        if handler.fid.size == expected_size_single_channel:
            handler.n_receivers = 1
        elif handler.fid.size == expected_size_multi_channel:
            # Keep original n_receivers
            pass
        else:
            # Try to infer correct parameters from actual data size
            # Calculate what n_acquisition should be if other params are correct
            remaining_size = handler.fid.size // (
                handler.n_repetitions * handler.n_averages * handler.n_slices * 1 * 1
            )

            # Check if it's a single channel case with different n_acquisition
            if remaining_size % handler.n_receivers == 0:
                # Multi-channel case - keep original n_receivers
                actual_n_acquisition = remaining_size // handler.n_receivers
            else:
                # Single channel case
                handler.n_receivers = 1
                actual_n_acquisition = remaining_size

            # Update n_acquisition if it differs significantly
            if (
                hasattr(handler, "n_acquisition")
                and abs(actual_n_acquisition - handler.n_acquisition) > 1
            ):
                print(
                    f"Warning: Adjusting n_acquisition from {handler.n_acquisition} to {actual_n_acquisition} based on actual FID size"
                )
                handler.n_acquisition = actual_n_acquisition

        try:
            fid = np.reshape(
                handler.fid,
                (
                    handler.n_repetitions,
                    handler.n_averages,
                    handler.n_slices,
                    1,
                    1,
                    handler.n_receivers,  # Use potentially adjusted n_receivers
                    handler.n_acquisition,
                ),
            )
            fid = np.mean(fid, axis=1)
            fid = np.reshape(
                fid,
                (
                    handler.n_repetitions,
                    handler.n_slices,
                    1,
                    1,
                    handler.n_receivers,  # Use potentially adjusted n_receivers
                    handler.n_acquisition,
                ),
            )
        except ValueError as e:
            # If the first reshape fails, try the alternative shape without averages
            try:
                fid = np.reshape(
                    handler.fid,
                    (
                        handler.n_repetitions,
                        handler.n_slices,
                        1,
                        1,
                        handler.n_receivers,  # Use potentially adjusted n_receivers
                        handler.n_acquisition,
                    ),
                )
            except ValueError as e2:
                # Restore original n_receivers before raising error
                handler.n_receivers = original_n_receivers
                raise RuntimeError(
                    f"Singlepulse Reconstruction of raw data failed.\n"
                    f"Please check the input data or the reconstruction strategy.\n"
                    f"FID size: {handler.fid.size}, "
                    f"Expected shape: ({handler.n_repetitions}, {handler.n_averages}, "
                    f"{handler.n_slices}, 1, 1, {handler.n_receivers}, {handler.n_acquisition})\n"
                    f"Original error: {e}\n"
                    f"Alternative error: {e2}"
                )
        finally:
            # Restore original n_receivers after the reshape operation
            handler.n_receivers = original_n_receivers

        fid = np.transpose(fid, (5, 0, 1, 2, 4, 3))
        return fid


class BrukerCsiRecoStrategy(BrukerRawRecoStrategy):
    """Reconstruction strategy for Bruker CSI data."""

    def reconstruct(self, handler, **reco_options) -> xr.DataArray:
        # reorder kspace points
        # check if no reconstruction is requested
        no_reconstruction = reco_options.get("no_reconstruction", False)
        if no_reconstruction:
            logger.info(
                "[BrukerCsiRecoStrategy] No reconstruction performed as per settings."
            )
            return None

        DATA_PRECISION = self._parse_data_precision(
            reco_options.get("data_precision", np.float16)
        )
        handler = self._data_precision(handler=handler, DATA_PRECISION=DATA_PRECISION)
        append_parameters = reco_options.get("append_parameters", True)

        postprocess_fid = reco_options.get("postprocess_fid", False)
        reshape_fid = reco_options.get("reshape_fid", False)
        set_kspace_xarray = reco_options.get("set_kspace_xarray", True)
        shift_xarray = reco_options.get("shift_xarray", True)
        set_reco_data = reco_options.get("set_reco_data", True)
        db = reco_options.get("db", True)
        data_type = reco_options.get("data_type", "job")

        try:
            # sort complex k-space data, write into xarray kspace_xarray:
            kspace_xarray = self.sort_kspace(
                handler=handler, data_type=data_type, db=False, write_to_object=False
            )
            # Extract numpy array for handler.kspace (expected by reco_rawdata)
            handler.kspace = kspace_xarray.values
            # Also store the xarray version for later use
            handler.kspace_xarray = kspace_xarray

        except Exception as e:
            err_msg, err_type = str(e), e.__class__.__name__

            msg = f"[ExpNum={handler.ExpNum}] The following error occured during "
            msg += f"handler.sort_kspace():\n    {err_type}: {err_msg}\n"

            handler.kspace_xarray = None
            raise ValueError(
                f"{msg}\nPlease check the input data and parameters for correctness."
            )

        # Use the reco_rawdata from the super class
        reco_array = self.reco_rawdata(
            postprocess_fid=postprocess_fid,  # reco_options.get("pos False,
            reshape_fid=reshape_fid,  # <- kspace was already set and reshaped
            set_kspace_xarray=set_kspace_xarray,
            shift_xarray=shift_xarray,
            set_reco_data=set_reco_data,
            handler=handler,
            db=db,
        )
        if append_parameters:
            reco_array = self._append_parameters(array=reco_array, handler=handler)
        return reco_array

    def sort_kspace(
        self,
        handler,
        data_type="job",
        cut_off=0,
        db=False,
        input_data=None,
        write_to_object=False,
    ):
        """
        Reconstruct a complex dataset comparable to the `2dseq` file from Paravision.

        This function reads raw data from either a `.job` or `.fid` file, processes it,
        and organizes it into a structured k-space representation. It supports 2D and 3D
        datasets and handles different phase encoding schemes (e.g., linear, centric, or custom).

        Parameters
        ----------
        data_type : str, optional
            Specifies the type of raw data file to read. Valid options are:

            - "job" (default): Read from the `.job` file.
            - "fid": Read from the `.fid` file.

        cut_off : int, optional
            Number of entries to cut off from the beginning of each FID (Free Induction Decay).
            Default is `0`.
        db : bool, optional
            If `True`, enables debug mode, which includes additional logging and plotting
            for troubleshooting. Default is `False`.
        input_data : np.ndarray, optional
            Optional input data array. If provided, the function skips reading from the
            `.job` or `.fid` file and uses this data instead. Default is `None`.
        write_to_object : bool, optional
            If `True`, the processed k-space data is written to the object's internal storage.
            If `False`, the data is returned without modifying the object. Default is `True`.

        Returns
        -------
        kspace_xarray : xarray.DataArray
            A structured xarray object containing the sorted k-space data. The dimensions are:

            - `acquisition_idx`: Index of the acquisition (FID points).
            - `slice_idx`: Index of the slice (for 3D datasets).
            - `read_idx`: Index of the readout direction.
            - `phase_idx`: Index of the phase encoding direction.
            - `repetition_idx`: Index of the repetition (for multi-averaged datasets).
            - `channel_idx`: Index of the receiver channel (for multi-channel data).

        Raises
        ------
        Warning
            If the specified `data_type` is not recognized or if the input data cannot be loaded.
        NotImplementedError
            If an unsupported encoding scheme is encountered.

        Notes
        -----
        1. Supported encoding schemes include linear and centric encoding. Custom schemes
        (e.g., spiral) are partially implemented but may require further testing.
        2. Debug mode (`db=True`) is useful for troubleshooting and visualizing the k-space
        sorting process. It provides scatter plots of the sampling positions and logs
        key intermediate steps.
        3. The `cut_off` parameter allows users to discard the initial points of each FID,
        which may contain artifacts or no signal.

        """

        # Use binned_statistic_2d to sort the 1D data into a 2D grid based on
        # the x and y coordinates
        from scipy.stats import binned_statistic_2d

        if input_data is None:
            if data_type == "job":
                try:
                    fid = handler.rawdatajob0
                    logger.debug("loaded job file")
                except BaseException:
                    logger.warning("couldn't load job file, trying to load .fid file")
                    fid = handler.fid
                    logger.debug("loaded fid file")
            elif data_type == "fid":
                try:
                    fid = self.fid
                    logger.debug("loaded fid file")
                except BaseException:
                    logger.warning("couldn't load fid file, trying to load .job file")
                    try:
                        fid = handler.rawdatajob0
                        logger.debug("loaded job file")
                    except BaseException:
                        logger.warning("loading fid does not work")
            else:
                logger.warning(
                    "unknown data_type parameter, trying to continue with .job file"
                )
                try:
                    fid = handler.rawdatajob0
                    logger.debug("loaded job file")
                except BaseException:
                    try:
                        fid = handler.rawdatajob0
                        logger.debug("loaded job file")
                    except BaseException:
                        logger.warning("loading fid does not work")

        else:
            fid = input_data
        # planned order is fid-y-x-z-nr-nchannels

        n_repetitions = handler.n_repetitions
        n_read = handler.matrix[0]
        n_phase = handler.matrix[1]

        # check if 3D or 2D:
        n_slices = handler.n_slices

        n_acquisition = handler.n_acquisition
        n_receivers = handler.n_receivers

        # get CSI phase  info
        enc_order, read_idx, phase_idx = self.get_phase_encoding_order(handler=handler)

        # init empty array:
        kspace_array = np.zeros(
            (n_acquisition, n_phase, n_read, n_slices, n_repetitions, n_receivers)
        )

        read_orient = handler.read_orientation
        data = fid.copy()

        # reshape fid file
        data = np.reshape(
            data,
            ((n_repetitions, n_slices, n_phase, n_read, n_receivers, n_acquisition)),
        )
        data = np.transpose(data, (5, 1, 2, 3, 0, 4))

        data = np.reshape(
            data,
            (n_acquisition, n_slices, n_read * n_phase, n_repetitions, n_receivers),
        )

        # get sampling phase encode indices:
        read_idx = list(read_idx.T + 0.5)
        phase_idx = list(phase_idx.T + 0.5)

        # Define bins of grid:
        read_bins = list(range(n_read + 1))
        phase_bins = list(range(n_phase + 1))

        # init empty array:
        kspace_array = np.zeros(
            (n_read, n_phase, n_acquisition, n_repetitions, n_slices, n_receivers),
            dtype=data.dtype,
        )

        # now reco differs in case linear or centric is used
        if enc_order == "LINEAR_ENC LINEAR_ENC":
            # do linear encoding reco
            logger.debug("Encoding order", enc_order)
            if handler.n_receivers > 1:
                logger.debug(
                    "This is dual channel data, performing phasing first, then csi reco..."
                )
                # do phasing
            else:
                logger.debug("This is single channel data")

                # If nr is 1, add an extra dimension to data
                if n_repetitions == 1:
                    data = data[..., np.newaxis]

                if db is True:
                    from matplotlib.ticker import MultipleLocator

                    # # Calculate statistics
                    statistic, read_edge, phase_edge, bin_number = binned_statistic_2d(
                        read_idx, phase_idx, None, "count", bins=[read_bins, phase_bins]
                    )

                    plt.figure(figsize=(12, 5))
                    ax = plt.subplot(1, 1, 1)
                    # ax.set_xlim(np.min(binx),np.max(binx))
                    # ax.set_ylim(np.min(biny),np.max(biny))
                    ax.grid(which="both")
                    ax.xaxis.set_major_locator(MultipleLocator(1))
                    ax.yaxis.set_major_locator(MultipleLocator(1))
                    ax.scatter(read_idx, phase_idx)

                for ic in range(n_receivers):  # number of channels
                    for iz in range(n_slices):  # 3rd dimension
                        for ia in range(n_acquisition):  # FID points
                            for ir in range(n_repetitions):  # repetitions
                                (
                                    statistic,
                                    read_edge,
                                    phase_edge,
                                    bin_number,
                                ) = binned_statistic_2d(
                                    read_idx,
                                    phase_idx,
                                    np.squeeze(data[ia, iz, :, ir, ic]),
                                    "mean",
                                    bins=[read_bins, phase_bins],
                                )
                                kspace_array[:, :, ia, ir, iz, ic] = np.reshape(
                                    np.squeeze(statistic), [n_read, n_phase]
                                )

                kspace_array = np.transpose(kspace_array, [2, 4, 0, 1, 3, 5])

                kspace_array = np.roll(
                    np.roll(kspace_array, shift=1, axis=2), shift=1, axis=3
                )

        elif enc_order == "CENTRIC_ENC CENTRIC_ENC":
            logger.critical(
                f"CSI enc_order = {enc_order} has not been properly reviewed!"
            )
            # do centric encoding reco
            if handler.n_receivers > 1:
                logger.debug(
                    "This is dual channel data, performing phasing first, then csi reco..."
                )

            else:
                logger.debug(
                    "We don't have dual channel data here, proceeding with CSI reco"
                )
                # this now applies to measurement where we don't have two channels
                # e.g. 31mm coil measurements
                # reorder k space according to sampling
                old_matrix_coords = np.zeros((n_phase, n_read))

                if read_orient == "L_R":
                    # Create a mesh grid
                    read_grid, phase_grid = np.meshgrid(read_idx, phase_idx)
                    # Flatten the grids to get 1D arrays
                    read_idx = read_grid.flatten()
                    phase_idx = phase_grid.flatten()

                    if db is True:
                        logger.critical(f"sampling position x: {read_idx}")
                        logger.critical(f"sampling position y: {phase_idx}")

                        logger.critical(f"bin edge x: {read_bins}")
                        logger.critical(f"bin edge y: {phase_bins}")

                        logger.critical(f"data.shape = {data.shape}")
                        logger.critical(f"k_space_array.shape = {kspace_array.shape}")

                        from matplotlib.ticker import MultipleLocator

                        # Calculate statistics
                        (
                            statistic,
                            read_edge,
                            phase_edge,
                            bin_number,
                        ) = binned_statistic_2d(
                            read_idx,
                            phase_idx,
                            None,
                            "count",
                            bins=[read_bins, phase_bins],
                        )

                        print(statistic)
                        plt.figure(figsize=(12, 5))
                        ax = plt.subplot(1, 1, 1)
                        # ax.set_xlim(np.min(binx),np.max(binx))
                        # ax.set_ylim(np.min(biny),np.max(biny))
                        ax.grid(which="both")
                        ax.xaxis.set_major_locator(MultipleLocator(1))
                        ax.yaxis.set_major_locator(MultipleLocator(1))
                        ax.scatter(read_idx, phase_idx)

                    for ic in range(n_receivers):  # number of channels
                        for iz in range(n_slices):  # 3rd dimension
                            for ia in range(n_acquisition):
                                for ir in range(n_repetitions):  # repetitions
                                    (
                                        statistic,
                                        read_edge,
                                        phase_edge,
                                        bin_number,
                                    ) = binned_statistic_2d(
                                        read_idx,
                                        phase_idx,
                                        np.squeeze(data[ia, iz, :, ir, ic]),
                                        "mean",
                                        bins=[read_bins, phase_bins],
                                    )

                                    kspace_array[:, :, ia, ir, iz, ic] = np.reshape(
                                        np.squeeze(statistic), [n_read, n_phase]
                                    )

                elif read_orient == "A_P":
                    c = 0
                    for ny in phase_idx:
                        for nx in read_idx:
                            old_matrix_coords[ny, nx] = c
                            c += 1

                    # extract fids from dataset , i.e. cut
                    get_fids = []
                    for n in np.arange(0, handler.rawdatajob0.shape[0], n_acquisition):
                        # transform the long FID into an array were every entry
                        # has the FID of a certain pixel
                        get_fids.append(
                            handler.rawdatajob0[cut_off + n : n + n_acquisition]
                        )
                    # make an array
                    # update fidsz in case we decide to loose the first 70 entries of each fid as there is
                    # no signal there
                    n_acquisition = n_acquisition - cut_off
                    get_fids = np.array(get_fids)

                    for idx in np.arange(0, get_fids.shape[0], 1):
                        placement_idx = np.where(old_matrix_coords == idx)
                        # found the index where each fid needs to go
                        # the first one in the rawdata file is the center of k space
                        # and so on outwards
                        nx = placement_idx[0][0]
                        ny = placement_idx[1][0]
                        # possibly change the dimensions of k space array
                        kspace_array[:, nx, ny] = get_fids[int(idx), :]
                    kspace_array = np.transpose(kspace_array, [0, 2, 1])
                else:
                    logger.debug(
                        read_orient,
                        " Orientation not know, no k-space reordering performed",
                    )
                logger.debug("k-space_shape", kspace_array.shape)
                # to match echoes - z - x - y - reps - chans
                kspace_array = np.transpose(kspace_array, [2, 4, 0, 1, 3, 5])

        elif enc_order == "luca_centric":
            # If nr is 1, add an extra dimension to d
            if n_repetitions == 1:
                data = data[..., np.newaxis]

            if db is True:
                from matplotlib.ticker import MultipleLocator

                # # Calculate statistics
                statistic, read_edge, phase_edge, bin_number = binned_statistic_2d(
                    read_idx, phase_idx, None, "count", bins=[read_bins, phase_bins]
                )

                print(statistic)
                plt.figure(figsize=(12, 5))
                ax = plt.subplot(1, 1, 1)
                # ax.set_xlim(np.min(binx),np.max(binx))
                # ax.set_ylim(np.min(biny),np.max(biny))
                ax.grid(which="both")
                ax.xaxis.set_major_locator(MultipleLocator(1))
                ax.yaxis.set_major_locator(MultipleLocator(1))
                ax.scatter(read_idx, phase_idx)
            for ic in range(n_receivers):  # number of channels
                for iz in range(n_slices):  # 3rd dimension
                    for ia in range(n_acquisition):
                        for ir in range(n_repetitions):  # repetitions
                            (
                                statistic,
                                read_edge,
                                phase_edge,
                                bin_number,
                            ) = binned_statistic_2d(
                                read_idx,
                                phase_idx,
                                np.squeeze(data[ia, iz, :, ir, ic]),
                                "mean",
                                bins=[read_bins, phase_bins],
                            )
                            kspace_array[:, :, ia, ir, iz, ic] = np.reshape(
                                np.squeeze(statistic), [n_read, n_phase]
                            )
            kspace_array = np.transpose(kspace_array, [2, 4, 0, 1, 3, 5])

        elif enc_order == "luca_spiral":
            logger.critical(f"Encoding order {enc_order} used, not tested yet!!!")

            # If nr is 1, add an extra dimension to d
            if n_repetitions == 1:
                data = data[..., np.newaxis]

            if db is True:
                from matplotlib.ticker import MultipleLocator

                # # Calculate statistics
                statistic, read_edge, phase_edge, bin_number = binned_statistic_2d(
                    read_idx, phase_idx, None, "count", bins=[read_bins, phase_bins]
                )
                plt.figure(figsize=(12, 5))
                ax = plt.subplot(1, 1, 1)
                ax.scatter(read_idx, phase_idx)
            for ic in range(n_receivers):  # number of channels
                for iz in range(n_slices):  # 3rd dimension
                    for ia in range(n_acquisition):
                        for ir in range(n_repetitions):  # repetitions
                            (
                                statistic,
                                read_edge,
                                phase_edge,
                                bin_number,
                            ) = binned_statistic_2d(
                                read_idx,
                                phase_idx,
                                np.squeeze(data[ia, iz, :, ir, ic]),
                                "mean",
                                bins=[read_bins, phase_bins],
                            )
                            kspace_array[:, :, ia, ir, iz, ic] = np.reshape(
                                np.squeeze(statistic), [n_read, n_phase]
                            )
            kspace_array = np.transpose(kspace_array, [2, 4, 0, 1, 3, 5])

        else:
            logger.critical(f"Encoding order {enc_order} used, not implemented yet")

        logger.debug(f"kspace_array.shape after sorting: {kspace_array.shape}")
        logger.debug(f"handler.sampling_time_axis_s={handler.sampling_time_axis_s}")

        # remove trailing zeros:
        kspace_array = kspace_array[cut_off:, ...]
        kspace_array = np.transpose(kspace_array, [0, 4, 1, 3, 5, 2])
        dims = [
            "acquisition_idx",
            "repetition_idx",
            "slice_idx",
            "phase_idx",
            "channel_idx",
            "read_idx",
        ]
        coords = {
            "acquisition_idx": handler.sampling_time_axis_s,
            "repetition_idx": np.array(np.arange(handler.n_repetitions)),
            "slice_idx": (
                [
                    "read_idx",
                    "phase_idx",
                    "slice_idx",
                ],
                handler.rps_grid_mm[2],
            ),
            "phase_idx": (
                [
                    "read_idx",
                    "phase_idx",
                    "slice_idx",
                ],
                handler.rps_grid_mm[1],
            ),
            "channel_idx": np.array(np.arange(handler.n_receivers)),
            "read_idx": (
                [
                    "read_idx",
                    "phase_idx",
                    "slice_idx",
                ],
                handler.rps_grid_mm[0],
            ),
        }
        # generate "empty" xarray:
        kspace_xarray = self.set_kspace_xarray(
            handler=handler,
            data=kspace_array,
            dims=dims,
            coords=coords,
            write_to_object=False,
        )

        return kspace_xarray

    def get_kspace_order_matrix(self, handler):
        """
        Extracts a matrix containing the index of the order of the k-space points as their values,
        specifically for centric encoding.

        This function uses the encoding order information to create a matrix where each
        element represents the order in which that k-space point was sampled in a centric
        encoding scheme.

        Returns
        -------
        order_matrix : numpy.ndarray
            A 2D numpy array where each element is the index of when that k-space point
            was sampled. The shape of the matrix is (n_phase, n_read).

        Notes
        -----
        This function assumes centric encoding and that the `get_encoding_order` method
        returns indices representing the distance from the center of k-space.

        Examples
        --------
        >>> order_matrix = self.extract_kspace_order_matrix()
        >>> plt.imshow(order_matrix)
        >>> plt.colorbar()
        >>> plt.title('K-space Sampling Order (Centric Encoding)')
        >>> plt.show()
        """
        # Get encoding order information
        enc_order, read_idx, phase_idx = self.get_phase_encoding_order(handler=handler)

        # Get matrix dimensions
        n_read = handler.matrix[0]
        n_phase = handler.matrix[1]

        # Initialize the order matrix
        order_matrix = np.zeros((n_phase, n_read), dtype=int)
        c = 0
        for i, p in enumerate(phase_idx):
            for j, r in enumerate(read_idx):
                order_matrix[p, r] = c
                c += 1

        return order_matrix

    def get_phase_encoding_order(self, handler):
        """

        Returns
        -------
        encoding order (string), x_indices, y_indices
        """
        # get dimensions:
        ysz = handler.method["PVM_Matrix"][1]
        xsz = handler.method["PVM_Matrix"][0]

        # check for encoding type:
        if (
            handler.method["Method"] == "<Bruker:CSI>"
            or handler.method["Method"] == "<User:csi_fix>"
        ):
            enc_order = handler.method["PVM_EncOrder"]
            if enc_order == "LINEAR_ENC LINEAR_ENC":
                # for later reordering into an 2D array:
                x_indices = handler.method["PVM_EncSteps0"] + int(np.floor(xsz / 2.0))
                y_indices = handler.method["PVM_EncSteps1"] + int(np.floor(ysz / 2.0))
                # for later reordering into an 2D array:
                # reorder k space according to sampling

                x_indices = handler.method["PVM_EncSteps0"] + int(np.floor(xsz / 2.0))
                y_indices = handler.method["PVM_EncSteps1"] + int(np.floor(ysz / 2.0))
                # print(f"x_indices = {x_indices}")
                # Create a mesh grid
                x_grid, y_grid = np.meshgrid(x_indices, y_indices)

                # print(f"x_grid = {x_grid}")
                # Flatten the grids to get 1D arrays
                x_indices = x_grid.flatten()
                # print(f"x_indices = {x_indices}")
                y_indices = y_grid.flatten()
            elif enc_order == "CENTRIC_ENC CENTRIC_ENC":
                # for later reordering into an 2D array:
                # reorder k space according to sampling

                x_indices = handler.method["PVM_EncSteps0"] + int(np.floor(xsz / 2.0))
                y_indices = handler.method["PVM_EncSteps1"] + int(np.floor(ysz / 2.0))
                # Create a mesh grid
                x_grid, y_grid = np.meshgrid(x_indices, y_indices)

                # Flatten the grids to get 1D arrays
                x_indices = x_grid.flatten()
                y_indices = y_grid.flatten()

        elif handler.method["Method"] in [
            "<User:lucaCSI>",
            "<User:lucaCSI2>",
            "<User:lucaCSI3>",
            "<User:lucaCSI4>",
        ]:
            if handler.method["CentricEncOrder_OnOff"] == "On":
                enc_order = "luca_centric"
                y_indices = handler.method["CentricEncOrderMatrixy"]
                x_indices = handler.method["CentricEncOrderMatrixx"]
                if np.min(y_indices) < 0:
                    y_indices = y_indices + np.abs(np.min(y_indices))
                    x_indices = x_indices + np.abs(np.min(x_indices))
            elif handler.method["SpiralEncOrder_OnOff"] == "On":
                enc_order = "luca_spiral"
                y_indices = handler.method["SpiralEncOrderMatrixy"]
                x_indices = handler.method["SpiralEncOrderMatrixx"]
                if np.min(y_indices) < 0:
                    y_indices = y_indices + np.abs(np.min(y_indices))
                    x_indices = x_indices + np.abs(np.min(x_indices))
            else:
                enc_order = handler.method["PVM_EncOrder"]
                logger.warning(f"Can't determine encoding method, assuming {enc_order}")

                if enc_order == "LINEAR_ENC LINEAR_ENC":
                    # for later reordering into an 2D array:
                    x_indices = handler.method["PVM_EncSteps0"] + int(
                        np.floor(xsz / 2.0)
                    )
                    y_indices = handler.method["PVM_EncSteps1"] + int(
                        np.floor(ysz / 2.0)
                    )
                    # for later reordering into an 2D array:
                    # reorder k space according to sampling

                    x_indices = handler.method["PVM_EncSteps0"] + int(
                        np.floor(xsz / 2.0)
                    )
                    y_indices = handler.method["PVM_EncSteps1"] + int(
                        np.floor(ysz / 2.0)
                    )
                    # print(f"x_indices = {x_indices}")
                    # Create a mesh grid
                    x_grid, y_grid = np.meshgrid(x_indices, y_indices)

                    # print(f"x_grid = {x_grid}")
                    # Flatten the grids to get 1D arrays
                    x_indices = x_grid.flatten()
                    # print(f"x_indices = {x_indices}")
                    y_indices = y_grid.flatten()
                elif enc_order == "CENTRIC_ENC CENTRIC_ENC":
                    # for later reordering into an 2D array:
                    # reorder k space according to sampling

                    x_indices = handler.method["PVM_EncSteps0"] + int(
                        np.floor(xsz / 2.0)
                    )
                    y_indices = handler.method["PVM_EncSteps1"] + int(
                        np.floor(ysz / 2.0)
                    )
                    # Create a mesh grid
                    x_grid, y_grid = np.meshgrid(x_indices, y_indices)

                    # Flatten the grids to get 1D arrays
                    x_indices = x_grid.flatten()
                    y_indices = y_grid.flatten()

        # same as stock
        elif handler.method["Method"] == "<User:geoffCSI>":
            enc_order = handler.method["PVM_EncOrder"]
            # for later reordering into an 2D array:
            y_indices = handler.method["PVM_EncSteps1"] + int(np.floor(ysz / 2.0))
            x_indices = handler.method["PVM_EncSteps0"] + int(np.floor(xsz / 2.0))
        else:
            logger.warning("Can't determine encoding method")
            enc_order = None
            x_indices = np.linspace(0, xsz - 1, xsz)
            y_indices = np.linspace(0, ysz - 1, ysz)
        return enc_order, x_indices, y_indices


# ---------------------------------------------------------------
# ----------------       SIEMENS RECO     -----------------------
# ---------------------------------------------------------------
class SiemensRawRecoStrategy(BaseRecoStrategy):
    """Reconstruction strategy for raw Siemens data."""

    def reconstruct(self, handler, **reco_options) -> xr.DataArray:
        """
        Perform the reconstruction for Siemens raw data.
        This is currently a placeholder and needs to be implemented.
        """
        print("SiemensRawRecoStrategy.reconstruct called")
        # Placeholder for Siemens reconstruction logic.
        # This will involve processing the data from handler.twix
        # and converting it into an xarray.DataArray.
        raise NotImplementedError(
            "Siemens raw data reconstruction is not yet implemented."
        )


class SiemensCsiRecoStrategy(BaseRecoStrategy):
    """Reconstruction strategy for Siemens CSI data."""

    def reconstruct(self, handler, reco_options: dict) -> xr.DataArray:
        # 1. For Siemens, the phase-encoding is standard. No reordering needed.
        k_space = handler

        # 2. Perform Fourier Transform
        image_data = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(k_space)))

        # 3. Create and return the xarray.DataArray
        # ... (xarray creation logic) ...

        # For demonstration, returning a placeholder
        return xr.DataArray(np.abs(image_data), dims=("z", "y", "x"), name="reco_image")


class AgilentSpectroRecoStrategy(BaseRecoStrategy):
    """Reconstruction strategy for Agilent/Varian spectroscopy data."""

    def reconstruct(self, handler, **kwargs) -> xr.DataArray:
        if getattr(handler, "fid", None) is None:
            handler.load_data(getattr(handler, "path", None))

        fid = getattr(handler, "fid", None)
        if fid is None:
            raise ValueError("Agilent handler has no FID loaded")

        fid = self._ensure_fid_shape(np.asarray(fid))
        repetitions, channels, original_points = fid.shape
        debug_log_recon_strategy(message=f"kwargs: {kwargs}")
        debug_log_recon_strategy(message=f"fid.shape: {fid.shape}")
        append_parameters = kwargs.get("append_parameters", True)
        processing_log = []

        digital_filter_points = kwargs.get(
            "digital_filter_points",
            getattr(handler, "digital_filter_points", 0) or 0,
        )
        apply_digital_filter = kwargs.get("apply_digital_filter", True)
        remove_dc = kwargs.get("remove_dc", True)

        apodization = kwargs.get("apodization")
        lb_hz = kwargs.get(
            "line_broadening_hz",
            getattr(handler, "line_broadening_hz", 0.0) or 0.0,
        )
        gb_hz = kwargs.get(
            "gaussian_broadening_hz",
            getattr(handler, "gaussian_broadening_hz", 0.0) or 0.0,
        )

        zero_fill_to = kwargs.get("zero_fill_to")
        zero_fill_factor = kwargs.get("zero_fill_factor")
        if zero_fill_to is None and zero_fill_factor is None:
            default_zf = getattr(handler, "zero_fill_points", None)
            if default_zf:
                zero_fill_to = int(default_zf)

        combine_channels = kwargs.get("combine_channels", True)
        retain_channel_axis = kwargs.get("retain_channel_axis", False)
        combine_method = kwargs.get("combine_method", "rss")

        apply_phasing = kwargs.get("apply_phasing", False)
        phase0_deg = kwargs.get(
            "phase0_deg", getattr(handler, "phase0_deg", 0.0) or 0.0
        )
        phase1_deg = kwargs.get(
            "phase1_deg", getattr(handler, "phase1_deg", 0.0) or 0.0
        )

        dwell_time = getattr(handler, "dwell_time_s", None)
        if dwell_time is None or dwell_time == 0:
            bandwidth = getattr(handler, "bandwidth_hz", None)
            dwell_time = 1.0 / bandwidth if bandwidth else 1.0

        fid, sampling_time_axis_s = self._apply_digital_filter(
            fid,
            dwell_time,
            digital_filter_points,
            apply_digital_filter,
            processing_log,
        )

        fid = self._maybe_remove_dc(fid, remove_dc, processing_log)

        fid = self._apply_apodization(
            fid,
            dwell_time,
            apodization,
            lb_hz,
            gb_hz,
            processing_log,
        )

        fid, sampling_time_axis_s = self._apply_zero_fill(
            fid,
            sampling_time_axis_s,
            dwell_time,
            zero_fill_to,
            zero_fill_factor,
            processing_log,
        )

        spectral = np.fft.fftshift(np.fft.fft(fid, axis=-1), axes=-1)

        if apply_phasing and (phase0_deg or phase1_deg):
            spectral = self._apply_phase_correction(
                spectral, phase0_deg=phase0_deg, phase1_deg=phase1_deg
            )
            processing_log.append(
                f"phase_correction: rp={phase0_deg:.2f}deg lp={phase1_deg:.2f}deg"
            )

        spectral, channel_axis_retained = self._combine_channels(
            spectral,
            combine_channels=combine_channels,
            retain_axis=retain_channel_axis,
            combine_method=combine_method,
            processing_log=processing_log,
        )

        freq_axis_hz = getattr(handler, "freq_axis_hz", None)
        freq_axis_ppm = getattr(handler, "freq_axis_ppm", None)
        if freq_axis_hz is None and freq_axis_ppm is None:
            freq_axis_hz, freq_axis_ppm = self._build_frequency_axes(
                spectral.shape[-1],
                dwell_time,
                handler,
            )

        dims = ["repetition_idx"]
        coords = {
            "repetition_idx": (
                "repetition_idx",
                np.arange(spectral.shape[0]),
                {"unit": "index"},
            ),
        }

        if channel_axis_retained:
            dims.append("channel_idx")
            coords["channel_idx"] = (
                "channel_idx",
                np.arange(spectral.shape[1]),
                {"unit": "index"},
            )

        dims.append("acquisition_idx")
        coords["acquisition_idx"] = (
            "acquisition_idx",
            np.arange(spectral.shape[-1]),
            {"unit": "index"},
        )

        if freq_axis_hz is not None:
            coords["frequency_axis_hz"] = (
                "acquisition_idx",
                freq_axis_hz,
                {"unit": "Hz"},
            )

        if freq_axis_ppm is not None:
            coords["freq_axis_ppm"] = (
                "acquisition_idx",
                freq_axis_ppm,
                {"unit": "ppm"},
            )

        if sampling_time_axis_s is not None:
            coords["acquisition_time_s"] = (
                "acquisition_idx",
                sampling_time_axis_s,
                {"unit": "s"},
            )

        spectral_attrs = {
            "domain": "spectral",
            "signal_domain": "spectral",
            "bandwidth_hz": getattr(handler, "bandwidth_hz", None),
            "dwell_time_s": dwell_time,
            "b0_tesla": getattr(handler, "b0_tesla", None),
            "nucleus": getattr(handler, "nucleus", None),
            "channel_combination": combine_method if combine_channels else "none",
            "processing_log": processing_log,
            "sequence_variant": getattr(handler, "sequence_variant", None),
        }

        if sampling_time_axis_s is not None:
            spectral_attrs["time_axis_s"] = sampling_time_axis_s

        reco_data = xr.DataArray(
            spectral,
            dims=tuple(dims),
            coords=coords,
            attrs=spectral_attrs,
            name="agilent_spectrum",
        )

        if append_parameters:
            reco_data = self._append_parameters(array=reco_data, handler=handler)

        handler.set_primary_data(reco_data, "reco_data")
        return reco_data

    @staticmethod
    def _ensure_fid_shape(fid: np.ndarray) -> np.ndarray:
        if fid.ndim == 1:
            fid = fid[np.newaxis, np.newaxis, :]
        elif fid.ndim == 2:
            fid = fid[np.newaxis, :, :]
        elif fid.ndim > 3:
            fid = fid.reshape(fid.shape[0], fid.shape[1], -1)
        return fid.astype(np.complex64, copy=False)

    def _apply_digital_filter(
        self,
        fid: np.ndarray,
        dwell_time: float,
        lsfid_points: int,
        apply_filter: bool,
        processing_log,
    ):
        if apply_filter and lsfid_points:
            if lsfid_points >= fid.shape[-1]:
                raise ValueError("Digital filter removal would drop all Agilent points")
            fid = fid[..., lsfid_points:]
            processing_log.append(
                f"digital_filter: removed first {lsfid_points} points"
            )

        time_axis = np.arange(fid.shape[-1]) * dwell_time
        return fid, time_axis

    @staticmethod
    def _maybe_remove_dc(
        fid: np.ndarray, remove_dc: bool, processing_log
    ) -> np.ndarray:
        if remove_dc:
            fid = fid - fid.mean(axis=-1, keepdims=True)
            processing_log.append("dc_offset_removed")
        return fid

    def _apply_apodization(
        self,
        fid: np.ndarray,
        dwell_time: float,
        apodization,
        lb_hz: float,
        gb_hz: float,
        processing_log,
    ) -> np.ndarray:
        window, mode = self._build_apodization_window(
            fid.shape[-1], dwell_time, apodization, lb_hz, gb_hz
        )
        if window is not None:
            fid = fid * window
            processing_log.append(f"apodization:{mode}")
        return fid

    def _apply_zero_fill(
        self,
        fid: np.ndarray,
        time_axis: np.ndarray,
        dwell_time: float,
        zero_fill_to,
        zero_fill_factor,
        processing_log,
    ):
        target_points = fid.shape[-1]
        if zero_fill_to is not None:
            target_points = max(target_points, int(zero_fill_to))
        elif zero_fill_factor is not None:
            factor = float(zero_fill_factor)
            if factor < 1.0:
                raise ValueError("zero_fill_factor must be >= 1.0")
            target_points = int(np.ceil(fid.shape[-1] * factor))

        if target_points > fid.shape[-1]:
            pad = target_points - fid.shape[-1]
            fid = np.pad(fid, ((0, 0), (0, 0), (0, pad)), mode="constant")
            time_axis = np.arange(fid.shape[-1]) * dwell_time
            processing_log.append(f"zero_fill:{fid.shape[-1]}")

        return fid, time_axis

    @staticmethod
    def _build_apodization_window(
        n_points: int,
        dwell_time: float,
        mode,
        lb_hz: float,
        gb_hz: float,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if not mode or str(mode).lower() in {"none", "", "off"}:
            return None, None

        mode = str(mode).lower()
        t = np.arange(n_points) * dwell_time

        if mode in {"auto", "default"}:
            if lb_hz > 0:
                mode = "exponential"
            elif gb_hz > 0:
                mode = "gaussian"
            else:
                return None, None

        if mode == "exponential":
            lb = lb_hz if lb_hz > 0 else 1.0
            window = np.exp(-np.pi * lb * t)
        elif mode == "gaussian":
            gb = gb_hz if gb_hz > 0 else 1.0
            window = np.exp(-((gb * t) ** 2))
        elif mode == "hamming":
            window = np.hamming(n_points)
        elif mode == "hann":
            window = np.hanning(n_points)
        elif mode == "blackman":
            window = np.blackman(n_points)
        else:
            raise ValueError(f"Unsupported apodization mode '{mode}'")

        return window.astype(np.float32, copy=False), mode

    @staticmethod
    def _apply_phase_correction(
        spectral: np.ndarray, phase0_deg: float, phase1_deg: float
    ) -> np.ndarray:
        phase0 = np.deg2rad(phase0_deg)
        phase1 = np.deg2rad(phase1_deg)
        n_points = spectral.shape[-1]
        phase_axis = np.linspace(-0.5, 0.5, n_points, endpoint=False)
        phase_ramp = phase0 + phase1 * phase_axis
        correction = np.exp(1j * phase_ramp)
        return spectral * correction

    def _combine_channels(
        self,
        spectral: np.ndarray,
        *,
        combine_channels: bool,
        retain_axis: bool,
        combine_method: str,
        processing_log,
    ):
        if not combine_channels or spectral.shape[1] == 1:
            return spectral, spectral.ndim == 3

        method = (combine_method or "rss").lower()
        if method in {"none", "off"}:
            return spectral, spectral.ndim == 3

        if method == "rss":
            combined = np.sqrt(np.sum(np.abs(spectral) ** 2, axis=1, keepdims=True))
            combined = combined.astype(np.float32, copy=False)
        elif method == "sum":
            combined = np.sum(spectral, axis=1, keepdims=True)
        elif method == "mean":
            combined = np.mean(spectral, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported coil combination method '{combine_method}'")

        processing_log.append(f"coil_combination:{method}")

        if retain_axis:
            return combined, True

        combined = np.squeeze(combined, axis=1)
        return combined, False

    @staticmethod
    def _build_frequency_axes(
        n_points: int, dwell_time: float, handler
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        freq_hz = np.fft.fftshift(np.fft.fftfreq(n_points, d=dwell_time))

        freq_ppm = None
        spectrometer_freq = getattr(handler, "spectrometer_frequency_mhz", None)
        if spectrometer_freq:
            freq_ppm = freq_hz / spectrometer_freq
            ref_freq = getattr(handler, "reference_frequency_mhz", None)
            if ref_freq:
                offset_ppm = (spectrometer_freq - ref_freq) / spectrometer_freq
                freq_ppm = freq_ppm + offset_ppm

        return freq_hz, freq_ppm
