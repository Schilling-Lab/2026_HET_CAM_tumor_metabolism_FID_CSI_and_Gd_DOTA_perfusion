"""
Temporal processing and time axis calculations.

This module contains all time-related functions extracted from utils_general.py
as part of the Phase 2 refactoring to create focused utility modules.
"""

from typing import Any, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from ..system.logging import init_default_logger

logger = init_default_logger(__name__)


def get_dwell_time_s(
    data_obj: Any = None,
    bw_hz: float = None,
    sampling_time_axis_s: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate the sampling duration per point in [s] when a spectrum/FID was acquired.

    Parameters
    ----------
        data_obj: data_obj, optional
            THe data object containing information about the sampling
        bw_hz: float
            spectral bandwidth in [Hz]
        sampling_time_axis_s: ndarray
            sampling time axis in [s]
    Returns
    -------
    sampling duration per point

    """
    dwell_time_s = 0
    if data_obj is None:
        if bw_hz is not None and bw_hz > 0:
            dwell_time_s = 1 / bw_hz
        elif sampling_time_axis_s is not None and isinstance(
            sampling_time_axis_s, np.ndarray
        ):
            dwell_time_s = sampling_time_axis_s[1] - sampling_time_axis_s[0]
        else:
            logger.critical("Pass either data_obj or bw_hz!")
    else:
        # CSI:
        dwell_time_s = data_obj.get("dwell_time_s", 0)

    return dwell_time_s


def calc_sampling_time_axis(
    data_obj=None,
    dwell_time_s=0,
    npoints=0,
    bw_hz=0,
    no_points=None,
    bandwidth_hz=None,
    start_with_0=True,
    t_echo_s=None,
):
    """
    This function returns sampling time axis (in an FID e.g.) in [s].

    This function maintains backward compatibility with the original interface
    while supporting new parameter names.

    Parameters
    ----------
    data_obj : object, optional
        Data object containing timing parameters (original interface)
    npoints : int, optional
        Number of points of the time axis (original interface)
    bw_hz : float, optional
        Spectral bandwidth in [Hz] (original interface)
    no_points : int, optional
        Number of time points (new interface). Default is 512.
    bandwidth_hz : float, optional
        Spectral width in Hz (new interface). Default is 3000.
    start_with_0 : bool, optional
        Whether to start the time axis at 0. Default is True.
    t_echo_s : float, optional
        Echo time in seconds for starting offset.
    dwell_time_s : float, optional
        Sampling time step in seconds. If provided, overrides bandwidth_hz.

    Returns
    -------
    array_like
        Time axis in seconds

    """
    # Handle original interface (data_obj based)
    if data_obj is not None:
        # Check if sampling_time_axis_s is already available on the object
        # This preserves sequence-specific axes (e.g. MGE with TE offsets)
        precalc_axis = getattr(data_obj, "sampling_time_axis_s", None)
        if precalc_axis is not None and len(precalc_axis) > 0:
            # If npoints is not specified or matches, use the pre-calculated axis
            if npoints == 0 or npoints is None or len(precalc_axis) == npoints:
                return np.asarray(precalc_axis)

        # Extract parameters from data object
        if dwell_time_s <= 0:
            dwell_time_s = get_dwell_time_s(data_obj=data_obj, bw_hz=bw_hz)

        if npoints == 0 or npoints is None:
            try:
                # CSI/spectroscopy data
                npoints = data_obj.n_acquisition - data_obj.cut_off
            except (AttributeError, TypeError):
                # Try to get from method parameters
                # npoints = getattr(data_obj, "method", {}).get("PVM_DigNp", 512)
                raise ValueError(
                    "Could not determine number of points from data_obj. Please provide npoints."
                )

        # Create time axis using original logic
        sampling_time_axis_s = np.linspace(0, dwell_time_s * (npoints - 1), npoints)
        return sampling_time_axis_s

    # Handle new interface or parameter fallbacks
    if no_points is not None:
        npoints = no_points
    elif npoints == 0:
        npoints = 512

    if bandwidth_hz is not None:
        bw_hz = bandwidth_hz
    elif bw_hz == 0:
        bw_hz = 3000

    if dwell_time_s is not None:
        dt = dwell_time_s
    elif dwell_time_s > 0:
        dt = dwell_time_s
    else:
        dt = 1.0 / bw_hz

    if start_with_0:
        time_axis = np.arange(npoints) * dt
        if t_echo_s is not None:
            time_axis += t_echo_s
    else:
        time_axis = np.arange(1, npoints + 1) * dt
        if t_echo_s is not None:
            time_axis += t_echo_s

    return time_axis


def calc_per_image_timeaxis_s(data_obj=None, start_with_0=True):
    """Calculate the time range [in s] on which the data was acquired.

    Parameters
    ----------
    data_obj : object, optional
        Data object containing timing parameters (original interface)

    """
    if data_obj is None:
        return None

    try:
        nr = data_obj.n_repetitions
        image_repetition_time = data_obj.scan_duration_s / nr

        if start_with_0:
            return np.linspace(0, (nr - 1) * image_repetition_time, nr)
        else:
            return np.linspace(image_repetition_time, nr * image_repetition_time, nr)
    except BaseException:
        return None


def calc_inversion_time_multi_slice(data_obj):
    """Calculate time increment on the inversion time for each slice in FAIR sequence."""
    if data_obj is None:
        raise ValueError("Data object is None")

    ref_pulse_time = data_obj.get("refpulse_duration_ms", 0)
    if ref_pulse_time == 0 or not ref_pulse_time:
        return 0

    fat_sup_time = data_obj.method.get("PVM_FatSupModuleTime", 0)
    mag_trans_time = data_obj.method.get("PVM_MagTransModuleTime", 0)
    fov_sat_time = data_obj.method.get("PVM_FovSatModuleTime", 0)
    rep_spoiler_time = data_obj.method.get("RepetitionSpoiler", [0, 0, 0])[2]

    amp_enable_time = data_obj.method.get("PVM_FatSupAmpEnable", 0)
    exc_pulse_half_time = data_obj.rfpulse_duration_ms / 2
    rare_factor = data_obj.rare_factor
    echo_time = data_obj.echo_time_ms

    grad_stab_time = data_obj.method.get("GradStab", 0)
    rise_time = 0.169  # Bruker default rise time in ms

    # This is taken from the FAIR_RARE Bruker sequence. other sequences will have different slice_TI_increment
    slice_TI_increment = (
        0.02
        + fat_sup_time
        + mag_trans_time
        + fov_sat_time
        + rep_spoiler_time
        + amp_enable_time
        + exc_pulse_half_time
        + echo_time * rare_factor
        + (echo_time - ref_pulse_time - 2 * grad_stab_time) / 2
        + 2.0
        + rise_time
    )
    return slice_TI_increment


def create_time_axis(
    start_time: float, end_time: float, dt: float, include_end: bool = True
) -> np.ndarray:
    """
    Create a time axis with specified parameters.

    Parameters
    ----------
    start_time : float
        Start time in seconds
    end_time : float
        End time in seconds
    dt : float
        Time step in seconds
    include_end : bool, optional
        Whether to include the end time point. Default is True.

    Returns
    -------
    array_like
        Time axis in seconds

    """
    if include_end:
        n_points = int((end_time - start_time) / dt) + 1
        time_axis = np.linspace(start_time, end_time, n_points)
    else:
        time_axis = np.arange(start_time, end_time, dt)

    return time_axis


def resample_time_series(
    time_old: np.ndarray,
    data_old: np.ndarray,
    time_new: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """
    Resample time series data to new time axis.

    Parameters
    ----------
    time_old : array_like
        Original time axis
    data_old : array_like
        Original data values
    time_new : array_like
        New time axis for resampling
    method : str, optional
        Interpolation method ('linear', 'cubic', 'nearest'). Default is 'linear'.

    Returns
    -------
    array_like
        Resampled data on new time axis

    """
    # Handle multi-dimensional data
    if data_old.ndim > 1:
        # Assume time is first dimension
        data_new = np.zeros((len(time_new),) + data_old.shape[1:])

        # Flatten extra dimensions for vectorized interpolation
        original_shape = data_old.shape
        data_flat = data_old.reshape(original_shape[0], -1)
        data_new_flat = np.zeros((len(time_new), data_flat.shape[1]))

        for i in range(data_flat.shape[1]):
            interp_func = interp1d(
                time_old, data_flat[:, i], kind=method, bounds_error=False, fill_value=0
            )
            data_new_flat[:, i] = interp_func(time_new)

        data_new = data_new_flat.reshape((len(time_new),) + original_shape[1:])
    else:
        # 1D data
        interp_func = interp1d(
            time_old, data_old, kind=method, bounds_error=False, fill_value=0
        )
        data_new = interp_func(time_new)

    return data_new


def interpolate_temporal_data(
    time_axis: np.ndarray, data: np.ndarray, target_dt: float, method: str = "linear"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate temporal data to achieve target time resolution.

    Parameters
    ----------
    time_axis : array_like
        Original time axis
    data : array_like
        Data to interpolate (time should be first dimension)
    target_dt : float
        Target time step in seconds
    method : str, optional
        Interpolation method. Default is 'linear'.

    Returns
    -------
    tuple
        (new_time_axis, interpolated_data)

    """
    # Create new time axis with target resolution
    time_start = time_axis[0]
    time_end = time_axis[-1]
    new_time_axis = create_time_axis(time_start, time_end, target_dt, include_end=True)

    # Interpolate data
    interpolated_data = resample_time_series(time_axis, data, new_time_axis, method)

    return new_time_axis, interpolated_data


def smooth_temporal_data(
    data: np.ndarray, window_size: int, axis: int = 0, method: str = "moving_average"
) -> np.ndarray:
    """
    Smooth temporal data using specified method.

    Parameters
    ----------
    data : array_like
        Data to smooth
    window_size : int
        Size of smoothing window
    axis : int, optional
        Temporal axis. Default is 0.
    method : str, optional
        Smoothing method ('moving_average', 'gaussian'). Default is 'moving_average'.

    Returns
    -------
    array_like
        Smoothed data

    """
    if method == "moving_average":
        # Use convolution for moving average
        kernel = np.ones(window_size) / window_size

        if data.ndim == 1:
            smoothed = np.convolve(data, kernel, mode="same")
        else:
            # Apply along specified axis
            smoothed = np.apply_along_axis(
                lambda x: np.convolve(x, kernel, mode="same"), axis, data
            )

    elif method == "gaussian":
        from scipy import ndimage

        sigma = window_size / 4.0  # Convert window size to sigma
        smoothed = ndimage.gaussian_filter1d(data, sigma, axis=axis)

    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    return smoothed
