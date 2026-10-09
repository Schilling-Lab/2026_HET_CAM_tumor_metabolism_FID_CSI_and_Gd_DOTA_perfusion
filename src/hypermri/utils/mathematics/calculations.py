"""
Mathematical calculations and curve fitting utilities.

This module contains basic mathematical functions extracted from utils_general.py
as part of the Phase 2 refactoring to create focused utility modules.
"""

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from ..system.logging import init_default_logger

logger = init_default_logger(__name__)


def get_histogram(input_data=None, number_of_bins: str = 10):
    """Compute histogram bin centers and counts.

    Parameters
    ----------
    input_data : array_like
        Values to bin.
    number_of_bins : int, optional
        Number of histogram bins. Default is 10.

    Returns
    -------
    x_data : numpy.ndarray
        Bin centers.
    y_data : numpy.ndarray
        Sample count per bin.
    bin_size : float
        Width of each bin.

    Notes
    -----
    NaN values in ``input_data`` are ignored when accumulating the histogram counts.
    """
    if np.isnan(input_data).any():
        min_val = np.nanmin(input_data)
        max_val = np.nanmax(input_data)
    else:
        min_val = np.min(input_data)
        max_val = np.max(input_data)
    bin_size = (max_val - min_val) / number_of_bins
    x_data = np.linspace(min_val, max_val, number_of_bins)
    y_data = np.zeros_like(x_data)

    if len(input_data.shape) > 1:
        for i in np.arange(0, input_data.shape[0], 1):
            for j in np.arange(0, input_data.shape[1], 1):
                if np.isnan(input_data[i, j]):
                    # handling nan values so that they don't contribute to the
                    # histogram
                    pass
                else:
                    idx = np.abs(x_data - input_data[i, j]).argmin()
                    y_data[idx] += 1
    else:
        for j in np.arange(0, input_data.shape[0], 1):
            idx = (np.abs(x_data - input_data[j])).argmin()
            y_data[idx] += 1
    return x_data, y_data, bin_size


def gaussian(x, amplitude, mean, sigma):
    """Evaluate a Gaussian curve.

    Parameters
    ----------
    x : array_like
        Independent variable.
    amplitude : float
        Peak amplitude of the Gaussian.
    mean : float
        Center position of the Gaussian.
    sigma : float
        Standard deviation (spread) of the Gaussian.

    Returns
    -------
    numpy.ndarray
        Gaussian values evaluated at ``x``.
    """
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2))


def compute_weighted_mean_and_std(input_data=None, weights=None):
    """Compute weighted statistics for a sample.

    Parameters
    ----------
    input_data : array_like
        Sample values.
    weights : array_like
        Non-negative weights assigned to ``input_data``.

    Returns
    -------
    tuple of float
        ``(weighted_mean, weighted_std)`` giving the weighted mean and
        standard deviation.
    """
    input_data = np.asarray(input_data)
    weights = np.asarray(weights)

    # Compute weighted mean
    weighted_mean = np.average(input_data, weights=weights)

    # Compute weighted variance
    weighted_var = np.average((input_data - weighted_mean) ** 2, weights=weights)
    weighted_std = np.sqrt(weighted_var)

    return weighted_mean, weighted_std


def find_max_and_index(data):
    """Locate the maximum value and its indices.

    Parameters
    ----------
    data : array_like or xarray.DataArray
        Input array, optionally multi-dimensional.

    Returns
    -------
    tuple
        ``(max_value, max_indices)`` where ``max_value`` is the maximum element
        and ``max_indices`` is a tuple of the indices at which it occurs.
    """
    if isinstance(data, xr.DataArray):
        abs_array = np.abs(data)
        max_value = abs_array.max().item()
        # Get flat index and unravel using shape
        flat_idx = abs_array.argmax().item()
        max_index = np.unravel_index(flat_idx, abs_array.shape)
    else:
        abs_array = np.abs(data)
        max_value = np.max(abs_array)
        max_index = np.unravel_index(np.argmax(abs_array), abs_array.shape)

    return max_value, max_index


def calc_sampling_time_axis(dwell_time_s=0, npoints=0, data_obj=None, bw_hz=0):
    """Create the sampling time axis associated with a dwell time.

    Parameters
    ----------
    dwell_time_s : float
        Dwell time between consecutive samples in seconds.
    npoints : int
        Number of points in the acquisition.
    data_obj : object, optional
        Data object to extract acquisition parameters from.
    bw_hz : float, optional
        Spectral bandwidth in Hz.

    Returns
    -------
    numpy.ndarray
        Equally spaced time axis in seconds.
    """
    if dwell_time_s <= 0:
        if bw_hz > 0:
            dwell_time_s = 1 / bw_hz
        else:
            dwell_time_s = 1e-3

    if data_obj is not None:
        if npoints == 0 or npoints is None:
            npoints = data_obj.n_acquisition - getattr(data_obj, "cut_off", 0)

    return np.linspace(0, dwell_time_s * (npoints - 1), npoints)


def polynomial_fit(x, y, degree=1, weights=None):
    """Fit a polynomial to data with optional weighting.

    Parameters
    ----------
    x : array_like
        Independent variable.
    y : array_like
        Dependent variable.
    degree : int, optional
        Degree of the polynomial to fit. Default is 1.
    weights : array_like, optional
        Relative weights applied to each observation. Default is ``None``.

    Returns
    -------
    tuple
        ``(coefficients, polynomial)`` where ``coefficients`` are the fitted
        polynomial coefficients and ``polynomial`` is a ``numpy.poly1d``
        callable representing the fit.
    """
    coeffs = np.polyfit(x, y, degree, w=weights)
    poly_func = np.poly1d(coeffs)
    return coeffs, poly_func


def moving_average(data, window_size):
    """Compute a simple moving average.

    Parameters
    ----------
    data : array_like
        Input data sequence.
    window_size : int
        Length of the averaging window.

    Returns
    -------
    numpy.ndarray
        Moving average values. The output length is ``len(data) - window_size + 1``.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def apply_gaussian_smoothing(input_data=None, sigma: float = 1.0):
    """Apply Gaussian smoothing to data.

    Parameters
    ----------
    input_data : array_like
        Input data to smooth.
    sigma : float, optional
        Standard deviation of the Gaussian kernel. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Smoothed data array.
    """
    from scipy import ndimage

    return ndimage.gaussian_filter(input_data, sigma=sigma)


def calculate_fwhm(x, y, method="interpolation"):
    """Compute the full width at half maximum of a peak.

    Parameters
    ----------
    x : array_like
        Sample positions.
    y : array_like
        Sampled peak values.
    method : {"interpolation", "fit"}, optional
        Algorithm used to determine the FWHM. Default is ``"interpolation"``.

    Returns
    -------
    float
        Estimated full width at half maximum.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported options.
    """
    # Find peak
    peak_idx = np.argmax(y)
    peak_val = y[peak_idx]
    half_max = peak_val / 2

    if method == "interpolation":
        # Find points where y crosses half maximum
        indices = np.where(y >= half_max)[0]
        if len(indices) < 2:
            return np.nan

        left_idx = indices[0]
        right_idx = indices[-1]

        # Interpolate for more precise FWHM
        if left_idx > 0:
            left_x = np.interp(
                half_max, [y[left_idx - 1], y[left_idx]], [x[left_idx - 1], x[left_idx]]
            )
        else:
            left_x = x[left_idx]

        if right_idx < len(x) - 1:
            right_x = np.interp(
                half_max,
                [y[right_idx], y[right_idx + 1]],
                [x[right_idx], x[right_idx + 1]],
            )
        else:
            right_x = x[right_idx]

        return right_x - left_x

    elif method == "fit":
        # Fit Gaussian and extract FWHM
        try:
            # Initial guess
            amplitude_guess = peak_val
            mean_guess = x[peak_idx]
            sigma_guess = (x[-1] - x[0]) / 10  # Rough estimate

            popt, _ = curve_fit(
                gaussian, x, y, p0=[amplitude_guess, mean_guess, sigma_guess]
            )

            # FWHM = 2.355 * sigma for Gaussian
            return 2.355 * abs(popt[2])

        except Exception as e:
            logger.warning(f"Gaussian fit failed for FWHM calculation: {e}")
            return np.nan

    else:
        raise ValueError("Method must be 'interpolation' or 'fit'")


def part_func(input_data=None, signal_part="Magnitude"):
    """Extract a chosen component from a complex-valued signal.

    Parameters
    ----------
    input_data : array_like, optional
        Complex-valued samples from which to extract a component.
    signal_part : {"Real", "Imaginary", "Phase", "Magnitude"}, optional
        Component to extract. Any other value defaults to ``"Magnitude"``.

    Returns
    -------
    numpy.ndarray
        Requested component of ``input_data``. Returns ``None`` when
        ``input_data`` is ``None``.

    Notes
    -----
    The function silently falls back to the magnitude if an unsupported
    ``signal_part`` is provided.
    """
    if input_data is None:
        return None

    match signal_part:
        case "Real":
            return np.real(input_data)
        case "Imaginary":
            return np.imag(input_data)
        case "Phase":
            return np.angle(input_data)
        case "Magnitude":
            return np.abs(input_data)
        case _:
            return np.abs(input_data)
