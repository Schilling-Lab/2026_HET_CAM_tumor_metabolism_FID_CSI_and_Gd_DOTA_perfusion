"""Core spectroscopy utilities shared across fitting workflows."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from astropy.modeling.core import Fittable1DModel
from scipy.optimize import curve_fit

from ...utils.system.logging import debug_log, init_default_logger

DEBUG_FILE = "/tmp/analysis_spec_core.log"
DEBUG_LOG = True


def debug_log_spec_core(
    message: str = "", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG
):
    """Debug logging system for spectroscopy core module."""
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
    )


logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


def lorentzian(x, FWHM, center_position, peak_height) -> np.ndarray:
    """Evaluate a Lorentzian line shape.

    Parameters
    ----------
    x : array_like
        Frequency axis in the same units as ``center_position``.
    FWHM : float
        Full width at half maximum of the peak.
    center_position : float
        Peak centre location in the units of ``x``.
    peak_height : float
        Peak amplitude at the centre.

    Returns
    -------
    numpy.ndarray
        Lorentzian profile evaluated at ``x``.
    """

    return peak_height * FWHM**2 / ((FWHM**2 + (2 * x - 2 * center_position) ** 2))


def sqrt_lorentzian(x, FWHM, center_position, peak_height):
    """Evaluate the square-root Lorentzian profile used for peak envelopes."""

    return peak_height / np.sqrt((1 + 3 * (x - center_position) ** 2 / (FWHM / 2) ** 2))


def generate_fid(
    time_axis_s: Optional[np.ndarray] = None,
    amplitude: float = 1.0,
    t2_s: float = 0.02,
    freq_Hz: float = 0.0,
    phase_rad: float = 0.0,
    dwell_time_s: float = 5e-4,
    npoints: int = 1000,
    noise_std: float = 0.0,
    sum_fids: bool = True,
) -> Optional[np.ndarray]:
    """
    Generate a FID (Free Induction Decay)
    Parameters
    ----------
    time_axis_s: 1D array
        time axis in seconds
    amplitude: float/ND-array
        FID amplitude in [a.u.]
    T2_s: float/ND-array
        T2 in seconds
    freq0_Hz: float/ND-array
        frequecy of FID in Hertz
    phase0_rad: float/ND-array
        phase of FID in Radian
    dwell_time_s: float
        dwell time in microseconds. If no time_axis_s is passed, time axis can be calculated from dwell time and npoints
    npoints: int
        Number of Points of FID used to calculate the time axis using dwell_time_s
    noise_amplitude: float/ND-array
        standard deviation of normal distributed noise
    sum_fids: if more than one number for the parameters (amplitude, T2_s, ...) is passed, multiple fids are generated.
        If True, the FIDs will be summed, else an array of fids is generated

    Returns
    -------
    A 1D (complex) FID

    Examples
    --------
    >>> from hypermri.analysis.core import apply_lb, generate_fid
    >>> from hypermri.utils.time.temporal import calc_sampling_time_axis

    >>> dt = 5e-4
    >>> npoints=1000
    >>> time_axis_s = calc_sampling_time_axis(fid_start=0, fid_end=npoints, dwell_time_s=dt)

    >>> f = np.fft.fftshift(np.fft.fftfreq(npoints, d=dt))
    >>> lb_hz=20.0
    >>> fid = generate_fid(time_axis_s=time_axis_s, T2_s=0.1, freq0_Hz=100.0, phase0_rad=0.5, dwell_time_s=dt, npoints=npoints)
    >>> spec_lb = apply_lb(input_data=fid, lb_hz=lb_hz, dwell_time_s=dt, signal_domain="temporal")

    >>> plt.figure()
    >>> plt.subplot(1,2,1)
    >>> plt.plot(time_axis_s, np.real(fid))
    >>> plt.plot(time_axis_s, np.imag(fid))
    >>> plt.plot(time_axis_s, np.abs(fid))
    >>> plt.xlabel('t [s]')

    >>> plt.subplot(1,2,2)
    >>> plt.plot(f, np.abs(np.fft.fftshift(np.fft.fft(fid)))/np.max(np.abs(np.fft.fftshift(np.fft.fft(fid)))),label='original FID')
    >>> plt.plot(f, np.abs(spec_lb)/np.max(np.abs(spec_lb)),label=f'linebroadened FID {lb_hz}Hz')
    >>> plt.xlabel('f [Hz]')
    >>> plt.legend()
    """

    if time_axis_s is None:
        from ...utils.time.temporal import calc_sampling_time_axis

        # calculate the time_axis in seconds:
        time_axis_s = calc_sampling_time_axis(
            npoints=npoints, dwell_time_s=dwell_time_s
        )
    else:
        pass
    npoints = len(time_axis_s)

    # Convert all parameters to numpy arrays, handling list and tuple inputs
    amplitude = np.atleast_1d(np.array(amplitude))
    t2_s = np.atleast_1d(np.array(t2_s))
    freq_Hz = np.atleast_1d(np.array(freq_Hz))
    phase_rad = np.atleast_1d(np.array(phase_rad))
    # Determine the target shape based on the most complex parameter
    target_shape = np.broadcast(amplitude, t2_s, freq_Hz, phase_rad).shape
    # Broadcast parameters to the target shape
    amplitude = np.broadcast_to(amplitude, target_shape)
    t2_s = np.broadcast_to(t2_s, target_shape)
    freq_Hz = np.broadcast_to(freq_Hz, target_shape)
    phase_rad = np.broadcast_to(phase_rad, target_shape)

    # Initialize the output FID array with the temporal dimension first
    fid = np.zeros((npoints, *target_shape), dtype="complex")

    # Generate the FID signal for each set of parameters
    for index in np.ndindex(target_shape):
        idx = (slice(None),) + index
        exp_decay = np.exp(-time_axis_s / t2_s[index])
        oscillation = np.exp(
            1j * (2 * np.pi * freq_Hz[index] * time_axis_s + phase_rad[index])
        )
        noise = np.random.normal(
            size=npoints,
            scale=noise_std,
        ) + 1j * np.random.normal(
            size=npoints,
            scale=noise_std,
        )
        fid[idx] = amplitude[index] * exp_decay * oscillation + noise

    if sum_fids:
        fid = np.sum(fid, axis=-1)

    return fid


def norm(x, a=0, b=1):
    """Norm an array to interval [a,b] with a<b. Default range is [0,1].

    To convert spectroscopy x-axis from ppm to Hz:

        x_Hz = norm(x_ppm - 4.7, -1, 1) * (scan.method['PVM_SpecSWH'] / 2
    """
    assert a < b

    min_x = np.min(x)
    max_x = np.max(x)

    normed_x_01 = (x - min_x) / (max_x - min_x)  # normed array from 0 to 1
    normed_x_custom = (b - a) * normed_x_01 + a  # normed array within [a,b]

    return normed_x_custom


def freq_to_index(freq: Optional[float], freq_range: Optional[np.ndarray]) -> int:
    """Calculate the index of a given frequency in a given frequency range.

    Args:
        freq (float): frequency in [a.u.]
        freq_range (array): frequency range in same [a.u.]
    Returns
        index (int): index of the frequency in the given frequency range.
    """
    index = np.argmin(np.abs(freq_range - freq))

    return index


def get_bw_from_freq_axis(
    freq_axis: Optional[np.ndarray], data_obj: Any = None, unit: str = "Hz"
) -> float:
    """
    Calculate the bandwidth from a frequency axis, taking into account the width of the sampling points.

    Bandwidth is calculated as the absolute difference between the last and first points of the frequency axis
    plus the width of one sampling point. This approach accounts for the fact that the outer points of the frequency
    axis represent sampling points, which have a certain width, hence the bandwidth is not simply the difference
    between these points.

    Parameters
    ----------
    freq_axis : np.ndarray, optional
        An array of frequency points. If provided, `data_obj` should not be passed. Default is None.
    data_obj : MRIExperiment, optional
        A data object from which the frequency axis can be extracted if `freq_axis` is not directly provided.
        The function `get_freq_axis` must be defined elsewhere to use this parameter. Default is None.
    unit : str, optional
        The unit of the frequency axis. This parameter is used when extracting the frequency axis from `data_obj`.
        Default is "Hz".

    Raises
    ------
    ValueError
        If neither `freq_axis` nor `data_obj` is provided, or if both are provided simultaneously.

    Returns
    -------
    float
        The calculated bandwidth of the frequency axis, incorporating the width of the sampling points.

    Notes
    -----
    The function relies on either a direct input of the frequency axis via `freq_axis` or extraction from a data
    object through a predefined `get_freq_axis` function, which should understand the `unit` parameter. The
    calculation adjusts for the sampling point width by adding the step size between the first two frequency
    points to the difference between the last and first frequency points.
    """
    # Ensure that either freq_axis or data_obj is provided, but not both
    if freq_axis is None and data_obj is None:
        raise ValueError("have to pass either freq_axis OR data_obj")

    if freq_axis is not None and data_obj is not None:
        raise ValueError("Do not pass freq_axis AND data_obj")

    # If freq_axis is provided directly, calculate the bandwidth using it
    if freq_axis is not None:
        # Calculate the sampling step as the absolute difference between the
        # first two points
        freq_axis_sampling_step = np.abs(freq_axis[1] - freq_axis[0])
        # Calculate the bandwidth as the absolute difference between the last
        # and first points
        freq_axis_bw = np.abs(freq_axis[-1] - freq_axis[0])
        # Adjust the bandwidth by adding the width of one sampling point
        bw = freq_axis_bw + freq_axis_sampling_step
        return bw

    # If a data object is provided, extract the frequency axis from it first
    if data_obj is not None:
        # Extract frequency axis from data object, considering the specified
        # unit
        freq_axis = get_freq_axis(data_obj=data_obj, unit=unit)
        # Repeat the bandwidth calculation as done for the directly provided
        # freq_axis
        freq_axis_sampling_step = np.abs(freq_axis[1] - freq_axis[0])
        freq_axis_bw = np.abs(freq_axis[-1] - freq_axis[0])
        bw = freq_axis_bw + freq_axis_sampling_step
        return bw


def get_freq_axis(
    data_obj: Any = None,
    unit: str = "ppm",
    cut_off: Optional[int] = None,
    npoints: Optional[int] = None,
    centered_at_0: bool = False,
    dwell_time_s: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Get frequency axis of given spectroscopy data_obj in unit of ppm or Hz.

    Parameters
    ----------
    data_obj: Any object

    unit : string
        Hz or ppm, Default value is ppm,

    cut_off : int
        number of points to remove from frequency axis [for stock Bruker 13c expirements]

    npoints: int
        number of points the frequency axis should have.

    Returns
    -------
    ppm_axis : np.ndarray
        Frequency axis of measurement in unit of ppm.
    """

    # Calculate frequency axis in ppm or Hz if no data_obj is passed
    # --------------------------------------------------------------
    debug_log_spec_core(
        f"get_freq_axis called with unit={unit}, cut_off={cut_off}, data_obj={data_obj}, npoints={npoints}, dwell_time_s={dwell_time_s}"
    )
    if data_obj is None:
        if dwell_time_s is None or npoints is None:
            Warning("if data_obj=None pass dwell_time_s and npoints!")
            return None
        if unit == "ppm":
            raise ValueError("if data_obj=None this can only return unit=Hz axis!")
            return None

        if isinstance(npoints, tuple) and len(npoints) == 1:
            npoints = npoints[0]

        if cut_off is None:
            cut_off = 0

        # define frequency axis in Hz
        hz_axis = np.fft.fftshift(np.fft.fftfreq(npoints, d=dwell_time_s))
        if centered_at_0 is True:
            hz_axis = center_freq_axis(freq_axis=hz_axis)
            return hz_axis
        else:
            return hz_axis
    else:
        if cut_off is None:
            cut_off = data_obj.get("cut_off", 0)

    # Calculate frequency axis in ppm or Hz if data_obj is passed:
    # ------------------------------------------------------------
    if not hasattr(data_obj, "bandwidth_hz"):
        debug_log_spec_core(
            "data_obj has no attribute bandwidth_hz, cannot calculate freq axis"
        )
        return None

    # if spectral range should be centered:
    if centered_at_0 is None:
        # usually ppm ranges should not be centered around 0 ppm:
        if unit == "ppm":
            centered_at_0 = False
        # usually Hz ranges should be centered around 0 Hz:
        elif unit == "Hz":
            centered_at_0 = False
        else:
            centered_at_0 = False

    # define center frequency:
    center_ppm = data_obj.get("center_ppm", 0)
    debug_log_spec_core(f"center_ppm={center_ppm}")

    # define bandwidth [Hz]:
    bw_hz = data_obj.bandwidth_hz
    if bw_hz is None:
        return []

    # sampling rate [s]:
    from ...utils.time.temporal import get_dwell_time_s

    # get sampling duration [s]
    dt = get_dwell_time_s(data_obj=data_obj)
    debug_log_spec_core(f"dt={dt}")

    # if no number of points was specified, use scan info, else use passed
    # number of points:
    if npoints is None:
        # number of acquisition points:
        freq_points = int(data_obj.n_acquisition - cut_off)
    else:
        freq_points = npoints

    debug_log_spec_core(f"freq_points={freq_points}")

    # define frequency axis in [Hz]
    hz_axis = np.fft.fftshift(np.fft.fftfreq(freq_points, d=dt))

    if unit.lower() == "ppm":
        # define bandwidth [ppm]:
        bw_ppm = getattr(data_obj, "bandwidth_ppm", None)
        debug_log_spec_core(f"bw_ppm={bw_ppm} ppm")

        if bw_ppm is None or bw_ppm == 0:
            debug_log_spec_core("bandwidth_ppm is None or 0, cannot calculate ppm axis")
            return None

        # converison factor:
        hz_to_ppm = bw_hz / bw_ppm

        debug_log_spec_core(f"hz_to_ppm={hz_to_ppm} Hz/ppm")

        # translate to [ppm]
        ppm_axis = hz_axis / hz_to_ppm

        debug_log_spec_core(f"ppm_axis (before centering)={ppm_axis}")

        # add offset:
        ppm_axis = ppm_axis + center_ppm

        # center around 0:
        if centered_at_0:
            ppm_axis = center_freq_axis(freq_axis=ppm_axis)
        debug_log_spec_core(f"ppm_axis (after centering)={ppm_axis}")
        return ppm_axis

    elif unit.lower() == "hz":
        # center around 0 Hz:
        if centered_at_0:
            hz_axis = center_freq_axis(freq_axis=hz_axis)

        return hz_axis

    else:
        Warning(f"unknown frequency unit {unit}")


def norm_spectrum_to_snr(spec, bg_indices=[0, 100]):
    """
    Norms input spectrum to background region (default is first 100 entries, can be changed)
    Parameters
    ----------
    spec: array
        shape: (Repetitions, Spectral_acquisition_points), contains spectra for a given method.
    bg_indices: list, optional
    two entry list with indices of background region, default is [0,100]
    Returns
    -------
    normed_spec: array
    """
    normed_spec = np.zeros_like(spec)
    for n in range(0, spec.shape[0], 1):
        normed_spec[n, :] = (
            spec[n, :] - np.mean(spec[n, bg_indices[0] : bg_indices[1]])
        ) / np.std(spec[n, bg_indices[0] : bg_indices[1]])
    return normed_spec


def fit_spectrum(
    ppm_axis,
    spectrum,
    peak_positions,
    SNR_cutoff=1,
    plot=False,
    norm_to_snr_before_fit=False,
    bg_region_first_spec=[1200, 1800],
    min_fwhm_ppm=0.01,
    fit_func="lorentzian",
):
    """
    Fits lorentzian to a spectrum at the desired peak locations
    Parameters
    ----------
    ppm_axis: numpy array
        The ppm-scale of the measurements to be examined.
    spectrum: numpy array
        N-dimensional spectrum to be fitted.
    peak_positions: list
        Contains the peak positions where fits should occur in ppm.
    SNR_cutoff: float
        SNR value below which peaks should not be fitted, default is 5.
    plot: bool
        Select if plotting for checking is needed
    norm_to_snr_before_fit: bool
        Select if you want to norm the spectra to a background region of the first repetition before fitting.
    bg_region_first_spec: list
        indices of the background region from which the snr is calculated.
    fit_func: str, default is 'lorentzian'
    can also be sqrt_lorentzian
    Returns
    -------
    peak_coeff: numpy array
        coefficients for a 3 parametric lorentzian fit model, shape: (NRepetitions, Number_of_peaks, 3)
        peak_coeff[0]: peak FWHM
        peak_coeff[1]: peak position (ppm)
        peak_coeff[2]: peak SNR

    peak_errors: numpy array
        errors for fit coefficients calculated from the covariance matrix, shape: (NRepetitions, Number_of_peaks, 3)
    """

    if fit_func == "lorentzian":
        fit_function = lorentzian
        fit_func_str = "Sqrt Lorentzian"
    elif fit_func == "sqrt_lorentzian":
        fit_function = sqrt_lorentzian
        fit_func_str = "Sqrt Lorentzian"
    else:
        logger.error(
            "Fit function %s not implemented - proceeding to fit using normal lorentzian",
            fit_func,
        )
        fit_function = lorentzian

    def find_range(axis, ppm):
        return np.argmin(np.abs(axis - ppm))

    # Norming to SNR
    # use first repetition where there should be no signal as reference region
    # for SNR

    if norm_to_snr_before_fit:
        # norm to background region of first repetition
        spec_norm = np.zeros_like(spectrum)
        for n in range(0, spectrum.shape[0], 1):
            spec_norm[n, :] = (
                spectrum[n, :]
                - np.mean(
                    spectrum[0, bg_region_first_spec[0] : bg_region_first_spec[1]]
                )
            ) / np.std(spectrum[0, bg_region_first_spec[0] : bg_region_first_spec[1]])
    else:
        # just baseline correction
        spec_norm = np.zeros_like(spectrum)
        for n in range(0, spectrum.shape[0], 1):
            spec_norm[n, :] = spectrum[n, :] - np.mean(
                spectrum[0, bg_region_first_spec[0] : bg_region_first_spec[1]]
            )
    # defining in with region we fit
    width = 1
    # number of reps
    NR = spectrum.shape[0]
    N_peaks = len(peak_positions)
    # defining region in which we will fit

    peak_coeff = np.zeros((NR, N_peaks, 3))
    peak_covariance = np.zeros((NR, N_peaks, 3, 3))
    peak_errors = np.zeros((NR, N_peaks, 3))

    for repetition in range(NR):
        for peak_number, peak_center in enumerate(peak_positions):
            peak_roi = [
                find_range(ppm_axis, peak_center - width),
                find_range(ppm_axis, peak_center + width),
            ]

            try:
                (
                    peak_coeff[repetition, peak_number],
                    peak_covariance[repetition, peak_number],
                ) = curve_fit(
                    fit_function,
                    ppm_axis[peak_roi[0] : peak_roi[1]],
                    spec_norm[repetition, peak_roi[0] : peak_roi[1]],
                    bounds=(
                        [
                            min_fwhm_ppm,
                            peak_center - width / 2.0,
                            np.min(spec_norm[repetition]),
                        ],
                        [5, peak_center + width / 2.0, np.max(spec_norm[repetition])],
                    ),
                )
                peak_errors[repetition, peak_number] = np.sqrt(
                    np.diag(peak_covariance[repetition, peak_number])
                )

            except RuntimeError:
                peak_coeff[repetition, peak_number] = None
                peak_covariance[repetition, peak_number] = None
                peak_errors[repetition, peak_number] = None

            # clean up badly fitted peaks
    for peak in range(peak_coeff.shape[1]):
        for repetition in range(peak_coeff.shape[0]):
            peak_snr = peak_coeff[repetition, peak][2]
            peak_errors[repetition, peak][2]
            if peak_snr > SNR_cutoff:
                # peak needs to have an SNR greater than a certain value
                pass
            else:
                peak_coeff[repetition, peak] = [0, 0, 0]

    if plot is True:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        @widgets.interact(repetition=(0, NR - 1, 1))
        def update(repetition=0):
            [l.remove() for l in ax[0].lines]
            [l.remove() for l in ax[0].lines]

            # Plotting for QA
            combined_fits = 0
            for peak_number, peak in enumerate(peak_positions):
                combined_fits += lorentzian(
                    ppm_axis,
                    peak_coeff[repetition, peak_number][0],
                    peak_coeff[repetition, peak_number][1],
                    peak_coeff[repetition, peak_number][2],
                )
            ax[0].plot(
                ppm_axis,
                spec_norm[repetition, :],
                linewidth="0.5",
                color="r",
                label="Data",
            )
            ax[0].plot(
                ppm_axis,
                combined_fits,
                linestyle="dashed",
                color="k",
                label=str(fit_func_str) + "fit",
            )

            ax[0].set_title("Repetition " + str(repetition))
            for peak_number, peak in enumerate(peak_positions):
                print(str(peak), peak_coeff[repetition, peak_number])
            ax[0].set_xlim([190, 160])
            ax[0].set_ylim(
                [np.min(spec_norm[repetition, :]), np.max(spec_norm[repetition, :])]
            )
            ax[0].set_ylabel("SNR")
            ax[0].set_xlabel(r"$\sigma$[ppm]")

            ax[0].legend()

            [l.remove() for l in ax[1].lines]
            [l.remove() for l in ax[1].lines]

            # Plotting for QA

            for peak_number, peak in enumerate(peak_positions):
                ax[1].plot(
                    ppm_axis,
                    lorentzian(
                        ppm_axis,
                        peak_coeff[repetition, peak_number][0],
                        peak_coeff[repetition, peak_number][1],
                        peak_coeff[repetition, peak_number][2],
                    )
                    - 20 * peak_number,
                    color="C" + str(peak_number),
                    label=str(peak) + " ppm",
                )
            ax[1].set_ylabel("I [a.u.]")
            ax[1].set_yticks([])
            ax[1].set_xlabel(r"$\sigma$[ppm]")
            ax[1].set_title("Repetition " + str(repetition))
            ax[1].legend()
            ax[1].set_xlim([190, 160])

        if norm_to_snr_before_fit:
            # plotting of background region for snr calculation
            fig_bg, bg_axis = plt.subplots(1)
            bg_axis.set_title("Background region - spectrum 0")
            points = np.linspace(0, spectrum[0, :].shape[0], spectrum[0, :].shape[0])
            bg_axis.plot(points, spectrum[0, :])

            bg_axis.fill_between(
                [points[bg_region_first_spec[0]], points[bg_region_first_spec[1]]],
                np.min(spectrum[0]),
                np.max(spectrum[0]),
                alpha=0.3,
                color="C1",
            )
        else:
            pass
    else:
        pass

    return peak_coeff, peak_errors


def integrate_fitted_spectrum(
    experiment_instance,
    ppm_axis,
    spectrum,
    peak_positions,
    peak_coeff,
    plot=False,
    plot_title=None,
    savepath=None,
    fit_func="lorentzian",
):
    """
    Integrates and displays a time curve for 1-D spectral data that was fitted using fit_spectrum
    Parameters
    ----------
    experiment_instance: MRIExperiment instance which contains meta data of the experiment that is being fitted
        This can be a NSPECT type for example.
    ppm_axis: numpy array
        The ppm-scale of the measurements to be examined.
    spectrum: numpy array
        N-dimensional spectrum to be fitted.
    peak_positions: list
        Contains the peak positions where fits where done.
    peak_coeff: numpy array
        coefficients for a 3 parametric lorentzian fit model, shape: (NRepetitions, Number_of_peaks, 3)
    plot: bool
        Select if plot for checking is wanted
    plot_title: str
        title of plot
    Returns
    -------
    peak_integrals: numpy array
        shape: (N_repetitions, N_peaks, 1)
    """
    # interpolate ppm-axis
    # number of reps
    if fit_func == "lorentzian":
        fit_function = lorentzian
    elif fit_func == "sqrt_lorentzian":
        fit_function = sqrt_lorentzian
    else:
        logger.error(
            "Fit function %s not implemented - proceeding to fit using normal lorentzian",
            fit_func,
        )
        fit_function = lorentzian
    NR = spectrum.shape[0]
    N_peaks = len(peak_positions)
    TR = experiment_instance.method["PVM_RepetitionTime"]
    time_scale = np.arange(0, TR * NR, TR) / 1000

    peak_integrals = np.zeros((NR, N_peaks))
    for repetition in range(NR):
        for peak_number, peak in enumerate(peak_positions):
            # integrate the fits
            peak_integrals[repetition, peak_number] = np.sum(
                np.abs(
                    fit_function(
                        ppm_axis,
                        peak_coeff[repetition, peak_number][0],
                        peak_coeff[repetition, peak_number][1],
                        peak_coeff[repetition, peak_number][2],
                    )
                )
            )
    peak_integrals = peak_integrals
    if plot:
        fig, ax = plt.subplots(1)
        for n in range(len(peak_positions)):
            ax.plot(
                time_scale, peak_integrals[:, n], label=str(peak_positions[n]) + " ppm"
            )
        ax.set_xlabel("Time [s] ")
        ax.set_ylabel("I [a.u.]")
        ax.legend()
        ax.set_title(plot_title)
    if savepath:
        plt.savefig(savepath + plot_title + "_timecurve.png")
    return peak_integrals, time_scale


def calculate_pyruvate_to_lactate_auc(lactate_timecurve, pyruvate_timecurve):
    """
    Calculate pyruvate to lactate area under the curve ratios for a given experiment.

    Parameters
    ----------
    lactate_timecurve: numpy array
    pyruvate_timecurve: numpy array

    Returns
    -------
    lac_pyr: float
        lactate/pyruvate ratio
    """
    pyruvate_auc = np.sum(pyruvate_timecurve)
    lactate_auc = np.sum(lactate_timecurve)
    lac_pyr = lactate_auc / pyruvate_auc

    return lac_pyr


def conv_func(
    name: str = None, lb_hz: float = 0.0, time_s: np.ndarray = None
) -> np.ndarray:
    """Define a convolution function (for linebroadening)

    Parameters
    ----------
    name: str
        Name of convolution function, e.g. "Gaussian", "Lorentzian", default is "Gaussian"
    lb_hz: float
        Linebroadening in Hz
    time_s: array
        time in seconds
    Returns
    ----------

    """
    if name == "Gaussian":
        filter = np.exp(-((np.pi * lb_hz * time_s) ** 2) / (2 * np.log(2)))

    elif name == "Lorentzian":
        filter = np.exp(-(lb_hz * time_s))

    elif name is None:
        # default Gaussian:
        filter = np.exp(-((np.pi * lb_hz * time_s) ** 2) / (2 * np.log(2)))
    else:
        # default Gaussian:
        filter = np.exp(-((np.pi * lb_hz * time_s) ** 2) / (2 * np.log(2)))
    return filter


def multi_dim_linebroadening(
    input_data: Any = None,
    lb_hz: float = 0.0,
    fid_start: int = 0,
    fid_end: int = 0,
    dwell_time_s: float = 0,
    conv_func_name: str = "Gaussian",
    signal_domain: str = "spectral",
    axis: int = 0,
    data_obj: Any = None,
) -> Optional[np.ndarray]:
    """
    Applies Gaussian linebroadening on spectra in all voxel

    Parameters
    ----------
    data_obj : MRIExperiment
    input_data: N-D array
        echoes - z - x -y -rep - chans
    lb_hz: float
        linebroadening in Hz
    fid_start: Integer
        Start index
    fid_end: Integer
        End index
    dwell_time_s: float
        sampling time per FID point [second]
    signal_domain: string
        whether data is in spectral or temporal (FID) domain

    Returns
    -------
    linebroadend_data: N-D array

    """
    if input_data is None:
        try:
            input_data = data_obj.reco_data
        except BaseException:
            raise Exception("provide either input_data or data_obj")

    else:
        pass

    if (
        dwell_time_s == 0 and data_obj is not None
    ):  # define sampling duration to do: check
        from ...utils.time.temporal import get_dwell_time_s

        dwell_time_s = get_dwell_time_s(data_obj=data_obj)
    elif dwell_time_s == 0 and data_obj is None:
        return None
    else:
        pass

    def wrapper_apply_lb(arr):
        return apply_lb(
            input_data=arr,
            lb_hz=lb_hz,
            fid_start=fid_start,
            fid_end=fid_end,
            dwell_time_s=dwell_time_s,
            conv_func_name=conv_func_name,
            signal_domain=signal_domain,
            data_obj=data_obj,
        )

    # Apply along the first axis (axis=0)
    linebroadend_data = np.apply_along_axis(wrapper_apply_lb, axis=axis, arr=input_data)

    return linebroadend_data


def multi_dim_zeropadding(
    input_data: np.ndarray = None,
    num_zp: int = 0,
    fid_start: int = 0,
    fid_end: int = 0,
    signal_domain: str = "spectral",
    data_obj: Any = None,
):
    """
    Zeropad multidimensional data along spectral domain

    Parameters
    ----------
    input_data: N-D array
        echoes - z - x -y -rep - chans
    num_zp: int
        number of points to add to the spectrum
    fid_start: Integer
        Start index
    fid_end: Integer
        End index
    signal_domain: string
        whether data is in spectral or temporal (FID) domain
    data_obj: Any

    Returns
    -------
    zeropadded data ([echoes + num_zp] - z - x -y -rep - chans)

    """
    if input_data is None:
        try:
            input_data = data_obj.seq2d_reordered
        except BaseException:
            try:
                input_data = data_obj.spec
            except BaseException:
                raise Exception("provide either input_data or spec_data")
    else:
        pass

    def wrapper_zeropad_data(arr):
        return zeropad_data(
            input_data=arr,
            num_zp=num_zp,
            fid_start=fid_start,
            fid_end=fid_end,
            signal_domain=signal_domain,
        )

    # Apply along the first axis (axis=0)
    zeropadded_data = np.apply_along_axis(wrapper_zeropad_data, axis=0, arr=input_data)
    return zeropadded_data


def apply_lb(
    input_data: np.array = None,
    lb_hz: float = 0.0,
    fid_start: int = 0,
    fid_end: int = 0,
    dwell_time_s: float = 0.0,
    conv_func_name: str = "Gaussian",
    signal_domain: str = "spectral",
    data_obj: Any = None,
    plot: bool = False,
) -> Optional[np.ndarray]:
    """
    Apply linebroadening to the input data.


    Parameters
    ----------
    input_data: 1D-array
        1D-spectrum or FID (use signal_domain to specify).
    lb_hz: float
        linebroadening in Hz.
    fid_start: int
        At which index the FID starts.
    fid_end: int
        At which index the FID ends.
    dwell_time_s: float
        dwell time in seconds.
    conv_func_name: {"Gaussian"}, string
        Which function to use in the convolution. Options: Gaussian, Lorentzian.
    signal_domain: {"spectral", "temporal"}, string
        In which domain the input data is.
    data_obj: hypermri object
        data object from which the sampling time can be extracted.
    plot: bool
        Select if plotting for checking is wanted.

    Returns
    -------
    A sepctrum on which linebroadening has been applied. If the input data was in the temporal domain (aka FID), a
    spectrum will be returned. If the data is in the spectral domain, a linebroadened spectrum will be returned.

    Examples
    --------
    >>> from hypermri.analysis.core import apply_lb, generate_fid
    >>> from hypermri.utils.time.temporal import calc_sampling_time_axis
    >>> dt = 5e-4
    >>> npoints=1000
    >>> time_axis_s = calc_sampling_time_axis(fid_start=0, fid_end=npoints, dwell_time_s=dt)

    >>> f = np.fft.fftshift(np.fft.fftfreq(npoints, d=dt))
    >>> lb_hz=20.0
    >>> fid = generate_fid(time_axis_s=time_axis_s, T2_s=0.1, freq0_Hz=100.0, phase0_rad=0.5, dwell_time_s=dt,
    >>> npoints=npoints)
    >>> spec_lb = apply_lb(input_data=fid, lb_hz=lb_hz, dwell_time_s=dt, signal_domain="temporal")

    >>> plt.figure()
    >>> plt.subplot(1,2,1)
    >>> plt.plot(time_axis_s, np.real(fid))
    >>> plt.plot(time_axis_s, np.imag(fid))
    >>> plt.plot(time_axis_s, np.abs(fid))
    >>> plt.xlabel('t [s]')

    >>> plt.subplot(1,2,2)
    >>> plt.plot(f, np.abs(np.fft.fftshift(np.fft.fft(fid)))/np.max(np.abs(np.fft.fftshift(np.fft.fft(fid)))),
    >>> label='original FID')
    >>> plt.plot(f, np.abs(spec_lb)/np.max(np.abs(spec_lb)),label=f'linebroadened FID {lb_hz}Hz')
    >>> plt.xlabel('f [Hz]')
    >>> plt.legend()
    """
    # check if data was input:
    if input_data is None:
        raise Exception("No input_data")
    else:
        pass

    input_data = np.atleast_1d(np.squeeze(input_data))

    # apply 1D Fourier-Transform
    if signal_domain == "temporal":
        pass
    elif signal_domain == "spectral":
        input_data = np.fft.ifft(np.fft.ifftshift(input_data, axes=(0,)), axis=0)

    # check dimensions of input data:
    if input_data.ndim == 1:
        if fid_end == 0:
            fid_end = input_data.shape[0]
        elif fid_end > input_data.shape[0]:
            fid_end = input_data.shape[0]
        elif fid_end < fid_start:
            raise Exception("fid_end has to be higher than fid_start")
        else:
            pass
    else:
        raise Exception("input_data has to have dim=1")

    if (
        dwell_time_s == 0 and data_obj is not None
    ):  # define sampling duration to do: check
        from ...utils.time.temporal import get_dwell_time_s

        dwell_time_s = get_dwell_time_s(
            data_obj=data_obj
        )  # data_obj.method["PVM_SpecDwellTime"] * 2.0 / 1000.0 / 1000.0
    elif dwell_time_s == 0 and data_obj is None:
        return None
    else:
        pass

    # get time axis in seconds
    if dwell_time_s == 0:
        raise Exception("sampling duration has to be > 0")
    else:
        from ...utils.time.temporal import calc_sampling_time_axis

        time_ax = calc_sampling_time_axis(
            npoints=len(input_data), dwell_time_s=dwell_time_s
        )

    # define filter:
    filter = conv_func(name=conv_func_name, lb_hz=lb_hz, time_s=time_ax)

    fid_times_dec_exp = input_data[fid_start:fid_end] * filter
    if signal_domain == "temporal":
        return_lb = fid_times_dec_exp
    elif signal_domain == "spectral":
        return_lb = np.fft.fftshift(np.fft.fft(fid_times_dec_exp, axis=0), axes=(0,))
    else:
        return_lb = np.fft.fftshift(np.fft.fft(fid_times_dec_exp, axis=0), axes=(0,))
        warnings.warn(
            "signal_domain not recognized - returning spectrum as output - use signal_domain='spectral' or 'temporal'"
        )
    if plot:
        _, ax = plt.subplots(1, 1)
        ax.plot(time_ax, np.real(input_data), label="original FID")
        ax.plot(time_ax, np.real(fid_times_dec_exp), label="linebroadened FID")
        ax.set_xlabel("t [s]")
        ax.legend()

    return return_lb


def zeropad_data(
    input_data: np.ndarray = None,
    num_zp: int = 0,
    fid_start: int = 0,
    fid_end: int = 0,
    signal_domain: str = "spectral",
):
    """
    Zeropad multidimensional data along temporal domain. Return data in the set domain.


    Parameters
    ----------
    input_data: 1D-array
        1D-spectrum
    num_zp: float
        number of points to add to the fid
    signal_domain: {"spectral", "temporal"}, string
        In which domain the input data is

    Returns
    -------
    Data that has been zeropadded. If input data was in the temporal domain (aka FID), a zeropadded FID is returned. If
    the data is in the spectral domain, a zeropadded spectrum will be returned.

    """
    # check if data was input:
    if input_data is None:
        raise Exception("No input_data")
    else:
        pass

    # apply 1D Fourier-Transform
    if signal_domain == "temporal":
        pass
    elif signal_domain == "spectral":
        input_data = np.fft.ifft(np.fft.ifftshift(input_data, axes=(0,)), axis=0)

    # check dimensions of input data:
    if input_data.ndim == 1:
        if fid_end == 0:
            fid_end = input_data.shape[0]
        elif fid_end > input_data.shape[0]:
            fid_end = input_data.shape[0]
        elif fid_end < fid_start:
            raise Exception("fid_end has to be higher than fid_start")
        else:
            pass
        fid = input_data[fid_start:fid_end]
    else:
        raise Exception("input_data has to have dim=1")

    # append zeros to the FID:
    fid_zp = np.append(fid, np.zeros((num_zp, 1), dtype=type(fid)))
    fid_zp = np.pad(fid, (0, num_zp), mode="constant", constant_values=(0, 0))
    if signal_domain == "temporal":
        return fid_zp
    elif signal_domain == "spectral":
        # apply 1D Fourier-Transform
        spec_zp = np.fft.fftshift(np.fft.fft(fid_zp, axis=0), axes=(0,))
        return spec_zp


# chemical shift in ppm:
METABOLITE_CS_PPM = {
    "pyruvate": 171.076,  # Lactate - 920Hz
    "lactate": 183.35,  # determined from repeated experiments
    "alanine": 176.5,  # https://www.pnas.org/doi/10.1073/pnas.0706235104
    "pyruvatehydrate": 179.5,  # https://www.pnas.org/doi/10.1073/pnas.0706235104
    "fumarate": 175.4,
    "malate1": 181.7,  # https://pubs.acs.org/doi/epdf/10.1021/jacs.9b10094
    "malate4": 180.5,  # https://pubs.acs.org/doi/epdf/10.1021/jacs.9b10094
    # https://pubmed.ncbi.nlm.nih.gov/30639333/ (fits human in vivo data @
    # 3T)
    "bicarbonate": 161.0,
    "urea": 163.5,  # trust me bro
    "co2": 124.5,  # https://www.pnas.org/doi/10.1073/pnas.0706235104
    "aspartate1": 176.92,  # doi: 10.3389/fphys.2021.792769
    "aspartate4": 180.20,  # doi: 10.3389/fphys.2021.792769
}


def get_metab_cs_ppm(
    metab: str = None, nucleus: str = "13c", pos: int = None, help: bool = False
) -> Optional[float]:
    """
    Get the literature value of the chemical shift of different metabolites

    Parameters
    ----------
    metab: str,
        The metabolite of interest.
    help: bool, optional
        Print the available metabolites.
    nucleus: str,
        The nucleus of the metabolite. So far only 13C is supported.
    pos: int
        which position of the metabolite

    Returns
    -------
    chemical shift (float) in ppm

    Examples
    --------
    >>> import hypermri.analysis.core as uts
    >>> uts.get_metab_cs_ppm(help=True)
    >>> pyruvate: 171.076 ppm
    >>> lactate: 183.35 ppm
    >>> alanine: 176.5 ppm
    >>> pyruvatehydrate: 179.5 ppm
    >>> fumarate: 175.4 ppm
    >>> malate1: 181.7 ppm
    >>> malate4: 180.5 ppm
    >>> bicarbonate: 161.0 ppm
    >>> co2: 124.5 ppm
    >>> urea: 163.5 ppm

    >>> import hypermri.analysis.core as uts
    >>> pyruvate_ppm = uts.get_metab_cs_ppm(metab="pyruvate")
    >>> print(pyruvate_ppm)
    """
    if metab == "" and help is True:
        pass
    elif metab != "":
        pass
    # valid nuclei:
    nuclei = [
        "13C",
    ]

    if metab is None and help is True:
        for m in METABOLITE_CS_PPM.keys():
            print(f"{m}: {METABOLITE_CS_PPM[m]} ppm")
        return None

    else:
        # Convert metab to title case (first letter capitalized)
        metab = metab.lower()

        # Checking for valid metabolite
        if metab in METABOLITE_CS_PPM:
            return METABOLITE_CS_PPM[metab]
        elif metab not in METABOLITE_CS_PPM and help is True:
            print(f"Invalid metabolite {metab}. Available metabolites are:")
            for m in METABOLITE_CS_PPM.keys():
                print(m)
            return None
        else:
            return None


def find_npeaks(
    input_data: np.ndarray = None,
    npeaks: int = None,
    freq_range: np.ndarray = None,
    plot: bool = False,
    find_peaks_params: dict = {},
    sort_order: str = None,
):
    """
    Finds and plots the specified number of highest peaks in a given spectrum.

    Parameters
    ----------
    input_data (array-like):
        The input data containing the spectrum.
    npeaks (int):
        The number of highest peaks to identify.
    freq_range (array-like):
        The frequency range corresponding to the input data.
    plot (bool):
        If True, plots the spectrum and the identified peaks.
    find_peaks_params (dict, optional):
        Additional parameters to pass to scipy's find_peaks function.
    sort_order (string, optional):
        return indices in "ascending", "descending" or non-sorted (default, None) order

    Example
    -------
    >>> find_npeaks(input_data=test_spec, npeaks=6, freq_range=fit_params['freq_axis_hz'], plot=True, find_peaks_params={'distance': 30})

    Returns
    -------
    array: Indices of the top peaks in the input data.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import find_peaks

    # Compatible colormap getter for different matplotlib versions
    try:
        from matplotlib import colormaps

        def get_colormap_colors(cmap_name, n):
            cmap = colormaps.get_cmap(cmap_name)
            return [cmap(i) for i in np.linspace(0, 1, n)]

    except ImportError:
        from matplotlib.cm import get_cmap

        def get_colormap_colors(cmap_name, n):
            cmap = get_cmap(cmap_name)
            return [cmap(i) for i in np.linspace(0, 1, n)]

    spec = np.abs(np.squeeze(input_data))

    # Finding peaks
    peaks, properties = find_peaks(spec, **find_peaks_params)

    # Extract the heights of these peaks
    peak_heights = np.squeeze(spec)[peaks]

    # Sort peaks by height in descending order and get the top n peaks
    top_peaks_indices = peaks[np.argsort(-peak_heights)][:npeaks]

    if sort_order == "descending":
        top_peaks_indices = np.sort(top_peaks_indices)[
            ::-1
        ]  # This sorts the array and then reverses it
    elif sort_order == "ascending":
        top_peaks_indices = np.sort(top_peaks_indices)  # This sorts the array
    else:
        pass

    if plot:
        if freq_range is None:
            freq_range = np.linspace(0, spec.shape[0] - 1, spec.shape[0])

        fig, ax = plt.subplots(1)
        ax.plot(freq_range, np.squeeze(np.abs(spec)), label="spec")

        # Get colors from colormap
        colors = get_colormap_colors("jet", len(top_peaks_indices))

        for k, i in enumerate(top_peaks_indices):
            color = colors[k]
            ax.vlines(freq_range[i], 0, spec[i], colors=color, label=f"Peak {k}")

        plt.legend()
        # Show plot (optional, depending on your environment)
        plt.show()

    return top_peaks_indices


def find_npeaks_nd(
    input_data=None,
    npeaks=None,
    freq_range=None,
    plot=False,
    find_peaks_params={},
    sort_order=None,
):
    """
    Finds the specified number of highest peaks along a specific dimension
    for each slice in N-dimensional spectroscopic data.

    Automatically detects the dimension named 'acquisition_idx' if input is
    an xarray DataArray, otherwise defaults to the *last* dimension. Finds
    'npeaks' independently for each 1D slice along the target dimension.

    Parameters
    ----------
    input_data (array-like or xarray.DataArray): The ND input data.
        E.g., shape (channels, reps, x, y, spectral_points).
    npeaks (int): The number of highest peaks to identify per 1D slice.
    freq_range (array-like, optional): The frequency range (or coordinates)
        corresponding to the peak-finding dimension. Used for plotting if
        data is not xarray or lacks coordinates on the target dim.
        If None and plotting is True, indices are used for the x-axis.
    plot (bool): If True, plots the spectrum and identified peaks for the
        *first* available slice (index [0, 0, ...] across non-target dims).
    find_peaks_params (dict, optional): Additional parameters passed to
        scipy.signal.find_peaks (e.g., {'height': 10, 'distance': 5}).
    sort_order (str, optional): Order of returned peak indices within each slice:
        "ascending", "descending", or None (original find_peaks order, typically ascending).

    Returns
    -------
    numpy.ndarray or None: An array containing the indices of the top peaks
        within the target dimension for each slice. The shape will be
        input_data.shape excluding the peak-finding dimension, followed by
        the npeaks dimension.
        (e.g., if input shape is (X, Y, Z) and peaks are found along Z (axis=-1),
         output shape is (X, Y, npeaks)).
        Indices are float type; NaN indicates fewer than npeaks were found.
        Returns None if input_data or npeaks is None or on error.

    Example:
    -------
    Assuming data is (time, voxels, spectral_points)
    And spectral_points is the last dimension (axis=-1)
    peak_indices = find_npeaks_nd(data, npeaks=5, find_peaks_params={'height': 100})
    * peak_indices shape will be (time, voxels, 5)
    * peak_indices[t, v, :] gives the 5 peak indices for time t, voxel v.

    * With xarray having dim 'acquisition_idx'
    * peak_indices_xr = find_npeaks_nd(data_xr, npeaks=3, plot=True)
    """
    try:
        from matplotlib import colormaps

        def get_colormap_colors(cmap_name, n):
            cmap = colormaps.get_cmap(cmap_name)
            return [cmap(i) for i in np.linspace(0, 1, n)]

    except ImportError:
        from matplotlib.cm import get_cmap

        def get_colormap_colors(cmap_name, n):
            cmap = get_cmap(cmap_name)
            return [cmap(i) for i in np.linspace(0, 1, n)]

    from scipy.signal import find_peaks

    if input_data is None or npeaks is None:
        print("Error: input_data and npeaks must be provided.")
        return None
    if not isinstance(npeaks, int) or npeaks <= 0:
        print("Error: npeaks must be a positive integer.")
        return None

    data_np = None
    target_axis = -1  # Default to LAST axis
    target_dim_name = None
    original_shape = None
    coords = None
    is_xarray_input = False

    # --- Input Type and Dimension Detection ---
    if isinstance(input_data, xr.DataArray):
        is_xarray_input = True
        original_shape = input_data.shape
        print("Input is xarray.DataArray. Checking for 'acquisition_idx' dimension.")
        if "acquisition_idx" in input_data.dims:
            target_dim_name = "acquisition_idx"
            try:
                # Use get_axis_num for robust axis index retrieval
                target_axis = input_data.get_axis_num(target_dim_name)
                print(f"Found '{target_dim_name}' at axis {target_axis}.")
                # Try to get coordinates automatically if freq_range not
                # provided
                if freq_range is None and target_dim_name in input_data.coords:
                    coords = input_data.coords[target_dim_name].values
                    print(
                        f"Using coordinates from '{target_dim_name}' for frequency axis."
                    )
            except (
                Exception
            ) as e:  # Catch potential errors (e.g., dim exists but not in main data)
                print(
                    f"Warning: Dimension '{target_dim_name}' found but failed to get axis index ({e}). Defaulting to last axis."
                )
                target_axis = -1  # Fallback to last axis
        else:
            print("Dimension 'acquisition_idx' not found. Defaulting to last axis.")
            target_axis = -1

        # If defaulted or didn't find coords initially, try getting coords from
        # the chosen axis
        if coords is None:
            # Ensure target_axis is valid before accessing dims
            if target_axis >= len(input_data.dims) or target_axis < -len(
                input_data.dims
            ):
                print(
                    f"Error: Fallback axis {target_axis} is invalid for dims {input_data.dims}. Cannot proceed."
                )
                return None
            target_dim_name = input_data.dims[
                target_axis
            ]  # Get name of the axis we ended up with
            if freq_range is None and target_dim_name in input_data.coords:
                coords = input_data.coords[target_dim_name].values
                print(
                    f"Using coordinates from dimension '{target_dim_name}' (axis {target_axis}) for frequency axis."
                )

        data_np = input_data.values  # Work with numpy array underneath
    else:
        # Assume numpy array or array-like
        try:
            data_np = np.asarray(input_data)  # Convert input to numpy array
            original_shape = data_np.shape
            target_axis = -1  # Default to last axis for numpy
            if is_xarray_input:  # It was xarray but 'acquisition_idx' wasn't found
                print("Defaulting to last axis for peak finding.")
            else:  # Was not xarray input
                print(
                    "Input is NumPy array or array-like. Defaulting to last axis for peak finding."
                )
        except Exception as e:
            print(f"Error: Could not convert input_data to NumPy array: {e}")
            return None

    # --- Validate Dimensions and Axis ---
    if data_np.ndim == 0:
        print("Error: Input data is scalar.")
        return None
    elif data_np.ndim == 1:
        # If input is 1D, ensure target_axis is 0
        target_axis = 0
        print("Input data is 1D. Finding peaks along the single axis (axis 0).")
    # Check axis validity after potential modification for 1D case
    if target_axis < -data_np.ndim or target_axis >= data_np.ndim:
        print(
            f"Error: Determined target_axis {target_axis} is invalid for data with {data_np.ndim} dimensions."
        )
        return None
    # Ensure target_axis is positive for apply_along_axis internal use and
    # consistent indexing
    target_axis_positive = target_axis % data_np.ndim

    # --- Helper function to find peaks in 1D ---
    def _find_peaks_1d_slice(spec_1d, npeaks, find_peaks_params, sort_order):
        """Internal function: returns integer indices, uses -1 for padding."""
        placeholder = -1  # Use -1 instead of NaN

        if spec_1d is None or spec_1d.size == 0 or np.all(np.isnan(spec_1d)):
            # Return array of placeholders with integer type
            return np.full(npeaks, placeholder, dtype=int)

        spec = np.abs(spec_1d)
        finite_mask = np.isfinite(spec)
        if not np.any(finite_mask):
            return np.full(npeaks, placeholder, dtype=int)
        spec_finite = np.where(finite_mask, spec, 0)

        try:
            peaks, properties = find_peaks(spec_finite, **find_peaks_params)
        except ValueError as e:
            print(
                f"Warning: find_peaks failed for a slice: {e}. Returning placeholders."
            )
            return np.full(npeaks, placeholder, dtype=int)

        if peaks.size == 0:
            # No peaks found, return placeholders
            return np.full(npeaks, placeholder, dtype=int)

        peak_heights = spec[peaks]
        sorted_peak_indices_by_height = np.argsort(-peak_heights)

        num_peaks_found = len(sorted_peak_indices_by_height)
        num_peaks_to_return = min(npeaks, num_peaks_found)

        top_peaks_indices = peaks[sorted_peak_indices_by_height[:num_peaks_to_return]]

        # Apply sorting by index value if requested
        if sort_order == "descending":
            top_peaks_indices = np.sort(top_peaks_indices)[::-1]
        elif sort_order == "ascending":
            top_peaks_indices = np.sort(top_peaks_indices)

        # Pad with placeholder (-1) if fewer than npeaks were found/selected
        # Ensure output array has integer type
        padded_indices = np.full(
            npeaks, placeholder, dtype=int
        )  # Initialize with placeholder
        # Assign the valid integer indices
        padded_indices[:num_peaks_to_return] = top_peaks_indices  # Already int

        return padded_indices

    # --- Apply the peak finding function along the target axis ---
    target_dim_size = data_np.shape[target_axis_positive]
    print(
        f"Applying peak finding along axis {target_axis_positive}(dimension size {target_dim_size}) across input shape {data_np.shape}..."
    )
    try:
        # Suppress potential RuntimeWarnings from NaN slices if desired
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            all_peak_indices = np.apply_along_axis(
                _find_peaks_1d_slice,
                axis=target_axis_positive,
                arr=data_np,
                # Pass additional arguments to the helper function:
                npeaks=npeaks,
                find_peaks_params=find_peaks_params,
                sort_order=sort_order,
            )
    except Exception as e:
        print(f"\nError during np.apply_along_axis: {e}")
        import traceback

        traceback.print_exc()
        return None

    print(f"Peak finding complete. Result shape: {all_peak_indices.shape}")

    # --- Plotting (optional, for the first slice) ---
    if plot:
        print("Plotting results for the first slice...")
        try:
            # Determine the frequency axis data for plotting
            plot_freq_axis = None
            # Default label
            axis_label = f"Index (Axis {target_axis_positive})"
            if coords is not None:
                # Ensure coords match the dimension length
                if len(coords) == target_dim_size:
                    plot_freq_axis = coords
                    axis_label = (
                        target_dim_name
                        if target_dim_name
                        else f"Axis {target_axis_positive} Coords"
                    )
                    print(f"Using coordinates for plot x-axis ({axis_label}).")
                else:
                    print(
                        f"Warning: Auto - detected coordinates length({len(coords)}) does not match target axis length({target_dim_size}). Plotting against indices."
                    )
                    plot_freq_axis = np.arange(target_dim_size)
            elif freq_range is not None:
                # Ensure provided freq_range matches the dimension length
                if len(freq_range) == target_dim_size:
                    plot_freq_axis = np.asarray(
                        freq_range
                    )  # Ensure numpy array for indexing
                    axis_label = "Provided freq_range"
                    print("Using provided freq_range for plot x-axis.")
                else:
                    print(
                        f"Warning: Provided freq_range length({len(freq_range)}) does not match target axis length({target_dim_size}). Plotting against indices."
                    )
                    plot_freq_axis = np.arange(target_dim_size)
            else:
                # Use indices if no frequency info available
                plot_freq_axis = np.arange(target_dim_size)
                print(
                    "Warning: No freq_range provided or found in coordinates. Plotting against indices."
                )

            # Prepare indices to select the first slice's data and peak results
            slice_spec_indices = [0] * data_np.ndim
            slice_peaks_indices = []
            slice_desc = []  # Description for plot title

            if data_np.ndim == 1:
                # Simple case: data is already 1D
                spec_to_plot = np.abs(data_np)
                # Result is already 1D (npeaks,)
                peaks_to_plot_indices_float = all_peak_indices
                slice_desc.append("1D Data")
            else:
                # ND case: Select the first element along all non-target axes
                current_peak_array_dim = 0  # Tracks position in the result array's dims
                for i in range(data_np.ndim):
                    if i == target_axis_positive:
                        slice_spec_indices[i] = slice(
                            None
                        )  # Get all along target axis for spectrum plot
                    else:
                        # Index 0 for the spectrum slice
                        slice_spec_indices[i] = 0
                        # Index 0 for the results slice (corresponding dim)
                        slice_peaks_indices.append(0)
                        # Get dimension name for description
                        dim_name = "N/A"
                        if is_xarray_input:
                            try:
                                dim_name = input_data.dims[i]
                            except IndexError:
                                pass
                        elif original_shape:  # Use original shape info if available
                            dim_name = f"Dim{i}"

                        slice_desc.append(f"{dim_name}=0")
                        current_peak_array_dim += 1  # Move to next dim in result array

                # Finally, select all npeaks along the last dimension of the
                # result array
                slice_peaks_indices.append(slice(None))

                slice_spec_indices_tuple = tuple(slice_spec_indices)
                slice_peaks_indices_tuple = tuple(slice_peaks_indices)

                # Get the spectrum data for the selected slice
                spec_to_plot = np.abs(data_np[slice_spec_indices_tuple])
                # Get the peak indices for the selected slice
                peaks_to_plot_indices_float = all_peak_indices[
                    slice_peaks_indices_tuple
                ]

            # Filter out NaNs from peak indices and convert to integer for
            # indexing
            valid_peaks_mask = ~np.isnan(peaks_to_plot_indices_float)
            peaks_to_plot_indices = peaks_to_plot_indices_float[
                valid_peaks_mask
            ].astype(int)
            num_peaks_plotted = len(peaks_to_plot_indices)

            # Create the plot
            fig, ax = plt.subplots(1, figsize=(10, 5))  # Wider plot
            plot_label = (
                f"Spectrum ({', '.join(slice_desc)})" if slice_desc else "Spectrum"
            )
            ax.plot(plot_freq_axis, spec_to_plot, label=plot_label)

            plot_title = f"Spectrum and Top {num_peaks_plotted}/{npeaks} Peaks"
            if slice_desc:
                plot_title += f" (Slice: {', '.join(slice_desc)})"

            if num_peaks_plotted > 0:
                # Get colors from colormap
                colors = get_colormap_colors("jet", num_peaks_plotted)
                # Get the heights of the peaks being plotted from the
                # *absolute* spectrum
                peak_heights_plot = spec_to_plot[peaks_to_plot_indices]

                for k, peak_idx in enumerate(peaks_to_plot_indices):
                    # Ensure index is within bounds (should be, but safety
                    # check)
                    if 0 <= peak_idx < len(plot_freq_axis):
                        color = colors[k]
                        freq_val = plot_freq_axis[peak_idx]
                        # Use .item() if freq_val might be 0-d array (e.g.,
                        # single point coord)
                        freq_val_item = (
                            freq_val.item()
                            if isinstance(freq_val, np.ndarray)
                            else freq_val
                        )

                        ax.vlines(
                            freq_val_item,
                            0,
                            peak_heights_plot[k],
                            colors=color,
                            linestyles="--",
                            label=f"Peak @ Idx {peak_idx}",
                        )
                        # Optional: Add text label for index near the peak top
                        ax.text(
                            freq_val_item,
                            peak_heights_plot[k],
                            f" {peak_idx}",
                            color=color,
                            va="bottom",
                            ha="center",
                            fontsize=8,
                            rotation=90,
                        )
                    else:
                        print(
                            f"Warning: Peak index {peak_idx} out of bounds for frequency axis, skipping plot marker."
                        )

                # Improve legend handling for many peaks
                handles, labels = ax.get_legend_handles_labels()
                # Keep only the first label ('Spectrum') and maybe one 'Peak'
                # label summary
                unique_labels = {}
                new_handles = []
                new_labels = []
                peak_label_added = False
                for handle, label in zip(handles, labels):
                    is_peak = "Peak @" in label
                    if not is_peak:  # Keep non-peak labels (like 'Spectrum')
                        if label not in unique_labels:
                            unique_labels[label] = handle
                            new_handles.append(handle)
                            new_labels.append(label)
                    elif not peak_label_added:  # Add only one summary label for peaks
                        unique_labels["Peak"] = handle  # Use a generic handle
                        new_handles.append(
                            plt.Line2D([0], [0], linestyle="--", color="gray")
                        )  # Generic peak line
                        new_labels.append(f"Detected Peaks ({num_peaks_plotted})")
                        peak_label_added = True

                ax.legend(new_handles, new_labels)

            else:
                ax.legend()  # Show legend even if only spectrum is plotted (no peaks found)

            ax.set_title(plot_title)
            ax.set_xlabel(axis_label)
            ax.set_ylabel("Absolute Intensity")
            ax.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            # Catch potential errors during slicing or plotting
            print(f"\nError during plotting: {e}")
            import traceback

            traceback.print_exc()
            print("Please check data dimensions, freq_range, and matplotlib setup.")

    return all_peak_indices


def freq_Hz_to_ppm(
    freq_Hz: float = 0.0,
    hz_axis: Optional[np.array] = None,
    ppm_axis: Optional[np.array] = None,
    ppm_axis_flipped: bool = False,
    data_obj: Any = None,
    ppm_centered_at_0: bool = False,
) -> Optional[Union[float, np.ndarray]]:
    """
    Converts frequency values from Hz to parts per million (ppm) based on provided Hz and ppm axes.

    This function allows for the conversion of frequency values from Hertz (Hz) to parts per million (ppm),
    which is often required in spectroscopy for chemical shift calculations. The conversion relies on
    matching frequency values in Hz to their corresponding values in ppm using provided frequency axes.
    The function can handle single values, lists, or arrays of frequencies to be converted.

    Parameters
    ----------
    freq_Hz : float or ndarray, optional
        Frequency value(s) in Hz to be converted. Can be a single value, list, or ndarray. Default is 0.
    hz_axis : ndarray, optional
        The frequency axis in Hz. Required if `data_obj` is not provided. Default is None.
    ppm_axis : ndarray, optional
        The frequency axis in ppm. Required if `data_obj` is not provided. Default is None.
    ppm_axis_flipped : bool, optional
        Set to True if the ppm axis is in descending order. This affects the conversion calculation.
        Default is False.
    data_obj : MRIExperiment, optional
        An object containing the Hz and ppm axes, used if either axis is not explicitly provided.
        Default is None.
    ppm_centered_at_0 : bool, optional
        If True, centers the ppm axis at 0. This parameter is considered in the conversion calculation.
        Default is False.

    Returns
    -------
    ndarray or float
        The converted frequency values in ppm. The return type matches the type of `freq_Hz` input
        (i.e., if `freq_Hz` is a single value, a single float is returned; if it's an array, an ndarray is returned).

    Raises
    ------
    ValueError
        If neither `hz_axis` nor `ppm_axis` is provided and `data_obj` is None.

    Examples
    --------
    >>> hz_axis = np.linspace(-2000, 2000, 4000)
    >>> ppm_axis = np.linspace(10, -10, 4000)
    >>> freq_Hz_to_ppm(freq_Hz=500, hz_axis=hz_axis, ppm_axis=ppm_axis)
    2.5
    """
    if hz_axis is None or ppm_axis is None:
        if data_obj is None:
            raise ValueError("Provide either hz_axis and ppm_axis or data_obj!")
            return None
        if hz_axis is None:
            hz_axis = get_freq_axis(
                data_obj=data_obj, unit="Hz"
            )  # Assuming get_freq_axis is defined elsewhere
        if ppm_axis is None:
            ppm_axis = get_freq_axis(
                data_obj=data_obj, unit="ppm"
            )  # Assuming get_freq_axis is defined elsewhere

    if ppm_axis_flipped:
        ppm_axis = np.flip(ppm_axis)

    if ppm_centered_at_0:
        ppm_axis = center_freq_axis(
            freq_axis=ppm_axis
        )  # Assuming center_freq_axis is defined elsewhere
        hz_axis = center_freq_axis(
            freq_axis=hz_axis
        )  # Assuming center_freq_axis is defined elsewhere

    # calculate a "conversion function" using linear regression
    from scipy.stats import linregress

    f_hz_to_ppm = linregress(hz_axis, ppm_axis)

    input_was_list = isinstance(freq_Hz, list)  # Check if input is a list

    # Ensure freq_Hz is an array for vectorized operations
    freq_Hz = np.asarray(freq_Hz)

    # Convert using the linear regression results
    freq_ppm = f_hz_to_ppm.intercept + f_hz_to_ppm.slope * freq_Hz

    # Return the output in the same type as the input
    if input_was_list:
        return freq_ppm.tolist()  # Convert back to list if the input was a list
    elif isinstance(freq_Hz, np.ndarray):
        return freq_ppm  # Keep as numpy array if the input was an array
    else:
        return float(freq_ppm)  # Convert to float if the input was a scalar


def freq_ppm_to_Hz(
    freq_ppm: float = 0,
    hz_axis: np.ndarray = None,
    ppm_axis: np.ndarray = None,
    ppm_axis_flipped: bool = False,
    data_obj: Any = None,
):
    """
    Converts frequency values from parts per million (ppm) to Hertz (Hz) based on provided Hz and ppm axes.

    This function facilitates the conversion of frequency values from parts per million (ppm) to Hertz (Hz),
    which is commonly required in spectroscopy for chemical shift calculations. The conversion is achieved by
    matching frequency values in ppm to their corresponding values in Hz using provided frequency axes. The function
    can handle single values, lists, or arrays of frequencies to be converted.

    Parameters
    ----------
    freq_ppm : float or ndarray, optional
        Frequency value(s) in ppm to be converted. Can be a single value, list, or ndarray. Default is 0.
    hz_axis : ndarray, optional
        The frequency axis in Hz. Required if `data_obj` is not provided. Default is None.
    ppm_axis : ndarray, optional
        The frequency axis in ppm. Required if `data_obj` is not provided. Default is None.
    ppm_axis_flipped : bool, optional
        Set to True if the ppm axis is in descending order. This affects the conversion calculation.
        Default is False.
    data_obj : MRIExperiment, optional
        An object containing the Hz and ppm axes, used if either axis is not explicitly provided.
        Default is None.

    Returns
    -------
    ndarray or float
        The converted frequency values in Hz. The return type matches the type of `freq_ppm` input
        (i.e., if `freq_ppm` is a single value, a single float is returned; if it's an array, an ndarray is returned).

    Raises
    ------
    ValueError
        If neither `hz_axis` nor `ppm_axis` is provided and `data_obj` is None.

    Examples
    --------
    >>> ppm_axis = np.linspace(10, -10, 4000)
    >>> hz_axis = np.linspace(-2000, 2000, 4000)
    >>> freq_ppm_to_Hz(freq_ppm=2.5, hz_axis=hz_axis, ppm_axis=ppm_axis)
    500
    """
    if hz_axis is None or ppm_axis is None:
        if data_obj is not None:
            if hz_axis is None:
                hz_axis = get_freq_axis(data_obj=data_obj, unit="Hz")
            if ppm_axis is None:
                ppm_axis = get_freq_axis(data_obj=data_obj, unit="ppm")
        else:
            return None

    # check if passed ppm axis was flipped:
    if ppm_axis_flipped:
        ppm_axis = np.flip(ppm_axis)

    # ---------------------------------------------------------------
    # calculate a "conversion function"
    from scipy.stats import linregress

    f_ppm_to_hz = linregress(ppm_axis, hz_axis)

    # make iterable:
    if isinstance(freq_ppm, float) or isinstance(freq_ppm, int):
        freq_ppm = [freq_ppm]

    freq_hz = []
    # find index of freq_ppm in freq_ppm axis:
    for f in freq_ppm:
        if f is None:
            freq_hz.append(None)
        else:
            freq_hz.append(f_ppm_to_hz.intercept + f_ppm_to_hz.slope * f)

    if len(freq_hz) == 1:
        return freq_hz[0]
    else:
        return freq_hz


def center_freq_axis(freq_axis: np.array = None) -> np.array:
    """
    Center the inout freq_axis around 0.

    Parameters
    ----------
    freq_axis : np.array
        The frequency axis to be centered.

    Returns
    -------
    np.array
        The centered frequency axis.

    """
    # calculate translation term:
    freq_axis_centered_around_0 = freq_axis
    # from 0 to max:
    freq_axis_centered_around_0 = freq_axis_centered_around_0 - np.min(
        freq_axis_centered_around_0
    )
    # center around 0:
    freq_axis_centered_around_0 = (
        freq_axis_centered_around_0 - freq_axis_centered_around_0[-1] / 2.0
    )

    return freq_axis_centered_around_0


def make_NDspec_6Dspec(input_data: np.ndarray = None, provided_dims: tuple = None):
    """
    Transforms an N-dimensional (ND) array to a 6-dimensional array, with the option to reorder the dimensions.

    Parameters
    ----------
    input_data : ndarray
        The input array to transform. Can be of any shape with dimensions N where N <= 6.
    provided_dims : tuple of ints, optional
        Specifies the new order of the original dimensions in the resulting 6D array. The length of this tuple should
        match the number of dimensions in `input_data`. Each element in the tuple represents the index of the dimension
        in the output 6D array. If `None`, the original order is preserved in the first N dimensions, and the rest are
        filled with singleton dimensions.

    Returns
    -------
    output_data : ndarray
        The resulting 6D array with possibly reordered dimensions. If the input array had fewer than 6 dimensions, the
        additional dimensions are singleton (i.e., of size 1).

    Examples
    --------
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> make_NDspec_6Dspec(a).shape
    (2, 3, 1, 1, 1, 1)

    >>> make_NDspec_6Dspec(a).shape
    (2, 3, 1, 1, 1, 1)

    >>> make_NDspec_6Dspec(a, provided_dims=(1,0)).shape
    (3, 2, 1, 1, 1, 1)

    >>> make_NDspec_6Dspec(a, provided_dims=(5,0)).shape
    (3, 1, 1, 1, 1, 2)

    >>> aa = np.arange(720).reshape((1,2,3,4,5,6))
    >>> make_NDspec_6Dspec(aa, provided_dims=(1,2,3,4,5,0)).shape
    (6, 1, 2, 3, 4, 5)
    """
    if input_data is None:
        raise ValueError("Input data cannot be None")

    # Expand input_data to 6D if it has fewer dimensions, using np.newaxis to
    # add singleton dimensions
    while np.ndim(input_data) < 6:
        input_data = np.expand_dims(input_data, axis=-1)

    # If provided_dims is None, use the existing order, otherwise calculate
    # the new order
    if provided_dims is None:
        provided_dims = range(np.ndim(input_data))

    if any(isinstance(elem, str) for elem in provided_dims):
        from ...utils.image.dimensions import get_indices_for_strings

        provided_dims = get_indices_for_strings(strings=provided_dims)

    else:
        provided_dims = list(provided_dims)

    # Calculate the positions of missing dimensions if the provided_dims does
    # not cover all 6 dimensions
    missing_dims = [dim for dim in range(6) if dim not in provided_dims]

    # Combine provided_dims with missing_dims to cover all 6 dimensions
    all_dims = provided_dims + missing_dims
    new_order = [all_dims.index(i) for i in range(len(all_dims))]

    # Reorder dimensions according to all_dims
    output_data = np.transpose(input_data, axes=new_order)

    return output_data


def fit_single_spectrum_500MHz(
    ppm_axis,
    spectrum,
    peak_positions,
    plot=False,
    min_fwhm_ppm=0.01,
    max_fwhm_ppm=0.1,
    fit_func="lorentzian",
    SNR_cutoff=0,
):
    """
    Fits lorentzian to a spectrum at the desired peak locations
    Parameters
    ----------
    ppm_axis: numpy array
        The ppm-scale of the measurements to be examined.
    spectrum: numpy array
        N-dimensional spectrum to be fitted.
    peak_positions: list
        Contains the peak positions where fits should occur in ppm.
    SNR_cutoff: float
        SNR value below which peaks should not be fitted, default is 5.
    plot: bool
        Select if plotting for checking is needed
    norm_to_snr_before_fit: bool
        Select if you want to norm the spectra to a background region of the first repetition before fitting.
    bg_region_first_spec: list
        indices of the background region from which the snr is calculated.
    fit_func: str, default is 'lorentzian'
    can also be sqrt_lorentzian
    Returns
    -------
    peak_coeff: numpy array
        coefficients for a 3 parametric lorentzian fit model, shape: (NRepetitions, Number_of_peaks, 3)
        peak_coeff[0]: peak FWHM
        peak_coeff[1]: peak position (ppm)
        peak_coeff[2]: peak SNR

    peak_errors: numpy array
        errors for fit coefficients calculated from the covariance matrix, shape: (NRepetitions, Number_of_peaks, 3)
    """

    if fit_func == "lorentzian":
        fit_function = lorentzian
        fit_func_str = "Sqrt Lorentzian"
    elif fit_func == "sqrt_lorentzian":
        fit_function = sqrt_lorentzian
        fit_func_str = "Sqrt Lorentzian"
    else:
        logger.error(
            "Fit function %s not implemented - proceeding to fit using normal lorentzian",
            fit_func,
        )
        fit_function = lorentzian

    def find_range(axis, ppm):
        return np.argmin(np.abs(axis - ppm))

    # Norming to SNR
    # use first repetition where there should be no signal as reference region
    # for SNR

    # defining in with region we fit
    width = 2

    N_peaks = len(peak_positions)
    # defining region in which we will fit

    peak_coeff = np.zeros((N_peaks, 3))
    peak_covariance = np.zeros((N_peaks, 3, 3))
    peak_errors = np.zeros((N_peaks, 3))

    for peak_number, peak_center in enumerate(peak_positions):
        peak_roi = [
            find_range(ppm_axis, peak_center - width),
            find_range(ppm_axis, peak_center + width),
        ]

        try:
            (
                peak_coeff[peak_number],
                peak_covariance[peak_number],
            ) = curve_fit(
                fit_function,
                ppm_axis[peak_roi[0] : peak_roi[1]],
                spectrum[peak_roi[0] : peak_roi[1]],
                bounds=(
                    [
                        min_fwhm_ppm,
                        peak_center - width / 2.0,
                        np.min(spectrum),
                    ],
                    [max_fwhm_ppm, peak_center + width / 2.0, np.max(spectrum)],
                ),
            )
            peak_errors[peak_number] = np.sqrt(np.diag(peak_covariance[peak_number]))

        except RuntimeError:
            peak_coeff[peak_number] = None
            peak_covariance[peak_number] = None
            peak_errors[peak_number] = None

        # clean up badly fitted peaks
    for peak in range(peak_coeff.shape[0]):
        peak_snr = peak_coeff[peak][0]
        peak_errors[peak][0]
        if peak_snr > SNR_cutoff:
            # peak needs to have an SNR greater than a certain value
            pass
        else:
            peak_coeff[peak] = [0, 0, 0]

    if plot is True:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ppm_axis_itp = np.linspace(np.min(ppm_axis), np.max(ppm_axis), 1000000)
        # Plotting for QA
        combined_fits = 0
        combined_fits_itp = 0
        for peak_number, peak in enumerate(peak_positions):
            combined_fits += lorentzian(
                ppm_axis,
                peak_coeff[peak_number][0],
                peak_coeff[peak_number][1],
                peak_coeff[peak_number][2],
            )
            combined_fits_itp += lorentzian(
                ppm_axis_itp,
                peak_coeff[peak_number][0],
                peak_coeff[peak_number][1],
                peak_coeff[peak_number][2],
            )
        ax[0].plot(
            ppm_axis,
            spectrum,
            linewidth="0.5",
            color="r",
            label="Data",
        )
        ax[0].plot(
            ppm_axis_itp,
            combined_fits_itp,
            linestyle="dashed",
            color="k",
            label=str(fit_func_str) + "fit",
        )

        for peak_number, peak in enumerate(peak_positions):
            print(str(peak), peak_coeff[peak_number])
        ax[0].set_xlim([190, 160])
        ax[0].set_ylim([np.min(spectrum), np.max(spectrum)])
        ax[0].set_ylabel("SNR")
        ax[0].set_xlabel(r"$\sigma$[ppm]")

        ax[0].legend()

        # Plotting for QA

        for peak_number, peak in enumerate(peak_positions):
            ax[1].plot(
                ppm_axis,
                lorentzian(
                    ppm_axis,
                    peak_coeff[peak_number][0],
                    peak_coeff[peak_number][1],
                    peak_coeff[peak_number][2],
                )
                - 20 * peak_number,
                color="C" + str(peak_number),
                label=str(np.round(peak, 1)) + " ppm",
            )
        ax[1].set_ylabel("I [a.u.]")
        ax[1].set_yticks([])
        ax[1].set_xlabel(r"$\sigma$[ppm]")

        ax[1].legend()
        ax[1].set_xlim([190, 160])

    return peak_coeff, peak_errors


class squareLorentzian(Fittable1DModel):
    """
    One dimensional sqrt Lorentzian model. The denominator is the sqrt of the original term.

    Parameters
    ----------
    amplitude : float or `~astropy.unit.Quantity`.
        Peak value - for a normalized profile (integrating to 1),
        set amplitude = 2 / (np.pi * fwhm)
    x_0 : float or `~astropy.unit.Quantity`.
        Position of the peak
    fwhm : float or `~astropy.unit.Quantity`.
        Full width at half maximum (FWHM)

    See Also
    --------
    Gaussian1D, Box1D, RickerWavelet1D

    Notes
    -----
    Either all or none of input ``x``, position ``x_0`` and ``fwhm`` must be provided
    consistently with compatible unit or as unitless numbers.

    Model formula:

    .. math::

        f(x) = \\frac{A \\gamma^{2}}{\\gamma^{2} + \\left(x - x_{0}\\right)^{2}}

    where :math:`\\gamma` is half of given FWHM.

    """

    from astropy.modeling import Parameter

    amplitude = Parameter(default=1, description="Peak value")
    x_0 = Parameter(default=0, description="Position of the peak")
    fwhm = Parameter(default=1, description="Full width at half maximum")

    @staticmethod
    def evaluate(x, amplitude, x_0, fwhm):
        """One dimensional Lorentzian model function"""

        # return amplitude * ((fwhm / 2.0) ** 2) / ((x - x_0) ** 2 + (fwhm /
        # 2.0) ** 2)
        return amplitude / np.sqrt(3 * ((x - x_0) / (fwhm / 2.0)) ** 2 + 1)

    @staticmethod
    def fit_deriv(x, amplitude, x_0, fwhm):
        """One dimensional Lorentzian model derivative with respect to parameters"""

        # d_amplitude = fwhm**2 / (4 * np.sqrt(fwhm**2/4 + (x - x_0)**2))
        # d_x_0       = (amplitude * fwhm**2 * (x - x_0)) / (4 * (fwhm**2/4 + (x - x_0)**2)**(3/2))

        # d_fwhm      = (-amplitude * fwhm**3 / (16 * (fwhm**2/4 + (x - x_0)**2)**(3/2))) + (amplitude * fwhm / (2 * np.sqrt(fwhm**2/4 + (x - x_0)**2)))

        d_amplitude = 1 / np.sqrt(3 * ((x - x_0) / (fwhm / 2.0)) ** 2 + 1)
        d_x_0 = (
            (-1 / 2)
            * amplitude
            / (3 * ((x - x_0) / (fwhm / 2.0)) ** 2 + 1) ** (3 / 2)
            * 6
            * (x - x_0)
            / (fwhm / 2.0) ** 2
        )

        d_fwhm = (
            (-1 / 2)
            * amplitude
            / (3 * ((x - x_0) / (fwhm / 2.0)) ** 2 + 1) ** (3 / 2)
            * (-6)
            * (x - x_0) ** 2
            / (fwhm / 2.0) ** 3
            * (1 / 2)
        )

        return [d_amplitude, d_x_0, d_fwhm]

    def bounding_box(self, factor=25):
        """Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high)``.

        Parameters
        ----------
        factor : float
            The multiple of FWHM used to define the limits.
            Default is chosen to include most (99%) of the
            area under the curve, while still showing the
            central feature of interest.

        """
        x0 = self.x_0
        dx = factor * self.fwhm

        return (x0 - dx, x0 + dx)

    @property
    def input_units(self):
        """Units of the independent variable expected by the model."""
        if self.x_0.unit is None:
            return None
        return {self.inputs[0]: self.x_0.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        """Map quantity units for parameters based on input/output data units."""
        return {
            "x_0": inputs_unit[self.inputs[0]],
            "fwhm": inputs_unit[self.inputs[0]],
            "amplitude": outputs_unit[self.outputs[0]],
        }
