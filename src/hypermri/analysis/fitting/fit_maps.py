"""Voxel-wise fitting helpers for parametric MRI maps.

The module provides light-weight helper functions used by the ``hp`` xarray
accessor to run non-linear fits across volumes. Each helper accepts NumPy
arrays and returns dictionaries so that the accessor can easily assemble
``xarray`` datasets with parameter estimates.
"""

import warnings
from itertools import product

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, least_squares
from tqdm.auto import tqdm

from hypermri.utils.system.logging import debug_log, init_default_logger

DB_MSG = True  # Change this to False to hide detailed error messages
DATA_PRECISION = np.float16


DEBUG_FILE = "/tmp/fit_maps_debug.log"
DEBUG_LOG = True


def debug_log_fit_maps(
    message="", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG, numpy_full_print=True
):
    """
    Debug logging system for quantitative fitting functions.

    Parameters
    ----------
    message : str, optional
        The debug message to log. Default is empty string.
    file_name : str, optional
        Path to the log file. Default is DEBUG_FILE constant.
    write_to_file : bool, optional
        Whether to write the log to a file. Default is DEBUG_LOG constant.
    numpy_full_print : bool, optional
        Whether to print full numpy arrays without truncation. Default is True.

    Returns
    -------
    None
    """
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
        numpy_full_print=numpy_full_print,
    )


logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


# --- Helper function for two-compartment model fitting ---
def _fit_single_voxel_2cm_helper(
    signal: np.ndarray = None,
    aif_signal: np.ndarray = None,
    time_points_s: np.ndarray = None,
    fit_start_time_s: np.ndarray = None,
    fit_stop_time_s: np.ndarray = None,
    fit_params: dict = {},
):
    """Fit the two-compartment model for a single voxel.

    Parameters
    ----------
    signal
        Tissue concentration time-course for the voxel under investigation.
    aif_signal
        Arterial input function sampled on the same time grid as ``signal``.
    time_points_s
        Time axis in seconds used to sample ``signal`` and ``aif_signal``.
    fit_start_time_s, fit_stop_time_s
        Optional temporal window that bounds the fitting process. ``None``
        defaults to the full time course.
    fit_params
        Dictionary-based override that can supply any of the above values at
        runtime.

    Returns
    -------
    dict
        Dictionary containing the estimated flow ``Fp`` and volume fractions
        ``ve`` and ``vb``.
    """
    aif_signal = fit_params.get("aif_signal", aif_signal)
    time_points_s = fit_params.get("time_points_s", time_points_s)
    fit_start_time_s = fit_params.get("fit_start_time_s", 0.0)
    fit_stop_time_s = fit_params.get("fit_stop_time_s", np.inf)
    signal = fit_params.get("signal", signal)

    signal = np.asarray(signal).squeeze()
    aif_signal = np.asarray(aif_signal).squeeze()
    time_points_s = np.asarray(time_points_s).squeeze()

    # --- Truncate the data to the desired fitting range ---
    start_index = np.argmax(time_points_s >= fit_start_time_s)
    if np.isinf(fit_stop_time_s) or fit_stop_time_s > time_points_s[-1]:
        stop_index = len(time_points_s)
    else:
        # Use searchsorted to get the index where the value would be inserted
        stop_index = np.searchsorted(time_points_s, fit_stop_time_s)

    if start_index >= stop_index:
        return {"Fp": np.nan, "ve": np.nan, "vb": np.nan}

    # Create truncated versions of the arrays for fitting
    time_points_fit = time_points_s[start_index:stop_index]
    tissue_signal_fit = signal[start_index:stop_index]
    # The AIF for the model must be the full AIF to correctly calculate the
    # convolution
    aif_fit_model = aif_signal[start_index:stop_index]

    time_step = (
        time_points_fit[1] - time_points_fit[0]
        if len(time_points_fit) > 1
        else time_points_s[1] - time_points_s[0]
    )

    # The model needs to be evaluated over the truncated time course,
    # but the convolution should still be handled carefully.
    def model_to_fit(t, Fp, ve, vb):
        return two_compartment_model_deprecated(t, Fp, ve, vb, aif_fit_model, time_step)

    # --- Set Initial Guesses and Bounds ---
    initial_guesses = [0.2, 0.2, 0.1]
    lower_bounds = [0, 0, 0]
    upper_bounds = [5.0, 1.0, 1.0]

    try:
        popt, _ = curve_fit(
            model_to_fit,
            time_points_fit,  # Fit over the truncated time
            tissue_signal_fit,  # Fit to the truncated signal
            p0=initial_guesses,
            bounds=(lower_bounds, upper_bounds),
        )
        fp_fit, ve_fit, vb_fit = popt
    except (RuntimeError, ValueError):
        fp_fit, ve_fit, vb_fit = np.nan, np.nan, np.nan

    return {"Fp": fp_fit, "ve": ve_fit, "vb": vb_fit}


# --- Helper function for Tofts fitting ---
def _fit_single_voxel_tofts_helper(
    t,
    c_t_measured,
    cp_func,
    model_type="1comp",
    x0_1comp=None,
    bounds_1comp=None,
    x0_2comp=None,
    bounds_2comp=None,
    fit_mask=None,
):
    """
    Fit Tofts (1-comp) or Extended Tofts (2-comp) model to a single voxel.
    """
    if x0_1comp is None:
        x0_1comp = [0.01, 0.1]
    if bounds_1comp is None:
        bounds_1comp = ([0, 0], [5, 5])
    if x0_2comp is None:
        x0_2comp = [0.01, 0.1, 0.01]
    if bounds_2comp is None:
        bounds_2comp = ([0, 0, 0], [5, 5, 1])

    # Ensure inputs are float64 to avoid casting errors in least_squares
    t = np.asarray(t, dtype=np.float64)
    c_t_measured = np.asarray(c_t_measured, dtype=np.float64)

    if np.all(c_t_measured <= 0):
        return (0, 0, 0) if model_type == "2comp" else (0, 0)

    # Helper to apply mask if present
    def residual_func(model_vals):
        res = model_vals - c_t_measured
        if fit_mask is not None:
            return res[fit_mask]
        return res

    if model_type == "1comp":
        res = least_squares(
            lambda p: residual_func(tofts_model_1comp(t, p[0], p[1], cp_func)),
            x0=np.array(x0_1comp, dtype=np.float64),
            bounds=bounds_1comp,
            ftol=1e-3,
        )
        return res.x
    elif model_type == "2comp":
        res = least_squares(
            lambda p: residual_func(
                extended_tofts_model_2comp(t, p[0], p[1], p[2], cp_func)
            ),
            x0=np.array(x0_2comp, dtype=np.float64),
            bounds=bounds_2comp,
            ftol=1e-3,
        )
        return res.x
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# --- Helper function for Patlak fitting ---
def _fit_single_voxel_patlak_helper(
    signal: np.ndarray = None,
    aif_signal: np.ndarray = None,
    time_points_s: np.ndarray = None,
    fit_params: dict = {},
):
    """
    Helper function to fit the Patlak model using linear regression.

    """
    # --- Unpack all possible filtering parameters from the dictionary ---
    from scipy.integrate import cumulative_trapezoid

    signal = fit_params.get("signal", signal)
    aif_signal = fit_params.get("aif_signal", aif_signal)
    time_points_s = fit_params.get("time_points_s", time_points_s)
    aif_threshold = fit_params.get("aif_threshold", 1e-6)

    # --- Validate time_points_s ---
    if time_points_s is None:
        raise ValueError(
            "time_points_s is required for Patlak fitting but was not provided. "
            "Please provide time values in seconds via fit_params['time_points_s']."
        )

    # Time-based fit range (defaults to include all times)
    fit_start_time_s = fit_params.get("fit_start_time_s", -np.inf)
    fit_stop_time_s = fit_params.get("fit_stop_time_s", np.inf)

    # Index-based fit range (defaults to include all indices)
    fit_start_index = fit_params.get("fit_start_index", 0)
    fit_stop_index = fit_params.get("fit_stop_index", None)

    # --- Prepare data arrays ---
    signal = np.asarray(signal).squeeze()
    aif_signal = np.asarray(aif_signal).squeeze()
    time_points_s = np.asarray(time_points_s).squeeze()
    num_points = len(time_points_s)

    # Check if time_points_s appears to be indices rather than actual time values
    # This is a heuristic: if values are sequential integers starting near 0, warn the user
    if len(time_points_s) > 2:
        if np.allclose(time_points_s, np.arange(len(time_points_s)), rtol=0.01):
            raise ValueError(
                "time_points_s appears to contain indices (0, 1, 2, ...) rather than actual "
                "time values in seconds. The Patlak model requires time values in seconds. "
                "If your coordinate contains indices, multiply by the repetition time (TR) or "
                "provide the actual acquisition times in seconds."
            )

    # --- Build a single boolean mask from all filter conditions ---
    valid_mask = np.ones(num_points, dtype=bool)
    all_indices = np.arange(num_points)

    valid_mask &= time_points_s >= fit_start_time_s
    valid_mask &= time_points_s <= fit_stop_time_s
    valid_mask &= all_indices >= fit_start_index
    if fit_stop_index is not None:
        # The stop index is exclusive, matching Python's slicing behavior
        valid_mask &= all_indices < fit_stop_index
    valid_mask &= aif_signal > aif_threshold

    # --- Check for sufficient data points after filtering ---
    if np.sum(valid_mask) < 2:  # Need at least two points to fit a line
        return {"K_trans": np.nan, "Vp": np.nan, "rss_patlak": np.nan}

    # --- Apply the mask to select data for the fit ---
    aif_filt = aif_signal[valid_mask]
    signal_filt = signal[valid_mask]

    # --- Calculate Patlak variables (X and Y) ---
    # Note: The integral must be calculated over the *full* AIF
    integral_aif = cumulative_trapezoid(aif_signal, time_points_s, initial=0)
    # Then select the valid points for the regression
    integral_aif_filt = integral_aif[valid_mask]

    patlak_x = integral_aif_filt / aif_filt
    patlak_y = signal_filt / aif_filt

    # --- Perform Linear Regression: Y = Vp * 1 + K_trans * X ---
    M = np.vstack([np.ones_like(patlak_x, dtype=float), patlak_x.astype(float)]).T

    try:
        coeffs, _, rank, _ = np.linalg.lstsq(M, patlak_y, rcond=None)

        if rank < 2:  # Underdetermined system
            return {"K_trans": np.nan, "Vp": np.nan, "rss_patlak": np.nan}

        vp_val = coeffs[0]
        ki_val = coeffs[1]

        predicted_patlak_y = M @ coeffs
        rss = np.sum((patlak_y - predicted_patlak_y) ** 2)

        if ki_val < 0:  # Clamp non-physical results to zero
            ki_val = 0.0

        return {"K_trans": ki_val, "Vp": vp_val, "rss_patlak": rss}

    except np.linalg.LinAlgError:
        return {"K_trans": np.nan, "Vp": np.nan, "rss_patlak": np.nan}


def _fit_single_voxel_b1_period_sweep_helper(
    signal: np.ndarray,
    sqrt_refpower: np.ndarray,
    fit_params: dict,
) -> dict:
    """
    Fit the sinusoidal B1 model S = A * sin(pi / period * sqrt(P)) by sweeping candidate periods.

    Parameters
    ----------
    signal
        Complex or real-valued voxel measurements for a single location.
    sqrt_refpower
        Square-root RF power samples aligned with ``signal``.
    fit_params
        Dictionary containing the precomputed period grid and thresholds.

    Returns
    -------
    dict
        Best-fit parameters and diagnostics for the voxel.
        "amplitude", "period", "refpower_90deg_W", "rss_b1", "stop_index", "n_valid_samples"
    """

    period_grid = fit_params["_b1_period_grid"]
    min_valid_samples = int(fit_params.get("min_valid_samples", 3))

    # Check if neighbor period is available to restrict search range
    neighbor_period = fit_params.get("period", None)
    neighbor_period_search_width = float(
        fit_params.get("neighbor_period_search_width", 0.3)
    )

    if neighbor_period is not None and np.isfinite(neighbor_period):
        # Generate restricted period grid centered around neighbor period
        period_samples = int(fit_params.get("period_samples", 1000))
        period_min_local = neighbor_period * (1.0 - neighbor_period_search_width)
        period_max_local = neighbor_period * (1.0 + neighbor_period_search_width)

        # Respect global min/max constraints if they exist
        global_period_min = fit_params.get("period_min", None)
        global_period_max = fit_params.get("period_max", None)
        if global_period_min is not None:
            period_min_local = max(period_min_local, global_period_min)
        if global_period_max is not None:
            period_max_local = min(period_max_local, global_period_max)

        # Generate restricted grid
        period_grid = np.linspace(period_min_local, period_max_local, period_samples)
    else:
        debug_log_fit_maps("Using full period grid (no neighbor information)")

    signal = np.asarray(signal, dtype=complex).squeeze()
    sqrt_refpower = np.asarray(sqrt_refpower, dtype=float).squeeze()

    finite_mask = np.isfinite(signal) & np.isfinite(sqrt_refpower)
    signal_valid = signal[finite_mask]
    sqrt_refpower_valid = sqrt_refpower[finite_mask]
    n_valid = signal_valid.size

    stop_index = int(fit_params.get("stop_index", n_valid))
    min_stop_index = int(fit_params.get("min_stop_index", 2))
    if stop_index < min_stop_index:
        stop_index = min_stop_index

    signal_valid = signal_valid[: stop_index + 1]
    sqrt_refpower_valid = sqrt_refpower_valid[: stop_index + 1]

    if n_valid < min_valid_samples or period_grid.size == 0:
        debug_log_fit_maps("Insufficient valid samples or empty period grid.")
        return {
            "amplitude": np.nan,
            "period": np.nan,
            "refpower_90deg_W": np.nan,
            "rss_b1": np.nan,
            "stop_index": np.nan,
            "n_valid_samples": float(n_valid),
        }

    debug_log_fit_maps(f"sqrt_refpower_valid: {sqrt_refpower_valid}")

    # Generate a matrix containing sin(pi / period * sqrt(P)) for all periods
    sin_matrix = np.sin(
        np.pi / period_grid[np.newaxis, :] * sqrt_refpower_valid[:, np.newaxis]
    )

    # Solve for amplitudes using least squares for each candidate period
    numerator = np.sum(np.conjugate(sin_matrix) * signal_valid[:, np.newaxis], axis=0)
    denom = np.sum(np.abs(sin_matrix) ** 2, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        amps = np.divide(
            numerator,
            denom,
            out=np.zeros_like(numerator, dtype=complex),
            where=denom > 0,
        )

    # Compute residuals for each candidate period
    residuals = np.sum(
        np.abs(signal_valid[:, np.newaxis] - sin_matrix * amps[np.newaxis, :]) ** 2,
        axis=0,
    )
    import matplotlib.pyplot as plt
    from IPython.display import display

    valid_period_mask = denom > 0
    if not np.any(valid_period_mask):
        return {
            "amplitude": np.nan,
            "period": np.nan,
            "refpower_90deg_W": np.nan,
            "rss_b1": np.nan,
            "stop_index": np.nan,
            "n_valid_samples": float(n_valid),
        }

    masked_residuals = np.where(valid_period_mask, residuals, np.inf)
    best_idx = int(np.argmin(masked_residuals))

    best_period = float(period_grid[best_idx])
    best_amp = amps[best_idx]
    best_residual = float(residuals[best_idx])

    if neighbor_period is not None and np.isfinite(neighbor_period):
        debug_log_fit_maps(
            f"Using neighbor-informed period search: "
            f"center={neighbor_period:.3f}, range=[{period_min_local:.3f}, {period_max_local:.3f}], best_period={best_period:.3f}",
        )

    # Reference power for 90° flip: sin argument = π/2 => sqrt(P) = period / 2
    # debug_log_fit_maps(f"1/ refpower_90deg (pre-adjustment) = {refpower_90deg:.3f} W")
    from hypermri.analysis.anatomical.b1_mapping import (
        refpow_from_sin_fit,
        refpow_parameter_adjustment,
    )

    refpower_90deg = refpow_from_sin_fit(periods=best_period)
    debug_log_fit_maps(
        f"1/ refpower_90deg (pre-adjustment) = {refpower_90deg:.3f} W",
        write_to_file=False,
    )
    refpower_90deg = refpow_parameter_adjustment(
        refpower_W=refpower_90deg,
        trf_ms=fit_params.get("trf_ms", 1),
        sint=fit_params.get("sint", 1),
        fa_deg=fit_params.get("fa_deg", 90),
    )
    debug_log_fit_maps(
        f"2/ refpower_90deg (pre-adjustment) = {refpower_90deg:.3f} W",
        write_to_file=False,
    )
    plot = False
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(period_grid, residuals)
        ax[0].axvline(best_period, color="r", linestyle="--", label="Best Period")
        ax[0].set_xlabel("Period")
        ax[0].set_ylabel("Sum of Squared Residuals")
        ax[0].set_title("B1 Period Sweep Fit for Single Voxel")
        ax[1].plot(
            sqrt_refpower_valid, signal_valid.real, "o", label="Data (Real Part)"
        )
        ax[1].plot(
            sqrt_refpower_valid,
            (best_amp * np.sin(np.pi / best_period * sqrt_refpower_valid)).real,
            "-",
            label="Fitted Model (Real Part)",
        )
        ax[1].axvline(best_period / 2.0, color="g", linestyle="--", label="90° Power")
        ax[1].axvline(
            sqrt_refpower_valid[stop_index - 1],
            color="orange",
            linestyle="--",
            label="Stop Index",
        )
        ax[1].legend()
        display(fig)
        plt.pause(1)
        plt.close(fig)

    if np.any(finite_mask):
        finite_indices = np.nonzero(finite_mask)[0]
        stop_index = float(finite_indices.max())
        float(finite_indices.min())
    else:
        stop_index = np.nan

    return {
        "amplitude": best_amp,
        "period": best_period,
        "refpower_90deg_W": refpower_90deg,
        "rss_b1": best_residual,
        "stop_index": stop_index,
        "n_valid_samples": float(n_valid),
    }


# --- Helper function for T1 fitting ---
def _fit_t1_inversion_recovery_helper(
    signal: np.ndarray = None,
    inversion_times_ms: np.ndarray = None,
    t1_values_ms: np.ndarray = None,
    fit_baseline: bool = False,
    fit_params: dict = {},
):
    """
    Helper function to find the best T1 by minimizing the sum of squared residuals.
    Fits either S(t) = A*(1-2*exp(-t/T1)) + B (if fit_baseline=True) or
    S(t) = A*(1-2*exp(-t/T1)) (if fit_baseline=False).
    A and B can be complex if the input signal is complex.

    Args:
        inversion_times_ms (np.ndarray): 1D array of inversion times in milliseconds.
        signal (np.ndarray): 1D complex or real signal array for a single voxel.
        t1_values_to_test (np.ndarray): 1D array of T1 values (in ms) to test.
        fit_baseline (bool): If True, fit the baseline parameter B. If False, assume B=0.

    Returns
        dict: Dictionary containing the best fit parameters {'t1': ..., 'A': ..., 'B': ..., 'rss_t1': ...}.
            Returns None if no valid T1 could be found or fitting fails.
            Note: B will be 0.0 if fit_baseline is False. A and B will be complex
            if the input signal is complex.

    Info
    ----
        The bounds can be calculated as follows for the baseline model S(t) = A(1-2*exp(-t/T1)) + B
        S(t) = (M0 + M0') / 2 * (1 - 2 * exp(-t/T1)) + (M0 - M0') / 2
        where M0' is the initial Magnetization (S(t=0) = -M0') after inversion pulse. The inversion pulse can be imperfect, so let's assume
        * 0.9 M0 <= M0' <= M0
        Depending on the T1, we might not sample the fully recovered signal M0 with our longet inversion time timax
        So it follwos (assuming perfect inversion for now):
        * Smax = S(timax) = M0 * (1 - 2 * exp(-timax / T1))
        * [e.g. T1 = 3500ms, timax = 400ms]:
        * S(4000ms) = M0 * (1 - 2 * exp(4000/3500)) = 0.36 * M0
        ->  0.36 M0 <= Smax <= M0
        Putting all together we end at:
        * 0.95*M0 <= A <= M0
        * 0 <= B <= 0.05 * M0
        ----
        --> 0.95*Smax <= A <= (1/0.36)=2.7 Smax
        --> 0 <= B <= (0.05 * 2.7) 0.135 Smax

    """
    inversion_times_ms = fit_params.get("inversion_times_ms", inversion_times_ms)
    t1_values_ms = fit_params.get("t1_values_ms", t1_values_ms)
    fit_baseline = fit_params.get("fit_baseline", fit_baseline)
    signal = fit_params.get("signal", signal)
    # Ensure signal is numpy array and squeezed for max calculation
    signal = np.asarray(signal).squeeze()
    s_ref = np.max(np.abs(signal)) if signal.size > 0 else 1.0

    # For L2 regularization
    fit_params.get("regularization_alpha", 0.0)

    # For hard bounds (B, A)
    # Default is no bounds (-inf, inf)
    bounds_B = fit_params.get("bounds_B", (-np.inf, np.inf))
    bounds_A = fit_params.get("bounds_A", (-np.inf, np.inf))

    # Support for relative bounds (percentages)
    rel_bounds_B = fit_params.get("rel_bounds_B", None)
    rel_bounds_A = fit_params.get("rel_bounds_A", None)

    # Automatically calculate the relative bounds (percentages):
    auto_calc_rel_bound = fit_params.get("auto_calc_rel_bound", False)
    if auto_calc_rel_bound:
        # Maximum inversion time [ms]:
        timax = np.max(inversion_times_ms)
        # Maximum assumed T1 [ms]:
        t1max = np.max(t1_values_ms)
        # Maximum recovered signal if t1max:
        max_recov_sig = np.abs(1 - 2 * np.exp(-timax / t1max))  # [0-1]
        # inversion imperfection [0-1], 0 perfect inversion, 1 90 degree instead of 180:
        inversion_imperfection = 0.1
        # limits for A:
        lb_A = 1 - inversion_imperfection / 2
        ub_A = 1 / max(max_recov_sig, 0.1)  # avoid division by 0
        rel_bounds_A = [lb_A, ub_A]
        # limit for B:
        lb_B = 0
        ub_B = inversion_imperfection / 2 * (1 / max(max_recov_sig, 0.1))
        rel_bounds_B = [lb_B, ub_B]
        # print(f"Limits A: {lb_A} - {ub_A}\nLimits B: {lb_B} - {ub_B}")

    if rel_bounds_B is not None:
        if isinstance(rel_bounds_B, (int, float)):
            # Assume symmetric range around 0 for baseline
            bounds_B = (-rel_bounds_B * s_ref, rel_bounds_B * s_ref)
        else:
            # Scale tuple directly
            bounds_B = (rel_bounds_B[0] * s_ref, rel_bounds_B[1] * s_ref)

    if rel_bounds_A is not None:
        if isinstance(rel_bounds_A, (int, float)):
            # Assume range around 100% for amplitude
            bounds_A = ((1.0 - rel_bounds_A) * s_ref, (1.0 + rel_bounds_A) * s_ref)
        else:
            # Scale tuple directly
            bounds_A = (rel_bounds_A[0] * s_ref, rel_bounds_A[1] * s_ref)

    # Check if any bounds are finite
    use_lsq_linear = any(np.isfinite(bounds_B)) or any(np.isfinite(bounds_A))
    print(
        f"DEBUG: use_lsq_linear={use_lsq_linear}, bounds_A={bounds_A}, bounds_B={bounds_B}"
    )

    t1_values_ms = np.asarray(t1_values_ms).squeeze()
    inversion_times_ms = np.asarray(inversion_times_ms).squeeze()

    rss_values = []
    coeffs_for_t1_values = []

    # Determine the appropriate NaN value based on signal type
    if np.iscomplexobj(signal):
        nan_param_val = np.nan + 0j * np.nan
    else:
        nan_param_val = np.nan

    for t1_guess in t1_values_ms:
        if t1_guess <= 0:
            rss_values.append(np.inf)
            coeffs_for_t1_values.append(
                [nan_param_val, nan_param_val] if fit_baseline else [nan_param_val]
            )
            continue

        # Model basis is real
        exp_term = np.exp(-inversion_times_ms / float(t1_guess))

        if fit_baseline:
            # M columns: [ones, exp(-t/T1)]
            M = np.vstack(
                [
                    np.ones_like(inversion_times_ms, dtype=float),
                    # Basis for B'
                    exp_term,  # Basis for A'
                ]
            ).T
        else:
            # M column: [1 - 2*exp(-t/T1)]
            model_basis_no_baseline = 1 - 2 * exp_term
            M = model_basis_no_baseline.reshape(-1, 1)

        try:
            # np.linalg.lstsq can handle complex 'signal' (b in Ax=b)
            # and will produce complex 'coeffs' (x) if 'signal' is complex,
            # even if 'M' (A in Ax=b) is real.
            coeffs, residuals, rank, s = np.linalg.lstsq(M, signal, rcond=None)

            predicted_signals = M @ coeffs
            # RSS: sum of squared magnitudes of residuals
            rss = np.sum(np.abs(signal - predicted_signals) ** 2)

            rss_values.append(rss)
            coeffs_for_t1_values.append(coeffs)

        except np.linalg.LinAlgError:
            rss_values.append(np.inf)
            coeffs_for_t1_values.append(
                [nan_param_val, nan_param_val] if fit_baseline else [nan_param_val]
            )

    if not rss_values or np.all(
        np.isinf(rss_values) | np.isnan(rss_values)
    ):  # check for nan RSS too
        return {
            "t1": np.nan,
            "A": nan_param_val,
            "B": nan_param_val,
            "rss_t1": np.nan,
        }

    min_rss_index = np.nanargmin(
        rss_values
    )  # Use nanargmin to ignore NaNs in rss_values
    best_t1 = t1_values_ms[min_rss_index]
    best_coeffs = coeffs_for_t1_values[min_rss_index]
    min_rss = rss_values[min_rss_index]

    # Initialize A and B with appropriate NaN type
    best_A = nan_param_val
    best_B = nan_param_val

    if fit_baseline:
        # S(t) = C0*1 + C1*exp(-t/T1)
        # Model: S(t) = B + A*(1 - 2*exp(-t/T1)) = (B+A) - 2*A*exp(-t/T1)
        # C0 = B+A
        # C1 = -2A  => A = -C1/2
        # B = C0 - A = C0 + C1/2
        if len(best_coeffs) == 2 and not np.any(np.isnan(best_coeffs)):
            C0 = best_coeffs[0]  # This is B' in S(t) = B' + A' * exp(-t/T1)
            C1 = best_coeffs[1]  # This is A'
            best_A = -C1 / 2.0
            best_B = C0 + C1 / 2.0
        # If coeffs are nan, A and B remain nan_param_val
    else:
        # S(t) = C0 * (1 - 2*exp(-t/T1))
        # Model: S(t) = A * (1 - 2*exp(-t/T1))
        # C0 = A
        if len(best_coeffs) == 1 and not np.any(np.isnan(best_coeffs)):
            C0 = best_coeffs[0]
            best_A = C0
            best_B = 0.0  # B is fixed to 0 (real zero)
        # If coeff is nan, A remains nan_param_val, B should also be nan_param_val (or 0.0 if A is also nan?)
        # If A is nan, B being 0 is by definition of fit_baseline=False,
        # but if the fit failed to determine A, B should ideally also reflect that uncertainty or fixed nature.
        # Current logic: if A is nan_param_val, B also remains nan_param_val unless set to 0.0.
        # Let's ensure B is nan if A is nan when fit_baseline=False and fit
        # failed.
        if np.any(np.isnan(best_coeffs)):  # if A could not be determined
            best_B = nan_param_val  # ensure B is also nan

    return {"t1": best_t1, "A": best_A, "B": best_B, "rss_t1": min_rss}


# --- Helper function for ADC fitting ---
def _fit_adc_monoexponential_helper(
    signal: np.ndarray = None,
    b_values: np.ndarray = None,
    signal_threshold: float = 1e-5,
    fit_params: dict = {},
):
    """
    Fit ADC using a mono-exponential model via log-linear regression.
    Fits ln(S(b)) = ln(S0) - b * ADC.

    Parameters
    ----------
    b_values: np.ndarray
        1D array of b-values (e.g., in s/mm^2).
    signal: np.ndarray
        1D real, positive signal array for a single voxel. Typically magnitude data.
    signal_threshold: float
        Minimum signal value to include in the fit. Points below this (after taking abs) are excluded.

    Returns
    -------
        best fit parameters: dict
        Dictionary containing the best fit parameters {'adc': ..., 's0': ..., 'rss_adc': ...}.
        Returns NaNs if fitting fails or insufficient data.

    """
    b_values = fit_params.get("b_values", b_values)
    signal_threshold = fit_params.get("signal_threshold", signal_threshold)
    signal = fit_params.get("signal", signal)

    # Ensure signal is magnitude and positive for log
    magnitude_signal = np.abs(signal)

    # Filter out b-values where signal is too low or b-value is not positive (optional for b_values > 0)
    # For ADC, typically b-values are >= 0.
    valid_indices = (magnitude_signal > signal_threshold) & (b_values >= 0)

    if np.sum(valid_indices) < 2:  # Need at least two points to fit a line
        return {"adc": np.nan, "s0": np.nan, "rss_adc": np.nan}

    b_values_filt = b_values[valid_indices]
    log_signal_filt = np.log(magnitude_signal[valid_indices])

    # Design matrix M for y = c0*x0 + c1*x1 where x0=1, x1=-b
    # We want to solve for [ln(S0), ADC]
    # So, M = [ [1, -b1], [1, -b2], ... ]
    M = np.vstack(
        [np.ones_like(b_values_filt, dtype=float), -b_values_filt.astype(float)]
    ).T

    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(M, log_signal_filt, rcond=None)

        if rank < 2:  # Underdetermined system
            return {"adc": np.nan, "s0": np.nan, "rss_adc": np.nan}

        log_S0 = coeffs[0]
        adc_val = coeffs[1]
        s0_val = np.exp(log_S0)

        # Calculate sum of squared residuals on the log-transformed signal
        predicted_log_signals = M @ coeffs
        rss = np.sum((log_signal_filt - predicted_log_signals) ** 2)

        # Ensure ADC is not negative (physical constraint, though fit might
        # yield it)
        if adc_val < 0:  # Or handle as needed, e.g. clamp to 0 or mark as bad fit
            adc_val = np.nan  # Or 0.0, depending on preference

        return {"adc": adc_val, "s0": s0_val, "rss_adc": rss}

    except np.linalg.LinAlgError:
        return {"adc": np.nan, "s0": np.nan, "rss_adc": np.nan}
    except (
        ValueError
    ):  # Catches log of zero/negative if not filtered enough, though threshold should help
        return {"adc": np.nan, "s0": np.nan, "rss_adc": np.nan}


# --- Generalized single voxel fitting function ---
def _fit_single_voxel_quantitative(
    indices: tuple,
    input_data: xr.DataArray,
    non_dynamic_dims: list,
    dynamic_dim_name: str,  # e.g., "inversion_idx" or "bvalue_idx"
    dim_to_index_pos: dict,
    fit_type: str,  # e.g. "t1" or "adc"
    fit_params: dict = {},
):
    """
    Fits the specified model (T1, ADC, Perfusion, B1) for a single voxel. This function
    is usually called by fit_quantitative_maps() in a parallelized manner.

    Parameters
    ----------
    indices : tuple
        Tuple of indices corresponding to the non-dynamic dimensions of the input_data.
    input_data : xr.DataArray
        The full input data array containing the dynamic dimension.
    non_dynamic_dims : list
        List of dimension names that are not dynamic (e.g., spatial dimensions).
    dynamic_dim_name : str
        Name of the dynamic dimension (e.g., "inversion_idx" or "bvalue_idx").
    dim_to_index_pos : dict
        Mapping from dimension names to their positions in the indices tuple.
    fit_type : str
        Type of fit to perform. Either "t1", "adc", "patlak", "2cm" or "b1".
    fit_params : dict, optional
        Dictionary of fitting parameters specific to the fit_type.

    """
    current_selection_isel = dict(zip(non_dynamic_dims, indices))
    signal_1d_da = input_data.isel(current_selection_isel)

    # --- Extract the corresponding dynamic coordinates (TI times or b-values) ---
    dynamic_coords_for_fit = None
    dyn_coord_array = input_data.coords[dynamic_dim_name]

    try:
        coord_selection = {}
        for coord_dim in dyn_coord_array.dims:
            if coord_dim in non_dynamic_dims:
                coord_dim_pos = dim_to_index_pos[coord_dim]
                coord_selection[coord_dim] = indices[coord_dim_pos]
            elif coord_dim == dynamic_dim_name:
                coord_selection[coord_dim] = slice(None)
            else:  # Fallback for other coord dimensions not being iterated or dynamic
                coord_selection[coord_dim] = 0

        # Enhanced diagnostic logging for FAIR_RARE / inversion recovery data
        if dynamic_dim_name == "inversion_idx" and "slice_idx" in dyn_coord_array.dims:
            logger.info(
                f"[DIAGNOSTIC] Coordinate selection for T1 fitting:\n"
                f"  - Indices: {indices}\n"
                f"  - Non-dynamic dims: {non_dynamic_dims}\n"
                f"  - Dynamic dim: {dynamic_dim_name}\n"
                f"  - Coordinate array dims: {dyn_coord_array.dims}\n"
                f"  - Coordinate array shape: {dyn_coord_array.shape}\n"
                f"  - Coordinate selection: {coord_selection}\n"
                f"  - dim_to_index_pos: {dim_to_index_pos}"
            )

        dynamic_coords_for_fit = dyn_coord_array.isel(coord_selection).values.astype(
            float
        )  # ensure float

        # Log selected coordinates
        if dynamic_dim_name == "inversion_idx":
            logger.debug(
                f"Dynamic coords for fit at indices {indices}: {dynamic_coords_for_fit}"
            )
    except Exception as e:
        logger.error(
            f"Error selecting dynamic coordinates for indices {indices}, dim {dynamic_dim_name}: {e}"
        )
        return (None, indices)

    if (
        signal_1d_da.ndim == 1
        and dynamic_dim_name in signal_1d_da.dims
        and signal_1d_da.sizes[dynamic_dim_name] == len(dynamic_coords_for_fit)
    ):
        signal_1d_values = (
            signal_1d_da.values
        )  # Should already be float or complex from input_data

        fit_results = None
        if fit_type == "t1":
            fit_params["inversion_times_ms"] = dynamic_coords_for_fit
            fit_results = _fit_t1_inversion_recovery_helper(
                signal=signal_1d_values,
                fit_params=fit_params,
            )
        elif fit_type == "adc":
            fit_results = _fit_adc_monoexponential_helper(
                signal=signal_1d_values,
                fit_params=fit_params,
            )
        elif fit_type == "patlak":
            fit_results = _fit_single_voxel_patlak_helper(
                signal=signal_1d_values,
                fit_params=fit_params,
            )
        elif fit_type == "2cm":
            fit_results = _fit_single_voxel_2cm_helper(
                signal=signal_1d_values,
                fit_params=fit_params,
            )
        elif fit_type == "tofts":
            t1_map = fit_params.get("t1_map", None)
            s0_map = fit_params.get("s0_map", None)
            tr_s = fit_params.get("tr_s")
            fa_deg = fit_params.get("fa_deg")
            r1 = fit_params.get("r1", 3.5)
            cp_func = fit_params.get("cp_func")
            model_mode = fit_params.get("model_mode", "both")

            fit_times = dynamic_coords_for_fit
            c_t = signal_1d_values

            if (
                t1_map is not None
                and s0_map is not None
                and tr_s is not None
                and fa_deg is not None
            ):
                # Safety check: if maps are numpy and don't match the full indices length
                # (e.g. due to alignment issues), try to use just the indices they do have
                if isinstance(t1_map, np.ndarray) and t1_map.ndim != len(indices):
                    # Fallback: if they are 3D (read, slice, phase), try to find those indices
                    # This is a last-resort to prevent crash
                    try:
                        t1v = t1_map[indices[-t1_map.ndim :]]
                        s0v = s0_map[indices[-s0_map.ndim :]]
                    except:
                        return (None, indices)
                else:
                    t1v = t1_map[indices]
                    s0v = s0_map[indices]

                if t1v > 0 and t1v < 5000 and s0v > 1:
                    c_t = signal_to_concentration(
                        signal_1d_values, s0v, t1v / 1000.0, tr_s, fa_deg, r1
                    )
                    if np.max(c_t) < 0.001:
                        return (None, indices)
                else:
                    return (None, indices)

            fit_results = {}

            # --- Calculate fit mask based on params ---
            fit_start_idx = fit_params.get("fit_start_index", 0)
            fit_end_idx = fit_params.get("fit_end_index", None)
            fit_start_time = fit_params.get("fit_start_time_s", -np.inf)
            fit_end_time = fit_params.get("fit_end_time_s", np.inf)
            fit_indices = fit_params.get("fit_indices", None)
            fit_times_include = fit_params.get("fit_times_s", None)

            n_points = len(fit_times)
            valid_mask = np.ones(n_points, dtype=bool)

            # Index slicing
            if fit_start_idx > 0:
                valid_mask[:fit_start_idx] = False
            if fit_end_idx is not None:
                valid_mask[fit_end_idx:] = False

            # Time windowing
            valid_mask &= (fit_times >= fit_start_time) & (fit_times <= fit_end_time)

            # Specific indices/times inclusion
            if fit_indices is not None:
                new_mask = np.zeros(n_points, dtype=bool)
                new_mask[fit_indices] = True
                valid_mask &= new_mask

            if fit_times_include is not None:
                new_mask = np.zeros(n_points, dtype=bool)
                # Find nearest indices for requested times
                for t_val in np.atleast_1d(fit_times_include):
                    idx = np.argmin(np.abs(fit_times - t_val))
                    new_mask[idx] = True
                valid_mask &= new_mask

            if model_mode in ["1comp", "both"]:
                p1 = _fit_single_voxel_tofts_helper(
                    fit_times, c_t, cp_func, "1comp", fit_mask=valid_mask
                )
                fit_results["Ktrans_1c"] = p1[0]
                fit_results["kep_1c"] = p1[1]

            if model_mode in ["2comp", "both"]:
                p2 = _fit_single_voxel_tofts_helper(
                    fit_times, c_t, cp_func, "2comp", fit_mask=valid_mask
                )
                fit_results["Ktrans_2c"] = p2[0]
                fit_results["kep_2c"] = p2[1]
                fit_results["vp_2c"] = p2[2]
                fit_results["ve_2c"] = np.clip(p2[0] / p2[1], 0, 1) if p2[1] > 0 else 0

        elif fit_type == "b1":
            # sqrt_refpower = np.sqrt(np.clip(dynamic_coords_for_fit, a_min=0.0, a_max=None))
            # fit_results = _fit_single_voxel_b1_period_sweep_helper(
            #     signal=signal_1d_values,
            #     sqrt_refpower=sqrt_refpower,
            #     fit_params=fit_params,
            # )
            print(
                "B1 fitting not yet implemented in this context, using region growing."
            )
        else:
            print(f"Warning: Unknown fit_type '{fit_type}' for indices {indices}.")
            return (None, indices)

        return (fit_results, indices)
    else:
        # print(f"Warning: Data not suitable for fitting at indices {indices}. Signal dim: {signal_1d_da.ndim}, dynamic dim: {dynamic_dim_name}, signal size: {signal_1d_da.sizes.get(dynamic_dim_name, -1)}, coord size: {len(dynamic_coords_for_fit if dynamic_coords_for_fit is not None else [])}")
        return (None, indices)


# --- Parameter setup function ---
def def_fit_params(
    input_data: xr.DataArray = None,
    fit_type: str = "t1",
    fit_params: dict = None,
    return_params_info: bool = False,
) -> xr.Dataset:
    """Prepare a Dataset of parameters for quantitative fitting.

    This function organizes all fitting parameters, storing global settings
    as Dataset attributes and creating voxel-specific parameter maps if needed.

    Parameters
    ----------
    input_data : xr.DataArray, optional
        The measurement data, used as a template for creating parameter maps.
        Can be None if return_params_info=True.
    fit_type : str
        Type of fit to perform. Either "t1", "adc", "patlak", "2cm", or "b1".
    fit_params : dict, optional
        A dictionary with user-defined parameter overrides. If None, default
        parameters are generated.
    return_params_info : bool
        If True, prints information about all potential parameters for the
        selected fit_type and returns None. When True, input_data can be None.

    Returns
    -------
    xr.Dataset or None
        A comprehensive Dataset containing all fitting parameters, ready for use
        with fit_quantitative_maps. Returns None if return_params_info=True.

    """
    if return_params_info:
        _print_params_info(fit_type)
        return None

    if input_data is None:
        raise ValueError("input_data is required when return_params_info=False")

    if fit_params is None:
        fit_params = {}

    # --- Define global parameters stored as Dataset attributes ---
    attrs = {
        "fit_type": fit_type,
        "use_multiprocessing": fit_params.get("use_multiprocessing", True),
        "number_of_cpu_cores": fit_params.get("number_of_cpu_cores", -1),
        "db_mode": fit_params.get("db_mode", False),
    }

    # Add fit-type-specific parameters
    if fit_type == "t1":
        t1_values_s = fit_params.get("t1_values_s", np.linspace(0.01, 10, 500))
        dynamic_dim = fit_params.get("dynamic_dim", "inversion_idx")
        attrs.update(
            {
                "dynamic_dim": dynamic_dim,
                "t1_values_s": t1_values_s,
                "t1_values_ms": t1_values_s * 1000.0,
                "fit_baseline": fit_params.get("fit_baseline", False),
                "inversion_times_ms": fit_params.get(
                    "inversion_times_ms", input_data.coords.get(dynamic_dim, None)
                ),
            }
        )

    elif fit_type == "adc":
        dynamic_dim = fit_params.get("dynamic_dim", "bvalue_idx")
        attrs.update(
            {
                "dynamic_dim": dynamic_dim,
                "signal_threshold": fit_params.get("signal_threshold", 1e-5),
                "b_values": fit_params.get(
                    "b_values", input_data.coords.get(dynamic_dim, None)
                ),
            }
        )

    elif fit_type == "patlak":
        aif_signal = fit_params.get("aif_signal", None)
        if aif_signal is None:
            raise ValueError(
                "No AIF signal (arterial input function) provided for Patlak fitting. "
                "Please provide 'aif_signal' in fit_params."
            )

        dynamic_dim = fit_params.get("dynamic_dim", "repetition_idx")
        attrs.update(
            {
                "dynamic_dim": dynamic_dim,
                "time_points_s": fit_params.get(
                    "time_points_s", input_data.coords.get(dynamic_dim, None)
                ),
                "aif_signal": aif_signal,
                "fit_start_time_s": fit_params.get("fit_start_time_s", -np.inf),
                "fit_stop_time_s": fit_params.get("fit_stop_time_s", np.inf),
                "fit_start_index": fit_params.get("fit_start_index", 0),
                "fit_stop_index": fit_params.get("fit_stop_index", None),
                "aif_threshold": fit_params.get("aif_threshold", 1e-6),
            }
        )

        # Warn if input data is complex
        if np.iscomplexobj(input_data.values):
            warnings.warn(
                "Input data for Patlak fitting is complex-valued. The Patlak model expects "
                "real-valued concentration data. Consider taking the magnitude or real part "
                "of the signal before fitting.",
                UserWarning,
                stacklevel=2,
            )

    elif fit_type == "2cm":
        aif_signal = fit_params.get("aif_signal", None)
        if aif_signal is None:
            raise ValueError(
                "No AIF signal (arterial input function) provided for 2CM fitting. "
                "Please provide 'aif_signal' in fit_params."
            )

        dynamic_dim = fit_params.get("dynamic_dim", "repetition_idx")
        attrs.update(
            {
                "dynamic_dim": dynamic_dim,
                "time_points_s": fit_params.get(
                    "time_points_s", input_data.coords.get(dynamic_dim, None)
                ),
                "aif_signal": aif_signal,
                "fit_start_time_s": fit_params.get("fit_start_time_s", 0.0),
                "fit_stop_time_s": fit_params.get("fit_stop_time_s", np.inf),
            }
        )

    elif fit_type == "tofts":
        aif_signal = fit_params.get("aif_signal", None)
        aif_times = fit_params.get("aif_times", None)
        if aif_signal is None or aif_times is None:
            raise ValueError(
                "Both 'aif_signal' and 'aif_times' must be provided for Tofts fitting."
            )

        # Create AIF interpolator
        cp_func = interp1d(
            aif_times, aif_signal, kind="linear", fill_value="extrapolate"
        )

        dynamic_dim = fit_params.get("dynamic_dim", "repetition_idx")
        attrs.update(
            {
                "dynamic_dim": dynamic_dim,
                "time_points_s": fit_params.get(
                    "time_points_s", input_data.coords.get(dynamic_dim, None)
                ),
                "cp_func": cp_func,
                "t1_map": fit_params.get("t1_map", None),
                "s0_map": fit_params.get("s0_map", None),
                "tr_s": fit_params.get("tr_s", None),
                "fa_deg": fit_params.get("fa_deg", None),
                "r1": fit_params.get("r1", 3.5),
                "model_mode": fit_params.get(
                    "model_mode", "both"
                ),  # "1comp", "2comp", or "both"
            }
        )

    elif fit_type == "b1":
        # B1-specific parameters
        fa_deg = fit_params.get("fa_deg", None)
        trf_ms = fit_params.get("trf_ms", None)
        sint = fit_params.get("sint", None)
        dynamic_dim = fit_params.get("dynamic_dim")
        if dynamic_dim is None:
            candidate_dims = fit_params.get(
                "dynamic_dim_candidates",
                ["rfpower_idx", "power_idx", "rf_power_idx", "rfpower"],
            )
            dynamic_dim = next(
                (dim for dim in candidate_dims if dim in input_data.dims), None
            )
            if dynamic_dim is None:
                raise ValueError(
                    "Unable to determine dynamic dimension for B1 parameters. "
                    "Provide 'dynamic_dim' in fit_params."
                )

        coord_refpower = input_data.coords.get(dynamic_dim, None)
        if coord_refpower is None:
            refpower_values = fit_params.get("refpower_W", None)
        else:
            if coord_refpower.ndim == 1 and coord_refpower.dims[0] == dynamic_dim:
                refpower_values = coord_refpower.values
            else:
                coord_sel = {
                    dim: 0 for dim in coord_refpower.dims if dim != dynamic_dim
                }
                coord_sel[dynamic_dim] = slice(None)
                refpower_values = coord_sel.values

        fit_params.setdefault("dynamic_dim", dynamic_dim)
        fit_params.setdefault("period_min", fit_params.get("period_min", 0.2))
        fit_params.setdefault("period_max", fit_params.get("period_max", None))
        fit_params.setdefault("period_samples", fit_params.get("period_samples", 180))

        attrs.update(
            {
                "dynamic_dim": dynamic_dim,
                "refpower_W": fit_params.get("refpower_W", refpower_values),
                "fa_deg": fa_deg,
                "trf_ms": trf_ms,
                "sint": sint,
                "period_min": fit_params.get("period_min"),
                "period_max": fit_params.get("period_max"),
                "period_samples": fit_params.get("period_samples"),
            }
        )
    else:
        raise ValueError(
            f"Unsupported fit_type: '{fit_type}'. Must be 't1', 'adc', 'patlak', '2cm', 'tofts', or 'b1'."
        )

    # Create empty Dataset with attributes
    fit_params_ds = xr.Dataset(attrs=attrs)

    return fit_params_ds


def _print_params_info(fit_type: str):
    """Print parameter information for a given fit_type."""
    params_info = {
        "t1": {
            "required_fit_params": {
                "t1_values_s": "Array of T1 values (in seconds) to test during grid search",
            },
            "optional_fit_params": {
                "fit_baseline": "If True, fits baseline parameter B in model S(t) = A*(1-2*exp(-t/T1)) + B. Default: False",
                "use_multiprocessing": "Enable parallel processing. Default: True",
                "number_of_cpu_cores": "Number of CPU cores for parallel processing (-1 uses all). Default: -1",
            },
            "output_parameters": {
                "t1": "Longitudinal relaxation time (seconds)",
                "A": "Signal amplitude parameter (can be complex if input is complex)",
                "B": "Baseline offset parameter (can be complex if input is complex, 0 if fit_baseline=False)",
                "rss_t1": "Residual sum of squares for the fit",
            },
            "model_equation": "S(t) = A * (1 - 2 * exp(-t / T1)) + B",
        },
        "adc": {
            "required_fit_params": {
                "b_values": "Array of b-values (e.g., in s/mm^2) corresponding to the diffusion weighting",
            },
            "optional_fit_params": {
                "signal_threshold": "Minimum signal value to include in fit (avoids log(0)). Default: 1e-5",
                "use_multiprocessing": "Enable parallel processing. Default: True",
                "number_of_cpu_cores": "Number of CPU cores for parallel processing (-1 uses all). Default: -1",
            },
            "output_parameters": {
                "adc": "Apparent diffusion coefficient (typically in mm^2/s)",
                "s0": "Signal at b=0 (baseline signal intensity)",
                "rss_adc": "Residual sum of squares for the fit (in log-space)",
            },
            "model_equation": "S(b) = S0 * exp(-b * ADC)",
        },
        "patlak": {
            "required_fit_params": {
                "aif_signal": "Arterial input function (AIF) sampled at the same time points as input_data",
                "time_points_s": "Time axis in seconds corresponding to the dynamic acquisition",
            },
            "optional_fit_params": {
                "fit_start_time_s": "Start time (in seconds) for fitting window. Default: -inf (all times)",
                "fit_stop_time_s": "Stop time (in seconds) for fitting window. Default: inf (all times)",
                "fit_start_index": "Start index for fitting window. Default: 0",
                "fit_stop_index": "Stop index (exclusive) for fitting window. Default: None (all indices)",
                "aif_threshold": "Minimum AIF value to include in fit (avoids division by small numbers). Default: 1e-6",
                "use_multiprocessing": "Enable parallel processing. Default: True",
                "number_of_cpu_cores": "Number of CPU cores for parallel processing (-1 uses all). Default: -1",
            },
            "output_parameters": {
                "K_trans": "Transfer constant from plasma to tissue (mL/min/g or similar units)",
                "Vp": "Plasma volume fraction",
                "rss_patlak": "Residual sum of squares for the linear regression",
            },
            "model_equation": "Ct(T) = K_trans * integral(Cp(t)dt) + Vp * Cp(T)",
        },
        "2cm": {
            "required_fit_params": {
                "aif_signal": "Arterial input function (AIF) sampled at the same time points as input_data",
                "time_points_s": "Time axis in seconds corresponding to the dynamic acquisition",
            },
            "optional_fit_params": {
                "fit_start_time_s": "Start time (in seconds) for fitting window. Default: 0.0",
                "fit_stop_time_s": "Stop time (in seconds) for fitting window. Default: inf (all times)",
                "use_multiprocessing": "Enable parallel processing. Default: True",
                "number_of_cpu_cores": "Number of CPU cores for parallel processing (-1 uses all). Default: -1",
            },
            "output_parameters": {
                "Fp": "Plasma flow (mL/min/g)",
                "ve": "Extravascular extracellular space volume fraction",
                "vb": "Blood volume fraction",
            },
            "model_equation": "Two-compartment renal filtration model from Zollner et al. (2019)",
        },
        "tofts": {
            "required_fit_params": {
                "aif_signal": "Arterial input function concentration values",
                "aif_times": "Time points in seconds for the AIF",
            },
            "optional_fit_params": {
                "t1_map": "Numpy array of pre-contrast T1 values (ms)",
                "s0_map": "Numpy array of baseline signal values",
                "tr_s": "Repetition time in seconds",
                "fa_deg": "Flip angle in degrees",
                "r1": "Relaxivity constant (default 3.5)",
                "model_mode": "One of '1comp', '2comp', or 'both' (default 'both')",
                "fit_start_index": "First repetition index to include (default 0)",
                "fit_end_index": "Index after the last repetition to include (default None)",
                "fit_start_time_s": "Start time in seconds for the window (default -inf)",
                "fit_end_time_s": "End time in seconds for the window (default inf)",
                "fit_indices": "List of specific repetition indices to include",
                "fit_times_s": "List of specific time points in seconds to include",
            },
            "output_parameters": {
                "Ktrans_1c": "Ktrans from 1-compartment Tofts model",
                "kep_1c": "kep from 1-compartment Tofts model",
                "Ktrans_2c": "Ktrans from 2-compartment Extended Tofts model",
                "kep_2c": "kep from 2-compartment Extended Tofts model",
                "vp_2c": "vp from 2-compartment Extended Tofts model",
                "ve_2c": "ve (Ktrans/kep) from 2-compartment model",
            },
            "model_equation": "Ct(t) = vp*Cp(t) + Ktrans * integral(Cp(tau)*exp(-kep*(t-tau))dtau)",
        },
        "b1": {
            "required_fit_params": {
                "refpower_W": "Array of reference power values in Watts corresponding to the RF power sweep",
                "fa_deg": "Flip angle in degrees (e.g., 90 for 90° pulse). Required for absolute B1 calibration",
                "trf_ms": "RF pulse duration in milliseconds. Required for reference power calculation",
                "sint": "Integration factor (dimensionless, related to RF pulse shape). Required for reference power calculation",
            },
            "optional_fit_params": {
                "period_samples": "Number of period values tested in each iteration. Default: 500",
                "use_region_growing": "Use region-growing algorithm to propagate initial guesses from neighbors. Default: False",
                "seed_voxel": "Seed voxel specification (tuple or dict indexed by xarray dims, e.g. {'read_idx': 32, 'phase_idx': 32}). Default: auto-detect",
                "stop_index_extra": "Additional samples beyond the inferred 180° cutoff to include in the fit window. Default: 0",
                "stage1_niter": "Iterations used for the first (full) sweep of the two-stage fit. Default: max(niter, 10)",
                "stage1_npoints": "Period samples used for the first sweep. Default: max(period_samples, 1500)",
                "store_debug_info": "Store iteration details (stop_indices, r2 maps) for debugging. Default: False",
                "use_multiprocessing": "Enable parallel processing. Default: True",
                "number_of_cpu_cores": "Number of CPU cores for parallel processing (-1 uses all). Default: -1",
            },
            "output_parameters": {
                "amplitude": "Complex signal amplitude from sinusoidal fit",
                "period": "Fitted period parameter (in sqrt(power) units, related to B1 field strength)",
                "refpower_90deg_W": "Reference power required for 90° flip angle (Watts), adjusted for pulse parameters",
                "rss_b1": "Residual sum of squares for the sinusoidal fit (minimum R² from optimization)",
            },
            "model_equation": "S(sqrt(P)) = A * sin(π / period * sqrt(P))",
        },
    }

    if fit_type not in params_info:
        raise ValueError(
            f"Unsupported fit_type: '{fit_type}'. Must be one of {list(params_info.keys())}"
        )

    info = params_info[fit_type]

    # Print parameter information in a nice format
    print("=" * 80)
    print(f"Parameter Information for fit_type='{fit_type}'")
    print("=" * 80)

    print(f"\nModel Equation:")
    print(f"  {info['model_equation']}")

    print(f"\n{'Required Parameters in fit_params:':-<80}")
    for param, description in info["required_fit_params"].items():
        print(f"  • {param}")
        print(f"      {description}")

    print(f"\n{'Optional Parameters in fit_params:':-<80}")
    for param, description in info["optional_fit_params"].items():
        print(f"  • {param}")
        print(f"      {description}")

    print(f"\n{'Output Parameters (in returned Dataset):':-<80}")
    for param, description in info["output_parameters"].items():
        print(f"  • {param}")
        print(f"      {description}")

    print("=" * 80)


# --- Generalized main fitting function ---
def fit_quantitative_maps(
    input_data: xr.DataArray = None,
    fit_type: str = "t1",
    fit_params: dict = {},
    return_params_info: bool = False,
):
    """
    Fit quantitative parameters (T1, ADC, Patlak, 2CM, Tofts, B1) voxel-wise.

    Parameters
    ----------
    input_data : xr.DataArray, optional
        Input data with a 'dynamic' dimension (e.g., 'inversion_idx' for T1,
        'bvalue_idx' for ADC, 'rfpower_idx' for B1) and its corresponding coordinate.
        Can be None if return_params_info=True.
    fit_type : str
        Type of fit to perform. Either "t1", "adc", "patlak", "2cm", "tofts", or "b1".
    fit_params : dict
        Dictionary containing fitting parameters.
        Required:

        - 'dynamic_dim': str, Name of the dynamic dimension (e.g., "inversion_idx", "bvalue_idx").

        Optional general:

        - 'use_multiprocessing': bool (default: True)
        - 'number_of_cpu_cores': int (default: -1)

        For fit_type='t1':
        - 't1_values_s': np.ndarray, T1 values (in seconds) to test.
        - 'fit_baseline': bool, If True, fit T1 baseline.

        For fit_type='adc':
        - 'signal_threshold': float (default: 1e-5), signal threshold for ADC fit.

        For fit_type='patlak' or '2cm':
        - 'aif_signal': AIF signal array (required)
        - 'time_points_s': Time points in seconds

        For fit_type='tofts':
        - 'aif_signal': Arterial Input Function concentration values (required).
        - 'aif_times': Time points in seconds for the AIF (required).
        - 't1_map': np.ndarray or xr.DataArray, pre-contrast T1 map in ms.
        - 's0_map': np.ndarray or xr.DataArray, baseline signal map.
        - 'n_precontrast': int, number of pre-contrast frames to auto-calculate s0_map if missing.
        - 'tr_s': float, repetition time in seconds (for signal-to-concentration).
        - 'fa_deg': float, flip angle in degrees (for signal-to-concentration).
        - 'r1': float, relaxivity (default 3.5).
        - 'model_mode': str, "1comp", "2comp" or "both" (default).
        - 'fit_start_index': int, first repetition index to include.
        - 'fit_end_index': int, index after the last repetition to include.
        - 'fit_start_time_s': float, start time in seconds for the window.
        - 'fit_end_time_s': float, end time in seconds for the window.
        - 'fit_indices': list, specific repetition indices to include.
        - 'fit_times_s': list, specific time points in seconds to include.

        For fit_type='b1':
        - 'period_samples': int (default: 500), length of period array for fitting
        - 'fa_deg': float (default: 90), flip angle in degrees
        - 'trf_ms': float (default: 1), RF pulse duration in ms
        - 'sint': float (default: 1), integration factor
        - 'use_region_growing': bool (default: False), use region-growing algorithm
        - 'seed_voxel': tuple or None, (x, y) coordinates of seed voxel for region growing
        - 'stop_index_extra': int (default: 0), additional samples beyond the 180° cutoff when determining stop indices
        - 'plot_region_growing_progress': bool (default: False), live-update region mask and stop index maps during fitting
        - 'region_growing_priority': str (default: 'fifo'), priority strategy for region growing ('fifo' or 'snr')

    return_params_info : bool
        If True, prints information about all potential parameters for the
        selected fit_type and returns None (does not perform the fit).
        When True, input_data can be None.

    Returns
    -------
    xr.Dataset or None:
        If return_params_info=False: An xarray Dataset containing the fitted parameter maps.
        - For T1: 't1', 'A', 'B', 'rss_t1'.
        - For ADC: 'adc', 's0', 'rss_adc'.
        - For Patlak: 'K_trans', 'Vp', 'rss_patlak'.
        - For 2CM: 'Fp', 've', 'vb'.
        - For Tofts: 'Ktrans_1c', 'kep_1c', 'Ktrans_2c', 'kep_2c', 'vp_2c', 've_2c'.
        - For B1: 'amplitude', 'period', 'refpower_90deg_W', 'rss_b1'.

        If return_params_info=True: None (prints parameter information instead).

    """
    if return_params_info:
        params_info = {
            "t1": {
                "required_fit_params": {
                    "t1_values_s": "Array of T1 values (in seconds) to test during grid search",
                },
                "optional_fit_params": {
                    "fit_baseline": "If True, fits baseline parameter B in model S(t) = A*(1-2*exp(-t/T1)) + B. Default: False",
                    "use_multiprocessing": "Enable parallel processing. Default: True",
                    "number_of_cpu_cores": "Number of CPU cores for parallel processing (-1 uses all). Default: -1",
                },
                "output_parameters": {
                    "t1": "Longitudinal relaxation time (seconds)",
                    "A": "Signal amplitude parameter (can be complex if input is complex)",
                    "B": "Baseline offset parameter (can be complex if input is complex, 0 if fit_baseline=False)",
                    "rss_t1": "Residual sum of squares for the fit",
                },
                "model_equation": "S(t) = A * (1 - 2 * exp(-t / T1)) + B",
            },
            "adc": {
                "required_fit_params": {
                    "b_values": "Array of b-values (e.g., in s/mm^2) corresponding to the diffusion weighting",
                },
                "optional_fit_params": {
                    "signal_threshold": "Minimum signal value to include in fit (avoids log(0)). Default: 1e-5",
                    "use_multiprocessing": "Enable parallel processing. Default: True",
                    "number_of_cpu_cores": "Number of CPU cores for parallel processing (-1 uses all). Default: -1",
                },
                "output_parameters": {
                    "adc": "Apparent diffusion coefficient (typically in mm^2/s)",
                    "s0": "Signal at b=0 (baseline signal intensity)",
                    "rss_adc": "Residual sum of squares for the fit (in log-space)",
                },
                "model_equation": "S(b) = S0 * exp(-b * ADC)",
            },
            "patlak": {
                "required_fit_params": {
                    "aif_signal": "Arterial input function (AIF) sampled at the same time points as input_data",
                    "time_points_s": "Time axis in seconds corresponding to the dynamic acquisition",
                },
                "optional_fit_params": {
                    "fit_start_time_s": "Start time (in seconds) for fitting window. Default: -inf (all times)",
                    "fit_stop_time_s": "Stop time (in seconds) for fitting window. Default: inf (all times)",
                    "fit_start_index": "Start index for fitting window. Default: 0",
                    "fit_stop_index": "Stop index (exclusive) for fitting window. Default: None (all indices)",
                    "aif_threshold": "Minimum AIF value to include in fit (avoids division by small numbers). Default: 1e-6",
                    "use_multiprocessing": "Enable parallel processing. Default: True",
                    "number_of_cpu_cores": "Number of CPU cores for parallel processing (-1 uses all). Default: -1",
                },
                "output_parameters": {
                    "K_trans": "Transfer constant from plasma to tissue (mL/min/g or similar units)",
                    "Vp": "Plasma volume fraction",
                    "rss_patlak": "Residual sum of squares for the linear regression",
                },
                "model_equation": "Ct(T) = K_trans * integral(Cp(t)dt) + Vp * Cp(T)",
            },
            "2cm": {
                "required_fit_params": {
                    "aif_signal": "Arterial input function (AIF) sampled at the same time points as input_data",
                    "time_points_s": "Time axis in seconds corresponding to the dynamic acquisition",
                },
                "optional_fit_params": {
                    "fit_start_time_s": "Start time (in seconds) for fitting window. Default: 0.0",
                    "fit_stop_time_s": "Stop time (in seconds) for fitting window. Default: inf (all times)",
                    "use_multiprocessing": "Enable parallel processing. Default: True",
                    "number_of_cpu_cores": "Number of CPU cores for parallel processing (-1 uses all). Default: -1",
                },
                "output_parameters": {
                    "Fp": "Plasma flow (mL/min/g)",
                    "ve": "Extravascular extracellular space volume fraction",
                    "vb": "Blood volume fraction",
                },
                "model_equation": "Two-compartment renal filtration model from Zollner et al. (2019)",
            },
            "b1": {
                "required_fit_params": {
                    "refpower_W": "Array of reference power values in Watts corresponding to the RF power sweep",
                    "fa_deg": "Flip angle in degrees (e.g., 90 for 90-degree pulse)",
                    "trf_ms": "RF pulse duration in milliseconds",
                    "sint": "RF pulse integration factor (dimensionless, related to pulse shape)",
                },
                "optional_fit_params": {
                    "period_samples": "Number of period values to test in each iteration. Default: 1000",
                    "use_region_growing": "Use region-growing algorithm starting from high-signal voxel. Default: False",
                    "seed_voxel": "Seed voxel coordinates (x, y) for region growing. Default: None (auto-detect)",
                    "plot_region_growing_progress": "Display live region mask and stop-index map updates during region growing. Default: False",
                    "region_growing_priority": "Queue strategy for region growing ('fifo' for breadth-first, 'snr' to visit higher-signal voxels first). Default: 'fifo'",
                    "use_multiprocessing": "Enable parallel processing. Default: True",
                    "number_of_cpu_cores": "Number of CPU cores for parallel processing (-1 uses all). Default: -1",
                },
                "output_parameters": {
                    "amplitude": "Complex signal amplitude from sinusoidal fit",
                    "period": "Fitted period parameter (sqrt(W) units)",
                    "refpower_90deg_W": "Reference power (W) required for 90° flip angle, adjusted for pulse parameters",
                    "rss_b1": "Residual sum of squares for the fit (minimum R²)",
                },
                "model_equation": "S(sqrt(P)) = A * sin(π / period * sqrt(P))",
            },
        }

        if fit_type not in params_info:
            raise ValueError(
                f"Unsupported fit_type: '{fit_type}'. Must be one of {list(params_info.keys())}"
            )

        info = params_info[fit_type]

        # Print parameter information in a nice format
        print("=" * 80)
        print(f"Parameter Information for fit_type='{fit_type}'")
        print("=" * 80)

        print(f"\nModel Equation:")
        print(f"  {info['model_equation']}")

        print(f"\n{'Required Parameters in fit_params:':-<80}")
        for param, description in info["required_fit_params"].items():
            print(f"  • {param}")
            print(f"      {description}")

        print(f"\n{'Optional Parameters in fit_params:':-<80}")
        for param, description in info["optional_fit_params"].items():
            print(f"  • {param}")
            print(f"      {description}")

        print(f"\n{'Output Parameters (in returned Dataset):':-<80}")
        for param, description in info["output_parameters"].items():
            print(f"  • {param}")
            print(f"      {description}")

        print("=" * 80)

        return None

    if input_data is None:
        raise ValueError("input_data is required when return_params_info=False")

    if not isinstance(input_data, xr.DataArray):
        raise TypeError(
            f"input_data must be an xarray.DataArray, but got {type(input_data)}. "
            "The fit_quantitative_maps function requires labeled xarray data to handle "
            "coordinates and parallelization. For fitting single NumPy arrays, please "
            "use the internal helper functions (e.g., _fit_t1_inversion_recovery_helper)."
        )

    use_multiprocessing = fit_params.get("use_multiprocessing", True)
    number_of_cpu_cores = fit_params.get("number_of_cpu_cores", -1)

    # Parameters for T1 fit
    model_name = None
    model_equation = None

    if fit_type == "t1":
        model_name = "t1_inversion_recovery"
        model_equation = (
            # Store the equation for human-readability
            "A * (1 - 2 * exp(-t / T1)) + B"
        )
        t1_values_s = fit_params.get("t1_values_s", np.linspace(0.01, 5, 500))
        t1_values_ms = t1_values_s * 1000.0  # Convert seconds to milliseconds
        dynamic_dim = fit_params.get("dynamic_dim", "inversion_idx")
        output_param_names = ["t1", "A", "B", "rss_t1"]

        fit_params["t1_values_ms"] = t1_values_ms
        fit_params["t1_values_s"] = t1_values_s
        fit_params["fit_baseline"] = fit_params.get("fit_baseline", False)
        fit_params["inversion_times_ms"] = input_data.coords.get(dynamic_dim, None)
        fit_params["dynamic_dim"] = dynamic_dim

        if fit_params.get("db_mode", False):
            print(f"Fitting {fit_type} with the following parameters:")
            print(f"  - t1_values_s: {t1_values_s}")
            print(f"  - fit_baseline: {fit_params['fit_baseline']}")
            print(f"  - inversion_times_ms: {fit_params['inversion_times_ms']}")
            print(f"  - output_param_names: {output_param_names}")

    elif fit_type == "adc":
        model_name = "monoexponential_adc"
        model_equation = (
            "s0 * exp(-b * adc)"  # Store the equation for human-readability
        )

        # ADC specific params for worker
        dynamic_dim = fit_params.get("dynamic_dim", "bvalue_idx")
        output_param_names = ["adc", "s0", "rss_adc"]

        fit_params["adc_signal_threshold"] = fit_params.get(
            "adc_signal_threshold", 1e-5
        )
        fit_params["b_values"] = fit_params.get(
            "b_values", input_data.coords.get(dynamic_dim, None)
        )
        fit_params["dynamic_dim"] = dynamic_dim
        if fit_params.get("db_mode", False):
            print(f"Fitting {fit_type} with the following parameters:")
            print(f"  - b_values: {fit_params['b_values']}")
            print(f"  - adc_signal_threshold: {fit_params['adc_signal_threshold']}")

    elif fit_type == "patlak":
        model_name = "patlak"
        model_equation = "Ct(T) = K_trans * integral(Cp(t)dt) + Vp * Cp(T)"
        dynamic_dim = "repetition_idx"
        output_param_names = ["K_trans", "Vp", "rss_patlak"]

        # Patlak specific params for worker
        # Try to get time_points_s from multiple sources in order of preference:
        # 1. Explicitly provided in fit_params
        # 2. 'time' coordinate (new, contains frame_time_s)
        # 3. 'repetition_idx' coordinate (old, might be indices or time values)
        fit_params["time_points_s"] = fit_params.get(
            "time_points_s",
            input_data.coords.get(
                "time", input_data.coords.get("repetition_idx", None)
            ),
        )
        fit_params["aif_signal"] = fit_params.get("aif_signal", None)
        fit_params["fit_start_time_s"] = fit_params.get("fit_start_time_s", 0.0)
        fit_params["aif_threshold"] = fit_params.get("aif_threshold", 1e-6)
        fit_params["dynamic_dim"] = dynamic_dim

        # Check if AIF signal provided
        if fit_params["aif_signal"] is None:
            raise ValueError(
                "No AIF signal (arterial input function) provided for Patlak fitting. "
                "Please provide 'aif_signal' in fit_params."
            )

        # Check if time_points_s is valid
        if fit_params["time_points_s"] is None:
            raise ValueError(
                "No time points provided for Patlak fitting. The 'time_points_s' parameter is required "
                "and must contain actual time values in seconds (not indices).\n"
                "Options to fix this:\n"
                "1. Re-reconstruct your data with the latest version to automatically get a 'time' coordinate\n"
                "2. Provide it explicitly: fit_params['time_points_s'] = np.arange(n_points) * TR_in_seconds\n"
                "3. Use the 'time' coordinate if available: fit_params['time_points_s'] = data.coords['time'].values"
            )

        # Validate that time_points_s looks reasonable
        time_points_check = np.asarray(fit_params["time_points_s"]).squeeze()
        if len(time_points_check) > 2:
            if np.allclose(
                time_points_check, np.arange(len(time_points_check)), rtol=0.01
            ):
                # Check if the coordinate has a unit attribute indicating it's already in seconds
                coord_obj = input_data.coords.get("repetition_idx", None)
                coord_unit = (
                    coord_obj.attrs.get("unit", "unknown")
                    if coord_obj is not None and hasattr(coord_obj, "attrs")
                    else "unknown"
                )

                if coord_unit == "s":
                    # This shouldn't happen - the coordinate claims to be in seconds but looks like indices
                    raise ValueError(
                        f"The 'repetition_idx' coordinate has unit='{coord_unit}' but contains values that look like "
                        "indices (0, 1, 2, ...). This is likely a data integrity issue. Please check your data reconstruction."
                    )
                else:
                    # Check if 'time' coordinate exists and has proper values
                    has_time_coord = "time" in input_data.coords
                    time_coord_msg = ""
                    if has_time_coord:
                        time_coord_msg = "\nNote: Your data has a 'time' coordinate, but it was not used. Try: fit_params['time_points_s'] = data.coords['time'].values"

                    raise ValueError(
                        "The 'time_points_s' parameter appears to contain indices (0, 1, 2, ...) rather than "
                        "actual time values in seconds. The Patlak model requires time values in seconds.\n"
                        "Options to fix:\n"
                        "1. Re-reconstruct your data with the latest version to get a 'time' coordinate automatically\n"
                        "2. Manually provide time values: fit_params['time_points_s'] = np.arange(n) * TR_in_seconds\n"
                        "3. Convert repetition_idx to time: fit_params['time_points_s'] = data.coords['repetition_idx'].values * TR"
                        + time_coord_msg
                    )

        # Warn if input data is complex for Patlak fit
        if np.iscomplexobj(input_data.values):
            warnings.warn(
                "Input data for Patlak fitting is complex-valued. The Patlak model expects "
                "real-valued concentration data. Consider taking the magnitude or real part "
                "of the signal before fitting.",
                UserWarning,
                stacklevel=2,
            )

        if fit_params.get("db_mode", False):
            print(f"Fitting {fit_type} with the following parameters:")
            time_pts = fit_params["time_points_s"]
            if hasattr(time_pts, "values"):
                time_pts_preview = (
                    time_pts.values[:5] if len(time_pts.values) > 5 else time_pts.values
                )
            else:
                time_pts_preview = time_pts[:5] if len(time_pts) > 5 else time_pts
            print(f"  - time_points_s (first 5): {time_pts_preview}")
            print(
                f"  - time_points_s range: {np.min(time_pts)} to {np.max(time_pts)} seconds"
            )
            aif_sig = fit_params["aif_signal"]
            if hasattr(aif_sig, "values"):
                aif_preview = (
                    aif_sig.values[:5] if len(aif_sig.values) > 5 else aif_sig.values
                )
            else:
                aif_preview = aif_sig[:5] if len(aif_sig) > 5 else aif_sig
            print(f"  - aif_signal (first 5): {aif_preview}")
            print(f"  - fit_start_time_s: {fit_params['fit_start_time_s']}")

    elif fit_type == "2cm":
        model_name = "two_compartment_model"
        model_equation = (
            "Two-compartment renal filtration model from Zollner et al. (2019)"
        )
        # 2 compartment specific params for worker
        dynamic_dim = "repetition_idx"
        output_param_names = ["Fp", "ve", "vb"]
        fit_params["time_points_s"] = fit_params.get(
            "time_points_s", input_data.coords.get("repetition_idx", None)
        )
        fit_params["aif_signal"] = fit_params.get(
            "aif_signal", input_data.coords.get("aif_signal", None)
        )
        fit_params["fit_start_time_s"] = fit_params.get("fit_start_time_s", 0.0)
        fit_params["fit_stop_time_s"] = fit_params.get("fit_stop_time_s", np.inf)

        # Check if AIF signal provided
        if fit_params["aif_signal"] is None:
            raise ValueError(
                "No AIF signal (arterial input function) provided for 2CM fitting. "
                "Please provide 'aif_signal' in fit_params."
            )

        if fit_params.get("db_mode", False):
            print(f"Fitting {fit_type} with the following parameters:")
            print(f"  - time_points_s: {fit_params['time_points_s']}")
            print(f"  - aif_signal: {fit_params['aif_signal']}")
            print(f"  - fit_start_time_s: {fit_params['fit_start_time_s']}")
            print(f"  - fit_stop_time_s: {fit_params['fit_stop_time_s']}")

    elif fit_type == "tofts":
        model_name = "tofts_model"
        model_equation = (
            "Ct(t) = vp*Cp(t) + Ktrans * integral(Cp(tau)*exp(-kep*(t-tau))dtau)"
        )
        dynamic_dim = "repetition_idx"

        model_mode = fit_params.get("model_mode", "both")
        if model_mode == "1comp":
            output_param_names = ["Ktrans_1c", "kep_1c"]
        elif model_mode == "2comp":
            output_param_names = ["Ktrans_2c", "kep_2c", "vp_2c", "ve_2c"]
        else:
            output_param_names = [
                "Ktrans_1c",
                "kep_1c",
                "Ktrans_2c",
                "kep_2c",
                "vp_2c",
                "ve_2c",
            ]

        fit_params["time_points_s"] = fit_params.get(
            "time_points_s", input_data.coords.get("repetition_idx", None)
        )

        aif_signal = fit_params.get("aif_signal")
        aif_times = fit_params.get("aif_times")
        if aif_signal is None or aif_times is None:
            raise ValueError(
                "Tofts fitting requires both 'aif_signal' and 'aif_times' in fit_params."
            )

        # Ensure they are 1D numpy arrays to prevent multi-dimensional interpolation output
        aif_signal = np.asarray(aif_signal).squeeze()
        aif_times = np.asarray(aif_times).squeeze()

        if aif_signal.ndim != 1 or aif_times.ndim != 1:
            raise ValueError(
                f"Tofts fitting requires 1D AIF signal and times. Got shapes {aif_signal.shape} and {aif_times.shape}"
            )

        # Optional: Slice AIF before interpolation (e.g. to remove artifacts at start)
        # Default to the same window as the fit if not specified
        aif_start = fit_params.get(
            "aif_start_index", fit_params.get("fit_start_index", 0)
        )
        aif_end = fit_params.get("aif_end_index", fit_params.get("fit_end_index"))

        if (aif_start is not None and aif_start > 0) or aif_end is not None:
            aif_signal_length = len(aif_signal)

            aif_signal = aif_signal[aif_start:aif_end]
            aif_times = aif_times[aif_start:aif_end]
            print(
                f"Slicing AIF: start_index={aif_start}, end_index={aif_end}, {aif_signal_length} -> {len(aif_signal)}"
            )

        fit_params["cp_func"] = interp1d(
            aif_times, aif_signal, kind="linear", fill_value="extrapolate"
        )

        # Update fit_params with necessary metadata for the worker
        fit_params.update(
            {
                "dynamic_dim": dynamic_dim,
                "time_points_s": fit_params.get(
                    "time_points_s", input_data.coords.get("repetition_idx", None)
                ),
                "t1_map": fit_params.get("t1_map", None),
                "s0_map": fit_params.get("s0_map", None),
                "tr_s": fit_params.get("tr_s", None),
                "fa_deg": fit_params.get("fa_deg", None),
                "r1": fit_params.get("r1", 3.5),
                "model_mode": fit_params.get(
                    "model_mode", "both"
                ),  # "1comp", "2comp", or "both"
                "fit_start_index": fit_params.get("fit_start_index", 0),
                "fit_end_index": fit_params.get("fit_end_index", None),
                "fit_start_time_s": fit_params.get("fit_start_time_s", -np.inf),
                "fit_end_time_s": fit_params.get("fit_end_time_s", np.inf),
            }
        )

        # --- Handle S0 Map Calculation ---
        if fit_params.get("s0_map") is None:
            n_pre = fit_params.get("n_precontrast", 0)
            if n_pre > 0:
                print(f"Calculating S0 map from first {n_pre} repetitions...")
                s0_map = input_data.isel({dynamic_dim: slice(0, n_pre)}).mean(
                    dim=dynamic_dim
                )
                fit_params["s0_map"] = s0_map
            else:
                warnings.warn(
                    "No s0_map or n_precontrast provided. Fitting will proceed without signal-to-concentration conversion (assuming input is already concentration)."
                )

    elif fit_type == "b1":
        model_name = "b1_sinusoidal"
        model_equation = "S(sqrt(P)) = A * sin(π / period * sqrt(P))"
        dynamic_dim = fit_params.get("dynamic_dim")
        if dynamic_dim is None:
            candidate_dims = fit_params.get(
                "dynamic_dim_candidates",
                ["rfpower_idx", "power_idx", "rf_power_idx", "rfpower"],
            )
            dynamic_dim = next(
                (dim for dim in candidate_dims if dim in input_data.dims), None
            )
            if dynamic_dim is None:
                raise ValueError(
                    "Unable to determine the dynamic RF power dimension for B1 fitting. "
                    "Please supply 'dynamic_dim' in fit_params."
                )

        if dynamic_dim not in input_data.dims:
            raise ValueError(
                f"Dynamic dimension '{dynamic_dim}' not found in input_data dims {input_data.dims}."
            )

        output_param_names = [
            "amplitude",
            "period",
            "refpower_90deg_W",
            "rss_b1",
            "stop_index",
            "n_valid_samples",
        ]

        power_coord = input_data.coords.get(dynamic_dim, None)
        power_values = None
        if power_coord is not None:
            if power_coord.ndim == 1 and power_coord.dims[0] == dynamic_dim:
                power_values = power_coord.values.astype(float)
            else:
                coord_sel = {dim: 0 for dim in power_coord.dims if dim != dynamic_dim}
                coord_sel[dynamic_dim] = slice(None)
                power_values = power_coord.isel(coord_sel).values.astype(float)

            power_values = np.asarray(power_values, dtype=float).squeeze()
            if power_values.size == 0:
                raise ValueError(
                    "RF power coordinate appears empty after selection. "
                    "Check the dynamic dimension setup."
                )
            fit_params.setdefault("refpower_W", power_values)
        elif "refpower_W" in fit_params:
            power_values = np.asarray(fit_params["refpower_W"], dtype=float).squeeze()
        else:
            raise ValueError(
                f"No RF power coordinate found for '{dynamic_dim}'. Provide "
                "'refpower_W' in fit_params."
            )

        period_grid = fit_params.get("period_grid", None)
        if period_grid is None:
            period_min = float(fit_params.get("period_min", 0.2))
            period_max = fit_params.get("period_max", None)
            period_samples = int(fit_params.get("period_samples", 1000))

            if period_samples < 2:
                raise ValueError(
                    "B1 fitting requires at least two period samples. "
                    "Increase 'period_samples' in fit_params."
                )

            sqrt_power_values = np.sqrt(np.clip(power_values, a_min=0.0, a_max=None))
            finite_sqrt = sqrt_power_values[np.isfinite(sqrt_power_values)]
            debug_log_fit_maps(f"B1 fit: finite sqrt power values: {finite_sqrt}")
            if finite_sqrt.size == 0:
                finite_sqrt = np.array([1.0], dtype=float)

            if period_max is None:
                # TO DO: Consider making the 2.5 factor adjustable via fit_params
                period_max = float(np.nanmax(finite_sqrt) * 2.5)
                debug_log_fit_maps(
                    f"B1 fit: auto-set period_max to {period_max} based on data"
                )
                if not np.isfinite(period_max) or period_max <= 0:
                    period_max = 2.5

            if period_min <= 0:
                period_min = 0.1

            if period_max <= period_min:
                period_max = period_min * 1.5

            period_grid = np.linspace(period_min, period_max, period_samples)
            debug_log_fit_maps(
                f"B1 fit: generated period_grid from {period_min} to {period_max} with {period_samples} samples"
            )
        else:
            period_grid = np.asarray(period_grid, dtype=float).squeeze()
            if period_grid.ndim != 1 or period_grid.size < 2:
                raise ValueError(
                    "'period_grid' must be a 1D array with at least two entries."
                )

        fit_params["_b1_period_grid"] = np.asarray(period_grid, dtype=float)
        fit_params["dynamic_dim"] = dynamic_dim
        fit_params.setdefault("min_valid_samples", 3)
        fit_params.setdefault("period_min", float(period_grid[0]))
        fit_params.setdefault("period_max", float(period_grid[-1]))
        fit_params.setdefault("period_samples", int(period_grid.size))

        # Seed informed region growing setup
        seed_voxel = fit_params.get("seed_voxel", None)
        fit_params.get("use_region_growing", False)

        debug_log_fit_maps(f"=" * 80)
        debug_log_fit_maps(f"===== B1 Fitting: Seed Initialization =====")

        # Fit Seed voxel to determine inits for region growing
        if seed_voxel is not None:
            seed_voxel_signal = input_data.isel(seed_voxel).squeeze().values
            debug_log_fit_maps(
                f"B1 fit: seed voxel at {seed_voxel} has signal {seed_voxel_signal}"
            )

            # fig, ax = plt.subplots(1,4,figsize=(10,5))

            # 2 fit iterations, first to get rough period estimate, second to set stop index
            for k in range(4):

                # Perform a seed voxel fit as an initial check
                # seed_fit_results = "amplitude", "period", "refpower_90deg_W", "rss_b1", "stop_index", "n_valid_samples"
                seed_fit_results = _fit_single_voxel_b1_period_sweep_helper(
                    signal=seed_voxel_signal,
                    sqrt_refpower=sqrt_power_values,
                    fit_params=fit_params,
                )
                n_valid_samples = seed_fit_results.get(
                    "n_valid_samples", len(power_values)
                )

                period = seed_fit_results.get("period", None)

                # --- Fit only up to period (180) + stop_index_extra ---
                stop_index_extra = fit_params.get("stop_index_extra", 0)
                # Turn period (sqrt(W)) into stop index directly
                # period_idx = np.searchsorted(sqrt_power_values, period)
                # This takes the closest index to the period value (lower end)
                period_idx = np.searchsorted(
                    sqrt_power_values, period
                )  # + 1 # +1 to include period point
                # This takes the closest index to the period value (either side)
                # period_idx = np.argmin(np.abs(sqrt_power_values - period)) + 1 # +1 to include period point
                stop_index = period_idx + stop_index_extra

                # ax[k].scatter(sqrt_power_values, np.real(seed_voxel_signal), label='Data', color='blue')
                # ax[k].plot(sqrt_power_values, np.real(seed_fit_results['amplitude'] * np.sin(np.pi / period * sqrt_power_values)))
                # ax[k].axvline(x=sqrt_power_values[stop_index], color='red', linestyle='--', label='Stop Index')
                # ax[k].legend()

                # Ensure stop_index does not exceed n_valid_samples
                if stop_index > n_valid_samples:
                    stop_index = n_valid_samples

                debug_log_fit_maps(
                    f"B1 fit: seed voxel initial period fit result: {period}, stop_index: {stop_index}"
                )

                fit_params["stop_index"] = stop_index
            seed_fit_results["stop_index"] = stop_index
            seed_fit_results["_b1_period_grid"] = fit_params["_b1_period_grid"]
            fit_params["seed_fit_results"] = seed_fit_results

            # display(fig)
            # plt.pause(2)

        if fit_params.get("db_mode", False):
            print(f"Fitting {fit_type} with the following parameters:")
            print(f"  - dynamic_dim: {dynamic_dim}")
            print(f"  - period_min: {fit_params['period_min']}")
            print(f"  - period_max: {fit_params['period_max']}")
            print(f"  - period_samples: {fit_params['period_samples']}")

    else:
        raise ValueError(
            f"Unsupported fit_type: '{fit_type}'. Must be 't1', 'adc', 'patlak', '2cm', or 'b1'."
        )

    non_dynamic_dims = [dim for dim in input_data.dims if dim != dynamic_dim]
    non_dynamic_dims_set = set(non_dynamic_dims)

    def _sanitize_dim_coordinate(dim_name: str):
        """
        Return a coordinate that no longer depends on the dynamic dimension.

        Some reconstruction utilities (e.g., stacking with inserted scans)
        can attach dimension coordinates that vary along the dynamic axis.
        Those coordinates cannot be attached to the fit result because the
        dynamic dimension is removed in the output DataArrays. When this
        happens we try to drop the dynamic dimension if all slices are
        identical; otherwise we fall back to a simple index coordinate.
        """
        if dim_name not in input_data.coords:
            return None

        coord_da = input_data.coords[dim_name]
        coord_dims_set = set(coord_da.dims)

        if coord_dims_set.issubset(non_dynamic_dims_set):
            return coord_da

        if dynamic_dim in coord_dims_set:
            try:
                dyn_axis = coord_da.dims.index(dynamic_dim)
                values = np.asarray(coord_da.values)
                reference = np.take(values, 0, axis=dyn_axis)
                reference = np.expand_dims(reference, axis=dyn_axis)
                reference = np.broadcast_to(reference, values.shape)

                if np.issubdtype(values.dtype, np.number) or np.issubdtype(
                    values.dtype, np.bool_
                ):
                    identical = np.allclose(values, reference, equal_nan=True)
                else:
                    identical = np.array_equal(values, reference)

                if identical:
                    reduced = coord_da.isel({dynamic_dim: 0})
                    if dynamic_dim in reduced.dims:
                        reduced = reduced.squeeze(dim=dynamic_dim)
                    return reduced
            except Exception:
                pass

        warnings.warn(
            f"Dropping coordinate '{dim_name}' from fit outputs because it depends on the "
            f"dynamic dimension '{dynamic_dim}'. Consider sanitizing coordinates before fitting "
            "if this information is needed.",
            UserWarning,
            stacklevel=2,
        )

        return xr.DataArray(
            np.arange(input_data.sizes[dim_name]),
            dims=(dim_name,),
            name=dim_name,
        )

    # --- Handle Single Voxel Case (no other dimensions) ---
    if not non_dynamic_dims:
        print(
            f"Input data is 1D along {dynamic_dim}. Performing single {fit_type.upper()} fit."
        )
        signal_1d_values = input_data.values

        # Extract dynamic coordinates
        dyn_coord_array = input_data.coords[dynamic_dim]
        dynamic_coords_for_fit = None
        if dyn_coord_array.ndim == 1 and dyn_coord_array.dims[0] == dynamic_dim:
            dynamic_coords_for_fit = dyn_coord_array.values.astype(float)
        else:  # Attempt to extract if coordinate is multi-dimensional but matches input_data's single dynamic_dim
            try:
                # This might need more robust handling based on actual coord structure
                # For a truly 1D input_data along dynamic_dim, its coord should
                # also be 1D along dynamic_dim
                coord_sel = {d: 0 for d in dyn_coord_array.dims if d != dynamic_dim}
                coord_sel[dynamic_dim] = slice(None)
                dynamic_coords_for_fit = dyn_coord_array.isel(coord_sel).values.astype(
                    float
                )

            except Exception as e:
                raise ValueError(
                    f"Could not extract 1D dynamic coordinates from {dyn_coord_array.dims} for single voxel {fit_type} fit: {e}"
                )

        if dynamic_coords_for_fit is None:
            raise ValueError(
                f"Failed to extract dynamic_coords for single voxel {fit_type} fit."
            )

        fit_results_dict = None
        if fit_type == "t1":
            fit_results_dict = _fit_t1_inversion_recovery_helper(
                signal=signal_1d_values,
                fit_params=fit_params,
            )
        elif fit_type == "adc":
            fit_results_dict = _fit_adc_monoexponential_helper(
                signal=signal_1d_values,
                fit_params=fit_params,
            )
        elif fit_type == "patlak":
            fit_results_dict = _fit_single_voxel_patlak_helper(
                signal=signal_1d_values,
                fit_params=fit_params,
            )
        elif fit_type == "2cm":
            fit_results_dict = _fit_single_voxel_2cm_helper(
                signal=signal_1d_values,
                fit_params=fit_params,
            )
        elif fit_type == "b1":
            # to be implemented
            pass

        if fit_results_dict is not None:
            # For complex A, B make sure dtype is handled
            results_xr_dict = {}
            for key, val in fit_results_dict.items():
                dtype_val = np.float64
                if key in ["A", "B"] and np.iscomplexobj(
                    val
                ):  # A or B could be complex from T1 fit if signal complex
                    dtype_val = input_data.dtype  # Keep original complex dtype
                elif key in ["A", "B"] and not np.iscomplexobj(
                    input_data.values
                ):  # Ensure A, B are float if input is float
                    dtype_val = np.float64
                results_xr_dict[key] = xr.DataArray(val, dtype=dtype_val)
            return xr.Dataset(results_xr_dict)
        else:
            results_xr_dict = {
                param: xr.DataArray(np.nan, dtype=np.float64)
                for param in output_param_names
            }

            return xr.Dataset(results_xr_dict)

    # --- Setup for Multiple Voxels (Parallel or Sequential) ---
    shape_to_iterate = tuple(input_data.sizes[dim] for dim in non_dynamic_dims)
    total_voxels = np.prod(shape_to_iterate) if shape_to_iterate else 1

    print(
        f"Preparing to process {total_voxels} voxels for {fit_type.upper()} fitting..."
    )

    index_iterator = list(
        product(*(range(input_data.sizes[dim]) for dim in non_dynamic_dims))
    )
    dim_to_index_pos = {dim: i for i, dim in enumerate(non_dynamic_dims)}

    read_idx_pos = (
        non_dynamic_dims.index("read_idx") if "read_idx" in non_dynamic_dims else None
    )
    phase_idx_pos = (
        non_dynamic_dims.index("phase_idx") if "phase_idx" in non_dynamic_dims else None
    )
    coord_to_indices = {}
    if read_idx_pos is not None and phase_idx_pos is not None:
        for idx_tuple in index_iterator:
            coord_to_indices[(idx_tuple[read_idx_pos], idx_tuple[phase_idx_pos])] = (
                idx_tuple
            )

    # --- Execute Tasks (Parallel or Sequential) ---
    print(f"Executing {total_voxels} tasks for {fit_type.upper()} fitting...")

    # For Tofts, pre-process maps to ensure they are numpy and aligned
    if fit_type == "tofts":
        for map_key in ["t1_map", "s0_map"]:
            val = fit_params.get(map_key)
            if isinstance(val, xr.DataArray):
                # Align with input_data's spatial dimensions
                try:
                    # template is the spatial footprint of input_data
                    template = input_data.isel({dynamic_dim: 0}, drop=True)

                    # Physical coordinate alignment for spatial dimensions if counts differ
                    spatial_dims = ["read_idx", "phase_idx", "slice_idx"]

                    # 0. Temporarily drop spatial coordinates to avoid conflicts during indexing
                    # but keep a backup of physical coordinates for mapping logic
                    coords_backup = {
                        d: val.coords[d] for d in spatial_dims if d in val.coords
                    }
                    val_no_coords = val.drop_vars(
                        [d for d in spatial_dims if d in val.coords]
                    )

                    for dim in spatial_dims:
                        if (
                            dim in val.dims
                            and dim in template.dims
                            and val.sizes[dim] != template.sizes[dim]
                        ):

                            def get_repr_coords(da, dim_name, backup_coords=None):
                                c = da.coords.get(dim_name)
                                if c is None and backup_coords:
                                    c = backup_coords.get(dim_name)

                                if c is not None:
                                    if c.ndim == 3:
                                        if dim_name == "read_idx":
                                            return c.values[
                                                :, c.shape[1] // 2, c.shape[2] // 2
                                            ]
                                        elif dim_name == "phase_idx":
                                            return c.values[
                                                c.shape[0] // 2, :, c.shape[2] // 2
                                            ]
                                        elif dim_name == "slice_idx":
                                            return c.values[
                                                c.shape[0] // 2, c.shape[1] // 2, :
                                            ]
                                    elif c.ndim == 1:
                                        return c.values
                                return np.arange(da.sizes[dim_name])

                            val_c = get_repr_coords(val, dim, coords_backup)
                            tmpl_c = get_repr_coords(template, dim)
                            mapping = [np.argmin(np.abs(val_c - tc)) for tc in tmpl_c]

                            val_no_coords = val_no_coords.isel({dim: mapping})

                    # Re-assign coordinates from template to ensure exact match
                    for dim in spatial_dims:
                        if dim in template.coords:
                            val_no_coords = val_no_coords.assign_coords(
                                {dim: template.coords[dim]}
                            )

                    val = val_no_coords

                    # 1. Align remaining coordinates using nearest neighbor
                    # 2. Broadcast to add any missing dimensions from input_data
                    val_aligned = val.reindex_like(template, method="nearest")
                    val_broadcasted = val_aligned.broadcast_like(template)

                    # Ensure only non_dynamic_dims remain by dropping extra dims
                    to_drop = [
                        d for d in val_broadcasted.dims if d not in non_dynamic_dims
                    ]
                    if to_drop:
                        val_broadcasted = val_broadcasted.isel(
                            {d: 0 for d in to_drop}, drop=True
                        )

                    # 3. Transpose to ensure order matches non_dynamic_dims
                    val_ordered = val_broadcasted.transpose(*non_dynamic_dims)

                    fit_params[map_key] = val_ordered.values
                except Exception as e:
                    warnings.warn(
                        f"Could not align {map_key} with input data: {e}. "
                        "Falling back to raw values."
                    )
                    fit_params[map_key] = val.values

    results_list = []
    first_iteration_list = []

    # Define a helper function for a single task to keep the loops clean
    def run_single_task(indices):
        return _fit_single_voxel_quantitative(
            indices,
            input_data,
            non_dynamic_dims,
            dynamic_dim,
            dim_to_index_pos,
            fit_type,
            fit_params,
        )

    if fit_type == "b1":
        debug_log_fit_maps(f"fit_params={fit_params}", write_to_file=False)
        if read_idx_pos is None or phase_idx_pos is None:
            raise ValueError(
                "B1 region growing requires 'read_idx' and 'phase_idx' dimensions."
            )
        fit_params["index_iterator"] = index_iterator
        region_growing_output = region_growing_b1_fitting(
            input_data=input_data, fit_params=fit_params
        )
        # Handle single, dual, and triple return values (with optional debug info)
        debug_info = None
        if isinstance(region_growing_output, tuple):
            if len(region_growing_output) == 3:
                # Three return values: fit_results, first_iteration_results, debug_info
                fit_results, first_iteration_results, debug_info = region_growing_output
            elif len(region_growing_output) == 2:
                # Two return values: could be (fit_results, first_iteration_results)
                # or (fit_results, debug_info) depending on settings
                if fit_params.get("return_debug_info", False):
                    # If debug info was requested and only 2 values returned,
                    # it means two-iteration fit is disabled
                    fit_results, debug_info = region_growing_output
                    first_iteration_results = None
                else:
                    fit_results, first_iteration_results = region_growing_output
            else:
                fit_results = region_growing_output[0]
                first_iteration_results = None
        else:
            fit_results = region_growing_output
            first_iteration_results = None

        for coord, fit_res in fit_results.items():
            indices = coord_to_indices.get(coord, None)
            if indices is None:
                debug_log_fit_maps(
                    f"Skipping coordinate {coord} not present in index iterator mapping."
                )
                continue
            results_list.append((fit_res, indices))

            # Add first iteration result if available
            if first_iteration_results is not None and coord in first_iteration_results:
                first_iteration_list.append((first_iteration_results[coord], indices))

        # Store debug info in fit_params for potential access by caller
        if debug_info is not None:
            fit_params["_debug_info"] = debug_info
    else:
        if use_multiprocessing:
            print(f"Using joblib Parallel with {number_of_cpu_cores} cores.")
            # Create a list of delayed tasks specifically for Parallel
            tasks_for_parallel = [
                delayed(run_single_task)(indices) for indices in index_iterator
            ]
            results_list = Parallel(n_jobs=number_of_cpu_cores)(
                tqdm(
                    tasks_for_parallel,
                    desc=f"Processing voxels ({fit_type.upper()}, parallel)",
                )
            )
        else:
            print("Executing sequentially.")
            # Directly call the function in a simple loop
            for indices in tqdm(
                index_iterator,
                desc=f"Processing voxels({fit_type.upper()}, sequential)",
            ):
                results_list.append(run_single_task(indices))

    print("Fitting complete. Collecting results...")

    # --- Collect and Structure Results ---
    # Preserve ALL coordinates from input_data, not just those matching non_dynamic_dims
    result_coords = {}

    # First, add coordinates that match the non_dynamic_dims exactly
    for dim in non_dynamic_dims:
        sanitized_coord = _sanitize_dim_coordinate(dim)
        if sanitized_coord is not None:
            result_coords[dim] = sanitized_coord

    # Then, add any multidimensional coordinates (like x, y, z grids)
    # that depend on the non_dynamic_dims but aren't dimension coordinates themselves
    for coord_name, coord_data in input_data.coords.items():
        if coord_name not in result_coords:
            # Check if this coordinate's dimensions are a subset of non_dynamic_dims
            # This handles 2D/3D coordinate grids like x(read_idx, phase_idx)
            coord_dims_set = set(coord_data.dims)
            non_dyn_dims_set = set(non_dynamic_dims)

            # Include if all coordinate dims are in non_dynamic_dims
            # OR if it's a scalar coordinate (no dimensions)
            if coord_dims_set.issubset(non_dyn_dims_set) or len(coord_dims_set) == 0:
                result_coords[coord_name] = coord_data

    # Determine dtype for parameters like A, B, S0 based on input data
    # For T1 fit, A and B can be complex if input signal is complex and model supports it.
    # For ADC fit, S0 is typically real (derived from magnitude signal).
    complex_param_dtype = (
        input_data.dtype if np.iscomplexobj(input_data.values) else np.float64
    )

    fitted_param_arrays = {}
    for param_name in output_param_names:
        dtype = np.float64  # Default for t1, adc, rss
        if fit_type == "t1" and param_name in ["A", "B"]:
            dtype = complex_param_dtype
        elif fit_type == "adc" and param_name == "s0":  # S0 from ADC fit is real
            dtype = np.float64  # np.exp(log_S0) will be real if log_S0 is real
        elif fit_type == "b1" and param_name == "amplitude":
            dtype = complex_param_dtype  # B1 amplitude is complex
        elif fit_type == "b1" and param_name == "stop_index":
            dtype = np.float64  # store stop index as float to allow NaNs

        fitted_param_arrays[param_name] = np.full(shape_to_iterate, np.nan, dtype=dtype)

    # Store iteration arrays if requested (B1 only)
    iteration_arrays = {}

    for fit_result_dict, indices in results_list:
        if fit_result_dict is not None:
            for param_name in output_param_names:
                fitted_param_arrays[param_name][indices] = fit_result_dict.get(
                    param_name, np.nan
                )

            # Store iteration data if available
            if "all_periods" in iteration_arrays and "all_periods" in fit_result_dict:
                iteration_arrays["all_periods"][indices] = fit_result_dict[
                    "all_periods"
                ]
                iteration_arrays["all_r2"][indices] = fit_result_dict["all_r2"]

    # --- Process first iteration results if available ---
    first_iter_param_arrays = {}
    if first_iteration_list and fit_type == "b1":
        # Create arrays for first iteration results with "_iter1" suffix
        for param_name in output_param_names:
            dtype = np.float64
            if param_name == "amplitude":
                dtype = complex_param_dtype
            elif param_name == "stop_index":
                dtype = np.float64
            first_iter_param_arrays[f"{param_name}_iter1"] = np.full(
                shape_to_iterate, np.nan, dtype=dtype
            )

        # Fill in first iteration values
        for first_iter_dict, indices in first_iteration_list:
            if first_iter_dict is not None:
                for param_name in output_param_names:
                    first_iter_param_arrays[f"{param_name}_iter1"][indices] = (
                        first_iter_dict.get(param_name, np.nan)
                    )

    # --- Create result DataArrays ---
    output_dataset_dict = {}
    for param_name, data_array in fitted_param_arrays.items():
        output_dataset_dict[param_name] = xr.DataArray(
            data_array, coords=result_coords, dims=non_dynamic_dims, name=param_name
        )

    # Add first iteration arrays if available
    for param_name, data_array in first_iter_param_arrays.items():
        output_dataset_dict[param_name] = xr.DataArray(
            data_array, coords=result_coords, dims=non_dynamic_dims, name=param_name
        )

    # Add iteration arrays if they were stored
    if iteration_arrays:
        # Iteration arrays have extra dimensions: (...spatial..., niter, npoints)
        iter_dims = non_dynamic_dims + ["iteration_idx", "period_idx"]
        for iter_name, iter_array in iteration_arrays.items():
            output_dataset_dict[iter_name] = xr.DataArray(
                iter_array, dims=iter_dims, name=iter_name
            )

    # Add debug info arrays if they were returned from region growing
    if fit_type == "b1" and "_debug_info" in fit_params:
        debug_info = fit_params["_debug_info"]

        # Debug maps are 2D (read_idx, phase_idx only)
        # Create simple 1D index coordinates for them
        stop_index_shape = debug_info.get("stop_index_map", np.array([])).shape

        if len(stop_index_shape) == 2:
            debug_dims = ["read_idx", "phase_idx"]
            debug_coords = {
                "read_idx": np.arange(stop_index_shape[0]),
                "phase_idx": np.arange(stop_index_shape[1]),
            }

            # Add stop_index_map
            if "stop_index_map" in debug_info:
                output_dataset_dict["stop_index_map_iter1"] = xr.DataArray(
                    debug_info["stop_index_map"],
                    coords=debug_coords,
                    dims=debug_dims,
                    name="stop_index_map_iter1",
                )

            # Add region_mask
            if "region_mask" in debug_info:
                output_dataset_dict["region_mask"] = xr.DataArray(
                    debug_info["region_mask"].astype(np.float32),
                    coords=debug_coords,
                    dims=debug_dims,
                    name="region_mask",
                )

        # Store init_params_map in dataset attributes for reference
        # (can't easily convert to DataArray since it has varying structure per voxel)
        if "init_params_map" in debug_info:
            # We'll add this as an attribute to the dataset after it's created
            pass

    output_dataset = xr.Dataset(output_dataset_dict)

    # Add init_params_map to dataset attributes if available
    if fit_type == "b1" and "_debug_info" in fit_params:
        debug_info = fit_params["_debug_info"]
        if "init_params_map" in debug_info:
            # Store a summary or indication that init params are available
            output_dataset.attrs["has_init_params_map"] = True
            # The actual init_params_map dict is kept in fit_params["_debug_info"]["init_params_map"]
            # and can be accessed by the caller

    # --- Add metadata to the final dataset ---
    # First, copy ALL relevant attributes from input_data
    attrs_to_copy = [
        "patient_pos",
        "slice_orientation",
        "read_orientation",
        "read_direction",
        "phase_direction",
        "slice_direction",
        "slice_thickness_mm",
        "git_commit_hash",
        "spat_dim",
        "signal_domain",
        "domain",
        "unit",
        "matrix",
        "fov_mm",
        "voxel_size_mm",
        "n_slices",
        "n_channels",
        "acquisition_time_s",
        "description",
    ]

    for attr_name in attrs_to_copy:
        if attr_name in input_data.attrs:
            output_dataset.attrs[attr_name] = input_data.attrs[attr_name]

    # Then add/override with fit-specific metadata
    output_dataset.attrs["model_name"] = model_name
    output_dataset.attrs["model_equation"] = model_equation

    # Ensure git_commit_hash is present
    if "git_commit_hash" not in output_dataset.attrs:
        try:
            output_dataset.attrs["git_commit_hash"] = input_data.attrs.get(
                "git_commit_hash", "unknown"
            )
        except AttributeError:
            output_dataset.attrs["git_commit_hash"] = "unknown"

    # It's also good practice to store the coordinates used for the fit
    if fit_type == "adc":
        b_values_used = fit_params.get("b_values")
        if b_values_used is not None:
            output_dataset.attrs["b_values"] = list(b_values_used)
    elif fit_type == "patlak":
        time_points_s = fit_params.get("time_points_s")
        dynamic_dim = "repetition_idx"
        if time_points_s is not None:
            output_dataset.attrs["time_points_s"] = xr.DataArray(
                time_points_s, dims=[dynamic_dim]
            )

        aif_signal = fit_params.get("aif_signal")
        if aif_signal is not None:
            output_dataset.attrs["aif_signal"] = xr.DataArray(
                aif_signal, dims=[dynamic_dim]
            )
    elif fit_type == "tofts":
        time_points_s = fit_params.get("time_points_s")
        dynamic_dim = "repetition_idx"
        if time_points_s is not None:
            output_dataset.attrs["time_points_s"] = xr.DataArray(
                time_points_s, dims=[dynamic_dim]
            )

        output_dataset.attrs["model_mode"] = fit_params.get("model_mode", "both")
        output_dataset.attrs["tr_s"] = fit_params.get("tr_s")
        output_dataset.attrs["fa_deg"] = fit_params.get("fa_deg")
        output_dataset.attrs["r1"] = fit_params.get("r1", 3.5)

        # Store windowing parameters
        output_dataset.attrs["fit_start_index"] = fit_params.get("fit_start_index", 0)
        output_dataset.attrs["fit_end_index"] = fit_params.get("fit_end_index")
        output_dataset.attrs["aif_start_index"] = fit_params.get("aif_start_index", 0)
        output_dataset.attrs["aif_end_index"] = fit_params.get("aif_end_index")

        aif_signal = fit_params.get("aif_signal")
        aif_times = fit_params.get("aif_times")
        if aif_signal is not None:
            if aif_times is not None:
                # Store both in a DataArray with coords to allow re-interpolation
                output_dataset.attrs["aif_signal"] = xr.DataArray(
                    aif_signal,
                    dims=["aif_idx"],
                    coords={"aif_times": ("aif_idx", aif_times)},
                )
            else:
                output_dataset.attrs["aif_signal"] = xr.DataArray(
                    aif_signal, dims=[dynamic_dim]
                )
    elif fit_type == "t1":
        inversion_times_coord = input_data.coords.get(dynamic_dim)
        if inversion_times_coord is not None:
            # Storing as a data variable preserves all dimensions
            output_dataset.attrs["inversion_times_ms"] = inversion_times_coord
    elif fit_type == "2cm":
        time_points_s = fit_params.get("time_points_s")
        dynamic_dim = "repetition_idx"  # The time dimension
        if time_points_s is not None:
            output_dataset.attrs["time_points_s"] = xr.DataArray(
                time_points_s, dims=[dynamic_dim]
            )

        aif_signal = fit_params.get("aif_signal")
        if aif_signal is not None:
            output_dataset.attrs["aif_signal"] = xr.DataArray(
                aif_signal, dims=[dynamic_dim]
            )
    elif fit_type == "b1":
        refpower_W = fit_params.get("refpower_W")
        if refpower_W is not None:
            output_dataset.attrs["refpower_W"] = list(refpower_W)
        output_dataset.attrs["fa_deg"] = fit_params.get("fa_deg", 90)
        output_dataset.attrs["trf_ms"] = fit_params.get("trf_ms", 1)
        output_dataset.attrs["sint"] = fit_params.get("sint", 1)

    return output_dataset


@xr.register_dataset_accessor("hp")
class fit_accessor:
    """``xarray`` accessor that orchestrates model reconstructions."""

    def __init__(self, xarray_obj):
        """Attach the accessor to an :class:`xarray.Dataset`.

        Parameters
        ----------
        xarray_obj
            Dataset created by one of the fitting pipelines.
        """
        self._obj = xarray_obj
        self._model_reconstructors = {
            "monoexponential_adc": monoexponential_adc_model,
            "t1_inversion_recovery": inversion_recovery_model,
            "patlak": patlak_model,
            "tofts_model": tofts_model_reconstruction,
            "two_compartment_model": two_compartment_model_deprecated,
            "FID": fid_model,
            "b1_sinusoidal": b1_sinusoidal_model,
        }

    def reconstruct(self, voxel_coords: dict = None, **kwargs) -> xr.DataArray:
        """Reconstruct fitted signals for whole volumes or individual voxels.

        Parameters
        ----------
        voxel_coords
            Mapping of dimension names to indices that identify a single voxel.
            When ``None`` the reconstruction spans the full dataset.
        **kwargs
            Additional parameters required by the underlying model reconstructor,
            for example ``inversion_times_ms`` for inversion recovery models.

        Returns
        -------
        xarray.DataArray
            Reconstructed signal curves. When ``voxel_coords`` are provided the
            array is one-dimensional across the dynamic axis; otherwise it matches
            the spatial layout of the original dataset.

        Raises
        ------
        NotImplementedError
            If the dataset advertises an unsupported model via ``model_name``.
        ValueError
            If the required auxiliary parameters are missing from the dataset
            attributes or the supplied keyword arguments.
        """
        model_name = self._obj.attrs.get("model_name")
        if not model_name or model_name not in self._model_reconstructors:
            raise NotImplementedError(
                f"Reconstruction for model '{model_name}' is not implemented or model_name is missing, available models: {list( self._model_reconstructors.keys())}"
            )

        reconstructor = self._model_reconstructors[model_name]

        # Select the data for the specific voxel if coordinates are provided
        data_to_reconstruct = (
            self._obj.isel(voxel_coords) if voxel_coords else self._obj
        )

        # --- Gather parameters from the selected dataset and kwargs ---
        params = {}
        dynamic_dim_attr = data_to_reconstruct.attrs.get("dynamic_dim")
        dynamic_dim_override = kwargs.get("dynamic_dim")

        if model_name == "monoexponential_adc":
            # Prefer 'dynamic_dim' override, then 'b_values' from kwargs,
            # then 'b_values' from attrs, then try coordinate extraction.
            b_values = dynamic_dim_override
            if b_values is None:
                b_values = kwargs.get(
                    "b_values", data_to_reconstruct.attrs.get("b_values")
                )

            if b_values is None:
                # Attempt to extract from dynamic_dim coordinate if available
                dim_name = dynamic_dim_attr or "bvalue_idx"
                if dim_name in data_to_reconstruct.coords:
                    b_values = data_to_reconstruct.coords[dim_name]

            if b_values is None:
                raise ValueError(
                    "b-values must be provided via `dynamic_dim` or `b_values` kwarg, "
                    f"exist in dataset attrs, or be available as a coordinate for '{dim_name}'."
                )
            params = {
                "s0": data_to_reconstruct["s0"],
                "adc": data_to_reconstruct["adc"],
                "b_values": b_values,
            }

        elif model_name == "t1_inversion_recovery":
            # Prefer 'dynamic_dim' override, then 'inversion_times_ms' from kwargs,
            # then from attrs, then try coordinate extraction.
            full_inversion_times = dynamic_dim_override
            if full_inversion_times is None:
                full_inversion_times = kwargs.get(
                    "inversion_times_ms",
                    self._obj.attrs.get("inversion_times_ms"),
                )

            if full_inversion_times is None:
                dim_name = dynamic_dim_attr or "inversion_idx"
                if dim_name in self._obj.coords:
                    full_inversion_times = self._obj.coords[dim_name]

            if full_inversion_times is None:
                raise ValueError(
                    "Inversion times must be provided via `dynamic_dim` or `inversion_times_ms` kwarg, "
                    f"exist in dataset attrs, or be available as a coordinate for '{dim_name}'."
                )

            # If reconstructing a single voxel, select the correct inversion
            # times for that voxel
            if voxel_coords and hasattr(full_inversion_times, "dims"):
                # We only need to select along dimensions that exist in
                # full_inversion_times
                slice_selector = {
                    dim: voxel_coords[dim]
                    for dim in full_inversion_times.dims
                    if dim in voxel_coords
                }
                inversion_times_for_model = full_inversion_times.isel(slice_selector)
            else:
                # For full dataset reconstruction, use the full array or provided array
                inversion_times_for_model = full_inversion_times

            params = {
                "A": data_to_reconstruct["A"],
                "B": data_to_reconstruct["B"],
                "t1": data_to_reconstruct["t1"],
                "inversion_times_ms": inversion_times_for_model,
            }

        elif model_name == "patlak":
            time_points_s = dynamic_dim_override
            if time_points_s is None:
                time_points_s = kwargs.get(
                    "time_points_s", data_to_reconstruct.attrs.get("time_points_s")
                )
            aif_signal = kwargs.get(
                "aif_signal", data_to_reconstruct.attrs.get("aif_signal")
            )

            if time_points_s is None or aif_signal is None:
                raise ValueError(
                    "time_points_s (via `dynamic_dim` or `time_points_s`) and aif_signal must be provided via kwargs or exist in dataset attrs."
                )

            # Ensure they are numpy arrays for the model function
            time_points_s_np = (
                time_points_s.values
                if isinstance(time_points_s, xr.DataArray)
                else np.asarray(time_points_s)
            )
            aif_signal_np = (
                aif_signal.values
                if isinstance(aif_signal, xr.DataArray)
                else np.asarray(aif_signal)
            )

            params = {
                "K_trans": data_to_reconstruct["K_trans"],
                "Vp": data_to_reconstruct["Vp"],
                "time_points_s": time_points_s_np,
                "aif_signal": aif_signal_np,
            }

        elif model_name == "tofts_model":
            time_points_s = dynamic_dim_override
            if time_points_s is None:
                time_points_s = kwargs.get(
                    "time_points_s", data_to_reconstruct.attrs.get("time_points_s")
                )
            aif_signal = kwargs.get(
                "aif_signal", data_to_reconstruct.attrs.get("aif_signal")
            )

            if time_points_s is None or aif_signal is None:
                raise ValueError(
                    "time_points_s (via `dynamic_dim` or `time_points_s`) and aif_signal must be provided via kwargs or exist in dataset attrs for Tofts reconstruction."
                )

            # Ensure they are numpy arrays for the model function
            time_points_s_np = (
                time_points_s.values
                if isinstance(time_points_s, xr.DataArray)
                else np.asarray(time_points_s)
            )

            aif_times_np = None
            if (
                isinstance(aif_signal, xr.DataArray)
                and "aif_times" in aif_signal.coords
            ):
                aif_times_np = aif_signal.coords["aif_times"].values

            aif_signal_np = (
                aif_signal.values
                if isinstance(aif_signal, xr.DataArray)
                else np.asarray(aif_signal)
            )

            # Signal space parameters
            reco_space = str(kwargs.get("reco_space", "concentration")).lower()
            s0_val = None
            t10_val = None

            if reco_space == "signal":
                # Try to get s0 and t10 for this voxel
                if "s0_map" in data_to_reconstruct:
                    s0_val = float(data_to_reconstruct["s0_map"])
                elif "s0" in data_to_reconstruct:
                    s0_val = float(data_to_reconstruct["s0"])

                if "t1_map" in data_to_reconstruct:
                    t10_val = float(data_to_reconstruct["t1_map"])
                elif "t1" in data_to_reconstruct:
                    t10_val = float(data_to_reconstruct["t1"])

            params = {
                "time_points_s": time_points_s_np,
                "aif_signal": aif_signal_np,
                "aif_times": aif_times_np,
                "model_mode": kwargs.get(
                    "model_mode", data_to_reconstruct.attrs.get("model_mode", "both")
                ),
                "Ktrans_1c": data_to_reconstruct.get("Ktrans_1c"),
                "kep_1c": data_to_reconstruct.get("kep_1c"),
                "Ktrans_2c": data_to_reconstruct.get("Ktrans_2c"),
                "kep_2c": data_to_reconstruct.get("kep_2c"),
                "vp_2c": data_to_reconstruct.get("vp_2c"),
                "ve_2c": data_to_reconstruct.get("ve_2c"),
                "s0": s0_val,
                "t10_ms": t10_val,
                "tr_s": kwargs.get("tr_s", data_to_reconstruct.attrs.get("tr_s")),
                "fa_deg": kwargs.get("fa_deg", data_to_reconstruct.attrs.get("fa_deg")),
                "r1": kwargs.get("r1", data_to_reconstruct.attrs.get("r1", 3.5)),
                "reco_space": reco_space,
            }

        elif model_name == "two_compartment_model":
            time_points_s = dynamic_dim_override
            if time_points_s is None:
                time_points_s = data_to_reconstruct.attrs["time_points_s"].values

            time_step = (
                (time_points_s[1] - time_points_s[0]) if len(time_points_s) > 1 else 1.0
            )
            params = {
                "Fp": data_to_reconstruct["Fp"],
                "ve": data_to_reconstruct["ve"],
                "vb": data_to_reconstruct["vb"],
                "time_points_s": time_points_s,
                "aif_signal": data_to_reconstruct.attrs["aif_signal"].values,
                "time_step_s": time_step,
            }
        elif model_name == "FID":
            # Allow callers to override the sampling axis when reconstructing.
            sampling_time_axis = dynamic_dim_override
            if sampling_time_axis is None:
                sampling_time_axis = kwargs.get(
                    "sampling_time_axis_s",
                    kwargs.get(
                        "t", data_to_reconstruct.attrs.get("sampling_time_axis_s")
                    ),
                )

            if sampling_time_axis is None:
                raise ValueError(
                    "FID reconstruction requires `dynamic_dim` or `sampling_time_axis_s` via kwargs or dataset attrs"
                )

            # FID model parameters
            params = {
                "amplitude": data_to_reconstruct["fit_amps"],
                "freq": data_to_reconstruct["fit_freqs"],
                "T2_star": data_to_reconstruct["fit_t2s"],
                "t": np.asarray(sampling_time_axis),
                "db_mode": False,
            }
        elif model_name == "b1_sinusoidal":
            # Allow user to specify custom RF powers for reconstruction
            refpower_W = dynamic_dim_override
            if refpower_W is None:
                refpower_W = kwargs.get(
                    "refpower_W", data_to_reconstruct.attrs.get("refpower_W")
                )

            if refpower_W is None:
                raise ValueError(
                    "B1 reconstruction requires 'dynamic_dim' or 'refpower_W' via kwargs or dataset attrs"
                )

            # B1 model parameters
            params = {
                "refpower_W": np.asarray(refpower_W),
                "amplitude": data_to_reconstruct["amplitude"],
                "period": data_to_reconstruct["period"],
                "db_mode": kwargs.get("db_mode", False),
            }

        return reconstructor(**params)


# -- Mono-exponential ADC model ---
def monoexponential_adc_model(
    b_values: np.ndarray, s0: xr.DataArray, adc: xr.DataArray, db_mode: bool = False
) -> xr.DataArray:
    """
    Calculate the signal for a mono-exponential ADC model.
    S(b) = S0 * exp(-b * ADC)
    """
    if db_mode:
        print(f"Calculating mono-exponential ADC model:")
        print(f"  - b_values: {b_values}")
        print(f"  - s0: {s0}")
        print(f"  - adc: {adc}")

    b_da = xr.DataArray(b_values, dims=["bvalue_idx"], coords={"bvalue_idx": b_values})
    if db_mode:
        print(f"  - b_da: {b_da}")
    return s0 * np.exp(-b_da * adc)


# -- Two-compartment model from Zollner et al. ---
def two_compartment_model_deprecated(
    time_points_s, Fp, ve, vb, aif_signal, time_step_s
):
    """
    Two-compartment renal filtration model from Zollner et al. (DEPRECATED)

    .. deprecated:: 0.1
        Use the new Tofts or Extended Tofts models instead.

    This function operates in two modes based on the input types:

    1.  **Fitting Mode**: If Fp, ve, vb are scalar numbers, it computes the
        model for a single voxel and returns a 1D NumPy array for use
        with fitting routines like scipy.optimize.curve_fit.
    2.  **Reconstruction Mode**: If Fp, ve, vb are xarray.DataArrays, it
        efficiently computes the model for an entire image using vectorized
        FFT convolution and returns an xarray.DataArray.

    Parameters
    ----------
    time_points_s : numpy.ndarray
        1D array of time points in seconds.
    Fp : float or xarray.DataArray
        Plasma flow value(s).
    ve : float or xarray.DataArray
        Extravascular extracellular space volume fraction(s).
    vb : float or xarray.DataArray
        Blood volume fraction(s).
    aif_signal : numpy.ndarray or xarray.DataArray
        1D array representing the arterial input function over time.
    time_step_s : float
        The time step (in seconds) between points in the time series.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        The calculated tissue concentration curve(s).

    """
    from scipy.signal import fftconvolve

    is_fitting_mode = isinstance(Fp, (float, np.number))
    time_dim_name = "repetition_idx"
    time_da = xr.DataArray(
        time_points_s, dims=[time_dim_name], coords={time_dim_name: time_points_s}
    )

    aif_np = (
        aif_signal.values
        if isinstance(aif_signal, xr.DataArray)
        else np.asarray(aif_signal)
    )
    time_step = time_step_s.item() if hasattr(time_step_s, "item") else time_step_s

    # This will broadcast correctly for both scalar and DataArray inputs
    safe_ratio = xr.where(ve != 0, Fp / ve, 0)
    exp_term = np.exp(-safe_ratio * time_da)
    impulse_response = Fp * exp_term

    if is_fitting_mode:
        # For fitting, impulse_response is a 1D DataArray.
        convolution = (
            fftconvolve(aif_np, impulse_response.values, mode="full") * time_step
        )
        convolution_truncated = convolution[: len(time_points_s)]
        tissue_concentration = convolution_truncated + vb * aif_np
        return tissue_concentration  # Return numpy array for curve_fit
    else:
        # For reconstruction, impulse_response is multi-dimensional.
        # We use fftconvolve which is much faster than looping.
        time_axis_num = impulse_response.get_axis_num(time_dim_name)

        # Reshape aif_np to be broadcastable for convolution along the time
        # axis
        aif_shape_for_broadcast = [1] * impulse_response.ndim
        aif_shape_for_broadcast[time_axis_num] = -1
        aif_reshaped = aif_np.reshape(aif_shape_for_broadcast)

        convolution_np = (
            fftconvolve(
                impulse_response.values, aif_reshaped, mode="full", axes=time_axis_num
            )
            * time_step
        )

        # Truncate the result along the time axis
        slicer = [slice(None)] * convolution_np.ndim
        slicer[time_axis_num] = slice(0, len(time_points_s))
        convolution_truncated_np = convolution_np[tuple(slicer)]

        convolution_da = xr.DataArray(
            convolution_truncated_np,
            dims=impulse_response.dims,
            coords=impulse_response.coords,
        )

        aif_da = xr.DataArray(
            aif_np, dims=[time_dim_name], coords={time_dim_name: time_points_s}
        )
        tissue_concentration = convolution_da + vb * aif_da
        return tissue_concentration


# -- Patlak model ---
def patlak_model(
    time_points_s: np.ndarray,
    aif_signal: np.ndarray,
    K_trans: xr.DataArray,
    Vp: xr.DataArray,
) -> xr.DataArray:
    """
    Reconstructs the tissue concentration curve from Patlak model parameters.
    C_t(T) = K_i * integral_0^T(C_p(t)dt) + V_p * C_p(T)
    """
    time_dim_name = "repetition_idx"
    time_da = xr.DataArray(
        time_points_s, dims=[time_dim_name], coords={time_dim_name: time_points_s}
    )
    aif_da = xr.DataArray(
        aif_signal, dims=[time_dim_name], coords={time_dim_name: time_points_s}
    )

    integral_aif = cumulative_trapezoid(aif_signal, time_points_s, initial=0)
    integral_aif_da = xr.DataArray(
        integral_aif, dims=[time_dim_name], coords=time_da.coords
    )

    # Reconstruct the signal using broadcasting
    reconstructed_signal = K_trans * integral_aif_da + Vp * aif_da
    return reconstructed_signal


def tofts_model_reconstruction(
    time_points_s: np.ndarray,
    aif_signal: np.ndarray,
    aif_times: np.ndarray = None,
    model_mode: str = "both",
    Ktrans_1c: xr.DataArray = None,
    kep_1c: xr.DataArray = None,
    Ktrans_2c: xr.DataArray = None,
    kep_2c: xr.DataArray = None,
    vp_2c: xr.DataArray = None,
    ve_2c: xr.DataArray = None,
    # Optional parameters for signal-space reconstruction
    s0: float = None,
    t10_ms: float = None,
    tr_s: float = None,
    fa_deg: float = None,
    r1: float = 3.5,
    reco_space: str = "concentration",
) -> xr.DataArray:
    """
    Reconstructs the tissue concentration curve from Tofts model parameters.
    Supports both 1-compartment and 2-compartment (extended) Tofts models.
    Can also reconstruct in signal space if s0, t10_ms, tr_s, and fa_deg are provided.
    """
    from scipy.interpolate import interp1d

    if aif_times is not None:
        cp_func = interp1d(
            aif_times, aif_signal, kind="linear", fill_value="extrapolate"
        )
    else:
        cp_func = interp1d(
            time_points_s, aif_signal, kind="linear", fill_value="extrapolate"
        )

    results = []
    names = []

    # Helper to check if a DataArray has valid data (not all NaN)
    def is_valid(da):
        return da is not None and not np.all(np.isnan(da.values))

    def maybe_convert_to_signal(conc_curve):
        if str(reco_space).lower() == "signal" and all(
            v is not None for v in [s0, t10_ms, tr_s, fa_deg]
        ):
            return concentration_to_signal(
                conc_curve, s0, t10_ms / 1000.0, tr_s, fa_deg, r1
            )
        return conc_curve

    time_dim_name = "repetition_idx"

    if model_mode in ["1comp", "both"]:
        if is_valid(Ktrans_1c) and is_valid(kep_1c):
            # For single voxel (most common case for interactive plotting)
            if Ktrans_1c.size == 1:
                y1_conc = tofts_model_1comp(
                    time_points_s, float(Ktrans_1c), float(kep_1c), cp_func
                )
                y1 = maybe_convert_to_signal(y1_conc)
                results.append(
                    xr.DataArray(
                        y1,
                        dims=[time_dim_name],
                        coords={time_dim_name: time_points_s},
                        name="Tofts (1-comp)",
                    )
                )
                names.append("Tofts (1-comp)")

    if model_mode in ["2comp", "both"]:
        if is_valid(Ktrans_2c) and is_valid(kep_2c) and is_valid(vp_2c):
            if Ktrans_2c.size == 1:
                y2_conc = extended_tofts_model_2comp(
                    time_points_s,
                    float(Ktrans_2c),
                    float(kep_2c),
                    float(vp_2c),
                    cp_func,
                )
                y2 = maybe_convert_to_signal(y2_conc)
                results.append(
                    xr.DataArray(
                        y2,
                        dims=[time_dim_name],
                        coords={time_dim_name: time_points_s},
                        name="Extended Tofts (2-comp)",
                    )
                )
                names.append("Extended Tofts (2-comp)")

    if not results:
        # Return zeros if no valid models found
        return xr.DataArray(
            np.zeros_like(time_points_s),
            dims=[time_dim_name],
            coords={time_dim_name: time_points_s},
        )

    if len(results) == 1:
        return results[0]
    else:
        # Concatenate along a new model dimension to allow plotting both
        return xr.concat(results, dim=xr.DataArray(names, name="model_type"))


# -- Inverion Recovery model ---
def inversion_recovery_model(
    inversion_times_ms: np.ndarray,
    A: xr.DataArray,
    B: xr.DataArray,
    t1: xr.DataArray,
    db_mode: bool = False,
) -> xr.DataArray:
    """
    Reconstructs the signal for a T1 inversion recovery experiment.
    S(t) = A * (1 - 2 * exp(-t / T1)) + B
    """

    if db_mode:
        print(f"Calculating inversion recovery model:")
        print(f"  - inversion_times_ms: {inversion_times_ms}")
        print(f"  - A: {A}")
        print(f"  - B: {B}")
        print(f"  - t1: {t1}")

    return A * (1 - 2 * np.exp(-inversion_times_ms / t1)) + B


# -- B1 Sinusoidal model ---
def b1_sinusoidal_model(
    refpower_W: np.ndarray,
    amplitude: xr.DataArray,
    period: xr.DataArray,
    db_mode: bool = False,
) -> xr.DataArray:
    """
    Reconstruct B1 signal from fitted sinusoidal parameters.

    The B1 sinusoidal model relates signal to RF power through:
    S(sqrt(P)) = A * sin(π / period * sqrt(P))

    where P is the RF power in Watts, A is the complex amplitude,
    and period is the fitted period parameter.

    Parameters
    ----------
    refpower_W : np.ndarray
        1D array of RF power values in Watts to reconstruct signal for
    amplitude : xr.DataArray
        Complex amplitude parameter from B1 fitting (can be multi-dimensional)
    period : xr.DataArray
        Period parameter from B1 fitting (same dimensions as amplitude)
    db_mode : bool, optional
        If True, print debugging information. Default is False.

    Returns
    -------
    xr.DataArray
        Reconstructed signal with dimensions from amplitude/period plus 'rfpower_idx'

    Examples
    --------
    >>> refpower_W = np.linspace(0.001, 0.1, 20)
    >>> reconstructed = b1_sinusoidal_model(refpower_W, amp_map, period_map)
    """
    if db_mode:
        print(f"Calculating B1 sinusoidal model:")
        print(f"  - refpower_W: {refpower_W}")
        print(f"  - amplitude shape: {amplitude.shape}")
        print(f"  - period shape: {period.shape}")

    # Create xarray for RF power with sqrt transformation
    sqrt_refpow_da = xr.DataArray(
        np.sqrt(refpower_W), dims=["rfpower_idx"], coords={"rfpower_idx": refpower_W}
    )

    if db_mode:
        print(f"  - sqrt_refpow_da shape: {sqrt_refpow_da.shape}")

    # Calculate sinusoidal signal (broadcasts automatically)
    signal = amplitude * np.sin(np.pi / period * sqrt_refpow_da)

    if db_mode:
        print(f"  - output signal shape: {signal.shape}")

    return signal


# -- FID model ---
def fid_model(
    amplitude: xr.DataArray,
    freq: xr.DataArray,
    T2_star: xr.DataArray,
    t: np.ndarray,
    db_mode: bool = False,
) -> xr.DataArray:
    """
    Calculate the Free Induction Decay (FID) signal for given frequencies and T2* relaxation times.
    """

    if db_mode:
        print(f"\n--- Entering fid_model ---")
        print(f"  Input 'amplitude' dims: {amplitude.dims}, shape: {amplitude.shape}")
        print(f"  Input 'freq' dims: {freq.dims}, shape: {freq.shape}")
        print(f"  Input 'T2_star' dims: {T2_star.dims}, shape: {T2_star.shape}")
        print(f"  Input 't' (numpy array) shape: {t.shape}")

    # Convert 't' (numpy array) to an xarray.DataArray with 'acquisition_idx'
    # dimension.
    t_da = xr.DataArray(t, dims=["acquisition_idx"], coords={"acquisition_idx": t})
    if db_mode:
        print(f"  Converted 't_da' dims: {t_da.dims}, shape: {t_da.shape}")

    # Explicitly broadcast 't_da' to match the dimensions of 'amplitude'.
    # This ensures 't_aligned' has all necessary dimensions for correct
    # alignment.
    t_aligned = t_da.broadcast_like(amplitude)
    if db_mode:
        print(
            f"  't_aligned' (broadcasted 't') dims: {t_aligned.dims}, shape: {t_aligned.shape}"
        )

    # Perform the FID calculation.
    # All operands are now xarray DataArrays with explicit named dimensions.
    sig = (
        amplitude
        * np.exp(1j * 2.0 * np.pi * freq * t_aligned)
        * np.exp(-t_aligned / T2_star)
    )

    if db_mode:
        print(
            f"  Output 'sig' dims: {sig.dims}, shape: {sig.shape}, dtype: {sig.dtype}"
        )
        print(f"--- Exiting fid_model ---\n")
    return sig


# -- Signal to Concentration Conversion --
def signal_to_concentration(S_t, S_0, T10_s, TR_s, flip_angle_deg, r1):
    theta = np.deg2rad(flip_angle_deg)
    E10 = np.exp(-TR_s / T10_s)
    M0 = (S_0 * (1 - np.cos(theta) * E10)) / (np.sin(theta) * (1 - E10))
    A_t = (S_t - S_0) / (M0 * np.sin(theta))
    B = (1 - E10) / (1 - np.cos(theta) * E10)
    sum_ab = np.minimum(A_t + B, 0.98)  # Saturation protection
    arg_log = (1 - sum_ab) / (1 - np.cos(theta) * sum_ab)
    arg_log = np.where(arg_log <= 1e-5, 1e-4, arg_log)
    R1_t = -(1.0 / TR_s) * np.log(arg_log)
    R10 = 1.0 / T10_s
    return np.maximum((1.0 / r1) * (R1_t - R10), 0)


def concentration_to_signal(C_t, S_0, T10_s, TR_s, flip_angle_deg, r1):
    """
    Convert concentration back to signal using the spoiled gradient echo (SPGR) model.
    Inverse of signal_to_concentration.
    """
    theta = np.deg2rad(flip_angle_deg)
    R10 = 1.0 / T10_s
    R1_t = R10 + r1 * C_t
    E1_t = np.exp(-TR_s * R1_t)
    E10 = np.exp(-TR_s * R10)

    # Calculate M0 from baseline signal S_0 and T10
    M0 = (S_0 * (1 - np.cos(theta) * E10)) / (np.sin(theta) * (1 - E10))

    # Calculate signal S_t from concentration-dependent E1_t
    S_t = M0 * np.sin(theta) * (1 - E1_t) / (1 - np.cos(theta) * E1_t)
    return S_t


# Alias for backward compatibility with pickled objects
two_compartment_model = two_compartment_model_deprecated


# -- Tofts Model (1-Compartment) --
def tofts_model_1comp(t, Ktrans, kep, Cp_func):
    cp_vals = Cp_func(t)
    # The impulse response must start at t=0 for the convolution
    t_relative = t - t[0]
    impulse = np.exp(-kep * t_relative)
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    conv = np.convolve(cp_vals, impulse, mode="full")[: len(t)] * dt
    return Ktrans * conv


# -- Extended Tofts Model (2-Compartment) --
def extended_tofts_model_2comp(t, Ktrans, kep, vp, Cp_func):
    return vp * Cp_func(t) + tofts_model_1comp(t, Ktrans, kep, Cp_func)


def region_growing_b1_fitting(
    input_data: xr.DataArray,
    fit_params: dict,
    max_iterations: int = None,
    connectivity: int = 4,
):
    """
    Perform region-growing B1 fitting starting from a seed voxel.

    Parameters
    ----------
    input_data : xr.DataArray
        Complex-valued MRI data with spatial and RF power dimensions.
        fit_params : dict
        Dictionary containing fitting parameters.

        Required keys:

        - 'seed_voxel': tuple (read_idx, phase_idx) or dict with 'read_idx'/'phase_idx'
        - 'seed_fit_results': precomputed fit results for the seed voxel
        - 'refpower_W': array of RF power values

        Optional keys:

        - 'use_two_iteration_fit': bool, default True. If True, performs two-iteration
            fitting and returns both results.
        - 'use_neighbor_init_values': bool, default True. If True, uses neighbor-averaged
            parameters (period, amplitude, etc.) as initial guesses for each new voxel.
            If False, uses the same initial parameters for all voxels.
        - 'seed_first_iteration_result': dict, first iteration result for seed voxel
        - 'stop_index_extra': int, extra samples to include beyond 180° power
        - 'min_stop_index': int, minimum stop index
        - 'neighbor_period_search_width': float, search width for period refinement
        - 'region_growing_priority': str, queue priority mode ('fifo', 'snr', etc.)
        - 'plot_region_growing_progress': bool, show live progress visualization
        - 'return_debug_info': bool, default False. If True, returns debug information
            including stop_index_map, init_params_map, and region_mask.

        max_iterations : int, optional
        Maximum number of voxels to process. If None, processes all voxels.
    connectivity : int, default 4
        Neighbor connectivity (4 or 8).

    Returns
    -------
    fit_results : dict
        Dictionary mapping (read_idx, phase_idx) to final fit results.
    first_iteration_results : dict, optional
        Only returned if 'use_two_iteration_fit' is True and results are available.
        Dictionary mapping (read_idx, phase_idx) to first iteration fit results.
    debug_info : dict, optional
        Only returned if 'return_debug_info' is True. Contains:
        - 'stop_index_map': 2D numpy array of stop indices
        - 'init_params_map': dict mapping (read_idx, phase_idx) to init parameters
        - 'region_mask': 2D boolean array of processed voxels

    """
    index_iterator = fit_params.get("index_iterator", None)
    seed_voxel = fit_params.get("seed_voxel", None)
    seed_fit_result = fit_params.get("seed_fit_results", None)
    return_debug_info = fit_params.get("return_debug_info", False)

    if seed_voxel is None:
        raise ValueError("region_growing_b1_fitting requires a 'seed_voxel' entry.")

    if isinstance(seed_voxel, dict):
        seed_read_idx = seed_voxel.get("read_idx", None)
        seed_phase_idx = seed_voxel.get("phase_idx", None)
    else:
        try:
            seed_read_idx, seed_phase_idx = seed_voxel
        except (TypeError, ValueError):
            raise ValueError(
                "seed_voxel must be either a dict with 'read_idx'/'phase_idx' keys or a tuple of (read_idx, phase_idx)."
            )

    if seed_read_idx is None or seed_phase_idx is None:
        raise ValueError(
            "Seed voxel indices could not be determined. "
            "Ensure 'seed_voxel' provides both 'read_idx' and 'phase_idx'."
        )

    if seed_fit_result is None:
        raise ValueError(
            "region_growing_b1_fitting requires precomputed 'seed_fit_results' for the seed voxel."
        )

    import heapq
    from collections import deque

    # Extract 2D spatial data (squeeze other singleton dimensions)
    squeeze_dims = [
        d
        for d in input_data.dims
        if d not in ["read_idx", "phase_idx", "rfpower_idx"]
        and input_data.sizes[d] == 1
    ]
    debug_log_fit_maps(f"Squeezing dimensions for 2D extraction: {squeeze_dims}")
    if squeeze_dims:
        data_2d = input_data.squeeze(dim=squeeze_dims, drop=True)
    else:
        data_2d = input_data

    # Get dimensions
    read_size = data_2d.sizes["read_idx"]
    phase_size = data_2d.sizes["phase_idx"]

    region_mask = np.zeros((read_size, phase_size), dtype=bool)

    fit_results = {}
    fit_results[(seed_read_idx, seed_phase_idx)] = dict(seed_fit_result)
    region_mask[seed_read_idx, seed_phase_idx] = True

    # Store first iteration results if two-iteration fitting is enabled
    first_iteration_results = {}
    if fit_params.get("use_two_iteration_fit", True):
        # For seed voxel, we might not have first iteration stored
        seed_first_iter = fit_params.get("seed_first_iteration_result", None)
        if seed_first_iter is not None:
            first_iteration_results[(seed_read_idx, seed_phase_idx)] = dict(
                seed_first_iter
            )

    stop_index_map = np.full((read_size, phase_size), np.nan, dtype=float)
    stop_index_map[seed_read_idx, seed_phase_idx] = float(
        seed_fit_result.get("stop_index", np.nan)
    )

    # Initialize storage for init parameters if debug info is requested
    init_params_map = {} if return_debug_info else None

    sqrt_refpower = np.sqrt(
        np.clip(
            np.asarray(fit_params["refpower_W"], dtype=float), a_min=0.0, a_max=None
        )
    )

    # Define neighbor offsets
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    def get_neighbor_fit_params(read_idx, phase_idx, method="median"):
        # Aggregate parameters from already-fitted neighbors to guide the next fit
        neighbor_params = {
            "amplitude": [],
            "period": [],
            "refpower_90deg_W": [],
            "stop_index": [],
        }

        # Collect parameters from 3x3 neighborhood
        for dr in [-1, 0, 1]:
            for dp in [-1, 0, 1]:
                nbr_r = read_idx + dr
                nbr_p = phase_idx + dp
                debug_log_fit_maps(
                    f"  Checking neighbor at ({nbr_r}, {nbr_p})", write_to_file=False
                )

                # Check if neighbor is in bounds and has been fitted
                if (
                    0 <= nbr_r < read_size
                    and 0 <= nbr_p < phase_size
                    and (nbr_r, nbr_p) in fit_results
                ):

                    fit_res = fit_results[(nbr_r, nbr_p)]

                    # Only use valid fits
                    if not np.isnan(fit_res["period"]):
                        neighbor_params["amplitude"].append(
                            np.abs(fit_res["amplitude"])
                        )
                        debug_log_fit_maps(
                            f"    Neighbor amplitude: {neighbor_params['amplitude']}",
                            write_to_file=False,
                        )
                        neighbor_params["period"].append(fit_res["period"])
                        neighbor_params["refpower_90deg_W"].append(
                            fit_res["refpower_90deg_W"]
                        )
                        if "stop_index" in fit_res and not np.isnan(
                            fit_res["stop_index"]
                        ):
                            neighbor_params["stop_index"].append(fit_res["stop_index"])

        # Calculate means
        mean_params = {}
        for key, values in neighbor_params.items():
            debug_log_fit_maps(
                f"Neighbor values for {key}: {values}", write_to_file=False
            )
            if len(values) > 0:
                if method == "median":
                    mean_params[key] = np.nanmedian(values)
                else:
                    mean_params[key] = np.mean(values)
            else:
                mean_params[key] = np.nan

        mean_params["n_neighbors"] = sum(
            1 for val in neighbor_params["period"] if not np.isnan(val)
        )
        debug_log_fit_maps(
            f"Computed mean neighbor params: {mean_params}", write_to_file=False
        )

        return mean_params

    def build_fit_params_with_neighbors(base_params, neighbor_params):
        merged = dict(base_params)

        # Check if we should use neighbor information
        use_neighbor_init = base_params.get("use_neighbor_init_values", True)

        if not use_neighbor_init:
            # Don't merge neighbor parameters, just return base params
            return merged

        # Merge neighbor parameters into base params
        for key, value in neighbor_params.items():
            if key == "n_neighbors":
                continue
            if value is None:
                continue
            if isinstance(value, (float, int, np.floating, np.integer)):
                if np.isnan(value):
                    continue
            merged[key] = value
        return merged

    stop_index_extra_global = int(fit_params.get("stop_index_extra", 0))
    min_stop_index_global = int(fit_params.get("min_stop_index", 2))

    def fit_voxel_with_refinement(signal: np.ndarray = None, base_params: dict = None):
        """
        Run initial fit, derive refined stop index, and optionally refit to mimic seed procedure.

        Parameters
        ----------
        signal : np.ndarray
            Signal data for the voxel to fit.
        base_params : dict
            Base fitting parameters.

        Returns
        -------
        refined_result : dict
            Final fit results (either first or second iteration depending on settings).
        stop_index_val : float
            The stop index value used.
        first_iteration_result : dict or None
            Results from the first iteration if two-iteration fitting is enabled, None otherwise.
        """
        # Check if we should perform two-iteration fitting
        use_two_iterations = base_params.get("use_two_iteration_fit", True)

        first_params = dict(base_params)
        first_result = _fit_single_voxel_b1_period_sweep_helper(
            signal=signal,
            sqrt_refpower=sqrt_refpower,
            fit_params=first_params,
        )

        refined_result = dict(first_result)
        refined_result["_b1_period_grid"] = fit_params.get("_b1_period_grid")
        stop_index_val = float(first_result.get("stop_index", np.nan))

        # Store first iteration result if two-iteration fitting is enabled
        first_iteration_result = None
        if use_two_iterations:
            first_iteration_result = dict(first_result)

        period = first_result.get("period", np.nan)
        if np.isfinite(period) and use_two_iterations:
            # refpower_idx = int(np.searchsorted(sqrt_refpower, refpower))
            stop_index_extra = int(
                base_params.get("stop_index_extra", stop_index_extra_global)
            )
            min_stop_index = int(
                base_params.get("min_stop_index", min_stop_index_global)
            )

            # --- Fit only up to period (180) + stop_index_extra ---
            # Turn period (sqrt(W)) into stop index directly

            # This takes the closest index to the period value (lower end)
            period_idx = np.searchsorted(
                sqrt_refpower, period
            )  # + 1 # +1 to include period point
            # This takes the closest index to the period value (either side)
            # period_idx = np.argmin(np.abs(sqrt_refpower - period)) + 1 # +1 to include period point
            stop_index = period_idx + stop_index_extra

            if sqrt_refpower.size > 0:
                stop_index = min(stop_index, sqrt_refpower.size - 1)
            stop_index = max(stop_index, min_stop_index)

            refined_params = dict(base_params)
            refined_params["stop_index"] = stop_index

            refined_result = _fit_single_voxel_b1_period_sweep_helper(
                signal=signal,
                sqrt_refpower=sqrt_refpower,
                fit_params=refined_params,
            )
            refined_result["_b1_period_grid"] = fit_params.get("_b1_period_grid")
            refined_result["stop_index"] = float(stop_index)
            stop_index_val = float(stop_index)
        else:
            refined_result["stop_index"] = float(stop_index_val)

        return refined_result, stop_index_val, first_iteration_result

    pending_voxels = set()
    if index_iterator is not None:
        # Only keep voxels that correspond to spatial dimensions we operate on
        for indices in index_iterator:
            if isinstance(indices, tuple) and len(indices) >= 2:
                pending_voxels.add((indices[0], indices[1]))
            elif isinstance(indices, dict):
                r = indices.get("read_idx")
                p = indices.get("phase_idx")
                if r is not None and p is not None:
                    pending_voxels.add((r, p))
        if not pending_voxels:
            pending_voxels = None

    priority_mode = fit_params.get("region_growing_priority", "fifo")
    if isinstance(priority_mode, str):
        priority_mode = priority_mode.lower()
    else:
        priority_mode = "fifo"
    use_priority_queue = priority_mode in {"snr", "signal", "max_signal", "max_snr"}

    def compute_signal_priority(signal):
        values = np.abs(np.asarray(signal).ravel())
        finite = np.isfinite(values)
        if not finite.any():
            return float("-inf")
        return float(np.nanmean(values[finite]))

    if use_priority_queue:
        queue = []

        def queue_empty():
            return len(queue) == 0

        def enqueue_voxel(coord, signal=None):
            priority = 0.0
            if signal is not None:
                priority = compute_signal_priority(signal)
            else:
                signal_local = data_2d.isel(
                    read_idx=coord[0], phase_idx=coord[1]
                ).values
                priority = compute_signal_priority(signal_local)
            heapq.heappush(queue, (-priority, coord))

        def dequeue_voxel():
            _, coord = heapq.heappop(queue)
            return coord

    else:
        queue = deque()

        def queue_empty():
            return len(queue) == 0

        def enqueue_voxel(coord, signal=None):
            queue.append(coord)

        def dequeue_voxel():
            return queue.popleft()

    seed_signal = data_2d.isel(read_idx=seed_read_idx, phase_idx=seed_phase_idx).values
    enqueue_voxel((seed_read_idx, seed_phase_idx), seed_signal)
    iterations = 0
    plot_progress = fit_params.get("plot_region_growing_progress", False)
    fig = None
    region_im = None
    stop_index_im = None
    display_handle = None
    if plot_progress:
        try:
            import matplotlib.pyplot as plt
            from IPython.display import display

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            region_im = axes[0].imshow(region_mask, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("Region Mask")
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            stop_index_im = axes[1].imshow(stop_index_map, cmap="viridis")
            axes[1].set_title("Stop Index")
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            plt.tight_layout()
            display_handle = display(fig, display_id=True)
        except Exception:
            plot_progress = False

    while not queue_empty() and (max_iterations is None or iterations < max_iterations):
        curr_read, curr_phase = dequeue_voxel()
        debug_log_fit_maps(
            f"Processing voxel at read_idx={curr_read}, phase_idx={curr_phase}",
            write_to_file=False,
        )
        if pending_voxels and (curr_read, curr_phase) in pending_voxels:
            pending_voxels.discard((curr_read, curr_phase))

        for dr, dp in offsets:
            nbr_read = curr_read + dr
            nbr_phase = curr_phase + dp

            # Validate bounds and not already processed
            if (
                0 <= nbr_read < read_size
                and 0 <= nbr_phase < phase_size
                and not region_mask[nbr_read, nbr_phase]
            ):
                # debug_log_fit_maps(
                #     f"  Fitting neighbor voxel at read_idx={nbr_read}, phase_idx={nbr_phase}"
                # )

                # Derive initial parameters from existing neighbors
                neighbor_params = get_neighbor_fit_params(nbr_read, nbr_phase)
                # debug_log_fit_maps(f"neighbor_params={neighbor_params}")

                current_fit_params = build_fit_params_with_neighbors(
                    fit_params, neighbor_params
                )

                # Store init parameters if debug info is requested
                if return_debug_info:
                    init_params_map[(nbr_read, nbr_phase)] = {
                        "neighbor_params": dict(neighbor_params),
                        "merged_fit_params": {
                            k: v
                            for k, v in current_fit_params.items()
                            if k
                            in [
                                "amplitude",
                                "period",
                                "refpower_90deg_W",
                                "stop_index",
                                "use_neighbor_init_values",
                                "neighbor_period_search_width",
                            ]
                        },
                    }

                signal = data_2d.isel(read_idx=nbr_read, phase_idx=nbr_phase).values

                fit_res, stop_idx_val, first_iter_res = fit_voxel_with_refinement(
                    signal=signal, base_params=current_fit_params
                )

                fit_results[(nbr_read, nbr_phase)] = fit_res
                if first_iter_res is not None:
                    first_iteration_results[(nbr_read, nbr_phase)] = first_iter_res
                region_mask[nbr_read, nbr_phase] = True
                stop_index_map[nbr_read, nbr_phase] = stop_idx_val
                enqueue_voxel((nbr_read, nbr_phase), signal)
                if pending_voxels and (nbr_read, nbr_phase) in pending_voxels:
                    pending_voxels.discard((nbr_read, nbr_phase))
                if (
                    plot_progress
                    and region_im is not None
                    and stop_index_im is not None
                ):
                    region_im.set_data(region_mask)
                    stop_index_im.set_data(stop_index_map)
                    finite_stop = np.isfinite(stop_index_map)
                    if np.any(finite_stop):
                        vmin = float(np.nanmin(stop_index_map[finite_stop]))
                        vmax = float(np.nanmax(stop_index_map[finite_stop]))
                        if vmax > vmin:
                            stop_index_im.set_clim(vmin=vmin, vmax=vmax)
                    if display_handle is not None:
                        display_handle.update(fig)
                    else:
                        plt.pause(0.0001)

        iterations += 1

        if (
            (not queue_empty())
            and max_iterations is not None
            and iterations >= max_iterations
        ):
            debug_log_fit_maps(
                f"Reached max_iterations={max_iterations}; stopping region growing."
            )
            break

        # If the queue is empty but there are remaining voxels, start a new region
        if queue_empty():
            next_voxel = None
            if pending_voxels:
                try:
                    next_voxel = next(iter(pending_voxels))
                except StopIteration:
                    next_voxel = None
            else:
                # Fallback: scan entire grid for unvisited voxels
                for r in range(read_size):
                    for p in range(phase_size):
                        if not region_mask[r, p]:
                            next_voxel = (r, p)
                            break
                    if next_voxel:
                        break

            if next_voxel is not None:
                debug_log_fit_maps(
                    f"Queue empty; starting new region from voxel {next_voxel}."
                )
                nbr_read, nbr_phase = next_voxel
                neighbor_params = get_neighbor_fit_params(nbr_read, nbr_phase)
                current_fit_params = build_fit_params_with_neighbors(
                    fit_params, neighbor_params
                )

                # Store init parameters if debug info is requested
                if return_debug_info:
                    init_params_map[(nbr_read, nbr_phase)] = {
                        "neighbor_params": dict(neighbor_params),
                        "merged_fit_params": {
                            k: v
                            for k, v in current_fit_params.items()
                            if k
                            in [
                                "amplitude",
                                "period",
                                "refpower_90deg_W",
                                "stop_index",
                                "use_neighbor_init_values",
                                "neighbor_period_search_width",
                            ]
                        },
                    }

                signal = data_2d.isel(read_idx=nbr_read, phase_idx=nbr_phase).values
                fit_res, stop_idx_val, first_iter_res = fit_voxel_with_refinement(
                    signal=signal, base_params=current_fit_params
                )
                fit_results[(nbr_read, nbr_phase)] = fit_res
                if first_iter_res is not None:
                    first_iteration_results[(nbr_read, nbr_phase)] = first_iter_res
                region_mask[nbr_read, nbr_phase] = True
                stop_index_map[nbr_read, nbr_phase] = stop_idx_val
                enqueue_voxel((nbr_read, nbr_phase), signal)
                if pending_voxels and (nbr_read, nbr_phase) in pending_voxels:
                    pending_voxels.discard((nbr_read, nbr_phase))
                if (
                    plot_progress
                    and region_im is not None
                    and stop_index_im is not None
                ):
                    region_im.set_data(region_mask)
                    stop_index_im.set_data(stop_index_map)
                    finite_stop = np.isfinite(stop_index_map)
                    if np.any(finite_stop):
                        vmin = float(np.nanmin(stop_index_map[finite_stop]))
                        vmax = float(np.nanmax(stop_index_map[finite_stop]))
                        if vmax > vmin:
                            stop_index_im.set_clim(vmin=vmin, vmax=vmax)
                    if display_handle is not None:
                        display_handle.update(fig)
                    else:
                        plt.pause(0.01)

    # Prepare return values
    use_two_iteration_fit = fit_params.get("use_two_iteration_fit", True)

    if return_debug_info:
        debug_info = {
            "stop_index_map": stop_index_map,
            "init_params_map": init_params_map,
            "region_mask": region_mask,
        }

        if use_two_iteration_fit and first_iteration_results:
            return fit_results, first_iteration_results, debug_info
        else:
            return fit_results, debug_info
    else:
        # Original return behavior
        if use_two_iteration_fit and first_iteration_results:
            return fit_results, first_iteration_results
        else:
            return fit_results
