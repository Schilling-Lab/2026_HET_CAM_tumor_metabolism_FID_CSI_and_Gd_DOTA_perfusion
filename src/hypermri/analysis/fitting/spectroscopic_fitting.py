"""Spectroscopy fitting utilities and interactive helpers.

The module collects numerical helpers, widgets, and fitting algorithms used to
analyse spectroscopic Free Induction Decay (FID) signals. Functions expose a
NumPy-style API so they can be composed in notebooks as well as automated
pipelines.
"""

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from IPython.display import clear_output, display

if TYPE_CHECKING:
    from hypermri.core.experiment import MRIExperiment

from ...compatibility import assign_frequency_axis, resolve_frequency_axis
from ...utils.system.logging import debug_log, init_default_logger
from ..spectroscopy.core import (
    freq_Hz_to_ppm,
    freq_ppm_to_Hz,
    get_metab_cs_ppm,
    make_NDspec_6Dspec,
)

# Import fit_maps to register the hp accessor on xarray datasets
from . import fit_maps  # noqa: F401

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")

# This flag enables writing debug information to a file
DEBUG_FILE = "/tmp/spectroscopic_fitting_debug.log"
DEBUG_LOG = True


def debug_log_spectroscopic_fitting(message, enabled=DEBUG_LOG):
    """Log mask application debug messages to /tmp/mask_application_debug.log with timestamp."""
    debug_log(
        message=message,
        file_name=DEBUG_FILE,
        write_to_file=enabled,
    )


# -->pseudo-inv-fitting-->pseudo-inv-fitting-->pseudo-inv-fitting-->pseudo-inv-fitting-->pseudo-inv-fitting-->
def fid_basefunc(freq, T2_star, t):
    """
    Calculate the Free Induction Decay (FID) signal for given frequencies and T2* relaxation times.

    Parameters
    ----------
    freq : array_like
        Frequencies (in Hz) for which to calculate the FID signal.
    T2_star : array_like
        T2* relaxation times (in seconds) corresponding to each frequency in `freq`.
    t : array_like or scalar
        Time points (in seconds) at which the FID signal is calculated.

    Returns
    -------
    numpy.ndarray
        An array of complex values representing the FID signal at each time point for each
        frequency and T2* pair.

    Notes
    -----
    The FID signal is calculated using the formula:
    s(t) = exp(-1i * 2 * pi * freq * t) * exp(-t / T2_star)

    Examples
    --------
    >>> freq = [500, 600]
    >>> T2_star = [0.05, 0.03]
    >>> t = np.linspace(0, 0.1, 100)
    >>> signal = basefunc(freq, T2_star, t)
    """
    if not isinstance(freq, list) and not isinstance(freq, np.ndarray):
        freq = [freq]
    if not isinstance(T2_star, list) and not isinstance(T2_star, np.ndarray):
        T2_star = [T2_star]

    # avoid divide by 0 issues:
    if isinstance(T2_star, list) and 0 in T2_star:
        sig = np.zeros_like(
            [
                np.exp(1j * 2.0 * np.pi * f * t) * np.exp(-t / 1.0)
                for f, T2 in zip(freq, T2_star)
            ]
        )
    else:
        sig = np.array(
            [
                np.exp(1j * 2.0 * np.pi * f * t) * np.exp(-t / T2)
                for f, T2 in zip(freq, T2_star)
            ]
        )
    return sig


def fid_basefunc_jacobian(freq, T2_star):
    """
    Calculate the Jacobian of the basefunc with respect to its parameters.

    This function is a placeholder and currently returns True. It should be implemented
    to return the actual Jacobian matrix of the basefunc with respect to frequency and T2*.

    Parameters
    ----------
    freq : array_like
        Frequencies for which to calculate the Jacobian.
    T2_star : array_like
        T2* relaxation times for which to calculate the Jacobian.

    Returns
    -------
    bool
        Currently returns True. Should be implemented to return the Jacobian matrix.

    Notes
    -----
    This function needs to be properly implemented to return the Jacobian matrix
    for optimization algorithms that require it.
    """


def calc_X(x, intercept=1):
    """
    Define a parameter Matrix X for linear regression models.

    This function prepares the input x into a format suitable for linear regression,
    optionally including an intercept term.

    Parameters
    ----------
    x : array_like
        Input data to be transformed into the parameter matrix.
    intercept : int, optional
        If 1 (default), includes a column of ones for the intercept term.
        If 0, no intercept term is included.

    Returns
    -------
    numpy.ndarray
        The parameter matrix X, where each row corresponds to an observation
        and each column to a parameter (including the intercept if specified).

    Notes
    -----
    This function handles various input shapes and dimensions, making it versatile
    for different types of regression problems.
    """
    if x.shape[0] == 1:
        x = x.T
    # X contains the sampling points [[1, t0], [1, t1], ..., [1, tN]]
    if isinstance(x[0], np.ndarray):
        if len(x[0]) > 1:
            x_arr = []
            if intercept != 0:
                for k in range(x.shape[0]):
                    xi = x[k, :]
                    x_arr.append([intercept] + [x for x in xi])
                return np.array(x_arr)
            else:
                # default path for fitting spectra:
                for k in range(x.shape[0]):
                    xi = x[k, :]
                    x_arr.append([x for x in xi])
                return np.array(x_arr)

        if intercept != 0:
            return np.array([[intercept] + [xi[0]] for xi in x])
        else:
            return np.array([[xi[0]] for xi in x])
    else:
        if intercept != 0:
            return np.array([[intercept, xi] for xi in x])
        else:
            return np.array([[xi] for xi in x])


def calc_beta(X, y, x=None, intercept=1):
    """
    Calculate the beta coefficients for a linear regression model.

    This function solves the equation Y = X * beta using the pseudoinverse method.

    Parameters
    ----------
    X : array_like
        The design matrix of shape (n_samples, n_features).
    y : array_like
        The target values of shape (n_samples,).
    x : array_like, optional
        Raw input data. If provided and X is None, calc_X will be used to create X.
    intercept : int, optional
        Passed to calc_X if x is provided. Default is 1.

    Returns
    -------
    numpy.ndarray
        The calculated beta coefficients.

    Notes
    -----
    This function uses numpy's pseudoinverse (pinv) to solve for beta,
    which is suitable for both overdetermined and underdetermined systems.
    """
    if X is None and x is not None:
        X = calc_X(x, intercept=intercept)
    return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)


def eta(X=None, beta=[0, 1], x=None):
    """
    Calculate the predicted values for a linear model.

    Parameters
    ----------
    X : array_like, optional
        The design matrix. If None, it will be calculated from x using calc_X.
    beta : array_like, optional
        The coefficients of the linear model. Default is [0, 1].
    x : array_like, optional
        Raw input data to be used if X is None.

    Returns
    -------
    numpy.ndarray
        The predicted values based on X and beta.

    Notes
    -----
    This function calculates X.dot(beta) to produce the predicted values.
    It ensures X is a 2D array before performing the dot product.
    """
    if X is None:
        X = calc_X(x)
    if X.ndim == 1:
        X = X[:, np.newaxis]  # Ensure X is a column vector if 1D
    return X.dot(beta)


def S(y, X, b):
    """
    Calculate the residual sum of squares for a linear model.

    Parameters
    ----------
    y : array_like
        The true target values.
    X : array_like
        The design matrix.
    b : array_like
        The coefficients of the linear model.

    Returns
    -------
    float
        The residual sum of squares.

    Notes
    -----
    This function computes ||y - X.dot(b)||^2, which is the squared
    Euclidean 2-norm of the residual vector.
    """
    # define residual sum of squares:
    return np.linalg.norm(y - X.dot(b)) ** 2


def s_square(y, X, b, N, P):
    """Estimate the residual variance for an overdetermined model.

    Parameters
    ----------
    y : array_like
        Observed response vector.
    X : array_like
        Design matrix used to construct the linear model.
    b : array_like
        Model coefficients estimated from ``y`` and ``X``.
    N : int
        Number of observations in ``y``.
    P : int
        Number of fitted parameters contained in ``b``.

    Returns
    -------
    float
        Residual mean square value :math:`S/(N-P)`.

    Raises
    ------
    AssertionError
        If the system is not overdetermined (``N <= P``).
    """

    assert N > P
    return S(y, X, b) / (N - P)


def fishers_f(N, P, alpha):
    """Fishers'F distribution for the upper alpha quantile with P and N-P
    # degrees of freedom"""

    # example from Watts: fishers_f(N=28, P=2, alpha=0.05) = 3.37
    assert N > P
    from scipy import stats

    return stats.f.ppf(1 - alpha, P, N - P)


def students_t(N, P, alpha):
    """ "Students T distribution for alpha/2 and N-P degrees of freedom:"""
    assert N > P
    from scipy import stats

    return stats.t.ppf(1 - alpha / 2, N - P)


def standard_error(X, p, s=None, y=None, b=None, N=None, P=None):
    """Compute the standard error for parameter ``p`` in a linear model.

    Parameters
    ----------
    X : array_like
        Design matrix of shape ``(N, P)``.
    p : int
        Index of the parameter whose uncertainty should be evaluated.
    s : float, optional
        Pre-computed standard deviation of the residuals. When ``None`` the
        value is derived from ``y`` and ``b`` via :func:`s_square`.
    y : array_like, optional
        Observed response vector, required when ``s`` is not provided.
    b : array_like, optional
        Estimated model coefficients, required when ``s`` is not provided.
    N : int, optional
        Number of observations. Required when ``s`` is derived from ``y``.
    P : int, optional
        Number of estimated parameters. Required when ``s`` is derived from
        ``y``.

    Returns
    -------
    float
        Standard error associated with the parameter at index ``p``.
    """

    if s is None:
        s = np.sqrt(s_square(y=y, X=X, b=b, N=N, P=P))

    term2 = np.sqrt(np.linalg.inv(X.T.dot(X)).diagonal()[p])
    return s * term2


def mci(b, p, N, P, X, y, alpha):
    """
    marginal confidence intervall
    Parameters
    ----------
    b
    p
    N
    P
    X
    y
    alpha

    Returns
    -------

    """
    return standard_error(X=X, p=p, y=y, b=b, N=N, P=P) * students_t(
        N=N, P=P, alpha=alpha
    )


def cb(x, y, X, b, N, P, alpha):
    """
    Calculate the confidence bands

    Parameters
    ----------
    x: derivative matrix (N x P)
    y: measured signal (N)
    X: ...
    b: estimated values
    N: Number of observations (measured points)
    P: Number of parameters:
    alpha: confidence interval (100-alpha) %

    Returns
    -------

    """
    if x.shape[0] > 2:
        band = []
        for k in range(x.shape[0]):
            band.append(
                np.sqrt(s_square(y=y, X=X, b=b, N=N, P=P))
                * np.sqrt(x[k, :].T.dot(np.linalg.inv(X.T.dot(X))).dot(x[k, :]))
                * np.sqrt(P * fishers_f(N, P, alpha))
            )

        return np.array(band)
    else:
        # b1 + b2 +/ s * sqrt(
        return (
            np.sqrt(s_square(y=y, X=X, b=b, N=N, P=P))
            * np.sqrt(x.T.dot(np.linalg.inv(X.T.dot(X))).dot(x))
            * np.sqrt(P * fishers_f(N, P, alpha))
        )


def fid_func_variable_peaks(t, *params):
    """
    Calculate the free induction decay (FID) signal for a variable number of peaks.

    Parameters
    ----------
    t : numpy.ndarray
        An array of time points at which the FID signal is calculated.
    *params : float
        A sequence of parameters for each peak in the FID signal. The parameters for each peak
        are given in the order: frequency (w0_1, w0_2, ..., w0_N), decay time (T2_1, T2_2, ..., T2_N),
        and amplitude (amp_1, amp_2, ..., amp_N), where N is the number of peaks.

    Returns
    -------
    numpy.ndarray
        The calculated FID signal at each time point in `t`.

    Notes
    -----
    The function assumes a complex signal representation, where each peak in the FID signal
    is modeled as a decaying exponential modulated by a sinusoidal function of its frequency.
    """
    # Calculate number of peaks from parameters
    num_peaks = len(params) // 3
    amp = params[:num_peaks]
    w0 = params[num_peaks : 2 * num_peaks]
    t2 = params[2 * num_peaks :]

    # Initialize signal
    sig = 0.0 + 1j * 0.0
    # Accumulate signal from each peak
    for a, w, t22 in zip(amp, w0, t2):
        sig += a * np.exp(1j * 2.0 * np.pi * w * t) * np.exp(-t / t22)
    return sig


def jac_fid_func_variable_peaks_real(t, *params):
    """
    Calculate the Jacobian matrix of the real part of the FID signal
    for a variable number of peaks with respect to its parameters.

    Parameters
    ----------
    t : numpy.ndarray
        An array of time points at which the Jacobian is calculated.
    *params : float
        Parameters for the FID signal as expected by `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        The Jacobian matrix of the real part of the FID signal with respect
        to the parameters.

    Notes
    -----
    The Jacobian matrix is structured with rows corresponding to parameters
    (in the order of amplitudes, frequencies, and T2 values) and columns
    corresponding to time points in `t`.
    """
    num_peaks = len(params) // 3
    amp = params[:num_peaks]
    w0 = params[num_peaks : 2 * num_peaks]
    t2 = params[2 * num_peaks :]

    jac = np.empty((3 * num_peaks, t.size), dtype=complex)

    c = 0
    # Compute derivatives with respect to amplitudes
    for w, t22 in zip(w0, t2):
        jac[c, :] = np.exp(1j * 2.0 * np.pi * w * t) * np.exp(-t / t22)
        c += 1

    # Compute derivatives with respect to frequencies
    for a, w, t22 in zip(amp, w0, t2):
        jac[c, :] = (
            a
            * (1j * 2.0 * np.pi * t)
            * np.exp(1j * 2.0 * np.pi * w * t)
            * np.exp(-t / t22)
        )
        c += 1

    # Compute derivatives with respect to T2 values
    for a, w, t22 in zip(amp, w0, t2):
        jac[c, :] = (
            a * (-t / (t22**2)) * np.exp(1j * 2.0 * np.pi * w * t) * np.exp(-t / t22)
        )
        c += 1

    return np.real(jac)


def jac_fid_func_variable_peaks_imag(t, *params):
    """
    Computes the Jacobian of the imaginary part of the FID signal with respect to its parameters,
    for a variable number of peaks.

    Similar to `jac_fid_func_variable_peaks_real`, but focuses on the imaginary part of the signal.

    Parameters
    ----------
    t : numpy.ndarray
        A 1D numpy array of time points.
    *params : tuple
        Parameters of the FID signal as detailed for `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        The Jacobian matrix of the imaginary part of the FID signal, with respect to the parameters,
        evaluated at each time point in `t`.
    """

    num_peaks = len(params) // 3
    amp = params[:num_peaks]
    w0 = params[num_peaks : 2 * num_peaks]
    t2 = params[2 * num_peaks :]

    jac = np.empty((3 * num_peaks, t.size), dtype=complex)

    c = 0

    # amplitudes:
    for w, t22 in zip(w0, t2):
        jac[c, :] = np.exp(1j * 2.0 * np.pi * w * t) * np.exp(-t / t22)
        c += 1

    # fequencies:
    for a, w, t22 in zip(amp, w0, t2):
        # print(f"a={a}, w={w}, t2={t22}")
        jac[c, :] = (
            a
            * (1j * 2.0 * np.pi * t)
            * np.exp(1j * 2.0 * np.pi * w * t)
            * np.exp(-t / t22)
        )
        c += 1
    # T2s:
    for a, w, t22 in zip(amp, w0, t2):
        jac[c, :] = (
            a * (-t / (t22**2)) * np.exp(1j * 2.0 * np.pi * w * t) * np.exp(-t / t22)
        )
        c += 1
    return np.imag(jac)


def std_squared_real(input_data, t, *params):
    """
    Calculate the squared standard deviation of the real part of the FID signal from the observed signal.

    This function is useful for estimating the noise level in the real component of the signal,
    based on the difference between the observed signal and the model predicted by `fid_func_variable_peaks`.

    Parameters
    ----------
    input_data : numpy.ndarray
        The observed FID signal as a numpy array.
    t : numpy.ndarray
        A 1D numpy array of time points at which the signal and model are evaluated.
    *params : tuple
        Parameters of the FID signal as expected by `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        A numpy array representing the squared standard deviation of the real part of the FID signal.
    """

    def fid_func_variable_peaks_real_part(t, *params):
        return np.real(fid_func_variable_peaks(t, *params))

    n = len(params)
    m = len(t)
    if n + 1 >= m:
        raise ValueError(f"m={m} has to be bigger than n={n}+1")
    std = (
        1
        / (m - n + 1)
        * (np.real(input_data) - fid_func_variable_peaks_real_part(t, *params)).T
        @ (np.real(input_data) - fid_func_variable_peaks_real_part(t, *params))
    )
    std = std * np.eye(m)
    return std


def std_squared_imag(input_data, t, *params):
    """
    Calculate the squared standard deviation of the imaginary part of the FID signal from the observed signal.

    This function is useful for estimating the noise level in the imaginary component of the signal,
    based on the difference between the observed signal and the model predicted by `fid_func_variable_peaks`.

    Parameters
    ----------
    input_data : numpy.ndarray
        The observed FID signal as a numpy array.
    t : numpy.ndarray
        A 1D numpy array of time points at which the signal and model are evaluated.
    *params : tuple
        Parameters of the FID signal as expected by `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        A numpy array representing the squared standard deviation of the imaginary part of the FID signal.
    """

    def fid_func_variable_peaks_imag_part(t, *params):
        return np.imag(fid_func_variable_peaks(t, *params))

    n = len(params)
    m = len(t)
    if n + 1 >= m:
        raise ValueError(f"m={m} has to be bigger than n={n}+1")
    std = (
        1
        / (m - n + 1)
        * (np.imag(input_data) - fid_func_variable_peaks_imag_part(t, *params)).T
        @ (np.imag(input_data) - fid_func_variable_peaks_imag_part(t, *params))
    )
    std = std * np.eye(m)
    return std


def weights_real(input_data, t, *params):
    """
    Computes weighting coefficients for the real part of the FID signal based on its noise characteristics.

    These weights are inversely proportional to the variance of the noise in the real part of the signal,
    useful for weighted least squares fitting or analysis.

    Parameters
    ----------
    input_data : numpy.ndarray
        The observed FID signal.
    t : numpy.ndarray
        A 1D numpy array of time points.
    *params : tuple
        Parameters of the FID signal as expected by `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        A diagonal matrix of weights for the real part of the FID signal.
    """

    std = std_squared_real(input_data, t, *params)
    m = len(t)
    weights = np.zeros_like(std)
    # Take the inverse of the diagonal of std, which represents variance
    np.fill_diagonal(weights, 1 / np.diag(std))
    return weights


def weights_imag(input_data, t, *params):
    """
    Computes weighting coefficients for the imagninary part of the FID signal based on its noise characteristics.

    These weights are inversely proportional to the variance of the noise in the imagninary part of the signal,
    useful for weighted least squares fitting or analysis.

    Parameters
    ----------
    input_data : numpy.ndarray
        The observed FID signal.
    t : numpy.ndarray
        A 1D numpy array of time points.
    *params : tuple
        Parameters of the FID signal as expected by `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        A diagonal matrix of weights for the imagninary part of the FID input_data.
    """

    std = std_squared_imag(input_data, t, *params)

    weights = np.zeros_like(std)
    # Take the inverse of the diagonal of std, which represents variance
    np.fill_diagonal(weights, 1 / np.diag(std))
    return weights


def parameter_covariance_real(input_data, t, *params):
    """
    Estimates the covariance matrix of the parameters for the real part of the FID signal.

    This covariance matrix is useful for understanding the confidence intervals and correlations
    between parameters estimated from the real part of the FID signal.

    Parameters
    ----------
    input_data : numpy.ndarray
        The observed FID signal.
    t : numpy.ndarray
        A 1D numpy array of time points.
    *params : tuple
        Parameters of the FID signal as expected by `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        The covariance matrix of the parameters estimated from the real part of the FID signal.
    """

    jac = jac_fid_func_variable_peaks_real(t, *params)
    weights = weights_real(input_data, t, *params)
    # print(f"jac.shape={jac.shape}")
    # print(f"weights.shape={weights.shape}")

    # The multiplication should ensure that the weights are applied correctly
    temp = jac @ weights @ jac.T
    param_cov = np.linalg.inv(temp)
    return param_cov


def parameter_covariance_imag(input_data, t, *params):
    """
    Estimates the covariance matrix of the parameters for the imaginary part of the FID signal.

    This covariance matrix is useful for understanding the confidence intervals and correlations
    between parameters estimated from the imaginary part of the FID signal.

    Parameters
    ----------
    input_data : numpy.ndarray
        The observed FID signal.
    t : numpy.ndarray
        A 1D numpy array of time points.
    *params : tuple
        Parameters of the FID signal as expected by `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        The covariance matrix of the parameters estimated from the imaginary part of the FID signal.
    """

    jac = jac_fid_func_variable_peaks_imag(t, *params)
    weights = weights_imag(input_data, t, *params)
    # print(f"jac.shape={jac.shape}")
    # print(f"weights.shape={weights.shape}")

    # The multiplication should ensure that the weights are applied correctly
    temp = jac @ weights @ jac.T
    param_cov = np.linalg.inv(temp)
    return param_cov


def estimate_std_fit_results(
    popt, fit_params, input_data, verbose=False, full_output=False
):
    """
    Estimates the standard deviation of fit results for both real and imaginary parts of the FID signal.

    This function calculates the standard deviation of the fitted parameters, providing an insight into
    the precision of the fit.

    Parameters
    ----------
    popt : numpy.ndarray
        Optimized parameters from fitting the FID signal.
    fit_params : dict
        Dictionary containing fit parameters including the time axis (`sampling_time_axis_s`).
    input_data : numpy.ndarray
        The observed FID signal.
    verbose : bool, optional
        If True, print detailed information about the amplitude, frequency, and T2 estimates.
    full_output : bool, optional
        If True, return the detailed standard deviations of the fit results.

    Returns
    -------
    numpy.ndarray
        Standard deviation of the fitted parameters. If `full_output` is True, returns detailed standard deviations.
    """
    # define measured signal as FID:
    if fit_params["signal_domain"] == "spectral":
        input_data = np.fft.ifft(np.fft.ifftshift(input_data))
    else:
        pass

    try:
        param_cov = parameter_covariance_real(
            input_data, fit_params["sampling_time_axis_s"], *popt
        )
        cov = np.diag(param_cov)
        std_real = np.sqrt(cov)
    except BaseException:
        std_real = np.empty((len(popt), len(popt)))

    try:
        param_cov = parameter_covariance_imag(
            input_data, fit_params["sampling_time_axis_s"], *popt
        )
        cov = np.diag(param_cov)
        std_imag = np.sqrt(cov)
    except BaseException:
        std_imag = np.empty((len(popt), len(popt)))

    num_peaks = len(popt) // 3

    # Caclulate the mean of real and imaginary part
    # Suppress runtime warnings for overflow
    with np.errstate(over="ignore", invalid="ignore"):
        std = np.sqrt(std_real**2 + std_imag**2) / np.sqrt(2)

    # 4 = (complex) amplitude, freq, T2, (magnitude) amplitude
    std_res = np.zeros((4, num_peaks), dtype=complex)

    std = np.reshape(std, [-1, num_peaks])
    std_real = np.reshape(std_real, [-1, num_peaks])
    std_imag = np.reshape(std_imag, [-1, num_peaks])

    # complex amplitude:
    std_res[0, :] = std_real[0, :] + 1j * std_imag[0, :]
    # frequency
    std_res[1, :] = std[1, :]
    # T2
    std_res[2, :] = std[2, :]
    # Magnitude Amplitude:
    std_res[3, :] = std[0, :]

    popt_res = np.reshape(popt, [-1, num_peaks])

    std_res = np.transpose(std_res, (1, 0))
    popt_res = np.transpose(popt_res, (1, 0))

    if verbose:
        print("Amplitude Estimates:")
        for i in range(num_peaks):
            print(
                f"Amp({i + 1}): {np.abs(np.round(popt_res[i, 0], 2)):>5} +/- "
                f"{np.round(std_res[i, 3], 2)} a.u."
            )
            print(
                f"Amp({i + 1}): {np.round(popt_res[i, 0], 2):>5} +/- "
                f"{np.round(std_res[i, 0], 2)} a.u."
            )

        print("\nFrequency Estimates:")
        for i in range(num_peaks):
            print(
                f"freq({i + 1}): {np.real(np.round(popt_res[i, 1], 2)):>5} +/- "
                f"{np.round(np.real(std_res[i, 1]), 2)} Hz"
            )

        print("\nT2 Estimates:")
        for i in range(num_peaks):
            print(
                f"T2({i + 1}): {np.real(np.round(1000 * popt_res[i, 2], 2)):>5} +/- "
                f"{np.round(1000 * np.real(std_res[i, 2]), 2)} ms"
            )

    if full_output:
        return std_res
    else:
        return std_res


def fit_params_metabs_setup_tool(
    input_data=None, fit_params={}, data_obj: "MRIExperiment" = None, plot_params={}
):
    """
    Interactive viewer for 6D MRI spectroscopy input_data + helper to set up the
    parameters metabs_freqs_Hz, metabs_freqs_ppm, metabs_freqs_Hz by clicking and
    metabs_freq_ranges by dragging, using the right mouse button.

    This function creates an interactive plot for visualizing and analyzing
    6D MRI spectroscopy input_data. It allows navigation through different dimensions
    of the input_data, switching between frequency unit (Hz and ppm), and selecting
    different input_data types (real, imaginary, phase, magnitude).

    Parameters
    ----------
    input_data : ndarray
        6D array containing MRI spectroscopy input_data.
        Dimensions should be in the order:
        (spectroscopic, z, x, y, repetitions, channels)
    fit_params : dict
        Dictionary containing fitting parameters and metabolite information.
        Required keys:
        - 'metabs': list of str, metabolite names
        - 'metabs_freqs_Hz': list of float, metabolite frequencies in Hz
        - 'metabs_freqs_ppm': list of float, metabolite frequencies in ppm
        - 'freq_axis_hz': array-like, frequency range in Hz
        - 'freq_axis_ppm': array-like, frequency range in ppm
        - 'metabs_freq_ranges': list of float, frequency ranges in Hz for each metabolite
    data_obj : MRIExperiment
    plot_params = {}
        Dictionary containing plot parameters.
        Required keys:
        - 'figsize': tuple, figure size (width, height)


    Returns
    -------
    None
        This function doesn't return any value but creates an interactive plot.

    Notes
    -----
    The viewer includes the following interactive features:
    - Navigation through z-slices, repetitions, and channels
    - Switching between Hz and ppm for frequency axis
    - Selecting different metabolites (highlighted with original color, others grayed out)
    - Clicking on the x-y image to update the spectrum
    - Clicking on the spectrum to set metabolite frequencies (automatically updates both Hz and ppm)
    - Right-clicking and dragging to select a spectral range for each metabolite
    - Choosing between real, imaginary, phase, and magnitude input_data display
    - Adaptive scaling for different input_data types
    - Table display of all metabolite frequencies and ranges

    The plot consists of three main elements:
    1. X-Y View: Shows a 2D slice of the input_data
    2. Spectrum: Shows the spectrum at the selected x-y position
    3. Frequency Table: Displays frequencies and ranges for all metabolites
    """
    from ipywidgets import interact
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import RectangleSelector

    try:
        from matplotlib import colormaps

        get_cmap = colormaps.get_cmap  # Matplotlib 3.10+
    except ImportError:
        from matplotlib.cm import get_cmap  # Matplotlib <3.10

    if isinstance(input_data, xr.DataArray):
        new_order = (
            "acquisition_idx",
            "slice_idx",
            "read_idx",
            "phase_idx",
            "repetition_idx",
            "channel_idx",
        )

        # Use the .transpose() method to reorder the dimensions
        input_data = input_data.transpose(*new_order)
        fit_params["provided_dims"] = [0, 1, 2, 3, 4, 5]
    else:
        if np.ndim(input_data) != 6:
            provided_dims = fit_params.get("provided_dims", None)
            provided_dims_ind = fit_params.get("provided_dims_ind", None)
            if provided_dims is None and provided_dims_ind is None:
                provided_dims = np.array((np.ndim(input_data), 1))
                provided_dims[0] = 0  # spec
                if np.ndim(input_data) > 1:
                    provided_dims[1] = 4  # "reps"
                if np.ndim(input_data) > 2:
                    provided_dims[2] = 1  # "z"
                if np.ndim(input_data) > 3:
                    provided_dims[3] = 2  # "x"
                if np.ndim(input_data) > 4:
                    provided_dims[4] = 3  # "y"
                if np.ndim(input_data) > 5:
                    provided_dims[5] = 5  # "chans"
            fit_params["provided_dims"] = provided_dims
            input_data = make_NDspec_6Dspec(
                input_data=input_data, provided_dims=fit_params["provided_dims"]
            )
    signal_domain = fit_params.get("signal_domain", "spectral")
    fit_params["signal_domain"] = signal_domain
    if signal_domain == "spectral":
        pass
    else:
        input_data = np.fft.fftshift(np.fft.fft(input_data, axis=0), axes=0)
    input_data = np.rot90(
        input_data,
        k=1,
        axes=(2, 3),
    )

    current_z, current_rep, current_channel = (
        input_data.shape[1] // 2,
        input_data.shape[4] // 2,
        0,
    )
    current_x, current_y = input_data.shape[2] // 2, input_data.shape[3] // 2
    freq_axis_mode = "Hz"
    data_type = "Magnitude"
    figsize = plot_params.get("figsize", (15, 10))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    plt.subplots_adjust(bottom=0.25)

    # Prepare frequency axis
    freq_hz = fit_params["freq_axis_hz"]
    freq_ppm = fit_params["freq_axis_ppm"]
    metabs = fit_params.get("metabs", "peak1")
    metabs_freqs_Hz = fit_params.get("metabs_freqs_Hz", 0.0)
    fit_params["metabs_freqs_Hz"] = metabs_freqs_Hz
    metabs_freq_ranges = fit_params.get("metabs_freq_ranges", 30.0)
    fit_params["metabs_freq_ranges"] = metabs_freq_ranges
    fit_params["metabs"] = metabs
    # Generate a color map for metabolite lines
    color_map = get_cmap("tab10")
    metab_colors = [color_map(i) for i in np.linspace(0, 1, len(fit_params["metabs"]))]

    # Add vertical lines for metabolites with different colors and labels
    metab_lines = []
    metab_labels = []
    metab_ranges = []

    for i, (metab, freq, color, range_hz) in enumerate(
        zip(
            fit_params["metabs"],
            fit_params["metabs_freqs_Hz"],
            metab_colors,
            fit_params["metabs_freq_ranges"],
        )
    ):
        if freq is None:
            fit_params["metabs_freqs_Hz"][i] = 0.0

        line = ax2.axvline(
            freq, color="gray", linestyle="--", linewidth=0.5, visible=False
        )
        label = ax2.text(
            freq,
            0,
            metab,
            color="gray",
            rotation=90,
            va="bottom",
            ha="right",
            visible=False,
        )
        range_rect = ax2.add_patch(
            Rectangle(
                (freq - range_hz / 2, 0),
                range_hz,
                1,
                facecolor=color,
                alpha=0.2,
                visible=False,
            )
        )
        metab_lines.append(line)
        metab_labels.append(label)
        metab_ranges.append(range_rect)

    def get_display_data(d):
        if data_type == "Real":
            return np.real(d)
        elif data_type == "Imaginary":
            return np.imag(d)
        elif data_type == "Phase":
            return np.angle(d)
        else:  # Magnitude
            return np.abs(d)

    def update_plots(z, rep, channel):
        nonlocal current_z, current_rep, current_channel
        current_z, current_rep, current_channel = z, rep, channel

        display_data = get_display_data(input_data[:, z, :, :, rep, channel])

        # Update image
        im.set_data(np.mean(display_data, axis=0))
        vmin, vmax = np.percentile(np.mean(display_data, axis=0), [2, 98])
        im.set_clim(vmin, vmax)

        # Update spectrum
        spectrum_data = display_data[:, current_x, current_y]
        spectrum_line.set_ydata(spectrum_data)

        # Update y-axis limits for spectrum
        if data_type == "Phase":
            ax2.set_ylim(-np.pi, np.pi)
        else:
            ax2.set_ylim(np.min(spectrum_data), 1.1 * np.max(spectrum_data))

        update_metab_lines()
        update_freq_table()

    def on_click_xy(event):
        nonlocal current_x, current_y
        print(f"Clicked at x={event.xdata}, y={event.ydata}")
        if event.inaxes == ax1:
            current_x, current_y = int(event.ydata), int(event.xdata)  # Swap x and y
            spectrum_data = get_display_data(
                input_data[
                    :, current_z, current_x, current_y, current_rep, current_channel
                ]
            )
            spectrum_line.set_ydata(spectrum_data)
            ax2.set_title(
                f"Spectrum at X:{current_y}, Y:{current_x}"
            )  # Swap x and y in the title
            # print(f"current_y, current_x={current_y, current_x}")
            # try:

            #     point_highlight.set_data(current_y, current_x)  # Swap x and y
            # except ValueError:
            #     raise ValueError(
            #         f"current_y={current_y}, current_x={current_x}, input_data.shape={input_data.shape}"
            #     )

            # Update y-axis limits for spectrum
            if data_type == "Phase":
                ax2.set_ylim(-np.pi, np.pi)
            else:
                ax2.set_ylim(np.min(spectrum_data), 1.1 * np.max(spectrum_data))

            fig.canvas.draw_idle()

    def on_click_spectrum(event):
        if event.inaxes == ax2 and event.button == 1:  # Left click
            freq = event.xdata
            metab_str = metab_selector.value
            metab_idx = fit_params["metabs"].index(metab_str)
            if freq_axis_mode == "Hz":
                fit_params["metabs_freqs_Hz"][metab_idx] = freq
                fit_params["initial_metabs_freqs_Hz"][metab_idx] = freq
                fit_params["metabs_freqs_ppm"][metab_idx] = freq_Hz_to_ppm(
                    freq_Hz=[freq], hz_axis=freq_hz, ppm_axis=freq_ppm
                )[0]

                fit_params["freq_range_lower_bound_Hz"][metab_idx] = (
                    fit_params["initial_metabs_freqs_Hz"][metab_idx]
                    - fit_params["metabs_freq_ranges"][metab_idx] / 2
                )
                fit_params["freq_range_upper_bound_Hz"][metab_idx] = (
                    fit_params["initial_metabs_freqs_Hz"][metab_idx]
                    + fit_params["metabs_freq_ranges"][metab_idx] / 2
                )
            else:
                fit_params["metabs_freqs_ppm"][metab_idx] = freq
                fit_params["metabs_freqs_Hz"][metab_idx] = freq_ppm_to_Hz(
                    freq_ppm=[freq], ppm_axis=freq_ppm, hz_axis=freq_hz
                )[0]
                fit_params["initial_metabs_freqs_Hz"][metab_idx] = fit_params[
                    "metabs_freqs_Hz"
                ][metab_idx]

                fit_params["freq_range_lower_bound_Hz"][metab_idx] = (
                    fit_params["initial_metabs_freqs_Hz"][metab_idx]
                    - fit_params["metabs_freq_ranges"][metab_idx] / 2
                )
                fit_params["freq_range_upper_bound_Hz"][metab_idx] = (
                    fit_params["initial_metabs_freqs_Hz"][metab_idx]
                    + fit_params["metabs_freq_ranges"][metab_idx] / 2
                )

            update_metab_lines()
            update_freq_table()

    def on_select_range(eclick, erelease):
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        metab_str = metab_selector.value
        metab_idx = fit_params["metabs"].index(metab_str)
        new_range = abs(x2 - x1)
        if freq_axis_mode == "ppm":
            # Convert range from ppm to Hz
            new_range = (
                new_range
                * (fit_params["freq_axis_hz"][-1] - fit_params["freq_axis_hz"][0])
                / (fit_params["freq_axis_ppm"][-1] - fit_params["freq_axis_ppm"][0])
            )
        fit_params["metabs_freq_ranges"][metab_idx] = new_range
        fit_params["freq_range_lower_bound_Hz"][metab_idx] = (
            fit_params["initial_metabs_freqs_Hz"][metab_idx]
            - fit_params["metabs_freq_ranges"][metab_idx] / 2
        )
        fit_params["freq_range_upper_bound_Hz"][metab_idx] = (
            fit_params["initial_metabs_freqs_Hz"][metab_idx]
            + fit_params["metabs_freq_ranges"][metab_idx] / 2
        )
        update_metab_lines()
        update_freq_table()

        # Clear the RectangleSelector
        rs.set_active(False)
        rs.set_visible(False)
        fig.canvas.draw_idle()

        # Reactivate the RectangleSelector for future use
        rs.set_active(True)

    def update_metab_lines():
        y_positions = np.linspace(
            0.1, np.max(spectrum_line.get_data()[1]), len(metab_lines)
        )  # Distribute labels vertically

        sorted_indices = sorted(
            range(len(metab_lines)),
            key=lambda k: (
                fit_params["metabs_freqs_Hz"][k]
                if freq_axis_mode == "Hz"
                else fit_params["metabs_freqs_ppm"][k]
            ),
            reverse=(freq_axis_mode == "ppm"),
        )

        selected_metab = metab_selector.value
        for i, idx in enumerate(sorted_indices):
            freq_hz = fit_params["metabs_freqs_Hz"][idx]
            freq_ppm = fit_params["metabs_freqs_ppm"][idx]
            range_hz = fit_params["metabs_freq_ranges"][idx]

            if freq_axis_mode == "Hz":
                freq = freq_hz
                range_start = freq - range_hz / 2
                range_end = freq + range_hz / 2
            else:
                freq = freq_ppm
                range_ppm = (
                    range_hz
                    * (fit_params["freq_axis_ppm"][-1] - fit_params["freq_axis_ppm"][0])
                    / (fit_params["freq_axis_hz"][-1] - fit_params["freq_axis_hz"][0])
                )
                range_start = freq - range_ppm / 2
                range_end = freq + range_ppm / 2

            metab_lines[idx].set_xdata([freq])
            metab_lines[idx].set_visible(True)

            if fit_params["metabs"][idx] == selected_metab:
                metab_lines[idx].set_color(metab_colors[idx])
                metab_labels[idx].set_color(metab_colors[idx])
            else:
                metab_lines[idx].set_color("gray")
                metab_labels[idx].set_color("gray")

            label = metab_labels[idx]
            label.set_position((freq, y_positions[i]))
            label.set_visible(True)

            metab_ranges[idx].set_xy((range_start, 0))
            metab_ranges[idx].set_width(range_end - range_start)
            metab_ranges[idx].set_height(ax2.get_ylim()[1])
            metab_ranges[idx].set_visible(True)

        fig.canvas.draw_idle()

    def update_freq_table():
        ax3.clear()
        ax3.axis("off")
        table_data = []
        for i, metab in enumerate(fit_params["metabs"]):
            freq_hz = fit_params["metabs_freqs_Hz"][i]
            freq_ppm = fit_params["metabs_freqs_ppm"][i]
            range_hz = fit_params["metabs_freq_ranges"][i]
            table_data.append(
                [metab, f"{freq_hz:.2f}", f"{freq_ppm:.2f}", f"±{range_hz / 2:.2f}"]
            )

        table = ax3.table(
            cellText=table_data,
            colLabels=["Metabolite", "Freq (Hz)", "Freq (ppm)", "Range (±Hz)"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Highlight the selected metabolite
        selected_metab = metab_selector.value
        selected_idx = fit_params["metabs"].index(selected_metab)
        for i in range(len(table_data[0])):
            table[(selected_idx + 1, i)].set_facecolor(metab_colors[selected_idx])
            table[(selected_idx + 1, i)].set_text_props(weight="bold")

        ax3.set_title("Metabolite Frequencies and Ranges")
        fig.canvas.draw_idle()

    def switch_freq_axis(change):
        nonlocal freq_axis_mode
        freq_axis_mode = change.new
        if freq_axis_mode == "Hz":
            spectrum_line.set_xdata(freq_hz)
            ax2.set_xlim(freq_hz[0], freq_hz[-1])
            ax2.set_xlabel("Frequency (Hz)")
        else:
            spectrum_line.set_xdata(freq_ppm)
            ax2.set_xlim(freq_ppm[-1], freq_ppm[0])  # Flip x-axis for ppm
            ax2.set_xlabel("Frequency (ppm)")
        update_metab_lines()
        update_freq_table()

    def switch_data_type(change):
        nonlocal data_type
        data_type = change.new
        update_plots(current_z, current_rep, current_channel)

    def on_metab_select(change):
        update_metab_lines()
        update_freq_table()

    # Initial plot setup
    display_data = get_display_data(
        input_data[:, current_z, :, :, current_rep, current_channel]
    )
    im = ax1.imshow(np.mean(display_data, axis=0), cmap="viridis")
    # ax1.set_title("X-Y View")
    ax1.set_title(f"input_data.shape={input_data.shape}")
    (point_highlight,) = ax1.plot(
        current_x, current_y, "ro", markersize=10, fillstyle="none"
    )

    spectrum_data = display_data[:, current_x, current_y]
    (spectrum_line,) = ax2.plot(freq_hz, spectrum_data)
    ax2.set_title("Spectrum")
    ax2.set_xlabel("Frequency (Hz)")

    # Set initial color limits and y-axis limits
    vmin, vmax = np.percentile(display_data, [2, 98])
    im.set_clim(vmin, vmax)
    ymin, ymax = np.percentile(spectrum_data, [1, 99])
    y_range = ymax - ymin
    ax2.set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)

    fig.canvas.mpl_connect("button_press_event", on_click_xy)
    fig.canvas.mpl_connect("button_press_event", on_click_spectrum)

    # Set up RectangleSelector for range selection
    rs = RectangleSelector(
        ax2,
        on_select_range,
        useblit=True,
        button=[3],  # right click
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=False,
    )  # Changed to False

    # Ensure the RectangleSelector stays active
    ax2.set_navigate(True)
    ax2.set_navigate_mode(None)

    # Create widgets
    z_slider = widgets.IntSlider(
        min=0, max=input_data.shape[1] - 1, step=1, value=current_z, description="Z:"
    )
    rep_slider = widgets.IntSlider(
        min=0,
        max=input_data.shape[4] - 1,
        step=1,
        value=current_rep,
        description="Repetition:",
    )
    channel_slider = widgets.IntSlider(
        min=0,
        max=input_data.shape[5] - 1,
        step=1,
        value=current_channel,
        description="Channel:",
    )
    freq_axis_toggle = widgets.ToggleButtons(
        options=["Hz", "ppm"], value="Hz", description="Frequency Axis:"
    )
    metab_selector = widgets.Dropdown(
        options=fit_params["metabs"],
        value=fit_params["metabs"][0],
        description="Metabolite:",
    )
    data_type_selector = widgets.Dropdown(
        options=["Magnitude", "Real", "Imaginary", "Phase"],
        value="Magnitude",
        description="Data Type:",
    )

    freq_axis_toggle.observe(switch_freq_axis, names="value")
    data_type_selector.observe(switch_data_type, names="value")
    metab_selector.observe(on_metab_select, names="value")

    interact(update_plots, z=z_slider, rep=rep_slider, channel=channel_slider)

    display(widgets.HBox([freq_axis_toggle, metab_selector, data_type_selector]))

    # Initial update of metabolite lines and frequency table
    update_metab_lines()
    update_freq_table()

    # Keep a reference to rs to prevent garbage collection
    plt.gcf().rs = rs

    plt.tight_layout()


def interactive_setup_fit_params(
    data_obj: "MRIExperiment" = None, input_data=None, fit_params=None
):
    """
    Interactively set up fit parameters for data analysis.

    This function guides the user through a series of steps to set up fit parameters
    for data analysis. It includes dimension assignment with confirmation,
    signal domain selection with plot visualization and confirmation, cut-off
    selection, and metabolite selection.

    Parameters
    ----------
    data_obj : MRIExperiment, optional
        A data object (e.g., MRIExperiment) containing the input data.
    input_data : array-like, optional
        The input data array if not provided via data_obj.
    fit_params : dict, optional
        Initial fit parameters. If None, an empty dictionary is used.

    Returns
    -------
    dict
        Updated fit parameters based on user interactions.

    Notes
    -----
    This function uses IPython widgets for interactive elements and assumes
    it's being run in a Jupyter notebook environment. The function performs
    the following steps:

    1. Check for data availability
    2. Dimension assignment with confirmation
    3. Signal domain selection with plot visualization
    4. Cut-off selection for signal processing
    5. Metabolite selection and frequency assignment

    The function updates the fit_params dictionary throughout the process,
    including keys such as 'provided_dims', 'signal_domain', 'cut_off',
    'metabs', 'metabs_freqs_ppm', and 'metabs_freqs_Hz'.

    Examples
    --------
    >>> data_obj = MRIExperiment()  # Assuming MRIExperiment is a valid data object
    >>> fit_params = interactive_setup_fit_params(data_obj=data_obj)
    >>> print(fit_params)
    {
        'provided_dims': ['spectroscopic', 'z', 'x', 'y'],
        'signal_domain': 'temporal',
        'cut_off': 100,
        'metabs': ['pyruvate', 'lactate'],
        'metabs_freqs_ppm': [171.076, 183.35],
        ...
    }
    """

    if fit_params is None:
        fit_params = {}
    fit_params = def_fit_params(fit_params=fit_params, data_obj=data_obj)
    fit_params["nsamplepoints"] = len(input_data.coords["acquisition_idx"])

    def assign_metabolites():
        nonlocal fit_params
        print("-->assign_metabolites() was called")
        # Create widgets
        frequency_unit = widgets.RadioButtons(
            options=["PPM", "Hz"], description="Frequency Unit:", disabled=False
        )

        metabolite_box = widgets.VBox()
        custom_metab_box = widgets.VBox()
        custom_metabs_display = widgets.Output()
        confirm_button = widgets.Button(description="Confirm Metabolites")

        # Keep track of custom metabolites
        custom_metabolites = {}

        # Function to update metabolites based on frequency unit
        def update_metabolites(change):
            nonlocal metabolite_box
            larmor_frequency = 298.03  # MHz for 13C at 7T

            if change["new"] == "Hz":
                metabolites_cs = {
                    "pyruvate": 0,
                    "OMPD1": 1,
                    "OMPD5_1": 2,
                    "OMPD5_2": 3,
                    "OMPD5_3": 4,
                }
            else:  # PPM --> taken from utils_spectroscopy
                metabolites_cs = {
                    "pyruvate": 171.076,  # Lactate - 920Hz
                    "lactate": 183.35,  # determined from repeated experiments
                    "alanine": 176.5,  # https://www.pnas.org/doi/10.1073/pnas.0706235104
                    "pyruvatehydrate": 179.5,  # https://www.pnas.org/doi/10.1073/pnas.0706235104
                    "fumarate": 175.4,
                    "malate1": 181.7,  # https://pubs.acs.org/doi/epdf/10.1021/jacs.9b10094
                    "malate4": 180.5,  # https://pubs.acs.org/doi/epdf/10.1021/jacs.9b10094
                    # https://pubmed.ncbi.nlm.nih.gov/30639333/ (fits human in
                    # vivo data @ 3T)
                    "bicarbonate": 161.0,
                    "urea": 163.5,  # trust me bro
                    "co2": 124.5,  # https://www.pnas.org/doi/10.1073/pnas.0706235104
                    "aspartate1": 176.92,  # doi: 10.3389/fphys.2021.792769
                    "aspartate4": 180.20,  # doi: 10.3389/fphys.2021.792769
                }

            # Add custom metabolites to the list
            metabolites_cs.update(custom_metabolites)

            metabolite_checkboxes = [
                widgets.Checkbox(
                    description=f"{name}({value: .3f} {'Hz' if change['new'] == 'Hz' else 'ppm'})",
                    value=name
                    in custom_metabolites,  # Set custom metabolites as checked
                )
                for name, value in metabolites_cs.items()
            ]
            metabolite_box.children = metabolite_checkboxes

        # Connect the update function to the frequency unit widget
        frequency_unit.observe(update_metabolites, names="value")

        # Create input boxes for custom metabolites
        new_metab_name = widgets.Text(
            description="Name:", placeholder="Enter metabolite name"
        )
        new_metab_value = widgets.FloatText(
            description="Value:", placeholder="Enter value", value=-9999
        )
        add_button = widgets.Button(description="Add Metabolite")

        def add_custom_metabolite(b):
            name = new_metab_name.value
            value = new_metab_value.value
            if name and value is not None and value != -9999:
                custom_metabolites[name] = value
                update_metabolites({"new": frequency_unit.value})
                with custom_metabs_display:
                    print(f"Added: {name} at {value: .3f} {frequency_unit.value}")
                new_metab_name.value = ""
                new_metab_value.value = -9999

        add_button.on_click(add_custom_metabolite)

        custom_metab_box.children = [new_metab_name, new_metab_value, add_button]

        def on_confirm_button_clicked(b):
            nonlocal fit_params  # Add this line to access fit_params
            selected_metabolites = {}
            for checkbox in metabolite_box.children:
                if checkbox.value:
                    name, value_str = checkbox.description.split(" (")
                    # Extract only the numeric part
                    value = float(value_str.split()[0])
                    selected_metabolites[name] = value
            fit_params["metabs"] = [m for m in selected_metabolites.keys()]
            if frequency_unit.value == "PPM":
                fit_params["metabs_freqs_ppm"] = [
                    v for v in selected_metabolites.values()
                ]
                fit_params["metabs_freqs_Hz"] = None

            else:
                fit_params["metabs_freqs_ppm"] = None
                fit_params["metabs_freqs_Hz"] = [
                    v for v in selected_metabolites.values()
                ]

            # Update T2 values based on the selected metabolites
            if len(fit_params["max_t2_s"]) > len(fit_params["metabs"]):
                fit_params["max_t2_s"] = fit_params["max_t2_s"][
                    : len(fit_params["metabs"])
                ]
                fit_params["min_t2_s"] = fit_params["min_t2_s"][
                    : len(fit_params["metabs"])
                ]
                fit_params["metabs_t2s"] = fit_params["metabs_t2s"][
                    : len(fit_params["metabs"])
                ]
            else:
                pass

            fit_params = def_fit_params(fit_params=fit_params, data_obj=data_obj)
            print("Selected metabolites:")
            for metab, value in selected_metabolites.items():
                print(f"{metab}: {value:.3f} {frequency_unit.value}")
            confirm_button.disabled = True
            for checkbox in metabolite_box.children:
                checkbox.disabled = True

            # Disable the confirm button
            confirm_button.disabled = True
            plot_params = {}
            plot_params["figsize"] = (10, 5)

            fit_params_metabs_setup_tool(
                input_data=input_data,
                fit_params=fit_params,
                data_obj=data_obj,
                plot_params=plot_params,
            )

            # Create a new button for final confirmation
            final_confirm_button = widgets.Button(
                description="Confirm Final Parameters"
            )
            final_output = widgets.Output()

            def on_final_confirm_clicked(b):
                new_fit_params = def_fit_params(
                    fit_params=fit_params, data_obj=data_obj
                )
                fit_params.update(
                    new_fit_params
                )  # Merge changes into existing fit_params
                fit_params["test"] = "test"
                final_confirm_button.disabled = True

            final_confirm_button.on_click(on_final_confirm_clicked)

            # Display the updated parameters and the final confirm button
            with final_output:
                print(
                    "\nPlease review the parameters and click 'Confirm Final Parameters' to proceed."
                )

            display(final_output)
            display(final_confirm_button)

        confirm_button.on_click(on_confirm_button_clicked)

        # Display widgets
        display(frequency_unit)
        display(widgets.HBox([metabolite_box, custom_metab_box]))
        display(custom_metabs_display)
        display(confirm_button)

        # Trigger initial update
        update_metabolites({"new": "PPM"})

    # Execute the setup steps
    # assign_signal_domain()
    assign_metabolites()

    return fit_params


def def_fit_params_check_consistency(
    fit_params: dict = None, data_obj: "MRIExperiment" = None
):
    """
    Ensure that the entries in the `fit_params` dictionary are consistent.
    This function flattens any list or array values to a single level.

    Parameters
    ----------
    fit_params : dict, optional
        A dictionary containing the fitting parameters. Each key-value pair represents a parameter and its value.
        This function ensures that no parameter value is a nested list or array.
    data_obj : MRIExperiment, optional
        An object representing the spectroscopy scan. This object can provide additional context or default values
        for the fitting parameters, aiding in the consistency check and correction process.

    Returns
    -------
    dict
        A dictionary containing the fitting parameters after ensuring that all parameter values are consistent.

    Notes
    -----
    - The specific actions taken depend on the nature of the parameter and the context provided by `data_obj` if available.
    - This function is part of a larger preprocessing step to prepare fitting parameters for spectroscopy data analysis.

    """

    def flatten_array(arr):
        """
        Flatten nested lists or arrays into a single level.

        Parameters
        ----------
        arr : list or np.ndarray
            A potentially nested list or array to be flattened.

        Returns
        -------
        list or np.ndarray
            A flattened version of the input list or array.

        """
        if isinstance(arr, np.ndarray):
            # If it's a numpy array, use .flatten() to flatten it
            # return arr.flatten()
            return arr
        elif isinstance(arr, list):
            # If it's a list, flatten it using a list comprehension
            return (
                [item for sublist in arr for item in sublist]
                if any(isinstance(i, list) for i in arr)
                else arr
            )
        else:
            return arr

    keys_to_flatten = [
        "metabs_freqs_Hz",
        "metabs_freqs_ppm",
        "metabs_t2s",
        "max_t2_s",
        "min_t2_s",
    ]
    for k in keys_to_flatten:
        fit_params[k] = flatten_array((fit_params.get(k, [])))
    for k in keys_to_flatten:
        try:
            fit_params[k] = fit_params[k][: len(fit_params["metabs"])]
        except BaseException:
            pass
    return fit_params


def get_param_from_fit_params(param, voxel_idx, fit_params, input_data):
    """
    Extract voxel-specific or global fit parameter.

    Parameters
    ----------
    param : list or ndarray
        Parameter to extract (can be per-voxel or per-metabolite).
    voxel_idx : tuple
        The index of the voxel to extract the parameter for, if applicable.
    fit_params : dict
        Dictionary containing fit parameters, including the number of metabolites.
    input_data: ndarray
        size of the input_data to be fitted.

    Returns
    -------
    The extracted parameter, either voxel-specific or global.

    """
    if isinstance(voxel_idx, list):
        voxel_idx = tuple(voxel_idx)
    # Print statements for debugging
    n_metabs = len(fit_params["metabs"])  # Get number of metabolites
    # print(voxel_idx)
    # if isinstance(param, np.ndarray):
    # print(param.shape)
    if isinstance(param, np.ndarray):
        # Case 1: ND array with voxel-specific data (same shape as input_data)
        if param.shape == input_data.shape:
            return param[voxel_idx]
        # Case 2: ND array where last dimension corresponds to number of
        # metabolites
        elif param.shape[-1] == n_metabs:
            # Check if param has one more dimension than input_data
            if param.ndim == 7:  # 6 dimensions + 1 for metabolites
                # Select based on the voxel index, excluding the last dimension
                if len(voxel_idx) < 7:
                    voxel_idx_7d = voxel_idx + (0,)
                else:
                    voxel_idx_7d = voxel_idx
                # print(f"voxel_idx_7d={voxel_idx_7d}")
                # print(f"param.shape={param.shape}")
                return param[voxel_idx_7d[:-1]].squeeze()
            else:
                return param  # Return as-is if it doesn't have extra dimension

    elif isinstance(param, list):
        # Case 3: List with the same number of elements as the number of
        # metabolites
        if len(param) == n_metabs:
            return param
        else:
            # In case it's not per-metabolite, return the value for this voxel
            return param[voxel_idx]

    # Default: Return the parameter as-is if no conditions are met
    return param


def fit_func_pseudo_inv_error(X=None, y=None, beta=None, alpha=0.05):
    """
    Calculate the marginal confidence interval (mci) [(100-alpha) %].

    Parameters
    ----------
    X (np.ndarray):
        parameter matrix
    y (np.ndarray):
        measured input_data
    beta:
        estimates fit coefficients
    alpha:
        confidence interval

    Returns
    -------
    marginal confidence interval

    """
    # error estimation of beta:
    intercept = 0

    # number of observations:
    N = X.shape[0]

    # number of regressor variables:
    P = X.shape[1]

    # residual sum of squares:
    rSS = S(y=y, X=X, b=beta)

    # residual mean squares:
    rmS = s_square(y=y, X=X, b=beta, N=N, P=P)

    # Fishers'F
    FF = fishers_f(N=N, P=P, alpha=alpha)

    # standard error:
    sb1 = standard_error(X=X, p=0, s=np.sqrt(rmS))

    if intercept == 0:
        sb2 = 0
    else:
        sb2 = standard_error(X=X, p=1, s=np.sqrt(rmS))

    calc_mci = True
    if calc_mci:
        # marginal confidence interval:
        mcis = [mci(b=beta, p=p, N=N, P=P, X=X, y=y, alpha=alpha) for p in range(P)]

        if intercept == 0:
            mci2 = 0
        else:
            mci2 = mci(b=beta, p=1, N=N, P=P, X=X, y=y, alpha=alpha)

    calc_cb = False
    if calc_cb:
        if intercept == 0:
            cb1 = np.ones_like(X)
        else:
            x_temp = calc_X(x=np.linspace(np.min(X), np.max(X), X.shape[0]))
            cb1 = cb(x=x_temp, y=y, X=X, b=beta, N=N, P=P, alpha=alpha)
    return mcis


def solve_for_complex_amplitudes(
    input_data: np.ndarray = None,
    freqs: np.ndarray = None,
    metabs_t2s: np.ndarray = None,
    sampling_time_axis_s: np.ndarray = None,
    signal_domain: str = "spectral",
    db: bool = False,
    pseudo_inv: bool = True,
):
    """
    Solve for the complex amplitudes of FID signals using linear least-squares.

    Given a set of non-linear parameters (frequencies and T2* values), this
    function constructs a basis set of FID signals. It then solves the linear
    system `y = M * a` to find the complex amplitudes `a` that best fit the
    measured `input_data` (y) to the basis set (M).

    The solution is found using the pseudo-inverse method, which is robust and
    efficient for this linear least-squares problem.

    Parameters
    ----------
    input_data : np.ndarray
        The measured signal, either in the time or spectral domain.
    freqs : np.ndarray
        An array of frequencies (Hz) for each metabolite's basis signal.
    metabs_t2s : np.ndarray
        An array of T2* relaxation times (s) for each metabolite.
    sampling_time_axis_s : np.ndarray
        The time points (s) at which the signal was sampled.
    signal_domain : str, optional
        The domain of `input_data`. Must be 'spectral' or 'temporal'.
        If 'spectral', the data is converted to the time domain via iFFT.
        Defaults to 'spectral'.
    db : bool, optional
        If True, return intermediate arrays for debugging. Defaults to False.
    pseudo_inv : bool, optional
        If True (default), uses a direct pseudo-inverse method. If False, uses
        the normal equations. The direct method is generally preferred.

    Returns
    -------
    final_signal : np.ndarray
        The reconstructed time-domain signal, composed of the basis signals
        scaled by the fitted amplitudes.
    beta : np.ndarray
        The array of complex amplitudes found for each basis signal.

    Notes
    -----
    The model for the measured signal :math:`S(t)` is a linear combination of
    basis functions, where each basis function is a decaying complex exponential:

    .. math::
        S(t) = \\sum_{k=1}^{N} A_k \\cdot \\left[ \\exp(i \\omega_k t) \\exp\\left(-\\frac{t}{T_{2,k}^*}\\right) \\right]

    This can be expressed in matrix form as :math:`\\mathbf{y} = \\mathbf{M} \\mathbf{a}`, where:
    - :math:`\\mathbf{y}` is the vector of measured signal points.
    - :math:`\\mathbf{M}` is the matrix whose columns are the basis signals for each metabolite.
    - :math:`\\mathbf{a}` is the vector of unknown complex amplitudes.

    The amplitude vector :math:`\\mathbf{a}` is solved using the pseudo-inverse of :math:`\\mathbf{M}`:

    .. math::
        \\mathbf{a} = \\mathbf{M}^+ \\mathbf{y}

    """

    # generate signal:
    sim_signal = fid_basefunc(freq=freqs, T2_star=metabs_t2s, t=sampling_time_axis_s)

    # define measured signal as FID:
    if signal_domain == "spectral":
        meas_signal = np.fft.ifft(np.fft.ifftshift(input_data))
    else:
        meas_signal = input_data

    # "faster" version
    if pseudo_inv:
        # define a parameter/derivate Matrix X (for error estimation):
        # Compute the pseudo-inverse of m
        m_pseudo_inv = np.linalg.pinv(sim_signal.T)

        # Compute amplitudes 'beta' using the pseudo-inverse
        beta = m_pseudo_inv.dot(meas_signal)
    else:
        # See 1988, Bates, Watts

        # define a parameter/derivate Matrix X:
        X = calc_X(x=sim_signal.T, intercept=0)

        # Solve the Equation Y = X * beta --> beta  = (X.T X)^-1 X.T y
        beta = calc_beta(X=X, y=meas_signal)

    # Compute 'a' using the pseudo-inverse
    final_signal = sim_signal.T * beta

    # return intermediate arrays for debugging:
    if db:
        return final_signal, beta, sim_signal, m_pseudo_inv
    else:
        return final_signal, beta


# All below are adjusted to work with xarrays (.reco_data)


def _def_fit_params_calc_sampling(input_data: xr.DataArray, fit_params: dict) -> dict:
    """
    Calculate sampling and frequency-axis parameters from the input data.

    This internal helper function extracts essential time and frequency-domain
    information from the data's coordinates and attributes, which is
    foundational for the subsequent fitting steps.

    Parameters
    ----------
    input_data : xr.DataArray
        An xarray.DataArray containing the measurement data.
        It must have a coordinate for the spectral dimension.
    fit_params : dict
        A dictionary of user-provided fitting parameters.

    Returns
    -------
    dict
        The updated fit_params dictionary populated with calculated sampling
        and frequency information.

    """
    spectral_dim = fit_params.get("spectral_dim", "acquisition_idx")
    if spectral_dim not in input_data.coords:
        raise ValueError(
            f"Spectral coordinate '{spectral_dim}' not found in input_data."
        )

    # --- 1. Get user parameters or set sensible defaults ---
    cut_off = fit_params.get("cut_off", 0)
    b0 = fit_params.get("b0", 7)  # Field strength in Tesla
    nucleus = fit_params.get("nucleus", "13c")

    # --- 2. Determine time axis and sampling interval ---
    full_time_coords = input_data.coords[spectral_dim].values
    nsamplepoints_total = len(full_time_coords)

    # Ensure the cut-off value is within the valid range of indices.
    cut_off = max(0, min(cut_off, nsamplepoints_total - 1))

    # Apply the cut-off to define the final time axis used for fitting.
    sampling_time_axis_s = (full_time_coords - full_time_coords[0])[cut_off:]
    nsamplepoints = len(sampling_time_axis_s)

    # Calculate the sampling interval (dt). Fallback for single-point data.
    if nsamplepoints > 1:
        dwell_time_s = np.abs(sampling_time_axis_s[1] - sampling_time_axis_s[0])
    else:
        # If only one point, infer dt from the original unfiltered data.
        dwell_time_s = (
            np.abs(full_time_coords[1] - full_time_coords[0])
            if nsamplepoints_total > 1
            else 1.0
        )
    signal_domain = fit_params.get("signal_domain", None)
    if signal_domain is None:
        signal_domain = input_data.attrs.get("signal_domain", "temporal")

    # --- 3. Determine frequency axes (Hz and PPM) and bandwidth ---
    input_attrs = getattr(input_data, "attrs", None)

    # Get the frequency range in Hz from data attributes, or calculate it.
    freq_axis_hz = resolve_frequency_axis(input_attrs, "Hz")
    if freq_axis_hz is None:
        freq_axis_hz = np.fft.fftshift(np.fft.fftfreq(nsamplepoints, d=dwell_time_s))

    # This import is here to avoid circular dependencies.
    from ..spectroscopy.core import get_bw_from_freq_axis

    bw_Hz = get_bw_from_freq_axis(freq_axis=freq_axis_hz)

    # Get the PPM axis from data attributes.
    freq_axis_ppm = fit_params.get("freq_axis_ppm", None)
    if freq_axis_ppm is None:
        freq_axis_ppm = resolve_frequency_axis(input_attrs, "ppm")

    # --- 4. Update and return the dictionary ---
    fit_params.update(
        {
            "nsamplepoints": nsamplepoints,
            "dwell_time_s": dwell_time_s,
            "sampling_time_axis_s": sampling_time_axis_s,
            "freq_axis_hz": freq_axis_hz,
            "freq_axis_ppm": freq_axis_ppm,
            "bw_Hz": bw_Hz,
            "cut_off": cut_off,
            "b0": b0,
            "nucleus": nucleus,
            "signal_domain": signal_domain,
        }
    )
    return fit_params


def _def_fit_params_get_metabs(fit_params: dict) -> dict:
    """
    Define and harmonize metabolite frequency parameters.

    This function processes user-provided metabolite frequencies, calculates
    missing values (Hz from ppm or vice-versa), and checks for consistency if
    both are provided. If no frequency is given for a metabolite, it attempts
    to retrieve a default chemical shift from an internal library.

    Parameters
    ----------
    fit_params : dict
        A dictionary of fitting parameters. Must contain 'freq_axis_hz' and
        'freq_axis_ppm'. It can optionally contain 'metabs',
        'metabs_freqs_ppm', and 'metabs_freqs_Hz'.

    Returns
    -------
    dict
        The updated fit_params dictionary with verified and consistent
        metabolite frequency information.

    Raises
    ------
    ValueError
        If 'freq_axis_hz' or 'freq_axis_ppm' is not in fit_params.

    """
    # Default metabolites if not provided by the user.
    metabs = fit_params.get("metabs", ["pyruvate", "lactate", "alanine"])
    if isinstance(metabs, str):
        metabs = [metabs]
    num_metabs = len(metabs)

    # Get user-provided frequencies, default to lists of Nones
    user_freqs_ppm = fit_params.get("metabs_freqs_ppm")
    if user_freqs_ppm is None:
        user_freqs_ppm = [None] * num_metabs
    elif not isinstance(user_freqs_ppm, (list, np.ndarray, tuple)):
        user_freqs_ppm = [user_freqs_ppm]

    user_freqs_Hz = fit_params.get("metabs_freqs_Hz")
    if user_freqs_Hz is None:
        user_freqs_Hz = [None] * num_metabs
    elif not isinstance(user_freqs_Hz, (list, np.ndarray, tuple)):
        user_freqs_Hz = [user_freqs_Hz]

    # Ensure the user-provided lists have the correct length
    if len(user_freqs_ppm) != num_metabs:
        user_freqs_ppm = [None] * num_metabs
        warnings.warn("Length of 'metabs_freqs_ppm' does not match 'metabs'. Ignoring.")
    if len(user_freqs_Hz) != num_metabs:
        user_freqs_Hz = [None] * num_metabs
        warnings.warn("Length of 'metabs_freqs_Hz' does not match 'metabs'. Ignoring.")

    final_freqs_ppm = [0.0] * num_metabs
    final_freqs_Hz = [0.0] * num_metabs

    hz_axis = fit_params["freq_axis_hz"]
    ppm_axis = fit_params["freq_axis_ppm"]

    for i, metab_name in enumerate(metabs):
        ppm_val = user_freqs_ppm[i]
        hz_val = user_freqs_Hz[i]

        if ppm_val is not None and hz_val is not None:
            # Both are provided, check for consistency
            final_freqs_ppm[i] = ppm_val
            calculated_hz = freq_ppm_to_Hz(ppm_val, hz_axis, ppm_axis)

            if calculated_hz is None:
                warnings.warn(
                    f"Could not convert ppm {ppm_val} to Hz for {metab_name}. Using provided Hz value {hz_val} as fallback."
                )
                final_freqs_Hz[i] = hz_val
            else:
                if not np.isclose(calculated_hz, hz_val, atol=1e-2):
                    warnings.warn(
                        f"Warning for {metab_name}: Provided ppm({ppm_val}) and Hz({hz_val}) do not match (calculated Hz: {calculated_hz: .2f}). Using ppm value as primary."
                    )
                final_freqs_Hz[i] = calculated_hz

        elif ppm_val is not None:
            # Only ppm is provided
            final_freqs_ppm[i] = ppm_val
            calculated_hz = freq_ppm_to_Hz(ppm_val, hz_axis, ppm_axis)
            if calculated_hz is None:
                warnings.warn(
                    f"Could not convert ppm {ppm_val} to Hz for {metab_name}. Defaulting to 0.0 Hz."
                )
                final_freqs_Hz[i] = 0.0
                debug_log_spectroscopic_fitting(
                    message=f"Could not convert ppm {ppm_val} to Hz for {metab_name}.\nhz_axis={hz_axis}\nppm_axis={ppm_axis}",
                )

            else:
                final_freqs_Hz[i] = calculated_hz

        elif hz_val is not None:
            # Only Hz is provided
            final_freqs_Hz[i] = hz_val
            try:
                calculated_ppm = freq_Hz_to_ppm(hz_val, hz_axis, ppm_axis)
            except Exception as e:
                calculated_ppm = None
            if calculated_ppm is None:
                warnings.warn(
                    f"Could not convert Hz {hz_val} to ppm for {metab_name}. Defaulting to 0.0 ppm."
                )
                final_freqs_ppm[i] = 0.0
            else:
                final_freqs_ppm[i] = calculated_ppm

        else:
            # Neither is provided, get default
            print(f"No frequency defined for {metab_name}, getting default value.")
            default_ppm = get_metab_cs_ppm(metab_name)
            if default_ppm is not None:
                final_freqs_ppm[i] = default_ppm
                calculated_hz = freq_ppm_to_Hz(default_ppm, hz_axis, ppm_axis)
                if calculated_hz is None:
                    warnings.warn(
                        f"Could not convert default ppm {default_ppm} to Hz for {metab_name}. Defaulting to 0.0 Hz."
                    )
                    final_freqs_Hz[i] = 0.0
                else:
                    final_freqs_Hz[i] = calculated_hz
            else:
                warnings.warn(
                    f"⚠️ Could not find default chemical shift for '{metab_name}'. Please check spelling or provide frequency manually. Setting to 0.0."
                )
                final_freqs_ppm[i] = 0.0
                final_freqs_Hz[i] = 0.0

    fit_params.update(
        {
            "metabs": metabs,
            "metabs_freqs_ppm": final_freqs_ppm,
            "metabs_freqs_Hz": final_freqs_Hz,
        }
    )
    return fit_params


def _print_spectroscopic_params_info():
    """Print information about all available parameters for spectroscopic fitting."""
    print("=" * 80)
    print("Parameter Information for Spectroscopic Fitting (fit_data_pseudo_inv)")
    print("=" * 80)

    print("\nModel:")
    print("  Iterative pseudo-inverse fitting of Free Induction Decay (FID) signals")
    print(
        "  to extract metabolite concentrations, frequencies, and T2* relaxation times."
    )

    print(f"\n{'Required Parameters in fit_params:':-<80}")
    required_params = {
        "metabs": "List of metabolite names to fit (e.g., ['Pyruvate', 'Lactate', 'Alanine'])",
        "metabs_freqs_ppm": "List of metabolite frequencies in ppm (same length as metabs)",
    }
    for param, description in required_params.items():
        print(f"  • {param}")
        print(f"      {description}")

    print(f"\n{'Optional Parameters in fit_params:':-<80}")
    optional_params = {
        "b0": "Magnetic field strength in Tesla (e.g., 7.0 for 7T)",
        "bw_Hz": "Bandwidth in Hz. Default: calculated from dwell time",
        "cut_off": "Frequency cutoff for filtering. Default: None",
        "dwell_time_s": "Dwell time in seconds. Default: calculated from input_data",
        "fit_range_repetitions": "Range of repetitions to fit. Default: range(1)",
        "freq_axis_hz": "Frequency axis in Hz. Default: auto-calculated",
        "freq_axis_ppm": "Frequency axis in ppm. Default: auto-calculated",
        "initial_freqs_Hz": "Initial frequency guesses in Hz (per metabolite). Default: from metabs_freqs_ppm",
        "initial_t2s_s": "Initial T2* values in seconds (per metabolite). Default: 0.025",
        "max_t2_s": "Maximum T2* value in seconds. Default: 0.07",
        "metabs_freq_ranges": "Frequency search range in Hz (per metabolite). Default: 70.0",
        "metabs_t2_ranges": "T2* search range in seconds (per metabolite). Default: 0.04",
        "metabs_t2s": "Expected T2* values in seconds (per metabolite). Default: 0.025",
        "min_t2_s": "Minimum T2* value in seconds. Default: 0.005",
        "niter": "Number of iterative zoom refinements. Default: 1",
        "nucleus": "Nucleus being measured (e.g., '1H', '13C', '31P')",
        "npoints": "Number of grid points for parameter search. Default: 31",
        "nsamplepoints": "Number of sampling points. Default: from input_data",
        "pseudo_inv": "Use pseudo-inverse method. Default: True",
        "rep_fitting": "Repetition index for fitting. Default: 11",
        "sampling_time_axis_s": "Time axis in seconds. Default: auto-calculated",
        "show_tqdm": "Show progress bar. Default: True",
        "signal_domain": "Domain of the signal ('temporal' or 'spectral'). Default: 'temporal'",
        "skip_fitting": "Boolean DataArray to skip specific metabolites/voxels. Default: False everywhere",
        "snr_threshold": "Amplitude threshold for metabolite exclusion. Metabolites with |amplitude| below threshold are refitted without. Default: None",
        "spectral_dim": "Name of the spectral dimension. Default: 'acquisition_idx'",
        "zoomfactor": "Factor by which to zoom search grid per iteration. Default: 1.5",
    }
    for param, description in optional_params.items():
        print(f"  • {param}")
        print(f"      {description}")

    print(f"\n{'Output Parameters (in returned Dataset):':-<80}")
    output_params = {
        "fit_amps": "Fitted amplitudes for each metabolite",
        "fit_freqs": "Fitted frequencies in Hz for each metabolite",
        "fit_t2s": "Fitted T2* relaxation times in seconds for each metabolite",
        "fit_fwhm_Hz": "Full-width at half-maximum in Hz (derived from T2*)",
        "fit_r2_freq": "R-squared values from frequency fitting",
        "fit_r2_t2": "R-squared values from T2* fitting",
        "fit_low_snr_refitted": "Boolean flag indicating metabolites excluded due to low amplitude",
    }
    for param, description in output_params.items():
        print(f"  • {param}")
        print(f"      {description}")

    print("=" * 80)
    print("\nUsage Example:")
    print("  fit_params_ds = def_fit_params(")
    print("      input_data=my_data,")  # my_data is an xarray.DataArray
    print("      fit_params={")
    print("          'metabs': ['Pyruvate', 'Lactate'],")
    print("          'metabs_freqs_ppm': [171.0, 183.0],")
    print("      }")
    print("  )")
    print(
        "  result = fit_data_pseudo_inv(input_data=my_data, fit_params=fit_params_ds)"
    )
    print("=" * 80)


def def_fit_params(
    input_data: xr.DataArray = None,
    fit_params: dict = None,
    return_params_info: bool = False,
) -> xr.Dataset:
    """
    Prepare a Dataset of parameters for spectral fitting.

    This function is the primary entry point for setting up a fit. It organizes
    all parameters, storing global settings (like field strength or nucleus)
    as Dataset attributes and voxel-specific parameters (like initial frequency
    guesses or T2* ranges) as DataArrays. The input data is used as a template
    for the spatial dimensions of these parameter maps.

    Parameters
    ----------
    input_data : xr.DataArray, optional
        The measurement data, used as a template for creating the parameter maps.
        Can be None if return_params_info=True.
    fit_params : dict, optional
        A dictionary with user-defined parameter overrides. If None, default
        parameters are generated.
    return_params_info : bool, optional
        If True, prints information about all available parameters and returns None.
        When True, input_data can be None.

    Returns
    -------
    xr.Dataset or None
        A comprehensive Dataset containing all fitting parameters, ready for use
        with the fitting functions. Returns None if return_params_info=True.

    """
    if return_params_info:
        _print_spectroscopic_params_info()
        return None

    if input_data is None:
        raise ValueError("input_data is required when return_params_info=False")

    if fit_params is None:
        fit_params = {}

    # --- 1. Calculate base sampling and metabolite parameters ---
    fit_params = _def_fit_params_calc_sampling(input_data, fit_params)
    fit_params = _def_fit_params_get_metabs(fit_params)

    # --- 2. Define global parameters to be stored as Dataset attributes ---
    # These are parameters that are constant across all voxels for a given fit.
    attrs = {
        "b0": fit_params.get("b0"),
        "nucleus": fit_params.get("nucleus"),
        "bw_Hz": fit_params.get("bw_Hz"),
        "cut_off": fit_params.get("cut_off"),
        "fit_range_repetitions": fit_params.get("fit_range_repetitions", range(1)),
        "freq_axis_hz": fit_params.get("freq_axis_hz"),
        "freq_axis_ppm": fit_params.get("freq_axis_ppm"),
        "metabs": fit_params.get("metabs"),
        "metabs_freqs_ppm": fit_params.get("metabs_freqs_ppm"),
        "metabs_freqs_Hz": fit_params.get("metabs_freqs_Hz"),
        "metabs_t2s": fit_params.get("metabs_t2s"),
        "niter": fit_params.get("niter", 1),
        "npoints": fit_params.get("npoints", 31),
        "nsamplepoints": fit_params.get("nsamplepoints"),
        "pseudo_inv": fit_params.get("pseudo_inv", True),
        "rep_fitting": fit_params.get("rep_fitting", 11),
        "dwell_time_s": fit_params.get("dwell_time_s"),
        "show_tqdm": fit_params.get("show_tqdm", True),
        "signal_domain": fit_params.get("signal_domain", "temporal"),
        "snr_threshold": fit_params.get("snr_threshold", None),
        "spectral_dim": fit_params.get("spectral_dim", "acquisition_idx"),
        "sampling_time_axis_s": fit_params.get("sampling_time_axis_s"),
        "zoomfactor": fit_params.get("zoomfactor", 1.5),
    }

    # --- 3. Create voxel-specific parameter maps (DataArrays) ---
    # These parameters can vary per voxel, e.g., different initial T2* values.
    spectral_dim = attrs["spectral_dim"]
    metabolites = attrs["metabs"]
    # Create a template slice with the same spatial dimensions as the input
    # data.
    template_slice = input_data.isel({spectral_dim: 0}, drop=True)

    data_vars = {}

    # Define other parameters that will become DataArrays with metabolite
    # dimension.
    map_param_defaults = {
        "metabs_freq_ranges": 70.0,  # Search range for frequency in Hz
        "metabs_t2_ranges": 0.04,  # Search range for T2* in seconds
        "min_t2_s": 0.005,  # Lower bound for T2* fit
        "max_t2_s": 0.07,  # Upper bound for T2* fit
    }

    for key, default_val in map_param_defaults.items():
        user_val = fit_params.get(key, default_val)
        param_map_base = xr.full_like(template_slice, 0.0, dtype=np.float64)
        # Allow user to specify a list of values per metabolite or a single
        # value for all.
        metab_values = (
            user_val if isinstance(user_val, list) else [user_val] * len(metabolites)
        )
        metab_da = xr.DataArray(
            metab_values, dims=["metabolite"], coords={"metabolite": metabolites}
        )
        data_vars[key] = param_map_base + metab_da

    # Track whether individual metabolites should be skipped during fitting.
    skip_template = xr.full_like(data_vars["metabs_freq_ranges"], False, dtype=bool)
    if "skip_fitting" in fit_params and isinstance(
        fit_params["skip_fitting"], xr.DataArray
    ):
        data_vars["skip_fitting"] = fit_params["skip_fitting"].astype(bool)
    else:
        data_vars["skip_fitting"] = skip_template

    # --- 4. Initialize 'initial_t2s_s' map with robust handling ---
    # This allows for various input types: a single value, a list, or a
    # DataArray.
    initial_t2s_val = fit_params.get(
        "initial_t2s_s", fit_params.get("metabs_t2s", 0.025)
    )
    t2s_map_base = xr.full_like(template_slice, 0.0, dtype=np.float64)

    if isinstance(initial_t2s_val, xr.DataArray):
        data_vars["initial_t2s_s"] = initial_t2s_val
    else:
        # Handle scalar or list inputs, ensuring the final list matches the
        # number of metabolites.
        if isinstance(initial_t2s_val, list):
            metab_values = (initial_t2s_val + [0.025] * len(metabolites))[
                : len(metabolites)
            ]
            metab_values = [
                0.025 if np.isnan(v) else v for v in metab_values
            ]  # Replace any NaNs
        else:
            metab_values = [initial_t2s_val] * len(metabolites)

        metab_da_t2s = xr.DataArray(
            metab_values, dims=["metabolite"], coords={"metabolite": metabolites}
        )
        data_vars["initial_t2s_s"] = t2s_map_base + metab_da_t2s

    # --- 5. Initialize the metabolite-specific frequency map ---
    initial_freqs_map = xr.full_like(template_slice, 0.0, dtype=np.float64)

    # Ensure metabolites is a list
    if metabolites is None:
        metabolites = []

    # Ensure metabs_freqs_Hz has the correct length
    metab_freqs_values = fit_params.get("metabs_freqs_Hz")
    if metab_freqs_values is None:
        metab_freqs_values = []

    if len(metab_freqs_values) != len(metabolites):
        metab_freqs_values = [0.0] * len(metabolites)  # Fallback

    metab_da_freqs = xr.DataArray(
        metab_freqs_values,
        dims=["metabolite"],
        coords={"metabolite": metabolites},
    )
    data_vars["initial_freqs_Hz"] = initial_freqs_map + metab_da_freqs

    # --- 6. Assemble the Dataset ---
    fit_params_ds = xr.Dataset(data_vars, attrs=attrs)

    # --- 7. Calculate and add derived parameters used by the fitting algorithm ---
    # These are calculated from the primary parameters for convenience.
    fit_params_ds["freq_range_lower_bound_Hz"] = (
        fit_params_ds["initial_freqs_Hz"] - fit_params_ds["metabs_freq_ranges"] / 2
    )
    fit_params_ds["freq_range_upper_bound_Hz"] = (
        fit_params_ds["initial_freqs_Hz"] + fit_params_ds["metabs_freq_ranges"] / 2
    )
    # Final resolution of the frequency and T2* search grids after iterative
    # zooming.
    fit_params_ds["res_freqs_Hz"] = (
        fit_params_ds.metabs_freq_ranges
        / (attrs["zoomfactor"] ** attrs["niter"])
        / attrs["npoints"]
    )
    fit_params_ds["res_t2s_s"] = (
        fit_params_ds.metabs_t2_ranges
        / (attrs["zoomfactor"] ** attrs["niter"])
        / attrs["npoints"]
    )

    fit_params_ds.attrs["metabs_description"] = (
        "A complete set of parameters for spectral fitting."
    )

    fit_params_ds.attrs = attrs

    return fit_params_ds


def _fit_single_spectrum_pseudo_inv(
    data_slice: np.ndarray,
    initial_freqs_Hz: np.ndarray,
    initial_t2s_s: np.ndarray,
    metabs_freq_ranges: np.ndarray,
    metabs_t2_ranges: np.ndarray,
    min_t2_s: np.ndarray,
    max_t2_s: np.ndarray,
    freq_range_lower_bound_Hz: np.ndarray,
    freq_range_upper_bound_Hz: np.ndarray,
    skip_fitting_mask: np.ndarray | None = None,
    db=False,
    **kwargs,
) -> tuple:
    """
    Fit a single spectrum (voxel) using an iterative pseudo-inverse method.

    This function serves as a wrapper for `xr.apply_ufunc`, orchestrating the
    fitting process for a single voxel's data. It iteratively refines
    metabolite frequencies and T2* values by alternating between fitting each
    parameter while holding the other constant. The search range for each
    parameter is progressively narrowed in each iteration. To prevent fit
    degradation, the results of an iteration are only accepted if they
    improve the overall R-squared value compared to the previous best fit.

    Parameters
    ----------
    data_slice : np.ndarray
        The 1D temporal or spectral data for a single voxel.
    initial_freqs_Hz : np.ndarray
        Initial frequency guesses for each metabolite in this voxel.
    initial_t2s_s : np.ndarray
        Initial T2* guesses for each metabolite in this voxel.
    metabs_freq_ranges : np.ndarray
        Initial frequency search ranges (± Hz) for each metabolite.
    metabs_t2_ranges : np.ndarray
        Initial T2* search ranges (± s) for each metabolite.
    min_t2_s : np.ndarray
        Hard lower bounds for T2* values for each metabolite.
    max_t2_s : np.ndarray
        Hard upper bounds for T2* values for each metabolite.
    freq_range_lower_bound_Hz : np.ndarray
        Hard lower bounds for frequency search for each metabolite.
    freq_range_upper_bound_Hz : np.ndarray
        Hard upper bounds for frequency search for each metabolite.
    skip_fitting_mask : np.ndarray or None, optional
        Boolean array indicating which metabolites should be skipped during
        fitting (True = skip). If None, all metabolites are fitted.
    db : bool, optional
        If True, enables debug printing. Default is False.
    **kwargs : dict
        Global fitting parameters passed from `xr.apply_ufunc`. Expected keys
        include `metabs` (list of metabolite names), `sampling_time_axis_s`,
        `rep_fitting` (number of iterative repetitions), `zoomfactor` (factor
        to reduce search range per repetition), `niter`, `npoints`,
        `pseudo_inv`, and `signal_domain`.

    Returns
    -------
    tuple
        A tuple containing the final fitted parameters and diagnostics:
        - best_fit_amps (np.ndarray): Complex amplitudes from the best fit.
          Shape: (n_metabs,).
        - best_freq_Hz (np.ndarray): Frequencies (Hz) from the best fit.
          Shape: (n_metabs,).
        - best_t2_s (np.ndarray): T2* values (s) from the best fit.
          Shape: (n_metabs,).
        - combined_r2_freq (np.ndarray): R-squared data for frequency fits
          across all repetitions. Shape: (n_metabs, rep_fitting, niter,
          npoints, 2).
        - combined_r2_t2 (np.ndarray): R-squared data for T2* fits across
          all repetitions. Shape: (n_metabs, rep_fitting, niter, npoints, 2).

    Notes
    -----
    The function handles empty or all-zero `data_slice` by returning arrays
    filled with NaNs. It also includes an optimization for "frozen"
    parameters (where search ranges are zero), performing only a single
    amplitude calculation.

    The core logic involves a loop that runs for `rep_fitting` repetitions.
    In each repetition:
    1. Frequencies are fitted while T2* values are held constant.
    2. T2* values are fitted while frequencies are held constant.
    3. The search ranges for both parameters are reduced by `zoomfactor`.
    4. The parameters are only updated if the R-squared of the current fit
       is better than the best R-squared found so far.

    """

    if db:
        print("Fitting single spectrum with pseudo-inverse method...")
        print(f"Data slice shape: {data_slice.shape}")
        print(f"Initial frequencies (Hz): {initial_freqs_Hz}")
        print(f"Initial T2* values (s): {initial_t2s_s}")
        print(f"Frequency ranges (Hz): {metabs_freq_ranges}")
        print(f"T2* ranges (s): {metabs_t2_ranges}")
        print(f"Min T2* (s): {min_t2_s}")
        print(f"Max T2* (s): {max_t2_s}")
        print(f"Frequency lower bound (Hz): {freq_range_lower_bound_Hz}")
        print(f"Frequency upper bound (Hz): {freq_range_upper_bound_Hz}")
    # --- 1. Assemble the parameter dictionary for this specific voxel ---
    fit_params_voxel = kwargs.copy()
    fit_params_voxel.update(
        {
            "metabs_freqs_Hz": initial_freqs_Hz,
            "metabs_t2s": initial_t2s_s,
            "metabs_freq_ranges": metabs_freq_ranges,
            "metabs_t2_ranges": metabs_t2_ranges,
            "min_t2_s": min_t2_s,
            "max_t2_s": max_t2_s,
            "freq_range_lower_bound_Hz": freq_range_lower_bound_Hz,
            "freq_range_upper_bound_Hz": freq_range_upper_bound_Hz,
        }
    )

    metabs_all = list(kwargs.get("metabs", []))
    total_metabs = len(initial_freqs_Hz)
    if not metabs_all or len(metabs_all) != total_metabs:
        metabs_all = [f"metab_{i}" for i in range(total_metabs)]

    rep_fitting = fit_params_voxel.get("rep_fitting", 4)
    niter = fit_params_voxel.get("niter", 1)
    npoints = fit_params_voxel.get("npoints", 10)

    initial_freqs_full = np.asarray(initial_freqs_Hz, dtype=np.float64)
    initial_t2s_full = np.asarray(initial_t2s_s, dtype=np.float64)
    freq_ranges_full = np.asarray(metabs_freq_ranges, dtype=np.float64)
    t2_ranges_full = np.asarray(metabs_t2_ranges, dtype=np.float64)
    min_t2_full = np.asarray(min_t2_s, dtype=np.float64)
    max_t2_full = np.asarray(max_t2_s, dtype=np.float64)
    lower_bound_full = np.asarray(freq_range_lower_bound_Hz, dtype=np.float64)
    upper_bound_full = np.asarray(freq_range_upper_bound_Hz, dtype=np.float64)

    if skip_fitting_mask is None:
        skip_mask = np.zeros(total_metabs, dtype=bool)
    else:
        temp_mask = np.asarray(skip_fitting_mask, dtype=bool).reshape(-1)
        if temp_mask.size != total_metabs:
            skip_mask = np.resize(temp_mask, total_metabs)
        else:
            skip_mask = temp_mask

    active_idx = np.where(~skip_mask)[0]
    inactive_idx = np.where(skip_mask)[0]

    def _empty_result():
        empty_amps = np.zeros(total_metabs, dtype=np.complex128)
        freq_result = initial_freqs_full.copy()
        t2_result = initial_t2s_full.copy()
        r2_shape = (total_metabs, rep_fitting, niter, npoints, 2)
        empty_r2 = np.full(r2_shape, np.nan, dtype=np.float64)
        empty_snr_flag = np.zeros(total_metabs, dtype=bool)
        return (
            empty_amps,
            freq_result,
            t2_result,
            empty_r2,
            empty_r2.copy(),
            empty_snr_flag,
        )

    # --- 2. Handle empty voxels to avoid unnecessary computation ---
    if np.all(np.isnan(data_slice)) or np.all(data_slice == 0):
        return _empty_result()

    if active_idx.size == 0:
        return _empty_result()

    if inactive_idx.size > 0:
        initial_freqs_Hz = initial_freqs_full[active_idx]
        initial_t2s_s = initial_t2s_full[active_idx]
        metabs_freq_ranges = freq_ranges_full[active_idx]
        metabs_t2_ranges = t2_ranges_full[active_idx]
        min_t2_s = min_t2_full[active_idx]
        max_t2_s = max_t2_full[active_idx]
        freq_range_lower_bound_Hz = lower_bound_full[active_idx]
        freq_range_upper_bound_Hz = upper_bound_full[active_idx]
        metabs_active = [metabs_all[i] for i in active_idx]
    else:
        metabs_active = metabs_all

    fit_params_voxel["metabs"] = metabs_active
    fit_params_voxel["metabs_freqs_Hz"] = initial_freqs_Hz.copy()
    fit_params_voxel["metabs_t2s"] = initial_t2s_s.copy()
    fit_params_voxel["metabs_freq_ranges"] = metabs_freq_ranges.copy()
    fit_params_voxel["metabs_t2_ranges"] = metabs_t2_ranges.copy()
    fit_params_voxel["min_t2_s"] = min_t2_s.copy()
    fit_params_voxel["max_t2_s"] = max_t2_s.copy()
    fit_params_voxel["freq_range_lower_bound_Hz"] = freq_range_lower_bound_Hz.copy()
    fit_params_voxel["freq_range_upper_bound_Hz"] = freq_range_upper_bound_Hz.copy()

    # --- 3. Perform iterative fitting by calling the core functions ---
    # This loop alternates between fitting frequency and T2* for a set number
    # of repetitions.
    all_r2_freq_data = []
    all_r2_t2_data = []

    # Track best results across iterations to avoid worsening fits
    best_freq_r2 = np.inf
    best_freq_Hz = None
    best_t2_r2 = None
    best_t2_s = None
    best_fit_amps = None

    debug_log_spectroscopic_fitting(
        f"""Starting iterative fitting with {rep_fitting} repetitions.\n
Initial frequencies (Hz): {initial_freqs_Hz}\nInitial T2* values (s): {initial_t2s_s}
Frequency ranges (Hz): {metabs_freq_ranges}\nT2* ranges (s): {metabs_t2_ranges}
Min T2* (s): {min_t2_s}\nMax T2* (s): {max_t2_s}
Frequency lower bound (Hz): {freq_range_lower_bound_Hz}\nFrequency upper bound (Hz): {freq_range_upper_bound_Hz}
                                    """
    )

    # Check if parameters are frozen (per metabolite)
    freq_ranges_frozen = metabs_freq_ranges <= 1e-10  # Boolean array per metabolite
    t2_ranges_frozen = metabs_t2_ranges <= 1e-10  # Boolean array per metabolite

    # If ALL metabolites have both parameters frozen, perform minimal computation
    if np.all(freq_ranges_frozen) and np.all(t2_ranges_frozen):
        # Calculate amplitudes with current (frozen) parameters
        _, fit_amps = solve_for_complex_amplitudes(
            input_data=data_slice,
            freqs=initial_freqs_Hz,
            metabs_t2s=initial_t2s_s,
            sampling_time_axis_s=fit_params_voxel["sampling_time_axis_s"],
            pseudo_inv=fit_params_voxel.get("pseudo_inv", True),
            signal_domain=fit_params_voxel["signal_domain"],
        )

        # Create minimal R2 data arrays
        n_metabs = len(fit_params_voxel["metabs"])
        niter = fit_params_voxel.get("niter", 1)
        npoints = fit_params_voxel.get("npoints", 10)

        # Return results with initial values (no optimization performed)
        empty_r2_freq = np.zeros((n_metabs, 1, niter, npoints, 2))
        empty_r2_t2 = np.zeros((n_metabs, 1, niter, npoints, 2))

        if inactive_idx.size > 0:
            amps_full = np.zeros(total_metabs, dtype=np.complex128)
            amps_full[active_idx] = fit_amps
            freq_full = initial_freqs_full.copy()
            freq_full[active_idx] = initial_freqs_Hz
            t2_full = initial_t2s_full.copy()
            t2_full[active_idx] = initial_t2s_s
            empty_freq_full = np.full(
                (total_metabs, 1, niter, npoints, 2), np.nan, dtype=np.float64
            )
            empty_t2_full = np.full(
                (total_metabs, 1, niter, npoints, 2), np.nan, dtype=np.float64
            )
            empty_freq_full[active_idx, ...] = empty_r2_freq
            empty_t2_full[active_idx, ...] = empty_r2_t2
            empty_snr_flag_full = np.zeros(total_metabs, dtype=bool)
            return (
                amps_full,
                freq_full,
                t2_full,
                empty_freq_full,
                empty_t2_full,
                empty_snr_flag_full,
            )

        empty_snr_flag = np.zeros(n_metabs, dtype=bool)
        return (
            fit_amps,
            initial_freqs_Hz,
            initial_t2s_s,
            empty_r2_freq,
            empty_r2_t2,
            empty_snr_flag,
        )

    for rep_iter in range(rep_fitting):
        # debug_log_spectroscopic_fitting(
        #     message=f"data_slice={data_slice}\nfit_params_voxel={fit_params_voxel}",
        # )
        # Fit frequency, holding T2* constant
        _, freq_Hz, _, r2_freq = fit_freq_pseudo_inv(
            input_data=data_slice, fit_params=fit_params_voxel
        )

        # Calculate overall R2 for frequency fit (sum of minimum R2 values per metabolite)
        current_freq_r2 = np.sum(
            [
                np.min(r2_freq[i, -1, :, 1])
                for i in range(len(fit_params_voxel["metabs"]))
            ]
        )

        # Only update frequency parameters if R2 improves (or first iteration)
        if best_freq_r2 is None or current_freq_r2 < best_freq_r2:
            best_freq_r2 = current_freq_r2
            best_freq_Hz = freq_Hz.copy()
            fit_params_voxel["metabs_freqs_Hz"] = freq_Hz
        else:
            # Keep previous best frequency values
            fit_params_voxel["metabs_freqs_Hz"] = best_freq_Hz.copy()

        # Update the reduced frequency search ranges for the next iteration
        if rep_iter < rep_fitting:  # Don't update ranges on the last iteration
            zoomfactor = fit_params_voxel.get("zoomfactor", 1.5)
            fit_params_voxel["metabs_freq_ranges"] = (
                fit_params_voxel["metabs_freq_ranges"] / zoomfactor
            )
        all_r2_freq_data.append(r2_freq)

        # Fit T2*, holding frequency constant
        _, t2_s, fit_amps, r2_t2 = fit_t2_pseudo_inv(
            input_data=data_slice, fit_params=fit_params_voxel
        )

        # Calculate overall R2 for T2* fit (sum of minimum R2 values per metabolite)
        current_t2_r2 = np.sum(
            [np.min(r2_t2[i, -1, :, 1]) for i in range(len(fit_params_voxel["metabs"]))]
        )

        # Only update T2* parameters if R2 improves (or first iteration)
        if best_t2_r2 is None or current_t2_r2 < best_t2_r2:
            best_t2_r2 = current_t2_r2
            best_t2_s = t2_s.copy()
            best_fit_amps = fit_amps.copy()
            fit_params_voxel["metabs_t2s"] = t2_s
        else:
            # Keep previous best T2* values and use them for next iteration
            fit_params_voxel["metabs_t2s"] = best_t2_s.copy()

        debug_log_spectroscopic_fitting(
            f"Best frequencies (Hz) after iteration {rep_iter+1}: {best_freq_Hz}\nBest T2* (s): {best_t2_s}"
        )

        # Update the reduced T2* search ranges for the next iteration
        if rep_iter < rep_fitting:  # Don't update ranges on the last iteration
            fit_params_voxel["metabs_t2_ranges"] = (
                fit_params_voxel["metabs_t2_ranges"] / zoomfactor
            )
        all_r2_t2_data.append(r2_t2)

    # shape (n_metabs, rep_fitting, niter, npoints, 2)
    combined_r2_freq = np.stack(all_r2_freq_data, axis=1)
    combined_r2_t2 = np.stack(all_r2_t2_data, axis=1)

    # --- 4. Amplitude-based refitting (if threshold is provided) ---
    snr_threshold = fit_params_voxel.get("snr_threshold", None)
    low_snr_mask_active = np.zeros(len(metabs_active), dtype=bool)

    if snr_threshold is not None and snr_threshold > 0:
        # Check amplitude magnitude directly against threshold
        amp_magnitudes = np.abs(best_fit_amps)
        low_snr_mask_active = amp_magnitudes < snr_threshold

        debug_log_spectroscopic_fitting(
            f"Amplitude magnitudes: {amp_magnitudes}\nLow amplitude metabolites (indices): {np.where(low_snr_mask_active)[0]}"
        )

        # If any metabolites have low amplitude, refit without them
        if np.any(low_snr_mask_active):
            high_snr_idx = np.where(~low_snr_mask_active)[0]

            if high_snr_idx.size > 0:
                # Store original search ranges
                original_freq_ranges = metabs_freq_ranges.copy()
                original_t2_ranges = metabs_t2_ranges.copy()

                # Create new fit params with only high-SNR metabolites
                refit_params = fit_params_voxel.copy()
                refit_params["metabs"] = [
                    fit_params_voxel["metabs"][i] for i in high_snr_idx
                ]
                refit_params["metabs_freqs_Hz"] = best_freq_Hz[high_snr_idx].copy()
                refit_params["metabs_t2s"] = best_t2_s[high_snr_idx].copy()
                refit_params["metabs_freq_ranges"] = original_freq_ranges[
                    high_snr_idx
                ].copy()
                refit_params["metabs_t2_ranges"] = original_t2_ranges[
                    high_snr_idx
                ].copy()
                refit_params["min_t2_s"] = fit_params_voxel["min_t2_s"][
                    high_snr_idx
                ].copy()
                refit_params["max_t2_s"] = fit_params_voxel["max_t2_s"][
                    high_snr_idx
                ].copy()
                refit_params["freq_range_lower_bound_Hz"] = fit_params_voxel[
                    "freq_range_lower_bound_Hz"
                ][high_snr_idx].copy()
                refit_params["freq_range_upper_bound_Hz"] = fit_params_voxel[
                    "freq_range_upper_bound_Hz"
                ][high_snr_idx].copy()

                # Re-run fitting loop for high-SNR metabolites
                refit_r2_freq_data = []
                refit_r2_t2_data = []
                refit_best_freq_r2 = np.inf
                refit_best_freq_Hz = None
                refit_best_t2_r2 = None
                refit_best_t2_s = None
                refit_best_fit_amps = None

                for rep_iter in range(rep_fitting):
                    # Fit frequency
                    _, refit_freq_Hz, _, refit_r2_freq = fit_freq_pseudo_inv(
                        input_data=data_slice, fit_params=refit_params
                    )

                    refit_current_freq_r2 = np.sum(
                        [
                            np.min(refit_r2_freq[i, -1, :, 1])
                            for i in range(len(refit_params["metabs"]))
                        ]
                    )

                    if (
                        refit_best_freq_r2 is None
                        or refit_current_freq_r2 < refit_best_freq_r2
                    ):
                        refit_best_freq_r2 = refit_current_freq_r2
                        refit_best_freq_Hz = refit_freq_Hz.copy()
                        refit_params["metabs_freqs_Hz"] = refit_freq_Hz
                    else:
                        refit_params["metabs_freqs_Hz"] = refit_best_freq_Hz.copy()

                    if rep_iter < rep_fitting - 1:
                        zoomfactor = refit_params.get("zoomfactor", 1.5)
                        refit_params["metabs_freq_ranges"] = (
                            refit_params["metabs_freq_ranges"] / zoomfactor
                        )
                    refit_r2_freq_data.append(refit_r2_freq)

                    # Fit T2*
                    _, refit_t2_s, refit_fit_amps, refit_r2_t2 = fit_t2_pseudo_inv(
                        input_data=data_slice, fit_params=refit_params
                    )

                    refit_current_t2_r2 = np.sum(
                        [
                            np.min(refit_r2_t2[i, -1, :, 1])
                            for i in range(len(refit_params["metabs"]))
                        ]
                    )

                    if (
                        refit_best_t2_r2 is None
                        or refit_current_t2_r2 < refit_best_t2_r2
                    ):
                        refit_best_t2_r2 = refit_current_t2_r2
                        refit_best_t2_s = refit_t2_s.copy()
                        refit_best_fit_amps = refit_fit_amps.copy()
                        refit_params["metabs_t2s"] = refit_t2_s
                    else:
                        refit_params["metabs_t2s"] = refit_best_t2_s.copy()

                    if rep_iter < rep_fitting - 1:
                        refit_params["metabs_t2_ranges"] = (
                            refit_params["metabs_t2_ranges"] / zoomfactor
                        )
                    refit_r2_t2_data.append(refit_r2_t2)

                # Update results for high-SNR metabolites
                refit_combined_r2_freq = np.stack(refit_r2_freq_data, axis=1)
                refit_combined_r2_t2 = np.stack(refit_r2_t2_data, axis=1)

                best_fit_amps[high_snr_idx] = refit_best_fit_amps
                best_freq_Hz[high_snr_idx] = refit_best_freq_Hz
                best_t2_s[high_snr_idx] = refit_best_t2_s
                combined_r2_freq[high_snr_idx, ...] = refit_combined_r2_freq
                combined_r2_t2[high_snr_idx, ...] = refit_combined_r2_t2

            # Zero out low-amplitude metabolite amplitudes (outside the if block)
            best_fit_amps[low_snr_mask_active] = 0.0

    # Return the best results across all iterations
    if inactive_idx.size > 0:
        amps_full = np.zeros(total_metabs, dtype=np.complex128)
        amps_full[active_idx] = best_fit_amps
        freq_full = initial_freqs_full.copy()
        freq_full[active_idx] = best_freq_Hz
        t2_full = initial_t2s_full.copy()
        t2_full[active_idx] = best_t2_s
        r2_freq_full = np.full(
            (total_metabs, rep_fitting, niter, npoints, 2), np.nan, dtype=np.float64
        )
        r2_t2_full = np.full(
            (total_metabs, rep_fitting, niter, npoints, 2), np.nan, dtype=np.float64
        )
        r2_freq_full[active_idx, ...] = combined_r2_freq
        r2_t2_full[active_idx, ...] = combined_r2_t2

        # Map low_snr_mask_active back to full array
        low_snr_mask_full = np.zeros(total_metabs, dtype=bool)
        low_snr_mask_full[active_idx] = low_snr_mask_active

        return (
            amps_full,
            freq_full,
            t2_full,
            r2_freq_full,
            r2_t2_full,
            low_snr_mask_full,
        )

    return (
        best_fit_amps,
        best_freq_Hz,
        best_t2_s,
        combined_r2_freq,
        combined_r2_t2,
        low_snr_mask_active,
    )


def fit_data_pseudo_inv(
    input_data: xr.DataArray,
    fit_params: xr.Dataset,
) -> xr.Dataset:
    """Fits spectral data across all voxels using an iterative pseudo-inverse method.

    This is the main "workhorse" function that orchestrates the fit. It uses
    `xarray.apply_ufunc` to efficiently apply the fitting logic
    (`_fit_single_spectrum_pseudo_inv`) to every spectrum in the input data,
    leveraging Dask for parallelization.

    Parameters
    ----------
    input_data : xr.DataArray
        The input spectral data as an xarray.DataArray.
    fit_params : xr.Dataset
        The configuration Dataset, typically created by `def_fit_params`,
        containing all necessary parameters for the fit.

    Returns
    -------
    xr.Dataset
        An xarray.Dataset containing the fit results. This includes DataArrays for
        `fit_amps`, `fit_freqs`, `fit_t2s`, and the derived `fit_fwhm_Hz` for each
        metabolite across all voxels. It also contains the R-squared values
        from the fitting process (`fit_r2_freq`, `fit_r2_t2`), and a boolean flag
        `fit_low_snr_refitted` indicating which metabolites were excluded due to
        low amplitude during refitting (if snr_threshold was provided in fit_params).
    """
    # --- 1. Define dimensions and align all inputs for apply_ufunc ---
    global_params = fit_params.attrs
    spectral_dim = global_params.get("spectral_dim", "acquisition_idx")
    # All dimensions that are NOT the spectral dimension are loop dimensions.
    loop_dims = [dim for dim in input_data.dims if dim != spectral_dim]

    # Ensure all inputs have the same dimension order for apply_ufunc.
    aligned_input = input_data.transpose(*loop_dims, spectral_dim)

    if "skip_fitting" not in fit_params:
        skip_template = xr.full_like(fit_params["initial_freqs_Hz"], False, dtype=bool)
        fit_params = fit_params.assign(skip_fitting=skip_template)

    # List of voxel-specific parameter maps to pass to the fitting function.
    param_keys = [
        "initial_freqs_Hz",
        "initial_t2s_s",
        "metabs_freq_ranges",
        "metabs_t2_ranges",
        "min_t2_s",
        "max_t2_s",
        "freq_range_lower_bound_Hz",
        "freq_range_upper_bound_Hz",
        "skip_fitting",
    ]
    aligned_param_maps = [
        fit_params[key].transpose(*loop_dims, "metabolite") for key in param_keys
    ]

    # --- 2. Use apply_ufunc to apply the fitting function across all voxels ---
    fit_results_tuple = xr.apply_ufunc(
        _fit_single_spectrum_pseudo_inv,
        aligned_input,
        *aligned_param_maps,
        input_core_dims=[[spectral_dim]] + [["metabolite"]] * len(aligned_param_maps),
        # Corresponds to amps, freqs, t2s, r2_freq, r2_t2, low_snr_refitted
        output_core_dims=[
            ["metabolite"],  # fit_amps
            ["metabolite"],  # fit_freqs
            ["metabolite"],  # fit_t2s
            ["metabolite", "rep_fitting", "niter", "npoints", "r2_data"],  # r2_freq
            ["metabolite", "rep_fitting", "niter", "npoints", "r2_data"],  # r2_t2
            ["metabolite"],  # low_snr_refitted
        ],
        exclude_dims={spectral_dim},
        dask="parallelized",  # Enable parallel processing
        vectorize=True,
        output_dtypes=[
            np.complex128,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            bool,
        ],
        kwargs=global_params,  # Pass global settings here
    )

    # --- 3. Assemble the final results dataset ---
    fit_amps, fit_freqs, fit_t2s, fit_r2_freq, fit_r2_t2, low_snr_refitted = (
        fit_results_tuple
    )
    result_ds = xr.Dataset(
        {
            "fit_amps": fit_amps,
            "fit_freqs": fit_freqs,
            "fit_t2s": fit_t2s,
            "fit_r2_freq": fit_r2_freq,
            "fit_r2_t2": fit_r2_t2,
            "fit_low_snr_refitted": low_snr_refitted,
        }
    )

    # --- 4. Add attributes and derived variables ---
    result_ds.attrs.update(global_params)
    result_ds["fit_amps"].attrs["unit"] = "a.u."
    result_ds["fit_freqs"].attrs["unit"] = "Hz"
    result_ds["fit_t2s"].attrs["unit"] = "s"
    result_ds["fit_fwhm_Hz"] = 1 / (np.pi * result_ds["fit_t2s"])
    result_ds["fit_fwhm_Hz"].attrs["unit"] = "Hz"
    result_ds["fit_r2_freq"].attrs["unit"] = "R2"
    result_ds["fit_r2_freq"].attrs[
        "description"
    ] = "R2 (squared error) values for frequency fitting across iterations and search + points. Last dimension: [R2_value, frequency_Hz]"
    result_ds["fit_r2_t2"].attrs["unit"] = "R2"
    result_ds["fit_r2_freq"].attrs[
        "description"
    ] = "R2 (squared error) values for T2* fitting across iterations and search + points. Last dimension: [R2_value, T2*_s]"
    result_ds["fit_low_snr_refitted"].attrs[
        "description"
    ] = "Boolean flag indicating metabolites that were excluded and refitted due to low amplitude (|amplitude| < snr_threshold)"
    result_ds.attrs["model_name"] = "FID"
    fit_params_attrs = getattr(fit_params, "attrs", None)
    freq_axis_ppm = (
        resolve_frequency_axis(fit_params_attrs, "ppm") if fit_params_attrs else None
    )
    freq_axis_hz = (
        resolve_frequency_axis(fit_params_attrs, "Hz") if fit_params_attrs else None
    )
    assign_frequency_axis(result_ds.attrs, freq_axis_ppm, "ppm")
    assign_frequency_axis(result_ds.attrs, freq_axis_hz, "Hz")
    result_ds.attrs["signal_domain"] = fit_params.attrs.get("signal_domain", None)
    result_ds.attrs["domain"] = fit_params.attrs.get("domain", None)
    result_ds.attrs["spectral_dim"] = fit_params.attrs.get(
        "spectral_dim", "acquisition_idx"
    )
    result_ds.attrs["git_commit_hash"] = fit_params.attrs.get("git_commit_hash", None)

    return result_ds


def fit_freq_pseudo_inv(
    input_data: np.ndarray = None, fit_params: dict = None
) -> tuple:
    """
    Fit metabolite frequencies for a single spectrum via grid search.

    This function iteratively refines the frequency estimate for each
    metabolite by searching over a progressively smaller range. It is designed
    to be called by a wrapper function that prepares the parameters for a
    single voxel. The optimization separates metabolites with fixed ("frozen")
    frequencies from those being actively fitted to improve efficiency.

    Parameters
    ----------
    input_data : np.ndarray, optional
        The 1D data for a single spectrum. Default is None.
    fit_params : dict, optional
        A dictionary containing all necessary parameters for the fit, including
        both global settings and voxel-specific initial guesses and ranges.
        Default is None.

    Returns
    -------
    fitted_spectral : np.ndarray
        The final fitted signal in the spectral domain, composed of all
        metabolite contributions.
    current_freqs_Hz : np.ndarray
        An array of the final fitted frequencies in Hz for each metabolite.
    complex_amps : np.ndarray
        An array of the final complex amplitudes for each metabolite.
    all_r2_freq : np.ndarray
        A multi-dimensional array storing the R-squared (goodness of fit)
        values for each frequency point in the search grid across all
        iterations. The shape is (n_metabs, niter, npoints, 2).

    """
    # --- 1. Unpack necessary parameters from the dictionary ---
    metabs = fit_params["metabs"]
    npoints = fit_params["npoints"]
    niter = fit_params["niter"]
    signal_domain = fit_params["signal_domain"]
    sampling_time_axis_s = fit_params["sampling_time_axis_s"]
    nsamplepoints = fit_params["nsamplepoints"]
    cut_off = fit_params["cut_off"]
    use_pseudo_inv = fit_params["pseudo_inv"]
    db = fit_params.get("db", False)  # Debug flag

    # Voxel-specific parameter arrays
    current_freqs_Hz = fit_params["metabs_freqs_Hz"].copy()
    metabs_t2s = fit_params["metabs_t2s"]
    metabs_freq_ranges = fit_params["metabs_freq_ranges"].copy()
    freq_range_lower_bound_Hz = fit_params["freq_range_lower_bound_Hz"]
    freq_range_upper_bound_Hz = fit_params["freq_range_upper_bound_Hz"]

    # if db:
    all_r2_freq = np.zeros((len(metabs), niter, npoints, 2))

    # --- 2. Prepare input signal ---
    # Apply cut-off and ensure correct number of points.
    if nsamplepoints != input_data.shape[0] or cut_off != 0:
        if signal_domain == "spectral":
            temp_signal = np.fft.ifft(np.fft.ifftshift(input_data))
            temp_signal = temp_signal[cut_off : cut_off + nsamplepoints]
            input_data = np.fft.fftshift(np.fft.fft(temp_signal))
        else:  # Temporal domain
            input_data = input_data[cut_off : cut_off + nsamplepoints]

    # The cost function (R2) is calculated in the time domain.
    if signal_domain == "spectral":
        input_data_temporal = np.fft.ifft(np.fft.ifftshift(input_data))
    else:
        input_data_temporal = input_data.copy()

    # --- 3. Optimization: Separate frozen from unfrozen metabolites ---
    # Identify frozen metabolites (those with zero frequency ranges)
    frozen_metab_mask = metabs_freq_ranges <= 1e-10
    unfrozen_metab_indices = np.where(~frozen_metab_mask)[0]
    frozen_metab_indices = np.where(frozen_metab_mask)[0]

    # Pre-calculate frozen metabolite contributions (cache)
    frozen_signal_cache = None
    if len(frozen_metab_indices) > 0:
        frozen_freqs = current_freqs_Hz[frozen_metab_indices]
        frozen_t2s = metabs_t2s[frozen_metab_indices]
        if len(frozen_freqs) > 0:
            # Calculate signal contribution from frozen metabolites only once
            # This does not actually fit anything, just computes the signal
            # based on the frozen parameters.
            frozen_signal_cache, _ = solve_for_complex_amplitudes(
                input_data=input_data_temporal,
                freqs=frozen_freqs,
                metabs_t2s=frozen_t2s,
                sampling_time_axis_s=sampling_time_axis_s,
                pseudo_inv=use_pseudo_inv,
                signal_domain="temporal",
            )

    # --- 4. Iterative Fitting Loop ---
    r2 = np.zeros((len(metabs), npoints))
    # Iterate over niter (not really used)
    for i in range(niter):
        # Iterate over each metabolite to find its best frequency
        for m, metab in enumerate(metabs):
            freqs = current_freqs_Hz.copy()
            # Define the search grid for the current iteration
            search_range_Hz = np.linspace(
                current_freqs_Hz[m] - metabs_freq_ranges[m] / 2,
                current_freqs_Hz[m] + metabs_freq_ranges[m] / 2,
                npoints,
            )

            if np.all(search_range_Hz == search_range_Hz[0]):  # Skip if range is zero
                all_r2_freq[m, i, :, 0] = current_freqs_Hz[m]
                all_r2_freq[m, i, :, 1] = 0

            # Calculate R2 for each frequency point in the search grid
            for f, freq in enumerate(search_range_Hz):
                freqs[m] = freq

                # Optimized signal calculation: separate frozen from unfrozen
                if len(unfrozen_metab_indices) > 0:
                    # Get unfrozen metabolite parameters for current test
                    unfrozen_freqs = freqs[unfrozen_metab_indices]
                    unfrozen_t2s = metabs_t2s[unfrozen_metab_indices]

                    # Calculate signal from unfrozen metabolites only
                    unfrozen_signal, _ = solve_for_complex_amplitudes(
                        input_data=input_data_temporal,
                        freqs=unfrozen_freqs,
                        metabs_t2s=unfrozen_t2s,
                        sampling_time_axis_s=sampling_time_axis_s,
                        pseudo_inv=use_pseudo_inv,
                        signal_domain="temporal",
                    )

                    # Combine cached frozen signal with current unfrozen signal
                    if frozen_signal_cache is not None:
                        # Concatenate signals and sum along metabolite axis
                        total_fitted_signal = np.concatenate(
                            [frozen_signal_cache, unfrozen_signal], axis=1
                        )
                    else:
                        total_fitted_signal = unfrozen_signal

                    fitted_signal_sum = np.sum(total_fitted_signal, axis=1)
                else:
                    # All metabolites are frozen - use cached signal only
                    fitted_signal_sum = (
                        np.sum(frozen_signal_cache, axis=1)
                        if frozen_signal_cache is not None
                        else np.zeros_like(input_data_temporal)
                    )

                # Calculate squared error
                r2[m, f] = _estimate_diff(
                    y=input_data_temporal,
                    y_hat=fitted_signal_sum,
                )
                # if db:
                all_r2_freq[m, i, f, 0] = freqs[m]
                all_r2_freq[m, i, f, 1] = r2[m, f]

            # Update the frequency to the best one found in this iteration
            best_freq = search_range_Hz[np.argmin(r2[m, :])]
            current_freqs_Hz[m] = np.clip(
                best_freq, freq_range_lower_bound_Hz[m], freq_range_upper_bound_Hz[m]
            )

    # --- 4. Final Calculation ----
    # Calculate final signal and amplitudes for ALL metabolites with optimized parameters
    fitted_signal, complex_amps = solve_for_complex_amplitudes(
        input_data=input_data_temporal,
        freqs=current_freqs_Hz,  # Now includes optimized unfrozen frequencies
        metabs_t2s=metabs_t2s,  # Unchanged for all metabolites
        sampling_time_axis_s=sampling_time_axis_s,
        pseudo_inv=use_pseudo_inv,
        signal_domain="temporal",
    )

    fitted_spectral = np.fft.fftshift(np.fft.fft(fitted_signal, axis=0))
    return (fitted_spectral, current_freqs_Hz, complex_amps, all_r2_freq)


def fit_t2_pseudo_inv(input_data: np.ndarray = None, fit_params: dict = None) -> tuple:
    """
    Fit metabolite T2* values for a single spectrum via grid search.

    This function iteratively refines the T2* estimate for each
    metabolite by searching over a progressively smaller range. It is designed
    to be called by a wrapper function that prepares the parameters for a
    single voxel. The optimization separates metabolites with fixed ("frozen")
    T2* values from those being actively fitted to improve efficiency.

    Parameters
    ----------
    input_data : np.ndarray, optional
        The 1D data for a single spectrum. Default is None.
    fit_params : dict, optional
        A dictionary containing all necessary parameters for the fit, including
        both global settings and voxel-specific initial guesses and ranges.
        Default is None.

    Returns
    -------
    fitted_spectral : np.ndarray
        The final fitted signal in the spectral domain, composed of all
        metabolite contributions.
    current_t2s_s : np.ndarray
        An array of the final fitted T2* values in seconds for each metabolite.
    complex_amps : np.ndarray
        An array of the final complex amplitudes for each metabolite.
    all_r2_t2 : np.ndarray
        A multi-dimensional array storing the R-squared (goodness of fit)
        values for each T2* point in the search grid across all
        iterations. The shape is (n_metabs, niter, npoints, 2).

    """
    # --- 1. Unpack necessary parameters from the dictionary ---
    metabs = fit_params["metabs"]
    npoints = fit_params["npoints"]
    niter = fit_params["niter"]
    signal_domain = fit_params["signal_domain"]
    sampling_time_axis_s = fit_params["sampling_time_axis_s"]
    nsamplepoints = fit_params["nsamplepoints"]
    cut_off = fit_params["cut_off"]
    use_pseudo_inv = fit_params["pseudo_inv"]

    # Voxel-specific parameter arrays
    metabs_freqs_Hz = fit_params["metabs_freqs_Hz"]
    current_t2s_s = fit_params["metabs_t2s"].copy()
    metabs_t2_ranges = fit_params["metabs_t2_ranges"].copy()
    min_t2_s = fit_params["min_t2_s"]
    max_t2_s = fit_params["max_t2_s"]

    # if db:
    all_r2_t2 = np.zeros((len(metabs), niter, npoints, 2))

    # --- 2. Prepare input signal ---
    # Apply cut-off and ensure correct number of points.
    if nsamplepoints != input_data.shape[0] or cut_off != 0:
        if signal_domain == "spectral":
            temp_signal = np.fft.ifft(np.fft.ifftshift(input_data))
            temp_signal = temp_signal[cut_off : cut_off + nsamplepoints]
            input_data = np.fft.fftshift(np.fft.fft(temp_signal))
        else:  # Temporal domain
            input_data = input_data[cut_off : cut_off + nsamplepoints]

    # The cost function (R2) is calculated in the time domain.
    if signal_domain == "spectral":
        input_data_temporal = np.fft.ifft(np.fft.ifftshift(input_data))
    else:
        input_data_temporal = input_data.copy()

    # --- 3. Optimization: Separate frozen from unfrozen metabolites ---
    # Identify frozen metabolites (those with zero T2* ranges)
    frozen_metab_mask = metabs_t2_ranges <= 1e-10
    unfrozen_metab_indices = np.where(~frozen_metab_mask)[0]
    frozen_metab_indices = np.where(frozen_metab_mask)[0]

    # Pre-calculate frozen metabolite contributions (cache)
    frozen_signal_cache = None
    if len(frozen_metab_indices) > 0:
        frozen_freqs = metabs_freqs_Hz[frozen_metab_indices]
        frozen_t2s = current_t2s_s[frozen_metab_indices]
        if len(frozen_freqs) > 0:
            # Calculate signal contribution from frozen metabolites only once
            # This does not actually fit anything, just computes the signal
            # based on the frozen parameters.
            frozen_signal_cache, _ = solve_for_complex_amplitudes(
                input_data=input_data_temporal,
                freqs=frozen_freqs,
                metabs_t2s=frozen_t2s,
                sampling_time_axis_s=sampling_time_axis_s,
                pseudo_inv=use_pseudo_inv,
                signal_domain="temporal",
            )

    # --- 4. Iterative Fitting Loop ---
    # initialize R2 array
    r2 = np.zeros((len(metabs), npoints))

    # Iterate over niter (not really used )
    for i in range(niter):
        # Iterate over each metabolite to find its best T2*
        for m, metab in enumerate(metabs):
            t2s = current_t2s_s.copy()
            # Define the search grid of T2* offsets for the current iteration
            search_range_s = np.linspace(
                -metabs_t2_ranges[m] / 2, metabs_t2_ranges[m] / 2, npoints
            )

            # Skip frozen metabolites (those with zero T2* ranges)
            if metabs_t2_ranges[m] <= 1e-10:
                # Fill R2 array with zeros for frozen metabolite and continue to next
                all_r2_t2[m, i, :, 0] = current_t2s_s[m]
                all_r2_t2[m, i, :, 1] = 0
                continue

            # Calculate R2 for each T2* point in the search grid
            for t, t2_offset in enumerate(search_range_s):
                t2s[m] = current_t2s_s[m] + t2_offset
                if t2s[m] > 0:  # T2* must be positive
                    # Optimized signal calculation: separate frozen from unfrozen
                    if len(unfrozen_metab_indices) > 0:
                        # Get unfrozen metabolite parameters for current test
                        unfrozen_freqs = metabs_freqs_Hz[unfrozen_metab_indices]
                        unfrozen_t2s = t2s[unfrozen_metab_indices]

                        # Calculate signal from unfrozen metabolites only
                        unfrozen_signal, _ = solve_for_complex_amplitudes(
                            input_data=input_data_temporal,
                            freqs=unfrozen_freqs,
                            metabs_t2s=unfrozen_t2s,
                            sampling_time_axis_s=sampling_time_axis_s,
                            pseudo_inv=use_pseudo_inv,
                            signal_domain="temporal",
                        )

                        # Combine cached frozen signal with current unfrozen signal
                        if frozen_signal_cache is not None:
                            # Concatenate signals and sum along metabolite axis
                            total_fitted_signal = np.concatenate(
                                [frozen_signal_cache, unfrozen_signal], axis=1
                            )
                        else:
                            total_fitted_signal = unfrozen_signal

                        fitted_signal_sum = np.sum(total_fitted_signal, axis=1)
                    else:
                        # All metabolites are frozen - use cached signal only
                        fitted_signal_sum = (
                            np.sum(frozen_signal_cache, axis=1)
                            if frozen_signal_cache is not None
                            else np.zeros_like(input_data_temporal)
                        )

                    # Calculate squared error
                    r2[m, t] = _estimate_diff(
                        y=input_data_temporal,
                        y_hat=fitted_signal_sum,
                    )
                else:
                    # Assign infinite error to invalid T2* values
                    r2[m, t] = np.inf

                all_r2_t2[m, i, t, 0] = t2s[m]
                all_r2_t2[m, i, t, 1] = r2[m, t]

            # Update the T2* to the best one found in this iteration
            best_t2 = current_t2s_s[m] + search_range_s[np.argmin(r2[m, :])]
            current_t2s_s[m] = np.clip(best_t2, min_t2_s[m], max_t2_s[m])

    # --- 4. Final Calculation ---
    # Calculate final signal and amplitudes with the best-fit T2* values.
    fitted_signal, complex_amps = solve_for_complex_amplitudes(
        input_data=input_data_temporal,
        freqs=metabs_freqs_Hz,
        metabs_t2s=current_t2s_s,
        sampling_time_axis_s=sampling_time_axis_s,
        pseudo_inv=use_pseudo_inv,
        signal_domain="temporal",
    )

    fitted_spectral = np.fft.fftshift(np.fft.fft(fitted_signal, axis=0))

    return (fitted_spectral, current_t2s_s, complex_amps, all_r2_t2)


def _estimate_diff(y, y_hat, method="sse"):
    """Quantify the difference between observed and fitted signals.

    Parameters
    ----------
    y : array_like
        Observed signal.
    y_hat : array_like
        Fitted signal with the same shape as ``y``.
    method : {"sse", "r2"}
        Error metric to use. ``"sse"`` returns the sum of squared errors,
        ``"r2"`` returns the coefficient of determination.

    Returns
    -------
    float
        Requested error metric.

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    """

    if method == "sse":
        # Sum needs to be applied across the metabolite axis
        sse = np.sum(np.abs(y - y_hat) ** 2)
        return sse
    elif method == "r2":
        # Does not work reliably with complex and oscillatory data
        sse = np.sum(np.abs(y - y_hat) ** 2)
        sst = np.sum(np.abs(y - np.mean(y)) ** 2)
        r2 = 1 - sse / sst if sst != 0 else 0
        return r2
    else:
        raise ValueError(f"Unknown method '{method}' for estimating difference.")


def add_metab_to_fit_params_old(
    new_freq: float | np.ndarray | xr.DataArray | None,
    new_t2_s: float,
    new_metab_name: str = "new_metabolite",
    fit_params: xr.Dataset = None,
    fit_results: xr.Dataset = None,
    new_freq_range: float = 30.0,
    new_t2_range: float = 0.04,
    new_min_t2_s: float = 0.005,
    new_max_t2_s: float = 0.1,
    new_freq_unit: str = "Hz",
    reference_metabolite: str | None = None,
    freq_offset: float = 0.0,
    freq_offset_unit: str = "Hz",
    freeze_other_freqs: bool = True,
    freeze_other_t2s: bool = True,
) -> xr.Dataset:
    """
    Add a new metabolite to a ``fit_params`` Dataset for subsequent fitting.

    This helper is typically called from the interactive review UI once the
    operator has identified an additional peak to model. The function builds the
    new metabolite row, broadcasts the supplied initial parameters across every
    voxel in the dataset, and (optionally) collapses the search ranges of the
    existing metabolites to keep them fixed for the follow-up optimisation. When
    ``fit_results`` are supplied, their fitted frequency/T2* values are used as
    the starting point for the previously-known peaks, while the new metabolite
    receives the user-specified ``new_freq``/``new_t2_s`` everywhere.

    Parameters
    ----------
    new_freq : float or array-like, optional
        Initial frequency guess for the new peak. If ``reference_metabolite`` is
        provided this value is ignored and the initial frequency is derived from
        the reference frequencies plus ``freq_offset``. Default is None.
    new_t2_s : float
        Initial T2* guess for the new peak (in seconds).
    new_metab_name : str, optional
        Name for the new metabolite (e.g., 'urea'). Default is "new_metabolite".
    fit_params : xr.Dataset, optional
        The original `fit_params` Dataset. Required if `fit_results` is not
        provided.
    fit_results : xr.Dataset, optional
        Results from a previous fit. If provided, `fit_freqs` and `fit_t2s` are
        used as the initial guesses for the next fit.
    new_freq_range : float, optional
        Frequency search range for the new peak (in Hz or ppm, depending on
        `new_freq_unit`). Default is 30.0.
    new_t2_range : float, optional
        T2* search range for the new peak (in seconds). Default is 0.04.
    new_min_t2_s : float, optional
        Minimum allowed T2* for the new peak. Default is 0.005.
    new_max_t2_s : float, optional
        Maximum allowed T2* for the new peak. Default is 0.1.
    new_freq_unit : str, optional
        Unit of `new_freq` and `new_freq_range`. Can be 'Hz' or 'ppm'.
        Default is 'Hz'.
    reference_metabolite : str, optional
        Name of an existing metabolite whose fitted/initial frequencies should be
        copied as the starting point for the new metabolite. Requires
        ``fit_params`` or ``fit_results`` to contain the metabolite.
    freq_offset : float, optional
        Constant offset to apply to the reference metabolite's frequencies. The
        offset is interpreted in units defined by ``freq_offset_unit``. Default
        is 0.0.
    freq_offset_unit : str, optional
        Unit of ``freq_offset``. Can be 'Hz' or 'ppm'. Default is 'Hz'.
    freeze_other_freqs : bool, optional
        If True, sets the frequency search range of existing peaks to 0.
        Default is True.
    freeze_other_t2s : bool, optional
        If True, sets the T2* search range of existing peaks to 0.
        Default is True.

    Returns
    -------
    xr.Dataset
        A new, updated `xr.Dataset` of fitting parameters including the new peak.

    """
    if fit_results is None and fit_params is None:
        raise ValueError("Either fit_params or fit_results must be provided.")

    if reference_metabolite is None and new_freq is None:
        raise ValueError(
            "Either new_freq must be provided or reference_metabolite must be set."
        )

    allowed_units = {"hz", "ppm"}
    if freq_offset_unit.lower() not in allowed_units:
        raise ValueError(
            f"Unknown freq_offset_unit '{freq_offset_unit}'. Expected one of {allowed_units}."
        )

    # --- 1. Determine the base parameters for *existing* metabolites ---
    # Use previous fit results as the new starting point, or fall back to
    # original params.
    if fit_results is not None:
        base_params = fit_params.copy(
            deep=True
        )  # Start with a copy to preserve all fields
        base_params["initial_freqs_Hz"] = fit_results["fit_freqs"]
        base_params["initial_t2s_s"] = fit_results["fit_t2s"]
    else:  # If no results, start from the provided fit_params
        base_params = fit_params.copy(deep=True)

    # --- 2. Convert frequency-related inputs when PPM units are requested ---
    source_for_axes = fit_params if fit_params is not None else fit_results

    hz_axis = getattr(source_for_axes, "freq_axis_hz", None)
    ppm_axis = getattr(source_for_axes, "freq_axis_ppm", None)

    if isinstance(source_for_axes, (xr.Dataset, xr.DataArray)):
        if hz_axis is None:
            hz_axis = resolve_frequency_axis(source_for_axes.attrs, "Hz")
        if ppm_axis is None:
            ppm_axis = resolve_frequency_axis(source_for_axes.attrs, "ppm")

    if isinstance(source_for_axes, xr.Dataset):
        if hz_axis is None and "freq_axis_hz" in source_for_axes:
            hz_axis = source_for_axes["freq_axis_hz"].values
        if ppm_axis is None and "freq_axis_ppm" in source_for_axes:
            ppm_axis = source_for_axes["freq_axis_ppm"].values

    def _require_axes():
        if hz_axis is None or ppm_axis is None:
            raise ValueError(
                "Frequency axis information is required to convert from ppm to Hz."
            )

    def _ppm_delta_to_Hz(delta_ppm: float) -> float:
        """Convert a delta specified in ppm into Hz using the available axes."""
        base = freq_ppm_to_Hz(freq_ppm=0.0, hz_axis=hz_axis, ppm_axis=ppm_axis)
        offset = freq_ppm_to_Hz(freq_ppm=delta_ppm, hz_axis=hz_axis, ppm_axis=ppm_axis)
        base_val = float(np.asarray(base))
        offset_val = float(np.asarray(offset))
        return offset_val - base_val

    def _convert_ppm_value(value):
        if value is None:
            return None
        if isinstance(value, xr.DataArray):
            array_val = np.asarray(value.data)
            return _convert_ppm_ndarray(array_val)
        if isinstance(value, np.ndarray):
            return _convert_ppm_ndarray(value)
        converted = freq_ppm_to_Hz(freq_ppm=value, hz_axis=hz_axis, ppm_axis=ppm_axis)
        return float(np.asarray(converted))

    def _convert_ppm_ndarray(array_val: np.ndarray) -> np.ndarray:
        flat = np.asarray(array_val).ravel().tolist()
        converted = freq_ppm_to_Hz(freq_ppm=flat, hz_axis=hz_axis, ppm_axis=ppm_axis)
        return np.asarray(converted, dtype=float).reshape(array_val.shape)

    use_reference = reference_metabolite is not None

    if new_freq_unit.lower() == "ppm":
        _require_axes()
        new_freq_range = np.abs(_ppm_delta_to_Hz(new_freq_range))
        if not use_reference:
            new_freq = _convert_ppm_value(new_freq)

    freq_offset_Hz = freq_offset
    if freq_offset_unit.lower() == "ppm":
        _require_axes()
        freq_offset_Hz = _ppm_delta_to_Hz(freq_offset)

    if isinstance(freq_offset_Hz, (np.ndarray, list)):
        freq_offset_Hz = float(np.asarray(freq_offset_Hz))
    else:
        freq_offset_Hz = float(freq_offset_Hz)

    if use_reference:
        if reference_metabolite not in base_params.metabolite.values:
            raise ValueError(
                f"reference_metabolite '{reference_metabolite}' is not present in the fit parameters."
            )
        reference_freqs = base_params["initial_freqs_Hz"].sel(
            metabolite=reference_metabolite, drop=True
        )
        new_freq = np.asarray(reference_freqs.data) + freq_offset_Hz
    elif new_freq is None:
        raise ValueError(
            "new_freq must be provided when reference_metabolite is not specified."
        )

    if isinstance(new_freq, xr.DataArray):
        new_freq = np.asarray(new_freq.data)
    elif isinstance(new_freq, list):
        new_freq = np.asarray(new_freq)

    if isinstance(new_freq, np.ndarray):
        new_freq = np.asarray(new_freq, dtype=float)
        if new_freq.ndim == 0:
            new_freq = float(new_freq.item())

    if isinstance(new_freq_range, np.ndarray):
        if new_freq_range.size != 1:
            raise ValueError("new_freq_range must be a scalar value.")
        new_freq_range = float(new_freq_range.reshape(-1)[0])
    else:
        new_freq_range = float(new_freq_range)

    # --- 3. Create a Dataset containing only the new peak's parameters ---
    template = base_params.isel(metabolite=0, drop=True)
    new_peak_data = xr.Dataset(coords={"metabolite": [new_metab_name]})
    target_dims = template["initial_freqs_Hz"].dims

    # --- Handle Frequency (new_freq) ---
    if isinstance(new_freq, (np.ndarray, xr.DataArray)):
        # If input is an array, expand its dimensions to match the template
        input_dims = target_dims[: new_freq.ndim]
        temp_da = xr.DataArray(new_freq, dims=input_dims)
        missing_dims = [d for d in target_dims if d not in input_dims]

        # Create the correctly-dimensioned DataArray
        aligned_da = temp_da.expand_dims(missing_dims).transpose(*target_dims)

        # Assign the final DataArray to the new dataset
        new_peak_data["initial_freqs_Hz"] = aligned_da
    else:
        # If input is a scalar, use full_like for simplicity
        new_peak_data["initial_freqs_Hz"] = xr.full_like(
            template["initial_freqs_Hz"], new_freq
        )

    # --- Handle T2 (new_t2_s) ---
    # Repeat the same logic for the T2 value
    if isinstance(new_t2_s, (np.ndarray, xr.DataArray)):
        input_dims_t2 = target_dims[: new_t2_s.ndim]
        temp_da_t2 = xr.DataArray(new_t2_s, dims=input_dims_t2)
        missing_dims_t2 = [d for d in target_dims if d not in input_dims_t2]
        aligned_da_t2 = temp_da_t2.expand_dims(missing_dims_t2).transpose(*target_dims)
        new_peak_data["initial_t2s_s"] = aligned_da_t2
    else:
        new_peak_data["initial_t2s_s"] = xr.full_like(
            template["initial_t2s_s"], new_t2_s
        )

    # --- For remaining params, scalar values are expected ---
    new_peak_data["metabs_freq_ranges"] = xr.full_like(
        template["metabs_freq_ranges"], new_freq_range
    )
    new_peak_data["metabs_t2_ranges"] = xr.full_like(
        template["metabs_t2_ranges"], new_t2_range
    )
    new_peak_data["min_t2_s"] = xr.full_like(template["min_t2_s"], new_min_t2_s)
    new_peak_data["max_t2_s"] = xr.full_like(template["max_t2_s"], new_max_t2_s)
    if "skip_fitting" in base_params:
        new_peak_data["skip_fitting"] = xr.full_like(
            template["initial_freqs_Hz"], False, dtype=bool
        )

    # --- 4. Concatenate the existing and new peak parameters ---
    updated_params = xr.concat([base_params, new_peak_data], dim="metabolite")

    if "skip_fitting" in updated_params:
        skip_da = updated_params["skip_fitting"]
        if np.issubdtype(skip_da.dtype, np.floating):
            skip_da = skip_da.fillna(0.0)
        updated_params["skip_fitting"] = skip_da.astype(bool)

    # --- 5. Apply the freezing logic to old peaks if requested ---
    is_old_peak = updated_params.metabolite != new_metab_name
    if freeze_other_freqs:
        updated_params["metabs_freq_ranges"] = xr.where(
            is_old_peak, 0.0, updated_params["metabs_freq_ranges"]
        )
    if freeze_other_t2s:
        updated_params["metabs_t2_ranges"] = xr.where(
            is_old_peak, 0.0, updated_params["metabs_t2_ranges"]
        )

    # --- 6. Recalculate derived parameters and update global attributes ---
    updated_params["freq_range_lower_bound_Hz"] = (
        updated_params["initial_freqs_Hz"] - updated_params["metabs_freq_ranges"] / 2
    )
    updated_params["freq_range_upper_bound_Hz"] = (
        updated_params["initial_freqs_Hz"] + updated_params["metabs_freq_ranges"] / 2
    )

    # Carry over the attributes from the original params
    source_attrs = fit_params.attrs if fit_params is not None else fit_results.attrs
    updated_params.attrs = source_attrs.copy()
    # Update the metabolite list in the attributes
    updated_params.attrs["metabs"] = updated_params.metabolite.values.tolist()

    return updated_params


def add_metab_to_fit_params(
    new_freq: float | np.ndarray | xr.DataArray | None,
    new_t2_s: float,
    new_metab_name: str = "new_metabolite",
    fit_params: xr.Dataset = None,
    fit_results: xr.Dataset = None,
    new_freq_range: float = 30.0,
    new_t2_range: float = 0.04,
    new_min_t2_s: float = 0.005,
    new_max_t2_s: float = 0.1,
    new_freq_unit: str = "Hz",
    reference_metabolite: str | None = None,
    freq_offset: float = 0.0,
    freq_offset_unit: str = "Hz",
    reference_repetitions: int | slice | range | Sequence[int] | str | None = None,
    freeze_other_freqs: bool = True,
    freeze_other_t2s: bool = True,
) -> xr.Dataset:
    """
    Add a new metabolite to a ``fit_params`` Dataset for subsequent fitting.

    This helper is typically called from the interactive review UI once the
    operator has identified an additional peak to model. The function builds the
    new metabolite row, broadcasts the supplied initial parameters across every
    voxel in the dataset, and (optionally) collapses the search ranges of the
    existing metabolites to keep them fixed for the follow-up optimisation. When
    ``fit_results`` are supplied, their fitted frequency/T2* values are used as
    the starting point for the previously-known peaks, while the new metabolite
    receives the user-specified ``new_freq``/``new_t2_s`` everywhere.

    Parameters
    ----------
    new_freq : float or array-like, optional
        Initial frequency guess for the new peak. If ``reference_metabolite`` is
        provided this value is ignored and the initial frequency is derived from
        the reference frequencies plus ``freq_offset``. Default is None.
    new_t2_s : float
        Initial T2* guess for the new peak (in seconds).
    new_metab_name : str, optional
        Name for the new metabolite (e.g., 'urea'). Default is "new_metabolite".
    fit_params : xr.Dataset, optional
        The original `fit_params` Dataset. Required if `fit_results` is not
        provided.
    fit_results : xr.Dataset, optional
        Results from a previous fit. If provided, `fit_freqs` and `fit_t2s` are
        used as the initial guesses for the next fit.
    new_freq_range : float, optional
        Frequency search range for the new peak (in Hz or ppm, depending on
        `new_freq_unit`). Default is 30.0.
    new_t2_range : float, optional
        T2* search range for the new peak (in seconds). Default is 0.04.
    new_min_t2_s : float, optional
        Minimum allowed T2* for the new peak. Default is 0.005.
    new_max_t2_s : float, optional
        Maximum allowed T2* for the new peak. Default is 0.1.
    new_freq_unit : str, optional
        Unit of `new_freq` and `new_freq_range`. Can be 'Hz' or 'ppm'.
        Default is 'Hz'.
    reference_metabolite : str, optional
        Name of an existing metabolite whose fitted/initial frequencies should be
        copied as the starting point for the new metabolite. Requires
        ``fit_params`` or ``fit_results`` to contain the metabolite.
    freq_offset : float, optional
        Constant offset to apply to the reference metabolite's frequencies. The
        offset is interpreted in units defined by ``freq_offset_unit``. Default
        is 0.0.
    freq_offset_unit : str, optional
        Unit of ``freq_offset``. Can be 'Hz' or 'ppm'. Default is 'Hz'.
    reference_repetitions : int, slice, range, sequence of int or str, optional
        Repetition index/indices used to derive the reference frequency when
        ``reference_metabolite`` is provided. If omitted, the fitted reference
        frequency from each repetition is used directly, which preserves the
        current behaviour. If a single repetition is selected, that repetition's
        fitted frequency is reused for all repetitions. If multiple repetitions
        are selected, their median fitted frequency is reused for all
        repetitions. If set to ``"max_fit_amp"``, the repetition with the
        largest fitted amplitude magnitude of the reference metabolite is chosen
        and reused for all repetitions.
    freeze_other_freqs : bool, optional
        If True, sets the frequency search range of existing peaks to 0.
        Default is True.
    freeze_other_t2s : bool, optional
        If True, sets the T2* search range of existing peaks to 0.
        Default is True.

    Returns
    -------
    xr.Dataset
        A new, updated `xr.Dataset` of fitting parameters including the new peak.

    """
    if fit_results is None and fit_params is None:
        raise ValueError("Either fit_params or fit_results must be provided.")

    if reference_metabolite is None and new_freq is None:
        raise ValueError(
            "Either new_freq must be provided or reference_metabolite must be set."
        )

    allowed_units = {"hz", "ppm"}
    if freq_offset_unit.lower() not in allowed_units:
        raise ValueError(
            f"Unknown freq_offset_unit '{freq_offset_unit}'. Expected one of {allowed_units}."
        )

    # --- 1. Determine the base parameters for *existing* metabolites ---
    # Use previous fit results as the new starting point, or fall back to
    # original params.
    if fit_results is not None:
        base_params = fit_params.copy(
            deep=True
        )  # Start with a copy to preserve all fields
        try:
            merged_freqs = fit_results["fit_freqs"].combine_first(
                base_params["initial_freqs_Hz"]
            )
            merged_t2s = fit_results["fit_t2s"].combine_first(
                base_params["initial_t2s_s"]
            )
        except KeyError as exc:
            raise KeyError(
                "fit_results must contain 'fit_freqs' and 'fit_t2s' variables."
            ) from exc

        # Preserve any metabolites that exist only in fit_params while still
        # reusing fitted values for overlapping metabolites.
        base_params["initial_freqs_Hz"] = merged_freqs.transpose(
            *base_params["initial_freqs_Hz"].dims
        )
        base_params["initial_t2s_s"] = merged_t2s.transpose(
            *base_params["initial_t2s_s"].dims
        )
    else:  # If no results, start from the provided fit_params
        base_params = fit_params.copy(deep=True)

    # --- 2. Convert frequency-related inputs when PPM units are requested ---
    source_for_axes = fit_params if fit_params is not None else fit_results

    hz_axis = getattr(source_for_axes, "freq_axis_hz", None)
    ppm_axis = getattr(source_for_axes, "freq_axis_ppm", None)

    if isinstance(source_for_axes, (xr.Dataset, xr.DataArray)):
        if hz_axis is None:
            hz_axis = resolve_frequency_axis(source_for_axes.attrs, "Hz")
        if ppm_axis is None:
            ppm_axis = resolve_frequency_axis(source_for_axes.attrs, "ppm")

    if isinstance(source_for_axes, xr.Dataset):
        if hz_axis is None and "freq_axis_hz" in source_for_axes:
            hz_axis = source_for_axes["freq_axis_hz"].values
        if ppm_axis is None and "freq_axis_ppm" in source_for_axes:
            ppm_axis = source_for_axes["freq_axis_ppm"].values

    def _require_axes():
        if hz_axis is None or ppm_axis is None:
            raise ValueError(
                "Frequency axis information is required to convert from ppm to Hz."
            )

    def _ppm_delta_to_Hz(delta_ppm: float) -> float:
        """Convert a delta specified in ppm into Hz using the available axes."""
        base = freq_ppm_to_Hz(freq_ppm=0.0, hz_axis=hz_axis, ppm_axis=ppm_axis)
        offset = freq_ppm_to_Hz(freq_ppm=delta_ppm, hz_axis=hz_axis, ppm_axis=ppm_axis)
        base_val = float(np.asarray(base))
        offset_val = float(np.asarray(offset))
        return offset_val - base_val

    def _convert_ppm_value(value):
        if value is None:
            return None
        if isinstance(value, xr.DataArray):
            array_val = np.asarray(value.data)
            return _convert_ppm_ndarray(array_val)
        if isinstance(value, np.ndarray):
            return _convert_ppm_ndarray(value)
        converted = freq_ppm_to_Hz(freq_ppm=value, hz_axis=hz_axis, ppm_axis=ppm_axis)
        return float(np.asarray(converted))

    def _convert_ppm_ndarray(array_val: np.ndarray) -> np.ndarray:
        flat = np.asarray(array_val).ravel().tolist()
        converted = freq_ppm_to_Hz(freq_ppm=flat, hz_axis=hz_axis, ppm_axis=ppm_axis)
        return np.asarray(converted, dtype=float).reshape(array_val.shape)

    use_reference = reference_metabolite is not None

    if new_freq_unit.lower() == "ppm":
        new_freq_range_arr = np.asarray(new_freq_range)
        if np.any(np.abs(new_freq_range_arr) > 1.0):
            warnings.warn(
                "new_freq_range is larger than 1 ppm. If you meant to specify the search range in Hz, convert it to ppm before calling add_metab_to_fit_params; otherwise the search range may become very large.",
                UserWarning,
                stacklevel=2,
            )
        _require_axes()
        new_freq_range = np.abs(_ppm_delta_to_Hz(new_freq_range))
        if not use_reference:
            new_freq = _convert_ppm_value(new_freq)

    freq_offset_Hz = freq_offset
    if freq_offset_unit.lower() == "ppm":
        _require_axes()
        freq_offset_Hz = _ppm_delta_to_Hz(freq_offset)

    if isinstance(freq_offset_Hz, (np.ndarray, list)):
        freq_offset_Hz = float(np.asarray(freq_offset_Hz))
    else:
        freq_offset_Hz = float(freq_offset_Hz)

    def _get_repetition_dim(data_array: xr.DataArray) -> str | None:
        if "repetition_idx" in data_array.dims:
            return "repetition_idx"
        return next((dim for dim in data_array.dims if "rep" in dim.lower()), None)

    def _select_reference_repetitions(reference_freqs_da: xr.DataArray) -> xr.DataArray:
        if reference_repetitions is None:
            return reference_freqs_da

        repetition_dim = _get_repetition_dim(reference_freqs_da)
        if repetition_dim is None:
            raise ValueError(
                "reference_repetitions was provided, but no repetition dimension was found."
            )

        if isinstance(reference_repetitions, str):
            if reference_repetitions != "max_fit_amp":
                raise ValueError(
                    "reference_repetitions must be an index, slice, range, sequence of indices, or 'max_fit_amp'."
                )
            if fit_results is None or "fit_amps" not in fit_results:
                raise ValueError(
                    "reference_repetitions='max_fit_amp' requires fit_results with a 'fit_amps' variable."
                )
            reference_amps = fit_results["fit_amps"].sel(
                metabolite=reference_metabolite, drop=True
            )
            amp_repetition_dim = _get_repetition_dim(reference_amps)
            if amp_repetition_dim != repetition_dim:
                raise ValueError(
                    "Unable to align repetition dimension between fit_freqs and fit_amps for reference selection."
                )
            selected_idx = np.abs(reference_amps).argmax(dim=repetition_dim)
            selected = reference_freqs_da.isel({repetition_dim: selected_idx})
            return selected.broadcast_like(reference_freqs_da)

        if isinstance(reference_repetitions, slice):
            selected = reference_freqs_da.isel({repetition_dim: reference_repetitions})
        elif isinstance(reference_repetitions, range):
            selected = reference_freqs_da.isel(
                {repetition_dim: list(reference_repetitions)}
            )
        elif isinstance(reference_repetitions, Sequence) and not isinstance(
            reference_repetitions, (str, bytes)
        ):
            selected = reference_freqs_da.isel(
                {repetition_dim: list(reference_repetitions)}
            )
        else:
            selected = reference_freqs_da.isel(
                {repetition_dim: int(reference_repetitions)}
            )

        if repetition_dim in selected.dims:
            if selected.sizes[repetition_dim] == 0:
                raise ValueError(
                    "reference_repetitions selected no repetitions for the reference metabolite."
                )
            selected = selected.median(dim=repetition_dim, skipna=True)

        return selected.broadcast_like(reference_freqs_da)

    if use_reference:
        if reference_metabolite not in base_params.metabolite.values:
            raise ValueError(
                f"reference_metabolite '{reference_metabolite}' is not present in the fit parameters."
            )
        reference_freqs = base_params["initial_freqs_Hz"].sel(
            metabolite=reference_metabolite, drop=True
        )
        reference_freqs = _select_reference_repetitions(reference_freqs)
        new_freq = np.asarray(reference_freqs.data) + freq_offset_Hz
    elif new_freq is None:
        raise ValueError(
            "new_freq must be provided when reference_metabolite is not specified."
        )

    if isinstance(new_freq, xr.DataArray):
        new_freq = np.asarray(new_freq.data)
    elif isinstance(new_freq, list):
        new_freq = np.asarray(new_freq)

    if isinstance(new_freq, np.ndarray):
        new_freq = np.asarray(new_freq, dtype=float)
        if new_freq.ndim == 0:
            new_freq = float(new_freq.item())

    if isinstance(new_freq_range, np.ndarray):
        if new_freq_range.size != 1:
            raise ValueError("new_freq_range must be a scalar value.")
        new_freq_range = float(new_freq_range.reshape(-1)[0])
    else:
        new_freq_range = float(new_freq_range)

    # --- 3. Create a Dataset containing only the new peak's parameters ---
    template = base_params.isel(metabolite=0, drop=True)
    new_peak_data = xr.Dataset(coords={"metabolite": [new_metab_name]})
    target_dims = template["initial_freqs_Hz"].dims

    # --- Handle Frequency (new_freq) ---
    if isinstance(new_freq, (np.ndarray, xr.DataArray)):
        # If input is an array, expand its dimensions to match the template
        input_dims = target_dims[: new_freq.ndim]
        temp_da = xr.DataArray(new_freq, dims=input_dims)
        missing_dims = [d for d in target_dims if d not in input_dims]

        # Create the correctly-dimensioned DataArray
        aligned_da = temp_da.expand_dims(missing_dims).transpose(*target_dims)

        # Assign the final DataArray to the new dataset
        new_peak_data["initial_freqs_Hz"] = aligned_da
    else:
        # If input is a scalar, use full_like for simplicity
        new_peak_data["initial_freqs_Hz"] = xr.full_like(
            template["initial_freqs_Hz"], new_freq
        )

    # --- Handle T2 (new_t2_s) ---
    # Repeat the same logic for the T2 value
    if isinstance(new_t2_s, (np.ndarray, xr.DataArray)):
        input_dims_t2 = target_dims[: new_t2_s.ndim]
        temp_da_t2 = xr.DataArray(new_t2_s, dims=input_dims_t2)
        missing_dims_t2 = [d for d in target_dims if d not in input_dims_t2]
        aligned_da_t2 = temp_da_t2.expand_dims(missing_dims_t2).transpose(*target_dims)
        new_peak_data["initial_t2s_s"] = aligned_da_t2
    else:
        new_peak_data["initial_t2s_s"] = xr.full_like(
            template["initial_t2s_s"], new_t2_s
        )

    # --- For remaining params, scalar values are expected ---
    new_peak_data["metabs_freq_ranges"] = xr.full_like(
        template["metabs_freq_ranges"], new_freq_range
    )
    new_peak_data["metabs_t2_ranges"] = xr.full_like(
        template["metabs_t2_ranges"], new_t2_range
    )
    new_peak_data["min_t2_s"] = xr.full_like(template["min_t2_s"], new_min_t2_s)
    new_peak_data["max_t2_s"] = xr.full_like(template["max_t2_s"], new_max_t2_s)
    if "skip_fitting" in base_params:
        new_peak_data["skip_fitting"] = xr.full_like(
            template["initial_freqs_Hz"], False, dtype=bool
        )

    # --- 4. Concatenate the existing and new peak parameters ---
    updated_params = xr.concat([base_params, new_peak_data], dim="metabolite")

    if "skip_fitting" in updated_params:
        skip_da = updated_params["skip_fitting"]
        if np.issubdtype(skip_da.dtype, np.floating):
            skip_da = skip_da.fillna(0.0)
        updated_params["skip_fitting"] = skip_da.astype(bool)

    # --- 5. Apply the freezing logic to old peaks if requested ---
    is_old_peak = updated_params.metabolite != new_metab_name
    if freeze_other_freqs:
        updated_params["metabs_freq_ranges"] = xr.where(
            is_old_peak, 0.0, updated_params["metabs_freq_ranges"]
        )
    if freeze_other_t2s:
        updated_params["metabs_t2_ranges"] = xr.where(
            is_old_peak, 0.0, updated_params["metabs_t2_ranges"]
        )

    # --- 6. Recalculate derived parameters and update global attributes ---
    updated_params["freq_range_lower_bound_Hz"] = (
        updated_params["initial_freqs_Hz"] - updated_params["metabs_freq_ranges"] / 2
    )
    updated_params["freq_range_upper_bound_Hz"] = (
        updated_params["initial_freqs_Hz"] + updated_params["metabs_freq_ranges"] / 2
    )

    # Carry over the attributes from the original params
    source_attrs = fit_params.attrs if fit_params is not None else fit_results.attrs
    updated_params.attrs = source_attrs.copy()
    # Update the metabolite list in the attributes
    updated_params.attrs["metabs"] = updated_params.metabolite.values.tolist()

    return updated_params


def initialise_fit_params_from_results_old(
    fit_params: xr.Dataset,
    fit_results: xr.Dataset,
    freq_ranges_hz: float | Sequence[float] | np.ndarray | xr.DataArray | None = None,
    t2_ranges_s: float | Sequence[float] | np.ndarray | xr.DataArray | None = None,
) -> xr.Dataset:
    """Reuse fitted parameters as the next iteration's initial guesses.

    All voxel-wise ``initial_freqs_Hz`` and ``initial_t2s_s`` entries are updated
    with the values contained in ``fit_results``. Optional range arguments allow
    specifying metabolite-level search widths which are broadcast across all
    voxels.

    Parameters
    ----------
    fit_params : xr.Dataset
        Original parameter dataset used for the previous fit. A deep copy is
        returned with updated initial values.
    fit_results : xr.Dataset
        Dataset returned by :func:`fit_data_pseudo_inv`, providing the fitted
        frequencies and T2* values.
    freq_ranges_hz : float, sequence, ndarray or xr.DataArray, optional
        Frequency search ranges to assign (± Hz). If a scalar is provided the
        same range is used for every metabolite. Sequences or 1D arrays must have
        length equal to the metabolite dimension. ``xr.DataArray`` inputs must
        include the ``metabolite`` dimension. Default keeps the existing ranges.
    t2_ranges_s : float, sequence, ndarray or xr.DataArray, optional
        T2* search ranges (± s) applied with the same broadcasting rules as
        ``freq_ranges_hz``.

    Returns
    -------
    xr.Dataset
        Updated fitting parameters, ready for another optimisation round.

    """

    if fit_params is None or fit_results is None:
        raise ValueError("fit_params and fit_results must both be provided.")

    updated_params = fit_params.copy(deep=True)

    # --- Align fitted frequencies and T2* with the parameter layout ---
    target_dims = updated_params["initial_freqs_Hz"].dims

    try:
        fitted_freqs = fit_results["fit_freqs"].transpose(*target_dims)
        fitted_t2s = fit_results["fit_t2s"].transpose(*target_dims)
    except KeyError as exc:
        raise KeyError(
            "fit_results must contain 'fit_freqs' and 'fit_t2s' variables."
        ) from exc
    except ValueError as exc:
        raise ValueError(
            "Unable to align fit_results with fit_params; check dimensions."
        ) from exc

    updated_params["initial_freqs_Hz"] = fitted_freqs
    updated_params["initial_t2s_s"] = fitted_t2s

    def _broadcast_ranges(
        values: float | Sequence[float] | np.ndarray | xr.DataArray | None,
        template: xr.DataArray,
        param_name: str,
    ) -> xr.DataArray:
        if values is None:
            return template
        metab_coord = template.coords.get("metabolite")
        metab_dim = "metabolite"
        n_metabs = template.sizes[metab_dim]

        if isinstance(values, xr.DataArray):
            if metab_dim not in values.dims:
                raise ValueError(
                    f"{param_name} DataArray must contain a 'metabolite' dimension."
                )
            return values.broadcast_like(template)

        values_arr = np.asarray(values)
        if values_arr.ndim == 0:
            return xr.full_like(template, float(values_arr))
        if values_arr.shape[0] != n_metabs:
            raise ValueError(
                f"{param_name} must have length {n_metabs} to match metabolites."
            )
        per_metab = xr.DataArray(
            values_arr,
            dims=[metab_dim],
            coords={metab_dim: metab_coord},
        )
        return xr.zeros_like(template) + per_metab

    # --- Assign search ranges if requested ---
    updated_params["metabs_freq_ranges"] = _broadcast_ranges(
        freq_ranges_hz,
        updated_params["metabs_freq_ranges"],
        "freq_axis_hz",
    )
    updated_params["metabs_t2_ranges"] = _broadcast_ranges(
        t2_ranges_s,
        updated_params["metabs_t2_ranges"],
        "t2_ranges_s",
    )

    # --- Recalculate derived limits ---
    updated_params["freq_range_lower_bound_Hz"] = (
        updated_params["initial_freqs_Hz"] - updated_params["metabs_freq_ranges"] / 2
    )
    updated_params["freq_range_upper_bound_Hz"] = (
        updated_params["initial_freqs_Hz"] + updated_params["metabs_freq_ranges"] / 2
    )

    return updated_params


def initialise_fit_params_from_results(
    fit_params: xr.Dataset,
    fit_results: xr.Dataset,
    freq_ranges_hz: float | Sequence[float] | np.ndarray | xr.DataArray | None = None,
    t2_ranges_s: float | Sequence[float] | np.ndarray | xr.DataArray | None = None,
) -> xr.Dataset:
    """Reuse fitted parameters as the next iteration's initial guesses.

    All voxel-wise ``initial_freqs_Hz`` and ``initial_t2s_s`` entries are updated
    with the values contained in ``fit_results``. Optional range arguments allow
    specifying metabolite-level search widths which are broadcast across all
    voxels.

    Parameters
    ----------
    fit_params : xr.Dataset
        Original parameter dataset used for the previous fit. A deep copy is
        returned with updated initial values.
    fit_results : xr.Dataset
        Dataset returned by :func:`fit_data_pseudo_inv`, providing the fitted
        frequencies and T2* values.
    freq_ranges_hz : float, sequence, ndarray or xr.DataArray, optional
        Frequency search ranges to assign (± Hz). If a scalar is provided the
        same range is used for every metabolite. Sequences or 1D arrays must have
        length equal to the metabolite dimension. ``xr.DataArray`` inputs must
        include the ``metabolite`` dimension. Default keeps the existing ranges.
    t2_ranges_s : float, sequence, ndarray or xr.DataArray, optional
        T2* search ranges (± s) applied with the same broadcasting rules as
        ``freq_ranges_hz``.

    Returns
    -------
    xr.Dataset
        Updated fitting parameters, ready for another optimisation round.

    """
    if fit_params is None or fit_results is None:
        raise ValueError("fit_params and fit_results must both be provided.")

    updated_params = fit_params.copy(deep=True)

    # --- Align fitted frequencies and T2* with the parameter layout ---
    target_dims = updated_params["initial_freqs_Hz"].dims

    try:
        fitted_freqs = fit_results["fit_freqs"].transpose(*target_dims)
        fitted_t2s = fit_results["fit_t2s"].transpose(*target_dims)
    except KeyError as exc:
        raise KeyError(
            "fit_results must contain 'fit_freqs' and 'fit_t2s' variables."
        ) from exc
    except ValueError as exc:
        raise ValueError(
            "Unable to align fit_results with fit_params; check dimensions."
        ) from exc

    updated_params["initial_freqs_Hz"] = fitted_freqs
    updated_params["initial_t2s_s"] = fitted_t2s

    def _broadcast_ranges(
        values: float | Sequence[float] | np.ndarray | xr.DataArray | None,
        template: xr.DataArray,
        param_name: str,
    ) -> xr.DataArray:
        if values is None:
            return template
        metab_coord = template.coords.get("metabolite")
        metab_dim = "metabolite"
        n_metabs = template.sizes[metab_dim]

        if isinstance(values, xr.DataArray):
            if metab_dim not in values.dims:
                raise ValueError(
                    f"{param_name} DataArray must contain a 'metabolite' dimension."
                )
            return values.broadcast_like(template)

        values_arr = np.asarray(values)
        if values_arr.ndim == 0:
            return xr.full_like(template, float(values_arr))
        if values_arr.shape[0] != n_metabs:
            raise ValueError(
                f"{param_name} must have length {n_metabs} to match metabolites."
            )
        per_metab = xr.DataArray(
            values_arr,
            dims=[metab_dim],
            coords={metab_dim: metab_coord},
        )
        return xr.zeros_like(template) + per_metab

    # --- Assign search ranges if requested ---
    updated_params["metabs_freq_ranges"] = _broadcast_ranges(
        freq_ranges_hz,
        updated_params["metabs_freq_ranges"],
        "freq_axis_hz",
    )
    updated_params["metabs_t2_ranges"] = _broadcast_ranges(
        t2_ranges_s,
        updated_params["metabs_t2_ranges"],
        "t2_ranges_s",
    )

    # --- Recalculate derived limits ---
    updated_params["freq_range_lower_bound_Hz"] = (
        updated_params["initial_freqs_Hz"] - updated_params["metabs_freq_ranges"] / 2
    )
    updated_params["freq_range_upper_bound_Hz"] = (
        updated_params["initial_freqs_Hz"] + updated_params["metabs_freq_ranges"] / 2
    )

    return updated_params


def interactive_fit_params_setup(
    input_data: xr.DataArray,
    initial_params: dict = None,
) -> xr.Dataset:
    """
    Launch an interactive wizard to create a `fit_params` Dataset.

    This function first prompts the user to select metabolites via checkboxes.
    After confirmation, it launches a more advanced visual tool that allows for
    fine-tuning of voxel-specific parameters (like initial frequencies and
    search ranges) by clicking on image and spectral plots.

    Parameters
    ----------
    input_data: xr.DataArray
        The input spectral data.
    initial_params: dict
        A dictionary of initial parameter overrides.

    Returns
    -------
    fit_params: xr.DartaArray
        The fully configured `fit_params` xarray.Dataset once the visual tool is closed.
    """
    if initial_params is None:
        initial_params = {}
    output = widgets.Output()  # To capture and display outputs cleanly

    # --- Widget Setup ---
    known_metabolites = {
        "pyruvate": 171.1,
        "lactate": 183.2,
        "alanine": 176.7,
        "bicarbonate": 161.2,
        "urea": 163.5,
    }
    # Create a checkbox for each known metabolite.
    metab_checkboxes = [
        widgets.Checkbox(description=name, value=(name in ["pyruvate", "lactate"]))
        for name in known_metabolites.keys()
    ]
    confirm_button = widgets.Button(
        description="Confirm Metabolites & Launch Visual Tool", button_style="success"
    )

    # Use a dictionary as a mutable container to retrieve the result from the
    # callback.
    final_params_container = {}

    # --- Callback Function ---
    def on_confirm_clicked(b):
        """This function runs when the confirm button is clicked."""
        # Get selected metabolites from checkboxes.
        selected_metabs = [cb.description for cb in metab_checkboxes if cb.value]
        if not selected_metabs:
            with output:
                clear_output(wait=True)
                print("Error: Please select at least one metabolite.")
            return

        # Prepare parameters for the main setup function.
        params_for_def = initial_params.copy()
        params_for_def["metabs"] = selected_metabs

        # Create the initial `fit_params` Dataset.
        fit_params = def_fit_params(input_data, params_for_def)

        # Launch the visual tool in the output area.
        with output:
            clear_output(wait=True)
            print("Launching visual parameter setup tool...")
            print(
                "Instructions: Click image to select a voxel. Click spectrum to set peak frequency."
            )
            print(
                "Right-click and drag on the spectrum to set the frequency search range."
            )

            # The visual tool modifies the `fit_params` object in-place.
            fit_params_metabs_setup_tool(input_data, fit_params)

            # Store the final, configured object in the container.
            final_params_container["fit_params"] = fit_params
            print(
                "\nVisual setup complete. The function has returned the configured Dataset."
            )

    confirm_button.on_click(on_confirm_clicked)

    # --- Display Widgets and Wait for Completion ---
    display(
        widgets.VBox(
            [
                widgets.Label("Select metabolites to include in the fit:"),
                *metab_checkboxes,
                confirm_button,
            ]
        ),
        output,
    )

    # This loop effectively waits for the user to finish with the interactive tool.
    # It pauses execution of the script until the callback has placed the result
    # into the container.
    import time

    while "fit_params" not in final_params_container:
        time.sleep(0.5)

    return final_params_container["fit_params"]
