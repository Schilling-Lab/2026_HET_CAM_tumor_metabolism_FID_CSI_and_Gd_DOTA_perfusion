import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats


def rayleigh_pdf_fit(x, mu, sigma):
    """Evaluate a Rayleigh probability density function.

    Parameters
    ----------
    x : array_like
        Sample locations at which the PDF is evaluated.
    mu : float
        Location parameter of the Rayleigh distribution.
    sigma : float
        Scale parameter of the Rayleigh distribution.

    Returns
    -------
    numpy.ndarray
        Rayleigh PDF values corresponding to ``x``.

    Notes
    -----
    This function is tailored for use with :func:`scipy.optimize.curve_fit`.
    """

    return stats.rayleigh.pdf(x, loc=mu, scale=sigma)


def fit_rayleigh(input_data=None, plot=True, **kwargs):
    """Fit a Rayleigh distribution to the magnitude of complex data.

    Parameters
    ----------
    input_data : array_like, optional
        Complex-valued data from which the magnitude will be modelled. When
        ``None`` a :class:`ValueError` is raised.
    plot : bool, optional
        If ``True`` the sample magnitude and fitted PDF are plotted. Default is
        ``True``.
    **kwargs
        Additional keyword arguments:

        ``bins`` : int, optional
            Number of histogram bins. Default is 30.
        ``density`` : bool, optional
            If ``True`` the histogram is normalised. Default is ``True``.
        ``p0`` : tuple of float, optional
            Initial guesses for ``(mu, sigma)``.

    Returns
    -------
    popt : numpy.ndarray
        Optimal parameters ``(mu, sigma)`` of the Rayleigh distribution.
    pcov : numpy.ndarray
        Estimated covariance matrix of ``popt``.

    Raises
    ------
    ValueError
        If ``input_data`` is ``None``.

    Notes
    -----
    Histograms are computed on the absolute value of ``input_data``. When
    plotting is enabled the function creates two subplots: one for the signal
    magnitude and one for the histogram with the fitted PDF.
    """

    if input_data is None:
        raise ValueError("input_data has to be passed!")

    input_data = np.abs(
        input_data
    ).flatten()  # Ensure input is flattened and magnitude is taken

    # Extract or set default keyword arguments
    bins = kwargs.get("bins", 30)
    plot_density = kwargs.get(
        "density", True
    )  # Renamed to avoid confusion with histogram output
    p0 = kwargs.get("p0", [np.mean(input_data), np.std(input_data)])

    # Compute histogram density and bins
    hist_density, bins = np.histogram(input_data, bins=bins, density=plot_density)
    centers = (bins[:-1] + bins[1:]) / 2
    xlin = np.linspace(bins[0], bins[-1], 100)

    # Fit Rayleigh distribution
    popt, pcov = optimize.curve_fit(rayleigh_pdf_fit, centers, hist_density, p0=p0)

    if plot:
        fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(12, 6))
        ax[0].plot(input_data)
        ax[0].set_title("Signal Magnitude")
        ax[1].hist(input_data, bins=bins, density=plot_density)
        ax[1].plot(xlin, rayleigh_pdf_fit(xlin, *popt), color="r")
        ax[1].axvline(np.mean(input_data), color="k", label="Mean")
        ax[1].axvline(popt[0], color="r", label="Fitted Mu")
        ax[1].axvline(popt[1], color="g", label="Fitted Sigma")
        ax[1].set_title("Histogram Magnitude")
        ax[1].legend()

    return popt, pcov
