from typing import Tuple, Union

import numpy as np

# Larmor Frequencies (MHz) at 7 Tesla from here:
# https://kodu.ut.ee/~laurit/AK2/NMR_tables_Bruker2012.pdf
LAMOR_FREQUENCY_H1 = 300.130
LAMOR_FREQUENCY_C13 = 75.468
LAMOR_FREQUENCY_P31 = 121.495

# [pK, δHA, δA = δHA - δ_range] - Reference: Urea
PARAMS_OMPD_C1 = 6.5450, 8.4919, 8.4919 - 0.4644

# [pK, δHA, δA = δHA + δ_range] - Reference: Urea
PARAMS_OMPD_C5 = 6.5450, 12.4801, 12.4801 + 3.2975

# [pK, δHA, δA]
PARAMS_PHOSPHOR = [6.77, 3.23, 5.70]

# [pK, δHA, δA] # Reference Urea?
PARAMS_DIMEZA = [7.3, 172.5, 177.71]

#
PARAMS_DIMEZA_NO_UREA = [7.3, 172.5 - 163.5, 177.71 - 163.5]


def pH2ppm(ph_value, params):
    """Reversed Henderson-Hasselbach relationship to convert pH to chemical shift."""
    pK, dHA, dA = params
    return dHA + (dA - dHA) / (1 + 10 ** (-(ph_value - pK)))


def ppm2pH(
    freq_diff_ppm: Union[float, np.ndarray] = None,
    params: Tuple[float, float, float] = None,
) -> Union[float, np.ndarray]:
    """Henderson-Hasselbach relationship to convert chemical shift to pH.

    Parameters
    ----------
    freq_diff_ppm : float or np.ndarray
        Frequency difference of the pH sensitive peaks in ppm.
    params : tuple of float
        Tuple of three floats: (pKa, dHA_ppm, dA_ppm)
        where dHA_ppm is the chemical shift of the protonated form,
        and dA_ppm is the chemical shift of the deprotonated form.

    Returns
    -------
    float or np.ndarray
        The calculated pH value(s).

    Examples
    --------
    >>> from hypermri.functions.pHconverter import ppm2pH, PARAMS_OMPD_C5
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> freq_diff_ppm = np.linspace(0, 14, 100)
    >>> pH = ppm2pH(freq_diff_ppm, PARAMS_OMPD_C5)
    >>> plt.plot(freq_diff_ppm, pH)
    >>> plt.xlabel('Chemical Shift (ppm)')
    >>> plt.ylabel('pH')
    >>> plt.title('pH vs Chemical Shift for OMPD C5')

    >>> # After fitting data (metabolite names Urea and OMPD5, data_obj csi)
    >>> from hypermri.analysis.spectroscopy.core import freq_Hz_to_ppm
    >>> freq_ph = freq_Hz_to_ppm(fit_results['fit_freqs'].sel(metabolite="OMPD5").squeeze(), data_obj=csi)
    >>> freq_urea = freq_Hz_to_ppm(fit_results['fit_freqs'].sel(metabolite="Urea").squeeze(), data_obj=csi)
    >>> fig, ax = plt.subplots(1)
    >>> ax.imshow(ppm2pH(freq_diff_ppm = freq_ph-freq_urea, params=PARAMS_OMPD_C5)))

    """
    pKa, dHA_ppm, dA_ppm = params
    return pKa + np.log10((freq_diff_ppm - dHA_ppm) / (dA_ppm - freq_diff_ppm))


def freq2pH(frequency, params, lamor_frequency):
    """Henderson-Hasselbach relationship to convert chemical shift (given in Hz) to pH."""
    return ppm2pH(frequency / lamor_frequency, params)


def pH2freq(ppm, params, lamor_frequency):
    """Henderson-Hasselbach relationship to convert pH to chemical shift (given in Hz)."""
    return ppm2pH(ppm, params) * lamor_frequency


def OMPD5_freq2pH(freq, params=PARAMS_OMPD_C5, lamor_frequency=LAMOR_FREQUENCY_C13):
    """Convert frequency of OMPD C5 peak into pH."""
    return freq2pH(freq, params, lamor_frequency)
