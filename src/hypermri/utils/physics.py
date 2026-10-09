"""
Physics-related functions for MRI calculations.

This module contains functions for:
- Magnetic field conversions (gauss_to_kHz, kHz_to_gauss, etc.)
- Gyromagnetic ratios (get_gmr, etc.)
- Nuclear physics (larmor frequency, zeeman energy, thermal polarization)
- Hyperpolarization (hyperpolarized_Mz_T1, etc.)
- Angle conversions (rad_to_deg, deg_to_rad, etc.)
- Flip angle corrections (flipangle_corr, etc.)
- Temperature conversions (celsius_to_kelvin, kelvin_to_celsius)
- RF power calculations (calc_B1_amp_G, calc_fa_deg, rfpower_from_refpower)
"""

# Physics constants
GAMMA_RATIOS = {
    "1H": 267522187,
    "1H (in H2O)": 267522187,
    "2H": 41066225,
    "3H": 285349779,
    "3He": 203789473,
    "7Li": 103977013,
    "13C": 67283474,
    "14N": 19337792,
    "15N": 27126030,
    "17O": 36281518,
    "19F": 251815948,
    "23Na": 70808493,
    "27Al": 69763060,
    "29Si": 53190356,
    "31P": 108394000,
    "57Fe": 8680624,
    "89Y": 13162791,
    "129Xe": 74521644,
    "195Pt": 57715900,
}


import warnings

import matplotlib.pyplot as plt
import numpy as np

from ..utils.system.logging import init_default_logger

logger = init_default_logger(__name__)


def flipangle_corr(T1_obs: float, flipangle: float, TR: float):
    """
    Corrects the observed flipangle for low flipangle experiments with a certain repetition time.

    Parameters
    ----------
    T1_obs : float
        Observed T1 decay constant in seconds, usually around 50 for
        [1-13C]pyruvate.
    flipangle : float
        Flipangle in degree.
    TR : float
        Repetition time in seconds.

    Returns
    -------
    T1 : float
        corrected T1 decay constant

    """
    T1 = 1 / ((1 / T1_obs) + (np.log(np.cos(flipangle * np.pi / 180)) / TR))
    return T1


def gauss_to_kHz(B1_G: float = 0.0, nucleus: str = "13c"):
    """
    Calculate B1_G (Gauss) in kHz.

    Parameters
    ----------
    B1_G: float
        The B1 amplitude in Gauss
    nucleus: str
        can be 1h, 2h, 13c, ...

    Returns
    -------
    The B1 amplitude in kHz

    """
    gmr = get_gmr(nucleus=nucleus, unit="MHz_T")
    return B1_G * gmr


def kHz_to_gauss(B1_kHz: float = 0.0, nucleus: str = "13c"):
    """
    Calculate B1_kHz in Gauss.

    Parameters
    ----------
    B1_kHz: float
        The B1 amplitude in kHz
    nucleus: str
        can be 1h, 2h, 13c, ...

    Returns
    -------
    The B1 amplitude in kHz

    """
    gmr = get_gmr(nucleus=nucleus, unit="MHz_T")
    return B1_kHz / gmr


def deg_to_rad(angle_deg: float = 0.0):
    """
    Convert an angle from degrees to radians.

    Parameters
    ----------
    angle_deg : float, optional
        Angle in degrees. Default is 0.

    Returns
    -------
    float
        The angle in radians.
    """
    return angle_deg / 180 * np.pi


def rad_to_deg(angle_rad: float = 0.0):
    """
    Convert an angle from radians to degrees.

    Parameters
    ----------
    angle_rad : float, optional
        Angle in radians. Default is 0.

    Returns
    -------
    float
        The angle in degrees.
    """
    return angle_rad * 180 / np.pi


def calc_B1_amp_G(
    fa_deg: float = 90.0, trf_ms: float = 1.0, nucleus: str = "13c", sint: float = 1.0
):
    """
    Calculate the B1 amplitude in Gauss.

    Parameters
    ----------
    fa_deg : float, optional
        The flip angle in degrees. Default is 90.
    trf_ms : float, optional
        The pulse duration in milliseconds. Default is 1.
    nucleus : str, optional
        The nucleus type, e.g., '1h', '2h', '13c'. Default is '13c'.
    sint : float, optional
        Integration factor (0-1). Represents the area the RF pulse takes up compared to a block pulse of the same length.
        Default is 1 for a block pulse. For example, sint = 0.762 for a Gaussian of 1ms and fwhm=1/e.

    Returns
    -------
    float
        The B1 amplitude in Gauss [G]. Use `Gauss_to_kHz` to convert the amplitude to kHz.

    """
    # check if integration factor is 0:
    if sint == 0:
        logger.critical("Can't divide by 0, please change integration factor sint!")
        return None
    # check if pulse duration in [ms] is 0:
    if trf_ms == 0:
        logger.warning("Can't divide by 0, please change pulse duration trf_ms!")
        return None

    # gyromagnetic ratio in radian / Tesla / second:
    gmr_rad_Ts = get_gmr(nucleus=nucleus, unit="rad_Ts", db=False)

    # flip-angle  in radian:
    fa_rad = deg_to_rad(angle_deg=fa_deg)

    return (
        fa_rad / (gmr_rad_Ts * trf_ms * 1e-3 * sint) * 1e4
    )  # 1e4 to convert Tesla to Gauss


def calc_fa_deg(
    rfpow_W: float = 1.0,
    reference_pow_W: float = 1.0,
    trf_ms: float = 1.0,
    sint: float = 1.0,
    fa_deg: float = 90.0,
):
    """
    Calculate the flip angle in degrees based on the RF pulse amplitude, reference power, pulse duration, and integration factor.

    Parameters
    ----------
    rfpow_W : float
        RF pulse amplitude in watts.
    reference_pow_W : float
        Reference power for a 1ms pulse with sint=1 and block pulse, resulting in a 90-degree flip angle.
    trf_ms : float
        Pulse duration in milliseconds.
    sint : float
        Integration factor.
    fa_deg: float
        set flipangle in degrees.

    Returns
    -------
    float
        The flip angle in degrees.
    """
    assert trf_ms > 0, "Pulse duration trf_ms must greater 0"
    assert sint > 0, "Integration factor must greater 0"
    # print(f"calc_fa_deg: rfpow_W={rfpow_W}, reference_pow_W={reference_pow_W}, trf_ms={trf_ms}, sint={sint}, fa_deg={fa_deg}")
    # calculate flipangle from input parameters:
    return fa_deg * trf_ms * sint * np.sqrt(rfpow_W / reference_pow_W)


def calc_fa_deg_from_B1(b1_G=0.0, trf_ms=1.0, nucleus="13c", sint=1.0):
    """
    Calculate the flip angle in degrees based on the B1 amplitude, pulse duration, and integration factor.

    Parameters
    ----------
    B1_G : float
        B1 amplitude in Gauss.
    trf_ms : float
        Pulse duration in milliseconds.
    nucleus : str
        Nucleus type, e.g., '1h', '2h', '13c'.
    sint : float
        Integration factor (0-1). Represents the area the RF pulse takes up compared to a block pulse of the same length.
        Default is 1 for a block pulse. For example, sint = 0.762 for a Gaussian of 1ms and fwhm=1/e.

    Returns
    -------
    float
        The flip angle in degrees.
    """
    # gyromagnetic ratio in radian / Tesla / second:
    gmr_rad_Ts = get_gmr(nucleus=nucleus, unit="rad_Ts", db=False)

    # Calculate flip angle in radians
    fa_rad = (
        b1_G * gmr_rad_Ts * trf_ms * 1e-3 * sint / 1e4
    )  # 1e4 to convert Gauss to Tesla

    # Convert to degrees
    fa_deg = rad_to_deg(angle_rad=fa_rad)

    return fa_deg


def get_gmr(nucleus: str = "13c", unit: str = "MHz_T", db: bool = False):
    """
    Get the gyromagnetic ratio of nearly any nucleus.

    Parameters
    ----------
    nucleus: str
        can be 1h, 2h, 13c, ...
    unit: str
        Either MHz/T (MHz_T, "gamma_bar") or 10e6 rad/T/s (rad_Ts)

    Returns
    -------
    The gyromagnetic ratio in standard unit

    """

    # valid nuclei:
    nuclei = [
        "1H",
        "1H (in H2O)",
        "2H",
        "3H",
        "3He",
        "7Li",
        "13C",
        "14N",
        "15N",
        "17O",
        "19F",
        "23Na",
        "27Al",
        "29Si",
        "31P",
        "57Fe",
        "63Cu",
        "67Zn",
        "129Xe",
    ]

    nucleus = nucleus.lower()

    # Checking for valid nucleus and unit
    valid_nuclei = [nuc.lower() for nuc in nuclei]
    valid_units = ["rad_Ts", "MHz_T"]

    # Dictionary with gmr values (taken from wikipedia)
    # https://en.wikipedia.org/wiki/Gyromagnetic_ratio
    gmr_values = {
        "1h_rad_Ts": 267.52218744 * 1e6,
        "1h_MHz_T": 42.577478518,
        "1h_in_h2o_rad_Ts": 267.5153151 * 1e6,
        "1h_in_h2o_MHz_T": 42.57638474,
        "2h_rad_Ts": 41.065 * 1e6,
        "2h_MHz_T": 6.536,
        "3h_rad_Ts": 285.3508 * 1e6,
        "3h_MHz_T": 45.415,
        "3he_rad_Ts": -203.7894569 * 1e6,
        "3he_MHz_T": -32.43409942,
        "7li_rad_Ts": 103.962 * 1e6,
        "7li_MHz_T": 16.546,
        "13c_rad_Ts": 67.2828 * 1e6,
        "13c_MHz_T": 10.7084,
        "14n_rad_Ts": 19.331 * 1e6,
        "14n_MHz_T": 3.077,
        "15n_rad_Ts": -27.116 * 1e6,
        "15n_MHz_T": -4.316,
        "17o_rad_Ts": -36.264 * 1e6,
        "17o_MHz_T": -5.772,
        "19f_rad_Ts": 251.815 * 1e6,
        "19f_MHz_T": 40.078,
        "23na_rad_Ts": 70.761 * 1e6,
        "23na_MHz_T": 11.262,
        "27al_rad_Ts": 69.763 * 1e6,
        "27al_MHz_T": 11.103,
        "29si_rad_Ts": -53.19 * 1e6,
        "29si_MHz_T": -8.465,
        "31p_rad_Ts": 108.291 * 1e6,
        "31p_MHz_T": 17.235,
        "57fe_rad_Ts": 8.681 * 1e6,
        "57fe_MHz_T": 1.382,
        "63cu_rad_Ts": 71.118 * 1e6,
        "63cu_MHz_T": 11.319,
        "67zn_rad_Ts": 16.767 * 1e6,
        "67zn_MHz_T": 2.669,
        "129xe_rad_Ts": -73.997 * 1e6,
        "129xe_MHz_T": -11.777,
    }

    # Constructing the key for the dictionary lookup
    key = f"{nucleus.lower()}_{unit}"

    # check if passed input is valid:
    if nucleus.lower() not in valid_nuclei:
        return f"Unknown nucleus: {nucleus}. Please provide a valid nucleus."
    elif unit not in valid_units:
        return f"Unknown unit: {unit}. Please provide a valid unit (either 'rad_Ts' or 'MHz_T')."
    else:
        pass

    # print which unit is passed:
    if db:
        if unit == "MHz_T":
            print("in [MHz/T]")
        elif unit == "rad_Ts":
            print("in [10e6 rad/T/s]")
        else:
            print("unknown unit")

    # Returning the value if the key exists, otherwise return None
    return gmr_values.get(key, None)


def rfpower_from_refpower(refpower_W=1, fa_deg=90, trf_ms=1, sint=1):
    """
    Calculate the RF power necessary for a pulse.

    Parameters
    ----------
    refpower_W : float, optional
        Reference power for a 90-degree flip angle, 1ms pulse duration, and block pulse (sint=1). Default is 1.
    fa_deg : float, optional
        Flip angle in degrees. Default is 90.
    trf_ms : float, optional
        Pulse duration in milliseconds. Default is 1.
    sint : float, optional
        Integration factor (0-1, with 1 representing a block pulse). Default is 1.

    Returns
    -------
    float
        The RF power in watts.

    """
    assert trf_ms > 0, "Pulse duration trf_ms must greater 0"
    assert sint > 0, "Integration factor must greater 0"
    return refpower_W * (fa_deg / (90 * trf_ms * sint)) ** 2


def celsius_to_kelvin(celsius=0):
    """
    Convert a temperature from Celsius to Kelvin.

    Parameters
    ----------
    celsius : float, optional
        Temperature in Celsius. Default is 0.

    Returns
    -------
    float
        Temperature in Kelvin.
    """
    assert np.all(celsius >= -273.15), "temperature must be greater -273.15C"
    kelvin = celsius + 273.15
    return kelvin


def kelvin_to_celsius(kelvin=273.15):
    """
    Convert a temperature from Kelvin to Celsius.

    Parameters
    ----------
    kelvin : float, optional
        Temperature in Kelvin. Default is 273.15.

    Returns
    -------
    float
        Temperature in Celsius.

    """
    # assert np.all(kelvin >= 0), "temperature must be greater/equal 0K"
    celsius = kelvin - 273.15
    return celsius


def get_zeeman_energy_split(nucleus="13c", fieldstrength_T=7):
    """
    Calculate the energy difference between E_alpha and E_beta of a spin of a nucleus at a given field strength.

    Parameters
    ----------
    nucleus : str, optional
        The type of nucleus (e.g., "13c"). Default is "13c".
    fieldstrength_T : float, optional
        Field strength in Tesla. Default is 7.

    Returns
    -------
    float
        The energy difference in Joules.

    """
    assert fieldstrength_T > 0, "fieldstrength must be greater than 0"

    # define constants:
    planck_constant = 6.62607015e-34  # Planck constant in J*s
    # Larmor frequency in Hz
    larmor_freq_Hz = get_larmor_freq(
        nucleus=nucleus, fieldstrength_T=fieldstrength_T, unit="Hz"
    )
    return planck_constant * larmor_freq_Hz


def get_larmor_freq(nucleus="13c", fieldstrength_T=7, unit="Hz"):
    """
    Calculate the Larmor frequency for a given nucleus at a given field strength.

    Parameters
    ----------
    nucleus : str, optional
        The type of nucleus (e.g., "13c"). Default is "13c".
    fieldstrength_T : float, optional
        Field strength in Tesla. Default is 7.
    unit : str, optional
        Unit for the frequency, either "Hz" or "rad". Default is "Hz".

    Returns
    -------
    float
        The Larmor frequency in the specified unit.

    """
    if unit == "Hz":
        gmr = get_gmr(nucleus=nucleus, unit="MHz_T") * 1e6
    elif unit == "rad":
        gmr = get_gmr(nucleus=nucleus, unit="rad_Ts")
    else:
        raise Exception(f"Unknown unit = {unit}, has to be either Hz or rad!")

    return gmr * fieldstrength_T


def get_thermal_polarisation_ratio(
    nucleus="13c", temperature=230, fieldstrength_T=7, temperature_unit="K"
):
    """
    Calculate the polarisation ratio of spins of a given nucleus at a specified temperature and field strength.

    Parameters
    ----------
    nucleus : str, optional
        The type of nucleus (e.g., "13c"). Default is "13c".
    temperature : float, optional
        Temperature in Kelvin or Celsius. Default is 230.
    fieldstrength_T : float, optional
        Field strength in Tesla. Default is 7.
    temperature_unit : str, optional
        Temperature unit, either "K" (Kelvin) or "C" (Celsius). Default is "K".

    Returns
    -------
    float
        The polarisation ratio.

    """
    assert fieldstrength_T > 0, "fieldstrength must be greater than 0"

    # define constants:
    planck_constant = 6.62607015e-34  # Planck constant in J*s
    boltzmann_constant = 1.380649e-23  # Boltzmann constant in J/K
    h_bar = planck_constant / (2 * np.pi)  # Reduced Planck constant

    if temperature_unit == "K":
        pass
    elif temperature_unit == "C":
        temperature = celsius_to_kelvin(celsius=temperature)
    else:
        raise ValueError("temperature unit must be either K or C!")

    # calculate energy of zeeman splitting at fieldstrength:
    dE = get_zeeman_energy_split(nucleus=nucleus, fieldstrength_T=fieldstrength_T)

    population_ratio = np.exp(-dE / (boltzmann_constant * temperature))

    return population_ratio


def get_thermal_polarisation_ppm(
    nucleus="13c", temperature=230, fieldstrength_T=7, temperature_unit="K"
):
    """
    Get the number of spins (in ppm) that contribute to the NMR signal.

    Parameters
    ----------
    nucleus : str, optional
        The type of nucleus (e.g., "13c"). Default is "13c".
    temperature : float, optional
        Temperature in Kelvin or Celsius. Default is 230.
    fieldstrength_T : float, optional
        Field strength in Tesla. Default is 7.
    temperature_unit : str, optional
        Temperature unit, either "K" (Kelvin) or "C" (Celsius). Default is "K".

    Returns
    -------
    float
        The polarisation in ppm.

    """
    # ratio of spin up / spin down:
    population_ratio = get_thermal_polarisation_ratio(
        nucleus=nucleus,
        temperature=temperature,
        fieldstrength_T=fieldstrength_T,
        temperature_unit=temperature_unit,
    )

    # polarisation in ppm:
    polarisation_ppm = (1 - population_ratio) * 1e6

    return polarisation_ppm


def hyperpolarized_Mz_flipangle(
    M0_au=1,
    n_excitations=100,
    flipangle_deg=10,
):
    """
    Calculate the hyperpolarized magnetization decay due to RF pulses alone.

    This function uses an iterative approach to model the Mz before each
    pulse and the Mxy generated by each pulse.

    Parameters
    ----------
    M0_au : float
        Initial magnetization in arbitrary unit.
    n_excitations : int
        Number of excitations (pulses).
    flipangle_deg : float or np.ndarray
        Flip angle in degrees. Can be a single value for a constant scheme,
        or an array of values for a variable scheme.

    Returns
    -------
    Mz : np.ndarray
        Calculated longitudinal magnetization (Mz) immediately before each pulse.
    Mxy : np.ndarray
        Calculated transverse magnetization (Mxy) immediately after each pulse.

    """
    # Handle the flip angle input
    if np.isscalar(flipangle_deg):
        flip_angles_rad = np.deg2rad(np.full(n_excitations, flipangle_deg))
    else:
        if len(flipangle_deg) != n_excitations:
            raise ValueError("Length of flipangle_deg array must equal n_excitations.")
        flip_angles_rad = np.deg2rad(np.array(flipangle_deg))

    # Initialize arrays
    Mz = np.zeros(n_excitations)
    Mxy = np.zeros(n_excitations)

    # Set the initial magnetization
    current_mz = M0_au

    # Loop through each excitation
    for n in range(n_excitations):
        # Mz immediately before the pulse
        Mz[n] = current_mz
        # Mxy generated by the pulse
        Mxy[n] = current_mz * np.sin(flip_angles_rad[n])
        # Mz remaining immediately after the pulse
        current_mz *= np.cos(flip_angles_rad[n])

    return Mz, Mxy


def hyperpolarized_Mz_T1(M0_au, t_s, T1_s):
    """
    Calculate the hyperpolarized magnetization decay due to T1 relaxation alone.

    Parameters
    ----------
    M0_au : float
        Initial magnetization in arbitrary unit.
    t_s : array_like
        Time points in seconds.
    T1_s : float
        T1 relaxation time in seconds.

    Returns
    -------
    Mz : np.ndarray
        Calculated longitudinal magnetization (Mz) values.
    Mxy : np.ndarray
        Transverse magnetization (Mxy), which is zero for T1 decay alone.

    """
    Mz = M0_au * np.exp(-np.array(t_s) / T1_s)
    # T1 relaxation affects Mz only; it does not generate transverse
    # magnetization.
    Mxy = np.zeros_like(Mz)
    return Mz, Mxy


def estimate_noise(
    input_data: np.ndarray = None,
    mask_dict: dict = None,
    mask_key: str = None,
    mask: str = None,
    masks_drawn_on="axial",
    slice_ind=None,
    signal_domain="temporal",
    component="complex",
    in_mask=False,
    plot=False,
    **kwargs,
):
    """
    Estimate the standard deviation of noise from MRI images.

    This function calculates the standard deviation of noise in MRI images
    using a specified mask or performs the calculations on all voxels if no mask is provided.
    Assumes first dimension to be in spectral dimension (to temporal=FID).
    The returned standard deviation of the noise is calculated from the temporal dimension
    to be easily usable with fitted amplitudes from utils_fitting.fit_data_pseudo_inv.

    Parameters
    ----------
    input_data : ndarray, optional
        The MRI data from which to estimate noise. The data should be a 5D
        ndarray.
    mask_dict : dict, optional
        A dictionary containing masks. A specific mask is selected using `mask_key`.
    mask_key : str, optional
        The key corresponding to the mask to be used from `mask_dict`.
    mask : ndarray, optional
        An alternative to using `mask_dict` and `mask_key`. Directly specify the mask to use.
        All voxels will be considered if no mask is provided.
    masks_drawn_on : str, default 'axial'
        The orientation of the masks.
    slice_ind : int, optional
        The specific slice index to use from `input_data`. Ignored if no mask is provided.
    signal_domain : str, optional
        input_data signal domain, can be spectral (default) or temporal. If spectral, an inverse FT is performed.
    in_mask: bool, optional
        Whether the noise is in the mask or outside the mask. Default is False.
    plot : bool, default False
        If True, plots the mask, real and imaginary components of the noise signal, and their histograms.
    kwargs : dict, optional
        Additional keyword arguments for extended functionality. Supported keywords include `return_all` to return all results.

    Returns
    -------
    std_noise : float
        The estimated standard deviation of the noise, calculated as the average of the standard
        deviations of the real and imaginary components of the signal.

    """
    from ..utils.mathematics.utils_fit import fit_rayleigh, rayleigh_pdf_fit

    # Validate input_data
    if input_data is None:
        raise ValueError("input_data must be provided!")

    # Process mask or use all voxels if no mask is provided
    if mask is None:
        if mask_dict is not None and mask_key is not None:
            try:
                mask = mask_dict[mask_key]
            except KeyError:
                raise ValueError(f"Cannot load mask_key {mask_key} from mask_dict!")
        else:
            # Use all voxels and skip slice operations
            mask = None
            warnings.warn("No mask provided, using all voxels for noise estimation!")

    # Fourier transform based on signal domain
    if signal_domain == "spectral":
        input_data = np.fft.ifft(np.fft.ifftshift(input_data, axes=0), axis=0)

    # If mask is provided, use mask operations; otherwise, just flatten data
    if mask is not None:
        mask_inv = mask if in_mask else np.ones_like(mask) - mask

        # Apply mask and calculate noise
        noise = apply_mask(
            input_data,
            mask=mask_inv,
            mask_slice_ind=slice_ind,
            return_nans=True,
            masks_drawn_on=masks_drawn_on,
            bitwise=True,
        )
        noise_signal = noise[~np.isnan(noise)].flatten()
    else:
        noise_signal = input_data.flatten()

    # Binned dataset for noise distribution
    _, bins = np.histogram(np.abs(noise_signal), bins=30, density=1.0)
    # centers = (bins[:-1] + bins[1:]) / 2
    xlin = np.linspace(bins[0], bins[-1], 100)

    popt, _ = fit_rayleigh(input_data=np.abs(noise_signal), plot=False)

    # Calculate the mean and standard deviation of noise
    std_real = np.nanstd(np.real(noise_signal))
    std_imag = np.nanstd(np.imag(noise_signal))
    mean_real = np.nanmean(np.real(noise_signal))
    mean_imag = np.nanmean(np.imag(noise_signal))

    # Plotting section, if requested
    if plot:
        _, ax = plt.subplots(1, 5, tight_layout=True, figsize=(8, 3))

        if mask is not None:
            if masks_drawn_on == "axial":
                im = ax[0].imshow(
                    np.rot90(np.squeeze(mask_inv[0, slice_ind, :, :, 0, 0]))
                )
                plt.colorbar(im, ax=ax[0], label="Mask Coverage", shrink=0.3)
                ax[0].set_title("Inverse Signal Mask")
            else:
                im = ax[0].imshow(
                    np.rot90(np.squeeze(mask_inv[0, :, :, slice_ind, 0, 0]))
                )
                plt.colorbar(im, ax=ax[0], label="Mask Coverage", shrink=0.3)
                ax[0].set_title("Inverse Signal Mask")
        else:
            ax[0].text(0.5, 0.5, "No Mask Applied", ha="center", va="center")
            ax[0].set_title("All Voxels Used")

        ax[1].plot(np.real(noise_signal), color="blue")
        ax[1].plot(np.imag(noise_signal), color="red", alpha=0.5)
        ax[1].set_title("Signal Real/Imaginary")

        ax[2].hist(np.real(noise_signal), bins=30, color="blue", density=1.0)
        ax[2].hist(np.imag(noise_signal), bins=30, color="red", alpha=0.5, density=1)
        ax[2].set_title("Histogram Real/Imaginary")

        ax[3].plot(np.abs(noise_signal))
        ax[3].set_title("Signal Magnitude")

        ax[4].hist(np.abs(noise_signal), bins=30, density=1)
        ax[4].plot(xlin, rayleigh_pdf_fit(xlin, *popt), color="r")
        ax[4].axvline(np.mean(np.abs(noise_signal)), color="k")
        ax[4].axvline(popt[0], color="r")
        ax[4].axvline(popt[1], color="r")
        ax[4].set_title("Histogram Magnitude")
        plt.show()

    # Output the mean and standard deviation of noise
    print(f"Mean noise: {mean_real:.2f} + {mean_imag:.2f}*1i")
    print(f"Mean |noise|: {np.mean(np.abs(noise_signal))}")
    print(f"Standard deviation of noise: σ = {std_real:.2f} + {std_imag:.2f}*1i")
    print(f"Rayleigh Fit of |noise|: σ = {popt[1]:.2f}")

    # Determine the noise standard deviation based on the component setting
    if component == "complex":
        std_noise = (std_real + std_imag) / 2
    elif component in ["magnitude", "mag"]:
        std_noise = popt[1]
    else:
        std_noise = (std_real + std_imag) / 2

    if "return_all" in kwargs and kwargs["return_all"]:
        results = {}
        results["mean_real"] = mean_real
        results["mean_imag"] = mean_imag
        results["std_real"] = std_real
        results["std_imag"] = std_imag
        results["std_noise"] = std_noise
        results["rayleigh_fit"] = popt
        return results

    return std_noise


def hyperpolarized_Mz_flipangle_T1(
    M0_au: float = 1e5,
    n_excitations: int = 100,
    TR_s: float = 0.20,
    T1_s: float = 30.0,
    flipangle_deg: float = None,
    plot: bool = True,
    interactive: bool = False,
    data_obj=None,
    plot_params: dict = {},
):
    """
    Calculate and optionally plot the hyperpolarized signal considering both
    flip angle and T1 relaxation using an accurate iterative model.

    Parameters
    ----------
    M0_au : float, optional
        Initial magnetization. Default is 1e5.
    n_excitations : int, optional
        Number of excitations. Default is 100.
    TR_s : float, optional
        Repetition time in seconds. Default is 0.2.
    T1_s : float, optional
        T1 relaxation time in seconds. Default is 30.
    flipangle_deg : float, np.ndarray, optional
        Flip angle in degrees (scalar or array). Defaults to 5.
    plot : bool, optional
        If True, create a plot. Default is True.
    interactive : bool, optional
        If True, create an interactive plot. Default is False.
    data_obj : hypermri object, optional
        If passed, parameters will be taken from the object.
    plot_params : dict, optional
        Additional parameters for plotting.

    Returns
    -------
    Mz : ndarray
        Calculated longitudinal magnetization Mz values before each pulse.
    Mxy : ndarray
        Calculated transverse magnetization Mxy values after each pulse.
    t_s : ndarray
        Time points in seconds.

    """
    # Parameter handling from data_obj or defaults
    if data_obj:
        TR_s = data_obj.repetition_time_ms / 1000.0
        n_excitations = np.prod(data_obj.matrix) * data_obj.n_repetitions
        if flipangle_deg is None:
            flipangle_deg = data_obj.flipangle_deg

    if flipangle_deg is None:
        flipangle_deg = 12.0  # Default value if still None

    # This is the core calculation function, now separate for clarity
    def _calculate_signal(m0, t1, tr, fa_deg, n_exc):
        # Handle scalar or array flip angle input
        if np.isscalar(fa_deg):
            flip_angles_rad = np.deg2rad(np.full(n_exc, fa_deg))
        else:
            if len(fa_deg) != n_exc:  # Ensure array has correct length
                fa_deg_resampled = np.interp(
                    np.arange(n_exc), np.linspace(0, n_exc, len(fa_deg)), fa_deg
                )
            else:
                fa_deg_resampled = fa_deg
            flip_angles_rad = np.deg2rad(np.array(fa_deg_resampled))

        # Initialize arrays
        t = np.arange(n_exc) * tr
        mz = np.zeros(n_exc)
        mxy = np.zeros(n_exc)

        current_mz = m0
        for n in range(n_exc):
            mz[n] = current_mz
            mxy[n] = current_mz * np.sin(flip_angles_rad[n])
            mz_after_pulse = current_mz * np.cos(flip_angles_rad[n])
            current_mz = mz_after_pulse * np.exp(-tr / t1)

        return mz, mxy, t

    # Perform initial calculation
    Mz, Mxy, t_s = _calculate_signal(M0_au, T1_s, TR_s, flipangle_deg, n_excitations)

    if plot:
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
        ]
        figsize = plot_params.get("figsize", (8, 5))

        if interactive:
            from IPython.display import display
            from ipywidgets import (
                Checkbox,
                FloatSlider,
                HBox,
                IntSlider,
                Layout,
                Output,
                VBox,
            )

            # Widgets
            w_m0 = FloatSlider(
                min=1e4, max=1e6, step=1e4, value=M0_au, description="M0"
            )
            w_exc = IntSlider(
                min=10,
                max=500,
                step=10,
                value=n_excitations,
                description="# Excitations",
            )
            w_tr = FloatSlider(
                min=0.01, max=2, step=0.001, value=TR_s, description="TR (s)"
            )
            w_t1 = FloatSlider(min=1, max=100, step=1, value=T1_s, description="T1 (s)")
            w_fa = FloatSlider(
                min=0,
                max=90,
                step=1,
                value=flipangle_deg if np.isscalar(flipangle_deg) else 12,
                description="Flip Angle (°)",
            )
            w_comp = Checkbox(value=True, description="Show components")

            plot_out = Output()

            plt.ioff()
            fig, ax1 = plt.subplots(figsize=figsize, constrained_layout=True)
            ax2 = ax1.twiny()
            plt.ion()

            with plot_out:
                display(fig.canvas)

            def update_plot(change=None):
                M0_au_w = w_m0.value
                n_excitations_w = w_exc.value
                TR_s_w = w_tr.value
                T1_s_w = w_t1.value
                flipangle_deg_w = w_fa.value
                show_components = w_comp.value

                mz_plot, mxy_plot, t_plot = _calculate_signal(
                    M0_au_w, T1_s_w, TR_s_w, flipangle_deg_w, n_excitations_w
                )

                ax1.clear()
                ax2.clear()

                # Plot Mz and Mxy
                ax1.plot(t_plot, mz_plot, label="$M_z$ (pre-pulse)", color=colors[0])
                ax1.plot(
                    t_plot,
                    mxy_plot,
                    label="$M_{xy}$ (signal)",
                    color=colors[-1],
                    linestyle="--",
                    marker="o",
                    markersize=4,
                )

                # Optional: show the individual decay components
                if show_components:
                    mz_t1_only, _ = hyperpolarized_Mz_T1(M0_au_w, t_plot, T1_s_w)
                    ax1.plot(
                        t_plot,
                        mz_t1_only,
                        label=f"T1 Decay Only (T1={T1_s_w:.1f}s)",
                        color=colors[1],
                        linestyle=":",
                        alpha=0.8,
                    )
                    mz_fa_only, _ = hyperpolarized_Mz_flipangle(
                        M0_au_w, n_excitations_w, flipangle_deg_w
                    )
                    ax1.plot(
                        t_plot,
                        mz_fa_only,
                        label=f"Flip Angle Decay Only ({flipangle_deg_w:.1f}°)",
                        color=colors[2],
                        linestyle=":",
                        alpha=0.8,
                    )

                eff_t1_val_at_m0e = M0_au_w / np.e
                eff_T1_s_idx = np.argmin(np.abs(mz_plot - eff_t1_val_at_m0e))
                eff_T1_s = t_plot[eff_T1_s_idx]

                ax1.vlines(
                    eff_T1_s,
                    0,
                    eff_t1_val_at_m0e,
                    color="r",
                    linestyle="--",
                    label=rf"$M_0/e$ (Effective T1 $\approx$ {eff_T1_s:.1f}s)",
                )

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Magnetization (a.u.)")
                ax1.set_title("Hyperpolarized Magnetization Simulation")
                ax1.legend()
                ax1.set_ylim(0, M0_au_w * 1.1)
                ax1.set_xlim(0, t_plot[-1] if len(t_plot) > 1 else 1)

                ax2.set_xlim(ax1.get_xlim())
                tick_indices = np.linspace(0, n_excitations_w - 1, 11, dtype=int)
                ax2.set_xticks(t_plot[tick_indices])
                ax2.set_xticklabels(tick_indices)
                ax2.set_xlabel("Excitation Number")

                fig.canvas.draw_idle()

            # Observe changes
            for w in [w_m0, w_exc, w_tr, w_t1, w_fa, w_comp]:
                w.observe(update_plot, names="value")

            # Initial Plot
            update_plot()

            # Layout
            sidebar = VBox(
                [w_m0, w_exc, w_tr, w_t1, w_fa, w_comp],
                layout=Layout(width="300px", margin="0 20px 0 0"),
            )
            ui = HBox([sidebar, plot_out])
            display(ui)

        else:
            # Non-interactive plot
            plt.ioff()
            fig, ax1 = plt.subplots(figsize=figsize)
            plt.ion()
            ax2 = ax1.twiny()

            flipangle_deg if np.isscalar(flipangle_deg) else "VFA"

            # Use the inner function for logic, but need to adapt args slightly since update_plot expects widgets
            # Actually easier to just copy the plotting logic briefly or adapt the call
            mz_plot, mxy_plot, t_plot = Mz, Mxy, t_s  # Already calculated above

            ax1.plot(t_plot, mz_plot, label="$M_z$ (pre-pulse)", color=colors[0])
            ax1.plot(
                t_plot,
                mxy_plot,
                label="$M_{xy}$ (signal)",
                color=colors[-1],
                linestyle="--",
                marker="o",
                markersize=4,
            )

            # Components (always on for non-interactive default?) - let's keep it simple as before
            # The original code passed show_components=True for non-interactive
            mz_t1_only, _ = hyperpolarized_Mz_T1(M0_au, t_plot, T1_s)
            ax1.plot(
                t_plot,
                mz_t1_only,
                label=f"T1 Decay Only (T1={T1_s:.1f}s)",
                color=colors[1],
                linestyle=":",
                alpha=0.8,
            )

            mz_fa_only, _ = hyperpolarized_Mz_flipangle(
                M0_au,
                n_excitations,
                flipangle_deg if np.isscalar(flipangle_deg) else flipangle_deg[0],
            )  # Simplified for VFA array handling in this call
            ax1.plot(
                t_plot,
                mz_fa_only,
                label=f"Flip Angle Decay Only",
                color=colors[2],
                linestyle=":",
                alpha=0.8,
            )

            eff_t1_val_at_m0e = M0_au / np.e
            eff_T1_s_idx = np.argmin(np.abs(mz_plot - eff_t1_val_at_m0e))
            eff_T1_s = t_plot[eff_T1_s_idx]

            ax1.vlines(
                eff_T1_s,
                0,
                eff_t1_val_at_m0e,
                color="r",
                linestyle="--",
                label=rf"$M_0/e$ (Effective T1 $\approx$ {eff_T1_s:.1f}s)",
            )

            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Magnetization (a.u.)")
            ax1.set_title("Hyperpolarized Magnetization Simulation")
            ax1.legend()
            ax1.set_ylim(0, M0_au * 1.1)
            ax1.set_xlim(0, t_plot[-1] if len(t_plot) > 1 else 1)

            ax2.set_xlim(ax1.get_xlim())
            tick_indices = np.linspace(0, n_excitations - 1, 11, dtype=int)
            ax2.set_xticks(t_plot[tick_indices])
            ax2.set_xticklabels(tick_indices)
            ax2.set_xlabel("Excitation Number")

            plt.show()

    if interactive:
        return
    else:
        return Mz, Mxy, t_s
