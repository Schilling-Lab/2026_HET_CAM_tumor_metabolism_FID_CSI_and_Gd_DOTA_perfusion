"""
Calibration functions for MR spectroscopy.

Contains functions for temperature and pH calibration based on chemical shifts.
"""

import numpy as np

from ..system.logging import init_default_logger

logger = init_default_logger(__name__)


def ppm_from_temp_and_concentration(c="5mM", t=0):
    """
    Calculate the ppm value based on the concentration and temperature.

    Parameters
    ----------
    c : str, optional
        The concentration of the solution. Supported values are '5mM', '10mM', '20mM', '50mM', '100mM', '200mM', '400mM', '600mM'.
    t : float, optional
        The temperature in degrees Celsius.

    Returns
    -------
    float
        The calculated ppm value.

    Raises
    ------
    ValueError
        If an unsupported concentration is provided.
    """
    # Mapping concentrations to their respective parameters
    concentration_params = {
        "5mM": (-0.01297, 12.687991248319072),
        "10mM": (-0.01298, 12.69612502856522),
        "20mM": (-0.01311, 12.712149898413177),
        "50mM": (-0.01339, 12.755002271910573),
        "100mM": (-0.01373, 12.814997258304782),
        "200mM": (-0.01441, 12.926556654978315),
        "400mM": (-0.01544, 13.118609741470681),
        "600mM": (-0.01623, 13.288926529656862),
    }

    if c in concentration_params:
        slope, intercept = concentration_params[c]
        return slope * t + intercept
    else:
        raise ValueError(
            "Unsupported concentration. Please choose from '5mM', '10mM', '20mM', '50mM', '100mM', '200mM', '400mM', '600mM'."
        )


def get_temperature_from_frequency(
    frequency: float = None,
    calibration_type: str = "blood",
    frequency_is_ppm: bool = True,
):
    """
    Returns the temperature of a pyruvate-lactate solution for a desired calibration function.

    Parameters
    ----------
    frequency: float
        Pyruvate-lactate relative chemical shift in Hz or ppm.
    calibration_type: str, optional
        Type of calibration to use. Default is 'blood'.
    frequency_is_ppm: bool, optional
        If True, frequency is in ppm. If False, frequency is in Hz (at 7T).

    Returns
    -------
    temperature, temperature_error: float
        Calculated temperature and its error.
    """

    def linear(f, pitch, crossing):
        return (f - crossing) / pitch

    def err_linear(f, pitch, pitch_error, crossing, crossing_err):
        return np.sqrt(
            (crossing_err / pitch) ** 2 + (pitch_error * ((crossing - f) / pitch)) ** 2
        )

    if (
        calibration_type
        not in ["cryo_thermal", "LDH", "Blood", "Blood_meaned", "LDH_meaned"]
        and frequency_is_ppm == False
    ):
        raise ValueError(
            "You must input the frequency in ppm for calibrations at 11.9T"
        )
    if frequency_is_ppm:
        if calibration_type == "cryo_thermal":
            pitch = -0.016321184066004
            pitch_error = 0.001151351715244
            crossing = 12.843448632366941
            crossing_error = 4.4252367604e-05
        elif calibration_type == "LDH":
            pitch = -0.01472863465268
            pitch_error = 0.009713604636182
            crossing = 12.780660470812476
            crossing_error = 0.000357109774727
        elif calibration_type == "LDH_meaned":
            pitch = -0.014386495546791
            pitch_error = 0.056925861243265
            crossing = 12.763982363066608
            crossing_error = 0.00202276880719
        elif calibration_type == "Blood":
            pitch = -0.013011837418234
            pitch_error = 0.03850878645128
            crossing = 12.737055542464372
            crossing_error = 0.001205824445164
        elif calibration_type == "Blood_meaned":
            pitch = -0.014365470412533
            pitch_error = 0.067207631500598
            crossing = 12.765264460856539
            crossing_error = 0.002331100420903
        elif calibration_type == "5mM":
            pitch = -0.012967023642931
            pitch_error = 0.001611918011057
            crossing = 12.687991248319072
            crossing_error = 0.000499970341949
        elif calibration_type == "10mM":
            pitch = -0.012976898165725
            pitch_error = 0.003458085864921
            crossing = 12.69612502856522
            crossing_error = 0.001072727320076
        elif calibration_type == "20mM":
            pitch = -0.013109496965684
            pitch_error = 0.00236649843376
            crossing = 12.712149898413177
            crossing_error = 0.000740674010679
        elif calibration_type == "50mM":
            pitch = -0.013385022660291
            pitch_error = 0.003073372872851
            crossing = 12.755002271910573
            crossing_error = 0.000978830935923
        elif calibration_type == "100mM":
            pitch = -0.013731512844978
            pitch_error = 0.005107782357626
            crossing = 12.814997258304782
            crossing_error = 0.001661063045042
        elif calibration_type == "200mM":
            pitch = -0.014408217740115
            pitch_error = 0.007887358921608
            crossing = 12.926556654978315
            crossing_error = 0.00266816618565
        elif calibration_type == "400mM":
            pitch = -0.01544148517688
            pitch_error = 0.009940461235411
            crossing = 13.118609741470681
            crossing_error = 0.003551089800098
        elif calibration_type == "600mM":
            pitch = -0.016231456018516
            pitch_error = 0.012464416792304
            crossing = 13.288926529656862
            crossing_error = 0.00462054712916
        else:
            logger.warning("Calibration type not recognized, defaulting to Blood at 7T")
            pitch = -0.013011837418234
            pitch_error = 0.03850878645128
            crossing = 12.737055542464372
            crossing_error = 0.001205824445164

    else:
        # input in Hz at 7T
        if calibration_type == "cryo_thermal":
            pitch = -1.232111910557836
            pitch_error = 0.001151351782786
            crossing = 969.5721817181965
            crossing_error = 4.4252370713e-05
        elif calibration_type == "LDH":
            pitch = -1.111887813410422
            pitch_error = 0.009713604908939
            crossing = 964.8322034375896
            crossing_error = 0.000357109790553
        elif calibration_type == "LDH_meaned":
            pitch = -1.086059180561179
            pitch_error = 0.056925856223268
            crossing = 963.57314641691
            crossing_error = 0.00202276880719
        elif calibration_type == "Blood":
            pitch = -0.982284162038326
            pitch_error = 0.038508790923977
            crossing = 961.5404010527442
            crossing_error = 0.001205824603659
        elif calibration_type == "Blood_meaned":
            pitch = -1.084472026996133
            pitch_error = 0.067207630777528
            crossing = 963.6699360643906
            crossing_error = 0.002331100486608
        else:
            logger.warning("Calibration type not recognized, defaulting to Blood at 7T")
            pitch = -0.982284162038326
            pitch_error = 0.038508790923977
            crossing = 961.5404010527442
            crossing_error = 0.001205824603659

    temperature, temperature_error = linear(frequency, pitch, crossing), err_linear(
        frequency, pitch, pitch_error, crossing, crossing_error
    )

    return temperature, temperature_error


def get_pH_from_OMPD(peak_diff_ppm, err_peak_diff):
    """
    Calculate pH for Z-OMPD from the chemical shift difference in OMPD1 and OMPD 5 peaks

    Parameters
    ----------
    peak_diff_ppm: float
        Difference between OMPD 1 and 5 peak in ppm.
    err_peak_diff: float
        Error in difference between OMPD 1 and 5 peak from fit.

    Returns
    -------
    pH, pH_error
    """
    pKa = 6.5450
    A = 3.2975
    B = 4.4801
    err_ph = np.abs(
        (A / (np.log(10) * (B - peak_diff_ppm) * (A + B - peak_diff_ppm)))
        * err_peak_diff
    )
    return pKa - np.log10((A / (peak_diff_ppm - B)) - 1), err_ph


def get_pH(peak_diff, err_peak_diff, compound="ompd"):
    """
    Calculate pH from chemical shift difference.

    Parameters
    ----------
    peak_diff: float
        Difference between peaks in ppm.
    err_peak_diff: float
        Error in difference between peaks.
    compound: str
        Type of compound (e.g., 'ompd').

    Returns
    -------
    pH, pH_error

    """
    if compound.lower() == "ompd":
        return get_pH_from_OMPD(peak_diff, err_peak_diff)
    else:
        raise ValueError("Unsupported compound. Currently only 'ompd' is supported.")
