import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as co
from scipy.optimize import curve_fit

from hypermri.utils.mathematics.calculations import get_histogram
from hypermri.utils.physics import flipangle_corr


def calculate_polarization_level_magnitude_spectra(
    thermal_data,
    hyper_data,
    time_to_diss,
    bg_region_hyper=[90, 100],
    bg_region_thermal=[90, 100],
    molarity_hyper=0.08,
    molarity_thermal=0.08,
    T1_for_backcalculation=False,
    linebroadening=0,
    integration_width_hyper=3,
    integration_width_thermal=3,
    select_peak_ppm_thermal=False,
    select_peak_ppm_hyper=False,
    first_spec=0,
    take_one_thermal_spec=False,
    magnetic_field=7,
    Temperature=18,
    print_output=True,
    plot=True,
):
    """
    Calculate the polarization level of a hyperpolarized sample by comparing a thermal dataset with a hyperpolarized one. Common sequences used are Singlepulse or NSPECT.

    Parameters
    ----------
    thermal_data : BrukerExp instance (e.g. NSPECT, Singlepulse)
        Contains thermal reference spectra.
    hyper_data : BrukerExp instance
        Contains hyperpolarized spectra with a given TR
    timepoint : float
        Time to where polarization level is backcalculated to.
    bg_region_hyper : list, optional
        ppm values where background region is taken from, by default [0,10]
    bg_region_thermal : list, optional
        ppm values where background region is taken from, by default [0,10]
    molarity_hyper : float, optional
        Molarity of hyperpolarized sample in mols / l, for pyruvate this is
        80mM, by default 0.08
    molarity_thermal : float, optional
        Molarity of thermal sample, by default 0.08
    T1_for_backcalculation : bool/float, optional
        Gives us the option to use an externally known T1 for the decay outside the bore in seconds.
        If False, it uses the calculated/flipangle corrected T1 inside the bore.
    linebroadening : float, optional
        Linebroadening applied to both spectra before integration, default is 0 Hz.
    integration_width_hyper : float, optional
        Integration width around peak of hyper spectra in ppm, default is 3.
    integration_width_thermal : float, optional
        Integration width around peak of thermal spectrum in ppm, default is 3.
    first_spec : int, optional
        First repetition that is used, by default 0.
    B_field : int, optional
        Magnetic field in spectrometer in Tesla, by default 1
    Temp : int, optional
        Temperature in the bore in degree Celsius, by default 28.5
    print_output : bool,optional
        Prints results into notebook, by default True

    Returns
    -------
    Polarization_level : float
        Polarization level in hyperpolarized state at time_to_diss in percent.
    T1_hyper_corr : float
        T1 decay constant in seconds, corrected for flipangle and TR.
    SNR_thermal : float
        Thermal SNR from measurement, corrected for flipangle and averages.
    SNR_hyper : float
        Hyperpolarized SNR backcalculated to time_to_diss, corrected for flipangle.
    Pol_lvl_thermal : float
        Thermal polarization level (NOT IN PERCENT)

    """
    # Step 1
    # calculate thermal polarization level according to Boltzmann
    Pol_lvl_thermal = np.tanh(
        co.hbar * 67.2828 * 1e6 * magnetic_field / (2 * co.k * (273.15 + Temperature))
    )

    # Step 2: calculate hyperpolarized SNR dependent on time
    # Norm all spectra to background region
    # Integrate the peak to obtain SNR array
    # fit it to exponential decay function
    # correct the T1 decay constant through the flipangle and TR of the sequence used to monitor the
    # hyperpolarized decay
    # convert input flip angle to radians
    FA_hyper = float(hyper_data.method["ExcPulse1"][2]) * np.pi / 180.0
    # get the number of spectra for hyper measurement
    Nspec_hyper = int(hyper_data.n_repetitions)
    # Repetition time
    TR_hyper = float(hyper_data.method["PVM_RepetitionTime"]) / 1000  # into s

    ppm_axis_hyper = hyper_data.get_spec_non_localized_spectroscopy(cut_off=70)[0]
    lower_bound_bg_hyper = np.abs(ppm_axis_hyper - bg_region_hyper[0])
    lower_bound_bg_index_hyper = np.argmin(
        lower_bound_bg_hyper - np.min(lower_bound_bg_hyper)
    )
    upper_bound_bg_hyper = np.abs(ppm_axis_hyper - bg_region_hyper[1])
    upper_bound_bg_index_hyper = np.argmin(
        upper_bound_bg_hyper - np.min(upper_bound_bg_hyper)
    )
    bg_region_hyper_indices = [lower_bound_bg_index_hyper, upper_bound_bg_index_hyper]
    bg_region_hyper_indices.sort()

    hyper_spectra = hyper_data.get_spec_non_localized_spectroscopy(
        LB=linebroadening, cut_off=70
    )[1]
    hyper_normed = np.zeros_like(hyper_spectra)
    for spectrum in range(Nspec_hyper):
        hyper_normed[spectrum, :] = (
            hyper_spectra[spectrum, :]
            - np.mean(hyper_spectra[spectrum, bg_region_hyper[0] : bg_region_hyper[1]])
        ) / np.std(hyper_spectra[spectrum, bg_region_hyper[0] : bg_region_hyper[1]])

    # integrate a selected peak
    if select_peak_ppm_hyper:
        center_ppm_hyper = select_peak_ppm_hyper
    else:
        # otherwise find largest peak
        center_hyper = np.squeeze(np.where(hyper_normed - np.max(hyper_normed) == 0))[1]
        center_ppm_hyper = ppm_axis_hyper[center_hyper]

    lower_bound_integration_ppm_hyper = np.abs(
        ppm_axis_hyper - (center_ppm_hyper - integration_width_hyper)
    )
    lower_bound_integration_index_hyper = np.argmin(
        lower_bound_integration_ppm_hyper - np.min(lower_bound_integration_ppm_hyper)
    )
    upper_bound_integration_ppm_hyper = np.abs(
        ppm_axis_hyper - (center_ppm_hyper + integration_width_hyper)
    )
    upper_bound_integration_index_hyper = np.argmin(
        upper_bound_integration_ppm_hyper - np.min(upper_bound_integration_ppm_hyper)
    )
    # from this we calculate the integrated peak region
    # sorted so that lower index is first
    integrated_peak_roi_hyper = [
        lower_bound_integration_index_hyper,
        upper_bound_integration_index_hyper,
    ]
    integrated_peak_roi_hyper.sort()

    time_of_first_spec = first_spec * TR_hyper
    SNR_hyper = np.sum(
        hyper_normed[
            first_spec, integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]
        ]
    )

    # fit exponential to hyper SNR to backcalculate
    def exponential(x, M, T1, offset):
        return M * np.exp(-x / T1) + offset

    # integrate the spectra, not the normalized spectra for the T1 fit, as
    # otherwhise there could be fit issues due to high noise
    Hyper_Signal_for_T1_fit = np.array(
        [
            np.sum(
                hyper_spectra[
                    spectrum,
                    integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1],
                ]
            )
            for spectrum in range(first_spec, Nspec_hyper)
        ]
    )
    # define the time axis during which the scans took place
    hyp_time_axis = np.arange(first_spec * TR_hyper, TR_hyper * Nspec_hyper, TR_hyper)
    # Fit
    coeff, err = curve_fit(
        exponential,
        hyp_time_axis,
        Hyper_Signal_for_T1_fit,
        p0=(np.max(Hyper_Signal_for_T1_fit), 50, np.mean(Hyper_Signal_for_T1_fit)),
    )
    # flipangle correct for time outside bore

    flipangle_corr_T1 = flipangle_corr(coeff[1], FA_hyper, TR_hyper)
    error_t1 = np.sqrt(np.diag(err))[1]
    # backcalculate to time of dissolution
    if T1_for_backcalculation:
        SNR_hyper_backcalculated = exponential(
            -time_to_diss, SNR_hyper, T1_for_backcalculation, 0
        )
    else:
        SNR_hyper_backcalculated = exponential(
            -time_to_diss, SNR_hyper, flipangle_corr_T1, 0
        )
    # now norm thermal
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    # Step 3 Calculate thermal SNR
    FA_thermal = float(thermal_data.method["ExcPulse1"][2]) * np.pi / 180.0
    ppm_axis_thermal = thermal_data.get_spec_non_localized_spectroscopy(
        linebroadening, 70
    )[0]
    lower_bound_bg_thermal = np.abs(ppm_axis_thermal - bg_region_thermal[0])
    lower_bound_bg_index_thermal = np.argmin(
        lower_bound_bg_thermal - np.min(lower_bound_bg_thermal)
    )
    upper_bound_bg_thermal = np.abs(ppm_axis_thermal - bg_region_thermal[1])
    upper_bound_bg_index_thermal = np.argmin(
        upper_bound_bg_thermal - np.min(upper_bound_bg_thermal)
    )
    bg_region_thermal_indices = [
        lower_bound_bg_index_thermal,
        upper_bound_bg_index_thermal,
    ]
    bg_region_thermal_indices.sort()

    Nspec_thermal = float(thermal_data.n_repetitions) * float(
        thermal_data.method["PVM_NAverages"]
    )
    # option to only use one thermal spectrum
    if take_one_thermal_spec:
        Nspec_thermal = 1
        therm_spectra = thermal_data.get_spec_non_localized_spectroscopy(
            linebroadening, 70
        )[1][0]
        thermal_normed = (
            therm_spectra
            - np.mean(
                therm_spectra[
                    bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                ]
            )
        ) / np.std(
            therm_spectra[bg_region_thermal_indices[0] : bg_region_thermal_indices[1]]
        )
    else:
        # mean all thermal spectra we have
        if thermal_data.n_repetitions > 1:
            # mean thermal spectra if we have multiple that need to be averaged by us
            # i.e. Repetitions instead of Averages
            # number of thermal spectra
            Nspec_thermal = thermal_data.n_repetitions
            therm_fids = thermal_data.get_spec_non_localized_spectroscopy(
                linebroadening, 70
            )[2]
            # average them
            therm_fid_averaged = np.mean(therm_fids, axis=0)
            # calculate spectrum
            therm_spectrum = np.abs(np.fft.fftshift(np.fft.fft(therm_fid_averaged)))
            # norm to background noise
            thermal_normed = (
                therm_spectrum
                - np.mean(
                    therm_spectrum[
                        bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                    ]
                )
            ) / np.std(
                therm_spectrum[
                    bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                ]
            )

        elif thermal_data.method["PVM_NAverages"] > 1:
            # in case we have averages
            therm_spectra = thermal_data.get_spec_non_localized_spectroscopy(
                linebroadening, 70
            )[1]
            thermal_normed = (
                therm_spectra
                - np.mean(
                    therm_spectra[
                        bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                    ]
                )
            ) / np.std(
                therm_spectra[
                    bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                ]
            )
        else:
            pass

    # in case we want to integrate a specific peak
    if select_peak_ppm_thermal:
        center_ppm_thermal = select_peak_ppm_thermal
    else:
        # find largest peak
        center_thermal = np.squeeze(
            np.where(thermal_normed - np.max(thermal_normed) == 0)
        )
        print(np.where(thermal_normed - np.max(thermal_normed) == 0))
        center_ppm_thermal = ppm_axis_thermal[center_thermal]

    #  integrate around peak

    lower_bound_integration_ppm_thermal = np.abs(
        ppm_axis_thermal - (center_ppm_thermal - integration_width_thermal)
    )
    lower_bound_integration_index_thermal = np.argmin(
        lower_bound_integration_ppm_thermal
        - np.min(lower_bound_integration_ppm_thermal)
    )
    upper_bound_integration_ppm_thermal = np.abs(
        ppm_axis_thermal - (center_ppm_thermal + integration_width_thermal)
    )
    upper_bound_integration_index_thermal = np.argmin(
        upper_bound_integration_ppm_thermal
        - np.min(upper_bound_integration_ppm_thermal)
    )
    # from this we calculate the integrated peak region
    # sorted so that lower index is first
    integrated_peak_roi_thermal = [
        lower_bound_integration_index_thermal,
        upper_bound_integration_index_thermal,
    ]
    integrated_peak_roi_thermal.sort()

    # print(integrated_peak_roi_thermal)
    SNR_thermal = np.sum(
        thermal_normed[integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]]
    )
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    # Step 4: Correction factors
    Receiver_Gain_thermal = thermal_data.acqp["RG"]
    Receicer_Gain_hyper = hyper_data.acqp["RG"]

    correction_factor = (
        np.sqrt(Nspec_thermal)
        * (np.sin(FA_thermal) / np.sin(FA_hyper))
        * (molarity_thermal / molarity_hyper)
        * (Receiver_Gain_thermal / Receicer_Gain_hyper)
    )
    enhancement_factor = (SNR_hyper_backcalculated / SNR_thermal) * correction_factor
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    # Step 5: compare and plot results

    Polarization_level = Pol_lvl_thermal * enhancement_factor
    Polarization_level_at_first_spec = (
        Pol_lvl_thermal * (SNR_hyper / SNR_thermal) * correction_factor
    )
    Polarization_level = np.round(Polarization_level * 100, 1)
    Polarization_level_at_first_spec = np.round(
        Polarization_level_at_first_spec * 100, 1
    )

    if print_output is True:
        print("--------------------------------------------------------------")
        print(
            "Corrected observed T1=",
            np.round(coeff[1]),
            " s, for a flipangle of ",
            FA_hyper * 180 / np.pi,
            " ° and a TR of ",
            TR_hyper,
            " s ",
        )
        print("Resulting in T1_corr = ", np.round(flipangle_corr_T1, 1), " s")

        print(
            "Receiver Gain difference - Hyper RX Gain = ",
            Receicer_Gain_hyper,
            " vs Thermal RX Gain = ",
            Receiver_Gain_thermal,
        )
        print(
            "Molarity  difference - Hyper Sample 13C Molarity = ",
            molarity_hyper,
            " vs Thermal Sample 13C Molarity  = ",
            molarity_thermal,
        )
        print(
            "Number of spectra  difference - Hyper Scan 1 sample vs Thermal scan ",
            Nspec_thermal,
            " sample",
        )
        print(
            "Flipangle difference correction - Hyper flip angle ",
            FA_hyper * 180 / np.pi,
            " ° - vs Thermal flip angle ",
            FA_thermal * 180 / np.pi,
            " °",
        )
        print(
            "Enhancement factor from thermal to hyper",
            "{:.1e}".format(enhancement_factor),
        )
        if T1_for_backcalculation:
            print(
                "Externally used T1 from other fit function = ",
                np.round(T1_for_backcalculation, 1),
            )
        else:
            print(
                "T1_hyper_corr = ",
                np.round(flipangle_corr_T1, 1),
                "pm",
                np.round(error_t1, 1),
                " s",
            )

        print(
            "SNR_thermal normed to Molarity and Number of spectra",
            np.round(
                SNR_thermal
                * (molarity_hyper / molarity_thermal)
                / np.sqrt(Nspec_thermal),
                1,
            ),
        )
        print(
            "SNR_thermal / correction factor = ",
            np.round(SNR_thermal / correction_factor, 3),
        )
        print("--------------------------------------------------------------")
        print("THERMAL Polarization = ", Pol_lvl_thermal)
        print("SNR_thermal = ", np.round(SNR_thermal, 1))
        print("SNR_hyper_backcalculated = ", np.round(SNR_hyper_backcalculated, 1))
        print("SNR_hyper_at_first_spec = ", np.round(SNR_hyper, 1))

        time_of_first_spec = TR_hyper * first_spec
        print(
            "HYPER - Polarization level of first spec at T = ",
            time_of_first_spec,
            " s, is ",
            Polarization_level_at_first_spec,
            " %",
        )
        print(
            "HYPER - Polarization level at T = ",
            time_of_first_spec - time_to_diss,
            " s, is ",
            Polarization_level,
            " %",
        )
        print("--------------------------------------------------------------")
    else:
        pass

    if plot is True:
        fig, ax = plt.subplots(1, tight_layout=True, figsize=(6, 4))
        backcalc_axis = np.arange(
            time_of_first_spec - time_to_diss, TR_hyper * Nspec_hyper, TR_hyper
        )
        if T1_for_backcalculation:
            ax.plot(
                backcalc_axis,
                exponential(
                    backcalc_axis,
                    Polarization_level,
                    T1_for_backcalculation,
                    Pol_lvl_thermal,
                ),
            )
            ax.scatter(
                time_of_first_spec,
                exponential(
                    time_of_first_spec,
                    Polarization_level,
                    T1_for_backcalculation,
                    Pol_lvl_thermal,
                ),
                label="First measurement point",
                color="C1",
            )
        else:
            ax.plot(
                backcalc_axis,
                exponential(
                    backcalc_axis,
                    Polarization_level,
                    flipangle_corr_T1,
                    Pol_lvl_thermal,
                ),
            )
            ax.scatter(
                time_of_first_spec,
                exponential(
                    time_of_first_spec,
                    Polarization_level,
                    flipangle_corr_T1,
                    Pol_lvl_thermal,
                ),
                label="First measurement point",
                color="C1",
            )
        ax.set_xlabel("Time since start of experiment [s]")
        ax.set_ylabel(r"Polarization level [$\%$]")
        ax.set_title("Polarization level")
        ax.legend()

        fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(10, 4))
        ax[0].plot(thermal_data.ppm_axis, thermal_normed)
        ax[0].set_xlabel("ppm")
        ax[0].set_title("Thermal spectrum - " + str(Nspec_thermal) + " averages")
        ax[0].set_ylabel("SNR")
        ax[0].fill_between(
            [
                thermal_data.ppm_axis[bg_region_thermal_indices[0]],
                thermal_data.ppm_axis[bg_region_thermal_indices[1]],
            ],
            np.min(thermal_normed),
            np.max(thermal_normed),
            alpha=0.3,
            color="C2",
            label="Background",
        )
        ax[0].fill_between(
            [
                thermal_data.ppm_axis[integrated_peak_roi_thermal[0]],
                thermal_data.ppm_axis[integrated_peak_roi_thermal[1]],
            ],
            np.min(thermal_normed),
            np.max(thermal_normed),
            alpha=0.3,
            color="C1",
            label="Peak integration",
        )

        ax[1].plot(ppm_axis_hyper, hyper_normed[first_spec])
        ax[1].set_xlabel("ppm")
        ax[1].set_title("First hyper spectrum")
        ax[1].set_ylabel("SNR")
        ax[1].fill_between(
            [
                ppm_axis_hyper[bg_region_hyper_indices[0]],
                ppm_axis_hyper[bg_region_hyper_indices[1]],
            ],
            np.min(hyper_normed[first_spec]),
            np.max(hyper_normed[first_spec]),
            alpha=0.3,
            color="C2",
            label="Background",
        )
        ax[1].fill_between(
            [
                ppm_axis_hyper[integrated_peak_roi_hyper[0]],
                ppm_axis_hyper[integrated_peak_roi_hyper[1]],
            ],
            np.min(hyper_normed[first_spec]),
            np.max(hyper_normed[first_spec]),
            alpha=0.3,
            color="C1",
            label="Peak integration",
        )
        ax[1].legend()

        ax[2].set_title("Hyper Signal for T1 fit")
        ax[2].scatter(hyp_time_axis, Hyper_Signal_for_T1_fit, label="Data points")
        backcalc_axis = np.arange(
            time_of_first_spec - time_to_diss, TR_hyper * Nspec_hyper, TR_hyper
        )
        ax[2].plot(
            hyp_time_axis,
            exponential(hyp_time_axis, coeff[0], coeff[1], coeff[2]),
            label="Fit - T1=" + str(np.round(coeff[1], 1)) + "s",
        )
        if T1_for_backcalculation:
            ax[2].plot(
                backcalc_axis,
                exponential(backcalc_axis, coeff[0], T1_for_backcalculation, coeff[2]),
                label="T1 manual input ="
                + str(np.round(T1_for_backcalculation, 1))
                + "s",
            )
        else:
            ax[2].plot(
                backcalc_axis,
                exponential(backcalc_axis, coeff[0], flipangle_corr_T1, coeff[2]),
                label="T1 corrected =" + str(np.round(flipangle_corr_T1, 1)) + "s",
            )

        ax[2].legend()
        ax[2].set_ylabel("Hyper Signal [a.u.]")
        ax[2].set_xlabel("Time since start of experiment [s]")
        ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        # second plot showing background levels

        fig2, ax2 = plt.subplots(2, 2, figsize=(10, 4), tight_layout=True)
        ax2[0, 0].plot(
            ppm_axis_thermal[
                bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
            ],
            thermal_normed[bg_region_thermal_indices[0] : bg_region_thermal_indices[1]],
        )
        ax2[0, 1].plot(
            ppm_axis_hyper[bg_region_hyper_indices[0] : bg_region_hyper_indices[1]],
            hyper_normed[
                first_spec, bg_region_hyper_indices[0] : bg_region_hyper_indices[1]
            ],
        )

        ax2[0, 0].set_title("Thermal BG region")
        ax2[0, 1].set_title("Hyper BG region")
        bg_region_thermal_indices.sort()
        x_data, y_data, bin_size = get_histogram(
            thermal_normed[bg_region_thermal_indices[0] : bg_region_thermal_indices[1]],
            25,
        )
        ax2[1, 0].bar(x_data, y_data, width=bin_size)
        x_data, y_data, bin_size = get_histogram(
            hyper_normed[
                first_spec, bg_region_hyper_indices[0] : bg_region_hyper_indices[1]
            ],
            25,
        )
        ax2[1, 1].bar(x_data, y_data, width=bin_size)
        ax2[1, 0].set_title("Thermal BG - Histogram")
        ax2[1, 1].set_title("Hyper BG - Histogram")
        ax2[1, 0].set_xlabel("SNR val")
        ax2[1, 1].set_xlabel("SNR val")
        ax2[0, 0].set_xlabel("ppm")
        ax2[0, 1].set_xlabel("ppm")
        ax2[1, 0].set_ylabel("Nr Points")
        ax2[1, 1].set_ylabel("Nr Points")

        # third plot showing peak integration regions

        fig2, ax2 = plt.subplots(2, 2, figsize=(10, 4), tight_layout=True)
        ax2[0, 0].plot(
            ppm_axis_thermal[
                integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]
            ],
            thermal_normed[
                integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]
            ],
        )
        ax2[0, 1].plot(
            ppm_axis_hyper[integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]],
            hyper_normed[
                first_spec, integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]
            ],
        )

        ax2[0, 0].set_title("Thermal Signal region")
        ax2[0, 1].set_title("Hyper Signal region")
        x_data, y_data, bin_size = get_histogram(
            thermal_normed[
                integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]
            ],
            100,
        )
        ax2[1, 0].bar(x_data, y_data, width=bin_size)
        x_data, y_data, bin_size = get_histogram(
            hyper_normed[
                first_spec, integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]
            ],
            100,
        )
        ax2[1, 1].bar(x_data, y_data, width=bin_size)
        ax2[1, 0].set_title("Thermal Signal - Histogram")
        ax2[1, 1].set_title("Hyper Signal - Histogram")
        ax2[1, 0].set_xlabel("SNR val")
        ax2[1, 1].set_xlabel("SNR val")
        ax2[0, 0].set_xlabel("ppm")
        ax2[0, 1].set_xlabel("ppm")
        ax2[1, 0].set_ylabel("Nr Points")
        ax2[1, 1].set_ylabel("Nr Points")

    else:
        pass

    return (
        Polarization_level,
        SNR_thermal,
        SNR_hyper_backcalculated,
        Pol_lvl_thermal,
        correction_factor,
    )


def calculate_polarization_level_phased_real_spectra(
    thermal_data,
    hyper_data,
    time_to_diss,
    bg_region_hyper=[90, 100],
    bg_region_thermal=[90, 100],
    molarity_hyper=0.08,
    molarity_thermal=0.08,
    T1_for_backcalculation=False,
    linebroadening=0,
    integration_width_hyper=3,
    integration_width_thermal=3,
    select_peak_ppm_thermal=False,
    select_peak_ppm_hyper=False,
    first_spec=0,
    thermal_phase_input=False,
    hyper_phase_input=False,
    take_one_thermal_spec=False,
    magnetic_field=7,
    Temperature=18,
    print_output=True,
    plot=True,
):
    """
    Calculate the polarization level of a hyperpolarized sample by comparing a thermal dataset with a hyperpolarized one. Common sequences used are Singlepulse or NSPECT.

    Parameters
    ----------
    thermal_data : BrukerExp instance (e.g. NSPECT, Singlepulse)
        Contains thermal reference spectra.
    hyper_data : BrukerExp instance
        Contains hyperpolarized spectra with a given TR
    timepoint : float
        Time to where polarization level is backcalculated to.
    bg_region_hyper : list, optional
        ppm values where background region is taken from, by default [0,10]
    bg_region_thermal : list, optional
        ppm values where background region is taken from, by default [0,10]
    molarity_hyper : float, optional
        Molarity of hyperpolarized sample in mols / l, for pyruvate this is
        80mM, by default 0.08
    molarity_thermal : float, optional
        Molarity of thermal sample, by default 0.08
    T1_for_backcalculation : bool/float, optional
        Gives us the option to use an externally known T1 for the decay outside the bore in seconds.
        If False, it uses the calculated/flipangle corrected T1 inside the bore.
    linebroadening : float, optional
        Linebroadening applied to both spectra before integration, default is 0 Hz.
    integration_width_hyper : float, optional
        Integration width around peak of hyper spectra in ppm, default is 3.
    integration_width_thermal : float, optional
        Integration width around peak of thermal spectrum in ppm, default is 3.
    first_spec : int, optional
        First repetition that is used, by default 0.
    thermal_phase_input: float, optional
        Phase correction factor for thermal spectra in degree (0-360), obtained from phase_fid-function,
        by default False, i.e. no phase correction.
    hyper_phase_input: float, optional
        Phase correction factor for hyper spectra in degree (0-360), obtained from phase_fid-function,
         by default False, i.e. no phase correction.
    B_field : int, optional
        Magnetic field in spectrometer in Tesla, by default 1
    Temp : int, optional
        Temperature in the bore in degree Celsius, by default 28.5
    print_output : bool,optional
        Prints results into notebook, by default True

    Returns
    -------
    Polarization_level : float
        Polarization level in hyperpolarized state at time_to_diss in percent.
    T1_hyper_corr : float
        T1 decay constant in seconds, corrected for flipangle and TR.
    SNR_thermal : float
        Thermal SNR from measurement, corrected for flipangle and averages.
    SNR_hyper : float
        Hyperpolarized SNR backcalculated to time_to_diss, corrected for flipangle.
    Pol_lvl_thermal : float
        Thermal polarization level (NOT IN PERCENT)

    """
    # Step 1
    # calculate thermal polarization level according to Boltzmann
    Pol_lvl_thermal = np.tanh(
        co.hbar * 67.2828 * 1e6 * magnetic_field / (2 * co.k * (273.15 + Temperature))
    )

    # Step 2: calculate hyperpolarized SNR dependent on time
    # Norm all spectra to background region
    # Integrate the peak to obtain SNR array
    # fit it to exponential decay function
    # correct the T1 decay constant through the flipangle and TR of the sequence used to monitor the
    # hyperpolarized decay
    # convert input flip angle to radians
    FA_hyper = float(hyper_data.method["ExcPulse1"][2]) * np.pi / 180.0
    # get the number of spectra for hyper measurement
    Nspec_hyper = int(hyper_data.n_repetitions)
    # Repetition time
    TR_hyper = float(hyper_data.method["PVM_RepetitionTime"]) / 1000  # into s

    (
        ppm_axis_hyper,
        mag_spec_hyper,
        fids_hyper,
        hyper_complex_spec,
    ) = hyper_data.get_spec_non_localized_spectroscopy(cut_off=70)
    lower_bound_bg_hyper = np.abs(ppm_axis_hyper - bg_region_hyper[0])
    lower_bound_bg_index_hyper = np.argmin(
        lower_bound_bg_hyper - np.min(lower_bound_bg_hyper)
    )
    upper_bound_bg_hyper = np.abs(ppm_axis_hyper - bg_region_hyper[1])
    upper_bound_bg_index_hyper = np.argmin(
        upper_bound_bg_hyper - np.min(upper_bound_bg_hyper)
    )
    bg_region_hyper_indices = [lower_bound_bg_index_hyper, upper_bound_bg_index_hyper]
    bg_region_hyper_indices.sort()

    # take first spectrum thats wanted
    hyper_complex_spec_1 = hyper_complex_spec[first_spec]

    # perform baseline correction
    hyper_complex_spec_1_baseline_corr = hyper_complex_spec_1 - np.mean(
        hyper_complex_spec_1[bg_region_hyper_indices[0] : bg_region_hyper_indices[1]]
    )
    # phase the real spectrum
    Integrals_hyp = []
    phases = np.linspace(0, 360, 1000)
    for phase in phases:
        itgl = np.max(
            np.real(
                hyper_complex_spec_1_baseline_corr
                * np.exp(1j * (phase * np.pi) / 180.0)
            )
        )
        Integrals_hyp.append(itgl)
    # take the real part of the spectrum
    if hyper_phase_input is False:
        final_phase_hyper = phases[
            np.argmin(np.abs(Integrals_hyp - np.max(Integrals_hyp)))
        ]
    else:
        final_phase_hyper = hyper_phase_input
    # apply phasing:
    first_hyper_spec = np.real(
        hyper_complex_spec_1_baseline_corr
        * np.exp(1j * (final_phase_hyper * np.pi) / 180.0)
    )
    # norm to background region
    hyper_normed = (
        first_hyper_spec
        - np.mean(
            first_hyper_spec[bg_region_hyper_indices[0] : bg_region_hyper_indices[1]]
        )
    ) / np.std(
        first_hyper_spec[bg_region_hyper_indices[0] : bg_region_hyper_indices[1]]
    )
    hyper_spectra = []
    # phase all spectra and dont norm them for T1 fitting
    for spectrum in range(Nspec_hyper):
        spec = np.fft.fftshift(np.fft.fft(fids_hyper[spectrum, :]))
        hyper_spectra.append(
            np.real(spec * np.exp(1j * (final_phase_hyper * np.pi) / 180.0))
        )
    hyper_spectra = np.array(hyper_spectra)
    # integrate a selected peak
    if select_peak_ppm_hyper:
        center_ppm_hyper = select_peak_ppm_hyper
    else:
        # otherwise find largest peak
        center_hyper = np.squeeze(np.where(hyper_normed - np.max(hyper_normed) == 0))
        center_ppm_hyper = ppm_axis_hyper[center_hyper]

    lower_bound_integration_ppm_hyper = np.abs(
        ppm_axis_hyper - (center_ppm_hyper - integration_width_hyper)
    )
    lower_bound_integration_index_hyper = np.argmin(
        lower_bound_integration_ppm_hyper - np.min(lower_bound_integration_ppm_hyper)
    )
    upper_bound_integration_ppm_hyper = np.abs(
        ppm_axis_hyper - (center_ppm_hyper + integration_width_hyper)
    )
    upper_bound_integration_index_hyper = np.argmin(
        upper_bound_integration_ppm_hyper - np.min(upper_bound_integration_ppm_hyper)
    )
    # from this we calculate the integrated peak region
    # sorted so that lower index is first
    integrated_peak_roi_hyper = [
        lower_bound_integration_index_hyper,
        upper_bound_integration_index_hyper,
    ]
    integrated_peak_roi_hyper.sort()

    time_of_first_spec = first_spec * TR_hyper
    SNR_hyper = np.sum(
        hyper_normed[integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]]
    )

    # fit exponential to hyper SNR to backcalculate
    def exponential(x, M, T1, offset):
        return M * np.exp(-x / T1) + offset

    # integrate the spectra, not the normalized spectra for the T1 fit, as
    # otherwhise there could be fit issues due to high noise
    Hyper_Signal_for_T1_fit = np.array(
        [
            np.sum(
                hyper_spectra[
                    spectrum,
                    integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1],
                ]
            )
            for spectrum in range(first_spec, Nspec_hyper)
        ]
    )

    # define the time axis during which the scans took place
    hyp_time_axis = np.arange(first_spec * TR_hyper, TR_hyper * Nspec_hyper, TR_hyper)
    # Fit
    coeff, err = curve_fit(
        exponential,
        hyp_time_axis,
        Hyper_Signal_for_T1_fit,
        p0=(np.max(Hyper_Signal_for_T1_fit), 50, np.mean(Hyper_Signal_for_T1_fit)),
    )
    # flipangle correct for time outside bore

    flipangle_corr_T1 = flipangle_corr(coeff[1], FA_hyper, TR_hyper)
    error_t1 = np.sqrt(np.diag(err))[1]
    # backcalculate to time of dissolution
    if T1_for_backcalculation:
        SNR_hyper_backcalculated = exponential(
            -time_to_diss, SNR_hyper, T1_for_backcalculation, 0
        )
    else:
        SNR_hyper_backcalculated = exponential(
            -time_to_diss, SNR_hyper, flipangle_corr_T1, 0
        )
    # now norm thermal
    # Step 3 Calculate thermal SNR
    FA_thermal = float(thermal_data.method["ExcPulse1"][2]) * np.pi / 180.0
    (
        ppm_axis_thermal,
        mag_spec_thermal,
        fids_thermal,
        complex_spec_thermal,
    ) = thermal_data.get_spec_non_localized_spectroscopy(linebroadening, 70)
    lower_bound_bg_thermal = np.abs(ppm_axis_thermal - bg_region_thermal[0])
    lower_bound_bg_index_thermal = np.argmin(
        lower_bound_bg_thermal - np.min(lower_bound_bg_thermal)
    )
    upper_bound_bg_thermal = np.abs(ppm_axis_thermal - bg_region_thermal[1])
    upper_bound_bg_index_thermal = np.argmin(
        upper_bound_bg_thermal - np.min(upper_bound_bg_thermal)
    )
    bg_region_thermal_indices = [
        lower_bound_bg_index_thermal,
        upper_bound_bg_index_thermal,
    ]
    bg_region_thermal_indices.sort()

    Nspec_thermal = float(thermal_data.n_repetitions) * float(thermal_data.n_averages)
    # option to only use one thermal spectrum
    if take_one_thermal_spec:
        Nspec_thermal = 1
        complex_spec_thermal_single = complex_spec_thermal[0]
        complex_spec_thermal_single_baselinecorr = (
            complex_spec_thermal_single
            - np.mean(
                complex_spec_thermal_single[
                    bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                ]
            )
        )
        if thermal_phase_input is False:
            Integrals_therm = []
            phases = np.linspace(0, 360, 1000)
            for phase in phases:
                itgl = np.max(
                    np.real(
                        complex_spec_thermal_single_baselinecorr
                        * np.exp(1j * (phase * np.pi) / 180.0)
                    )
                )
                Integrals_therm.append(itgl)

            final_phase_therm = phases[
                np.argmin(np.abs(Integrals_therm - np.max(Integrals_therm)))
            ]
        else:
            final_phase_therm = thermal_phase_input
        phased_therm_spec = np.real(
            complex_spec_thermal_single_baselinecorr
            * np.exp(1j * (final_phase_therm * np.pi) / 180.0)
        )
        thermal_normed = (
            phased_therm_spec
            - np.mean(
                phased_therm_spec[
                    bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                ]
            )
        ) / np.std(
            phased_therm_spec[
                bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
            ]
        )
    else:
        # mean all thermal spectra we have
        if thermal_data.n_repetitions > 1:
            # mean thermal spectra if we have multiple that need to be averaged by us
            # i.e. Repetitions instead of Averages
            # number of thermal spectra
            Nspec_thermal = thermal_data.n_repetitions
            # average them
            therm_fid_averaged = np.mean(fids_thermal, axis=0)
            # calculate spectrum
            complex_spec_thermal_averaged = np.fft.fftshift(
                np.fft.fft(therm_fid_averaged)
            )
            complex_spec_thermal_averaged_baselinecorr = (
                complex_spec_thermal_averaged
                - np.mean(
                    complex_spec_thermal_averaged[
                        bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                    ]
                )
            )
            if thermal_phase_input is False:
                Integrals_therm = []
                phases = np.linspace(0, 360, 1000)
                for phase in phases:
                    itgl = np.max(
                        np.real(
                            complex_spec_thermal_averaged_baselinecorr
                            * np.exp(1j * (phase * np.pi) / 180.0)
                        )
                    )
                    Integrals_therm.append(itgl)

                final_phase_therm = phases[
                    np.argmin(np.abs(Integrals_therm - np.max(Integrals_therm)))
                ]
            else:
                final_phase_therm = thermal_phase_input
            phased_complex_spec_thermal_averaged = np.real(
                complex_spec_thermal_averaged_baselinecorr
                * np.exp(1j * (final_phase_therm * np.pi) / 180.0)
            )
            # norm to background noise
            thermal_normed = (
                phased_complex_spec_thermal_averaged
                - np.mean(
                    phased_complex_spec_thermal_averaged[
                        bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                    ]
                )
            ) / np.std(
                phased_complex_spec_thermal_averaged[
                    bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                ]
            )

        elif thermal_data.method["PVM_NAverages"] > 1:
            # in case we have averages
            complex_spec_thermal_averaged_baselinecorr = complex_spec_thermal - np.mean(
                complex_spec_thermal[
                    bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                ]
            )
            if thermal_phase_input is False:
                Integrals_therm = []
                phases = np.linspace(0, 360, 1000)
                for phase in phases:
                    itgl = np.max(
                        np.real(
                            complex_spec_thermal_averaged_baselinecorr
                            * np.exp(1j * (phase * np.pi) / 180.0)
                        )
                    )
                    Integrals_therm.append(itgl)

                final_phase_therm = phases[
                    np.argmin(np.abs(Integrals_therm - np.max(Integrals_therm)))
                ]
            else:
                final_phase_therm = thermal_phase_input
            phased_complex_spec_thermal_averaged = np.real(
                complex_spec_thermal_averaged_baselinecorr
                * np.exp(1j * (final_phase_therm * np.pi) / 180.0)
            )
            thermal_normed = (
                phased_complex_spec_thermal_averaged
                - np.mean(
                    phased_complex_spec_thermal_averaged[
                        bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                    ]
                )
            ) / np.std(
                phased_complex_spec_thermal_averaged[
                    bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                ]
            )
        else:
            pass

    # in case we want to integrate a specific peak
    if select_peak_ppm_thermal:
        center_ppm_thermal = select_peak_ppm_thermal
    else:
        # find largest peak
        center_thermal = np.squeeze(
            np.where(thermal_normed - np.max(thermal_normed) == 0)
        )
        print(np.where(thermal_normed - np.max(thermal_normed) == 0))
        center_ppm_thermal = ppm_axis_thermal[center_thermal]

    #  integrate around peak

    lower_bound_integration_ppm_thermal = np.abs(
        ppm_axis_thermal - (center_ppm_thermal - integration_width_thermal)
    )
    lower_bound_integration_index_thermal = np.argmin(
        lower_bound_integration_ppm_thermal
        - np.min(lower_bound_integration_ppm_thermal)
    )
    upper_bound_integration_ppm_thermal = np.abs(
        ppm_axis_thermal - (center_ppm_thermal + integration_width_thermal)
    )
    upper_bound_integration_index_thermal = np.argmin(
        upper_bound_integration_ppm_thermal
        - np.min(upper_bound_integration_ppm_thermal)
    )
    # from this we calculate the integrated peak region
    # sorted so that lower index is first
    integrated_peak_roi_thermal = [
        lower_bound_integration_index_thermal,
        upper_bound_integration_index_thermal,
    ]
    integrated_peak_roi_thermal.sort()

    # print(integrated_peak_roi_thermal)
    SNR_thermal = np.sum(
        thermal_normed[integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]]
    )
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    # Step 4: Correction factors
    Receiver_Gain_thermal = thermal_data.acqp["RG"]
    Receicer_Gain_hyper = hyper_data.acqp["RG"]

    correction_factor = (
        np.sqrt(Nspec_thermal)
        * (np.sin(FA_thermal) / np.sin(FA_hyper))
        * (molarity_thermal / molarity_hyper)
        * (Receiver_Gain_thermal / Receicer_Gain_hyper)
    )
    enhancement_factor = (SNR_hyper_backcalculated / SNR_thermal) * correction_factor
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    # Step 5: compare and plot results

    Polarization_level = Pol_lvl_thermal * enhancement_factor
    Polarization_level_at_first_spec = (
        Pol_lvl_thermal * (SNR_hyper / SNR_thermal) * correction_factor
    )
    Polarization_level = np.round(Polarization_level * 100, 1)
    Polarization_level_at_first_spec = np.round(
        Polarization_level_at_first_spec * 100, 1
    )

    if print_output is True:
        print("--------------------------------------------------------------")
        print(
            "Corrected observed T1=",
            np.round(coeff[1]),
            " s, for a flipangle of ",
            FA_hyper * 180 / np.pi,
            " ° and a TR of ",
            TR_hyper,
            " s ",
        )
        print("Resulting in T1_corr = ", np.round(flipangle_corr_T1, 1), " s")

        print(
            "Receiver Gain difference - Hyper RX Gain = ",
            Receicer_Gain_hyper,
            " vs Thermal RX Gain = ",
            Receiver_Gain_thermal,
        )
        print(
            "Molarity  difference - Hyper Sample 13C Molarity = ",
            molarity_hyper,
            " vs Thermal Sample 13C Molarity  = ",
            molarity_thermal,
        )
        print(
            "Number of spectra  difference - Hyper Scan 1 sample vs Thermal scan ",
            Nspec_thermal,
            " sample",
        )
        print(
            "Flipangle difference correction - Hyper flip angle ",
            FA_hyper * 180 / np.pi,
            " ° - vs Thermal flip angle ",
            FA_thermal * 180 / np.pi,
            " °",
        )
        print(
            "Enhancement factor from thermal to hyper",
            "{:.1e}".format(enhancement_factor),
        )
        if T1_for_backcalculation:
            print(
                "Externally used T1 from other fit function = ",
                np.round(T1_for_backcalculation, 1),
            )
        else:
            print(
                "T1_hyper_corr = ",
                np.round(flipangle_corr_T1, 1),
                "pm",
                np.round(error_t1, 1),
                " s",
            )

        print(
            "SNR_thermal normed to Molarity and Number of spectra",
            np.round(
                SNR_thermal
                * (molarity_hyper / molarity_thermal)
                / np.sqrt(Nspec_thermal),
                1,
            ),
        )
        print(
            "SNR_thermal / correction factor = ",
            np.round(SNR_thermal / correction_factor, 3),
        )
        print("--------------------------------------------------------------")
        print("THERMAL Polarization = ", Pol_lvl_thermal)
        print("SNR_thermal = ", np.round(SNR_thermal, 1))
        print("SNR_hyper_backcalculated = ", np.round(SNR_hyper_backcalculated, 1))
        print("SNR_hyper_at_first_spec = ", np.round(SNR_hyper, 1))

        time_of_first_spec = TR_hyper * first_spec
        print(
            "HYPER - Polarization level of first spec at T = ",
            time_of_first_spec,
            " s, is ",
            Polarization_level_at_first_spec,
            " %",
        )
        print(
            "HYPER - Polarization level at T = ",
            time_of_first_spec - time_to_diss,
            " s, is ",
            Polarization_level,
            " %",
        )
        print("--------------------------------------------------------------")
    else:
        pass

    if plot is True:
        fig, ax = plt.subplots(1, tight_layout=True, figsize=(6, 4))
        backcalc_axis = np.arange(
            time_of_first_spec - time_to_diss, TR_hyper * Nspec_hyper, TR_hyper
        )
        if T1_for_backcalculation:
            ax.plot(
                backcalc_axis,
                exponential(
                    backcalc_axis,
                    Polarization_level,
                    T1_for_backcalculation,
                    Pol_lvl_thermal,
                ),
                label="Manual input T1 = "
                + str(np.round(T1_for_backcalculation, 1))
                + " s",
            )

            p_lvl_at_first_spec = exponential(
                time_of_first_spec,
                Polarization_level,
                T1_for_backcalculation,
                Pol_lvl_thermal,
            )
            ax.plot(
                backcalc_axis,
                exponential(
                    backcalc_axis,
                    (p_lvl_at_first_spec - Pol_lvl_thermal)
                    / (np.exp(-time_of_first_spec / flipangle_corr_T1)),
                    flipangle_corr_T1,
                    Pol_lvl_thermal,
                ),
                label="Flipangle corrected T1 = "
                + str(np.round(flipangle_corr_T1, 1))
                + " s",
            )

            ax.scatter(
                time_of_first_spec,
                exponential(
                    time_of_first_spec,
                    Polarization_level,
                    T1_for_backcalculation,
                    Pol_lvl_thermal,
                ),
                label="First measurement point",
                color="C1",
            )
        else:
            ax.plot(
                backcalc_axis,
                exponential(
                    backcalc_axis,
                    Polarization_level,
                    flipangle_corr_T1,
                    Pol_lvl_thermal,
                ),
            )
            ax.scatter(
                time_of_first_spec,
                exponential(
                    time_of_first_spec,
                    Polarization_level,
                    flipangle_corr_T1,
                    Pol_lvl_thermal,
                ),
                label="First measurement point",
                color="C1",
            )
        ax.set_xlabel("Time since start of experiment [s]")
        ax.set_ylabel(r"Polarization level [$\%$]")
        ax.set_title("Polarization level")
        ax.legend()

        fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(10, 4))
        ax[0].plot(thermal_data.ppm_axis, thermal_normed)
        ax[0].set_xlabel("ppm")
        ax[0].set_title("Thermal spectrum - " + str(Nspec_thermal) + " averages")
        ax[0].set_ylabel("SNR")
        ax[0].fill_between(
            [
                thermal_data.ppm_axis[bg_region_thermal_indices[0]],
                thermal_data.ppm_axis[bg_region_thermal_indices[1]],
            ],
            np.min(thermal_normed),
            np.max(thermal_normed),
            alpha=0.3,
            color="C2",
            label="Background",
        )
        ax[0].fill_between(
            [
                thermal_data.ppm_axis[integrated_peak_roi_thermal[0]],
                thermal_data.ppm_axis[integrated_peak_roi_thermal[1]],
            ],
            np.min(thermal_normed),
            np.max(thermal_normed),
            alpha=0.3,
            color="C1",
            label="Peak integration",
        )

        ax[1].plot(ppm_axis_hyper, hyper_normed)
        ax[1].set_xlabel("ppm")
        ax[1].set_title("First hyper spectrum")
        ax[1].set_ylabel("SNR")
        ax[1].fill_between(
            [
                ppm_axis_hyper[bg_region_hyper_indices[0]],
                ppm_axis_hyper[bg_region_hyper_indices[1]],
            ],
            np.min(hyper_normed),
            np.max(hyper_normed),
            alpha=0.3,
            color="C2",
            label="Background",
        )
        ax[1].fill_between(
            [
                ppm_axis_hyper[integrated_peak_roi_hyper[0]],
                ppm_axis_hyper[integrated_peak_roi_hyper[1]],
            ],
            np.min(hyper_normed),
            np.max(hyper_normed),
            alpha=0.3,
            color="C1",
            label="Peak integration",
        )
        ax[1].legend()

        ax[2].set_title("Hyper Signal for T1 fit")
        ax[2].scatter(hyp_time_axis, Hyper_Signal_for_T1_fit, label="Data points")
        backcalc_axis = np.arange(
            time_of_first_spec - time_to_diss, TR_hyper * Nspec_hyper, TR_hyper
        )
        ax[2].plot(
            hyp_time_axis,
            exponential(hyp_time_axis, coeff[0], coeff[1], coeff[2]),
            label="Fit - T1=" + str(np.round(coeff[1], 1)) + "s",
        )

        ax[2].legend()
        ax[2].set_ylabel("Hyper Signal [a.u.]")
        ax[2].set_xlabel("Time since start of experiment [s]")
        ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        # second plot showing background levels

        fig2, ax2 = plt.subplots(2, 2, figsize=(10, 4), tight_layout=True)
        ax2[0, 0].plot(
            ppm_axis_thermal[
                bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
            ],
            thermal_normed[bg_region_thermal_indices[0] : bg_region_thermal_indices[1]],
        )
        ax2[0, 1].plot(
            ppm_axis_hyper[bg_region_hyper_indices[0] : bg_region_hyper_indices[1]],
            hyper_normed[bg_region_hyper_indices[0] : bg_region_hyper_indices[1]],
        )

        ax2[0, 0].set_title("Thermal BG region")
        ax2[0, 1].set_title("Hyper BG region")
        bg_region_thermal_indices.sort()
        x_data, y_data, bin_size = get_histogram(
            thermal_normed[bg_region_thermal_indices[0] : bg_region_thermal_indices[1]],
            25,
        )
        ax2[1, 0].bar(x_data, y_data, width=bin_size)
        x_data, y_data, bin_size = get_histogram(
            hyper_normed[bg_region_hyper_indices[0] : bg_region_hyper_indices[1]],
            25,
        )
        ax2[1, 1].bar(x_data, y_data, width=bin_size)
        ax2[1, 0].set_title("Thermal BG - Histogram")
        ax2[1, 1].set_title("Hyper BG - Histogram")
        ax2[1, 0].set_xlabel("SNR val")
        ax2[1, 1].set_xlabel("SNR val")
        ax2[0, 0].set_xlabel("ppm")
        ax2[0, 1].set_xlabel("ppm")
        ax2[1, 0].set_ylabel("Nr Points")
        ax2[1, 1].set_ylabel("Nr Points")

        # third plot showing peak integration regions

        fig2, ax2 = plt.subplots(2, 2, figsize=(10, 4), tight_layout=True)
        ax2[0, 0].plot(
            ppm_axis_thermal[
                integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]
            ],
            thermal_normed[
                integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]
            ],
        )
        ax2[0, 1].plot(
            ppm_axis_hyper[integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]],
            hyper_normed[integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]],
        )

        ax2[0, 0].set_title("Thermal Signal region")
        ax2[0, 1].set_title("Hyper Signal region")
        x_data, y_data, bin_size = get_histogram(
            thermal_normed[
                integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]
            ],
            100,
        )
        ax2[1, 0].bar(x_data, y_data, width=bin_size)
        x_data, y_data, bin_size = get_histogram(
            hyper_normed[integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]],
            100,
        )
        ax2[1, 1].bar(x_data, y_data, width=bin_size)
        ax2[1, 0].set_title("Thermal Signal - Histogram")
        ax2[1, 1].set_title("Hyper Signal - Histogram")
        ax2[1, 0].set_xlabel("SNR val")
        ax2[1, 1].set_xlabel("SNR val")
        ax2[0, 0].set_xlabel("ppm")
        ax2[0, 1].set_xlabel("ppm")
        ax2[1, 0].set_ylabel("Nr Points")
        ax2[1, 1].set_ylabel("Nr Points")

    else:
        pass

    return (
        Polarization_level,
        Polarization_level_at_first_spec,
        SNR_thermal,
        SNR_hyper_backcalculated,
        Pol_lvl_thermal,
        enhancement_factor,
        flipangle_corr_T1,
    )


def calculate_polarization_level_phased_real_spectra_not_normed_to_SNR(
    thermal_data,
    hyper_data,
    time_to_diss,
    molarity_hyper=0.08,
    molarity_thermal=0.08,
    T1_for_backcalculation=False,
    linebroadening=0,
    integration_width_hyper=3,
    integration_width_thermal=3,
    select_peak_ppm_thermal=False,
    select_peak_ppm_hyper=False,
    first_spec=0,
    thermal_phase_input=False,
    hyper_phase_input=False,
    take_one_thermal_spec=False,
    magnetic_field=7,
    Temperature=18,
    print_output=True,
    plot=True,
):
    """
    Calculate the polarization level of a hyperpolarized sample
    by comparing a thermal dataset with a hyperpolarized one. Common sequences used are Singlepulse or NSPECT.

    Parameters
    ----------
    thermal_data : BrukerExp instance (e.g. NSPECT, Singlepulse)
        Contains thermal reference spectra.
    hyper_data : BrukerExp instance
        Contains hyperpolarized spectra with a given TR
    timepoint : float
        Time to where polarization level is backcalculated to.
    bg_region_hyper : list, optional
        ppm values where background region is taken from, by default [0,10]
    bg_region_thermal : list, optional
        ppm values where background region is taken from, by default [0,10]
    molarity_hyper : float, optional
        Molarity of hyperpolarized sample in mols / l, for pyruvate this is
        80mM, by default 0.08
    molarity_thermal : float, optional
        Molarity of thermal sample, by default 0.08
    T1_for_backcalculation : bool/float, optional
        Gives us the option to use an externally known T1 for the decay outside the bore in seconds.
        If False, it uses the calculated/flipangle corrected T1 inside the bore.
    linebroadening : float, optional
        Linebroadening applied to both spectra before integration, default is 0 Hz.
    integration_width_hyper : float, optional
        Integration width around peak of hyper spectra in ppm, default is 3.
    integration_width_thermal : float, optional
        Integration width around peak of thermal spectrum in ppm, default is 3.
    first_spec : int, optional
        First repetition that is used, by default 0.
    thermal_phase_input: float, optional
        Phase correction factor for thermal spectra in degree (0-360), obtained from phase_fid-function,
        by default False, i.e. no phase correction.
    hyper_phase_input: float, optional
        Phase correction factor for hyper spectra in degree (0-360), obtained from phase_fid-function,
         by default False, i.e. no phase correction.
    B_field : int, optional
        Magnetic field in spectrometer in Tesla, by default 1
    Temp : int, optional
        Temperature in the bore in degree Celsius, by default 28.5
    print_output : bool,optional
        Prints results into notebook, by default True

    Returns
    -------
    Polarization_level : float
        Polarization level in hyperpolarized state at time_to_diss in percent.
    T1_hyper_corr : float
        T1 decay constant in seconds, corrected for flipangle and TR.
    SNR_thermal : float
        Thermal SNR from measurement, corrected for flipangle and averages.
    SNR_hyper : float
        Hyperpolarized SNR backcalculated to time_to_diss, corrected for flipangle.
    Pol_lvl_thermal : float
        Thermal polarization level (NOT IN PERCENT)
    """
    # Step 1
    # calculate thermal polarization level according to Boltzmann
    Pol_lvl_thermal = np.tanh(
        co.hbar * 67.2828 * 1e6 * magnetic_field / (2 * co.k * (273.15 + Temperature))
    )

    # Step 2: calculate hyperpolarized SNR dependent on time
    # Norm all spectra to background region
    # Integrate the peak to obtain SNR array
    # fit it to exponential decay function
    # correct the T1 decay constant through the flipangle and TR of the sequence used to monitor the
    # hyperpolarized decay
    # convert input flip angle to radians
    FA_hyper = float(hyper_data.method["ExcPulse1"][2]) * np.pi / 180.0
    # get the number of spectra for hyper measurement
    Nspec_hyper = int(hyper_data.n_repetitions)
    # Repetition time
    TR_hyper = float(hyper_data.method["PVM_RepetitionTime"]) / 1000  # into s

    (
        ppm_axis_hyper,
        mag_spec_hyper,
        fids_hyper,
        hyper_complex_spec,
    ) = hyper_data.get_spec_non_localized_spectroscopy(cut_off=70)

    # take first spectrum thats wanted
    hyper_complex_spec_1 = hyper_complex_spec[first_spec]

    # perform baseline correction
    hyper_complex_spec_1_baseline_corr = hyper_complex_spec_1 - np.mean(
        hyper_complex_spec_1
    )
    # phase the real spectrum
    Integrals_hyp = []
    phases = np.linspace(0, 360, 1000)
    for phase in phases:
        itgl = np.max(
            np.real(
                hyper_complex_spec_1_baseline_corr
                * np.exp(1j * (phase * np.pi) / 180.0)
            )
        )
        Integrals_hyp.append(itgl)
    # take the real part of the spectrum
    if hyper_phase_input is False:
        final_phase_hyper = phases[
            np.argmin(np.abs(Integrals_hyp - np.max(Integrals_hyp)))
        ]
    else:
        final_phase_hyper = hyper_phase_input
    # apply phasing:
    first_hyper_spec = np.real(
        hyper_complex_spec_1_baseline_corr
        * np.exp(1j * (final_phase_hyper * np.pi) / 180.0)
    )
    # norm to background region
    hyper_normed = first_hyper_spec
    hyper_spectra = []
    # phase all spectra and dont norm them for T1 fitting
    for spectrum in range(Nspec_hyper):
        spec = np.fft.fftshift(np.fft.fft(fids_hyper[spectrum, :]))
        hyper_spectra.append(
            np.real(spec * np.exp(1j * (final_phase_hyper * np.pi) / 180.0))
        )
    hyper_spectra = np.array(hyper_spectra)
    # integrate a selected peak
    if select_peak_ppm_hyper:
        center_ppm_hyper = select_peak_ppm_hyper
    else:
        # otherwise find largest peak
        center_hyper = np.squeeze(np.where(hyper_normed - np.max(hyper_normed) == 0))
        center_ppm_hyper = ppm_axis_hyper[center_hyper]

    lower_bound_integration_ppm_hyper = np.abs(
        ppm_axis_hyper - (center_ppm_hyper - integration_width_hyper)
    )
    lower_bound_integration_index_hyper = np.argmin(
        lower_bound_integration_ppm_hyper - np.min(lower_bound_integration_ppm_hyper)
    )
    upper_bound_integration_ppm_hyper = np.abs(
        ppm_axis_hyper - (center_ppm_hyper + integration_width_hyper)
    )
    upper_bound_integration_index_hyper = np.argmin(
        upper_bound_integration_ppm_hyper - np.min(upper_bound_integration_ppm_hyper)
    )
    # from this we calculate the integrated peak region
    # sorted so that lower index is first
    integrated_peak_roi_hyper = [
        lower_bound_integration_index_hyper,
        upper_bound_integration_index_hyper,
    ]
    integrated_peak_roi_hyper.sort()

    time_of_first_spec = first_spec * TR_hyper
    SNR_hyper = np.sum(
        hyper_normed[integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]]
    )

    # fit exponential to hyper SNR to backcalculate
    def exponential(x, M, T1, offset):
        return M * np.exp(-x / T1) + offset

    # integrate the spectra, not the normalized spectra for the T1 fit, as
    # otherwhise there could be fit issues due to high noise
    Hyper_Signal_for_T1_fit = np.array(
        [
            np.sum(
                hyper_spectra[
                    spectrum,
                    integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1],
                ]
            )
            for spectrum in range(first_spec, Nspec_hyper)
        ]
    )

    # define the time axis during which the scans took place
    hyp_time_axis = np.arange(first_spec * TR_hyper, TR_hyper * Nspec_hyper, TR_hyper)
    # Fit
    coeff, err = curve_fit(
        exponential,
        hyp_time_axis,
        Hyper_Signal_for_T1_fit,
        p0=(np.max(Hyper_Signal_for_T1_fit), 50, np.mean(Hyper_Signal_for_T1_fit)),
    )
    # flipangle correct for time outside bore

    flipangle_corr_T1 = flipangle_corr(coeff[1], FA_hyper, TR_hyper)
    error_t1 = np.sqrt(np.diag(err))[1]
    # backcalculate to time of dissolution
    if T1_for_backcalculation:
        SNR_hyper_backcalculated = exponential(
            -time_to_diss, SNR_hyper, T1_for_backcalculation, 0
        )
    else:
        SNR_hyper_backcalculated = exponential(
            -time_to_diss, SNR_hyper, flipangle_corr_T1, 0
        )
    # now norm thermal
    # Step 3 Calculate thermal SNR
    FA_thermal = float(thermal_data.method["ExcPulse1"][2]) * np.pi / 180.0
    (
        ppm_axis_thermal,
        mag_spec_thermal,
        fids_thermal,
        complex_spec_thermal,
    ) = thermal_data.get_spec_non_localized_spectroscopy(linebroadening, 70)

    Nspec_thermal = float(thermal_data.n_repetitions) * float(
        thermal_data.method["PVM_NAverages"]
    )
    # option to only use one thermal spectrum
    if take_one_thermal_spec:
        Nspec_thermal = 1
        complex_spec_thermal_single = complex_spec_thermal[0]
        complex_spec_thermal_single_baselinecorr = (
            complex_spec_thermal_single - np.mean(complex_spec_thermal_single)
        )
        if thermal_phase_input is False:
            Integrals_therm = []
            phases = np.linspace(0, 360, 1000)
            for phase in phases:
                itgl = np.max(
                    np.real(
                        complex_spec_thermal_single_baselinecorr
                        * np.exp(1j * (phase * np.pi) / 180.0)
                    )
                )
                Integrals_therm.append(itgl)

            final_phase_therm = phases[
                np.argmin(np.abs(Integrals_therm - np.max(Integrals_therm)))
            ]
        else:
            final_phase_therm = thermal_phase_input
        phased_therm_spec = np.real(
            complex_spec_thermal_single_baselinecorr
            * np.exp(1j * (final_phase_therm * np.pi) / 180.0)
        )
        thermal_normed = phased_therm_spec

    else:
        # mean all thermal spectra we have
        if thermal_data.n_repetitions > 1:
            # mean thermal spectra if we have multiple that need to be averaged by us
            # i.e. Repetitions instead of Averages
            # number of thermal spectra
            Nspec_thermal = thermal_data.n_repetitions
            # average them
            therm_fid_averaged = np.sum(fids_thermal, axis=0)
            # calculate spectrum
            complex_spec_thermal_averaged = np.fft.fftshift(
                np.fft.fft(therm_fid_averaged)
            )
            complex_spec_thermal_averaged_baselinecorr = (
                complex_spec_thermal_averaged - np.mean(complex_spec_thermal_averaged)
            )
            if thermal_phase_input is False:
                Integrals_therm = []
                phases = np.linspace(0, 360, 1000)
                for phase in phases:
                    itgl = np.max(
                        np.real(
                            complex_spec_thermal_averaged_baselinecorr
                            * np.exp(1j * (phase * np.pi) / 180.0)
                        )
                    )
                    Integrals_therm.append(itgl)

                final_phase_therm = phases[
                    np.argmin(np.abs(Integrals_therm - np.max(Integrals_therm)))
                ]
            else:
                final_phase_therm = thermal_phase_input
            phased_complex_spec_thermal_averaged = np.real(
                complex_spec_thermal_averaged_baselinecorr
                * np.exp(1j * (final_phase_therm * np.pi) / 180.0)
            )
            # norm to background noise
            thermal_normed = phased_complex_spec_thermal_averaged

        elif thermal_data.method["PVM_NAverages"] > 1:
            # in case we have averages
            complex_spec_thermal_averaged_baselinecorr = complex_spec_thermal - np.mean(
                complex_spec_thermal
            )
            if thermal_phase_input is False:
                Integrals_therm = []
                phases = np.linspace(0, 360, 1000)
                for phase in phases:
                    itgl = np.max(
                        np.real(
                            complex_spec_thermal_averaged_baselinecorr
                            * np.exp(1j * (phase * np.pi) / 180.0)
                        )
                    )
                    Integrals_therm.append(itgl)

                final_phase_therm = phases[
                    np.argmin(np.abs(Integrals_therm - np.max(Integrals_therm)))
                ]
            else:
                final_phase_therm = thermal_phase_input
            phased_complex_spec_thermal_averaged = np.real(
                complex_spec_thermal_averaged_baselinecorr
                * np.exp(1j * (final_phase_therm * np.pi) / 180.0)
            )
            thermal_normed = phased_complex_spec_thermal_averaged

        else:
            pass

    # in case we want to integrate a specific peak
    if select_peak_ppm_thermal:
        center_ppm_thermal = select_peak_ppm_thermal
    else:
        # find largest peak
        center_thermal = np.squeeze(
            np.where(thermal_normed - np.max(thermal_normed) == 0)
        )
        print(np.where(thermal_normed - np.max(thermal_normed) == 0))
        center_ppm_thermal = ppm_axis_thermal[center_thermal]

    #  integrate around peak

    lower_bound_integration_ppm_thermal = np.abs(
        ppm_axis_thermal - (center_ppm_thermal - integration_width_thermal)
    )
    lower_bound_integration_index_thermal = np.argmin(
        lower_bound_integration_ppm_thermal
        - np.min(lower_bound_integration_ppm_thermal)
    )
    upper_bound_integration_ppm_thermal = np.abs(
        ppm_axis_thermal - (center_ppm_thermal + integration_width_thermal)
    )
    upper_bound_integration_index_thermal = np.argmin(
        upper_bound_integration_ppm_thermal
        - np.min(upper_bound_integration_ppm_thermal)
    )
    # from this we calculate the integrated peak region
    # sorted so that lower index is first
    integrated_peak_roi_thermal = [
        lower_bound_integration_index_thermal,
        upper_bound_integration_index_thermal,
    ]
    integrated_peak_roi_thermal.sort()

    # print(integrated_peak_roi_thermal)
    SNR_thermal = np.sum(
        thermal_normed[integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]]
    )
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    # Step 4: Correction factors
    Receiver_Gain_thermal = thermal_data.acqp["RG"]
    Receicer_Gain_hyper = hyper_data.acqp["RG"]

    correction_factor = (
        Nspec_thermal
        * (np.sin(FA_thermal) / np.sin(FA_hyper))
        * (molarity_thermal / molarity_hyper)
        * (Receiver_Gain_thermal / Receicer_Gain_hyper)
    )
    enhancement_factor = (SNR_hyper_backcalculated / SNR_thermal) * correction_factor
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    # Step 5: compare and plot results

    Polarization_level = Pol_lvl_thermal * enhancement_factor
    Polarization_level_at_first_spec = (
        Pol_lvl_thermal * (SNR_hyper / SNR_thermal) * correction_factor
    )
    Polarization_level = np.round(Polarization_level * 100, 1)
    Polarization_level_at_first_spec = np.round(
        Polarization_level_at_first_spec * 100, 1
    )

    if print_output is True:
        print("--------------------------------------------------------------")
        print(
            "Corrected observed T1=",
            np.round(coeff[1]),
            " s, for a flipangle of ",
            FA_hyper * 180 / np.pi,
            " ° and a TR of ",
            TR_hyper,
            " s ",
        )
        print("Resulting in T1_corr = ", np.round(flipangle_corr_T1, 1), " s")

        print(
            "Receiver Gain difference - Hyper RX Gain = ",
            Receicer_Gain_hyper,
            " vs Thermal RX Gain = ",
            Receiver_Gain_thermal,
        )
        print(
            "Molarity  difference - Hyper Sample 13C Molarity = ",
            molarity_hyper,
            " vs Thermal Sample 13C Molarity  = ",
            molarity_thermal,
        )
        print(
            "Number of spectra  difference - Hyper Scan 1 sample vs Thermal scan ",
            Nspec_thermal,
            " sample",
        )
        print(
            "Flipangle difference correction - Hyper flip angle ",
            FA_hyper * 180 / np.pi,
            " ° - vs Thermal flip angle ",
            FA_thermal * 180 / np.pi,
            " °",
        )
        print(
            "Enhancement factor from thermal to hyper",
            "{:.1e}".format(enhancement_factor),
        )
        if T1_for_backcalculation:
            print(
                "Externally used T1 from other fit function = ",
                np.round(T1_for_backcalculation, 1),
            )
        else:
            print(
                "T1_hyper_corr = ",
                np.round(flipangle_corr_T1, 1),
                "pm",
                np.round(error_t1, 1),
                " s",
            )

        print(
            "SNR_thermal normed to Molarity and Number of spectra",
            np.round(
                SNR_thermal
                * (molarity_hyper / molarity_thermal)
                / np.sqrt(Nspec_thermal),
                1,
            ),
        )
        print(
            "SNR_thermal / correction factor = ",
            np.round(SNR_thermal / correction_factor, 3),
        )
        print("--------------------------------------------------------------")
        print("THERMAL Polarization = ", Pol_lvl_thermal)
        print("SNR_thermal = ", np.round(SNR_thermal, 1))
        print("SNR_hyper_backcalculated = ", np.round(SNR_hyper_backcalculated, 1))
        print("SNR_hyper_at_first_spec = ", np.round(SNR_hyper, 1))

        time_of_first_spec = TR_hyper * first_spec
        print(
            "HYPER - Polarization level of first spec at T = ",
            time_of_first_spec,
            " s, is ",
            Polarization_level_at_first_spec,
            " %",
        )
        print(
            "HYPER - Polarization level at T = ",
            time_of_first_spec - time_to_diss,
            " s, is ",
            Polarization_level,
            " %",
        )
        print("--------------------------------------------------------------")
    else:
        pass

    if plot is True:
        fig, ax = plt.subplots(1, tight_layout=True, figsize=(6, 4))
        backcalc_axis = np.arange(
            time_of_first_spec - time_to_diss, TR_hyper * Nspec_hyper, TR_hyper
        )
        if T1_for_backcalculation:
            ax.plot(
                backcalc_axis,
                exponential(
                    backcalc_axis,
                    Polarization_level,
                    T1_for_backcalculation,
                    Pol_lvl_thermal,
                ),
            )
            ax.scatter(
                time_of_first_spec,
                exponential(
                    time_of_first_spec,
                    Polarization_level,
                    T1_for_backcalculation,
                    Pol_lvl_thermal,
                ),
                label="First measurement point",
                color="C1",
            )
        else:
            ax.plot(
                backcalc_axis,
                exponential(
                    backcalc_axis,
                    Polarization_level,
                    flipangle_corr_T1,
                    Pol_lvl_thermal,
                ),
            )
            ax.scatter(
                time_of_first_spec,
                exponential(
                    time_of_first_spec,
                    Polarization_level,
                    flipangle_corr_T1,
                    Pol_lvl_thermal,
                ),
                label="First measurement point",
                color="C1",
            )
        ax.set_xlabel("Time since start of experiment [s]")
        ax.set_ylabel(r"Polarization level [$\%$]")
        ax.set_title("Polarization level")
        ax.legend()

        fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(10, 4))
        ax[0].plot(thermal_data.ppm_axis, thermal_normed)
        ax[0].set_xlabel("ppm")
        ax[0].set_title("Thermal spectrum - " + str(Nspec_thermal) + " averages")
        ax[0].set_ylabel("I [a.u.]")

        ax[0].fill_between(
            [
                thermal_data.ppm_axis[integrated_peak_roi_thermal[0]],
                thermal_data.ppm_axis[integrated_peak_roi_thermal[1]],
            ],
            np.min(thermal_normed),
            np.max(thermal_normed),
            alpha=0.3,
            color="C1",
            label="Peak integration",
        )

        ax[1].plot(ppm_axis_hyper, hyper_normed)
        ax[1].set_xlabel("ppm")
        ax[1].set_title("First hyper spectrum")
        ax[1].set_ylabel("I [a.u.]")

        ax[1].fill_between(
            [
                ppm_axis_hyper[integrated_peak_roi_hyper[0]],
                ppm_axis_hyper[integrated_peak_roi_hyper[1]],
            ],
            np.min(hyper_normed),
            np.max(hyper_normed),
            alpha=0.3,
            color="C1",
            label="Peak integration",
        )
        ax[1].legend()

        ax[2].set_title("Hyper Signal for T1 fit")
        ax[2].scatter(hyp_time_axis, Hyper_Signal_for_T1_fit, label="Data points")
        backcalc_axis = np.arange(
            time_of_first_spec - time_to_diss, TR_hyper * Nspec_hyper, TR_hyper
        )
        ax[2].plot(
            hyp_time_axis,
            exponential(hyp_time_axis, coeff[0], coeff[1], coeff[2]),
            label="Fit - T1=" + str(np.round(coeff[1], 1)) + "s",
        )
        if T1_for_backcalculation:
            ax[2].plot(
                backcalc_axis,
                exponential(backcalc_axis, coeff[0], T1_for_backcalculation, coeff[2]),
                label="T1 manual input ="
                + str(np.round(T1_for_backcalculation, 1))
                + "s",
            )
        else:
            ax[2].plot(
                backcalc_axis,
                exponential(backcalc_axis, coeff[0], flipangle_corr_T1, coeff[2]),
                label="T1 corrected =" + str(np.round(flipangle_corr_T1, 1)) + "s",
            )

        ax[2].legend()
        ax[2].set_ylabel("Hyper Signal [a.u.]")
        ax[2].set_xlabel("Time since start of experiment [s]")
        ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        # third plot showing peak integration regions

        fig2, ax2 = plt.subplots(2, 2, figsize=(10, 4), tight_layout=True)
        ax2[0, 0].plot(
            ppm_axis_thermal[
                integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]
            ],
            thermal_normed[
                integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]
            ],
        )
        ax2[0, 1].plot(
            ppm_axis_hyper[integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]],
            hyper_normed[integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]],
        )

        ax2[0, 0].set_title("Thermal Signal region")
        ax2[0, 1].set_title("Hyper Signal region")
        x_data, y_data, bin_size = get_histogram(
            thermal_normed[
                integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]
            ],
            100,
        )
        ax2[1, 0].bar(x_data, y_data, width=bin_size)
        x_data, y_data, bin_size = get_histogram(
            hyper_normed[integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]],
            100,
        )
        ax2[1, 1].bar(x_data, y_data, width=bin_size)
        ax2[1, 0].set_title("Thermal Signal - Histogram")
        ax2[1, 1].set_title("Hyper Signal - Histogram")
        ax2[1, 0].set_xlabel("Signal value [a.u.]")
        ax2[1, 1].set_xlabel("Signal value [a.u.]")
        ax2[0, 0].set_xlabel("ppm")
        ax2[0, 1].set_xlabel("ppm")
        ax2[1, 0].set_ylabel("Nr Points")
        ax2[1, 1].set_ylabel("Nr Points")

    else:
        pass

    return (
        Polarization_level,
        Polarization_level_at_first_spec,
        SNR_thermal,
        SNR_hyper_backcalculated,
        Pol_lvl_thermal,
        enhancement_factor,
        flipangle_corr_T1,
    )
