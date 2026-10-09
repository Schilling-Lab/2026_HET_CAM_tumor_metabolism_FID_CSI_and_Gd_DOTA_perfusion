import glob
import json
from pathlib import Path

import cv2
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nrrd
import numpy as np
import pandas as pd
import pydicom
from IPython.display import display
from ipywidgets import fixed
from scipy.optimize import curve_fit
from scipy.stats import levene, linregress, normaltest, ranksums, ttest_ind

from ..system.logging import LOG_MODES, init_default_logger, underoverline

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")
# globally change log level, default is critical
logger.setLevel(LOG_MODES["critical"])


def tolerate(x, x_av, x_std, n=1.0):
    """
    Define a tolerance interval for a datapoint, as n*std from the mean value.
    If tolerance is not met, take mean value.

    Parameters
    ----------
    x: float
        x-value to study.
    x_av: float
        Average value of the dataset from which x came.
    x_std: float
        Standard deviation from the dataset from which x came.
    n: float (1.0, Optional)
        Datapoints more than n*std from x_av will not be tolerated.

    Returns
    -------
    x OR x_av: float

    """
    if x < x_av - x_std * n:
        return x_av
    elif x > x_av + x_std * n:
        return x_av
    else:
        return x


def axr_map_from_adcs(
    adc_stacked, tms=None, guess=np.array([0.8, 10**-3]), tolerance=4, slice=0
):
    """
    Calculate the AXR map from the ADC maps. Calculate time taken to do so. Read by various methods.

    Parameters
    ----------
    adc_stacked: np-ndarray
        ADC maps, shape [y,x,tm], where the first slice should be the filter off ADC map.
    tms: np.array (None, Optional)
        Array of mixing times, stacked as with adc_stacked.
    guess: np.ndarray  ([sigma, axr] = [0.8, 10 ** -3], Optional)
        Initial guess for the scipy.curve_fit best fit parameters. 1x2 array (sigma, AXR). Good guesses for the AXR
        will vary for different FEXI datasets, and are typically in the range of 10e-4.
    tolerance: float (4, Optional)
        tolerance factor for AXR fit. Datapoints more than tolerance*std away from the ADC mean in the slice will be
        replaced by the slice average.
    slice: int (0, Optional)
        Slice index for which to calculate AXR map.

    Returns
    -------
    axr: np.ndarray
        AXR map of dimensions [x,y].
    axr_err: np.ndarray
        Error in the AXR map, shape [y,x].
    sigma: np.ndarray
        Filter efficiency. See FEXI documentation for further details.

    """

    # Dimensions of ADC filter ON.
    dims = np.shape(adc_stacked)
    # Empty AXR map.
    axr = np.zeros(shape=(dims[0], dims[1], dims[3]))
    # Empty filter efficiency map (y-intercept)
    sigma = np.zeros(shape=(dims[0], dims[1], dims[3]))
    # Empty error map for AXR.
    axr_err = np.zeros(shape=(dims[0], dims[1], dims[3]))
    # Average of each ADC slice.
    adc_av_stacked = np.nanmean(adc_stacked, axis=(0, 1))
    # Std of each ADC slice.
    adc_std_stacked = np.nanstd(adc_stacked, axis=(0, 1))

    def func_axr(x, a, b):  # Function to fit/plot (AXR).
        return adc_stacked[k, j, 0, slice] * (1 - a * np.exp(-x * b))

    # tic = time.perf_counter()                                           #
    # Start timer.
    count = 0  # Count removed datapoints.
    for z in range(dims[3]):
        for j in range(dims[1]):
            for k in range(dims[0]):  # For each pixel.
                # Pixel data. Necessary to redefine, because it will be edited
                # for the fit.
                data = np.array([])
                data_tms = np.array([])
                if (
                    np.isnan(adc_stacked[k, j, 1:, z]).any() == False
                ):  # Check for unmasked pixels in ROI.
                    for t in range(len(tms[1:])):
                        data = np.append(
                            data,
                            tolerate(
                                adc_stacked[k, j, 1:, z][t],
                                adc_av_stacked[1:, z][t],
                                adc_std_stacked[1:, z][t],
                                n=tolerance,
                            ),
                        )
                        data_tms = np.append(data_tms, tms[1:][t])
                    try:
                        # Perform least sq. fit.
                        popt, pcov = curve_fit(func_axr, data_tms, data, guess)
                        axr[k, j, z] = popt[1]
                        sigma[k, j, z] = popt[0]
                        axr_err[k, j, z] = np.sqrt(np.diag(pcov))[1]
                    except BaseException:
                        logger.warning(f"AXR fit failed for pixel: {j, k, z}")
                        axr[k, j, z] = np.nan
                        sigma[k, j, z] = np.nan
                        axr_err[k, j, z] = np.nan
                else:  # Masked pixels
                    axr[k, j, z] = np.nan
                    sigma[k, j, z] = np.nan
                    axr_err[k, j, z] = np.nan

    try:
        x_0, y_0, z_0 = np.where(axr < 0)
        for i in range(len(x_0)):
            if axr_err[x_0[i], y_0[i], z_0[i]] < 1:  # Small errors, good fit.
                axr[x_0, y_0, z_0[i]] = 0
                axr_err[x_0, y_0, z_0[i]] = 0
                sigma[x_0, y_0, z_0[i]] = 0

            else:  # Big errors, failed fit.
                axr[x_0, y_0, z_0[i]] = np.nan
                axr_err[x_0, y_0, z_0[i]] = np.nan
                sigma[x_0, y_0, z_0[i]] = np.nan

        logger.debug(
            f"{len(x_0)} pixels suppressed due to negative AXR values, using fitting errors."
        )

    except BaseException:
        logger.warning(f"Failed to suppress pixels with negative AXR values.")

    # Rather than thresholding, use fitting errors to mask pixels.
    try:
        # Identify pixels with high errors.
        x_2, y_2, z_2 = np.where(axr_err > 1)

        # Fits have clearly failed, so set to NaN.
        axr[x_2, y_2, z_2] = np.nan
        axr_err[x_2, y_2, z_2] = np.nan
        sigma[x_2, y_2, z_2] = np.nan

        logger.debug(f"{len(x_2)} pixels thresholded due to silly AXR fits.")
    except BaseException:
        logger.warning(f"Failed to supress pixels with failed fitting.")

    return axr, axr_err, sigma


def fexi_pixel(
    off,
    on,
    j,
    k,
    factor=1,
    start_tm=None,
    stop_tm=None,
    roi_path=None,
    tolerance=1,
    alt_dims=False,
    alpha_fit=0.5,
    colour_fit="red",
    colour_data="black",
    save_path=None,
):
    """
    Calculate the ADC and AXR within a single pixel.

    Parameters
    ----------
    off: np.ndarray
        Filter OFF FEXI object.
    on: np.ndarray
        Filter ON FEXI object.
    j: int
        x-coordinate of pixel of interest.
    k: int
        y-coordinate of pixel of interest.
    factor: float (1, Optional)
        Datapoints more than factor*std away from the median signal values will be excluded from the ADC fit.
    start_tm: int (None, Optional)
        Index of the lowest mixing time to be used in the AXR model. By default, all are included.
    stop_tm: int (None, Optional)
        Index of the highest mixing time to be used in the AXR model. By default, all are included.
    roi_path: np.ndarray (None, Optional)
        File path to ROI. Optional.
    tolerance: float (4, Optional)
        Tolerance factor for AXR fit. Datapoints more than tolerance*std away from the ADC mean in the slice will be
        replaced by the slice average.
    alt_dims: bool (False, Optional)
            Alternate data acquisition protocol, where mixing times are cycled through with each repetition. Change to
            True if data was acquired with alternate protocol.
    alpha_fit: float (0.5, Optional)
        Opacity of the plotted fit.
    colour_fit: str ('red', Optional)
        Colour of the plotted fit line.
    colour_data: str ('black', Optional)
        Colour of the plotted data points.
    save_path: str (None, Optional)
        Filepath to which to save the main figure.


    Returns
    -------

    """
    # Image dimensions.
    dims = np.shape(off.img)
    if roi_path is None:
        logger.debug("No ROI provided. Proceeding with no mask.")
        roi = np.ones(shape=(150, 150))
    else:
        try:
            roi = read_mask_from_file(roi_path)
            logger.debug(f"Retrieved ROI from {roi_path}")
        except BaseException:
            logger.debug(f"Failed to retrieve ROI from {roi_path}")

    roi = cv2.resize(roi, dsize=(dims[0], dims[0]), interpolation=cv2.INTER_NEAREST)

    off.config(
        roi=roi, start_tm=start_tm, stop_tm=stop_tm, alt_dims=alt_dims, factor=factor
    )
    on.config(
        roi=roi, start_tm=start_tm, stop_tm=stop_tm, alt_dims=alt_dims, factor=factor
    )
    # Stack ADC maps for filter OFF and ON images [x,y,tm,z].
    adc = np.dstack((off.adc, on.adc))
    # Stack mixing times for filter OFF and ON images [tm].
    tms = np.append(off.mixing_times, on.mixing_times)
    # Stack s0 for filter ON and OFF images [x,y,tm,z].
    s0 = np.dstack((off.s0, on.s0))
    # Calculate AXR map for all slices [x,y,z].
    axr, axr_err, sigma = axr_map_from_adcs(adc, tms, tolerance=tolerance)

    def func_adc(x, a, b):  # Function to fit/plot (ADC).
        return -a * x + b

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), layout="constrained")

    fig.suptitle(f"FEXI Analysis for Pixel ({j},{k})")
    ax[0].set_box_aspect(1)  # Force square subplots.
    ax[1].set_box_aspect(1)

    def update_all_plots(
        time,
        slice,
    ):
        print("Running Update!")
        ax[0].cla()
        ax[1].cla()
        ax[0].set_xlabel("$t_{m}\\ [ms]$")
        ax[1].set_xlabel("$b\\ [s\\ mm^{{-2}}]$")
        ax[0].set_ylabel("$ADC\\ [x10^{-4}\\ mm^{2}\\ s^{-1}]$")
        ax[1].set_ylabel("$log(S_{b})\\ [A.U.]$")
        ax[0].set_title(f"AXR = {axr[k, j, slice - 1] * 10 ** 3:.3f} [s$^{{-1}}$]")
        ax[1].set_title(
            f"ADC = {adc[k, j, time, slice - 1] * 10**4:.3f} [x10$^{{-4}}$ mm$^{{2}}$ s$^{{-1}}$]"
        )

        adc_data_off = off.get_adc_data(k, j, 0, slice - 1)
        adc_data_on = on.get_adc_data(k, j, time - 1, slice - 1)

        adc_data_off_ = exclude_outliers(
            adc_data_off[0], adc_data_off[1], off.b_values, factor=factor
        )
        adc_data_on_ = exclude_outliers(
            adc_data_on[0], adc_data_on[1], on.b_values, factor=factor
        )
        if time == 0:  # Select filter off or on based on tm.
            adc_data = adc_data_off
            adc_data_ = adc_data_off_  # Minus outliers.
        else:
            adc_data = adc_data_on
            adc_data_ = adc_data_on_  # Minus outliers.

        # Dummy b-val data for plotting ADC fit.
        x__ = np.linspace(np.min(on.b_values), np.max(on.b_values), 100)
        # ADC fit to plot.
        fit_adc = func_adc(
            x=x__, a=adc[k, j, time, slice - 1], b=s0[k, j, time, slice - 1]
        )
        # ADC with outliers.
        ax[1].scatter(adc_data[0], adc_data[1], color="gray", marker="x", alpha=0.3)
        # ADC datapoints.
        ax[1].scatter(adc_data_[0], adc_data_[1], color=colour_data, marker="x")
        # Plot ADC fit.
        ax[1].plot(x__, fit_adc, color=colour_fit, alpha=alpha_fit)
        # Dummy tm data for plotting AXR fit.
        x_ = np.linspace(np.min(tms[1:]), np.max(tms[1:]), 100)

        # # Function to fit for average AXR.
        def func_axr(x, a, b):
            return adc[k, j, 0, 0] * (1 - a * np.exp(-x * b))

        # # Fit AXR to ADC datapoints.
        # params_axr, pcov_axr = curve_fit(
        #     func_axr, tms[1:],
        #     adc[k, j, 1:, 0],
        #     p0=guess_axr
        # )
        # Plot AXR fit.
        ax[0].plot(
            x_,
            func_axr(x_, sigma[k, j, 0], axr[k, j, 0]) * 10**4,
            color=colour_fit,
            alpha=alpha_fit,
            label=f"AXR={axr[k, j, 0] * 10**3:.3f}[s$^{{-1}}$]",
        )

        # ADC datapoints in AXR subplot.
        ax[0].scatter(
            tms, adc[k, j, :, slice - 1] * 10**4, marker="x", color=colour_data
        )
        # Highlght point corresponding to tm.
        ax[0].scatter(
            tms[time], adc[k, j, time, slice - 1] * 10**4, marker="x", color="orange"
        )

        # Connecting line after filter application.
        ax[0].plot(
            (tms[0], x_[0]),
            (
                adc[k, j, :, slice - 1][0] * 10**4,
                func_axr(x_, sigma[k, j, 0], axr[k, j, 0])[0] * 10**4,
            ),
            color=colour_fit,
            alpha=alpha_fit,
        )
        if save_path is not None:
            try:
                plt.savefig(f"{save_path}.svg", format="svg")
                logger.debug(f"Successfully saved to {save_path}")
            except BaseException:
                logger.warning(f"Could not save to {save_path}")

        return None

    # Slider to cycle through mixing times in ADC images.
    tm_slider = widgets.IntSlider(
        value=0, min=0, max=np.shape(adc)[2] - 1, description="tm: "
    )
    # Slider to cycle thru slices.
    slice_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=np.shape(adc)[3],
        description="Slice: ",
    )
    # Put sliders in a HBox.
    ui = widgets.HBox(
        [tm_slider, slice_slider],
        layout=widgets.Layout(display="flex"),
    )
    # Display our cool ui.
    display(ui)

    out = widgets.interactive_output(
        update_all_plots, {"time": tm_slider, "slice": slice_slider}
    )

    return None


def fexi(
    off=None,
    on=None,
    anat=None,
    guess_adc=[6, 10**-3],
    guess_axr=[0.8, 10**-4],
    factor=1,
    roi_path=None,
    bg=True,
    stats=False,
    plot=True,
    tolerance=1,
    start_tm=None,
    stop_tm=None,
    alt_dims=False,
    delete=([]),
    export_slice=0,
    title="",
    save_path=None,
    denoise=None,
    cmap_adc="jet",
    cmap_axr="inferno",
    return_maps=False,
):
    """
    Function to analyse FEXI parameters. ROI should be supplied for meaningful output.

    Parameters
    ----------
    off: np.ndarray
        Filter OFF FEXI object.
    on: np.ndarray
        Filter ON FEXI object.
    anat: np.ndarray
        MRIExperiment RARE object, for use as anatomical backgorund image in plotting.
    guess_adc: np.ndarray  (optional)
        Initial guess for the scipy.curve_fit best fit parameters. 1x2 array (S0, ADC). Good guesses for the ADC
        will vary for different FEXI datasets, and are typically in the order of 10e-3.
    guess_axr: np.ndarray  (optional)
        Initial guess for the scipy.curve_fit best fit parameters. 1x2 array (sigma, AXR). Good guesses for the AXR
        will vary for different FEXI datasets, and are typically in the order of 10e-4.
    factor: float (2)
        Datapoints more than factor*std away from the median signal values will be excluded from the ADC fit.
    roi_path: str (None)
        File path to ROI. Optional.
    bg: bool (True)
        If True, parametric maps will be displayed over the anatomical image in the background.
    stats: bool (False)
        If True, select attributes will be printed from the FEXI object.
    plot: bool (Ttrue, Optional)
        Set to False to run FEXI computations, but not create the figure.
    tolerance: float
        tolerance for AXR fitting function.
    start_tm: int (None)
        Index of the lowest mixing time to be used in ther AXR model. By default, all are included.
    stop_tm: int (None)
        Index of the highest mixing time to be used in ther AXR model. By default, all are included.
    alt_dims: bool (False)
        Set to True if data was acquired with alternative acquisition dimensions.
    delete: np.ndarray
        Array of datapoints to delete and exclude from fitting.
    export_slice: int
        slice for which to export FEXI dataframe. Temporary. Ultimately slices will be saved iteratively.
    title: str
        Title for main plot.
    save_path: str
        Filepath to which to save the main figure.
    denoise: str (None, optional)
        De-noising type to perform on raw data. to be read by FEXI.denoise().
    return_maps: bool (False)
        If True, return ADC and AXR maps as np.ndarray instead of key-value data series.

    Returns
    -------
    data_series: Pandas data series
        Indexed data series containing calculated ADC and AXR values.
    adc, axr: np.ndarray
        ADC and AXR maps.
    """

    allowed_anatomical_sequences = ["FLASH", "RARE"]

    if off is None:
        logger.critical("No filter OFF object provided.")
        raise SystemExit(0)
    elif lookup_sequence_class_name(off) != "FEXI":
        logger.warning("Filter OFF object is not a FEXI object.")
        raise SystemExit(0)
    elif off.filter != "off":
        logger.warning("Filter OFF object is not as it seems... Aborting.")
        raise SystemExit(0)
    if on is None:
        logger.warning("No filter ON object provided.")
        raise SystemExit(0)
    elif lookup_sequence_class_name(on) != "FEXI":
        logger.warning("Filter ON object is not a FEXI object.")
        raise SystemExit(0)
    elif on.filter != "on":
        logger.warning("Filter ON object is not as it seems... Aborting.")
        raise SystemExit(0)

    dims = np.shape(off.img)
    if roi_path is None:
        logger.warning(
            "No ROI provided, proceeding with no mask. Supply ROI path for meaningful analysis."
        )
        roi = np.ones(shape=(150, 150))
    else:
        try:
            roi = read_mask_from_file(roi_path)
            logger.debug(f"Retrieved ROI from {roi_path}")
        except BaseException:
            logger.debug(
                f"Failed to retrieve ROI from {roi_path}. Proceeding without masking."
            )

    roi = cv2.resize(roi, dsize=(dims[0], dims[0]), interpolation=cv2.INTER_NEAREST)
    # Configure FEXI objects (rearrange dims, mask, calculate adc, trim tm
    # dimension).
    off.config(
        roi=roi,
        factor=factor,
        start_tm=start_tm,
        stop_tm=stop_tm,
        alt_dims=alt_dims,
        guess=guess_adc,
        denoise=denoise,
    )
    on.config(
        roi=roi,
        factor=factor,
        start_tm=start_tm,
        stop_tm=stop_tm,
        alt_dims=alt_dims,
        guess=guess_adc,
        denoise=denoise,
    )
    # Exclude anomalous coordinate points.
    off.exclude(coords=delete)
    on.exclude(coords=delete)
    if anat is not None:
        if lookup_sequence_class_name(anat) not in allowed_anatomical_sequences:
            logger.warning(
                f"Anatomical image type '{lookup_sequence_class_name(anat)}' unsupported. Supported sequences: {allowed_anatomical_sequences}. Aborting."
            )
            raise SystemExit(0)
        else:  # anatomical image ok. Proceeding.
            logger.debug("Anatomical image provided.")
            # Take middle slice from anatomical.
            anat = sort_anatomical(anat)
            # Align anatomcial.
            anat = align_anatomical(img_1=anat, img_2=on.img[:, :, 0, 0, 0, 0])
            logger.debug("Anatomical image realigned.")
    else:  # No anatomical image given.
        bg = False
        logger.debug("Proceeding with no anatomical image.")

    # Stack ADC maps for filter OFF and ON images.
    adc = np.dstack((off.adc, on.adc))
    # Stack mixing times for filter OFF and ON images.
    tms = np.append(off.mixing_times, on.mixing_times)
    # Calculate AXR map.
    axr, axr_err, sigma = axr_map_from_adcs(
        adc,
        tms,
        tolerance=tolerance,
    )

    # Average ADC values in ROI.
    adc_avs = np.nanmean(adc, axis=(0, 1))

    # Function to fit for average AXR.
    def func(x, a, b):
        return adc_avs[0, 0] * (1 - a * np.exp(-x * b))

    # Calculate AXR fit parameters for the ADCs in the ROI.
    try:
        # Fit AXR to average ADCS in ROI.
        params, pcov = curve_fit(func, tms[1:], adc_avs[1:, export_slice], p0=guess_axr)
        perr = np.sqrt(np.diag(pcov))
        # [axr,sigma,axr_err,sigma_err]
        axr_fit_params = [params[1], params[0], perr[1], perr[0]]
        logger.debug("AXR successfully fitted for average ADC in ROI.")
    except BaseException:
        logger.warning(
            f"Could not fit average AXR data: {tms[1:], adc_avs[1:, export_slice]}"
        )
        axr_fit_params = [0, 0, 0, 0]

    if stats:
        on.print_stats()

    # Important parameters.
    adc_av_roi = np.nanmean(adc, axis=(0, 1))
    adc_std_roi = np.nanstd(adc, axis=(0, 1))
    axr_av_pixelwise = np.nanmean(axr, axis=(0, 1))
    axr_std_pixelwise = np.nanstd(axr, axis=(0, 1))
    sig_av_pixelwise = np.nanmean(sigma, axis=(0, 1))
    sig_std_pixelwise = np.nanstd(sigma, axis=(0, 1))

    n_decimal = 2  # Number of decimal points to which to round saved data.

    data = np.array(
        (
            axr_av_pixelwise[export_slice] * 10**3,
            axr_std_pixelwise[export_slice] * 10**3,
            sig_av_pixelwise[export_slice] * 100,
            sig_std_pixelwise[export_slice] * 100,
            axr_fit_params[0] * 10**3,
            axr_fit_params[2] * 10**3,
            axr_fit_params[1] * 100,
            axr_fit_params[3] * 100,
        )
    )
    index = [
        "AXR av. [s^1]",
        "AXR std. [s^1]",
        "Filter eff. av. [%]",
        "Filter eff. std. [%]",
        "AXR ROI av. [s^1]",
        "AXR ROI std. [s^1]",
        "Filter eff. ROI av. [%]",
        "Filter eff. ROI std. [%]",
    ]
    for tm in range(len(tms)):
        data = np.append(data, adc_av_roi[tm, export_slice] * 10**4)
        data = np.append(data, adc_std_roi[tm, export_slice] * 10**4)
        index = np.append(index, f"ADC$_{tm}$ av. [10^4 mm^2 s^-1]")
        index = np.append(
            index,
            f"ADC$_{tm}$ std. [10^4 mm^2 s^-1]",
        )

    # Return panda series, for use in dataframe.
    import pandas as pd

    data_series = pd.Series(data, index=index)
    print("")
    print("FEXI ANALYSIS")
    underoverline(text="FEXI Parameters (from average ADC in ROI):")
    print(
        f"Average AXR: {axr_fit_params[0] * 10 ** 3: .3f} + /- {axr_fit_params[2] * 10 ** 3: .3f}[s ^ (-1)]"
    )
    print(
        f"Average filter eff.: {axr_fit_params[1] * 100: .2f} + /- {axr_fit_params[3] * 100: .2f} [ % ]"
    )
    print("")
    underoverline(text="FEXI Parameters (pixelwise):")
    print(
        f"Average AXR: {axr_av_pixelwise[export_slice] * 10 ** 3: .3f} + /- {axr_av_pixelwise[export_slice] * 10 ** 3: .3f}[s ^ (-1)]"
    )
    print(
        f"Median AXR:              {np.nanmedian(axr[:, :, export_slice]) * 10 ** 3:.3f} [s^(-1)]"
    )
    print(
        f"Average filter eff.: {sig_av_pixelwise[export_slice] * 100: .2f} + /- {sig_std_pixelwise[export_slice] * 100: .2f} [ % ]"
    )
    print("")

    if plot:
        plot_fexi(
            adc_map=adc,
            axr_map=axr,
            sigma_map=sigma,
            anat_map=anat,
            adc_avs=adc_avs,
            t_mix=tms,
            title=title,
            axr_fit_params=axr_fit_params,
            bg=bg,
            save_path=save_path,
            cmap_adc=cmap_adc,
            cmap_axr=cmap_axr,
        )
    if return_maps:
        return adc, axr
    else:
        return data_series


def plot_fexi(
    adc_map,
    axr_map,
    sigma_map,
    anat_map,
    adc_avs,
    t_mix,
    title,
    slice=0,
    bg=True,
    axr_fit_params=None,
    save_path=None,
    cmap_adc="jet",
    cmap_axr="inferno",
):
    """
    Function called by various methods to visualise FEXI analysis. Not to be called by user.

    Parameters
    ----------
    adc_map: np.ndarray
        ADC maps, for filter off and on, stacked. Shape [y,x,N_tm +1]
    axr_map: np.ndarray
        AXR map, shape [y,x]
    sigma_map: np.ndarray
        Filter efficiency map, shape [y,x]
    anat_map: np.ndarray
        Anatomical image for background plots. The lower resolution parametric maps will be interpolated to the same
        resolution as the anatomical images.
    adc_avs: np.ndarray
        Average ADC value in the ROI for ech region of interest.
    t_mix: np.ndarray
        Array of mixing times, for filter off and onm, stacked.
    title: str
        Title for the FEXI plot, to be taken from whatever method calls plot_fexi.
    slice: int
        Slice to display (if applicable).
    bg: bool (True)
        If True, parametric maps will be displayed over the anatomical image in the background.
    axr_fit_params: np.ndarray  (optional)
        Supplied by calling function.
    save_path: str
        Filepath to which to save the main figure.

    Returns
    -------
    """
    # Interpolate to anatomical img resolution.
    if bg:
        adc_map_ = np.zeros(
            shape=(
                np.shape(anat_map)[1],
                np.shape(anat_map)[0],
                np.shape(adc_map)[2],
                np.shape(adc_map)[3],
            )
        )
        axr_map_ = np.zeros(
            shape=(np.shape(anat_map)[1], np.shape(anat_map)[0], np.shape(adc_map)[3])
        )
        sigma_map_ = np.zeros(
            shape=(np.shape(anat_map)[1], np.shape(anat_map)[0], np.shape(adc_map)[3])
        )
        interp_type = cv2.INTER_NEAREST
        # Interpolate all parameter maps to resolution of anatomical images,
        # for each slice.
        for z in range(np.shape(adc_map)[-1]):
            adc_map_[:, :, :, z] = cv2.resize(
                adc_map[:, :, :, z],
                dsize=(np.shape(anat_map)[1], np.shape(anat_map)[0]),
                interpolation=interp_type,
            )
            axr_map_[:, :, z] = cv2.resize(
                axr_map[:, :, z],
                dsize=(np.shape(anat_map)[1], np.shape(anat_map)[0]),
                interpolation=interp_type,
            )
            sigma_map_[:, :, z] = cv2.resize(
                sigma_map[:, :, z],
                dsize=(np.shape(anat_map)[1], np.shape(anat_map)[0]),
                interpolation=interp_type,
            )
    else:
        adc_map_ = adc_map
        axr_map_ = axr_map
        sigma_map_ = sigma_map

    # Standard deviations of the ADC ROI.
    adc_std = np.nanstd(adc_map, axis=(0, 1))
    # Create figure.
    fig, ([ax0, ax1], [ax2, ax3]) = plt.subplots(
        nrows=2, ncols=2, figsize=(8, 8), facecolor="white"
    )
    # Figure title.
    fig.suptitle(f"FEXI Analysis: {title}")
    #  Adjust subplot spacing.
    plt.subplots_adjust(wspace=0.5, hspace=0.01)
    """----------- inital plots / cbars ----------- """
    # Plot ForeGround images.
    fg_adc = ax0.imshow(
        adc_map_[:, :, 0, slice] * 10**4,
        vmin=0,
        vmax=12,
        alpha=1,
        cmap=cmap_adc,
    )
    fg_axr = ax2.imshow(
        axr_map_[:, :, slice] * 10**3,
        alpha=1,
        cmap=cmap_axr,  # Show AXR map, in SI unit.
        origin="upper",
        vmin=np.nanmin(axr_map_[:, :, slice] * 10**3),
        vmax=np.nanmax(axr_map_[:, :, slice] * 10**3),
    )
    fg_sig = ax3.imshow(
        sigma_map_[:, :, slice] * 100, alpha=1, origin="upper", vmin=0, vmax=100
    )

    # Create colourbars.
    cbar_adc = colourbar(fg_adc)
    cbar_axr = colourbar(fg_axr)
    cbar_sig = colourbar(fg_sig)
    # Label colourbars.
    cbar_adc.set_label(
        r"x$10^{-4}\ mm^{2}\ s^{-1}$", rotation=270, loc="center", labelpad=15
    )
    cbar_sig.set_label(r"$\%$", rotation=270, loc="center", labelpad=15)
    cbar_axr.set_label(r"$s^{-1}$", rotation=270, loc="center", labelpad=15)

    def update_all_plots(time=0, max_adc=12, max_axr=5, max_sig=50, slice=0, alph=0.8):
        # Format subplots in figure.
        ax0.cla()
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax0.set_title(f"ADC")
        ax1.set_title(f"ROI Averages")  # Set title.
        ax2.set_title(f"AXR")  # Title
        ax3.set_title(f"$\\sigma$")  # Title

        if bg:  # Plot background anatomical images.
            bg_adc_update = ax0.imshow(anat_map, cmap="gray", origin="upper")
            bg_axr_update = ax2.imshow(anat_map, cmap="gray", origin="upper")
            bg_sig_update = ax3.imshow(anat_map, cmap="gray", origin="upper")

        # Plot updates foreground images.
        fg_adc_update = ax0.imshow(
            adc_map_[:, :, time, slice - 1] * 10**4,
            vmin=4,
            vmax=max_adc,
            alpha=alph,
            cmap=cmap_adc,
            zorder=1,
        )
        fg_axr_update = ax2.imshow(
            axr_map_[:, :, slice - 1] * 10**3,
            vmin=0,
            vmax=max_axr,
            alpha=alph,
            cmap=cmap_axr,
            origin="upper",
        )
        fg_sigma_update = ax3.imshow(
            sigma_map_[:, :, slice - 1] * 100,
            vmin=0,
            vmax=max_sig,
            alpha=alph,
            origin="upper",
            zorder=1,
        )
        # Update colour limits.
        fg_axr.set_clim([0, max_axr])
        fg_sig.set_clim([1, max_sig])
        ax0.annotate(
            text=f"$t_{{m}} = {t_mix[time]}$ [ms]",
            xy=[1, np.shape(adc_map_[:, :, time, slice - 1])[1] - 2],
            color="white",
        )
        ax0.set_xlabel(
            rf"$ADC_{{av}}={np.nanmean(adc_map[:, :, time, slice - 1]) * 10 ** 4: .4f}\ [x10 ^ {{-4}}\ mm ^ {{2}}\ s ^ {{-1}}]$"
            # +- {np.nanstd(adc_map[:, :, time]) * 10 ** 4:.4f} [x10^{{-4}}\ mm^{{2}}\ s^{{-1}}]$'
        )
        ax1.set_xlabel("$t_{m}\\ [ms]$")
        ax2.set_xlabel(
            rf"$AXR_{{av}} = {np.nanmean(axr_map[:, :, slice - 1] * 10 ** 3):.4f}\ [s^{{-1}}]$"
        )
        ax3.set_xlabel(
            rf"$\sigma_{{av}} = {np.nanmean(sigma_map[:, :, slice - 1] * 100):.4f}\ [\%]$"
        )

        """------------ Panel 2 -------------"""

        def func(x, a, b):  # Function to fit for average AXR.
            return adc_avs[0, 0] * (1 - a * np.exp(-x * b))

        # Simulated x-data.
        X_ = np.linspace(t_mix[1], t_mix[-1], 100)
        # Plot fit.
        ax1.plot(
            X_,
            func(X_, axr_fit_params[1], axr_fit_params[0]) * 10**4,
            color="red",
            label="Best fit",
            zorder=0,
            linestyle="dashed",
        )
        # Plot average ADCs' errorbars.
        ax1.errorbar(
            t_mix[1:],
            adc_avs[1:, slice - 1] * 10**4,
            yerr=adc_std[1:, slice - 1] * 10**4,
            color="gray",
            marker=".",
            capsize=5,
            linestyle="",
            zorder=5,
            alpha=0.3,
        )
        # Plot average ADCs on top.
        ax1.scatter(
            t_mix[1:],
            adc_avs[1:, slice - 1] * 10**4,
            color="black",
            marker=".",
            linestyle="",
            zorder=10,
            alpha=1,
        )
        # Plot Filter off average ADC.
        ax1.hlines(
            y=adc_avs[0, slice - 1] * 10**4,
            xmin=0,
            xmax=t_mix[-1],
            color="black",
            linestyles="dashed",
            label="Filter OFF",
        )
        # Display label in legend.
        ax1.legend(loc="lower right")
        asp = (
            np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0]
        )  # Force aspect ratio to conform.
        ax1.set_aspect(asp)
        ax1.set_ylabel("$ADC\\ [x10^{-4}\\ mm^{2}\\ s^{-1}]$")  # y-label.

    """ ----------------------- Sliders ----------------------- """
    # Slider to cycle through mixing times in ADC images.
    tm_slider = widgets.IntSlider(
        value=0, min=0, max=np.shape(adc_map)[2] - 1, description="Mixing time: "
    )
    # Slider to change opacity of parameter maps.
    alpha_slider = widgets.FloatSlider(
        value=1,
        min=0,
        max=1,
        step=0.1,
        description="Opacity: ",
    )
    # Slider for upper limit of AXR colourbar.
    axr_slider = widgets.IntSlider(
        value=8,
        min=1,
        max=25,
        description="AXR: ",
        # max = int(np.nanmax(axr_map * 10 ** 3)),
    )
    # Slider for upper limit of ADC colourbar.
    adc_slider = widgets.IntSlider(
        value=12,
        min=1,
        max=25,
        description="ADC: ",
        # max = int(np.nanmax(axr_map * 10 ** 3)),
    )
    sigma_slider = widgets.IntSlider(
        value=50,
        min=1,
        max=100,
        description="$\\sigma$: ",
    )
    # Slider for upper limit of AXR colourbar.
    slice_slider = widgets.IntSlider(
        value=0,
        min=1,
        max=np.shape(adc_map)[3],
        description="Slice: ",
        # max = int(np.nanmax(axr_map * 10 ** 3)),
    )
    # Put sliders in a HBox.
    row_1 = widgets.HBox([adc_slider, axr_slider, sigma_slider])
    row_2 = widgets.HBox(
        [tm_slider, slice_slider, alpha_slider],
    )
    ui = widgets.VBox(
        [row_1, row_2],
        layout=widgets.Layout(display="flex"),
    )

    # Display our cool ui.
    display(ui)

    out = widgets.interactive_output(
        update_all_plots,
        {
            "time": tm_slider,
            "max_adc": adc_slider,
            "max_axr": axr_slider,
            "max_sig": sigma_slider,
            "slice": slice_slider,
            "alph": alpha_slider,
        },
    )

    if save_path is not None:
        try:
            plt.savefig(f"{save_path}.svg", format="svg")
            logger.debug(f"Successfully saved to {save_path}")
        except BaseException:
            logger.warning(f"Could not save to {save_path}")
    else:
        logger.debug("Figure not saved.")

    return None


def box_plot(norm, alt, tms, title="", filepath=None):
    # Button to save fig as .svg file.
    button = widgets.Button(description="Save Fig.")
    # Put sliders in a HBox.
    ui = widgets.HBox(
        [button],
        layout=widgets.Layout(display="flex"),
    )
    # Display our cool ui.
    display(ui)

    # Remove NaN values, boxplot does not like them very much.
    datasets = [norm.dropna(), alt.dropna()]

    # Set x-positions for boxes
    x_pos_range = np.arange(len(datasets)) / (len(datasets) - 1)
    x_pos = (x_pos_range * 0.5) + 0.75

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    # Properties for box plot.
    colour = "black"
    colours = ["red", "blue"]
    alpha = 1
    linewidth = 1
    linestyle = "-"

    boxprops = dict(
        linestyle=linestyle,
        color=colour,
        linewidth=linewidth,
        alpha=alpha,
        facecolor="None",
    )
    whiskerprops = dict(
        linestyle=linestyle, color=colour, linewidth=linewidth, alpha=alpha
    )
    capprops = dict(linestyle=linestyle, color=colour, linewidth=linewidth, alpha=alpha)
    medianprops = dict(
        linestyle=linestyle, color=colour, linewidth=linewidth, alpha=alpha
    )
    # flierprops     = dict(marker='--', markerfacecolor='black', markersize=12,markeredgecolor='none')
    meanpointprops = dict(
        marker="x", markeredgecolor="black", markerfacecolor="firebrick"
    )
    meanlineprops = dict(linestyle="-", linewidth=2.5, color="black")
    series = []

    for i, data in enumerate(datasets):
        bp = ax.boxplot(
            np.array(data),
            sym="",
            whis=[0, 100],
            widths=0.6 / len(datasets),
            labels=list(datasets[0]),
            positions=[x_pos[i] + j * 1 for j in range(len(data.T))],
            patch_artist=True,
            boxprops=boxprops,
            medianprops=medianprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            # meanlineprops=meanlineprops,
            meanline=True,
            # showmeans=True
        )

        # Adding to dataframe.
        if i == 0:
            acq = "norm"
        else:
            acq = "alt"
        index = []
        medians = []

        # i refers to the dataset, ie norm or alt. tm refers to the mixing
        # time.
        for tm in range(len(tms)):
            med = bp["medians"][tm].get_ydata()[0]
            medians.append(med)
            index.append(f"tm$_{{{tm}}}$")
        s = pd.Series(medians, index=index)
        series.append(s)

    ax.set_title(title)
    ax.set_ylabel("ADC [$x10^{-4}mm^{2}s^{-1}$]")
    ax.set_xlabel("$t_{m}$")
    ax.grid(visible=True, which="minor")
    ax.set_xticks(np.arange(len(list(datasets[0]))) + 1)
    plt.gca().xaxis.set_minor_locator(
        ticker.FixedLocator(np.array(range(len(list(datasets[0])) + 1)) + 0.5)
    )
    plt.gca().tick_params(axis="x", which="minor", length=4)
    plt.gca().tick_params(axis="x", which="major", length=0)
    ax.set_xlim([0.5, len(list(datasets[0])) + 0.5])

    p_values = []

    # Plot datapoints.
    for tm in range(len(tms)):
        for acq in range(len(datasets)):
            data = datasets[acq]

            y = data[str(tm)]
            x = np.random.normal(tm + 1, 0.04, len(y))

            if acq == 0:
                offset = -0.25  # norm.
                ttest_x = y
            else:
                offset = +0.25  # alt.
                ttest_y = y

            ax.scatter(x + offset, y, marker=".", alpha=0.1, color=colours[acq])

        med_x = np.median(ttest_x)
        med_y = np.median(ttest_y)
        p_val_t = ttest_ind(ttest_x, ttest_y)[1]
        p_val_ranksums = ranksums(ttest_x, ttest_y)[1]
        p_val_normaltest = normaltest(ttest_x)[1]
        p_val_normaltest_alt = normaltest(ttest_y)[1]
        p_val_levene = levene(ttest_x, ttest_y)[1]
        print(f"Mixing time: {tm}")
        print("------------------")
        print(f"Sample size:          n = {np.shape(ttest_x)}")
        print(f"Sample size:          n = {np.shape(ttest_y)}")
        print(f"Median (standard):        {med_x}")
        print(f"Median (alt):             {med_y}")
        print(f"Normal (standard)  test:  {p_val_normaltest:.15f}")
        print(f"Normal (alternate) test:  {p_val_normaltest_alt:.15f}")
        print(f"Levene test:              {p_val_levene:.8f}")
        print(f"t-test:                   {p_val_t:.15f}")
        print(f"rank-sums test:           {p_val_ranksums:.15f}")

        print()

        # p_val_normaltest
        p_values.append(p_val_t)
        # print(f'tm:{tm} ({tms[tm]}ms),p:{p_val:.4f}')
        # ax.annotate(f'$p={p_val:.4f}$',xy = (tm+0.75,4.8),alpha=0.4)
    # print(p_values)

    def save_button_clicked(b):
        # So as not to overwrite figure, append tm to file names.
        if filepath:
            try:
                plt.savefig(f"{filepath}.svg", format="svg", dpi=300)
                print("Figure saved!")
            except BaseException:
                print("Could not save figure!")
        else:
            print("To save figure, please specify filepath.")

    button.on_click(save_button_clicked)

    return series


def dti_boxplot(data, labels, title=""):
    # Properties for box plot.
    colour = "black"
    alpha = 0.4
    linewidth = 0.6
    linestyle = "-"
    boxprops = dict(
        linestyle=linestyle,
        color=colour,
        linewidth=linewidth,
        alpha=alpha,
        facecolor="None",
    )
    whiskerprops = dict(
        linestyle=linestyle, color=colour, linewidth=linewidth, alpha=alpha
    )
    capprops = dict(linestyle=linestyle, color=colour, linewidth=linewidth, alpha=alpha)
    medianprops = dict(
        linestyle=linestyle, color=colour, linewidth=linewidth, alpha=alpha
    )

    plt.figure()
    plt.title(title)
    plt.xlabel("Mouse ID")
    plt.ylabel("ADC $[x10^{-4}mm^{2}s^{-1}]$")
    bp = plt.boxplot(
        data,
        sym="",
        whis=[0, 100],
        # widths=0.6 / len(data),
        labels=labels,
        # positions=[x_pos[i] + j * 1 for j in range(len(data.T))],
        patch_artist=True,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        # meanlineprops=meanlineprops,
        meanline=True,
        # showmeans=True
    )

    for i in range(len(data)):
        plt.scatter(
            x=np.random.normal(i + 1, 0.04, len(data[i])),
            y=data[i],
            marker=".",
            alpha=0.1,
            color="gray",
        )
        minimum = min(arr.min() for arr in data)
        maximum = max(arr.max() for arr in data)

        plt.annotate(
            f"$\\sigma={np.nanstd(data[i]):.3f}$",
            xy=(i + 0.75, maximum + 0.5),
            alpha=0.4,
        )

    plt.ylim(minimum - 0.5, maximum + 1)
    plt.show()
    return None


def correlate_adc(
    off,
    on,
    off_alt,
    on_alt,
    roi=np.ones(shape=(32, 32)),
    filepath=None,
    return_datapoints=True,
    cmap_adc="spring",
):
    """
    Create a correlation plot between the ADC values for each pixel between two different FEXI acquisitions, with different
    acquisition methods.

    Parameters
    ----------
    off: FEXI object
        Filter OFF FEXI object, normal acquisition.
    off_alt: FEXI object
        Filter OFF FEXI object, alternate acquisition.
    on: FEXI object
        Filter ON FEXI object, normal acquisition.
    on_alt: FEXI object
        Filter ON FEXI object, alternate acquisition.
    roi: np.ndarray
        The ROI drawn for FEXI analysis.
    filepath: string
        filepath to which to save the current figure to. Filter value (off/on) and mixing time will be appended to file
        name.
    return_datapoints: bool
        If true, the adc datapoints will be returned as a flattened array for further use.
    cmap_adc: string
        Colour map which the plots will adopt.

    Returns
    -------

    """
    # Check filters are the same.
    if off.filter != off_alt.filter and on.filter != on_alt.filter:
        print(
            "Filters do not match. Please supply object with the same filter parameter."
        )

    # Configure FEXI objects (rearrange dims, mask, calculate adc, trim tm
    # dimension).
    off.config(roi=roi, alt_dims=False)
    off_alt.config(roi=roi, alt_dims=True)
    on.config(roi=roi, alt_dims=False)
    on_alt.config(roi=roi, alt_dims=True)
    # Stack ADC maps for filter OFF and ON images.
    adc = np.dstack((off.adc, on.adc))
    adc_alt = np.dstack((off_alt.adc, on_alt.adc))
    # Stack mixing times for filter OFF and ON images.
    tms = np.append(off.mixing_times, on.mixing_times)

    # Slider to cycle through mixing times in ADC images.
    tm_slider = widgets.IntSlider(
        value=0, min=0, max=len(tms) - 1, description="Mixing time: "
    )
    # Button to save fig as .svg file.
    button = widgets.Button(description="Save Fig.")

    # Put sliders in a HBox.
    ui = widgets.HBox(
        [tm_slider, button],
        layout=widgets.Layout(display="flex"),
    )
    # Display our cool ui.
    display(ui)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    # Figure title.
    fig.suptitle(f"ADC Correlation Analysis")
    #  Adjust subplot spacing.
    plt.subplots_adjust(wspace=0.5, hspace=0.01)
    # Initialise plots.
    init_norm = ax[0].imshow(
        adc[:, :, 0] * 10**4,
        vmin=np.nanmin([adc * 10**4, adc_alt * 10**4]),
        vmax=np.nanmax([adc * 10**4, adc_alt * 10**4]),
        cmap=cmap_adc,
    )
    init_alt = ax[2].imshow(
        adc_alt[:, :, 0] * 10**4,
        vmin=np.nanmin([adc * 10**4, adc_alt * 10**4]),
        vmax=np.nanmax([adc * 10**4, adc_alt * 10**4]),
        cmap=cmap_adc,
    )
    # Create colourbars.
    colourbar(init_norm)
    colourbar(init_alt)

    def update(t=0):
        ax[0].cla()
        ax[1].cla()
        ax[2].cla()

        # Flatten ADC maps.
        x = adc[:, :, t].flatten() * 10**4
        y = adc_alt[:, :, t].flatten() * 10**4

        # Dummy data for plotting.
        np.linspace(np.min(x), np.max(x), 100)

        # identify and remove NaN values from both arrays.
        nan_coords = np.argwhere(np.isnan(x)).flatten()
        x = np.delete(x, nan_coords)
        y = np.delete(y, nan_coords)

        nan_coords = np.argwhere(np.isnan(y)).flatten()
        x = np.delete(x, nan_coords)
        y = np.delete(y, nan_coords)

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        # print(f'p-val: {p_value}')

        ax[0].set_title("Normal Acq.")
        ax[1].set_title("Correlation Plot.")
        ax[2].set_title("Alternative Acq.")
        ax[0].set_xlabel(
            f"ADC$_{{av}}$={np.nanmean(adc[:, :, t]) * 10 ** 4:.3f} $[x10^{{-4}}mm^{{2}}s^{{-1}}]$"
        )
        ax[1].set_xlabel("ADC (Normal Acq.) $[x10^{-4}mm^{2}s^{-1}]$")
        ax[1].set_ylabel("ADC (Alt. Acq.)   $[x10^{-4}mm^{2}s^{-1}]$")
        ax[2].set_xlabel(
            f"ADC$_{{av}}$={np.nanmean(adc_alt[:, :, t]) * 10 ** 4:.3f} $[x10^{{-4}}mm^{{2}}s^{{-1}}]$"
        )
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[2].set_xticks([])
        ax[2].set_yticks([])

        ax[0].imshow(
            adc[:, :, t] * 10**4,
            vmin=np.nanmin([adc * 10**4, adc_alt * 10**4]),
            vmax=np.nanmax([adc * 10**4, adc_alt * 10**4]),
            cmap=cmap_adc,
        )
        ax[1].plot(x, x, color="red", alpha=0.5)
        ax[1].scatter(
            x,
            y,
            marker=".",
            alpha=0.5,
            color="black",
            label=f"r={r_value:.3f}",
            # \nR$^{{2}}$={r_value**2:.3f}'
        )
        ax[2].imshow(
            adc_alt[:, :, t] * 10**4,
            vmin=np.nanmin([adc * 10**4, adc_alt * 10**4]),
            vmax=np.nanmax([adc * 10**4, adc_alt * 10**4]),
            cmap=cmap_adc,
        )

        from matplotlib.ticker import FormatStrFormatter

        ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        asp = (
            np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
        )  # Force aspect ratio to conform.
        ax[1].set_aspect(asp)
        ax[1].legend()

    plt.show()
    out = widgets.interactive_output(update, {"t": tm_slider})

    def save_button_clicked(b):
        # So as not to overwrite figure, append tm to file names.
        if tm_slider.value == 0:
            filter = "off"
        else:
            filter = "on"
        if filepath:
            try:
                plt.savefig(
                    f"{filepath}_filter={filter}_tm={tm_slider.value}.svg",
                    format="svg",
                    dpi=300,
                )
            except BaseException:
                print("Could not save figure.")

    button.on_click(save_button_clicked)

    if return_datapoints:
        return_norm = np.split(adc[:, :, :].flatten(), indices_or_sections=5, axis=0)
        return_alt = np.split(adc_alt[:, :, :].flatten(), indices_or_sections=5, axis=0)
        return return_norm, return_alt


def fexi_summary(
    df_1,
    df_2,
    df_3,
    subj_id="",
    days=[0, 1, 2, 4],
    regions=["Region 1", "Region 2", "Region 3"],
    title=None,
    ylim_axr=[-0.5, 8.8],
):
    """
    Function to plot a 3x3 summary of FEXI analysis from a single subject in three different ROIs. Designed for use in
    the FEXI GBM pilot study.

    Parameters
    ----------
    df_1: pandas dataframe
        Dataframe containing the series returned from four different sessions in ROI 1.
    df_2: pandas dataframe
        Dataframe containing the series returned from four different sessions in ROI 2.
    df_3: pandas dataframe
        Dataframe containing the series returned from four different sessions in ROI 3.
    subj_id: str
        ID of the subject imaged, for use in the plot title.
    days: list
        Days on which imaging was performed.
    regions: list
        Names of the three ROIs.
    title: str (None)
        Figure suptitle.
    ylim_axr: list  ([-1,10])
        Upper and lower limits for the y-axes of the AXR plots.

    Returns
    -------

    """
    # ADCs from dataframes.
    mean_1 = np.mean(df_1.iloc[8])
    mean_2 = np.mean(df_2.iloc[8])
    mean_3 = np.mean(df_3.iloc[8])
    std_1 = np.std(df_1.iloc[8])
    std_2 = np.std(df_2.iloc[8])
    std_3 = np.std(df_3.iloc[8])

    means = [mean_1, mean_2, mean_3]
    stds = [std_1, std_2, std_3]

    # For setting axis values.
    # x = [0, 1, 2, '', 4]
    # xi = list(range(len(x)))
    xlim = [-0.5, days[-1] + 0.5]

    print("ADC Average for the week:")
    print("-------------------------")

    fig, ax = plt.subplots(
        nrows=2, ncols=3, figsize=(10, 7), sharex="col", sharey="row"
    )
    if title:
        fig.suptitle(f"{title} \n{subj_id}")
    fig.supxlabel("Days post-Radiotherapy")

    # Labels for y-axis, either side.
    y_labels_left = ["AXR [$s^{-1}$]", "ADC [$x10^{-4}mm^{2}s^{-1}$]"]

    # Plotting parameters.
    colour = "black"
    marker = "."
    alpha = 0.7
    alpha_err = 0.3

    for i in range(3):  # iterate over columns.
        # Top row.
        ax[0, i].set_title(f"{regions[i]}")
        ax[0, i].set_box_aspect(1)
        # ax[0, i].set_xticks(xi, x)
        ax[0, i].set_xlim(xlim[0], xlim[1])
        ax[0, i].set_ylim(ylim_axr[0], ylim_axr[1])

        # Bottom row.
        ax[1, i].set_box_aspect(1)
        ax[1, i].set_xlabel("Day")
        # ax[1, i].set_xticks(xi, x)
        ax[1, i].set_xlim(xlim[0], xlim[1])

        # Plot means.
        ax[1, i].hlines(
            y=means[i],
            xmin=xlim[0],
            xmax=xlim[1],
            alpha=0.2,
            color="gray",
            linestyle="dashed",
        )

        # Print ADC means.
        print(f"{regions[i]}: ADCav = {means[i]:.4f} +/- {stds[i]:.4f}")

    # y-labels.
    ax[0, 0].set_ylabel(y_labels_left[0])
    ax[1, 0].set_ylabel(y_labels_left[1])

    # Plot AXR datapoints, no error bars.
    ax[0, 0].errorbar(
        x=days, y=df_1.iloc[4], marker=marker, linestyle="", alpha=alpha, color=colour
    )
    ax[0, 1].scatter(
        x=days, y=df_2.iloc[4], marker=marker, linestyle="", alpha=alpha, color=colour
    )
    ax[0, 2].errorbar(
        x=days, y=df_3.iloc[4], marker=marker, linestyle="", alpha=alpha, color=colour
    )
    # Plot ADC datapoints with errorbars.
    ax[1, 0].errorbar(
        x=days,
        y=df_1.iloc[8],
        yerr=df_1.iloc[9],
        marker=marker,
        linestyle="",
        capsize=5,
        alpha=alpha_err,
        color="gray",
    )
    ax[1, 1].errorbar(
        x=days,
        y=df_2.iloc[8],
        yerr=df_2.iloc[9],
        marker=marker,
        linestyle="",
        capsize=5,
        alpha=alpha_err,
        color="gray",
    )
    ax[1, 2].errorbar(
        x=days,
        y=df_3.iloc[8],
        yerr=df_3.iloc[9],
        marker=marker,
        linestyle="",
        capsize=5,
        alpha=alpha_err,
        color="gray",
    )
    # Plot ADC datapoints on top, without errorbars.
    ax[1, 0].scatter(
        x=days, y=df_1.iloc[8], marker=marker, linestyle="", alpha=alpha, color=colour
    )
    ax[1, 1].scatter(
        x=days, y=df_2.iloc[8], marker=marker, linestyle="", alpha=alpha, color=colour
    )
    ax[1, 2].scatter(
        x=days, y=df_3.iloc[8], marker=marker, linestyle="", alpha=alpha, color=colour
    )

    plt.subplots_adjust(wspace=0.1, hspace=0.01)

    plt.show()

    return None


def analyse_adc(
    off,
    on,
    anat=None,
    guess_adc=[6, 10**-3],
    roi_path=np.array([]),
    colours=["red", "blue", "green"],
    ylim=[10, 12.5],
    markers=["o", "o", "o"],
    factor=1,
    offsets=[-10, 0, +10],
    filepath=None,
):
    dims = np.shape(off.img)

    # Retrieve ROI
    if roi_path is None:
        logger.debug("No ROI provided. Proceeding with no mask.")
        roi = np.ones(shape=(150, 150))
    else:
        try:
            roi = read_mask_from_file(roi_path)
            logger.debug(f"Retrieved ROI from {roi_path}")
        except BaseException:
            logger.debug(f"Failed to retrieve ROI from {roi_path}")

    # Resize ROI.
    roi = cv2.resize(roi, dsize=(dims[0], dims[0]), interpolation=cv2.INTER_NEAREST)
    off.config(roi=roi, factor=factor)
    for obj in on:
        obj.config(roi=roi, factor=factor)

    # Sort anatomical image.
    if anat is not None:
        logger.debug("Anatomical image provided.")
        # Take middle slice from anatomical.
        anat = sort_anatomical(anat)
        # Align anatomcial.
        anat = align_anatomical(img_1=anat, img_2=on[0].img[:, :, 0, 0, 0, 0])
        logger.debug("Anatomical image realigned.")

    tms = np.append(off.mixing_times, on[0].mixing_times)
    # Figure.
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    fig.suptitle(f"ADC Analysis")

    adc_avs_all = np.array([])
    adc_std_all = np.array([])

    for i in range(len(on)):
        ax.set_xlabel("$t_{m}\\ [ms]$")
        ax.set_ylabel("$ADC\\ [x10^{{-4}}\\ mm^{2}\\ s^{-1}]$")
        ax.set_ylim(ylim[0], ylim[1])
    for i in range(len(on)):
        ax.set_title(f"$b_{{f}}={on[i].bf:.0f}$ s mm$^{{-2}}$")
        adc_avs = np.array([])
        adc_std = np.array([])
        adc_avs = np.append(adc_avs, np.nanmean(on[i].adc, axis=(0, 1)))
        adc_std = np.append(adc_std, np.nanstd(on[i].adc, axis=(0, 1)))

        adc_avs_all = np.append(adc_avs_all, adc_avs)
        adc_std_all = np.append(adc_std_all, adc_std)

        ax.errorbar(
            tms[1:] + offsets[i],
            adc_avs * 10**4,
            yerr=adc_std * 10**4,  # Plot average ADCs
            color=colours[i],
            marker=markers[i],
            capsize=5,
            linestyle="",
            zorder=10,
            alpha=0.5,
            # label=f'Filter ON',
        )
        # hline for filter off value.
    ax.axhline(
        y=np.nanmean(off.adc) * 10**4,
        color="black",
        alpha=0.3,
        linestyle="--",
        # label='Filter OFF'
    )
    # Fill-between plots filter OFF ADC and std.
    y1 = (np.nanmean(off.adc) + np.nanstd(off.adc)) * 10**4
    y2 = (np.nanmean(off.adc) - np.nanstd(off.adc)) * 10**4
    ax.fill_between(
        x=[tms[0] - 50, tms[-1] + 50],
        y1=y1,
        y2=y2,
        color="gray",
        alpha=0.2,
    )
    ax.set_xlim(tms[0] - 20, tms[-1] + 20)
    ax.set_xticks([7.458, 57.458, 107.458, 157.458, 207.458, 257.458, 307.458])
    fig.tight_layout()
    if filepath is not None:
        try:
            plt.savefig(f"{filepath}.svg", format="svg", dpi=300)
        except BaseException:
            logger.warning("Could not save figure.")
    return None


def mean_diffusivity(
    objs_off,
    objs_on,
    anat,
    start_tm=None,
    stop_tm=None,
    roi=np.array([]),
    factor=1,
    bg=True,
):
    """
    Calculate mean diffusivity maps from diffusion measurments in three orthogonal directions.

    Parameters
    ----------
    objs_off: np.ndarray
        Array of filter OFF objects, with diffusion gradients in orthogonal directions.
    objs_on: np.ndarray
        Array of filter ON objects, with diffusion gradients in orthogonal directions.
    anat: Bruker Obj
        Anatomical image for background.
    start_tm: int (None)
        Index of the lowest mixing time to be used in ther AXR model. By default, all are included.
    stop_tm: int (None)
        Index of the highest mixing time to be used in ther AXR model. By default, all are included.
    roi: np.ndarray (None)
        ROI mask for thr dataset. Should have shape [y,x]. Optional.
    factor: float (2)
        Datapoints more than factor*std away from the median signal values will be excluded from the ADC fit.
    bg: bool (True)
        If True, parametric maps will be displayed over the anatomical image in the background.
    """
    anat = sort_anatomical(
        anat
    )  # Take middle slice, average over repetitions if necessary.

    for obj in objs_off:  # Configure objects for FEXI analysis.
        obj.config(
            roi,
            start_tm,
            stop_tm,
            factor=factor,
        )

    # Order filter-on dimensions, apply mask, and calculate ADC map.
    for obj in objs_on:
        obj.config(
            roi,
            start_tm,
            stop_tm,
            factor=factor,
        )

    adc_1 = np.dstack((objs_off[0].adc, objs_on[0].adc))
    adc_2 = np.dstack((objs_off[1].adc, objs_on[1].adc))
    adc_3 = np.dstack((objs_off[2].adc, objs_on[2].adc))
    tms = np.append(objs_off[0].mixing_times, objs_on[0].mixing_times)

    mean_adc_map = np.zeros(shape=np.shape(adc_1))
    for t in range(len(tms)):
        slices_to_average = np.dstack((adc_1[:, :, t], adc_2[:, :, t], adc_3[:, :, t]))
        mean_adc_map[:, :, t] = np.mean(slices_to_average, axis=2)

    axr, axr_err, sig = axr_map_from_adcs(
        mean_adc_map,
        tms,
    )

    plot_fexi(
        adc_map=mean_adc_map,
        axr_map=axr,
        sigma_map=sig,
        anat_map=anat,
        t_mix=tms,
        bg=bg,
        title="Mean Diffusivity",
    )


def testing_fexi_dropdown(
    off,
    on,
    anat=None,
    guess_adc=[6, 10**-3],
    guess_axr=[0.8, 10**-4],
    factor=1,
    roi_dict=[],
    bg=False,
    stats=True,
    tolerance=1,
    start_tm=None,
    stop_tm=None,
    alpha=0.5,
):
    roi_selector_options = ["1", "2", "3"]

    roi_selector = widgets.Dropdown(
        options=roi_selector_options, value="1", description="ROI:"
    )
    row1 = widgets.HBox([roi_selector])
    display(row1)
    fexi(
        off,
        on,
        anat=anat,
        guess_adc=guess_adc,
        guess_axr=guess_axr,
        factor=factor,
        roi=roi_dict[roi_selector.value],
        bg=bg,
        stats=stats,
        tolerance=tolerance,
        start_tm=start_tm,
        stop_tm=stop_tm,
        alpha=alpha,
    )


def sort_anatomical(anat_obj):
    """
    Takes anatomical image obj, shape [y,x, slice, rep]. Averages over repetitions, and takes middle slice for
    displaying in FEXI analysis.

    Parameters
    ----------
    obj: Bruker Obj
        Anatomical image, as provided by the scanner.

    Returns
    -------
    anat_img:
        Middle slice of the image object, averaged over repetitions, with an extra (empty) dimension added as is
        required for some methods in the FEXI analysis.

    """
    # Extract image file from object.
    anat_img = anat_obj.seq2d
    # Extract image dimensions.
    dims_anat = np.shape(anat_img)
    # If there are repetitions in the file, average over them.
    if len(dims_anat) > 3:
        print("Found repetitions in anatomical image data. Taking average.")
        anat_img = np.nanmean(anat_img, axis=-1)  # but why tho.
    slice = dims_anat[2] // 2  # Find middle slice.
    # print(f'Taking slice {slice+1} of {dims_anat[-1]} from anatomical image.')
    anat_img = anat_img[:, :, slice]
    anat_img = np.expand_dims(anat_img, axis=-1)
    return anat_img


def align_anatomical(img_1, img_2):
    """
    Function takes two images acquired with different matrix sizes. img_1 is alligned to img_2; img_2 is unchanged.

    Parameters
    ----------
    img_1: np.ndarray
        Image to shift.
    img_2: np.ndarray
        Reference image to which img_1 will be alligned.
    """
    dims_1 = np.shape(img_1)
    dims_2 = np.shape(img_2)
    a = [0.5, 0.5]  # coords in high res img.
    b = [
        dims_1[0] / (dims_2[0] * 2),
        dims_1[0] / (dims_2[0] * 2),
    ]  # coords in low res img.
    x = np.abs(a[0] - b[0])
    y = np.abs(a[1] - b[1])
    d = np.sqrt(x**2 + y**2)  # distance to shift (Pythagoras).
    from scipy.ndimage import shift

    img_1_alligned = shift(img_1, shift=d, mode="wrap")

    return img_1_alligned


def exclude_outliers(x, y, bvalues, factor=1):
    """
    Takes data for ADC fit and excludes values more than std*factor away from median value of the repetitions.

    Parameters
    ----------
    x: np.ndarray
        x-data to fit.
    y: np.ndarray
        y-data to fit.
    bvalues: np.ndarray
        Array of b-values used in the filter ON acquisition.
    factor: float (1)
        Datapoints more than factor*std away from the median signal values will be excluded from the ADC fit.

    Returns
    -------
    x_new: np.ndarray
        New x-coordinate array with outliers removed.
    y_new: np.ndarray
        New y-coordinate array with outliers removed.
    """
    y_new = np.array([])  # New arrays minus the outliers.
    x_new = np.array([])

    for b in bvalues:
        ydata = y[x == b]  # Separate data by b-values.
        xdata = x[x == b]
        median = np.median(ydata)  # Median.
        sigma = np.std(ydata)  # Std dev.
        low = median - factor * sigma  # Lower data limit.
        high = median + factor * sigma  # Upper data limit.
        """
        print('b:              ', b)
        print('median:         ', median,'std:', sigma)
        print('Deleting below: ', low)
        print('Deleting above: ', high)
        print('Values too low:  ', ydata[too_low])  # Check visually that the numbers make sense.
        print('Values too high: ', ydata[too_high])
        """
        too_low = np.where(ydata <= low)[0]  # Indicies of points too low.
        too_high = np.where(ydata >= high)[0]  # Indicies of points too high.
        to_delete = np.append(  # Concatonate arrays of indicies to delete.
            too_low, too_high
        )
        ydata = np.delete(ydata, to_delete)  # Delete outlaying y-data.
        xdata = np.delete(xdata, to_delete)  # Delete corresponding x-data.
        x_new = np.append(x_new, xdata)
        y_new = np.append(y_new, ydata)
    # print('Shape of data with outliers:   ', x.shape, y.shape)
    # print('Shape of data without outliers:', x_new.shape, y_new.shape)
    return x_new, y_new

    # def draw_roi(anat_obj, n_rois=3, filepath=None):
    """
    Draw roi on anatomical image. Need only supply the anatomical OBJECT, and the number of rois to be drawn.

    Parameters
    ----------
    anat_obj: RARE obj, or other Brucker obj.
        Anatomical image object from Bruker scanner. No preprocessing required.
    n_rois: int (3)
        Maximum number of ROIs that can be drawn and saved.
    filepath: str (None)
        Filepath to which to save the mask, if the save button is pressed.

    Returns
    -------
    segm_list: dict
        Segmenter list to be read by utils_anatomical.get_masks()
    """

    # Retrieve middle slice from anatomical image object.
    img_to_segment = sort_anatomical(anat_obj)
    # Create segmenter list.
    segm_list = ut_anat.get_segmenter_list(img_to_segment, n_rois=n_rois)
    # Draw and save ROIs drawn into segm_list.
    ut_anat.draw_masks_on_anatomical(segm_list)
    # Button to save fig as .svg file.
    button = widgets.Button(description="Save Mask.")
    # Put sliders in a HBox.
    ui = widgets.HBox(
        [button],
        layout=widgets.Layout(display="flex"),
    )
    # Display our cool ui.
    display(ui)

    def save_button_clicked(b):
        if filepath:
            try:
                mask = ut_anat.get_masks(segm_list)["1"][:, :, 0]
                np.savetxt(f"{filepath}", mask, delimiter=",")
                print(f"Mask saved to {filepath}. Dimensions {np.shape(mask)}")
                # plt.savefig(f"{filepath}.svg", format='svg', dpi=300)
            except BaseException:
                print("Could not save mask.")

    button.on_click(save_button_clicked)

    return segm_list


# def get_roi(roi_list, id=1):
#     """
#     Retrieve mask from roi_list, and interpolate to a given image size.

#     Parameters
#     ----------
#     segm_list: dict
#         Dictionary of segmenter objects.
#     id: int (1)
#         Dictionary index of mask to retrieve.
#     interp: int (32)
#         Number of pixels to which the mask will be interpolated. Should match FEXI image dimensions.

#     Returns
#     -------
#     mask: np.ndarray
#         ROI mask to be applied to FEXI images.
#     """

#     # Retrieve mask.
#     mask = ut_anat.get_masks(roi_list)[str(id)]

#     return mask


def read_mask_from_file(filepath):
    """
    Function to read masks that have been saved to a text file, for example by the draw_roi() function.

    Parameters
    ----------
    filepath: str
        Full filepath to the mask file to be read.

    Returns
    -------
    mask: np.ndarray
        Mask array for use in FEXI analysis. Has shape [y,x,chan].

    """

    try:
        mask = np.genfromtxt(filepath, delimiter=",")
        np.shape(mask)
        logger.debug(f"Reading mask from filepath {filepath}.")
    except BaseException:
        logger.debug(
            f"Could not load mask from filepath {filepath}. \nNo pixels will be masked."
        )
        mask = np.ones(shape=(140, 140))

    return mask


def inspect_anatomical(anat_obj):
    """
    Function to voew anatomical images. Interactive widget with features to scroll through slices.

    Parameters
    ----------
    anat_obj: RARE object (or similar; must have <<seq2d>> attribute.)

    Returns
    -------
    """
    from matplotlib.transforms import Affine2D

    allowed_anatomical_sequences = ["FLASH", "RARE"]
    if anat_obj is not None:
        if lookup_sequence_class_name(anat_obj) not in allowed_anatomical_sequences:
            logger.warning(
                f"Image type '{lookup_sequence_class_name(anat_obj)}' unsupported. Supported sequences: {allowed_anatomical_sequences}. Aborting."
            )
            raise SystemExit(0)
        else:  # anatomical image ok. Proceeding.
            logger.debug("Supported anatomical image provided. Proceeding.")
    else:
        logger.warning(
            f"Please supply anatomical image object. Supported sequences: {allowed_anatomical_sequences}. Aborting."
        )
        raise SystemExit(0)

    img = anat_obj.seq2d
    slices = img.shape[2]
    fig, ax = plt.subplots(nrows=1, ncols=1)

    def update(i, c, r, n):
        ax.cla()
        ax.set_title(f"slice {i + 1} of {n}")
        Affine2D().rotate_deg(r)
        ax.imshow(
            img[:, :, i],
            cmap="gray",
            origin="upper",
            vmax=c * np.nanmax(img),
        )

    slice_slider = widgets.IntSlider(
        value=0, min=0, max=slices - 1, description="Slice: "
    )
    contrast_slider = widgets.FloatSlider(
        value=1, min=0.01, max=1, step=0.01, description="Contrast: "
    )
    rotate_slider = widgets.IntSlider(
        value=0, min=0, max=270, step=45, description="Rotate: "
    )
    # Put sliders in a HBox.
    ui = widgets.HBox(
        [slice_slider, contrast_slider, rotate_slider],
        layout=widgets.Layout(display="flex"),
    )
    # Display our cool ui.
    display(ui)
    out = widgets.interactive_output(
        update,
        {
            "i": slice_slider,
            "c": contrast_slider,
            "r": rotate_slider,
            "n": fixed(slices),
        },
    )
    return None


def colourbar(mappable):
    """
    Method to map a colour bar to an image without distorting image proportions.

    Parameters
    ----------
    mappable: AxesImageImage
        Object to which the colour bar is mapped.

    Returns
    -------
    cbar
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def fexi_multisl(
    off=None,
    on=None,
    anat=None,
    guess_adc=[6, 10**-3],
    guess_axr=[0.8, 10**-4],
    factor=1,
    roi_path=None,  # change description
    bg=True,
    stats=False,
    plot=True,
    tolerance=1,
    start_tm=None,
    stop_tm=None,
    alt_dims=False,
    delete=([]),
    export_slice=0,
    title="",
    save_path=None,
    denoise=None,
    complex_val=False,
):
    """
    Function to analyse FEXI parameters. ROI should be supplied for meaningful output.

    Parameters
    ----------
    off: np.ndarray
        Filter OFF FEXI object.
    on: np.ndarray
        Filter ON FEXI object.
    anat: np.ndarray
        MRIExperiment RARE object, for use as anatomical backgorund image in plotting.
    guess_adc: np.ndarray  (optional)
        Initial guess for the scipy.curve_fit best fit parameters. 1x2 array (S0, ADC). Good guesses for the ADC
        will vary for different FEXI datasets, and are typically in the order of 10e-3.
    guess_axr: np.ndarray  (optional)
        Initial guess for the scipy.curve_fit best fit parameters. 1x2 array (sigma, AXR). Good guesses for the AXR
        will vary for different FEXI datasets, and are typically in the order of 10e-4.
    factor: float (2)
        Datapoints more than factor*std away from the median signal values will be excluded from the ADC fit.
    roi_path: str (None)
        File path to ROI. Optional.
    bg: bool (True)
        If True, parametric maps will be displayed over the anatomical image in the background.
    stats: bool (False)
        If True, select attributes will be printed from the FEXI object.
    plot: bool (Ttrue, Optional)
        Set to False to run FEXI computations, but not create the figure.
    tolerance: float
        tolerance for AXR fitting function.
    start_tm: int (None)
        Index of the lowest mixing time to be used in ther AXR model. By default, all are included.
    stop_tm: int (None)
        Index of the highest mixing time to be used in ther AXR model. By default, all are included.
    alt_dims: bool (False)
        Set to True if data was acquired with alternative acquisition dimensions.
    delete: np.ndarray
        Array of datapoints to delete and exclude from fitting.
    export_slice: int
        slice for which to export FEXI dataframe. Temporary. Ultimately slices will be saved iteratively.
    title: str
        Title for main plot.
    save_path: str
        Filepath to which to save the main figure.
    denoise: str (None, optional)
        De-noising type to perform on raw data. to be read by FEXI.denoise().
    complex_val: bool (False, optional)
            Parameter to deal with different recostruction. If False the seq2d contains real data, if True the the recostruction
            gives the 2 channel of the quadrature detection with complex values.


    Returns
    -------
    data_series: pandas data series
        Indexed data series containing all of the calculated FEXI parameters.
    """

    if off is None:
        logger.warning("No filter OFF object provided.")
        raise SystemExit(0)
    elif off.filter != "off":
        logger.warning("Filer OFF object is not as it seems...")
        raise SystemExit(0)
    if on is None:
        logger.warning("No filter ON object provided.")
        raise SystemExit(0)
    elif on.filter != "on":
        logger.warning("Filer ON object is not as it seems...")
        raise SystemExit(0)
    dims = np.shape(off.img)
    nslice = off.nslices

    anat_img = anat.Load_2dseq_file(recon_num=1)

    if roi_path is None:
        logger.debug("No ROI provided. Proceeding with no mask.")
        roi = np.ones(shape=(np.shape(anat_img)[0], np.shape(anat_img)[1], nslice))
    else:
        mask_dict = from_mask_file_to_roi_list(
            roi_path,
            nslice,
            anat_dim1=np.shape(anat_img)[0],
            anat_dim2=np.shape(anat_img)[1],
        )
        mask = from_roilist_to_mask_multisl(mask_dict)
        print("MASK COMPUTED")
        roi = mask
    if len(np.shape(roi)) < 3:
        roi = np.expand_dims(roi, axis=-1)
    roi_resized = []
    for i in range(nslice):
        resized_img = cv2.resize(
            roi[:, :, i], dsize=(dims[0], dims[0]), interpolation=cv2.INTER_NEAREST
        )
        roi_resized.append(resized_img)
    roi_resized = np.array(roi_resized)
    roi_resized = np.moveaxis(roi_resized, 0, -1)

    # Configure FEXI objects (rearrange dims, mask, calculate adc, trim tm
    # dimension).
    off.config_j(
        roi=roi_resized,
        factor=factor,
        start_tm=start_tm,
        stop_tm=stop_tm,
        alt_dims=alt_dims,
        guess=guess_adc,
        denoise=denoise,
        complex_val=complex_val,
    )
    on.config_j(
        roi=roi_resized,
        factor=factor,
        start_tm=start_tm,
        stop_tm=stop_tm,
        alt_dims=alt_dims,
        guess=guess_adc,
        denoise=denoise,
        complex_val=complex_val,
    )
    # Exclude anomalous coordinate points.
    off.exclude(coords=delete)
    on.exclude(coords=delete)
    if anat is not None:
        logger.debug("Anatomical image provided.")

        anat_fromseq = anat.Load_2dseq_file(recon_num=1)
        anat_img = np.copy(anat_fromseq)
        dims_anat = np.shape(anat_img)
        if (
            len(dims_anat) > 3
        ):  # If there are repetitions in the file, average over them.
            print("Found repetitions in anatomical image data. Taking average.")
            anat_img = np.nanmean(anat_img, axis=-1)

        for i in range(nslice):
            anat_img[:, :, i] = align_anatomical(
                img_1=anat_img[:, :, i], img_2=on.img[:, :, 0, 0, 0, 0]
            )
    else:
        bg = False
    # Stack ADC maps for filter OFF and ON images.
    adc = np.dstack((off.adc, on.adc))
    np.dstack((off.adc_err, on.adc_err))
    print("adc shape ", np.shape(adc))
    # Stack mixing times for filter OFF and ON images.
    tms = np.append(off.mixing_times, on.mixing_times)
    # Calculate AXR map.
    axr, axr_err, sigma = axr_map_from_adcs_multisl(
        adc,
        tms,
        tolerance=tolerance,
    )

    # Average ADC values in ROI.
    adc_avs = np.nanmean(adc, axis=(0, 1))

    # Header.
    # from IPython.display import display, Markdown
    # display(Markdown('![figure](/Users/bendanzertum/Desktop/FEXI/header.svg)'))
    n_sl = np.shape(off.img)[6]
    adc_avs_res = np.array(adc_avs)
    for sl in range(n_sl):
        adc_avs_res[:, sl] = adc_avs_res[:, sl] / adc_avs[0, sl]

    # Function to fit for average AXR.
    def func(x, a, b):
        return 1 - a * np.exp(-x * b)

    # Calculate AXR fit parameters for the ADCs in the ROI.
    datas = []

    try:
        tms_res = tms / 1000.0
        fit_val = np.zeros((n_sl, (np.shape(on.img)[5])))
        # Fit AXR to average ADCS in ROI.
        axr_fit_params = np.zeros((n_sl, 4))

        for sl in range(n_sl):
            param, pcov = curve_fit(
                func, tms_res[1:], adc_avs_res[1:, sl], p0=guess_axr
            )
            perr = np.sqrt(np.diag(pcov))
            axr_fit_params[sl, :] = [param[1], param[0], perr[1], perr[0]]
            print("axr, err axr", param[1], perr[1])
            print("sigma, err sigma", param[0], perr[0])
            fit_val[sl, :] = func(tms_res[1:], param[0], param[1]) * adc_avs[0, sl]
            datas.append(
                {
                    "axr_ROI": param[1],
                    "axr_ROI_error": perr[1],
                    "sigma_ROI": param[0],
                    "sigma_ROI_error": perr[0],
                }
            )

        logger.debug("AXR successfully fitted for average ADC in ROI.")
    except BaseException:
        logger.warning(
            f"Could not fit average AXR data: {tms[1:], adc_avs[1:, export_slice]}"
        )
        axr_fit_params[sl, :] = [0, 0, 0, 0]

    import pandas as pd

    df = pd.DataFrame(datas)

    if stats:
        on.print_stats()

    # Important parameters.
    adc_av_roi = np.nanmean(adc, axis=(0, 1))
    adc_std_roi = np.nanstd(adc, axis=(0, 1))
    axr_av_pixelwise = np.nanmean(axr, axis=(0, 1))
    axr_std_pixelwise = np.nanstd(axr, axis=(0, 1))
    sig_av_pixelwise = np.nanmean(sigma, axis=(0, 1))
    sig_std_pixelwise = np.nanstd(sigma, axis=(0, 1))

    n_decimal = 2  # Number of decimal points to which to round saved data.

    data = np.array(
        (
            axr_av_pixelwise[export_slice],
            axr_std_pixelwise[export_slice],
            sig_av_pixelwise[export_slice] * 100,
            sig_std_pixelwise[export_slice] * 100,
            axr_fit_params[export_slice, 0],
            axr_fit_params[export_slice, 2],
            axr_fit_params[export_slice, 1] * 100,
            axr_fit_params[export_slice, 3] * 100,
        )
    )
    index = [
        "AXR av. [s^1]",
        "AXR std. [s^1]",
        "Filter eff. av. [%]",
        "Filter eff. std. [%]",
        "AXR ROI av. [s^1]",
        "AXR ROI std. [s^1]",
        "Filter eff. ROI av. [%]",
        "Filter eff. ROI std. [%]",
    ]

    for tm in range(len(tms)):
        data = np.append(data, adc_av_roi[tm, export_slice] * 10**4)
        data = np.append(data, adc_std_roi[tm, export_slice] * 10**4)
        index = np.append(index, f"ADC$_{tm}$ av. [10^4 mm^2 s^-1]")
        index = np.append(
            index,
            f"ADC$_{tm}$ std. [10^4 mm^2 s^-1]",
        )

    # Return panda series, for use in dataframe.
    """
    import pandas as pd
    data_series = pd.Series(
        data,
        index = index
    )
    """
    print("")
    print("FEXI ANALYSIS")
    underoverline(text="FEXI Parameters (from average ADC in ROI):")
    print(
        f"Average AXR: {axr_fit_params[export_slice, 0]: .3f} + /- {axr_fit_params[export_slice, 2]: .3f}[s ^ (-1)]"
    )
    print(
        f"Average filter eff.: {axr_fit_params[export_slice, 1] * 100: .2f} + /- {axr_fit_params[export_slice, 3] * 100: .2f} [% ]"
    )
    print("")
    underoverline(text="FEXI Parameters (pixelwise):")
    print(
        f"Average AXR: {axr_av_pixelwise[export_slice] * 10 ** 3: .3f} + /- {axr_av_pixelwise[export_slice] * 10 ** 3: .3f}[s ^ (-1)]"
    )
    print(
        f"Median AXR:              {np.nanmedian(axr[:, :, export_slice]) * 10 ** 3:.3f} [s^(-1)]"
    )
    print(
        f"Average filter eff.: {sig_av_pixelwise[export_slice] * 100: .2f} + /- {sig_std_pixelwise[export_slice] * 100: .2f} [ % ]"
    )
    print("")

    if plot:
        # print('axr_fit_params in plot == True ',axr_fit_params)
        print("fit_val (value of the curve fitted) ", fit_val * 10**4)
        plot_fexi_multisl(
            adc_map=adc,
            axr_map=axr,
            fit_val=fit_val,
            sigma_map=sigma,
            anat_map=anat_img,
            adc_avs=adc_avs,
            t_mix=tms,
            title=title,
            slice=export_slice,
            axr_fit_params=axr_fit_params,
            bg=bg,
            save_path=save_path,
        )

    # return panda series with only the ROI method!!!
    return df


def plot_fexi_multisl(
    adc_map,
    axr_map,
    fit_val,
    sigma_map,
    anat_map,
    adc_avs,
    t_mix,
    title,
    slice=0,
    bg=True,
    axr_fit_params=None,
    save_path=None,
):
    """
    Function called by various methods to visualise FEXI analysis. Not to be called by user.

    Parameters
    ----------
    adc_map: np.ndarray
        ADC maps, for filter off and on, stacked. Shape [y,x,N_tm +1]
    axr_map: np.ndarray
        AXR map, shape [y,x]
    sigma_map: np.ndarray
        Filter efficiency map, shape [y,x]
    anat_map: np.ndarray
        Anatomical image for background plots. The lower resolution parametric maps will be interpolated to the same
        resolution as the anatomical images.
    adc_avs: np.ndarray
        Average ADC value in the ROI for ech region of interest.
    t_mix: np.ndarray
        Array of mixing times, for filter off and onm, stacked.
    title: str
        Title for the FEXI plot, to be taken from whatever method calls plot_fexi.
    slice: int
        Slice to display (if applicable).
    bg: bool (True)
        If True, parametric maps will be displayed over the anatomical image in the background.
    axr_fit_params: np.ndarray  (optional)
        Supplied by calling function.
    save_path: str
        Filepath to which to save the main figure.
    fit_val: TO DO

    Returns
    -------
    """
    # Interpolate to anatomical img resolution.
    if bg:
        adc_map_ = np.zeros(
            shape=(
                np.shape(anat_map)[1],
                np.shape(anat_map)[0],
                np.shape(adc_map)[2],
                np.shape(adc_map)[3],
            )
        )
        axr_map_ = np.zeros(
            shape=(np.shape(anat_map)[1], np.shape(anat_map)[0], np.shape(adc_map)[3])
        )
        sigma_map_ = np.zeros(
            shape=(np.shape(anat_map)[1], np.shape(anat_map)[0], np.shape(adc_map)[3])
        )
        interp_type = cv2.INTER_NEAREST
        # Interpolate all parameter maps to resolution of anatomical images,
        # for each slice.
        for z in range(np.shape(adc_map)[-1]):
            adc_map_[:, :, :, z] = cv2.resize(
                adc_map[:, :, :, z],
                dsize=(np.shape(anat_map)[1], np.shape(anat_map)[0]),
                interpolation=interp_type,
            )
            axr_map_[:, :, z] = cv2.resize(
                axr_map[:, :, z],
                dsize=(np.shape(anat_map)[1], np.shape(anat_map)[0]),
                interpolation=interp_type,
            )
            sigma_map_[:, :, z] = cv2.resize(
                sigma_map[:, :, z],
                dsize=(np.shape(anat_map)[1], np.shape(anat_map)[0]),
                interpolation=interp_type,
            )
    else:
        adc_map_ = adc_map
        axr_map_ = axr_map
        sigma_map_ = sigma_map

    # Standard deviations of the ADC ROI.
    adc_std = np.nanstd(adc_map, axis=(0, 1))
    # Create figure.
    fig, ([ax0, ax1], [ax2, ax3]) = plt.subplots(
        nrows=2, ncols=2, figsize=(8, 8), facecolor="white"
    )
    # Figure title.
    fig.suptitle(f"FEXI Analysis: {title}")
    #  Adjust subplot spacing.
    plt.subplots_adjust(wspace=0.5, hspace=0.01)
    """----------- inital plots / cbars ----------- """
    # Plot ForeGround images.
    fg_adc = ax0.imshow(
        adc_map_[:, :, 0, slice] * 10**4, vmin=0, vmax=12, alpha=1, cmap="spring"
    )
    fg_axr = ax2.imshow(
        axr_map_[:, :, slice] * 10**3,
        alpha=1,
        cmap="inferno",  # Show AXR map, in SI unit.
        origin="upper",
        vmin=np.nanmin(axr_map_[:, :, slice] * 10**3),
        vmax=np.nanmax(axr_map_[:, :, slice] * 10**3),
    )
    fg_sig = ax3.imshow(
        sigma_map_[:, :, slice] * 100, alpha=1, origin="upper", vmin=0, vmax=100
    )

    # Create colourbars.
    cbar_adc = colourbar(fg_adc)
    cbar_axr = colourbar(fg_axr)
    cbar_sig = colourbar(fg_sig)
    # Label colourbars.
    cbar_adc.set_label(
        r"x$10^{-4}\ mm^{2}\ s^{-1}$", rotation=270, loc="center", labelpad=15
    )
    cbar_sig.set_label(r"$\%$", rotation=270, loc="center", labelpad=15)
    cbar_axr.set_label(r"$s^{-1}$", rotation=270, loc="center", labelpad=15)

    def update_all_plots(
        time=0, max_adc=12, max_axr=5, max_sig=50, slice=slice, alph=0.8
    ):
        # Format subplots in figure.
        ax0.cla()
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax0.set_title(f"ADC")
        ax1.set_title(f"ROI Averages")  # Set title.
        ax2.set_title(f"AXR")  # Title
        ax3.set_title(f"$\\sigma$")  # Title
        # test = (func(tm_res, axr_fit_params[slice-1,1],axr_fit_params[slice-1,0],slice-1))
        # * 10 ** 4
        # ax3.set_title(f'{test}')  # Title

        if bg:  # Plot background anatomical images.
            bg_adc_update = ax0.imshow(
                anat_map[:, :, slice - 1], cmap="gray", origin="upper"
            )
            bg_axr_update = ax2.imshow(
                anat_map[:, :, slice - 1], cmap="gray", origin="upper"
            )
            bg_sig_update = ax3.imshow(
                anat_map[:, :, slice - 1], cmap="gray", origin="upper"
            )

        # slice = slice + 1
        # Plot updates foreground images.
        fg_adc_update = ax0.imshow(
            adc_map_[:, :, time, slice - 1] * 10**4,
            vmin=4,
            vmax=max_adc,
            alpha=alph,
            cmap="spring",
            zorder=1,
        )
        fg_axr_update = ax2.imshow(
            axr_map_[:, :, slice - 1] * 10**3,
            vmin=0,
            vmax=max_axr,
            alpha=alph,
            cmap="inferno",
            origin="upper",
        )
        fg_sigma_update = ax3.imshow(
            sigma_map_[:, :, slice - 1] * 100,
            vmin=0,
            vmax=max_sig,
            alpha=alph,
            origin="upper",
            zorder=1,
        )
        # Update colour limits.
        fg_axr.set_clim([0, max_axr])
        fg_sig.set_clim([1, max_sig])
        ax0.annotate(
            text=f"$t_{{m}} = {t_mix[time]}$ [ms]",
            xy=[1, np.shape(adc_map_[:, :, time, slice - 1])[1] - 2],
            color="white",
        )
        ax0.set_xlabel(
            rf"$ADC_{{av}}={np.nanmean(adc_map[:, :, time, slice - 1]) * 10 ** 4: .4f}\ [x10 ^ {{-4}}\ mm ^ {{2}}\ s ^ {{-1}}]$"
            # +- {np.nanstd(adc_map[:, :, time]) * 10 ** 4:.4f} [x10^{{-4}}\ mm^{{2}}\ s^{{-1}}]$'
        )
        ax1.set_xlabel("$t_{m}\\ [ms]$")
        ax2.set_xlabel(
            rf"$AXR_{{av}} = {np.nanmean(axr_map[:, :, slice - 1] * 10 ** 3):.4f}\ [s^{{-1}}]$"
        )
        ax3.set_xlabel(
            rf"$\sigma_{{av}} = {np.nanmean(sigma_map[:, :, slice - 1] * 100):.4f}\ [\%]$"
        )
        ax3.set_xlabel(
            rf"$\sigma_{av}  = {(fit_val[slice - 1, 0] * 10 ** 4):.4f}\ [\%]$"
        )

        """------------ Panel 2 -------------"""

        ax1.plot(
            t_mix[1:],
            fit_val[slice - 1, :] * 10**4,
            color="red",
            label="Best fit",
            zorder=0,
            linestyle="dashed",
        )
        # Plot average ADCs' errorbars.
        ax1.errorbar(
            t_mix[1:],
            adc_avs[1:, slice - 1] * 10**4,
            yerr=adc_std[1:, slice - 1] * 10**4,
            color="gray",
            marker=".",
            capsize=5,
            linestyle="",
            zorder=5,
            alpha=0.3,
        )
        # Plot average ADCs on top.
        ax1.scatter(
            t_mix[1:],
            adc_avs[1:, slice - 1] * 10**4,
            color="black",
            marker=".",
            linestyle="",
            zorder=10,
            alpha=1,
        )
        # Plot Filter off average ADC.
        ax1.hlines(
            y=adc_avs[0, slice - 1] * 10**4,
            xmin=0,
            xmax=t_mix[-1],
            color="black",
            linestyles="dashed",
            label="Filter OFF",
        )
        # Display label in legend.
        ax1.legend(loc="lower right")
        asp = (
            np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0]
        )  # Force aspect ratio to conform.
        ax1.set_aspect(asp)
        ax1.set_ylabel("$ADC\\ [x10^{-4}\\ mm^{2}\\ s^{-1}]$")  # y-label.

    """ ----------------------- Sliders ----------------------- """
    # Slider to cycle through mixing times in ADC images.
    tm_slider = widgets.IntSlider(
        value=0, min=0, max=np.shape(adc_map)[2] - 1, description="Mixing time: "
    )
    # Slider to change opacity of parameter maps.
    alpha_slider = widgets.FloatSlider(
        value=1,
        min=0,
        max=1,
        step=0.1,
        description="Opacity: ",
    )
    # Slider for upper limit of AXR colourbar.
    axr_slider = widgets.IntSlider(
        value=8,
        min=1,
        max=25,
        description="AXR: ",
        # max = int(np.nanmax(axr_map * 10 ** 3)),
    )
    # Slider for upper limit of ADC colourbar.
    adc_slider = widgets.IntSlider(
        value=12,
        min=1,
        max=25,
        description="ADC: ",
        # max = int(np.nanmax(axr_map * 10 ** 3)),
    )
    sigma_slider = widgets.IntSlider(
        value=50,
        min=1,
        max=100,
        description="$\\sigma$: ",
    )
    # Slider for upper limit of AXR colourbar.
    slice_slider = widgets.IntSlider(
        value=0,
        min=1,
        max=np.shape(adc_map)[3],
        description="Slice: ",
        # max = int(np.nanmax(axr_map * 10 ** 3)),
    )
    # Put sliders in a HBox.
    row_1 = widgets.HBox([adc_slider, axr_slider, sigma_slider])
    row_2 = widgets.HBox(
        [tm_slider, slice_slider, alpha_slider],
    )
    ui = widgets.VBox(
        [row_1, row_2],
        layout=widgets.Layout(display="flex"),
    )

    # Display our cool ui.
    display(ui)

    out = widgets.interactive_output(
        # update_all_plots(slice = slice),
        update_all_plots,
        {
            "time": tm_slider,
            "max_adc": adc_slider,
            "max_axr": axr_slider,
            "max_sig": sigma_slider,
            "slice": slice_slider,
            "alph": alpha_slider,
        },
    )

    if save_path is not None:
        try:
            plt.savefig(f"{save_path}.svg", format="svg")
            logger.debug(f"Successfully saved to {save_path}")
        except BaseException:
            logger.warning(f"Could not save to {save_path}")
    else:
        logger.debug("Figure not saved.")

    return None


def axr_map_from_adcs_multisl(
    adc_stacked, tms=None, guess=np.array([0.8, 10**-3]), tolerance=3, slice=0
):
    """
    Calculate the AXR map from the ADC maps. Calculate time taken to do so. Read by various methods.

    Parameters
    ----------
    adc_stacked: np-ndarray
        ADC maps, shape [y,x,tm], where the first slice should be the filter off ADC map.
    tms: np.array (None, Optional)
        Array of mixing times, stacked as with adc_stacked.
    guess: np.ndarray  ([sigma, axr] = [0.8, 10 ** -3], Optional)
        Initial guess for the scipy.curve_fit best fit parameters. 1x2 array (sigma, AXR). Good guesses for the AXR
        will vary for different FEXI datasets, and are typically in the range of 10e-4.
    tolerance: float (4, Optional)
        tolerance factor for AXR fit. Datapoints more than tolerance*std away from the ADC mean in the slice will be
        replaced by the slice average.
    slice: int (0, Optional)
        Slice index for which to calculate AXR map.

    Returns
    -------
    axr: np.ndarray
        AXR map of dimensions [x,y].
    axr_err: np.ndarray
        Error in the AXR map, shape [y,x].
    sigma: np.ndarray
        Filter efficiency. See FEXI documentation for further details.
    """

    # Dimensions of ADC filter ON.
    dims = np.shape(adc_stacked)
    # Empty AXR map.
    axr = np.zeros(shape=(dims[0], dims[1], dims[3]))
    # Empty filter efficiency map (y-intercept)
    sigma = np.zeros(shape=(dims[0], dims[1], dims[3]))
    # Empty error map for AXR.
    axr_err = np.zeros(shape=(dims[0], dims[1], dims[3]))
    adc_stacked_resc = np.zeros(dims)
    # print('dims', dims)
    for sl in range(dims[3]):
        for x in range(dims[0]):
            for y in range(dims[1]):
                for tm in range(dims[2]):
                    if np.isnan(adc_stacked[x, y, tm, sl]) == False:
                        adc_stacked_resc[x, y, tm, sl] = (
                            adc_stacked[x, y, tm, sl] / adc_stacked[x, y, 0, sl]
                        )

    adc_stacked = adc_stacked_resc
    adc_stacked = np.where(adc_stacked == 0.0, np.nan, adc_stacked)

    # Use the non-NaN indices to select non-NaN elements
    # Average of each ADC slice.
    adc_av_stacked = np.nanmean(adc_stacked, axis=(0, 1))
    # Std of each ADC slice.
    adc_std_stacked = np.nanstd(adc_stacked, axis=(0, 1))

    def func_axr(x, a, b):  # Function to fit/plot (AXR).
        return 1 - a * np.exp(-x * b)

    # tic = time.perf_counter()                                           #
    # Start timer.
    count = 0  # Count removed datapoints.
    for z in range(dims[3]):
        for j in range(dims[1]):
            for k in range(dims[0]):  # For each pixel.
                # Pixel data. Necessary to redefine, because it will be edited
                # for the fit.
                data = np.array([])
                data_tms = np.array([])
                # print(adc_stacked[k, j, 1:,z])
                if (
                    np.isnan(adc_stacked[k, j, 1:, z]).any() == False
                ):  # Check for unmasked pixels in ROI.
                    # print('adc stacked ',adc_stacked[k, j, 1:,z])
                    for t in range(len(tms[1:])):
                        data = np.append(
                            data,
                            tolerate(
                                adc_stacked[k, j, 1:, z][t],
                                adc_av_stacked[1:, z][t],
                                adc_std_stacked[1:, z][t],
                                n=tolerance,
                            ),
                        )
                        data_tms = np.append(data_tms, (tms[1:][t]) / 1000.0)
                    try:
                        popt, pcov = curve_fit(func_axr, data_tms, data, guess)
                        axr[k, j, z] = popt[1]
                        sigma[k, j, z] = popt[0]

                        axr_err[k, j, z] = np.sqrt(np.diag(pcov))[1]
                    except BaseException:
                        logger.warning(f"AXR fit failed for pixel: {j, k, z}")
                        axr[k, j, z] = np.nan
                        sigma[k, j, z] = np.nan
                        axr_err[k, j, z] = np.nan
                else:  # Masked pixels
                    axr[k, j, z] = np.nan
                    sigma[k, j, z] = np.nan
                    axr_err[k, j, z] = np.nan
    axr = axr / 1000.0
    axr_err = axr_err / 1000.0
    sigma = sigma / 1000.0
    # Set negative AXR values to zero, assuming the fit error is small.
    non_nan_indices = np.isnan(axr) == False
    try:
        axr[axr < 0] = 0
        axr_err[axr_err < 0] = 0
        sigma[sigma < 0] = 0

    except BaseException:
        logger.warning(f"Failed to suppress pixels with negative AXR values.")

    # Use the non-NaN indices to select non-NaN elements
    # print('axr after zero setting ', axr[non_nan_indices] )
    # Rather than thresholding, use fitting errors to mask pixels.
    try:
        # Identify pixels with high errors.
        print("ricorda non ottimale...")
        x_2, y_2, z_2 = np.where(axr_err > 5)

        # Fits have clearly failed, so set to NaN.
        axr[x_2, y_2, z_2] = np.nan
        axr_err[x_2, y_2, z_2] = np.nan
        sigma[x_2, y_2, z_2] = np.nan

        logger.debug(f"{len(x_2)} pixels thresholded due to silly AXR fits.")
    except BaseException:
        logger.warning(f"Failed to supress pixels with failed fitting.")
    pp = np.isnan(axr) == False

    return axr, axr_err, sigma


def exclude_outliers_jacopo(x, y, bvalues, factor=1):
    """
    Takes data for ADC fit and excludes values more than n*factor away from mean of repetitions.

    Parameters
    ----------
    x: np.ndarray
        x-data to fit.
    y: np.ndarray
        y-data to fit.
    bvalues: np.ndarray
        Array of b-values used in the filter ON acquisition.
    factor: float (1)
        Datapoints more than factor*std away from the median signal values will be excluded from the ADC fit.

    Returns
    -------
    x_new: np.ndarray
        New x-coordinate array with outliers removed.
    y_new: np.ndarray
        New y-coordinate array with outliers removed.
    """
    x_new = np.array([])
    y_new = np.array([])
    x_mean = np.array([])
    y_mean = np.array([])
    x_sigma = np.array([])
    y_sigma = np.array([])
    for b in bvalues:
        ydata = y[x == b]  # Separate data by b-values.
        xdata = x[x == b]
        # if (np.isnan(ydata)==False):
        # print('excout bef ydata',ydata)
        median = np.median(ydata)  # Median.
        # mean = np.mean(ydata)  # Mean.
        # median = mean
        sigma = np.std(ydata)  # Std dev.
        low = median - factor * sigma  # Lower data limit.
        high = median + factor * sigma  # Upper data limit.
        """
        print('b:              ', b)
        print('median:         ', median,'std:', sigma)
        print('Deleting below: ', low)
        print('Deleting above: ', high)
        print('Values too low:  ', ydata[too_low])  # Check visually that the numbers make sense.
        print('Values too high: ', ydata[too_high])
        """
        too_low = np.where(ydata < low)[0]  # Indicies of points too low.
        too_high = np.where(ydata > high)[0]  # Indicies of points too high.
        to_delete = np.append(
            too_low, too_high
        )  # Concatonate arrays of indicies to delete.
        ydata = np.delete(ydata, to_delete)  # Delete outlaying y-data.
        xdata = np.delete(xdata, to_delete)  # Delete corresponding x-data.
        x_new = np.append(x_new, xdata)
        y_new = np.append(y_new, ydata)

        x_mean = np.append(x_mean, np.mean(xdata))
        y_mean = np.append(y_mean, np.mean(ydata))
        x_sigma = np.append(x_sigma, np.std(xdata))
        y_sigma = np.append(y_sigma, np.std(ydata))

    return x_new, y_new

    # this function take segm_list and trasform it in a np.array (x,y,slice) of the masks
    # NB works only for i ROI region TO DO: implement


def from_roilist_to_mask_multisl(dict_mask):
    sorted_keys = sorted(dict_mask.keys(), key=int)
    matrices = [dict_mask[key] for key in sorted_keys]
    array_3d = np.squeeze(np.stack(matrices, axis=2))

    return array_3d


def from_mask_file_to_roi_list(filepath, nslice, anat_dim1, anat_dim2):
    print("nslice", nslice)
    try:
        """
        mask = np.genfromtxt(
            filepath,
            delimiter=','
        )
        """
        mask = np.loadtxt(filepath, delimiter=",")
        # PUT NOT HARD CODED JACOPO
        # reshaped_mask = mask.reshape((140, 140, nslice)) # for mice
        # reshaped_mask = mask.reshape((256, 256, nslice)) # for yeast
        reshaped_mask = mask.reshape((anat_dim1, anat_dim2, nslice))
        print("fatto reshape")
        logger.debug(f"Reading mask from filepath {filepath}.")
    except BaseException:
        print("COULD NOT LOAD FROM FILE PATH {filepath}")
        logger.debug(
            f"Could not load mask from filepath {filepath}. \nNo pixels will be masked."
        )
        # reshaped_mask = np.ones(
        #    shape=(140,140,nslice)
        # )
        reshaped_mask = np.ones(shape=(anat_dim1, anat_dim2, nslice))

    dict = {"1": reshaped_mask}
    return dict


def lookup_sequence_class_name(obj):
    """
    Finds name of sequence class for given scan obj using the config file.

    Parameters
    ----------
    obj: np.ndarray
        Bruker image object for which to determine the sequence class name.

    Returns
    -------
    Name of object sequence class.

    """
    try:
        # In a dict of lists, finds the key corresponding to the list
        # containing x. Returns the first key found.
        return next(k for k, v in load_json().items() if obj.type == v or obj.type in v)
    except StopIteration:
        return "MRIExperiment"


def load_json():
    """
    Loads SmartLoadMeasurements.json, containing dictionary of sequence class names. Called by lookup_sequence_class_name().

    Parameters
    ----------

    Returns
    -------
    Content of SmartLoadMeasurements.json.

    """
    config_path = Path(__file__).parents[1] / "config/SmartLoadMeasurements.json"

    with open(config_path) as file:
        text = json.load(file)
        return text


def get_days(df):
    import re

    # Extract days of study.
    days = []
    for item in df.columns:
        days = np.append(days, int(re.findall(r"\d+", item)[0]))

    days = days.astype(int)
    return days


def put_all_mice_together(
    df_list=None,
    title="No Title Specified",
    sub_titles=["ADC av. in the ROI.", "AXR from fitting ADC av."],
    mice=None,
    colours=None,
    ylabels=["ADC [$mm^{2}\\ s^{-1}}$]", "AXR [$s^{-1}$]", "AXR [$s^{-1}$]"],
    alpha_mice=0.5,
    alpha_avs=0.5,
    save=None,
    xlim=[-0.5, 13.5],
    ylim=[0, 13],
):
    if df_list is None:
        print("ERROR: No data supplied! Aborting.")
    elif mice is None:
        print("ERROR: Please supply animal names! Aborting.")
    else:
        print(f"Dataframes: {len(df_list)}")
        print(f"Names:      {np.shape(mice)}")
        print(f"Colours:    {np.shape(colours)}")

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        fig.suptitle(title)
        fig.supxlabel("Days post-Radiotherapy")

        adc_vals = np.array([])
        axr_vals = np.array([])

        # Loop through dataframes in the list. (iterating over mice)
        for i in range(len(df_list)):
            # Extract days from dfs.
            days_int = get_days(df_list[i])
            axes[0].plot(
                days_int, df_list[i].iloc[8], color=colours[i], alpha=alpha_mice
            )  # ADCs.
            axes[1].plot(
                days_int, df_list[i].iloc[4], color=colours[i], alpha=alpha_mice
            )  # AXRs from fitting average ADCs in the ROIs.

            axes[0].scatter(
                days_int,
                df_list[i].iloc[8],
                label=mice[i],
                color=colours[i],
                alpha=alpha_mice,
            )  # ADCs scatter.
            axes[1].scatter(
                days_int,
                df_list[i].iloc[4],
                label=mice[i],
                color=colours[i],
                alpha=alpha_mice,
            )  # AXRs scatter.

            # Save values for averaging.
            adc_vals = np.append(adc_vals, df_list[i].iloc[8])
            axr_vals = np.append(axr_vals, df_list[i].iloc[4])

        # Calculate averages and stds.
        avs_adc = np.nanmean(adc_vals)
        avs_axr = np.nanmean(axr_vals)
        np.nanstd(adc_vals)
        np.nanstd(axr_vals)
        print(avs_adc)

        # Plot horizontal lines for weekly averages.
        axes[0].axhline(
            y=np.nanmean(avs_adc), linestyle="dashed", color="black", alpha=alpha_avs
        )
        axes[1].axhline(
            y=np.nanmean(avs_axr), linestyle="dashed", color="black", alpha=alpha_avs
        )

        from matplotlib.ticker import MaxNLocator

        # Loop over axes, format.
        for k in range(2):
            # axes[k].set_xticks(xi, x)
            # axes[k].set_xlim(-0.5, 4.5)
            axes[k].set_ylabel(ylabels[k])
            axes[k].set_title(sub_titles[k])
            axes[k].set_box_aspect(1)
            axes[k].set_ylim(ylim[0], ylim[1])
            axes[k].set_xlim(xlim[0], xlim[1])

            axes[k].xaxis.set_major_locator(MaxNLocator(integer=True))

        axes[0].legend()
        plt.show()

        if save:
            plt.savefig(save, format="svg")
    try:
        return adc_vals, axr_vals
    except BaseException:
        return None


def inspect_treatment(
    orient="axial",
    dicom_path="",
    tumour_path="",
    tumour_ir_path="",
):
    """
    Rudimentary function for inspecting radiotherapy treatment planning files. This function was written to process the
    data files from the SARRP PC in the TranslaTum. Data (re-)orientation procedure may vary for different data sets.

    Although the function requests file paths to tumour volumes, in theory any volumes can be plotted over the
    anatomical images providing dimensions conform.

    Parameters
    ----------
    orient: str
        Image orientation to plot. Supported orientations ['axial','sagittal','coronal'].
    dicom_path: str
        Path to folder containing anatomical DICOM images (i.e. CT, MRI...).
    tumour_path:
        Path to tumour volume file (tested for .nrrd file types).
    tumour_ir_path: str
        Path to irradiated tumour volume file (tested for .nrrd file types).

    Returns
    -------

    """
    try:
        plots = []
        for f in glob.glob(dicom_path):
            ds = pydicom.dcmread(f)
            pix = ds.pixel_array  # Extract image file.
            plots.append(pix)  # Add to list of slices.
            # Convert list of plots to numpy array.
            plots_arr = np.array(plots)
            plots_arr = plots / np.nanmax(plots_arr)  # Normalise images.
        logger.debug("Successfully loaded DICOM image files.")
    except BaseException:
        logger.critical(f"Unable to load DICOM files. Aborting...")
        raise SystemExit(0)

    try:
        tumour, header_tumour = nrrd.read(tumour_path)  # Load tumour volume.
        tumour_ir, header_tumour_ir = nrrd.read(
            tumour_ir_path
        )  # Load irradiated tumour volume.
        logger.debug("Successfully loaded tumour volume files.")
    except BaseException:
        logger.critical(f"Unable to load tumour volume files. Aborting...")
        raise SystemExit(0)

    tumour = tumour.T  # Transpose tumour volume files.
    tumour_ir = tumour_ir.T

    if orient == "axial":
        n_slices = np.shape(plots_arr)[2]
    elif orient == "sagittal":
        n_slices = np.shape(plots_arr)[1]
    elif orient == "coronal":
        n_slices = np.shape(plots_arr)[0]
    else:
        logger.critical(f"Orientation {orient} not supported. Aborting...")
        raise SystemExit(0)

    slice_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=n_slices,
        description="Slice: ",
    )
    ui = widgets.HBox(
        [slice_slider],
        layout=widgets.Layout(display="flex"),
    )
    display(ui)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9), layout="constrained")
    fig.suptitle(f"Treatment Planning ({orient} plane)")
    import matplotlib.colors as colours

    cmap_tumour = colours.ListedColormap(["none", "blue"])
    cmap_tumour_ir = colours.ListedColormap(["none", "red"])

    def update_ax(
        i=1,
    ):
        ax.cla()
        ax.imshow(plots_arr[:, i - 1, :], cmap="gray", vmin=0, vmax=1)
        ax.imshow(
            tumour[:, i - 1, :],
            cmap=cmap_tumour,
            alpha=0.8,
        )
        ax.imshow(tumour_ir[:, i - 1, :], cmap=cmap_tumour_ir, alpha=0.8)
        plt.show()

    def update_cor(
        j=1,
    ):
        ax.cla()
        ax.imshow(np.flipud(plots_arr[j - 1, :, :]), cmap="gray", vmin=0, vmax=1)
        ax.imshow(np.flipud(tumour[j - 1, :, :]), cmap=cmap_tumour, alpha=0.8)
        ax.imshow(np.flipud(tumour_ir[j - 1, :, :]), cmap=cmap_tumour_ir, alpha=0.8)
        plt.show()

    def update_sag(
        k=1,
    ):
        ax.cla()
        ax.imshow((plots_arr[:, :, k - 1]), cmap="gray", vmin=0, vmax=1)
        ax.imshow((tumour[:, :, k - 1]), cmap=cmap_tumour, alpha=0.8)
        ax.imshow((tumour_ir[:, :, k - 1]), cmap=cmap_tumour_ir, alpha=0.8)
        plt.show()

    # Output.
    if orient == "axial":
        out = widgets.interactive_output(
            update_ax,
            {
                "i": slice_slider,
            },
        )
        ax.imshow(plots_arr[:, 0, :], cmap="gray", vmin=0, vmax=1)
    elif orient == "coronal":
        out = widgets.interactive_output(
            update_cor,
            {
                "j": slice_slider,
            },
        )
        ax.imshow(np.flipud(plots_arr[0, :, :]), cmap="gray", vmin=0, vmax=1)
    elif orient == "sagittal":
        out = widgets.interactive_output(
            update_sag,
            {
                "k": slice_slider,
            },
        )
        ax.imshow((plots_arr[:, :, 0]), cmap="gray", vmin=0, vmax=1)

    plt.show()
    return None
