"""Spectroscopy fitting helpers built on top of Astropy models."""

import time

import numpy as np
from astropy.modeling import fitting, models
from astropy.modeling.models import custom_model
from joblib import Parallel, cpu_count, delayed
from tqdm.auto import tqdm

from ...utils.physics import get_gmr
from ...utils.system.logging import init_default_logger
from ..spectroscopy.core import freq_to_index, get_freq_axis, get_metab_cs_ppm

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


# -->astropy-fitting-->astropy-fitting-->astropy-fitting-->astropy-fitting-->astropy-fitting-->astropy-fitting-->
def fit_single_point_apy(
    data_obj=None,
    data_point=None,
    f_init=None,
    freq_range=None,
    verblevel=0,
    fit_params=None,
):
    """
    Fit a model (in f_init) to the data data_point (1D) along freq_range.

    Parameters
    ----------
    data_obj: hypermri-sequence object

    data_point : ndarray
        1D spectral data point.

    f_init : object
        Initial model for fitting.

    freq_range : ndarray
        Frequency values corresponding to data_point.

    verblevel : int, optional
        Verbose level of fit, 0 = off.

    fit_params : dict, optional
        Additional fit parameters.

    Returns
    -------
    fit_model_result : object
        Fitted model for the provided data point.

    """

    if data_point is None:
        Warning("data_point is None!")
        return None

    if fit_params is None:
        fit_params = generate_default_fit_params_apy(
            data_obj=data_obj, input_data=data_point
        )

    # FIXME move these (the ones that have to be set to avoid crashing) to a
    # standard value defining function that is always called:
    fit_method = fit_params.get("fit_method", "SLSQPLSQFitter")

    # define frequency range:
    # print(freq_range)
    if freq_range is not None:
        pass
    else:
        freq_range = fit_params.get(
            "freq_range",
            get_freq_axis(
                data_obj=data_obj,
                unit=fit_params["freq_unit"],
                npoints=data_point.shape[0],
            ),
        )

    # data_point = data_point[fit_params["fit_range"][0]:fit_params["fit_range"][1]]
    # freq_range =
    # freq_range[fit_params["fit_range"][0]:fit_params["fit_range"][1]]

    # normalize data:
    if fit_params["fit_area_opt"]:
        data_point = data_point / fit_params["fit_area_opt_factor"]

    # match the initial fit parameters to the data (input_data)
    f_init = match_fit_init_apy(
        data_obj=data_obj,
        fit_init=f_init,
        input_data=data_point,
        fit_params=fit_params,
    )

    # choose fitting method:
    if fit_method == "LevMarLSQFitter":
        fit_sol = fitting.LevMarLSQFitter()
        fit_model_result = fit_sol(
            f_init,
            freq_range,
            data_point,
        )
    elif fit_method == "SLSQPLSQFitter":
        fit_sol = fitting.SLSQPLSQFitter()
        fit_model_result = fit_sol(
            f_init,
            freq_range,
            data_point,
            verblevel=verblevel,
            iter=fit_params["fit_maxiter"],
            epsilon=fit_params["fit_eps"],
        )
    elif fit_method == "LinearLSQFitter":
        fit_sol = fitting.LinearLSQFitter()
        fit_model_result = fit_sol(
            f_init,
            freq_range,
            data_point,
        )
    else:
        # set fitting method:
        fit_params["fit_method"] = "SLSQPLSQFitter"
        fit_sol = fitting.SLSQPLSQFitter()
        fit_model_result = fit_sol(
            f_init,
            freq_range,
            data_point,
            verblevel=verblevel,
            iter=fit_params["fit_maxiter"],
            epsilon=fit_params["fit_eps"],
        )

    # save fit parameters:
    fit_params["fit_info"] = fit_sol.fit_info

    return fit_model_result


def fit_spectra_apy(
    data_obj=None,
    input_data=None,
    number_of_cpu_cores=None,
    use_multiprocessing=True,
    fit_params=None,
):
    """
    Fit spectra onto a CSI dataset using the astropy modeling and fitting library.

    Parameters
    ----------
    input_data : array-like, optional
        CSI data in format [freq - z - x - y - reps - channels].
        - if None or "seq2d", self.seq2d is used
        - if "csi_image", self.csi_image is used.

    number_of_cpu_cores : int, optional
        Number of CPU cores to be used for multiprocessing.
        If None (Default), the number of cores is selected automatically.

    use_multiprocessing : bool, default=True
        If True, use multiprocessing using multiple CPU cores.

    fit_params : dict, optional
        Dictionary containing fit parameters.
        - e.g. bounds: `{"amplitude_1": {"bounds": (0.0, 0.1)} ,"x_0_1": {"bounds": (178, 180)}}`

    Returns
    -------
    fit_results : dict
        Dictionary containing the fitted model results and other relevant parameters.
        - `data`: Model results.
        - `metabs`: List of metabolites.
        - `nrange`: Range of repetitions.
        - `freq_range`: Frequency range.
        - `lb`: Linebroadening.

    Examples
    --------
    Define Gaussian linebroadening:
    >>> linebroadening_Hz = 0

    # Generate linebroadened CSI dataset
    >>> spec_data_00Hz = luca_csi.multi_dim_linebroadening(lb=linebroadening_Hz)

    Specify metabolites for fitting:
    >>> metabs = ["Pyruvate","Lactate", "Alanine", "PyruvateHydrate"]

    Initialize fit parameters and bounds:
    >>> fit_params = {
    >>>    "lb": linebroadening_Hz,
    >>>    "fwhm_1.value":0.5, "fwhm_2.value":0.5,
    >>>    "fwhm_3.value":0.5, "fwhm_4.value":0.5,
    >>>    "fwhm_1.bounds":(0.2, 2), "fwhm_2.bounds":(0.2, 2),
    >>>    "fwhm_3.bounds":(0.2, 2), "fwhm_4.bounds":(0.2, 2)
    >>> }

    Perform fitting
    >>> fit_model_results_lb00Hz = luca_csi.fit_spectra_astropy2(
    >>>    metabs=metabs,
    >>>    fit_reps=[0,0],
    >>>    fit_params=fit_params,
    >>>    use_multiprocessing=True
    )
    ```

    """
    # import relevat functions from fitting utility:

    if input_data is None:
        # ParaVision reco data:
        try:
            spec_data = data_obj.spec
        except BaseException:
            try:
                spec_data = data_obj.seq2d_reordered
            except BaseException:
                raise ValueError("No input_data provided")
    else:
        spec_data = input_data

    spec_data = np.abs(spec_data)  # / np.max(np.abs(spec_data))

    if fit_params is None:
        fit_params = {}

    # generate + fill fit_params
    fit_params = generate_default_fit_params_apy(
        data_obj=data_obj, input_data=spec_data, fit_params=fit_params
    )

    freq_range = get_freq_axis(
        data_obj=data_obj, unit="ppm", npoints=input_data.shape[0]
    )

    # prepare data (linebroadening):
    # ------------------------------------------------------------------------------
    lb = fit_params.get("lb", 0)

    # metabolites:
    metabs = fit_params.get("metabs")

    # number of peaks:
    np.size(metabs)

    # set output level (0=silent)
    verblevel = fit_params.get("verblevel", 0)

    # repetitions:
    fit_reps = fit_params.get("fit_reps", [0, -1])

    # all repetitions:
    if fit_reps == [0, -1]:
        # for init size of empty array:
        nr = spec_data.shape[4]

        # range over which to iterate
        nrange = range(nr)

    # only one repetition
    elif fit_reps[0] == fit_reps[1]:
        nrange = range(fit_reps[0], fit_reps[1] + 1)
        nr = 1

    else:
        nr = fit_reps[1] - fit_reps[0] + 1
        nrange = range(fit_reps[0], fit_reps[1] + 1)

    # fitted_model = np.apply_along_axis(wrapper_fit_spec, axis=0,
    # arr=spec_data)
    fit_model_results = np.empty(
        (
            1,
            spec_data.shape[1],
            spec_data.shape[2],
            spec_data.shape[3],
            spec_data.shape[4],
            spec_data.shape[5],
        ),
        dtype=object,
    )
    # Parallel computing to speed up fitting for multidimensional data
    if use_multiprocessing:
        try:
            # try to set the number of usable cpu cores to the amount of
            # available cores
            if number_of_cpu_cores is None:
                # int-divide by 2 because hyperthreading cores don't
                # count/help:
                number_of_cpu_cores = cpu_count() // 2

            else:
                pass
            logger.debug("Using %d cpu cores" % number_of_cpu_cores)
        except BaseException:
            use_multiprocessing = False
            number_of_cpu_cores = 1
            logger.warning("Using %d cpu cores" % number_of_cpu_cores)

    # to measure duration of fitting:
    time.time()

    if use_multiprocessing is True:
        # reshape data:
        reshaped_data = spec_data[:, :, :, :, nrange, :].reshape(spec_data.shape[0], -1)

        # Fixme tqdm has trouble with CPU bound parallel operations...
        index_list = tqdm(
            range(reshaped_data.shape[1]),
            desc="Fitting progress on %d cores" % number_of_cpu_cores,
            leave=True,
        )

        f_init = generate_fit_init_apy(
            data_obj=data_obj,
            input_data=reshaped_data,
            fit_params=fit_params,
        )

        # Prepare the arguments for each function call
        args = [
            (
                data_obj,
                np.squeeze(reshaped_data[:, i]),
                f_init,
                freq_range,
                verblevel,
                fit_params,
            )
            for i in index_list
        ]

        results = Parallel(n_jobs=number_of_cpu_cores)(
            delayed(fit_single_point_apy)(*arg) for arg in args
        )

        # reshape fit results (still a sol model) to original data shape
        fit_model_results[:, :, :, :, nrange, :] = np.reshape(
            results,
            [
                1,
                spec_data.shape[1],
                spec_data.shape[2],
                spec_data.shape[3],
                nr,
                spec_data.shape[5],
            ],
        )

    else:
        # Initialize the tqdm object for the outermost loop
        pbar = tqdm(
            range(spec_data.shape[1] * spec_data.shape[2] * spec_data.shape[3] * nr),
            desc="Progress: ",
            leave=True,
        )

        f_init = generate_fit_init_apy(
            data_obj=data_obj,
            input_data=spec_data,
            fit_params=fit_params,
        )

        for r in nrange:
            for x in range(spec_data.shape[2]):
                for y in range(spec_data.shape[3]):
                    # Check if there are any non-finite values in the data
                    if np.any(~np.isfinite(np.squeeze(spec_data[:, 0, x, y, r, 0]))):
                        logger.error("There are non-finite values in the data.")
                    else:
                        pass

                    fit_model_result = fit_single_point_apy(
                        data_obj=data_obj,
                        data_point=np.squeeze(spec_data[:, 0, x, y, r, 0]),
                        f_init=f_init,
                        freq_range=freq_range,
                        verblevel=verblevel,
                        fit_params=fit_params,
                    )
                    fit_model_results[0, 0, x, y, r, 0] = fit_model_result
                    pbar.update()

        pbar.close()
    if fit_params["show_tqdm"]:
        # print(fit_params)
        # print(f"Fitting took: {np.round(time.time() - st, 2)}s")
        pass
    else:
        pass
    fit_results = {}
    fit_results["data"] = fit_model_results
    fit_results["metabs"] = metabs
    fit_results["nrange"] = nrange
    fit_results["freq_range"] = freq_range
    fit_results["lb"] = lb
    return fit_results


def extract_fit_results_apy(
    data_obj=None, fit_params=None, fit_results=None, metabs=None
):
    """
    Returns the fitted spectra and maps of amplitude, frequency, fwhm, background, AUC and AUC without background for
    the metabolites metabs when the fit_results (output from fit_spectra_astropy2) are passed in

    Parameters
    ----------
    metabs: string
        list of metabolites of which spectra and maps should be generated
    fit_results: ND array
        the fitted model output (output from fit_spectra_astropy2)
    freq_range: 1D array
        array of spectral frequencies that was fitted against (default get_ppm())

    Returns
    -------
    A ND Array of fit spectra (same dimensions as seq2d_reordered, Maps of fit-parameters of the different metabolites

    Examples
    --------
    .. code-block:: python

        # define Gaussian linebroadening:
        linebroadening_Hz = 0

        # Generate linebroadened CSI dataset (for later comparison)
        csi_data_00Hz = luca_csi.multi_dim_linebroadening(lb=linebroadening_Hz)

        # which metabolites should be fit:
        metabs = ["Pyruvate","Lactate", "Alanine", "PyruvateHydrate"]

        # init fit parameters + bounds:
        # init fit parameters + bounds:
        fit_params = {"lb": linebroadening_Hz,
                      "fwhm_1.value":0.5,
                      "fwhm_2.value":0.5,
                      "fwhm_3.value":0.5,
                      "fwhm_4.value":0.5,
                      "fwhm_1.bounds":(0.2, 2),
                      "fwhm_2.bounds":(0.2, 2),
                      "fwhm_3.bounds":(0.2, 2),
                      "fwhm_4.bounds":(0.2, 2)}


        # perform fitting
        fit_model_results_lb00Hz = luca_csi.fit_spectra_astropy2(metabs=metabs,
                                                                 fit_reps=[0,0],
                                                                 fit_params=fit_params)
                                                             use_multiprocessing=True)


    # Extract spectra and fit results (maps of amplitude, frequency,
    # fullwidth-halfmax ...
    fit_spectra_00Hz, fit_res_00Hz = luca_csi.extract_fit_results(metabs=metabs,
                                                                  fit_results=fit_model_results_lb00Hz)

    """
    if fit_results is None:
        return None

    if metabs is None:
        metabs = fit_params["metabs"]
    else:
        if isinstance(metabs, "list"):
            pass
        else:
            metabs = list((metabs,))

    freq_range = fit_params.get(
        "freq_range", get_freq_axis(data_obj=data_obj, unit="ppm")
    )

    # exctract data:
    fit_model_results = fit_results["data"]

    # init empty dict to save the fit results per metabolite:
    fit_maps = {}

    # init empty array to fill with fitted spectra:
    fit_spectra = np.zeros(
        (
            freq_range.shape[0],
            fit_model_results.shape[1],
            fit_model_results.shape[2],
            fit_model_results.shape[3],
            fit_model_results.shape[4],
            fit_model_results.shape[5],
        )
    )

    from copy import deepcopy

    # Initialize the dictionary
    for k, m in enumerate(metabs):
        if m == "":
            m = str(k)
        fit_maps[m] = {
            "amplitude": np.zeros(fit_model_results.shape),
            "x_0": np.zeros(fit_model_results.shape),
            "fwhm": np.zeros(fit_model_results.shape),
            "nfloor": np.zeros(fit_model_results.shape),
            "AUC_background": np.zeros(fit_model_results.shape),
            "AUC_no_background": np.zeros(fit_model_results.shape),
        }

        if fit_params["lineshape"] == "Voigt":
            # Gaussian linewidth:
            fit_maps[m]["fwhm_G"] = np.zeros(fit_model_results.shape)

    # Loop through fit_model_results to populate the dictionary
    # Assuming fit_model_results is a numpy array, change the loop accordingly
    # if it's a dictionary
    for c in range(fit_model_results.shape[5]):
        for z in range(fit_model_results.shape[1]):
            for r in range(fit_model_results.shape[4]):
                for x in range(fit_model_results.shape[2]):
                    for y in range(fit_model_results.shape[3]):
                        # You may need to adjust the indices based on your
                        # actual data shape
                        try:
                            fit_model = fit_model_results[0, z, x, y, r, c].copy()
                        except BaseException:
                            fit_model = None

                        # generate fit maps:
                        for k, m in enumerate(metabs):
                            if m == "":
                                m = str(k)
                            if fit_model is not None:
                                try:
                                    if (fit_params["lineshape"] == "Lorentzian") or (
                                        fit_params["lineshape"] == "sqrtLorentzian"
                                    ):
                                        fit_maps[m]["amplitude"][0, z, x, y, r, c] = (
                                            getattr(
                                                fit_model, f"amplitude_{k + 1}"
                                            ).value
                                        )
                                        fit_maps[m]["x_0"][0, z, x, y, r, c] = getattr(
                                            fit_model, f"x_0_{k + 1}"
                                        ).value
                                        fit_maps[m]["fwhm"][0, z, x, y, r, c] = getattr(
                                            fit_model, f"fwhm_{k + 1}"
                                        ).value
                                        fit_maps[m]["nfloor"][0, z, x, y, r, c] = (
                                            getattr(fit_model, "nfloor").value
                                        )

                                        f_temp = deepcopy(fit_model)
                                        for kk in range(np.size(metabs)):
                                            # Assuming the attributes are
                                            # 1-indexed
                                            if k == kk:
                                                continue  # Skip the one you want to keep
                                            setattr(f_temp, f"amplitude_{kk + 1}", 0)

                                        fit_maps[m]["AUC_background"][
                                            0, z, x, y, r, c
                                        ] = np.sum(f_temp(freq_range))
                                        fit_maps[m]["AUC_no_background"][
                                            0, z, x, y, r, c
                                        ] = np.sum(
                                            f_temp(freq_range)
                                            - np.squeeze(
                                                fit_maps[m]["nfloor"][0, z, x, y, r, c]
                                            )
                                        )
                                    elif fit_params["lineshape"] == "Voigt":
                                        fit_maps[m]["amplitude"][0, z, x, y, r, c] = (
                                            getattr(
                                                fit_model, f"amplitude_{k + 1}"
                                            ).value
                                        )
                                        fit_maps[m]["x_0"][0, z, x, y, r, c] = getattr(
                                            fit_model, f"x_0_{k + 1}"
                                        ).value
                                        fit_maps[m]["fwhm"][0, z, x, y, r, c] = getattr(
                                            fit_model, f"fwhm_{k + 1}"
                                        ).value
                                        fit_maps[m]["fwhm_G"][0, z, x, y, r, c] = (
                                            getattr(fit_model, f"fwhm_G_{k + 1}").value
                                        )
                                        fit_maps[m]["nfloor"][0, z, x, y, r, c] = (
                                            getattr(fit_model, "nfloor").value
                                        )

                                        f_temp = deepcopy(fit_model)
                                        for kk in range(np.size(metabs)):
                                            # Assuming the attributes are
                                            # 1-indexed
                                            if k == kk:
                                                continue  # Skip the one you want to keep
                                            setattr(f_temp, f"amplitude_{kk + 1}", 0)

                                        fit_maps[m]["AUC_background"][
                                            0, z, x, y, r, c
                                        ] = np.sum(f_temp(freq_range))
                                        fit_maps[m]["AUC_no_background"][
                                            0, z, x, y, r, c
                                        ] = np.sum(
                                            f_temp(freq_range)
                                            - np.squeeze(
                                                fit_maps[m]["nfloor"][0, z, x, y, r, c]
                                            )
                                        )

                                except AttributeError as e:
                                    print(
                                        f"Attribute not found for metabolite {m} at indices {z}, {x}, {y}, {r}, {c}: {e}"
                                    )
                                    fit_maps[m]["amplitude"][0, z, x, y, r, c] = 0.0
                                    fit_maps[m]["x_0"][0, z, x, y, r, c] = 0.0
                                    fit_maps[m]["fwhm"][0, z, x, y, r, c] = 0.0
                                    fit_maps[m]["nfloor"][0, z, x, y, r, c] = 0.0
                                    fit_maps[m]["AUC_background"][
                                        0, z, x, y, r, c
                                    ] = 0.0
                                    fit_maps[m]["AUC_no_background"][
                                        0, z, x, y, r, c
                                    ] = 0.0
                            else:
                                # print(
                                #     f"fit_model is None at indices {z}, {x}, {y}, {r}, {c}"
                                # )
                                fit_maps[m]["amplitude"][0, z, x, y, r, c] = 0.0
                                fit_maps[m]["x_0"][0, z, x, y, r, c] = 0.0
                                fit_maps[m]["fwhm"][0, z, x, y, r, c] = 0.0
                                fit_maps[m]["nfloor"][0, z, x, y, r, c] = 0.0
                                fit_maps[m]["AUC_background"][0, z, x, y, r, c] = 0.0
                                fit_maps[m]["AUC_no_background"][0, z, x, y, r, c] = 0.0

                        # generate fit_spectra:
                        for k, m in enumerate(fit_params["metabs"]):
                            # search independent of lower/uppercase:
                            if m.lower() in [x.lower() for x in metabs]:
                                pass
                            else:
                                # set the amplitude of the non-wanted peaks to
                                # 0
                                name = f"amplitude_{k + 1}"
                                getattr(
                                    fit_model,
                                    name,
                                ).value = 0

                        if fit_model is None:
                            fit_spectra[:, z, x, y, r, c] = np.zeros_like(freq_range)
                        else:
                            fit_spectra[:, z, x, y, r, c] = fit_model(freq_range)

    return fit_spectra, fit_maps


def generate_fit_init_apy(
    data_obj=None,
    input_data=None,
    fit_params=None,
):
    """
    Generate initial fit parameters for spectral data based on provided metabolites and constraints.

    This function dynamically sets up Lorentzian functions based on the number of metabolites (peaks)
    specified in the fit_params. It initializes the fitting parameters and sets bounds for each parameter
    based on provided constraints or defaults.

    Parameters
    ----------
    input_data : ndarray, optional
        Spectral data used to derive some of the initial parameters, especially for setting amplitude.
        If not provided, it defaults to class-based CSI data.

    fit_params : dict, optional
        Dictionary containing fit parameters and constraints. It can specify metabolites, linebroadening,
        initial values, bounds, and the expected lineshape. If certain parameters are not provided,
        the function uses default values.

        Example structure::

            {
                "metabs": ["Pyruvate","Lactate", "Alanine", "PyruvateHydrate"],
                ...
            }

    Returns
    -------
    f_init : object
        Initialized model for fitting.

    Notes
    -----
    - The function uses Astropy's custom_model for defining the dynamic sum of Lorentzians.
    - It's designed to be flexible with the number of peaks, allowing for easy scalability.

    """
    # define fit functions + init fir params::
    # ------------------------------------------------------------------------------
    if input_data is None:
        if data_obj is not None:
            # ParaVision reco data:
            spec_data = data_obj.seq2d_reordered
        else:
            Warning("provide either input_data or spec_data")
    else:
        spec_data = input_data

    # fill fit_params dictonary with default values:
    fit_params = generate_default_fit_params_apy(
        data_obj=data_obj, input_data=input_data, fit_params=fit_params
    )

    # get metabolites:
    metabs = fit_params.get("metabs", ["", "", "", "", ""])

    # get linebroadening (use 0 if nothing was set)
    # lb has to be >0 for lineshape Voigt to work
    lb_Hz = fit_params.get("lb_Hz", 20)

    if fit_params["freq_unit"] == "ppm":
        # linebroadening factor:  Hz --> ppm (13c, @ 7T)
        lb = lb_Hz / (get_gmr("13c") * 7)
    else:
        lb = lb_Hz

    fit_params["lb"] = lb

    # number of peaks:
    n_peaks = len(metabs)

    # Retrieve predefined ppm values or use default values <-> x_0 definition
    # prefered over metab
    x_0_init = [
        fit_params.get(f"x_0_{k + 1}.value", get_metab_cs_ppm(metab=metab))
        for k, metab in enumerate(metabs)
    ]

    # Initialize amplitudes using either provided values or half the max of
    # the data
    amplitude_init = [
        fit_params.get(f"amplitude_{k + 1}.value", np.max(np.abs(spec_data) / 2.0))
        for k in range(n_peaks)
    ]

    # Initialize FWHM (full width at half maximum) using provided values or
    # default to 1
    fwhm_init = [
        fit_params.get(
            f"fwhm_{k + 1}.value",
            1,
        )
        for k in range(n_peaks)
    ]

    # Initialize FWHM (full width at half maximum) using provided values or
    # default to 1
    fwhm_G_init = [
        fit_params.get(
            f"fwhm_G_{k + 1}.value",
            lb,
        )
        for k in range(n_peaks)
    ]

    # Range for the bounds of x_0 values (defaults to 1)
    x_0_bounds_range = [
        fit_params.get(f"x_0_{k + 1}.bounds_range", 1) for k in range(n_peaks)
    ]

    lineshape = fit_params["lineshape"]
    if lineshape == "Lorentzian":
        # Initialize parameters based on predefined values or default ones

        # Retrieve predefined ppm values or use default values <-> x_0 definition prefered over metab
        # Initialize parameters based on predefined values or default ones

        # Define the model: A dynamic sum of Lorentzians
        @custom_model
        def sum_of_lorentzians(
            x,
            amplitude_1=1.0,
            x_0_1=10.0,
            fwhm_1=1.0,
            amplitude_2=1.0,
            x_0_2=30.0,
            fwhm_2=1.0,
            amplitude_3=1.0,
            x_0_3=40.0,
            fwhm_3=1.0,
            amplitude_4=1.0,
            x_0_4=50.0,
            fwhm_4=1.0,
            amplitude_5=1.0,
            x_0_5=60.0,
            fwhm_5=1.0,
            nfloor=0,
        ):
            # Initialize the total value to 0
            total = 0

            # Group parameters into lists for easier iteration
            amplitudes = [
                amplitude_1,
                amplitude_2,
                amplitude_3,
                amplitude_4,
                amplitude_5,
            ]
            x_0s = [x_0_1, x_0_2, x_0_3, x_0_4, x_0_5]
            fwhms = [fwhm_1, fwhm_2, fwhm_3, fwhm_4, fwhm_5]

            # Calculate the total sum of Lorentzians based on the number of
            # peaks
            for i in range(n_peaks):
                total += models.Lorentz1D(
                    amplitude=amplitudes[i], x_0=x_0s[i], fwhm=fwhms[i]
                )(x)

            # Add the floor (baseline) to the total
            return total + nfloor

        # Initialize the model with provided or default parameters
        initial_args = {}
        for i in range(n_peaks):
            initial_args[f"amplitude_{i + 1}"] = amplitude_init[i]
            initial_args[f"x_0_{i + 1}"] = x_0_init[i]
            initial_args[f"fwhm_{i + 1}"] = fwhm_init[i]
        f_init = sum_of_lorentzians(**initial_args)

    elif lineshape == "sqrtLorentzian":
        from ...utils.utils_spectroscopy import squareLorentzian

        # Define the model: A dynamic sum of Lorentzians
        @custom_model
        def sum_of_sqrtlorentzians(
            x,
            amplitude_1=1.0,
            x_0_1=10.0,
            fwhm_1=1.0,
            amplitude_2=1.0,
            x_0_2=30.0,
            fwhm_2=1.0,
            amplitude_3=1.0,
            x_0_3=40.0,
            fwhm_3=1.0,
            amplitude_4=1.0,
            x_0_4=50.0,
            fwhm_4=1.0,
            amplitude_5=1.0,
            x_0_5=60.0,
            fwhm_5=1.0,
            nfloor=0,
        ):
            # Initialize the total value to 0
            total = 0

            # Group parameters into lists for easier iteration
            amplitudes = [
                amplitude_1,
                amplitude_2,
                amplitude_3,
                amplitude_4,
                amplitude_5,
            ]
            x_0s = [x_0_1, x_0_2, x_0_3, x_0_4, x_0_5]
            fwhms = [fwhm_1, fwhm_2, fwhm_3, fwhm_4, fwhm_5]

            # Calculate the total sum of Lorentzians based on the number of
            # peaks
            for i in range(n_peaks):
                total += squareLorentzian(
                    amplitude=amplitudes[i], x_0=x_0s[i], fwhm=fwhms[i]
                )(x)

            # Add the floor (baseline) to the total
            return total + nfloor

        # Initialize the model with provided or default parameters
        initial_args = {}
        for i in range(n_peaks):
            initial_args[f"amplitude_{i + 1}"] = amplitude_init[i]
            initial_args[f"x_0_{i + 1}"] = x_0_init[i]
            initial_args[f"fwhm_{i + 1}"] = fwhm_init[i]
        f_init = sum_of_sqrtlorentzians(**initial_args)

    elif lineshape == "Voigt":
        # Initialize parameters based on predefined values or default ones
        # Define the model: A dynamic sum of Lorentzians
        @custom_model
        def sum_of_voigts(
            x,
            amplitude_1=1.0,
            x_0_1=10.0,
            fwhm_1=1.0,
            fwhm_G_1=1.0,
            amplitude_2=1.0,
            x_0_2=30.0,
            fwhm_2=1.0,
            fwhm_G_2=1.0,
            amplitude_3=1.0,
            x_0_3=40.0,
            fwhm_3=1.0,
            fwhm_G_3=1.0,
            amplitude_4=1.0,
            x_0_4=50.0,
            fwhm_4=1.0,
            fwhm_G_4=1.0,
            amplitude_5=1.0,
            x_0_5=60.0,
            fwhm_5=1.0,
            fwhm_G_5=1.0,
            nfloor=0,
        ):
            # Initialize the total value to 0
            total = 0

            # Group parameters into lists for easier iteration
            amplitudes = [
                amplitude_1,
                amplitude_2,
                amplitude_3,
                amplitude_4,
                amplitude_5,
            ]
            x_0s = [x_0_1, x_0_2, x_0_3, x_0_4, x_0_5]
            # Lorentzian line width:
            fwhm_Ls = [fwhm_1, fwhm_2, fwhm_3, fwhm_4, fwhm_5]
            # Gaussian linewidth:
            fwhm_Gs = [fwhm_G_1, fwhm_G_2, fwhm_G_3, fwhm_G_4, fwhm_G_5]

            # Calculate the total sum of Lorentzians based on the number of
            # peaks
            for i in range(n_peaks):
                total += models.Voigt1D(
                    amplitude_L=amplitudes[i],
                    x_0=x_0s[i],
                    fwhm_L=fwhm_Ls[i],
                    fwhm_G=fwhm_Gs[i],
                )(x)

            # Add the floor (baseline) to the total
            return total + nfloor

        # Initialize the model with provided or default parameters
        initial_args = {}
        for i in range(n_peaks):
            initial_args[f"amplitude_{i + 1}"] = amplitude_init[i]
            initial_args[f"x_0_{i + 1}"] = x_0_init[i]
            initial_args[f"fwhm_{i + 1}"] = fwhm_init[i]
            initial_args[f"fwhm_G_{i + 1}"] = fwhm_G_init[i]
        f_init = sum_of_voigts(**initial_args)

    else:
        Warning(f"unknwon lineshape: {lineshape}")
        return None

    # Set parameter bounds directly on the model's attributes
    for i in range(n_peaks):
        # Fix amplitude values and set them to 0.0 for peaks that have no
        # defined x_0_{i+1} (frequency)
        if x_0_init[i]:
            # Define bounds for amplitude, x_0, and fwhm for each peak
            getattr(f_init, f"amplitude_{i + 1}").bounds = fit_params.get(
                f"amplitude_{i + 1}.bounds", (0.0, np.inf)
            )
            getattr(f_init, f"x_0_{i + 1}").bounds = fit_params.get(
                f"x_0_{i + 1}.bounds",
                (
                    x_0_init[i] - x_0_bounds_range[i],
                    x_0_init[i] + x_0_bounds_range[i],
                ),
            )
            getattr(f_init, f"fwhm_{i + 1}").bounds = fit_params.get(
                f"fwhm_{i + 1}.bounds", (0.1, 10)
            )
            # fix Gaussian linewidth:
            if lineshape == "Voigt":
                getattr(f_init, f"fwhm_G_{i + 1}").fixed = fit_params.get(
                    f"fwhm_G_{i + 1}.fixed", True
                )
                # set Gaussian linewidth value:
                getattr(f_init, f"fwhm_G_{i + 1}").value = fit_params.get(
                    f"fwhm_G_{i + 1}.value", lb
                )

        else:
            getattr(f_init, f"amplitude_{i + 1}").fixed = fit_params.get(
                f"amplitude_{i + 1}.fixed", True
            )
            getattr(f_init, f"amplitude_{i + 1}").value = fit_params.get(
                f"amplitude_{i + 1}.value", 0.0
            )
            # the stuff between the 2 lines is just defined to avoid calling a non-defined parameter:
            # ------------------------------------------------------------
            # Define bounds for amplitude, x_0, and fwhm for each peak
            getattr(f_init, f"amplitude_{i + 1}").bounds = fit_params.get(
                f"amplitude_{i + 1}.bounds", (0.0, np.inf)
            )
            getattr(f_init, f"x_0_{i + 1}").bounds = fit_params.get(
                f"x_0_{i + 1}.bounds",
                (
                    0,  # simple fix, needs better solution later:
                    1,
                ),
            )
            getattr(f_init, f"fwhm_{i + 1}").bounds = fit_params.get(
                f"fwhm_{i + 1}.bounds", (0.1, 10)
            )
            # ------------------------------------------------------------

    # Set bounds for the floor (baseline) if it exists
    if hasattr(f_init, "nfloor"):
        f_init.nfloor.fixed = fit_params.get("nfloor.fixed", False)
    if f_init.nfloor.fixed:  # Set bounds for the floor (baseline) if it exists
        pass
    else:
        f_init.nfloor.bounds = fit_params.get(
            "nfloor.bounds", (0.0, 0.5 * amplitude_init[0])
        )

    return f_init


def generate_default_fit_params_apy(data_obj=None, input_data=None, fit_params=None):
    """
    Generate default parameters for fitting. Also helpful to get an overview of available
    parameters.

    Parameters
    ----------
    data_obj : object, optional
        Object containing data information, mainly used for fetching frequency range.

    input_data : ndarray, optional
        Spectral data used for certain default computations like area of data.

    fit_params : dict, optional
        Initial structure of fit parameters, to which this function can add or modify defaults.

    Returns
    -------
    fit_params : dict
        Updated dictionary with set default values.

    """
    if fit_params is None:
        fit_params = {}

    if input_data is None:
        if data_obj is not None:
            input_data = data_obj.seq2d
        else:
            pass

    # get unit (default to ppm)
    fit_params["freq_unit"] = fit_params.get("freq_unit", "ppm")

    # get expected lineshape (use Lorentzian if nothing was set):
    fit_params["lineshape"] = fit_params.get("lineshape", "sqrtLorentzian")

    # decide wether to normalize the data during fit process to area = 1
    # (highly recommended)
    fit_params["normalize_data"] = fit_params.get("normalize_data", True)

    # get metabolites:
    metabs = fit_params.get("metabs", ["", "", "", "", ""])
    while len(metabs) < 5:
        metabs.append("")

    # write metabolite list into fit_params:
    fit_params["metabs"] = metabs

    if input_data is None:
        if data_obj is not None:
            n_acquisition = data_obj.n_acquisition
        else:
            # default:
            n_acquisition = 200
    else:
        n_acquisition = input_data.shape[0]

    # get frequency range
    freq_range = fit_params.get(
        "fit_params.freq_range",
        get_freq_axis(
            data_obj=data_obj, unit=fit_params["freq_unit"], npoints=n_acquisition
        ),
    )
    fit_params["freq_range"] = freq_range

    if freq_range is None:
        Warning("freq_range is None, provide either scan or freq_range!")

    # set default fit method to Sequential Least Square Programming
    fit_method = fit_params.get("fit_method", "SLSQPLSQFitter")
    fit_params["fit_method"] = fit_method

    # epsilson (quasi step-size for SLSQP)
    if fit_method == "SLSQPLSQFitter":
        eps = fit_params.get("fit_eps", np.sqrt(np.finfo(float).eps))
        fit_params["fit_eps"] = eps
    else:  # for now use same epsilon for fitting:
        eps = fit_params.get("fit_eps", np.sqrt(np.finfo(float).eps))
        fit_params["fit_eps"] = eps

    # if the area should be "Normalized" This is really helpful when using
    # SLSQPLSQFitter
    fit_params["fit_area_opt"] = fit_params.get("fit_area_opt", True)

    # set default max number of iterations to 200:
    fit_params["fit_maxiter"] = fit_params.get("fit_maxiter", 200)

    # set default linebroadening (in Hz) to 20Hz:
    fit_params["lb_Hz"] = fit_params.get("lb_Hz", 20)

    # whether a second round of fitting with the main peak subtracted should
    # be performde:
    fit_params["fit_subtract_main_peak"] = fit_params.get(
        "fit_subtract_main_peak", False
    )

    # store area of spectrum:
    if input_data is None:
        fit_params["area_data"] = None
    else:
        fit_params["area_data"] = np.sum(input_data)

    # set b0 field strength:
    fit_params["b0"] = fit_params.get("b0", 7)

    # set nucleus:
    fit_params["nuc"] = fit_params.get("nuc", "13c")

    if input_data is None:
        # Maxiumum area value will be 100 (good for SLSQPLSQFitter)
        fit_params["fit_area_opt_factor"] = fit_params.get("fit_area_opt_factor", 1)

    else:
        # Maxiumum area value will be 100 (good for SLSQPLSQFitter)
        fit_params["fit_area_opt_factor"] = fit_params.get(
            "fit_area_opt_factor", np.max(np.sum(np.abs(input_data), axis=0)) / 100.0
        )

    # fit_params["fit_freq_range"] = fit_params.get("fit_freq_range",
    # [fit_params["freq_range"][0], fit_params["freq_range"][-1]] )

    # initialize fit parameters:
    for k in range(len(metabs)):
        # full - width half - max( in ppm)
        fit_params[f"fwhm_{k + 1}.value"] = fit_params.get(
            f"fwhm_{k + 1}.value",
            0.2,
        )
        # bounds for full - width half - max ( in ppm)
        fit_params[f"fwhm_{k + 1}.bounds"] = fit_params.get(
            f"fwhm_{k + 1}.bounds", (0.1, 2)
        )
        # fwhm of Gaussian (in case lineshape is Voigt)
        fit_params[f"fwhm_G_{k + 1}.value"] = fit_params.get(
            f"fwhm_G_{k + 1}.value",
            fit_params["lb_Hz"] / (get_gmr(fit_params["nuc"]) * fit_params["b0"]),
        )
        # Range for the bounds of x_0 values (defaults to 1)
        fit_params[f"x_0_{k + 1}.bounds_range"] = fit_params.get(
            f"x_0_{k + 1}.bounds_range", 1
        )
    return fit_params


def match_fit_init_apy(
    data_obj=None,
    fit_init=None,
    input_data=None,
    fit_params=None,
):
    """
    Adjusts the initial fit parameters to match the input data.
    This is mainly to adjust amplitudes and other initial parameters closer to what's observed in the actual data.

    Parameters
    ----------
    data_obj : object, optional
        Object containing data information.

    fit_init : object
        Initial fitting parameters.

    input_data : ndarray
        1D spectral data.

    freq_range : ndarray, optional
        Frequency range corresponding to the input data.

    fit_params : dict, optional
        Dictionary of fit parameters.

    Returns
    -------
    fit_init : object
        Updated fitting parameters after matching with input data.

    """
    if input_data is None:
        Warning("input_data is None!")
        return fit_init

    metabs = fit_params.get("metabs")

    # get number of peaks
    n_peaks = np.size(metabs)

    # get Gaussian linewidth if it was applied:
    from ...utils.physics import get_gmr

    # get nucleus
    nuc = fit_params.get("nuc", "13c")

    # get field strength
    b0 = fit_params.get("b0", 7)

    # turn Hz to ppm
    lb = fit_params.get("lb", 0) / (get_gmr(nuc) * b0)

    freq_range = fit_params.get("freq_range", None)

    # take magnitude of data in case it's complex:
    if isinstance(input_data, complex):
        input_data = np.abs(input_data)

    # get frequency range:
    if freq_range is None:
        freq_range = get_freq_axis(
            data_obj=data_obj, unit="ppm", npoints=input_data.shape[0]
        )

    fit_params["freq_range"] = freq_range

    # define range in which should be fitted:
    fit_freq_range = fit_params.get("fit_freq_range", [freq_range[0], freq_range[-1]])

    # define range:
    fit_range = [0, 0]
    fit_range[0] = np.argmin(np.abs(freq_range - fit_freq_range[0]))
    fit_range[1] = np.argmin(np.abs(freq_range - fit_freq_range[-1]))
    fit_params["fit_range"] = fit_params.get("fit_range", fit_range)

    # get fitting method:
    fit_method = fit_params.get("fit_method", "SLSQPLSQFitter")

    # set fitting epsilon depending on data size:
    if fit_method == "SLSQPLSQFitter":
        eps = fit_params.get("fit_eps", np.sqrt(np.finfo(float).eps))
        if eps == np.sqrt(np.finfo(float).eps):
            fit_params["fit_eps"] = eps
        else:
            # instead of scaling data, scale stepsze (does not work really
            # well):
            fit_params["fit_eps"] = np.sum(np.abs(input_data)) * eps

    # get expected lineshape (use Lorentzian if nothing was set):
    lineshape = fit_params.get("lineshape", "Lorentzian")

    if (
        lineshape == "Lorentzian"
        or lineshape == "sqrtLorentzian"
        or lineshape == "Voigt"
    ):
        # set bounds for amplitude:
        for k in range(n_peaks):  # Assuming k starts from 1
            if getattr(fit_init, f"amplitude_{k + 1}") > 0:
                # find maximum in range around frequency peak
                f = freq_to_index(
                    freq=getattr(fit_init, f"x_0_{k + 1}"),
                    freq_range=freq_range,
                )
                f_b = (
                    freq_to_index(
                        getattr(
                            getattr(
                                fit_init,
                                f"x_0_{k + 1}",
                            ),
                            "bounds",
                        )[0],
                        freq_range=freq_range,
                    ),
                    freq_to_index(
                        getattr(
                            getattr(
                                fit_init,
                                f"x_0_{k + 1}",
                            ),
                            "bounds",
                        )[1],
                        freq_range=freq_range,
                    ),
                )

                # set init value for amplitude:
                setattr(
                    fit_init,
                    f"amplitude_{k + 1}",
                    np.max(
                        input_data[
                            max(0, f - (f_b[1] - f_b[0])) : min(
                                len(freq_range) - 1, f + (f_b[1] - f_b[0])
                            )
                            + 1,
                        ]
                    )
                    - np.min(input_data),  # estimate for background
                )

                if lineshape == "Voigt":
                    # set init value for Gaussian linewidth:
                    setattr(
                        fit_init,
                        f"fwhm_G_{k + 1}",
                        lb,
                    )

        setattr(
            fit_init,
            "nfloor",
            np.min(input_data),
        )

    else:
        pass
    return fit_init
