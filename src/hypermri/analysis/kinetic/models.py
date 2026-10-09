"""Kinetic models and helpers for hyperpolarised 13C spectroscopy analyses."""

##########################################################################

# Kinetic modeling of hyperpolarized [1-13C]pyruvate and [1-13C]lactate signals

##########################################################################


# =========================================================================================================================================================
# Import necessary packages
# =========================================================================================================================================================

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy
from numpy.linalg import pinv
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# =========================================================================================================================================================
# Definition of necessary functions
# =========================================================================================================================================================


def pyruvate_input_function(t, M_pyr_const, t_inj, sigma):
    return M_pyr_const / 2 * (1 + np.arctan((t - t_inj) / sigma) * 2 / np.pi)


def pyruvate_input_function_time_derivative(t, M_pyr_const, t_inj, sigma):
    return M_pyr_const / (np.pi * sigma) * 1 / (1 + ((t - t_inj) / sigma) ** 2)


def print_vector(b, symbol):
    """Pretty-print a vector equation using SymPy symbols."""
    s = sympy.MatrixSymbol(symbol, b.shape[0], 1)
    t = sympy.MatrixSymbol("1", A.shape[1], 1)

    sympy.init_printing(use_unicode=False)
    sympy.pprint(sympy.Eq(s, sympy.Matrix(b) * t))


def print_matrix(A, symbol):
    """Pretty-print a matrix equation for visual inspection."""
    s = sympy.MatrixSymbol(symbol, A.shape[0], 1)
    t = sympy.MatrixSymbol("1", A.shape[1], 1)

    sympy.init_printing(use_unicode=False)
    sympy.pprint(sympy.Eq(s, sympy.Matrix(A) * t))


def print_linear_system_of_equations(A, b):
    """Display the linear system :math:`Ax = b` using SymPy pretty printing."""
    x = sympy.MatrixSymbol("x", A.shape[0], 1)

    sympy.init_printing(use_unicode=False)
    sympy.pprint(sympy.Eq(sympy.Matrix(b), sympy.Matrix(A) * x))


def append_df_to_excel(df, basic_path, excelsheet_name):
    """
    Append data row to existing Excel sheet.

    Parameters
    ----------
    df : panadas dataframe
        data row to be appended to Excel sheet

    basic_path : string
        filepath of targeted Excel sheet

    excelsheet_name : string
        name of targeted Excel sheet
    """

    df_excel = pd.read_excel(os.path.join(basic_path, excelsheet_name))
    result = pd.concat([df_excel, df], ignore_index=True)
    result.to_excel(os.path.join(basic_path, excelsheet_name), index=False)

    # Adjust column width
    writer = pd.ExcelWriter(excelsheet_name, engine="xlsxwriter")

    result.to_excel(writer, sheet_name="Sheet1")
    worksheet = writer.sheets["Sheet1"]
    for idx, col in enumerate(result):
        series = result[col]
        max_len = max((series.astype(str).map(len).max(), len(str(series.name)))) + 1
        worksheet.set_column(idx + 1, idx + 1, max_len)
        
    worksheet.autofit()
    writer.close()


def kinetic_modeling_of_HP_pyruvate_and_lactate_after_injection_using_RK4_ODE_integration_and_lstsquaresfit(
    t,
    M_pyr,
    M_lac,
    M_lac_scaling_factor,
    flipangle,
    t_repetition,
    t_min_modeling,
    lower_coeff_bounds,
    upper_coeff_bounds,
    basic_path,
    sample_ID,
    excelsheet_name=r"Overview_kinetic_modeling_fit_parameters.xlsx",
    fitmodel="with reverse conversion",
    plot=True,
):
    """
    Get conversion and relaxation rates of hyperpolarized pyruvate and lactate after hyperpolarized pyruvate injection.

    The kinetic model can be visualized as follows::

                            forward conversion
                                k_pyr_lac
        M_pyr         <=========================>        M_lac
                                k_lac_pyr
                            reverse conversion
        ||                                               ||
        \\/                                               \\/
        r_pyr                                            r_lac
        = 1/T_1_pyr                                      = 1/T_1_lac

    Linear system of differential equations::

        d/dt   | M_pyr |  =  | -k_pyr_lac-r_pyr        k_lac_pyr    |  °  | M_pyr |  +  | dM_pyr_in/dt |
               | M_lac |     |     k_pyr_lac       -k_lac_pyr-r_lac |     | M_lac |     |       0      |

    With M_pyr_in the input function delivering hyperpolarized pyruvate into the intracellular compartments.
    M_pyr_in is approximated as a smoothed Heavy-side step function::

        M_pyr_in     = M_pyr_const/2 ° (1 + 2/pi ° arctan((t - t_inj)/sigma))
        dM_pyr_in/dt = M_pyr_const/(pi°sigma) ° 1/(1 + ((t - t_inj)/sigma)^2)

    Transposed linear system of differential equations::

        d/dt   | M_pyr ,  M_lac |  =  | M_pyr ,  M_lac | °  | -k_pyr_lac-r_pyr        k_pyr_lac    |  +  | dM_pyr_in/dt,  0  |
                                                            |     k_lac_pyr       -k_lac_pyr-r_lac |

    Parameters
    ----------
    t : 1D array, float
        sampled time points [s]

    M_pyr : 1D array, float
        time development of magnetization of hyperpolarized 13C pyruvate [a. u.]

    M_lac : 1D array, float
        time development of magnetization of hyperpolarized 13C lactate [a. u.]

    M_lac_scaling_factor : float
        scaling factor for plotting M_lac

    flipangle : float
        applied flip angle of the radiofrequency pulse [°]

    t_repetition : float
        time between two subsequent radiofrequency pulses [s]

    t_min_modeling : float
        starting time for kinetic modeling (larger than injection time) [s]

    lower_coeff_bounds : list, float
        lower fitting bounds for the fit parameters.

        Structure::

                     [k_pyr_lac, k_lac_pyr, r_pyr, r_lac, M_pyr_const, t_inj, sigma, M_pyr_0, M_lac_0]
            indices  [0,         1,         2,     3,     4,           5,     6,     7,       8      ]
            unit    [1/s,       1/s,       1/s,   1/s,   a. u.,       s,     s,     a. u.,   a. u.  ]
            example  [0.000,     0.000,     0.00,  0.00,  0.5,         15,    0.0,   0.0,     0.0    ]

    upper_coeff_bounds : list, float
        upper fitting bounds for the fit parameters.

        Structure::

                     [k_pyr_lac, k_lac_pyr, r_pyr, r_lac, M_pyr_const, t_inj, sigma, M_pyr_0, M_lac_0]
            indices  [0,         1,         2,     3,     4,           5,     6,     7,       8      ]
            unit    [1/s,       1/s,       1/s,   1/s,   a. u.,       s,     s,     a. u.,   a. u.  ]
            example  [0.060,     0.002,     0.10,  0.10,  2.0,         30,    2.0,   1.0,     1.0    ]

    basic_path : string
        filepath for saving fit parameter results and fit plot

    sample_ID : string
        characterization of the sample

    excelsheet_name: string
        name of the Excel sheet to which the fit results should be appended

    fitmodel : 'with reverse conversion' or 'without reverse conversion'
        flag for choosing different kinetic models

    plot : boolean
        flag for turning on or off plotting

    Returns
    -------
    fit_results: pandas dataframe

    """

    # Define modeling region
    modeling_region = np.where(t >= t_min_modeling)

    # Truncate arrays to modeling region, normalize pyruvate and lactate data
    # and apply scaling factor to lactate data
    t_trunc = t[modeling_region]
    M_pyr_norm_trunc = M_pyr[modeling_region] / np.max(M_pyr)
    M_lac_norm_trunc = M_lac[modeling_region] / np.max(M_pyr)

    # Define the pyruvate input function time derivative
    def pyruvate_input_function_time_derivative(t, M_pyr_const, t_inj, sigma):
        return M_pyr_const / (np.pi * sigma) * 1 / (1 + ((t - t_inj) / sigma) ** 2)

    if fitmodel == "with reverse conversion":
        # Define ODE describing conversion and relaxation
        def ODE(
            t,
            state,
            k_pyr_lac,
            k_lac_pyr,
            r_pyr,
            r_lac,
            M_pyr_const,
            t_inj,
            sigma,
            M_pyr_0,
            M_lac_0,
        ):
            M_pyr, M_lac = state

            dM_pyr_dt = (
                -(k_pyr_lac + r_pyr) * M_pyr
                + k_lac_pyr * M_lac
                + pyruvate_input_function_time_derivative(t, M_pyr_const, t_inj, sigma)
            )
            dM_lac_dt = -(k_lac_pyr + r_lac) * M_lac + k_pyr_lac * M_pyr

            return [dM_pyr_dt, dM_lac_dt]

        # Define ODE integration
        def kinetics(
            t,
            k_pyr_lac,
            k_lac_pyr,
            r_pyr,
            r_lac,
            M_pyr_const,
            t_inj,
            sigma,
            M_pyr_0,
            M_lac_0,
        ):
            P = (
                k_pyr_lac,
                k_lac_pyr,
                r_pyr,
                r_lac,
                M_pyr_const,
                t_inj,
                sigma,
                M_pyr_0,
                M_lac_0,
            )

            return np.hstack(
                [
                    odeint(
                        ODE, [M_pyr_0, M_lac_0], t[: t.shape[0] // 2], P, tfirst=True
                    )[:, 0],
                    odeint(
                        ODE, [M_pyr_0, M_lac_0], t[t.shape[0] // 2 :], P, tfirst=True
                    )[:, 1],
                ]
            )

        # Apply least squares fit of the measured data to the integrated ODE
        P, P_covariance = curve_fit(
            kinetics,
            np.hstack([t_trunc, t_trunc]),
            np.hstack([M_pyr_norm_trunc, M_lac_norm_trunc]),
            bounds=(lower_coeff_bounds, upper_coeff_bounds),
        )

        P_errors = np.sqrt(np.diag(P_covariance))

        # Assign parameter names to fit results
        k_pyr_lac = P[0]
        k_pyr_lac_err = P_errors[0]
        k_lac_pyr = P[1]
        k_lac_pyr_err = P_errors[1]

        r_pyr = P[2]
        r_pyr_err = P_errors[2]
        r_lac = P[3]
        r_lac_err = P_errors[3]

        T_1_pyr = 1 / (P[2] + np.log(np.cos(flipangle * np.pi / 180)) / t_repetition)
        T_1_pyr_err = P_errors[2] / P[2] * T_1_pyr
        T_1_lac = 1 / (P[3] + np.log(np.cos(flipangle * np.pi / 180)) / t_repetition)
        T_1_lac_err = P_errors[3] / P[3] * T_1_lac

        M_pyr_const = P[4]
        M_pyr_const_err = P_errors[4]
        t_inj = P[5]
        t_inj_err = P_errors[5]
        sigma = P[6]
        sigma_err = P_errors[6]
        M_pyr_0 = P[7]
        M_pyr_0_err = P_errors[7]
        M_lac_0 = P[8]
        M_lac_0_err = P_errors[8]

    elif fitmodel == "without reverse conversion":
        # Define ODE describing conversion and relaxation
        def ODE(
            t,
            state,
            k_pyr_lac,
            r_pyr,
            r_lac,
            M_pyr_const,
            t_inj,
            sigma,
            M_pyr_0,
            M_lac_0,
        ):
            M_pyr, M_lac = state

            dM_pyr_dt = -(
                k_pyr_lac + r_pyr
            ) * M_pyr + pyruvate_input_function_time_derivative(
                t, M_pyr_const, t_inj, sigma
            )
            dM_lac_dt = -r_lac * M_lac + k_pyr_lac * M_pyr

            return [dM_pyr_dt, dM_lac_dt]

        # Define ODE integration
        def kinetics(
            t, k_pyr_lac, r_pyr, r_lac, M_pyr_const, t_inj, sigma, M_pyr_0, M_lac_0
        ):
            P = (k_pyr_lac, r_pyr, r_lac, M_pyr_const, t_inj, sigma, M_pyr_0, M_lac_0)

            return np.hstack(
                [
                    odeint(
                        ODE, [M_pyr_0, M_lac_0], t[: t.shape[0] // 2], P, tfirst=True
                    )[:, 0],
                    odeint(
                        ODE, [M_pyr_0, M_lac_0], t[t.shape[0] // 2 :], P, tfirst=True
                    )[:, 1],
                ]
            )

        # Apply least squares fit of the measured data to the integrated ODE
        P, P_covariance = curve_fit(
            kinetics,
            np.hstack([t_trunc, t_trunc]),
            np.hstack([M_pyr_norm_trunc, M_lac_norm_trunc]),
            bounds=(lower_coeff_bounds, upper_coeff_bounds),
        )

        P_errors = np.sqrt(np.diag(P_covariance))

        # Assign parameter names to fit results
        k_pyr_lac = P[0]
        k_pyr_lac_err = P_errors[0]
        k_lac_pyr = 0
        k_lac_pyr_err = 0

        r_pyr = P[1]
        r_pyr_err = P_errors[1]
        r_lac = P[2]
        r_lac_err = P_errors[2]

        T_1_pyr = 1 / (P[1] + np.log(np.cos(flipangle * np.pi / 180)) / t_repetition)
        T_1_pyr_err = P_errors[1] / P[1] * T_1_pyr
        T_1_lac = 1 / (P[2] + np.log(np.cos(flipangle * np.pi / 180)) / t_repetition)
        T_1_lac_err = P_errors[2] / P[2] * T_1_lac

        M_pyr_const = P[3]
        M_pyr_const_err = P_errors[3]
        t_inj = P[4]
        t_inj_err = P_errors[4]
        sigma = P[5]
        sigma_err = P_errors[5]
        M_pyr_0 = P[6]
        M_pyr_0_err = P_errors[6]
        M_lac_0 = P[7]
        M_lac_0_err = P_errors[7]

    # Summarize fit results in dictionary
    results_dict = {
        "sample_ID": [sample_ID],
        "t_min_modeling [s]": [t_min_modeling],
        "k_pyr_lac [1/s]": [k_pyr_lac],
        "k_pyr_lac_err [1/s]": [k_pyr_lac_err],
        "k_lac_pyr [1/s]": [k_lac_pyr],
        "k_lac_pyr_err [1/s]": [k_lac_pyr_err],
        "r_pyr [1/s]": [r_pyr],
        "r_pyr_err [1/s]": [r_pyr_err],
        "r_lac [1/s]": [r_lac],
        "r_lac_err [1/s]": [r_lac_err],
        "T_1_pyr [s]": [T_1_pyr],
        "T_1_pyr_err [s]": [T_1_pyr_err],
        "T_1_lac [s]": [T_1_lac],
        "T_1_lac_err [s]": [T_1_lac_err],
        "M_pyr_const [a. u.]": [M_pyr_const],
        "M_pyr_const_err [a. u.]": [M_pyr_const_err],
        "t_inj [s]": [t_inj],
        "t_inj_err [s]": [t_inj_err],
        "sigma [s]": [sigma],
        "sigma_err [s]": [sigma_err],
        "M_pyr_0 [a. u.]": [M_pyr_0],
        "M_pyr_0_err [a. u.]": [M_pyr_0_err],
        "M_lac_0 [a. u.]": [M_lac_0],
        "M_lac_0_err [a. u.]": [M_lac_0_err],
    }

    # Display fit results in pandas dataframe
    # results_dataframe = pd.DataFrame(results_dict)
    # display(results_dataframe)

    # Append fit results to Excel sheet
    if not os.path.exists(os.path.join(basic_path, excelsheet_name)):
        results_dataframe_row = pd.DataFrame(
            {
                "sample_ID": [],
                "t_min_modeling [s]": [],
                "k_pyr_lac [1/s]": [],
                "k_pyr_lac_err [1/s]": [],
                "k_lac_pyr [1/s]": [],
                "k_lac_pyr_err [1/s]": [],
                "r_pyr [1/s]": [],
                "r_pyr_err [1/s]": [],
                "r_lac [1/s]": [],
                "r_lac_err [1/s]": [],
                "T_1_pyr [s]": [],
                "T_1_pyr_err [s]": [],
                "T_1_lac [s]": [],
                "T_1_lac_err [s]": [],
                "M_pyr_const [a. u.]": [],
                "M_pyr_const_err [a. u.]": [],
                "t_inj [s]": [],
                "t_inj_err [s]": [],
                "sigma [s]": [],
                "sigma_err [s]": [],
                "M_pyr_0 [a. u.]": [],
                "M_pyr_0_err [a. u.]": [],
                "M_lac_0 [a. u.]": [],
                "M_lac_0_err [a. u.]": [],
            }
        )

        results_dataframe_row.to_excel(
            os.path.join(basic_path, excelsheet_name), index=False
        )

    results_dataframe_row = pd.DataFrame(
        {
            "sample_ID": [sample_ID],
            "t_min_modeling [s]": [t_min_modeling],
            "k_pyr_lac [1/s]": [k_pyr_lac],
            "k_pyr_lac_err [1/s]": [k_pyr_lac_err],
            "k_lac_pyr [1/s]": [k_lac_pyr],
            "k_lac_pyr_err [1/s]": [k_lac_pyr_err],
            "r_pyr [1/s]": [r_pyr],
            "r_pyr_err [1/s]": [r_pyr_err],
            "r_lac [1/s]": [r_lac],
            "r_lac_err [1/s]": [r_lac_err],
            "T_1_pyr [s]": [T_1_pyr],
            "T_1_pyr_err [s]": [T_1_pyr_err],
            "T_1_lac [s]": [T_1_lac],
            "T_1_lac_err [s]": [T_1_lac_err],
            "M_pyr_const [a. u.]": [M_pyr_const],
            "M_pyr_const_err [a. u.]": [M_pyr_const_err],
            "t_inj [s]": [t_inj],
            "t_inj_err [s]": [t_inj_err],
            "sigma [s]": [sigma],
            "sigma_err [s]": [sigma_err],
            "M_pyr_0 [a. u.]": [M_pyr_0],
            "M_pyr_0_err [a. u.]": [M_pyr_0_err],
            "M_lac_0 [a. u.]": [M_lac_0],
            "M_lac_0_err [a. u.]": [M_lac_0_err],
        }
    )

    append_df_to_excel(results_dataframe_row, basic_path, excelsheet_name)

    # Plot the fitting results

    if plot:
        title = r"Kinetic modeling of HP [1-$^{13}$C]pyruvate and [1-$^{13}$C]lactate magnetization"

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=[14, 4.8])

        x11 = t
        x12 = t
        x13 = np.arange(t_trunc[0], t_trunc[-1], 0.01)
        x14 = np.arange(t_trunc[0], t_trunc[-1], 0.01)

        xmin1 = -30
        xmax1 = 240

        y11 = M_pyr / np.max(M_pyr)
        y12 = M_lac / np.max(M_pyr) * M_lac_scaling_factor
        y13 = odeint(ODE, [M_pyr_0, M_lac_0], x13, tuple(P), tfirst=True)[:, 0]
        y14 = (
            odeint(ODE, [M_pyr_0, M_lac_0], x14, tuple(P), tfirst=True)[:, 1]
            * M_lac_scaling_factor
        )

        ymin1 = -0.3
        ymax1 = 1.5

        ax1.set_title(title)
        ax1.set_ylabel(r"signal [a. u.]")
        ax1.set_xlabel(r"time [s]")

        ax1.scatter(
            x11,
            y11,
            color="black",
            s=5,
            label=r"HP [1-$^{13}$C]pyruvate, measurement",
            marker="x",
        )
        if M_lac_scaling_factor != 1:
            ax1.scatter(
                x12,
                y12,
                color="red",
                s=5,
                label=r"{} x ".format(M_lac_scaling_factor)
                + r"HP [1-$^{13}$C]lactate, measurement",
                marker="x",
            )
        else:
            ax1.scatter(
                x12,
                y12,
                color="red",
                s=5,
                label=r"HP [1-$^{13}$C]lactate, measurement",
                marker="x",
            )

        ax1.plot(
            x13,
            y13,
            color="black",
            lw=1.5,
            linestyle="-",
            label=r"HP [1-$^{13}$C]pyruvate, " + "model",
        )
        if M_lac_scaling_factor != 1:
            ax1.plot(
                x14,
                y14,
                color="red",
                lw=1.5,
                linestyle="-",
                label=r"{} x ".format(M_lac_scaling_factor)
                + r"HP [1-$^{13}$C]lactate, "
                + "model",
            )
        else:
            ax1.plot(
                x14,
                y14,
                color="red",
                lw=1.5,
                linestyle="-",
                label=r"HP [1-$^{13}$C]lactate, " + "model",
            )

        ax1.set_xlim(xmin1, xmax1)
        ax1.set_ylim(ymin1, ymax1)

        # ax1.axvline(0, color='black', linestyle='dashed', linewidth=1)
        # ax1.axhline(0, color='black', linestyle='dashed', linewidth=1)

        ax1.grid()

        leg = ax1.legend(loc="upper right", framealpha=1, title=sample_ID)
        # leg = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        leg.get_frame().set_edgecolor("black")

        #######################################################################

        # ax2 = ax1.twinx()
        # ax2.get_xaxis().set_visible(False)
        # ax2.get_yaxis().set_visible(False)

        ax2.set_axis_off()

        handles2, labels2 = ax2.get_legend_handles_labels()

        handles2.append(patches.Patch(color="none", label=""))

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$t_{\mathrm{min, modeling}}$ = "
                + "{:.1f}".format(t_min_modeling)
                + " s",
            )
        )

        handles2.append(patches.Patch(color="none", label=""))

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$k_{\mathrm{PL}}$ = "
                + "({:.4f}".format(k_pyr_lac)
                + r" $\pm$ "
                + "{:.4f}) 1/s".format(k_pyr_lac_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$k_{\mathrm{LP}}$ = "
                + "({:.4f}".format(k_lac_pyr)
                + r" $\pm$ "
                + "{:.4f}) 1/s".format(k_lac_pyr_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$r_{\mathrm{P}}$ = "
                + "({:.3f}".format(r_pyr)
                + r" $\pm$ "
                + "{:.3f}) 1/s".format(r_pyr_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$r_{\mathrm{L}}$ = "
                + "({:.3f}".format(r_lac)
                + r" $\pm$ "
                + "{:.3f}) 1/s".format(r_lac_err),
            )
        )

        handles2.append(patches.Patch(color="none", label=""))

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$T_{\mathrm{1,P}}$ = "
                + "({:.2f}".format(T_1_pyr)
                + r" $\pm$ "
                + "{:.2f}) s".format(T_1_pyr_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$T_{\mathrm{1,L}}$ = "
                + "({:.2f}".format(T_1_lac)
                + r" $\pm$ "
                + "{:.2f}) s".format(T_1_lac_err),
            )
        )

        handles2.append(patches.Patch(color="none", label=""))

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$M_{\mathrm{P,const}}$ = "
                + "({:.2f}".format(M_pyr_const)
                + r" $\pm$ "
                + "{:.2f})".format(M_pyr_const_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$t_{\mathrm{inj}}$ = "
                + "({:.2f}".format(t_inj)
                + r" $\pm$ "
                + "{:.2f}) s".format(t_inj_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$\sigma_{\mathrm{P,in}}$ = "
                + "({:.2f}".format(sigma)
                + r" $\pm$ "
                + "{:.2f}) s".format(sigma_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$M_{\mathrm{P,0}}$ = "
                + "({:.2f}".format(M_pyr_0)
                + r" $\pm$ "
                + "{:.2f})".format(M_pyr_0_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$M_{\mathrm{L,0}}$ = "
                + "({:.2f}".format(M_lac_0)
                + r" $\pm$ "
                + "{:.2f})".format(M_lac_0_err),
            )
        )

        leg2 = ax2.legend(
            handles=handles2,
            ncols=1,
            title="Fit parameters, {}".format(fitmodel),
            loc="center left",
        )  # , bbox_to_anchor=(0, 1.5))
        leg2.get_frame().set_edgecolor("black")

        #######################################################################

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                basic_path,
                sample_ID
                + r"_RK4_ODE_integration_lstsqufit_solver_"
                + fitmodel.replace(" ", "_"),
            )
        )

        #########################################################################################################################################################################
        title = r"Kinetic modeling of HP [1-$^{13}$C]pyruvate and [1-$^{13}$C]lactate magnetization"

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=[14, 4.8])

        x11 = t
        x12 = t
        # x13 = np.arange(t_trunc[0], t_trunc[-1], 0.01)
        # x14 = np.arange(t_trunc[0], t_trunc[-1], 0.01)

        xmin1 = -30
        xmax1 = 240
        x13 = np.arange(xmin1, xmax1, 0.01)
        x14 = np.arange(xmin1, xmax1, 0.01)

        y11 = M_pyr / np.max(M_pyr)
        y12 = M_lac / np.max(M_pyr) * M_lac_scaling_factor
        y13 = 0.3 * pyruvate_input_function(x13, M_pyr_const, t_inj, sigma)
        y14 = 3 * pyruvate_input_function_time_derivative(
            x14, M_pyr_const, t_inj, sigma
        )

        ymin1 = -0.3
        ymax1 = 1.5

        ax1.set_title(title)
        ax1.set_ylabel(r"signal [a. u.]")
        ax1.set_xlabel(r"time [s]")

        ax1.scatter(
            x11,
            y11,
            color="black",
            s=5,
            label=r"HP [1-$^{13}$C]pyruvate, measurement",
            marker="x",
        )
        if M_lac_scaling_factor != 1:
            ax1.scatter(
                x12,
                y12,
                color="red",
                s=5,
                label=r"{} x ".format(M_lac_scaling_factor)
                + r"HP [1-$^{13}$C]lactate, measurement",
                marker="x",
            )
        else:
            ax1.scatter(
                x12,
                y12,
                color="red",
                s=5,
                label=r"HP [1-$^{13}$C]lactate, measurement",
                marker="x",
            )

        ax1.plot(
            x13,
            y13,
            color="dodgerblue",
            lw=1.5,
            linestyle="-",
            label=r"0.3 x $M_{\mathrm{P, in}}$ ($t$) [a. u.]",
        )
        ax1.plot(
            x14,
            y14,
            color="blue",
            lw=1.5,
            linestyle="-",
            label=r"3 x d$M_{\mathrm{P, in}}$/d$t$ ($t$) [a. u./s]",
        )

        ax1.set_xlim(xmin1, xmax1)
        ax1.set_ylim(ymin1, ymax1)

        ax1.axvline(
            t_inj,
            color="black",
            linestyle="dashed",
            linewidth=1,
            label=r"$t_{\mathrm{inj}}$ [s]",
        )
        ax1.axhline(
            np.max(y14) / 2,
            (t_inj - xmin1) / (xmax1 - xmin1),
            (t_inj + sigma - xmin1) / (xmax1 - xmin1),
            color="green",
            linewidth=1,
            label=r"$\sigma_{\mathrm{P,in}}$ [s]",
        )

        ax1.grid()

        leg = ax1.legend(loc="upper right", framealpha=1, title=sample_ID)
        # leg = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        leg.get_frame().set_edgecolor("black")

        #########################################################################################################################################################################

        # ax2 = ax1.twinx()
        # ax2.get_xaxis().set_visible(False)
        # ax2.get_yaxis().set_visible(False)

        ax2.set_axis_off()

        handles2, labels2 = ax2.get_legend_handles_labels()

        handles2.append(patches.Patch(color="none", label=""))

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$t_{\mathrm{min, modeling}}$ = "
                + "{:.1f}".format(t_min_modeling)
                + " s",
            )
        )

        handles2.append(patches.Patch(color="none", label=""))

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$k_{\mathrm{PL}}$ = "
                + "({:.4f}".format(k_pyr_lac)
                + r" $\pm$ "
                + "{:.4f}) 1/s".format(k_pyr_lac_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$k_{\mathrm{LP}}$ = "
                + "({:.4f}".format(k_lac_pyr)
                + r" $\pm$ "
                + "{:.4f}) 1/s".format(k_lac_pyr_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$r_{\mathrm{P}}$ = "
                + "({:.3f}".format(r_pyr)
                + r" $\pm$ "
                + "{:.3f}) 1/s".format(r_pyr_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$r_{\mathrm{L}}$ = "
                + "({:.3f}".format(r_lac)
                + r" $\pm$ "
                + "{:.3f}) 1/s".format(r_lac_err),
            )
        )

        handles2.append(patches.Patch(color="none", label=""))

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$T_{\mathrm{1,P}}$ = "
                + "({:.2f}".format(T_1_pyr)
                + r" $\pm$ "
                + "{:.2f}) s".format(T_1_pyr_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$T_{\mathrm{1,L}}$ = "
                + "({:.2f}".format(T_1_lac)
                + r" $\pm$ "
                + "{:.2f}) s".format(T_1_lac_err),
            )
        )

        handles2.append(patches.Patch(color="none", label=""))

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$M_{\mathrm{P,const}}$ = "
                + "({:.2f}".format(M_pyr_const)
                + r" $\pm$ "
                + "{:.2f})".format(M_pyr_const_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$t_{\mathrm{inj}}$ = "
                + "({:.2f}".format(t_inj)
                + r" $\pm$ "
                + "{:.2f}) s".format(t_inj_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$\sigma_{\mathrm{P,in}}$ = "
                + "({:.2f}".format(sigma)
                + r" $\pm$ "
                + "{:.2f}) s".format(sigma_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$M_{\mathrm{P,0}}$ = "
                + "({:.2f}".format(M_pyr_0)
                + r" $\pm$ "
                + "{:.2f})".format(M_pyr_0_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$M_{\mathrm{L,0}}$ = "
                + "({:.2f}".format(M_lac_0)
                + r" $\pm$ "
                + "{:.2f})".format(M_lac_0_err),
            )
        )

        leg2 = ax2.legend(
            handles=handles2,
            ncols=1,
            title="Fit parameters, {}".format(fitmodel),
            loc="center left",
        )  # , bbox_to_anchor=(0, 1.5))
        leg2.get_frame().set_edgecolor("black")

        #########################################################################################################################################################################

        plt.tight_layout()
        plt.savefig(os.path.join(basic_path, sample_ID + r"_pyruvate_input_function"))

    return results_dict


def kinetic_modeling_of_HP_pyruvate_and_lactate_after_injection_using_RK4_ODE_integration_and_lstsquaresfit_without_pyruvate_input_function(
    t,
    M_pyr,
    M_lac,
    M_lac_scaling_factor,
    flipangle,
    t_repetition,
    t_min_modeling,
    lower_coeff_bounds,
    upper_coeff_bounds,
    basic_path,
    sample_ID,
    excelsheet_name=r"Overview_kinetic_modeling_fit_parameters.xlsx",
    fitmodel="with reverse conversion",
    plot=True,
):
    """
    Get conversion and relaxation rates of hyperpolarized pyruvate and lactate after hyperpolarized pyruvate injection.

                        forward conversion
                            k_pyr_lac
    M_pyr         <=========================>        M_lac
                            k_lac_pyr
                        reverse conversion
    ||                                               ||
    \\/                                               \\/
    r_pyr                                            r_lac
    = 1/T_1_pyr                                      = 1/T_1_lac

    Linear system of differential equations:

    d/dt   | M_pyr |  =  | -k_pyr_lac-r_pyr        k_lac_pyr    |  °  | M_pyr |
           | M_lac |     |     k_pyr_lac       -k_lac_pyr-r_lac |     | M_lac |

    Transposed linear system of differential equations:

    d/dt   | M_pyr ,  M_lac |  =  | M_pyr ,  M_lac | °  | -k_pyr_lac-r_pyr        k_pyr_lac    |
                                                        |     k_lac_pyr       -k_lac_pyr-r_lac |

    Parameters
    ----------
    t : 1D array, float
        sampled time points [s]

    M_pyr : 1D array, float
        time development of magnetization of hyperpolarized 13C pyruvate [a. u.]

    M_lac : 1D array, float
        time development of magnetization of hyperpolarized 13C lactate [a. u.]

    M_lac_scaling_factor : float
        scaling factor for plotting M_lac

    flipangle : float
        applied flip angle of the radiofrequency pulse [°]

    t_repetition : float
        time between two subsequent radiofrequency pulses [s]

    t_min_modeling : float
        starting time for kinetic modeling (larger than injection time) [s]

    lower_coeff_bounds : list, float
        lower fitting bounds for the fit parameters:

                 [k_pyr_lac, k_lac_pyr, r_pyr, r_lac, M_pyr_0, M_lac_0]
        indices  [0,         1,         2,     3,     4,       5      ]
        unit    [1/s,       1/s,       1/s,   1/s,   a. u.,   a. u.  ]
        example  [0.000,     0.000,     0.00,  0.00,  0.0,     0.0    ]

    upper_coeff_bounds : list, float
        upper fitting bounds for the fit parameters:

                 [k_pyr_lac, k_lac_pyr, r_pyr, r_lac, M_pyr_0, M_lac_0]
        indices  [0,         1,         2,     3,     4,       5      ]
        unit    [1/s,       1/s,       1/s,   1/s,   a. u.,   a. u.  ]
        example  [0.060,     0.002,     0.10,  0.10,  1.0,     1.0    ]

    basic_path : string
        filepath for saving fit parameter results and fit plot

    sample_ID : string
        characterization of the sample

    excelsheet_name: string
        name of the Excel sheet to which the fit results should be appended

    fitmodel : 'with reverse conversion' or 'without reverse conversion'
        flag for choosing different kinetic models

    plot : boolean
        flag for turning on or off plotting

    Returns
    -------
    fit_results: pandas dataframe

    """

    # Define modeling region
    modeling_region = np.where(t >= t_min_modeling)

    # Truncate arrays to modeling region, normalize pyruvate and lactate data
    # and apply scaling factor to lactate data
    t_trunc = t[modeling_region]
    M_pyr_norm_trunc = M_pyr[modeling_region] / np.max(M_pyr)
    M_lac_norm_trunc = M_lac[modeling_region] / np.max(M_pyr)

    if fitmodel == "with reverse conversion":
        # Define ODE describing conversion and relaxation
        def ODE(
            t,
            state,
            k_pyr_lac,
            k_lac_pyr,
            r_pyr,
            r_lac,
            M_pyr_0,
            M_lac_0,
        ):
            M_pyr, M_lac = state

            dM_pyr_dt = -(k_pyr_lac + r_pyr) * M_pyr + k_lac_pyr * M_lac
            dM_lac_dt = -(k_lac_pyr + r_lac) * M_lac + k_pyr_lac * M_pyr

            return [dM_pyr_dt, dM_lac_dt]

        # Define ODE integration
        def kinetics(
            t,
            k_pyr_lac,
            k_lac_pyr,
            r_pyr,
            r_lac,
            M_pyr_0,
            M_lac_0,
        ):
            P = (
                k_pyr_lac,
                k_lac_pyr,
                r_pyr,
                r_lac,
                M_pyr_0,
                M_lac_0,
            )

            return np.hstack(
                [
                    odeint(
                        ODE, [M_pyr_0, M_lac_0], t[: t.shape[0] // 2], P, tfirst=True
                    )[:, 0],
                    odeint(
                        ODE, [M_pyr_0, M_lac_0], t[t.shape[0] // 2 :], P, tfirst=True
                    )[:, 1],
                ]
            )

        # Apply least squares fit of the measured data to the integrated ODE
        P, P_covariance = curve_fit(
            kinetics,
            np.hstack([t_trunc, t_trunc]),
            np.hstack([M_pyr_norm_trunc, M_lac_norm_trunc]),
            bounds=(lower_coeff_bounds, upper_coeff_bounds),
        )

        P_errors = np.sqrt(np.diag(P_covariance))

        # Assign parameter names to fit results
        k_pyr_lac = P[0]
        k_pyr_lac_err = P_errors[0]
        k_lac_pyr = P[1]
        k_lac_pyr_err = P_errors[1]

        r_pyr = P[2]
        r_pyr_err = P_errors[2]
        r_lac = P[3]
        r_lac_err = P_errors[3]

        T_1_pyr = 1 / (P[2] + np.log(np.cos(flipangle * np.pi / 180)) / t_repetition)
        T_1_pyr_err = P_errors[2] / P[2] * T_1_pyr
        T_1_lac = 1 / (P[3] + np.log(np.cos(flipangle * np.pi / 180)) / t_repetition)
        T_1_lac_err = P_errors[3] / P[3] * T_1_lac

        M_pyr_0 = P[4]
        M_pyr_0_err = P_errors[4]
        M_lac_0 = P[5]
        M_lac_0_err = P_errors[5]

    elif fitmodel == "without reverse conversion":
        # Define ODE describing conversion and relaxation
        def ODE(
            t,
            state,
            k_pyr_lac,
            r_pyr,
            r_lac,
            M_pyr_0,
            M_lac_0,
        ):
            M_pyr, M_lac = state

            dM_pyr_dt = -(k_pyr_lac + r_pyr) * M_pyr
            dM_lac_dt = -r_lac * M_lac + k_pyr_lac * M_pyr

            return [dM_pyr_dt, dM_lac_dt]

        # Define ODE integration
        def kinetics(t, k_pyr_lac, r_pyr, r_lac, M_pyr_0, M_lac_0):
            P = (k_pyr_lac, r_pyr, r_lac, M_pyr_0, M_lac_0)

            return np.hstack(
                [
                    odeint(
                        ODE, [M_pyr_0, M_lac_0], t[: t.shape[0] // 2], P, tfirst=True
                    )[:, 0],
                    odeint(
                        ODE, [M_pyr_0, M_lac_0], t[t.shape[0] // 2 :], P, tfirst=True
                    )[:, 1],
                ]
            )

        # Apply least squares fit of the measured data to the integrated ODE
        P, P_covariance = curve_fit(
            kinetics,
            np.hstack([t_trunc, t_trunc]),
            np.hstack([M_pyr_norm_trunc, M_lac_norm_trunc]),
            bounds=(lower_coeff_bounds, upper_coeff_bounds),
        )

        P_errors = np.sqrt(np.diag(P_covariance))

        # Assign parameter names to fit results
        k_pyr_lac = P[0]
        k_pyr_lac_err = P_errors[0]
        k_lac_pyr = 0
        k_lac_pyr_err = 0

        r_pyr = P[1]
        r_pyr_err = P_errors[1]
        r_lac = P[2]
        r_lac_err = P_errors[2]

        T_1_pyr = 1 / (P[1] + np.log(np.cos(flipangle * np.pi / 180)) / t_repetition)
        T_1_pyr_err = P_errors[1] / P[1] * T_1_pyr
        T_1_lac = 1 / (P[2] + np.log(np.cos(flipangle * np.pi / 180)) / t_repetition)
        T_1_lac_err = P_errors[2] / P[2] * T_1_lac

        M_pyr_0 = P[3]
        M_pyr_0_err = P_errors[3]
        M_lac_0 = P[4]
        M_lac_0_err = P_errors[4]

    # Summarize fit results in dictionary
    results_dict = {
        "sample_ID": [sample_ID],
        "t_min_modeling [s]": [t_min_modeling],
        "k_pyr_lac [1/s]": [k_pyr_lac],
        "k_pyr_lac_err [1/s]": [k_pyr_lac_err],
        "k_lac_pyr [1/s]": [k_lac_pyr],
        "k_lac_pyr_err [1/s]": [k_lac_pyr_err],
        "r_pyr [1/s]": [r_pyr],
        "r_pyr_err [1/s]": [r_pyr_err],
        "r_lac [1/s]": [r_lac],
        "r_lac_err [1/s]": [r_lac_err],
        "T_1_pyr [s]": [T_1_pyr],
        "T_1_pyr_err [s]": [T_1_pyr_err],
        "T_1_lac [s]": [T_1_lac],
        "T_1_lac_err [s]": [T_1_lac_err],
        "M_pyr_0 [a. u.]": [M_pyr_0],
        "M_pyr_0_err [a. u.]": [M_pyr_0_err],
        "M_lac_0 [a. u.]": [M_lac_0],
        "M_lac_0_err [a. u.]": [M_lac_0_err],
    }

    # Display fit results in pandas dataframe
    # results_dataframe = pd.DataFrame(results_dict)
    # display(results_dataframe)

    # Append fit results to Excel sheet
    if not os.path.exists(os.path.join(basic_path, excelsheet_name)):
        results_dataframe_row = pd.DataFrame(
            {
                "sample_ID": [],
                "t_min_modeling [s]": [],
                "k_pyr_lac [1/s]": [],
                "k_pyr_lac_err [1/s]": [],
                "k_lac_pyr [1/s]": [],
                "k_lac_pyr_err [1/s]": [],
                "r_pyr [1/s]": [],
                "r_pyr_err [1/s]": [],
                "r_lac [1/s]": [],
                "r_lac_err [1/s]": [],
                "T_1_pyr [s]": [],
                "T_1_pyr_err [s]": [],
                "T_1_lac [s]": [],
                "T_1_lac_err [s]": [],
                "M_pyr_0 [a. u.]": [],
                "M_pyr_0_err [a. u.]": [],
                "M_lac_0 [a. u.]": [],
                "M_lac_0_err [a. u.]": [],
            }
        )

        results_dataframe_row.to_excel(
            os.path.join(basic_path, excelsheet_name), index=False
        )

    results_dataframe_row = pd.DataFrame(
        {
            "sample_ID": [sample_ID],
            "t_min_modeling [s]": [t_min_modeling],
            "k_pyr_lac [1/s]": [k_pyr_lac],
            "k_pyr_lac_err [1/s]": [k_pyr_lac_err],
            "k_lac_pyr [1/s]": [k_lac_pyr],
            "k_lac_pyr_err [1/s]": [k_lac_pyr_err],
            "r_pyr [1/s]": [r_pyr],
            "r_pyr_err [1/s]": [r_pyr_err],
            "r_lac [1/s]": [r_lac],
            "r_lac_err [1/s]": [r_lac_err],
            "T_1_pyr [s]": [T_1_pyr],
            "T_1_pyr_err [s]": [T_1_pyr_err],
            "T_1_lac [s]": [T_1_lac],
            "T_1_lac_err [s]": [T_1_lac_err],
            "M_pyr_0 [a. u.]": [M_pyr_0],
            "M_pyr_0_err [a. u.]": [M_pyr_0_err],
            "M_lac_0 [a. u.]": [M_lac_0],
            "M_lac_0_err [a. u.]": [M_lac_0_err],
        }
    )

    append_df_to_excel(results_dataframe_row, basic_path, excelsheet_name)

    # Plot the fitting results

    if plot:
        title = r"Kinetic modeling of HP [1-$^{13}$C]pyruvate and [1-$^{13}$C]lactate magnetization"

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=[14, 4.8])

        x11 = t
        x12 = t
        x13 = np.arange(t_trunc[0], t_trunc[-1], 0.01)
        x14 = np.arange(t_trunc[0], t_trunc[-1], 0.01)

        # xmin1 = 0
        # xmax1 = 1

        y11 = M_pyr / np.max(M_pyr)
        y12 = M_lac / np.max(M_pyr) * M_lac_scaling_factor
        y13 = odeint(ODE, [M_pyr_0, M_lac_0], x13, tuple(P), tfirst=True)[:, 0]
        y14 = (
            odeint(ODE, [M_pyr_0, M_lac_0], x14, tuple(P), tfirst=True)[:, 1]
            * M_lac_scaling_factor
        )

        ax1.set_title(title)
        ax1.set_ylabel(r"signal [a. u.]")
        ax1.set_xlabel(r"time [s]")

        ax1.scatter(
            x11,
            y11,
            color="black",
            s=5,
            label=r"HP [1-$^{13}$C]pyruvate, measurement",
            marker="x",
        )
        if M_lac_scaling_factor != 1:
            ax1.scatter(
                x12,
                y12,
                color="red",
                s=5,
                label=r"{} x ".format(M_lac_scaling_factor)
                + r"HP [1-$^{13}$C]lactate, measurement",
                marker="x",
            )
        else:
            ax1.scatter(
                x12,
                y12,
                color="red",
                s=5,
                label=r"HP [1-$^{13}$C]lactate, measurement",
                marker="x",
            )

        ax1.plot(
            x13,
            y13,
            color="black",
            lw=1.5,
            linestyle="-",
            label=r"HP [1-$^{13}$C]pyruvate, " + "model",
        )
        if M_lac_scaling_factor != 1:
            ax1.plot(
                x14,
                y14,
                color="red",
                lw=1.5,
                linestyle="-",
                label=r"{} x ".format(M_lac_scaling_factor)
                + r"HP [1-$^{13}$C]lactate, "
                + "model",
            )
        else:
            ax1.plot(
                x14,
                y14,
                color="red",
                lw=1.5,
                linestyle="-",
                label=r"HP [1-$^{13}$C]lactate, " + "model",
            )

        # ax1.set_xlim(xmin1, xmax1)
        # ax1.set_ylim(ymin1, ymax1)

        # ax1.axvline(0, color='black', linestyle='dashed', linewidth=1)
        # ax1.axhline(0, color='black', linestyle='dashed', linewidth=1)

        ax1.grid()

        leg = ax1.legend(framealpha=1, title=sample_ID)
        # leg = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        leg.get_frame().set_edgecolor("black")

        #######################################################################

        # ax2 = ax1.twinx()
        # ax2.get_xaxis().set_visible(False)
        # ax2.get_yaxis().set_visible(False)

        ax2.set_axis_off()

        handles2, labels2 = ax2.get_legend_handles_labels()

        handles2.append(patches.Patch(color="none", label=""))

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$t_{\mathrm{min, modeling}}$ = "
                + "{:.1f}".format(t_min_modeling)
                + " s",
            )
        )

        handles2.append(patches.Patch(color="none", label=""))

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$k_{\mathrm{PL}}$ = "
                + "({:.4f}".format(k_pyr_lac)
                + r" $\pm$ "
                + "{:.4f}) 1/s".format(k_pyr_lac_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$k_{\mathrm{LP}}$ = "
                + "({:.4f}".format(k_lac_pyr)
                + r" $\pm$ "
                + "{:.4f}) 1/s".format(k_lac_pyr_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$r_{\mathrm{P}}$ = "
                + "({:.3f}".format(r_pyr)
                + r" $\pm$ "
                + "{:.3f}) 1/s".format(r_pyr_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$r_{\mathrm{L}}$ = "
                + "({:.3f}".format(r_lac)
                + r" $\pm$ "
                + "{:.3f}) 1/s".format(r_lac_err),
            )
        )

        handles2.append(patches.Patch(color="none", label=""))

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$T_{\mathrm{1,P}}$ = "
                + "({:.2f}".format(T_1_pyr)
                + r" $\pm$ "
                + "{:.2f}) s".format(T_1_pyr_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$T_{\mathrm{1,L}}$ = "
                + "({:.2f}".format(T_1_lac)
                + r" $\pm$ "
                + "{:.2f}) s".format(T_1_lac_err),
            )
        )

        handles2.append(patches.Patch(color="none", label=""))

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$M_{\mathrm{P,0}}$ = "
                + "({:.2f}".format(M_pyr_0)
                + r" $\pm$ "
                + "{:.2f})".format(M_pyr_0_err),
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$M_{\mathrm{L,0}}$ = "
                + "({:.2f}".format(M_lac_0)
                + r" $\pm$ "
                + "{:.2f})".format(M_lac_0_err),
            )
        )

        leg2 = ax2.legend(
            handles=handles2,
            ncols=1,
            title="Fit parameters, {}".format(fitmodel),
            loc="center left",
        )  # , bbox_to_anchor=(0, 1.5))
        leg2.get_frame().set_edgecolor("black")

        #######################################################################

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                basic_path,
                sample_ID
                + r"_RK4_ODE_integration_lstsqufit_solver_without_input_function_"
                + fitmodel.replace(" ", "_"),
            )
        )

    return results_dict


def kinetic_modeling_of_HP_pyruvate_and_lactate_after_injection_using_pseudoinverse_of_measured_signal(
    t,
    M_pyr,
    M_lac,
    M_lac_scaling_factor,
    flipangle,
    t_repetition,
    t_min_modeling,
    basic_path,
    sample_ID,
    excelsheet_name=r"Overview_kinetic_modeling_fit_parameters.xlsx",
    fitmodel="with reverse conversion",
    plot=True,
):
    """
    Get conversion and relaxation rates of hyperpolarized pyruvate and lactate after hyperpolarized pyruvate injection.

                        forward conversion
                            k_pyr_lac
    M_pyr         <=========================>        M_lac
                            k_lac_pyr
                        reverse conversion
    ||                                               ||
    \\/                                               \\/
    r_pyr                                            r_lac
    = 1/T_1_pyr                                      = 1/T_1_lac

    Linear system of differential equations:

    d/dt   | M_pyr |  =  | -k_pyr_lac-r_pyr        k_lac_pyr    |  °  | M_pyr |
           | M_lac |     |     k_pyr_lac       -k_lac_pyr-r_lac |     | M_lac |

    Transposed linear system of differential equations:

    d/dt   | M_pyr ,  M_lac |  =  | M_pyr ,  M_lac | °  | -k_pyr_lac-r_pyr        k_pyr_lac    |
                                                        |     k_lac_pyr       -k_lac_pyr-r_lac |

    ... with used matrix nomenclature:

              B                =          A          °                     X
    (can be calculated from A)         (known)                  (unknown model parameters)

    Parameters
    ----------
    t : 1D array, float
        sampled time points [s]

    M_pyr : 1D array, float
        time development of magnetization of hyperpolarized 13C pyruvate [a. u.]

    M_lac : 1D array, float
        time development of magnetization of hyperpolarized 13C lactate [a. u.]

    M_lac_scaling_factor : float
        scaling factor for plotting M_lac

    flipangle : float
        applied flip angle of the radiofrequency pulse [°]

    t_repetition : float
        time between two subsequent radiofrequency pulses [s]

    t_min_modeling : float
        starting time for kinetic modeling (larger than injection time) [s]

    basic_path : string
        filepath for saving fit parameter results and fit plot

    sample_ID : string
        characterization of the sample

    excelsheet_name: string
        name of the Excel sheet to which the fit results should be appended

    fitmodel : 'with reverse conversion' or 'without reverse conversion'
        flag for choosing different kinetic models

    plot : boolean
        flag for turning on or off plotting

    Returns
    -------
    fit_results: pandas dataframe

    """

    # Define modeling region
    modeling_region = np.where(t >= t_min_modeling)

    # Truncate arrays to modeling region, normalize pyruvate and lactate data
    t_trunc = t[modeling_region]
    M_pyr_norm_trunc = M_pyr[modeling_region] / np.max(M_pyr)
    M_lac_norm_trunc = M_lac[modeling_region] / np.max(M_pyr)

    # Define matrices of linear system of differential equations
    # (last point of modeling region is lost due to numerical differentiation (forward difference))
    A = np.vstack([M_pyr_norm_trunc[:-1], M_lac_norm_trunc[:-1]]).T

    dt = t_trunc[1:] - t_trunc[:-1]
    dM_pyr_norm_trunc = M_pyr_norm_trunc[1:] - M_pyr_norm_trunc[:-1]
    dM_lac_norm_trunc = M_lac_norm_trunc[1:] - M_lac_norm_trunc[:-1]
    B = np.vstack([dM_pyr_norm_trunc / dt, dM_lac_norm_trunc / dt]).T

    # Solve the linear system of differential equations
    if fitmodel == "with reverse conversion":
        X = pinv(A) @ B
        # print_matrix(X, 'X')

        k_pyr_lac = X[0, 1]
        k_lac_pyr = X[1, 0]
        r_pyr = -X[0, 0]
        r_lac = -X[1, 1]

        T_1_pyr = 1 / (
            r_pyr - k_pyr_lac + np.log(np.cos(flipangle * np.pi / 180)) / t_repetition
        )
        T_1_lac = 1 / (
            r_lac - k_lac_pyr + np.log(np.cos(flipangle * np.pi / 180)) / t_repetition
        )

        t_model = np.linspace(t_trunc[0], t_trunc[-1], 10000)
        dt_model = t_model[1] - t_model[0]

        A_model = np.zeros((10000, 2))
        A_model[0, 0] = M_pyr_norm_trunc[0]
        A_model[0, 1] = M_lac_norm_trunc[0]

        for i, ti in enumerate(t_model[:-1]):
            A_model[i + 1] = A_model[i] + (A_model[i] @ X) * dt_model

        # Remove normalization of pyruvate and lactate data
        M_pyr_model = A_model[:, 0]  # * np.max(M_pyr)
        M_lac_model = A_model[:, 1]  # * np.max(M_pyr)

    elif fitmodel == "without reverse conversion":
        X = pinv(A) @ B

        def exp_decay(t, M_0, r):
            return M_0 * np.exp(-r * t)

        M_pyr_exp_decay_coeff, M_pyr_exp_decay_coeff_cov = curve_fit(
            exp_decay, t_trunc, M_pyr_norm_trunc, bounds=([0, 1e-3], [1e2, 1])
        )

        X[0, 0] = -M_pyr_exp_decay_coeff[1]
        X[1, 0] = 0
        # print_matrix(X, 'X')

        k_pyr_lac = X[0, 1]
        k_lac_pyr = X[1, 0]
        r_pyr = -X[0, 0]
        r_lac = -X[1, 1]

        T_1_pyr = 1 / (
            r_pyr - k_pyr_lac + np.log(np.cos(flipangle * np.pi / 180)) / t_repetition
        )
        T_1_lac = 1 / (
            r_lac - k_lac_pyr + np.log(np.cos(flipangle * np.pi / 180)) / t_repetition
        )

        t_model = np.linspace(t_trunc[0], t_trunc[-1], 10000)
        dt_model = t_model[1] - t_model[0]

        M_pyr_model = exp_decay(
            t_model, M_pyr_exp_decay_coeff[0], M_pyr_exp_decay_coeff[1]
        )

        A_model = np.zeros((10000, 2))
        A_model[:, 0] = M_pyr_model
        A_model[0, 1] = M_lac_norm_trunc[0]

        for i, ti in enumerate(t_model[:-1]):
            A_model[i + 1, 1] = A_model[i, 1] + (A_model[i] @ X)[1] * dt_model

        # Remove normalization of pyruvate and lactate data
        # M_pyr_model*= np.max(M_pyr)
        M_lac_model = A_model[:, 1]  # * np.max(M_pyr)

    # Summarize fit results in dictionary
    results_dict = {
        "sample_ID": [sample_ID],
        "t_min_modeling [s]": [t_min_modeling],
        "k_pyr_lac [1/s]": [k_pyr_lac],
        "k_lac_pyr [1/s]": [k_lac_pyr],
        "r_pyr [1/s]": [r_pyr],
        "r_lac [1/s]": [r_lac],
        "T_1_pyr [s]": [T_1_pyr],
        "T_1_lac [s]": [T_1_lac],
    }

    # Display fit results in pandas dataframe
    # results_dataframe = pd.DataFrame(results_dict)
    # display(results_dataframe)

    # Append fit results to Excel sheet
    if not os.path.exists(os.path.join(basic_path, excelsheet_name)):
        results_dataframe_row = pd.DataFrame(
            {
                "sample_ID": [],
                "t_min_modeling [s]": [],
                "k_pyr_lac [1/s]": [],
                "k_lac_pyr [1/s]": [],
                "r_pyr [1/s]": [],
                "r_lac [1/s]": [],
                "T_1_pyr [s]": [],
                "T_1_lac [s]": [],
            }
        )

        results_dataframe_row.to_excel(
            os.path.join(basic_path, excelsheet_name), index=False
        )

    results_dataframe_row = pd.DataFrame(
        {
            "sample_ID": [sample_ID],
            "t_min_modeling [s]": [t_min_modeling],
            "k_pyr_lac [1/s]": [k_pyr_lac],
            "k_lac_pyr [1/s]": [k_lac_pyr],
            "r_pyr [1/s]": [r_pyr],
            "r_lac [1/s]": [r_lac],
            "T_1_pyr [s]": [T_1_pyr],
            "T_1_lac [s]": [T_1_lac],
        }
    )

    append_df_to_excel(results_dataframe_row, basic_path, excelsheet_name)

    # Plot the fitting results

    if plot:
        title = r"Kinetic modeling of HP [1-$^{13}$C]pyruvate and [1-$^{13}$C]lactate magnetization"

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=[14, 4.8])

        x11 = t
        x12 = t
        x13 = t_model
        x14 = t_model

        # xmin1 = 0
        # xmax1 = 1

        y11 = M_pyr / np.max(M_pyr)
        y12 = M_lac / np.max(M_pyr) * M_lac_scaling_factor
        y13 = M_pyr_model
        y14 = M_lac_model * M_lac_scaling_factor

        # ymin1 = 0
        # ymax1 = 1

        ax1.set_title(title)
        ax1.set_ylabel(r"signal [a. u.]")
        ax1.set_xlabel(r"time [s]")

        ax1.scatter(
            x11,
            y11,
            color="black",
            s=5,
            label=r"HP [1-$^{13}$C]pyruvate, measurement",
            marker="x",
        )
        ax1.scatter(
            x12,
            y12,
            color="red",
            s=5,
            label=r"{} x ".format(M_lac_scaling_factor)
            + r"HP [1-$^{13}$C]lactate, measurement",
            marker="x",
        )

        ax1.plot(
            x13,
            y13,
            color="black",
            lw=1.5,
            linestyle="-",
            label=r"HP [1-$^{13}$C]pyruvate, " + "model",
        )
        ax1.plot(
            x14,
            y14,
            color="red",
            lw=1.5,
            linestyle="-",
            label=r"{} x ".format(M_lac_scaling_factor)
            + r"HP [1-$^{13}$C]lactate, "
            + "model",
        )

        # ax1.set_xlim(xmin1, xmax1)
        # ax1.set_ylim(ymin1, ymax1)

        # ax1.axvline(0, color='black', linestyle='dashed', linewidth=1)
        # ax1.axhline(0, color='black', linestyle='dashed', linewidth=1)

        ax1.grid()

        leg = ax1.legend(framealpha=1, title=sample_ID)
        # leg = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        leg.get_frame().set_edgecolor("black")

        #######################################################################

        # ax2 = ax1.twinx()
        # ax2.get_xaxis().set_visible(False)
        # ax2.get_yaxis().set_visible(False)

        ax2.set_axis_off()

        handles2, labels2 = ax2.get_legend_handles_labels()

        handles2.append(patches.Patch(color="none", label=""))

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$t_{\mathrm{min, modeling}}$ = "
                + "{:.1f}".format(t_min_modeling)
                + " s",
            )
        )

        handles2.append(patches.Patch(color="none", label=""))

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$k_{\mathrm{PL}}$ = " + "{:.4f}".format(k_pyr_lac) + " 1/s",
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$k_{\mathrm{LP}}$ = " + "{:.4f}".format(k_lac_pyr) + " 1/s",
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$r_{\mathrm{P}}$ = " + "{:.3f}".format(r_pyr) + " 1/s",
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$r_{\mathrm{L}}$ = " + "{:.3f}".format(r_lac) + " 1/s",
            )
        )

        handles2.append(patches.Patch(color="none", label=""))

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$T_{\mathrm{1,P}}$ = " + "{:.2f}".format(T_1_pyr) + " s",
            )
        )

        handles2.append(
            patches.Patch(
                color="none",
                label=r"$T_{\mathrm{1,L}}$ = " + "{:.2f}".format(T_1_lac) + " s",
            )
        )

        leg2 = ax2.legend(
            handles=handles2,
            ncols=1,
            title="Modeling parameters, {}".format(fitmodel),
            loc="center left",
        )  # , bbox_to_anchor=(0, 1.5))
        leg2.get_frame().set_edgecolor("black")

        #######################################################################

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                basic_path,
                sample_ID + r"_pseudoinverse_solver_" + fitmodel.replace(" ", "_"),
            )
        )

    return results_dict
