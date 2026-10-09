import sys

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import Bounds, least_squares
from tqdm.auto import tqdm

from ..utils.system.logging import init_default_logger

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")

try:
    pass
except ModuleNotFoundError:
    # logger.info("UMPIRE package not found.")
    pass


LAMOR_FREQUENCY_H1 = 300.130
LAMOR_FREQUENCY_C13 = 75.468

np.set_printoptions(threshold=sys.maxsize)


class MEpHISDO_Solver:

    def __init__(
        self,
        max_num_compartments=3,
        signal_data=None,
        echo_timings=None,
        uMGE_object=None,
    ):

        # load data
        self.__load_data(signal_data, uMGE_object, echo_timings)

        # initialize number of expected frequencies
        self.max_num_compartments = max_num_compartments

        if self.__sanityCheck():
            self.num_Echoes = self.signal_data.shape[0]

    def __sanityCheck(self):

        if self.signal_data.shape[0] != self.echo_timings.size:
            raise Exception(
                "Given number of echo timings "
                + str(self.echo_timings.size)
                + " does not match data size of "
                + str(self.signal_data.shape[0])
            )
        elif self.signal_data.shape[0] <= self.max_num_compartments * 1.5 + 2 + 1:
            raise Exception(
                str(self.signal_data.shape[0])
                + " echoes are not enough to solve for "
                + str(self.max_num_compartments)
                + " frequencies. At least "
                + str(self.max_num_compartments * 1.5 + 2 + 2)
                + " echoes are required"
            )
        elif self.signal_data.ndim != 4:
            raise Exception(
                "Input signal data does not match the required format. It has to be in the format of echo, read, phase, slice"
            )
        else:
            return True

    def __load_data(self, signal_data, uMGE_object, echo_timings):
        if isinstance(signal_data, type(None)) and not isinstance(
            uMGE_object, type(None)
        ):
            self.signal_data = uMGE_object.data
            self.uMGE = uMGE_object
            self.echo_timings = uMGE_object.TEs / 1000.0
            Warning("Loaded data from uMGE object!")

        elif isinstance(signal_data, type(None)) and isinstance(
            uMGE_object, type(None)
        ):
            raise Exception("No data provided!")
        else:
            if isinstance(echo_timings, type(None)):
                raise Exception("No echo time AND no uMGE object provided!")
            else:
                self.signal_data = signal_data
                self.echo_timings = echo_timings
                Warning("Loaded separate data!")
                self.uMGE = None

    def __check_inits_and_bounds(self):

        if len(self.frequ_inits) != len(self.frequ_boundaries.ub):
            raise Exception(
                "The number of frequency inits ("
                + str(len(self.frequ_inits))
                + ") does not match the number of frequency boundaries ("
                + str(len(self.frequ_boundaries.ub))
                + ")!"
            )

        elif len(self.amplitude_inits) * 2 != len(self.amplitude_boundaries.ub):
            raise Exception(
                "The number of amplitude inits ("
                + str(len(self.amplitude_inits))
                + ") does not match the number of amplitude boundaries ("
                + str(len(self.amplitude_boundaries.ub))
                + ")!"
            )

        elif len(self.t2_star_boundary.ub) != 1:
            raise Exception(
                "The number of T2* inits ("
                + str(len(self.t2_star_init))
                + ") does not match the number of T2* boundaries ("
                + str(len(self.t2_star_boundary.ub))
                + ")!"
            )

        if len(self.frequ_inits) != self.max_num_compartments:
            raise Exception(
                "The number of "
                + str(len(self.frequ_inits))
                + "frequency inits are not sufficient to solve for "
                + str(self.max_num_compartments)
                + " compartments!"
            )

        elif self.c1_contrib != -np.inf:
            if len(self.frequ_inits) != len(self.amplitude_inits) + 1:
                raise Exception(
                    "The number of "
                    + str(len(self.frequ_inits))
                    + "frequency inits does not match the number of "
                    + str(len(self.amplitude_inits))
                    + " amplitude inits if C1-contribution should be considererd!"
                )
            else:
                return True
        else:
            if len(self.frequ_inits) != len(self.amplitude_inits):
                raise Exception(
                    "The number of "
                    + str(len(self.frequ_inits))
                    + "frequency inits does not match the number of "
                    + str(len(self.amplitude_inits))
                    + " amplitude inits!"
                )
            else:
                return True

    def signal_model(self, x, num_compartments):
        """
        Method defining the signal model, which was adapted from "Multicoil Dixon Chemical Species Separation With an
        Iterative Least-Squares Estimation Method" by Scott B. Reeder et al.
        The singal model is as follows: s(t) = sum(complex_ampl*e^(i*2*pi*f_m*t)) * e^(-t/T2*)
        To make it processable by least_squares() the real and imaginary signal components are separated.

        Parameters
        ----------
        x : np.ndarray
            One-dimensional array of shape (n,) containing parameter estimates.

        num_compartments : int
            Number of expected pH-compartments in a specific voxel

        Returns
        ----------
        model : np.ndarray
            Modelled signal of all echoes based on estimated parameters x.
            Real and imaginary signal components are separated for each echo.
            Layout: [echo1_real, echo1_imag, echo2_real...]
        """

        model = np.zeros(self.echo_timings.size * 2)

        for idx_t, t in enumerate(self.echo_timings):

            for metab_idx in range(0, num_compartments):
                f_idx = num_compartments * 2 + metab_idx

                model[idx_t * 2] = model[idx_t * 2] + (
                    x[metab_idx * 2] * np.cos(2 * np.pi * x[f_idx] * t)
                    - x[metab_idx * 2 + 1] * np.sin(2 * np.pi * x[f_idx] * t)
                )
                model[idx_t * 2 + 1] = model[idx_t * 2 + 1] + (
                    x[metab_idx * 2] * np.sin(2 * np.pi * x[f_idx] * t)
                    + x[metab_idx * 2 + 1] * np.cos(2 * np.pi * x[f_idx] * t)
                )

            if self.c1_contrib != -np.inf:
                model[idx_t * 2] = model[idx_t * 2] + (
                    x[-2] * np.cos(2 * np.pi * self.c1_contrib * t)
                    - x[-1] * np.sin(2 * np.pi * self.c1_contrib * t)
                )
                model[idx_t * 2 + 1] = model[idx_t * 2 + 1] + (
                    x[-2] * np.sin(2 * np.pi * self.c1_contrib * t)
                    + x[-1] * np.cos(2 * np.pi * self.c1_contrib * t)
                )

            model[idx_t * 2] = model[idx_t * 2] * np.exp(-t / x[num_compartments * 3])
            model[idx_t * 2 + 1] = model[idx_t * 2 + 1] * np.exp(
                -t / x[num_compartments * 3]
            )

        return model

    def fun(self, x, num_compartments, y):
        """
        Function defining the difference between measured and fitted echo data.
        The function is used by least_squares() for optimization.
        To make it processable by least_squares() the real and imaginary signal components are separated.

        Parameters
        ----------
        x : np.ndarray
            One-dimensional array of shape (n,) containing parameter estimates.

        num_compartments : int
            Number of expected pH-compartments in a specific voxel

        y : np.ndarray
            One-dimensional array of shape (num_echoes*2,) containing signal values of all echoes in one voxel devided into real and imaginary component.

        Returns
        ----------
        np.ndarray
           One-dimensional array of shape (m,).
           Contains the difference between measured and fitted echo data.
        """

        return self.signal_model(x, num_compartments) - y

    def jac(self, x, num_compartments, y):
        """
        Method of computing the Jacobian matrix of the signal model(an m-by-n matrix, where element (i, j) is the partial derivative of f[i] with respect to x[j]).
        size of m --> number of echoes*2
        size of n --> number of parameters (3*num_compartments + 1)
        The signal model was adapted from "Multicoil Dixon Chemical Species Separation With an Iterative Least-Squares Estimation Method" by Scott B. Reeder et al.
        The Jacobian is automatically scaled with the given number of pH-compartments.
        The function is used by least_squares() for the optimization.

        Parameters
        ----------
        x : np.ndarray
            One-dimensional array of shape (n,) containing parameter estimates.

        num_compartments : int
            Number of expected pH-compartments in a specific voxel

        y : np.ndarray
            One-dimensional array of shape (num_echoes*2,) containing signal values of all echoes in one voxel devided into real and imaginary component.

        Returns
        ----------
        J : np.ndarray
            Jacobian matrix of size m-by-n.
            Has the following column layout [real_ampl1, imag_ampl1, real_ampl2, ... ,f1, f2, ..., T2*]
            Has the following row layout [echo1_real, echo1_imag, echo2_real...]
        """

        J = np.zeros((self.echo_timings.size * 2, x.size))

        for idx_t, t in enumerate(self.echo_timings):

            for metab_idx in range(0, num_compartments):

                f_idx = num_compartments * 2 + metab_idx

                # partial derivative with respect to real amplitude
                J[idx_t * 2, metab_idx * 2] = np.cos(2 * np.pi * x[f_idx] * t) * np.exp(
                    -t / x[num_compartments * 3]
                )
                J[idx_t * 2 + 1, metab_idx * 2] = np.sin(
                    2 * np.pi * x[f_idx] * t
                ) * np.exp(-t / x[num_compartments * 3])

                # partial derivative with respect to imaginary amplitude
                J[idx_t * 2, metab_idx * 2 + 1] = -np.sin(
                    2 * np.pi * x[f_idx] * t
                ) * np.exp(-t / x[num_compartments * 3])
                J[idx_t * 2 + 1, metab_idx * 2 + 1] = np.cos(
                    2 * np.pi * x[f_idx] * t
                ) * np.exp(-t / x[num_compartments * 3])

                # partial derivative with respect to frequency
                J[idx_t * 2, f_idx] = (
                    2
                    * np.pi
                    * t
                    * (
                        -x[metab_idx * 2] * np.sin(2 * np.pi * x[f_idx] * t)
                        - x[metab_idx * 2 + 1] * np.cos(2 * np.pi * x[f_idx] * t)
                    )
                    * np.exp(-t / x[num_compartments * 3])
                )
                J[idx_t * 2 + 1, f_idx] = (
                    2
                    * np.pi
                    * t
                    * (
                        x[metab_idx * 2] * np.cos(2 * np.pi * x[f_idx] * t)
                        - x[metab_idx * 2 + 1] * np.sin(2 * np.pi * x[f_idx] * t)
                    )
                    * np.exp(-t / x[num_compartments * 3])
                )

                # partial derivative with respect to T2*
                J[idx_t * 2, num_compartments * 3] = J[
                    idx_t * 2, num_compartments * 3
                ] + (
                    x[metab_idx * 2] * np.cos(2 * np.pi * x[f_idx] * t)
                    - x[metab_idx * 2 + 1] * np.sin(2 * np.pi * x[f_idx] * t)
                )
                J[idx_t * 2 + 1, num_compartments * 3] = J[
                    idx_t * 2 + 1, num_compartments * 3
                ] + (
                    x[metab_idx * 2] * np.sin(2 * np.pi * x[f_idx] * t)
                    + x[metab_idx * 2 + 1] * np.cos(2 * np.pi * x[f_idx] * t)
                )

            if self.c1_contrib != -np.inf:
                # for partial derivative with respect to T2*
                J[idx_t * 2, num_compartments * 3] = J[
                    idx_t * 2, num_compartments * 3
                ] + (
                    x[-2] * np.cos(2 * np.pi * self.c1_contrib * t)
                    - x[-1] * np.sin(2 * np.pi * self.c1_contrib * t)
                )
                J[idx_t * 2 + 1, num_compartments * 3] = J[
                    idx_t * 2 + 1, num_compartments * 3
                ] + (
                    x[-2] * np.sin(2 * np.pi * self.c1_contrib * t)
                    + x[-1] * np.cos(2 * np.pi * self.c1_contrib * t)
                )

                # partial derivative with respect to real c1-amplitude
                J[idx_t * 2, -2] = np.cos(2 * np.pi * self.c1_contrib * t) * np.exp(
                    -t / x[num_compartments * 3]
                )
                J[idx_t * 2 + 1, -2] = np.sin(2 * np.pi * self.c1_contrib * t) * np.exp(
                    -t / x[num_compartments * 3]
                )

                # partial derivative with respect to imaginary c1-amplitude
                J[idx_t * 2, -1] = -np.sin(2 * np.pi * self.c1_contrib * t) * np.exp(
                    -t / x[num_compartments * 3]
                )
                J[idx_t * 2 + 1, -1] = np.cos(2 * np.pi * self.c1_contrib * t) * np.exp(
                    -t / x[num_compartments * 3]
                )

            J[idx_t * 2, num_compartments * 3] = (
                J[idx_t * 2, num_compartments * 3]
                * (t / (x[num_compartments * 3] ** 2))
                * np.exp(-t / x[num_compartments * 3])
            )
            J[idx_t * 2 + 1, num_compartments * 3] = (
                J[idx_t * 2 + 1, num_compartments * 3]
                * (t / (x[num_compartments * 3] ** 2))
                * np.exp(-t / x[num_compartments * 3])
            )

        return J

    def find_best_fit(
        self,
        solution_array,
        threshold_ampl_ratio=0.1,
        threshold_frequency_proximity=10,
        threshold_frequency_opposing_sign=5,
    ):
        """
        Function determines the best fit to the signal data from all fits with different number of compartments.
        The fit results are proritized by the lowest fit loss.
        Fit results which reulted in frequency values closer than a threshold or
        an amplitude ratio of the highest and lowest amplitude smaller than a threshold
        or two similar frequencies (below threshold_frequency_opposing_sign) with opposing sign are disregarded.

        Parameters
        ----------
        solution_array : np.ndarray
            Array containing fit results using all possible compartment sizes.
            The array must have the shape of (max_num_compartments, fit_results)

        threshold_ampl_ratio : float
            If the ratio between the highest and lowest fitted amplitude lies under this threshold, the solution is being disregarded.
            Compartments with too low amplitudes cannot be fitted reliably and therefore are being disregarded.

        threshold_frequency_proximity : float
            If two fitted frequency values are closer than this threshold, the solution is being disregarded.
            Too narrow frequency values cannot be distinguished reliably and therefore are being disregarded.

        threshold_frequency_opposing_sign : float
            If two fitted frequency values have opposing sign and have similar frequency values (closer than this threshold), the solution is being disregarded.
            This prevents fits where two similar frequencies with opposing signs are fitted instead of a frequency close to zero.

        Returns
        ----------
        compartment_index : int
           Index of the most promising solution in the solution_array.
           If all soutions were discarded because of the restrictions, this value is None
        """
        sorted_indexes = np.argsort(solution_array[:, -1])

        for index in sorted_indexes:
            amplitude_fits = solution_array[
                index, self.max_num_compartments : 2 * self.max_num_compartments
            ]
            frequency_fits = solution_array[index, : self.max_num_compartments]

            # calculate amplitude ratio of largest and smallest compartment
            ratio = np.nanmin(np.abs(amplitude_fits)) / np.nanmax(
                np.abs(amplitude_fits)
            )

            # second validity check
            if (
                ratio >= threshold_ampl_ratio
                and not self.check_frequency_proximity(
                    frequency_fits, threshold_frequency_proximity
                )
                and not self.check_opposing_frequencies(
                    frequency_fits, threshold_frequency_opposing_sign
                )
            ):
                return index

        return None

    def find_inits(self, num_comps):
        """
        Function returns init and boundary arrays for the least squares fit based on the init and boundary value input to the solve function.
        For best perfomance, the init and boundary values should be sorted in order from most to least expected frequency compartment.

        Parameters
        ----------
        num_comps : int
            Number of compartments to be fitted

        Returns
        ----------
        inits : np.array
           Init values for the fit having the following order: amplitude, frequency, t2*, c1-amplitude (optional)

        bounds : scipy.optimize.Bounds
            Boundary values for the fit having the following order: amplitude, frequency, t2*, c1-amplitude (optional)
        """

        num_total_inits = len(self.amplitude_inits)

        # separate amplitude inits into real and imaginary part
        ampl_inits = np.empty((num_total_inits * 2))
        for idx, init in enumerate(self.amplitude_inits):
            ampl_inits[idx * 2] = np.real(init)
            ampl_inits[idx * 2 + 1] = np.imag(init)

        if self.c1_contrib != -np.inf:

            # return init values and boundaries in following order: amplitude, frequency, t2*, c1-amplitude (optional)
            return (
                np.append(
                    np.append(
                        np.append(
                            np.append(
                                ampl_inits[:num_comps],
                                ampl_inits[
                                    num_total_inits : num_total_inits + num_comps
                                ],
                            ),
                            ampl_inits[-2:],
                        ),
                        self.frequ_inits[:num_comps],
                    ),
                    self.t2_star_init,
                ),
                Bounds(
                    np.append(
                        np.append(
                            np.append(
                                np.append(
                                    self.amplitude_boundaries.lb[:num_comps],
                                    self.amplitude_boundaries.lb[
                                        num_total_inits : num_total_inits + num_comps
                                    ],
                                ),
                                self.amplitude_boundaries.lb[-2:],
                            ),
                            self.frequ_boundaries.lb[:num_comps],
                        ),
                        self.t2_star_boundary.lb,
                    ),
                    np.append(
                        np.append(
                            np.append(
                                np.append(
                                    self.amplitude_boundaries.ub[:num_comps],
                                    self.amplitude_boundaries.ub[
                                        num_total_inits : num_total_inits + num_comps
                                    ],
                                ),
                                self.amplitude_boundaries.ub[-2:],
                            ),
                            self.frequ_boundaries.ub[:num_comps],
                        ),
                        self.t2_star_boundary.ub,
                    ),
                ),
            )
        else:
            return (
                np.append(
                    np.append(
                        np.append(
                            ampl_inits[:num_comps],
                            ampl_inits[num_total_inits : num_total_inits + num_comps],
                        ),
                        self.frequ_inits[:num_comps],
                    ),
                    self.t2_star_init,
                ),
                Bounds(
                    np.append(
                        np.append(
                            np.append(
                                self.amplitude_boundaries.lb[:num_comps],
                                self.amplitude_boundaries.lb[
                                    num_total_inits : num_total_inits + num_comps
                                ],
                            ),
                            self.frequ_boundaries.lb[:num_comps],
                        ),
                        self.t2_star_boundary.lb,
                    ),
                    np.append(
                        np.append(
                            np.append(
                                self.amplitude_boundaries.ub[:num_comps],
                                self.amplitude_boundaries.ub[
                                    num_total_inits : num_total_inits + num_comps
                                ],
                            ),
                            self.frequ_boundaries.ub[:num_comps],
                        ),
                        self.t2_star_boundary.ub,
                    ),
                ),
            )

    def determine_most_promising_frequ_init(
        self, num_comps, fitted_amplitudes, fitted_frequencies
    ):
        """
        Function returns most promising frequency and amplitude fit values based on the fitted amplitudes for initialisation of the next fit.
        It sorts the fit results by absolute amplitude and returns the fitted amplitude and frequency values as init for the next fit with less compartments.
        The boundary values are just reduced by size and not sorted.

        Parameters
        ----------
        num_comps : int
            Number of compartments expected at next fit.

        fitted_amplitudes : np.ndarray (complex)
            Complex amplitude of each pH compartment fitted by MEpHISDO.

        fitted_frequencies : np.ndarray
            Frequency of each pH compartment fitted by MEpHISDO.

        Returns
        ----------
        inits : np.array
           Init values for the fit having the following order: amplitude, frequency, t2*

        bounds : scipy.optimize.Bounds
            Boundary values for the fit having the following order: amplitude, frequency, t2*
        """
        # try:
        num_total_ampl = len(fitted_amplitudes)

        if np.isnan(np.sum(fitted_amplitudes)):
            return self.find_inits(num_comps)
        else:
            # sort by absolute amplitudes in descending order
            sorted_indices_ampl = np.argsort(-np.abs(fitted_amplitudes))

            x0 = np.append(
                np.append(
                    np.append(
                        fitted_amplitudes[sorted_indices_ampl].real[:num_comps],
                        fitted_amplitudes[sorted_indices_ampl].imag[:num_comps],
                    ),
                    fitted_frequencies[sorted_indices_ampl][:num_comps],
                ),
                self.t2_star_init,
            )
            bounds = Bounds(
                np.append(
                    np.append(
                        np.append(
                            self.amplitude_boundaries.lb[:num_comps],
                            self.amplitude_boundaries.lb[
                                num_total_ampl : num_total_ampl + num_comps
                            ],
                        ),
                        self.frequ_boundaries.lb[:num_comps],
                    ),
                    self.t2_star_boundary.lb,
                ),
                np.append(
                    np.append(
                        np.append(
                            self.amplitude_boundaries.ub[:num_comps],
                            self.amplitude_boundaries.ub[
                                num_total_ampl : num_total_ampl + num_comps
                            ],
                        ),
                        self.frequ_boundaries.ub[:num_comps],
                    ),
                    self.t2_star_boundary.ub,
                ),
            )

        return x0.real, bounds
        # except:
        #    print(sorted_indices_ampl)
        #    print(fitted_amplitudes)

    def check_frequency_proximity(self, frequencies, threshold):
        return any(
            abs(frequencies[i] - frequencies[j]) < threshold
            for i in range(len(frequencies))
            for j in range(i + 1, len(frequencies))
        )

    def check_opposing_frequencies(self, frequencies, threshold):
        return any(
            abs(frequencies[i] - (frequencies[j] * -1)) < threshold
            for i in range(len(frequencies))
            for j in range(i + 1, len(frequencies))
        )

    def solve_one_voxel(
        self, signal_data, num_compartments, inits, boundaries, **least_squares_kwargs
    ):
        """
        Fits signal model to echo data of one voxel using scipy.optimize.least_squares().
        The signal model was adapted from "Multicoil Dixon Chemical Species Separation With an Iterative Least-Squares Estimation Method" by Scott B. Reeder et al.
        The singal model is as follows: s(t) = sum(complex_ampl*e^(i*2*pi*f_m*t)) * e^(-t/T2*).
        The optimization allows the definition of boundary conditions that can be chosen to meet the expected frequency values, signal amplitudes and T2* of Z-OMPD.

        Parameters
        ----------
        signal_data : np.ndarray
            One-dimensional array containing complex echo signal data recorded in one voxel.

        num_compartments : int
            Number of expected pH-compartments in a specific voxel

        inits : np.ndarray
            One-dimensional array of shape (num_compartments*3 + 1,) containing parameter estimates for amplitude frequency and T2*.
            Estimates must be provided in the following order: [ampl1_real, ampl1_imag, ampl2_real, ... , f1, f2, ... , T2*]

        boundaries : Bound_class or np.ndarray
            Here, boundary conditions for the estimated parameters can be provided.
            It can be either:
                1. An instance of Bounds class
                2. Lower and upper bounds on estimated parameters. Each array must match the size of "inits" or be a scalar, in the latter case a bound will be the same for all variables.
            Boundaries must be provided in the same order as "inits": [ampl1_real, ampl1_imag, ampl2_real, ... , f1, f2, ... , T2*]
            Use np.inf with an appropriate sign to disable bounds on all or some variables.

        **least_squares_kwargs : dictionary
            Can be used to overwrite the default arguments of least_squares() defined in this function.

        Returns
        ----------
        solution_array : np.ndarray (complex)
            Contains final estimates for frequencies, amplitudes and T2* fit, sorted by the size of the estimated frequencies.
            Also contains the optimality of the fit.
            Has the following layout [f1, f2, ..., fm, ampl1, ampl2, ..., amplm, T2*, optimality]
        """

        # devide signal data in real and imaginary component to make it processable with least_squares
        y = np.empty(signal_data.size * 2)

        y[::2] = np.real(signal_data)
        y[1::2] = np.imag(signal_data)

        # optimize
        solution = least_squares(
            self.fun,
            inits,
            jac=self.jac,
            args=(num_compartments, y),
            verbose=0,
            method="trf",
            max_nfev=10000,
            bounds=boundaries,
            x_scale="jac",
            xtol=None,
            ftol=1e-10,
            gtol=1e-10,
            tr_solver="exact",
            loss="soft_l1",
            **least_squares_kwargs,
        )

        if self.c1_contrib != -np.inf:
            expand_idx = 1
            c1_amplitude_estimates = solution.x[-2] + 1j * solution.x[-1]
        else:
            expand_idx = 0

        # postprocess solution
        if solution.success:

            complex_amplitude_estimates = (
                solution.x[: num_compartments * 2 : 2]
                + 1j * solution.x[1 : num_compartments * 2 : 2]
            )

            frequency_estimates = solution.x[
                num_compartments * 2 : num_compartments * 3
            ]

            t2_star_estimate = solution.x[-1 - (expand_idx * 2)]

            # sort by frequency size
            sorted_indices = np.argsort(frequency_estimates)

            complex_amplitude_estimates = np.array(complex_amplitude_estimates)[
                sorted_indices
            ]
            frequency_estimates = np.array(frequency_estimates)[sorted_indices]

            solution_array = frequency_estimates
            solution_array = np.append(solution_array, complex_amplitude_estimates)
            solution_array = np.append(solution_array, t2_star_estimate)

            if self.c1_contrib != -np.inf:
                solution_array = np.append(solution_array, c1_amplitude_estimates)

            solution_array = np.append(solution_array, solution.cost)

            return solution_array

        else:
            return [float("nan") for i in range(num_compartments * 2 + 2 + expand_idx)]

    def solve(
        self,
        frequ_inits=np.array([250.0, 185.0, 80.0]),
        frequ_boundaries=Bounds([-100, -100, -100], [400, 400, 400]),
        amplitude_inits=np.array([40 + 1j, 20 + 1j, 10 + 1j]),
        amplitude_boundaries=Bounds(
            [-200, -200, -200, -200, -200, -200], [200, 200, 200, 200, 200, 200]
        ),
        t2_star_init=0.013,
        t2_star_boundary=Bounds([0.001], [0.1]),
        threshold_ampl_ratio=0.1,
        threshold_frequency_proximity=10,
        optional_signal_data=None,
        save_path=None,
        c1_contrib=-np.inf,
        **least_squares_kwargs,
    ):

        self.frequ_inits = frequ_inits
        self.frequ_boundaries = frequ_boundaries
        self.amplitude_inits = amplitude_inits
        self.amplitude_boundaries = amplitude_boundaries
        self.t2_star_init = t2_star_init
        self.t2_star_boundary = t2_star_boundary
        self.c1_contrib = c1_contrib

        if self.__check_inits_and_bounds():

            # in case optional data is given, reinitialize signal_data
            if isinstance(optional_signal_data, type(None)):
                signal = self.signal_data
            else:
                if self.__sanityCheck(optional_signal_data, self.echo_timings):
                    signal = optional_signal_data
                    Warning("Solve with optional data.")

            # read data dimensions and safe indexes
            _, dim_read, dim_phase, dim_slice = signal.shape
            index_list = list(np.ndindex(dim_read, dim_phase, dim_slice))
            index_list = tqdm(index_list, desc="Calculating Voxels", leave=True)

            # if UMPIRE object is provided, get SNR mask
            if isinstance(self.uMGE, type(None)):
                snr_mask = np.full((dim_read, dim_phase, dim_slice), False)
            elif not isinstance(self.uMGE, type(None)) and "snr" in self.uMGE.mask_dict:
                snr_mask = self.uMGE.mask_dict["snr"]
            else:
                snr_mask = np.full((dim_read, dim_phase, dim_slice), False)

            if self.c1_contrib != -np.inf:
                raise Exception("Fitting for c1 contribution currently not supported")
            else:
                c1 = 0

            # dummy function for one voxel, that gets serialized
            def calc_voxel(idx, snr_mask):

                # check for for SNR
                if not snr_mask[(..., *idx)]:

                    solution_matrix = np.array([])

                    x0, bounds = self.find_inits(self.max_num_compartments)

                    # start with highest number of expected compartments
                    num_frequencies = self.max_num_compartments

                    # calculate solution for all possible compartment sizes
                    for iterations in range(1, self.max_num_compartments + 1)[::-1]:

                        solution = self.solve_one_voxel(
                            signal[(..., *idx)], num_frequencies, x0, bounds
                        )

                        # update inits and bounds based on fitted values
                        x0, bounds = self.determine_most_promising_frequ_init(
                            num_frequencies - 1,
                            solution[num_frequencies : 2 * num_frequencies],
                            solution[:num_frequencies],
                        )

                        # fill to max_compartments
                        for itr in range(self.max_num_compartments - iterations):
                            solution = np.insert(
                                solution, num_frequencies + itr, float("nan")
                            )
                            solution = np.insert(
                                solution, 2 * num_frequencies + itr + 1, float("nan")
                            )

                        solution_matrix = np.append(solution_matrix, solution)

                        num_frequencies -= 1

                    solution_matrix = solution_matrix.reshape(
                        self.max_num_compartments, -1
                    )

                    solution_matrix = np.flip(solution_matrix, axis=0)

                    # find best fitting solution
                    idx_best_fit = self.find_best_fit(
                        solution_matrix,
                        threshold_ampl_ratio,
                        threshold_frequency_proximity,
                    )

                    if not isinstance(idx_best_fit, type(None)):
                        return solution_matrix[idx_best_fit]
                    else:
                        return [
                            float("nan")
                            for i in range(self.max_num_compartments * 2 + 2 + c1)
                        ]

                else:
                    return [
                        float("nan")
                        for i in range(self.max_num_compartments * 2 + 2 + c1)
                    ]

            # serial solve for mutliple voxels in parallel
            solution_serial = Parallel(n_jobs=-1)(
                delayed(calc_voxel)(idx, snr_mask) for idx in index_list
            )

            # for debugging uncomment
            # for idx in index_list:
            #   solution_serial = calc_voxel(idx, snr_mask)
            # print(solution_serial)

            # save result as np.array and reshape
            solution_matrix = np.array(solution_serial).reshape(
                dim_read, dim_phase, dim_slice, -1
            )

            self.solution_matrix = np.moveaxis(solution_matrix[:, :, :, :], -1, 0)

            if not isinstance(save_path, type(None)):
                np.save(save_path, self.solution_matrix)

            return self.solution_matrix


def create_test_signal(
    phasor_amplitudes, frequencies, echo_timigs, noise_scale=0.0, t2_star_decay=None
):
    """
    Function creates a n signals with given phasor_amplitudes, field_offset, T2_star, theta for initialized
    echo_spacing and metabolite frequencies.

    Parameters
    ----------
    phasor_amplitudes : np.ndarray
        Amplitude of each signal peak contributing to the signal.
        This is what the solver tries to reconstruct from the test signals.
        Array size has to match the number of given frequencies

    frequencies : np.array
        Frequency of each signal peak contributing to the signal.
        This is what the solver tries to reconstruct from the test signals.
        Array size has to match the number of given phasor_amplitudes

    echo_timigs : np.array
        Echo spacing used to generate test signals.

    noise_scale : float, optional
        Standard deviation (spread or “width”) of the normal distributed noise alternating the signals.
        If noise = 0.0 no noise is added.

    t2_star_decay : float, optional
        If not None, the signal also contains T2* contributions.

    Returns
    ----------
    test_signals : np.ndarray
        Array of n test signals with contributions of all pH compartments, field changes and normal distributed noise.
    """

    if frequencies.size != phasor_amplitudes.size:
        print("Size of given frequencies does not match size of given amplitudes.")
    else:

        test_signals = np.zeros(echo_timigs.size, dtype=complex)

        for idx in range(echo_timigs.size):
            if noise_scale != 0.0:
                noise = (
                    np.random.normal(scale=noise_scale)
                    + np.random.normal(scale=noise_scale) * 1j
                )
                print("Noise: " + str(noise))

            for idx_a, amplitude in enumerate(phasor_amplitudes):
                test_signals[idx] = test_signals[idx] + amplitude * np.exp(
                    1j * 2 * np.pi * frequencies[idx_a] * echo_timigs[idx]
                )

            # include T2* signal decay
            if not isinstance(t2_star_decay, type(None)):
                test_signals[idx] = test_signals[idx] * np.exp(
                    -echo_timigs[idx] / t2_star_decay
                )

            # add noise
            if noise_scale != 0.0:
                test_signals[idx] = test_signals[idx] + noise

        # expand dimensions to fit requirement of input
        test_signals = np.expand_dims(test_signals, -1)
        test_signals = np.expand_dims(test_signals, -1)
        test_signals = np.expand_dims(test_signals, -1)

        return test_signals
