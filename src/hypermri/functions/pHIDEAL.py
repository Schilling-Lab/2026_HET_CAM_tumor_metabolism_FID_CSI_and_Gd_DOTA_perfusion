import sys

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..utils.system.logging import init_default_logger

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")

try:
    from umpire import UMPIRE
except ModuleNotFoundError:
    # logger.info("UMPIRE package not found.")
    pass

from ..utils.image.interpolation import (
    downsample_masked_array,
    interpolate_masked_3d_array,
)

LAMOR_FREQUENCY_H1 = 300.130
LAMOR_FREQUENCY_C13 = 75.468

np.set_printoptions(threshold=sys.maxsize)


class pHIDEAL_Solver:

    def __init__(
        self,
        max_num_frequencies=3,
        signal_data=None,
        echo_timings=None,
        uMGE_object=None,
    ):

        # load data
        self.__load_data(signal_data, uMGE_object, echo_timings)

        # initialize number of expected frequencies
        self.max_num_frequencies = max_num_frequencies

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
        elif self.signal_data.shape[0] <= self.max_num_frequencies * 1.5:
            raise Exception(
                str(self.signal_data.shape[0])
                + " echoes are not enough to solve for "
                + str(self.max_num_frequencies)
                + " frequencies. At least "
                + str(self.max_num_frequencies * 1.5 + 1)
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
            raise Exception("no data for IDEAL solver given")
        else:
            if isinstance(echo_timings, type(None)):
                raise Exception("No echo time AND no uMGE object provided!")
            else:
                self.signal_data = signal_data
                self.echo_timings = echo_timings
                Warning("Loaded separate data!")
                self.uMGE = None

    def __check_init_frequency_estimates(self, offsets, use_UMPRIRE_freq_for_init):

        if use_UMPRIRE_freq_for_init and isinstance(self.uMGE, type(None)):
            raise Exception(
                "No uMGE object provided to initialize frequency estimates!"
            )
        elif use_UMPRIRE_freq_for_init:

            # check if uMGE_object conains UMPIRE field map
            if np.count_nonzero(self.uMGE._fmap) == 0:
                raise Exception(
                    "No field map for initialization if the frequency estimates found! Call umpire() and calculate_fmap() functions first."
                )
            else:
                Warning("Taking UMPIRE field map to initialize frequency estimates!")
                return True

        elif isinstance(offsets, type(None)):
            raise Exception(
                "No frequency offsets found and usage of uMGE object disabled. Either a numpy array with initial frequency estimates or a uMGE object with a calculated fieldmap!"
            )

        elif not isinstance(offsets, type(None)):
            if offsets.size < self.max_num_frequencies:
                raise Exception(
                    "Not enough frequency offsets provided to initialize max number of "
                    + str(self.max_num_frequencies)
                    + " frequency estimates!"
                )
            else:
                return True

    def define_S(self, signal_data):
        """
        Function takes singnal data and builds a matrix with it.
        In contrast to the paper "Multicoil Dixon Chemical Species Separation With an Iterative Least-Squares
        Estimation Method" by Scott B. Reeder et al. no corrections will be done

        Parameters
        ----------
        signal_data : np.ndarray
            Array containing measured signal.

        Returns
        ----------
        s_matrix : np.ndarray
            Array of shape number of echoes*2 containing measured signal.
            Signal is split into real signal components (Index 0 to number of echoes-1) and imaginary signal components (Index number of echoes to (number of echoes*2)-1)
        """

        s_matrix = np.zeros((signal_data.shape[0] * 2))

        for idx_s, s in enumerate(signal_data):

            s_matrix[idx_s] = np.real(s)
            s_matrix[idx_s + signal_data.shape[0]] = np.imag(s)

            np.squeeze(s_matrix)

        return s_matrix

    def define_S_corrected(self, signal_data, field_map_value=None, t_2_star=None):
        """
        Function takes singnal data and extracts true signal contributed by metabolites by mutliplying with inverse of measured field_offset and T2*.
        For more information see paper "Multicoil Dixon Chemical Species Separation With an Iterative Least-Squares
        Estimation Method" by Scott B. Reeder et al.

        Parameters
        ----------
        signal_data : np.ndarray
            Array containing measured signal.

        field_map_value : float
            Value of field map at this voxel.

        t_2_star : float
           T2* in this voxel in [s].

        Returns
        ----------
        s_corrected : np.ndarray
            Array of shape number of echoes*2 containing corrected measured signal.
            Signal is split into real signal components (Index 0 to number of echoes-1) and imaginary signal components (Index number of echoes to (number of echoes*2)-1)
        """

        s_corrected = np.zeros((signal_data.shape[0] * 2))

        for idx_s, s in enumerate(signal_data):

            t_n = self.echo_timings[idx_s]

            s_buffer = s

            # field map correction
            # s_buffer = s_buffer * np.exp(-2 * np.pi * field_map_value * t_n * 1j)

            # correction for T2*
            if not isinstance(t_2_star, type(None)):
                s_buffer = s_buffer * np.exp(t_n / t_2_star)
            # else:
            #     s_buffer = s_buffer * np.exp(t_n/0.013)

            s_corrected[idx_s] = np.real(s_buffer)
            s_corrected[idx_s + signal_data.shape[0]] = np.imag(s_buffer)

            np.squeeze(s_corrected)

        return s_corrected

    def define_A(self, frequency_estimate):
        """
        Function defines matrix A that can be used to calculate real and imaginary signal components based on the signal equation.
        For more information see paper "Multicoil Dixon Chemical Species Separation With an Iterative Least-Squares
        Estimation Method" by Scott B. Reeder et al.

         Parameters
        ----------
        frequency_estimate : np.ndarray
            Array containing estimates for the frequencies of each ph compartment.

        Returns
        ----------
        matrix_A : np.ndarray
            Matrix used to calculate real and imaginary signal components.
            Row 0 to num_Echoes-1 is of the form [cos(2*np.pi*f_1*t_n) -sin(2*np.pi*f_1*t_n) ... cos(2*np.pi*f_M*t_n) -sin(2*np.pi*f_M*t_n)], representing real signal component.
            Row num_Echoes to (num_Echoes*2)-1 is of the form [sin(2*np.pi*f_1*t_n) cos(2*np.pi*f_1*t_n) ... sin(2*np.pi*f_M*t_n) cos(2*np.pi*f_M*t_n)], representing imaginary signal component.
        """

        matrix_A = np.zeros((self.num_Echoes * 2, frequency_estimate.size * 2))

        # fill real parts
        for idx_t, t_n in enumerate(self.echo_timings):

            for idx_f, f in enumerate(frequency_estimate):

                c = np.sin(2 * np.pi * f * t_n)
                d = np.cos(2 * np.pi * f * t_n)

                # fill real parts
                matrix_A[idx_t, idx_f * 2] = d
                matrix_A[idx_t, (idx_f * 2) + 1] = -c

                # fill imaginary parts
                matrix_A[idx_t + self.num_Echoes, idx_f * 2] = c
                matrix_A[idx_t + self.num_Echoes, (idx_f * 2) + 1] = d

        return matrix_A

    def define_S_diff(self, signal_matrix, estimated_amplitudes, matrix_A):
        """
        Function defines difference between true signal corrected by estimated T2* and estimated signal contributions of all metabolites.
        For more information see paper "Multicoil Dixon Chemical Species Separation With an Iterative Least-Squares
        Estimation Method" by Scott B. Reeder et al.

        Parameters
        ----------
        signal_matrix : np.ndarray
            Array of measured signal. Can be created by using define_S().
            Signal is split into real signal components (Index 0 to number of echoes-1) and imaginary signal components (Index number of echoes to (number of echoes*2)-1)


        estimated_amplitudes : np.ndarray
            Estimated amplitudes of all metabolites contributing to signal.
            Array size: 2*num_frequencies.
            Layout: [ampl_real_1, ampl_imag_1, ampl_real_2, ampl_imag_2, ... , ampl_real_M, ampl_imag_M]

        matrix_A : np.ndarray
            Matrix used to calculate real and imaginary signal components.
            Row 1 to num_Echoes is of the form [cos(2*np.pi*f_1*t_n) -sin(2*np.pi*f_1*t_n) ... cos(2*np.pi*f_M*t_n) -sin(2*np.pi*f_M*t_n)], representing real signal component.
            Row num_Echoes+1 to num_Echoes*2 is of the form [sin(2*np.pi*f_1*t_n) cos(2*np.pi*f_1*t_n) ... sin(2*np.pi*f_M*t_n) cos(2*np.pi*f_M*t_n)], representing imaginary signal component.
            A can be defined using function define_A().

        Returns
        ----------
        S_diff : np.ndarray
            Array of shape number of echoes*2 containing difference between true and estimated signal.
            Signal is split into real signal components (Index 0 to number of echoes-1) and imaginary signal components (Index number of echoes to (number of echoes*2)-1)
        """

        S_diff = np.zeros(shape=signal_matrix.shape)

        for idx_s, s in enumerate(signal_matrix):
            S_diff[idx_s] = s - (estimated_amplitudes @ matrix_A[idx_s, :])

        return S_diff

    def define_B(
        self, matrix_A, estimated_amplitudes, frequency_estimate, t2_star=None
    ):
        """
        Function defines matrix B containing first order taylor expansion of the signal equation.
        The Taylor expansion was made accoring to the paper "Multicoil Dixon Chemical Species Separation With an Iterative Least-Squares
        Estimation Method" by Scott B. Reeder et al. but altered to solve not for global frequency offset but individual frequency contributions of all expected compartemens.
        B is used to calculate frequency estimates and signal amplitudes of contributing signal compartments.

        Parameters
        ----------
        matrix_A : np.ndarray
            Matrix used to calculate real and imaginary signal components.
            Row 1 to num_Echoes is of the form [cos(2*np.pi*f_1*t_n) -sin(2*np.pi*f_1*t_n) ... cos(2*np.pi*f_M*t_n) -sin(2*np.pi*f_M*t_n)], representing real signal component.
            Row num_Echoes+1 to num_Echoes*2 is of the form [sin(2*np.pi*f_1*t_n) cos(2*np.pi*f_1*t_n) ... sin(2*np.pi*f_M*t_n) cos(2*np.pi*f_M*t_n)], representing imaginary signal component.
            A can be defined using function define_A().

        estimated_amplitudes : np.ndarray
            Estimated amplitudes of all metabolites contributing to signal.
            Array size: 2*num_frequencies.
            Layout: [ampl_real_1, ampl_imag_1, ampl_real_2, ampl_imag_2, ... , ampl_real_M, ampl_imag_M]

        t2_star : float
            Estimated T2* value in [ms]


        Returns
        ----------
        matrix_B : np.ndarray
            Matrix of shape num_Echoes*2 x (3*num_frequencies + 1)
            Contains the firs derivative of the parameters that have to be estimated.
            Layout: [delta f_1, delta f_2 ... delta f_M,  T2*, ampl_real_1, ampl_imag_1, ampl_real_2, ampl_imag_2, ... , ampl_real_M, ampl_imag_M]
            The first num_Echoes rows contain real part of signal equation.
            The last num_Echoes rows contain the imaginary part of the signal equation.
        """

        # taylor aproximation of metabolite contributions
        matrix_B = matrix_A

        # define a new column for each frequency that has to be estimated + one column if T2* is also estimated
        if not isinstance(t2_star, type(None)):
            new_columns = np.zeros((matrix_A.shape[0], frequency_estimate.size + 1))
        else:
            new_columns = np.zeros((matrix_A.shape[0], frequency_estimate.size))

        for idx_t, t_n in enumerate(self.echo_timings):

            for idx_f, f in enumerate(frequency_estimate):

                c = np.cos(2 * np.pi * f * t_n)
                d = np.sin(2 * np.pi * f * t_n)

                # fill real signal contributions
                new_columns[idx_t, idx_f] = (
                    2
                    * np.pi
                    * t_n
                    * (
                        -estimated_amplitudes[idx_f * 2] * d
                        - estimated_amplitudes[idx_f * 2 + 1] * c
                    )
                )

                # fill imaginary signal contributions
                new_columns[idx_t + self.num_Echoes, idx_f] = (
                    2
                    * np.pi
                    * t_n
                    * (
                        estimated_amplitudes[idx_f * 2] * c
                        - estimated_amplitudes[idx_f * 2 + 1] * d
                    )
                )

                # add column for T2*
                if not isinstance(t2_star, type(None)):

                    # fill real signal contributions
                    new_columns[idx_t, frequency_estimate.size] = (
                        new_columns[idx_t, frequency_estimate.size]
                        + estimated_amplitudes[idx_f * 2] * c
                        - estimated_amplitudes[idx_f * 2 + 1] * d
                    )

                    # fill imaginary signal contributions
                    new_columns[
                        idx_t + self.num_Echoes, frequency_estimate.size
                    ] = new_columns[
                        idx_t + self.num_Echoes, frequency_estimate.size
                    ] + (
                        estimated_amplitudes[idx_f * 2] * d
                        + estimated_amplitudes[idx_f * 2 + 1] * c
                    )

            # add column for T2*
            if not isinstance(t2_star, type(None)):

                # fill real signal contributions
                new_columns[idx_t, frequency_estimate.size] = new_columns[
                    idx_t, frequency_estimate.size
                ] * (t_n / (t2_star) ** 2)

                # fill imaginary signal contributions
                new_columns[idx_t + self.num_Echoes, frequency_estimate.size] = (
                    new_columns[idx_t + self.num_Echoes, frequency_estimate.size]
                    * (t_n / (t2_star) ** 2)
                )

        # append columns for frequency estimate to matrix_B
        matrix_B = np.append(new_columns, matrix_B, axis=1)

        return matrix_B

    def solve_one_voxel(
        self,
        signal_data_one_voxel,
        init_frequency_estimates,
        t2_star_init=None,
        stop_t2_star_offset=0.005,
        stop_f_offset=0.5,
        max_iterations=100,
        f_map_value=None,
    ):
        """
        Function solving IDEAL problem for given initial guesses and echo data of one pixel.
        Requires correct initialization of pHIDEAL class.

        Parameters
        ----------
        signal_data_one_voxel : np.ndarray
            One-dimensional array containing complex signal values of all echoes in one voxel.

        init_frequency_estimates : np.ndarray
            Values to initialize frequency values for optimization.
            The size of init_frequency_estimates determines the number of frequencies and amplitudes that will be fitted in this voxel.

        t2_star_init : float, optional
            If not None, the pHIDEAL algorithm also optimizes for T2*

        stop_t2_star_offset, float, optional
            Stop criterion for T2*. Default is 0.005
            Stops iterating if delta of T2* is below the given value and delta of all f is below stop_f_offset.

        stop_f_offset, float, optional
            Stop criterion for all frequency values. Default is 0.5
            Stops iterating if delta of all frequencies being fitted are below the given value and delta of T2* is below the given value (if t2_star_init not None).

        max_iterations : int, optional
            Max number of iterations performed before stop.
            Default is 100.

        f_map_value : float, optional
            If provided, the signal can be corrected by the given value to account for B0-field inhomogeneities.
            Currently disabled. B0-corrections are done by separate function in umpireMGE class.

        Returns
        ----------
        solution : np.ndarray (complex)
            Contains final estimates for frequencies, amplitudes and optionally T2* fit.
            Also contains the number of iterations required for the fit.
            Has the following layout [f1, f2, ..., fm, ampl1, ampl2, ..., amplm, T2*, num_iterations]
        """

        stop_criterion = False
        num_iterations = 1

        estimated_frequencies = init_frequency_estimates

        num_frequencies = estimated_frequencies.size

        estimated_t2_star = t2_star_init

        # correct the signal for given frequency offset measured by field map
        # S = self.define_S(signal_data_one_voxel)#self.define_S_corrected(signal_data_one_voxel, f_map_value)

        try:
            while not stop_criterion:

                # T2* correct the signal if T2* not none
                s_corrected = self.define_S_corrected(
                    signal_data_one_voxel, t_2_star=estimated_t2_star
                )

                A = self.define_A(estimated_frequencies)

                estimated_amplitudes = (
                    np.linalg.matrix_power(np.transpose(A) @ A, -1)
                    @ np.transpose(A)
                    @ s_corrected
                )

                s_diff = self.define_S_diff(s_corrected, estimated_amplitudes, A)

                B = self.define_B(A, estimated_amplitudes, estimated_frequencies)

                y = (
                    np.linalg.matrix_power(np.transpose(B) @ B, -1)
                    @ np.transpose(B)
                    @ s_diff
                )

                # dummy to see if all deltas of the estimated frequencies are below the stop criterion
                dummy_stop = True

                for idx_f, f in enumerate(estimated_frequencies):
                    estimated_frequencies[idx_f] = (
                        estimated_frequencies[idx_f] + y[idx_f]
                    )

                    # check for stop criterion
                    if np.abs(y[idx_f]) > stop_f_offset:
                        dummy_stop = False

                # check T2*
                if not isinstance(t2_star_init, type(None)):

                    estimated_t2_star = (
                        estimated_t2_star + y[estimated_frequencies.size]
                    )

                    if np.abs(y[estimated_frequencies.size]) > stop_t2_star_offset:
                        dummy_stop = False

                stop_criterion = dummy_stop

                # check if stop criterions reached
                if num_iterations == max_iterations:
                    stop_criterion = True

                num_iterations += 1

            # calculate estimated amplitudes for the last time
            A = self.define_A(estimated_frequencies)

            estimated_amplitudes = (
                np.linalg.matrix_power(np.transpose(A) @ A, -1)
                @ np.transpose(A)
                @ s_corrected
            )

            complex_amplitudes = np.zeros(
                (int((len(estimated_amplitudes) / 2)),), dtype=complex
            )

            # make amplitudes complex
            for idx in range(0, len(estimated_amplitudes), 2):
                complex_amplitudes[int(idx / 2)] = (
                    estimated_amplitudes[idx] + 1j * estimated_amplitudes[idx + 1]
                )

            # sort with size
            sorted_indices = np.argsort(estimated_frequencies)

            complex_amplitudes = np.array(complex_amplitudes)[sorted_indices]
            estimated_frequencies = np.array(estimated_frequencies)[sorted_indices]

            solution = estimated_frequencies

            solution = np.append(solution, complex_amplitudes)

            if not isinstance(t2_star_init, type(None)):
                solution = np.append(solution, estimated_t2_star)

            solution = np.append(solution, num_iterations)

            return solution

        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                Warning("Singular matrix occured!")
                if not isinstance(t2_star_init, type(None)):
                    return [float("nan") for i in range(num_frequencies * 2 + 2)]
                else:
                    return [float("nan") for i in range(num_frequencies * 2 + 1)]
            else:
                raise

    def solve(
        self,
        t2_star_init=None,
        optional_signal_data=None,
        stop_f_offset=0.5,
        max_iterations=300,
        field_map=None,
        save_path=None,
        frequ_inits=np.array([-90.7, 16.0, 55.9]),
        use_UMPRIRE_freq_for_init=True,
        discard_similar=True,
        fit_threshold=0.3,
    ):
        """
        Function reconstructs frequencies for given initial guesses and echo data of an entire volume
        by calling solve_one_voxel function for all voxels in the volume.
        Requires correct initialization of pHIDEAL class.

        Parameters
        ----------
        optional_data : np.ndarray, optional
            In case pHIDEAL solver should no perform the calculations on the initial signal_data (e.g on test data),
            other signal points can be passed here.
            The optional data must have the dimensions echo, read, phase, slice.

        stop_f_offset : float, optional
            Stop criterion for field offset estimates of all frequncies. Default is 0.05
            Stops iterating if delta of all field offset is below the given value.

        max_iterations : int, optional
            Max number of iterations performed before stop.
            Default is 100.

        field_map : BrukerExp of field map sequence : optional
            In case prior knowledge of field map should be used as initial values.
            If None, the initial values provided are being used.

        save_path : string, optional
            Can be used to save solution in a file defined by save_path

        frequ_inits : np.ndarray, optional
            Values to initialize frequency values for optimization.
            Must have at least the size of max_num_frequencies

        use_UMPRIRE_freq_for_init : bool, optional
            Enables initialization of frequency values for optimization based on frequency determined using the UMPIRE algorithm.

        t2_star_init : float, optional
            If not None, the pHIDEAL algorithm also optimizes for T2*

        discard_similar : bool, optional
            If True, similar frequency values determined by pHIDEAL will be discarded, such that only one of them is left.

        fit_threshold : float, optional
            Threshold used by check_result function.
            If MSE of fitted unwrapped phase and measured unwrapped phase is smaller than fit_threshold, the fit result will be counted as valid.

        Returns
        ----------
        solution : np.ndarray
            Contains solution for estimated frequencies, all complex amplitudes of phasors of contributing metabolites and optionally T2*.
            Layout: [f_1, f_2 ... f_M, ampl1, ampl2, ..., ampl_M, T2*] for each voxel
        """

        # sanity check frequency estimates
        if self.__check_init_frequency_estimates(
            frequ_inits, use_UMPRIRE_freq_for_init
        ):

            # in case optional data is given, reinitialize signal_data
            if isinstance(optional_signal_data, type(None)):
                signal = self.signal_data
            else:
                if self.__sanityCheck(optional_signal_data, self.echo_timings):
                    signal = optional_signal_data
                    Warning("Solve with optional data.")

            if not isinstance(t2_star_init, type(None)):
                t2 = 1
            else:
                t2 = 0

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

            # dummy function for one voxel, that gets serialized
            def calc_voxel(
                idx, f_map_value, snr_mask, use_UMPRIRE_freq_for_init, frequ_inits
            ):

                # check for for SNR
                if not snr_mask[(..., *idx)]:

                    num_frequencies = self.max_num_frequencies

                    for iterations in range(1, self.max_num_frequencies + 1):
                        if (
                            isinstance(self.uMGE, type(None))
                            or not use_UMPRIRE_freq_for_init
                        ):
                            init_frequency_estimates = self.initialize_frequ_estimates(
                                num_frequencies, frequ_inits=frequ_inits
                            )
                        else:
                            init_frequency_estimates = self.initialize_frequ_estimates(
                                num_frequencies,
                                frequency_value=self.uMGE._fmap[(..., *idx)],
                                frequ_inits=frequ_inits,
                            )

                        solution = self.solve_one_voxel(
                            signal_data_one_voxel=signal[(..., *idx)],
                            init_frequency_estimates=init_frequency_estimates,
                            f_map_value=f_map_value[(..., *idx)],
                            t2_star_init=t2_star_init,
                            stop_f_offset=stop_f_offset,
                            max_iterations=max_iterations,
                        )

                        # get unwrapped phase values
                        unwrapped_phase = np.copy(
                            self.uMGE.unwrapped_phase[(..., *idx)]
                        )

                        # check if one of the solution values lies out of the expected amplitude and frequency range
                        if np.isnan(solution).all() or not self.check_result(
                            solution,
                            num_frequencies,
                            unwrapped_phase,
                            fit_threshold=fit_threshold,
                        ):

                            # if 1 frequency reached and result not good, return nan solution
                            if self.max_num_frequencies - iterations == 0:
                                empty_solution = [
                                    float("nan")
                                    for i in range(
                                        self.max_num_frequencies * 2 + 1 + t2
                                    )
                                ]
                                return empty_solution
                            else:
                                num_frequencies = self.max_num_frequencies - iterations

                                # init_frequency_estimates = self.determine_most_promising_frequ_init(init_frequency_estimates, solution[num_frequencies:2*num_frequencies], solution[:num_frequencies])
                        else:
                            # merge similar frequency values
                            if discard_similar:
                                for idx, is_closein in enumerate(
                                    np.isclose(
                                        solution[0], solution[1:num_frequencies], atol=3
                                    )
                                ):
                                    if is_closein:
                                        Warning(
                                            "Calculated frequency values were close, leading to discarding of one! If this behaviour is unwanted set parameter 'discard_similar' in pHIDEAL() to False"
                                        )
                                        solution[idx + 1] = float("nan")
                                        solution[idx + num_frequencies + 1] = float(
                                            "nan"
                                        )

                            # fill to max_num_frequencies
                            for itr in range(
                                self.max_num_frequencies - num_frequencies
                            ):
                                solution = np.insert(
                                    solution, num_frequencies + itr, float("nan")
                                )
                                solution = np.insert(
                                    solution,
                                    2 * num_frequencies + itr + 1,
                                    float("nan"),
                                )

                            return solution
                else:
                    empty_solution = [
                        float("nan")
                        for i in range(self.max_num_frequencies * 2 + 1 + t2)
                    ]
                    return empty_solution

            # check if field_map is available
            if not isinstance(field_map, type(None)) and not isinstance(
                self.uMGE, type(None)
            ):

                fmap = self.init_field_map(field_map, method="resize", scalefor13c=True)

            else:
                fmap = np.full((1, dim_read, dim_phase, dim_slice), 0)

            # serial solve for mutliple voxels in parallel
            solution_serial = Parallel(n_jobs=4)(
                delayed(calc_voxel)(
                    idx, fmap, snr_mask, use_UMPRIRE_freq_for_init, frequ_inits
                )
                for idx in index_list
            )

            # for debugging uncomment
            # for idx in index_list:
            #    solution_serial = calc_voxel(idx, fmap, snr_mask, use_UMPRIRE_freq_for_init, frequ_inits)
            #   print(solution_serial)

            # save result as np.array and reshape
            solution_matrix = np.array(solution_serial).reshape(
                dim_read, dim_phase, dim_slice, -1
            )

            self.solution_matrix = np.moveaxis(solution_matrix[:, :, :, :-1], -1, 0)

            # print mean number of iterations
            print(
                "Mean Number of iterations: "
                + str(np.nanmean(np.real(solution_matrix[:, :, :, -1])))
            )

            if not isinstance(save_path, type(None)):
                np.save(save_path, self.solution_matrix)

            return self.solution_matrix

    def evaluation_solve(
        self,
        optional_signal_data=None,
        stop_f_offset=0.5,
        max_iterations=300,
        field_map=None,
        save_path=None,
        frequ_inits=np.array([-90.7, 16.0, 55.9]),
        use_UMPRIRE_freq_for_init=True,
        t2_star_init=None,
        discard_similar=True,
        fit_threshold=0.3,
    ):
        """
        Same function as solve() but solves and saves solutions for all three compartments.
        Is used for evaltuation of fit

        Parameters
        ----------
        optional_data : np.ndarray, optional
            In case pHIDEAL solver should no perform the calculations on the initial signal_data (e.g on test data),
            other signal points can be passed here.
            The optional data must have the dimensions echo, read, phase, slice.

        stop_f_offset : float, optional
            Stop criterion for field offset estimates of all frequncies. Default is 0.05
            Stops iterating if delta of all field offset is below the given value.

        max_iterations : int, optional
            Max number of iterations performed before stop.
            Default is 100.

        field_map : BrukerExp of field map sequence : optional
            In case prior knowledge of field map should be used as initial values.
            If None, the initial values provided are being used.

        save_path : string, optional
            Can be used to save solution in a file defined by save_path


        frequ_inits : np.ndarray, optional
            Values to initialize frequency values for optimization.
            Must have at least the size of max_num_frequencies

        use_UMPRIRE_freq_for_init : bool, optional
            Enables initialization of frequency values for optimization based on frequency determined using the UMPIRE algorithm.

        t2_star_init : float, optional
            If not None, the pHIDEAL algorithm also optimizes for T2*

        discard_similar : bool, optional
            If True, similar frequency values determined by pHIDEAL will be discarded, such that only one of them is left.

        fit_threshold : float, optional
            Threshold used by check_result function.
            If MSE of fitted unwrapped phase and measured unwrapped phase is smaller than fit_threshold, the fit result will be counted as valid.

        Returns
        ----------
        solution : np.ndarray
            Contains solution for estimated frequencies and all complex amplitudes of phasors of contributing metabolites.
            Layout: [f_1, f_2 ... f_M, ampl1, ampl2, ..., ampl_M, T2*, num_iterations]*max_num_frequencies
        """

        # sanity check frequency estimates
        if self.__check_init_frequency_estimates(
            frequ_inits, use_UMPRIRE_freq_for_init
        ):

            # in case optional data is given, reinitialize signal_data
            if isinstance(optional_signal_data, type(None)):
                signal = self.signal_data
            else:
                if self.__sanityCheck(optional_signal_data, self.echo_timings):
                    signal = optional_signal_data
                    Warning("Solve with optional data.")

            if not isinstance(t2_star_init, type(None)):
                t2 = 1
            else:
                t2 = 0

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

            # dummy function for one voxel, that gets serialized
            def calc_voxel(
                idx, f_map_value, snr_mask, use_UMPRIRE_freq_for_init, frequ_inits
            ):

                # array to save solution for all compartment sizes
                all_solutions = np.empty(
                    (self.max_num_frequencies, self.max_num_frequencies * 2 + 1 + t2),
                    dtype=complex,
                )
                all_solutions.fill(np.nan)

                # check for for SNR
                if not snr_mask[(..., *idx)]:

                    num_frequencies = self.max_num_frequencies

                    if (
                        not isinstance(self.uMGE, type(None))
                        and use_UMPRIRE_freq_for_init
                    ):
                        init_frequ_estimates = self.initialize_frequ_estimates_w_UMPIRE(
                            self.uMGE._fmap[(..., *idx)],
                            self.max_num_frequencies,
                            frequ_inits,
                        )
                    else:
                        init_frequ_estimates = frequ_inits

                    for iterations in range(1, self.max_num_frequencies + 1):

                        solution = self.solve_one_voxel(
                            signal_data_one_voxel=signal[(..., *idx)],
                            init_frequency_estimates=init_frequ_estimates,
                            f_map_value=f_map_value[(..., *idx)],
                            t2_star_init=t2_star_init,
                            stop_f_offset=stop_f_offset,
                            max_iterations=max_iterations,
                        )

                        print(init_frequ_estimates)

                        init_frequ_estimates = self.determine_most_promising_frequ_init(
                            init_frequ_estimates,
                            solution[num_frequencies : 2 * num_frequencies],
                            solution[:num_frequencies],
                        )

                        # get unwrapped phase values
                        # unwrapped_phase = np.copy(self.uMGE.unwrapped_phase[(...,*idx)])

                        # check if one of the solution values lies out of the expected amplitude and frequency range
                        if np.isnan(solution).all():

                            empty_solution = [
                                float("nan")
                                for i in range(self.max_num_frequencies * 2 + 1 + t2)
                            ]
                            all_solutions[iterations - 1, :] = (
                                empty_solution  # return empty_solution
                            )

                        else:
                            # fill to max_num_frequencies
                            for itr in range(
                                self.max_num_frequencies - num_frequencies
                            ):
                                solution = np.insert(
                                    solution, num_frequencies + itr, float("nan")
                                )
                                solution = np.insert(
                                    solution,
                                    2 * num_frequencies + itr + 1,
                                    float("nan"),
                                )

                            all_solutions[iterations - 1, :] = (
                                solution  # return solution
                            )

                        num_frequencies = self.max_num_frequencies - iterations
                else:
                    empty_solution = [
                        float("nan")
                        for i in range(self.max_num_frequencies * 2 + 1 + t2)
                    ]
                    all_solutions[0:3, :] = empty_solution  # return empty_solution

                return all_solutions

            # check if field_map is available
            if not isinstance(field_map, type(None)) and not isinstance(
                self.uMGE, type(None)
            ):

                fmap = self.init_field_map(field_map, method="resize", scalefor13c=True)

            else:
                fmap = np.full((1, dim_read, dim_phase, dim_slice), 0)

            # serial solve for mutliple voxels in parallel
            # solution_serial = Parallel(n_jobs=4)(
            #    delayed(calc_voxel)(idx, fmap, snr_mask, use_UMPRIRE_freq_for_init, offsets) for idx in index_list
            # )

            # for debugging uncomment
            for idx in index_list:
                # if idx == (23,26,5):
                solution_serial = calc_voxel(
                    idx, fmap, snr_mask, use_UMPRIRE_freq_for_init, frequ_inits
                )
                print("Solution: " + str(solution_serial))

            # save result as np.array and reshape
            solution_matrix = np.array(solution_serial).reshape(
                dim_read, dim_phase, dim_slice, 3, 7 + t2
            )

            self.solution_matrix = (
                solution_matrix  # np.moveaxis(solution_matrix[:, :, :, :-1], -1, 0)
            )

            # print mean number of iterations
            # print(
            #    "Mean Number of iterations: "
            #    + str(np.nanmean(np.real(solution_matrix[:, :, :, -1])))
            # )

            if not isinstance(save_path, type(None)):
                np.save(save_path, self.solution_matrix)

            return self.solution_matrix

    def initialize_frequ_estimates_w_UMPIRE(
        self, UMPRIRE_frequency_value, num_frequencies, frequ_inits
    ):
        """
        Function returns frequency init values based on provided UMPIRE frequency fit and provided frequency init values.
        It determines the value in freq_inits closest to the UMPIRE value, calculates difference and adds for the other values in frequ_inits
        the difference between the closest value in freq_inits and the UMPIRE value.
        Returns num_frequencies inital frequency estimates.

         Parameters
        ----------
        UMPRIRE_frequency_value : float
            Frequency value determined by UMPIRE algorithm.

        num_frequencies : int
            Number of frequency initial frequency values

        freq_inits : np.ndarray
            Must contain 3 values for frequency initialization. Best would be expected frequencies for pH  6.2, 7.0, 7.5

        Returns
        ----------
        np.ndarray
            Frequency values to initialize pHIDEAL
        """

        if frequ_inits.size == 3:
            # determine to which init frequency value is closest to UMPRIRE_frequency_value
            idx_closest_value = min(
                range(len(frequ_inits)),
                key=lambda i: abs(frequ_inits[i] - UMPRIRE_frequency_value),
            )

            # calculate difference
            frequ_diff = UMPRIRE_frequency_value - frequ_inits[idx_closest_value]

            if idx_closest_value == 0:
                init_array = np.array(
                    [
                        UMPRIRE_frequency_value,
                        frequ_inits[1] + frequ_diff,
                        frequ_inits[2] + frequ_diff,
                    ]
                )
            elif idx_closest_value == 1:
                init_array = np.array(
                    [
                        frequ_inits[0] + frequ_diff,
                        UMPRIRE_frequency_value,
                        frequ_inits[2] + frequ_diff,
                    ]
                )
            elif idx_closest_value == 2:
                init_array = np.array(
                    [
                        frequ_inits[0] + frequ_diff,
                        frequ_inits[1] + frequ_diff,
                        UMPRIRE_frequency_value,
                    ]
                )

            if num_frequencies == 1:
                return np.array([UMPRIRE_frequency_value])
            elif num_frequencies == 2:
                if idx_closest_value == 0:
                    return init_array[:2]
                else:
                    return init_array[1:]
            elif num_frequencies == 3:
                return init_array
        else:
            raise Exception("Please provid 3 frequency init values.")

    def determine_most_promising_frequ_init(
        self,
        frequ_inits,
        fitted_amplitudes,
        fitted_frequencies,
        valid_frequency_range=np.array([-250, 200]),
    ):
        """
        Function returns most promising frequency init values based on the fitted amplitudes and frequencies.
        It first sorts the fit results by absolute amplitude. Then it checks if the fitted frequeny lies within the expected frequency range.
        If not, the frequency init gets penalized and moved to the end of the list. After sorting, the last init value in the list is discarded.
        This function can be used if the fit result is considered as not valid and the pH compartment size has to be reduced.

         Parameters
        ----------
        frequ_inits : np.ndarray
            Array containing the initial frequency estimates used for to fit the provided amplitudes.

        fitted_amplitudes : np.ndarray (complex)
            Complex amplitude of each pH compartment fitted by pHIDEAL.

        fitted_frequencies : np.ndarray
            Frequency of each pH compartment fitted by pHIDEAL.

        valid_frequency_range : np.ndarray, optional
            Range of frequencies that are counted as valid fit results.
            The provided frequency range is based on the shifting range of Z-OMPD in the expected pH range plus a margin to account for B0-inhomogeneities.

        Returns
        ----------
        np.ndarray
            Frequency values used to initialize frequency in pHIDEAL, without the init frequency value resulting in the lowest signal amplitude.
        """

        if fitted_amplitudes.size != frequ_inits.size:
            raise Exception(
                "There has been an issue in the frequency init determination."
                + str(frequ_inits.size)
                + " frequency init do not match "
                + str(fitted_amplitudes.size)
                + " fitted amplitudes!"
            )
        else:

            # sort by absolute amplitudes in descending order
            sorted_indices_ampl = np.argsort(-np.abs(fitted_amplitudes))

            # apply sorted indices to frequency inits and determined frequencies
            init_pre_sorted = frequ_inits[sorted_indices_ampl]
            freq_pre_sorted = fitted_frequencies[sorted_indices_ampl]

            final_sorted_indices = np.arange(0, fitted_amplitudes.size)

            # prioritize inits that resulted in valid frequencies
            for idx, sorted_freq in enumerate(freq_pre_sorted):

                # add punishment if frequency not within valid frequency range
                if (
                    sorted_freq < valid_frequency_range[0]
                    or sorted_freq > valid_frequency_range[1]
                ):

                    final_sorted_indices[idx] += 10

            # retrieve final order
            final_sorted_indices = np.argsort(final_sorted_indices)

            # return the most promising frequency inits
            return init_pre_sorted[final_sorted_indices][:-1]

    def init_field_map(self, ref_fmap, method="resize", scalefor13c=True):
        """Handels given reference field map and converts from 1H to 13C frequence. Resizes reference map if necessray."""
        if scalefor13c:
            gyro_ratios = LAMOR_FREQUENCY_H1 / LAMOR_FREQUENCY_C13
        else:
            gyro_ratios = 1

        print(f"Using gyromagnetic ratio of {gyro_ratios:.2f}")
        if ref_fmap.shape != self.uMGE.fmap.shape:
            if method == "resize":
                logger.warning(
                    f"Resizing... {ref_fmap.shape} -> {self.uMGE.fmap.shape}"
                )
                ref_fmap = interpolate_masked_3d_array(ref_fmap, self.uMGE.fmap.shape)
            elif method == "downsample":
                logger.warning(
                    f"Downsampling... {ref_fmap.shape} -> {self.uMGE.fmap.shape}"
                )
                down_factors = np.array(ref_fmap.shape) / np.array(self.uMGE.fmap.shape)
                for d in down_factors:
                    if not float(d).is_integer():
                        assert False, f"Downsampling factors: {down_factors}"
                ref_fmap = downsample_masked_array(
                    ref_fmap, tuple(down_factors.astype(int))
                )
            else:
                raise ValueError("Not a valid Method.")

        if hasattr(ref_fmap, "mask") and ref_fmap.mask.any():
            logger.warning(f"Replacing masked values with zero...")
            ref_fmap = ref_fmap.filled(0)

        return ref_fmap / gyro_ratios

    def check_result(
        self,
        solution_matrix,
        num_frequencies,
        unwrapped_phase,
        fit_threshold,
        valid_frequency_range=np.array([-250, 200]),
    ):
        """
        Function checks validity of the fit results based on prior knowledge of expected frequency range
        and unwrapped phase evolution of the measured signal in a two-step approach:
            First: It is assured that the determined frequency values lie within the valid_frequency_range
            Second: Multiple signal echoes are simulated based on the determined values of frequency and amplitude. Their phase is unwrapped using UMPIRE
                    and the mean squared error of the simulated and measured unwrapped signal phase is calculated.
                    If the MSE is smaller than provided fit_threshold, the result is counted as valid.

         Parameters
        ----------
        solution_matrix : np.ndarray (complex)
            Array containing final estimates for frequencies, amplitudes and optionally T2* fit as returned by solve_one_voxel().

        num_frequencies : int
            Number of pH compartments considered for this fit.

        unwrapped_phase : np.ndarray
            Unwrapped phase of measured signal.

        fit_threshold : float
            Threshold under which the MSE of measured and simulated unwrapped phase has to lie to count result as valid.

        valid_frequency_range : np.ndarray, optional
            Range of frequencies that are counted as valid fit results.
            The provided frequency range is based on the shifting range of Z-OMPD in the expected pH range plus a margin to account for B0-inhomogeneities.

        Returns
        ----------
        bool
            Tells result of validity check. If True, the fit result is counted as valid.
            If False, the fit result is counted as not valid.
        """

        # check if determined frequency values lie within the given range
        for idx in range(num_frequencies):

            if (
                solution_matrix[idx] < valid_frequency_range[0]
                or solution_matrix[idx] > valid_frequency_range[1]
            ):
                return False

        # check if fitted phase evolution is close to the calculated phase evolution
        if (
            self.calculate_fit_error(solution_matrix, unwrapped_phase, num_frequencies)
            > fit_threshold
        ):
            return False

            # if np.abs(solution_matrix[idx+num_frequencies]) <= 0.15*np.nanmax(np.abs(solution_matrix[num_frequencies:2*num_frequencies])):
            #    return False

        return True

    def calculate_fit_error(self, solution, unwrapped_phase, num_frequencies):
        """
        Function simlates signal from estimated frequencies and amplitudes, unwrappes the simulated signal phase using UMPIRE
        and calulates mean squared error of unwrapped simulated signal phase and provided unwrapped measured signal phase.
        The result of this function is used in check_result()

         Parameters
        ----------
        solution : np.ndarray (complex)
            Array containing final estimates for frequencies, amplitudes and optionally T2* fit as returned by solve_one_voxel().

        unwrapped_phase : np.ndarray
            Unwrapped phase of measured signal.

        num_frequencies : int
            Number of pH compartments considered for this fit.

        Returns
        ----------
        float
           Mean squared error of unwrapped simulated signal phase and provided unwrapped measured signal phase
        """

        # load frequencies and amplitudes
        amplitudes = solution[num_frequencies : num_frequencies * 2]
        amplitudes = amplitudes[~np.isnan(amplitudes)]
        frequencies = solution[:num_frequencies]
        frequencies = frequencies[~np.isnan(frequencies)]

        # create test signal with given values and plot phases
        sim_signal = np.squeeze(
            create_test_signal(amplitudes, frequencies, self.echo_timings)
        )

        # test of unwrapped phase
        unwrapped_phases_solution = UMPIRE(sim_signal, self.uMGE.TEs)

        # return mean square error
        return (np.square(unwrapped_phase - unwrapped_phases_solution)).mean()

    # init offsets correspond to pH of 6.2, 7.0, 7.5
    def initialize_frequ_estimates(
        self, num_frequencies, frequ_inits, frequency_value=None, use_UMPIRE=True
    ):

        if not isinstance(frequency_value, type(None)) or not use_UMPIRE:
            if num_frequencies == 1:
                return np.array([frequency_value])  # +offsets[1]
            elif num_frequencies == 2:
                return np.array(
                    [frequency_value + frequ_inits[1], frequency_value + frequ_inits[2]]
                )
            elif num_frequencies == 3:
                return np.array(
                    [
                        frequency_value + frequ_inits[0],
                        frequency_value + frequ_inits[1],
                        frequency_value + frequ_inits[2],
                    ]
                )
            else:
                raise Exception(
                    "Solving for "
                    + str(num_frequencies)
                    + " frequencies currently not supported!"
                )
        else:
            if num_frequencies == 1:
                return np.array([frequ_inits[1]])
            elif num_frequencies == 2:
                return np.array([frequ_inits[1], frequ_inits[2]])
            elif num_frequencies == 3:
                return frequ_inits
            else:
                raise Exception(
                    "Solving for "
                    + str(num_frequencies)
                    + " frequencies currently not supported!"
                )


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
