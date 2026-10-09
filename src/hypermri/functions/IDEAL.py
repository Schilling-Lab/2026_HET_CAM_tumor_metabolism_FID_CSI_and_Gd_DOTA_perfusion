import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from joblib import Parallel, delayed
from mpl_interactions import image_segmenter
from tqdm.auto import tqdm

from hypermri.utils import plot_colorbar

from ..utils.system.logging import init_default_logger

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


class IDEAL_Solver:
    def __init__(
        self,
        relative_frequencies,
        bssfp_object=None,
        signal_data=None,
        echo_spacing=None,
    ):
        # sanity check
        if isinstance(signal_data, type(None)) and not isinstance(
            bssfp_object, type(None)
        ):
            signal_data = bssfp_object.Reconstructed_data
        elif isinstance(signal_data, type(None)) and isinstance(
            bssfp_object, type(None)
        ):
            raise Exception("no data for IDEAL solver given")

        if isinstance(echo_spacing, type(None)) and not isinstance(
            bssfp_object, type(None)
        ):
            echo_spacing = bssfp_object.method["EchoSpacing"] / 1000.0
        elif isinstance(echo_spacing, type(None)) and isinstance(
            bssfp_object, type(None)
        ):
            raise Exception("no echo spacing for IDEAL solver given")

        if signal_data.shape[0] <= relative_frequencies.size:
            print(
                "Not enough echoes to determine contributions of "
                + str(relative_frequencies.size)
                + " metabolites."
            )
            print(
                "Please ensure you chose the rigth format for signal_data: (#echoes, #other_dimensions)."
            )
        elif signal_data.ndim < 2:
            print(
                "Please ensure you chose the rigth format for signal_data: (#echoes, other_dimensions)."
            )
        else:
            self.signal_data = signal_data[:, :, :, :, :, 0]
            self.echo_spacing = echo_spacing
            self.relative_frequencies = relative_frequencies
            self.f_offset = None
            self.T2_star = None
            self.theta = None
            self.num_metabolites = relative_frequencies.size
            self.num_Echoes = signal_data.shape[0]
            self.num_transients = 0  # number of transients that are solved for by IDEAL
            self.solve_for = None
            self.bssfp_object = bssfp_object

    def define_S_corrected(self, signal_data):
        """
        Function takes singnal data and extracts true signal contributed by metabolites by mutliplying with inverse of estimated field_offset, theta and T2* offset.
        For more information see paper "Multicoil Dixon Chemical Species Separation With an Iterative Least-Squares
        Estimation Method" by Scott B. Reeder et al.

        Parameters
        ----------
        signal_data : np.ndarray
            Array containing measured signal.

        Returns
        ----------
        s_corrected : np.ndarray
            Array of shape num_Metabolites*2 containing corrected measured signal.
            Signal is split into real signal components (Index 0 to num_Metabolites-1) and imaginary signal components (Index num_Metabolites to (num_Metabolites*2)-1)
        """

        s_corrected = np.zeros((signal_data.shape[0] * 2))

        for idx_s, s in enumerate(signal_data):
            t_n = (
                -((self.num_Echoes - 1 - (idx_s * 2)) / 2) * self.echo_spacing
            )  # centers t around 1/TR --> t=0 at 1/TR

            s_buffer = s

            # correct field_offset
            if "Field_offset" in self.solve_for:
                s_buffer = s_buffer * np.exp(-2 * np.pi * self.f_offset * t_n * 1j)

            # correct T2*
            # s_buffer = s_buffer * np.exp((np.abs(-((self.num_Echoes - 1 - (idx_s * 2)) / 2)) * self.echo_spacing) / self.T2_star)

            # correct theta
            if "Theta" in self.solve_for:
                s_buffer = s_buffer * np.exp(-1 * self.theta * ((-1) ** (idx_s + 1)))

            s_corrected[idx_s] = np.real(s_buffer)
            s_corrected[idx_s + signal_data.shape[0]] = np.imag(s_buffer)

            np.squeeze(s_corrected)

        return s_corrected

    def define_A(self):
        """
        Function defines matrix A that can be used to calculate real and imaginary signal components.
        For more information see paper "Multicoil Dixon Chemical Species Separation With an Iterative Least-Squares
        Estimation Method" by Scott B. Reeder et al.

        Returns
        ----------
        matrix_A : np.ndarray
            Matrix used to calculate real and imaginary signal components.
            Row 0 to num_Echoes-1 is of the form [cos(2*np.pi*f_1*t_n) -sin(2*np.pi*f_1*t_n) ... cos(2*np.pi*f_M*t_n) -sin(2*np.pi*f_M*t_n)], representing real signal component.
            Row num_Echoes to (num_Echoes*2)-1 is of the form [sin(2*np.pi*f_1*t_n) cos(2*np.pi*f_1*t_n) ... sin(2*np.pi*f_M*t_n) cos(2*np.pi*f_M*t_n)], representing imaginary signal component.
        """

        matrix_A = np.zeros((self.num_Echoes * 2, self.num_metabolites * 2))

        # fill real parts
        for idx_t in range(self.num_Echoes):
            t_n = -((self.num_Echoes - 1 - (idx_t * 2)) / 2) * self.echo_spacing

            for idx_f, f in enumerate(self.relative_frequencies):
                # fill real parts
                matrix_A[idx_t, idx_f * 2] = np.cos(2 * np.pi * f * t_n)
                matrix_A[idx_t, (idx_f * 2) + 1] = -np.sin(2 * np.pi * f * t_n)

                # fill imaginary parts
                matrix_A[idx_t + self.num_Echoes, idx_f * 2] = np.sin(
                    2 * np.pi * f * t_n
                )
                matrix_A[idx_t + self.num_Echoes, (idx_f * 2) + 1] = np.cos(
                    2 * np.pi * f * t_n
                )

        return matrix_A

    def define_S_diff(self, s_corrected, estimated_amplitudes, matrix_A):
        """
        Function defines difference between true signal corrected by estimated field offset and estimated signal contributions of all metabolites.
        For more information see paper "Multicoil Dixon Chemical Species Separation With an Iterative Least-Squares
        Estimation Method" by Scott B. Reeder et al.

        Parameters
        ----------
        s_corrected : np.ndarray
            Array of measured signal corrected with estimated field_offset.
            Signal is split into real signal components (Index 0 to num_Metabolites-1) and imaginary signal components (Index num_Metabolites to (num_Metabolites*2)-1)


        estimated_amplitudes : np.ndarray
            Estimated amplitudes of all metabolites contributing to signal.
            Array size: 2*num_Metabolites.
            Layout: [ampl_real_1, ampl_imag_1, ampl_real_2, ampl_imag_2, ... , ampl_real_M, ampl_imag_M]

        matrix_A : np.ndarray
            Matrix used to calculate real and imaginary signal components.
            Row 1 to num_Echoes is of the form [cos(2*np.pi*f_1*t_n) -sin(2*np.pi*f_1*t_n) ... cos(2*np.pi*f_M*t_n) -sin(2*np.pi*f_M*t_n)], representing real signal component.
            Row num_Echoes+1 to num_Echoes*2 is of the form [sin(2*np.pi*f_1*t_n) cos(2*np.pi*f_1*t_n) ... sin(2*np.pi*f_M*t_n) cos(2*np.pi*f_M*t_n)], representing imaginary signal component.
            A can be defined using function define_A().

        Returns
        ----------
        S_diff : np.ndarray
            Array of shape num_Metabolites*2 containing difference between true and estimated signal.
            Signal is split into real signal components (Index 0 to num_Metabolites-1) and imaginary signal components (Index num_Metabolites to (num_Metabolites*2)-1)
        """

        S_diff = np.zeros(shape=s_corrected.shape)

        for idx_s, s in enumerate(s_corrected):
            S_diff[idx_s] = s - (estimated_amplitudes @ matrix_A[idx_s, :])

        return S_diff

    def define_B(self, matrix_A, estimated_amplitudes):
        """
        Function defines matrix B containing first order taylor expansion of the signal equation.
        Used to calculate delta_T2*, delta_theta, delta_f_offset and delta of signal amplitudes of contributing metabolites.
        For more information see paper "Multicoil Dixon Chemical Species Separation With an Iterative Least-Squares
        Estimation Method" by Scott B. Reeder et al.

        Parameters
        ----------
        matrix_A : np.ndarray
            Matrix used to calculate real and imaginary signal components.
            Row 1 to num_Echoes is of the form [cos(2*np.pi*f_1*t_n) -sin(2*np.pi*f_1*t_n) ... cos(2*np.pi*f_M*t_n) -sin(2*np.pi*f_M*t_n)], representing real signal component.
            Row num_Echoes+1 to num_Echoes*2 is of the form [sin(2*np.pi*f_1*t_n) cos(2*np.pi*f_1*t_n) ... sin(2*np.pi*f_M*t_n) cos(2*np.pi*f_M*t_n)], representing imaginary signal component.
            A can be defined using function define_A().


        estimated_amplitudes : np.ndarray
            Estimated amplitudes of all metabolites contributing to signal.
            Array size: 2*num_Metabolites.
            Layout: [ampl_real_1, ampl_imag_1, ampl_real_2, ampl_imag_2, ... , ampl_real_M, ampl_imag_M]

        Returns
        ----------
        matrix_B : np.ndarray
            Matrix of shape num_Echoes*2 x (1+num_Metabolites)
            The first num_Echoes rows contain real part of signal equation.
            The last num_Echoes rows contain the imaginary part of the signal equation.
        """
        # taylor aproximation of metabolite contributions
        matrix_B = matrix_A

        # first column of matrix containing taylor aproximation of field offset
        if "Field_offset" in self.solve_for:
            column_field_offset = np.zeros((matrix_A.shape[0], 1))

        if "Theta" in self.solve_for:
            column_theta_real = np.zeros((matrix_A.shape[0], 1))
            column_theta_imag = np.zeros((matrix_A.shape[0], 1))

        for idx in range(self.num_Echoes):
            t_n = -((self.num_Echoes - 1 - (idx * 2)) / 2) * self.echo_spacing

            for idx_f, f in enumerate(self.relative_frequencies):
                if "Field_offset" in self.solve_for:
                    # fill real signal contributions of metabolites
                    column_field_offset[idx, 0] = column_field_offset[idx, 0] + (
                        -estimated_amplitudes[idx_f * 2] * np.sin(2 * np.pi * f * t_n)
                        - estimated_amplitudes[idx_f * 2 + 1]
                        * np.cos(2 * np.pi * f * t_n)
                    )

                    # fill imaginary signal contributions of metabolites
                    column_field_offset[idx + self.num_Echoes, 0] = column_field_offset[
                        idx + self.num_Echoes, 0
                    ] + (
                        estimated_amplitudes[idx_f * 2] * np.cos(2 * np.pi * f * t_n)
                        - estimated_amplitudes[idx_f * 2 + 1]
                        * np.sin(2 * np.pi * f * t_n)
                    )

                if "Theta" in self.solve_for:
                    # fill real signal contributions of metabolites
                    column_theta_real[idx, 0] = column_theta_real[idx, 0] + (
                        estimated_amplitudes[idx_f * 2] * np.cos(2 * np.pi * f * t_n)
                        - estimated_amplitudes[idx_f * 2 + 1]
                        * np.sin(2 * np.pi * f * t_n)
                    )

                    column_theta_imag[idx, 0] = column_theta_imag[idx, 0] + (
                        -estimated_amplitudes[idx_f * 2] * np.sin(2 * np.pi * f * t_n)
                        - estimated_amplitudes[idx_f * 2 + 1]
                        * np.cos(2 * np.pi * f * t_n)
                    )

                    # fill imaginary signal contributions of metabolites
                    column_theta_real[idx + self.num_Echoes, 0] = column_theta_real[
                        idx + self.num_Echoes, 0
                    ] + (
                        estimated_amplitudes[idx_f * 2] * np.sin(2 * np.pi * f * t_n)
                        + estimated_amplitudes[idx_f * 2 + 1]
                        * np.cos(2 * np.pi * f * t_n)
                    )
                    column_theta_imag[idx + self.num_Echoes, 0] = column_theta_imag[
                        idx + self.num_Echoes, 0
                    ] + (
                        estimated_amplitudes[idx_f * 2] * np.cos(2 * np.pi * f * t_n)
                        - estimated_amplitudes[idx_f * 2 + 1]
                        * np.sin(2 * np.pi * f * t_n)
                    )

            if "Field_offset" in self.solve_for:
                # apply inner derivative of field offset to real components
                column_field_offset[idx, 0] = (
                    column_field_offset[idx, 0] * 2 * np.pi * t_n
                )
                # apply inner derivative of field offset to imaginary
                # components
                column_field_offset[idx + self.num_Echoes, 0] = (
                    column_field_offset[idx + self.num_Echoes, 0] * 2 * np.pi * t_n
                )

            if "Theta" in self.solve_for:
                # apply inner derivative of theta to real components
                column_theta_real[idx, 0] = column_theta_real[idx, 0] * (-1) ** (
                    idx + 1
                )
                column_theta_imag[idx, 0] = column_theta_imag[idx, 0] * (-1) ** (
                    idx + 1
                )

                # apply inner derivative of theta to imaginary components
                column_theta_real[idx + self.num_Echoes, 0] = column_theta_real[
                    idx + self.num_Echoes, 0
                ] * (-1) ** (idx + 1)
                column_theta_imag[idx + self.num_Echoes, 0] = column_theta_imag[
                    idx + self.num_Echoes, 0
                ] * (-1) ** (idx + 1)

        # second column of matrix containing taylor aproximation of T2* shortening
        # column_T2_star = np.zeros((matrix_A.shape[0], 1))

        # fill real parts
        # for idx in range(self.num_Echoes):
        #    for idx_f, f in enumerate(self.relative_frequencies):
        #        column_T2_star[idx, 0] = column_T2_star[idx, 0] + (
        #            estimated_amplitudes[idx_f * 2]
        #            * np.cos(2 * np.pi * f * np.abs(-((self.num_Echoes - 1 - (idx * 2)) / 2)) * self.echo_spacing)
        #            - estimated_amplitudes[idx_f * 2 + 1]
        #            * np.sin(2 * np.pi * f * np.abs(-((self.num_Echoes - 1 - (idx * 2)) / 2)) * self.echo_spacing)
        #        )

        #    column_T2_star[idx, 0] = column_T2_star[idx, 0] * (
        #        (np.abs(-((self.num_Echoes - 1 - (idx * 2)) / 2)) * self.echo_spacing) / np.power(self.T2_star, 2)
        #    )

        # fill imaginary parts
        # for idx in range(self.num_Echoes):
        #    for idx_f, f in enumerate(self.relative_frequencies):
        #        column_T2_star[idx + self.num_Echoes, 0] = column_T2_star[
        #            idx + self.num_Echoes, 0
        #        ] + (
        #            estimated_amplitudes[idx_f * 2]
        #            * np.sin(2 * np.pi * f * np.abs(-((self.num_Echoes - 1 - (idx * 2)) / 2)) * self.echo_spacing)
        #           + estimated_amplitudes[idx_f * 2 + 1]
        #          * np.cos(2 * np.pi * f * np.abs(-((self.num_Echoes - 1 - (idx * 2)) / 2)) * self.echo_spacing)
        #     )

        # column_T2_star[idx + self.num_Echoes, 0] = column_T2_star[
        #   idx + self.num_Echoes, 0
        #    ] * ((np.abs(-((self.num_Echoes - 1 - (idx * 2)) / 2)) * self.echo_spacing) / np.power(self.T2_star, 2))
        # first_columns = np.append(column_field_offset, column_T2_star, axis=1)

        if "Field_offset" in self.solve_for and "Theta" in self.solve_for:
            first_columns = np.append(column_field_offset, column_theta_real, axis=1)
            first_columns = np.append(first_columns, column_theta_imag, axis=1)

        elif "Field_offset" in self.solve_for and "Theta" not in self.solve_for:
            first_columns = column_field_offset

        elif "Field_offset" not in self.solve_for and "Theta" in self.solve_for:
            first_columns = np.append(column_theta_real, column_theta_imag, axis=1)

        matrix_B = np.append(first_columns, matrix_B, axis=1)

        return matrix_B

    def solve_one_voxel(
        self,
        signal,
        init_f_offset=0.0,
        init_T2_star=0.01,
        init_Theta=0 + 0j,
        stop_f_offset=1,
        stop_T2_star=0.05,
        stop_theta=0.1 + 0.1j,
        max_iterations=25,
    ):
        """
        Function solving IDEAL problem for given initial guesses and echo data of one pixel.
        Requires correct initialization of IDEAL class.

        Parameters
        ----------
        stop_f_offset : float, optional
            Stop criterion for field offset. Default is 0.0001
            Stops iterating if delta of field offset is below the given value.

        stop_theta : float, optional
            Stop criterion for effects of bipolar gradients. Default is 1.0
            Stops iterating if delta of theta is below the given value.

        stop_T2_star : float, optional
            Stop criterion for T2*. Default is 1.0
            Stops iterating if delta of T2* is below the given value.

        stop_theta : float, optional
            Stop criterion for theta. Default is 1.0 + 1.0j
            Stops iterating if delta of theta is below the given value.

        max_iterations : int, optional
            Max number of iterations performed before stop.
            Default is 25.

        optional_data : np.ndarray, optional
            In case IDEAL solver should no perform the calculations on the initial signal_data (e.g on test data),
            other signal points can be passed here.
            The optional data should be a one-dimensional array.

        Returns
        ----------
        solution : np.ndarray
            Contains solution for field_offset, T2_star, theta and all complex amplitudes of phasors of contributing metabolites.
        """

        self.f_offset = init_f_offset
        self.T2_star = init_T2_star
        self.theta = init_Theta

        stop_criterion = False
        num_iterations = 1

        while not stop_criterion:
            # re estimate signal with new field offset
            corrected_S = self.define_S_corrected(signal)
            A = self.define_A()

            estimated_amplitudes = (
                np.linalg.matrix_power(np.transpose(A) @ A, -1)
                @ np.transpose(A)
                @ corrected_S
            )

            s_diff = self.define_S_diff(corrected_S, estimated_amplitudes, A)

            B = self.define_B(A, estimated_amplitudes)

            y = (
                np.linalg.matrix_power(np.transpose(B) @ B, -1)
                @ np.transpose(B)
                @ s_diff
            )
            buffer = 0
            if "Field_offset" in self.solve_for:
                self.f_offset = self.f_offset + y[buffer]
                buffer += 1
            # self.T2_star = self.T2_star + y[1]

            if "Theta" in self.solve_for:
                self.theta = (self.theta.real + y[buffer]) + 1j * (
                    self.theta.imag + y[buffer + 1]
                )

            # check if stop criterions reached
            if num_iterations == max_iterations:
                stop_criterion = True
            elif "Field_offset" in self.solve_for and "Theta" in self.solve_for:
                if (
                    np.abs(y[0]) < stop_f_offset
                    and np.abs(y[1]) < stop_theta.real
                    and np.abs(y[2]) < stop_theta.imag
                ):
                    stop_criterion = True

            elif "Field_offset" in self.solve_for and "Theta" not in self.solve_for:
                if np.abs(y[0]) < stop_f_offset:
                    stop_criterion = True

            elif "Field_offset" not in self.solve_for and "Theta" in self.solve_for:
                if np.abs(y[0]) < stop_theta.real and np.abs(y[1]) < stop_theta.imag:
                    stop_criterion = True

            num_iterations += 1

        # build solution matrix
        if "Field_offset" in self.solve_for and "Theta" in self.solve_for:
            solution = np.append(self.theta, self.f_offset)

        elif "Field_offset" in self.solve_for and "Theta" not in self.solve_for:
            solution = self.f_offset

        elif "Field_offset" not in self.solve_for and "Theta" in self.solve_for:
            solution = self.theta

        # calculate estimated amplitudes for the last time
        corrected_S = self.define_S_corrected(signal)
        A = self.define_A()

        estimated_amplitudes = (
            np.linalg.matrix_power(np.transpose(A) @ A, -1)
            @ np.transpose(A)
            @ corrected_S
        )

        complex_amplitudes = np.zeros(
            (int((len(estimated_amplitudes) / 2)),), dtype=complex
        )

        # make amplitudes complex
        for idx in range(0, len(estimated_amplitudes), 2):
            complex_amplitudes[int(idx / 2)] = (
                estimated_amplitudes[idx] + 1j * estimated_amplitudes[idx + 1]
            )

        solution = np.append(solution, complex_amplitudes)
        solution = np.append(solution, num_iterations)

        return solution

    def solve(
        self,
        init_f_offset=0.0,
        init_T2_star=0.01,
        init_Theta=0 + 0j,
        stop_f_offset=1.0,
        stop_T2_star=0.05,
        stop_theta=0.1 + 0.1j,
        max_iterations=25,
        optional_data=None,
        solve_for=["Field_offset", "Theta"],
        field_map=None,
        save_path=None,
        correct_chemical_shift=False,
    ):
        """
        Function solving IDEAL problem for given initial guesses and echo data of an entire volume
        by calling solve_one_voxel function for all voxels in the volume.
        Requires correct initialization of IDEAL class.

        Parameters
        ----------
        init_f_offset : float, optional
            Initial guess for field inhomogeneities shifting echoes by a phase.
            Default is 0.0.

        init_T2_star : float, optional
            Initial guess for shortened T2* relaxation. Default is 0.01
            Cannot be zero due to deivision by zero.

        init_Theta : complex, optional
            Initial guess for effects of bipolar gradients like phase errors, eddy currents and amplitude modulations.
            Default is 0.0 + 0.0j.

        stop_f_offset : float, optional
            Stop criterion for field offset. Default is 0.0001
            Stops iterating if delta of field offset is below the given value.

        stop_theta : float, optional
            Stop criterion for effects of bipolar gradients. Default is 1.0
            Stops iterating if delta of theta is below the given value.

        stop_T2_star : float, optional
            Stop criterion for T2*. Default is 1.0
            Stops iterating if delta of T2* is below the given value.

        max_iterations : int, optional
            Max number of iterations performed before stop.
            Default is 25.

        optional_data : np.ndarray, optional
            In case IDEAL solver should no perform the calculations on the initial signal_data (e.g on test data),
            other signal points can be passed here.
            The optional data should be a one-dimensional array.

        solve_for : list[string], optional
            Parameters that are being optized for.
            Possible parameters:
                - Field_offset
                - Theta

        field_map : BrukerExp of field map sequence : optional
            In case prior knowledge of field map should be used as initial values.
            If None, the initial values provided are being used.

        save_path : string, optional
            Can be used to save solution in a file defined by save_path

        Returns
        ----------
        solution : np.ndarray
            Contains solution for field_offset, T2_star, theta and all complex amplitudes of phasors of contributing metabolites.
        """

        if "Field_offset" not in solve_for and "Theta" not in solve_for:
            (
                "Please choose transients to solve for (possibilities: Field_offset, Theta) and give them as string list in solve_for."
            )
        else:
            for s in solve_for:
                if s != "Field_offset" and s != "Theta":
                    Warning("Unknown transient " + s + " given in solve_for!")

            self.solve_for = solve_for
            self.num_transients = len(solve_for)

            # check if sufficient amount of echoes given to solve for wanted
            # num of transients
            if self.num_Echoes < (self.num_transients + self.num_metabolites + 1):
                Warning(
                    str(self.num_Echoes)
                    + " echoes are not enough to solve for "
                    + str(self.num_transients + self.num_metabolites)
                    + " unknowns!"
                )

            # in case optional data is given, reinitialize signal_data
            if isinstance(optional_data, np.ndarray):
                self.signal_data = optional_data
                self.num_Echoes = optional_data.shape[0]
                print("Solve with optional data.")

            # read data dimensions and safe indexes
            _, dim_read, dim_phase, dim_slice, dim_rep = self.signal_data.shape
            index_list = list(np.ndindex(dim_read, dim_phase, dim_slice, dim_rep))
            index_list = tqdm(index_list, desc="Calculating Voxels", leave=True)

            # dummy function for one voxel, that gets serialized
            def calc_voxel(idx, f_map_value):
                solution = self.solve_one_voxel(
                    self.signal_data[(..., *idx)],
                    f_map_value[(..., *idx)],
                    init_T2_star,
                    init_Theta,
                    stop_f_offset,
                    stop_T2_star,
                    stop_theta,
                    max_iterations,
                )

                return solution

            # check if field_map is available
            if not isinstance(field_map, type(None)):
                fmap = field_map.seq2d

                # reorder field map to match bssfp data -> only works for
                # scanner reconstructed bssfp data!
                fmap = np.squeeze(fmap)
                fmap = np.transpose(fmap, (1, 0, 2))
                fmap = np.flip(fmap, axis=2)

                # add extra dimensions
                fmap = np.expand_dims(fmap, 0)
                fmap = np.repeat(fmap[..., np.newaxis], dim_rep, axis=-1)
            else:
                fmap = np.full(
                    (1, dim_read, dim_phase, dim_slice, dim_rep), init_f_offset
                )

            # serial solve for mutliple voxels in parallel
            solution_serial = Parallel(n_jobs=4)(
                delayed(calc_voxel)(idx, fmap) for idx in index_list
            )

            # save result as np.array and reshape
            solution_matrix = np.array(solution_serial).reshape(
                dim_read, dim_phase, dim_slice, dim_rep, -1
            )

            self.solution_matrix = np.moveaxis(solution_matrix[:, :, :, :, :-1], -1, 0)

            # print mean number of iterations
            print(
                "Mean Number of iterations: "
                + str(np.mean(np.real(solution_matrix[:, :, :, :, -1])))
            )

            # correct for chemical shift
            if correct_chemical_shift:
                self.solution_matrix = self.correct_chemical_shift()

            if not isinstance(save_path, type(None)):
                np.save(save_path, self.solution_matrix)

            return self.solution_matrix

    def create_test_signal(
        self,
        phasor_amplitudes,
        n,
        field_offset=0.0,
        T2_star=0.0,
        theta=0.0 + 0.0j,
        noise_scale=0.0,
    ):
        """
        Function creates a n signals with given phasor_amplitudes, field_offset, T2_star, theta for initialized
        echo_spacing and metabolite frequencies.

        Parameters
        ----------
        phasor_amplitudes : np.ndarray
            Amplitude of each metabolite contributing to the signal.
            This is what the solver tries to reconstruct from the test signals.
            Array size has to match the number of given metabolites

        n : int
            Number of test signals to be generated.



        field_offset : float, optional
            Field offset shifting signals by a phase in [Hz].
            This is what the solver tries to reconstruct from the test signals.

        T2_star : float, optional
            Shortened T2* relaxation time in [s].
            This is what the solver tries to reconstruct from the test signals.

        theta : complex, optional
            Effects of bipolar gradients like phase errors, eddy currents and amplitude modulations.
            This is what the solver tries to reconstruct from the test signals.

        noise : float, optional
            Standard deviation (spread or “width”) of the normal distributed noise alternating the signals.
            If noise = 0.0 no noise is added.

        Returns
        ----------
        test_signals : np.ndarray
            Array of n test signals with contributions of all metabolites, field changes and normal distributed noise.
        """

        test_signals = np.zeros(n, dtype=complex)

        for idx in range(n):
            if noise_scale != 0.0:
                noise = (
                    np.random.normal(scale=noise_scale)
                    + np.random.normal(scale=noise_scale) * 1j
                )
                print("Noise: " + str(noise))

            for idx_a, amplitude in enumerate(phasor_amplitudes):
                test_signals[idx] = test_signals[idx] + amplitude * np.exp(
                    1j
                    * 2
                    * np.pi
                    * self.relative_frequencies[idx_a]
                    * -((self.num_Echoes - 1 - (idx * 2)) / 2)
                    * self.echo_spacing
                )

            # apply field offset
            if field_offset != 0.0:
                test_signals[idx] = test_signals[idx] * (
                    np.exp(
                        (1j * 2 * np.pi * field_offset)
                        * -((self.num_Echoes - 1 - (idx * 2)) / 2)
                        * self.echo_spacing
                    )
                )

            # apply T2* error
            if T2_star != 0.0:
                test_signals[idx] = test_signals[idx] * (
                    np.exp(
                        (-1 / T2_star)
                        * np.abs(-((self.num_Echoes - 1 - (idx * 2)) / 2))
                        * self.echo_spacing
                    )
                )

            # apply theta
            if theta != 0.0:
                test_signals[idx] = test_signals[idx] * (
                    np.exp(theta * ((-1) ** (idx + 1)))
                )

            # add noise
            if noise_scale != 0.0:
                test_signals[idx] = test_signals[idx] + noise

        return test_signals

    def get_amplitude_ratios(self):
        """
        Function returns ratios of metabolite contributions calculated by IDEAL.

        Returns
        ----------
        ratios : np.ndarray
            Ratios of metabolite contributions.

        """

        if self.solution_matrix is None:
            print(
                "There is no solution matrix available. Please run solve function first."
            )
        else:
            ratios = np.abs(self.solution_matrix[self.num_transients, ...]) / np.abs(
                self.solution_matrix[self.num_transients + 1, ...]
            )

        return ratios

    def assesment_plot(
        self,
        sol_metab1,
        sol_metab2,
        csi=None,
        integration_intervall_metabs=None,
        roi_names=("Acetone", "Water", "Acetone+Water"),
        csi_min_intensity=0.25,
        plot_all=False,
        slice_idx=None,
    ):
        """
        Function creates an interactive plot for cormarison of metabolite ratios from CSI and metabolite ratios calculated by IDEAL.
        The plot allows to draw regions of interests in which the mean of the selected pixels will be displayed.
        Masks can be saved and loaded using the buttons. The textbox defines the file name.
        Until now only 2D plots can be displayed.

        Parameters
        ----------
        sol_metab1 : np.array, complex
            Solution calculated by IDEAL for 1st metabolite.
            Should contain only one slice.

        sol_metab2 : np.array, complex
            Solution calculated by IDEAL for 2nd metabolite.
            Should contain only one slice.

        csi : CSI class, optional
            CSI sequence used for comparison.
            If no csi input, then only the ratios of the IDEAL metabolite solutions are plotted.

        integration_intervall_metabs : np.array, optional
            Defines intervalls over which the integral of the CSI data will be calculated (in Hz).
            Each integral should stand for one metabolite.
            Required shape: ((int1_start, int1_end),(int2_start, int2_end),...)
            Has to be given if CSI should be used


        roi_names : np.array (string), optional
            Names of individual ROIS that can be selected.
            Default: ("Acetone", "Water", "Acetone+Water")

        csi_min_intensity : float, optional
           Min intensity value that will be taken into the integral of the CSI.
           Default: 0.25

        """

        # integrate csi values for given intervalls
        if (
            not isinstance(integration_intervall_metabs, type(None))
            and len(integration_intervall_metabs) < 2
            and len(integration_intervall_metabs[0]) != 2
        ):
            print(
                "The intervall dimensions do not fit. Please give them in form of: ((int1_start, int1_end),(int2_start, int2_end),...)"
            )
        else:
            # define base parameters
            parameter_dict = {
                "vmin": 0,
                "vmax": 10,
            }

            line_properties = {"color": "red", "linewidth": 1}

            if not isinstance(slice_idx, type(None)):
                slice_idx_2 = (
                    slice_idx[0] + 1,
                    slice_idx[1],
                    slice_idx[2],
                    slice_idx[3],
                )

                IDEAL_metab1 = np.abs(sol_metab1[slice_idx])
                IDEAL_metab2 = np.abs(sol_metab2[slice_idx_2])
            else:
                IDEAL_metab1 = np.abs(sol_metab1)
                IDEAL_metab2 = np.abs(sol_metab2)

            # if CSI is provided, plot ratios between CSI and IDEAL solution
            if not isinstance(csi, type(None)):
                csi_metab1 = np.rot90(
                    csi.integrate_intervall(
                        integration_intervall_metabs[0][0],
                        integration_intervall_metabs[0][1],
                        min_intensity=csi_min_intensity,
                    )[::-1, :]
                )

                csi_metab_2 = np.rot90(
                    csi.integrate_intervall(
                        integration_intervall_metabs[1][0],
                        integration_intervall_metabs[1][1],
                        min_intensity=csi_min_intensity,
                    )[::-1, :]
                )

                csi_ratio = csi_metab_2 / csi_metab1

                # calculate ratios of IDEAL solution
                IDEAL_ratio = IDEAL_metab2 / IDEAL_metab1

                total_ratio = csi_ratio  # define base parameters
                parameter_dict = {
                    "vmin": np.min(total_ratio),
                    "vmax": np.max(total_ratio),
                }

                if plot_all:
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                    ax2.imshow(IDEAL_ratio, **parameter_dict)
                    ax2.set_title("IDEAL ratio")
                    ax3.imshow(csi_ratio, **parameter_dict)
                    ax3.set_title("CSI ratio")
                else:
                    fig, ax1 = plt.subplots()

                ax1.set_title("IDEAL/CSI")

            # if CSI is not provided, just plot ratios of IDEAL solution
            else:
                fig, ax1 = plt.subplots()
                ax1.set_title("Ratios of metabolites")

                total_ratio = IDEAL_metab2  # / IDEAL_metab1

                parameter_dict = {
                    "vmin": np.min(total_ratio),
                    "vmax": np.max(total_ratio),
                }

            # create segmenter object
            segmenter = image_segmenter(
                total_ratio[:, :],
                figsize=(8, 5),
                # mask_colors=("red", "pink", "green"),
                nclasses=len(roi_names),
                lineprops=line_properties,
                mask_alpha=0.76,
                ax=ax1,
                **parameter_dict,
            )

            # function calculates mean of all pixels for every mask
            def calculate_mean():
                # get masks
                for roi in range(0, len(roi_names)):
                    test_mask = segmenter.mask == roi + 1
                    num_selected_pixels = test_mask.sum()

                    if num_selected_pixels > 2:
                        # apply mask on data
                        applied_mask = total_ratio * test_mask

                        # calculate mean
                        mean = (
                            np.sum(applied_mask)
                            - np.max(applied_mask)
                            - np.min(applied_mask)
                        ) / (num_selected_pixels - 2)
                    else:
                        mean = 0.0

                    print("Ratio of " + roi_names[roi] + ": " + str(mean))

            def plot_ratios(eraser_mode, roi_key):
                roi_number = roi_names.index(roi_key) + 1
                segmenter.erasing = eraser_mode
                segmenter.current_class = roi_number
                calculate_mean()

            # init widgets
            class_selector = widgets.Dropdown(options=roi_names, description="ROI name")
            erasing_button = widgets.Checkbox(value=False, description="Erasing")
            save_button = widgets.Button(description="Save Masks")
            load_button = widgets.Button(description="Load Masks")
            file_name_box = widgets.Text(
                value="mask",
                placeholder="filename",
                description="Mask Filename:",
                disabled=False,
            )

            ui = widgets.HBox(
                [
                    erasing_button,
                    class_selector,
                    save_button,
                    load_button,
                    file_name_box,
                ],
                layout=widgets.Layout(display="flex"),
            )

            sliders = widgets.interactive_output(
                plot_ratios,
                {
                    "eraser_mode": erasing_button,
                    "roi_key": class_selector,
                },
            )

            # save current mask in file (only one mask can be saved)
            def on_save(b):
                segmenter.mask.tofile(str(file_name_box.value) + ".dat")
                with sliders:
                    print("Saved masks to file " + str(file_name_box.value) + ".dat!")

            # load mask
            def on_load(b):
                if os.path.exists(str(file_name_box.value) + ".dat"):
                    segmenter.mask = np.reshape(
                        np.fromfile(str(file_name_box.value) + ".dat"),
                        segmenter.mask.shape,
                    )

                    for i in range(segmenter.nclasses + 1):
                        idx = segmenter.mask == i
                        if i == 0:
                            segmenter._overlay[idx] = [0, 0, 0, 0]
                        else:
                            segmenter._overlay[idx] = segmenter.mask_colors[i - 1]

                    segmenter._mask.set_data(segmenter._overlay)
                else:
                    logger.error(
                        "No file called " + str(file_name_box.value) + ".dat to load"
                    )

            save_button.on_click(on_save)
            load_button.on_click(on_load)

        display(ui, sliders)

    def plot_ratio_curves(
        self,
        solution_arrays,
        mask_name,
        num_rois,
        x_values,
        slice_idx=np.s_[1, 14, :, :, 0],
        slice_idx_add_slice=None,
        noise_array=None,
        ext_means=None,
    ):
        """
        Function takes given IDEAL solutions, calculates ratios between two metabolites
        and calculates mean value of ratios for given mask. Those values are fitted to a line and plotted on a curve.
        Requires adjustment of slice to apply the mask on and adjustment of x-values for fit.


        Parameters
        ----------
        solution_arrays : tuple (solution_title (string), IDEAL solution matrix)
            N solutions with names the masks are applied on and curves are plotted for.

        mask_name : string
            File name of mask applied on solution matrix.
            Can be created using assesment_plot function.

        num_rois : int
            Number of rois saved in mask.
            Corresponds to the number of mean values calculated for plot.

        x_axis : np.array(float)
            X-axis for plot. Values also used for fit

        slice_idx : np.s_, optional
            Numpy slice of solution for first metabolite.
            Slice of second metabolite will be taken automatically.

        slice_idx_add_slice : np.s_, optional
            In case second slice required for calculating curve plot.

        noise_array : np.array(int), optional
            Allows to substract noise levels for entire slice.
            Has to be of size len(solution_arrays).

        ext_means : tuple (title (string), list of float), optional
            Allows to put external values into plot (e.g. from CSI).
            First element of tuple is title for legend.
            The second element is a list of mean values.

        Returns
        ----------
        Plot of fitted curves for all solution matrices given as input.
        """
        if os.path.exists(mask_name + ".dat"):
            fig, ax = plt.subplots()

            slice_idx_2 = (
                slice_idx[0] + 1,
                slice_idx[1],
                slice_idx[2],
                slice_idx[3],
            )

            x_axis = x_values
            x_axis = np.insert(x_axis, 0, 0.0)

            for idx, solution in enumerate(solution_arrays):
                if not isinstance(noise_array, type(None)):
                    noise = noise_array[idx]
                else:
                    noise = 0

                if len(solution[1].shape) == 3:
                    sol_metab_1 = np.abs(solution[1][0, :, :])
                    sol_metab_2 = np.abs(solution[1][1, :, :])
                else:
                    sol_metab_1 = np.abs(solution[1][slice_idx]) - noise
                    sol_metab_2 = np.abs(solution[1][slice_idx_2]) - noise

                ratios = sol_metab_2 / sol_metab_1

                mean_values = []

                # load mask
                mask = np.reshape(
                    np.fromfile(str(mask_name) + ".dat"),
                    sol_metab_1.shape,
                )

                if not isinstance(slice_idx_add_slice, type(None)):
                    slice_idx_add_slice_2 = (
                        slice_idx_add_slice[0] + 1,
                        slice_idx_add_slice[1],
                        slice_idx_add_slice[2],
                        slice_idx_add_slice[3],
                    )

                    sol_metab_1_slice2 = (
                        np.abs(solution[1][slice_idx_add_slice]) - noise
                    )
                    sol_metab_2_slice2 = (
                        np.abs(solution[1][slice_idx_add_slice_2]) - noise
                    )

                    ratios_slice2 = sol_metab_2_slice2 / sol_metab_1_slice2

                    mask2 = np.reshape(
                        np.fromfile("mask_coronal_DMSO_Urea_3_45_larger" + ".dat"),
                        sol_metab_1.shape,
                    )

                for roi in range(0, num_rois):
                    if (
                        roi == 0
                        or roi == 3
                        # or roi == 4
                        or isinstance(slice_idx_add_slice, type(None))
                    ):
                        test_mask = mask == roi + 1
                        num_selected_pixels = test_mask.sum()

                        if num_selected_pixels > 2:
                            applied_mask = ratios * test_mask

                            # calculate mean
                            mean = (
                                np.sum(applied_mask)
                                - np.max(applied_mask)
                                - np.min(applied_mask)
                            ) / (num_selected_pixels - 2)

                            mean_values.append(mean)
                        else:
                            mean = 0.0
                            mean_values.append(mean)

                    else:
                        test_mask = mask2 == roi + 1
                        num_selected_pixels = test_mask.sum()

                        if num_selected_pixels > 2:
                            applied_mask = ratios_slice2 * test_mask

                            # calculate mean
                            mean = (
                                np.sum(applied_mask)
                                - np.max(applied_mask)
                                - np.min(applied_mask)
                            ) / (num_selected_pixels - 2)

                            mean_values.append(mean)
                        else:
                            mean = 0.0
                            mean_values.append(mean)

                # fit line and add to plot
                fit = np.polyfit(
                    x_values,
                    mean_values,
                    1,
                    full=True,
                )

                print("SSR of " + solution[0] + ": " + str(fit[1][0]))

                ax.scatter(x_values, mean_values, alpha=0.5)
                ax.plot(x_axis, fit[0][0] * x_axis + fit[0][1], label=solution[0])

                print(mean_values)

            # add external fit
            if not isinstance(ext_means, type(None)):
                for ext_value in ext_means:
                    # fit line and add to plot
                    fit = np.polyfit(
                        x_values,
                        ext_value[1],
                        1,
                        full=True,
                    )

                    print("SSR of " + ext_value[0] + ": " + str(fit[1][0]))
                    ax.scatter(x_values, ext_value[1], alpha=0.5)
                    ax.plot(x_axis, fit[0][0] * x_axis + fit[0][1], label=ext_value[0])

            ax.set_ylabel("Ratio [Urea/DMSO]")
            ax.set_xlabel("concentration [mol/l]")
            ax.legend()

        else:
            logger.error("No file called " + mask_name + ".dat to load")

        # plot curves for all means

    def plot_results(
        self,
        axlist,
        values_to_plot=("Metabolites"),
        fig=None,
        value_lims=(100000, 5000000, 0, 10),
        parameter_dict=None,
        display_ui=True,
        optional_data=None,
    ):
        if optional_data is not None:
            self.solution_matrix = optional_data

        # check dimensions of solution
        if len(self.solution_matrix.shape) == 4:
            is_3D = False
        elif len(self.solution_matrix.shape) == 5:
            is_3D = True
        else:
            print("Solution Matrix dimensions not correct.")

        # use default paramter dictionary if none is provided
        if not parameter_dict:
            parameter_dict = {
                "vmin": value_lims[0],
                "vmax": value_lims[1],
            }

        # create 2D plot
        if not is_3D:
            if len(axlist) != self.solution_matrix.shape[0] + 1:
                print(
                    "Not enough ax objects in axlist for 2D plot, please give "
                    + str(self.solution_matrix.shape[0] + 1)
                    + " axes!"
                )
            elif fig is None:
                print("2D plot requires a figure object.")
            else:
                for ax in axlist:
                    ax.clear()

            parameter_dict = {
                "vmin": value_lims[0],
                "vmax": value_lims[1],
            }

            # plot field offset
            axlist[0].imshow(np.real(self.solution_matrix[0, :, :]))
            plot_colorbar(fig, axlist[0], np.real(self.solution_matrix[0, :, :]))

            # plot T2*
            # axlist[1].imshow(np.real(self.solution_matrix[1, :, :]))
            # plot_colorbar(fig, axlist[1], np.real(self.solution_matrix[1, :, :]))

            # plot metabolite amplitudes
            for idx, metabolite in enumerate(
                self.solution_matrix[self.num_transients :, :, :]
            ):
                axlist[idx + self.num_transients].imshow(
                    np.abs(metabolite), **parameter_dict
                )
                plot_colorbar(
                    fig,
                    axlist[idx + self.num_transients],
                    (parameter_dict["vmin"], parameter_dict["vmax"]),
                )

            # plot ratio image
            axlist[-1].imshow(
                self.get_amplitude_ratios(),
                vmin=value_lims[2],
                vmax=value_lims[3],
            )
            plot_colorbar(fig, axlist[-1], (value_lims[2], value_lims[3]))

            axlist[0].set_title("Field Offset [Hz]")
            # axlist[1].set_title("T2*")
            axlist[-1].set_title("Amplitude Ratios")

        if is_3D:
            # read image dimensions
            dim_read = self.solution_matrix.shape[1]
            dim_phase = self.solution_matrix.shape[2]
            dim_slice = self.solution_matrix.shape[3]
            dim_rep = self.solution_matrix.shape[4]

            # if len(axlist) != 3 * (self.solution_matrix.shape[0]) + 3:
            #   print(
            #        "Not enough ax objects in axlist for 3D plot, please give 3*Number of Metabolites axes!"
            #   )
            # else:

            def plot_all(
                slice_nr_ax,
                slice_nr_cor,
                slice_nr_sag,
                colorbar_min,
                colorbar_max,
                colorbar_f_offset_min,
                colorbar_f_offset_max,
                nr_rep,
            ):
                for ax in axlist:
                    ax.clear()

                ax_offset = 0  # used to count how many ax objects already filled

                # plot field offset
                if "F_offset" in values_to_plot:
                    parameter_dict_f_offset = {
                        "vmin": np.min((colorbar_f_offset_min, colorbar_f_offset_max)),
                        "vmax": np.max((colorbar_f_offset_min, colorbar_f_offset_max)),
                    }

                    plot_img(
                        slice_nr_ax,
                        slice_nr_cor,
                        slice_nr_sag,
                        np.real(self.solution_matrix[0, :, :, :, :]),
                        axlist[ax_offset : ax_offset + 3],
                        parameter_dict_f_offset,
                        "Field offset ",
                        nr_rep,
                    )
                    ax_offset += 3

                if "Theta" in values_to_plot:
                    parameter_dict_f_offset = {
                        "vmin": np.min((colorbar_f_offset_min, colorbar_f_offset_max)),
                        "vmax": np.max((colorbar_f_offset_min, colorbar_f_offset_max)),
                    }

                    plot_img(
                        slice_nr_ax,
                        slice_nr_cor,
                        slice_nr_sag,
                        np.abs(self.solution_matrix[1, :, :, :, :]),
                        axlist[ax_offset : ax_offset + 3],
                        parameter_dict_f_offset,
                        "Theta",
                        nr_rep,
                    )
                    ax_offset += 3

                # plot T2*
                if "T2*" in values_to_plot:
                    plot_img(
                        slice_nr_ax,
                        slice_nr_cor,
                        slice_nr_sag,
                        np.real(self.solution_matrix[2, :, :, :, :]),
                        axlist[ax_offset : ax_offset + 3],
                        parameter_dict,
                        "T2* ",
                        nr_rep,
                    )
                    ax_offset += 3

                # plot metabolites
                if "Metabolites" in values_to_plot:
                    parameter_dict_metab = {
                        "vmin": np.min((colorbar_min, colorbar_max)),
                        "vmax": np.max((colorbar_min, colorbar_max)),
                    }

                    for idx in range(
                        self.solution_matrix.shape[0] - self.num_transients
                    ):
                        plot_img(
                            slice_nr_ax,
                            slice_nr_cor,
                            slice_nr_sag,
                            np.abs(
                                self.solution_matrix[
                                    idx + self.num_transients, :, :, :, :
                                ]
                            ),
                            axlist[ax_offset : ax_offset + 3],
                            parameter_dict_metab,
                            "Metabolite " + str(idx),
                            nr_rep,
                        )

                        ax_offset += 3

                # plot images of amplitude ratios
                if "Ratios" in values_to_plot:
                    ratios = self.get_amplitude_ratios()
                    mean = np.nanmean(ratios)
                    parameter_dict_ratios = {
                        "vmin": mean / 2,
                        "vmax": mean * 2,
                    }
                    plot_img(
                        slice_nr_ax,
                        slice_nr_cor,
                        slice_nr_sag,
                        ratios,
                        axlist[ax_offset : ax_offset + 3],
                        parameter_dict_ratios,
                        "Ratios",
                    )
                    ax_offset += 3

            # define image plotting function
            def plot_img(
                slice_nr_ax,
                slice_nr_cor,
                slice_nr_sag,
                image_data,
                axes,
                parameter_dict,
                title,
                nr_rep,
            ):
                # plot images of calculated amplitudes
                axes[0].imshow(
                    np.rot90(image_data[slice_nr_ax, :, :, nr_rep], 3),
                    extent=[
                        (dim_phase // 2),
                        -(dim_phase // 2),
                        (dim_slice // 2),
                        -(dim_slice // 2),
                    ],
                    **parameter_dict,
                )
                axes[0].set_aspect(dim_slice / dim_phase)
                axes[0].axvline(
                    slice_nr_sag - (dim_slice // 2),
                    color="white",
                    linewidth=1,
                    linestyle="dashed",
                )
                axes[0].axhline(
                    slice_nr_cor - (dim_phase // 2),
                    color="white",
                    linewidth=1,
                    linestyle="dashed",
                )
                axes[0].set_xlabel("phase")
                axes[0].set_ylabel("slice")
                axes[0].set_title(title + " - Axial")

                axes[1].imshow(
                    np.rot90(image_data[:, :, slice_nr_cor, nr_rep], 2),
                    extent=[
                        dim_phase // 2,
                        -dim_phase // 2,
                        -dim_read // 2,
                        dim_read // 2,
                    ],
                    **parameter_dict,
                )
                axes[1].set_aspect(dim_phase / dim_read)
                axes[1].axhline(
                    slice_nr_ax - (dim_read // 2),
                    color="white",
                    linewidth=1,
                    linestyle="dashed",
                )
                axes[1].axvline(
                    slice_nr_sag - (dim_phase // 2),
                    color="white",
                    linewidth=1,
                    linestyle="dashed",
                )
                axes[1].set_xlabel("phase")
                axes[1].set_ylabel("read")
                axes[1].set_title(title + " - Coronal")

                axes[2].imshow(
                    np.rot90(image_data[:, slice_nr_sag, :, nr_rep], 2),
                    extent=[
                        dim_slice // 2,
                        -dim_read // 2,
                        -dim_read // 2,
                        dim_read // 2,
                    ],
                    **parameter_dict,
                )
                axes[2].set_aspect(dim_slice / dim_read)
                axes[2].axhline(
                    slice_nr_ax - (dim_read // 2),
                    color="white",
                    linewidth=1,
                    linestyle="dashed",
                )
                axes[2].axvline(
                    slice_nr_cor - (dim_slice // 2),
                    color="white",
                    linewidth=1,
                    linestyle="dashed",
                )
                axes[2].set_xlabel("slice")
                axes[2].set_ylabel("read")
                axes[2].set_title(title + " - Sagital")

            # create interactive sliders for  slices in all dimensions
            slice_slider_ax = widgets.IntSlider(
                value=dim_read // 2,
                min=0,
                max=dim_read - 1,
                description="Slice Axial: ",
            )
            slice_slider_cor = widgets.IntSlider(
                value=dim_slice // 2,
                min=0,
                max=dim_slice - 1,
                description="Slice Coronal: ",
            )
            slice_slider_sag = widgets.IntSlider(
                value=dim_phase // 2,
                min=0,
                max=dim_phase - 1,
                description="Slice Sagital: ",
            )

            slider_rep = widgets.IntSlider(
                value=dim_rep // 2,
                min=0,
                max=dim_rep - 1,
                description="Repetition: ",
            )

            int_box_colorbar_min = widgets.BoundedIntText(
                value=0,
                min=0,
                max=10**100,
                step=1,
                description="Min Colorbar:",
                disabled=False,
            )

            int_box_colorbar_max = widgets.BoundedIntText(
                value=100,
                min=0,
                step=1,
                max=10**100,
                description="Max Colorbar:",
                disabled=False,
            )

            int_box_f_offset_min = widgets.BoundedIntText(
                value=0,
                min=-2000,
                step=1,
                max=2000,
                description="Max Colorbar:",
                disabled=False,
            )

            int_box_f_offset_max = widgets.BoundedIntText(
                value=100,
                min=-2000,
                step=1,
                max=2000,
                description="Max Colorbar:",
                disabled=False,
            )

            # put both sliders inside a HBox for nice alignment  etc.
            ui_ax = widgets.VBox(
                [slice_slider_ax, int_box_colorbar_min],
                layout=widgets.Layout(display="flex"),
            )

            ui_cor = widgets.VBox(
                [slice_slider_cor, int_box_colorbar_max],
                layout=widgets.Layout(display="flex"),
            )

            ui_sag = widgets.VBox(
                [slice_slider_sag, slider_rep],
                layout=widgets.Layout(display="flex"),
            )

            colorbar_boxes = widgets.HBox((int_box_f_offset_min, int_box_f_offset_max))

            main_ui = widgets.HBox((ui_ax, ui_cor, ui_sag))

            ui = widgets.VBox(
                [main_ui, colorbar_boxes],
                layout=widgets.Layout(display="flex"),
            )

            # connect plotting function with sliders
            # Note: The output is unused here because it is sufficient to display
            #       the HBox (stored in ui variable). Without the ui display you
            #       would have to call the display() function on the output of
            #       widgets.interactive_output().
            out = widgets.interactive_output(
                plot_all,
                {
                    "slice_nr_ax": slice_slider_ax,
                    "slice_nr_cor": slice_slider_cor,
                    "slice_nr_sag": slice_slider_sag,
                    "colorbar_min": int_box_colorbar_min,
                    "colorbar_max": int_box_colorbar_max,
                    "colorbar_f_offset_min": int_box_f_offset_min,
                    "colorbar_f_offset_max": int_box_f_offset_max,
                    "nr_rep": slider_rep,
                },
            )

            # This displays the Hbox containing the slider widgets.
            if display_ui:
                display(ui, out)
            else:
                return ui

    def easy_solve(self, field_map):
        """
        Function tries to calculate singla contributions of all metabolites by simple matrix inversion.
        Not sufficient for real measured signal

        Parameters
        ----------
        signal_data : np.ndarray
            Array containing measured signal.

        Returns
        ----------
        matrix_R : np.ndarray
            Array of shape num_Metabolites containing amplitudes of all metabolites.
        """

        def easy_solve_one_voxel(signal_data, field_offset, echo_spacing):
            matrix_S = signal_data
            num_Echoes = signal_data.shape[0]

            s_corrected = np.zeros((matrix_S.shape[0] * 2))

            for idx_s, s in enumerate(matrix_S):
                t_n = (
                    -((num_Echoes - 1 - (idx_s * 2)) / 2) * echo_spacing
                )  # centers t around 1/TR --> t=0 at 1/TR

                s_buffer = s
                s_buffer = s_buffer * np.exp(-2 * np.pi * field_offset * t_n * 1j)

                s_corrected[idx_s] = np.real(s_buffer)
                s_corrected[idx_s + matrix_S.shape[0]] = np.imag(s_buffer)

                np.squeeze(s_corrected)

            A = self.define_A()

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

            return complex_amplitudes

        # read data dimensions and safe indexes
        _, dim_read, dim_phase, dim_slice, dim_rep = self.signal_data.shape
        index_list = list(np.ndindex(dim_read, dim_phase, dim_slice, dim_rep))
        index_list = tqdm(index_list, desc="Calculating Voxels", leave=True)

        fmap = np.transpose(
            np.rot90(field_map.seq2d[:, :, :, 0], 1, axes=(0, 2)), (1, 2, 0)
        )
        fmap = np.expand_dims(fmap, 0)
        fmap = np.repeat(fmap[..., np.newaxis], dim_rep, axis=-1)

        # dummy function for one voxel, that gets serialized
        def calc_voxel(idx, f_map_value, echo_spacing):
            solution = easy_solve_one_voxel(
                self.signal_data[(..., *idx)],
                f_map_value[(..., *idx)],
                echo_spacing,
            )

            return solution

        # serial solve for mutliple voxels in parallel
        solution_serial = Parallel(n_jobs=4)(
            delayed(calc_voxel)(idx, fmap, self.echo_spacing) for idx in index_list
        )

        # save result as np.array and reshape
        solution_matrix = np.array(solution_serial).reshape(
            dim_read, dim_phase, dim_slice, dim_rep, -1
        )

        self.solution_matrix = np.moveaxis(solution_matrix[:, :, :, :, :-1], -1, 0)

        return self.solution_matrix

    def load_solution(self, load_path, num_transients):
        """
        Loads precalculated solution matrix from file.

        Parameters
        ----------
        load_path : string
            Relative path from notebook location to saved solution matrix.

        Returns
        ----------
        solution_matrix : np.ndarray
            Loaded solution matrix.
        """

        self.num_transients = num_transients

        loaded_matrix = np.load(str(load_path) + ".npy")

        # check if loaded matrix suits initialized parameters
        if loaded_matrix.ndim != 5:
            print("Matrix dimensions have to be of size 5.")
        else:
            self.solution_matrix = loaded_matrix

        return self.solution_matrix

    def correct_chemical_shift(
        self,
    ):
        if isinstance(self.bssfp_object, type(None)):
            raise Exception("chemical shift correction requires bssfp_object")
        else:
            metab_data = self.solution_matrix[self.num_transients :, :, :, :, :]

            acq_BW = self.bssfp_object.method["PVM_EffSWh"]
            num_read_pixels = self.bssfp_object.method["PVM_Matrix"][0]
            receive_f = self.bssfp_object.method["ReadFreqList"]

            from scipy.ndimage import shift

            # init empty array:
            metab_data_shifted_real = np.zeros_like(metab_data)
            metab_data_shifted_imag = np.zeros_like(metab_data)

            # calculate chemical shift based on acquisition BW and frequency
            # offsets
            chem_shifts = [0] * len(self.relative_frequencies)

            bw_per_pixel = acq_BW / num_read_pixels

            for idx, metab_f in enumerate(self.relative_frequencies):
                chem_shifts[idx] = (metab_f - receive_f) / bw_per_pixel

            # loop over metabolites:
            for e in range(metab_data.shape[0]):
                # loop over repetitions:
                for r in range(metab_data.shape[4]):
                    # loop images in slice dim:
                    for s in range(metab_data.shape[3]):
                        image = np.squeeze(metab_data[e, :, :, s, r])
                        metab_data_shifted_real[e, :, :, s, r] = shift(
                            np.real(image), [-chem_shifts[e], 0], mode="wrap"
                        )
                        metab_data_shifted_imag[e, :, :, s, r] = shift(
                            np.imag(image), [-chem_shifts[e], 0], mode="wrap"
                        )
            metab_data_shifted = metab_data_shifted_real + 1j * metab_data_shifted_imag

            for idx in range(self.num_transients):
                metab_data_shifted = np.insert(
                    metab_data_shifted,
                    0,
                    self.solution_matrix[idx, :, :, :, :],
                    axis=0,
                )

            return metab_data_shifted


def define_P(rel_frequencies, num_Echoes, echo_spacing, T2_star_relax=None):
    """
    Function defines complex matrix P containing exponential part of phasors contributing to signal over all echoes.

    Returns
    ----------
    matrix_P : np.ndarray
        Complex matrix of shape num_Echoes x num_metabolites.
    """

    matrix_P = np.zeros((num_Echoes, rel_frequencies.size), dtype=complex)

    for idx_f, f in enumerate(rel_frequencies):
        for idx_t in range(num_Echoes):
            matrix_P[idx_t, idx_f] = np.exp(2 * np.pi * f * idx_t * echo_spacing * 1j)

            if not isinstance(T2_star_relax, type(None)):
                matrix_P[idx_t, idx_f] = matrix_P[idx_t, idx_f] * np.exp(
                    -(idx_t * echo_spacing) / T2_star_relax
                )

    return matrix_P


def calc_NSA(
    rel_frequencies,
    num_Echoes,
    echo_shift_start,
    echo_shift_end,
    step_size,
    T2_star_relax=None,
    save_path=None,
):
    """
    Calculate and plots NSA (number of signal averages, see Rheeder et al.
    Least-Squares Chemical Shift Separation for 13C Metabolic Imaging for math)

    Parameters
    ----------
    rel_frequencies : np.ndarray
        Array of relative metabolite frequencies (in Hz) that shall contribute to the signal.

    num_Echoes : int
        Number of echoes to calculate NSA for.

    echo_shift_start : float
        Min. value of echo (in ms) spacing the NSA is calculated for.

    echo_shift_end : float
        Max. value of echo spacing (in ms) the NSA is calculated for.

    step_size : float
        Step size of echo spacing (in ms) the NSA is calculated for.

    T2_star_relax : float, optional
        Optional T2* decay considered for NSA calulations.

    Returns
    ----------
    nsa_values : np.ndarray
        All nsa values calculated shape: (echo spacing, metabolite)
    """

    echo_shifts = np.arange(
        (echo_shift_start + step_size) / 1000, (echo_shift_end) / 1000, step_size / 1000
    )

    nsa_values = np.zeros((echo_shifts.size, rel_frequencies.size))

    # for echo_spacing in echo_shifts:
    for idx_es, echo_spacing in enumerate(echo_shifts):
        matrix_p = define_P(
            rel_frequencies, num_Echoes, echo_spacing, T2_star_relax=T2_star_relax
        )

        for idx in range(rel_frequencies.size):
            nsa_values[idx_es, idx] = 1 / np.real(
                np.linalg.matrix_power(matrix_p.conj().T @ matrix_p, -1)[idx, idx]
            )

    # plot graph
    fig, ax = plt.subplots(figsize=(15, 6))

    def plot(slider_pos):
        ax.clear()

        if slider_pos != 0:
            average_NSA = (
                np.sum(nsa_values[np.where(echo_shifts < slider_pos / 1000)[0][-1], :])
                / rel_frequencies.size
            )
        else:
            average_NSA = 0

        print(
            "Average NSA of all metabolites for ES="
            + str(slider_pos)
            + "ms: "
            + str(average_NSA)
        )

        for idx in range(rel_frequencies.size):
            ax.plot(
                echo_shifts * 1000,
                nsa_values[:, idx],
                label="Metab. f=" + str(rel_frequencies[idx]) + " Hz",
            )
            ax.set_ylabel("NSA")
            ax.set_xlabel("Echo Spacing in [ms]")
            ax.set_title("NSA for " + str(num_Echoes) + " Echoes")
            ax.legend()

            if slider_pos != 0:
                metab_NSA = nsa_values[
                    np.where(echo_shifts < slider_pos / 1000)[0][-1], idx
                ]
            else:
                metab_NSA = 0

            print(
                "NSA Metab. f=" + str(rel_frequencies[idx]) + " Hz: " + str(metab_NSA)
            )

        ax.axvline(
            slider_pos,
            linewidth=1,
            linestyle="dashed",
        )

        fig.tight_layout()

    slider = widgets.FloatSlider(
        value=1.79,
        min=0,
        max=echo_shift_end,
        step=step_size,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        layout=widgets.Layout(width="50%"),
    )

    out = widgets.interactive_output(
        plot,
        {
            "slider_pos": slider,
        },
    )

    if not isinstance(save_path, type(None)):
        fig.savefig(save_path + ".svg")

    display(slider, out)

    return nsa_values


# def correct_chemical_shift(self, recieve_f, recieve_BW):


# def define_inverse_D(self):
#     """
#     Function defines inverse of complex diagonal matrix D containing all effects affecting the pure signal:
#         - field inhomogeneities shift echoes by a phase (accounted for by variable f_offset)
#         - shortened T2* relaxation
#         - effects of bipolar gradients like phase errors, eddy currents and amplitude modulations (accounted for by variable theta)

#     Returns
#     ----------
#     matrix_D : np.ndarray
#         Complex diagonal matrix of shape num_Echoes x num_Echoes.
#     """

#     matrix_D = np.zeros((self.num_Echoes, self.num_Echoes), dtype=complex)

#     for idx in range(self.num_Echoes):
#         matrix_D[idx, idx] = np.exp(
#             (
#                 -1
#             )  # since D diagonal matrix, multiplication with -1 results in inverse of matrix D (--> (e^x)^-1 = e^(-1*x))
#             * ((1j * 2 * np.pi * self.f_offset) - (1 / self.T2_star))
#             * idx
#             * self.echo_spacing
#             + (((-1) ** (idx + 1)) * self.theta)
#         )

#     return matrix_D

# def define_B(self, estimated_signal_data, P):
#     """
#     Function defines complex matrix B containing first order taylor expansion of the signal equation.
#     Used to calculate delta_T2*, delta_theta, delta_f_offset and delta of signal amplitudes of contributing metabolites.
#     For more information see paper "Multicoil Dixon Chemical Species Separation With an Iterative Least-Squares
#     Estimation Method" by Scott B. Reeder et al.

#     Parameters
#     ----------
#     echo_data : np.ndarray
# Complex array containing all signals/echoes of one measurement location.

#     P : np.ndarray
#         Complex matrix of shape num_Echoes x num_metabolites, containing exponential part of phasors
# contributing to signal over all echoes. Can be defined using function
# define_P().

#     Returns
#     ----------
#     matrix_B : np.ndarray
#         Complex matrix of shape num_Echoes x (3+num_Metabolites)
#     """

#     matrix_B = np.zeros((self.num_Echoes, 3 + self.num_metabolites), dtype=complex)

#     for idx_s, s in enumerate(estimated_signal_data):
#         matrix_B[idx_s, 0] = 1j * 2 * np.pi * idx_s * self.echo_spacing * s
#         matrix_B[idx_s, 1] = (
#             np.sqrt(idx_s * self.echo_spacing * idx_s * self.echo_spacing)
#             / (self.T2_star * self.T2_star)
#         ) * s
#         matrix_B[idx_s, 2] = ((-1) ** idx_s) * s
#         for idx in range(self.num_metabolites):
#             matrix_B[idx_s, idx + 3] = P[idx_s, idx]

#     return matrix_B

# def solve(
#     self,
#     stop_f_offset=1.0,
#     stop_theta=1.0,
#     stop_T2_star=1.0,
#     max_iterations=25,
#     optional_data=None,
# ):
#     """
#     Function solving IDEAL problem for given initial guesses and echo data.
#     Requires correct initialization of IDEAL class.

#     Parameters
#     ----------
#     stop_f_offset : float, optional
#         Stop criterion for field offset. Default is 1.0
#         Stops iterating if delta of field offset is below the given value.

#     stop_theta : float, optional
#         Stop criterion for effects of bipolar gradients. Default is 1.0
#         Stops iterating if delta of theta is below the given value.

#     stop_T2_star : float, optional
#         Stop criterion for T2*. Default is 1.0
#         Stops iterating if delta of T2* is below the given value.

#     max_iterations : int, optional
#         Max number of iterations performed before stop.
#         Default is 25.

#     optional_data : np.ndarray, optional
#         In case IDEAL solver should no perform the calculations on the initial signal_data (e.g on test data),
#         other signal points can be passed here.
#         The optional data should be a one-dimensional array.

#     Returns
#     ----------
#     solution : np.ndarray
#         Contains solution for field_offset, T2_star, theta and all complex amplitudes of phasors of contributing metabolites.
#     """

#     stop_criterion = False
#     num_iterations = 1

#     # build matrix S containing signals of all echoes
#     if not isinstance(optional_data, np.ndarray):
#         signal_matrix = self.signal_data[:, 0]
#     else:
#         signal_matrix = optional_data
#         self.num_Echoes = optional_data.shape[0]

#     # first create initial matricies P and D_0
#     matrix_P = self.define_P()
#     inverse_matrix_D = self.define_inverse_D()

#     # print("Matrix P " + str(matrix_P))

#     # calculate first guess of complex amplitudes based on first guesses of T2*, field offset and theta
#     matrix_R = (
#         np.linalg.matrix_power(np.transpose(matrix_P) @ matrix_P, -1)
#         @ np.transpose(matrix_P)
#         @ inverse_matrix_D
#         @ signal_matrix
#     )

#     while not stop_criterion:

#         print("Itreration number " + str(num_iterations))

#         estimated_signal = matrix_P @ matrix_R

#         matrix_B = self.define_B(estimated_signal, matrix_P)
#         deltas = (
#             np.linalg.matrix_power((np.transpose(matrix_B) @ matrix_B), -1)
#             @ np.transpose(matrix_B)
#             @ (  # here we calculate the difference between the true signal corrected by the estimate of all disturbances and the estimated signal P*R
#                 (inverse_matrix_D @ signal_matrix) - estimated_signal
#             )
#         )
#         # print("Matrix D " + str(inverse_matrix_D))
#         # print(estimated_signal)

#         # update guesses for field_offset, T2*, theta and complex amplitudes
#         self.f_offset = self.f_offset + np.abs(deltas[0])
#         self.T2_star = self.T2_star + np.abs(deltas[1])
#         self.theta = self.theta + deltas[2]

#         for idx, r in enumerate(matrix_R):
#             matrix_R[idx] = r + deltas[idx + 3]

#         # check if stop criterions reached
#         if num_iterations == max_iterations:
#             stop_criterion = True
#         elif (
#             np.abs(deltas[0]) < stop_f_offset
#             and np.abs(deltas[1]) < stop_T2_star
#             and np.abs(deltas[2]) < stop_theta
#         ):
#             stop_criterion = True

#         # reinitialize matrix D with new parameters
#         inverse_matrix_D = self.define_inverse_D()

#         # stop_criterion = True
#         solution = np.array(
#             [self.f_offset, self.T2_star, self.theta], dtype=complex
#         )
#         solution = np.concatenate((solution, matrix_R))

#         print("Solution: " + str(solution))

#         num_iterations += 1

#     # build solution matrix
#     solution = np.array([self.f_offset, self.T2_star, self.theta], dtype=complex)
#     solution = np.concatenate((solution, matrix_R))

#     return solution
