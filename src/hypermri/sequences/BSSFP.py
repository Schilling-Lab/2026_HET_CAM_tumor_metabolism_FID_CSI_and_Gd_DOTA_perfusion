"""Balanced SSFP anatomical sequence implementation for HyperMRI."""

import os
import traceback
from copy import deepcopy

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.patches import Rectangle

from hypermri.core.experiment import MRIExperiment

from ..utils.io.files import load_plot_params
from ..utils.mathematics.calculations import get_histogram
from ..utils.system.logging import init_default_logger
from ..utils.system.objects import add_counter, get_counter
from .base_anatomy import BaseAnatomy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


class BSSFP(BaseAnatomy):
    """Balanced steady-state free precession anatomy scan with coil-aware reconstruction helpers."""

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        """Initialise the BSSFP sequence wrapper and trigger reconstructions."""
        # Extract sequence-specific kwargs before calling parent constructor
        reco_options = kwargs.get("reco_options", {})
        minimal_loading = reco_options.get("minimal_loading", False)

        super().__init__(mri_exp_obj, *args, **kwargs)

        if not minimal_loading and not self._data_array_loaded:
            try:
                self.reconstruct(reco_type="seq2d", **reco_options)
            except Exception:
                traceback.print_exc()

        if not minimal_loading:
            try:
                self.reconstruct(reco_type="raw", **reco_options)
            except Exception:
                traceback.print_exc()

        # print(self.method)
        self.n_echoes = self.get_Nechoes()
        self.Reconstructed_data = None

    def get_matrixDimensions(self) -> int:
        """Get dimensions of acquisition matrix."""
        SpatialDim = self.method["PVM_SpatDimEnum"]

        if SpatialDim == "2D":
            out = self.method["PVM_SPackArrNSlices"]
        elif SpatialDim == "3D":
            out = self.method["PVM_Matrix"][-1]
        else:
            raise NotImplementedError(
                f"PVM_SpatDimEnum as an unknown value: {SpatialDim}."
            )

        return out

    def get_Nechoes(self) -> int:
        """Return the number of echoes acquired by the sequence."""
        if "PVM_NEchoImages" in self.method:
            return self.method["PVM_NEchoImages"]
        else:
            return self.method["PVM_NMovieFrames"]

    def reco_scrambled_phase(self, input_data=None, path_to_coord=None, plot_db=False):
        """
        Reconstruct bSSFP data that was acquired with scrambled phase encode order

        Parameters
        ----------
        input_data: .fid file of the acquired data

        """
        try:
            from matplotlib import colormaps

            get_cmap = colormaps.get_cmap  # Matplotlib 3.10+
        except ImportError:
            from matplotlib.cm import get_cmap  # Matplotlib <3.10
        if input_data is None:
            input_data = self.fid
        else:
            pass

        if path_to_coord is None:
            raise Exception("path_to_coord parameter is empty")
        else:
            pass

        # load path_to_coord:
        import pandas as pd

        # Load the coordinates
        try:
            df = pd.read_csv(path_to_coord, sep=" ", header=None, names=["x", "y"])
        except BaseException:
            raise Exception("Coordinates could not be loaded!")

        # parameters:
        mat = self.matrix
        n_repetitions = self.n_repetitions

        data = input_data.copy()

        # not sure wether it's mat[2], mat[1] or mat[1], mat[2]:
        # factor of 4 will be removed later:
        d = np.reshape(data, [n_repetitions, mat[2], mat[1], 4, mat[0]])

        # only use first out of 4 entries (rest should be empty):

        # If nr is 1, add an extra dimension to d
        if n_repetitions == 1:
            d = d[..., np.newaxis]

        # not sure wether it's mat[2], mat[1] or mat[1], mat[2]:
        num_bins = [mat[2], mat[1]]
        matrix = np.zeros_like(d)

        if plot_db is True:
            import matplotlib.colors as colors

            # Create a colormap
            cmap = get_cmap("jet")
            norm = colors.Normalize(vmin=0, vmax=len(df) - 1)

            # Create the plot
            fig, ax = plt.subplots()

            # Scatter plot with points connected and colored based on their
            # index
            for i in range(len(df) - 1):
                ax.plot(
                    df.iloc[i : i + 2, 0], df.iloc[i : i + 2, 1], color=cmap(norm(i))
                )

            # Display the plot
            plt.show()

        # Use binned_statistic_2d to sort the 1D data into a 2D grid based on
        # the x and y coordinates
        from scipy.stats import binned_statistic_2d

        if n_repetitions == 1:
            for r in range(n_repetitions):
                for k in range(mat[0]):
                    statistic, x_edge, y_edge, binnumber = binned_statistic_2d(
                        df["y"],
                        df["x"],
                        values=np.squeeze(d[r, :, :, 0, k]).flatten(),
                        statistic="mean",
                        bins=num_bins,
                    )
                    matrix[r, :, :, 0, k] = statistic[..., np.newaxis]
        else:
            for r in range(n_repetitions):
                for k in range(mat[0]):
                    statistic, x_edge, y_edge, binnumber = binned_statistic_2d(
                        df["y"],
                        df["x"],
                        values=np.squeeze(d[r, :, :, 0, k]).flatten(),
                        statistic="mean",
                        bins=num_bins,
                    )
                    matrix[r, :, :, 0, k] = statistic

        num_unscramle_counts = get_counter(
            data_obj=self,
            counter_name="unscramble",
        )

        # if was not saved yet:
        if num_unscramle_counts == 0:
            # save sorted matrix as fid
            matrix = np.reshape(matrix, (mat[1] * mat[2] * n_repetitions, -1))
            matrix = self.postprocess_fid_file(input_data=matrix)
            self.fid = matrix

            add_counter(
                data_obj=self,
                counter_name="unscramble",
                n_counts=1,
            )
        else:
            # reshape data
            matrix = np.reshape(matrix, (mat[1] * mat[2] * n_repetitions, -1))
            # remove 0s
            matrix = self.postprocess_fid_file(input_data=matrix)

        # bring data into [echoes, z, y, x, repetitions, channels] format
        matrix = np.reshape(
            matrix,
            (
                self.n_repetitions,
                self.matrix[1],
                self.matrix[2],
                self.matrix[0],
            ),
        )
        print(matrix.shape)
        matrix = np.expand_dims(matrix, axis=(4, 5))
        print(matrix.shape)
        matrix = np.transpose(matrix, (4, 3, 1, 2, 0, 5))

        self.image = self.bssfp_fft(input_data=matrix, axes=(1, 2, 3))

        return matrix

    def define_grid(self, fov=None, mat=None):
        """
        Define a grid with the points as the centers of the voxels.

        Parameters
        ----------
        fov:
            Field of View in mm [fovx, fovy, fovz]
        mat:
            matrix size in [fovx, fovy, fovz]

        Returns
        -------
        nd-arrays, defining center of voxels

        """

        if fov is None:
            fov = np.array(self.define_imageFOV_parameters())
        if mat is None:
            mat = np.array(self.define_imagematrix_parameters())
        res = fov / mat

        # init:
        ext_1 = ext_2 = ext_3 = None
        if (len(fov) > 0) and (len(mat) > 0):
            ext_1 = np.linspace(
                -fov[0] / 2 + res[0] / 2, fov[0] / 2 - res[0] / 2, mat[0]
            )
        if (len(fov) > 1) and (len(mat) > 1):
            ext_2 = np.linspace(
                -fov[1] / 2 + res[1] / 2, fov[1] / 2 - res[1] / 2, mat[1]
            )
        if (len(fov) > 2) and (len(mat) > 2):
            ext_3 = np.linspace(
                -fov[2] / 2 + res[2] / 2, fov[2] / 2 - res[2] / 2, mat[2]
            )

        return ext_1, ext_2, ext_3

    def interpolate_bssfp(
        self,
        bssfp_data=None,
        interp_size=None,
        interp_method="linear",
        use_multiprocessing=False,
        dtype=None,
        number_of_cpu_cores=None,
    ):
        """
         Interpolate the input data (bssfp_data) onto a new grid in the same range as
         bssfp_data but points in each dimension are described by interp_size. Uses scipy's RegularGridInterpolator
         method. Multithreading can be activated.

         Parameters
         ----------
         bssfp_data: ND array
             Expected to have the order Echoes, Read, phase, slice, repetitions, channels
         interp_size: tuple
             Describes the interpolated data size.
             Has to be 6D: number of echhoes, points in  Read, phase, slice; number of repetitions, channels
         interp_method: str
             interpolation method that is used. Supported are "linear", "nearest",
             # if scipy==1.10.0: "slinear", "cubic", "quintic" and "pchip".
         use_multiprocessing: bool
             toggles multiprocessing
         dtype: strs
             should give you the option to change the dtype of the to be interpolated array to reduce the size

         number_of_cpu_cores:
             Number of CPU cores to if multi processing is desired

         Examples
         --------
         # interpolate 1st and 2nd dimension onto a 10x higher grid:

         # get extend in x-y-z:

         >>> x, y, z = dnp_bssfp.define_grid()
         >>> interp_factor = 10

         >>> xres = np.linspace(x[0], x[-1], interp_factor*len(x))
         >>> yres = np.linspace(y[0], y[-1], interp_factor*len(y))

         >>> zres = np.linspace(z[0], z[-1], len(z))
         >>> t = np.arange(0,dnp_bssfp_pv_reco.shape[4],1)

        >>> test_nearest = dnp_bssfp.interpolate_bssfp(bssfp_data=dnp_bssfp_pv_reco,
        >>>                     interp_size=(1,len(xres),len(yres),len(zres),len(t),2),
        >>>                    interp_method="nearest")


         >>> plt.figure()
         >>> for k in range(10):
         >>> ax = plt.subplot(3,4,k+1)
         >>> ax.imshow(np.squeeze(np.mean(test_linear[:,:,:,k,:,:], axis=3)))

         # interpolate every 2nd image (lactate) onto a 4x higher time resoltion:

         # get extend in x-y-z:

         >>> x, y, z = dnp_bssfp.define_grid()
         >>> interp_factor = 4

         >>> xres = np.linspace(x[0], x[-1], len(x))
         >>> yres = np.linspace(y[0], y[-1], len(y))
         >>> zres = np.linspace(z[0], z[-1], len(z))

         >>> t = np.arange(0,dnp_bssfp_pv_reco.shape[4],1)
         >>> tres = np.linspace(z[0], t[-1], interp_factor*len(t))


         >>> test_lac_interp = dnp_bssfp.interpolate_bssfp(bssfp_data=dnp_bssfp_pv_reco[:,:,:,:,::2,:],
         >>>                     interp_size=(1,len(xres),len(yres),len(zres),len(tres),2),
         >>>                     interp_method="linear")
         >>> plt.figure()
         >>> reps = range(100)
         >>> for k in reps:
         >>> ax = plt.subplot(10,10,k+1)
         >>> ax.imshow(np.squeeze(test_lac_interp[0,:,:,5,k+150,0]))

        """
        import numpy as np
        from scipy.interpolate import interpn

        if use_multiprocessing:
            try:
                from joblib import Parallel, cpu_count, delayed
                from tqdm.auto import tqdm

                # try to set the number of usable cpu cores to the amount of
                # available cores
                if number_of_cpu_cores is None:
                    # int-divide by 2 because hyperthreading cores dont
                    # count/help:
                    number_of_cpu_cores = cpu_count() // 2
                else:
                    pass

            except BaseException:
                use_multiprocessing = False

        import time

        # init empty output
        input_data = None

        # init empty output
        if bssfp_data is None:
            input_data = self.Reconstructed_data
        else:
            input_data = bssfp_data

        if interp_size is None:
            Warning("Don't know what to do")
            interp_size = input_data.shape

        # check method:
        allowed_interp_methods = ["linear", "nearest", "splinef2d", "cubic"]
        if interp_method not in allowed_interp_methods:
            logger.critical(
                "uknown interpolation method: "
                + interp_method
                + ", using linear instead."
            )
            interp_method = "linear"

        if dtype is None:
            dtype = bssfp_data.dtype

        # duration:
        t = np.arange(0, input_data.shape[4], 1)
        t_num = input_data.shape[4]

        # cant loop over repetitions if this also should
        # be interpolated ...
        if len(t) == interp_size[4]:
            pass
        else:
            use_multiprocessing = False

        # get extend in x-y-z:
        x, y, z = self.define_grid()
        # number of echoes:
        e_num = input_data.shape[0]
        # range of echoes:
        e = np.linspace(1, e_num, input_data.shape[0])
        # number of channels:
        c_num = input_data.shape[5]
        # range of channels:
        c = np.linspace(1, c_num, input_data.shape[5])

        # points to interpolate onto:
        # echoes:
        eres = np.linspace(1, e[-1], interp_size[0])
        # points in space:
        xres = np.linspace(x[0], x[-1], interp_size[1])
        yres = np.linspace(y[0], y[-1], interp_size[2])
        zres = np.linspace(z[0], z[-1], interp_size[3])
        # repetitions:
        tres = np.linspace(t[0], t[-1], interp_size[4])

        # channels:
        if e_num == 1:
            eres = np.linspace(1, e_num, input_data.shape[0])
        else:
            eres = np.linspace(1, e[-1], interp_size[0])

        # channels:
        if c_num == 1:
            cres = np.linspace(1, c_num, input_data.shape[5])
        else:
            cres = c

        # get the start time
        st = time.time()

        if use_multiprocessing:
            # generate grids (to interpolate onto)
            egres, xgres, ygres, zgres, cgres = np.meshgrid(
                eres, xres, yres, zres, cres, indexing="ij"
            )
            # init empty array:
            interpolated_data = np.zeros(
                (len(eres), len(xres), len(yres), len(zres), len(t), len(cres)),
                dtype=dtype,
            )

            # index list (time range)
            index_list = list(np.ndindex(len(t)))

            # create progress bar
            index_list = tqdm(index_list, desc="interpolation progress", leave=True)

            # define interpolationm functionm
            def interpolate_image(it):
                # the conditions speed up the interpolation quite a bit
                # if there are less dimensions (1 echo e.g.)

                # if more than 1 echo and more than 1 channel:
                if e_num > 1 and c_num > 1:
                    interpolated_data_timepoint = np.squeeze(
                        interpn(
                            points=(e, x, y, z, c),
                            values=np.squeeze(input_data[:, :, :, :, it, :]),
                            xi=(egres, xgres, ygres, zgres, cgres),
                            method=interp_method,
                        )
                    )
                # if more than 1 echo and 1 channel:
                elif e_num > 1 and c_num == 0:
                    interpolated_data_timepoint = np.squeeze(
                        interpn(
                            points=(e, x, y, z),
                            values=np.squeeze(input_data[:, :, :, :, it, 0]),
                            xi=(egres, xgres, ygres, zgres),
                            method=interp_method,
                        )
                    )
                # if 1 echo and more than 1 channel:
                elif e_num == 1 and c_num > 1:
                    interpolated_data_timepoint = np.squeeze(
                        interpn(
                            points=(x, y, z, c),
                            values=np.squeeze(input_data[0, :, :, :, it, :]),
                            xi=(xgres, ygres, zgres, cgres),
                            method=interp_method,
                        )
                    )
                # if 1 echo and 1 channel:
                elif e_num == 1 and c_num == 1:
                    interpolated_data_timepoint = np.squeeze(
                        interpn(
                            points=(x, y, z),
                            values=np.squeeze(input_data[0, :, :, :, it, 0]),
                            xi=(xgres, ygres, zgres),
                            method=interp_method,
                        )
                    )

                return interpolated_data_timepoint

            # interpolate multiple timesteps in parallel:
            interpolated_data_tuple = Parallel(n_jobs=number_of_cpu_cores)(
                delayed(interpolate_image)(it) for it in index_list
            )

            # if more than 1 echo and more than 1 channel:
            for it in range(len(t)):
                if e_num > 1 and c_num > 1:
                    interpolated_data[:, :, :, :, it, :] = interpolated_data_tuple[it]
                # if more than 1 echo and 1 channel:
                elif e_num > 1 and c_num == 0:
                    interpolated_data[:, :, :, :, it, 0] = interpolated_data_tuple[it]
                # if 1 echo and more than 1 channel:
                elif e_num == 1 and c_num > 1:
                    interpolated_data[0, :, :, :, it, :] = interpolated_data_tuple[it]
                # if 1 echo and 1 channel:
                elif e_num == 1 and c_num == 1:
                    interpolated_data[0, :, :, :, it, 0] = interpolated_data_tuple[it]

        elif use_multiprocessing == False and (t_num > 1):
            # generate grids (to interpolate onto)
            egres, xgres, ygres, zgres, tgres, cgres = np.meshgrid(
                eres, xres, yres, zres, tres, cres, indexing="ij"
            )

            # init empty array:
            interpolated_data = np.zeros(
                (
                    len(eres),
                    len(xres),
                    len(yres),
                    len(zres),
                    len(tres),
                    len(cres),
                ),
                dtype=dtype,
            )
            # if more than 1 echo and more than 1 channel:
            if e_num > 1 and c_num > 1:
                interpolated_data = np.squeeze(
                    interpn(
                        points=(e, x, y, z, t, c),
                        values=input_data[:, :, :, :, :, :],
                        xi=(egres, xgres, ygres, zgres, tgres, cgres),
                        method=interp_method,
                    )
                )
            # if more than 1 echo and 1 channel:
            elif e_num > 1 and c_num == 0:
                interpolated_data[:, :, :, :, :, 0] = np.squeeze(
                    interpn(
                        points=(e, x, y, z, t),
                        values=input_data[:, :, :, :, :, 0],
                        xi=(egres, xgres, ygres, zgres, tgres),
                        method=interp_method,
                    )
                )

            # if 1 echo and more than 1 channel:
            elif e_num == 1 and c_num > 1:
                interpolated_data[0, :, :, :, :, :] = np.squeeze(
                    interpn(
                        points=(x, y, z, t, c),
                        values=np.squeeze(input_data[0, :, :, :, :, :]),
                        xi=(xgres, ygres, zgres, tgres, cgres),
                        method=interp_method,
                    )
                )
            # if 1 echo and 1 channel:
            elif e_num == 1 and c_num == 1:
                interpolated_data[0, :, :, :, :, 0] = np.squeeze(
                    interpn(
                        points=(x, y, z, t),
                        values=input_data[0, :, :, :, :, 0],
                        xi=(xgres, ygres, zgres, tgres),
                        method=interp_method,
                    )
                )

        elif use_multiprocessing == False and (t_num == 1):
            # generate grids (to interpolate onto)
            egres, xgres, ygres, zgres, cgres = np.meshgrid(
                eres, xres, yres, zres, cres, indexing="ij"
            )
            # init empty array:
            interpolated_data = np.zeros(
                (len(eres), len(xres), len(yres), len(zres), len(tres), len(cres)),
                dtype=dtype,
            )
            # if more than 1 echo and more than 1 channel:
            if e_num > 1 and c_num > 1:
                interpolated_data[:, :, :, :, 0, :] = np.squeeze(
                    interpn(
                        points=(e, x, y, z, c),
                        values=input_data[:, :, :, :, 0, :],
                        xi=(egres, xgres, ygres, zgres, cgres),
                        method=interp_method,
                    )
                )

            # if more than 1 echo and  1 channel:
            elif e_num > 1 and c_num == 1:
                interpolated_data[:, :, :, :, 0, 0] = np.squeeze(
                    interpn(
                        points=(
                            e,
                            x,
                            y,
                            z,
                        ),
                        values=input_data[:, :, :, :, 0, 0],
                        xi=(egres, xgres, ygres, zgres),
                        method=interp_method,
                    )
                )
            # if 1 echo and more than 1 channel:
            elif e_num == 1 and c_num > 1:
                for it in range(len(t)):
                    interpolated_data[0, :, :, :, 0, :] = np.squeeze(
                        interpn(
                            points=(x, y, z, c),
                            values=np.squeeze(input_data[0, :, :, :, 0, :]),
                            xi=(xgres, ygres, zgres, cgres),
                            method=interp_method,
                        )
                    )
            # if 1 echo and 1 channel:
            elif e_num == 1 and c_num == 1:
                for it in range(len(t)):
                    interpolated_data[0, :, :, :, 0, 0] = np.squeeze(
                        interpn(
                            points=(x, y, z),
                            values=input_data[0, :, :, :, 0, 0],
                            xi=(xgres, ygres, zgres),
                            method=interp_method,
                        )
                    )

        # get the execution time
        et = time.time()
        elapsed_time = et - st
        # print excution time:
        logger.debug("Execution time:", elapsed_time, "seconds")

        return interpolated_data

    def calc_timeaxis(self, start_with_0=True):
        """
        Calculate the time range [in s] on which the bssfp data was acquired.

        Parameters
        ----------
        start_with_0 (bool)
            wether the time axis should start with 0 or 1 image repetition time

        Returns
        -------
            Time axis [in seconds]

        """
        time_axis = None
        # number of repetitions
        n_repetitions = self.n_repetitions
        # scan time per image in  seconds
        image_repetition_time = self.method["PVM_ScanTime"] / n_repetitions / 1000.0

        # time range start with 0:
        if start_with_0:
            time_axis = np.linspace(
                0, (n_repetitions - 1) * image_repetition_time, n_repetitions
            )
            # time range start with 1 image reptition time:
        else:
            time_axis = np.linspace(
                image_repetition_time,
                n_repetitions * image_repetition_time,
                n_repetitions,
            )

        return time_axis

    def calc_AUC(
        self,
        signal=None,
        time_axis=None,
        axlist=None,
        fig=None,
        display_ui=True,
        pyr_ind=0,
        mask_id=None,
        mask_key=None,
        signal_range_input=[None, None],
        noise_range_input=[None, None],
        noise_signal=None,
        apply_filter=True,
        savepath=None,
        savename=None,
    ):
        """
        Open a GUI that lets you set the range to calculate the signal range
        and the noise range of a signal vs. time curve. automatically calculates
        the area under the curve (AUC)


        Parameters
        ----------
        signal: 1xN (1 metabolite) or 2xN (2 metabolites) array. Usually extracted
            from a masked bssfp dataset (1xN) or (2xN)
        timeaxis: Time axis of the metabolites. If no input given, integer 1:N
            are used. (1xN)
        axlist: list of 2 pyplot axis: can be generated via
            fig, (ax1, ax2) = plt.subplots(1, 2,tight_layout=True)
            and set via:
            axlist = [ax1, ax2]
        pyr_ind: either 0 or 1, depending on which metabolite (pyr or lac) has been acquired
            first
        mask_id: str
            can be used to indicate from which mask the signal curves are.
        mask_key: str
            will be used to store the auc in a auc_ratio_dict under this key
        signal_range_input: list
            can store indices corresponding to starting/ending points for signal summation
        noise_range_input: list
            an store indices corresponding to starting/ending points for noise subtraction
        noise_signal: np.ndarray
            noise signal time curve that is subtracted from the signal.
        appyly_filter: bool
            If true, a savitzky Golay filter is applied to smooth the noise data

        Returns
        -------
        area under the curve of pyruvate, lactate and the ratio of lac/pyr
        auc_pyr, auc_lac, auc_ratio

        Example Usage:
        --------------
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,tight_layout=True, figsize=(12,4))

        roi_key = "kidneyL"
        phip_bssfp.calc_AUC(signal=(np.abs(phip_signal_pyr[roi_key]),
                                    np.abs(phip_signal_lac[roi_key])),
                            axlist = [ax1, ax2, ax3, ax4],
                            mask_key=roi_key,
                            signal_range_input=[4,51],
                            noise_range_input=[53,73],
                            noise_signal=np.abs(phip_signal_pyr['outside_ref']),
                            apply_filter=True)

        """

        # no signal --> exit
        if signal is None:
            return False

        # if 6D take average along spatial dimensions:
        if np.ndim(signal) == 6:
            signal = np.squeeze(np.nanmean(signal, axis=(1, 2, 3)))
        # if 7D take average along spatial dimensions (pyr, lac in first
        # dimension):
        if np.ndim(signal) == 7:
            signal = np.squeeze(np.nanmean(signal, axis=(2, 3, 4)))

        # if 6D take average along spatial dimensions:
        if np.ndim(noise_signal) == 6:
            noise_signal = np.squeeze(np.nanmean(noise_signal, axis=(1, 2, 3)))

        if time_axis is None:
            print(np.ndim(signal))
            if np.ndim(signal) == 1:
                time_axis = np.linspace(0, signal.shape[0] - 1, signal.shape[0])
            elif np.ndim(signal) == 2:
                time_axis = np.linspace(0, len(signal[0]) - 1, len(signal[0]))
            else:
                Warning("ndims signal has to be 1D or 2D!")
                return False

        if axlist is None:
            return False
        global auc_pyr, auc_lac, auc_ratio
        auc_pyr = 0.0
        auc_lac = 0.0
        auc_ratio = 0.0

        # define function that save the calculated AUCs:
        def save_auc(
            self,
            auc_lac=np.nan,
            auc_pyr=np.nan,
            auc_ratio=np.nan,
            mask_id=None,
            mask_key=None,
            signal_range=None,
            noise_range=None,
            auc_lac_dyn_noise=np.nan,
            auc_pyr_dyn_noise=np.nan,
            auc_ratio_dyn_noise=np.nan,
        ):
            # if a mask_id has been passed, store the calculated AUC under this
            # mask_id
            if mask_id is None:
                self.auc_ratio_nomask = auc_ratio
                self.auc_lac_nomask = auc_lac
                self.auc_pyr_nomask = auc_pyr

                # try to save the AUCs caluclated by subtracting the time
                # curves:
                if noise_signal is not None:
                    self.auc_ratio_dyn_noise_nomask = auc_ratio_dyn_noise
                    self.auc_lac_dyn_noise_nomask = auc_lac_dyn_noise
                    self.auc_pyr_dyn_noise_nomask = auc_pyr_dyn_noise
            # try to save the AUCs under the mask IDs
            else:
                # try to save AUC with mask ID:
                try:
                    self.auc_ratio[mask_id] = auc_ratio
                    self.auc_lac[mask_id] = auc_lac
                    self.auc_pyr[mask_id] = auc_pyr
                except BaseException:
                    try:
                        # init empty array for AUCs
                        setattr(self, "auc_ratio", np.zeros((10, 1)))
                        setattr(self, "auc_lac", np.zeros((10, 1)))
                        setattr(self, "auc_pyr", np.zeros((10, 1)))
                        # try to save AUC with mask ID:
                        # try to save AUC with mask ID:
                        self.auc_ratio[mask_id] = auc_ratio
                        self.auc_lac[mask_id] = auc_lac
                        self.auc_pyr[mask_id] = auc_pyr
                    except BaseException:
                        # save AUC without mask ID:
                        self.auc_ratio_nomask = auc_ratio
                        self.auc_lac_nomask = auc_lac
                        self.auc_pyr_nomask = auc_pyr
                # try to save the AUCs caluclated by subtracting the time
                # curves:
                if noise_signal is not None:
                    # try to save AUC with mask ID:
                    try:
                        self.auc_ratio_dyn_noise[mask_id] = auc_ratio_dyn_noise
                        self.auc_lac_dyn_noise[mask_id] = auc_lac_dyn_noise
                        self.auc_pyr_dyn_noise[mask_id] = auc_pyr_dyn_noise
                    except BaseException:
                        try:
                            # init empty array for AUCs
                            setattr(self, "auc_ratio_dyn_noise", np.zeros((10, 1)))
                            setattr(self, "auc_lac_dyn_noise", np.zeros((10, 1)))
                            setattr(self, "auc_pyr_dyn_noise", np.zeros((10, 1)))
                            # try to save AUC with mask ID:
                            # try to save AUC with mask ID:
                            self.auc_ratio_dyn_noise[mask_id] = auc_ratio_dyn_noise
                            self.auc_lac_dyn_noise[mask_id] = auc_lac_dyn_noise
                            self.auc_pyr_dyn_noise[mask_id] = auc_pyr_dyn_noise
                        except BaseException:
                            # save AUC without mask ID:
                            self.auc_ratio_dyn_noise_nomask = auc_ratio_dyn_noise
                            self.auc_lac_dyn_noise_nomask = auc_lac_dyn_noise
                            self.auc_pyr_dyn_noise_nomask = auc_pyr_dyn_noise

            if mask_key is not None:
                # try to save AUCs in dic via passed key:
                try:
                    self.auc_ratio_dic[mask_key] = auc_ratio
                    self.auc_lac_dic[mask_key] = auc_lac
                    self.auc_pyr_dic[mask_key] = auc_pyr
                except BaseException:
                    try:
                        # init empty dictonary:
                        setattr(self, "auc_ratio_dic", {})
                        setattr(self, "auc_lac_dic", {})
                        setattr(self, "auc_pyr_dic", {})
                        # save AUCs:
                        self.auc_ratio_dic[mask_key] = auc_ratio
                        self.auc_lac_dic[mask_key] = auc_lac
                        self.auc_pyr_dic[mask_key] = auc_pyr
                    except BaseException:
                        pass

                # try to save the AUCs caluclated by subtracting the time
                # curves:
                if noise_signal is not None:
                    # try to save AUCs in dic via passed key:
                    try:
                        self.auc_ratio_dyn_noise_dic[mask_key] = auc_ratio_dyn_noise
                        self.auc_lac_dyn_noise_dic[mask_key] = auc_lac_dyn_noise
                        self.auc_pyr_dyn_noise_dic[mask_key] = auc_pyr_dyn_noise
                    except BaseException:
                        try:
                            # init empty dictonary:
                            setattr(self, "auc_ratio_dyn_noise_dic", {})
                            setattr(self, "auc_lac_dyn_noise_dic", {})
                            setattr(self, "auc_pyr_dyn_noise_dic", {})
                            # save AUCs:
                            self.auc_ratio__dyn_noisedic[mask_key] = auc_ratio_dyn_noise
                            self.auc_lac_dyn_noise_dic[mask_key] = auc_lac_dyn_noise
                            self.auc_pyr_dyn_noise_dic[mask_key] = auc_pyr_dyn_noise
                        except BaseException:
                            pass

                # try to save signal/noise range via passed key:
                try:
                    self.auc_range_dic = {
                        "mask_key": {
                            "signal_range": signal_range,
                            "noise_range": noise_range,
                        }
                    }
                except BaseException:
                    try:
                        # init empty dictonary:
                        setattr(self, "auc_range_dic", {})
                        # save AUC ranges:
                        self.auc_range_dic = {
                            "mask_key": {
                                "signal_range": signal_range,
                                "noise_range": noise_range,
                            }
                        }
                    except BaseException:
                        pass

            else:
                pass

        # --------------------------- define plot function --------------------
        def plot_auc(
            signal_range=[0, 1],
            noise_range=[0, 1],
            subtract_noise=False,
            save_csv=False,
        ):
            # clear axis:
            for ax in axlist:
                ax.clear()

            fig.suptitle(savename)

            # Initialize dynamic noise AUC variables
            auc_lac_dyn_noise = np.nan
            auc_pyr_dyn_noise = np.nan
            auc_ratio_dyn_noise = np.nan

            # turn selected range into indices:
            noise_ind = [0] * 2
            noise_ind[0] = np.argmin(np.abs(time_axis - noise_range[0]))
            noise_ind[1] = np.argmin(np.abs(time_axis - noise_range[1]))

            signal_ind = [0] * 2
            signal_ind[0] = np.argmin(np.abs(time_axis - signal_range[0]))
            signal_ind[1] = np.argmin(np.abs(time_axis - signal_range[1]))

            # calc mean over noise range
            mean_noise = [0.0] * np.ndim(signal)
            if subtract_noise:
                # if pyruvate and lactate:
                if np.ndim(signal) == 2:
                    for s in range(np.ndim(signal)):
                        mean_noise[s] = np.mean(signal[s][noise_ind[0] : noise_ind[1]])
                elif np.ndim(signal) == 1:
                    mean_noise = np.mean(signal[noise_ind[0] : noise_ind[1]])
                else:
                    logger.critical("input signal has to be 1D or 2D")
            # calc mean average, use all signa channels:
            mean_noise = np.mean(mean_noise)
            signal_no_const_noise = signal - mean_noise

            # integrate signal:
            integrated_signal = [0.0] * (np.ndim(signal))
            for s in range(np.ndim(signal)):
                if np.ndim(signal) == 2:
                    integrated_signal[s] = np.sum(
                        signal_no_const_noise[s][signal_ind[0] : signal_ind[1]]
                    )
                elif np.ndim(signal) == 1:
                    integrated_signal = np.sum(
                        signal_no_const_noise[signal_ind[0] : signal_ind[1]]
                    )
                else:
                    logger.critical("input signal has to be 1D or 2D")

            if np.ndim(signal) > 1:
                if pyr_ind == 0:
                    auc_lac = integrated_signal[pyr_ind + 1]
                    auc_pyr = integrated_signal[pyr_ind]
                    auc_ratio = auc_lac / auc_pyr
                elif pyr_ind == 1:
                    auc_lac = integrated_signal[pyr_ind - 1]
                    auc_pyr = integrated_signal[pyr_ind]
                    auc_ratio = auc_lac / auc_pyr
                else:
                    Warning("Don't understand the indexing")
                    auc_lac = integrated_signal
                    auc_pyr = integrated_signal
                    auc_ratio = 1.0

            elif np.ndim(signal) == 1:
                auc_lac = 0.0
                auc_pyr = integrated_signal
                auc_ratio = 0.0
            else:
                logger.critical("input signal has to be 1D or 2D")

            logger.critical("const. noise AUC pyr : " + str(auc_pyr))
            logger.critical("const. noise AUC lac : " + str(auc_lac))
            logger.critical("const. noise AUC lac/pyr : " + str(auc_ratio))

            # subplot 1 -------------------------------------------------------
            # plot signal curves:
            if np.ndim(signal) == 2:
                for s in range(np.ndim(signal)):
                    axlist[0].plot(time_axis, signal[s])
            elif np.ndim(signal) == 1:
                axlist[0].plot(time_axis, signal)
            else:
                logger.critical("input signal has to be 1D or 2D")

            for r in range(2):
                # plot signal range:
                axlist[0].axvline(
                    signal_range[r],
                    color="black",
                    linewidth=1,
                    linestyle="dashed",
                    label="noise range",
                )
                # plot noise range:
                axlist[0].axvline(
                    noise_range[r],
                    color="red",
                    linewidth=1,
                    linestyle="dashed",
                    label="signal range",
                )

            # plot mean noise:
            axlist[0].axhline(
                mean_noise,
                color="red",
                linewidth=1,
                linestyle="dashed",
                label="signal range",
            )

            logger.critical(np.ndim(signal))
            # subplot 2 -------------------------------------------------------
            # plot integrated signal:
            if np.ndim(signal) == 2:
                for s in range(np.ndim(signal)):
                    axlist[1].plot(time_axis, np.cumsum(signal[s] - mean_noise))
                    axlist[1].axhline(
                        np.cumsum(signal[s] - mean_noise)[-1],
                        color="red",
                        linewidth=1,
                        linestyle="dashed",
                    )
            elif np.ndim(signal) == 1:
                axlist[1].plot(time_axis, np.cumsum(signal - mean_noise))
                axlist[1].axhline(
                    np.cumsum(signal - mean_noise)[-1],
                    color="red",
                    linewidth=1,
                    linestyle="dashed",
                )
            else:
                logger.critical("input signal has to be 1D or 2D")

            if save_button.value:
                filename = os.path.join(savepath, savename + "_time_and_aucs.csv")
                if np.ndim(signal) == 2:
                    data_to_save = np.column_stack(
                        (time_axis, signal[0] - mean_noise, signal[1] - mean_noise)
                    )
                    np.savetxt(filename, data_to_save, fmt="%f", delimiter=",")
                else:
                    data_to_save = np.column_stack((time_axis, signal - mean_noise))
                    np.savetxt(filename, data_to_save, fmt="%f", delimiter=",")

                plt.savefig(os.path.join(savepath, savename + "_time_and_aucs.png"))

            for r in range(2):
                # plot signal range:
                axlist[1].axvline(
                    signal_range[r],
                    color="black",
                    linewidth=1,
                    linestyle="dashed",
                )

            # axlist[0].legend("noise range", "signal range")  # ,
            # title="AUC ratio " + str(np.round(auc_ratio, 3)),
            # )
            # set labels signal over time:
            axlist[0].set_title("Signal over repetitions")
            axlist[0].set_ylabel("I [a.u.] ")
            axlist[0].set_xlabel("Time since start [s] ")
            # set labels integrated signal over time:
            axlist[1].set_xlabel("Time since start [s] ")
            axlist[1].legend()
            if mask_key is not None:
                axlist[1].set_title(mask_key + " - integr. signal")
            else:
                axlist[1].set_title("integrated signal")
            # draw grid:
            axlist[1].grid(color="r", linestyle="-", linewidth=0.25, which="major")
            axlist[1].grid(color="r", linestyle="--", linewidth=0.25, which="minor")

            # plot the signal with the loaded noise_range subtracted:
            if noise_signal is not None:
                # if filter should be applied
                if apply_filter is True:
                    # imort savitzky-golay filter:
                    from scipy.signal import savgol_filter

                    noise_signal_smooth = savgol_filter(noise_signal, 30, 3)
                else:
                    # don't apply filter:
                    noise_signal_smooth = noise_signal

                # subtract dymamic (loaded) noise:
                signal_no_dyn_noise = signal - noise_signal_smooth
                # init empty array:
                integrated_signal_dyn_noise = [0.0] * (np.ndim(signal))
                # integrate signal:
                if np.ndim(signal) == 2:
                    for s in range(np.ndim(signal)):
                        integrated_signal_dyn_noise[s] = np.sum(signal_no_dyn_noise[s])
                elif np.ndim(signal) == 1:
                    integrated_signal_dyn_noise = np.sum(signal_no_dyn_noise)
                else:
                    logger.critical("input signal has to be 1D or 2D")

                # plot signal with subtracted noise:
                if len(axlist) > 2:
                    if np.ndim(signal) == 2:
                        # plot signal subtracted noise:
                        for s in range(np.ndim(signal)):
                            axlist[2].plot(time_axis, signal_no_dyn_noise[s])
                            axlist[2].set_title("Signal (subtr. dyn noise)")
                            axlist[2].set_ylabel("I [a.u.]")
                            axlist[2].set_xlabel("Time [s]")
                    elif np.ndim(signal) == 1:
                        # plot signal subtracted noise:
                        axlist[2].plot(time_axis, signal_no_dyn_noise)
                        axlist[2].set_title("Signal (subtr. dyn noise)")
                        axlist[2].set_ylabel("I [a.u.]")
                        axlist[2].set_xlabel("Time [s]")
                    else:
                        logger.critical("input signal has to be 1D or 2D")

                # plot integrated signal with subtracted noise:
                if len(axlist) > 3:
                    # plot signal subtracted noise:
                    if np.ndim(signal) == 2:
                        for s in range(len(signal)):
                            axlist[3].plot(time_axis, np.cumsum(signal_no_dyn_noise[s]))
                    elif np.ndim(signal) == 1:
                        axlist[3].plot(time_axis, np.cumsum(signal_no_dyn_noise))
                    else:
                        logger.critical("input signal has to be 1D or 2D")

                    axlist[3].grid(
                        color="r", linestyle="-", linewidth=0.25, which="major"
                    )
                    axlist[3].grid(
                        color="r", linestyle="--", linewidth=0.25, which="minor"
                    )
                    axlist[3].set_title("Integrated Signal (subtr. dyn noise)")
                    axlist[3].set_xlabel("Time [s]")

                    if np.ndim(signal) == 2:
                        if pyr_ind == 0:
                            auc_lac_dyn_noise = integrated_signal_dyn_noise[pyr_ind + 1]
                            auc_pyr_dyn_noise = integrated_signal_dyn_noise[pyr_ind]
                            auc_ratio_dyn_noise = auc_lac_dyn_noise / auc_pyr_dyn_noise
                        elif pyr_ind == 1:
                            auc_lac_dyn_noise = integrated_signal_dyn_noise[pyr_ind - 1]
                            auc_pyr_dyn_noise = integrated_signal_dyn_noise[pyr_ind]
                            auc_ratio_dyn_noise = auc_lac_dyn_noise / auc_pyr_dyn_noise
                        else:
                            Warning("Don't understand the indexing")
                            auc_lac_dyn_noise = integrated_signal_dyn_noise
                            auc_pyr_dyn_noise = integrated_signal_dyn_noise
                            auc_ratio_dyn_noise = 1
                    elif np.ndim(signal) == 1:
                        auc_lac_dyn_noise = 0
                        auc_pyr_dyn_noise = integrated_signal_dyn_noise
                        auc_ratio_dyn_noise = 0

                    logger.critical("dyn. noise AUC pyr : " + str(auc_pyr_dyn_noise))
                    logger.critical("dyn. noise AUC lac : " + str(auc_lac_dyn_noise))
                    logger.critical(
                        "dyn. noise AUC lac/pyr : " + str(auc_ratio_dyn_noise)
                    )

                    # axlist[3].legend(
                    #     np.round(integrated_signal_dyn_noise, 3),
                    #     title="AUC ratio "
                    #     + str(
                    #         np.round(auc_ratio_dyn_noise, 3),
                    #     ),
                    # )

            # save the AUCs
            save_auc(
                self,
                auc_lac=auc_lac,
                auc_pyr=auc_pyr,
                auc_ratio=auc_ratio,
                mask_id=mask_id,
                mask_key=mask_key,
                signal_range=signal_range,
                noise_range=noise_range,
                auc_lac_dyn_noise=auc_lac_dyn_noise,
                auc_pyr_dyn_noise=auc_pyr_dyn_noise,
                auc_ratio_dyn_noise=auc_ratio_dyn_noise,
            )

            return auc_pyr, auc_lac, auc_ratio

        # -------------------------------define widgets -----------------------
        if signal_range_input[0] is None:
            # we have no inputs therefore take the entire signal range
            slider_signal_range_reps = widgets.IntRangeSlider(
                min=0.0,
                value=[0, np.max(time_axis)],
                max=np.max(time_axis),
                description="Signal range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            slider_noise_range_reps = widgets.IntRangeSlider(
                min=0.0,
                value=[0, np.max(time_axis)],
                max=np.max(time_axis),
                description="Noise range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
        else:
            slider_signal_range_reps = widgets.IntRangeSlider(
                min=0.0,
                value=[signal_range_input[0], signal_range_input[1]],
                max=np.max(time_axis),
                description="Signal range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            slider_noise_range_reps = widgets.IntRangeSlider(
                min=0.0,
                value=[noise_range_input[0], noise_range_input[1]],
                max=np.max(time_axis),
                description="Noise range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
        subtract_noise_checkbox = widgets.Checkbox(
            description="subtract noise", value=1
        )

        save_button = widgets.ToggleButton(description="Save time curves", value=0)
        # -----------------------------------------------------------------------
        out = widgets.interactive_output(
            plot_auc,
            {
                "signal_range": slider_signal_range_reps,
                "noise_range": slider_noise_range_reps,
                "subtract_noise": subtract_noise_checkbox,
                "save_csv": save_button,
            },
        )
        main_ui = widgets.VBox(
            [
                slider_signal_range_reps,
                slider_noise_range_reps,
                subtract_noise_checkbox,
                save_button,
            ]
        )

        if display_ui:
            display(main_ui, out)
        else:
            return main_ui

    def split_into_pyr_lac(
        self, input_data=None, pyr_ind=0, metab_axis=4, third_metab=False
    ):
        """
        Split the data into num_metab metabolites.

        Parameters
        ----------
        input_data: data that should be split
        pyr_ind: index of the pyruvate acquisition (either 0 or 1)
        metab_axis: axis on which the dataset should be split into pyruvate and
            lactate
        third_metab : in case data also contains data of 3rd metabolite (expected dimensions: pyr, 3rd metabolite, lac)
            returns num_metab datasets (pyr_data and lac_data)

        Returns
        -------
        pyr_data: data corresponding to pyruvate
        lac_data: data corresponding to lactate
        third_metab_data: data corresponding to 3rd metabolite (if third_metab=True)

        """
        # dont continue if there is no data:
        if input_data is None:
            return False

        # set pyruvate and lactate index:
        if third_metab:
            pyr_ind = 0
            thirdMetab_ind = 1
            lac_ind = 2
            dim_metab = 3
        elif pyr_ind == 0:
            lac_ind = 1
            dim_metab = 2
        elif pyr_ind == 1:
            lac_ind = 0
            dim_metab = 2
        else:
            Warning("Pyruvate index has to be 0 or 1!")
            return False

        # split the data into pyruvate and lactate data:
        # try:
        if metab_axis == 0:
            pyr_data = input_data[pyr_ind::dim_metab, :, :, :, :, :]
            lac_data = input_data[lac_ind::dim_metab, :, :, :, :, :]
            if third_metab:
                third_metab_data = input_data[thirdMetab_ind::dim_metab, :, :, :, :, :]
        if metab_axis == 1:
            pyr_data = input_data[:, pyr_ind::dim_metab, :, :, :, :]
            lac_data = input_data[:, lac_ind::dim_metab, :, :, :, :]
            if third_metab:
                third_metab_data = input_data[:, thirdMetab_ind::dim_metab, :, :, :, :]
        if metab_axis == 2:
            pyr_data = input_data[:, :, pyr_ind::dim_metab, :, :, :]
            lac_data = input_data[:, :, lac_ind::dim_metab, :, :, :]
            if third_metab:
                third_metab_data = input_data[:, :, thirdMetab_ind::dim_metab, :, :, :]
        if metab_axis == 3:
            pyr_data = input_data[:, :, :, pyr_ind::dim_metab, :, :]
            lac_data = input_data[:, :, :, lac_ind::dim_metab, :, :]
            if third_metab:
                third_metab_data = input_data[:, :, :, thirdMetab_ind::dim_metab, :, :]
        # default, consistent with used dimension convention:
        if metab_axis == 4:
            pyr_data = input_data[:, :, :, :, pyr_ind::dim_metab, :]
            lac_data = input_data[:, :, :, :, lac_ind::dim_metab, :]
            if third_metab:
                third_metab_data = input_data[:, :, :, :, thirdMetab_ind::dim_metab, :]
        if metab_axis == 5:
            pyr_data = input_data[:, :, :, :, :, pyr_ind::dim_metab]
            lac_data = input_data[:, :, :, :, :, lac_ind::dim_metab]
            if third_metab:
                third_metab_data = input_data[:, :, :, :, :, thirdMetab_ind::dim_metab]
        # except:
        #     pyr_data = None
        #     lac_data = None

        if third_metab:
            return pyr_data, lac_data, third_metab_data
        else:
            return pyr_data, lac_data

    def roi_signal(
        self,
        input_data=None,
        masks=None,
        mask_dict=None,
        mask_key=None,
        start_ind=None,
        end_ind=None,
        mean_per_slice=False,
    ):
        """
        Exctract signal from ROIs from the input_data. ROIs are either defined
        in the passed mask_dict and can be adressed with the mask_key argument or can be passed directly
        as masks.
        The complex data is summed along the time dimension and then a maksk is applied. Then the magntiude
        of the masked complex data is taken and summed up.

        Parameters
        ----------
        input_data: 6D or 7D numpy array with dimensions:
            [n_echoes, x, y, slices, n_repetitions, n_channels]
            or
            [n_metabolites, n_echoes, x, y, slices, n_repetitions, n_channels]
        masks: 3D numpy array with dimensions:
            [x, y, slices]
        mask_dict: dict
            dictionary containing multiple masks
        mask_key: str
            key to adress the desired mask in the mask_dict
        start_ind: int
            starting index along time dimension for signal averaging
        end_ind: int
            ending index along time dimension for signal averaging
        mean_per_slice: bool
            If True, the mean signal is calculated per slice first,
            then averaged over slices. If False, the mean is calculated over all
            voxels in the mask directly.

        Returns
        -------
        data_masked: averaged signal in the ROI defined by the masks

        Example
        -------
        dnp_signal_pyr_tumor_AUC =  dnp_bssfp.roi_signal(input_data=dnp_bssfp_itp_pyr,
                                                    mask_dict=mask_dict,
                                                    mask_key='tumor',
                                                    start_ind=0,
                                                    end_ind=dnp_bssfp_itp_pyr.shape[4],
                                                    mean_per_slice=False)

        """
        # no need to contiune if there is no data:
        if input_data is None:
            return False

        # load mask from dict if possible
        if mask_dict is not None:
            if mask_key is not None:
                masks = mask_dict[mask_key]

        # if mask is not defined:
        if masks is None:
            return False

        # empty ROI:
        if np.sum(masks) == 0:
            return np.nan

        # set range over which to average:
        if start_ind is None:
            start_ind = 0

        if end_ind is None:
            end_ind = input_data.shape[4]

        # Generate copy, otherwise mask_key in mask_dict will be overwritten!
        masks_copy = np.copy(masks)

        # Replace 0s with NaN:
        masks_copy[masks_copy == 0] = np.nan

        # average data along time dimension:
        # no need for nanmean because mean along time means either
        # signal or no signal (NaNs) arrays are averaged.
        data = np.abs(
            np.mean(input_data[:, :, :, :, start_ind:end_ind, :], axis=4, keepdims=True)
        )

        # to keep old method for now, does per slice averaging
        # before averaging per over mean of slices (wrong):
        if mean_per_slice is True:
            if np.nanmean(masks_copy) > 0:
                # Init empty masked data
                data_masked = np.empty(
                    (
                        input_data.shape[0],
                        input_data.shape[3],
                        input_data.shape[5],
                    )
                )

                # Fill the empty arrays
                # loop over echoes:
                for echo in range(0, input_data.shape[0], 1):
                    # loop over receive channels
                    for chan in range(0, input_data.shape[5], 1):
                        # loop over slices
                        for n_slice in range(0, input_data.shape[3], 1):
                            data_masked[echo, n_slice, chan] = np.nanmean(
                                data[echo, :, :, n_slice, 0, chan]
                                * masks_copy[:, :, n_slice]
                            )
                # sum over slices (Nan does not contribute)
                data_masked = np.nanmean(np.abs(data_masked))
            else:
                data_masked = np.nan
        else:
            if np.nanmean(masks_copy) > 0:
                # Init empty masked data
                data_masked = np.empty(
                    (
                        input_data.shape[0],
                        input_data.shape[5],
                    )
                )

                # Fill the empty arrays
                # loop over echoes:
                for echo in range(0, input_data.shape[0], 1):
                    # loop over receive channels
                    for chan in range(0, input_data.shape[5], 1):
                        data_masked[echo, chan] = np.nanmean(
                            data[echo, :, :, :, 0, chan] * masks_copy
                        )
                # sum over slices (Nan does not contribute)
                data_masked = np.nanmean(data_masked)
            else:
                data_masked = np.nan

        return data_masked

    def get_AUCR_and_std(
        self,
        input_data1=None,
        input_data2=None,
        input_data_orig=None,
        masks=None,
        mask_dict=None,
        mask_key=None,
        start_ind=None,
        end_ind=None,
        std=None,
    ):
        """
        Exctracts signal from ROIs from the input_data. ROIs are either defined
        in the passed mask_dict and can be adressed with the mask_key argument or can be passed directly
        as masks.
        The complex data is summed along the time dimension and then a maksk is applied. Then the magntiude
        of the masked complex data is taken and summed up.

        Parameters
        ----------
        input_data1: 6D or 7D numpy array with dimensions:
            [n_echoes, x, y, slices, n_repetitions, n_channels]
            or
            [n_metabolites, n_echoes, x, y, slices, n_repetitions, n_channels]
        input_data2: 6D or 7D numpy array with dimensions:
            [n_echoes, x, y, slices, n_repetitions, n_channels]
            or
            [n_metabolites, n_echoes, x, y, slices, n_repetitions, n_channels]
        input_data_orig: original (non-interpolated) dataset for
            getting the image shape
        masks: 3D numpy array with dimensions:
            [x, y, slices]
        mask_dict: dictionary containing multiple masks
        mask_key: key to adress the desired mask in the mask_dict
        start_ind: starting index along time dimension for signal averaging
        end_ind: ending index along time dimension for signal averaging
        std: standard deviation of the noise in the original dataset

        Returns
        -------
        AUCR: averaged AUCR in the ROI defined by the masks
        total_var_AUCR: total variance of the AUCR in the ROI defined by the masks

        # Example:
        ----------
        dnp_signal_pyr_tumor_AUC =  dnp_bssfp.roi_signal(input_data=dnp_bssfp_itp_pyr,
                                                    mask_dict=mask_dict,
                                                    mask_key='tumor',
                                                    start_ind=0,
                                                    end_ind=dnp_bssfp_itp_pyr.shape[4],
                                                    mean_per_slice=False)
        """
        # no need to contiune if there is no data:
        if input_data1 is None:
            return False

        if input_data2 is None:
            return False

        if std is None:
            means, stds = self.get_mean_std_of_voxel(
                input_data=input_data_orig, db=True
            )
            std = np.mean(stds)

        # if no original datasetset given:
        if input_data_orig is None:
            image_shape_noninterp = input_data1.shape[1:4]
        else:
            image_shape_noninterp = input_data_orig.shape[1:4]

        # load mask from dict if possible
        if mask_dict is not None:
            if mask_key is not None:
                masks = mask_dict[mask_key]

        # if mask is not defined:
        if masks is None:
            return False

        # empty ROI:
        if np.sum(masks) == 0:
            return np.nan

        # set range over which to average:
        if start_ind is None:
            start_ind = 0

        if end_ind is None:
            end_ind = input_data1.shape[4]

        # Generate copy, otherwise mask_key in mask_dict will be overwritten!
        masks_copy = np.copy(masks)

        # Replace 0s with NaN:
        masks_copy[masks_copy == 0] = np.nan

        # average data along time dimension:
        # no need for nanmean because mean along time means either
        # signal or no signal (NaNs) arrays are averaged.
        data1 = np.abs(
            np.mean(
                input_data1[:, :, :, :, start_ind:end_ind, :], axis=4, keepdims=True
            )
        )
        data2 = np.abs(
            np.mean(
                input_data2[:, :, :, :, start_ind:end_ind, :], axis=4, keepdims=True
            )
        )

        AUCR = data2 / data1
        if np.nanmean(masks_copy) > 0:
            # Init empty masked data
            data_masked = np.empty(
                (
                    input_data1.shape[0],
                    input_data1.shape[1],
                    input_data1.shape[2],
                    input_data1.shape[3],
                    input_data1.shape[5],
                )
            )

            # Calculate the AUCR for each original voxel:
            # loop over echoes:
            for echo in range(0, input_data1.shape[0], 1):
                # loop over receive channels
                for chan in range(0, input_data1.shape[5], 1):
                    data_masked[echo, :, :, :, chan] = (
                        AUCR[echo, :, :, :, 0, chan] * masks_copy
                    )

            # calculate the AUCR mean:
            AUCR_mean = np.nanmean(data_masked)
            logger.critical(AUCR_mean)
            AUCR_all = data_masked.ravel()
            logger.critical(np.nanmean(AUCR_all))
            # remove NaNs
            mask = ~np.isnan(AUCR_all)
            AUCR_all = AUCR_all[mask]
            logger.critical(AUCR_all)
            # Calculate the variance of the AUCR:
            var_AUCR = (std / AUCR_all) ** 2

            # get number of subvoxels (mask resolution) per
            # bSSFP voxel. Sorted.
            nvoxels = self.get_number_mask_voxels_in_bssfp_voxels(
                image_shape=image_shape_noninterp,
                masks=None,
                mask_dict=mask_dict,
                mask_key=mask_key,
                normalize=False,
                make_long=True,
            )

            total_var_AUCR = np.sqrt(np.sum(var_AUCR) / np.sum(nvoxels))
            AUCR = np.nanmean(data_masked)
            return AUCR, total_var_AUCR

        else:
            data_masked = np.nan

        return False

    def roi_signal_curve(
        self,
        input_data=None,
        masks=None,
        mask_dict=None,
        mask_key=None,
        mean_per_slice=False,
    ):
        """
        (Note: recommended to use apply_mask in utils_general)
        Exctracts time curves from ROIs from the input_data. ROIs are either defined
        in the passed mask_dict and can be adressed with the mask_key argument orcan be passed directly
        as masks. The masks are applied onto the input data and the resulting masked data is summed up along
        all dimension besides the repetition/time dimension, returning a 1xT array
        """

        # no need to continue if there is no data:
        if input_data is None:
            logger.critical("no input data!")
            return False

        # load mask from dict if possible
        if mask_dict is not None:
            if mask_key is not None:
                try:
                    masks = mask_dict[mask_key]
                except BaseException:
                    logger.critical("can't extract mask!")
                    return False

        # masks = np.flip(masks, axis=2)

        # if mask is not defined:
        if masks is None:
            logger.critical("no mask defined!")
            return False

        # if ROI is empty:
        if np.sum(masks) == 0:
            logger.critical(f"ROI {mask_key} is empty!")
            return np.zeros((input_data.shape[4]))

        # Generate copy, otherwise mask_key in mask_dict will be overwritten!
        masks_copy = np.copy(masks)

        # replace 0 with NaN
        masks_copy[masks_copy == 0] = np.nan

        try:
            # to keep old method for now, does per slice averaging
            # before averaging per over mean of slices (wrong):
            if mean_per_slice is True:
                if input_data.dtype == "complex":
                    # covert mask to complex mask
                    masks_copy = masks_copy.astype(complex)
                    # Init empty masked data
                    data_masked = np.empty(
                        (
                            input_data.shape[0],
                            input_data.shape[3],
                            input_data.shape[4],
                            input_data.shape[5],
                        ),
                        dtype=input_data.dtype,
                    )
                else:
                    # Init empty masked data
                    data_masked = np.empty(
                        (
                            input_data.shape[0],
                            input_data.shape[3],
                            input_data.shape[4],
                            input_data.shape[5],
                        )
                    )

                # Fill the empty arrays
                # loop over repetitions:
                for rep in range(0, input_data.shape[4], 1):
                    # loop over echoes:
                    for echo in range(0, input_data.shape[0], 1):
                        # loop over receive channels
                        for chan in range(0, input_data.shape[5], 1):
                            # loop over slices
                            for n_slice in range(0, input_data.shape[3], 1):
                                data_masked[echo, n_slice, rep, chan] = np.nanmean(
                                    input_data[echo, :, :, n_slice, rep, chan]
                                    * masks_copy[:, :, n_slice]
                                )
                # sum over slices (Nan does not contribute)
                data_masked = np.squeeze(
                    np.nanmean(
                        np.nanmean(
                            np.nanmean(data_masked, axis=0, keepdims=True),
                            axis=1,
                            keepdims=True,
                        ),
                        axis=3,
                    )
                )
            # use new way of masking:
            else:
                if input_data.dtype == "complex":
                    # covert mask to complex mask
                    masks_copy = masks_copy.astype(complex)
                    # Init empty masked data
                    data_masked = np.empty(
                        (
                            input_data.shape[0],  # N echoes
                            input_data.shape[4],  # N repetitions
                            input_data.shape[5],  # N receive channels
                        ),
                        dtype=input_data.dtype,  # complex dataset
                    )
                else:
                    # Init empty masked data
                    data_masked = np.empty(
                        (
                            input_data.shape[0],  # N echoes
                            input_data.shape[4],  # N repetitions
                            input_data.shape[5],  # N receive channels
                        )
                    )
                # loop over slices:
                for rep in range(0, input_data.shape[4], 1):
                    # loop over echoes:
                    for echo in range(0, input_data.shape[0], 1):
                        # loop over receive channels
                        for chan in range(0, input_data.shape[5], 1):
                            # apply mask:
                            data_masked[echo, rep, chan] = np.nanmean(
                                input_data[echo, :, :, :, rep, chan] * masks_copy
                            )
                # take average over echoes and receive channels:
                data_masked = np.squeeze(
                    np.nanmean(
                        np.nanmean(data_masked, axis=0, keepdims=True),
                        axis=2,
                    )
                )
        # return zeros (of length number of repetitions):
        except Exception:
            logger.critical("masking did not work --> returning 0s")
            data_masked = np.zeros((input_data.shape[4]))

        return data_masked

    def get_number_mask_voxels_in_bssfp_voxels(
        self,
        image_shape=None,
        masks=None,
        mask_dict=None,
        mask_key=None,
        normalize=False,
        make_long=False,
    ):
        """
        Input:

        image_shape: [z,x,y] of the noninterpolated dataset

        This function calculates the number of mask voxels per bSSFP voxel.
        This can be used to calculate the std. of a ROI. Can be inaccurate in case of
        matrix size differences like eg. masks 21 slices and bSSFP 12.
        (close to 2 but still rounded to 1)
        """
        # no need to contiune if there is no image_shape:
        if image_shape is None:
            return False

        # load mask from dict if possible
        if mask_dict is not None:
            if mask_key is not None:
                try:
                    masks = mask_dict[mask_key]
                except BaseException:
                    return False

        # masks = np.flip(masks, axis=2)

        # if mask is not defined:
        if masks is None:
            return False

        # empty ROI:
        if np.sum(masks) == 0:
            return np.nan

        # init empty:
        coverage = np.nan

        # replace 0 with NaN
        # masks[masks == 0] = np.nan
        mask_depth, mask_height, mask_width = masks.shape
        image_depth, image_height, image_width = image_shape

        block_depth = mask_depth / image_depth
        block_height = mask_height / image_height
        block_width = mask_width / image_width

        coverage = np.zeros(image_shape)
        for i in range(image_depth):
            for j in range(image_height):
                for k in range(image_width):
                    # Compute the block in the original mask corresponding to
                    # this voxel
                    start_i = round(i * block_depth)
                    end_i = round((i + 1) * block_depth)
                    start_j = round(j * block_height)
                    end_j = round((j + 1) * block_height)
                    start_k = round(k * block_width)
                    end_k = round((k + 1) * block_width)

                    # Extract this block from the mask
                    mask_block = masks[start_i:end_i, start_j:end_j, start_k:end_k]

                    # Compute the coverage for this voxel
                    if normalize is True:
                        coverage[i, j, k] = np.mean(mask_block)
                    else:
                        coverage[i, j, k] = np.sum(mask_block)

        # calculate ratio of mask to bssfp data resolution:
        # z_ratio = int(masks.shape[0] / input_data.shape[1])
        # x_ratio = int(masks.shape[1] / input_data.shape[2])
        # y_ratio = int(masks.shape[2] / input_data.shape[3])

        # # init empty array:
        # mask_voxels_per_bssfp_voxel = np.zeros(input_data.shape[1:4])
        # # loop over bssfp voxels:
        # for y in range(input_data.shape[3]):
        #     for x in range(input_data.shape[2]):
        #         for z in range(input_data.shape[1]):
        #             mask_voxels_per_bssfp_voxel[z, x, y] = np.sum(
        #                 masks[
        #                     (z - 1) * z_ratio : z * z_ratio,
        #                     (x - 1) * x_ratio : x * x_ratio,
        #                     (y - 1) * y_ratio : y * y_ratio,
        #                 ]
        #             )

        # if signal should be normalized to a range from min/Nvoxels to 1.
        if make_long is True:
            # make 1D array:
            mask_voxels_per_bssfp_voxel_allvals = np.ravel(coverage)

            # remove 0s:
            mask_voxels_per_bssfp_voxel_allvals = mask_voxels_per_bssfp_voxel_allvals[
                mask_voxels_per_bssfp_voxel_allvals != 0
            ]

            # normalize to the maximum amount of mask voxels that a bSSFP voxel
            # can contain:
            nvoxels = np.sort(mask_voxels_per_bssfp_voxel_allvals)
            return nvoxels
        else:
            return coverage

    def get_mean_std_of_voxel(
        self,
        input_data=None,
        voxel_pos=None,
        echo=0,
        chan=0,
        start_ind=None,
        end_ind=None,
        db=False,
    ):
        """
        This function calculates the mean and standard deviation of
        the bssfp data (input_data) in the voxel at position voxel_pos

        Input:

        input_data: bSSFP data, expected order: echoes-read-phase-slice-repetitions-channels

        voxel_pos: voxel position in index read-phase-slice.
            If None is passed, 4 corner voxel are taken.

        echo: echo number, default = 0

        chan: acquisition channel number, default = 0

        Returns

        mean and standarad deviation
        """

        # dont continue if no input_data is passed:
        if input_data is None:
            return False

        # default voxel_pos:
        if voxel_pos is None:
            # take 4 corner voxels (on opposite of read start (can be affected from
            # small read matrixces (~16 points)))):
            voxel_pos = [[-1, 0, 0], [-1, 0, -1], [-1, -1, 0], [-1, -1, -1]]

        # use full range if no index was passed:
        if start_ind is None:
            start_ind = 0

        # use full range if no index was passed:
        if end_ind is None:
            end_ind = input_data.shape[4]

        # if 1 voxel position was passed:
        if np.ndim(voxel_pos) == 1:
            stds = np.std(
                np.real(
                    input_data[
                        echo,
                        voxel_pos[0],
                        voxel_pos[1],
                        voxel_pos[2],
                        start_ind:end_ind,
                        chan,
                    ]
                )
            )
            # np.std cant work with complex data:
            if input_data.dtype == complex:
                stds = stds + 1j * np.std(
                    np.imag(
                        input_data[
                            echo,
                            voxel_pos[0],
                            voxel_pos[1],
                            voxel_pos[2],
                            start_ind:end_ind,
                            chan,
                        ]
                    )
                )
            # get mean of voxel signa:
            means = np.mean(
                input_data[
                    echo,
                    voxel_pos[0],
                    voxel_pos[1],
                    voxel_pos[2],
                    start_ind:end_ind,
                    chan,
                ]
            )

        # if more than 1 voxel position was passed:
        elif np.ndim(voxel_pos) == 2:
            # init empty array
            stds = np.zeros((np.shape(voxel_pos)[0], 1), dtype=input_data.dtype)
            means = np.zeros((np.shape(voxel_pos)[0], 1), dtype=input_data.dtype)
            for p in range(np.shape(voxel_pos)[0]):
                stds[p] = np.std(
                    np.real(
                        input_data[
                            echo,
                            voxel_pos[p][0],
                            voxel_pos[p][1],
                            voxel_pos[p][2],
                            start_ind:end_ind,
                            chan,
                        ]
                    )
                )
                # np.std cant work with complex data:
                if input_data.dtype == complex:
                    stds[p] = stds[p] + 1j * np.std(
                        np.imag(
                            input_data[
                                echo,
                                voxel_pos[p][0],
                                voxel_pos[p][1],
                                voxel_pos[p][2],
                                start_ind:end_ind,
                                chan,
                            ]
                        )
                    )
                means[p] = np.mean(
                    input_data[
                        echo,
                        voxel_pos[p][0],
                        voxel_pos[p][1],
                        voxel_pos[p][2],
                        start_ind:end_ind,
                        chan,
                    ]
                )

        else:
            Warning("voxel_pos has to be a 1D or 2D array:")
            return False

        if db is True:
            logger.critical("calculated mean from edge voxels is: " + str(means))
            logger.critical("calculated std from edge voxels is: " + str(stds))

        return means, stds

    def get_std_of_ROI(
        self,
        input_data=None,
        masks=None,
        mask_dict=None,
        mask_key=None,
        std=None,
        pyr_lac=True,
        db=False,
    ):
        """ """

        # no need to contiune if there is no data:
        if input_data is None:
            return False

        # load mask from dict if possible
        if mask_dict is not None:
            if mask_key is not None:
                try:
                    masks = mask_dict[mask_key]
                except BaseException:
                    return False

        # empty ROI:
        if np.sum(masks) == 0:
            Warning("Empty mask!")
            return np.nan

        # get std from data if no std is passed:
        if std is None:
            means, stds = self.get_mean_std_of_voxel(input_data=input_data, db=db)
            std = np.mean(stds)

        # get normalized_mask_voxels_in_bssfp_voxels. This sorted array
        # goes from 0 to 1. 0 means that no mask voxel was in the bSSFP voxel,
        # 1 means that the bSSFP voxel was completly filled by the mask voxel.
        # This acts a weight:
        normalized_mask_voxels_in_bssfp_voxels = (
            self.get_number_mask_voxels_in_bssfp_voxels(
                image_shape=input_data.shape[1:4],
                masks=None,
                mask_dict=mask_dict,
                mask_key=mask_key,
                normalize=True,
                make_long=True,
            )
        )

        normed_weights = normalized_mask_voxels_in_bssfp_voxels
        # Error propagation:
        # std_ROI = (
        #     1
        #     / (np.sum(normed_weights) / len(normed_weights))
        #     * np.sqrt(
        #         np.sum(
        #             (
        #                 (
        #                     normed_weights
        #                     / len(normed_weights)
        #                     * std
        #                     * 1
        #                     / np.sqrt(input_data.shape[4])
        #                 )
        #                 ** 2
        #             )
        #         )
        #     )
        # )

        # take into account that the AUC values was averaged:
        # divide by 2 because only have the repetitions is averaged (each)
        # if pyr_lac is True:
        #     std = std / np.sqrt(input_data.shape[4] / 2)
        # else:
        #     std = std / np.sqrt(input_data.shape[4])

        std_ROI = std * np.sqrt(np.sum(normed_weights**2) / len(normed_weights) ** 2)

        return std_ROI

    def get_AUCR_std(
        self,
        std=None,
        std_lac=None,
        AUC_pyr=None,
        AUC_lac=None,
    ):
        """Propagate measurement uncertainty for the lactate/pyruvate AUC ratio."""
        # if no AUC for pyruvate was passed, return NaN
        if AUC_pyr is None:
            return np.nan

        # if no AUC for lactate was passed, return NaN
        if AUC_lac is None:
            return np.nan

        # if no std was passed:
        if std is None:
            return np.nan
        # set standard dev. of pyruvate to std:
        else:
            std_pyr = std

        # use std for lactate if no specifig lactate std was passed:
        if std_lac is None:
            std_lac = std

        # calculate AUC ratio:
        AUCR = AUC_lac / AUC_pyr
        # Error propagation division:
        std_AUCR = AUCR * np.sqrt((std_lac / AUC_lac) ** 2 + (std_pyr / AUC_pyr) ** 2)

        return std_AUCR

    def combine_multichannel(
        self, input_data=None, phase=None, chan_dim=None, rssq=False, db=False
    ):
        """
        Quickly computes the fftshift(fft(fftshift(input_data))) along the input axis.
        If input_data is not set, returns the fft along axis of the fid file

        Parameters
        ----------
        input_data : (complex) ndarray (None)
            data that should be fourier-transformed.
            If blank, self.fid is used

        phase: float None
            Phase [degree] that should be used to combine the 2 channels.
            If non is given the phase will be automatically determined

        axes : tuple (0,1,2)
            axes along the fourier transform should be performed

        rssq : root sum of squares

        db : bool (False)
            Prints debugging info if True
        """

        # if no input dat was given, use fid
        if input_data is None:
            fid = self.fid
            logger.debug("using self.fid")
        else:
            fid = input_data
            logger.debug("using input data")
        logger.debug("dimensions of input_data = " + str(fid.ndim))

        # do not continue if only 1 channel input:
        if input_data.shape[5] == 1:
            # logger.critical("only 1 channel put in...")
            return input_data

        # if no channel dimension was given use 6th dimension for channel:
        if chan_dim is None:
            chan_dim = 5
            if chan_dim <= fid.ndim:
                chan_dim = chan_dim
            else:
                chan_dim_before = chan_dim
                chan_dim = fid.ndim
                Warning(
                    "Unexpected array dimensions: "
                    + str(fid.ndim)
                    + ".Default channel dimension: "
                    + str(chan_dim_before)
                    + ". Using"
                    + " the last array dimension as channel dimension"
                )
        else:
            if chan_dim <= fid.ndim - 1:
                chan_dim = chan_dim
            else:
                Warning(
                    "Inconsistent array dimensions: "
                    + str(fid.ndim)
                    + "and desired channel dimension: "
                    + str(chan_dim)
                    + ". Using"
                    + " the last array dimension as channel dimension"
                )
                chan_dim = fid.ndim

        # test if the array has more than 1 channel:
        if fid.shape[chan_dim - 1] > 1:
            max_val = np.max(fid)
            logger.debug("Max val is " + str(max_val))
            max_index = np.unravel_index(np.argmax(fid), fid.shape)
            logger.debug("Max index is " + str(max_index))
        else:
            Warning("Data has only " + fid.ndim[chan_dim] + " in the channel dimension")

        logger.debug("chan_dim = " + str(chan_dim))

        # use root mean square:
        if rssq is True:
            fid_combined = np.sqrt(
                np.abs(fid[:, :, :, :, :, [0]]) ** 2
                + np.abs(fid[:, :, :, :, :, [1]]) ** 2
            )
        # combine both channels using a phase term applied on channel 2
        else:
            # no phase input, find phase that maximizes the signal
            if phase is None:
                # find max in array, compare signal from max for phaseing
                # assuming the order Echoes, Read, phase, slice, repetitions,
                # channels in the data:
                ch1_max = np.sum(
                    np.sum(np.sum(fid[max_index[0], :, :, :, max_index[4], 0]))
                )
                ch2_max = np.sum(
                    np.sum(np.sum(fid[max_index[0], :, :, :, max_index[4], 1]))
                )

                # define phase array from -pi to pi
                phase_range_rad = np.linspace(-np.pi, np.pi, 1000)
                phase_range_deg = phase_range_rad / np.pi * 180
                # combine both channels:
                ch12_combined = ch1_max + ch2_max * np.exp(1j * phase_range_rad)
                max_phase_ind = np.argmax(np.abs(ch12_combined))
                max_phase_rad_val = phase_range_rad[max_phase_ind]
                max_phase_deg_val = phase_range_deg[max_phase_ind]

                # plot result
                if db:
                    from matplotlib import pyplot as plt

                    fig, ax = plt.subplots()
                    ax.plot(phase_range_deg, np.real(ch12_combined))
                    ax.plot(phase_range_deg, np.real(ch12_combined))
                    ax.plot(
                        phase_range_deg[max_phase_ind], max(np.abs(ch12_combined)), "rx"
                    )
                    ax.title.set_text("Max phase = " + str(max_phase_deg_val))

                # combine channels:
                fid_combined = np.zeros((fid.shape[:-1] + (1,)), dtype=fid.dtype)
                # use this [] indexing to keep dimension
                fid_combined = fid[:, :, :, :, :, [0]] + fid[
                    :, :, :, :, :, [1]
                ] * np.exp(1j * max_phase_rad_val)

            # use input phase:
            else:
                max_phase_rad_val = phase / 180 * np.pi
                # combine channels:
                fid_combined = np.zeros((fid.shape[:-1] + (1,)), dtype=fid.dtype)
                # use this [] indexing to keep dimension
                fid_combined = fid[:, :, :, :, :, [0]] + fid[
                    :, :, :, :, :, [1]
                ] * np.exp(1j * max_phase_rad_val)

        return fid_combined

    def get_extent_bssfp(self):
        """
        Calculate the extent of a bssfp for axial, coronal and sagittal displaying and
        adds them as attributes to the class:

        self.cor_ext
        self.ax_ext
        self.sag_ext

        """
        read_orient = self.method["PVM_SPackArrReadOrient"]
        read_offset = self.method["PVM_SPackArrReadOffset"]
        phase_offset = self.method["PVM_SPackArrPhase1Offset"]
        slice_offset = self.method["PVM_SPackArrSliceOffset"]
        self.matrix[0]
        self.matrix[1]
        self.matrix[2]
        mm_read = self.method["PVM_Fov"][0]
        mm_phase = self.method["PVM_Fov"][1]
        mm_slice = self.method["PVM_Fov"][2]
        if self.method["PVM_SPackArrSliceOrient"] == "coronal":
            if read_orient == "H_F":
                self.cor_ext = [
                    -mm_phase / 2 + phase_offset,
                    mm_phase / 2 + phase_offset,
                    -mm_read / 2 + read_offset,
                    mm_read / 2 + read_offset,
                ]
                self.ax_ext = [
                    -mm_phase / 2 + phase_offset,
                    mm_phase / 2 + phase_offset,
                    -mm_slice / 2 + slice_offset,
                    mm_slice / 2 + slice_offset,
                ]
                self.sag_ext = [
                    -mm_read / 2 + read_offset,
                    mm_read / 2 + read_offset,
                    -mm_slice / 2 + slice_offset,
                    mm_slice / 2 + slice_offset,
                ]

        else:
            raise NotImplementedError(
                self.method["PVM_SPackArrSliceOrient"], "not implemented yet"
            )

    def find_noise_region(self):
        """This function is supposed to find a region containing no signal that can be used as a reference of noise"""
        return True

    def generate_figures(
        self,
        bssfp_data=None,
        coronal_image=None,
        axial_image=None,
        axlist=None,
        fig=None,
        path_to_params=None,
        path_to_save=None,
        param_dict=None,
        pyr_ind=0,
        fig_prefix="fig_",
    ):
        """

        Load figure defining parameter file, plot the data and save it.
        Use the parameters as you would use them in plot3d_new2

        Parameters
        ----------
        path_to_params: str
            parameter (json) file
        path_to_save: str
            folder where the figures should be saved
        param_dict: dict
            parameters

        Examples
        --------

        similat to plot3d_new2, but with additional parameters path_to_params, path_to_save
        and param_dict

        >>> fig, (ax1, ax2, ax3) = plt.subplots(1, 3,tight_layout=True)

        >>> phantom_bssfp.generate_figures(bssfp_data= dnp_bssfp_pv_complex_reco_combined_shift,
        >>>               coronal_image=coronal,
        >>>               axial_image=axial,
        >>>               axlist=[ax1, ax2, ax3],
        >>>               path_to_params="lac_sig.json",
        >>>               fig=fig,
        >>>               path_to_save="figures_looped_lac\\",
        >>>               param_dict={'rep1': range(1,200,2)},
        >>>               fig_prefix="lac_")

        """

        # load json parameter file:
        plot_params = load_plot_params(param_file=path_to_params, data_obj=self)

        if plot_params is None:
            return
        if path_to_save is None:
            return
        if param_dict is None:
            return

        # generate path if none was passed:
        if path_to_save is None:
            path_to_save = "looped_figures/"

        # make it path if not:
        if path_to_save[-1] != "/":
            path_to_save = path_to_save + "/"

        # loop over parameters that should be looped through
        figure_counter = 0
        for key in param_dict:
            # generate copy to not reload each time
            plot_params_plot = plot_params
            # loop over values of plot parameter:
            for value in param_dict[key]:
                figure_counter = figure_counter + 1
                plot_params_plot["fig_name"] = (
                    path_to_save + fig_prefix + "%0.3d" % figure_counter
                )
                plot_params_plot[key] = value
                # call plotting function with new parameters:
                self.plot3D_new2(
                    bssfp_data=bssfp_data,
                    coronal_image=coronal_image,
                    axial_image=axial_image,
                    axlist=axlist,
                    fig=fig,
                    pyr_ind=pyr_ind,
                    plot_params=plot_params_plot,
                    save_fig=True,
                )

    def norm_bssfp_to_background_and_divide_by_mean_for_SSI(
        self,
        bssfp_data,
        coronal_object,
        slice_num_cor,
        axlist=None,
        bg_pixels_x=[0, 3],
        bg_pixels_y=[1, 8],
        num_hist_bins=25,
    ):
        """
        Function to prepare bssfp data for Structural Similarity comparison.

        Parameters
        ----------

        bssfp_data: 3D array
            Already summed in time or just for one repetition prepared bssfp image data
            for one metabolite. Shape is (phase, read, slice)
        coronal_object: BrukerExp instance
            coronal reference data
        slice_num_cor: int
            slice number of desired bssfp slice
        axlist: list
            contains axis elements to plot into
        bg_pixels_x: list, two entries
            background region choice in x-dimension, to be plotted as a Rectangle on bssfp image in this function
        bg_pixels_y: list, two entries
            background region choice in y-dimension, to be plotted as a Rectangle on bssfp image in this function
        num_hist_bins : int
            number of histogram bins in histogram plot

        Returns
        -------

        background value
        """
        if axlist is None:
            logger.warning(
                "Need to pass axlist (use fig, (ax1, ax2, ax3) = plt.subplots(1, 3) \
                        to generate and call function with: \
                        axlist=[ax1, ax2, ax3])"
            )
            return False
        if coronal_object.__module__.startswith("hypermri."):
            pass
        else:
            logger.critical("coronal_object must be a type BrukerExp object")

        # assumes data of shape (read, phase,slice)

        def save_background_val(self, background_val, background_std):
            setattr(self, "ssi_background_value", {})
            self.ssi_background_value = background_val
            setattr(self, "ssi_background_std", {})
            self.ssi_background_std = background_std

        def plot(
            slice_num_cor, bg_pixels_x, bg_pixels_y, num_hist_bins, cor_image_window
        ):
            print(
                "N-pixels=",
                str(
                    (bg_pixels_y[1] - bg_pixels_y[0])
                    * (bg_pixels_x[1] - bg_pixels_x[0])
                ),
            )
            for ax in axlist:
                ax.clear()
            axlist[0].imshow(
                bssfp_data[:, :, slice_num_cor],
                extent=[
                    0,
                    bssfp_data[:, :, slice_num_cor].shape[1],
                    bssfp_data[:, :, slice_num_cor].shape[0],
                    0,
                ],
            )
            axlist[0].add_patch(
                Rectangle(
                    (np.min(bg_pixels_x[0:1]), np.min(bg_pixels_y[0:1])),
                    bg_pixels_x[1] - bg_pixels_x[0],
                    bg_pixels_y[1] - bg_pixels_y[0],
                    color="r",
                    fc="r",
                    alpha=0.4,
                )
            )

            coronal_slice_corresponding_to_bssfp_slice = self.find_closest_slice(
                coronal_obj=coronal_object, bssfp_pos=[0, 0, slice_num_cor]
            )[2]

            axlist[1].imshow(
                coronal_object.seq2d[:, :, coronal_slice_corresponding_to_bssfp_slice],
                cmap="bone",
                vmin=cor_image_window[0],
                vmax=cor_image_window[1],
            )

            x_data, y_data, bin_size = get_histogram(
                bssfp_data[
                    bg_pixels_y[0] : bg_pixels_y[1], bg_pixels_x[0] : bg_pixels_x[1], 0
                ],
                num_hist_bins,
            )
            axlist[2].bar(x_data, y_data, width=bin_size)

            background_val = np.mean(
                bssfp_data[
                    bg_pixels_y[0] : bg_pixels_y[1],
                    bg_pixels_x[0] : bg_pixels_x[1],
                    slice_num_cor,
                ]
            )
            background_std = np.std(
                bssfp_data[
                    bg_pixels_y[0] : bg_pixels_y[1],
                    bg_pixels_x[0] : bg_pixels_x[1],
                    slice_num_cor,
                ]
            )

            axlist[2].vlines(background_val, 0, np.max(y_data), color="r", label="mean")
            axlist[2].legend()
            axlist[2].set_xlabel("Pixel value")
            axlist[2].set_ylabel("Num of Pix")
            axlist[2].set_title("Histogram")
            axlist[0].set_title("bssfp slice " + str(slice_num_cor))
            axlist[1].set_title(
                "Coronal ref slice " + str(coronal_slice_corresponding_to_bssfp_slice)
            )
            save_background_val(self, background_val, background_std)
            print(background_val, background_std)

        # UI components

        num_histogram_bins_slider = widgets.IntSlider(
            min=1,
            value=10,
            max=75,
            description="Histogram bins",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )

        slice_num_cor_slider = widgets.IntSlider(
            min=0.0,
            value=slice_num_cor,
            max=bssfp_data.shape[2] - 1,
            description="Coronal slice number",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        cor_image_windowing_slider = widgets.IntRangeSlider(
            min=0,
            value=[np.min(coronal_object.seq2d), np.max(coronal_object.seq2d)],
            max=np.max(coronal_object.seq2d),
            description="Coronal img windowing",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        bg_pixels_x_slider = widgets.IntRangeSlider(
            min=0.0,
            value=[bg_pixels_x[0], bg_pixels_x[1]],
            max=bssfp_data.shape[1] - 1,
            description="Bg pixels x",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        bg_pixels_y_slider = widgets.IntRangeSlider(
            min=0.0,
            value=[bg_pixels_y[0], bg_pixels_y[1]],
            max=bssfp_data.shape[0] - 1,
            description="Bg pixels y",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )

        out = widgets.interactive_output(
            plot,
            {
                "slice_num_cor": slice_num_cor_slider,
                "bg_pixels_x": bg_pixels_x_slider,
                "bg_pixels_y": bg_pixels_y_slider,
                "num_hist_bins": num_histogram_bins_slider,
                "cor_image_window": cor_image_windowing_slider,
            },
        )
        main_ui = widgets.VBox(
            [
                slice_num_cor_slider,
                bg_pixels_x_slider,
                bg_pixels_y_slider,
                num_histogram_bins_slider,
                cor_image_windowing_slider,
            ]
        )
        display(main_ui, out)

    def find_sig_range_reps(
        self, dnp_bssfp_data, phip_bssfp_data, signal_range_dict, axlist=None
    ):
        # TODO: In the future one could rewrite this function so that it only loads one bssfp dataset and then
        # comparing is then done in a second function that calls two times this
        # function
        """
        Finds the repetitions where signal starts and stops for two bssfp datasets.

        This function was developed for the 2023 paper on comparing PHIP and DNP
        for HP pyruvate.
        This function assumes data of the shape: [ Echoes, Read, phase, slice, repetitions, channels]
        i.e.: dnp_bssfp_pv_complex_reco_combined_shift.shape = (1, 16, 12, 10, 250, 1)

        Parameters
        ----------

        dnp_bssfp_data: array-like
            Contains multi rep, 3D bssfp data, usually this is only pyruvate or lactate channel
        phip_bssfp_data:array-like
            Contains multi rep, 3D bssfp data, usually this is only pyruvate or lactate channel
        axlist: list
            contains subplots to plot into, i.e. [ax1,ax2,ax3]
        signal_range_dict: dict
            has two keys: "phip_signal_range_reps", "dnp_signal_range_reps"
            each of them has a list with two entries specifying the indices of the repetitions which should be normed

        Returns
        -------

        dnp_images[:,:,:,dnp_max_sign_rep]: dnp bssfp image stack at the highest signal repetition
            Has shape (Read, phase, slice)

        phip_images[:,:,:,phip_max_sign_rep]: phip bssfp image stack at the highest signal repetition
            has shape (Read, phase, slice)

        Plot of the process and result
        Example
        ------

        >>> dnp_sig_reps,phip_sig_reps = find_sig_range_reps(dnp_bssfp_pv_complex_reco_combined_shift,phip_bssfp_pv_complex_reco_combined_shift)
        """
        if axlist is None:
            logger.critical(
                "Need to pass axlist (use fig, (ax1, ax2, ax3) = plt.subplots(1, 3) \
                        to generate and call function with: \
                        axlist=[ax1, ax2, ax3])"
            )
            return False
        if signal_range_dict is None:
            logger.critical("Need to pass a signal range dictionary")
            return False
        else:
            setattr(self, "original_dict", deepcopy(signal_range_dict))
            setattr(self, "signal_range_dict", signal_range_dict)

        # checking if we have echoes or more than one channel
        if phip_bssfp_data.shape[0] == 1:
            pass
        else:
            logger.critical(
                "phip_bssfp_data.shape[0] is larger than 1, this function only works for single echo data"
            )
        if phip_bssfp_data.shape[5] == 1:
            pass
        else:
            logger.critical(
                "phip_bssf_data.shape[5] is larger than 1, this function only works for single channel data"
            )

        if dnp_bssfp_data.shape[0] == 1:
            pass
        else:
            logger.critical(
                "dnp_bssfp_data.shape[0] is larger than 1, this function only works for single echo data"
            )
        if dnp_bssfp_data.shape[5] == 1:
            pass
        else:
            logger.critical(
                "dnp_bssfp_data.shape[5] is larger than 1, this function only works for single channel data"
            )

        # shorten the array to remove echoes and multi channel dimensions
        phip_images = np.squeeze(phip_bssfp_data)
        dnp_images = np.squeeze(dnp_bssfp_data)
        global itgl_dnp, itgl_phip
        itgl_dnp = []
        itgl_phip = []
        for rep in range(phip_images.shape[3]):
            itgl_phip.append(np.sum(np.abs(phip_images[:, :, :, rep])))
            itgl_dnp.append(np.sum(np.abs(dnp_images[:, :, :, rep])))
        # max sig rep for phip and dnp
        max_sig_rep_phip = np.squeeze(np.argmin(np.abs(itgl_phip - np.max(itgl_phip))))
        max_sig_rep_dnp = np.squeeze(np.argmin(np.abs(itgl_dnp - np.max(itgl_dnp))))

        # plot results
        def plot_sig_range(dnp_sig_range, phip_sig_range):
            for ax in axlist:
                ax.clear()

            axlist[0].plot(itgl_dnp, label="dnp")
            axlist[0].plot(itgl_phip, label="phip")

            axlist[0].axvline(
                dnp_sig_range[0], 0, np.max(itgl_dnp), color="C0", linestyle="dashed"
            )
            axlist[0].axvline(
                dnp_sig_range[1], 0, np.max(itgl_dnp), color="C0", linestyle="dashed"
            )

            axlist[0].axvline(
                phip_sig_range[0], 0, np.max(itgl_phip), color="C1", linestyle="dashed"
            )
            axlist[0].axvline(
                phip_sig_range[1], 0, np.max(itgl_phip), color="C1", linestyle="dashed"
            )

            axlist[0].legend()
            axlist[0].set_xlabel("Repetition")
            axlist[0].set_ylabel("Sum over all pixels and slices")

            axlist[1].imshow(
                np.sum(
                    np.abs(phip_images[:, :, :, phip_sig_range[0] : phip_sig_range[1]]),
                    axis=(2, 3),
                )
            )
            axlist[2].imshow(
                np.sum(
                    np.abs(dnp_images[:, :, :, dnp_sig_range[0] : dnp_sig_range[1]]),
                    axis=(2, 3),
                )
            )
            axlist[1].set_title("PHIP")
            axlist[2].set_title("DNP")
            axlist[0].grid(color="k", linestyle="-", linewidth=0.25, which="major")
            axlist[0].grid(color="k", linestyle="--", linewidth=0.25, which="minor")

            # return
            # dnp_images[:,:,:,dnp_sig_range[0]:dnp_sig_range[1]],phip_images[:,:,:,phip_sig_range[0]:phip_sig_range[1]]

        # check if the dropdown value changes and then update all
        def dropdown_eventhandler(change):
            if str(change.new) == "20 reps around peak maxima":
                self.signal_range_dict.update(
                    {"phip": [max_sig_rep_phip - 10, max_sig_rep_phip + 10]}
                )
                self.signal_range_dict.update(
                    {"dnp": [max_sig_rep_dnp - 10, max_sig_rep_dnp + 10]}
                )
            elif str(change.new) == "2 before, 18 after peak maxima":
                self.signal_range_dict.update(
                    {"phip": [max_sig_rep_phip - 2, max_sig_rep_phip + 18]}
                )
                self.signal_range_dict.update(
                    {"dnp": [max_sig_rep_dnp - 2, max_sig_rep_dnp + 18]}
                )

            elif str(change.new) == "Manual setting":
                self.signal_range_dict.update(self.original_dict)

            else:
                pass
            phip_slider_signal_range_reps.value = self.signal_range_dict["phip"]
            dnp_slider_signal_range_reps.value = self.signal_range_dict["dnp"]
            plot_sig_range(
                [self.signal_range_dict["dnp"][0], self.signal_range_dict["dnp"][1]],
                [self.signal_range_dict["phip"][0], self.signal_range_dict["phip"][1]],
            )

        options_dropdown = widgets.Dropdown(
            options=[
                "Manual setting",
                "20 reps around peak maxima",
                "2 before, 18 after peak maxima",
            ],
            value="Manual setting",
        )
        options_dropdown.observe(dropdown_eventhandler)

        # check if slider value changes and then update the dictionary
        def slider_change_eventhandler(change):
            self.signal_range_dict.update({"phip": phip_slider_signal_range_reps.value})
            self.signal_range_dict.update({"dnp": dnp_slider_signal_range_reps.value})

        phip_slider_signal_range_reps = widgets.IntRangeSlider(
            min=0.0,
            value=[
                self.signal_range_dict["phip"][0],
                self.signal_range_dict["phip"][1],
            ],
            max=phip_images.shape[3] - 1,
            description="PHIP Signal range:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        dnp_slider_signal_range_reps = widgets.IntRangeSlider(
            min=0.0,
            value=[self.signal_range_dict["dnp"][0], self.signal_range_dict["dnp"][1]],
            max=dnp_images.shape[3] - 1,
            description="DNP Signal range:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        phip_slider_signal_range_reps.observe(slider_change_eventhandler)
        dnp_slider_signal_range_reps.observe(slider_change_eventhandler)

        out = widgets.interactive_output(
            plot_sig_range,
            {
                "dnp_sig_range": dnp_slider_signal_range_reps,
                "phip_sig_range": phip_slider_signal_range_reps,
            },
        )
        main_ui = widgets.VBox(
            [
                dnp_slider_signal_range_reps,
                phip_slider_signal_range_reps,
                options_dropdown,
            ]
        )
        display(main_ui, out)

    def find_high_snr_slices(
        self,
        image_data,
        snr_per_slice,
        axlist=None,
        slice_default_indices_cor=None,
        slice_default_indices_sag=None,
        slice_default_indices_ax=None,
    ):
        """
        Finds the slices where there is sufficient signal to compute a SSI for a bssfp dataset.

        This function was developed for the 2023 paper on comparing PHIP and DNP
        for HP pyruvate.

        Parameters
        ----------
        image_data: np.ndarray
            shape (read, phase, slice), contains already preprocessed bssfp magnitude data for displaying
        snr_per_slice: dict
            contains snr per slice for three keys: coronal, axial, sagittal
        axlist: list
            contains subplots to plot into, i.e. [ax1,ax2,ax3]
        slice_indices: list, default None.
            if this dataset was already processed, one can hard code the correct slices here

        Returns
        -------

        """
        if axlist is None:
            logger.critical(
                "Need to pass axlist (use fig, (ax1,ax2,ax3) = plt.subplots(1, 3) \
                        to generate and call function with: \
                        axlist=[ax1,ax2,ax3]])"
            )
            return False
        setattr(self, "slice_signal_coronal", np.zeros((2, 1)))
        setattr(self, "slice_signal_axial", np.zeros((2, 1)))
        setattr(self, "slice_signal_sagittal", np.zeros((2, 1)))

        def plot_slice_range(
            slice_indices_cor, slice_indices_ax, slice_indices_sag, orientation
        ):
            for ax in axlist:
                ax.clear()
            if orientation == "coronal":
                axlist[0].plot(snr_per_slice["coronal"])
                axlist[0].axvline(
                    slice_indices_cor[0],
                    0,
                    np.max(snr_per_slice["coronal"]),
                    color="C0",
                    linestyle="dashed",
                )
                axlist[0].axvline(
                    slice_indices_cor[1],
                    0,
                    np.max(snr_per_slice["coronal"]),
                    color="C0",
                    linestyle="dashed",
                )

                self.slice_signal_coronal[0] = slice_indices_cor[0]
                self.slice_signal_coronal[1] = slice_indices_cor[1]

                axlist[0].set_title("Coronal")
                axlist[1].imshow(image_data[:, :, slice_indices_cor[0]])
                axlist[2].imshow(image_data[:, :, slice_indices_cor[1]])

            elif orientation == "sagittal":
                axlist[0].plot(snr_per_slice["sagittal"])
                axlist[0].axvline(
                    slice_indices_sag[0],
                    0,
                    np.max(snr_per_slice["sagittal"]),
                    color="C0",
                    linestyle="dashed",
                )
                axlist[0].axvline(
                    slice_indices_sag[1],
                    0,
                    np.max(snr_per_slice["sagittal"]),
                    color="C0",
                    linestyle="dashed",
                )

                self.slice_signal_sagittal[0] = slice_indices_sag[0]
                self.slice_signal_sagittal[1] = slice_indices_sag[1]

                axlist[0].set_title("sagittal")
                axlist[1].imshow(image_data[:, slice_indices_sag[0], :])
                axlist[2].imshow(image_data[:, slice_indices_sag[1], :])

            elif orientation == "axial":
                axlist[0].plot(snr_per_slice["axial"])
                axlist[0].axvline(
                    slice_indices_ax[0],
                    0,
                    np.max(snr_per_slice["axial"]),
                    color="C0",
                    linestyle="dashed",
                )
                axlist[0].axvline(
                    slice_indices_ax[1],
                    0,
                    np.max(snr_per_slice["axial"]),
                    color="C0",
                    linestyle="dashed",
                )

                self.slice_signal_axial[0] = slice_indices_ax[0]
                self.slice_signal_axial[1] = slice_indices_ax[1]

                axlist[0].set_title("axial")
                axlist[1].imshow(image_data[slice_indices_ax[0], :, :])
                axlist[2].imshow(image_data[slice_indices_ax[1], :, :])

            axlist[1].set_title("First slice ")
            axlist[2].set_title("Last slice")

            axlist[0].set_ylabel("Sum over all pixels and slices")
            axlist[0].set_xlabel("Slice")
            axlist[0].grid(color="k", linestyle="-", linewidth=0.25, which="major")
            axlist[0].grid(color="k", linestyle="--", linewidth=0.25, which="minor")

        # interactive input UI functions
        if slice_default_indices_cor:
            signal_range_slices_slider_cor = widgets.IntRangeSlider(
                min=0.0,
                value=[slice_default_indices_cor[0], slice_default_indices_cor[1]],
                max=image_data.shape[2] - 1,
                description="Coronal slice range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            self.slice_signal_coronal = slice_default_indices_cor
        else:
            signal_range_slices_slider_cor = widgets.IntRangeSlider(
                min=0.0,
                value=[0, image_data.shape[2] - 1],
                max=image_data.shape[2] - 1,
                description="Coronal slice range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            self.slice_signal_coronal = [0, image_data.shape[2] - 1]
        if slice_default_indices_ax:
            signal_range_slices_slider_ax = widgets.IntRangeSlider(
                min=0.0,
                value=[slice_default_indices_ax[0], slice_default_indices_ax[1]],
                max=image_data.shape[0] - 1,
                description="Axial slice range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            self.slice_signal_axial = slice_default_indices_ax
        else:
            signal_range_slices_slider_ax = widgets.IntRangeSlider(
                min=0.0,
                value=[0, image_data.shape[0] - 1],
                max=image_data.shape[0] - 1,
                description="Axial slice range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            self.slice_signal_axial = [0, image_data.shape[0] - 1]
        if slice_default_indices_sag:
            signal_range_slices_slider_sag = widgets.IntRangeSlider(
                min=0.0,
                value=[slice_default_indices_sag[0], slice_default_indices_sag[1]],
                max=image_data.shape[1] - 1,
                description="Sagittal slice range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            self.slice_signal_sagittal = slice_default_indices_sag
        else:
            signal_range_slices_slider_sag = widgets.IntRangeSlider(
                min=0.0,
                value=[0, image_data.shape[1] - 1],
                max=image_data.shape[1] - 1,
                description="Sagittal slice range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            self.slice_signal_sagittal = [0, image_data.shape[1] - 1]
        orientation_rb = widgets.RadioButtons(
            options=["coronal", "axial", "sagittal"],
            value="coronal",
            description="Orientation",
            disabled=False,
        )

        out = widgets.interactive_output(
            plot_slice_range,
            {
                "orientation": orientation_rb,
                "slice_indices_cor": signal_range_slices_slider_cor,
                "slice_indices_ax": signal_range_slices_slider_ax,
                "slice_indices_sag": signal_range_slices_slider_sag,
            },
        )
        first_ui = widgets.VBox([orientation_rb])
        second_ui = widgets.VBox(
            [
                signal_range_slices_slider_cor,
                signal_range_slices_slider_ax,
                signal_range_slices_slider_sag,
            ]
        )
        main_ui = widgets.HBox([first_ui, second_ui])
        display(main_ui, out)

    def find_noise_region(self):
        """This function is supposed to find a region containing no signal that can be used as a reference of noise"""
        return True

    # @deprecated("return_corresponding_anatomical_slice not usable anymore") # will be possible to use in 3.13
    def return_corresponding_anatomical_slice(
        self, coronal_image, slice_nr_cor, bssfp_data, plot_res=False
    ):
        """
        Finds the corresponding coronal slice of an anatomical reference to a specific bssfp scan.

        Parameters
        ----------
        coronal_image: BrukerExp instance
        slice_nr_cor: int
        bssfp_data: np.ndarray

        Returns
        -------
        cor_slice_ind: int
            slice index for anatomical reference image that corresponds to the slice_nr_cor input from a bssfp slice
        """
        dim_read_cor = coronal_image.matrix[0]  # was z
        dim_phase_cor = coronal_image.matrix[1]  # was y
        if coronal_image.method["PVM_SpatDimEnum"] == "3D":
            dim_slice_cor = coronal_image.matrix[2]  # was x
        else:
            # was x
            dim_slice_cor = coronal_image.method["PVM_SPackArrNSlices"]

        # FOV:
        mm_read_cor = coronal_image.method["PVM_Fov"][0]
        mm_phase_cor = coronal_image.method["PVM_Fov"][1]
        mm_slice_gap_cor = coronal_image.method["PVM_SPackArrSliceGap"]

        if coronal_image.method["PVM_SpatDimEnum"] == "3D":
            mm_slice_cor = coronal_image.method["PVM_Fov"][2]
        else:
            mm_slice_cor = coronal_image.method["PVM_SliceThick"]  # was x
            mm_slice_cor = mm_slice_cor * dim_slice_cor + mm_slice_gap_cor * (
                dim_slice_cor - 1
            )

        dim_read_bssfp = bssfp_data.shape[1]  # was z
        dim_phase_bssfp = bssfp_data.shape[2]  # was y
        dim_slice_bssfp = bssfp_data.shape[3]  # was x

        mm_read_bssfp = self.method["PVM_Fov"][0]
        mm_phase_bssfp = self.method["PVM_Fov"][1]
        mm_slice_bssfp = self.method["PVM_Fov"][2]

        cor_grid = self.define_grid(
            mat=np.array((dim_read_cor, dim_phase_cor, dim_slice_cor)),
            fov=np.array((mm_read_cor, mm_phase_cor, mm_slice_cor)),
        )

        bssfp_grid = self.define_grid(
            mat=np.array((dim_read_bssfp, dim_phase_bssfp, dim_slice_bssfp)),
            fov=np.array((mm_read_bssfp, mm_phase_bssfp, mm_slice_bssfp)),
        )
        coronal_image = coronal_image.seq2d

        if slice_nr_cor >= dim_slice_bssfp:
            logger.critical(
                "Input coronal bssfp slice number cannot exceed " + str(dim_slice_bssfp)
            )
            return None
        else:
            # input slice nr from bssfp slices
            ax_slice_ind, cor_slice_ind = self.find_closest_slice(
                axial_anat_grid=None,
                coronal_anat_grid=cor_grid,
                bssfp_grid=bssfp_grid,
                bssfp_pos=[
                    None,
                    None,
                    slice_nr_cor,
                ],
            )
            if plot_res:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(
                    np.sum(np.abs(bssfp_data[0, :, :, slice_nr_cor, :, 0]), axis=2)
                )
                ax[1].imshow(coronal_image[:, :, cor_slice_ind], cmap="bone")
                ax[0].set_title("Bssfp, coronal direction, slice " + str(slice_nr_cor))
                ax[1].set_title("Coronal image, slice " + str(cor_slice_ind))
            return cor_slice_ind

    def plot3D(self, axlist, fig, parameter_dict=None, display_ui=True):
        """A helper function for an interactive image plot.

        Recommended usage:
            import matplotlib.pyplot as plt
            from hypermri.sequences import BSSFP
            from IPython.display import display

            # create  BSSFP object
            bSSFP1 = BSSFP(...)

            # create figure and axes objects to plot into
            fig, (ax1,ax2,ax3) = plt.subplots(ncols=3)

            # plot mge data into given axes object, provide custom parameter
            plot_params = {'cmap': 'viridis'}
            bSSFP1.plot3D([ax1,ax2,ax3], plot_params)

            # note: you can still add stuff to the axes object afterwards
            ax1.set_title("Axial")
            ax2.set_title("Coronal")
            ax3.set_title("sagittal")

            # render plot
            plt.show()

        Parameters
        ----------
        axlist : list of Axes
            The axes to draw to. Recommended to generate using the command:
            'fig, ax = plt.subplots()'.
            3D plot requires 3 axes. 1st axial, 2nd coronal, 3rd sagittal

        param_dict : dict
            Dictionary of keyword arguments to pass to ax.imshow.

        display_ui: bool
            Default is True. Directly calls IPython.display on the interactive
            sliders and renders them. If False, returns the ui object instead of
            displaying it.

        Returns
        ----------
        ui : Hbox, optional
            The ipywidgets HBox containing the interactive sliders is returned
            only if the option 'display_ui' is set to False.
        """
        # check if there is data to plot
        if not isinstance(self.Reconstructed_data, np.ndarray):
            logger.debug(
                "There is no reconstructed data available. Please call the function BSSFP.reconstruction first!"
            )

        else:
            if self.method["PVM_SpatDimEnum"] != "3D":
                logger.debug("3D print function but no 3D data available!")
            else:
                # read image dimensions
                dim_read = self.matrix[0]
                dim_phase = self.matrix[1]
                dim_slice = self.matrix[2]

                # use default paramter dictionary if none is provided
                if not parameter_dict:
                    parameter_dict = {"cmap": "bone"}

                # define image plotting function
                def plot_img(slice_nr_ax, slice_nr_cor, slice_nr_sag, echo_nr):
                    if len(axlist) != 3:
                        logger.debug(
                            "Not enough ax objects in axlist for 3D plot, please give 3 axes!"
                        )
                    else:
                        for ax in axlist:
                            ax.clear()

                        axlist[0].imshow(
                            np.rot90(
                                np.abs(
                                    self.Reconstructed_data[
                                        echo_nr, slice_nr_ax, :, :, 0, 0
                                    ]
                                ),
                                3,
                            ),
                            extent=[
                                (dim_phase // 2),
                                -(dim_phase // 2),
                                (dim_slice // 2),
                                -(dim_slice // 2),
                            ],
                            **parameter_dict,
                        )
                        axlist[0].set_aspect(dim_slice / dim_phase)
                        axlist[0].axvline(
                            slice_nr_sag - (dim_slice // 2),
                            color="white",
                            linewidth=1,
                            linestyle="dashed",
                        )
                        axlist[0].axhline(
                            slice_nr_cor - (dim_phase // 2),
                            color="white",
                            linewidth=1,
                            linestyle="dashed",
                        )
                        axlist[0].set_xlabel("phase")
                        axlist[0].set_ylabel("slice")
                        axlist[0].set_title("Axial")

                        axlist[1].imshow(
                            np.rot90(
                                np.abs(
                                    self.Reconstructed_data[
                                        echo_nr, :, :, slice_nr_cor, 0, 0
                                    ]
                                ),
                                2,
                            ),
                            extent=[
                                dim_phase // 2,
                                -dim_phase // 2,
                                -dim_read // 2,
                                dim_read // 2,
                            ],
                            **parameter_dict,
                        )
                        axlist[1].set_aspect(dim_phase / dim_read)
                        axlist[1].axhline(
                            slice_nr_ax - (dim_read // 2),
                            color="white",
                            linewidth=1,
                            linestyle="dashed",
                        )
                        axlist[1].axvline(
                            slice_nr_sag - (dim_phase // 2),
                            color="white",
                            linewidth=1,
                            linestyle="dashed",
                        )
                        axlist[1].set_xlabel("phase")
                        axlist[1].set_ylabel("read")
                        axlist[1].set_title("Coronal")

                        axlist[2].imshow(
                            np.rot90(
                                np.abs(
                                    self.Reconstructed_data[
                                        echo_nr, :, slice_nr_sag, :, 0, 0
                                    ]
                                ),
                                2,
                            ),
                            extent=[
                                dim_slice // 2,
                                -dim_read // 2,
                                -dim_read // 2,
                                dim_read // 2,
                            ],
                            **parameter_dict,
                        )
                        axlist[2].set_aspect(dim_slice / dim_read)
                        axlist[2].axhline(
                            slice_nr_ax - (dim_read // 2),
                            color="white",
                            linewidth=1,
                            linestyle="dashed",
                        )
                        axlist[2].axvline(
                            slice_nr_cor - (dim_slice // 2),
                            color="white",
                            linewidth=1,
                            linestyle="dashed",
                        )
                        axlist[2].set_xlabel("slice")
                        axlist[2].set_ylabel("read")
                        axlist[2].set_title("sagittal")

                # create interactive sliders for  slices in all dimensions
                slice_slider_ax = widgets.IntSlider(
                    value=dim_read // 2,
                    min=0,
                    max=dim_read - 1,
                    description="Slice Axial: ",
                )
                slice_slider_cor = widgets.IntSlider(
                    value=dim_phase // 2,
                    min=0,
                    max=dim_phase - 1,
                    description="Slice Coronal: ",
                )
                slice_slider_sag = widgets.IntSlider(
                    value=dim_slice // 2,
                    min=0,
                    max=dim_slice - 1,
                    description="Slice sagittal: ",
                )

                # create interactive slider for echoes
                echo_slider = widgets.IntSlider(
                    value=0, min=0, max=self.n_echoes - 1, description="Echo: "
                )

                # put both sliders inside a HBox for nice alignment  etc.
                ui_ax = widgets.VBox(
                    [slice_slider_ax, echo_slider],
                    layout=widgets.Layout(display="flex"),
                )

                ui_cor = widgets.VBox(
                    [slice_slider_cor],
                    layout=widgets.Layout(display="flex"),
                )

                ui_sag = widgets.VBox(
                    [slice_slider_sag],
                    layout=widgets.Layout(display="flex"),
                )

                main_ui = widgets.HBox((ui_ax, ui_cor, ui_sag))

                # connect plotting function with sliders
                # Note: The output is unused here because it is sufficient to display
                #       the HBox (stored in ui variable). Without the ui display you
                #       would have to call the display() function on the output of
                #       widgets.interactive_output().
                out = widgets.interactive_output(
                    plot_img,
                    {
                        "slice_nr_ax": slice_slider_ax,
                        "slice_nr_cor": slice_slider_cor,
                        "slice_nr_sag": slice_slider_sag,
                        "echo_nr": echo_slider,
                    },
                )

                # This displays the Hbox containing the slider widgets.
                if display_ui:
                    display(main_ui, out)
                else:
                    return main_ui

    def plot3D_new(
        self,
        axlist,
        anat_object=None,
        input_data=None,
        parameter_dict=None,
        display_ui=True,
        pyr_ind=0,
    ):
        """A helper function for an interactive image plot.

        Recommended usage:
            import matplotlib.pyplot as plt
            from hypermri.sequences import BSSFP
            from IPython.display import display

            # create  BSSFP object
            bSSFP1 = BSSFP(...)

            # create figure and axes objects to plot into
            fig, (ax1,ax2,ax3) = plt.subplots(ncols=3)

            # plot mge data into given axes object, provide custom parameter
            plot_params = {'cmap': 'viridis'}
            bSSFP1.plot3D([ax1,ax2,ax3], plot_params)

            # note: you can still add stuff to the axes object afterwards
            ax1.set_title("Axial")
            ax2.set_title("Coronal")
            ax3.set_title("sagittal")

            # render plot
            plt.show()

        Parameters
        ----------
        axlist : list of Axes
            The axes to draw to. Recommended to generate using the command:
            'fig, ax = plt.subplots()'.
            3D plot requires 3 axes. 1st axial, 2nd coronal, 3rd sagittal
        anat_object : BrukerExp instance
            contains anatomical reference images, usually coronal.
        input_data: nd-array (None)
            nd-array to plot, data is supposed to sorted in the order
            Echoes, Read, phase, slice, repetitions, channels

        param_dict : dict
            Dictionary of keyword arguments to pass to ax.imshow.

        display_ui: bool
            Default is True. Directly calls IPython.display on the interactive
            sliders and renders them. If False, returns the ui object instead of
            displaying it.

        Returns
        ----------
        ui : Hbox, optional
            The ipywidgets HBox containing the interactive sliders is returned
            only if the option 'display_ui' is set to False.
        """
        from IPython.display import clear_output

        # check if there is data to plot
        if input_data is None:
            if not isinstance(self.Reconstructed_data, np.ndarray):
                logger.debug(
                    "There is no reconstructed data available. "
                    + "Please call the function BSSFP.reconstruction first!"
                )
            else:
                data = self.Reconstructed_data
        else:
            data = input_data

        if self.method["PVM_SpatDimEnum"] != "3D":
            logger.debug("3D print function but no 3D data available!")
        else:
            self.data_to_plot = data
            # read image dimensions
            dim_read = self.matrix[0]  # was z
            dim_phase = self.matrix[1]  # was y
            dim_slice = self.matrix[2]  # was x
            mm_read = self.method["PVM_Fov"][0]
            mm_phase = self.method["PVM_Fov"][1]
            mm_slice = self.method["PVM_Fov"][2]
            n_repetitions = self.n_repetitions
            n_receivers = self.n_receivers
            slice_thick_cor = mm_slice / dim_slice
            slice_thick_ax = mm_read / dim_read
            slice_thick_sag = mm_phase / dim_phase
            read_orientation = self.read_orientation
            self.method["PVM_SPackArrReadOffset"]
            self.method["PVM_SPackArrPhase1Offset"]
            self.method["PVM_SPackArrSliceOffset"]
            if not read_orientation == "H_F":
                # checking if we have the right read orientation
                # if this error comes up you have to implement different
                # orientations possibly
                logger.debug(
                    "Careful this function is not evaluated for this read orientation"
                )
            else:
                pass

            # if anat_object is None:
            #     # transform the input data
            #     (
            #         transformed_cor,
            #         self.data_to_plot,
            #         anat_cor_ext,
            #         bssfp_ext,
            #     ) = orient_coronal_and_bssfp_for_plot(anat_object, self, input_data)
            # else:
            #     pass
            anat_cor_ext = None

            # hardcode for now:
            transformed_cor = anat_object

            # use default paramter dictionary if none is provided
            if not parameter_dict:
                parameter_dict = {"cmap": "viridis"}

            # define image plotting function
            def plot_img(
                slice_nr_ax,
                slice_nr_cor,
                slice_nr_sag,
                echo_nr,
                rep1,
                chan,
                rep2=None,
                plot_style="Abs",
                experiment_style="Perfusion",
                pyr_lac="Pyr",
                anat_overlay="Metab",
            ):
                if len(axlist) < 3:
                    logger.debug(
                        "Not enough ax objects in axlist for 3D plot, please give 3 axes!"
                    )
                else:
                    # init range:
                    rep_range = rep1
                    # choose plotstyle:
                    if plot_style == "Abs":
                        bssfp_image_data = np.absolute(self.data_to_plot)
                    elif plot_style == "Real":
                        bssfp_image_data = np.real(self.data_to_plot)
                    elif plot_style == "Imag":
                        bssfp_image_data = np.imag(self.data_to_plot)
                    elif plot_style == "Phase":
                        bssfp_image_data = np.angle(self.data_to_plot)
                    else:
                        # default
                        bssfp_image_data = np.absolute(self.data_to_plot)

                    # init empty arrays:
                    global pyr_data, lac_data
                    # deterMine experiment style:
                    if experiment_style == "Perfusion":
                        pyr_data = bssfp_image_data[:, :, :, :, :, :]
                    elif experiment_style == "Pyr + Lac":
                        pyr_data = bssfp_image_data[:, :, :, :, pyr_ind::2, :]
                        lac_data = bssfp_image_data[:, :, :, :, pyr_ind + 1 :: 2, :]
                    else:
                        Warning(
                            "unknown measurement style "
                            + str(plot_opts_measurement.value)
                        )

                    # determine which contrast to use:
                    if pyr_lac == "Pyr":
                        bssfp_image_data = pyr_data
                    elif (pyr_lac == "Lac") and (experiment_style == "Pyr + Lac"):
                        bssfp_image_data = lac_data
                    elif (pyr_lac == "Lac/Pyr") and (experiment_style == "Pyr + Lac"):
                        bssfp_image_data = lac_data / pyr_data
                    else:
                        bssfp_image_data = pyr_data
                        Warning("unknown Contrast" + str(plot_opts_metabolite.value))

                    # takes mean over both repetitions:
                    if rep_avg_checkbox.value:
                        # valid range:
                        if rep2 > rep1:
                            rep_range = range(rep1, rep2)
                            # assuming the order Echoes, Read, phase, slice,
                            # repetitions, channels in the data:
                            bssfp_image_data = np.mean(
                                bssfp_image_data[:, :, :, :, rep_range, :],
                                axis=4,
                                keepdims=True,
                            )
                            rep_range = 0
                        # use start repetition
                        else:
                            rep_range = rep1
                    # take start repetition value:
                    else:
                        rep_range = rep1

                    # clear axis:
                    for ax in axlist:
                        ax.clear()

                    # axial view:
                    axlist[0].imshow(
                        np.squeeze(
                            (
                                bssfp_image_data[
                                    echo_nr, slice_nr_ax, :, :, rep_range, chan
                                ]
                            )
                        ),
                        extent=self.ax_ext,
                        **parameter_dict,
                    )
                    axlist[0].set_xlabel("mm (phase)")
                    axlist[0].set_ylabel("mm (slice)")

                    axlist[0].set_title("Axial")
                    # axlist[0].set_aspect(dim_y / dim_z)

                    axlist[0].axhline(
                        mm_slice / 2 - slice_nr_cor * slice_thick_cor,
                        color="white",
                        linewidth=1,
                        linestyle="solid",
                    )
                    axlist[0].axvline(
                        mm_phase / 2 - slice_nr_sag * slice_thick_sag,
                        color="white",
                        linewidth=1,
                        linestyle="dashed",
                    )

                    # sagittal view:
                    axlist[1].imshow(
                        np.squeeze(
                            (
                                bssfp_image_data[
                                    echo_nr, :, slice_nr_sag, :, rep_range, chan
                                ]
                            )
                        ),
                        extent=self.sag_ext,
                        **parameter_dict,
                    )
                    axlist[1].set_xlabel("mm (read)")
                    axlist[1].set_ylabel("mm (slice)")

                    axlist[1].set_title("Sagittal")
                    # axlist[0].set_aspect(dim_y / dim_z)

                    axlist[1].axhline(
                        mm_slice / 2 - slice_nr_cor * slice_thick_cor,
                        color="white",
                        linewidth=1,
                        linestyle="solid",
                    )
                    axlist[1].axvline(
                        mm_read / 2 - slice_nr_ax * slice_thick_ax,
                        color="white",
                        linewidth=1,
                        linestyle="dashed",
                    )

                    # coronal view:

                    if anat_overlay == "Metab + Anat":
                        axlist[2].imshow(
                            np.squeeze((transformed_cor[:, :, slice_nr_cor])),
                            extent=anat_cor_ext,
                            cmap="bone",
                        )
                        # coronal bssfp data
                        axlist[2].imshow(
                            np.squeeze(
                                (
                                    bssfp_image_data[
                                        echo_nr, :, :, slice_nr_cor, rep_range, chan
                                    ]
                                )
                            ),
                            extent=self.cor_ext,
                            alpha=0.3,
                            **parameter_dict,
                        )
                        axlist[2].set_title("Coronal")
                        axlist[2].set_xlabel("mm")
                        axlist[2].set_ylabel("mm")
                    elif anat_overlay == "None":
                        axlist[2].imshow(
                            np.squeeze(
                                (
                                    bssfp_image_data[
                                        echo_nr, :, :, slice_nr_cor, rep_range, chan
                                    ]
                                )
                            ),
                            extent=self.cor_ext,
                            **parameter_dict,
                        )

                    elif anat_overlay == "Anat":
                        axlist[2].imshow(
                            np.squeeze((transformed_cor[:, :, slice_nr_cor])),
                            extent=anat_cor_ext,
                            cmap="bone",
                        )

                    # axlist[0].set_aspect(dim_y / dim_z)
                    axlist[2].set_xlabel("mm (phase)")
                    axlist[2].set_ylabel("mm (read)")
                    axlist[2].set_title("Coronal")
                    axlist[2].axhline(
                        mm_read / 2 - slice_nr_ax * slice_thick_ax,
                        color="white",
                        linewidth=1,
                        linestyle="solid",
                    )
                    axlist[2].axvline(
                        mm_phase / 2 - slice_nr_sag * slice_thick_sag,
                        color="white",
                        linewidth=1,
                        linestyle="dashed",
                    )

                    # plot time curve
                    if len(axlist) > 3:
                        if pyr_lac == "Pyr":
                            axlist[3].plot(
                                np.squeeze(
                                    (
                                        pyr_data[
                                            echo_nr,
                                            slice_nr_ax,
                                            slice_nr_sag,
                                            slice_nr_cor,
                                            :,
                                            chan,
                                        ]
                                    )
                                ),
                                linewidth=2,
                            )
                            # only possible if pyruvate and lactate were
                            # measured
                            if experiment_style == "Pyr + Lac":
                                axlist[3].plot(
                                    np.squeeze(
                                        (
                                            lac_data[
                                                echo_nr,
                                                slice_nr_ax,
                                                slice_nr_sag,
                                                slice_nr_cor,
                                                :,
                                                chan,
                                            ]
                                        )
                                    ),
                                    linewidth=1,
                                )
                        # only possible if pyruvate and lactate were measured
                        elif (experiment_style == "Pyr + Lac") and (pyr_lac == "Lac"):
                            axlist[3].plot(
                                np.squeeze(
                                    (
                                        pyr_data[
                                            echo_nr,
                                            slice_nr_ax,
                                            slice_nr_sag,
                                            slice_nr_cor,
                                            :,
                                            chan,
                                        ]
                                    )
                                ),
                                linewidth=1,
                            )
                            axlist[3].plot(
                                np.squeeze(
                                    (
                                        lac_data[
                                            echo_nr,
                                            slice_nr_ax,
                                            slice_nr_sag,
                                            slice_nr_cor,
                                            :,
                                            chan,
                                        ]
                                    )
                                ),
                                linewidth=2,
                            )
                        elif (experiment_style == "Pyr + Lac") and (
                            pyr_lac == "Lac/Pyr"
                        ):
                            axlist[3].plot(
                                np.squeeze(
                                    (
                                        lac_data[
                                            echo_nr,
                                            slice_nr_ax,
                                            slice_nr_sag,
                                            slice_nr_cor,
                                            :,
                                            chan,
                                        ]
                                    )
                                )
                                / np.squeeze(
                                    (
                                        pyr_data[
                                            echo_nr,
                                            slice_nr_ax,
                                            slice_nr_sag,
                                            slice_nr_cor,
                                            :,
                                            chan,
                                        ]
                                    )
                                ),
                                linewidth=1,
                            )
                        else:
                            Warning("dont know what to do")
                            axlist[3].plot(
                                np.squeeze(
                                    (
                                        pyr_data[
                                            echo_nr,
                                            slice_nr_ax,
                                            slice_nr_sag,
                                            slice_nr_cor,
                                            :,
                                            chan,
                                        ]
                                    )
                                ),
                                linewidth=2,
                            )

                        axlist[3].axvline(
                            rep1,
                            color="black",
                            linewidth=1,
                            linestyle="dashed",
                        )
                        # draw 2nd line:
                        if (rep2 > rep1) and (rep_avg_checkbox.value):
                            axlist[3].axvline(
                                rep2,
                                color="black",
                                linewidth=1,
                                linestyle="dashed",
                            )

            global rep_avg_checkbox_option
            # create interactive sliders for  slices in all dimensions
            rep_avg_checkbox = widgets.Checkbox(description="Mean Reps")
            rep_avg_checkbox_option = widgets.VBox(
                layout=widgets.Layout(display="flex")
            )

            plot_opts_anat_overlay = widgets.RadioButtons(
                options=["Metab", "Metab + Anat", "Anat"],
                value="Metab",
                description="Anat overlay",
                disabled=False,
                layout=widgets.Layout(width="15%"),
            )

            # create interactive slider for echoes
            echo_slider = widgets.IntSlider(
                value=0, min=0, max=self.n_echoes - 1, description="Echo: "
            )

            slice_slider_rep1 = widgets.IntSlider(
                value=0,
                min=0,
                max=n_repetitions - 1,
                description="Rep. Start: ",
            )
            slice_slider_rep2 = widgets.IntSlider(
                value=0,
                min=0,
                max=n_repetitions - 1,
                description="Rep. End: ",
            )
            # set the minimum of the end reptition to the value of the start
            # reptition
            widgetLink = widgets.jslink(
                (slice_slider_rep1, "value"), (slice_slider_rep2, "min")
            )

            plot_opts_part = widgets.RadioButtons(
                options=["Abs", "Real", "Imag", "Phase"],
                value="Abs",
                description="Plot style:",
                disabled=False,
                layout=widgets.Layout(width="10%"),
            )

            plot_opts_metabolite = widgets.RadioButtons(
                options=["Pyr", "Lac", "Lac/Pyr"],
                value="Pyr",
                description="Metabolite:",
                disabled=True,  # activate if plot_opts_measurement is set to Pyr+Lac
                layout=widgets.Layout(width="10%"),
            )

            plot_opts_measurement = widgets.RadioButtons(
                options=["Perfusion", "Pyr + Lac"],
                value="Perfusion",  # Defaults to 'pineapple'
                description="Experiment:",
                disabled=False,
                layout=widgets.Layout(width="10%"),
            )

            # init sliders:
            slice_slider_ax = widgets.IntSlider(
                value=mm_read // 2,
                min=0,
                max=dim_read - 1,
                description="Slice Axial: ",
            )
            slice_slider_cor = widgets.IntSlider(
                value=mm_slice // 2,
                min=0,
                max=dim_slice - 1,
                description="Slice Coronal: ",
            )
            slice_slider_sag = widgets.IntSlider(
                value=mm_phase // 2,
                min=0,
                max=dim_phase - 1,
                description="Slice sagittal: ",
            )
            slice_slider_chan = widgets.IntSlider(
                value=0,
                min=0,
                max=n_receivers - 1,
                description="Channel: ",
            )

            def set_rep_range(args):
                """
                Sets repetition range and contrast options (Pyr, Lac, ...) depending
                on the experiment type
                """
                # changed from perfusion to pyr+lac
                if args["new"] == "Pyr + Lac":
                    slice_slider_rep1.max = np.ceil(n_repetitions / 2) - 1
                    slice_slider_rep2.max = np.ceil(n_repetitions / 2) - 1

                    slice_slider_rep1.value = np.ceil(slice_slider_rep1.value / 2)
                    slice_slider_rep2.value = np.ceil(slice_slider_rep2.value / 2)

                    plot_opts_metabolite.disabled = False
                elif args["new"] == "Perfusion":
                    slice_slider_rep1.max = n_repetitions - 1
                    slice_slider_rep2.max = n_repetitions - 1

                    slice_slider_rep1.value = slice_slider_rep1.value * 2
                    slice_slider_rep2.value = slice_slider_rep2.value * 2

                    plot_opts_metabolite.value = "Pyr"
                    plot_opts_metabolite.disabled = True
                else:
                    Warning("Unkown experiment type" + str(args["new"]))

            # change options depending on experiment type:
            plot_opts_measurement.observe(set_rep_range, names="value")

            # generate boxes:
            ui_sag = widgets.VBox(
                [
                    slice_slider_sag,
                    slice_slider_rep1,
                    slice_slider_rep2,
                    rep_avg_checkbox,
                    plot_opts_anat_overlay,
                    plot_opts_measurement,
                ],
                layout=widgets.Layout(width="33%"),
            )
            ui_ax = widgets.VBox(
                [slice_slider_ax, echo_slider, plot_opts_part],
                layout=widgets.Layout(width="33%"),
            )
            ui_cor = widgets.VBox(
                [slice_slider_cor, slice_slider_chan, plot_opts_metabolite],
                layout=widgets.Layout(width="33%"),
            )

            # def plot_ui(args):
            clear_output()
            # connect plotting function with sliders
            # Note: The output is unused here because it is sufficient to display
            #       the HBox (stored in ui variable). Without the ui display you
            #       would have to call the display() function on the output of
            #       widgets.interactive_output().def update_ui
            out = widgets.interactive_output(
                plot_img,
                {
                    "slice_nr_ax": slice_slider_ax,
                    "slice_nr_cor": slice_slider_cor,
                    "slice_nr_sag": slice_slider_sag,
                    "echo_nr": echo_slider,
                    "rep1": slice_slider_rep1,
                    "chan": slice_slider_chan,
                    "rep2": slice_slider_rep2,
                    "plot_style": plot_opts_part,
                    "experiment_style": plot_opts_measurement,
                    "pyr_lac": plot_opts_metabolite,
                    "anat_overlay": plot_opts_anat_overlay,
                },
            )
            main_ui = widgets.HBox([ui_ax, ui_cor, ui_sag])

            # This displays the Hbox containing the slider widgets.
            if display_ui:
                display(main_ui, out)
            else:
                return main_ui
