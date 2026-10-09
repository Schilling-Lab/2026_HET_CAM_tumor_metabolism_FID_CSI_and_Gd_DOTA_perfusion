"""
Base class used to read in one Siemens experiment.
"""

import os
from datetime import datetime
from pathlib import Path

import git
import mapvbvd
import numpy as np
import xarray as xr

from hypermri.compatibility import assign_frequency_axis
from hypermri.utils.system.logging import LOG_MODES, init_default_logger
from hypermri.vendorhandler import VendorHandler

# for paper_summary() method only


# Set a global variable to control debug message visibility
DB_MSG = True  # Change this to False to hide detailed error messages
DATA_PRECISION = np.float16


logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


class SiemensHandler(VendorHandler):
    """Base class for experiments conducted on Siemens systems."""

    def __init__(self, source, load_data=True, log_mode="warning"):
        """
        Initialize the siemensexp class.

        Parameters
        ----------
        source : str or SiemensExp-object
            The complete path to measurement/experiment folder
            In case 'source' is a SiemensExp object the path is automatically
            extracted form the object. That makes reloading a SiemensExp easier.

        Raises
        ------
        FileNotFoundError
            If path provided does not lead to a directory containing a .dat file.

        """
        logger.setLevel(LOG_MODES[log_mode])

        input_path = source.path if isinstance(source, SiemensHandler) else source

        dat_file_path = self.__find_dat_file(input_path)

        if dat_file_path:
            self.path = Path(dat_file_path)
            self.twix = mapvbvd.mapVBVD(str(self.path))

            self.hdr = self.twix.hdr
            self.method = self.hdr["MeasYaps"]
            self.acqp = self.hdr["Config"]
            self.visu = self.hdr["Meas"]

            # Initialize properties to None or default values
            self._rps_axes_mm = None
            self._rps_extent_mm = None
            self._rps_grid_mm = None
            self._xyz_axes_mm = None
            self._xyz_extent_mm = None
            self._xyz_grid_mm = None
            self.scanner_source = "Siemens"

            # ---------- Scan parameter files: ----------
            # ExpNum: Derived from MeasUID, or a placeholder
            self.ExpNum = (
                int(self.visu.get("MeasUid", 0)) if self.visu.get("MeasUid", 0) else -1
            )
            print(f"[DEBUG] SiemensExp: ExpNum set to {self.ExpNum}")

            # define the measurement type (placeholder for now)
            self.type = self.method.get(("tSequenceFileName",), None)
            if self.type is not None:
                self.type = self.type.strip('"')
                if "Siemens" in self.type:  # Placeholder logic
                    self.method_type = "stock"
                else:
                    self.method_type = "custom"
            else:
                self.method_type = "unknown"

            # define Nucleus:
            self.nucleus = self.method.get(
                ("sTXSPEC", "asNucleusInfo", "0", "tNucleus"), "1H"
            ).strip('"')

            self.is_spectroscopic = (
                self.method.get(("ucCSI",), "0x0") == "0x1"
                or self.method.get(("ucSVSSeq",), "0x0") == "0x1"
            )

            # Number of slice packs:
            self.n_slice_packs = self.method.get(("sSliceArray", "lSize"), 1)

            # set cutoff in spectroscopy:
            if self.is_spectroscopic:
                self.cut_off = (
                    70  # Default value, needs to be confirmed from Siemens headers
                )
            else:
                self.cut_off = 0

            # the time the SiemensExp object was last edited on the scanner
            self.timestamp = datetime.fromtimestamp(self.acqp.get("tTimeStamp", 0))

            # ---------- Scan subject ----------
            self.subject_name = self.visu.get("tPatientName", None)
            if self.subject_name is not None:
                self.subject_name = self.subject_name.strip('"')
            self.subject_id = self.visu.get("PatientID", None)
            if self.subject_id is not None:
                self.subject_id = self.subject_id.strip('"')
            self.subject_position = self.visu.get("tPatientPosition", None)
            if self.subject_position is not None:
                self.subject_position = self.subject_position.strip('"')
            self.subject_type = None  # Not directly available in provided headers

            # ---------- Scan Geometry ----------
            self.spat_dim = self.method.get(("sKSpace", "ucDimension"), None)
            if self.spat_dim == 2:
                self.spat_dim = "2D"
            elif self.spat_dim == 3:
                self.spat_dim = "3D"
            else:
                self.spat_dim = None

            # list of names of all recon folders inside 'pdata' (not applicable
            # for Siemens)
            self.recon_names = []
            # add number of receiver channels
            self.n_receivers = self.method.get(
                ("sProtConsistencyInfo", "lMaximumNofRxReceiverChannels"), 1
            )

            # add number of slices
            self.n_slices = self.method.get(("sSliceArray", "lSize"), 1)
            # slice thickess:
            self.slice_thickness_mm = self.method.get(
                ("sSliceArray", "asSlice", "0", "dThickness"), 0
            )
            # gap between slices:
            self.slice_gap_mm = 0  # Placeholder, couldn't find a clear equivalent
            self.acquisition_matrix = [
                self.visu.get("iNoOfFourierColumns", 0),
                self.visu.get("iNoOfFourierLines", 0),
                self.visu.get("iNoOfFourierPartitions", 0),
            ]
            self.patient_pos = self.visu.get("tPatientPosition", "HFS").strip('"')
            self.read_orientation = None  # Not directly available
            self.slice_orientation = None  # Not directly available
            self.slice_order = np.atleast_1d(0)  # Placeholder

            self.fov_mm = [
                self.method.get(("sSliceArray", "asSlice", "0", "dReadoutFOV"), 0),
                self.method.get(("sSliceArray", "asSlice", "0", "dPhaseFOV"), 0),
            ]
            # For PRESS, CSI etc.
            if self.is_spectroscopic and self.fov_mm[0] == 0:
                self.fov_mm = [
                    self.method.get(("sSpecPara", "sVoI", "dReadoutFOV"), 0),
                    self.method.get(("sSpecPara", "sVoI", "dPhaseFOV"), 0),
                    self.method.get(("sSpecPara", "sVoI", "dThickness"), 0),
                ]
            # add matrix:
            self.matrix = [
                self.method.get(("sKSpace", "lBaseResolution"), 1),
                self.method.get(("sKSpace", "lPhaseEncodingLines"), 1),
                self.n_slices,
            ]
            if self.is_spectroscopic:
                self.matrix = np.array([1, 1, 1])

            # add encoding matrix (actual acquired points, can be affected by
            # Partial Fourier e.g.):
            self.enc_matrix = [
                self.method.get(("sKSpace", "lBaseResolution"), 1),
                self.method.get(("sKSpace", "lPhaseEncodingLines"), 1),
                self.method.get(("sKSpace", "lPartitions"), 1),
            ]

            # add number of repetitions:
            self.n_repetitions = self.method.get(("lRepetitions",), 1)
            # add number of averages:
            self.n_averages = self.method.get(("lAverages",), 1)
            # add number of echoes:
            self.n_echoes = self.method.get(("lContrasts",), 1)
            # add number of spectral points:
            self.n_acquisition = self.method.get(("sSpecPara", "lVectorSize"), 1)
            print(f"[DEBUG] SiemensExp: n_acquisition set to {self.n_acquisition}")
            # add number of variable TRs:
            self.inversion_time_ms = self.method.get(("alTI", "0"), 0)
            if isinstance(
                self.inversion_time_ms, str
            ):  # Handle cases where it's a string of values
                self.inversion_time_ms = np.array(
                    [float(x) for x in self.inversion_time_ms.split()]
                )
            else:
                self.inversion_time_ms = np.atleast_1d(self.inversion_time_ms)

            self.n_inversion_time = (
                len(self.inversion_time_ms) if self.inversion_time_ms is not None else 1
            )

            # add resolution:
            if self.fov_mm is None or self.matrix is None or 0 in self.matrix:
                self.resolution_mm = None
            else:
                try:
                    self.resolution_mm = np.array(self.fov_mm) / np.array(self.matrix)
                except BaseException:
                    self.resolution_mm = None

            self.read_offset_mm = None
            self.phase_offset_mm = None
            self.phase_0offset_mm = None
            self.phase_1offset_mm = None
            self.slice_offset_mm = None
            self.slice_offset_mm_array = None

            # ---------- Scan RF Coil ----------
            self.coil_receive = self.visu.get(
                "tTransmittingCoil", None
            )  # Placeholder, often combined in Siemens
            self.coil_transmit = self.visu.get(
                "tTransmittingCoil", None
            )  # Placeholder, often combined in Siemens

            # ---------- More Sequence parameters ----------
            position_matrix = np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            )  # Default identity
            # Add logic to adjust position_matrix based on self.patient_pos if
            # needed
            self.position_matrix = position_matrix

            self.b_values = self.method.get(("sDiffusion", "alBValue", "0"), 0)
            if isinstance(self.b_values, str):
                self.b_values = np.array([float(x) for x in self.b_values.split()])
            else:
                self.b_values = np.atleast_1d(self.b_values)

            self.n_voxels = self.method.get(
                ("sSpecPara", "lVectorSize"), 1
            )  # Using vector size as a proxy
            self.rare_factor = self.method.get(("sFastImaging", "lTurboFactor"), 1)

            self.scan_duration_s = self.method.get(("lTotalScanTimeSec",), 0)

            dwell_time = self.method.get(("sRXSPEC", "alDwellTime", "0"), 0)
            if dwell_time > 0:
                self.bandwidth_hz = (1 / dwell_time) * 1e9 / 2
            else:
                self.bandwidth_hz = None

            self.echo_time_ms = self.method.get(("alTE", "0"), 0) / 1000
            self.repetition_time_ms = self.method.get(("alTR", "0"), 0) / 1000
            self.frame_time_s = None  # Requires calculation
            self.sampling_time_axis_s = None  # Requires calculation

            # ---------- RF Pulse parameters ----------
            # Siemens headers are complex for RF pulses, mapping basic ones
            self.rf_pulse_params = self.method.get(("sTXSPEC", "aRFPULSE", "0"), None)
            if self.rf_pulse_params is not None:
                self.rfpulse_duration_ms = self.rf_pulse_params.get(
                    "flDuration", 0
                )  # Placeholder
                self.rfpulse_bandwidth_Hz = 0  # Placeholder
                self.flipangle_deg = self.method.get(("adFlipAngleDegree", "0"), 0)
                self.rfpulse_is_calculated = False  # Placeholder
                self.rfpulse_sharpness = 0  # Placeholder
                self.rfpulse_bwfac_Hzms = 0  # Placeholder
                self.rfpulse_sint = 0  # Placeholder
                self.rfpulse_pint = 0  # Placeholder
                self.rfpulse_type = self.rf_pulse_params.get("tName", "").strip(
                    '"'
                )  # Placeholder
                self.rfpulse_rpfac_percent = 0  # Placeholder
                self.rfpulse_peakpow_W = 0  # Placeholder
                self.rfpulse_shape = self.rf_pulse_params.get("tName", "").strip(
                    '"'
                )  # Placeholder
                self.rfpulse_refpow_W = 0  # Placeholder
            else:
                self.rfpulse_duration_ms = 0
                self.rfpulse_bandwidth_Hz = 0
                self.flipangle_deg = self.method.get(("adFlipAngleDegree", "0"), 0)
                self.rfpulse_is_calculated = False
                self.rfpulse_sharpness = 0
                self.rfpulse_bwfac_Hzms = 0
                self.rfpulse_sint = 0
                self.rfpulse_pint = 0
                self.rfpulse_type = ""
                self.rfpulse_rpfac_percent = 0
                self.rfpulse_peakpow_W = 0
                self.rfpulse_shape = ""
                self.rfpulse_refpow_W = 0

            self.ref_pulse_params = None  # Not directly available
            self.refpulse_duration_ms = 0
            self.git_commit_hash = self.__get_git_commit_hash()
            self.refpulse_bandwidth_Hz = 0
            self.refpulse_flipangle_deg = 0
            self.refpulse_is_calculated = False
            self.refpulse_sharpness = 0
            self.refpulse_bwfac_Hzms = 0
            self.refpulse_sint = 0
            self.refpulse_pint = 0
            self.refpulse_type = ""
            self.refpulse_rpfac_percent = 0
            self.refpulse_peakpow_W = 0
            self.refpulse_shape = ""
            self.refpulse_refpow_W = 0

            if load_data:
                self.fid = self.postprocess_fid_file()
                self.dims = [f"dim_{i}" for i in range(len(self.fid.shape))]
                self.set_kspace_xarray(self.fid, dims=self.dims, coords=None)
                self.set_reco_data(self.kspace_xarray, axes_names=self.dims[:2])

        else:
            raise FileNotFoundError(
                f"Path {input_path} is not a valid Siemens experiment path (no .dat file found)."
            )

    def __repr__(self):
        return f"< {__package__}.{self.__class__.__name__} name = {self.path.name} >"

    def __str__(self):
        return f"{self.path.name}"

    def __gt__(self, other):
        return self.timestamp > other.timestamp

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __find_dat_file(self, path_str) -> str | None:
        """
        Find a .dat file from a given path.

        Parameters
        ----------
        path_str: str
            A string that is either a path to a directory or a .dat file.

        Returns
        -------
        path: str
            The path to the .dat file as a string, or None if not found.

        """
        path = Path(path_str)
        if path.is_file() and path.suffix == ".dat":
            return str(path)
        if path.is_dir():
            for f in path.iterdir():
                if f.is_file() and f.suffix == ".dat":
                    return str(f)  # Return the first one found
        return None

    def load_fid_file(self):
        """Read fid file and return it as np.array."""
        return self.fid

    def open(self, paramfile="method"):
        """Open method file in default text editor for '.json' files."""
        import pprint
        import subprocess
        import sys
        import tempfile
        import time

        tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False)

        def open_file_platfrom_independent(filename):
            if sys.platform == "win32":
                os.startfile(filename)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, filename])

        try:
            # print(f"Dumping temporary file here:", tmp.name)
            pretty_str = pprint.pformat(getattr(self, paramfile))
            tmp.write(pretty_str.encode("ascii"))
        finally:
            tmp.close()
            open_file_platfrom_independent(tmp.name)
            time.sleep(2)
            os.unlink(tmp.name)

    def __get_git_commit_hash(
        self,
    ):
        """
        Get the current git commit hash of the repository containing this script.

        Returns
        -------
        str
            The git commit hash, or an error message if not found.
        """
        try:
            # Get the directory where this script (siemensexp.py) is located.
            script_path = os.path.dirname(os.path.abspath(__file__))

            # Start the search from the script's directory.
            repo = git.Repo(script_path, search_parent_directories=True)

            return repo.head.object.hexsha

        except (git.InvalidGitRepositoryError, git.exc.NoSuchPathError):
            # This will be triggered if the search fails.
            return "Could not find a git repository."
        except NameError:
            # __file__ is not defined in an interactive context (e.g., a notebook cell)
            # Fallback to the original method for interactive use.
            try:
                repo = git.Repo(search_parent_directories=True)
                return repo.head.object.hexsha
            except git.InvalidGitRepositoryError:
                return "Not a git repository (interactive mode)."

    def set_kspace_xarray(self, data, dims, coords=None, write_to_object=True):
        """
        Initialize or update the kspace_xarray with given data, dimensions, and coordinates.

        This method creates a new xarray DataArray if one does not exist, or updates
        the existing one if it does. It enables flexible handling of k-space data
        with various shapes and dimensions.

        Parameters
        ----------
        data : numpy.ndarray
            The k-space data to be stored in the xarray.
        dims : list of str
            List of dimension names.
        coords : dict, optional
            Dictionary of coordinates for each dimension. If None, default coordinates
            are created with np.arange() for each dimension based on the shape of the data.
        write_to_object : bool, optional
            If True (default), the generated k-space xarray is written to self.kspace_xarray,
            otherwise the generated k-space xarray is returned.

        Raises
        ------
        ValueError
            If the number of dimensions does not match the data shape.

        Returns
        -------
        xr.DataArray
            An xarray DataArray containing the k-space data, dimensions, and coordinates.

        Notes
        -----
        - If coords is not specified, it is automatically generated with np.arange() based on the shape of the data.
        - This method can be used for both the initial creation and subsequent updates of the xarray.

        """
        # Validate dimensions
        if len(data.shape) != len(dims):
            raise ValueError(
                f"Number of dimensions({len(dims)}) must match the data shape({len(data.shape)})"
            )

        # Create default coordinates if not specified
        if coords is None:
            coords = {dim: np.arange(size) for dim, size in zip(dims, data.shape)}

        # Create or update xarray DataArray
        try:
            kspace_xarray = xr.DataArray(data, dims=dims, coords=coords)
        except BaseException:
            raise ValueError(
                f"""\nCould not create xarray DataArray, shape of: \nData: {
                    data.shape},
                \nDimensions: {dims}, \nCoordinates: {coords}\n
                """
            )

        if write_to_object:
            self.kspace_xarray = kspace_xarray
        else:
            return kspace_xarray

    def ft(self, xarray, axes_names=["read_idx", "phase_idx"], ift=False):
        """
        Quickly computes the fftshift(fft(fftshift(xarray.data))) along the input axis.

        Parameters
        ----------
        xarray : (complex) xarray
            Data that should be Fourier-transformed.

        axes_names : list of strings
            Axes names along the Fourier transform should be performed.

        ift : boolean
            Use inverse Fourier transform and inverese fftshifts.

        Returns
        -------
        xarray dataarray with Fourier-transformed .data
        """
        axes_names = np.atleast_1d(axes_names)
        axes_idx = [xarray.get_axis_num(axis_name) for axis_name in axes_names]

        ft_array = xarray.to_numpy().copy()

        try:
            if ift:
                for d in range(len(axes_idx)):
                    ft_array = np.fft.ifftshift(
                        np.fft.ifft(
                            np.fft.ifftshift(
                                ft_array,
                                axes=(axes_idx[d],),
                            ),
                            axis=axes_idx[d],
                        ),
                        axes=(axes_idx[d],),
                    )
                    logger.debug("ift along " + str(axes_idx[d]))
            else:
                for d in range(len(axes_idx)):
                    ft_array = np.fft.fftshift(
                        np.fft.fft(
                            np.fft.fftshift(
                                ft_array,
                                axes=(axes_idx[d],),
                            ),
                            axis=axes_idx[d],
                        ),
                        axes=(axes_idx[d],),
                    )
                    logger.debug("ft along " + str(axes_idx[d]))
        except BaseException:
            Warning("Could not perform ffts!")

        ft_xarray = xarray.copy()
        ft_xarray.data = ft_array

        return ft_xarray

    def set_reco_data(self, kspace_xarray, axes_names, ift=False, write_to_object=True):
        """
        Calculate the reco_data with given kspace_xarray.

        Parameters
        ----------
        kspace_xarray : xarray
            The k-space data to be Fourier-transformed.
        ft_axis_name_list : list of str
            List of dimension names to be Fourier-transformed.
        ift: bool
            whether to calculate the inverse Fourier transform.
        write_to_object: bool
            whether to write the reco_data to self

        Returns
        -------
        xarray dataarray containing the spatial domain signal
        """
        # generate a copy of the kspace xarray:
        reco_data = kspace_xarray.copy()

        # Keep track of the signal domain:
        if "acquisition_idx" in axes_names:
            if kspace_xarray.attrs["signal_domain"] == "temporal":
                reco_data.attrs["signal_domain"] = "spectral"
            elif kspace_xarray.attrs["signal_domain"] == "spectral":
                reco_data.attrs["signal_domain"] = "temporal"
            else:
                reco_data.attrs["signal_domain"] = None

        # perform the Fourier transform along the specified axes
        reco_data.data = self.ft(xarray=kspace_xarray, axes_names=axes_names, ift=ift)
        if (
            "read_idx" in axes_names
            or "phase_idx" in axes_names
            or "slice_idx" in axes_names
        ):
            reco_data.attrs["domain"] = "spatial"
        if hasattr(self, "freq_axis_ppm"):
            assign_frequency_axis(reco_data.attrs, self.freq_axis_ppm, "ppm")
        if hasattr(self, "freq_axis_hz"):
            assign_frequency_axis(reco_data.attrs, self.freq_axis_hz, "Hz")

        # write the xarray to the object or return it:
        if write_to_object:
            self.reco_data = reco_data
        else:
            return reco_data

    def set_seq2d_rps_xarray(self, data, dims, coords=None):
        """
        Initialize or update the seq2d_rps_xarray with given data, dimensions, and coordinates.

        This method creates a new xarray DataArray if one doesn't exist, or updates
        the existing one if it does. It allows for flexible handling of seq2d data
        with different shapes and dimensions.

        Parameters
        ----------
        data : numpy.ndarray
            The k-space data to be stored in the xarray.
        dims : list of str
            List of dimension names.
        coords : dict, optional
            Dictionary of coordinates for each dimension. If None, default
            coordinates will be created using np.arange() for each dimension
            based on the shape of the data.

        Raises
        ------
        ValueError
            If the number of dimensions doesn't match the data shape.

        Notes
        -----
        - If coords is not provided, it will be automatically generated using
          np.arange() based on the shape of the data.
        - This method can be used both for initial creation and subsequent updates of the xarray.
        """

        # Validate dimensions
        if len(data.shape) != len(dims):
            raise ValueError(
                f"Number of dimensions({len(dims)}) must match the length of the data shape(len={len( data.shape)}, shape={data.shape})"
            )

        if DATA_PRECISION == np.float32:
            if np.any(np.iscomplex(data)):
                data = data.astype(np.complex64)
            else:
                data = data.astype(np.float32)
        elif DATA_PRECISION == np.float64:
            if np.any(np.iscomplex(data)):
                data = data.astype(np.complex128)
            else:
                data = data.astype(np.float64)
        else:
            raise ValueError(
                f"Unsupported data precision: {DATA_PRECISION}. Supported types are np.float32 and np.float64."
            )

        # Create default coordinates if not provided
        if coords is None:
            coords = {dim: np.arange(size) for dim, size in zip(dims, data.shape)}

        # Create or update the xarray DataArray
        try:
            self.seq2d_rps_xarray = xr.DataArray(data, dims=dims, coords=coords)
        except BaseException:
            raise ValueError(
                f"""\nKonnte kein xarray DataArray erstellen, Form der: \nDaten: {
                    data.shape},
                \nDimensionen: {dims}, \nKoordinaten: {coords}\n
                """
            )

    def rearrange_reco_data_xarray(self, input_data=None, force_orient=False):
        """Rearrange the reco_data xarray according to the patient position and slice orientation to match the seq2d."""
        if input_data is None:
            reco_data = self.reco_data.copy()
        else:
            reco_data = input_data

        # make sure to onlt rearange data once unless it's explicitly
        # requested:
        num_rearrange = reco_data.attrs.get("num_rearrange", 0)
        if num_rearrange == 0:
            reco_data.attrs["num_rearrange"] = 0
        elif num_rearrange > 0 and force_orient is False:
            return
        elif num_rearrange > 0 and force_orient is True:
            pass

        return reco_data

    # def set_seq2d_xyz_xarray(self, seq2d_rps_xarray, return_seq2d_xyz=False):
    #     seq2d_rps_xarray.attrs["patient_pos"]
    #     seq2d_rps_xarray.attrs["slice_orientation"]

    #     np.round(seq2d_rps_xarray.attrs["read_direction"])
    #     np.round(seq2d_rps_xarray.attrs["phase_direction"])
    #     np.round(seq2d_rps_xarray.attrs["slice_direction"])

    #     seq2d_xyz_xarray = seq2d_rps_xarray.copy()

    #     if return_seq2d_xyz:
    #         return seq2d_xyz_xarray
    #     else:
    #         self.seq2d_xyz_xarray = seq2d_xyz_xarray

    def postprocess_fid_file(self, input_data=None):
        """
        Applies standard Siemens post-processing to the raw data.
        This includes oversampling removal and ramp sampling regridding.
        """
        self.twix.image.flagRemoveOS = True
        # Deactivating ramp sampling regridding as it can cause an exception
        # if trajectory data is not available.
        # self.twix.image.flagRampSampRegrid = True
        self.twix.image.is_sorted = False
        return self.twix.image[:]
