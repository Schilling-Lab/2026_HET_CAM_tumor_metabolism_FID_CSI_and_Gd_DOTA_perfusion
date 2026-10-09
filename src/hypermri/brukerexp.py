"""
Base class used to read in one Bruker experiment/<expno> directory.

This module provides functionality for reading and processing Bruker MRI data.
It handles the specific directory structure and file formats used by Bruker
ParaVision systems.

Notes
-----
Have a look at the ParaVision Manual for more information on the scan-data folder
structure.

Includes parts of functions adapted from brukerMRI package
(https://github.com/jdoepfert/brukerMRI).
"""

import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional

import git
import numpy as np
import pandas as pd
import xarray as xr
from brukerapi.dataset import Dataset

from hypermri.mixins import StandardizedMRIMixin
from hypermri.utils.physics import get_gmr
from hypermri.utils.system.logging import LOG_MODES, debug_log, init_default_logger
from hypermri.utils.time.temporal import calc_per_image_timeaxis_s
from hypermri.vendorhandler import VendorHandler

# Set a global variable to control info message visibility
DB_MSG = True  # Change this to False to hide detailed error messages
DATA_PRECISION = np.float16


DEBUG_FILE = "/tmp/bruker_exp_debug.log"
DEBUG_LOG = True


def debug_log_brukerexp(message="", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG):
    """Debug logging system for brukerexp"""
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
    )


logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


class BrukerHandler(VendorHandler, StandardizedMRIMixin):
    """
    Base class for experiments conducted on Bruker preclinical systems.

    ParaVision dataset paths have the following form:

    ``<DataPath>/<name>/<expno>/pdata/<procno>``

    Parameters
    ----------
    DataPath : str
        Arbitrary path in the directory tree that is used as root for ParaVision datasets.
    name : str
        Directory containing all information about a study. The name is created by
        ParaVision. Maximum length of the directory name is 64 characters.
    expno : str
        Directory containing all information to describe one experiment.
        The directory name is the experiment number (usually an integer).
        Data from every new mri-sequence executed during the MR scan is stored
        in a new expno directory.
    procno : str
        Directory containing all information about a reconstructed or derived
        image series of the experiment. Several procnos may exist which contain
        different derived image series (e.g., different reconstructions, parameter
        images, etc.). The directory name is the processed images number.

    Notes
    -----
    This class reads in a single <expno> directory.

    Attributes
    ----------
    timestamp : datetime.datetime
        The time point at which the scan:
        - finished running
        - was aborted
        - was last edited without being executed
        - was created

        Note: You can compare two MRIExperiment objects using '<' and '>'
        to find out which one has the latest timepoint.

    Methods
    -------
    Load_fid-file()
        Load the FID file.
    Load_rawdatajob0_file()
        Load the rawdata.job0 file.
    Load_2dseq_file()
        Load the 2dseq reconstructed image file.
    """

    @property
    def id(self) -> str:
        return self.ExpNum

    @property
    def freq_axis_ppm(self) -> Optional[np.ndarray]:
        # Return cached value if available, otherwise compute it
        if hasattr(self, "_freq_axis_ppm") and self._freq_axis_ppm is not None:
            return self._freq_axis_ppm

        from hypermri.analysis.spectroscopy.core import get_freq_axis

        self._freq_axis_ppm = get_freq_axis(data_obj=self, unit="ppm")
        return self._freq_axis_ppm

    @property
    def freq_axis_hz(self) -> Optional[np.ndarray]:
        # Return cached value if available, otherwise compute it
        if hasattr(self, "_freq_axis_hz") and self._freq_axis_hz is not None:
            return self._freq_axis_hz

        from hypermri.analysis.spectroscopy.core import get_freq_axis

        self._freq_axis_hz = get_freq_axis(data_obj=self, unit="Hz")
        return self._freq_axis_hz

    @property
    def dwell_times_s(self) -> float:
        # Return cached value if available, otherwise compute it
        if hasattr(self, "dwell_time_s") and self.dwell_time_s is not None:
            return self.dwell_time_s

        from hypermri.utils.time.temporal import get_dwell_time_s

        return get_dwell_time_s(data_obj=self)

    @property
    def type(self) -> str:
        return self.sequence_name

    @property
    def has_dicom_reco(self) -> bool:
        """Check if any reconstruction has DICOM files available."""
        if not hasattr(self, "recon_names") or not self.recon_names:
            return False

        for recon in self.recon_names:
            dicom_dir = self.path / "pdata" / str(recon) / "dicom"
            # Check if directory exists and is not empty
            if dicom_dir.exists() and dicom_dir.is_dir():
                try:
                    # Check for at least one file, ignoring hidden files
                    if any(
                        f.is_file() and not f.name.startswith(".")
                        for f in dicom_dir.iterdir()
                    ):
                        return True
                except (OSError, PermissionError):
                    continue
        return False

    def __init__(self, source, load_data: bool = False, log_mode: str = "error"):
        """
        Initialize brukerexp class.

        Parameters
        ----------
        source : str or MRIExperiment-object
            The complete path to measurement/experiment folder
        load_data: bool
            Whether to load the data
        log_mode: str
            Which log mode to use.

        """
        # Initialize mixins first
        super().__init__()

        try:
            logger.setLevel(LOG_MODES[log_mode])
            exp_folder_path = (
                source if not isinstance(source, BrukerHandler) else str(source.path)
            )
            if not self.is_valid_scan_dir(exp_folder_path):
                raise FileNotFoundError(
                    f"Folder {exp_folder_path} does not exist or is not a bruker experiment."
                )

            # Use standardized path handling
            self.path = exp_folder_path

            self.load_parameters(source)
            if load_data:
                self.load_data(source)

        except Exception as e:
            logger.error(f"Error initializing BrukerHandler for {source}: {e}")
            # Optionally, re-raise the exception if you want the program to stop
            # raise e

        # Initialize cached frequency axes as None - will be computed on first access
        self._freq_axis_ppm = None
        self._freq_axis_hz = None

    def _parse_parameters(self, source) -> Dict[str, Any]:
        if hasattr(source, "path"):
            path = source.path
        else:
            path = source

        self.path = path

        # ---------- Scan parameter files: ----------
        self.method = self.__ReadParamFile("method")
        self.acqp = self.__ReadParamFile("acqp")
        self.visu = self.__ReadParamFile("visu_pars")
        self.reco = self.__ReadParamFile(os.path.join("pdata", "1", "reco"))

        experiment_number = self.__getVisuExperimentNumber()
        self.ExpNum = experiment_number
        self.experiment_number = experiment_number
        self.vendor = "bruker"

        sequence_name = self.__GetMeasurementType()
        self.sequence_name = sequence_name

        nucleus = self.method.get("PVM_Nucleus1", None)
        if nucleus is not None and len(nucleus) >= 2:
            nucleus = nucleus[1:-1]
        self.nucleus = nucleus

        if sequence_name is not None and len(sequence_name) >= 7:
            self.method_type = "stock" if sequence_name[1:7] == "Bruker" else "custom"
        else:
            self.method_type = None

        self.is_spectroscopic = self.method.get("PVM_EncSpectroscopy", None) != "No"
        self.n_slice_packs = self.method.get("PVM_NSPacks", 1)
        if self.is_spectroscopic:
            self.cut_off = 70 if self.method_type == "stock" else 0
        else:
            self.cut_off = 0

        self.timestamp = self.method.get("MetaTimeStamp")

        def _strip_braces(value):
            if value is None:
                return None
            if (
                isinstance(value, str)
                and len(value) >= 2
                and value[0] == value[-1] == '"'
            ):
                return value[1:-1]
            return value

        self.subject_name = _strip_braces(self.visu.get("VisuSubjectName", None))
        self.subject_id = _strip_braces(self.visu.get("VisuSubjectId", None))
        self.subject_position = self.visu.get("VisuSubjectPosition", None)
        self.subject_type = self.visu.get("VisuSubjectType", None)
        if self.subject_type == "Quadruped":
            self.subject_type = "Rodent"

        spatial_dimension = self.method.get("PVM_SpatDimEnum", None)
        if spatial_dimension is not None:
            spatial_dimension = spatial_dimension[1:-1]
        self.spatial_dimension = spatial_dimension
        self.spat_dim = spatial_dimension

        self.recon_names = self.__FindRecons()
        self.n_receivers = self.method.get("PVM_EncNReceivers", 1)

        try:
            self.n_slices = self.method.get("PVM_SPackArrNSlices", 1)[0]
        except BaseException:
            self.n_slices = self.method.get("PVM_SPackArrNSlices", None)

        if self.n_slices is None:
            try:
                self.n_slices = self.method.get("SliceNum", 1)
            except BaseException:
                self.n_slices = 1

        self.slice_thickness_mm = self.method.get("PVM_SliceThick", 0)
        try:
            self.slice_gap_mm = self.method.get("PVM_SPackArrSliceGap", 0)[0]
        except BaseException:
            self.slice_gap_mm = self.method.get("PVM_SPackArrSliceGap", 0)

        acquisition_matrix_raw = self.acqp.get("ACQ_grad_matrix", None)
        if acquisition_matrix_raw is not None:
            try:
                self.acquisition_matrix = acquisition_matrix_raw[0]
            except Exception:
                self.acquisition_matrix = acquisition_matrix_raw
        self.acquisition_matrix = self.acquisition_matrix

        self.patient_position = self.acqp.get("ACQ_patient_pos", None)
        self.patient_pos = self.patient_position

        self.read_orientation = self.method.get("PVM_SPackArrReadOrient", None)
        self.slice_orientation = self.method.get("PVM_SPackArrSliceOrient", None)
        self.slice_order = np.atleast_1d(self.method.get("PVM_ObjOrderList", 0))
        try:
            self.reference_frequency_mhz = self.method.get("PVM_FrqRef", None)[0]
        except:
            self.reference_frequency_mhz = self.method.get("PVM_FrqRef", None)
        try:
            self.b0_tesla = self.reference_frequency_mhz / get_gmr(nucleus=self.nucleus)
        except:
            self.b0_tesla = None

        self.fov_mm = self.method.get("PVM_Fov", None)
        if self.is_spectroscopic and self.fov_mm is None:
            vox_arr_size = self.method.get("PVM_VoxArrSize", None)
            if vox_arr_size is not None:
                try:
                    self.fov_mm = np.array([float(f) for f in vox_arr_size[0]])
                except Exception:
                    self.fov_mm = vox_arr_size

        self.matrix = self.method.get("PVM_Matrix", None)
        if self.is_spectroscopic:
            self.matrix = np.array([1, 1, 1])

        self.encoding_matrix = self.method.get("PVM_EncMatrix", None)
        self.enc_matrix = self.encoding_matrix

        if not self.is_spectroscopic and self.matrix is not None:
            if spatial_dimension != "3D" and self.fov_mm is not None:
                slice_fov_mm = (
                    self.slice_thickness_mm * self.n_slices
                    + self.slice_gap_mm * (self.n_slices - 1)
                )
                self.fov_mm = np.append(self.fov_mm, slice_fov_mm)
                self.matrix = np.append(self.matrix, self.n_slices)

        self.center_ppm = self.method.get("PVM_FrqWorkPpm", [0.0, 0.0])[0]

        if spatial_dimension == "2D" and self.encoding_matrix is not None:
            self.encoding_matrix = np.append(self.encoding_matrix, self.n_slices)
            self.enc_matrix = self.encoding_matrix

        self.measurement_time = self._get_measurement_time()

        self.n_repetitions = self.method.get("PVM_NRepetitions", 1)
        self.n_averages = self.method.get("PVM_NAverages", 1)
        if sequence_name == "<Bruker:EPSI>":
            self.n_echoes = self.method.get("SpecSize", 1)
        else:
            self.n_echoes = self.method.get("PVM_NEchoImages", 1)
        self.n_acquisition = self.method.get("PVM_SpecMatrix", 1)

        self.inversion_time_ms = self.method.get("PVM_FairTIR_Arr", None)
        if self.inversion_time_ms is not None:
            self.n_inversion_time = len(self.inversion_time_ms)
        else:
            self.n_inversion_time = 1
            self.inversion_time_ms = 0

        if self.fov_mm is None or self.matrix is None:
            self.resolution_mm = None
        else:
            try:
                self.resolution_mm = self.fov_mm / self.matrix
            except Exception:
                self.resolution_mm = None

        try:
            self.read_offset_mm = self.method.get("PVM_SPackArrReadOffset", 0)[0]
            self.phase_offset_mm = self.method.get("PVM_SPackArrPhase1Offset", 0)[0]
            self.phase_0_offset_mm = self.method.get("PVM_SPackArrPhase0Offset", 0)[0]
            self.phase_1_offset_mm = self.method.get("PVM_SPackArrPhase1Offset", 0)[0]
            self.slice_offset_mm = self.method.get("PVM_SPackArrSliceOffset", 0)[0]
        except Exception:
            self.read_offset_mm = self.method.get("PVM_SPackArrReadOffset", 0)
            self.phase_offset_mm = self.method.get("PVM_SPackArrPhase1Offset", 0)
            self.phase_0_offset_mm = self.method.get("PVM_SPackArrPhase0Offset", 0)
            self.phase_1_offset_mm = self.method.get("PVM_SPackArrPhase1Offset", 0)
            self.slice_offset_mm = self.method.get("PVM_SPackArrSliceOffset", 0)

        self.phase_0offset_mm = self.phase_0_offset_mm
        self.phase_1offset_mm = self.phase_1_offset_mm

        self.slice_offset_mm_array = self.method.get("PVM_SliceOffset", 0)

        self.coil_receive = self.visu.get("VisuCoilReceiveName", None)
        self.coil_transmit = self.visu.get("VisuCoilTransmitName", None)

        transpose_read_phase = self.reco.get("RECO_transposition", None)
        if transpose_read_phase is None:
            self.transpose_read_phase = False
        elif np.sum(transpose_read_phase) > 0:
            self.transpose_read_phase = True
        else:
            self.transpose_read_phase = False

        self.visucore_extent = self.visu.get("VisuCoreExtent", None)
        self.visucore_thickness = self.visu.get("VisuCoreFrameThickness", None)
        self.visucore_size = self.visu.get("VisuCoreSize", None)
        self.visucore_position = self.visu.get("VisuCorePosition", None)
        self.visucore_orientation = self.visu.get("VisuCoreOrientation", None)
        self.spat_resol = self.method.get("PVM_SpatResol", None)

        position_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if self.subject_type == "Material":
            # Mentioned in Bruker manual
            position_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        else:
            match self.patient_position:
                # Mentioned in Bruker manual
                case "Feet_Supine":
                    pass
                # Taken from ParaVision 7
                case "Head_Supine":
                    position_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                # Taken from ParaVision 7
                case "Feet_Prone":
                    position_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                # Taken from ParaVision 7
                case "Head_Prone":
                    position_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                case _:
                    pass

        self.position_matrix = position_matrix

        self.fat_water_separation = self.method.get("PVM_FwsOnOff", "Off")
        self.b_values = self.method.get("PVM_DwEffBval", 0.0)
        self.n_voxels = self.method.get("PVM_NVoxels", 1)
        self.rare_factor = self.method.get("PVM_RareFactor", 1)

        self.echo_spacing_ms = self.method.get("EchoSpacing", 0.0)
        self.effective_te_ms = self.method.get("EffectiveTE", 0.0)

        self.scan_duration_s = self.method.get("PVM_ScanTime", 0.0) * 1e-3
        self.bandwidth_hz = self.method.get("PVM_EffSwh", None)
        if self.bandwidth_hz is None:
            self.bandwidth_hz = self.method.get("PVM_SpecSWH", None)
        self.bandwidth_ppm = self.method.get("PVM_SpecSW", None)
        self.echo_time_ms = self.method.get("PVM_EchoTime", None)
        self.repetition_time_ms = self.method.get("PVM_RepetitionTime", None)

        self.frame_time_s = calc_per_image_timeaxis_s(data_obj=self, start_with_0=False)
        self.dwell_time_s = self.method.get("PVM_DigDw", None)
        if self.dwell_time_s is not None:
            self.dwell_time_s = self.dwell_time_s / 1000.0
        try:
            from hypermri.utils.mathematics.calculations import calc_sampling_time_axis

            self.sampling_time_axis_s = calc_sampling_time_axis(
                dwell_time_s=self.dwell_time_s, npoints=self.n_acquisition
            )
        except Exception:
            self.sampling_time_axis_s = None

        # Excitation pulse parameters
        self.rf_pulse_params = self.method.get("ExcPulse1", None)
        if self.rf_pulse_params is not None:
            self.rfpulse_duration_ms = self.rf_pulse_params[0]
            self.rfpulse_bandwidth_hz = self.rf_pulse_params[1]
            self.flipangle_deg = self.rf_pulse_params[2]
            rfpulse_calc = self.rf_pulse_params[3]
            self.rfpulse_is_calculated = rfpulse_calc != "No"
            self.rfpulse_sharpness = self.rf_pulse_params[4]
            self.rfpulse_bwfac_hzms = self.rf_pulse_params[5]
            self.rfpulse_sint = self.rf_pulse_params[6]
            self.rfpulse_pint = self.rf_pulse_params[7]
            self.rfpulse_type = self.rf_pulse_params[8]
            self.rfpulse_rpfac_percent = self.rf_pulse_params[9]
            self.rfpulse_peakpow_w = self.rf_pulse_params[10]
            self.rfpulse_shape = (
                self.rf_pulse_params[11].replace(".exc>", "").replace("<", "")
            )
            self.rfpulse_refpow_w = self.method.get("PVM_RefPowCh1", 0.0)

        self.rfpulse_bandwidth_Hz = self.rfpulse_bandwidth_hz
        self.rfpulse_bwfac_Hzms = self.rfpulse_bwfac_hzms
        self.rfpulse_peakpow_W = self.rfpulse_peakpow_w
        self.rfpulse_refpow_W = self.rfpulse_refpow_w

        # Refocusing pulse parameters
        self.ref_pulse_params = self.method.get("RefPulse1", None)
        if self.ref_pulse_params is not None:
            self.refpulse_duration_ms = self.ref_pulse_params[0]
            self.refpulse_bandwidth_hz = self.ref_pulse_params[1]
            self.refpulse_flipangle_deg = self.ref_pulse_params[2]
            refpulse_calc = self.ref_pulse_params[3]
            self.refpulse_is_calculated = refpulse_calc != "No"
            self.refpulse_sharpness = self.ref_pulse_params[4]
            self.refpulse_bwfac_hzms = self.ref_pulse_params[5]
            self.refpulse_sint = self.ref_pulse_params[6]
            self.refpulse_pint = self.ref_pulse_params[7]
            self.refpulse_type = self.ref_pulse_params[8]
            self.refpulse_rpfac_percent = self.ref_pulse_params[9]
            self.refpulse_peakpow_w = self.ref_pulse_params[10]
            self.refpulse_shape = (
                self.ref_pulse_params[11].replace(".exc>", "").replace("<", "")
            )
            self.refpulse_refpow_w = self.method.get("PVM_RefPowCh1", 0.0)

        self.refpulse_bandwidth_Hz = self.refpulse_bandwidth_hz
        self.refpulse_bwfac_Hzms = self.refpulse_bwfac_hzms
        self.refpulse_peakpow_W = self.refpulse_peakpow_w
        self.refpulse_refpow_W = self.refpulse_refpow_w

        self.git_commit_hash = self.__get_git_commit_hash()

        params = {key: None for key in self.parameter_keys()}
        params.update(
            {
                "experiment_number": experiment_number,
                "timestamp": self.timestamp,
                "vendor": self.vendor,
                "nucleus": self.nucleus,
                "method_type": self.method_type,
                "sequence_name": self.sequence_name,
                "is_spectroscopic": self.is_spectroscopic,
                "cut_off": self.cut_off,
                "git_commit_hash": self.git_commit_hash,
                "subject_name": self.subject_name,
                "subject_id": self.subject_id,
                "subject_position": self.subject_position,
                "subject_type": self.subject_type,
                "patient_position": self.patient_position,
                "spatial_dimension": self.spatial_dimension,
                "n_slice_packs": self.n_slice_packs,
                "n_slices": self.n_slices,
                "slice_thickness_mm": self.slice_thickness_mm,
                "slice_gap_mm": self.slice_gap_mm,
                "slice_orientation": self.slice_orientation,
                "slice_order": self.slice_order,
                "slice_offset_mm": self.slice_offset_mm,
                "slice_offset_mm_array": self.slice_offset_mm_array,
                "read_orientation": self.read_orientation,
                "read_offset_mm": self.read_offset_mm,
                "phase_offset_mm": self.phase_offset_mm,
                "phase_0_offset_mm": self.phase_0_offset_mm,
                "phase_1_offset_mm": self.phase_1_offset_mm,
                "position_matrix": self.position_matrix,
                "fov_mm": self.fov_mm,
                "matrix": self.matrix,
                "encoding_matrix": self.encoding_matrix,
                "resolution_mm": self.resolution_mm,
                "center_ppm": self.center_ppm,
                "measurement_time": self.measurement_time,
                "n_receivers": self.n_receivers,
                "n_repetitions": self.n_repetitions,
                "n_averages": self.n_averages,
                "n_echoes": self.n_echoes,
                "n_acquisition": self.n_acquisition,
                "n_voxels": self.n_voxels,
                "n_inversion_time": self.n_inversion_time,
                "inversion_time_ms": self.inversion_time_ms,
                "rare_factor": self.rare_factor,
                "scan_duration_s": self.scan_duration_s,
                "bandwidth_hz": self.bandwidth_hz,
                "bandwidth_ppm": self.bandwidth_ppm,
                "echo_time_ms": self.echo_time_ms,
                "repetition_time_ms": self.repetition_time_ms,
                "frame_time_s": self.frame_time_s,
                "dwell_time_s": self.dwell_time_s,
                "sampling_time_axis_s": self.sampling_time_axis_s,
                "echo_spacing_ms": self.echo_spacing_ms,
                "effective_te_ms": self.effective_te_ms,
                "fat_water_separation": self.fat_water_separation,
                "b_values": self.b_values,
                "coil_receive": self.coil_receive,
                "coil_transmit": self.coil_transmit,
                "rf_pulse_params": self.rf_pulse_params,
                "rfpulse_duration_ms": self.rfpulse_duration_ms,
                "rfpulse_bandwidth_hz": self.rfpulse_bandwidth_hz,
                "rfpulse_is_calculated": self.rfpulse_is_calculated,
                "rfpulse_sharpness": self.rfpulse_sharpness,
                "rfpulse_bwfac_hzms": self.rfpulse_bwfac_hzms,
                "rfpulse_sint": self.rfpulse_sint,
                "rfpulse_pint": self.rfpulse_pint,
                "rfpulse_type": self.rfpulse_type,
                "rfpulse_rpfac_percent": self.rfpulse_rpfac_percent,
                "rfpulse_peakpow_w": self.rfpulse_peakpow_w,
                "rfpulse_shape": self.rfpulse_shape,
                "rfpulse_refpow_w": self.rfpulse_refpow_w,
                "flipangle_deg": self.flipangle_deg,
                "ref_pulse_params": self.ref_pulse_params,
                "refpulse_duration_ms": self.refpulse_duration_ms,
                "refpulse_bandwidth_hz": self.refpulse_bandwidth_hz,
                "refpulse_flipangle_deg": self.refpulse_flipangle_deg,
                "refpulse_is_calculated": self.refpulse_is_calculated,
                "refpulse_sharpness": self.refpulse_sharpness,
                "refpulse_bwfac_hzms": self.refpulse_bwfac_hzms,
                "refpulse_sint": self.refpulse_sint,
                "refpulse_pint": self.refpulse_pint,
                "refpulse_type": self.refpulse_type,
                "refpulse_rpfac_percent": self.refpulse_rpfac_percent,
                "refpulse_peakpow_w": self.refpulse_peakpow_w,
                "refpulse_shape": self.refpulse_shape,
                "refpulse_refpow_w": self.refpulse_refpow_w,
                "recon_names": self.recon_names,
                "acquisition_matrix": self.acquisition_matrix,
            }
        )

        return params

    def load_data(self, source):
        """
        Loads the data from the source path into standardized data arrays.
        """
        total_start_time = time.time()
        logger.info(f"[TIMING] Starting data loading for scan {self.ExpNum}")

        # load raw data into the instance
        start_time = time.time()
        self.fid = self.load_fid_file()
        fid_time = time.time() - start_time
        logger.info(f"[TIMING] FID loading took {fid_time:.3f}s")
        logger.debug(
            f"BrukerHandler.load_data: after loading FID, self.fid type: {type( self.fid)}"
        )
        logger.debug(f"BrukerHandler.load_data: handler id: {id(self)}")
        logger.debug(
            f"BrukerHandler.load_data: hasattr(self, 'fid'): {hasattr( self, 'fid')}"
        )

        start_time = time.time()
        self.rawdatajob0 = self.load_rawdatajob0_file()
        rawdata_time = time.time() - start_time
        logger.info(f"[TIMING] rawdata.job0 loading took {rawdata_time:.3f}s")

        # load reconstructed data stored in '.../pdata/1' and store using
        # standardized naming
        start_time = time.time()
        seq2d_data = self.load_2dseq_file()

        if hasattr(self, "transpose_read_phase") and self.transpose_read_phase:
            seq2d_data = np.transpose(
                seq2d_data, axes=(1, 0, *range(2, seq2d_data.ndim))
            )
        self.seq2d = seq2d_data  # Store raw seq2d data temporarily
        seq2d_time = time.time() - start_time
        logger.info(f"[TIMING] 2dseq loading took {seq2d_time:.3f}s")

        # Check if we have valid seq2d data
        start_time = time.time()
        has_valid_seq2d = False
        if seq2d_data is not None:
            if hasattr(seq2d_data, "__len__"):
                try:
                    has_valid_seq2d = len(seq2d_data) > 0
                except BaseException:
                    # For objects where len() might not work as expected
                    has_valid_seq2d = True
            else:
                # Scalar data
                has_valid_seq2d = True

        if has_valid_seq2d:
            # Convert to xarray if it's numpy array
            if isinstance(seq2d_data, np.ndarray):
                seq2d_data = xr.DataArray(seq2d_data)
            self.set_primary_data(seq2d_data, "scanner_data")
            # Only set legacy if we have valid data
            self._legacy_seq2d = seq2d_data
        else:
            # No valid seq2d data available
            self._legacy_seq2d = None

        # kspace_array is no longer stored permanently - used only during
        # reconstruction

        # Load DICOM files if available
        self.dicom = self.load_dicom_files()

        processing_time = time.time() - start_time
        logger.info(f"[TIMING] Data processing took {processing_time:.3f}s")

        total_time = time.time() - total_start_time
        logger.info(
            f"[TIMING] Total data loading took {total_time: .3f}s for scan {self.ExpNum}"
        )

    def load_dicom_files(self, recon_num: int = 1) -> list:
        """
        Load DICOM files from pdata directory if available.

        Parameters
        ----------
        recon_num : int, optional
            Reconstruction number to load DICOMs from, by default 1.

        Returns
        -------
        list
            List of loaded pydicom datasets, sorted by instance number.
        """
        try:
            import pydicom
        except ImportError:
            logger.warning("pydicom not installed. Cannot load DICOM files.")
            return []

        dicom_dir = self.path / "pdata" / str(recon_num) / "dicom"

        if not dicom_dir.exists() or not dicom_dir.is_dir():
            return []

        dicom_files = []
        try:
            # Find all .dcm files
            files = sorted(list(dicom_dir.glob("*.dcm")))
            if not files:
                # Try finding files without extension or uppercase
                files = sorted(
                    [
                        f
                        for f in dicom_dir.iterdir()
                        if f.is_file() and not f.name.startswith(".")
                    ]
                )

            if not files:
                return []

            logger.info(f"[TIMING] Loading {len(files)} DICOM files from {dicom_dir}")
            start_time = time.time()

            for f in files:
                try:
                    ds = pydicom.dcmread(f)
                    dicom_files.append(ds)
                except Exception as e:
                    logger.debug(f"Failed to load DICOM file {f}: {e}")

            # Sort by Instance Number if available
            dicom_files.sort(key=lambda x: getattr(x, "InstanceNumber", 0))

            logger.info(f"[TIMING] DICOM loading took {time.time() - start_time:.3f}s")

        except Exception as e:
            logger.warning(f"Error loading DICOM files: {e}")
            return []

        return dicom_files

    def load_reconstruction(self, recon_num: int = 1, set_as_primary: bool = True):
        """
        Load a specific pdata reconstruction and optionally set it as the primary data.

        This method allows loading different reconstructions (e.g., pdata/1, pdata/2, etc.)
        after the scan object has been initialized. Useful for comparing different
        reconstruction parameters or loading alternative processed data.

        Parameters
        ----------
        recon_num : int, optional
            The pdata reconstruction number to load (e.g., 1 for pdata/1, 2 for pdata/2).
            Default is 1.
        set_as_primary : bool, optional
            If True, sets the loaded reconstruction as the primary data (accessible via
            scan.data_array). If False, only returns the data without modifying the scan's
            primary data. Default is True.

        Returns
        -------
        xr.DataArray or None
            The loaded reconstruction as an xarray DataArray, or None if the reconstruction
            file does not exist or is empty.

        Raises
        ------
        FileNotFoundError
            If the requested pdata reconstruction directory or 2dseq file does not exist.

        Examples
        --------
        >>> # Load default reconstruction (pdata/1)
        >>> scan = study[5]
        >>>
        >>> # Load alternative reconstruction (pdata/2) and set as primary
        >>> scan.load_reconstruction(recon_num=2)
        >>> scan.data_array  # Now contains pdata/2 data
        >>>
        >>> # Load reconstruction without changing primary data (for comparison)
        >>> pdata3 = scan.load_reconstruction(recon_num=3, set_as_primary=False)
        >>> # Compare: scan.data_array still has pdata/2, pdata3 has pdata/3

        Notes
        -----
        - The method will also update the `reco` parameter file to match the loaded reconstruction
        - If transpose_read_phase is set, the loaded data will be transposed accordingly
        - The loaded data is converted to xarray.DataArray format with proper coordinates
        - Complex data types are preserved
        - Updates data_array (scanner data), not reco_data (reconstructed data)
        """
        logger.info(f"Loading reconstruction pdata/{recon_num} for scan {self.ExpNum}")

        # Check if the pdata directory and 2dseq file exist before attempting to load
        pdata_dir = self.path / "pdata" / str(recon_num)
        seq2d_file = pdata_dir / "2dseq"

        if not pdata_dir.exists():
            error_msg = (
                f"WARNING: Reconstruction directory 'pdata/{recon_num}' does not exist "
                f"for scan {self.ExpNum} at path: {self.path}\n"
                f"Available pdata directories: {[p.name for p in (self.path / 'pdata').glob('*') if p.is_dir()]}"
            )
            logger.warning(error_msg)
            raise FileNotFoundError(error_msg)

        if not seq2d_file.exists():
            error_msg = (
                f"WARNING: Reconstruction file 'pdata/{recon_num}/2dseq' does not exist "
                f"for scan {self.ExpNum} at path: {seq2d_file}"
            )
            logger.warning(error_msg)
            raise FileNotFoundError(error_msg)

        if os.path.getsize(seq2d_file) == 0:
            error_msg = (
                f"WARNING: Reconstruction file 'pdata/{recon_num}/2dseq' is empty (0 bytes) "
                f"for scan {self.ExpNum} at path: {seq2d_file}"
            )
            logger.warning(error_msg)
            return None

        # Use the reconstruction strategy to properly load and format the data
        # This ensures coordinates and complex data are handled correctly
        try:
            from hypermri.recon_strategy import Brukerseq2dRecoStrategy

            strategy = Brukerseq2dRecoStrategy()

            # Reconstruct with the specified recon_num
            # This will use _reconstruct_internal which handles all coordinate assignment
            seq2d_data = strategy.reconstruct(
                self, recon_num=recon_num, set_reco_file=True, no_reconstruction=False
            )

            if seq2d_data is None:
                error_msg = (
                    f"WARNING: No valid data found in pdata/{recon_num} for scan {self.ExpNum}. "
                    f"The file exists but contains no data."
                )
                logger.warning(error_msg)
                return None

        except Exception as e:
            logger.error(f"Failed to load reconstruction using strategy: {e}")
            logger.info("Falling back to basic loading method")

            # Fallback to basic loading if strategy fails
            seq2d_data = self.load_2dseq_file(recon_num=recon_num, set_reco_file=True)

            # Apply transpose if needed
            if hasattr(self, "transpose_read_phase") and self.transpose_read_phase:
                seq2d_data = np.transpose(
                    seq2d_data, axes=(1, 0, *range(2, seq2d_data.ndim))
                )

            # Check if we have valid data
            has_valid_data = False
            if seq2d_data is not None:
                if hasattr(seq2d_data, "__len__"):
                    try:
                        has_valid_data = len(seq2d_data) > 0
                    except BaseException:
                        has_valid_data = True
                else:
                    has_valid_data = True

            if not has_valid_data:
                error_msg = (
                    f"WARNING: No valid data found in pdata/{recon_num} for scan {self.ExpNum}. "
                    f"The file exists but contains no data."
                )
                logger.warning(error_msg)
                return None

            # Convert to xarray if it's a numpy array (basic fallback)
            if isinstance(seq2d_data, np.ndarray):
                seq2d_data = xr.DataArray(seq2d_data)

        # Optionally set as primary data (data_array, not reco_data)
        if set_as_primary:
            # Use "scanner_data" type to set data_array
            self.set_primary_data(seq2d_data, "scanner_data")
            self._legacy_seq2d = seq2d_data
            self.seq2d = seq2d_data

            # Load associated DICOM files
            self.dicom = self.load_dicom_files(recon_num=recon_num)

            logger.info(
                f"Set pdata/{recon_num} as primary data_array for scan {self.ExpNum}"
            )

        return seq2d_data

    def __str__(self):
        return f"{self.acqp['ACQ_scan_name'][1:-1]}"  # drop encasing arrows <>

    def __gt__(self, other):
        """This makes it easy to sort multiple experiments by their timestamp."""
        return self.timestamp > other.timestamp

    def __lt__(self, other):
        """This makes it easy to sort multiple experiments by their timestamp."""
        return self.timestamp < other.timestamp

    def get(self, attribute_name: str, default=None):
        """
        Gets an attribute value, returning a default if the attribute
        does not exist. This mimics dictionary .get() behavior.

        Parameters
        ----------
        attribute_name: str
            The name of the attribute to retrieve (e.g., 'name').
        default: str
            The value to return if the attribute is not found.

        Returns
        -------
        The attribute's value or the default.

        """
        # getattr is a built-in Python function that gets an attribute from an
        # object, with an optional default value.
        return getattr(self, attribute_name, default)

    @staticmethod
    def is_valid_scan_dir(path) -> bool:
        """Check if directory provided is a valid bruker scan directory."""
        if not os.path.isdir(path):
            return False

        content = os.listdir(path)

        if len(set(content).intersection(["pdata", "fid", "acqp", "method"])) > 0:
            if "acqp" not in content:
                return False
            if "method" not in content:
                return False
            return True
        return False

    @staticmethod
    def __is_valid_recon_dir(path) -> bool:
        """Check if directory provided is a 'pdata' reconstruction directory."""
        if os.path.isdir(path):
            content = os.listdir(path)
            if len(set(content).intersection(["2dseq"])) > 0:
                return True
        return False

    def __getVisuExperimentNumber(self) -> int:
        """Try to get experiment number. Returns -1 when fails.

        First try to convert folder name into experiment number.

        In case that fails, falls back to extracting the number from the
        ACQ_scan_name (see self.__str__) using regular expressions.

        In case no experiment number is found inside the name, return -1.
        """
        # try to get ExpNum from scan folder name
        if self.path.name.isdigit():
            return int(self.path.name)

        # try to match last (E...) in scan name using regex
        match = re.match(r".*\(E(\d+)\)", str(self))

        return int(match.group(1)) if match else -1

    def _get_measurement_time(self) -> Optional[datetime]:
        try:
            date_string = self.acqp["ACQ_time"]

            # Clean it up
            cleaned = date_string.strip("<>").replace(",", ".")

            # Parse with strptime
            return datetime.strptime(cleaned, "%Y-%m-%dT%H:%M:%S.%f%z")
        except:
            return None

    @staticmethod
    def __ParseSingleValue(val: str):
        """Extract a single value from a string.

        Parameters
        ----------
        value: str
            A string containing a single value.

        Returns
        -------
        An integer, a float, or a string without a newline character.

        Inspired by BrukerMRI package. https://github.com/jdoepfert/brukerMRI.

        """
        # check if int
        try:
            return int(val)
        except (TypeError, ValueError):
            pass

        # check if float
        try:
            return float(val)
        except (TypeError, ValueError):
            pass

        # otherwise remove newline character and return string
        return val.rstrip("\n")

    @staticmethod
    def __ParseArray(current_file, line):
        """
        Extract an array from a line in a file.

        Parameters
        ----------
        current_file: A file object to read from.
        line: str
            A string containing an array definition.

        Returns
        -------
        A NumPy array or a single value if the array only contains one element.

        Inspired by BrukerMRI package. https://github.com/jdoepfert/brukerMRI.

        """
        # extract the arraysize and convert it to numpy
        line = line[1:-2].replace(" ", "").split(",")
        array_size = np.array([int(x) for x in line])

        # then extract the next line
        val_list = current_file.readline().split()

        # if the line was a string, then return it directly
        try:
            float(val_list[0])
        except (TypeError, ValueError):
            return " ".join(val_list)

        # include potentially multiple lines
        while len(val_list) != np.prod(array_size):
            val_list += current_file.readline().split()

        # try converting to int, if error, then to float
        try:
            val_list = [int(x) for x in val_list]
        except ValueError:
            val_list = [float(x) for x in val_list]

        # convert to numpy array
        if len(val_list) > 1:
            return np.reshape(np.array(val_list), array_size)
        # or to plain number
        return val_list[0]

    def __ReadParamFile(self, file_name: str) -> dict:
        """
        Reads a Bruker MRI experiment's method or acqp file to a dictionary.

        Parameters
        ----------
        file_name: str
            A string containing the name of the file to read.

        Returns
        -------
        parsed_params: dict
            A dictionary containing the parsed parameters and their values.

        Inspired by BrukerMRI package. https://github.com/jdoepfert/brukerMRI.

        """
        param_dict = {}
        file_path = self.path / file_name

        if not file_path.exists() or os.path.getsize(file_path) == 0:
            return param_dict

        with open(file_path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break

                # when line contains parameter
                if line.startswith("##$"):
                    (param_name, current_line) = line[3:].split("=")  # split at "="

                    # if current entry (current_line) is arraysize
                    if current_line[0:2] == "( " and current_line[-3:-1] == " )":
                        value = self.__ParseArray(f, current_line)

                    # if current entry (current_line) is struct/list
                    elif current_line[0] == "(" and current_line[-3:-1] != " )":
                        # if necessary read in multiple lines
                        while current_line[-2] != ")":
                            current_line = current_line[0:-1] + f.readline()

                        # parse the values to a list
                        value = [
                            self.__ParseSingleValue(x)
                            for x in current_line[1:-2].split(", ")
                        ]
                    # otherwise current entry must be single string or number
                    else:
                        value = self.__ParseSingleValue(current_line)

                    # save parsed value to dict
                    param_dict[param_name] = value

                # This exists only to extract the TimeStamp from the method
                # file
                if line.startswith("$$ File finished by PARX at"):
                    time_str = line.split("at")[1].strip()
                    format_str = "%Y-%m-%d %H:%M:%S.%f %z"

                    time_obj = datetime.strptime(time_str, format_str)

                    param_dict["MetaTimeStamp"] = time_obj

        return param_dict

    def __GetMeasurementType(self) -> str:
        """Returns measurement type from method or acqp file as string"""
        # 1. Try Method from method file (most specific)
        method = self.method.get("Method")
        if method and method != "unknown":
            return method

        # 2. Try ACQ_method from acqp file (fallback for some ParaVision versions)
        acq_method = self.acqp.get("ACQ_method")
        if acq_method and acq_method != "unknown":
            return acq_method

        # 3. Try ACQ_scan_name from acqp file (least specific, but better than nothing)
        scan_name = self.acqp.get("ACQ_scan_name")
        if scan_name and scan_name != "unknown":
            return scan_name

        return "unknown"

    def __FindRecons(self):
        """Find all recon-folder in pdata and return their names in a list."""
        pdata_path = self.path / "pdata"
        pdata_content = os.listdir(pdata_path)

        recon_folder_names = []

        for recon_name in pdata_content:
            if self.__is_valid_recon_dir(pdata_path / recon_name):
                recon_folder_names.append(recon_name)

        recon_folder_names.sort()

        return recon_folder_names

    # ---------------------------------------------------
    # ------------ LOAD DATA AND PARAMETERS -------------
    # ---------------------------------------------------
    def load_fid_file(self):
        """Read fid file and return it as np.array.

        If multi slice data is loaded into topspin (i.e. SP with repetitions)
        the fid file is replaced by the ser file, which is loaded similarly.

        Therefore, in case there is no fid file but an ser file, try to load ser
        instead.

        Returns
        -------
        np.array
            complex fid file

        """
        start_time = time.time()
        raw_file = None

        fid_path = self.path / "fid"
        ser_path = self.path / "ser"

        if fid_path.exists() and os.path.getsize(fid_path) > 0:
            file_size_mb = os.path.getsize(fid_path) / 1024 / 1024
            logger.info(f"[TIMING] Loading fid file ({file_size_mb:.1f} MB)")
            io_start = time.time()
            with open(fid_path, "rb") as f:
                try:
                    raw_file = np.fromfile(f, dtype=np.int32, count=-1)
                except Exception as e:
                    err_msg, err_type = str(e), e.__class__.__name__
                    msg = f"The following error occured while loading '{fid_path}':\n"
                    msg += f"    {err_type}: {err_msg}"
                    logger.error(msg)
            io_time = time.time() - io_start
            logger.info(f"[TIMING] File I/O took {io_time:.3f}s")

        elif ser_path.exists() and os.path.getsize(ser_path) > 0:
            file_size_mb = os.path.getsize(ser_path) / 1024 / 1024
            logger.info(f"[TIMING] Loading ser file ({file_size_mb:.1f} MB)")
            io_start = time.time()
            with open(ser_path, "rb") as f:
                try:
                    raw_file = np.fromfile(f, dtype=np.int32)
                except Exception as e:
                    err_msg, err_type = str(e), e.__class__.__name__
                    msg = f"The following error occured while loading '{ser_path}':\n"
                    msg += f"    {err_type}: {err_msg}"
                    logger.error(msg)
            io_time = time.time() - io_start
            logger.info(f"[TIMING] File I/O took {io_time:.3f}s")
        else:
            logger.error(
                f"Both 'fid' and 'ser' files do not exist or are empty for scan {self.ExpNum}."
            )

        if raw_file is None:
            return np.array([])

        # turn into complex array and return - using fastest method
        complex_start = time.time()

        # Fast method: reshape to pairs and use dtype conversion
        # This is much faster than indexing for large arrays
        try:
            # Reshape int32 array to pairs, then create complex from real/imag
            # parts
            pairs = raw_file.reshape(-1, 2)
            complex_data = pairs[:, 0] + 1j * pairs[:, 1]
            logger.debug(
                f"[TIMING] Fast complex conversion(reshape) took {time.time() - complex_start: .3f}s"
            )
        except ValueError:
            # Fallback to slower method if reshape fails (odd number of
            # elements)
            logger.warning("Reshape failed, using slower complex conversion method")
            complex_data = raw_file[0::2] + 1j * raw_file[1::2]
            logger.debug(
                f"[TIMING] Fallback complex conversion took {time.time() - complex_start: .3f}s"
            )

        total_time = time.time() - start_time
        logger.info(f"[TIMING] load_fid_file total: {total_time:.3f}s")

        return complex_data

    def load_rawdatajob0_file(self):
        """
        Read the rawdata.job0 file and return it as a np.array.

        Returns
        -------
        complex_file: complex fid

        """
        file_path = self.path / "rawdata.job0"

        if file_path.exists() and os.path.getsize(file_path) > 0:
            with open(file_path, "rb") as f:
                raw_file = np.fromfile(f, dtype=np.int32, count=-1)
                # Fast complex conversion
                try:
                    pairs = raw_file.reshape(-1, 2)
                    complex_file = pairs[:, 0] + 1j * pairs[:, 1]
                except ValueError:
                    # Fallback for odd-sized arrays
                    complex_file = raw_file[0::2] + 1j * raw_file[1::2]
        else:
            # only info-level since this is quite often the case.
            logger.info(
                f"'rawdata.job0' file does not exist or is empty for scan {self.ExpNum}."
            )
            complex_file = np.array([])

        return complex_file

    def load_2dseq_file(
        self, recon_num: int = 1, set_reco_file: bool = False, data_precision=np.float16
    ):
        """
        Load 2dseq file and return it as np.array.

        To load in a specific reconstruction use the argument recon_num.
        Note: This method makes use of the brukerapi package.

        Parameters
        ----------
        recon_num : int, optional
            Name/number of reconstruction, by default 1.
        set_reco_file : bool, optional
            If True, the reconstruction info file reco is set to the
            loaded one, by default False.
        data_precision : type, optional
            The desired numpy data type for the loaded data (e.g., np.float16, np.float32, np.float64), by default np.float16.

        Returns
        -------
        out : np.array
            Array of the 2dseq file with specified precision.

        """
        start_time = time.time()

        def load_2dseq_reco(self, recon_num=1):
            return self.__ReadParamFile(os.path.join("pdata", f"{recon_num}", "reco"))

        file_path = self.path / "pdata" / str(recon_num) / "2dseq"

        if file_path.exists() and os.path.getsize(file_path) > 0:
            file_size_mb = os.path.getsize(file_path) / 1024 / 1024
            logger.info(f"[TIMING] Loading 2dseq file ({file_size_mb:.1f} MB)")

            if set_reco_file:
                reco_start = time.time()
                try:
                    self.reco = load_2dseq_reco(self, recon_num)
                except BaseException:
                    pass
                reco_time = time.time() - reco_start
                logger.info(f"[TIMING] Reco file loading took {reco_time: .3f}s")

            data_start = time.time()
            data = Dataset(file_path).data
            data_time = time.time() - data_start
            logger.info(f"[TIMING] Dataset loading took {data_time:.3f}s")

            # Apply data precision conversion
            precision_start = time.time()
            data = self._convert_data_precision(data, data_precision)
            precision_time = time.time() - precision_start
            logger.info(
                f"[TIMING] Data precision conversion took {precision_time:.3f}s"
            )

            total_time = time.time() - start_time
            logger.info(f"[TIMING] load_2dseq_file total: {total_time:.3f}s")

            return data
        else:
            # only info-level since this happens quite often and recon numbers
            # are already displayed with Study by default.
            logger.info(
                f"'2dseq' file does not exist or is empty for scan {self.ExpNum}."
            )
            return np.array([])

    def _convert_data_precision(self, array: np.ndarray, precision):
        """
        Convert array to specified precision, handling complex data appropriately.

        Parameters
        ----------
        array : np.ndarray
            Input array to convert
        precision : type
            The desired numpy data type (e.g., np.float16, np.float32, np.float64)

        Returns
        -------
        np.ndarray
            Array converted to the specified precision

        """
        if array is None or array.size == 0:
            return array

        # Validate precision
        if precision not in [np.float16, np.float32, np.float64]:
            logger.warning(
                f"Unsupported data precision: {precision}. Using np.float16."
            )
            precision = np.float16

        # Check if array is complex
        try:
            is_complex = np.any(np.iscomplex(array))
        except (TypeError, ValueError):
            is_complex = False

        if is_complex:
            # For complex data, convert to appropriate complex type
            if precision == np.float16:
                # complex64 uses float32 for real/imag parts (smallest complex type)
                return array.astype(np.complex64)
            elif precision == np.float32:
                return array.astype(np.complex64)
            elif precision == np.float64:
                return array.astype(np.complex128)
        else:
            # For real data, convert directly
            return array.astype(precision)

    def __get_git_commit_hash(
        self,
    ):
        """
        Get the current git commit hash of the repository containing this script.

        Returns
        -------
        git commit has: str
            The git commit hash, or an error message if not found.

        """
        try:
            # Get the directory where this script (brukerexp.py) is located.
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

    def open(self, paramfile: str = "method"):
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

    def summary(self):
        """
        Return summary table of frequency related parameter as pd.dataframe.
        """
        df = pd.DataFrame()

        paramlist = {
            "Reference Frequency [MHz]": "PVM_FrqRef",
            "Reference Frequency [ppm]": "PVM_FrqRefPpm",
            "Working Frequency [MHz]": "PVM_FrqWork",
            "Working Frequency [ppm]": "PVM_FrqWorkPpm",
            "Working Frequency Offset [Hz]": "PVM_FrqWorkOffset",
            "Working Frequency Offset [ppm]": "PVM_FrqWorkOffsetPpm",
        }

        for description, key in paramlist.items():
            try:
                df[description] = list(self.method[key])
            except KeyError:
                pass

        cols = ["Reference Frequency [MHz]"]
        df[cols] = df[df[cols] > 0][cols]
        df = df.dropna().T
