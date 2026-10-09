from ...utils.system.logging import init_default_logger

# initialize logger
logger = init_default_logger(__name__)

# try:
#     from umpire import UMPIRE, phase2frequency
# except ModuleNotFoundError:
#     # logger.info("UMPIRE package not found.")
#     pass

LAMOR_FREQUENCY_H1 = 300.130
LAMOR_FREQUENCY_C13 = 75.468


from ..MGE import MGE


class umpireMGE(MGE):
    """Multi-echo gradient echo sequence encapsulating TE-dependent reconstructions."""


# class umpireMGE(BaseSpectroscopy):
#     """Analysis class for the umpireMGE sequence.

#     Multi-gradient echo sequence for spectroscopic applications. The sequence acquires
#     multiple gradient echoes at different echo times, which are treated as spectroscopic
#     acquisition points. The echo times are automatically converted to acquisition_idx
#     dimension during reconstruction.
#     """

#     def __init__(
#         self,
#         mri_exp_obj: MRIExperiment,
#         *args,
#         **kwargs,
#     ):
#         """Accepts directory path or MRIExperiment object as input."""
#         # Extract sequence-specific kwargs before calling parent constructor
#         reco_options = kwargs.pop("reco_options", {})

#         super().__init__(mri_exp_obj, *args, **kwargs)

#         # Override is_spectroscopic flag for MGE sequences
#         # MGE sequences use multiple echoes as spectroscopic acquisition points
#         if hasattr(self, "handler") and hasattr(self.handler, "method"):
#             if "EffectiveTE" in self.handler.method:
#                 self.handler.is_spectroscopic = True
#                 logger.info(
#                     "umpireMGE: Set is_spectroscopic=True based on EffectiveTE presence"
#                 )

#                 # Calculate spectroscopic parameters from echo times
#                 effective_te_values = self.handler.method["EffectiveTE"]
#                 if len(effective_te_values) > 1:
#                     # Convert to seconds
#                     effective_te_s = np.array(effective_te_values) / 1000.0

#                     # Echo spacing is the spectroscopic dwell time
#                     delta_te = np.diff(effective_te_s)[0]

#                     # Override spectroscopic parameters
#                     self.handler.dwell_time_s = delta_te
#                     self.handler.bandwidth_hz = 1.0 / delta_te
#                     self.handler.n_acquisition = len(effective_te_values)
#                     self.handler.sampling_time_axis_s = effective_te_s

#                     # Calculate bandwidth in ppm if we have reference frequency
#                     if (
#                         hasattr(self.handler, "reference_frequency_mhz")
#                         and self.handler.reference_frequency_mhz
#                     ):
#                         self.handler.bandwidth_ppm = (
#                             self.handler.bandwidth_hz
#                             / self.handler.reference_frequency_mhz
#                         )

#                     logger.info(
#                         f"umpireMGE: Set spectroscopic parameters - "
#                         f"bandwidth_hz={self.handler.bandwidth_hz:.2f} Hz, "
#                         f"dwell_time_s={self.handler.dwell_time_s*1000:.3f} ms, "
#                         f"n_acquisition={self.handler.n_acquisition}"
#                     )

#                     # Update handler's params dict and re-sync to sequence object
#                     if hasattr(self.handler, "params"):
#                         # Update the handler's params dict directly
#                         self.handler.params["bandwidth_hz"] = self.handler.bandwidth_hz
#                         self.handler.params["bandwidth_ppm"] = (
#                             self.handler.bandwidth_ppm
#                         )
#                         self.handler.params["dwell_time_s"] = self.handler.dwell_time_s
#                         self.handler.params["n_acquisition"] = (
#                             self.handler.n_acquisition
#                         )
#                         self.handler.params["sampling_time_axis_s"] = (
#                             self.handler.sampling_time_axis_s
#                         )

#                         # Now re-sync parameters to sequence object
#                         try:
#                             handler_params = self.handler.get_parameters()
#                             if handler_params:
#                                 self.set_parameters(handler_params, allow_extra=True)
#                                 logger.debug(
#                                     "umpireMGE: Re-synchronized parameters after MGE spectroscopic setup"
#                                 )
#                         except Exception as exc:
#                             logger.debug(
#                                 f"umpireMGE: parameter re-synchronization skipped: {exc}"
#                             )

#         # set up logging level
#         #  Handler copying is now handled by BaseSpectroscopy
#         # Just verify the handler is available
#         if hasattr(self, "handler") and self.handler:
#             logger.debug(
#                 f"umpireMGE: handler available, has fid: {hasattr(self.handler, 'fid')}"
#             )
#         else:
#             logger.warning(
#                 "umpireMGE: No handler available after BaseSpectroscopy initialization"
#             )

#         # Copy any existing reconstructed data
#         if hasattr(mri_exp_obj, "data_array") and mri_exp_obj.data_array is not None:
#             self.data_array = mri_exp_obj.data_array
#         # kspace_array no longer stored permanently

#         # Perform reconstructions
#         try:
#             self.reconstruct(reco_type="seq2d", **reco_options)
#         except Exception as e:
#             logger.error(f"umpireMGE seq2d reconstruction failed: {e}")
#             # Don't raise - allow partial initialization

#         # Raw data processing:
#         try:
#             self.reconstruct(reco_type="raw", **reco_options)
#         except Exception as e:
#             logger.error(f"umpireMGE raw reconstruction failed: {e}")
#             # Don't raise - allow partial initialization

#         # # save some handy method parameters for easier access
#         # if self.method["PVM_SpatDimEnum"] == "<2D>":
#         #     logger.warning("2D Acquistion Found! Use at your own risk.")
#         #     slice_thick = self.method["PVM_SliceThick"]
#         #     nslices = self.method["PVM_SPackArrNSlices"]

#         #     fov_rp = self.method["PVM_Fov"]
#         #     self.pvm_fov = [*fov_rp, slice_thick * nslices]

#         #     res_rp = self.method["PVM_SpatResol"]
#         #     self.pvm_res = [*res_rp, slice_thick]

#         #     mat_rp = self.method["PVM_Matrix"]
#         #     self.pvm_matrix = [*mat_rp, nslices]
#         # else:
#         #     self.pvm_fov = self.method["PVM_Fov"]
#         #     self.pvm_res = self.method["PVM_SpatResol"]
#         #     self.pvm_matrix = self.method["PVM_Matrix"]
#         # self.pvm_orientation = self.method["PVM_SPackArrSliceOrient"]
#         # self.pvm_readorientation = self.method["PVM_SPackArrReadOrient"]

#         # # make sure our echo spacing is equal
#         # echo_time = self.method["PVM_EchoTime"]
#         # echo_spacing = self.method["EchoSpacing"]

#         if False:  # echo_time != echo_spacing:
#             raise Warning(
#                 f"PVM_EchoTime ({echo_time:.2f}ms)!= EchoSpacing ({echo_spacing:.2f}ms)"
#             )

#         # self.repetition = repetition

#         # # to keep track of different kinds of masks, they get combined in
#         # # self.mask
#         # self.mask_dict = {}

#         # (try to) load in complex data, order is: Echo Read Phase Slice
#         if isinstance(external_data, type(None)):
#             self.data = self.load_data(recon_num)
#         else:
#             self.slice_orientations = "coronal", "axial", "sagittal"
#             self.data = external_data

#         self.rawdatajob1 = self.Load_rawdatajob1_file()

#         # self._phase = np.angle(self.data)
#         # self._unwrapped_phase = np.zeros_like(self.data, dtype="float")
#         # self._magnitude = np.abs(self.data)

#         self._fmap = np.zeros_like(self.data[0], dtype="float")
#         self._fmap_err = np.zeros_like(self.data[0], dtype="float")
#         self._fit_results = np.zeros(
#             (*self.data[0].shape, 4), dtype="float"
#         )  # m, merr, t, terr
#         self._pHmap = np.zeros_like(self.data[0], dtype="float")
#         self._multi_comp_fmaps = np.zeros_like(self.data[:1, :, :, :], dtype="float")
#         self._multi_comp_amplitudes = np.zeros_like(
#             self.data[:1, :, :, :], dtype="float"
#         )
#         self._multi_comp_pHmaps = np.empty(([3, *self.data[0].shape]), dtype="float")
#         self._multi_comp_pHmaps.fill(np.nan)
#         self._multi_comp_mean_pH = np.zeros_like(self.data[0], dtype="float")
#         self._multi_comp_peak_pH = np.zeros_like(self.data[0], dtype="float")
#         self._T2_star_map = np.zeros_like(self.data[0], dtype="float")
#         self._fit_cost_map = np.zeros_like(self.data[0], dtype="float")
#         self._c1_ampl_map = np.zeros_like(self.data[0], dtype="complex")
#         self._f_offset = 0
#         self._max_num_comps_MEpHISDO = 0
#         self.useNechoes = slice(0, len(self.TEs), None)

#         # # bool value to keep track if we interpolated the data
#         # self.interpolated = False

#         # self._interp_phase = None
#         # self._interp_unwrapped_phase = None
#         # self._interp_magnitude = None

#         # # potentially used for snr scaling of the magnitude
#         # self._noise_stddev = None

#     @property
#     def mask(self):
#         """Returns a combination of all masks from 'mask_dict' as a single mask."""
#         if self.mask_dict:
#             return np.logical_or.reduce(list(self.mask_dict.values()))
#         return np.zeros_like(self._fmap)

#     @property
#     def fmap(self):
#         return np.ma.masked_array(data=self._fmap, mask=self.mask)

#     @property
#     def fmap_err(self):
#         return np.ma.masked_array(data=self._fmap_err, mask=self.mask)

#     @property
#     def fmap_err_rel(self):
#         """https://qastack.com.de/stats/86708"""
#         out = np.abs(self._fmap_err / self._fmap) * 100
#         return np.ma.masked_array(data=out, mask=self.mask)

#     @property
#     def magnitude(self):
#         mag = self._interp_magnitude if self.interpolated else self._magnitude

#         mask = np.repeat(self.mask[np.newaxis], len(self.TEs), axis=0)

#         std = self._noise_stddev or 1

#         return np.ma.masked_array(data=mag / std, mask=mask)

#     @property
#     def phase(self):
#         phase = self._interp_phase if self.interpolated else self._phase

#         mask = np.repeat(self.mask[np.newaxis], len(self.TEs), axis=0)

#         return np.ma.masked_array(data=phase, mask=mask)

#     @property
#     def unwrapped_phase(self):
#         if self.interpolated:
#             return self._interp_unwrapped_phase
#         return self._unwrapped_phase

#     @property
#     def pHmap(self):
#         out = np.ma.masked_array(data=self._pHmap, mask=self.mask)
#         if np.isnan(out).all():
#             logger.error("Masking of pH-Map aborted, all entries would be NaN.")
#         return out

#     @property
#     def T2_star(self):
#         out = np.ma.masked_array(
#             data=self._T2_star_fit_results[0].squeeze(), mask=self.mask
#         )
#         if np.isnan(out).all():
#             logger.error("Masking of T2*-Map aborted, all entries would be NaN.")
#         return out

#     @property
#     def spectral_data(self):

#         if len(self.rawdatajob1) == 0:
#             raise Exception(
#                 "No RawDataJob1 file found. Make sure you acquired additional spectra!"
#             )
#         else:
#             acq_points = int(self.method["NSpecPoints"])
#             num_spectra = int(len(self.rawdatajob1) / acq_points)

#             complex_spec = np.zeros((num_spectra, acq_points), dtype=complex)

#             ffts = lambda x: np.fft.fftshift(np.fft.fft(x))

#             for idx in range(num_spectra):
#                 complex_spec[idx, :] = ffts(
#                     self.rawdatajob1[idx * acq_points : acq_points + idx * acq_points]
#                 )

#             # calculate frequency axis
#             d = 1 / self.method["PVM_DigSw"]
#             frequ_axis = np.fft.fftshift(np.fft.fftfreq(acq_points, d=d))

#             return complex_spec, frequ_axis

#     def _useNechoes_input_handler(self, usr_input, verbose=False):
#         vprint = print if verbose else lambda *a, **k: None

#         if not usr_input:
#             vprint(f"Using all N={self.n_echoes} echoes.")
#             usr_input = slice(None)
#         elif isinstance(usr_input, int):
#             usr_input = slice(0, usr_input, None)
#             vprint(f"Using N={np.arange( 0, self.n_echoes, 1)[usr_input]} echoes.")
#         else:
#             vprint(f"Using N={np.arange( 0, self.n_echoes, 1)[usr_input]} echoes.")

#         return usr_input

#     def load_data(self, default_recon_num=-1):
#         """Load complex 2dseq reconstruction.

#         WARNING: Assumes the last recon is complex! Could be dangerous,
#                  bruker2complex throws an error if dimension != 2 though...
#         """
#         complex_recon_name = self.recon_names[default_recon_num]

#         # load data
#         data = self.load_2dseq_file(recon_num=complex_recon_name)

#         assert "complex" in str(data.dtype), "No complex data found!"
#         # convert into complex dtype - not necessary anymore see brukerapi@luca
#         # data = bruker2complex(data)

#         if self.repetition is not None:
#             print(f"Found {data.shape[-1]} repetitions. Using idx={self.repetition}.")
#             data = data[..., self.repetition]

#         if self.method["PVM_SpatDimEnum"] == "<2D>":
#             # only works for sag HF!
#             data = np.moveaxis(data, 0, -1)
#             data = np.moveaxis(data, 0, 1)
#             data = np.moveaxis(data, -2, -1)
#             self.slice_orientations = "coronal", "axial", "sagittal"
#             return data

#         # ensure the following axis order: read, phase, slice, echo
#         data = self.ensure_RPS_order(data)

#         # swap echo axis from last to first (bruker puts the echo dimension at
#         # the last axis per default)
#         data = np.moveaxis(data, -1, 0)

#         return data

#     def ensure_RPS_order(self, data):
#         """Swap data axis to ensure read, phase, slice order."""
#         assert (
#             self.acqp["ACQ_patient_pos"] == "Head_Prone"
#         ), f"Found {self.acqp['ACQ_patient_pos']} instead of 'Head_Prone'."

#         if self.n_slices > 1:
#             # Reorient the array, make read always first!
#             if self.pvm_orientation == "axial":
#                 if self.pvm_readorientation == "L_R":
#                     self.slice_orientations = "axial", "sagittal", "coronal"
#                     return data
#                 if self.pvm_readorientation in ("V_D", "A_P"):
#                     self.slice_orientations = "axial", "coronal", "sagittal"
#                     return np.moveaxis(data, 0, 1)

#             if self.pvm_orientation == "sagittal":
#                 if self.pvm_readorientation in ("V_D", "A_P"):
#                     self.slice_orientations = "sagittal", "coronal", "axial"
#                     return data
#                 if self.pvm_readorientation == "H_F":
#                     self.slice_orientations = "sagittal", "axial", "coronal"
#                     return np.moveaxis(data, 0, 1)

#             if self.pvm_orientation == "coronal":
#                 if self.pvm_readorientation == "L_R":
#                     self.slice_orientations = "coronal", "sagittal", "axial"
#                     return data
#                 if self.pvm_readorientation == "H_F":
#                     self.slice_orientations = "coronal", "axial", "sagittal"
#                     return np.moveaxis(data, 0, 1)

#         else:
#             if self.pvm_orientation == "axial":
#                 self.slice_orientations = ("axial",)
#                 if self.pvm_readorientation == "L_R":
#                     return data
#                 if self.pvm_readorientation == "A_P":
#                     return np.moveaxis(data, 0, 1)

#             if self.pvm_orientation == "sagittal":
#                 self.slice_orientations = ("sagittal",)
#                 if self.pvm_readorientation == "A_P":
#                     return data
#                 if self.pvm_readorientation == "H_F":
#                     return np.moveaxis(data, 0, 1)

#             if self.pvm_orientation == "coronal":
#                 self.slice_orientations = ("coronal",)
#                 if self.pvm_readorientation == "L_R":
#                     return data
#                 if self.pvm_readorientation == "H_F":
#                     return np.moveaxis(data, 0, 1)

#         raise NotImplementedError(
#             f"Got unknown orientation: '{self.pvm_orientation}'"
#             + f"and '{self.pvm_readorientation}'."
#         )

#     def umpire(
#         self,
#         usen_echoes=False,
#         DPD_filter_func="default",
#         magnitude_weighted_omega_star=False,
#         noreturn=True,
#     ):
#         # handle what echoes to use
#         self.usen_echoes = self._usen_echoes_input_handler(usen_echoes, verbose=True)

#         out = UMPIRE(
#             self.data[self.usen_echoes],
#             self.TEs[self.usen_echoes],
#             DPD_filter_func=DPD_filter_func,
#             magnitude_weighted_omega_star=magnitude_weighted_omega_star,
#         )

#         # store results
#         self._unwrapped_phase = out

#         if not noreturn:
#             return out

#     def MEpHISDO(
#         self,
#         max_num_compartments=3,
#         frequ_inits=np.array([250.0, 185.0, 80.0]),
#         frequ_boundaries=Bounds([-100, -100, -100], [400, 400, 400]),
#         amplitude_inits=np.array([40 + 1j, 20 + 1j, 10 + 1j]),
#         amplitude_boundaries=Bounds(
#             [-200, -200, -200, -200, -200, -200], [200, 200, 200, 200, 200, 200]
#         ),
#         t2_star_init=0.013,
#         t2_star_boundary=Bounds([0.001], [0.1]),
#         threshold_ampl_ratio=0.1,
#         threshold_frequency_proximity=10,
#         c1_contrib=-np.inf,
#         noreturn=True,
#         **least_squares_kwargs,
#     ):

#         MEpHISDO_solver = MEpHISDO_Solver(max_num_compartments, uMGE_object=self)

#         out = MEpHISDO_solver.solve(
#             frequ_inits=frequ_inits,
#             frequ_boundaries=frequ_boundaries,
#             amplitude_inits=amplitude_inits,
#             amplitude_boundaries=amplitude_boundaries,
#             t2_star_init=t2_star_init,
#             t2_star_boundary=t2_star_boundary,
#             threshold_ampl_ratio=threshold_ampl_ratio,
#             threshold_frequency_proximity=threshold_frequency_proximity,
#             c1_contrib=c1_contrib,
#             **least_squares_kwargs,
#         )

#         # store results
#         self._multi_comp_fmaps = out[:max_num_compartments, :, :, :].real
#         self._multi_comp_amplitudes = out[
#             max_num_compartments : 2 * max_num_compartments, :, :, :
#         ]

#         if c1_contrib != -np.inf:
#             self._c1_ampl_map = out[-2, :, :, :]

#         self._fit_cost_map = out[-1, :, :, :].real / np.max(np.abs(self.data), axis=0)

#         self._T2_star_map = out[2 * max_num_compartments, :, :, :].real

#         self._max_num_comps_MEpHISDO = max_num_compartments

#         if not noreturn:
#             return out

#     def pHIDEAL(
#         self,
#         num_frequencies=3,
#         field_map=None,
#         noreturn=True,
#         max_iterations=300,
#         stop_f_offset=0.5,
#         frequ_init=np.array([-90.7, 16.0, 55.9]),
#         use_UMPIRE_init=True,
#         fit_threshold=0.3,
#         discard_similar=True,
#         t_2_star_init=None,
#     ):
#         solver = pHIDEAL_Solver(
#             num_frequencies,
#             uMGE_object=self,
#         )

#         out = solver.solve(
#             t2_star_init=t_2_star_init,
#             field_map=field_map,
#             max_iterations=max_iterations,
#             stop_f_offset=stop_f_offset,
#             frequ_inits=frequ_init,
#             use_UMPRIRE_freq_for_init=use_UMPIRE_init,
#             discard_similar=discard_similar,
#             fit_threshold=fit_threshold,
#         )

#         # store results
#         self._multi_comp_fmaps = out[:num_frequencies, :, :, :].astype(float)
#         self._multi_comp_amplitudes = out[
#             num_frequencies : 2 * num_frequencies, :, :, :
#         ]

#         if not isinstance(t_2_star_init, type(None)):
#             self._T2_star_map = out[2 * num_frequencies, :, :, :].astype(float)

#         if not noreturn:
#             return out

#     def phase2frequency(
#         self,
#         usen_echoes=None,
#         load_bar=True,
#         keep_dims=True,
#         return_fit_results=False,
#         njobs=None,
#     ):
#         # check wether all echoes were used during self.umpire()
#         if usen_echoes is not None:
#             if len(self.data) > len(self.unwrapped_phase):
#                 AssertionError("usen_echoes already specified in self.umpire()!")
#             else:
#                 usen_echoes = self._usen_echoes_input_handler(usen_echoes)

#                 unwrapped_phase = self.unwrapped_phase[usen_echoes]
#                 TEs = self.TEs[usen_echoes]
#         else:
#             unwrapped_phase = self.unwrapped_phase
#             TEs = self.TEs

#         print(self.unwrapped_phase.shape)

#         return phase2frequency(
#             unwrapped_phase,
#             TEs,
#             load_bar=load_bar,
#             keep_dims=keep_dims,
#             return_fit_results=return_fit_results,
#             njobs=njobs,
#         )

#     def interpolate(self, interp_fac, method="nearest"):
#         """
#         interp_fac: int, np.ndarray
#         """

#         def i3d(arr, method=method):
#             return interpolate_3d_array(arr, new_dims, method=method)

#         # reset masks for interpolation process
#         self.mask_dict = {}

#         # calculate new dimensions
#         old_dims = np.array(self.data.shape[1:])  # skip echo dimension

#         # check if we have an int or a shape as interp_fac argument
#         if isinstance(interp_fac, int):
#             new_dims = old_dims * interp_fac
#         elif isinstance(interp_fac, (tuple, list, np.ndarray)):
#             new_dims = np.array(interp_fac)
#         else:
#             raise AssertionError("Not a valid interpolation factor. See the docs.")

#         self.interp_matrix = new_dims

#         # interpolate magnitude, phase and unwrapped phase:
#         interp_magnitude = np.zeros((self.n_echoes, *new_dims))
#         interp_phase = np.zeros((self.n_echoes, *new_dims))
#         interp_unwrapped_phase = np.zeros((self.n_echoes, *new_dims))

#         np.zeros(new_dims)
#         np.zeros(new_dims)

#         for i, _ in enumerate(self.TEs[self.usen_echoes]):
#             interp_phase[i] = i3d(self._phase[i])
#             interp_unwrapped_phase[i] = i3d(self._unwrapped_phase[i])
#             interp_magnitude[i] = i3d(self._magnitude[i])

#         self._interp_phase = interp_phase
#         self._interp_unwrapped_phase = interp_unwrapped_phase
#         self._interp_magnitude = interp_magnitude

#         # reset all other arrays based on old dimensions/information
#         # these arrays, are populated by (re-)calling the appropriate methods
#         self._fmap = np.zeros(new_dims)
#         self._fmap_err = np.zeros(new_dims)
#         self._fit_results = np.zeros((*self.pvm_matrix, 4))  # m, merr, t, terr
#         self._pHmap = np.zeros(new_dims)
#         self._fmap = np.zeros(new_dims)
#         self._multi_comp_fmaps = np.zeros((1, *new_dims))
#         self._multi_comp_pHmaps = np.zeros((1, *new_dims))
#         self._multi_comp_mean_pH = np.zeros(new_dims)
#         self._multi_comp_peak_pH = np.zeros(new_dims)

#         self.interpolated = True
#         print(f"{self.pvm_matrix} --> {new_dims}")

#     def interpolate_afterwards(self, interp_fac, method="nearest"):
#         def i3d(arr, method=method):
#             return interpolate_3d_array(arr, new_dims, method=method)

#         # reset masks for interpolation process
#         self.mask_dict = {}

#         # calculate new dimensions
#         old_dims = np.array(self.data.shape[1:])  # skip echo dimension

#         # check if we have an int or a shape as interp_fac argument
#         if isinstance(interp_fac, int):
#             new_dims = old_dims * interp_fac
#         elif isinstance(interp_fac, (tuple, list, np.ndarray)):
#             new_dims = np.array(interp_fac)
#         else:
#             raise AssertionError("Not a valid interpolation factor. See the docs.")

#         self.interp_matrix = new_dims

#         # interpolate magnitude, phase and unwrapped phase:
#         interp_magnitude = np.zeros((self.n_echoes, *new_dims))
#         interp_phase = np.zeros((self.n_echoes, *new_dims))
#         interp_unwrapped_phase = np.zeros((self.n_echoes, *new_dims))

#         np.zeros(new_dims)
#         np.zeros(new_dims)

#         for i, _ in enumerate(self.TEs[self.usen_echoes]):
#             interp_phase[i] = i3d(self._phase[i])
#             interp_unwrapped_phase[i] = i3d(self._unwrapped_phase[i])
#             interp_magnitude[i] = i3d(self._magnitude[i])

#         self._interp_phase = interp_phase
#         self._interp_unwrapped_phase = interp_unwrapped_phase
#         self._interp_magnitude = interp_magnitude

#         # interpolate fmap, fmap_err and pHmap:
#         self._fmap = i3d(self._fmap)
#         self._fmap_err = i3d(self._fmap_err)
#         self._pHmap = i3d(self._pHmap)

#         # interpolate _multi_comp_fmaps and _multi_comp_pHmaps
#         interp_multi_comp_fmaps = np.zeros((self._multi_comp_fmaps.shape[0], *new_dims))
#         interp_multi_comp_pHmaps = np.zeros(
#             (self._multi_comp_pHmaps.shape[0], *new_dims)
#         )
#         interp_multi_comp_amplitudes = np.zeros(
#             (self._multi_comp_amplitudes.shape[0], *new_dims)
#         )

#         for i in range(self._multi_comp_fmaps.shape[0]):
#             interp_multi_comp_fmaps[i] = i3d(self._multi_comp_fmaps[i])

#         for i in range(self._multi_comp_pHmaps.shape[0]):
#             interp_multi_comp_pHmaps[i] = i3d(self._multi_comp_pHmaps[i])

#         for i in range(self._multi_comp_amplitudes.shape[0]):
#             interp_multi_comp_amplitudes[i] = i3d(self._multi_comp_amplitudes[i])

#         self._multi_comp_fmaps = interp_multi_comp_fmaps
#         self._multi_comp_pHmaps = interp_multi_comp_pHmaps
#         self._multi_comp_amplitudes = interp_multi_comp_amplitudes
#         self._multi_comp_mean_pH = i3d(self._multi_comp_mean_pH)
#         self._multi_comp_peak_pH = i3d(self._multi_comp_peak_pH)
#         self._T2_star_map = i3d(self._T2_star_map)
#         self._fit_cost_map = i3d(self._fit_cost_map)

#         # interpolate fit results (not sure this is the correct way to do it):
#         interp_fit_results = np.zeros((*new_dims, 4))  # m, merr, t, terr
#         for i in range(4):
#             interp_fit_results[..., i] = i3d(self._fit_results[..., i])
#         self._fit_results = interp_fit_results

#         self.interpolated = True

#         print(f"{self.pvm_matrix} --> {new_dims}")

#     def calculate_fmap(self, usen_echoes=None, load_bar=True, njobs=None):
#         # generate frequency map from unwrapped phase images
#         fmap_out = self.phase2frequency(
#             usen_echoes=usen_echoes,
#             load_bar=load_bar,
#             return_fit_results=True,
#             njobs=njobs,
#         )

#         self._fmap = fmap_out[0]
#         self._fmap_err = fmap_out[1]
#         self._fit_results = fmap_out[2]

#     def mask_snr(self, echoNr=2):
#         """Interactive SNR masking interface.

#         Parameters to adjust:
#             1. 'Thr. SNR': The SNR threshold. Everything below gets masked.
#             2. 'Factor': Controlling the area size of the noise region.

#         After adjusting the parameters, click "SAVE MASK" to save the mask.

#         For silent masking, without any interaction use mask_slient() instead.
#         """
#         SNRMaskingPlotter(self, echoNr)

#     def mask_snr_silent(self, thr=10, factor=0.05, echoNr=2):
#         """SNR masking without any visualization."""
#         # magnitude of first echo only
#         mag = self.magnitude[echoNr]

#         SNR_scaled_mag, _, stddev = SNR(np.squeeze(mag), factor)

#         print(f"Std.Dev.: {stddev:.2f}")
#         print(f"SNR Min.: {np.min(SNR_scaled_mag):.2f}")
#         print(f"SNR Max.: {np.max(SNR_scaled_mag):.2f}")

#         mask_mag = createMaskFromThreshold(SNR_scaled_mag, thr)

#         self.mask_dict["snr"] = mask_mag
#         self._noise_stddev = stddev

#     def calculate_pH(self, titrationcurve=None, multi_comp=True):
#         """Calculate pH from frequency map. Assuming C5 peak.

#         titrationcurve : optional
#             Either None or 'andre'. Default is None, here the titration curve
#             parameter from M.Grashei and P.Wodtke published here
#             https://doi.org/10.1038/s41467-023-40747-3 were used.
#             If 'andre', the new titrationcurve acquired without Urea mixed in, is
#             used (see appendix Master Thesis.)
#         """
#         # [pK, δHA, δA = δHA + δ_range] - Reference: Urea
#         PARAMS_OMPD_C5_Urea = 6.5450, 12.4801, 12.4801 + 3.2975
#         # [pK, δHA, δA] of ZOMPD C5  - Reference: Urea
#         PARAMS_ANDRE_OMPD_Urea_C5 = 6.417854, 12.409269, 15.973497

#         if titrationcurve is None:
#             PARAMS = PARAMS_OMPD_C5_Urea
#         elif titrationcurve in ("andre", "Andre", "new"):
#             PARAMS = PARAMS_ANDRE_OMPD_Urea_C5
#         else:
#             AssertionError("titrationcurve takes None or 'andre' as input.")

#         f_0 = self.method["PVM_FrqWork"][0]

#         def ppm2pH(ppm, params):
#             """Henderson-Hasselbach relationship to convert chemical shift to pH."""
#             pK, dHA, dA = params
#             log_arg = (ppm - dHA) / (dA - ppm)

#             return pK + np.log10(log_arg)

#         ppmMap = self._fmap / f_0

#         pHmap = ppm2pH(ppmMap, PARAMS)

#         if np.isnan(pHmap).all():
#             logger.error("The pH conversion failed, all entries are NaN.")
#         else:
#             self._pHmap = pHmap

#         if multi_comp and np.count_nonzero(self._multi_comp_fmaps) != 0:

#             multi_comp_ppmMap = self._multi_comp_fmaps / f_0

#             mutli_comp_pHmap = ppm2pH(multi_comp_ppmMap, PARAMS)

#             if np.isnan(mutli_comp_pHmap).all():
#                 logger.error(
#                     "The pH conversion for the multi_comp results failed, all entries are NaN."
#                 )
#             else:
#                 self.sort_pH_by_compartment(mutli_comp_pHmap)
#                 # self._multi_comp_pHmaps = pHmap_pHIDEAL

#                 # calculate mean pH map
#                 self.calculate_mean_pH_map()

#     def calculate_mean_pH_map(self):

#         def calc_mean(idx):

#             weighted_pH = 0

#             index_not_nan = np.argwhere(~np.isnan(self._multi_comp_pHmaps[(..., *idx)]))

#             for map_idx in index_not_nan:
#                 weighted_pH = weighted_pH + (
#                     self._multi_comp_pHmaps[(map_idx, *idx)]
#                     * (
#                         np.abs(self._multi_comp_amplitudes[(map_idx, *idx)])
#                         / np.nansum(
#                             np.abs(self._multi_comp_amplitudes[(index_not_nan, *idx)])
#                         )
#                     )
#                 )

#             if weighted_pH == 0:
#                 return float("Nan")
#             else:
#                 return np.squeeze(weighted_pH)

#         _, dim_read, dim_phase, dim_slice = self._multi_comp_pHmaps.shape
#         index_list = list(np.ndindex(dim_read, dim_phase, dim_slice))

#         # serial solve for mutliple voxels in parallel
#         solution_serial = Parallel(n_jobs=4)(
#             delayed(calc_mean)(idx) for idx in index_list
#         )

#         mean_ph_map = np.array(solution_serial).reshape(
#             dim_read, dim_phase, dim_slice, -1
#         )

#         mean_ph_map = np.squeeze(mean_ph_map)

#         self._multi_comp_mean_pH = mean_ph_map

#     def calculate_peak_pH_map(self):

#         def calc_peak(idx):

#             index_not_nan = np.argwhere(~np.isnan(self._multi_comp_pHmaps[(..., *idx)]))

#             if index_not_nan.size != 0:

#                 max_index = np.nanargmax(
#                     np.abs(self._multi_comp_amplitudes[(..., *idx)])
#                 )

#                 return self._multi_comp_pHmaps[(max_index, *idx)]

#             else:
#                 return np.nan

#         _, dim_read, dim_phase, dim_slice = self._multi_comp_pHmaps.shape
#         index_list = list(np.ndindex(dim_read, dim_phase, dim_slice))

#         for index in index_list:
#             calc_peak(index)

#             # serial solve for mutliple voxels in parallel
#         solution_serial = Parallel(n_jobs=4)(
#             delayed(calc_peak)(idx) for idx in index_list
#         )

#         peak_ph_map = np.array(solution_serial).reshape(
#             dim_read, dim_phase, dim_slice, -1
#         )

#         peak_ph_map = np.squeeze(peak_ph_map)

#         self._multi_comp_peak_pH = peak_ph_map

#     def sort_pH_by_compartment(self, pH_map):

#         def sort_compartments(idx, pHmap):

#             pH_values = pHmap[(..., *idx)]
#             frequencies = self._multi_comp_fmaps[(..., *idx)]
#             amplitudes = self._multi_comp_amplitudes[(..., *idx)]

#             indexes_not_nan = np.argwhere(~np.isnan(pH_values))

#             output = np.empty(9, dtype="complex")
#             output.fill(np.nan)

#             if pH_values.size == 3 and indexes_not_nan.size == 3:

#                 output[:3] = pH_values
#                 output[3:6] = frequencies
#                 output[6:9] = amplitudes

#                 return output

#             elif indexes_not_nan.size == 1:

#                 pH_value = pH_values[indexes_not_nan]
#                 output[self.pH_comp_number(pH_value) - 1] = pH_value
#                 output[self.pH_comp_number(pH_value) - 1 + 3] = frequencies[
#                     indexes_not_nan
#                 ]
#                 output[self.pH_comp_number(pH_value) - 1 + 6] = amplitudes[
#                     indexes_not_nan
#                 ]

#             elif indexes_not_nan.size == 2:

#                 if self.pH_comp_number(
#                     pH_values[indexes_not_nan[1]]
#                 ) == self.pH_comp_number(pH_values[indexes_not_nan[0]]):

#                     if (
#                         self.pH_comp_number(pH_values[indexes_not_nan[1]]) == 1
#                         or self.pH_comp_number(pH_values[indexes_not_nan[0]]) == 2
#                     ):
#                         output[
#                             self.pH_comp_number(pH_values[indexes_not_nan[0]]) - 1
#                         ] = pH_values[indexes_not_nan[0]]
#                         output[self.pH_comp_number(pH_values[indexes_not_nan[1]])] = (
#                             pH_values[indexes_not_nan[1]]
#                         )

#                         output[
#                             self.pH_comp_number(pH_values[indexes_not_nan[0]]) - 1 + 3
#                         ] = frequencies[indexes_not_nan[0]]
#                         output[
#                             self.pH_comp_number(pH_values[indexes_not_nan[1]]) + 3
#                         ] = frequencies[indexes_not_nan[1]]

#                         output[
#                             self.pH_comp_number(pH_values[indexes_not_nan[0]]) - 1 + 6
#                         ] = amplitudes[indexes_not_nan[0]]
#                         output[
#                             self.pH_comp_number(pH_values[indexes_not_nan[1]]) + 6
#                         ] = amplitudes[indexes_not_nan[1]]
#                     elif self.pH_comp_number(pH_values[indexes_not_nan[0]]) == 3:
#                         output[
#                             self.pH_comp_number(pH_values[indexes_not_nan[0]]) - 2
#                         ] = pH_values[indexes_not_nan[0]]
#                         output[
#                             self.pH_comp_number(pH_values[indexes_not_nan[1]]) - 1
#                         ] = pH_values[indexes_not_nan[1]]

#                         output[
#                             self.pH_comp_number(pH_values[indexes_not_nan[0]]) - 2 + 3
#                         ] = frequencies[indexes_not_nan[0]]
#                         output[
#                             self.pH_comp_number(pH_values[indexes_not_nan[1]]) - 1 + 3
#                         ] = frequencies[indexes_not_nan[1]]

#                         output[
#                             self.pH_comp_number(pH_values[indexes_not_nan[0]]) - 2 + 6
#                         ] = amplitudes[indexes_not_nan[0]]
#                         output[
#                             self.pH_comp_number(pH_values[indexes_not_nan[1]]) - 1 + 6
#                         ] = amplitudes[indexes_not_nan[1]]

#                 else:
#                     output[self.pH_comp_number(pH_values[indexes_not_nan[0]]) - 1] = (
#                         pH_values[indexes_not_nan[0]]
#                     )
#                     output[self.pH_comp_number(pH_values[indexes_not_nan[1]]) - 1] = (
#                         pH_values[indexes_not_nan[1]]
#                     )

#                     output[
#                         self.pH_comp_number(pH_values[indexes_not_nan[0]]) - 1 + 3
#                     ] = frequencies[indexes_not_nan[0]]
#                     output[
#                         self.pH_comp_number(pH_values[indexes_not_nan[1]]) - 1 + 3
#                     ] = frequencies[indexes_not_nan[1]]

#                     output[
#                         self.pH_comp_number(pH_values[indexes_not_nan[0]]) - 1 + 6
#                     ] = amplitudes[indexes_not_nan[0]]
#                     output[
#                         self.pH_comp_number(pH_values[indexes_not_nan[1]]) - 1 + 6
#                     ] = amplitudes[indexes_not_nan[1]]

#             return output

#         _, dim_read, dim_phase, dim_slice = pH_map.shape
#         index_list = list(np.ndindex(dim_read, dim_phase, dim_slice))

#         # serial solve for mutliple voxels in parallel
#         solution_serial = Parallel(n_jobs=4)(
#             delayed(sort_compartments)(idx, pH_map) for idx in index_list
#         )

#         # for debugging uncomment
#         # for idx in index_list:
#         #   solution_serial = sort_compartments(idx, pH_map)
#         #    print(solution_serial)

#         sorted_solution = np.array(solution_serial).reshape(
#             dim_read, dim_phase, dim_slice, -1
#         )

#         self._multi_comp_pHmaps = np.moveaxis(
#             np.real(sorted_solution[:, :, :, :3]), -1, 0
#         )

#         self._multi_comp_fmaps = np.moveaxis(
#             np.real(sorted_solution[:, :, :, 3:6]), -1, 0
#         )

#         self._multi_comp_amplitudes = np.moveaxis(sorted_solution[:, :, :, 6:9], -1, 0)

#     def pH_comp_number(self, pH_value):

#         pH_compartmen_thr = np.array([6.8, 7.15])

#         if pH_value <= pH_compartmen_thr[0]:
#             return 1
#         elif pH_value > pH_compartmen_thr[0] and pH_value <= pH_compartmen_thr[1]:
#             return 2
#         elif pH_value > pH_compartmen_thr[1]:
#             return 3

#     @onlycallonce
#     def apply_frequency_offset(self, offset):
#         """Adds a given constant offset onto the frequency map."""
#         self._fmap += offset
#         self._multi_comp_fmaps += offset
#         self._f_offset = offset

#     def mask_abs_error(self, abs_error_thr=15):
#         """Mask array by a absolute fit error threshold."""
#         self.mask_dict["abs_error"] = self._fmap_err > abs_error_thr

#     @onlycallonce
#     def apply_1H_ref_map(self, ref_fmap, method="resize", scalefor13c=True):
#         """Substracts given reference from fmap. Resizes reference map if necessray."""
#         if scalefor13c:
#             gyro_ratios = LAMOR_FREQUENCY_H1 / LAMOR_FREQUENCY_C13
#         else:
#             gyro_ratios = 1

#         print(f"Using gyromagnetic ratio of {gyro_ratios:.2f}")
#         if ref_fmap.shape != self.fmap.shape:
#             if method == "resize":
#                 logger.warning(f"Resizing... {ref_fmap.shape} -> {self.fmap.shape}")
#                 ref_fmap = interpolate_masked_3d_array(ref_fmap, self.fmap.shape)
#             elif method == "downsample":
#                 logger.warning(f"Downsampling... {ref_fmap.shape} -> {self.fmap.shape}")
#                 down_factors = np.array(ref_fmap.shape) / np.array(self.fmap.shape)
#                 for d in down_factors:
#                     if not float(d).is_integer():
#                         assert False, f"Downsampling factors: {down_factors}"
#                 ref_fmap = downsample_masked_array(
#                     ref_fmap, tuple(down_factors.astype(int))
#                 )
#             else:
#                 raise ValueError("Not a valid Method.")

#         if hasattr(ref_fmap, "mask") and ref_fmap.mask.any():
#             logger.warning(f"Replacing masked values with zero...")
#             ref_fmap = ref_fmap.filled(0)

#         self._fmap -= ref_fmap / gyro_ratios

#         return ref_fmap / gyro_ratios

#     @onlycallonce
#     def apply_1H_ref_map_multi_comp(self, ref_fmap, method="resize", scalefor13c=True):
#         """Substracts given reference from fmap. Resizes reference map if necessray."""
#         if scalefor13c:
#             gyro_ratios = LAMOR_FREQUENCY_H1 / LAMOR_FREQUENCY_C13
#         else:
#             gyro_ratios = 1

#         print(f"Using gyromagnetic ratio of {gyro_ratios:.2f}")
#         if ref_fmap.shape != self.fmap.shape:
#             if method == "resize":
#                 logger.warning(f"Resizing... {ref_fmap.shape} -> {self.fmap.shape}")
#                 ref_fmap = interpolate_masked_3d_array(ref_fmap, self.fmap.shape)
#             elif method == "downsample":
#                 logger.warning(f"Downsampling... {ref_fmap.shape} -> {self.fmap.shape}")
#                 down_factors = np.array(ref_fmap.shape) / np.array(self.fmap.shape)
#                 for d in down_factors:
#                     if not float(d).is_integer():
#                         assert False, f"Downsampling factors: {down_factors}"
#                 ref_fmap = downsample_masked_array(
#                     ref_fmap, tuple(down_factors.astype(int))
#                 )
#             else:
#                 raise ValueError("Not a valid Method.")

#         if hasattr(ref_fmap, "mask") and ref_fmap.mask.any():
#             logger.warning(f"Replacing masked values with zero...")
#             ref_fmap = ref_fmap.filled(0)

#         self._multi_comp_fmaps -= ref_fmap / gyro_ratios

#         return ref_fmap / gyro_ratios

#     def plot_results(self, param_dict={}, anatref=None, old_data=False):
#         """Warpper for plot_results2D and plot_results3D."""
#         if old_data:
#             if self.Nslices == 1:
#                 return UmpirePlotter2D_old(self, parameters=param_dict)
#             else:
#                 return UmpirePlotter3D_old(self, parameters=param_dict, anatref=anatref)
#         else:

#             if self.Nslices == 1:
#                 return UmpirePlotter2D(self, parameters=param_dict)
#             else:
#                 return UmpirePlotter3D(self, parameters=param_dict, anatref=anatref)

#     def plot_spectrum(self, cut_off=0):

#         spectra, f_axis = self.spectral_data
#         plot_spectrum(spectra, f_axis)

#     @classmethod
#     def load(cls, filename):
#         """
#         Load an umpireMGE-object from a file and display the timestamp of the
#         saved object.

#         Parameters
#         ----------
#         filename: str
#             The name of the file to load the object from.

#         Returns
#         -------
#         umpireMGE:
#             The loaded object.

#         """
#         with open(filename, "rb") as f:
#             tmp = pickle.load(f)
#         print(f"Loading object saved on {tmp._timestamp}.")
#         return tmp

#     def save(self, filename):
#         """
#         Save the umpireMGE-object to a file, updating the timestamp and
#         displaying information about the saved object.

#         Parameters
#         ----------
#         filename: str
#             The name of the file to save the object to.

#         """
#         import datetime
#         import os

#         if filename.split(".")[-1] != "pickle":
#             filename += ".pickle"

#         self._filename = filename
#         self._timestamp = datetime.datetime.now()

#         with open(filename, "wb") as f:
#             try:
#                 pickle.dump(self, f, 4)
#             except pickle.PicklingError as perr:
#                 print(
#                     "Note: Sometimes pickle can conflict with the "
#                     "%autoreload extension.\n      If you altered a class "
#                     "method, try running 'save()' again or restart you kernel.\n"
#                 )
#                 raise perr

#         nbytes = os.path.getsize(filename)

#         if nbytes < 1e6:
#             sz = f"{nbytes * 1e-3:.2f} KB"
#         elif nbytes < 1e9:
#             sz = f"{nbytes * 1e-6:.2f} MB"
#         else:
#             sz = f"{nbytes * 1e-9:.2f} GB"

#         print(f"Object saved to '{filename}' with a size of {sz}.")
