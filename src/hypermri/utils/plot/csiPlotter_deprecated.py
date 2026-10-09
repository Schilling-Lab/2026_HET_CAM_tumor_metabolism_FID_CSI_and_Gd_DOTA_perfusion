# This just serves as a backup and will be removed in the future

# class CSIPlotter:
#     def calc_roi(self, key=0, extent=None, axial_image_shape=None):
#         """
#         Calculate the drawn ROI

#         Returns
#             _type_: _description_
#         """
#         from hypermri.utils.utils_general import translate_coordinates

#         if "mask_anat_x_indices" not in self.ROI[key]:
#             self.ROI[key]["mask_anat_x_indices"] = []
#             self.ROI[key]["mask_anat_y_indices"] = []

#         self.ROI[key]["mask_anat_x_indices"] = extent[0]
#         self.ROI[key]["mask_anat_y_indices"] = extent[1]

#         # translate coordinates into integers:
#         x_coordinates = translate_coordinates(
#             original_coords=self.ROI[key]["mask_anat_coords"][0],
#             fov=extent[0],
#             matrix_size=axial_image_shape[0],
#         )

#         y_coordinates = translate_coordinates(
#             original_coords=self.ROI[key]["mask_anat_coords"][1],
#             fov=extent[1],
#             matrix_size=axial_image_shape[1],
#         )
#         self.ROI[key]["mask_anat_x_indices"] = x_coordinates
#         self.ROI[key]["mask_anat_y_indices"] = y_coordinates
#         vertices = np.array(
#             [[[x, y] for x, y in zip(y_coordinates, x_coordinates)]], dtype=np.int32
#         )

#         # Create an empty mask with the same dimensions as the target image
#         mask = np.zeros((axial_image_shape), dtype=np.uint8)

#         # Fill the polygon in the mask
#         self.ROI[key]["mask_anat"] = np.fliplr(cv2.fillPoly(mask, vertices, 1))
#         self.ROI[key]["mask_csi"] = np.fliplr(
#             cv2.resize(
#                 self.ROI[key]["mask_anat"],
#                 [self.method["PVM_Matrix"][1], self.method["PVM_Matrix"][0]],
#             )
#         )
#         return True

#     def plot2d(
#         self,
#         axlist,
#         axial_image=None,
#         display_ui=True,
#         csi_data=None,
#         csi_fit_data=None,
#         fig=None,
#         plot_params=None,
#         save_fig=False,
#     ):
#         """
#         Plots a 2D Chemical Shift Imaging (CSI) dataset, optionally overlaid onto an anatomical image.
#         This function provides a comprehensive interface for visualization adjustments including
#         data interpolation, display settings, and dynamic plotting through interactive UI elements in Jupyter.

#         Parameters
#         ----------
#         axlist : list of matplotlib.axes.Axes
#             List of Axes objects where the plots will be drawn.
#         axial_image : hypermri.sequence object or ndarray, optional
#             An object containing the anatomical image data or a 6D numpy array (same dimensions as `csi_data`)
#             to be used as a background for overlays. If None, no anatomical image is displayed.
#         display_ui : bool, optional
#             If True (default), display interactive UI elements for dynamic plotting adjustments.
#         csi_data : ndarray, optional
#             6D array representing the CSI data to be plotted. If None, internal CSI data will be used.
#             Dimensions should match those of `axial_image` if it is a numpy array.
#         csi_fit_data : ndarray, optional
#             6D array of CSI fitted data to be plotted alongside the original data for comparison.
#             This array should have the same dimensions as `csi_data`.
#         fig : matplotlib.figure.Figure, optional
#             Figure object where the plots will be drawn. If None, the current figure will be used.
#         plot_params : dict, optional
#             Dictionary containing plot parameters for customization of the plot appearances such as color maps and overlays.
#             Can be used to regenerated images.
#         save_fig : bool, optional
#             If True, the generated plot will be saved to file. This requires 'fig' and 'plot_params' to be set.

#         Returns
#         -------
#         None or object
#             The function does not return any values but renders matplotlib plots directly to the provided Axes objects.
#             If 'display_ui' is False, it returns configured UI elements for external handling.


#         Example
#         -------
#         >>> csi_scan.reorient_anat(anatomical_obj_ax=flash);
#         >>> fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(5,5))
#         >>> csi_scan.plot2d(axlist=(ax1, ax2),
#         >>>     axial_image=flash,
#         >>>     display_ui=True,
#         >>>     csi_data=csi_scan.reco_data, # or csi_scan.reco_data
#         >>>     fig=fig,
#         >>>     plot_params=None,
#         >>>     save_fig=False)

#         """
#         # FIXME metab_ratio produces an error when Fum/Mal is selected
#         # FIXME saving of plots does not work intuitively
#         # FIXME documentation is missing

#         # check if there is data to plot
#         if csi_data is None:
#             if not isinstance(self.reco_data, np.ndarray):
#                 debug_log(
#                     "There is no reconstructed data available. "
#                     + "Please call the function bSSFP.reconstruction first!"
#                     + "--> Aborting"
#                 )
#                 return False
#             else:
#                 # get reconstructed data:
#                 data = self.reco_data
#         else:
#             # use input data:
#             data = csi_data

#         if np.ndim(csi_fit_data) == 7:
#             csi_fit_data = np.sum(csi_fit_data, axis=-1, keepdims=True)

#         # define CSI parameters:
#         # ------------------------------------------------------------
#         self.data_to_plot = data
#         self.org_data_to_plot = data
#         nechoes_csi = data.shape[0]
#         reps_csi = data.shape[4]
#         chans_csi = data.shape[5]

#         mm_read_csi, mm_phase_csi, mm_slice_csi = define_imageFOV_parameters(
#             data_obj=self
#         )
#         dim_read_csi, dim_phase_csi, dim_slice_csi = define_imagematrix_parameters(
#             data_obj=self
#         )

#         # check if csi_data is interpolated compared to the raw data
#         if (np.array(csi_data.shape) == np.array(self.reco_data.shape)).all() == False:
#             debug_log(
#                 "csi_data.shape does not match self.spatdom_spectra_xarray.shape. Assuming input data was interpolated."
#             )
#             if self.reco_data is not None:
#                 interpolation_matrix = np.array(
#                     np.array(csi_data.shape) / np.array(self.reco_data.shape),
#                     dtype=int,
#                 )
#                 read_itpl_factor = interpolation_matrix[1]
#                 phase_itpl_factor = interpolation_matrix[2]
#                 slice_itpl_factor = interpolation_matrix[3]
#                 echo_spectral_itpl_factor = interpolation_matrix[0]

#                 debug_log(
#                     "Changed CSI matrix size according to interpolation matrix: "
#                     + str(interpolation_matrix)
#                 )
#                 debug_log(
#                     "dim_read_csi changed to " + str(dim_read_csi * read_itpl_factor)
#                 )
#                 debug_log(
#                     "dim_phase_csi changed to " + str(dim_phase_csi * phase_itpl_factor)
#                 )
#                 debug_log(
#                     "dim_slice_csi changed to " + str(dim_slice_csi * slice_itpl_factor)
#                 )
#                 debug_log(
#                     "nechoes_csi changed to "
#                     + str(nechoes_csi * echo_spectral_itpl_factor)
#                 )
#                 debug_log(
#                     "Changed CSI matrix size according to interpolation matrix: "
#                     + str(interpolation_matrix)
#                 )
#                 logger.warning("dim_read_csi changed to " + str(dim_read_csi))
#                 logger.warning("dim_phase_csi changed to " + str(dim_phase_csi))
#                 logger.warning("dim_slice_csi changed to " + str(dim_slice_csi))
#                 logger.warning("nechoes_csi changed to " + str(nechoes_csi))
#             elif self.seq2d_reordered:
#                 interpolation_matrix = np.array(csi_data.shape) / np.array(
#                     self.seq2d_reordered.shape
#                 )
#                 read_itpl_factor = interpolation_matrix[1]
#                 phase_itpl_factor = interpolation_matrix[2]
#                 slice_itpl_factor = interpolation_matrix[3]
#                 echo_spectral_itpl_factor = interpolation_matrix[0]
#                 debug_log(
#                     "Changed CSI shape according to interpolation matrix: "
#                     + str(interpolation_matrix)
#                 )
#                 debug_log(
#                     "dim_read_csi changed to " + str(dim_read_csi * read_itpl_factor)
#                 )
#                 debug_log(
#                     "dim_phase_csi changed to " + str(dim_phase_csi * phase_itpl_factor)
#                 )
#                 debug_log(
#                     "dim_slice_csi changed to " + str(dim_slice_csi * slice_itpl_factor)
#                 )
#                 debug_log(
#                     "nechoes_csi changed to "
#                     + str(nechoes_csi * echo_spectral_itpl_factor)
#                 )
#                 debug_log(
#                     "Changed CSI shape according to interpolation matrix: "
#                     + str(interpolation_matrix)
#                 )
#                 logger.warning("dim_read_csi changed to " + str(dim_read_csi))
#                 logger.warning("dim_phase_csi changed to " + str(dim_phase_csi))
#                 logger.warning("dim_slice_csi changed to " + str(dim_slice_csi))
#                 logger.warning("nechoes_csi changed to " + str(nechoes_csi))
#             else:
#                 interpolation_matrix = np.ones_like(np.array(csi_data.shape))
#                 read_itpl_factor = 1
#                 phase_itpl_factor = 1
#                 slice_itpl_factor = 1
#                 echo_spectral_itpl_factor = 1
#                 debug_log(
#                     "Could not find self.spatdom_spectra_xarray or self.seq2d_reordered to determine if csi_data was interpolated beforehand."
#                 )
#         else:
#             interpolation_matrix = np.ones_like(np.array(csi_data.shape))
#             read_itpl_factor = 1
#             phase_itpl_factor = 1
#             slice_itpl_factor = 1
#             echo_spectral_itpl_factor = 1

#         dim_read_csi, dim_phase_csi, dim_slice_csi, nechoes_csi = (
#             dim_read_csi * read_itpl_factor,
#             dim_phase_csi * phase_itpl_factor,
#             dim_slice_csi * slice_itpl_factor,
#             nechoes_csi * echo_spectral_itpl_factor,
#         )

#         read_orient_csi = self.method["PVM_SPackArrReadOrient"]
#         # read_offset_csi = self.method["PVM_SPackArrReadOffset"]
#         # phase_offset_csi = self.method["PVM_SPackArrPhase1Offset"]
#         # slice_offset_csi = self.method["PVM_SPackArrSliceOffset"]

#         csi_grid = define_grid(
#             mat=np.array((dim_read_csi, dim_phase_csi, dim_slice_csi)),
#             fov=np.array((mm_read_csi, mm_phase_csi, mm_slice_csi)),
#         )

#         # define axial image parameters:
#         # ------------------------------------------------------------
#         # No Data - No Overlay:
#         if axial_image is None:
#             pass
#         # object contains no info about extent, assume same dimensions:
#         elif isinstance(axial_image, np.ndarray):
#             pass
#         # object contains info about extent:
#         elif axial_image.__module__.startswith("hypermri."):
#             mm_read_ax, mm_phase_ax, mm_slice_ax = define_imageFOV_parameters(
#                 data_obj=axial_image
#             )
#             (
#                 dim_anat_read_ax,
#                 dim_anat_phase_ax,
#                 dim_anat_slice_ax,
#             ) = define_imagematrix_parameters(data_obj=axial_image)

#             # read_orient_ax = self.method["PVM_SPackArrReadOrient"]
#             # read_offset_ax = self.method["PVM_SPackArrReadOffset"]
#             # phase_offset_ax = self.method["PVM_SPackArrPhase1Offset"]
#             # slice_offset_ax = self.method["PVM_SPackArrSliceOffset"]

#             # get grid describing the coronal slices: A_P
#             # ax_grid = define_grid(
#             #     mat=np.array((dim_anat_read_ax, dim_anat_phase_ax, dim_anat_slice_ax)),
#             #     fov=np.array((mm_read_ax, mm_phase_ax, mm_slice_ax)),
#             # )

#             axial_image.seq2d_oriented = shift_anat(
#                 csi_obj=self,
#                 anat_obj=axial_image,
#                 use_scipy_shift=True,
#             )

#             anat_image = axial_image.seq2d_oriented
#         else:
#             pass

#         # define axial image parameters:
#         # ------------------------------------------------------------
#         # No Data - No overlay:

#         # dim_read_plot = len(csi_grid[0])

#         # FIXME if no anatomical is loaded there is an error here
#         dim_read_anat = anat_image.shape[1]
#         dim_slice_plot = len(csi_grid[2])

#         # define plotting extents:
#         plotting_extent = get_plotting_extent(data_obj=self)
#         if (interpolation_matrix == 1).all():
#             self.plotting_grid = self.get_plotting_grid(csi_interpolation_matrix=None)
#         else:
#             self.plotting_grid = self.get_plotting_grid(
#                 csi_interpolation_matrix=interpolation_matrix
#             )

#         # define extent of datasets in space:
#         ax_csi_ext, sag_csi_ext, cor_csi_ext = get_extent(data_obj=self)
#         ax_anat_ext, sag_anat_ext, cor_anat_ext = get_extent(data_obj=axial_image)

#         ax_csi_ext_phase = np.linspace(
#             ax_csi_ext[0] - (ax_csi_ext[0] / self.data_to_plot.shape[2]),
#             ax_csi_ext[1] - (ax_csi_ext[1] / self.data_to_plot.shape[2]),
#             self.data_to_plot.shape[2],
#         )

#         ax_csi_ext_slice = np.linspace(
#             ax_csi_ext[2] - (ax_csi_ext[2] / self.data_to_plot.shape[3]),
#             ax_csi_ext[3] - (ax_csi_ext[3] / self.data_to_plot.shape[3]),
#             self.data_to_plot.shape[3],
#         )

#         # for checking if the axial slice is still in the CSI slice range:
#         cor_anat_range = np.linspace(cor_anat_ext[2], cor_anat_ext[3], dim_anat_read_ax)

#         if not read_orient_csi == "H_F":
#             # checking if we have the right read orientation
#             # if this error comes up you have to implement different
#             # orientations possibly
#             logger.debug(
#                 "Careful this function is not evaluated for this read orientation"
#             )
#         else:
#             pass

#         # get time axis
#         time_axis = calc_timeaxis(data_obj=self, start_with_0=True)

#         if plot_params is not None:
#             try:
#                 plot_params = load_plot_params(param_file=plot_params, data_obj=self)
#             except BaseException:
#                 plot_params = {}
#             if "fig_name" in plot_params:
#                 fig_name = plot_params["fig_name"]
#             else:
#                 fig_name = None

#         # define image plotting function
#         def plot_img(
#             slice_nr_ax=0,
#             slice_nr_cor=0,
#             slice_nr_sag=0,
#             freq_start=0,
#             freq_end=0,
#             rep1=0,
#             chan=0,
#             alpha_overlay=0.5,
#             rep2=None,
#             rep_avg=False,
#             freq_avg=False,
#             plot_style="Abs",
#             experiment_style="Spectropic",
#             anat_overlay="Metab",
#             metab_clim=[0, 1],
#             cmap="plasma",
#             interp_method="none",
#             interp_factor=1,
#             freq_units="ppm",
#             background=0.0,
#             lb=0.0,
#             freq_1=0.0,
#             freq_2=0.0,
#             freq_3=0.0,
#             freq_4=0.0,
#             ROI=0,
#             metab_ratio="1 (1 Metab)",
#             zero_as_Nan=False,
#         ):
#             """
#             This is the plotting function that is called whenever one of it's
#             parameters is changed:
#             Parameters
#             slice_nr_ax: axial slice number
#             slice_nr_cor: coronal slice number
#             slice_nr_sag: sagtial slice number (often not used)
#             freq_start: echo number (for me-bSSFP)
#             rep1: repetition that is shown. if rep_avg = True, start of range that is averaged (see rep2)
#             chan: Channel (for multichannel coil experiments)
#             alpha_overlay: the intenstiy of the overlay of the metabolic images onto the anatomical (0-1)
#             rep2: if rep_avg==True, this is the end of the range that is averaged (see rep1)
#             rep_avg: if True, the range from rep1 to rep2 is averaged
#             plot_style: can be magnitude, real, imaginary of phase
#             pyr_lac: which metabolite to show (default pyruvate was acquired first
#             anat_overlay: Can be metabolite only, overlay or anatomical only
#             metab_clim: range of values that is displayed (smaller are set to NaN)
#             cmap: colormap of metabolic images
#             interp_method: interpolation method with which the images are displayed
#             background: background signal that will be subtracted
#             """
#             # init range:
#             rep_range = rep1
#             # get frequency axis:
#             # if freq_units == "ppm":
#             #     freq_range = self.get_ppm()
#             # elif freq_units == "Hz":
#             #     freq_range = self.get_Hz()
#             # else:
#             #     freq_range = np.linspace(
#             #         0, self.data_to_plot.shape[0] - 1, self.data_to_plot.shape[0]
#             #     )

#             freq_range = self.freq_range

#             # generate a custom cmap:
#             # map_name, cmap = self.generate_custom_cmap(cmap_name=cmap)

#             # clear axis:
#             for ax in axlist:
#                 ax.clear()

#             # to avoid confusion as this is probably not universal (different read directions
#             # and patient_pos)... define this here so it can be made depending on those paramters:
#             # slice_nr_ax_csi = -(slice_nr_ax + 1)
#             slice_nr_ax_csi = slice_nr_ax

#             # parameters for interpolation:
#             interp_params = {}
#             interp_params["interp_method"] = interp_method
#             interp_params["interp_factor"] = interp_factor
#             interp_params["interp_cmap"] = cmap
#             interp_params["interp_threshold"] = metab_clim[0]

#             # get metabolic image:
#             metab_image = self.prep_metab_image_plot2d(
#                 data=None,
#                 plot_style=plot_style,
#                 rep_avg=rep_avg,
#                 rep1=rep1,
#                 rep2=rep2,
#                 freq_units=freq_units,
#                 freq_avg=freq_avg,
#                 freq_start=freq_start,
#                 freq_end=freq_end,
#                 slice_nr_ax_csi=slice_nr_ax_csi,
#                 background=background,
#                 chan=chan,
#                 metab_ratio=metab_ratio,
#                 freqs=[freq_1, freq_2, freq_3, freq_4],
#                 interp_params=interp_params,
#             )

#             # get spectrum:
#             csi_spec = self.prep_spectrum_plot2d(
#                 data=None,
#                 plot_style=plot_style,
#                 rep_avg=rep_avg,
#                 rep1=rep1,
#                 rep2=rep2,
#                 freq_units=freq_units,
#                 freq_avg=freq_avg,
#                 freq_start=freq_start,
#                 freq_end=freq_end,
#                 ix=slice_nr_sag,
#                 iy=slice_nr_cor,
#                 slice_nr_ax_csi=slice_nr_ax_csi,
#                 background=background,
#                 chan=chan,
#                 current_key=rangeslider_choose_ROI.value,
#             )

#             # get spectrum:
#             csi_fit_spec = self.prep_spectrum_plot2d(
#                 data=csi_fit_data,
#                 plot_style=plot_style,
#                 rep_avg=rep_avg,
#                 rep1=rep1,
#                 rep2=rep2,
#                 freq_units=freq_units,
#                 freq_avg=freq_avg,
#                 freq_start=freq_start,
#                 freq_end=freq_end,
#                 ix=slice_nr_sag,
#                 iy=slice_nr_cor,
#                 slice_nr_ax_csi=slice_nr_ax_csi,
#                 background=background,
#                 chan=chan,
#                 current_key=rangeslider_choose_ROI.value,
#             )

#             # get time curve:
#             freq_range_temp = freq_end - freq_start
#             csi_time = []
#             csi_time.append(
#                 self.prep_timecurve_plot2d(
#                     data=None,
#                     plot_style=plot_style,
#                     rep_avg=rep_avg,
#                     rep1=rep1,
#                     rep2=rep2,
#                     freq_units=freq_units,
#                     freq_avg=freq_avg,
#                     freq_start=freq_start,
#                     freq_end=freq_end,
#                     ix=slice_nr_sag,
#                     iy=slice_nr_cor,
#                     slice_nr_ax_csi=slice_nr_ax_csi,
#                     background=background,
#                     chan=chan,
#                 )
#             )

#             csi_fit_time = []
#             csi_fit_time.append(
#                 self.prep_timecurve_plot2d(
#                     data=csi_fit_data,
#                     plot_style=plot_style,
#                     rep_avg=rep_avg,
#                     rep1=rep1,
#                     rep2=rep2,
#                     freq_units=freq_units,
#                     freq_avg=freq_avg,
#                     freq_start=freq_start,
#                     freq_end=freq_end,
#                     ix=slice_nr_sag,
#                     iy=slice_nr_cor,
#                     slice_nr_ax_csi=slice_nr_ax_csi,
#                     background=background,
#                     chan=chan,
#                 )
#             )

#             text_freq_values = [
#                 text_freq_1.value,
#                 text_freq_2.value,
#                 text_freq_3.value,
#                 text_freq_4.value,
#             ]
#             for k in range(4):
#                 csi_time.append(
#                     self.prep_timecurve_plot2d(
#                         data=None,
#                         plot_style=plot_style,
#                         rep_avg=rep_avg,
#                         rep1=rep1,
#                         rep2=rep2,
#                         freq_units=freq_units,
#                         freq_avg=freq_avg,
#                         freq_start=text_freq_values[k] - freq_range_temp / 2,
#                         freq_end=text_freq_values[k] + freq_range_temp / 2,
#                         ix=slice_nr_sag,
#                         iy=slice_nr_cor,
#                         slice_nr_ax_csi=slice_nr_ax_csi,
#                         background=background,
#                         chan=chan,
#                     )
#                 )
#                 csi_fit_time.append(
#                     self.prep_timecurve_plot2d(
#                         data=csi_fit_data,
#                         plot_style=plot_style,
#                         rep_avg=rep_avg,
#                         rep1=rep1,
#                         rep2=rep2,
#                         freq_units=freq_units,
#                         freq_avg=freq_avg,
#                         freq_start=freq_start,
#                         freq_end=freq_end,
#                         ix=slice_nr_sag,
#                         iy=slice_nr_cor,
#                         slice_nr_ax_csi=slice_nr_ax_csi,
#                         background=background,
#                         chan=chan,
#                     )
#                 )

#             # find the points in the axial and coronal that are closest to the
#             # bssfp location:
#             # ax_slice_ind = sag_slice_ind = cor_slice_ind = None

#             ax_slice_ind = slice_nr_ax_csi
#             mask_ax = np.ones((self.data_to_plot.shape[1], self.data_to_plot.shape[2]))

#             # metabolic axial image:
#             metab_image = np.rot90(np.squeeze((metab_image)))

#             # if the colorange shoud be set automcatically:
#             if checkbox_auto_set_crange.value:
#                 # stupid but should do the trick for now:
#                 metab_clim = list(metab_clim)
#                 metab_clim[0] = np.nanmin(metab_image)
#                 metab_clim[1] = np.nanmax(metab_image)

#                 # don't show negative number in case of magntiude display
#                 # style:
#                 if plot_style == "Abs":
#                     metab_clim[0] = 0
#                 metab_clim = tuple(metab_clim)

#             # axial view: -----------------------------------------------------
#             if (anat_overlay == "Metab + Anat") and (anat_image is not None):
#                 # anatomical axial image:
#                 axlist[0].imshow(
#                     np.squeeze(np.rot90(anat_image[0, ax_slice_ind, :, :, 0, 0])),
#                     extent=plotting_extent,
#                     cmap="bone",
#                 )

#                 im_ax = axlist[0].imshow(
#                     metab_image,
#                     extent=plotting_extent,
#                     cmap=cmap,
#                     alpha=alpha_overlay,
#                     vmin=metab_clim[0],
#                     vmax=metab_clim[1],
#                 )

#                 # only plot ROI if checkbox_draw_roi.value is True:
#                 if (
#                     rangeslider_choose_ROI.value in self.ROI
#                     and "mask_anat_coords" in self.ROI[rangeslider_choose_ROI.value]
#                     and self.ROI[rangeslider_choose_ROI.value]["mask_anat_coords"][0]
#                 ):
#                     axlist[0].plot(
#                         self.ROI[rangeslider_choose_ROI.value]["mask_anat_coords"][0],
#                         self.ROI[rangeslider_choose_ROI.value]["mask_anat_coords"][1],
#                         color=self.ROI[rangeslider_choose_ROI.value]["color"],
#                     )

#             elif anat_overlay == "Metab":
#                 # metabolic axial image:
#                 im_ax = axlist[0].imshow(
#                     metab_image,
#                     extent=plotting_extent,
#                     cmap=cmap,
#                     vmin=metab_clim[0],
#                     vmax=metab_clim[1],
#                 )

#             elif (anat_overlay == "Anat") and (anat_image is not None):
#                 # anatomical axial image:
#                 axlist[0].imshow(
#                     np.squeeze(np.rot90(anat_image[0, ax_slice_ind, :, :, 0, 0])),
#                     extent=plotting_extent,
#                     cmap="bone",
#                 )

#             else:  # default show metab image
#                 im_ax = axlist[0].imshow(
#                     metab_image,
#                     extent=plotting_extent,
#                     cmap=cmap,
#                     alpha=alpha_overlay * mask_ax,
#                     vmin=metab_clim[0],
#                     vmax=metab_clim[1],
#                 )

#             # generate colorbar if fig object was passed:
#             if fig is None:
#                 # colorbar needs figure:
#                 pass

#             elif (fig is not None) and (anat_overlay != "Anat"):
#                 # nasty but otherwise plots new colorbar each time a paramter
#                 # is changed :/
#                 global cbar
#                 try:
#                     cbar.remove()
#                 except BaseException:
#                     pass
#                 cbar = fig.colorbar(im_ax, ax=axlist[0], fraction=0.033, pad=0.04)
#                 cbar.remove()
#                 cbar = fig.colorbar(
#                     im_ax,
#                     ax=axlist[0],
#                     fraction=0.033,
#                     pad=0.04,
#                 )
#                 # set colorbar range, use 3 ticks (min, mean, max)
#                 md = (metab_clim[1] + metab_clim[0]) / 2
#                 cbar.set_ticks([metab_clim[0], md, metab_clim[1]])
#                 cbar.set_ticklabels(
#                     np.round([metab_clim[0], md, metab_clim[1]]).astype(int)
#                 )
#             else:
#                 # dont plot colorbar if no metabolic data:
#                 try:
#                     # skip for now because this changes the size of the plotted images
#                     # which can be quite annoying:
#                     pass
#                     # remove if left over from previos plot:
#                     # cbar.remove()
#                 except BaseException:
#                     pass

#             # plot lines:
#             axlist[0].axhline(
#                 ax_csi_ext_slice[slice_nr_cor],
#                 color="white",
#                 linewidth=0.25,
#                 linestyle="solid",
#             )
#             axlist[0].axvline(
#                 ax_csi_ext_phase[
#                     slice_nr_sag
#                 ],  # mm_phase_csi / 2 - slice_nr_sag * slice_thick_sag_csi,
#                 color="white",
#                 linewidth=0.25,
#                 linestyle="dashed",
#             )

#             # plot labels:
#             axlist[0].set_xlabel("mm (read)")
#             axlist[0].set_ylabel("mm (phase)")

#             if cor_csi_ext[2] <= cor_anat_range[slice_nr_ax] <= cor_csi_ext[3]:
#                 axlist[0].set_title("Axial", color="black")
#             else:
#                 axlist[0].set_title("Axial", color="red")

#             # spectral view: -------------------------------------------
#             if len(axlist) > 1:
#                 axlist[1].plot(
#                     freq_range, csi_spec, linewidth=1, color="black", label="Data"
#                 )
#                 # add fitted spectrum to plot
#                 if csi_fit_data is not None:
#                     axlist[1].plot(
#                         freq_range,
#                         csi_fit_spec,
#                         linewidth=1,
#                         color="red",
#                         label="Fit",
#                     )
#                 axlist[1].axvline(
#                     freq_start,
#                     color="black",
#                     linewidth=1,
#                     linestyle="dashed",
#                 )
#                 axlist[1].axhline(
#                     0,
#                     color="black",
#                     linewidth=0.25,
#                     linestyle="dashed",
#                 )
#                 axlist[1].legend()
#                 # draw 2nd line:
#                 if (freq_end > freq_start) and (freq_avg_checkbox.value):
#                     axlist[1].axvline(
#                         freq_end,
#                         color="black",
#                         linewidth=1,
#                         linestyle="dashed",
#                     )
#                 elif (freq_end > freq_start) and (freq_avg_checkbox.value is False):
#                     axlist[1].axvline(
#                         freq_end,
#                         color="black",
#                         linewidth=0.25,
#                         linestyle="dashed",
#                     )

#                 # flip frequency axis if unit are ppm
#                 if plot_opts_freq_units.value == "ppm":
#                     axlist[1].invert_xaxis()
#                 else:
#                     pass

#                 if rangeslider_choose_ROI.value in self.ROI:
#                     if "mask_anat_coords" in self.ROI[rangeslider_choose_ROI.value]:
#                         axlist[1].set_title(
#                             self.ROI[rangeslider_choose_ROI.value]["name"]
#                         )
#                     else:
#                         axlist[1].set_title(str(slice_nr_sag) + " " + str(slice_nr_cor))
#                 else:
#                     axlist[1].set_title(str(slice_nr_sag) + " " + str(slice_nr_cor))

#                 axlist[1].set_xlabel(f"[{plot_opts_freq_units.value}]")
#                 freqs_on_off = [
#                     checkbox_freq_1.value,
#                     checkbox_freq_2.value,
#                     checkbox_freq_3.value,
#                     checkbox_freq_4.value,
#                 ]
#                 freqs_value = [
#                     text_freq_1.value,
#                     text_freq_2.value,
#                     text_freq_3.value,
#                     text_freq_4.value,
#                 ]
#                 for k in range(4):
#                     if freqs_on_off[k]:
#                         axlist[1].axvline(
#                             freqs_value[k],
#                             linewidth=1,
#                             color=rcParams["axes.prop_cycle"].by_key()["color"][k],
#                         )

#             # plot time curve ---------------------------------------------
#             if len(axlist) > 2:
#                 # axlist[2].plot(time_axis, csi_time, linewidth=1)
#                 axlist[2].plot(
#                     time_axis, csi_time[0], linewidth=1, color="black", label="Cursor"
#                 )
#                 for k, data_set in enumerate(csi_time[1::]):
#                     if freqs_on_off[k]:
#                         axlist[2].plot(
#                             time_axis,
#                             data_set,
#                             linewidth=1,
#                             color=rcParams["axes.prop_cycle"].by_key()["color"][k],
#                             label=f"freq {k}",
#                         )

#                     axlist[2].axvline(
#                         time_axis[rep1],
#                         color="black",
#                         linewidth=1,
#                         linestyle="dashed",
#                     )
#                     # draw 2nd line:
#                     if (rep2 > rep1) and (rep_avg_checkbox.value):
#                         axlist[2].axvline(
#                             time_axis[rep2],
#                             color="black",
#                             linewidth=1,
#                             linestyle="dashed",
#                         )
#                     elif (rep2 > rep1) and (rep_avg_checkbox.value is False):
#                         axlist[2].axvline(
#                             time_axis[rep2],
#                             color="black",
#                             linewidth=0.25,
#                             linestyle="dashed",
#                         )
#                     else:
#                         pass
#                     axlist[2].set_xlabel("t [s]")
#                     axlist[2].legend()
#                 if csi_fit_data is not None:
#                     axlist[2].plot(
#                         time_axis,
#                         csi_fit_time[0],
#                         linewidth=1,
#                         color="red",
#                         label="Fit",
#                     )

#             labelpad = 50  # Adjust this value as needed

#             # Function to format y-axis tick labels
#             # Function to format y-axis tick labels with fixed width
#             def format_y_tick_labels(ax, width=10):
#                 ticks = ax.get_yticks()
#                 formatted_labels = []
#                 for tick in ticks:
#                     if "e" in f"{tick:.1e}":
#                         formatted_label = f"{tick:.1e}"
#                     else:
#                         formatted_label = f"{tick:>{width}.4g}"
#                     formatted_labels.append(formatted_label)
#                 ax.set_yticklabels(formatted_labels, family="monospace")

#             # Apply custom formatting to each subplot
#             for ax in axlist[1:-1]:
#                 format_y_tick_labels(ax)

#         # --------------------------------------------------

#         global rep_avg_checkbox_option

#         # create interactive sliders for  slices in all dimensions
#         rep_avg_checkbox = widgets.Checkbox(
#             description="Avg. reps",
#             tooltip="Take average over repetitions",
#             layout=widgets.Layout(width="100px"),
#         )
#         # rep_avg_checkbox_option = widgets.VBox(layout=widgets.Layout(display="flex"))

#         freq_avg_checkbox = widgets.Checkbox(
#             description="Avg. freqs",
#             tooltip="Take average over frequency range",
#             layout=widgets.Layout(width="100px"),
#         )

#         slice_slider_rep1 = widgets.IntSlider(
#             value=0,
#             min=0,
#             max=reps_csi - 1,
#             description="Rep. Start: ",
#             # layout=widgets.Layout(width="25%"),
#         )
#         slice_slider_rep2 = widgets.IntSlider(
#             value=0,
#             min=0,
#             max=reps_csi - 1,
#             description="Rep. End: ",
#             # layout=widgets.Layout(width="25%"),
#         )

#         # set the minimum of the end reptition to the value of the start
#         # reptition
#         def rep_range_checker(args):
#             if slice_slider_rep1.value > slice_slider_rep2.value:
#                 slice_slider_rep2.value = slice_slider_rep1.value
#             else:
#                 pass
#             return True

#         slice_slider_rep1.observe(rep_range_checker, names="value")
#         slice_slider_rep2.observe(rep_range_checker, names="value")

#         # Different plot options:
#         # lets you choose the data shown:
#         plot_opts_part = widgets.Dropdown(
#             options=["Abs", "Real", "Imag", "Phase"],
#             value="Abs",
#             description="Plot style:",
#             disabled=False,
#             layout=widgets.Layout(width="20%"),
#         )

#         # lets you choose wether to show anatomical, metabolic or ratio:
#         plot_opts_anat_overlay = widgets.Dropdown(
#             options=["Metab", "Metab + Anat", "Anat"],
#             value="Metab + Anat",
#             description="Overlay",
#             disabled=False,
#             layout=widgets.Layout(width="20%"),
#         )

#         # lets you choose the overlay of the metabolic on the
#         # anatomical:
#         plot_opts_alpha_overlay = widgets.BoundedFloatText(
#             value=0.5,
#             min=0.0,
#             max=1.0,
#             step=0.1,
#             disabled=False,
#             description="alpha:",
#             continuous_update=False,
#             layout=widgets.Layout(width="10%"),
#         )

#         # set plot colorrange of metabolites:
#         plot_opts_metab_clim = widgets.FloatRangeSlider(
#             min=-10.0,
#             value=[0, np.max(data)],
#             max=np.max(data),
#             description="Crange:",
#             disabled=False,
#             continuous_update=False,
#             orientation="horizontal",
#             tooltip="Displayed colorrange",
#             readout=True,
#             readout_format=".1f",
#             layout=widgets.Layout(width="35%"),
#         )

#         # set frequency axis' unit:
#         plot_opts_freq_units = widgets.Dropdown(
#             options=["ppm", "Hz", "index"],
#             value="index",
#             description="Freq. unit:",
#             disabled=False,
#             layout=widgets.Layout(width="15%"),
#         )

#         # on frequney unit:
#         echo_slider_start = widgets.FloatSlider(
#             description="Freq. Start: ",
#             continuous_update=False,  # layout=widgets.Layout(width="200px")
#         )
#         echo_slider_end = widgets.FloatSlider(
#             description="Freq. End: ",
#             continuous_update=False,  # layout=widgets.Layout(width="200px")
#         )

#         def set_freq_slider(args):
#             # only valid if freq slider had value before:
#             if hasattr(args, "old"):
#                 freq_start_val_old = echo_slider_start.value
#                 freq_end_val_old = echo_slider_end.value
#                 # to get freq_range of previous setting:
#                 if args["old"] == "ppm":
#                     freq_range_old = get_freq_axis(
#                         data_obj=self, unit="ppm", npoints=self.data_to_plot.shape[0]
#                     )
#                     self.freq_unit = "ppm"
#                 # Hz frequency range:
#                 elif args["old"] == "Hz":
#                     freq_range_old = get_freq_axis(
#                         data_obj=self,
#                         unit="Hz",
#                         npoints=self.data_to_plot.shape[0],
#                     )
#                     self.freq_unit = "Hz"
#                 # index frequency rang:
#                 if args["old"] == "index":
#                     freq_range_old = np.linspace(0, nechoes_csi - 1, nechoes_csi)
#                     self.freq_unit = "index"
#             # use index rang:
#             else:
#                 args["old"] = args["new"]
#                 freq_start_val_old = 0
#                 freq_end_val_old = 1
#                 freq_range_old = np.linspace(0, nechoes_csi - 1, nechoes_csi)

#             # get index of previous settings:
#             echo_start_index = freq_to_index(
#                 freq=freq_start_val_old, freq_range=freq_range_old
#             )
#             echo_end_index = freq_to_index(
#                 freq=freq_end_val_old, freq_range=freq_range_old
#             )

#             ppm_axis = get_freq_axis(
#                 data_obj=self, unit="ppm", npoints=self.data_to_plot.shape[0]
#             )
#             hz_axis = get_freq_axis(
#                 data_obj=self, unit="Hz", npoints=self.data_to_plot.shape[0]
#             )

#             # set freq_range of new settings:
#             if args["new"] == "ppm":
#                 freq_range_new = ppm_axis
#             elif args["new"] == "Hz":
#                 freq_range_new = hz_axis
#             elif args["new"] == "index":
#                 freq_range_new = np.linspace(0, nechoes_csi - 1, nechoes_csi).astype(
#                     int
#                 )
#             else:
#                 pass

#             # this is quite messy but reliably changes the frequency range and
#             # keeps the set frequency values:
#             if args["new"] == args["old"]:
#                 echo_slider_end.max = freq_range_new[-1]
#                 echo_slider_end.min = freq_range_new[0]
#                 echo_slider_end.value = freq_range_new[echo_end_index]
#                 echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

#                 echo_slider_start.max = freq_range_new[-1]
#                 echo_slider_start.min = freq_range_new[0]
#                 echo_slider_start.value = freq_range_new[echo_start_index]
#                 echo_slider_start.step = freq_range_new[1] - freq_range_new[0]
#                 self.freq_range = freq_range_new
#                 return True

#             # set sliders accordingly to new settings:
#             # from "small" to "big"
#             if args["new"] == "Hz" and args["old"] in ("ppm", "index"):
#                 echo_slider_end.max = freq_range_new[-1]
#                 echo_slider_end.min = freq_range_new[0]
#                 echo_slider_end.value = freq_range_new[echo_end_index]
#                 echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

#                 echo_slider_start.max = freq_range_new[-1]
#                 echo_slider_start.min = freq_range_new[0]
#                 echo_slider_start.value = freq_range_new[echo_start_index]
#                 echo_slider_start.step = freq_range_new[1] - freq_range_new[0]

#             # "ppm bigger than number of echo index:"
#             if args["new"] == "ppm" and args["old"] == "index":
#                 if ppm_axis[-1] > nechoes_csi:
#                     echo_slider_end.max = freq_range_new[-1]
#                     echo_slider_end.value = freq_range_new[echo_end_index]
#                     echo_slider_end.min = freq_range_new[0]
#                     echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

#                     echo_slider_start.max = freq_range_new[-1]
#                     echo_slider_start.value = freq_range_new[echo_start_index]
#                     echo_slider_start.min = freq_range_new[0]
#                     echo_slider_start.step = freq_range_new[1] - freq_range_new[0]
#                 elif ppm_axis[-1] < nechoes_csi:
#                     echo_slider_end.value = freq_range_new[echo_end_index]
#                     echo_slider_end.max = freq_range_new[-1]
#                     echo_slider_end.min = freq_range_new[0]
#                     echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

#                     echo_slider_start.value = freq_range_new[echo_start_index]
#                     echo_slider_start.max = freq_range_new[-1]
#                     echo_slider_start.min = freq_range_new[0]
#                     echo_slider_start.step = freq_range_new[1] - freq_range_new[0]
#                 else:
#                     echo_slider_end.value = freq_range_new[echo_end_index]
#                     echo_slider_end.max = freq_range_new[-1]
#                     echo_slider_end.min = freq_range_new[0]
#                     echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

#                     echo_slider_start.value = freq_range_new[echo_start_index]
#                     echo_slider_start.max = freq_range_new[-1]
#                     echo_slider_start.min = freq_range_new[0]
#                     echo_slider_start.step = freq_range_new[1] - freq_range_new[0]

#             elif args["new"] == "ppm" and args["old"] == "Hz":
#                 echo_slider_start.value = freq_range_new[echo_start_index]
#                 echo_slider_start.min = freq_range_new[0]
#                 echo_slider_start.max = freq_range_new[-1]
#                 echo_slider_start.step = freq_range_new[1] - freq_range_new[0]

#                 echo_slider_end.value = freq_range_new[echo_end_index]
#                 echo_slider_end.min = freq_range_new[0]
#                 echo_slider_end.max = freq_range_new[-1]
#                 echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

#             elif args["new"] == "index" and args["old"] == "ppm":
#                 if nechoes_csi > ppm_axis[-1]:
#                     echo_slider_end.min = freq_range_new[0]
#                     echo_slider_end.max = freq_range_new[-1]
#                     echo_slider_end.value = freq_range_new[echo_end_index]
#                     echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

#                     echo_slider_start.min = freq_range_new[0]
#                     echo_slider_start.max = freq_range_new[-1]
#                     echo_slider_start.value = freq_range_new[echo_start_index]
#                     echo_slider_start.step = freq_range_new[1] - freq_range_new[0]

#                 elif ppm_axis[-1] > nechoes_csi:
#                     echo_slider_end.value = freq_range_new[echo_end_index]
#                     echo_slider_end.min = freq_range_new[0]
#                     echo_slider_end.max = freq_range_new[-1]
#                     echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

#                     echo_slider_start.value = freq_range_new[echo_start_index]
#                     echo_slider_start.min = freq_range_new[0]
#                     echo_slider_start.max = freq_range_new[-1]
#                     echo_slider_start.step = freq_range_new[1] - freq_range_new[0]

#                 else:
#                     echo_slider_end.value = freq_range_new[echo_end_index]
#                     echo_slider_end.min = freq_range_new[0]
#                     echo_slider_end.max = freq_range_new[-1]
#                     echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

#                     echo_slider_start.value = freq_range_new[echo_start_index]
#                     echo_slider_start.min = freq_range_new[0]
#                     echo_slider_start.max = freq_range_new[-1]
#                     echo_slider_start.step = freq_range_new[1] - freq_range_new[0]

#             elif args["new"] == "index" and args["old"] == "Hz":
#                 if nechoes_csi < hz_axis[-1]:
#                     echo_slider_end.value = int(freq_range_new[echo_end_index])
#                     echo_slider_end.min = int(freq_range_new[0])
#                     echo_slider_end.max = int(freq_range_new[-1])
#                     echo_slider_end.step = int(freq_range_new[1] - freq_range_new[0])

#                     echo_slider_start.value = int(freq_range_new[echo_start_index])
#                     echo_slider_start.min = int(freq_range_new[0])
#                     echo_slider_start.max = int(freq_range_new[-1])
#                     echo_slider_start.step = int(freq_range_new[1] - freq_range_new[0])

#                 elif hz_axis[-1] > nechoes_csi:
#                     echo_slider_end.max = int(freq_range_new[-1])
#                     echo_slider_end.value = int(freq_range_new[echo_end_index])
#                     echo_slider_end.min = int(freq_range_new[0])
#                     echo_slider_end.step = int(freq_range_new[1] - freq_range_new[0])

#                     echo_slider_start.max = int(freq_range_new[-1])
#                     echo_slider_start.value = int(freq_range_new[echo_start_index])
#                     echo_slider_start.min = int(freq_range_new[0])
#                     echo_slider_start.step = int(freq_range_new[1] - freq_range_new[0])

#                 else:
#                     echo_slider_end.max = int(freq_range_new[-1])
#                     echo_slider_end.value = int(freq_range_new[echo_end_index])
#                     echo_slider_end.min = int(freq_range_new[0])
#                     echo_slider_end.step = int(freq_range_new[1] - freq_range_new[0])

#                     echo_slider_start.max = freq_range_new[-1]
#                     echo_slider_start.value = freq_range_new[echo_start_index]
#                     echo_slider_start.min = freq_range_new[0]
#                     echo_slider_start.step = freq_range_new[1] - freq_range_new[0]

#             # translate freq1, freq2, ...
#             text_freq_1.value = np.round(
#                 freq_range_new[
#                     freq_to_index(freq=text_freq_1.value, freq_range=freq_range_old)
#                 ],
#                 4,
#             )
#             text_freq_1.step = freq_range_new[1] - freq_range_new[0]
#             text_freq_2.value = np.round(
#                 freq_range_new[
#                     freq_to_index(freq=text_freq_2.value, freq_range=freq_range_old)
#                 ],
#                 4,
#             )
#             text_freq_2.step = freq_range_new[1] - freq_range_new[0]
#             text_freq_3.value = np.round(
#                 freq_range_new[
#                     freq_to_index(freq=text_freq_3.value, freq_range=freq_range_old)
#                 ],
#                 4,
#             )
#             text_freq_3.step = freq_range_new[1] - freq_range_new[0]
#             text_freq_4.value = np.round(
#                 freq_range_new[
#                     freq_to_index(freq=text_freq_4.value, freq_range=freq_range_old)
#                 ],
#                 4,
#             )
#             text_freq_4.step = freq_range_new[1] - freq_range_new[0]

#             self.freq_range = freq_range_new

#         # change frequency range dependent on frequney unit:
#         plot_opts_freq_units.observe(set_freq_slider, names="value")

#         set_freq_slider({"new": "index"})

#         # lets you ccoose colormaps:
#         plot_opts_cmap = widgets.Dropdown(
#             options=[
#                 "cividis",
#                 "inferno",
#                 "magma",
#                 "plasma",
#                 "viridis",
#                 "jet",
#                 "Blues",
#                 "BuGn",
#                 "BuPu",
#                 "GnBu",
#                 "Greens",
#                 "Greys",
#                 "OrRd",
#                 "Oranges",
#                 "PuBu",
#                 "PuBuGn",
#                 "PuRd",
#                 "Purples",
#                 "RdPu",
#                 "Reds",
#                 "Wistia",
#                 "YlGn",
#                 "YlGnBu",
#                 "YlOrBr",
#                 "YlOrRd",
#                 "afmhot",
#                 "autumn",
#                 "binary",
#                 "bone",
#                 "cool",
#                 "copper",
#                 "gist_gray",
#                 "gist_heat",
#                 "gist_yarg",
#                 "rainbow",
#                 "gray",
#                 "hot",
#                 "pink",
#                 "spring",
#                 "summer",
#                 "winter",
#                 "custom_cold",
#             ],
#             value="plasma",
#             description="CM:",
#             tooltip="Colormap",
#             disabled=False,
#             layout=widgets.Layout(width="15%"),
#         )

#         # lets you ccoose interpolation methods:
#         plot_opts_interpolation_method = widgets.Dropdown(
#             options=[
#                 "none",
#                 "antialiased",
#                 "nearest",
#                 "bilinear",
#                 "bicubic",
#                 "spline16",
#                 "spline36",
#                 "hanning",
#                 "hamming",
#                 "hermite",
#                 "kaiser",
#                 "quadric",
#                 "catrom",
#                 "gaussian",
#                 "bessel",
#                 "mitchell",
#                 "sinc",
#                 "lanczos",
#                 "blackman",
#             ],
#             value="bilinear",
#             description="IM:",
#             tooltip="Interpolation method",
#             disabled=False,
#             layout=widgets.Layout(width="15%"),
#         )

#         plot_opts_ratio = widgets.Dropdown(
#             options=["1 (1 Metab)", "2/1 (Lac/Pyr)", "1/(2+3) (Mal/Fum)"],
#             value="1 (1 Metab)",
#             description="Ratio: ",
#             tooltip="Choose if you want to plot the ratio of metabolites",
#         )

#         # interpolation factor:
#         plot_opts_interpolation_factor = widgets.BoundedIntText(
#             value=1,
#             min=1,
#             max=100,
#             step=1,
#             disabled=False,
#             description="IF:",
#             tooltip="Interpolation factor",
#             layout=widgets.Layout(width="10%"),
#         )

#         # name of figure to save:
#         text_save_fig_name = widgets.Text(
#             value="fig_name.svg",
#             placeholder="Type something",
#             description="Name:",
#             disabled=False,
#             layout=widgets.Layout(width="25%"),
#         )

#         # save figure (in cwd):
#         button_save_fig = widgets.Button(
#             description="Save fig.",
#             disabled=False,
#             button_style="",  # 'success', 'info', 'warning', 'danger' or ''
#             tooltip="Save the figure",
#             icon="check",  # (FontAwesome names without the `fa-` prefix)
#             layout=widgets.Layout(width="50px"),
#         )

#         # Draw ROI in anatomical image:
#         checkbox_draw_roi = widgets.Checkbox(
#             description="Draw ROI",
#             value=False,
#             tooltip="Lets you draw the ROI in anatomical image",
#             layout=widgets.Layout(width="20%"),
#         )

#         # Draw ROI in anatomical image:
#         checkbox_auto_set_crange = widgets.Checkbox(
#             description="ACR",
#             value=True,
#             tooltip="automatically sets the colorange to min <-> max of available values",
#             layout=widgets.Layout(width="10%"),
#         )

#         # Draw ROI in anatomical image:
#         rangeslider_choose_ROI = widgets.BoundedIntText(
#             value=0,
#             min=0,
#             max=100,
#             step=1,
#             disabled=False,
#             tooltip="Choose ROI.",
#             description="ROI #:",
#             layout=widgets.Layout(width="15%"),
#         )
#         # Draw ROI in anatomical image:
#         text_ROI_name = widgets.Text(
#             value="ROI_name",
#             placeholder="Name of ROI #",
#             description="ROI Name:",
#             disabled=False,
#             layout=widgets.Layout(width="20%"),
#         )

#         # perfrom frequecy adjustment(in cwd):
#         button_ROI_clear = widgets.Button(
#             description="Clear ROI",
#             disabled=False,
#             button_style="",  # 'success', 'info', 'warning', 'danger' or ''
#             tooltip="Clear the drawn ROI",
#             layout=widgets.Layout(width="10%"),
#         )

#         # perfrom frequecy adjustment(in cwd):
#         button_freq_adj = widgets.Button(
#             description="Freq Adj.",
#             disabled=True,
#             button_style="",  # 'success', 'info', 'warning', 'danger' or ''
#             tooltip="Aligns all frequecencies in the chosen frequency range",
#             icon="check",  # (FontAwesome names without the `fa-` prefix)
#             layout=widgets.Layout(width="15%"),
#         )

#         # Background that is subtracted:
#         text_background = widgets.FloatText(
#             value=0.0,
#             description="BG:",
#             tooltip="Background (will be subtracted)",
#             continuous_update=False,
#             disabled=False,
#             layout=widgets.Layout(width="12%"),
#         )

#         # Linebroadening [Hz]
#         text_lb = widgets.FloatText(
#             value=0.0,
#             description="LB:",
#             disabled=False,
#             continuous_update=False,
#             tooltip="Linebroadening peak [Hz]",
#             layout=widgets.Layout(width="12%"),
#         )

#         # frequency 1
#         text_freq_1 = widgets.FloatText(
#             value=0.0,
#             description="Freq1:",
#             disabled=False,
#             layout=widgets.Layout(width="15%"),
#         )
#         checkbox_freq_1 = widgets.Checkbox(
#             value=False,
#             tooltip="Plot data from frequency 1",
#         )
#         text_freq_2 = widgets.FloatText(
#             value=0.0,
#             description="Freq2:",
#             disabled=False,
#             layout=widgets.Layout(width="15%"),
#         )
#         checkbox_freq_2 = widgets.Checkbox(
#             value=False,
#             tooltip="Plot data from frequency 2",
#         )
#         text_freq_3 = widgets.FloatText(
#             value=0.0,
#             description="Freq3:",
#             disabled=False,
#             layout=widgets.Layout(width="15%"),
#         )
#         checkbox_freq_3 = widgets.Checkbox(
#             value=False,
#             tooltip="Plot data from frequency 3",
#         )
#         text_freq_4 = widgets.FloatText(
#             value=0.0,
#             description="Freq4:",
#             disabled=False,
#             layout=widgets.Layout(width="15%"),
#         )
#         checkbox_freq_4 = widgets.Checkbox(
#             value=False,
#             tooltip="Plot data from frequency 4",
#         )

#         def perform_lb(args):
#             from ..utils.utils_spectroscopy import multi_dim_linebroadening

#             # linebroadening [Hz]
#             lb = args["new"]

#             # check if linebroadening values has changed:
#             if args["new"] is not args["old"]:
#                 input_data = self.org_data_to_plot.copy()

#                 # perform linebroadening
#                 self.data_to_plot = multi_dim_linebroadening(
#                     input_data=input_data,
#                     lb=lb,
#                     signal_domain="spectral",
#                     data_obj=self,
#                 )

#                 # ------------------------ plotting ------------------------
#                 # get plot parameters:
#                 plot_img_params = set_plot_params(self)

#                 # Extract the values from the widgets
#                 plot_img_values = {
#                     key: widget.value for key, widget in plot_img_params.items()
#                 }

#                 # Call the plot_img function with the extracted values
#                 plot_img(**plot_img_values)
#             else:
#                 pass

#         text_lb.observe(perform_lb, names="value")

#         def choose_ROI_name(args):
#             """Either initialize the ROI name with the "ROI_name_" + ROI ID, or
#             sets the field to the already existing ROI name with the ROI ID.
#             """
#             key = args["new"]

#             # check if a mask with this key exists:
#             if key not in self.ROI:
#                 self.ROI[key] = {}

#             # check if this key has a mask:
#             if "name" not in self.ROI[key]:
#                 self.ROI[key]["name"] = "ROI_name_" + str(key)
#             else:
#                 text_ROI_name.value = self.ROI[key]["name"]

#             plot_img_params = set_plot_params(self)
#             plot_img_params["ROI"].value = key
#             set_plot_params(self, params=plot_img_params)

#         # change ROI name deciding on which ROI is chosen:
#         rangeslider_choose_ROI.observe(choose_ROI_name, names="value")

#         def set_ROI_name(args):
#             """Sets the ROI name to the already existing ROI name with the ROI ID."""
#             # get the ROI ID:
#             key = rangeslider_choose_ROI.value

#             # check if a mask with this key exists:
#             if key not in self.ROI:
#                 self.ROI[key] = {}

#             # check if this key has a mask:
#             self.ROI[key]["name"] = text_ROI_name.value

#         # change ROI name deciding on which ROI is chosen:
#         text_ROI_name.observe(set_ROI_name, names="value")

#         # init sliders:
#         slice_slider_ax = widgets.IntSlider(
#             value=dim_read_anat // 2,
#             min=0,
#             max=dim_read_anat - 1,
#             description="Slice Axial: ",
#             style=dict(text_color="red"),  # doesnt work
#         )
#         slice_slider_cor = widgets.IntSlider(
#             value=dim_slice_plot // 2,
#             min=0,
#             max=dim_slice_plot - 1,
#             description="Slice Coronal: ",
#         )
#         slice_slider_sag = widgets.IntSlider(
#             value=dim_phase_csi // 2,
#             min=0,
#             max=dim_phase_csi - 1,
#             description="Slice sagittal: ",
#         )
#         slice_slider_chan = widgets.IntSlider(
#             value=0,
#             min=0,
#             max=chans_csi - 1,
#             description="Chan.:",
#             tooltip="Receive Channel",
#             layout=widgets.Layout(width="20%"),
#         )

#         def adjust_colorrange(args):
#             # Should adjust the colorrange depening on the chosen data type
#             # (real, imag, phase, magnitude)
#             input_data = self.org_data_to_plot
#             if args["new"] == "Abs":
#                 plot_opts_metab_clim.max = 1.2 * np.nanmax(np.abs(input_data))
#                 plot_opts_metab_clim.min = 1.2 * np.nanmin(np.abs(input_data))
#             if args["new"] == "Real":
#                 plot_opts_metab_clim.max = 1.2 * np.nanmax(np.real(input_data))
#                 plot_opts_metab_clim.min = 1.2 * np.nanmin(np.real(input_data))
#             if args["new"] == "Imag":
#                 plot_opts_metab_clim.max = 1.2 * np.nanmax(np.imag(input_data))
#                 plot_opts_metab_clim.min = 1.2 * np.nanmin(np.imag(input_data))
#             if args["new"] == "Phase":
#                 plot_opts_metab_clim.max = 2 * np.nanmax(np.angle(input_data))
#                 plot_opts_metab_clim.min = 2 * np.nanmin(np.angle(input_data))

#         plot_opts_part.observe(adjust_colorrange, names="value")

#         # combine widgets into a vertical box:
#         ui_colors = widgets.HBox(
#             [
#                 plot_opts_alpha_overlay,
#                 plot_opts_metab_clim,
#                 checkbox_auto_set_crange,
#                 plot_opts_cmap,
#                 plot_opts_interpolation_method,
#                 plot_opts_interpolation_factor,
#                 text_lb,
#                 text_background,
#             ],
#         )

#         # # combine widgets into a horizontal box::
#         ui_up_left = widgets.HBox(
#             [
#                 plot_opts_anat_overlay,
#                 plot_opts_part,
#                 plot_opts_ratio,
#                 plot_opts_freq_units,
#                 button_freq_adj,
#                 text_save_fig_name,
#                 button_save_fig,
#             ]
#         )
#         ui_up = ui_up_left
#         ui_rep1 = widgets.HBox(
#             [
#                 slice_slider_rep1,
#                 slice_slider_chan,
#                 slice_slider_ax,
#             ]
#         )
#         ui_ROI = widgets.HBox(
#             [
#                 slice_slider_rep2,
#                 checkbox_draw_roi,
#                 rangeslider_choose_ROI,
#                 text_ROI_name,
#                 button_ROI_clear,
#             ]
#         )

#         ui_reps = widgets.VBox([ui_rep1, ui_ROI])
#         ui_echoes = widgets.VBox([echo_slider_start, echo_slider_end])
#         ui_checkboxes = widgets.VBox([rep_avg_checkbox, freq_avg_checkbox])
#         ui_reps_echoes_chans = widgets.HBox([ui_echoes, ui_checkboxes, ui_reps])
#         ui_freqs = widgets.HBox(
#             [
#                 text_freq_1,
#                 checkbox_freq_1,
#                 text_freq_2,
#                 checkbox_freq_2,
#                 text_freq_3,
#                 checkbox_freq_3,
#                 text_freq_4,
#                 checkbox_freq_4,
#             ]
#         )

#         def clear_roi(args):
#             # clear the values that are stored in the ROI dictonary under the
#             # current chosen key + update the plot

#             # set dictonary to empty:
#             self.ROI[rangeslider_choose_ROI.value] = {}
#             # replace old ROI name with default name:
#             text_ROI_name.value = "ROI_name_" + str(rangeslider_choose_ROI.value)

#             # ------------------------ plotting ------------------------
#             # get plot parameters:
#             plot_img_params = set_plot_params(self)

#             # Extract the values from the widgets
#             plot_img_values = {
#                 key: widget.value for key, widget in plot_img_params.items()
#             }

#             # Call the plot_img function with the extracted values
#             plot_img(**plot_img_values)

#         button_ROI_clear.on_click(clear_roi)

#         def draw_roi_finish(args):
#             """calcuate the ROI when the drawing of the ROI has finished (tickbox off)"""
#             if args["new"]:
#                 pass
#             else:
#                 self.calc_roi(
#                     key=rangeslider_choose_ROI.value,
#                     extent=[axlist[0].get_xlim(), axlist[0].get_ylim()],
#                     axial_image_shape=anat_image.shape[1:3],
#                 )

#                 # ------------------------ plotting ------------------------
#                 # get plot parameters:
#                 plot_img_params = set_plot_params(self)

#                 # Extract the values from the widgets
#                 plot_img_values = {
#                     key: widget.value for key, widget in plot_img_params.items()
#                 }

#                 # Call the plot_img function with the extracted values
#                 plot_img(**plot_img_values)

#         checkbox_draw_roi.observe(draw_roi_finish, names="value")

#         global plot_img_params

#         def set_plot_params(self, params=None):
#             """
#             Set the plot parameters. Sets first the widget values to the parameter (if
#             ther are loaded values). Either way, after that the plot parameters are set to
#             the widget values
#             """
#             # sets the sliders to a specific value in case a parameter file was
#             # loaded
#             try:
#                 # set this first because it affects the repetition setting:
#                 slice_slider_ax.value = params["slice_nr_ax"]
#                 slice_slider_cor.value = params["slice_nr_cor"]
#                 slice_slider_sag.value = params["slice_nr_sag"]
#                 plot_opts_freq_units.value = params["freq_units"]
#                 echo_slider_start.value = params["freq_start"]
#                 echo_slider_end.value = params["freq_end"]
#                 slice_slider_rep1.value = params["rep1"]
#                 slice_slider_chan.value = params["chan"]
#                 plot_opts_alpha_overlay.value = params["alpha_overlay"]
#                 slice_slider_rep2.value = params["rep2"]
#                 rep_avg_checkbox.value = params["rep_avg"]
#                 freq_avg_checkbox.value = params["freq_avg"]
#                 plot_opts_part.value = params["plot_style"]
#                 plot_opts_anat_overlay.value = params["anat_overlay"]
#                 plot_opts_metab_clim.value = params["metab_clim"]
#                 plot_opts_cmap.value = params["cmap"]
#                 plot_opts_interpolation_method.value = params["interp_method"]
#                 text_background.value = params["background"]
#                 text_lb.value = params["lb"]
#                 plot_opts_interpolation_factor.value = params["interp_factor"]
#                 rangeslider_choose_ROI.value = params["ROI"]
#                 text_freq_1.value = params["freq_1"]
#                 text_freq_2.value = params["freq_2"]
#                 text_freq_3.value = params["freq_3"]
#                 text_freq_4.value = params["freq_4"]
#                 plot_opts_ratio.value = params["metab_ratio"]
#                 # plot_opts_0_as_NaN.value = params["zero_as_NaN"]
#             except BaseException:
#                 pass

#             # set plotting parameters to slider values:
#             plot_img_params = {
#                 "slice_nr_ax": slice_slider_ax,
#                 "slice_nr_cor": slice_slider_cor,
#                 "slice_nr_sag": slice_slider_sag,
#                 "freq_units": plot_opts_freq_units,
#                 "freq_start": echo_slider_start,
#                 "freq_end": echo_slider_end,
#                 "chan": slice_slider_chan,
#                 "alpha_overlay": plot_opts_alpha_overlay,
#                 "plot_style": plot_opts_part,
#                 "rep1": slice_slider_rep1,
#                 "rep2": slice_slider_rep2,
#                 "rep_avg": rep_avg_checkbox,
#                 "freq_avg": freq_avg_checkbox,
#                 "anat_overlay": plot_opts_anat_overlay,
#                 "metab_clim": plot_opts_metab_clim,
#                 "cmap": plot_opts_cmap,
#                 "interp_method": plot_opts_interpolation_method,
#                 "interp_factor": plot_opts_interpolation_factor,
#                 "background": text_background,
#                 "lb": text_lb,
#                 "ROI": rangeslider_choose_ROI,
#                 "freq_1": text_freq_1,
#                 "freq_2": text_freq_2,
#                 "freq_3": text_freq_3,
#                 "freq_4": text_freq_4,
#                 "metab_ratio": plot_opts_ratio,
#                 # "zero_as_NaN": plot_opts_0_as_NaN,
#             }
#             return plot_img_params

#         # Define the on_click function
#         def on_click(event):
#             if event.inaxes in axlist:
#                 # axial image:
#                 if event.inaxes == axlist[0]:
#                     if checkbox_draw_roi.value:
#                         # take click position:
#                         slice_x = event.xdata
#                         slice_y = event.ydata
#                         key = rangeslider_choose_ROI.value

#                         # check if a mask with this key exists:
#                         if key not in self.ROI:
#                             self.ROI[key] = {}

#                         # check if this key has a mask:
#                         if "mask_anat_coords" not in self.ROI[key]:
#                             self.ROI[key]["mask_anat_coords"] = ([], [])

#                         # check if this key has a mask_anat_coords:
#                         if "mask_anat" not in self.ROI[key]:
#                             self.ROI[key]["mask_anat"] = []

#                         # append click position to mask_anat_coords:
#                         self.ROI[key]["mask_anat_coords"][0].append(slice_x)
#                         self.ROI[key]["mask_anat_coords"][1].append(slice_y)

#                         if "color" not in self.ROI[key]:
#                             self.ROI[key]["color"] = plt.rcParams[
#                                 "axes.prop_cycle"
#                             ].by_key()["color"][key]

#                         # get plot parameters:
#                         plot_img_params = set_plot_params(self)

#                         # Extract the values from the widgets
#                         plot_img_values = {
#                             key: widget.value for key, widget in plot_img_params.items()
#                         }

#                         # Call the plot_img function with the extracted values
#                         plot_img(**plot_img_values)

#                     else:
#                         # get plot parameters:
#                         plot_img_params = set_plot_params(self)

#                         # take click position:
#                         slice_x = event.xdata
#                         slice_y = event.ydata

#                         # define position:
#                         if self.acqp["ACQ_patient_pos"] == "Head_Prone":
#                             if self.method["PVM_SPackArrSliceOrient"] == "axial":
#                                 if self.method["PVM_SPackArrReadOrient"] == "L_R":
#                                     # csi_grid_plot = [csi_grid[1], csi_grid[2]]
#                                     pos = (0, slice_x, slice_y)
#                                     pos = [pos[1], pos[2]]
#                             elif self.method["PVM_SPackArrSliceOrient"] == "sagittal":
#                                 if self.method["PVM_SPackArrReadOrient"] == "H_F":
#                                     # csi_grid_plot = [csi_grid[0], csi_grid[2]]
#                                     pos = (slice_x, 0, slice_y)
#                                     pos = [pos[0], pos[2]]
#                             elif self.method["PVM_SPackArrSliceOrient"] == "coronal":
#                                 if self.method["PVM_SPackArrReadOrient"] == "H_F":
#                                     # csi_grid_plot = [csi_grid[0], csi_grid[1]]
#                                     pos = (slice_x, slice_y, 0)
#                                     pos = [pos[0], pos[1]]

#                         # get csi grid:
#                         sag_min_ind = np.argmin(np.abs(self.plotting_grid[0] - slice_x))
#                         cor_min_ind = np.argmin(np.abs(self.plotting_grid[1] - slice_y))

#                         # set the 2 plot parameters:
#                         plot_img_params["slice_nr_sag"].value = sag_min_ind
#                         plot_img_params["slice_nr_cor"].value = cor_min_ind

#                         # update plot parameter dict:
#                         plot_img_params = set_plot_params(self, params=plot_img_params)
#                 # metabolic image:
#                 elif event.inaxes == axlist[1]:
#                     # get plot parameters:
#                     plot_img_params = set_plot_params(self)

#                     # take click position:
#                     freq_val = event.xdata
#                     sig_int = event.ydata

#                     # define position:
#                     # freq_1_before = freq_to_index(freq_val, self.freq_range)
#                     freq_start_before = plot_img_params["freq_start"].value
#                     freq_end_before = plot_img_params["freq_end"].value

#                     freq_diff_before = np.abs(freq_end_before - freq_start_before)

#                     if (freq_val + freq_diff_before) > self.freq_range[-1]:
#                         freq_end_after = self.freq_range[-1]
#                     else:
#                         freq_end_after = freq_val + freq_diff_before

#                     # set the plot parameters:
#                     plot_img_params["freq_start"].value = freq_val
#                     plot_img_params["freq_end"].value = freq_end_after

#                     # update plot parameter dict:
#                     plot_img_params = set_plot_params(self, params=plot_img_params)
#                 # time curve:
#                 elif event.inaxes == axlist[2]:
#                     # get plot parameters:
#                     plot_img_params = set_plot_params(self)

#                     # so rep2 can also be shifted:
#                     rep1_before = plot_img_params["rep1"].value
#                     rep2_before = plot_img_params["rep2"].value

#                     # take click position:
#                     time1_after = event.xdata

#                     rep1_after = np.argmin(np.abs(time_axis - time1_after))
#                     if rep1_after < 0:
#                         rep1_after = 0

#                     # calculate difference:
#                     rep_diff = rep1_after - rep1_before

#                     # apply difference to rep2
#                     rep1 = rep1_after
#                     rep2 = rep2_before + rep_diff

#                     # set the 2 plot parameters:
#                     plot_img_params["rep1"].value = rep1
#                     plot_img_params["rep2"].value = rep2

#                     # update plot parameter dict:
#                     plot_img_params = set_plot_params(self, params=plot_img_params)

#                 else:
#                     pass

#         # Connect the click event to the on_click function
#         fig.canvas.mpl_connect("button_press_event", on_click)
#         # define plot parameters:
#         plot_img_params = set_plot_params(self, params=plot_params)
#         # create a link to the plot_img function that is called whenever on the
#         # parameters in the {} list is changed:
#         out = widgets.interactive_output(
#             plot_img,
#             plot_img_params,
#         )
#         # main_ui = widgets.HBox([ui_ax, ui_cor, ui_sag])
#         main_ui = widgets.VBox([ui_up, ui_colors, ui_reps_echoes_chans, ui_freqs])

#         # function that is called when save fig. button is pressed:
#         # has to be defined after plot_img_params were defined:
#         def save_figure(args, fig_name=None):
#             import os

#             if fig_name is None:
#                 fig_name = text_save_fig_name.value[0:-4]
#             else:
#                 pass

#             # get path to folder on above acquisition data:
#             if hasattr(self, "savepath"):
#                 # FIXME this does not work properly
#                 path_parentfolder = self.savepath
#             else:
#                 # get path to folder on above acquisition data::
#                 path_parentfolder = str(self.path)[0 : str(self.path).rfind("\\")]
#             print(path_parentfolder)
#             # make it path if not:
#             if path_parentfolder[-1] != "/":
#                 path_to_save = path_parentfolder + "/"

#             logger.debug(path_parentfolder)

#             # save figure:
#             # fig = plt.gcf()
#             if 0 == 1:
#                 import pickle

#                 # with open(os.path.join("test.pickle")) as f:
#                 #     pickle.dumps(fig, f, "wb")

#                 with open(
#                     os.path.join(path_parentfolder, fig_name + ".pickle"),
#                     "wb",
#                 ) as f:
#                     pickle.dump(fig, f)

#             # save figure as vector graphic:
#             fig.savefig(os.path.join(path_parentfolder, fig_name + ".svg"))

#             # also save as png for preview:
#             fig.savefig(
#                 os.path.join(path_parentfolder, fig_name + ".png"),
#                 format="png",
#             )

#             plot_img_params_dict = {}
#             # fill dict:
#             for key, value in plot_img_params.items():
#                 # print(value.value)
#                 plot_img_params_dict[key] = value.value

#             # get figure size:
#             plot_img_params_dict["fig_size"] = (
#                 fig.get_size_inches() * fig.dpi
#             ).tolist()

#             # save plotting parameters in json:
#             import json

#             json.dump(
#                 plot_img_params_dict,
#                 open(
#                     os.path.join(path_parentfolder, fig_name + ".json"),
#                     "w",
#                 ),
#             )

#         import functools

#         # define callback for save figure button:
#         button_save_fig.on_click(functools.partial(save_figure, fig_name=None))

#         def run_frequency_adjust(
#             args, input_data, freq_peaks, freq_range, start_rep, stop_rep, ppm_hz
#         ):
#             self.data_to_plot, freq_map = self.frequency_adjust(
#                 input_data=input_data,
#                 freq_peaks=freq_peaks,
#                 freq_range=freq_range,
#                 start_rep=start_rep,
#                 stop_rep=stop_rep,
#                 ppm_hz=plot_opts_freq_units.value,
#             )

#             # get plot parameters:
#             plot_img_params = set_plot_params(self)

#             # Extract the values from the widgets
#             plot_img_values = {
#                 key: widget.value for key, widget in plot_img_params.items()
#             }

#             # Call the plot_img function with the extracted values
#             plot_img(**plot_img_values)
#             return True

#         button_freq_adj.on_click(
#             functools.partial(
#                 run_frequency_adjust,
#                 input_data=self.data_to_plot,
#                 freq_peaks=None,
#                 freq_range=[-50, 50],
#                 start_rep=None,
#                 stop_rep=None,
#                 ppm_hz="hz",
#             )
#         )

#         # # Perform 1 initial plot:
#         # if (plot_opts_anat_overlay.value == "Metab + Anat") and (
#         #     anat_image is not None
#         # ):
#         #     # anatomical axial image:
#         #     im_ax_anat axlist[0].imshow(
#         #         np.squeeze(np.random.randn(10, 10)),
#         #         extent=plotting_extent,
#         #         cmap="bone",
#         #     )
#         #
#         #     im_ax = axlist[0].imshow(
#         #         np.squeeze(np.random.randn(10, 10)),
#         #         extent=plotting_extent,
#         #         cmap="jet",
#         #         alpha=0.5,
#         #         vmin=0,
#         #         vmax=1,
#         #     )

#         # This displays the Hbox containing the slider widgets.
#         if display_ui:
#             display(main_ui, out)
#             if save_fig:
#                 save_figure(self, fig_name=fig_name)
#         else:
#             return main_ui

#     def prep_metab_image_plot2d(
#         self,
#         data=None,
#         plot_style="Abs",
#         rep_avg=False,
#         rep1=0,
#         rep2=1,
#         freq_units="index",
#         freq_avg=False,
#         freq_start=0,
#         freq_end=1,
#         slice_nr_ax_csi=0,
#         background=0,
#         chan=0,
#         metab_ratio="1 (1 Metab)",
#         freqs=[0, 0, 0, 0],
#         interp_params={},
#     ):
#         """
#         Prepares the metabolite image for plotting.

#         Parameters
#         ----------
#         data : array_like, optional
#             The data to plot. If None, the data to plot is taken from
#             self.data_to_plot. The default is None.
#         plot_style : str, optional
#             The style of the plot. The default is "Abs".
#         rep_avg : bool, optional
#             If True, the data is averaged over repetitions. The default is False.
#         rep1 : int, optional
#             The first repetition to average over. The default is 0.
#         rep2 : int, optional
#             The second repetition to average over. The default is 1.
#         freq_units : str, optional
#             The unit of the frequency axis. The default is "index".
#         freq_avg : bool, optional
#             If True, the data is averaged over echoes. The default is False.
#         freq_start : int, optional
#             The first frequency to average over to freq_end. The default is 0.
#         freq_end : int, optional
#             The last echo to average over. The default is 0.


#         """

#         # get interpolation parameters:

#         from ..utils.utils_general import img_interp

#         # interpoaltion factor (default = 1
#         interp_factor = interp_params.get("interp_factor", 1)
#         interp_cmap = interp_params.get("cmap", "jet")
#         interp_method = interp_params.get("interp_method", "bilinear")
#         interp_threshold = interp_params.get("interp_threshold", -np.inf)

#         # choose plot style:
#         if data is None:
#             data_to_plot = self.data_to_plot
#         else:
#             data_to_plot = data

#         ppm_axis = get_freq_axis(
#             data_obj=self, unit="ppm", npoints=self.data_to_plot.shape[0]
#         )
#         hz_axis = get_freq_axis(
#             data_obj=self,
#             unit="Hz",
#             npoints=self.data_to_plot.shape[0],
#         )
#         # get frequency axis:
#         if freq_units == "ppm":
#             freq_range = ppm_axis
#         elif freq_units == "Hz":
#             freq_range = hz_axis
#         else:
#             freq_range = np.linspace(
#                 0, data_to_plot.shape[0] - 1, data_to_plot.shape[0]
#             )

#         # translate frequency from unit to index:
#         freq_start_index = freq_to_index(freq=freq_start, freq_range=freq_range)
#         freq_end_index = freq_to_index(freq=freq_end, freq_range=freq_range)

#         # for repetitions:
#         if rep_avg:
#             if rep2 > rep1:
#                 rep_range = slice(
#                     rep1, rep2 + 1
#                 )  # +1 because the stop index is exclusive
#             else:
#                 # single value, but still a slice
#                 rep_range = slice(rep1, rep1 + 1)
#         else:
#             # single value, but still a slice
#             rep_range = slice(rep1, rep1 + 1)

#         # for echoes:
#         if freq_avg:
#             if freq_end > freq_start:
#                 echo_range = slice(
#                     freq_start_index, freq_end_index + 1
#                 )  # +1 because the stop index is exclusive
#             else:
#                 echo_range = slice(
#                     freq_start_index, freq_start_index + 1
#                 )  # single value, but still a slice
#         else:
#             echo_range = slice(
#                 freq_start_index, freq_start_index + 1
#             )  # single value, but still a slice

#         # hardcode for now:
#         slice_nr_ax_csi = 0
#         spatdom_xarray_data = np.mean(
#             data_to_plot[
#                 echo_range,
#                 slice(slice_nr_ax_csi, slice_nr_ax_csi + 1),
#                 :,
#                 :,
#                 rep_range,
#                 slice(chan, chan + 1),
#             ],
#             axis=4,
#             keepdims=True,
#         )

#         if plot_style == "Abs":
#             spatdom_xarray_data = np.absolute(spatdom_xarray_data)
#         elif plot_style == "Real":
#             spatdom_xarray_data = np.real(spatdom_xarray_data)
#         elif plot_style == "Imag":
#             spatdom_xarray_data = np.imag(spatdom_xarray_data)
#         elif plot_style == "Phase":
#             spatdom_xarray_data = np.angle(spatdom_xarray_data)
#         else:
#             # default
#             spatdom_xarray_data = np.absolute(spatdom_xarray_data)

#         # subbract background:
#         spatdom_xarray_data = spatdom_xarray_data - background

#         # 1 Metabolite:
#         if metab_ratio == "1 (1 Metab)":
#             reco_data = np.mean(
#                 spatdom_xarray_data,
#                 axis=0,
#                 keepdims=True,
#             )

#             # Interpolate the image by a factor:
#             reco_data = img_interp(
#                 input_data=np.squeeze(reco_data),
#                 interp_factor=interp_factor,
#                 cmap=interp_cmap,
#                 interp_method=interp_method,
#                 threshold=interp_threshold,
#                 overlay=True,
#             )

#         elif metab_ratio == "2/1 (Lac/Pyr)":
#             freq_range_temp = freq_end - freq_start
#             reco_data = []
#             for k in range(2):
#                 freq_start = (freqs[k] - freq_range_temp / 2,)
#                 freq_end = (freqs[k] + freq_range_temp / 2,)
#                 freq_start_index = freq_to_index(freq=freq_start, freq_range=freq_range)
#                 freq_end_index = freq_to_index(freq=freq_end, freq_range=freq_range)
#                 # for echoes:
#                 if freq_avg:
#                     if freq_end > freq_start:
#                         echo_range = slice(
#                             freq_start_index, freq_end_index + 1
#                         )  # +1 because the stop index is exclusive
#                     else:
#                         echo_range = slice(
#                             freq_start_index, freq_start_index + 1
#                         )  # single value, but still a slice
#                 else:
#                     echo_range = slice(
#                         freq_start_index, freq_start_index + 1
#                     )  # single value, but still a slice

#                 reco_data.append(
#                     np.mean(
#                         spatdom_xarray_data[
#                             echo_range,
#                             slice(slice_nr_ax_csi, slice_nr_ax_csi + 1),
#                             :,
#                             :,
#                             :,
#                             slice(chan, chan + 1),
#                         ],
#                         axis=0,
#                         keepdims=True,
#                     )
#                 )

#             spatdom_xarray_interp = []
#             for k in range(2):
#                 # Interpolate the image by a factor:
#                 spatdom_xarray_interp.append(
#                     img_interp(
#                         input_data=np.squeeze(reco_data[k]),
#                         interp_factor=interp_factor,
#                         cmap=interp_cmap,
#                         interp_method=interp_method,
#                         threshold=interp_threshold,
#                         overlay=True,
#                     )
#                 )
#             reco_data = spatdom_xarray_interp[1] / spatdom_xarray_interp[0]

#         else:
#             pass

#         return reco_data

#     def prep_spectrum_plot2d(
#         self,
#         data=None,
#         plot_style="Abs",
#         rep_avg=False,
#         rep1=0,
#         rep2=1,
#         freq_units="index",
#         freq_avg=False,
#         freq_start=0,
#         freq_end=1,
#         slice_nr_ax_csi=0,
#         ix=0,
#         iy=0,
#         background=0,
#         chan=0,
#         current_key=0,
#     ):
#         # choose plot style:
#         if data is None:
#             data_to_plot = self.data_to_plot
#         else:
#             data_to_plot = data

#         # subtract background:
#         spatdom_xarray_data = data_to_plot

#         ppm_axis = get_freq_axis(
#             data_obj=self, unit="ppm", npoints=self.data_to_plot.shape[0]
#         )
#         hz_axis = get_freq_axis(
#             data_obj=self,
#             unit="Hz",
#             npoints=self.data_to_plot.shape[0],
#         )
#         # get frequency axis:
#         if freq_units == "ppm":
#             freq_range = ppm_axis
#         elif freq_units == "Hz":
#             freq_range = hz_axis
#         else:
#             freq_range = np.linspace(
#                 0, spatdom_xarray_data.shape[0] - 1, spatdom_xarray_data.shape[0]
#             )

#         # translate frequency from unit to index:
#         freq_start_index = freq_to_index(freq=freq_start, freq_range=freq_range)
#         freq_end_index = freq_to_index(freq=freq_end, freq_range=freq_range)

#         # for repetitions:
#         if rep_avg:
#             if rep2 > rep1:
#                 rep_range = slice(
#                     rep1, rep2 + 1
#                 )  # +1 because the stop index is exclusive
#             else:
#                 # single value, but still a slice
#                 rep_range = slice(rep1, rep1 + 1)
#         else:
#             # single value, but still a slice
#             rep_range = slice(rep1, rep1 + 1)

#         # hardcode for now:
#         slice_nr_ax_csi = 0

#         # don't apply mask, take mean over repetitions:
#         csi_spec = np.mean(
#             spatdom_xarray_data[
#                 :,
#                 slice(slice_nr_ax_csi, slice_nr_ax_csi + 1),
#                 slice(ix, ix + 1),
#                 slice(iy, iy + 1),
#                 rep_range,
#                 slice(chan, chan + 1),
#             ],
#             axis=4,
#             keepdims=True,
#         )

#         if plot_style == "Abs":
#             csi_spec = np.absolute(csi_spec)
#         elif plot_style == "Real":
#             csi_spec = np.real(csi_spec)
#         elif plot_style == "Imag":
#             csi_spec = np.imag(csi_spec)
#         elif plot_style == "Phase":
#             csi_spec = np.angle(csi_spec)
#         else:
#             # default
#             csi_spec = np.absolute(csi_spec)

#         # lazy fix:
#         if np.ndim(spatdom_xarray_data) > 6:  # probably fit data ...
#             spatdom_xarray_data = spatdom_xarray_data[:, :, :, :, :, :, 0]

#         # whether to apply mask:
#         if current_key in self.ROI:
#             if "mask_csi" in self.ROI[current_key]:
#                 if np.sum(self.ROI[current_key]["mask_csi"]) > 0:
#                     # apply mask:
#                     spatdom_xarray_data = (
#                         spatdom_xarray_data
#                         * self.ROI[current_key]["mask_csi"][
#                             np.newaxis, np.newaxis, :, :, np.newaxis, np.newaxis
#                         ]
#                     )
#                     csi_spec = np.mean(
#                         np.mean(
#                             np.mean(
#                                 spatdom_xarray_data[
#                                     :,
#                                     slice(slice_nr_ax_csi, slice_nr_ax_csi + 1),
#                                     :,
#                                     :,
#                                     rep_range,
#                                     slice(chan, chan + 1),
#                                 ],
#                                 axis=4,
#                                 keepdims=True,
#                             ),
#                             axis=3,
#                             keepdims=True,
#                         ),
#                         axis=2,
#                         keepdims=True,
#                     )

#             else:
#                 # dont apply mask, take mean over repetitions:
#                 csi_spec = np.mean(
#                     spatdom_xarray_data[
#                         :,
#                         slice(slice_nr_ax_csi, slice_nr_ax_csi + 1),
#                         slice(ix, ix + 1),
#                         slice(iy, iy + 1),
#                         rep_range,
#                         slice(chan, chan + 1),
#                     ],
#                     axis=4,
#                     keepdims=True,
#                 )
#         else:
#             pass

#         csi_spec = csi_spec - -background

#         return np.squeeze(csi_spec)

#     def prep_timecurve_plot2d(
#         self,
#         data=None,
#         plot_style="Abs",
#         rep_avg=False,
#         rep1=0,
#         rep2=1,
#         freq_units="index",
#         freq_avg=False,
#         freq_start=0,
#         freq_end=1,
#         slice_nr_ax_csi=0,
#         ix=0,
#         iy=0,
#         background=0,
#         chan=0,
#     ):
#         # choose plot style:
#         if data is None:
#             data_to_plot = self.data_to_plot
#         else:
#             data_to_plot = data

#         if plot_style == "Abs":
#             spatdom_xarray_data = np.absolute(data_to_plot)
#         elif plot_style == "Real":
#             spatdom_xarray_data = np.real(data_to_plot)
#         elif plot_style == "Imag":
#             spatdom_xarray_data = np.imag(data_to_plot)
#         elif plot_style == "Phase":
#             spatdom_xarray_data = np.angle(data_to_plot)
#         else:
#             # default
#             spatdom_xarray_data = np.absolute(data_to_plot)

#         # subbract background:
#         spatdom_xarray_data = spatdom_xarray_data - background

#         ppm_axis = get_freq_axis(
#             data_obj=self, unit="ppm", npoints=self.data_to_plot.shape[0]
#         )
#         hz_axis = get_freq_axis(
#             data_obj=self,
#             unit="Hz",
#             npoints=self.data_to_plot.shape[0],
#         )
#         # get frequency axis:
#         if freq_units == "ppm":
#             freq_range = ppm_axis
#         elif freq_units == "Hz":
#             freq_range = hz_axis
#         else:
#             freq_range = np.linspace(
#                 0, spatdom_xarray_data.shape[0] - 1, spatdom_xarray_data.shape[0]
#             )

#         # translate frequency from unit to index:
#         freq_start_index = freq_to_index(freq=freq_start, freq_range=freq_range)
#         freq_end_index = freq_to_index(freq=freq_end, freq_range=freq_range)

#         # for echoes:
#         if freq_avg:
#             if freq_end > freq_start:
#                 echo_range = slice(
#                     freq_start_index, freq_end_index + 1
#                 )  # +1 because the stop index is exclusive
#             else:
#                 echo_range = slice(
#                     freq_start_index, freq_start_index + 1
#                 )  # single value, but still a slice
#         else:
#             echo_range = slice(
#                 freq_start_index, freq_start_index + 1
#             )  # single value, but still a slice

#         # hardcode for now:
#         slice_nr_ax_csi = 0

#         csi_spec = np.mean(
#             spatdom_xarray_data[
#                 echo_range,
#                 slice(slice_nr_ax_csi, slice_nr_ax_csi + 1),
#                 slice(ix, ix + 1),
#                 slice(iy, iy + 1),
#                 :,
#                 slice(chan, chan + 1),
#             ],
#             axis=0,
#             keepdims=True,
#         )

#         return np.squeeze(csi_spec)
