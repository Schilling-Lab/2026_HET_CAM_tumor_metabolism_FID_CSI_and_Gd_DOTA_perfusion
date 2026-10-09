# This just serves as a backup and will be removed in the future
# def plot3D_new2(
#     self,
#     axlist,
#     coronal_image=None,
#     axial_image=None,
#     bssfp_data=None,
#     parameter_dict=None,
#     display_ui=True,
#     pyr_ind=0,
#     fig=None,
#     plot_params=None,
#     save_fig=False,
#     thirdMetab=False,
#     fieldmap=None,
# ):
#     """

#     An interactive 3D plotting function for visualizing and analyzing bSSFP data.

#     Parameters
#     ----------
#     axlist : list of matplotlib.axes.Axes
#         List of axes to draw on. Should contain 3 axes for 3D plot:
#         1st for axial, 2nd for coronal, 3rd for time curves.
#     coronal_image : MRIExperiment instance, optional
#         Contains anatomical reference images, usually coronal.
#     axial_image : MRIExperiment instance, optional
#         Contains anatomical reference images, usually axial.
#     bssfp_data : ndarray, optional
#         Array to plot. Data should be sorted in the order:
#         Echoes, Read, Phase, Slice, Repetitions, Channels.
#         If None, uses self.Reconstructed_data.
#     parameter_dict : dict, optional
#         Dictionary of keyword arguments to pass to ax.imshow.
#     display_ui : bool, default True
#         If True, directly renders interactive sliders using IPython.display.
#         If False, returns the UI object instead.
#     pyr_ind : int, default 0
#         Index indicating if pyruvate was acquired first (before lactate).
#     fig : matplotlib.figure.Figure, optional
#         Figure object created during 'fig, ax = plt.subplots()' call.
#     plot_params : dict or str, optional
#         Plot parameters. Can be a dictionary or a path to a JSON file.
#     save_fig : bool, default False
#         If True, saves the figure immediately. Useful for batch processing.
#     thirdMetab : bool, default False
#         If True, plots a third metabolite (expected repetition order: Pyr, 3rd metabolite, Lac).
#     fieldmap : ndarray, optional
#         Fieldmap data for B0 visualization.

#     Returns
#     -------
#     widgets.HBox or None
#         If display_ui is False, returns the ipywidgets HBox containing interactive sliders.
#         Otherwise, returns None.

#     Notes
#     -----
#     This function provides an interactive interface for exploring 3D bSSFP data.
#     It allows for visualization of anatomical and metabolic data, with options
#     for overlay, color mapping, and time series analysis.

#     The function supports various plot customizations including:
#     - Slice selection in axial, coronal, and sagittal planes
#     - Echo and repetition selection
#     - Metabolite selection (Pyruvate, Lactate, ratio)
#     - Display style (Magnitude, Real, Imaginary, Phase)
#     - Color mapping and range adjustment
#     - Anatomical overlay options

#     Interactive features:
#     - Click on images to update slice positions
#     - Adjust repetition range for averaging
#     - Save figures with current settings

#     Examples
#     --------
#     >>> # Load data
#     >>> dirpath = 'path/to/data/'
#     >>> scans = hypermri.Study(dirpath)
#     >>> scan_id = '20231213_XX_mmt_'
#     >>> savepath = dirpath
#     >>>
#     >>> # Load and process data
#     >>> bssfp = scans[40]
#     >>> coronal = scans[41]
#     >>> axial = scans[31]
#     >>>
#     >>> # Reconstruct data
#     >>> bssfp.full_reco(axial=axial, coronal=coronal, interp=True)
#     >>>
#     >>> # Create figure and plot
#     >>> fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True, figsize=(12,5))
#     >>> bssfp.plot3D_new2(
#     ...     coronal_image=coronal,
#     ...     axial_image=axial,
#     ...     axlist=[ax1, ax2, ax3],
#     ...     plot_params='pyruvate_signal_mean_high_interp.json',
#     ...     fig=fig
#     ... )

#     See Also
#     --------
#     MRIExperiment : Class for handling Bruker experiment data
#     load_plot_params : Function to load plot parameters from a JSON file
#     """
#     from IPython.display import clear_output

#     from hypermri.utils.utils_general import define_grid as ut_define_grid
#     from hypermri.utils.utils_general import (
#         define_imageFOV_parameters as ut_define_imageFOV_parameters,
#     )
#     from hypermri.utils.utils_general import (
#         define_imagematrix_parameters as ut_define_imagematrix_parameters,
#     )

#     # import cProfile

#     clear_output()
#     # list of axes has to be passed:
#     if axlist is None:
#         logger.critical(
#             "Need to pass axlist (use fig, (ax1, ax2, ax3) = plt.subplots(1, 3) \
#                     to generate and call function with: \
#                     axlist=[ax1, ax2, ax3])"
#         )
#         return False

#     # # at least 3 axes have to be passed:
#     # if len(axlist) < 3:
#     #     logger.critical(
#     #         "Not enough ax objects in axlist for 3D plot, please give 3 axes!"
#     #     )
#     #     return False

#     # ony can plot 3D data so far:
#     if self.method["PVM_SpatDimEnum"] != "3D":
#         logger.critical(
#             "3D print function but no 3D data available!" + "--> Aborting"
#         )
#         return False

#     # check if there is data to plot
#     if bssfp_data is None:
#         if not isinstance(self.Reconstructed_data, np.ndarray):
#             logger.critical(
#                 "There is no reconstructed data available. "
#                 + "Please call the function bSSFP.reconstruction first!"
#                 + "--> Aborting"
#             )
#             return False
#         else:
#             # get reconstructed data:
#             data = self.seq2d
#     else:
#         # use input data:
#         data = bssfp_data

#     # define bSSFP parameters:
#     # ------------------------------------------------------------
#     self.data_to_plot = data
#     # read image dimensions --- bSSFP
#     dim_read_bssfp = data.shape[1]  # was z
#     dim_phase_bssfp = data.shape[2]  # was y
#     dim_slice_bssfp = data.shape[3]  # was x

#     self.method["PVM_Fov"][0]
#     self.method["PVM_Fov"][1]
#     self.method["PVM_Fov"][2]

#     reps_bssfp = data.shape[4]
#     chans_bssfp = data.shape[5]
#     nechoes_bssfp = data.shape[0]

#     # slice_thick_cor_bssfp = mm_slice_bssfp / dim_slice_bssfp
#     # slice_thick_ax_bssfp = mm_read_bssfp / dim_read_bssfp
#     # slice_thick_sag_bssfp = mm_phase_bssfp / dim_phase_bssfp

#     read_orient_bssfp = self.method["PVM_SPackArrReadOrient"]
#     # read_offset_bssfp = self.method["PVM_SPackArrReadOffset"]
#     # phase_offset_bssfp = self.method["PVM_SPackArrPhase1Offset"]
#     # slice_offset_bssfp = self.method["PVM_SPackArrSliceOffset"]

#     bssfp_grid = ut_define_grid(data_obj=self)

#     # ------AXIAL ANATOMICAL ------------------------------
#     # ------------------------------------------------------------
#     # init empty grid:
#     ax_grid = None
#     if axial_image is None:
#         axial_obj = None
#     # object contains no info about extent, assume same dimensions:
#     elif isinstance(axial_image, np.ndarray):
#         pass

#     # object contains info about extent:
#     elif axial_image.__module__.startswith("hypermri."):
#         # get grid describing the coronal slices:
#         ax_grid = ut_define_grid(data_obj=axial_image)

#         # save axial object before overwriting:
#         axial_obj = axial_image
#         # axial_image = axial_image.seq2d
#         ax_shift_counter = get_counter(
#             data_obj=axial_obj,
#             counter_name="shift_counter",
#         )
#         if ax_shift_counter > 0:
#             pass
#         else:
#             axial_obj.seq2d_oriented = shift_anat(
#                 csi_obj=self,
#                 anat_obj=axial_obj,
#                 use_scipy_shift=True,
#             )

#         axial_image = axial_obj.seq2d_oriented

#     else:
#         pass

#     # ------CORONAL ANATOMICAL ------------------------------
#     # init empty grid:
#     cor_grid = None
#     if coronal_image is None:
#         coronal_obj = None
#     # object contains no info about extent, assume same dimensions:
#     elif isinstance(coronal_image, np.ndarray):
#         pass
#     # object contains info about extent:
#     elif coronal_image.__module__.startswith("hypermri."):
#         # prepare coupling of coronal and bssfp data:
#         # get grid describing the coronal slices:
#         cor_grid = ut_define_grid(data_obj=coronal_image)

#         # save coronal object before overwriting:
#         coronal_obj = coronal_image
#         # coronal_image = coronal_image.seq2d
#         # axial_image = axial_image.seq2d
#         cor_shift_counter = get_counter(
#             data_obj=coronal_obj,
#             counter_name="shift_counter",
#         )

#         if cor_shift_counter == 0:
#             coronal_obj.seq2d_oriented = shift_anat(
#                 csi_obj=self,
#                 anat_obj=coronal_obj,
#                 use_scipy_shift=True,
#             )
#         coronal_image = coronal_obj.seq2d_oriented

#     else:
#         pass

#     # FOV - anat:
#     if axial_obj is not None:
#         fov_anat_read_plot, _, _ = ut_define_imageFOV_parameters(data_obj=axial_obj)

#         # matrix - anat:
#         mat_anat_read_plot, _, _ = ut_define_imagematrix_parameters(
#             data_obj=axial_obj
#         )
#         res_anat_read_plot = fov_anat_read_plot / mat_anat_read_plot
#     else:
#         res_anat_read_plot = None
#     if coronal_obj is not None:
#         (
#             mat_anat_slice_plot_z,
#             mat_anat_slice_plot_x,
#             mat_anat_slice_plot,
#         ) = ut_define_imagematrix_parameters(data_obj=coronal_obj)
#         (
#             fov_anat_slice_plot_z,
#             fov_anat_slice_plot_x,
#             fov_anat_slice_plot,
#         ) = ut_define_imageFOV_parameters(data_obj=coronal_obj)
#         res_anat_slice_plot = fov_anat_slice_plot / mat_anat_slice_plot
#     else:
#         res_anat_slice_plot = None

#     # calculate resolution - anat:
#     if res_anat_read_plot is None:
#         res_anat_read_plot = fov_anat_slice_plot_z / mat_anat_slice_plot_z
#     if res_anat_slice_plot is None:
#         pass

#     # FOV - bssfp:
#     fov_bssfp_read_plot, _, fov_bssfp_slice_plot = ut_define_imageFOV_parameters(
#         data_obj=self
#     )
#     # matrix - bssfp:
#     mat_bssfp_read_plot, _, mat_bssfp_slice_plot = ut_define_imagematrix_parameters(
#         data_obj=self
#     )

#     if not read_orient_bssfp == "H_F":
#         # checking if we have the right read orientation
#         # if this error comes up you have to implement different
#         # orientations possibly
#         logger.debug(
#             "Careful this function is not evaluated for this read orientation"
#         )
#     else:
#         pass

#     # ------FIELDMAP------------------------------
#     if fieldmap is not None:
#         from ..utils.utils_general import get_gmr

#         # define gyromagnetic ratios:
#         gmr_1h = get_gmr(nucleus="1h")  # MHz/T
#         gmr_13c = get_gmr(nucleus="13c")  # MHz/T
#         output_nuc = "13c"
#         fm = fieldmap.seq2d_oriented
#         if output_nuc == "1h":
#             pass
#         elif output_nuc == "13c":
#             fm = fm * gmr_13c / gmr_1h
#         else:
#             pass
#         fm = np.flip(fm, axis=1)
#         # define grid of fieldmap:
#         fieldmap_grid = ut_define_grid(data_obj=fieldmap)

#     # use default paramter dictionary if none is provided
#     if not parameter_dict:
#         parameter_dict = {"cmap": "magma"}

#     if plot_params is not None:
#         if isinstance(plot_params, dict):
#             pass
#         elif isinstance(plot_params, str):
#             try:
#                 plot_params = load_plot_params(
#                     param_file=plot_params, data_obj=self
#                 )
#             except BaseException:
#                 plot_params = {}
#         else:
#             plot_params = {}

#         if "fig_name" in plot_params:
#             fig_name = plot_params["fig_name"]
#         else:
#             fig_name = None

#     else:
#         fig_name = None

#     # ------------DEFINE EXTENT -----------------
#     # calc vals for hline and vline
#     cor_ext_phase = np.linspace(
#         self.cor_ext[0] - (self.cor_ext[0] / self.data_to_plot.shape[2]),
#         self.cor_ext[1] - (self.cor_ext[1] / self.data_to_plot.shape[2]),
#         self.data_to_plot.shape[2],
#     )

#     cor_ext_read = np.linspace(
#         self.cor_ext[2] - (self.cor_ext[2] / self.data_to_plot.shape[1]),
#         self.cor_ext[3] - (self.cor_ext[3] / self.data_to_plot.shape[1]),
#         self.data_to_plot.shape[1],
#     )

#     ax_ext_phase = np.linspace(
#         self.ax_ext[0] - (self.ax_ext[0] / self.data_to_plot.shape[2]),
#         self.ax_ext[1] - (self.ax_ext[1] / self.data_to_plot.shape[2]),
#         self.data_to_plot.shape[2],
#     )
#     ax_ext_slice = np.flip(
#         np.linspace(
#             self.ax_ext[2] - (self.ax_ext[2] / self.data_to_plot.shape[3]),
#             self.ax_ext[3] - (self.ax_ext[3] / self.data_to_plot.shape[3]),
#             self.data_to_plot.shape[3],
#         )
#     )

#     # define image plotting function
#     def plot_img(
#         slice_nr_ax=0,
#         slice_nr_cor=0,
#         slice_nr_sag=0,
#         echo_nr=0,
#         rep1=0,
#         chan=0,
#         alpha_overlay=0.5,
#         rep2=None,
#         rep_avg=False,
#         plot_style="Abs",
#         experiment_style="Perfusion",
#         pyr_lac="Pyr",
#         anat_overlay="Metab",
#         metab_clim=[0, 1],
#         cmap="plasma",
#         interp_method="none",
#         interp_factor=1,
#         background=0.0,
#         **kwargs,
#     ):
#         # cProfile.run("time-consuming_processing()")
#         """
#         This is the plotting function that is called whenever one of it's
#         parameters is changed:
#         Parameters
#         slice_nr_ax: axial slice number
#         slice_nr_cor: coronal slice number
#         slice_nr_sag: sagtial slice number (often not used)
#         echo_nr: echo number (for me-bSSFP)
#         rep1: repetition that is shown. if rep_avg = True, start of range that is averaged (see rep2)
#         chan: Channel (for multichannel coil experiments)
#         alpha_overlay: the intenstiy of the overlay of the metabolic images onto the anatomical (0-1)
#         rep2: if rep_avg==True, this is the end of the range that is averaged (see rep1)
#         rep_avg: if True, the range from rep1 to rep2 is averaged
#         plot_style: can be magnitude, real, imaginary of phase
#         pyr_lac: which metabolite to show (default pyruvate was acquired first
#         anat_overlay: Can be metabolite only, overlay or anatomical only
#         metab_clim: range of values that is displayed (smaller are set to NaN)
#         cmap: colormap of metabolic images
#         interp_method: interpolation method with which the images are displayed
#         background: background signal that will be subtracted
#         """
#         # init range:
#         rep_range = rep1
#         # choose plot style:
#         if plot_style == "Abs":
#             bssfp_image_data = np.absolute(self.data_to_plot)
#         elif plot_style == "Real":
#             bssfp_image_data = np.real(self.data_to_plot)
#         elif plot_style == "Imag":
#             bssfp_image_data = np.imag(self.data_to_plot)
#         elif plot_style == "Phase":
#             bssfp_image_data = np.angle(self.data_to_plot)
#         else:
#             # default
#             bssfp_image_data = np.absolute(self.data_to_plot)

#         # set color range:
#         # bssfp_image_data[bssfp_image_data < metab_clim[0]] = np.nan
#         # bssfp_image_data[bssfp_image_data > metab_clim[1]] = metab_clim[1]
#         # init empty arrays:
#         global pyr_data, lac_data
#         # deterMine experiment style:
#         if experiment_style == "Perfusion":
#             pyr_data = bssfp_image_data[:, :, :, :, :, :] - background
#         elif experiment_style == "Pyr + Lac" and not thirdMetab:
#             if pyr_lac == "Lac/Pyr":
#                 # background subtraction is handled further down
#                 pyr_data = bssfp_image_data[:, :, :, :, pyr_ind::2, :]
#                 lac_data = bssfp_image_data[:, :, :, :, pyr_ind + 1 :: 2, :]
#             else:
#                 pyr_data = bssfp_image_data[:, :, :, :, pyr_ind::2, :] - background
#                 lac_data = (
#                     bssfp_image_data[:, :, :, :, pyr_ind + 1 :: 2, :] - background
#                 )

#         elif thirdMetab:
#             if pyr_lac == "Lac/Pyr":
#                 # background subtraction is handled further down
#                 pyr_data = bssfp_image_data[:, :, :, :, pyr_ind::2, :]
#                 lac_data = bssfp_image_data[:, :, :, :, pyr_ind + 1 :: 2, :]
#             else:
#                 pyr_data = bssfp_image_data[:, :, :, :, pyr_ind::3, :] - background
#                 thirdMetab_data = (
#                     bssfp_image_data[:, :, :, :, pyr_ind + 1 :: 3, :] - background
#                 )
#                 lac_data = (
#                     bssfp_image_data[:, :, :, :, pyr_ind + 2 :: 3, :] - background
#                 )
#         else:
#             Warning("unknown measurement style " + str(plot_opts_measurement.value))

#         # determine which contrast to use:
#         if pyr_lac == "Pyr":
#             bssfp_image_data = pyr_data
#         elif (pyr_lac == "Lac") and (
#             (experiment_style == "Pyr + Lac")
#             or (experiment_style == "3 Metabolites")
#         ):
#             bssfp_image_data = lac_data
#         elif (pyr_lac == "3rd Metabolite") and (
#             experiment_style == "3 Metabolites"
#         ):
#             bssfp_image_data = thirdMetab_data
#         elif (pyr_lac == "Lac/Pyr") and (experiment_style == "Pyr + Lac"):
#             if rep_avg:
#                 # valid range:
#                 if rep2 > rep1:
#                     rep_range = range(rep1, rep2)
#                     # assuming the order Echoes, Read, phase, slice, repetitions, channels in the data:
#                     # filter low signal regions:
#                     bssfp_image_data = self.data_to_plot

#                     pyr_data = bssfp_image_data[:, :, :, :, pyr_ind::2, :]
#                     lac_data = bssfp_image_data[:, :, :, :, pyr_ind + 1 :: 2, :]

#                     lac_data = lac_data
#                     pyr_data = pyr_data

#                     lac_data = np.abs(
#                         np.nanmean(
#                             lac_data[:, :, :, :, rep_range, :],
#                             axis=4,
#                             keepdims=True,
#                         )
#                     )
#                     pyr_data = np.abs(
#                         np.nanmean(
#                             pyr_data[:, :, :, :, rep_range, :],
#                             axis=4,
#                             keepdims=True,
#                         )
#                     )

#                     pyr_signal_mask = pyr_data > background

#                     bssfp_image_data = lac_data / pyr_data
#                     bssfp_image_data = bssfp_image_data * pyr_signal_mask
#                     rep_range = 0
#                 # use start repetition
#                 else:
#                     rep_range = rep1
#             # take start repetition value:
#             else:
#                 rep_range = rep1
#                 bssfp_image_data = lac_data / pyr_data
#         else:
#             bssfp_image_data = pyr_data
#             Warning("unknown Contrast" + str(plot_opts_metabolite.value))

#         # takes mean over both repetitions:
#         if rep_avg:
#             # valid range:
#             if rep2 > rep1:
#                 # if ratio is chosen, the averaging has already been done
#                 if (pyr_lac == "Lac/Pyr") and (experiment_style == "Pyr + Lac"):
#                     pass

#                 else:
#                     rep_range = range(rep1, rep2)
#                     # assuming the order Echoes, Read, phase, slice,
#                     # repetitions, channels in the data:
#                     bssfp_image_data = np.mean(
#                         bssfp_image_data[:, :, :, :, rep_range, :],
#                         axis=4,
#                         keepdims=True,
#                     )
#                     rep_range = 0
#             # use start repetition
#             else:
#                 rep_range = rep1
#         # take start repetition value:
#         else:
#             rep_range = rep1

#         # generate a custom cmap:
#         map_name, cmap = self.generate_custom_cmap(cmap_name=cmap)

#         # clear axis:
#         for ax in axlist:
#             ax.clear()

#         # to avoid confusion as this is probably not universal (different read directions
#         # and patient_pos)... define this here so it can be made depending on those paramters:
#         # logger.critical(slice_nr_ax)
#         # slice_nr_ax_bssfp = ax_slice_calc(
#         #     nr_slices=axial_image.shape[1], input_index=slice_nr_ax
#         # )

#         # this has to be done to match the orientation of the coronal
#         # images. Unfortunately this is a bit confusing
#         slice_nr_ax_bssfp = -slice_nr_ax - 1
#         if axial_obj is None:
#             # TODO find better default:
#             anat_ax_slice_ind = None
#         else:
#             (
#                 anat_ax_slice_ind,
#                 _,
#                 _,
#             ) = self.find_closest_slice2(
#                 grid1=bssfp_grid,
#                 pos1=[
#                     slice_nr_ax_bssfp,
#                     slice_nr_sag,
#                     slice_nr_cor,
#                 ],
#                 grid2_obj=axial_obj,
#             )

#         if coronal_obj is None:
#             anat_cor_slice_ind = None
#         else:
#             (
#                 _,
#                 _,
#                 anat_cor_slice_ind,
#             ) = self.find_closest_slice2(
#                 grid1=bssfp_grid,
#                 pos1=[
#                     slice_nr_ax_bssfp,
#                     slice_nr_sag,
#                     slice_nr_cor,
#                 ],
#                 grid2_obj=coronal_obj,
#             )

#         # if axial and or coronal grid are not defined, use bssfp indices:
#         if anat_ax_slice_ind is None:
#             # roughly choose similar slice position
#             if axial_image is not None:
#                 anat_ax_slice_ind = int(
#                     slice_nr_ax_bssfp
#                     * axial_image.shape[1]
#                     / bssfp_image_data.shape[1]
#                 )
#             else:
#                 anat_ax_slice_ind = bssfp_image_data.shape[1]
#         if anat_cor_slice_ind is None:
#             if coronal_image is not None:
#                 # roughly choose similar slice position
#                 anat_cor_slice_ind = int(
#                     slice_nr_cor
#                     * coronal_image.shape[3]
#                     / bssfp_image_data.shape[3]
#                 )
#             else:
#                 anat_cor_slice_ind = bssfp_image_data.shape[3]

#         # flip index: dim_slice_bssfp
#         # logger.critical(coronal_image.shape)
#         # anat_cor_slice_ind = coronal_image.shape[3] - anat_cor_slice_ind + 1
#         # slice_nr_cor = dim_slice_bssfp - slice_nr_cor + 1

#         # generate mask for axial image:
#         mask_ax = np.rot90(
#             np.squeeze(
#                 (
#                     bssfp_image_data[
#                         echo_nr,
#                         slice_nr_ax_bssfp,
#                         :,
#                         :,
#                         rep_range,
#                         chan,
#                     ]
#                 )
#             )
#         ).copy()
#         mask_ax[mask_ax < metab_clim[0]] = 0
#         mask_ax[mask_ax > metab_clim[0]] = 1

#         # generate mask for axial image:
#         mask_cor = np.squeeze(
#             (bssfp_image_data[echo_nr, :, :, slice_nr_cor, rep_range, chan])
#         ).copy()

#         mask_cor[mask_cor < metab_clim[0]] = 0
#         mask_cor[mask_cor > metab_clim[0]] = 1

#         # function to perform image interpolation:
#         def img_interp(
#             metab_image=None, interp_factor=None, cmap=None, overlay=True
#         ):
#             # skip interpolation if interpolation factor is 1
#             if interp_factor == 1:
#                 interpolated = metab_image
#             else:
#                 # partially taken from:
#                 # https://stackoverflow.com/questions/52419509/get-interpolated-data-from-imshow
#                 import matplotlib.pyplot as plt
#                 from matplotlib.image import _resample
#                 from matplotlib.transforms import Affine2D

#                 # define new image size:
#                 out_dimensions = (
#                     metab_image.shape[0] * interp_factor,
#                     metab_image.shape[1] * interp_factor,
#                 )

#                 # needs to be scale later:
#                 metab_image_max = np.max(metab_image)
#                 # normalize (necessary for _resample)
#                 metab_image = metab_image / metab_image_max

#                 # generate axis object
#                 _, axs = plt.subplots(1)
#                 transform = Affine2D().scale(interp_factor, interp_factor)
#                 img = axs.imshow(
#                     metab_image, interpolation=interp_method, cmap=cmap
#                 )
#                 interpolated = _resample(
#                     img, metab_image, out_dimensions, transform=transform
#                 )
#                 # clear axis
#                 axs.clear()
#                 plt.close()
#                 # rescale
#                 interpolated = interpolated * metab_image_max
#                 # apply mask
#                 if overlay is True:
#                     interpolated[interpolated <= metab_clim[0]] = np.nan

#             return interpolated

#         if fieldmap is not None:
#             (
#                 ax_fm_ind,
#                 sag_fm_ind,
#                 cor_fm_ind,
#             ) = self.find_closest_slice2(
#                 grid1=bssfp_grid,
#                 grid2=fieldmap_grid,
#                 pos1=[slice_nr_ax_bssfp, slice_nr_sag, slice_nr_cor],
#             )

#             df_b0 = np.squeeze(fm[0, ax_fm_ind, sag_fm_ind, cor_fm_ind, 0, 0])
#         else:
#             df_b0 = None

#         # metabolic axial image:
#         metab_image = np.flip(
#             np.squeeze(
#                 (
#                     bssfp_image_data[
#                         echo_nr,
#                         slice_nr_ax_bssfp,
#                         :,
#                         :,
#                         rep_range,
#                         chan,
#                     ]
#                 )
#             ),
#             axis=1,
#         )

#         # Interpolate the image by a factor:
#         interpolated = img_interp(
#             metab_image=metab_image,
#             interp_factor=interp_factor,
#             cmap=cmap,
#             overlay=True,
#         )

#         # if the colorange shoud be set automcatically:
#         if checkbox_auto_set_crange.value:
#             # stupid but should do the trick for now:
#             metab_clim = list(metab_clim)
#             metab_clim[0] = np.nanmin(interpolated)
#             metab_clim[1] = np.nanmax(interpolated)

#             # don't show negative number in case of magntiude display
#             # style:
#             if plot_style == "Abs":
#                 metab_clim[0] = 0
#             metab_clim = tuple(metab_clim)

#         # axial view: -----------------------------------------------------
#         if (anat_overlay == "Metab + Anat") and (axial_image is not None):
#             # anatomical axial image:
#             axlist[0].imshow(
#                 np.squeeze(np.rot90(axial_image[0, anat_ax_slice_ind, :, :, 0, 0])),
#                 extent=self.ax_ext,
#                 cmap="bone",
#             )

#             im_ax = axlist[0].imshow(
#                 np.rot90(interpolated),
#                 extent=self.ax_ext,
#                 cmap=cmap,
#                 alpha=alpha_overlay,
#                 vmin=metab_clim[0],
#                 vmax=metab_clim[1],
#             )

#         elif anat_overlay == "Metab":
#             # metabolic axial image:
#             im_ax = axlist[0].imshow(
#                 np.rot90(interpolated),
#                 extent=self.ax_ext,
#                 cmap=cmap,
#                 vmin=metab_clim[0],
#                 vmax=metab_clim[1],
#             )

#         elif (anat_overlay == "Anat") and (axial_image is not None):
#             # anatomical axial image:
#             axlist[0].imshow(
#                 np.rot90(np.squeeze(axial_image[0, anat_ax_slice_ind, :, :, 0, 0])),
#                 extent=self.ax_ext,
#                 cmap="bone",
#             )

#         else:  # default show metab image
#             im_ax = axlist[0].imshow(
#                 np.rot90(interpolated),
#                 extent=self.ax_ext,
#                 cmap=cmap,
#                 alpha=alpha_overlay * mask_ax,
#                 vmin=metab_clim[0],
#                 vmax=metab_clim[1],
#             )

#         # generate colorbar if fig object was passed:
#         if fig is None:
#             # colorbar needs figure:
#             pass
#         elif (fig is not None) and (anat_overlay != "Anat"):
#             # nasty but otherwise plots new colorbar each time a paramter
#             # is changed :/
#             global cbar
#             try:
#                 cbar.remove()
#             except BaseException:
#                 pass
#             cbar = fig.colorbar(im_ax, ax=axlist[0], fraction=0.033, pad=0.04)
#             cbar.remove()
#             cbar = fig.colorbar(
#                 im_ax,
#                 ax=axlist[0],
#                 fraction=0.033,
#                 pad=0.04,
#             )
#             # set colorbar range, use 3 ticks (min, mean, max)
#             md = (metab_clim[1] + metab_clim[0]) / 2
#             cbar.set_ticks([metab_clim[0], md, metab_clim[1]])
#             cbar.set_ticklabels(
#                 np.round([metab_clim[0], md, metab_clim[1]]).astype(int)
#             )
#         else:
#             # dont plot colorbar if no metabolic data:
#             try:
#                 # skip for now because this changes the size of the plotted images
#                 # which can be quite annoying:
#                 pass
#                 # remove if left over from previos plot:
#                 # cbar.remove()
#             except BaseException:
#                 pass

#         # plot lines:
#         axlist[0].axhline(
#             ax_ext_slice[slice_nr_cor],
#             color="white",
#             linewidth=0.25,
#             linestyle="solid",
#         )
#         # plot lines:
#         axlist[0].axhline(
#             y=ax_ext_slice[slice_nr_cor] - res_anat_slice_plot / 2,
#             xmin=0,  # from 0 to 1 (1= all to the right)
#             xmax=0.1,  # from 0 to 1 (1= all to the right)
#             color="white",
#             linewidth=0.45,
#             linestyle="--",
#         )
#         # plot lines:
#         axlist[0].axhline(
#             ax_ext_slice[slice_nr_cor] + res_anat_slice_plot / 2,
#             xmin=0,  # from 0 to 1 (1= all to the right)
#             xmax=0.1,  # from 0 to 1 (1= all to the right)
#             color="white",
#             linewidth=0.45,
#             linestyle="--",
#         )

#         axlist[0].axvline(
#             ax_ext_phase[
#                 slice_nr_sag
#             ],  # mm_phase_bssfp / 2 - slice_nr_sag * slice_thick_sag_bssfp,
#             color="white",
#             linewidth=0.25,
#             linestyle="dashed",
#         )

#         # plot labels:
#         axlist[0].set_xlabel("mm (phase)")
#         axlist[0].set_ylabel("mm (slice)")
#         if df_b0 is not None:
#             axlist[0].set_title(
#                 "Axial "
#                 + str(anat_ax_slice_ind)
#                 + " df = "
#                 + str(round(df_b0, 1))
#                 + " Hz"
#             )

#         else:
#             axlist[0].set_title("Axial ")

#         # sagittal view:
#         # axlist[1].imshow(
#         #     np.squeeze(
#         #         (
#         #             bssfp_image_data[
#         #                 echo_nr, :, slice_nr_sag, :, rep_range, chan
#         #             ]
#         #         )
#         #     ),
#         #     extent=self.sag_ext,
#         #     **parameter_dict,
#         # )
#         # axlist[1].set_xlabel("mm (read)")
#         # axlist[1].set_ylabel("mm (slice)")

#         # axlist[1].set_title("Sagittal")
#         # # axlist[0].set_aspect(dim_y / dim_z)

#         # axlist[1].axhline(
#         #     mm_slice_bssfp / 2 - slice_nr_cor * slice_thick_cor_bssfp,
#         #     color="white",
#         #     linewidth=0.25,
#         #     linestyle="solid",
#         # )
#         # axlist[1].axvline(
#         #     mm_read_bssfp / 2 - slice_nr_ax * slice_thick_ax_bssfp,
#         #     color="white",
#         #     linewidth=0.25,
#         #     linestyle="dashed",
#         # )
#         # metabolic coronal image:
#         metab_image = np.squeeze(
#             (bssfp_image_data[echo_nr, :, :, slice_nr_cor, rep_range, chan])
#         )

#         # Interpolate the image by a factor:
#         interpolated = img_interp(
#             metab_image=metab_image,
#             interp_factor=interp_factor,
#             cmap=cmap,
#             overlay=True,
#         )

#         # if the colorange shoud be set automcatically:
#         if checkbox_auto_set_crange.value:
#             # stupid but should do the trick for now:
#             metab_clim = list(metab_clim)
#             metab_clim[0] = np.nanmin(interpolated)
#             metab_clim[1] = np.nanmax(interpolated)

#             # don't show negative number in case of magntiude display
#             # style:
#             if plot_style == "Abs":
#                 metab_clim[0] = 0
#             metab_clim = tuple(metab_clim)

#         # coronal view: -------------------------------------------
#         if (anat_overlay == "Metab + Anat") and (coronal_image is not None):
#             # anatomical coronal image:
#             axlist[1].imshow(
#                 np.squeeze((coronal_image[0, :, :, anat_cor_slice_ind, 0, 0])),
#                 extent=self.cor_ext,
#                 cmap="bone",
#             )

#             axlist[1].imshow(
#                 interpolated,
#                 extent=self.cor_ext,
#                 alpha=alpha_overlay,
#                 cmap=cmap,
#                 vmin=metab_clim[0],
#                 vmax=metab_clim[1],
#             )
#         elif anat_overlay == "Metab":
#             axlist[1].imshow(
#                 interpolated,
#                 extent=self.cor_ext,
#                 cmap=cmap,
#                 vmin=metab_clim[0],
#                 vmax=metab_clim[1],
#             )
#         elif (anat_overlay == "Anat") and (coronal_image is not None):
#             # anatomical coronal image:
#             axlist[1].imshow(
#                 np.squeeze((coronal_image[0, :, :, anat_cor_slice_ind, 0, 0])),
#                 extent=self.cor_ext,
#                 cmap="bone",
#             )
#         else:  # default show metabolic image
#             axlist[1].imshow(
#                 interpolated,
#                 extent=self.cor_ext,
#                 alpha=alpha_overlay * mask_cor,
#                 cmap=cmap,
#                 vmin=metab_clim[0],
#                 vmax=metab_clim[1],
#             )

#         # plot lines:
#         axlist[1].axhline(
#             cor_ext_read[
#                 slice_nr_ax
#             ],  # mm_read_bssfp / 2 - slice_nr_ax * slice_thick_ax_bssfp - 0.5,
#             color="white",
#             linewidth=0.25,
#             linestyle="solid",
#         )
#         # plot lines:
#         axlist[1].axhline(
#             y=cor_ext_read[slice_nr_ax] - res_anat_read_plot / 2,
#             xmin=0,  # from 0 to 1 (1= all to the right)
#             xmax=0.1,  # from 0 to 1 (1= all to the right)
#             color="white",
#             linewidth=0.45,
#             linestyle="--",
#         )
#         # plot lines:
#         axlist[1].axhline(
#             cor_ext_read[slice_nr_ax] + res_anat_read_plot / 2,
#             xmin=0,  # from 0 to 1 (1= all to the right)
#             xmax=0.1,  # from 0 to 1 (1= all to the right)
#             color="white",
#             linewidth=0.45,
#             linestyle="--",
#         )
#         axlist[1].axvline(
#             cor_ext_phase[
#                 slice_nr_sag
#             ],  # mm_phase_bssfp / 2 -  * slice_thick_sag_bssfp,
#             color="white",
#             linewidth=0.25,
#             linestyle="dashed",
#         )

#         # set labels:
#         axlist[1].set_xlabel("mm (phase)")
#         axlist[1].set_ylabel("mm (read)")
#         axlist[1].set_title("Coronal - " + str(anat_cor_slice_ind))

#         # plot time curve ---------------------------------------------
#         if len(axlist) > 2:
#             # pyruvate signal only:
#             if pyr_lac == "Pyr":
#                 axlist[2].plot(
#                     np.squeeze(
#                         (
#                             pyr_data[
#                                 echo_nr,
#                                 slice_nr_ax_bssfp,
#                                 slice_nr_sag,
#                                 slice_nr_cor,
#                                 :,
#                                 chan,
#                             ]
#                         )
#                     ),
#                     linewidth=2,
#                 )
#                 # only possible if pyruvate and lactate were measured
#                 if experiment_style == "Pyr + Lac":
#                     # add lactate:
#                     axlist[2].plot(
#                         np.squeeze(
#                             (
#                                 lac_data[
#                                     echo_nr,
#                                     slice_nr_ax_bssfp,
#                                     slice_nr_sag,
#                                     slice_nr_cor,
#                                     :,
#                                     chan,
#                                 ]
#                             )
#                         ),
#                         linewidth=1,
#                     )
#                 if experiment_style == "3 Metabolites":
#                     # add lactate:
#                     axlist[2].plot(
#                         np.squeeze(
#                             (
#                                 lac_data[
#                                     echo_nr,
#                                     slice_nr_ax_bssfp,
#                                     slice_nr_sag,
#                                     slice_nr_cor,
#                                     :,
#                                     chan,
#                                 ]
#                             )
#                         ),
#                         linewidth=1,
#                     )

#                     # add PyruvateHydrate:
#                     axlist[2].plot(
#                         np.squeeze(
#                             (
#                                 thirdMetab_data[
#                                     echo_nr,
#                                     slice_nr_ax_bssfp,
#                                     slice_nr_sag,
#                                     slice_nr_cor,
#                                     :,
#                                     chan,
#                                 ]
#                             )
#                         ),
#                         linewidth=1,
#                     )
#             # only possible if pyruvate and lactate were measured
#             elif (
#                 experiment_style == "Pyr + Lac"
#                 or experiment_style == "3 Metabolites"
#             ) and (pyr_lac == "Lac"):
#                 axlist[2].plot(
#                     np.squeeze(
#                         (
#                             pyr_data[
#                                 echo_nr,
#                                 slice_nr_ax_bssfp,
#                                 slice_nr_sag,
#                                 slice_nr_cor,
#                                 :,
#                                 chan,
#                             ]
#                         )
#                     ),
#                     linewidth=1,
#                 )
#                 # highlight lactate (linewidth):
#                 axlist[2].plot(
#                     np.squeeze(
#                         (
#                             lac_data[
#                                 echo_nr,
#                                 slice_nr_ax_bssfp,
#                                 slice_nr_sag,
#                                 slice_nr_cor,
#                                 :,
#                                 chan,
#                             ]
#                         )
#                     ),
#                     linewidth=2,
#                 )

#                 if experiment_style == "3 Metabolites":
#                     axlist[2].plot(
#                         np.squeeze(
#                             (
#                                 thirdMetab_data[
#                                     echo_nr,
#                                     slice_nr_ax_bssfp,
#                                     slice_nr_sag,
#                                     slice_nr_cor,
#                                     :,
#                                     chan,
#                                 ]
#                             )
#                         ),
#                         linewidth=1,
#                     )

#             # Plot ratio:
#             elif (experiment_style == "Pyr + Lac") and (pyr_lac == "Lac/Pyr"):
#                 axlist[2].plot(
#                     np.squeeze(
#                         (
#                             lac_data[
#                                 echo_nr,
#                                 slice_nr_ax_bssfp,
#                                 slice_nr_sag,
#                                 slice_nr_cor,
#                                 :,
#                                 chan,
#                             ]
#                         )
#                     )
#                     / np.squeeze(
#                         (
#                             pyr_data[
#                                 echo_nr,
#                                 slice_nr_ax_bssfp,
#                                 slice_nr_sag,
#                                 slice_nr_cor,
#                                 :,
#                                 chan,
#                             ]
#                         )
#                     ),
#                     linewidth=1,
#                 )

#             # if also PyruvateHydrate was measured
#             elif (experiment_style == "3 Metabolites") and (
#                 pyr_lac == "3rd Metabolite"
#             ):
#                 axlist[2].plot(
#                     np.squeeze(
#                         (
#                             pyr_data[
#                                 echo_nr,
#                                 slice_nr_ax_bssfp,
#                                 slice_nr_sag,
#                                 slice_nr_cor,
#                                 :,
#                                 chan,
#                             ]
#                         )
#                     ),
#                     linewidth=1,
#                 )

#                 axlist[2].plot(
#                     np.squeeze(
#                         (
#                             lac_data[
#                                 echo_nr,
#                                 slice_nr_ax_bssfp,
#                                 slice_nr_sag,
#                                 slice_nr_cor,
#                                 :,
#                                 chan,
#                             ]
#                         )
#                     ),
#                     linewidth=1,
#                 )

#                 axlist[2].plot(
#                     np.squeeze(
#                         (
#                             thirdMetab_data[
#                                 echo_nr,
#                                 slice_nr_ax_bssfp,
#                                 slice_nr_sag,
#                                 slice_nr_cor,
#                                 :,
#                                 chan,
#                             ]
#                         )
#                     ),
#                     linewidth=2,
#                 )

#             else:
#                 Warning("dont know what to do")
#                 axlist[2].plot(
#                     np.squeeze(
#                         (
#                             pyr_data[
#                                 echo_nr,
#                                 slice_nr_ax_bssfp,
#                                 slice_nr_sag,
#                                 slice_nr_cor,
#                                 :,
#                                 chan,
#                             ]
#                         )
#                     ),
#                     linewidth=2,
#                 )

#             axlist[2].axvline(
#                 rep1,
#                 color="black",
#                 linewidth=1,
#                 linestyle="dashed",
#             )
#             # draw 2nd line:
#             if (rep2 > rep1) and (rep_avg_checkbox.value):
#                 axlist[2].axvline(
#                     rep2,
#                     color="black",
#                     linewidth=1,
#                     linestyle="dashed",
#                 )
#             else:
#                 axlist[2].axvline(
#                     rep2,
#                     color="black",
#                     linewidth=0.25,
#                     linestyle="dashed",
#                 )

#             # axlist[2].plot(0, metab_clim[0], color="black", marker=".")
#             # axlist[2].plot(0, metab_clim[1], color="black", marker=".")
#             # axlist[2].set_ylim()
#             axlist[2].axhline(
#                 0,
#                 color="black",
#                 linewidth=0.5,
#                 linestyle="dashed",
#             )
#             axlist[2].set_box_aspect(1)

#     # --------------------------------------------------

#     global rep_avg_checkbox_option

#     # create interactive sliders for  slices in all dimensions
#     rep_avg_checkbox = widgets.Checkbox(description="Mean Reps")
#     rep_avg_checkbox_option = widgets.VBox(layout=widgets.Layout(display="flex"))
#     plot_images_individually_checkbox = widgets.Checkbox(
#         description="Ind.", layout=widgets.Layout(display="flex")
#     )
#     show_b0val_checkbox = widgets.Checkbox(description="Show B0")

#     # create interactive slider for echoes
#     echo_slider = widgets.IntSlider(
#         value=0, min=0, max=nechoes_bssfp - 1, description="Echo: "
#     )
#     slice_slider_rep1 = widgets.IntSlider(
#         value=0,
#         min=0,
#         max=reps_bssfp - 1,
#         description="Rep. Start: ",
#         layout=widgets.Layout(width="350px"),
#     )
#     slice_slider_rep2 = widgets.IntSlider(
#         value=0,
#         min=0,
#         max=reps_bssfp - 1,
#         description="Rep. End: ",
#         layout=widgets.Layout(width="350px"),
#     )

#     # set the minimum of the end reptition to the value of the start
#     # reptition

#     # set the minimum of the end reptition to the value of the start
#     # reptition
#     def rep_range_checker(args):
#         if slice_slider_rep1.value > slice_slider_rep2.value:
#             slice_slider_rep2.value = slice_slider_rep1.value
#         else:
#             pass
#         return True

#     slice_slider_rep1.observe(rep_range_checker, names="value")
#     slice_slider_rep2.observe(rep_range_checker, names="value")

#     if thirdMetab:
#         options_measurements = ["Perfusion", "Pyr + Lac", "3 Metabolites"]
#     else:
#         options_measurements = ["Perfusion", "Pyr + Lac"]

#     # Different plot options:
#     # lets you choose the type of experiment:
#     plot_opts_measurement = widgets.Dropdown(
#         options=options_measurements,
#         value="Perfusion",  # Defaults to 'pineapple'
#         description="Experiment:",
#         disabled=False,
#         layout=widgets.Layout(width="20%"),
#     )
#     # lets you choose the data shown:
#     plot_opts_part = widgets.Dropdown(
#         options=["Abs", "Real", "Imag", "Phase"],
#         value="Abs",
#         description="Plot style:",
#         disabled=False,
#         layout=widgets.Layout(width="10%"),
#     )

#     # lets you choose the metabolite shown:
#     plot_opts_metabolite = widgets.Dropdown(
#         options=["Pyr", "Lac", "3rd Metabolite"],
#         value="Pyr",
#         description="Metabolite:",
#         disabled=True,  # activate if plot_opts_measurement is set to Pyr+Lac
#         layout=widgets.Layout(width="15%"),
#     )
#     # lets you choose wether to show anatomical, metabolic or ratio:
#     if (coronal_image is None) and (axial_image is None):
#         plot_opts_anat_overlay = widgets.Dropdown(
#             options=["Metab", "Metab + Anat", "Anat"],
#             value="Metab",
#             description="Anat overlay",
#             disabled=True,
#             layout=widgets.Layout(width="20%"),
#         )
#     else:
#         plot_opts_anat_overlay = widgets.Dropdown(
#             options=["Metab", "Metab + Anat", "Anat"],
#             value="Metab",
#             description="Anat overlay",
#             disabled=False,
#             layout=widgets.Layout(width="20%"),
#         )
#     # lets you choose the overlay of the metabolic on the
#     # anatomical:
#     plot_opts_alpha_overlay = widgets.BoundedFloatText(
#         value=0.5,
#         min=0.0,
#         max=1.0,
#         step=0.1,
#         disabled=False,
#         description="alpha:",
#         layout=widgets.Layout(width="15%"),
#     )
#     # set plot colorrange of metabolites:
#     plot_opts_metab_clim = widgets.FloatRangeSlider(
#         min=-10.0,
#         value=[0, np.max(data)],
#         max=np.max(data),
#         description="Crange:",
#         disabled=False,
#         continuous_update=False,
#         orientation="horizontal",
#         readout=True,
#         readout_format=".1f",
#         layout=widgets.Layout(width="35%"),
#     )
#     # lets you ccoose colormaps:
#     plot_opts_cmap = widgets.Dropdown(
#         options=[
#             "cividis",
#             "inferno",
#             "magma",
#             "plasma",
#             "viridis",
#             "jet",
#             "Blues",
#             "BuGn",
#             "BuPu",
#             "GnBu",
#             "Greens",
#             "Greys",
#             "OrRd",
#             "Oranges",
#             "PuBu",
#             "PuBuGn",
#             "PuRd",
#             "Purples",
#             "RdPu",
#             "Reds",
#             "Wistia",
#             "YlGn",
#             "YlGnBu",
#             "YlOrBr",
#             "YlOrRd",
#             "afmhot",
#             "autumn",
#             "binary",
#             "bone",
#             "cool",
#             "copper",
#             "gist_gray",
#             "gist_heat",
#             "gist_yarg",
#             "rainbow",
#             "gray",
#             "hot",
#             "pink",
#             "spring",
#             "summer",
#             "winter",
#             "custom_cold",
#         ],
#         value="plasma",
#         description="Colormap:",
#         disabled=False,
#         layout=widgets.Layout(width="20%"),
#     )
#     # lets you ccoose interpolation methods:
#     plot_opts_interpolation_method = widgets.Dropdown(
#         options=[
#             "none",
#             "antialiased",
#             "nearest",
#             "bilinear",
#             "bicubic",
#             "spline16",
#             "spline36",
#             "hanning",
#             "hamming",
#             "hermite",
#             "kaiser",
#             "quadric",
#             "catrom",
#             "gaussian",
#             "bessel",
#             "mitchell",
#             "sinc",
#             "lanczos",
#             "blackman",
#         ],
#         value="bilinear",
#         description="Interp.method:",
#         disabled=False,
#         layout=widgets.Layout(width="20%"),
#     )

#     save_opts_fileformat_select = widgets.Select(
#         options=["svg", "png", "pickle"],
#         value="svg",
#         # rows=10,
#         description="FF:",
#         layout=widgets.Layout(width="8%"),
#         disabled=False,
#     )

#     plot_opts_interpolation_factor = widgets.BoundedIntText(
#         value=1,
#         min=1,
#         max=100,
#         step=1,
#         disabled=False,
#         description="Interp. fact:",
#         layout=widgets.Layout(width="10%"),
#     )
#     plot_opts_0_as_NaN = widgets.Checkbox(
#         value=False, description="NaN", disabled=False
#     )
#     # name of figure to save:
#     text_save_fig_name = widgets.Text(
#         value="fig_name.svg",
#         placeholder="Type something",
#         description="Name:",
#         disabled=False,
#         tooltip="name of figure (has to end with .png or .svg)",
#         layout=widgets.Layout(width="30%"),
#     )
#     # save figure (in cwd):
#     button_save_fig = widgets.Button(
#         description="Save fig.",
#         disabled=False,
#         button_style="",  # 'success', 'info', 'warning', 'danger' or ''
#         tooltip="Save the figure",
#         icon="check",  # (FontAwesome names without the `fa-` prefix)
#         layout=widgets.Layout(width="100px"),
#     )

#     text_background = widgets.FloatText(
#         value=0.0,
#         description="Background:",
#         disabled=False,
#         layout=widgets.Layout(width="12%"),
#     )
#     # Draw ROI in anatomical image:
#     checkbox_auto_set_crange = widgets.Checkbox(
#         description="ACR",
#         value=True,
#         tooltip="automatically sets the colorange to min <-> max of available values",
#         layout=widgets.Layout(width="10%"),
#     )

#     # def set_metab_clim():
#     #     if args["value"] == True:
#     #         plot_opts_metab_clim = widgets.FloatRangeSlider(
#     #         min=0.0,
#     #         value=[0, np.max(data)],
#     #         max=np.max(data),
#     #         description="M. range:",
#     #         disabled=False,
#     #         continuous_update=False,
#     #         orientation="horizontal",
#     #         readout=True,
#     #         readout_format=".1f",
#     #     else:
#     #         plot_opts_metab_clim = widgets.FloatRangeSlider(
#     #         min=0.0,
#     #         value=[0, np.max(data)],
#     #         max=np.max(data),
#     #         description="M. range:",
#     #         disabled=False,
#     #         continuous_update=False,
#     #         orientation="horizontal",
#     #         readout=True,
#     #         readout_format=".1f",
#     #     )

#     # rep_avg_checkbox.observe(set_metab_clim, names="value")

#     # init sliders:
#     slice_slider_ax = widgets.IntSlider(
#         value=mat_bssfp_read_plot // 2,
#         min=0,
#         max=mat_bssfp_read_plot - 1,
#         description="Slice Axial: ",
#     )
#     slice_slider_cor = widgets.IntSlider(
#         value=mat_bssfp_slice_plot // 2,
#         min=0,
#         max=mat_bssfp_slice_plot - 1,
#         description="Slice Coronal: ",
#     )
#     slice_slider_sag = widgets.IntSlider(
#         value=dim_phase_bssfp // 2,
#         min=0,
#         max=dim_phase_bssfp - 1,
#         description="Slice sagittal: ",
#     )
#     slice_slider_chan = widgets.IntSlider(
#         value=0,
#         min=0,
#         max=chans_bssfp - 1,
#         description="Channel: ",
#         layout=widgets.Layout(width="120px"),
#     )

#     def set_rep_range(args):
#         """
#         Sets repetition range and contrast options (Pyr, Lac, ...) depending
#         on the experiment type
#         """
#         # changed from perfusion to pyr+lac
#         if args["new"] == "Pyr + Lac" or args["new"] == "3 Metabolites":
#             if (np.ceil(reps_bssfp / 2) - 1) == 0:
#                 slice_slider_rep1.max = np.ceil(reps_bssfp / 2)
#                 slice_slider_rep2.max = np.ceil(reps_bssfp / 2)

#                 slice_slider_rep1.value = np.ceil(slice_slider_rep1.value / 2)
#                 slice_slider_rep2.value = np.ceil(slice_slider_rep2.value / 2)

#                 slice_slider_rep1.disabled = True
#                 slice_slider_rep2.disabled = True

#             else:
#                 slice_slider_rep1.disabled = False
#                 slice_slider_rep2.disabled = False

#                 slice_slider_rep1.max = np.ceil(reps_bssfp / 2) - 1
#                 slice_slider_rep2.max = np.ceil(reps_bssfp / 2) - 1

#                 slice_slider_rep1.value = np.ceil(slice_slider_rep1.value / 2)
#                 slice_slider_rep2.value = np.ceil(slice_slider_rep2.value / 2)

#             plot_opts_metabolite.disabled = False

#             if args["new"] == "Pyr + Lac":
#                 plot_opts_metabolite.options = ["Pyr", "Lac", "Lac/Pyr"]
#             else:
#                 plot_opts_metabolite.options = ["Pyr", "Lac", "3rd Metabolite"]

#         elif args["new"] == "Perfusion":
#             slice_slider_rep1.disabled = False
#             slice_slider_rep2.disabled = False

#             slice_slider_rep1.max = reps_bssfp - 1
#             slice_slider_rep2.max = reps_bssfp - 1

#             slice_slider_rep1.value = slice_slider_rep1.value * 2
#             slice_slider_rep2.value = slice_slider_rep2.value * 2

#             plot_opts_metabolite.value = "Pyr"
#             plot_opts_metabolite.disabled = True
#         else:
#             Warning("Unkown experiment type" + str(args["new"]))

#     # change options depending on experiment type:
#     plot_opts_measurement.observe(set_rep_range, names="value")

#     def adjust_colorrange(args):
#         # Should adjust the colorrange depening on the chosen data type
#         # (real, imag, phase, magnitude)

#         if args["new"] == "Abs":
#             plot_opts_metab_clim.max = 1.2 * np.max(np.abs(data))
#             plot_opts_metab_clim.min = 1.2 * np.min(np.abs(data))
#         if args["new"] == "Real":
#             plot_opts_metab_clim.max = 1.2 * np.max(np.real(data))
#             plot_opts_metab_clim.min = 1.2 * np.min(np.real(data))
#         if args["new"] == "Imag":
#             plot_opts_metab_clim.max = 1.2 * np.max(np.imag(data))
#             plot_opts_metab_clim.min = 1.2 * np.min(np.imag(data))
#         if args["new"] == "Phase":
#             plot_opts_metab_clim.max = 2 * np.max(np.angle(data))
#             plot_opts_metab_clim.min = 2 * np.min(np.angle(data))

#     plot_opts_part.observe(adjust_colorrange, names="value")

#     # combine widgets into a vertical box:
#     ui_colors = widgets.HBox(
#         [
#             plot_opts_alpha_overlay,
#             plot_opts_metab_clim,
#             checkbox_auto_set_crange,
#             plot_opts_cmap,
#             plot_opts_interpolation_method,
#             plot_opts_interpolation_factor,
#             text_background,
#         ],
#     )

#     # # combine widgets into a horizontal box::
#     ui_up_left = widgets.HBox(
#         [
#             plot_opts_measurement,
#             plot_opts_anat_overlay,
#             plot_opts_metabolite,
#             plot_opts_part,
#             text_save_fig_name,
#             button_save_fig,
#             plot_images_individually_checkbox,
#         ]
#     )
#     # ui_up_right = widgets.HBox([save_opts_fileformat_select])
#     # ui_up = widgets.VBox([ui_up_left, ui_up_right])
#     ui_up = ui_up_left
#     # # combine widgets and boxes into more boxes:
#     ui_slices = widgets.HBox([slice_slider_ax, slice_slider_sag, slice_slider_cor])
#     ui_reps = widgets.VBox([slice_slider_rep1, slice_slider_rep2, rep_avg_checkbox])
#     ui_reps_echoes_chans = widgets.HBox([echo_slider, slice_slider_chan, ui_reps])

#     def set_plot_params(self, params=None):
#         """
#         Set the plot parameters. Sets first the widget values to the parameter (if
#         ther are loaded values). Either way, after that the plot parameters are set to
#         the widget values
#         """
#         # sets the sliders to a specific value in case a parameter file was
#         # loaded (second value is default value if no value is defined)
#         try:
#             # set this first because it affects the repetition setting:
#             plot_opts_measurement.value = params.get(
#                 "experiment_style", "Perfusion"
#             )
#             slice_slider_ax.value = params.get("slice_nr_ax", 0)
#             slice_slider_cor.value = params.get("slice_nr_cor", 0)
#             slice_slider_sag.value = params.get("slice_nr_sag", 0)
#             echo_slider.value = params.get("echo_nr", 0)
#             slice_slider_rep1.value = params.get("rep1", 0)
#             slice_slider_chan.value = params.get("chan", 0)
#             plot_opts_alpha_overlay.value = params.get("alpha_overlay", 0.5)
#             slice_slider_rep2.value = params.get("rep2", 0)
#             rep_avg_checkbox.value = params.get("rep_avg", False)
#             plot_opts_part.value = params.get("plot_style", "Abs")
#             plot_opts_metabolite.value = params.get("pyr_lac", "Pyr")
#             plot_opts_anat_overlay.value = params.get("anat_overlay", "Metab")
#             plot_opts_metab_clim.value = params.get("metab_clim", np.inf)
#             plot_opts_cmap.value = params.get("cmap", "jet")
#             plot_opts_interpolation_method.value = params.get(
#                 "interp_method", "bilinear"
#             )
#             text_background.value = params.get("background", 0)
#             plot_opts_interpolation_factor.value = params.get("interp_factor", 1)
#             plot_images_individually_checkbox.value = params.get(
#                 "plot_images_individually", "False"
#             )
#         except BaseException:
#             pass

#         # set plotting parameters to slider values:
#         plot_img_params = {
#             "slice_nr_ax": slice_slider_ax,
#             "slice_nr_cor": slice_slider_cor,
#             "slice_nr_sag": slice_slider_sag,
#             "echo_nr": echo_slider,
#             "chan": slice_slider_chan,
#             "alpha_overlay": plot_opts_alpha_overlay,
#             "plot_style": plot_opts_part,
#             "experiment_style": plot_opts_measurement,
#             "rep1": slice_slider_rep1,
#             "rep2": slice_slider_rep2,
#             "rep_avg": rep_avg_checkbox,
#             "pyr_lac": plot_opts_metabolite,
#             "anat_overlay": plot_opts_anat_overlay,
#             "metab_clim": plot_opts_metab_clim,
#             "cmap": plot_opts_cmap,
#             "interp_method": plot_opts_interpolation_method,
#             "interp_factor": plot_opts_interpolation_factor,
#             "background": text_background,
#             "plot_images_individually": plot_images_individually_checkbox,
#             # "zero_as_NaN": plot_opts_0_as_NaN,
#         }
#         return plot_img_params

#     # Define the on_click function
#     def on_click(event):
#         if event.inaxes in axlist:
#             # axial image:
#             if event.inaxes == axlist[0]:
#                 # get plot parameters:
#                 plot_img_params = set_plot_params(self)

#                 # take click position:
#                 slice_x_sag = event.xdata
#                 slice_y_cor = event.ydata

#                 # define position:
#                 pos = (0, slice_x_sag, -slice_y_cor)

#                 # get bssfp grid:
#                 sag_min_ind = np.argmin(np.abs(bssfp_grid[1] - pos[1]))
#                 cor_min_ind = np.argmin(np.abs(bssfp_grid[2] - pos[2]))

#                 # setthe 2 plot parameters:
#                 plot_img_params["slice_nr_sag"].value = sag_min_ind
#                 plot_img_params["slice_nr_cor"].value = cor_min_ind
#                 # update plot parameter dict:
#                 plot_img_params = set_plot_params(self, params=plot_img_params)
#             # coronal image:
#             elif event.inaxes == axlist[1]:
#                 # get plot parameters:
#                 plot_img_params = set_plot_params(self)

#                 # take click position:
#                 slice_x_sag = event.xdata
#                 slice_y_ax = event.ydata

#                 # define position:
#                 pos = (slice_y_ax, slice_x_sag, 0)

#                 # get bssfp grid:
#                 sag_min_ind = np.argmin(np.abs(bssfp_grid[1] - pos[1]))
#                 ax_min_ind = np.argmin(np.abs(bssfp_grid[0] - pos[0]))

#                 # setthe 2 plot parameters:
#                 plot_img_params["slice_nr_sag"].value = sag_min_ind
#                 plot_img_params["slice_nr_ax"].value = ax_min_ind
#                 # update plot parameter dict:
#                 plot_img_params = set_plot_params(self, params=plot_img_params)
#             # time curve:
#             elif event.inaxes == axlist[2]:
#                 # get plot parameters:
#                 plot_img_params = set_plot_params(self)

#                 # so rep2 can also be shifted:
#                 rep1_before = plot_img_params["rep1"].value
#                 rep2_before = plot_img_params["rep2"].value

#                 # take click position:
#                 rep1_after = event.xdata

#                 # calculate difference:
#                 rep_diff = rep1_after - rep1_before

#                 # apply difference to rep2
#                 rep1 = rep1_after
#                 rep2 = rep2_before + rep_diff

#                 # set the 2 plot parameters:
#                 plot_img_params["rep1"].value = rep1
#                 plot_img_params["rep2"].value = rep2
#                 # update plot parameter dict:
#                 plot_img_params = set_plot_params(self, params=plot_img_params)

#             else:
#                 pass

#     # Connect the click event to the on_click function
#     fig.canvas.mpl_connect("button_press_event", on_click)
#     # define plot parameters:
#     plot_img_params = set_plot_params(self, params=plot_params)

#     # create a link to the plot_img function that is called whenever on the
#     # parameters in the {} list is changed:
#     out = widgets.interactive_output(
#         plot_img,
#         plot_img_params,
#     )
#     # main_ui = widgets.HBox([ui_ax, ui_cor, ui_sag])
#     main_ui = widgets.VBox([ui_up, ui_colors, ui_slices, ui_reps_echoes_chans])

#     # function that is called when save fig. button is pressed:
#     # has to be defined after plot_img_params were defined:
#     def save_figure(args, fig_name=None):
#         import os

#         if fig_name is None:
#             fig_name = text_save_fig_name.value[0:-4]
#         else:
#             pass

#         # get path to folder on above acquisition data:
#         if hasattr(self, "savepath"):
#             # FIXME this does not work properly
#             path_parentfolder = self.savepath
#         else:
#             # get path to folder on above acquisition data::
#             path_parentfolder = str(self.path)[0 : str(self.path).rfind("\\")]

#         # make it path if not:
#         if path_parentfolder[-1] != "/":
#             path_parentfolder + "/"

#         # save figure:
#         # fig = plt.gcf()
#         if 0 == 1:
#             import pickle

#             # with open(os.path.join("test.pickle")) as f:
#             #     pickle.dumps(fig, f, "wb")

#             with open(
#                 os.path.join(path_parentfolder, fig_name + ".pickle"),
#                 "wb",
#             ) as f:
#                 pickle.dump(fig, f)

#         plot_images_individually = plot_images_individually_checkbox.value

#         def save_subplot(ax, filename):
#             from matplotlib.transforms import Bbox

#             # Get the bounding box of the subplot in figure coordinates
#             bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
#                 fig.dpi_scale_trans.inverted()
#             )

#             # Pad the bounding box to ensure nothing is cut off
#             # You can adjust the padding as necessary
#             padding = 0.1  # Padding in inches
#             bbox_padded = Bbox.from_bounds(
#                 bbox.x0 - padding,
#                 bbox.y0 - padding,
#                 bbox.width + 2 * padding,
#                 bbox.height + 2 * padding,
#             )

#             # Save just the region of the figure that contains the subplot
#             fig.savefig(filename, bbox_inches=bbox_padded, transparent=True)

#         if plot_images_individually:
#             # Loop through each subplot and save it individually
#             for i, ax in enumerate(fig.get_axes()[:3]):
#                 # default order, might give wrong results if axlist < 3
#                 # elements
#                 if i == 0:
#                     subplot_name = "_axial"
#                 elif i == 1:
#                     subplot_name = "_coronal"
#                 elif i == 2:
#                     subplot_name = "_timecurve"
#                 else:
#                     subplot_name = f"_subplot_{i}"

#                 subplot_filename_svg = os.path.join(
#                     path_parentfolder, f"{fig_name}{subplot_name}.svg"
#                 )
#                 subplot_filename_png = os.path.join(
#                     path_parentfolder, f"{fig_name}{subplot_name}.png"
#                 )

#                 save_subplot(ax, subplot_filename_svg)
#                 save_subplot(ax, subplot_filename_png)

#         else:
#             # Save the whole figure as before
#             fig.savefig(os.path.join(path_parentfolder, fig_name + ".svg"))
#             fig.savefig(
#                 os.path.join(path_parentfolder, fig_name + ".png"), format="png"
#             )

#         # save plot infos (txt):
#         # with open(
#         #     os.path.join(path_parentfolder, text_save_fig_name.value[0:-3] + "txt"),
#         #     "w",
#         # ) as file:
#         #     for key, value in plot_img_params.items():
#         #         file.write(f"{key}: {value.value}\n")

#         # save plot infos (dict):
#         # init empty dict

#         plot_img_params_dict = {}
#         # fill dict:
#         for key, value in plot_img_params.items():
#             # print(value.value)
#             plot_img_params_dict[key] = value.value

#         # get figure size:
#         plot_img_params_dict["fig_size"] = (
#             fig.get_size_inches() * fig.dpi
#         ).tolist()

#         # save plotting parameters in json:
#         import json

#         json.dump(
#             plot_img_params_dict,
#             open(
#                 os.path.join(path_parentfolder, fig_name + ".json"),
#                 "w",
#             ),
#         )

#     import functools

#     # define callback for save figure button:
#     button_save_fig.on_click(functools.partial(save_figure, fig_name=None))

#     # This displays the Hbox containing the slider widgets.
#     if display_ui:
#         display(main_ui, out)
#         if save_fig:
#             save_figure(self, fig_name=fig_name)
#     else:
#         return main_ui
