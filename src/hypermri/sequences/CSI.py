from typing import Optional

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from tqdm.auto import tqdm

from hypermri.core.experiment import MRIExperiment

from ..analysis.spectroscopy.core import get_freq_axis, lorentzian
from ..utils.coordinates.grids import define_xyz_grid
from ..utils.coordinates.transformations import define_imagematrix_parameters
from ..utils.system.logging import LOG_MODES, debug_log, init_default_logger
from .base_spectroscopy import BaseSpectroscopy

# initialize logger
logger = init_default_logger(__name__)
logger.setLevel(LOG_MODES["Info"])
DEBUG_FILE = "/tmp/csi_debug.log"
DEBUG_LOG = True


def debug_log_csi(message, file_name=DEBUG_FILE, write_to_file=DEBUG_LOG):
    debug_log(message=message, file_name=file_name, write_to_file=write_to_file)


class CSI(BaseSpectroscopy):
    """Chemical shift imaging sequence with spectroscopy utilities.

    The class extends :class:`~hypermri.sequences.base_spectroscopy.BaseSpectroscopy`
    with helpers for channel-aware reconstructions (e.g. :meth:`combine_channels`),
    interactive signal filtering, and voxel-wise fitting visualisations used in the
    spectroscopy tutorials.
    """

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        # Extract sequence-specific kwargs before calling parent constructor
        reco_options = kwargs.pop("reco_options", {})

        super().__init__(mri_exp_obj, *args, **kwargs)

        # Check if FID data is available for CSI reconstruction
        # In minimal/lazy loading modes, we allow CSI to load even without FID data
        if not hasattr(self.handler, "fid") or self.handler.fid is None:
            logger.warning(
                "CSI: No FID data available - this may limit functionality but allowing initialization to continue"
            )
        else:
            try:
                self.reconstruct(reco_type="raw", **reco_options)
                logger.info("CSI: Raw reconstruction successful")
            except Exception as e:
                logger.error(f"CSI raw reconstruction failed: {e}")
                logger.error(
                    "CSI: Continuing with initialization but no reconstruction"
                )
                # Continue - this allows CSI to load successfully even if reconstruction fails

    # @property
    def Nsearchranges(self):
        return len(self.peaks)

    @property
    def data(self):
        """
        Return the primary data for CSI, preferring broadened data if available.
        Uses standardized data access patterns.
        """
        # Check for CSI-specific broadened data first
        if hasattr(self, "seq2d_broadend") and self.seq2d_broadend is not None:
            return self.seq2d_broadend

        # Fall back to primary data array
        if self.primary_data is not None:
            return self.primary_data

        # Legacy fallback
        if hasattr(self, "_legacy_seq2d") and self._legacy_seq2d is not None:
            return self._legacy_seq2d

        return None

    @property
    def data_array(self):
        """
        Alias for primary_data to maintain backward compatibility.
        CSI raw reconstruction stores data in reco_data, which primary_data accesses correctly.
        """
        return self.primary_data

    @data_array.setter
    def data_array(self, value):
        """Setter that delegates to the mixin's data_array setter."""
        # Call the parent's data_array setter
        BaseSpectroscopy.data_array.fset(self, value)

    def csi_fft(self, input_data=None, axes=(0, 2, 3), ift=False):
        """
        Quickly computes the fftshift(fft(fftshift(input_data))) along the input axis.
        If input_data is not provided, returns the fft along axis of the fid file

        Parameters
        ----------
        input_data : (complex) xarray (None)
            data that should be fourier-transformed.

        axes : tuple (0, 2, 3)
            axes along the fourier transform should be performed
            Axes are Spectroscopic, x, y.

        ift : use inverse Fourier transform and inverese fftshifts
        """

        if input_data is None:
            kspace_xarray = self.kspace_xarray
        else:
            kspace_xarray = input_data
        kspace_nparray = kspace_xarray.to_numpy()

        logger.debug("len(axes) = " + str(len(axes)))

        # init empty array:
        ft_kspace_nparray = kspace_nparray

        # perform ft along given axes
        try:
            if ift:
                for d in range(len(axes)):
                    ft_kspace_nparray = np.fft.ifftshift(
                        np.fft.ifft(
                            np.fft.ifftshift(
                                ft_kspace_nparray,
                                axes=(axes[d],),
                            ),
                            axis=axes[d],
                        ),
                        axes=(axes[d],),
                    )
                    logger.debug("ift along " + str(axes[d]))
            else:
                for d in range(len(axes)):
                    ft_kspace_nparray = np.fft.fftshift(
                        np.fft.fft(
                            np.fft.fftshift(
                                ft_kspace_nparray,
                                axes=(axes[d],),
                            ),
                            axis=axes[d],
                        ),
                        axes=(axes[d],),
                    )
                    logger.debug("ft along " + str(axes[d]))
        except BaseException:
            Warning("Could not perform ffts!")

        kspace_xarray.data = ft_kspace_nparray

        return kspace_xarray

    def plot_signal_decay(
        self,
        input_data: Optional[np.ndarray] = None,
        plot_params: Optional[dict] = None,
    ):
        """
        Interactive tool for analyzing signal decay and selecting optimal filter parameters.

        This function provides an interactive interface to visualize signal decay characteristics
        and determine optimal filtering parameters before reconstruction. It works with raw FID data
        and provides real-time preview of filter effects.

        Parameters
        ----------
        input_data : numpy.ndarray, optional
            The raw FID data array. If not provided, uses data from handler.

        Returns
        -------
        None
            The function updates `self.signal_end` and `self.filter_params` attributes
            which can then be used with filter_signal() and remove_noisy_kspace_samples().

        Notes
        -----
        This tool should be used as the first step in the processing workflow:
        1. Run plot_signal_decay() to analyze signal characteristics and select parameters
        2. Use the selected parameters with filter_signal() and/or remove_noisy_kspace_samples()
        3. Apply reconstruction with the processed data

        The interface shows:
        - Signal decay curves (mean absolute value and cumulative sum)
        - Original and filtered reconstruction previews
        - Interactive parameter sliders for different filter types
        - Click-to-select signal end point functionality
        """
        from IPython.display import display
        from ipywidgets import (
            Button,
            Checkbox,
            Dropdown,
            FloatSlider,
            HBox,
            IntSlider,
            Layout,
            VBox,
        )

        debug_log_csi(
            "============== plot_signal_decay was called! ============== \n",
        )

        current_image_freq_idx = None

        def _initialize_data_and_reconstruction():
            """Get raw data, setup reconstruction strategy, and prepare k-space data."""
            nonlocal input_data
            debug_log_csi(
                "--> Initializing data and reconstruction components\n _initialize_data_and_reconstruction()"
            )
            # Get raw data from handler if not provided
            if input_data is None:
                if (
                    hasattr(self.handler, "rawdatajob0")
                    and self.handler.rawdatajob0 is not None
                ):
                    input_data = self.handler.rawdatajob0.copy()
                elif hasattr(self.handler, "fid") and self.handler.fid is not None:
                    input_data = self.handler.fid.copy()
                else:
                    raise ValueError(
                        "No raw data available for analysis. Ensure CSI data is loaded properly."
                    )

            # Initialize reconstruction components
            original_kspace_array = None
            original_kspace_array_ft = None
            reco_strategy = None
            k_dist_norm = None
            k_dist = None

            # Check if reconstruction is possible
            if hasattr(self, "reconstruct") and hasattr(self, "handler"):
                try:
                    if (
                        hasattr(self, "kspace_xarray")
                        and self.kspace_xarray is not None
                    ):
                        original_kspace_array = self.kspace_xarray.copy()
                    elif hasattr(self, "reco_data") and self.reco_data is not None:
                        original_kspace_array = (
                            None  # Can't create k-space preview from reco data
                        )
                    else:
                        from ..recon_strategy import BrukerCsiRecoStrategy

                except Exception as e:
                    debug_log_csi(
                        f"Reconstruction preview failed ({e}), proceeding with signal analysis only"
                    )
                    original_kspace_array = None

            try:
                from ..recon_strategy import BrukerCsiRecoStrategy

                reco_strategy = BrukerCsiRecoStrategy()
            except:
                debug_log_csi(
                    "--> Could not create BrukerCsiRecoStrategy, proceeding with signal analysis only"
                )

            try:
                original_kspace_array = reco_strategy.sort_kspace(
                    handler=self.handler,
                    input_data=input_data,
                    write_to_object=False,
                )

                # Calculate k-space distances for spatial filtering
                try:
                    _, read_idx, phase_idx = reco_strategy.get_phase_encoding_order(
                        self.handler
                    )
                    # Center coordinates
                    center_x = self.handler.matrix[0] / 2
                    center_y = self.handler.matrix[1] / 2
                    # Calculate distances
                    k_dist = np.sqrt(
                        (read_idx - center_x) ** 2 + (phase_idx - center_y) ** 2
                    )

                    # Normalize distance
                    max_dist = np.max(k_dist)
                    if max_dist > 0:
                        k_dist_norm = k_dist / max_dist
                    else:
                        k_dist_norm = k_dist

                    if self.n_repetitions > 1 and k_dist_norm is not None:
                        k_dist_norm = np.tile(k_dist_norm, self.n_repetitions)
                except Exception as e:
                    debug_log_csi(f"Failed to calculate k-space distances: {e}")

            except Exception as e:
                debug_log_csi(f"--> k-space sorting failed: {e}")

            # Handle reconstruction preview if available
            if original_kspace_array is not None:
                try:
                    from hypermri.utils.coordinates.image_shifts import shift_image

                    original_kspace_array = shift_image(
                        input_data=original_kspace_array,
                        shift_method="phase",
                        data_obj=self,
                        force_shift=True,
                        domain="kspace",
                    )

                    if original_kspace_array.attrs["domain"] == "kspace":
                        original_kspace_array_ft = reco_strategy.set_reco_data(
                            kspace_xarray=original_kspace_array,
                            handler=self.handler,
                            axes_names=["read_idx", "phase_idx"],
                            write_to_object=False,
                            ift=False,
                        )
                    else:
                        original_kspace_array_ft = original_kspace_array.copy()

                    # Use xarray coordinate-based operations instead of axis indexing
                    original_kspace_array_ft = original_kspace_array_ft.roll(
                        read_idx=-1, phase_idx=-1
                    )

                    # NEW: FFT along acquisition dimension to get spectral data
                    if "acquisition_idx" in original_kspace_array_ft.dims:
                        acq_axis = original_kspace_array_ft.get_axis_num(
                            "acquisition_idx"
                        )
                        original_kspace_array_ft.data = np.fft.fftshift(
                            np.fft.fft(original_kspace_array_ft.data, axis=acq_axis),
                            axes=acq_axis,
                        )
                except Exception as e:
                    debug_log_csi(f"⚡️ Failed to create spatial domain preview: {e}")
                    original_kspace_array_ft = None

            return (
                input_data,
                original_kspace_array,
                original_kspace_array_ft,
                reco_strategy,
                k_dist,
                k_dist_norm,
            )

        def _prepare_signal_analysis():
            """Reshape data, compute signal metrics, and prepare for analysis."""
            debug_log_csi("--> Preparing signal analysis\n _prepare_signal_analysis()")
            # Reshape input_data for signal analysis
            n_acquisition = self.n_acquisition
            n_repetitions = self.n_repetitions

            if input_data.ndim == 1:
                n_total_points = len(input_data)
                n_acquisition = self.n_acquisition
                n_spatial_points = n_total_points // n_acquisition
                input_data_2d = np.reshape(
                    input_data, (n_spatial_points, n_acquisition)
                ).T
            else:
                input_data_2d = input_data.copy()
                try:
                    n_spatial_points = input_data_2d.size // n_acquisition
                except Exception:
                    n_acquisition = input_data_2d.shape[0]
                    n_spatial_points = input_data_2d.shape[1]

            # Group by repetitions if needed - store all repetitions for selection
            if n_repetitions > 1:
                n_points_per_rep = n_spatial_points // self.n_repetitions
                input_data_reps = np.reshape(
                    input_data_2d, (n_acquisition, self.n_repetitions, n_points_per_rep)
                )
                input_data_reps_orig = input_data_reps.copy()
                input_data_reps = np.mean(
                    input_data_reps[:, 0, :], axis=1, keepdims=True
                )
            else:
                input_data_reps = input_data_2d
                input_data_reps_orig = None
            debug_log_csi(
                f"Input data shape for signal analysis: {input_data_reps.shape}"
            )

            # Compute signal metrics
            if input_data_reps.ndim == 2:
                original_mean_abs_signal = np.mean(np.abs(input_data_reps), axis=0)
            else:
                original_mean_abs_signal = np.abs(input_data_reps)
            debug_log_csi(
                f"input_data_reps.ndim={input_data_reps.ndim}: original_mean_abs_signal.shape={original_mean_abs_signal.shape}"
            )
            mean_abs_signal = original_mean_abs_signal.copy()

            cumsum_signal = np.cumsum(mean_abs_signal)
            debug_log_csi(f"Initial cumsum_signal.shape={cumsum_signal.shape}")
            cumsum_signal = cumsum_signal / np.max(cumsum_signal)  # Normalize
            original_cumsum_signal = cumsum_signal.copy()
            n = len(cumsum_signal)

            return (
                input_data_2d,
                input_data_reps_orig,
                mean_abs_signal,
                cumsum_signal,
                n,
                n_acquisition,
                original_mean_abs_signal,
                original_cumsum_signal,
                n_spatial_points,
            )

        def _create_interactive_widgets(n):
            """Create all UI widgets with consolidated parameters."""
            debug_log_csi(
                "--> Creating interactive widgets\n _create_interactive_widgets()"
            )
            widget_configs = {
                "signal_end_slider": {
                    "value": n // 2,
                    "min": 0,
                    "max": n - 1,
                    "step": 1,
                    "description": "End Index",
                },
                "confirm_button": {"description": "Confirm Selection"},
                "filter_dropdown": {
                    "options": ["None", "Exponential", "Hamming", "Hann"],
                    "value": "None",
                    "description": "Filter Type",
                },
                "decay_constant_slider": {
                    "value": 0.01,
                    "min": 0.00,
                    "max": 0.01,
                    "step": 0.00001,
                    "description": "Decay Const",
                    "layout": Layout(height="100px", display="none"),
                    "readout_format": ".5f",
                },
                "alpha_slider": {
                    "value": 0.54,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "Alpha",
                    "layout": Layout(display="none"),
                },
                "center_position_slider": {
                    "value": n // 2,
                    "min": 0,
                    "max": n - 1,
                    "step": 1,
                    "description": "Center Pos",
                    "layout": Layout(display="none"),
                },
                "flipangle_correction_button": {
                    "description": "Flipangle correction",
                    "value": False,
                },
                "flipangle_slider": {
                    "value": self.flipangle_deg / 1.0,
                    "min": 0.0,
                    "max": 90.0,
                    "step": 0.5,
                    "description": "Flipangle",
                },
                "repetition_slider": {
                    "value": 0,
                    "min": 0,
                    "max": max(0, self.n_repetitions - 1),
                    "step": 1,
                    "description": "Repetition",
                    "layout": Layout(
                        display="none" if self.n_repetitions <= 1 else "block"
                    ),
                },
                "spatial_filter_dropdown": {
                    "options": ["None", "Gaussian", "Fermi"],
                    "value": "None",
                    "description": "Spatial Filter",
                },
                "spatial_width_slider": {
                    "value": 0.5,
                    "min": 0.05,
                    "max": 2.0,
                    "step": 0.05,
                    "description": "Spatial Width",
                    "layout": Layout(display="none"),
                },
            }

            # Create widgets based on type
            widget_types = {
                "signal_end_slider": IntSlider,
                "center_position_slider": IntSlider,
                "repetition_slider": IntSlider,
                "decay_constant_slider": FloatSlider,
                "alpha_slider": FloatSlider,
                "flipangle_slider": FloatSlider,
                "filter_dropdown": Dropdown,
                "flipangle_correction_button": Checkbox,
                "confirm_button": Button,
                "spatial_filter_dropdown": Dropdown,
                "spatial_width_slider": FloatSlider,
            }

            return {
                name: widget_types[name](**config)
                for name, config in widget_configs.items()
            }

        def _setup_plots_and_axes(
            n_acquisition,
            mean_abs_signal_original,
            mean_abs_signal_filtered,
            cumsum_signal_original,
            cumsum_signal_filtered,
            signal_end_slider,
        ):
            """Create matplotlib subplots and initialize all plot elements."""
            debug_log_csi("--> Setting up plots and axes\n _setup_plots_and_axes()")
            fig, ax = plt.subplots(3, 2, figsize=figsize)
            ax = ax.flatten()

            # Create signal plots with vertical lines
            (line_mean_abs_original,) = ax[0].plot(
                mean_abs_signal_original, label="Original"
            )
            (line_mean_abs_filtered,) = ax[0].plot(
                mean_abs_signal_filtered, label="Filtered", linestyle="--"
            )
            ax[0].set_title("Signal Decay (Mean Absolute Values)")
            ax[0].set_xlabel("Acquisition Index")
            ax[0].set_ylabel("Signal Amplitude")
            ax[0].legend(loc="best")
            vline_mean_abs = ax[0].axvline(
                x=signal_end_slider.value, color="r", linestyle="--"
            )

            (line_cumsum_original,) = ax[1].plot(
                cumsum_signal_original, label="Original"
            )
            (line_cumsum_filtered,) = ax[1].plot(
                cumsum_signal_filtered, label="Filtered", linestyle="--"
            )
            ax[1].set_title("Cumulative Signal Content")
            ax[1].set_xlabel("Acquisition Index")
            ax[1].set_ylabel("Normalized Cumulative Sum")
            ax[1].legend(loc="best")
            vline_cumsum = ax[1].axvline(
                x=signal_end_slider.value, color="r", linestyle="--"
            )

            # Get frequency axis for spectrum plotting
            try:
                from ..analysis.spectroscopy.core import get_freq_axis

                freq_axis = get_freq_axis(
                    data_obj=self, unit="ppm", npoints=n_acquisition
                )
            except Exception:
                freq_axis = np.arange(n_acquisition)

            # Initialize spectrum plots
            initial_spectrum = np.zeros(n_acquisition)
            (line_original_spectrum,) = ax[4].plot(freq_axis, np.real(initial_spectrum))
            (line_filtered_spectrum,) = ax[5].plot(freq_axis, np.real(initial_spectrum))

            # Add vertical lines for frequency selection
            vline_spectrum_original = ax[4].axvline(
                x=0, color="g", linestyle="--", visible=False
            )
            vline_spectrum_filtered = ax[5].axvline(
                x=0, color="g", linestyle="--", visible=False
            )

            spectrum_xlabel = (
                "Frequency (ppm)"
                if freq_axis.size == n_acquisition
                else "Acquisition Index"
            )
            for i, title in enumerate(
                [
                    "Original Spectrum (Highest Intensity Voxel)",
                    "Filtered Spectrum (Highest Intensity Voxel)",
                ]
            ):
                ax[4 + i].set_title(title)
                ax[4 + i].set_xlabel(spectrum_xlabel)
                ax[4 + i].set_ylabel("Signal Amplitude")
                if freq_axis.size == n_acquisition:
                    ax[4 + i].invert_xaxis()

                    # Set sensible x-limits to avoid excessive empty space
                    # Typical 13C range is around 160-200 ppm.
                    # We try to center on the data or a standard 13C window.
                    try:
                        data_min, data_max = np.min(freq_axis), np.max(freq_axis)
                        center_ppm = self.handler.get("center_ppm", 175)

                        # If center_ppm is in a typical 13C range, use a focused window
                        if 100 < center_ppm < 250:
                            # 40 ppm window is usually enough for hyperpolarized 13C
                            win_min, win_max = center_ppm - 25, center_ppm + 25
                            # Clip to actual axis range
                            ax[4 + i].set_xlim(
                                min(data_max, win_max), max(data_min, win_min)
                            )
                        else:
                            # Fallback: just use the data range if it's not huge
                            if (data_max - data_min) > 100:
                                # Still try to avoid 0 if possible
                                ax[4 + i].set_xlim(data_max, max(data_min, 100))
                    except:
                        pass

            plt.tight_layout()

            return {
                "fig": fig,
                "ax": ax,
                "freq_axis": freq_axis,
                "line_mean_abs_original": line_mean_abs_original,
                "line_mean_abs_filtered": line_mean_abs_filtered,
                "line_cumsum_original": line_cumsum_original,
                "line_cumsum_filtered": line_cumsum_filtered,
                "vline_mean_abs": vline_mean_abs,
                "vline_cumsum": vline_cumsum,
                "line_original_spectrum": line_original_spectrum,
                "line_filtered_spectrum": line_filtered_spectrum,
                "vline_spectrum_original": vline_spectrum_original,
                "vline_spectrum_filtered": vline_spectrum_filtered,
                "image_original": None,
                "image_original_filtered": None,
            }

        def _initialize_preview_images(plot_elements, original_kspace_array_ft):
            """Initialize reconstruction preview images."""
            debug_log_csi(
                "--> Initializing preview images\n _initialize_preview_images()"
            )
            if original_kspace_array_ft is not None:
                new_original_data = _to_display_image(original_kspace_array_ft)
                image_original = plot_elements["ax"][2].imshow(
                    new_original_data, cmap="gray"
                )
                plot_elements["ax"][2].set_title("Original Reconstruction")
                plot_elements["ax"][2].set_xlabel("Read Index")
                plot_elements["ax"][2].set_ylabel("Phase Index")

                filtered_image_ft = original_kspace_array_ft.copy()
                new_filtered_data = _to_display_image(filtered_image_ft)
                image_original_filtered = plot_elements["ax"][3].imshow(
                    new_filtered_data, cmap="gray"
                )
                plot_elements["ax"][3].set_title("Filtered Reconstruction")
                plot_elements["ax"][3].set_xlabel("Read Index")
                plot_elements["ax"][3].set_ylabel("Phase Index")

                # Set color limits
                try:
                    vmin = min(
                        np.nanmin(new_original_data), np.nanmin(new_filtered_data)
                    )
                    vmax = max(
                        np.nanmax(new_original_data), np.nanmax(new_filtered_data)
                    )
                    if vmin == vmax:
                        vmax = vmin + 1.0
                    image_original.set_clim(vmin=vmin, vmax=vmax)
                    image_original_filtered.set_clim(vmin=vmin, vmax=vmax)
                except Exception:
                    pass

                plot_elements["image_original"] = image_original
                plot_elements["image_original_filtered"] = image_original_filtered
                return filtered_image_ft
            else:
                # Show placeholder if reconstruction not available
                placeholder_text = "Reconstruction\nPreview\nNot Available"
                for i, title in enumerate(
                    ["Original Reconstruction", "Filtered Reconstruction"]
                ):
                    plot_elements["ax"][2 + i].text(
                        0.5,
                        0.5,
                        placeholder_text if i == 0 else "Filtered\n" + placeholder_text,
                        ha="center",
                        va="center",
                        transform=plot_elements["ax"][2 + i].transAxes,
                    )
                    plot_elements["ax"][2 + i].set_title(title)
                return None

        def _to_display_image(arr):
            """Robust conversion to 2D numpy array for imshow using xarray operations."""
            debug_log_csi("--> Converting to display image\n _to_display_image()")
            try:
                if arr is None:
                    return np.zeros((2, 2))

                # If xarray DataArray -> use coordinate-based operations
                if hasattr(arr, "dims") and hasattr(arr, "data"):
                    da = np.abs(arr) if hasattr(arr, "__abs__") else arr.copy()
                    if hasattr(da, "data"):
                        da.data = np.abs(da.data)

                    # Average all dims except spatial ones using xarray methods
                    keep_dims = {"read_idx", "phase_idx"}
                    avg_dims = [d for d in da.dims if d not in keep_dims]

                    if avg_dims:
                        try:
                            da = da.mean(dim=avg_dims)
                            final_dims = list(da.dims)
                            img = np.asarray(da.data)

                            if "read_idx" in final_dims and "phase_idx" in final_dims:
                                # Ensure proper orientation (phase, read) for imshow
                                if final_dims == ["read_idx", "phase_idx"]:
                                    img = img.transpose((1, 0))
                                else:
                                    try:
                                        p = final_dims.index("phase_idx")
                                        r = final_dims.index("read_idx")
                                        img = np.moveaxis(img, (p, r), (0, 1))
                                    except Exception:
                                        img = np.squeeze(img)
                            else:
                                img = np.squeeze(img)

                            img = np.squeeze(img)
                            if img.ndim == 2:
                                return img
                            elif img.ndim == 1:
                                side = int(np.ceil(np.sqrt(img.size)))
                                b = np.zeros((side, side), dtype=img.dtype)
                                b.flat[: img.size] = img
                                return b
                            elif img.ndim == 0:
                                return np.array([[img]])
                        except Exception:
                            pass  # Fall through to numeric fallback

                # Numeric fallback for non-xarray or failed xarray operations
                a = np.abs(np.asarray(getattr(arr, "data", arr)))
                if a.ndim > 2:
                    # Keep two largest axes, average the rest
                    axis_sizes = [(i, a.shape[i]) for i in range(a.ndim)]
                    sorted_axes = [
                        i
                        for i, _ in sorted(axis_sizes, key=lambda x: x[1], reverse=True)
                    ]
                    keep_axes = sorted_axes[:2]
                    axes_to_mean = tuple(i for i in range(a.ndim) if i not in keep_axes)
                    if axes_to_mean:
                        a = np.mean(a, axis=axes_to_mean)

                a = np.squeeze(a)
                if a.ndim == 2:
                    return a
                elif a.ndim == 1:
                    side = int(np.ceil(np.sqrt(a.size)))
                    b = np.zeros((side, side), dtype=a.dtype)
                    b.flat[: a.size] = a
                    return b
                elif a.ndim == 0:
                    return np.array([[a]])
                else:
                    # Final fallback - keep reducing dimensions
                    while a.ndim > 2:
                        a = np.mean(a, axis=-1)
                    return np.squeeze(a)
            except Exception:
                try:
                    out = np.squeeze(np.abs(np.asarray(getattr(arr, "data", arr))))
                    if out.ndim >= 2:
                        return (
                            out
                            if out.ndim == 2
                            else out.reshape((int(np.ceil(np.sqrt(out.size))), -1))[
                                :2, :2
                            ]
                        )
                    elif out.ndim == 1:
                        side = int(np.ceil(np.sqrt(out.size)))
                        b = np.zeros((side, side), dtype=out.dtype)
                        b.flat[: out.size] = out
                        return b
                except Exception:
                    pass
                return np.zeros((2, 2))

        def _get_representative_spectrum(data_array, rep_idx=0):
            """Extract spectrum from highest intensity voxel using xarray operations where possible."""
            debug_log_csi(
                "--> Extracting representative spectrum\n _get_representative_spectrum()"
            )
            debug_log_csi(
                f"rep_idx={rep_idx}, data_array.shape={getattr(data_array, 'shape', 'N/A')}"
            )
            from ..utils.mathematics.calculations import find_max_and_index

            try:
                if data_array is None:
                    return np.zeros(n_acquisition), (0, 0)

                # Data is already spectral (frequency domain)
                if hasattr(data_array, "data"):
                    data_abs = np.abs(data_array)
                    # Use xarray selection if available
                    if hasattr(data_array, "dims"):
                        _, max_ind = find_max_and_index(data_array)

                        # Map indices to dimension names
                        dim_indices = dict(zip(data_array.dims, max_ind))

                        # Remove acquisition_idx from selection to keep that dimension (the spectrum)
                        if "acquisition_idx" in dim_indices:
                            del dim_indices["acquisition_idx"]

                        # Handle repetition if present (override max finding for rep)
                        if "repetition_idx" in data_array.dims:
                            dim_indices["repetition_idx"] = rep_idx

                        spectrum = data_array.isel(dim_indices).squeeze().data

                        # Return indices relevant for spatial position (e.g. read/phase)
                        # This is for display purposes
                        display_inds = []
                        for d in ["read_idx", "phase_idx", "slice_idx"]:
                            if d in dim_indices:
                                display_inds.append(dim_indices[d])

                        return spectrum, tuple(display_inds)
                else:
                    data_abs = np.abs(data_array)

                # Fallback for numpy array or failed xarray logic
                _, max_ind = find_max_and_index(data_abs)
                debug_log_csi(f"Max index found at: {max_ind}")

                # Extract spectrum at max index
                # We use slice(None) for the first dimension (acquisition) and the specific indices for others
                # Assuming acquisition_idx is the first dimension (axis 0)
                slicer = [slice(None)] + list(max_ind[1:])

                if hasattr(data_array, "data"):
                    spectrum = data_array.data[tuple(slicer)].squeeze()
                else:
                    spectrum = data_array[tuple(slicer)].squeeze()

                debug_log_csi(f"Spectrum shape: {spectrum.shape}")
                return spectrum, max_ind

            except Exception as e:
                debug_log_csi(f"Error in get_representative_spectrum: {e}")
                return np.zeros(n_acquisition), (0, 0)

        def _collect_filter_parameters(widgets):
            """Centralized filter parameter collection to reduce code duplication."""
            debug_log_csi(
                "--> Collecting filter parameters\n _collect_filter_parameters()"
            )
            selected_filter = widgets["filter_dropdown"].value

            # Determine filter axis: 1 (spatial) for None/Truncation, 0 (time) for others
            filter_axis = 1 if selected_filter == "None" else 0

            filter_params = {
                "filter_type": selected_filter,
                "apply_flipangle_correction": widgets[
                    "flipangle_correction_button"
                ].value,
                "filter_axis": filter_axis,
            }

            if filter_params["apply_flipangle_correction"]:
                filter_params["flipangle_deg"] = widgets["flipangle_slider"].value

            filter_specific_params = {
                "Exponential": {
                    "decay_constant": widgets["decay_constant_slider"].value
                },
                "Hamming": {
                    "alpha": widgets["alpha_slider"].value,
                    "center_position": widgets["center_position_slider"].value,
                },
                "Hann": {"center_position": widgets["center_position_slider"].value},
                "None": {"signal_end": widgets["signal_end_slider"].value},
            }

            # Spatial filter parameters
            filter_params["spatial_filter_type"] = widgets[
                "spatial_filter_dropdown"
            ].value
            filter_params["spatial_width"] = widgets["spatial_width_slider"].value

            if selected_filter in filter_specific_params:
                filter_params.update(filter_specific_params[selected_filter])

            return filter_params, selected_filter

        # Ensure mutable defaults are created at runtime
        if plot_params is None:
            plot_params = {}

        # Extract figure size from plot_params or use default
        figsize = plot_params.get("figsize", (6, 6))

        # Initialize all components
        (
            input_data,
            original_kspace_array,
            original_kspace_array_ft,
            reco_strategy,
            k_dist,
            k_dist_norm,
        ) = _initialize_data_and_reconstruction()
        (
            input_data_2d,
            input_data_reps_orig,
            mean_abs_signal,
            cumsum_signal,
            n,
            n_acquisition,
            original_mean_abs_signal,
            original_cumsum_signal,
            n_spatial_points,
        ) = _prepare_signal_analysis()
        mean_abs_signal_original = original_mean_abs_signal.copy()
        cumsum_signal_original = original_cumsum_signal.copy()
        mean_abs_signal_filtered = mean_abs_signal.copy()
        cumsum_signal_filtered = cumsum_signal.copy()

        widgets = _create_interactive_widgets(n)
        plot_elements = _setup_plots_and_axes(
            n_acquisition,
            mean_abs_signal_original,
            mean_abs_signal_filtered,
            cumsum_signal_original,
            cumsum_signal_filtered,
            widgets["signal_end_slider"],
        )
        filtered_image_ft = _initialize_preview_images(
            plot_elements, original_kspace_array_ft
        )

        # Define update functions and event handlers
        def _update_signal():
            """Recompute signal analysis and filtered previews based on current parameters."""
            debug_log_csi("--> Updating signal analysis\n _update_signal()")
            # Use nonlocal to modify outer scope variables -> outer variables are influenced
            # by modifications in this function
            nonlocal mean_abs_signal_original
            nonlocal cumsum_signal_original
            nonlocal mean_abs_signal_filtered
            nonlocal cumsum_signal_filtered
            nonlocal n
            nonlocal filtered_image_ft
            nonlocal input_data_reps_orig

            filter_params, selected_filter = _collect_filter_parameters(widgets)
            current_rep = (
                widgets["repetition_slider"].value if self.n_repetitions > 1 else 0
            )

            # Apply Spatial Filtering if requested
            input_data_for_filtering = input_data_2d.copy()

            spatial_filter_type = filter_params.get("spatial_filter_type", "None")
            spatial_width = filter_params.get("spatial_width", 0.5)

            if k_dist_norm is not None and spatial_filter_type != "None":
                spatial_weights = np.ones_like(k_dist_norm)

                if spatial_filter_type == "Gaussian":
                    # Gaussian: exp(-0.5 * (r/sigma)^2)
                    spatial_weights = np.exp(-0.5 * (k_dist_norm / spatial_width) ** 2)
                elif spatial_filter_type == "Fermi":
                    # Fermi: 1 / (1 + exp((r - r0)/a)) - approximating with width
                    # Using width as cutoff radius, and fixed transition width
                    spatial_weights = 1.0 / (
                        1.0 + np.exp((k_dist_norm - spatial_width) / 0.05)
                    )

                # Apply weights to each time point (broadcasting over acquisition dim)
                input_data_for_filtering = (
                    input_data_for_filtering * spatial_weights[np.newaxis, :]
                )
                debug_log_csi(
                    f"Applied {spatial_filter_type} spatial filter with width {spatial_width}"
                )

            # Log filter-specific parameters
            filter_log_messages = {
                "apply_flipangle_correction": f"Flipangle degrees: {filter_params.get('flipangle_deg')}",
                "Exponential": f"Decay constant: {filter_params.get('decay_constant')}",
                "Hamming": f"Alpha: {filter_params.get('alpha')}, Center: {filter_params.get('center_position')}",
                "Hann": f"Center position: {filter_params.get('center_position')}",
                "None": f"Signal end: {filter_params.get('signal_end')}",
            }

            for condition, message in filter_log_messages.items():
                if (
                    condition == "apply_flipangle_correction"
                    and filter_params.get(condition)
                ) or (condition == selected_filter):
                    debug_log_csi(message)

            # Update signal analysis for current repetition
            if self.n_repetitions > 1 and input_data_reps_orig is not None:
                # To show k-space decay, average absolute values over acquisition dimension (axis=0)
                # Note: input_data_reps_orig is (n_acq, n_reps, n_spatial)
                # We want to show signal vs spatial index (k-space trajectory)
                # But plot_signal_decay usually plots signal vs acquisition time (FID decay).
                # The user mentioned "Signal Decay... I expect... exponential decay due to dephasing".
                # This usually refers to the FID decay (time domain).
                # However, the user also said "sequence start by encoding the central point and then coes out... I dont see an exponenital decay".
                # This implies they are looking at signal vs k-space index (spatial dimension).
                # Let's clarify: The plots created are:
                # ax[0]: "Signal Decay (Mean Absolute Values)" -> plot(mean_abs_signal)
                # mean_abs_signal was calculated as np.mean(np.abs(input_data), axis=0).
                # If input_data is (n_acq, n_spatial), axis=0 mean gives (n_spatial,).
                # So the x-axis of the plot IS the spatial index (k-space acquisition order).
                # So yes, we are plotting signal vs k-space index.

                # Original data for current rep
                current_rep_data = input_data_reps_orig[:, current_rep, :]
                mean_abs_signal_original = np.mean(np.abs(current_rep_data), axis=0)
            else:
                # To show k-space decay, average absolute values over acquisition dimension (axis=0)
                mean_abs_signal_original = np.mean(np.abs(input_data_2d), axis=0)

            # Recompute original signal analysis
            cumsum_signal_original = np.cumsum(mean_abs_signal_original)
            if np.max(cumsum_signal_original) > 0:
                cumsum_signal_original = cumsum_signal_original / np.max(
                    cumsum_signal_original
                )

            # Apply filtering for reconstruction preview
            filtered_data = None
            try:
                filtered_data = self.filter_signal(
                    input_data=input_data_for_filtering,
                    **filter_params,
                )
            except Exception as e:
                debug_log_csi(f"⚡️ Filter update failed: {e}")

            if filtered_data is not None:
                # Update filtered signal analysis
                try:
                    # filter_signal returns 1D array (block-by-block FIDs)
                    # We need to reshape it back to (n_acq, n_spatial) for analysis
                    filtered_data_2d = np.reshape(
                        filtered_data, (n_spatial_points, n_acquisition)
                    ).T

                    if self.n_repetitions > 1 and input_data_reps_orig is not None:
                        n_points_per_rep = n_spatial_points // self.n_repetitions
                        filtered_data_reps = np.reshape(
                            filtered_data_2d,
                            (n_acquisition, self.n_repetitions, n_points_per_rep),
                        )
                        # To show k-space decay, average absolute values over acquisition dimension (axis=0)
                        filtered_rep_data = np.mean(
                            np.abs(filtered_data_reps[:, current_rep, :]), axis=0
                        )
                    else:
                        # To show k-space decay, average absolute values over acquisition dimension (axis=0)
                        filtered_rep_data = np.mean(
                            np.abs(filtered_data_2d), axis=0, keepdims=False
                        )

                    mean_abs_signal_filtered = filtered_rep_data
                    cumsum_signal_filtered = np.cumsum(mean_abs_signal_filtered)
                    if np.max(cumsum_signal_filtered) > 0:
                        cumsum_signal_filtered = cumsum_signal_filtered / np.max(
                            cumsum_signal_filtered
                        )
                except Exception as e:
                    debug_log_csi(f"⚡️ Failed to update filtered signal metrics: {e}")
                    mean_abs_signal_filtered = mean_abs_signal_original.copy()
                    cumsum_signal_filtered = cumsum_signal_original.copy()
            else:
                mean_abs_signal_filtered = mean_abs_signal_original.copy()
                cumsum_signal_filtered = cumsum_signal_original.copy()

            if original_kspace_array_ft is not None and filtered_data is not None:
                try:
                    # Use xarray roll method to shift by -1 along both spatial dimensions
                    import copy

                    from ..recon_strategy import BrukerCsiRecoStrategy

                    reco_strategy = BrukerCsiRecoStrategy()

                    handler_filtered = copy.copy(self.handler)
                    handler_filtered.rawdatajob0 = filtered_data

                    filtered_image_ft = reco_strategy.reconstruct(
                        handler=handler_filtered
                    )

                    # NEW: FFT along acquisition dimension
                    if (
                        filtered_image_ft is not None
                        and "acquisition_idx" in filtered_image_ft.dims
                    ):
                        acq_axis = filtered_image_ft.get_axis_num("acquisition_idx")
                        filtered_image_ft.data = np.fft.fftshift(
                            np.fft.fft(filtered_image_ft.data, axis=acq_axis),
                            axes=acq_axis,
                        )
                except Exception as e:
                    debug_log_csi(f"⚡️ Reconstruction of filtered data failed: {e}")

        def _update_plot(end_segment_index):
            """Update all plot elements with current data and parameters."""
            debug_log_csi("--> Updating plots\n _update_plot()")
            _update_signal()

            # Update plot data
            try:
                plot_elements["line_mean_abs_original"].set_ydata(
                    mean_abs_signal_original
                )
                plot_elements["line_mean_abs_filtered"].set_ydata(
                    mean_abs_signal_filtered
                )
                plot_elements["line_cumsum_original"].set_ydata(cumsum_signal_original)
                plot_elements["line_cumsum_filtered"].set_ydata(cumsum_signal_filtered)
                plot_elements["vline_mean_abs"].set_xdata([end_segment_index])
                plot_elements["vline_cumsum"].set_xdata([end_segment_index])
            except Exception:
                pass

            # Update reconstruction preview images
            if (
                original_kspace_array_ft is not None
                and plot_elements["image_original"] is not None
            ):
                try:
                    # Slice if frequency selected
                    if current_image_freq_idx is not None:

                        def _slice_xarray(arr, idx):
                            if arr is None:
                                return None
                            if hasattr(arr, "dims"):
                                spatial_dims = {"read_idx", "phase_idx"}
                                spec_dims = [
                                    d for d in arr.dims if d not in spatial_dims
                                ]
                                if spec_dims:
                                    dim_name = spec_dims[0]
                                    if idx < arr.sizes[dim_name]:
                                        return arr.isel({dim_name: idx})
                            return arr

                        disp_original = _slice_xarray(
                            original_kspace_array_ft, current_image_freq_idx
                        )
                        disp_filtered = _slice_xarray(
                            filtered_image_ft, current_image_freq_idx
                        )
                    else:
                        disp_original = original_kspace_array_ft
                        disp_filtered = filtered_image_ft

                    new_original_data = _to_display_image(disp_original)
                    new_filtered_data = _to_display_image(disp_filtered)
                    plot_elements["image_original"].set_data(new_original_data)
                    plot_elements["image_original_filtered"].set_data(new_filtered_data)

                    # Update color limits
                    try:
                        vmin = min(
                            np.nanmin(new_original_data), np.nanmin(new_filtered_data)
                        )
                        vmax = max(
                            np.nanmax(new_original_data), np.nanmax(new_filtered_data)
                        )
                        if vmin == vmax:
                            vmax = vmin + 1.0
                        plot_elements["image_original"].set_clim(vmin=vmin, vmax=vmax)
                        plot_elements["image_original_filtered"].set_clim(
                            vmin=vmin, vmax=vmax
                        )
                    except Exception:
                        pass
                except Exception as e:
                    debug_log_csi(f"⚡️ Failed to update reconstruction previews: {e}")

            # Update representative spectra
            try:
                current_rep = (
                    widgets["repetition_slider"].value if self.n_repetitions > 1 else 0
                )

                if original_kspace_array_ft is not None:
                    original_spectrum, orig_pos = _get_representative_spectrum(
                        original_kspace_array_ft, current_rep
                    )
                    if filtered_image_ft is not None:
                        filtered_spectrum, filt_pos = _get_representative_spectrum(
                            filtered_image_ft, current_rep
                        )
                    else:
                        filtered_spectrum = np.zeros_like(original_spectrum)
                        filt_pos = orig_pos
                else:
                    original_spectrum = filtered_spectrum = np.zeros(n_acquisition)
                    orig_pos = filt_pos = (0, 0)

                debug_log_csi(
                    f"Original spectrum position: {orig_pos}, Filtered spectrum position: {filt_pos}"
                )
                # Update spectrum plots
                plot_elements["line_original_spectrum"].set_ydata(
                    np.real(original_spectrum)
                )
                plot_elements["line_filtered_spectrum"].set_ydata(
                    np.real(filtered_spectrum)
                )

                # Update titles
                rep_text = f" (Rep {current_rep})" if self.n_repetitions > 1 else ""
                plot_elements["ax"][4].set_title(
                    f"Original Spectrum{rep_text} - Voxel {orig_pos}"
                )
                plot_elements["ax"][5].set_title(
                    f"Filtered Spectrum{rep_text} - Voxel {filt_pos}"
                )

                # Adjust y-axis limits for spectra
                for spec_ax in [plot_elements["ax"][4], plot_elements["ax"][5]]:
                    spec_ax.relim()
                    # Only autoscale y-axis to preserve our custom x-limits
                    spec_ax.autoscale_view(scalex=False)
                debug_log_csi("Representative spectra updated successfully.")
            except Exception as e:
                debug_log_csi(f"⚡️ Failed to update representative spectra: {e}")

            # Update titles with current parameter values
            filter_type = widgets["filter_dropdown"].value
            current_rep = (
                widgets["repetition_slider"].value if self.n_repetitions > 1 else 0
            )
            rep_suffix = f" (Rep {current_rep})" if self.n_repetitions > 1 else ""

            title_formats = {
                "Exponential": (
                    f"Original{rep_suffix} (FA={widgets['flipangle_slider'].value:.1f}°)",
                    f"Filtered{rep_suffix} (decay={widgets['decay_constant_slider'].value:.3f})",
                ),
                "Hamming": (
                    f"Original{rep_suffix}",
                    f"Filtered{rep_suffix} (alpha={widgets['alpha_slider'].value:.2f})",
                ),
                "Hann": (f"Original{rep_suffix}", f"Filtered{rep_suffix} (Hann)"),
                "None": (
                    f"Original{rep_suffix}",
                    f"Truncated{rep_suffix} (end={end_segment_index})",
                ),
            }
            debug_log_csi(f"Updating titles for filter type: {filter_type}")

            if filter_type in title_formats:
                orig_title, filt_title = title_formats[filter_type]

                # Append frequency info if selected
                if current_image_freq_idx is not None:
                    freq_val = plot_elements["freq_axis"][current_image_freq_idx]
                    unit = (
                        "ppm"
                        if plot_elements["freq_axis"].size == n_acquisition
                        else "idx"
                    )
                    suffix = f"\n@ {freq_val:.2f} {unit}"
                    orig_title += suffix
                    filt_title += suffix

                plot_elements["ax"][2].set_title(orig_title)
                plot_elements["ax"][3].set_title(filt_title)
            debug_log_csi(f"Plot titles updated.")  # , plot_elements={plot_elements}")

            # Adjust plot limits and redraw
            try:
                for i in [0, 1]:
                    plot_elements["ax"][i].relim()
                    plot_elements["ax"][i].autoscale_view()
                    debug_log_csi(f"Axis {i} limits updated.")

                # Update frequency selection lines
                if current_image_freq_idx is not None:
                    freq_val = plot_elements["freq_axis"][current_image_freq_idx]
                    plot_elements["vline_spectrum_original"].set_xdata([freq_val])
                    plot_elements["vline_spectrum_filtered"].set_xdata([freq_val])
                    plot_elements["vline_spectrum_original"].set_visible(True)
                    plot_elements["vline_spectrum_filtered"].set_visible(True)
                else:
                    plot_elements["vline_spectrum_original"].set_visible(False)
                    plot_elements["vline_spectrum_filtered"].set_visible(False)

                plot_elements["fig"].canvas.draw_idle()
            except Exception:
                plt.draw()
            debug_log_csi("Plot update complete.")

        def _create_event_handler(param_name):
            """Factory function to create event handlers with consistent logging."""
            debug_log_csi(
                f"---> Creating event handler for {param_name}\n _create_event_handler()"
            )

            def handler(change):
                if change["name"] == "value":

                    if param_name == "filter":
                        # Special handling for filter dropdown
                        selected_filter = change["new"]
                        visibility_map = {
                            "Exponential": {
                                "decay_constant_slider": "block",
                                "alpha_slider": "none",
                                "center_position_slider": "none",
                            },
                            "Hamming": {
                                "decay_constant_slider": "none",
                                "alpha_slider": "block",
                                "center_position_slider": "block",
                            },
                            "Hann": {
                                "decay_constant_slider": "none",
                                "alpha_slider": "none",
                                "center_position_slider": "block",
                            },
                            "None": {
                                "decay_constant_slider": "none",
                                "alpha_slider": "none",
                                "center_position_slider": "none",
                            },
                        }
                        if selected_filter in visibility_map:
                            for widget_name, display_value in visibility_map[
                                selected_filter
                            ].items():
                                widgets[widget_name].layout.display = display_value
                        if selected_filter == "Hamming":
                            widgets["alpha_slider"].description = "Alpha"
                    _update_plot(widgets["signal_end_slider"].value)

                    if param_name == "spatial_filter":
                        selected_spatial = change["new"]
                        if selected_spatial != "None":
                            widgets["spatial_width_slider"].layout.display = "block"
                        else:
                            widgets["spatial_width_slider"].layout.display = "none"
                        _update_plot(widgets["signal_end_slider"].value)

            return handler

        def _setup_event_handlers():
            """Create and connect all widget event handlers."""

            def _onclick(event):
                """Handle plot click events."""
                if event.inaxes in [plot_elements["ax"][0], plot_elements["ax"][1]]:
                    end_segment_index = max(0, min(n - 1, int(event.xdata)))
                    widgets["signal_end_slider"].value = end_segment_index
                elif event.inaxes in [plot_elements["ax"][4], plot_elements["ax"][5]]:
                    # Spectrum plots -> update image frequency
                    clicked_x = event.xdata
                    freq_axis = plot_elements["freq_axis"]
                    if freq_axis is not None and freq_axis.size > 0:
                        idx = (np.abs(freq_axis - clicked_x)).argmin()
                        nonlocal current_image_freq_idx
                        current_image_freq_idx = idx
                        _update_plot(widgets["signal_end_slider"].value)

            def _on_confirm_selection(b):
                """Handle confirmation button click."""
                filter_params, _ = _collect_filter_parameters(widgets)
                self.signal_end = widgets["signal_end_slider"].value
                self.filter_params = filter_params
                print(f"Signal end index confirmed: {self.signal_end}")
                print(f"Filter parameters saved: {self.filter_params}")
                plot_elements["fig"].canvas.mpl_disconnect(cid)
                plt.close(plot_elements["fig"])

            # Connect click event
            cid = plot_elements["fig"].canvas.mpl_connect(
                "button_press_event", _onclick
            )

            # Create and connect event handlers using factory function
            handler_mappings = {
                "signal_end_slider": "signal_end",
                "filter_dropdown": "filter",
                "decay_constant_slider": "decay_constant",
                "alpha_slider": "alpha",
                "center_position_slider": "center_position",
                "repetition_slider": "repetition",
                "flipangle_slider": "flipangle",
                "flipangle_correction_button": "flipangle_or_checkbox",
                "spatial_filter_dropdown": "spatial_filter",
                "spatial_width_slider": "spatial_width",
            }

            for widget_name, param_name in handler_mappings.items():
                widgets[widget_name].observe(
                    _create_event_handler(param_name), names="value"
                )

            widgets["confirm_button"].on_click(_on_confirm_selection)
            return cid

        def _finalize_interface():
            """Organize widgets, display interface, and trigger initial update."""
            # Organize widgets in a cleaner layout
            filters_box = VBox(
                [
                    widgets["filter_dropdown"],
                    widgets["decay_constant_slider"],
                    widgets["alpha_slider"],
                    widgets["center_position_slider"],
                    widgets["flipangle_slider"],
                ]
            )
            spatial_filters_box = VBox(
                [
                    widgets["spatial_filter_dropdown"],
                    widgets["spatial_width_slider"],
                ]
            )
            controls_box = HBox(
                [
                    widgets["signal_end_slider"],
                    widgets["repetition_slider"],
                    filters_box,
                    spatial_filters_box,
                    widgets["confirm_button"],
                    widgets["flipangle_correction_button"],
                ]
            )

            display(controls_box)

            _update_plot(widgets["signal_end_slider"].value)
            plt.show()

        # Set up event handlers and finalize the interface
        cid = _setup_event_handlers()
        _finalize_interface()

    def combine_channels(
        self,
        input_data: xr.DataArray | None = None,
        channel_combination: str = "mean",
        adjust_phase: bool = True,
        plot: bool = False,
    ) -> xr.DataArray:
        """Combine receive channels after optional phase alignment.

        The method measures the phase offset between the raw FID traces of two
        receive channels, corrects the second channel if ``adjust_phase`` is true,
        and reconstructs the corrected data. The combined dataset is stored on the
        object as :attr:`reco_data_combined_channels` and returned for convenience.

        .. deprecated::
            This method is being refactored. For general channel combination,
            consider using :func:`hypermri.utils.mri.channels.combine_channels`.

        Parameters
        ----------
        input_data : xarray.DataArray, optional
            Spectroscopic data with ``channel_idx`` and ``acquisition_idx``
            dimensions. Defaults to :attr:`reco_data`.
        channel_combination : {"mean", "sum", "rss"}, optional
            How to combine the aligned channels. Default is "mean".
        adjust_phase : bool, optional
            If ``True`` (default) apply the measured phase difference to the second
            channel before combining. Disable when the hardware stack already
            delivers phase-matched channels.
        plot : bool, optional
            Show diagnostic figures for the raw and corrected FIDs.

        Returns
        -------
        xarray.DataArray
            Combined spectroscopic data with channel_idx dimension removed.

        Raises
        ------
        ValueError
            Raised when the input is missing ``channel_idx``/``acquisition_idx``
            dimensions or does not contain two channels.
        """
        import copy

        from hypermri.recon_strategy import BrukerCsiRecoStrategy

        if input_data is None:
            input_data = self.reco_data

        # Check for required dimensions
        if "channel_idx" not in input_data.dims:
            raise ValueError("Input data must have 'channel_idx' dimension")
        if "acquisition_idx" not in input_data.dims:
            raise ValueError("Input data must have 'acquisition_idx' dimension")

        # Get acquisition_idx size from the data
        acquisition_idx_length = input_data.sizes["acquisition_idx"]

        # Get number of channels
        n_channels = input_data.sizes["channel_idx"]

        # Check if we have 2 channels
        if n_channels != 2:
            raise ValueError(
                f"Input data must have at 2 channels for phase correction, has {n_channels}!"
            )

        # Optional plotting of reconstruction data
        if plot:
            fig, ax = plt.subplots(2, 2, figsize=(6, 4))
            ax = ax.flatten()

            image_ch1 = input_data.isel(channel_idx=0).squeeze()
            image_ch2 = input_data.isel(channel_idx=1).squeeze()
            print(f"Channel 1 shape: {image_ch1.shape}")

            # Plot original data
            ax[0].imshow(np.squeeze(np.real(image_ch1[0, ...])))
            ax[0].set_title("Ch1 (Image))")

            # Get axis indices for FFT
            read_idx = input_data.get_axis_num("read_idx")
            phase_idx = input_data.get_axis_num("phase_idx")

            # Perform 2D IFFT for channel 1
            kspace_ch1 = np.fft.fftshift(
                np.fft.ifft(np.fft.ifftshift(image_ch1, axes=read_idx), axis=read_idx),
                axes=read_idx,
            )
            kspace_ch1 = np.fft.fftshift(
                np.fft.ifft(
                    np.fft.ifftshift(kspace_ch1, axes=phase_idx), axis=phase_idx
                ),
                axes=phase_idx,
            )
            ax[1].imshow(np.squeeze(np.abs(kspace_ch1[0, ...])))
            ax[1].set_title("Ch1 (k-space)")

            # Perform 2D IFFT for channel 2
            kspace_ch2 = np.fft.fftshift(
                np.fft.ifft(np.fft.ifftshift(image_ch2, axes=read_idx), axis=read_idx),
                axes=read_idx,
            )
            kspace_ch2 = np.fft.fftshift(
                np.fft.ifft(
                    np.fft.ifftshift(kspace_ch2, axes=phase_idx), axis=phase_idx
                ),
                axes=phase_idx,
            )
            ax[2].imshow(np.squeeze(np.abs(image_ch2[0, ...])))
            ax[2].set_title("Ch2 (image)")
            ax[3].imshow(np.squeeze(np.abs(kspace_ch2[0, ...])))
            ax[3].set_title("Ch2 (k-space)")

            plt.tight_layout()
            plt.show()

        # Process raw FID data for phase correction
        fid = self.handler.rawdatajob0

        # Optional plotting of FID data

        # Calculate phase difference between channels
        phase_diff = np.angle(fid[acquisition_idx_length]) - np.angle(fid[0])
        phase_diff_degrees = phase_diff * 180 / np.pi

        print(
            f"Phase difference between channels: {phase_diff:.4f} rad ({phase_diff_degrees:.2f}°)"
        )
        if not adjust_phase:
            phase_diff = 0.0
            print("Phase adjustment disabled, proceeding without phase correction.")

        # Reshape FID data and separate channels
        fid = np.reshape(fid, (-1, acquisition_idx_length))

        fid_ch1 = fid[::2, :]  # Even indices (channel 1)
        fid_ch2 = fid[1::2, :]  # Odd indices (channel 2)

        # Apply phase correction to channel 2
        fid_ch2_phased = fid_ch2 * np.exp(-1j * phase_diff)

        # Reconstruct phase-corrected FID data
        fid_phased = np.zeros_like(fid, dtype=np.complex64)
        fid_phased[::2, :] = fid_ch1
        fid_phased[1::2, :] = fid_ch2_phased

        # Optional plotting of phase correction results
        if plot:
            fig, ax = plt.subplots(2, 3, figsize=(8, 4))
            ax = ax.flatten()

            # Plot real parts of first acquisition_idx points for each channel
            ax[0].plot(np.real(fid[0, :]))
            ax[0].set_title(f"Ch1 Real FID")
            ax[1].plot(np.real(fid[1, :]))
            ax[1].set_title(f"Ch2 Real FID")

            ax[2].plot(np.real(fid[0, :] + fid[1, :]), ".-", label="Original")

            ax[3].plot(np.real(fid_phased[0, :]))
            ax[3].set_title(f"Ch1 Real FID (Phased)")
            ax[4].plot(np.real(fid_phased[1, :]))
            ax[4].set_title(f"Ch2 Real FID (Phased)")
            ax[5].plot(
                np.real(fid[0, :] + fid[1, :]),
                ".-",
                label="Ch1 + Ch2 Original",
                alpha=0.5,
                linewidth=4,
                color="gray",
            )
            ax[5].plot(
                np.real(fid_phased[0, :] + fid_phased[1, :]),
                ".-",
                label="Ch1 + Ch2 Phased",
            )
            ax[5].legend()
            ax[5].set_title("Comparison Combined Channels")

            plt.tight_layout()
            plt.show()
        fid_phased = np.reshape(fid_phased, (-1,))
        # Reconstruct using BrukerCsiRecoStrategy
        reco_strategy = BrukerCsiRecoStrategy()

        # Preserve original rawdatajob0
        original_rawdatajob0 = (
            self.handler.rawdatajob0.copy()
            if hasattr(self.handler.rawdatajob0, "copy")
            else copy.deepcopy(self.handler.rawdatajob0)
        )

        # Modify handler directly
        self.handler.rawdatajob0 = fid_phased

        # Perform reconstruction & Store results in CSI object
        self.reco_data_combined_channels = reco_strategy.reconstruct(
            handler=self.handler
        )

        # Restore original data
        self.handler.rawdatajob0 = original_rawdatajob0

        # Use the new general-purpose channel combination utility
        from hypermri.utils.mri.channels import combine_channels

        # Get the original raw FID data (before reshaping)
        raw_fid = self.handler.rawdatajob0

        # Combine channels using phase-aligned method with the new utility
        # Note: The utility properly removes the channel_idx dimension
        self.reco_data_combined_channels = combine_channels(
            data_array=self.reco_data_combined_channels,
            method="phase_aligned",
            adjust_phase=adjust_phase,
            raw_fid_data=raw_fid,
        )

        # Store the combination method in attributes
        combination_method = "mean" if channel_combination != "sum" else "sum"

        # The utility already adds metadata, but we'll ensure description is complete
        if "description" not in self.reco_data_combined_channels.attrs:
            phase_info = self.reco_data_combined_channels.attrs.get(
                "phase_correction_deg", 0.0
            )
            if adjust_phase:
                self.reco_data_combined_channels.attrs["description"] = (
                    f"Channels combined using '{combination_method}' after phase alignment. "
                    f"Phase correction of {phase_info:.2f}° applied to channel 2."
                )
            else:
                self.reco_data_combined_channels.attrs["description"] = (
                    f"Channels combined using '{combination_method}' without phase correction."
                )

        return self.reco_data_combined_channels

    def create_filtered_data(
        self,
        end_index: int = None,
        *,
        filter_type: str = "None",
        decay_constant: float = 0.01,
        alpha: float = 0.54,
        center_position: int = None,
        apply_flipangle_correction: bool = False,
        flipangle_deg: float = None,
        filter_axis: int = 0,
        source_data: str = "reco_data",
        update_attributes: bool = True,
        filter_params: dict = None,
    ) -> xr.DataArray:
        """
        Create a filtered version of spectroscopic data using signal filtering and reconstruction.

        This method applies filtering to the underlying FID data (like plot_signal_decay workflow)
        and then reconstructs to create a filtered spectroscopic data array with the same
        structure as reco_data but with applied filtering effects.

        Parameters
        ----------
        end_index : int, optional
            Signal end index for truncation when filter_type="None".
            If not provided, uses self.signal_end if available.
        filter_type : str, optional
            Type of filter to apply. Options: "None", "Exponential", "Hamming", "Hann",
            by default "None"
        decay_constant : float, optional
            Decay constant for exponential filter, by default 0.01
        alpha : float, optional
            Alpha parameter for Hamming filter, by default 0.54
        center_position : int, optional
            Center position for Hamming/Hann filters. If None, uses middle of signal
        apply_flipangle_correction : bool, optional
            Whether to apply flipangle correction, by default False
        flipangle_deg : float, optional
            Flip angle in degrees for correction
        filter_axis : int, optional
            Axis along which to apply the filter. Default is 0.
        source_data : str, optional
            Source data to use as template. Options: "reco_data" or "data_array",
            by default "reco_data"
        update_attributes : bool, optional
            Whether to update time and frequency attributes, by default True
        filter_params : dict, optional
            Dictionary of filter parameters (alternative to individual parameters).
            Can contain keys from plot_signal_decay: filter_type, signal_end, etc.

        Returns
        -------
        xr.DataArray
            Filtered data array with same structure as source but with filtering applied

        Raises
        ------
        ValueError
            If filter parameters are invalid or source data is not available
        AttributeError
            If required methods or data are not available
        """
        # Use filter_params dict if provided (from plot_signal_decay workflow)
        if filter_params is not None:
            filter_type = filter_params.get("filter_type", filter_type)
            end_index = filter_params.get("signal_end", end_index)
            decay_constant = filter_params.get("decay_constant", decay_constant)
            alpha = filter_params.get("alpha", alpha)
            center_position = filter_params.get("center_position", center_position)
            apply_flipangle_correction = filter_params.get(
                "apply_flipangle_correction", apply_flipangle_correction
            )
            flipangle_deg = filter_params.get("flipangle_deg", flipangle_deg)
            filter_axis = filter_params.get("filter_axis", filter_axis)

        # Get source data for structure reference
        if source_data == "reco_data":
            if not hasattr(self, "reco_data") or self.reco_data is None:
                if hasattr(self, "data_array") and self.data_array is not None:
                    logger.warning(
                        "reco_data not available, falling back to data_array"
                    )
                    source = self.data_array
                else:
                    raise AttributeError(
                        "Neither reco_data nor data_array is available"
                    )
            else:
                source = self.reco_data
        elif source_data == "data_array":
            if not hasattr(self, "data_array") or self.data_array is None:
                raise AttributeError("data_array is not available")
            source = self.data_array
        else:
            raise ValueError(
                f"Invalid source_data: {source_data}. Must be 'reco_data' or 'data_array'"
            )

        # Validate source data structure
        if "acquisition_idx" not in source.dims:
            raise AttributeError(
                "Source data must have 'acquisition_idx' dimension for spectroscopic filtering"
            )

        logger.info(
            f"Creating filtered data with filter_type='{filter_type}', end_index={end_index}"
        )

        try:
            filtered_result = self.filter_signal(
                filter_type=filter_type,
                signal_end=end_index,
                decay_constant=decay_constant,
                alpha=alpha,
                center_position=center_position,
                apply_flipangle_correction=apply_flipangle_correction,
                flipangle_deg=flipangle_deg,
                filter_axis=filter_axis,
            )

            filtered_data = filtered_result

        except Exception as e:
            logger.warning(
                f"filter_signal method failed ({e}), falling back to simple truncation"
            )
            # Fallback to simple truncation
            filtered_data = source.isel(acquisition_idx=slice(0, end_index))

        try:
            # Use xarray roll method to shift by -1 along both spatial dimensions
            from ..recon_strategy import BrukerCsiRecoStrategy

            reco_strategy = BrukerCsiRecoStrategy()

            handler_filtered = self.handler  # .copy()
            handler_filtered.rawdatajob0 = filtered_data

            filtered_data = reco_strategy.reconstruct(handler=handler_filtered)
        except Exception as e:
            print(f"Could not reconstruct image: {e}")

        if update_attributes:
            # Update time-related attributes
            # This is necessary because filtering could change the effective time axis
            new_attrs = filtered_data.attrs.copy()

            # Add filtering metadata
            new_attrs["filter_type"] = filter_type
            new_attrs["filtered_end_index"] = end_index
            new_attrs["original_length"] = source.sizes["acquisition_idx"]

            if filter_type == "Exponential":
                new_attrs["decay_constant"] = decay_constant
                description_update = f"Data was filtered using {filter_type} filter, with parameters: {decay_constant}"
            elif filter_type == "Hamming":
                new_attrs["alpha"] = alpha
                new_attrs["center_position"] = center_position
                description_update = f"Data was filtered using {filter_type} filter, with parameters: {alpha}, {center_position}"
            elif filter_type == "Hann":
                new_attrs["center_position"] = center_position
                description_update = f"Data was filtered using {filter_type} filter, with parameters: {center_position}"
            else:
                description_update = f"Data was truncated to first {end_index} points without additional filtering"

            # Apply updated attributes
            filtered_data.attrs = new_attrs

            # Update description
            if "description" in filtered_data.attrs:
                filtered_data.attrs["description"] += f" | {description_update}"
            else:
                filtered_data.attrs["description"] = description_update

        return filtered_data

    def reset(self):
        """Resest to intital values."""
        # reset index for @onlycallonce
        self.__SNR_scaling_HasBeenCalled = False

        self.seq2d = self.load_2dseq_file()
        self.seq2d_broadend = np.copy(self.seq2d)

    def intensity_map(self):
        return np.sum(self.data, axis=0)

    def find_peaks(
        self, range_unit="ppm", searchrange=[], sr_index=0, height=0, **kwargs
    ):
        self.peaks[sr_index] = dict()
        self.peakproperties[sr_index] = dict()

        # get the configured x-axis values
        if range_unit == "ppm":
            xaxis = get_freq_axis(data_obj=self, unit="ppm", cut_off=0, npoints=None)
        elif range_unit == "Hz":
            xaxis = get_freq_axis(
                data_obj=self,
                unit="Hz",
                cut_off=0,
                npoints=None,
            )
        else:
            xaxis = np.arange(0, self.seq2d.shape[0], 1)

        if len(searchrange):

            def find_closest_index(x):
                return np.argmin(np.abs(x - xaxis))

            a = find_closest_index(searchrange[0])
            b = find_closest_index(searchrange[1])

            assert a < b, f"Illegal range! {searchrange}-->i_min={a} > i_max={b}."

        else:
            # that is, x[0, -1] == x[:]
            a, b = 0, -1

        # store info on searchrange and range unit

        self.searchrangeunit = range_unit
        self.searchranges[sr_index] = searchrange
        self.searchranges_idx[sr_index] = [a, b]

        def custom_find_peaks(spec):
            peaks, info = find_peaks(spec[a:b], height=height, **kwargs)
            peaks += a
            return peaks, info

        for idx, res in self.apply_to_every_spec(custom_find_peaks):
            tmp_peaks, tmp_properties = res

            # store peak location and peak properties
            self.peaks[sr_index][idx] = tmp_peaks
            self.peakproperties[sr_index][idx] = tmp_properties

    # --> move away from CSI?
    def fit_spectra(self, fit_params, data, xscale="ppm", plot=False, cut_off=0):
        """
        Fit lorentzian peak functions to reconstructed CSI data.

        This function used scipy.curvefit to fit lorentzians to found peaks in a 2dseq file from a csi experiment.

        Parameters
        ----------
        fit_params: dict
            has following keys:
            peak_positions: list of float, contains the positions of the peaks to be fitted in ppm (TODO implement for Hz as well)
            peak_heights: list of float, maximum height to which the lorentzian can be fitted
            peak_widths: float, upper bound for FWHM of peak to be fitted
            signal_threshold: float, everything greater than this SNR value is considered a peak, lower peaks will be discarded
            background_region: list of float, contains ppm values of a background region for baseline correction (TODO implement for Hz)
            reps_to_fit: list of int, numbers of the repetitons to be fitted. If set to True, all reps are fitted.

        data: nd.array
            Multi-dimensional array with CSI data to be fitted. As a default one can use the 2dseq file.
            Like:
             scans = hypermri.Study(path_to_day_folder)
             csi = scans[5]
             csi.fit_spectra(fit_params, csi.seq2d)
        xscale: str, default is ppm
            can also be Hz or None, then the index is chosen
        plot: bool, default is False.
            Wether or not to show the results.
        Returns
        -------

        """
        if xscale == "ppm":
            xaxis = get_freq_axis(data_obj=self, unit="ppm", cut_off=0, npoints=None)
            xlabel_text = r"$\sigma$ [ppm]"
        elif xscale == "Hz":
            xaxis = get_freq_axis(
                data_obj=self,
                unit="Hz",
                cut_off=0,
                npoints=None,
            )
            xlabel_text = r"$\sigma$ [Hz]"
        else:
            xaxis = np.arange(0, self.seq2d.shape[0], 1)
            xlabel_text = "array index"

        # we assume csi 2dseq data of form:
        # spectral_acq_points, x-dimension, y-dimension, repetitions
        logger.debug("Assuming input data of format")

        # Todo implement phasing of data

        n_repetitions = self.n_repetitions
        matrix = self.method["PVM_Matrix"]
        xdim = matrix[0]
        ydim = matrix[1]
        n_acquisition = self.n_acquisition
        spec_acq_time = self.method["PVM_SpecAcquisitionTime"]

        if data.shape == (n_acquisition, xdim, ydim, n_repetitions):
            logger.debug(
                "Assuming input data from 2dseq file of format (spectral_acq_points, x-dimension, y-dimension, repetitions)"
            )
        else:
            debug_log_csi(
                "Input data does not have format (spectral_acq_points, x-dimension, y-dimension, repetitions)"
            )

        # only selecting a specific number of repetitions to fit
        reps_to_fit = fit_params["reps_to_fit"]
        logger.debug(
            "csi.fit_spectra() -- Changed number of repetitions to fit to %d"
            + str(n_repetitions)
        )
        # fitting methods
        fit_method = fit_params["fit_method"]
        if fit_method not in ["curve_fit"]:
            logger.error(
                "Fit method "
                + str(fit_params["fit_method"])
                + " is not a valid fit method"
            )
        else:
            pass

        def find_range(axis, ppm):
            return np.argmin(np.abs(axis - ppm))

        background_region_indices = [
            find_range(xaxis, fit_params["background_region"][0]),
            find_range(xaxis, fit_params["background_region"][1]),
        ]
        # Interpolating the ppm axis
        x_axis_itp = np.linspace(np.min(xaxis), np.max(xaxis), 10000)

        N_peaks = len(fit_params["peak_positions"])
        #
        peak_coeff = np.zeros((N_peaks, 3, xdim, ydim, n_repetitions))
        peak_covariance = np.zeros((N_peaks, 3, 3, xdim, ydim, n_repetitions))
        peak_errors = np.zeros((N_peaks, 3, xdim, ydim, n_repetitions))
        # TODO make this parallel using multiprocessing
        for rep in tqdm(range(n_repetitions), desc="Fitting repetitions"):
            for x_coord in range(xdim):
                for y_coord in range(ydim):
                    # baseline correct each spectrum
                    spec_norm = data[:, x_coord, y_coord, rep] - np.mean(
                        data[
                            background_region_indices[0] : background_region_indices[1],
                            x_coord,
                            y_coord,
                            rep,
                        ]
                    )
                    # fit each peak
                    for peak_number, peak_center in enumerate(
                        fit_params["peak_positions"]
                    ):
                        peak_roi = [
                            find_range(xaxis, peak_center - fit_params["peak_width"]),
                            find_range(xaxis, peak_center + fit_params["peak_width"]),
                        ]
                        if rep in reps_to_fit:
                            try:
                                # Actual fitting happens here
                                if fit_method == "curve_fit":
                                    (
                                        peak_coeff[
                                            peak_number, :, x_coord, y_coord, rep
                                        ],
                                        peak_covariance[
                                            peak_number, :, :, x_coord, y_coord, rep
                                        ],
                                    ) = curve_fit(
                                        lorentzian,
                                        xaxis[peak_roi[0] : peak_roi[1]],
                                        spec_norm[peak_roi[0] : peak_roi[1]],
                                        bounds=(
                                            [
                                                0.01,
                                                peak_center
                                                - fit_params["peak_width"] / 2.0,
                                                0,
                                            ],
                                            [
                                                fit_params["peak_width"] * 2,
                                                peak_center
                                                + fit_params["peak_width"] / 2.0,
                                                fit_params["peak_heights"][peak_number],
                                            ],
                                        ),
                                    )
                                    peak_errors[
                                        peak_number, :, x_coord, y_coord, rep
                                    ] = np.sqrt(
                                        np.diag(
                                            peak_covariance[
                                                peak_number, :, :, x_coord, y_coord, rep
                                            ]
                                        )
                                    )
                                # TODO implement different fitting
                                else:
                                    pass
                            except RuntimeError:
                                peak_coeff[peak_number, :, x_coord, y_coord, rep] = 0
                                peak_covariance[
                                    peak_number, :, :, x_coord, y_coord, rep
                                ] = 0
                                peak_errors[peak_number, :, x_coord, y_coord, rep] = 0
                        else:
                            peak_coeff[peak_number, :, x_coord, y_coord, rep] = [
                                0,
                                0,
                                0,
                            ]
                        # clean up badly fitted peaks
                    # for peak in range(peak_coeff.shape[0]):
                    # FIXME this removes all fits so far
                    #    peak_height = peak_coeff[peak,:,x_coord,y_coord,rep][2]
                    #    peak_height_error = peak_errors[peak,:,x_coord,y_coord,rep][2]
                    #    if peak_height > fit_params['signal_threshold']:
                    #        # peak needs to have an SNR greater than a certain value
                    #        pass
                    #    else:
                    #        peak_coeff[peak,:,x_coord,y_coord,rep] = [0, 0, 0]

        if plot is True:
            plt.close("all")
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            @widgets.interact(
                x_coord=(0, xdim - 1, 1),
                y_coord=(0, ydim - 1, 1),
                rep=(0, n_repetitions - 1, 1),
            )
            def update(x_coord=xdim // 2, y_coord=ydim // 2, rep=0):
                ax[1].cla()
                ax[0].cla()
                displayed_spectrum = data[:, x_coord, y_coord, rep] - np.mean(
                    data[
                        background_region_indices[0] : background_region_indices[1],
                        x_coord,
                        y_coord,
                        rep,
                    ]
                )

                displayed_image = np.sum(data, axis=0)[:, :, rep]
                ax[0].imshow(displayed_image, cmap="plasma")
                ax[0].axhline(x_coord, color="w", linestyle="dashed")
                ax[0].axvline(y_coord, color="c", linestyle="dashed")
                ax[0].set_title("CSI summed over the spectrum")
                ax[0].set_xlabel("y_coord")
                ax[0].set_ylabel("x_coord")

                # Plotting for QA
                combined_fits = 0
                for peak_number, peak in enumerate(fit_params["peak_positions"]):
                    combined_fits += lorentzian(
                        x_axis_itp,
                        peak_coeff[peak_number, :, x_coord, y_coord, rep][0],
                        peak_coeff[peak_number, :, x_coord, y_coord, rep][1],
                        peak_coeff[peak_number, :, x_coord, y_coord, rep][2],
                    )
                    print(
                        "Peak at "
                        + str(peak)
                        + " has fit params: "
                        + str(peak_coeff[peak_number, :, x_coord, y_coord, rep])
                    )

                ax[1].plot(
                    xaxis,
                    displayed_spectrum,
                    linewidth="0.5",
                    color="r",
                    label="Data",
                )

                # ax[1].set_xlim([np.max(xaxis),np.min(xaxis)])
                # ax[1].set_ylim([np.min(displayed_spectrum-5),np.max(displayed_spectrum+5)])
                ax[1].plot(
                    x_axis_itp,
                    combined_fits,
                    linestyle="dashed",
                    color="k",
                    label="Lorentzian fit",
                )

                ax[1].set_ylabel("I [a.u]")
                ax[1].set_xlabel(xlabel_text)
                ax[1].legend()

        else:
            pass

        return peak_coeff, peak_errors

    def get_plotting_grid(self, csi_interpolation_matrix=None):
        """
        Return the grid that is used to plot the data.
        """
        # patient position (usually Head_Prone)
        patient_pos = self.acqp["ACQ_patient_pos"]
        read_orient = self.method["PVM_SPackArrReadOrient"]
        slice_orient = self.method["PVM_SPackArrSliceOrient"]
        mm_slice_gap = self.method["PVM_SPackArrSliceGap"]

        mm_read_csi, mm_phase_csi, mm_slice_csi = define_imageFOV_parameters(
            data_obj=self
        )
        dim_read_csi, dim_phase_csi, dim_slice_csi = define_imagematrix_parameters(
            data_obj=self
        )
        if csi_interpolation_matrix is not None:
            dim_read_csi, dim_phase_csi, dim_slice_csi = (
                dim_read_csi * csi_interpolation_matrix[1],
                dim_phase_csi * csi_interpolation_matrix[2],
                dim_slice_csi * csi_interpolation_matrix[3],
            )
            debug_log_csi(
                "Updated plotting grid according to interpolation matrix: "
                + str(csi_interpolation_matrix)
            )
            csi_grid = define_xyz_grid(
                mat=np.array((dim_read_csi, dim_phase_csi, dim_slice_csi)),
                fov=np.array((mm_read_csi, mm_phase_csi, mm_slice_csi)),
            )
        else:
            csi_grid = define_xyz_grid(
                mat=np.array((dim_read_csi, dim_phase_csi, dim_slice_csi)),
                fov=np.array((mm_read_csi, mm_phase_csi, mm_slice_csi)),
            )

        # offsets [mm]
        read_offset = self.method["PVM_SPackArrReadOffset"]
        phase_offset = self.method["PVM_SPackArrPhase1Offset"]
        slice_offset = self.method["PVM_SPackArrSliceOffset"]

        csi_grid = list(csi_grid)

        if patient_pos == "Head_Prone":
            if slice_orient == "axial":
                if read_orient == "L_R":
                    csi_grid[1] = csi_grid[1] + read_offset
                    csi_grid[2] = csi_grid[2] + phase_offset
                    csi_grid = [csi_grid[1], csi_grid[2]]
                elif read_orient == "A_P":
                    csi_grid[1] = csi_grid[1] + phase_offset
                    csi_grid[2] = csi_grid[2] + read_offset
                    csi_grid = [csi_grid[2], csi_grid[1]]
                else:
                    raise Exception(
                        "unknown read orientation: "
                        + read_orient
                        + "for slice_orient: "
                        + slice_orient
                    )

            # for now do the same to get values:
            elif slice_orient == "sagittal":
                if read_orient == "H_F":
                    csi_grid[2] = list(csi_grid)[2] + phase_offset
                    csi_grid[0] = list(csi_grid)[0] + read_offset
                    csi_grid = [csi_grid[0], csi_grid[2]]

                elif read_orient == "A_P":
                    csi_grid[2] = list(csi_grid)[2] + read_offset
                    csi_grid[0] = list(csi_grid)[0] + phase_offset
                    csi_grid = [csi_grid[2], csi_grid[0]]

                else:
                    raise Exception(
                        "unknown read orientation: "
                        + read_orient
                        + "for slice_orient: "
                        + slice_orient
                    )

            # for now do the same to get values:
            elif slice_orient == "coronal":
                if read_orient == "H_F":
                    csi_grid[1] = list(csi_grid)[1] + phase_offset
                    csi_grid[0] = list(csi_grid)[0] + read_offset
                    csi_grid = [csi_grid[0], csi_grid[1]]
                elif read_orient == "L_R":
                    csi_grid[1] = list(csi_grid)[1] + read_offset
                    csi_grid[0] = list(csi_grid)[0] + phase_offset
                    csi_grid = [csi_grid[1], csi_grid[0]]
                else:
                    raise Exception(
                        "unknown read orientation: "
                        + read_orient
                        + "for slice_orient: "
                        + slice_orient
                    )

            # for now do the same to get values:
            else:
                raise Exception("unknown slice orientation: " + slice_orient)

        # same as for head_prone, has to be changed later!!!!
        elif patient_pos == "Head_Supine":
            debug_log_csi(message="Head_Supine is not fully implemented yet.")
            if slice_orient == "axial":
                if read_orient == "L_R":
                    csi_grid = [csi_grid[1], csi_grid[2]]
                else:
                    raise Exception(
                        "unknown read orientation: "
                        + read_orient
                        + "for slice_orient: "
                        + slice_orient
                    )

            # for now do the same to get values:
            elif slice_orient == "sagittal":
                if read_orient == "H_F":
                    csi_grid = [csi_grid[0], csi_grid[2]]
                else:
                    raise Exception(
                        "unknown read orientation: "
                        + read_orient
                        + "for slice_orient: "
                        + slice_orient
                    )

            # for now do the same to get values:
            elif slice_orient == "coronal":
                if read_orient == "H_F":
                    csi_grid = [csi_grid[0], csi_grid[1]]
                else:
                    raise Exception(
                        "unknown read orientation: "
                        + read_orient
                        + "for slice_orient: "
                        + slice_orient
                    )

            # for now do the same to get values:
            else:
                raise Exception("unknown slice orientation: " + slice_orient)

        else:
            raise Exception("unknown patient_position: " + patient_pos)
        return csi_grid

    def define_imageFOV_parameters(data_obj=None):
        """
        Warning: Does not take into account the orientation and offsets of the
        object (yet)
        Calculate the FOV in mm.
        Return FOV in as mm_z, mm_x, mm_y, which correspond to z, x, y
        (scanner coord. system)

        Parameters
        ----------
        data_obj: hypermri-sequence object

        """
        # init empty return
        from warnings import Warning

        Warning("This function will be deprecated soon!")
        mm_z = mm_x = mm_y = None

        if data_obj is None:
            return mm_z, mm_x, mm_y

        # patient position (usually Head_Prone)
        patient_pos = data_obj.patient_position
        read_orient = data_obj.read_orientation
        slice_orient = data_obj.slice_orientation
        mm_slice_gap = data_obj.slice_gap_mm

        logger.debug(
            f"define_imageFOV_parameters: patient_pos={patient_pos}, read_orient={read_orient}, slice_orient={slice_orient}"
        )

        if patient_pos == "Head_Prone":
            # All axial slice options:
            if slice_orient == "axial":
                # axial slices with read in x-direction, phase in y and phase/slice
                # in z
                if read_orient == "L_R":
                    # x - dimension
                    mm_x = data_obj.fov_mm[0]
                    # y - dimension
                    mm_y = data_obj.fov_mm[1]
                    if data_obj.spatial_dimension == "3D":
                        mm_z = data_obj.fov_mm[2]
                    else:
                        dim_z, _, _ = define_imagematrix_parameters(data_obj=data_obj)
                        mm_z = data_obj.slice_thickness_mm
                        mm_z = mm_z * dim_z + mm_slice_gap * (dim_z - 1)
                # axial slices with read in y-direction, phase in x and phase/slice
                # in z
                elif read_orient == "A_P":
                    # x - dimension
                    mm_x = data_obj.fov_mm[1]
                    # y - dimension
                    mm_y = data_obj.fov_mm[0]
                    if data_obj.spatial_dimension == "3D":
                        mm_z = data_obj.fov_mm[2]
                    else:
                        dim_z, _, _ = define_imagematrix_parameters(data_obj=data_obj)
                        mm_z = data_obj.slice_thickness_mm
                        mm_z = mm_z * dim_z + mm_slice_gap * (dim_z - 1)

                else:
                    raise Exception(
                        f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                    )

            # All sagittal slice options:
            elif slice_orient == "sagittal":
                # sagittal slices with read in z-direction, phase in y and
                # phase/slice in x
                if read_orient == "H_F":
                    # x - dimension
                    mm_y = data_obj.fov_mm[1]
                    # z - dimension
                    mm_z = data_obj.fov_mm[0]
                    if data_obj.spatial_dimension == "3D":
                        mm_x = data_obj.fov_mm[2]
                    else:
                        _, dim_x, _ = define_imagematrix_parameters(data_obj=data_obj)
                        mm_x = data_obj.slice_thickness_mm
                        mm_x = mm_x * dim_x + mm_slice_gap * (dim_x - 1)
                # sagittal slices with read in y-direction, phase in z and
                # phase/slice in x
                elif read_orient == "A_P":
                    # x - dimension
                    mm_y = data_obj.fov_mm[0]
                    # z - dimension
                    mm_z = data_obj.fov_mm[1]
                    if data_obj.spatial_dimension == "3D":
                        mm_x = data_obj.fov_mm[2]
                    else:
                        _, dim_x, _ = define_imagematrix_parameters(data_obj=data_obj)
                        mm_x = data_obj.slice_thickness_mm
                        mm_x = mm_x * dim_x + mm_slice_gap * (mm_x - 1)
                else:
                    raise Exception(
                        f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                    )

            # All coronal slice options:
            elif slice_orient == "coronal":
                # coronal slices with read in z-direction, phase in x and
                # phase/slice in y
                if read_orient == "H_F":
                    # z - dimension
                    mm_z = data_obj.fov_mm[0]
                    # x - dimension
                    mm_x = data_obj.fov_mm[1]

                    if data_obj.spatial_dimension == "3D":
                        # y - dimension
                        mm_y = data_obj.fov_mm[2]
                    else:
                        _, _, dim_y = define_imagematrix_parameters(data_obj=data_obj)
                        mm_y = data_obj.slice_thickness_mm
                        mm_y = mm_y * dim_y + mm_slice_gap * (dim_y - 1)
                # coronal slices with read in x-direction, phase in z and
                # phase/slice in y
                elif read_orient == "L_R":
                    # z - dimension
                    mm_z = data_obj.fov_mm[0]
                    # x - dimension
                    mm_x = data_obj.fov_mm[1]
                    if data_obj.spatial_dimension == "3D":
                        # y - dimension
                        mm_y = data_obj.fov_mm[2]
                    else:
                        _, _, dim_y = define_imagematrix_parameters(data_obj=data_obj)
                        mm_y = data_obj.slice_thickness_mm
                        mm_y = mm_y * dim_y + mm_slice_gap * (dim_y - 1)

                else:
                    raise Exception(
                        f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                    )

            elif slice_orient == "axial sagittal coronal":
                raise Exception(
                    f"Slice orientation: {slice_orient} --> probably Localizer, not implemented yet!"
                )

            # for now do the same to get values:
            else:
                raise Exception("unknown slice orientation: " + slice_orient)

        # same as for head_prone, has to be changed later!!!!
        elif patient_pos == "Head_Supine":
            if slice_orient == "axial":
                if read_orient == "L_R":
                    # x - dimension
                    mm_x = data_obj.fov_mm[0]
                    # y - dimension
                    mm_y = data_obj.fov_mm[1]
                    if data_obj.spatial_dimension == "3D":
                        mm_z = data_obj.fov_mm[2]
                    else:
                        dim_z, _, _ = define_imagematrix_parameters(data_obj=data_obj)
                        mm_z = data_obj.slice_thickness_mm
                        mm_z = mm_z * dim_z + mm_slice_gap * (dim_z - 1)
                elif read_orient == "A_P":
                    # x - dimension
                    mm_x = data_obj.fov_mm[1]
                    # y - dimension
                    mm_y = data_obj.fov_mm[0]
                    if data_obj.spatial_dimension == "3D":
                        mm_z = data_obj.fov_mm[2]
                    else:
                        dim_z, _, _ = define_imagematrix_parameters(data_obj=data_obj)
                        mm_z = data_obj.slice_thickness_mm
                        mm_z = mm_z * dim_z + mm_slice_gap * (dim_z - 1)
                else:
                    raise Exception(
                        f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                    )
            # for now do the same to get values:
            elif slice_orient == "sagittal":
                if read_orient == "H_F":
                    # x - dimension
                    mm_y = data_obj.fov_mm[1]
                    # z - dimension
                    mm_z = data_obj.fov_mm[0]
                    if data_obj.spatial_dimension == "3D":
                        mm_x = data_obj.fov_mm[2]
                    else:
                        _, dim_x, _ = define_imagematrix_parameters(data_obj=data_obj)
                        mm_x = data_obj.slice_thickness_mm
                        mm_x = mm_x * dim_x + mm_slice_gap * (mm_x - 1)
                elif read_orient == "A_P":
                    # x - dimension
                    mm_y = data_obj.fov_mm[1]
                    # z - dimension
                    mm_z = data_obj.fov_mm[0]
                    if data_obj.spatial_dimension == "3D":
                        mm_x = data_obj.fov_mm[2]
                    else:
                        _, dim_x, _ = define_imagematrix_parameters(data_obj=data_obj)
                        mm_x = data_obj.slice_thickness_mm
                        mm_x = mm_x * dim_x + mm_slice_gap * (mm_x - 1)

                else:
                    raise Exception(
                        f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                    )

            # for now do the same to get values:
            elif slice_orient == "coronal":
                if read_orient == "H_F":
                    # x - dimension
                    mm_x = data_obj.fov_mm[1]
                    # z - dimension
                    mm_z = data_obj.fov_mm[0]
                    if data_obj.spatial_dimension == "3D":
                        mm_y = data_obj.fov_mm[2]
                    else:
                        _, _, dim_y = define_imagematrix_parameters(data_obj=data_obj)
                        mm_y = data_obj.slice_thickness_mm
                        mm_y = mm_y * dim_y + mm_slice_gap * (mm_y - 1)
                elif read_orient == "L_R":
                    # x - dimension
                    mm_x = data_obj.fov_mm[0]
                    # z - dimension
                    mm_z = data_obj.fov_mm[1]
                    if data_obj.spatial_dimension == "3D":
                        mm_y = data_obj.fov_mm[2]
                    else:
                        _, _, dim_y = define_imagematrix_parameters(data_obj=data_obj)
                        mm_y = data_obj.slice_thickness_mm
                        mm_y = mm_y * dim_y + mm_slice_gap * (mm_y - 1)

                else:
                    raise Exception(
                        f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                    )

            elif slice_orient == "axial sagittal coronal":
                raise Exception(
                    f"Slice orientation: {slice_orient} --> probably Localizer, not implemented yet!"
                )

            # for now do the same to get values:
            else:
                raise Exception("unknown slice orientation: " + slice_orient)

        elif patient_pos == "Foot_Prone":
            # All axial slice options:
            if slice_orient == "axial":
                # axial slices with read in x-direction, phase in y and phase/slice
                # in z
                if read_orient == "L_R":
                    # x - dimension
                    mm_x = data_obj.fov_mm[0]
                    # y - dimension
                    mm_y = data_obj.fov_mm[1]
                    if data_obj.spatial_dimension == "3D":
                        mm_z = data_obj.fov_mm[2]
                    else:
                        dim_z, _, _ = define_imagematrix_parameters(data_obj=data_obj)
                        mm_z = data_obj.slice_thickness_mm
                        mm_z = mm_z * dim_z + mm_slice_gap * (dim_z - 1)
                # axial slices with read in y-direction, phase in x and phase/slice
                # in z
                elif read_orient == "A_P":
                    # x - dimension
                    mm_x = data_obj.fov_mm[1]
                    # y - dimension
                    mm_y = data_obj.fov_mm[0]
                    if data_obj.spatial_dimension == "3D":
                        mm_z = data_obj.fov_mm[2]
                    else:
                        dim_z, _, _ = define_imagematrix_parameters(data_obj=data_obj)
                        mm_z = data_obj.slice_thickness_mm
                        mm_z = mm_z * dim_z + mm_slice_gap * (dim_z - 1)

                else:
                    raise Exception(
                        f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                    )

            # All sagittal slice options:
            elif slice_orient == "sagittal":
                # sagittal slices with read in z-direction, phase in y and
                # phase/slice in x
                if read_orient == "H_F":
                    # x - dimension
                    mm_y = data_obj.fov_mm[1]
                    # z - dimension
                    mm_z = data_obj.fov_mm[0]
                    if data_obj.spatial_dimension == "3D":
                        mm_x = data_obj.fov_mm[2]
                    else:
                        _, dim_x, _ = define_imagematrix_parameters(data_obj=data_obj)
                        mm_x = data_obj.slice_thickness_mm
                        mm_x = mm_x * dim_x + mm_slice_gap * (dim_x - 1)
                # sagittal slices with read in y-direction, phase in z and
                # phase/slice in x
                elif read_orient == "A_P":
                    # x - dimension
                    mm_y = data_obj.fov_mm[0]
                    # z - dimension
                    mm_z = data_obj.fov_mm[1]
                    if data_obj.spatial_dimension == "3D":
                        mm_x = data_obj.fov_mm[2]
                    else:
                        _, dim_x, _ = define_imagematrix_parameters(data_obj=data_obj)
                        mm_x = data_obj.slice_thickness_mm
                        mm_x = mm_x * dim_x + mm_slice_gap * (mm_x - 1)
                else:
                    raise Exception(
                        f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                    )

            # All coronal slice options:
            elif slice_orient == "coronal":
                # coronal slices with read in z-direction, phase in x and
                # phase/slice in y
                if read_orient == "H_F":
                    # z - dimension
                    mm_z = data_obj.fov_mm[0]
                    # x - dimension
                    mm_x = data_obj.fov_mm[1]

                    if data_obj.spatial_dimension == "3D":
                        # y - dimension
                        mm_y = data_obj.fov_mm[2]
                    else:
                        _, _, dim_y = define_imagematrix_parameters(data_obj=data_obj)
                        mm_y = data_obj.slice_thickness_mm
                        mm_y = mm_y * dim_y + mm_slice_gap * (dim_y - 1)
                # coronal slices with read in x-direction, phase in z and
                # phase/slice in y
                elif read_orient == "L_R":
                    # z - dimension
                    mm_z = data_obj.fov_mm[0]
                    # x - dimension
                    mm_x = data_obj.fov_mm[1]
                    if data_obj.spatial_dimension == "3D":
                        # y - dimension
                        mm_y = data_obj.fov_mm[2]
                    else:
                        _, _, dim_y = define_imagematrix_parameters(data_obj=data_obj)
                        mm_y = data_obj.slice_thickness_mm
                        mm_y = mm_y * dim_y + mm_slice_gap * (dim_y - 1)

                else:
                    raise Exception(
                        f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                    )

            elif slice_orient == "axial sagittal coronal":
                raise Exception(
                    f"Slice orientation: {slice_orient} --> probably Localizer, not implemented yet!"
                )

            # for now do the same to get values:
            else:
                raise Exception("unknown slice orientation: " + slice_orient)

        else:
            raise Exception("unknown patient_position: " + patient_pos)

        return mm_z, mm_x, mm_y
