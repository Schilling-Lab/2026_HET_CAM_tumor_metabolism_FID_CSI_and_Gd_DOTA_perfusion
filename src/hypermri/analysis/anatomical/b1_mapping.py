"""Interactive B1 mapping utilities for anatomical calibration experiments."""

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from hypermri.analysis.fitting.fit_maps import fit_quantitative_maps
from hypermri.utils.system.logging import debug_log, init_default_logger

if TYPE_CHECKING:
    from hypermri.core.experiment import MRIExperiment

DB_MSG = True  # Change this to False to hide detailed error messages
DATA_PRECISION = np.float16


DEBUG_FILE = "/tmp/b1_mapping_debug.log"
DEBUG_LOG = True


def debug_log_b1_mapping(
    message="", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG, numpy_full_print=True
):
    """
    Debug logging system for B1 mapping operations.

    Parameters
    ----------
    message : str, optional
        The debug message to log. Default is empty string.
    file_name : str, optional
        Path to the log file. Default is DEBUG_FILE constant.
    write_to_file : bool, optional
        Whether to write the log to a file. Default is DEBUG_LOG constant.
    numpy_full_print : bool, optional
        Whether to print full numpy arrays without truncation. Default is True.

    Returns
    -------
    None
    """
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
        numpy_full_print=numpy_full_print,
    )


logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


def fit_b1_surface_coil(
    input_data: xr.DataArray,
    *,
    fit_params: Optional[dict] = None,
    dynamic_dim: Optional[str] = None,
    sort_by_power: bool = True,
) -> xr.Dataset:
    """
    Fit voxel-wise B1 maps for surface-coil calibrations using the period sweep solver.

    Parameters
    ----------
    input_data
        Complex-valued FLASH calibration data with an RF power dimension.
    fit_params
        Optional overrides forwarded to :func:`fit_quantitative_maps`.
    dynamic_dim
        Name of the RF power dimension. When ``None`` the function attempts
        to infer it from common naming patterns such as ``rfpower_idx``.
    sort_by_power
        When True (default) the data are sorted along ``dynamic_dim`` to ensure
        monotonic power sampling before fitting.

    Returns
    -------
    xr.Dataset
        Dataset with ``amplitude``, ``period``, ``refpower_90deg_W``,
        ``rss_b1``, and diagnostic fields needed by ``interactive_b1_fit_viewer``.
    """
    if fit_params is None:
        fit_params = {}
    fit_params_local = dict(fit_params)  # shallow copy to avoid in-place mutation

    if dynamic_dim is not None:
        fit_params_local["dynamic_dim"] = dynamic_dim

    candidate_dims = fit_params_local.get(
        "dynamic_dim_candidates",
        ["rfpower_idx", "power_idx", "rf_power_idx", "rfpower"],
    )
    dynamic_dim_resolved = fit_params_local.get("dynamic_dim")
    if dynamic_dim_resolved is None:
        dynamic_dim_resolved = next(
            (dim for dim in candidate_dims if dim in input_data.dims), None
        )
        if dynamic_dim_resolved is None:
            raise ValueError(
                "Unable to determine RF power dimension. Supply 'dynamic_dim' in fit_params "
                "or pass it explicitly to fit_b1_surface_coil()."
            )
        fit_params_local["dynamic_dim"] = dynamic_dim_resolved

    if dynamic_dim_resolved not in input_data.dims:
        raise ValueError(
            f"Resolved dynamic dimension '{dynamic_dim_resolved}' not found in input_data dims {input_data.dims}."
        )

    if dynamic_dim_resolved not in input_data.coords:
        raise ValueError(
            f"Input data must provide coordinate values for '{dynamic_dim_resolved}'."
        )

    working_data = input_data
    if sort_by_power:
        coord = input_data.coords[dynamic_dim_resolved]
        if coord.ndim == 1:
            working_data = input_data.sortby(dynamic_dim_resolved)
        else:
            logger.warning(
                "Skipping sort along '%s' because its coordinate spans multiple dimensions.",
                dynamic_dim_resolved,
            )

    logger.info(
        "Starting B1 fitting on data with shape %s (dynamic dim '%s', length %d).",
        working_data.shape,
        dynamic_dim_resolved,
        working_data.sizes[dynamic_dim_resolved],
    )

    result_ds = fit_quantitative_maps(
        input_data=working_data,
        fit_type="b1",
        fit_params=fit_params_local,
    )

    logger.info("B1 fitting finished. Result variables: %s", list(result_ds.data_vars))
    return result_ds


def find_nth_highest_signal_voxel(input_data, n: int = 1, plot: bool = False):
    """
    Find the voxel with the nth highest signal after averaging along repetitions.

    Parameters
    ----------
    input_data : np.ndarray
        Input data with shape (n_powers, 1, 1, nx, ny, n_repetitions, 1) or (nx, ny).
        For 7D data, it represents multiple RF powers, spatial dimensions (nx, ny),
        repetitions, and singletons. For 2D data, it represents a single image (nx, ny).
    n : int, optional
        The rank of the voxel to return (1 for highest, 2 for second highest, etc.).
        Default is 1 (highest signal voxel).
    plot : bool, optional
        Whether to plot the signal and the selected voxel. Default is False.

    Returns
    -------
    tuple
        (x, y) coordinates of the nth highest signal voxel.

    Notes
    -----
    This function handles both 7D and 2D input data. It averages the signal over repetitions
    and then finds the voxel with the nth highest total signal magnitude across all powers.
    """
    # Average along repetitions (axis 5) and remove singleton dimensions
    if np.ndim(input_data) == 7:
        averaged_data = np.squeeze(np.nanmean(input_data, axis=(1, 2, 5, 6)))
    else:
        averaged_data = input_data

    # Sum signal magnitude across all powers
    total_signal = np.nansum(np.abs(averaged_data), axis=0)

    # Flatten the array and get the indices that would sort it in descending
    # order
    flat_indices = np.argsort(total_signal.ravel())[::-1]

    # Get the nth highest index (subtracting 1 because n starts from 1)
    nth_highest_flat_index = flat_indices[n - 1]

    # Convert the flat index back to 2D coordinates
    max_index = np.unravel_index(nth_highest_flat_index, total_signal.shape)

    if plot:
        plt.figure()
        plt.imshow(np.squeeze(np.abs(total_signal)), cmap="gray")
        plt.scatter(
            max_index[1],
            max_index[0],
            c="r",
            s=100,
            marker="x",
            label=f"{n}th highest signal voxel",
        )
        plt.legend()
        plt.show()

    return max_index


def interactive_b1_fit_viewer(
    b1_dataset: xr.Dataset,
    input_data: xr.DataArray,
    *,
    dynamic_dim: str = "rfpower_idx",
    static_indexers: Optional[dict] = None,
    abs_amplitude: bool = True,
    rfpulse_duration_ms: float = 1.0,
    rfpulse_sint: float = 1.0,
):
    """
    Launch an interactive viewer for inspecting B1 fits.

    Clicking on the amplitude image plots the raw signal, fitted curve,
    fit window (center/extent), and stop index for the selected voxel.

    Parameters
    ----------
    b1_dataset : xr.Dataset
        Output from ``fit_quantitative_maps`` containing at least the DataArrays
        ``amplitude``, ``period``, ``stop_index``, ``fit_window_center``,
        and ``fit_window_extent``.
    input_data : xr.DataArray
        Original measurement data passed to ``fit_quantitative_maps``.
    dynamic_dim : str, optional
        Name of the RF power dimension. Default is ``"rfpower_idx"``.
    static_indexers : dict, optional
        Dimension selections (passed to ``isel``) for reducing additional
        dimensions such as slices or coils before plotting.
    abs_amplitude : bool, optional
        When True (default), the displayed image is ``|amplitude|``.
        Set to False to view the real part instead.
    """
    import matplotlib.pyplot as plt

    if static_indexers is None:
        static_indexers = {}

    if "amplitude" not in b1_dataset:
        raise ValueError("b1_dataset must contain an 'amplitude' DataArray.")
    required_fields = {
        "period",
        "stop_index",
    }
    missing = required_fields.difference(b1_dataset.data_vars)
    if missing:
        raise ValueError(f"b1_dataset is missing required fields: {sorted(missing)}")

    if dynamic_dim not in input_data.dims:
        raise ValueError(f"dynamic_dim '{dynamic_dim}' not found in input_data.")

    # Reduce extra dimensions up front for both dataset and input data
    b1_ds_view = b1_dataset.isel(**static_indexers).squeeze(drop=True)
    input_view = input_data.isel(**static_indexers).squeeze(drop=True)

    spatial_dims = [dim for dim in b1_ds_view["amplitude"].dims if dim != dynamic_dim]
    if dynamic_dim in spatial_dims:
        spatial_dims.remove(dynamic_dim)
    if len(spatial_dims) != 2:
        raise ValueError(
            "Viewer requires exactly two non-dynamic spatial dimensions after applying static_indexers."
        )

    amp_da = b1_ds_view["amplitude"]
    period_da = b1_ds_view["period"]
    stop_da = b1_ds_view["stop_index"]
    refpower_90deg_W_da = b1_ds_view["refpower_90deg_W"]

    # cent_da = b1_ds_view["fit_window_center"]
    # ext_da = b1_ds_view["fit_window_extent"]

    if abs_amplitude:
        display_data = np.abs(amp_da.values)
    else:
        display_data = np.real(amp_da.values)

    if dynamic_dim not in input_view.coords:
        raise ValueError(
            f"input_data is missing coordinate values for '{dynamic_dim}'."
        )
    # sqrt_refpower = np.sqrt(np.asarray(input_view.coords[dynamic_dim].values, dtype=float))
    sqrt_refpower = np.asarray(input_view.coords[dynamic_dim].values, dtype=float)

    # Prepare figure with image and plot panels
    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(8, 5))
    im = ax_img.imshow(
        display_data,
        origin="lower",
        cmap="magma",
    )
    ax_img.set_title("B1 amplitude" + (" (abs)" if abs_amplitude else " (real)"))
    ax_img.set_xlabel(spatial_dims[1])
    ax_img.set_ylabel(spatial_dims[0])
    fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

    marker = ax_img.scatter(
        [], [], s=60, facecolors="none", edgecolors="cyan", linewidths=1.5, marker="x"
    )

    ax_plot.set_xlabel("sqrt(refpower) [W$^{1/2}$]")
    ax_plot.set_ylabel("Signal")
    ax_plot.set_title("Click a voxel to inspect fit")

    def update_plot(col_idx: int, row_idx: int):
        if (
            row_idx < 0
            or row_idx >= display_data.shape[0]
            or col_idx < 0
            or col_idx >= display_data.shape[1]
        ):
            return

        marker.set_offsets([[col_idx, row_idx]])

        isel_dict = {spatial_dims[0]: row_idx, spatial_dims[1]: col_idx}
        signal = input_view.isel(isel_dict).values
        amplitude = amp_da.isel(isel_dict).item()
        period = period_da.isel(isel_dict).item()
        stop_index_val = stop_da.isel(isel_dict).item()
        refpower_90deg_W_val = refpower_90deg_W_da.isel(isel_dict).item()
        refpower_90deg_W_val = (
            refpower_90deg_W_val / (rfpulse_duration_ms * rfpulse_sint) ** 2
        )

        # center_val = cent_da.isel(isel_dict).item()
        # extent_val = ext_da.isel(isel_dict).item()

        ax_plot.clear()
        ax_plot.set_xlabel("sqrt(RF Power) [W$^{1/2}$]")
        ax_plot.set_ylabel("Signal")

        ax_plot.plot(
            sqrt_refpower, np.real(signal), "o-", label="Signal (real)", linewidth=0.25
        )
        ax_plot.plot(
            sqrt_refpower,
            np.imag(signal),
            "o-",
            label="Signal (imag)",
            alpha=0.7,
            linewidth=0.25,
        )
        ax_plot.plot(
            sqrt_refpower,
            np.abs(signal),
            "o-",
            label="Signal (abs)",
            alpha=0.7,
            linewidth=1.5,
        )

        sqrt_refpower_high = np.linspace(sqrt_refpower.min(), sqrt_refpower.max(), 500)

        if np.isfinite(period) and period > 0:
            fit_curve = amplitude * np.sin(np.pi / period * np.sqrt(sqrt_refpower_high))
            ax_plot.plot(
                sqrt_refpower_high,
                np.real(fit_curve),
                "-",
                label="Fit (real)",
                linewidth=0.25,
            )
            ax_plot.plot(
                sqrt_refpower_high,
                np.imag(fit_curve),
                "-",
                label="Fit (imag)",
                alpha=0.7,
                linewidth=0.25,
            )
            ax_plot.plot(
                sqrt_refpower_high,
                np.abs(fit_curve),
                "-",
                label="Fit (abs)",
                alpha=0.77,
                linewidth=1.5,
            )

        # if np.isfinite(center_val) and np.isfinite(extent_val):
        # lower = max(center_val - extent_val, sqrt_refpower.min())
        # upper = min(center_val + extent_val, sqrt_refpower.max())
        # ax_plot.axvspan(lower, upper, color="orange", alpha=0.1, label="Fit window")
        # ax_plot.axvline(center_val, color="orange", linestyle="--", linewidth=1)

        ax_plot.axvline(refpower_90deg_W_val)

        if np.isfinite(stop_index_val):
            stop_idx = int(stop_index_val)
            stop_idx = np.clip(stop_idx, 0, sqrt_refpower.size - 1)
            ax_plot.axvline(
                sqrt_refpower[stop_idx],
                color="red",
                linestyle=":",
                linewidth=1.2,
                label=f"Stop index ({stop_idx})",
            )

        ax_plot.legend(loc="upper right")
        ax_plot.set_title(
            f"{spatial_dims[0]}={row_idx}, {spatial_dims[1]}={col_idx}, stop_idx={stop_index_val:.1f}\nReference Power={refpower_90deg_W_val}W"
        )
        plt.tight_layout()
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax_img or event.xdata is None or event.ydata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        update_plot(col, row)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    # return fig


def refpow_parameter_adjustment(
    refpower_W: float,
    data_obj: "MRIExperiment",
    trf_ms: float = 1,
    sint: float = 1,
    fa_deg: float = 90,
):
    """Scale reference power according to the applied pulse parameters.

    Parameters
    ----------
    refpower_W : float or array_like
        Reference power in Watts measured during calibration.
    data_obj : MRIExperiment, optional
        Sequence object that exposes ``rfpulse_duration_ms``, ``flipangle_deg``,
        and ``rfpulse_sint`` attributes. When provided the remaining parameters
        are read from the object.
    trf_ms : float, optional
        RF pulse duration in milliseconds when ``data_obj`` is ``None``.
    sint : float, optional
        Integrated RF pulse shape factor.
    fa_deg : float, optional
        Reference flip angle in degrees.

    Returns
    -------
    numpy.ndarray or float
        Adjusted reference power that compensates for the actual pulse setup.
    """
    if data_obj is None:
        pass
    else:
        trf_ms = data_obj.rfpulse_duration_ms
        fa_deg = data_obj.flipangle_deg
        sint = data_obj.rfpulse_sint

    refpower_parameter_adjusted_W = refpower_W * (90 * trf_ms * sint / fa_deg) ** 2
    return refpower_parameter_adjusted_W


def refpow_from_sin_fit(periods: float = 0):
    """Convert sine-fit periods into reference power estimates. This assumes that a blockpulse
    of 1ms duration and integration factor of 1 was used during calibration.

    If that is not the case, the resulting reference power values have to be adjusted
    using ``refpow_parameter_adjustment``.

    Fitted curve: S = A * sin(π / period * sqrt(refpower))
    Reference power for 90 degree flip angle: refpower_90deg = (0.5 * period) ** 2

    Parameters
    ----------
    periods : array_like or float
        Period values obtained from the sinusoidal calibration curve.

    Returns
    -------
    numpy.ndarray or float
        Reference power values in Watts computed as ``(0.5 * periods)**2``.
    """
    refpow = (0.5 * periods) ** 2
    return refpow


def plot_refpow_map_overlay(
    anatomical_image,
    refpow_map,
    anatomical_extent,
    b1_extent,
    fa_deg=90,
    trf_ms=1,
    sint=1,
    data_obj: "MRIExperiment" = None,
    rfpow_W=None,
    plot_params={},
):
    """
    Plot an interactive overlay of B1 map and resulting flip angle map onto an anatomical image using ipywidgets.

    Parameters
    ----------
    anatomical_image : ndarray
        Array containing the anatomical image data. Expected shape: (1, slices, x, y, 1, 1)
    refpow_map : ndarray
        Array containing the B1 map data (reference power map). Expected shape: (1, 1, x, y, 1, 1)
    anatomical_extent : tuple
        (left, right, bottom, top) values for the anatomical image extent.
    b1_extent : tuple
        (left, right, bottom, top) values for the B1 map extent.
    fa_deg : float, optional
        Initial defined flip angle for reference power. Default is 90.
    trf_ms : float, optional
        Initial pulse duration in milliseconds. Default is 1.
    sint : float, optional
        Initial integration factor. Default is 1.
    data_obj : MRIExperiment, optional
        Data object containing method parameters. Default is None.
    rfpow_W : float, optional
        Initial RF power in Watts. Default is 1.
    plot_params : dict, optional
        Additional plotting parameters. Default is an empty dictionary.

    Returns
    -------
    None
    """

    import ipywidgets as widgets
    from IPython.display import display
    from ipywidgets import HBox, Layout, VBox, fixed, interactive

    from ...utils.image.interpolation import img_interp
    from ...utils.physics import calc_fa_deg

    if data_obj is not None:
        fa_deg = data_obj.flipangle_deg
        trf_ms = data_obj.rfpulse_duration_ms
        sint = data_obj.rfpulse_sint
        rfpow_W = data_obj.rfpulse_peakpow_W

    if np.ndim(refpow_map) != 6:
        # assume 2D B1 map:
        refpow_map = np.expand_dims(refpow_map, axis=(0, 1, 4, 5))

    anatomical_image = np.rot90(anatomical_image, k=1, axes=(2, 3))
    refpow_map = np.rot90(refpow_map, k=1, axes=(2, 3))

    def mri_signal_intensity(flip_angle):
        return np.abs(np.sin(np.deg2rad(flip_angle)))

    def interpolate_refpow_map(refpow_map, factor):
        if factor == 1:
            return refpow_map
        refpow_map_temp = refpow_map.copy()
        refpow_map_temp = np.where(refpow_map_temp == np.nan, 0, refpow_map_temp)
        interpolated_refow_map = img_interp(
            input_data=refpow_map_temp, interp_factor=factor
        )
        return interpolated_refow_map

    figsize = plot_params.get("figsize", (9, 6))
    fig, ax = plt.subplots(figsize=figsize)

    def update(
        rfpow_W,
        alpha,
        interp_factor,
        fa_deg,
        trf_ms,
        sint,
        slice_index,
        map_type,
        mri_signal_mode,
    ):
        # Clear any previous output
        global click_info
        ax.clear()  # Clear the axes for the new plot
        # Create a text annotation for displaying click information

        click_info = ax.annotate(
            "Click in the image to get information",
            xy=(0, 0),
            xytext=(0, 0),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        click_info.set_fontsize(5)
        click_info.set_visible(True)

        if anatomical_image.shape[1] > 1:
            anatomical_slice = anatomical_image[0, slice_index, :, :, 0, 0]
        else:
            anatomical_slice = anatomical_image[0, 0, :, :, 0, 0]

        b1_slice = interpolate_refpow_map(refpow_map, interp_factor)[0, 0, :, :, 0, 0]

        # Plot the anatomical image
        im_anatomical = ax.imshow(
            np.squeeze(anatomical_slice),
            extent=anatomical_extent,
            cmap="bone",
        )
        flip_angle_map = calc_fa_deg(
            rfpow_W=rfpow_W,
            reference_pow_W=b1_slice,
            trf_ms=trf_ms,
            sint=sint,
            fa_deg=fa_deg,
        )
        if map_type == "B1 Map":
            overlay_data = b1_slice
        else:
            if mri_signal_mode:
                overlay_data = mri_signal_intensity(flip_angle_map)
            else:
                overlay_data = flip_angle_map

        # Plot the B1 map overlay
        im_overlay = ax.imshow(
            np.squeeze(overlay_data),
            extent=b1_extent,
            cmap="viridis",
            alpha=alpha,
        )

        title = f"RF Power: {rfpow_W: .3f} W, Set FA: {fa_deg: .1f}°, \nPulse Duration: {trf_ms: .1f} ms, Int. Factor: {sint: .2f}"
        if anatomical_image.shape[1] > 1:
            title = f"Slice {slice_index}, " + title
        ax.set_title(title)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata

            # Calculate the position in the refpow_map array
            col = int(
                (x - b1_extent[0]) / (b1_extent[1] - b1_extent[0]) * refpow_map.shape[2]
            )
            row = int(
                (b1_extent[3] - y) / (b1_extent[3] - b1_extent[2]) * refpow_map.shape[3]
            )
            flip_angle_map = calc_fa_deg(
                rf_power_widget.value,
                refpow_map,
                trf_widget.value,
                sint_widget.value,
                fa_widget.value,
            )

            if 0 <= row < refpow_map.shape[3] and 0 <= col < refpow_map.shape[2]:
                ref_power = refpow_map[0, 0, row, col, 0, 0]

                # Update the click_info annotation
                click_info.xy = (x, y)
                click_info.set_text(
                    f"Reference Power: {ref_power:.4f} W,\nFlip angle: {flip_angle_map[0, 0, row, col, 0, 0]:.2f}°"
                )
                click_info.set_visible(True)
                fig.canvas.draw_idle()
                print(f"x={x:.2f}, y={y:.2f}, ref_power={ref_power:.4f}")
            else:
                click_info.set_visible(False)
                fig.canvas.draw_idle()

            title = f"RF Power: {rf_power_widget.value: .3f} W, Set FA: {fa_widget.value: .1f}°, \nPulse Duration: {trf_widget.value: .1f} ms, Int. Factor: {sint_widget.value: .2f}\nReference Power: {ref_power: .4f}\nFlipangle: {flip_angle_map[ 0, 0, row, col, 0, 0]: .2f}°"
            ax.set_title(title)

    fig.canvas.mpl_connect("button_press_event", on_click)

    # Create ipywidgets with consistent widths
    slider_layout = Layout(width="600px")
    rf_power_widget = widgets.FloatSlider(
        value=rfpow_W,
        min=0.001,
        max=1.000,
        step=0.001,
        description="RF Power (W)",
        layout=slider_layout,
        readout_format=".3f",
    )
    alpha_widget = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1,
        step=0.01,
        description="Transparency",
        layout=slider_layout,
    )
    interp_widget = widgets.IntSlider(
        value=1, min=1, max=5, step=1, description="Interp Factor", layout=slider_layout
    )
    fa_widget = widgets.FloatSlider(
        value=fa_deg,
        min=0,
        max=180,
        step=1,
        description="Set FA (deg)",
        layout=slider_layout,
    )
    trf_widget = widgets.FloatSlider(
        value=trf_ms,
        min=0.1,
        max=10,
        step=0.1,
        description="Pulse Duration (ms)",
        layout=slider_layout,
    )
    sint_widget = widgets.FloatSlider(
        value=sint,
        min=0.1,
        max=1,
        step=0.01,
        description="Int. Factor",
        layout=slider_layout,
    )

    map_type_widget = widgets.RadioButtons(
        options=["B1 Map", "Flip Angle"],
        value="B1 Map",
        description="Map Type",
        layout=Layout(width="200px"),
    )
    mri_signal_widget = widgets.Checkbox(
        value=False, description="MRI Signal", layout=Layout(width="150px")
    )

    if anatomical_image.shape[1] > 1:
        slice_widget = widgets.IntSlider(
            value=0,
            min=0,
            max=anatomical_image.shape[1] - 1,
            step=1,
            description="Slice",
            layout=slider_layout,
        )
    else:
        slice_widget = fixed(0)

    # Organize widgets into two rows
    row1 = HBox([rf_power_widget, fa_widget, trf_widget])
    row2 = HBox([alpha_widget, interp_widget, sint_widget])

    if anatomical_image.shape[1] > 1:
        row2 = HBox([alpha_widget, interp_widget, sint_widget, slice_widget])

    control_column = VBox([map_type_widget, mri_signal_widget])

    # Combine all widgets
    all_widgets = VBox([row1, row2, control_column])

    # Create the interactive plot
    interactive_plot = interactive(
        update,
        rfpow_W=rf_power_widget,
        alpha=alpha_widget,
        interp_factor=interp_widget,
        fa_deg=fa_widget,
        trf_ms=trf_widget,
        sint=sint_widget,
        slice_index=slice_widget,
        map_type=map_type_widget,
        mri_signal_mode=mri_signal_widget,
    )

    # Display the interactive plot
    display(VBox([all_widgets, fig.canvas]))

    # Initial update to show the plot
    update(rfpow_W, 0.5, 1, fa_deg, trf_ms, sint, 0, "B1 Map", False)


def b1_calibration(
    data_obj: "MRIExperiment" = None, fit_params={}, plot=True, db=False, **kwargs
):
    """
    Perform B1 calibration based on provided data and fitting parameters.

    Parameters
    ----------
    data_obj : MRIExperiment, optional
        The data object containing MRI data and relevant acquisition parameters.
        Must contain attributes `method`, `n_repetitions`, `fid`, `rfpulse_duration_ms`, and `rfpulse_sint`.
        The `method` attribute is a dictionary containing pulse power information.

    fit_params : dict, optional
        A dictionary of fitting parameters to be passed to the spectral fitting function.
        If not provided, default parameters will be used. The dictionary may contain:

            - 'metabs': list of str, default is ["peak1"]
            - 'metabs_freqs_Hz': float, default is 0
            - 'cut_off': int, default is 70
            - 'provided_dims': list of str, default is ["reps", "spec"]
            - 'provided_dims_ind': list of int, default is [4, 0]
            - 'signal_domain': str, default is "temporal"

    plot : bool, optional
        If True, generates plots of the fitted data and calibration results. Default is True.

    db : bool, optional
        If True, additional diagnostic plots will be generated. Default is False.

    Returns
    -------
    refpower_W : float
        The fitted reference power (in Watts) corresponding to a 90-degree flip angle.

    Notes
    -----
    The function performs a sinusoidal fit of the signal amplitudes as a function of RF power.
    The reference power is determined from the fit and adjusted based on the pulse duration
    and integration factor.

    The method for generating the RF power array depends on the pulse method used in the
    experiment, specified in `data_obj.method['Method']`.

    Fitting is done using a pseudo-inverse approach to obtain the spectral fit,
    followed by a single voxel fit to determine the sinusoidal relationship between
    signal amplitudes and RF power.

    The user can visualize the results through plots, including the fitted sinusoidal curve
    and the corresponding RF power at the 90-degree flip angle.

    If `db` is enabled, additional plots of local R^2 values against fit periods
    are generated to provide diagnostic information about the fitting process.


    """

    rfpower_start = data_obj.method["PulseVectorStart"]
    rfpower_end = data_obj.method["PulseVectorEnd"]
    n_repetitions = data_obj.n_repetitions
    n_channels = data_obj.n_receivers
    rfpower_array = []
    if data_obj.method["Method"] == "<User:lucaB1shapelist_1>":
        rfpower_array = (
            np.linspace(np.sqrt(rfpower_start), np.sqrt(rfpower_end), n_repetitions)
            ** 2
        )
    else:
        rfpower_array = np.linspace(rfpower_start, rfpower_end, n_repetitions)
    fid = data_obj.fid
    local_stop = kwargs.get("local_stop", -1)
    initial_guess_cent = kwargs.get("initial_guess_cent", 3.1)
    kwargs.get("initial_guess_extent", 3.05)
    use_multiprocessing = kwargs.get("use_multiprocessing", True)
    if n_channels > 1:
        fid = np.reshape(fid, [data_obj.n_repetitions, data_obj.n_receivers, -1])
        from ..fitting.spectroscopic_fitting import def_fit_params, fit_data_pseudo_inv
        from ..spectroscopy.core import make_NDspec_6Dspec

        if fit_params == {}:
            fit_params["metabs"] = ["peak1"]
            fit_params["metabs_freqs_Hz"] = 0
            fit_params["cut_off"] = 70
            fit_params["signal_domain"] = "temporal"
        else:
            pass
        fit_params["provided_dims"] = ["reps", "channels", "spec"]
        fit_params["provided_dims_ind"] = [4, 5, 0]
        fid = np.reshape(fid, (n_repetitions, n_channels, -1))
        fid = make_NDspec_6Dspec(
            input_data=fid, provided_dims=fit_params["provided_dims"]
        )
        fit_params = def_fit_params(data_obj=data_obj, fit_params=fit_params)
        fit_repetitions = len(fit_params["fit_range_repetitions"])

        fit_results = fit_data_pseudo_inv(
            input_data=fid,
            fit_params=fit_params,
            data_obj=data_obj,
            use_multiprocessing=use_multiprocessing,
        )
        for n_chan in range(n_channels):
            print(f"Channel {n_chan + 1}/{n_channels}")
            if plot:
                from ...utils.plot.utils_plot import subplot_auto_size

                row, cols = subplot_auto_size(fit_repetitions)
                fig, ax = plt.subplots(row, cols, figsize=(10, 10))
                input_data_ft = np.fft.fftshift(
                    np.fft.fft(fit_results["input_data"].squeeze(), axis=0), axes=0
                )
                ax = ax.flatten()
                for r in range(fit_results["fit_spectrum"].squeeze().shape[1]):
                    ax[r].plot(
                        fit_params["freq_axis_hz"],
                        np.real(fit_results["fit_spectrum"].squeeze()[:, r, n_chan]),
                    )
                    if fit_params["signal_domain"] == "temporal":
                        ax[r].plot(
                            fit_params["freq_axis_hz"],
                            np.real(input_data_ft[:, r, n_chan]),
                            alpha=0.5,
                        )
                        ax[r].set_ylim(
                            1.1 * np.real(input_data_ft).min(),
                            1.1 * np.real(input_data_ft).max(),
                        )
                    else:
                        ax[r].plot(
                            fit_params["freq_axis_hz"],
                            np.real(fit_results["input_data"].squeeze()[:, r, n_chan]),
                            alpha=0.5,
                        )
                        ax[r].set_ylim(
                            1.1 * np.real(fit_results["input_data"]).min(),
                            1.1 * np.real(fit_results["input_data"]).max(),
                        )
                    ax[r].set_xlabel("Frequency [Hz]")
                    ax[r].set_title(f"RF Power {rfpower_array[r]:.4f} W")

                plt.tight_layout()
                plt.show()

            amplitude, period, fit_data, local_r2, local_per = fit_single_voxel(
                input_data=np.squeeze(
                    fit_results["fit_amps"].squeeze()[:, n_chan][:local_stop]
                ),
                refpower_W=rfpower_array[:local_stop],
                plot=plot,
                initial_guess_cent=initial_guess_cent,
                initial_guess_extent=initial_guess_cent,
                npoints=1000,
                niter=10,
                zoom_factor=1.75,
                stop_index_extra=int(fit_params.get("stop_index_extra", 0)),
            )
            # create the A matrix (modelled signal)

            locacl_fit = amplitude * np.sin(
                np.pi / period * np.sqrt(rfpower_array[:, np.newaxis])
            )
            refpower_W = refpow_from_sin_fit(periods=period)
            refpower_W = refpow_parameter_adjustment(
                refpower_W=refpower_W,
                trf_ms=data_obj.rfpulse_duration_ms,
                sint=data_obj.rfpulse_sint,
                fa_deg=90,
            )

            if db:
                from ...utils.plot.utils_plot import subplot_auto_size

                row, cols = subplot_auto_size(local_r2.shape[0])
                fig, ax = plt.subplots(row, cols)
                ax = ax.flatten()
                for a in range(local_r2.shape[0]):
                    ax[a].plot(local_per[a, :], local_r2[a, :])
                    ax[a].set_xlabel("Period")
                    ax[a].set_ylabel("R2")
                plt.tight_layout()

            if plot:
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].plot(
                    np.sqrt(rfpower_array),
                    np.real(fit_results["fit_amps"].squeeze()[:, n_chan]),
                    "o-",
                    label="Real part",
                )
                ax[0].plot(
                    np.sqrt(rfpower_array),
                    np.imag(fit_results["fit_amps"].squeeze()[:, n_chan]),
                    "--",
                    label="Imaginary part",
                )
                ax[0].plot(
                    np.sqrt(rfpower_array),
                    np.abs(fit_results["fit_amps"].squeeze()[:, n_chan]),
                    "--",
                    label="Magnitude",
                )
                ax[0].plot(
                    np.sqrt(rfpower_array),
                    np.abs(locacl_fit),
                    "--",
                    label="Fit",
                )

                ax[0].set_xlabel("sqrt(refpower) [W]")
                ax[0].set_ylabel("Signal")
                ax[0].axhline(0, color="gray", linestyle="--", linewidth=0.5)
                ax[0].legend()
                ax[0].set_title("B1 Calibration (sinusoidal fit)")
                ax[1].plot(
                    rfpower_array,
                    np.abs(fit_results["fit_amps"].squeeze()[:, n_chan]),
                    "*-",
                    label="Magnitude",
                )
                ax[1].plot(
                    rfpower_array,
                    np.real(fit_results["fit_amps"].squeeze()[:, n_chan]),
                    "--",
                    label="Real part",
                )
                ax[1].plot(
                    rfpower_array,
                    np.imag(fit_results["fit_amps"].squeeze()[:, n_chan]),
                    "--",
                    label="Imaginary part",
                )

                ax[1].set_xlabel("refpower [W]")
                ax[1].set_ylabel("Signal")
                ax[1].set_title(
                    f"B1 Calibration \n TRF: {data_obj.rfpulse_duration_ms}ms\n sint: {data_obj.rfpulse_sint}"
                )
                ax[1].axvline(
                    refpower_W,
                    label=f"Fitted refpower = {np.round(refpower_W, 4)}W",
                    color="red",
                )
                ax[1].axhline(0, color="gray", linestyle="--", linewidth=0.5)
                if local_stop != -1:
                    ax[1].axvline(
                        rfpower_array[local_stop],
                        label=f"Stop index = {local_stop}",
                        color="black",
                    )
                ax[1].legend()
                plt.tight_layout()
            print(f"Fitted refpower Ch{n_chan + 1}: {refpower_W:.6f} W")
            if data_obj.rfpulse_duration_ms != 1 or data_obj.rfpulse_sint != 1:
                print(
                    f"If the shown reference power (red line) is not at the peak of the sinusoidal fit [ ^= 90 degree flipangle], \nconsider that the pulse parameters are TRF = {data_obj.rfpulse_duration_ms} ms and integration factor={data_obj.rfpulse_sint} and that the reference power was \nscaled by these parameters."
                )
        return refpower_W
    else:
        from ..fitting.spectroscopic_fitting import def_fit_params, fit_data_pseudo_inv

        fid = np.reshape(fid, [data_obj.n_repetitions, -1])
        if fit_params == {}:
            fit_params["metabs"] = ["peak1"]
            fit_params["metabs_freqs_Hz"] = 0
            fit_params["cut_off"] = 70
            fit_params["signal_domain"] = "temporal"
        else:
            pass
        fit_params["provided_dims"] = ["reps", "spec"]
        fit_params["provided_dims_ind"] = [4, 0]
        fit_params = def_fit_params(data_obj=data_obj, fit_params=fit_params)

        fit_results = fit_data_pseudo_inv(
            input_data=fid,
            fit_params=fit_params,
            data_obj=data_obj,
            use_multiprocessing=use_multiprocessing,
        )

        fit_repetitions = len(fit_params["fit_range_repetitions"])

        if plot:
            from ...utils.plot.utils_plot import subplot_auto_size

            row, cols = subplot_auto_size(fit_repetitions)
            fig, ax = plt.subplots(row, cols, figsize=(10, 10))
            input_data_ft = np.fft.fftshift(
                np.fft.fft(fit_results["input_data"].squeeze(), axis=0), axes=0
            )
            ax = ax.flatten()
            for r in range(fit_repetitions):
                ax[r].plot(
                    fit_params["freq_axis_hz"],
                    np.real(fit_results["fit_spectrum"].squeeze()[:, r]),
                )
                if fit_params["signal_domain"] == "temporal":
                    ax[r].plot(
                        fit_params["freq_axis_hz"],
                        np.real(input_data_ft[:, r]),
                        alpha=0.5,
                    )
                    ax[r].set_ylim(
                        1.1 * np.real(input_data_ft).min(),
                        1.1 * np.real(input_data_ft).max(),
                    )
                else:
                    ax[r].plot(
                        fit_params["freq_axis_hz"],
                        np.real(fit_results["input_data"].squeeze()[:, r]),
                        alpha=0.5,
                    )
                    ax[r].set_ylim(
                        1.1 * np.real(fit_results["input_data"]).min(),
                        1.1 * np.real(fit_results["input_data"]).max(),
                    )
                ax[r].set_xlabel("Frequency [Hz]")
                ax[r].set_title(f"RF Power {rfpower_array[r]:.4f} W")

            plt.tight_layout()

        amplitude, period, fit_data, local_r2, local_per = fit_single_voxel(
            input_data=np.squeeze(fit_results["fit_amps"].squeeze()[:local_stop]),
            refpower_W=rfpower_array[:local_stop],
            plot=plot,
            initial_guess_cent=initial_guess_cent,
            initial_guess_extent=initial_guess_cent,
            npoints=1000,
            niter=10,
            zoom_factor=1.75,
            stop_index_extra=int(fit_params.get("stop_index_extra", 0)),
        )
        locacl_fit = amplitude * np.sin(
            np.pi / period * np.sqrt(rfpower_array[:, np.newaxis])
        )
        refpower_W = refpow_from_sin_fit(periods=period)
        refpower_W = refpow_parameter_adjustment(
            refpower_W=refpower_W,
            trf_ms=data_obj.rfpulse_duration_ms,
            sint=data_obj.rfpulse_sint,
            fa_deg=90,
        )

        if db:
            from ...utils.plot.utils_plot import subplot_auto_size

            row, cols = subplot_auto_size(local_r2.shape[0])
            fig, ax = plt.subplots(row, cols)
            ax = ax.flatten()
            for a in range(local_r2.shape[0]):
                ax[a].plot(local_per[a, :], local_r2[a, :])
                ax[a].set_xlabel("Period")
                ax[a].set_ylabel("R2")
            plt.tight_layout()

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(
                np.sqrt(rfpower_array),
                np.abs(np.squeeze(fit_results["fit_amps"])),
                "*-",
                label="Magnitude",
            )
            ax[0].plot(
                np.sqrt(rfpower_array),
                np.real(np.squeeze(fit_results["fit_amps"])),
                "--",
                label="Real part",
            )
            ax[0].plot(
                np.sqrt(rfpower_array),
                np.imag(np.squeeze(fit_results["fit_amps"])),
                "--",
                label="Imaginary part",
            )

            ax[0].plot(
                np.sqrt(rfpower_array),
                np.abs(locacl_fit),
                "--",
                label="Fit",
            )
            ax[0].set_xlabel("sqrt(refpower) [W]")
            ax[0].set_ylabel("Signal")
            ax[0].axhline(0, color="gray", linestyle="--", linewidth=0.5)
            ax[0].legend()
            ax[0].set_title("B1 Calibration (sinusoidal fit)")
            ax[1].plot(
                rfpower_array,
                np.real(np.squeeze(fit_results["fit_amps"])),
                "o-",
                label="Real part",
            )
            ax[1].plot(
                rfpower_array,
                np.imag(np.squeeze(fit_results["fit_amps"])),
                "o-",
                label="Imaginary part",
            )
            ax[1].plot(
                rfpower_array,
                np.abs(np.squeeze(fit_results["fit_amps"])),
                "--",
                label="Magnitude",
            )
            ax[1].plot(
                rfpower_array,
                np.abs(locacl_fit),
                "--",
                label="Fit",
            )
            ax[1].set_xlabel("refpower [W]")
            ax[1].set_ylabel("Signal")
            ax[1].set_title(
                f"B1 Calibration \n TRF: {data_obj.rfpulse_duration_ms}ms\n sint: {data_obj.rfpulse_sint}"
            )
            ax[1].axvline(
                refpower_W,
                label=f"Fitted refpower = {np.round(refpower_W, 4)}W",
                color="red",
            )
            ax[1].axhline(0, color="gray", linestyle="--", linewidth=0.5)
            if local_stop != -1:
                ax[1].axvline(
                    rfpower_array[local_stop],
                    label=f"Stop index = {local_stop}",
                    color="black",
                )
            ax[1].legend()
            plt.tight_layout()
            plt.show()  # Explicitly display the figure

        print(f"Fitted refpower: {refpower_W:.6f} W")
        if data_obj.rfpulse_duration_ms != 1 or data_obj.rfpulse_sint != 1:
            print(
                f"If the shown reference power (red line) is not at the peak of the sinusoidal fit [ ^= 90 degree flipangle], \nconsider that the pulse parameters are TRF = {data_obj.rfpulse_duration_ms} ms and integration factor={data_obj.rfpulse_sint} and that the reference power was \nscaled by these parameters."
            )

        return refpower_W
