import warnings
from abc import ABC, abstractmethod

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ipywidgets import (
    HTML,
    Checkbox,
    Dropdown,
    FloatSlider,
    HBox,
    IntSlider,
    Label,
    Layout,
    ToggleButtons,
    VBox,
    Widget,
    fixed,
)

from hypermri.utils.plot.utils_plot import common_cmaps


class BasePlotter(ABC):
    """
    An abstract base class for creating interactive plotters.

    This class provides a framework for setting up interactive plots using
    ipywidgets. It utilizes the Template Method design pattern to allow
    subclasses to define specific behaviors while sharing common setup and
    widget creation logic.

    Attributes
    ----------
    _denoised_cache : dict
        A cache to store results of denoising operations.
    _last_original_data_id : int
        The ID of the last processed data array to manage caching.
    state : list
        A list to hold the state of the plot, typically including the
        slicing dimension and clicked coordinates.
    widgets : dict
        A dictionary to store the created widget objects.
    plot_handles : object
        An object to store handles to Matplotlib plot elements.

    """

    def __init__(self):
        """Initializes the BasePlotter."""
        self._denoised_cache = {}
        self._last_original_data_id = None
        self.state = [None, None]  # [slicing_dim, clicked_indices]
        self.widgets = {}
        self.plot_handles = type("PlotHandles", (), {})()

    def _process_plot_params(self, plot_params: dict = None) -> dict:
        """
        Processes the input plot_params dictionary, applying default values
        where parameters are not explicitly provided.

        Parameters
        ----------
        plot_params : dict, optional
            A dictionary of plot parameters provided by the user.

        Returns
        -------
        dict
            A new dictionary with all plot parameters, including defaults.

        """
        if plot_params is None:
            plot_params = {}

        # Define default plot parameters
        default_params = {
            "colormap": "bone",
            "overlay_colormap": "magma",
            "continuous_update": True,
            "figsize": (16, 5),
            "vmin_perc": 0.0,
            "vmax_perc": 100.0,
            "combine_channels": True,
            "suptitle": None,
            "show_suptitle": True,
            "show_title": True,
            "default_spec_part": "Magnitude",
            "default_secondary_dim": None,
            "main_image_tick_labelsize": 8,
            "secondary_plot_tick_labelsize": 8,
            "main_image_xlabel": None,
            "main_image_ylabel": None,
            "secondary_plot_xlabel": None,
            "secondary_plot_ylabel": None,
            "title": None,
            "xlabel": None,
            "ylabel": None,
            "xlim": None,
            "ylim": None,
            "grid": False,
            "grid_linestyle": ":",
            "xticks": None,
            "yticks": None,
            "xticklabels": None,
            "yticklabels": None,
            "xticklabel_rotation": 0,
            "ytick_rotation": 0,
            "xtick_fontsize": 5,
            "ytick_fontsize": 5,
            "show_legend": False,
            "legend_loc": "best",
            "linecolor": "black",
            "overlay_linecolor": "red",
            "overlay_linecolormap": plt.colormaps["jet"],
            "linewidth": 0.5,
            "overlay_linewidth": 1,
            "aspect": None,
            "imshow_origin": "upper",
            "title_fontsize": 9,
            "suptitle_fontsize": 8,
            "xlabel_fontsize": 6,
            "ylabel_fontsize": 6,
            "legend_fontsize": 6,
            "cbar_label_fontsize": 6,
        }

        if type(default_params["overlay_linecolormap"]) is str:
            plt.colormaps["jet"]

        # Merge default parameters with user-provided parameters
        # User-provided parameters will override defaults
        processed_params = {**default_params, **plot_params}

        return processed_params

    def _initialize_plot_session(self, input_data, owner_obj, required_dims=None):
        """
        Validates input data and identifies plotting dimensions.

        This method serves as a common entry point for the `plot` method in
        subclasses. It handles the initial validation of the input `xr.DataArray`
        or `xr.Dataset`, identifies spatial and other dimensions, and determines
        the plot layout (e.g., 2D, 3D).

        Parameters
        ----------
        input_data : xr.DataArray or xr.Dataset or None
            The data to be plotted.
        owner_obj : object
            The parent object instance (e.g., Anatomy or Spectroscopy object)
            that holds default data arrays.
        required_dims : list, optional
            A list of dimension names that are required to be present in the
            final data_array.

        Returns
        -------
        dict
            A dictionary containing the validated `data_array` and identified
            dimensions such as `all_dims`, `spatial_dims`, `other_dims`,
            `slicing_dim`, `n_spatial_dims`, `is_3d_spatial`, and `candidates`.

        """
        all_dims = []
        spatial_dims = None
        other_dims = None
        slicing_dim = None
        n_spatial_dims = None
        is_3d_spatial = None
        sampling_time_axis_s = None

        # check if data is temporal (FID) or spectral
        is_temporal = False

        # check if data is spatial
        is_spatial = False

        # Initialize data_array to None
        data_array = None

        if input_data is not None:
            data_vars = input_data.data_vars

            if not data_vars:
                return {
                    "all_dims": all_dims,
                    "spatial_dims": spatial_dims,
                    "other_dims": other_dims,
                    "slicing_dim": slicing_dim,
                    "n_spatial_dims": n_spatial_dims,
                    "is_3d_spatial": is_3d_spatial,
                    "sampling_time_axis_s": sampling_time_axis_s,
                    "is_temporal": is_temporal,
                    "is_spatial": is_spatial,
                    "data_array": data_array,
                }

            # Set data_array from input_data
            data_array = input_data
            # --- 1. Data Validation and Selection ---

            if required_dims:
                for dim in required_dims:
                    if dim not in input_data.dims:
                        raise ValueError(
                            f"Required dimension '{dim}' not found in the data."
                        )

            # --- 2. Dimension Identification ---
            all_dims = list(input_data.dims)
            possible_spatial_dims = ["read_idx", "phase_idx", "slice_idx"]
            spatial_dims = [
                d
                for d in possible_spatial_dims
                if d in all_dims and input_data.sizes[d] > 1
            ]
            n_spatial_dims = len(spatial_dims)
            is_3d_spatial = n_spatial_dims == 3

            # Identify dimensions that are not spatial and need sliders
            other_dims = [d for d in all_dims if d not in spatial_dims]

            # Determine slicing dimension for 3D data
            slicing_dim = spatial_dims[-1] if is_3d_spatial else None

            sampling_time_axis_s = None
            try:
                sampling_time_axis_s = owner_obj.sampling_time_axis_s
            except BaseException:
                try:
                    sampling_time_axis_s = input_data.attrs.get(
                        "sampling_time_axis_s", None
                    )
                except BaseException:
                    pass

            if input_data.signal_domain == "temporal":
                is_temporal = True
            if input_data.domain == "spatial":
                is_spatial = True

        else:
            # input_data is None, try to get data from owner_obj
            if hasattr(owner_obj, "data_array") and owner_obj.data_array is not None:
                data_array = owner_obj.data_array
            elif hasattr(owner_obj, "reco_array") and owner_obj.reco_array is not None:
                data_array = owner_obj.reco_array
            else:
                raise AttributeError(
                    "Input data not provided and neither owner_obj.data_array nor owner_obj.reco_array found or are None. Cannot plot."
                )

            # Process the dimensions for the data we found
            if data_array is not None:
                # --- 1. Data Validation and Selection ---
                if required_dims:
                    for dim in required_dims:
                        if dim not in data_array.dims:
                            raise ValueError(
                                f"Required dimension '{dim}' not found in the data."
                            )

                # --- 2. Dimension Identification ---
                all_dims = list(data_array.dims)
                possible_spatial_dims = ["read_idx", "phase_idx", "slice_idx"]
                spatial_dims = [
                    d
                    for d in possible_spatial_dims
                    if d in all_dims and data_array.sizes[d] > 1
                ]
                n_spatial_dims = len(spatial_dims)
                is_3d_spatial = n_spatial_dims == 3

                # Identify dimensions that are not spatial and need sliders
                other_dims = [d for d in all_dims if d not in spatial_dims]

                # Determine slicing dimension for 3D data
                slicing_dim = spatial_dims[-1] if is_3d_spatial else None

                sampling_time_axis_s = None
                try:
                    sampling_time_axis_s = owner_obj.sampling_time_axis_s
                except BaseException:
                    try:
                        sampling_time_axis_s = data_array.attrs.get(
                            "sampling_time_axis_s", None
                        )
                    except BaseException:
                        pass

                if (
                    hasattr(data_array, "signal_domain")
                    and data_array.signal_domain == "temporal"
                ):
                    is_temporal = True
                if hasattr(data_array, "domain") and data_array.domain == "spatial":
                    is_spatial = True

        return {
            "all_dims": all_dims,
            "spatial_dims": spatial_dims,
            "other_dims": other_dims,
            "slicing_dim": slicing_dim,
            "n_spatial_dims": n_spatial_dims,
            "is_3d_spatial": is_3d_spatial,
            "sampling_time_axis_s": sampling_time_axis_s,
            "is_temporal": is_temporal,
            "is_spatial": is_spatial,
            "data_array": data_array,
        }

    def _standardize_input_to_dataset(self, data_container):
        """
        Converts input data into a standardized dictionary of DataArrays.

        - If input is a Dataset, returns a dict of its data variables.
        - If input is a DataArray, returns a dict with a single default key.
        - If input is None, returns None.

        Parameters
            data_container (xr.DataArray or xr.Dataset or None): The input data.

        Returns
            dict[str, xr.DataArray] or None: A dictionary of plottable DataArrays.
        """
        if data_container is None:
            # Return an empty Dataset
            return xr.Dataset()

        if isinstance(data_container, xr.Dataset):
            print("Input is a Dataset. Extracting all data variables.")
            # Return a dictionary comprehension of all data variables
            return data_container

        if isinstance(data_container, xr.DataArray):
            print("Input is a DataArray. Wrapping in a Dataset.")
            # Return a dict with a default key, e.g., 'signal'
            dataset = data_container.to_dataset(name="signal")
            dataset.attrs = data_container.attrs
            return dataset

        raise TypeError(
            f"Unsupported data type: {type(data_container)}. "
            "Please provide an xarray.DataArray or xarray.Dataset."
        )

    def _def_overlay_controls(self, **kwargs):
        """
        Defines widgets specific to the anatomy plot overlay.

        This implementation provides controls for the overlay's alpha, colormap,
        data source (if from a Dataset), and intensity range.

        Returns
        -------
        ipywidgets.VBox
            A VBox containing the anatomy-specific overlay control widgets.
        """
        overlay_data = kwargs.get("overlay_data")
        overlay_dataset_for_fit = kwargs.get("overlay_dataset_for_fit")
        continuous_update = kwargs.get("continuous_update")

        overlay_widget_list = []

        if overlay_data is not None:
            self.widgets["overlay_alpha"] = FloatSlider(
                value=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Overlay Alpha:",
                continuous_update=continuous_update,
                style={"description_width": "initial"},
            )
            overlay_widget_list.append(self.widgets["overlay_alpha"])

            if overlay_dataset_for_fit is not None:
                overlay_da_options = [
                    da_name
                    for da_name, da in overlay_dataset_for_fit.items()
                    if isinstance(da, xr.DataArray) and da.ndim > 1
                ]
                if overlay_da_options:
                    self.widgets["overlay_source_selector"] = Dropdown(
                        options=overlay_da_options,
                        value=(overlay_da_options[0]),
                        description="Overlay Param:",
                        style={"description_width": "initial"},
                    )
                    overlay_widget_list.append(self.widgets["overlay_source_selector"])

            valid_cmaps = [
                c for c in common_cmaps if c in plt.colormaps() or c in cmc.colormaps()
            ]
            self.widgets["overlay_cmap"] = Dropdown(
                options=valid_cmaps,
                value="magma",
                description="Overlay Cmap:",
                style={"description_width": "initial"},
            )
            overlay_widget_list.append(self.widgets["overlay_cmap"])

            self.widgets["overlay_range_mode"] = Dropdown(
                options=["Percentage", "Min/Max"],
                value="Percentage",
                description="Overlay Range:",
                style={"description_width": "initial"},
            )
            self.widgets["overlay_vmin"] = FloatSlider(
                value=0.0,
                min=0.0,
                max=100.0,
                step=0.1,
                description="Overlay Vmin [%]:",
                style={"description_width": "initial"},
            )
            self.widgets["overlay_vmax"] = FloatSlider(
                value=100.0,
                min=0.0,
                max=100.0,
                step=0.1,
                description="Overlay Vmax [%]:",
                style={"description_width": "initial"},
            )
            if "overlay_dataset_for_fit" in kwargs:
                overlay_metabolite_list = (
                    overlay_dataset_for_fit.metabolite.values.tolist()
                )
                self.widgets["overlay_selected_metabolites"] = Dropdown(
                    options=overlay_metabolite_list,
                    value=overlay_metabolite_list[0],
                    description="Overlay Mets:",
                    disabled=False,
                    rows=min(len(overlay_metabolite_list), 5),
                    style={"description_width": "initial"},
                )
            self.widgets["overlay_vmin"].style.handle_color = "lightblue"
            self.widgets["overlay_vmax"].style.handle_color = "lightgreen"
            overlay_widget_list.extend(
                [
                    self.widgets["overlay_selected_metabolites"],
                    self.widgets["overlay_range_mode"],
                    self.widgets["overlay_vmin"],
                    self.widgets["overlay_vmax"],
                ]
            )

        return VBox(overlay_widget_list)

    def _def_widgets(
        self,
        data_array,
        plot_params,
        slider_defaults,
        continuous_update,
        show_spatial_plot,
        slicing_dim,
        other_dims,
        channel_dim,
        is_overlay_present,
        **specific_kwargs,
    ):
        """
        Template method to create and layout all interactive widgets.

        This method defines the overall structure of the control panel. It
        creates common widgets (sliders, colormap, etc.) and then calls
        abstract methods to get subclass-specific widgets, which are then
        assembled into a final layout.

        Parameters
        ----------
        data_array : xr.DataArray
            The data array to be plotted, used to determine slider ranges.
        plot_params : dict
            A dictionary of plotting parameters.
        slider_defaults : dict
            Default values for the dimension sliders.
        continuous_update : bool
            Flag for continuous updates on sliders.
        show_spatial_plot : bool
            Flag indicating if a spatial plot is being shown.
        slicing_dim : str or None
            The dimension used for slicing in 3D plots.
        other_dims : list
            A list of non-spatial dimensions requiring sliders.
        channel_dim : str
            The name of the channel dimension.
        is_overlay_present : bool
            Flag indicating if overlay data is provided.
        **specific_kwargs : dict
            Additional keyword arguments to pass to the specific control
            definition methods.

        Returns
        -------
        tuple
            A tuple containing the final controls box (VBox) and the
            dictionary of all created widgets.
        """
        self.widgets = {}
        slider_list = []

        # --- 1. Common Widgets: Plane Label and Dimension Sliders ---
        self.widgets["plane_label"] = Label("Calculating initial view...")

        dims_with_sliders = []
        if slicing_dim:
            dims_with_sliders.append(slicing_dim)
        dims_with_sliders.extend(other_dims)

        unique_slider_dims = sorted(list(set(dims_with_sliders)))

        for dim_name in unique_slider_dims:
            if dim_name not in data_array.sizes:
                continue
            max_val = data_array.sizes[dim_name] - 1
            if max_val < 0:
                continue
            elif max_val == 0:
                self.widgets[dim_name] = fixed(0)
                continue

            default_val = slider_defaults.get(dim_name, max_val // 2)
            current_value = max(0, min(int(default_val), max_val))

            slider = IntSlider(
                min=0,
                max=max_val,
                step=1,
                value=current_value,
                description=f"{dim_name}:",
                continuous_update=continuous_update,
                layout=Layout(min_width="300px", width="auto"),
            )
            self.widgets[dim_name] = slider
            slider_list.append(slider)

        # --- 2. Common Widgets: Main Image Controls ---
        if show_spatial_plot:
            valid_cmaps = [
                c for c in common_cmaps if c in plt.colormaps() or c in cmc.colormaps()
            ]
            self.widgets["cmap"] = Dropdown(
                options=valid_cmaps,
                value=plot_params.get("colormap", "bone"),
                description="Colormap:",
            )
            self.widgets["range_mode"] = Dropdown(
                options=["Percentage", "Min/Max"],
                value="Percentage",
                description="Range Mode:",
                layout=Layout(width="auto"),
            )
            self.widgets["vmin"] = FloatSlider(
                value=plot_params.get("vmin_perc", 0.0),
                min=0,
                max=100,
                step=0.1,
                description="Vmin: [%]",
                continuous_update=False,
            )
            self.widgets["vmax"] = FloatSlider(
                value=plot_params.get("vmax_perc", 100.0),
                min=0,
                max=100,
                step=0.1,
                description="Vmax: [%]",
                continuous_update=False,
            )
            self.widgets["vmin"].style.handle_color = "lightblue"
            self.widgets["vmax"].style.handle_color = "lightgreen"
        else:
            self.widgets["cmap"] = fixed(plot_params.get("colormap", "bone"))
            self.widgets["range_mode"] = fixed("Percentage")
            self.widgets["vmin"] = fixed(plot_params.get("vmin_perc", 0.0))
            self.widgets["vmax"] = fixed(plot_params.get("vmax_perc", 100.0))

        is_multichannel = (
            channel_dim in data_array.dims and data_array.sizes[channel_dim] > 1
        )
        self.widgets["combine_channels"] = Checkbox(
            value=plot_params.get("combine_channels", False),
            description="Combine Channels (RSS)",
            indent=False,
            disabled=not is_multichannel,
        )
        self.widgets["interpolate"] = Checkbox(
            value=False, description="Interpolate Image", indent=False
        )
        self.widgets["denoise_components"] = IntSlider(
            value=0, min=0, max=100, description="DN Comps:"
        )
        self.widgets["denoise_components"].style.handle_color = "grey"

        # --- 3. Hook Methods: Get Subclass-Specific Controls ---
        # FIX: Explicitly pass necessary arguments to the hook methods.
        # The **specific_kwargs dictionary only contains arguments not explicitly
        # defined in _def_widgets signature. We need to add the ones that are.
        all_specific_kwargs = specific_kwargs.copy()
        all_specific_kwargs["data_array"] = data_array
        all_specific_kwargs["plot_params"] = plot_params
        all_specific_kwargs["continuous_update"] = continuous_update
        all_specific_kwargs["overlay_dataset_for_fit"] = specific_kwargs.get(
            "overlay_dataset_for_fit", None
        )

        specific_main_controls = self._def_specific_main_controls(**all_specific_kwargs)
        overlay_controls = self._def_overlay_controls(**all_specific_kwargs)

        # --- 4. Layout Assembly ---
        layout_rows = []
        if show_spatial_plot:
            layout_rows.append(HBox([self.widgets["plane_label"]]))

        # Sliders layout
        sliders_per_row = 3
        for i in range(0, len(slider_list), sliders_per_row):
            layout_rows.append(HBox(slider_list[i : i + sliders_per_row]))

        # Main controls box
        common_image_controls_list = [
            self.widgets.get(w)
            for w in ["cmap", "range_mode", "vmin", "vmax", "interpolate"]
        ]
        common_image_controls = [
            w for w in common_image_controls_list if isinstance(w, Widget)
        ]
        common_image_controls.append(self.widgets["denoise_components"])
        if is_multichannel:
            common_image_controls.append(self.widgets["combine_channels"])

        main_title = HTML(value="<b>Main Controls</b>")
        main_controls_box = VBox(
            [main_title, HBox([VBox(common_image_controls), specific_main_controls])],
            layout=Layout(border="solid 1px grey", padding="5px", margin="5px"),
        )

        # Overlay controls box
        overlay_box = VBox([])
        if is_overlay_present:
            overlay_title = HTML(value="<b>Overlay Controls</b>")
            overlay_box = VBox(
                [overlay_title, overlay_controls],
                layout=Layout(border="solid 1px grey", padding="5px", margin="5px"),
            )

        layout_rows.append(HBox([main_controls_box, overlay_box]))
        controls_box = VBox(layout_rows)

        return controls_box, self.widgets

    def _def_widgets_for_dataset(
        self,
        input_datasets=[],
        plot_params=None,
        slider_defaults=None,
        continuous_update=None,
        show_spatial_plot=None,
        slicing_dim=None,
        other_dims=None,
        channel_dim="",
        widget_prefix="",
        **specific_kwargs,
    ):
        """
        Creates and layouts interactive widgets for a list of xarray.Dataset objects.

        This function generates a unified set of sliders for all common dimensions
        and creates a separate control box for each data variable within the datasets,
        allowing for independent control of their visual properties.

        Parameters
        ----------
        input_datasets : List[xr.Dataset]
            A list of xarray Datasets to be plotted.
        plot_params : dict
            A dictionary of plotting parameters.
        slider_defaults : dict
            Default values for the dimension sliders.
        continuous_update : bool
            Flag for continuous updates on sliders.
        show_spatial_plot : bool
            Flag indicating if a spatial plot is being shown.
        slicing_dim : str or None
            The dimension used for slicing in 3D plots.
        other_dims : list
            A list of non-spatial dimensions requiring sliders.
        channel_dim : str
            The name of the channel dimension.
        **specific_kwargs : dict
            Additional keyword arguments for subclass-specific controls.

        Returns
        -------
        tuple
            A tuple containing the final controls box (VBox) and the
            dictionary of all created widgets.
        """
        self.widgets = {}
        all_dims_with_sliders = set(other_dims)
        if slicing_dim:
            all_dims_with_sliders.add(slicing_dim)

        # --- 1. Find all unique dimensions that need a slider ---
        max_sizes = {}
        all_data_vars = {}
        for i, ds in enumerate(input_datasets):
            for var_name, data_array in ds.data_vars.items():
                unique_var_name = f"ds{i}_{var_name}"
                all_data_vars[unique_var_name] = data_array
                for dim_name in all_dims_with_sliders:
                    if dim_name in data_array.sizes:
                        current_max = max_sizes.get(dim_name, 0)
                        max_sizes[dim_name] = max(
                            current_max, data_array.sizes[dim_name]
                        )

        # --- 2. Create Global Dimension Sliders ---
        slider_list = []
        self.widgets["plane_label"] = Label("Calculating initial view...")

        unique_slider_dims = sorted(list(max_sizes.keys()))

        for dim_name in unique_slider_dims:
            max_val = max_sizes[dim_name] - 1
            if max_val < 0:
                continue
            elif max_val == 0:
                self.widgets[dim_name] = fixed(0)
                continue

            default_val = slider_defaults.get(dim_name, max_val // 2)
            current_value = max(0, min(int(default_val), max_val))

            slider = IntSlider(
                min=0,
                max=max_val,
                step=1,
                value=current_value,
                description=f"{dim_name}:",
                continuous_update=continuous_update,
                layout=Layout(min_width="300px", width="auto"),
            )
            self.widgets[dim_name] = slider
            slider_list.append(slider)

        # --- 3. Create Individual Controls for Each DataArray ---
        data_array_controls = []
        for unique_var_name, data_array in all_data_vars.items():
            var_widgets = {}

            if show_spatial_plot:
                valid_cmaps = [
                    c
                    for c in common_cmaps
                    if c in plt.colormaps() or c in cmc.colormaps()
                ]
                var_widgets["cmap"] = Dropdown(
                    options=valid_cmaps,
                    value=plot_params.get("colormap", "bone"),
                    description="Colormap:",
                )
                var_widgets["range_mode"] = Dropdown(
                    options=["Percentage", "Min/Max"],
                    value="Percentage",
                    description="Range Mode:",
                    layout=Layout(width="auto"),
                )
                var_widgets["vmin"] = FloatSlider(
                    value=plot_params.get("vmin_perc", 0.0),
                    min=0,
                    max=100,
                    step=0.1,
                    description="Vmin: [%]",
                    continuous_update=False,
                )
                var_widgets["vmax"] = FloatSlider(
                    value=plot_params.get("vmax_perc", 100.0),
                    min=0,
                    max=100,
                    step=0.1,
                    description="Vmax: [%]",
                    continuous_update=False,
                )
                var_widgets["vmin"].style.handle_color = "lightblue"
                var_widgets["vmax"].style.handle_color = "lightgreen"
            else:
                var_widgets["cmap"] = fixed(plot_params.get("colormap", "bone"))
                var_widgets["range_mode"] = fixed("Percentage")
                var_widgets["vmin"] = fixed(plot_params.get("vmin_perc", 0.0))
                var_widgets["vmax"] = fixed(plot_params.get("vmax_perc", 100.0))

            is_multichannel = (
                channel_dim in data_array.dims and data_array.sizes[channel_dim] > 1
            )
            var_widgets["combine_channels"] = Checkbox(
                value=plot_params.get("combine_channels", False),
                description="Combine Channels (RSS)",
                indent=False,
                disabled=not is_multichannel,
            )

            # --- Hook for subclass-specific controls ---
            all_specific_kwargs = specific_kwargs.copy()
            all_specific_kwargs["data_array"] = data_array
            specific_widgets_dict = self._def_specific_main_controls(
                **all_specific_kwargs
            )

            # Add all widgets to the main dictionary with a unique key
            for w_name, w_obj in {**var_widgets, **specific_widgets_dict}.items():
                self.widgets[f"{unique_var_name}_{w_name}"] = w_obj

            # Assemble the controls for this DataArray into a bordered box
            title = HTML(
                value=f"< b > Controls for: < code > {unique_var_name.split( '_', 1)[1]} < /code > < / b >"
            )

            control_list = list(var_widgets.values()) + list(
                specific_widgets_dict.values()
            )

            data_array_box = VBox(
                [title, *control_list],
                layout=Layout(
                    border="solid 1px grey",
                    padding="5px",
                    margin="5px",
                    min_width="350px",
                ),
            )
            data_array_controls.append(data_array_box)

        # --- 4. Layout Assembly ---
        layout_rows = []
        if show_spatial_plot:
            layout_rows.append(HBox([self.widgets["plane_label"]]))

        # Global sliders layout
        sliders_title = HTML(value="<b>Global Dimension Controls</b>")

        # Add coordinate system toggle (only for spatial plots)
        slider_box_contents = [sliders_title]
        if show_spatial_plot:
            coord_system_toggle = ToggleButtons(
                options=["Auto", "XYZ", "RPS"],
                value="Auto",
                description="Coord System:",
                button_style="",
                tooltips=[
                    "Automatic coordinate detection",
                    "Scanner coordinates (X, Y, Z)",
                    "Acquisition coordinates (Read, Phase, Slice)",
                ],
            )
            self.widgets["coord_system"] = coord_system_toggle
            slider_box_contents.append(coord_system_toggle)
        else:
            # Add fixed widget for non-spatial plots
            self.widgets["coord_system"] = fixed("Auto")

        sliders_box = VBox(
            slider_box_contents + slider_list,
            layout=Layout(border="solid 1px grey", padding="5px", margin="5px"),
        )
        layout_rows.append(sliders_box)

        # Individual DataArray controls layout
        controls_per_row = 2
        for i in range(0, len(data_array_controls), controls_per_row):
            layout_rows.append(HBox(data_array_controls[i : i + controls_per_row]))

        final_controls_box = VBox(layout_rows)

        return final_controls_box, self.widgets

    def _vmin_vmax_range_check(
        self,
        range_mode,
        data_to_plot_spatial,
        vmin_slider,
        vmax_slider,
        min_sig,
        max_sig,
    ):
        """Helper to calculate vmin/vmax based on range mode and data.
        This function is intended to be used internally by plotters.
        """
        data_min_abs = np.nanmin(data_to_plot_spatial)
        data_max_abs = np.nanmax(data_to_plot_spatial)

        # Check if we need to reconfigure sliders due to mode switch
        if range_mode == "Min/Max" and "Abs" not in vmin_slider.description:
            # Update descriptions
            vmin_slider.description = "Vmin (Abs)"
            vmax_slider.description = "Vmax (Abs)"
            step = (max_sig - min_sig) / 200.0
            if step <= 0:
                step = 0.01

            # Only update ranges, not values, to avoid jumping
            if vmin_slider.min != min_sig or vmin_slider.max != max_sig:
                # Temporarily disable observe to prevent triggering during
                # setup
                with vmax_slider.hold_trait_notifications():
                    vmax_slider.min = min_sig
                    vmax_slider.max = max_sig
                    vmax_slider.step = step
                    vmax_slider.value = desired_vmax

                with vmin_slider.hold_trait_notifications():
                    vmin_slider.min = min_sig
                    vmin_slider.max = max_sig
                    vmin_slider.step = step
                    vmin_slider.value = desired_vmin

            vmin_val = vmin_slider.value
            vmax_val = vmax_slider.value
        elif range_mode == "Percentage" and "%" not in vmin_slider.description:
            # Switching TO Percentage mode: reconfigure sliders
            vmin_slider.description = "Vmin: [%]"
            vmax_slider.description = "Vmax: [%]"
            vmin_slider.min, vmax_slider.min = 0.0, 0.0
            vmin_slider.max, vmax_slider.max = 100.0, 100.0
            vmin_slider.step, vmax_slider.step = 0.1, 0.1
            vmin_slider.value = 0.0  # Reset to 0%
            vmax_slider.value = 100.0  # Reset to 100%

            vmin_val = vmin_slider.value
            vmax_val = vmax_slider.value

        # Calculate final vmin_val and vmax_val for imshow
        if range_mode == "Min/Max":
            vmin_val = vmin_slider.value
            vmax_val = vmax_slider.value
        else:  # Percentage mode
            vmax_for_perc = data_max_abs if data_max_abs > 0 else 1.0
            vmin_val = data_min_abs + vmin_slider.value / 100.0 * vmax_for_perc
            vmax_val = (vmax_slider.value / 100.0) * vmax_for_perc

        return vmin_val, vmax_val

    def _get_plot_image(
        self,
        data_array: xr.DataArray,
        is_spatial: bool,
        show_spatial_view: bool,
        space_dims: tuple = ("read_idx", "phase_idx", "slice_idx"),
    ) -> xr.DataArray:
        """
        Transforms data between image space and k-space.

        - If the view matches the data state (e.g., viewing k-space when data is k-space),
        it does nothing.
        - If they differ, it performs the correct FFT or IFFT.
        """

        # A transform is only needed if the desired view is different from the current data state.
        # is_spatial=True means k-space, show_spatial_view=True means image space.
        # is_spatial=False means image space, show_spatial_view=False means
        # k-space.
        if is_spatial == show_spatial_view:
            # Data is already in the desired state (e.g., k-space -> k-space or image -> image)
            # No transformation needed, just return the original array.
            return data_array

        # Don't do inv FT along slicedirection if data is multislice
        if data_array.n_slices > 1:
            space_dims = ("read_idx", "phase_idx")

        # --- A transform is required ---

        if not all(d in data_array.dims for d in space_dims):
            raise ValueError(
                f"Not all spatial dimensions {space_dims} found in DataArray."
            )

        # Get the integer axes corresponding to the spatial dimensions
        axis_nums = tuple(data_array.get_axis_num(d) for d in space_dims)

        # Perform the transform on the NumPy data, then wrap it back into an
        # xarray.DataArray using .copy() to preserve all coordinates and
        # attributes.
        if show_spatial_view:
            # --- k-space to image (IFFT) ---
            # The standard reconstruction: fftshift(ifftn(ifftshift(k-space)))
            kspace_data = np.fft.ifftshift(data_array.data, axes=axis_nums)
            image_data = np.fft.ifftn(kspace_data, axes=axis_nums)
            image_data_shifted = np.fft.fftshift(image_data, axes=axis_nums)
            return data_array.copy(data=image_data_shifted)
        else:
            # --- image to k-space (FFT) ---
            # The inverse of the reconstruction:
            # fftshift(fftn(ifftshift(image)))
            image_data = np.fft.ifftshift(data_array.data, axes=axis_nums)
            kspace_data = np.fft.fftn(image_data, axes=axis_nums)
            kspace_data_shifted = np.fft.fftshift(kspace_data, axes=axis_nums)
            return data_array.copy(data=kspace_data_shifted)

    def _calculate_extent(
        self,
        data_array,
        plot_dim_y,
        plot_dim_x,
        data_to_plot,
        selector,
        state,
        anatomy_obj=None,  # Added for anatomyPlotter specific extent calculation
    ):
        """
        Helper to calculate extent for imshow based on data coordinates.
        """
        extent, origin, aspect_ratio, xlabel, ylabel = (
            None,
            "upper",
            "equal",
            f"{plot_dim_x} [idx]",
            f"{plot_dim_y} [idx]",
        )
        use_coord_extent = False
        try:
            full_coords_x = data_array.coords[plot_dim_x]
            full_coords_y = data_array.coords[plot_dim_y]
            ny_plot, nx_plot = data_to_plot.shape

            if full_coords_x.ndim == 1 and full_coords_y.ndim == 1:  # 1D Coords
                x_vals, y_vals = full_coords_x.values, full_coords_y.values
                dx = abs(x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 1
                dy = abs(y_vals[1] - y_vals[0]) if len(y_vals) > 1 else 1
                extent = [
                    x_vals.min() - dx / 2.0,
                    x_vals.max() + dx / 2.0,
                    y_vals.min() - dy / 2.0,
                    y_vals.max() + dy / 2.0,
                ]
                origin = "lower"
                aspect_ratio = None
                xlabel = f"{plot_dim_x}[{full_coords_x.attrs.get( 'unit', 'val')}]"
                ylabel = f"{plot_dim_y}[{full_coords_y.attrs.get( 'unit', 'val')}]"
                use_coord_extent = True
            else:  # Multi-Dim Coords
                coord_selector = {}
                # Get current slicing dim from state
                slicing_dim_state = state[0]
                if slicing_dim_state and slicing_dim_state in selector:
                    if (
                        slicing_dim_state in full_coords_x.dims
                        and slicing_dim_state in full_coords_y.dims
                    ):
                        coord_selector[slicing_dim_state] = selector[slicing_dim_state]
                slice_coords_x = full_coords_x.isel(**coord_selector)
                slice_coords_y = full_coords_y.isel(**coord_selector)
                x_min, x_max = np.nanmin(slice_coords_x.values), np.nanmax(
                    slice_coords_x.values
                )
                # Filter selector to only include dims present in the
                # coordinates
                valid_coord_dims_x = [d for d in selector if d in full_coords_x.dims]
                valid_coord_dims_y = [d for d in selector if d in full_coords_y.dims]
                coord_selector_x = {
                    d: selector[d]
                    for d in valid_coord_dims_x
                    if d not in [plot_dim_x, plot_dim_y]
                }
                coord_selector_y = {
                    d: selector[d]
                    for d in valid_coord_dims_y
                    if d not in [plot_dim_x, plot_dim_y]
                }

                # Add slicing dim if it exists in coords and selector
                slicing_dim_state = state[0]
                if slicing_dim_state and slicing_dim_state in selector:
                    if slicing_dim_state in full_coords_x.dims:
                        coord_selector_x[slicing_dim_state] = selector[
                            slicing_dim_state
                        ]
                    if slicing_dim_state in full_coords_y.dims:
                        coord_selector_y[slicing_dim_state] = selector[
                            slicing_dim_state
                        ]

                slice_coords_x = full_coords_x.isel(**coord_selector_x)
                slice_coords_y = full_coords_y.isel(**coord_selector_y)

                # Check if sliced coords are still multi-dimensional before min/max
                # If they become 1D or scalar after selection, handle appropriately
                # For simplicity, we proceed assuming they might still be
                # multi-dim matching the plot data
                x_min, x_max = np.nanmin(slice_coords_x.values), np.nanmax(
                    slice_coords_x.values
                )
                y_min, y_max = np.nanmin(slice_coords_y.values), np.nanmax(
                    slice_coords_y.values
                )

                if np.all(np.isfinite([x_min, x_max, y_min, y_max])):
                    # Use actual shape of the *data* being plotted for dx/dy
                    # calculation
                    ny_plot_data, nx_plot_data = data_to_plot.shape
                    dx = (x_max - x_min) / (nx_plot_data - 1) if nx_plot_data > 1 else 0
                    dy = (y_max - y_min) / (ny_plot_data - 1) if ny_plot_data > 1 else 0
                    extent = [
                        x_min - dx / 2.0,
                        x_max + dx / 2.0,
                        y_min - dy / 2.0,
                        y_max + dy / 2.0,
                    ]
                    origin = "lower"
                    aspect_ratio = None
                    xlabel = f"{plot_dim_x}[{full_coords_x.attrs.get( 'unit', 'val')}]"
                    ylabel = f"{plot_dim_y}[{full_coords_y.attrs.get( 'unit', 'val')}]"
                    use_coord_extent = True
                else:
                    # Fallback for cases where coordinate extent cannot be
                    # reliably determined
                    use_coord_extent = False
        except Exception as e_extent:
            warnings.warn(
                f"Could not calculate extent from coordinates: {e_extent}. Falling back to index extent."
            )
            use_coord_extent = False

        if not use_coord_extent:
            ny_plot, nx_plot = data_to_plot.shape
            extent = [-0.5, nx_plot - 0.5, ny_plot - 0.5, -0.5]
            origin, aspect_ratio = "upper", None
            xlabel = f"{plot_dim_x} [index]"
            ylabel = f"{plot_dim_y} [index]"

        return extent, origin, aspect_ratio, xlabel, ylabel

    def _apply_spatial_transform(
        self,
        data_array: xr.DataArray,
        is_spatial: bool,
        show_spatial_view: bool,
        space_dims: tuple = ("read_idx", "phase_idx"),
    ):
        """
        Applies a spatial transform to the data_array based on its type and desired view.

        Parameters
            data_array: The input xarray.DataArray.
            is_spatial: True if the input is k-space data.
            show_spatial_view: True if the desired output is the image domain.

        Returns
            A new xarray.DataArray with the spatial transform applied.
        """
        if is_spatial and show_spatial_view:
            return data_array
        else:
            fft_axes = tuple(range(-len(space_dims), 0))
        if is_spatial and not show_spatial_view:

            def func(x):
                return np.fft.ifftshift(
                    np.fft.ifftn(np.fft.ifftshift(x, axes=fft_axes), axes=fft_axes),
                    axes=fft_axes,
                )

        elif not is_spatial and show_spatial_view:

            def func(x):
                return np.fft.fftshift(
                    np.fft.fftn(np.fft.fftshift(x, axes=fft_axes), axes=fft_axes),
                    axes=fft_axes,
                )

        processed_data = data_array.copy()
        processed_data = xr.apply_ufunc(
            func,
            processed_data,
            input_core_dims=[list(space_dims)],
            output_core_dims=[list(space_dims)],
            exclude_dims=set(space_dims),
            dask="parallelized",
        )
        return processed_data

    def _create_observer(
        self, slicing_dim_for_observer, data_obj=None, update_plot=None, state=None
    ):
        def observer(change):
            if change["new"] != change["old"]:
                # --- START: PROPOSED FIX ---
                # This is the moment the view changes. Reset the click
                # coordinates.
                state[0] = slicing_dim_for_observer
                state[1] = None  # Reset the stored click coordinates to None.
                # --- END: PROPOSED FIX ---

                if hasattr(update_plot, "click_marker") and update_plot.click_marker:
                    update_plot.click_marker.set_visible(False)
                if hasattr(update_plot, "ax_rep") and update_plot.ax_rep:
                    update_plot.ax_rep.cla()
                    update_plot.ax_rep.set_title("Click image for voxel data")
                    update_plot.ax_rep.grid(True, linestyle=":")
                    if (
                        hasattr(update_plot, "secondary_marker")
                        and update_plot.secondary_marker
                    ):
                        update_plot.secondary_marker.set_visible(False)
                    try:
                        update_plot.fig.canvas.draw_idle()
                    except BaseException:
                        pass
                if (
                    hasattr(data_obj, "clear_lasso_patch_ref")
                    and data_obj.clear_lasso_patch_ref
                ):
                    data_obj.clear_lasso_patch_ref(draw_update=True)

        return observer

    @abstractmethod
    def plot(self, *args, **kwargs):
        """Abstract plot method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement the plot method.")
