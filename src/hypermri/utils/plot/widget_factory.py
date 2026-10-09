"""Widget factory module for unified plotting system.

This module consolidates widget creation logic that was previously
scattered across multiple plotter classes.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from ipywidgets import (
    HTML,
    Checkbox,
    Dropdown,
    FloatSlider,
    FloatText,
    HBox,
    IntSlider,
    Layout,
    ToggleButton,
    VBox,
)

from hypermri.utils.plot.utils_plot import common_cmaps


class WidgetFactory:
    """Factory class for creating plot control widgets.

    This class provides methods to create various types of widgets
    for controlling plot appearance and data processing.
    """

    def __init__(self, processed_plot_params: Optional[Dict] = None):
        """Initialize the WidgetFactory.

        Parameters
        ----------
        processed_plot_params : Optional[Dict], optional
            Processed plot parameters for widget defaults
        """
        self.processed_plot_params = processed_plot_params or {}

    def create_dimension_slider(
        self,
        dim_name: str,
        max_val: int,
        default_val: Optional[int] = None,
        continuous_update: bool = True,
    ) -> IntSlider:
        """Create a slider for a data dimension.

        Parameters
        ----------
        dim_name : str
            Name of the dimension (e.g., 'slice_idx')
        max_val : int
            Maximum value for the slider
        default_val : Optional[int], optional
            Default value, defaults to max_val // 2
        continuous_update : bool, optional
            Whether to update continuously, by default True

        Returns
        -------
        IntSlider
            Created slider widget
        """
        if default_val is None:
            default_val = max_val // 2

        slider = IntSlider(
            min=0,
            max=max_val,
            step=1,
            value=max(0, min(int(default_val), max_val)),
            description=f"{dim_name}:",
            continuous_update=continuous_update,
            layout=Layout(max_width="300px", width="auto"),
        )

        return slider

    def create_colormap_dropdown(
        self, default_cmap: str = "bone", is_overlay: bool = False
    ) -> Dropdown:
        """Create a colormap selection dropdown.

        Parameters
        ----------
        default_cmap : str, optional
            Default colormap, by default "bone"
        is_overlay : bool, optional
            Whether this is for overlay data, by default False

        Returns
        -------
        Dropdown
            Created colormap dropdown
        """
        valid_cmaps = [c for c in common_cmaps if c in plt.colormaps()]

        description = "Overlay Colormap:" if is_overlay else "Colormap:"

        return Dropdown(
            options=valid_cmaps,
            value=default_cmap if default_cmap in valid_cmaps else valid_cmaps[0],
            description=description,
        )

    def create_range_mode_dropdown(self) -> Dropdown:
        """Create a range mode dropdown (Percentage vs Min/Max).

        Returns
        -------
        Dropdown
            Created range mode dropdown
        """
        return Dropdown(
            options=["Percentage", "Min/Max"],
            value="Percentage",
            description="Range Mode:",
            layout=Layout(width="auto"),
        )

    def create_vmin_vmax_sliders(
        self, vmin_perc: float = 0.0, vmax_perc: float = 100.0
    ) -> Tuple[FloatSlider, FloatSlider]:
        """Create vmin and vmax sliders for intensity range control.

        Parameters
        ----------
        vmin_perc : float, optional
            Default vmin percentage, by default 0.0
        vmax_perc : float, optional
            Default vmax percentage, by default 100.0

        Returns
        -------
        Tuple[FloatSlider, FloatSlider]
            Tuple of (vmin_slider, vmax_slider)
        """
        vmin_slider = FloatSlider(
            value=vmin_perc,
            min=0,
            max=100,
            step=0.1,
            description="Vmin: [%]",
            continuous_update=False,
        )
        vmin_slider.style.handle_color = "lightblue"

        vmax_slider = FloatSlider(
            value=vmax_perc,
            min=0,
            max=100,
            step=0.1,
            description="Vmax: [%]",
            continuous_update=False,
        )
        vmax_slider.style.handle_color = "lightgreen"

        return vmin_slider, vmax_slider

    def create_denoise_slider(
        self, default_value: int = 0, max_components: int = 20
    ) -> IntSlider:
        """Create a denoising components slider.

        Parameters
        ----------
        default_value : int, optional
            Default number of components, by default 0
        max_components : int, optional
            Maximum number of components, by default 20

        Returns
        -------
        IntSlider
            Created denoising slider
        """
        return IntSlider(
            value=default_value,
            min=0,
            max=max_components,
            description="Denoise Comp:",
        )

    def create_linebroadening_text(self, default_value: float = 0.0) -> FloatText:
        """Create a line broadening text input.

        Parameters
        ----------
        default_value : float, optional
            Default line broadening in Hz, by default 0.0

        Returns
        -------
        FloatText
            Created text input widget
        """
        return FloatText(
            value=default_value,
            description="LB (Hz):",
        )

    def create_npoints_slider(
        self, max_points: int, default_value: Optional[int] = None
    ) -> IntSlider:
        """Create an npoints slider for spectroscopy data.

        Parameters
        ----------
        max_points : int
            Maximum number of acquisition points
        default_value : Optional[int], optional
            Default value, defaults to max_points

        Returns
        -------
        IntSlider
            Created npoints slider
        """
        if default_value is None:
            default_value = max_points

        return IntSlider(
            value=default_value,
            max=max_points,
            min=1,
            description="NPoints:",
            tooltip=f"Number of points to display (max: {max_points})",
        )

    def create_spec_unit_toggle(self) -> ToggleButton:
        """Create a spectroscopy unit toggle (ppm/Hz).

        Returns
        -------
        ToggleButton
            Created toggle button
        """
        return ToggleButton(
            value=True,
            description="ppm/Hz",
            tooltip="Toggle between ppm (default) and Hz frequency units",
            layout=Layout(width="70px", height="25px"),
        )

    def create_view_toggle(self) -> ToggleButton:
        """Create a view toggle button (Spectrum/FID).

        Returns
        -------
        ToggleButton
            Created toggle button
        """
        return ToggleButton(
            value=False,
            description="Show FID",
            tooltip="Toggle between spectrum (default) and FID view",
            layout=Layout(width="70px", height="25px"),
        )

    def create_data_variable_dropdown(
        self, variable_names: List[str], default_var: Optional[str] = None
    ) -> Dropdown:
        """Create a data variable selection dropdown for multi-variable datasets.

        Parameters
        ----------
        variable_names : List[str]
            List of available variable names
        default_var : Optional[str], optional
            Default variable to select

        Returns
        -------
        Dropdown
            Created dropdown widget
        """
        if not variable_names:
            return None

        if default_var is None or default_var not in variable_names:
            default_var = variable_names[0]

        return Dropdown(
            options=variable_names,
            value=default_var,
            description="Variable:",
        )

    def create_metabolite_dropdown(
        self, metabolite_names: List[str]
    ) -> Optional[Dropdown]:
        """Create a metabolite selection dropdown for fitted data.

        Parameters
        ----------
        metabolite_names : List[str]
            List of metabolite names

        Returns
        -------
        Optional[Dropdown]
            Created dropdown widget, or None if no metabolites
        """
        if not metabolite_names:
            return None

        return Dropdown(
            options=metabolite_names,
            value=metabolite_names[0],
            description="Metabolite:",
        )

    def create_secondary_plot_dim_dropdown(
        self, dim_options: List[str]
    ) -> Optional[Dropdown]:
        """Create a dropdown for selecting secondary plot dimension.

        Parameters
        ----------
        dim_options : List[str]
            List of available dimension names

        Returns
        -------
        Optional[Dropdown]
            Created dropdown widget, or None if no options
        """
        if not dim_options:
            return None

        return Dropdown(
            options=dim_options,
            value=dim_options[0],
            description="2nd Plot Dim:",
            disabled=not dim_options,
        )

    def create_average_repetitions_checkbox(
        self, has_repetitions: bool = True
    ) -> Checkbox:
        """Create a checkbox for averaging repetitions.

        Parameters
        ----------
        has_repetitions : bool, optional
            Whether data has repetitions, by default True

        Returns
        -------
        Checkbox
            Created checkbox widget
        """
        return Checkbox(
            value=False,
            description="Average Reps",
            disabled=not has_repetitions,
        )

    def create_combine_channels_checkbox(self, has_channels: bool = True) -> Checkbox:
        """Create a checkbox for channel combination.

        Parameters
        ----------
        has_channels : bool, optional
            Whether data has multiple channels, by default True

        Returns
        -------
        Checkbox
            Created checkbox widget
        """
        return Checkbox(
            value=False,
            description="Combine Channels",
            disabled=not has_channels,
        )

    def create_spec_part_dropdown(self, default_part: str = "Magnitude") -> Dropdown:
        """Create a dropdown for data part selection.

        Parameters
        ----------
        default_part : str, optional
            Default data part, by default "Magnitude"

        Returns
        -------
        Dropdown
            Created dropdown widget
        """
        return Dropdown(
            options=["Magnitude", "Phase", "Real", "Imaginary"],
            value=default_part,
            description="Data Part:",
        )

    def create_overlay_alpha_slider(self, default_alpha: float = 0.5) -> FloatSlider:
        """Create a slider for overlay transparency.

        Parameters
        ----------
        default_alpha : float, optional
            Default alpha value, by default 0.5

        Returns
        -------
        FloatSlider
            Created alpha slider
        """
        slider = FloatSlider(
            value=default_alpha,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Overlay Alpha:",
            continuous_update=True,
        )
        slider.layout.max_width = "300px"
        return slider

    def create_show_base_checkbox(self) -> Checkbox:
        """Create a checkbox for showing/hiding base data.

        Returns
        -------
        Checkbox
            Created checkbox widget
        """
        return Checkbox(value=True, description="Show Base")

    def create_interpolate_image_checkbox(self) -> Checkbox:
        """Create a checkbox for image interpolation.

        Returns
        -------
        Checkbox
            Created checkbox widget
        """
        return Checkbox(
            value=False,
            description="Interpolate Image",
            tooltip="Use spline interpolation for smoother images",
        )

    def create_widget_group(
        self, title: str, widgets: List, border: bool = True
    ) -> VBox:
        """Create a grouped widget box with title.

        Parameters
        ----------
        title : str
            Title for the widget group
        widgets : List
            List of widgets to include in the group
        border : bool, optional
            Whether to show border, by default True

        Returns
        -------
        VBox
            Created widget group
        """
        title_widget = HTML(value=f"<b>{title}</b>")

        layout_kwargs = {"padding": "5px", "margin": "5px", "min_width": "350px"}
        if border:
            layout_kwargs["border"] = "solid 1px grey"

        return VBox([title_widget] + widgets, layout=Layout(**layout_kwargs))

    def create_control_row(self, widgets: List) -> HBox:
        """Create a horizontal row of widgets.

        Parameters
        ----------
        widgets : List
            List of widgets to arrange horizontally

        Returns
        -------
        HBox
            Created horizontal box
        """
        return HBox(widgets)

    def apply_font_size(self, widget, font_size: str = "10px") -> None:
        """Apply font size to a widget if supported.

        Parameters
        ----------
        widget : Widget
            Widget to modify
        font_size : str, optional
            Font size string, by default "10px"
        """
        try:
            widget.style.font_size = font_size
        except:
            pass
