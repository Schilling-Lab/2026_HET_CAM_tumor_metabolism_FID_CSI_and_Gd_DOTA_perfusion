from copy import deepcopy

import ipywidgets as widgets

# colorbar
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# cursor cross
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ...functions.MEpHISDO import create_test_signal
from ...utils.plot.plotterABC import PlotterABC
from ._plot_modes import *

DEFAULT_PARAMETER = {"interpolation": "none", "fmap": {"cmap": "plasma"}}


class UmpirePlotter3D(PlotterABC):
    """umpireMGE 3D plotter class

    What data is plotted depends on the mode selected. E.g. when the user selects
    the mode 'magnitude' magnitude images are shown. The data modes implemented
    so far are:

        - fmap: Generated frequency map
        - fmap_err: Error map from fitting (absolute error)
        - fmap_err_rel: Error map from fitting (relative error)
        - phMap: use self.calculate_pH() first, a map of pH instead of frequencies
        - magnitude: classical magnitude images
        - phase: classical phase images

    Each Mode can have its own colormap, title, colorbar label and color range.

    How to add a mode:
        1. create global mode object using the dataModeClass
        2. in __init__(): add mode using self.add_mode()
        3. in plot(): add where needed...
    """

    def __init__(self, uMGE, anatref=None, parameters=None):
        """Initialize plotter and directly call self.show()."""
        self.uMGE = uMGE
        self.anatref = anatref  # anatomical reference

        self.__init_modes(
            MODE_fmap,
            MODE_fmap_err,
            MODE_fmap_err_rel,
            MODE_pHmap,
            MODE_magnitude,
            MODE_phase,
            MODE_multi_comp_fmaps,
            MODE_multi_comp_pHmaps,
            MODE_multi_comp_mean_pH,
            MODE_multi_comp_peak_pH,
        )

        if False:  # hasattr(self.uMGE, 'T2_star'):
            self.__init_modes(MODE_T2_star)

        self.__init_parameters(parameters)
        self.__init_range_sliders()
        self.__init_sequence_params()

        self.setup_figure()
        self.setup_interactive()

        self.show()

    def __init_modes(self, *modes):
        """Store all modes given. See self.add_mode() for more info."""
        self.modes = dict()

        for mode in modes:
            self.add_mode(mode)

    def __init_parameters(self, parameters):
        """Plot (and mode) specific user parameters are processed here."""
        self.params = DEFAULT_PARAMETER

        if parameters is not None:
            self.params.update(parameters)

        # here we check wether there are any paramters dedicated for a specific
        # mode. In that case we overwrite the mode attribute with the given
        # values and delete the entry from the parameter dictionary.
        for key in list(self.params.keys()):
            if key in self.modes:
                self.edit_mode(key, self.params.pop(key))

    def __init_range_sliders(self):
        """Create a color range slider for every mode."""
        self.range_sliders = dict()

        for m in self.modes.values():
            self.add_range_slider(m)

    def __init_sequence_params(self):
        """Add handy PVM-parameters + add orientation-dependend labels and titles."""
        self.pvm_fov = self.uMGE.pvm_fov
        self.pvm_res = self.uMGE.pvm_res
        self.pvm_matrix = self.uMGE.pvm_matrix
        self.pvm_orientation = self.uMGE.pvm_orientation
        self.pvm_readorientation = self.uMGE.pvm_readorientation

        if self.uMGE.interpolated:
            self.pvm_matrix = self.uMGE.interp_matrix
            self.pvm_res = self.pvm_fov / self.uMGE.interp_matrix

        # those are somewhat indirect sequence parameter
        self.labels = "read", "phase", "slice"
        self.titles = self.uMGE.slice_orientations

    def add_title_and_label(self, ax, view):
        """Add title and axis labels to the main view and both projections"""
        c = {"read": "red", "phase": "green", "slice": "blue"}

        if view == "main":
            ax.set_title(self.titles[0])
            ax.set_xlabel(self.labels[1], color=c[self.labels[1]], fontweight="heavy")
            ax.set_ylabel(self.labels[0], color=c[self.labels[0]], fontweight="heavy")
        elif view == "proj1":
            ax.set_title(self.titles[1])
            ax.set_xlabel(self.labels[2], color=c[self.labels[2]], fontweight="heavy")
            ax.set_ylabel(self.labels[1], color=c[self.labels[1]], fontweight="heavy")
        elif view == "proj2":
            ax.set_title(self.titles[2])
            ax.set_xlabel(self.labels[2], color=c[self.labels[2]], fontweight="heavy")
            ax.set_ylabel(self.labels[0], color=c[self.labels[0]], fontweight="heavy")
        else:
            AssertionError(f"'{view}' is not a valid view.")

    def setup_figure(self):
        """Setup the figure and axes layout."""
        with plt.ioff():
            fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))

        self.fig = fig
        self.axes = list(axes.flatten())

        for i, ax in enumerate(self.axes):
            setattr(self, f"ax{i + 1}", ax)

        # give axes 2 a colorbar
        divider = make_axes_locatable(self.ax2)
        self.cax2 = divider.append_axes("right", size="5%", pad=0.1)
        self.caxes = [self.cax2]

        # give the anatomical images labels and titles
        # (labels to identify them, they don't show up in the plot)
        # (titles will show up in the plot)
        self.ax2.set_label("main")
        self.ax3.set_label("proj1")
        self.ax4.set_label("proj2")

    def add_mode(self, mode_obj):
        """Add a given mode to this class object and insert self.uMGE as source."""
        mode_obj.src = self.uMGE
        self.modes[mode_obj.name] = deepcopy(mode_obj)

    def set_mode(self, mode_name):
        """Make the given mode the currently active mode."""
        self.mode = self.modes[mode_name]

    def edit_mode(self, mode_name, params):
        """Overwrite attributes of a given mode with values from params."""
        assert mode_name in self.modes.keys(), f"'{mode_name}' is not a valid mode."
        for attr_name, attr_value in params.items():
            setattr(self.modes[mode_name], attr_name, attr_value)

    def clear(self):
        """Clear all ax objects."""
        for a in self.axes + self.caxes:
            a.clear()

    def show(self):
        """This method has to be called to make the plot appear."""
        self.fig.tight_layout()
        display(self.ui, self.sliders)
        plt.show(self.fig)

    def add_range_slider(self, mode, **kwargs):
        """Add a color range slider for a mode."""
        dmin, dmax = mode.min, mode.max

        if mode.default_range is not None:
            default_values = mode.default_range
        else:
            default_values = dmin, dmax

        slider_options = dict(
            value=default_values,
            min=dmin,
            max=dmax,
            step=(dmax - dmin) * 1e-3,
            description="Color-Range:",
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".0f",
            layout=widgets.Layout(width="650px"),
            style={"description_width": "initial"},
            **kwargs,
        )

        self.range_sliders[mode.name] = widgets.FloatRangeSlider(**slider_options)

    def get_extent(self, view):
        """Calculate the image extent for a given view.

              [:, :, x]         [x, :, :]         [:, x, :]
               b-----.           d-----.           b-----.
        main : |     |   proj1 : |     |   proj2 : |     |
               a,c---d           c,e---f           a,e---f

        Parameter
        ----------
        view : str {'main', 'proj1', 'proj2'}
            Returns extent of the image itself or one of its two projections.
        """
        read_ext, phase_ext, slice_ext = self.pvm_fov

        a = -read_ext / 2  # - 0.5
        b = read_ext / 2  # - 0.5
        c = -phase_ext / 2  # - 0.5
        d = phase_ext / 2  # - 0.5
        e = -slice_ext / 2  # - 0.5
        f = slice_ext / 2  # - 0.5

        # extent is:
        # left, right, bottom, top
        if view == "main":
            return (c, d, a, b)
        if view == "proj1":
            return (e, f, c, d)
        if view == "proj2":
            return (e, f, a, b)

    def pos2idx(self, pos, dim):
        """Converts position coordinates (FOV) into an array index (Matrix).

        Parameter
        ----------
        pos : float
            Position of an image point in FOV coordinates (usually millimeter)
            along ONE axis.

        dim : int
            Specifies the axis of the position. This is important because different
            axis can have different resolutions.
        """
        idx = (pos - self.pvm_res[dim] / 2 + self.pvm_fov[dim] / 2) / self.pvm_res[dim]
        return np.round(idx).astype("int")

    def idx2pos(self, idx, dim):
        """Converts array index (Matrix) into position coordinates (FOV).

        Parameter
        ----------
        idx : ing
            Position of an image point as a matrix indices of ONE axis.

        dim : int
            Specifies the axis of the position. This is important because different
            axis can have different resolutions/voxel sizes.
        """
        pos = idx * self.pvm_res[dim] - self.pvm_fov[dim] / 2 + self.pvm_res[dim] / 2
        return pos

    def draw_cursor(self, ax, x_tuple, y_tuple):
        """Draws a cursor cross onto the axes at given position.

        Parameter
        ----------
        ax : matplotlib axis object

        x_tuple : (x, dimx)
            Where x is the horizontal position and dimx is the image axis belonging
            to the position. This important because different axis can have different
            resolutions/voxel sizes.

        y_tuple : (y, dimy)
            See x_tuple.
        """
        # draw vertical and horizontal position lines
        x, dimx = x_tuple
        y, dimy = y_tuple

        x_off = self.pvm_res[dimx] / 2
        y_off = self.pvm_res[dimy] / 2
        ax.vlines(
            [x, x],
            ymin=[min(ax.get_ylim()), y + y_off],
            ymax=[y - y_off, max(ax.get_ylim())],
            color="lightgray",
            linewidth=1.4,
            linestyle="dashed",
        )
        ax.hlines(
            [y, y],
            xmin=[min(ax.get_xlim()), x + x_off],
            xmax=[x - x_off, max(ax.get_xlim())],
            color="lightgray",
            linewidth=1.4,
            linestyle="dashed",
        )
        ax.add_patch(
            Rectangle(
                (x - x_off, y - y_off),
                self.pvm_res[dimx],
                self.pvm_res[dimy],
                fc="none",
                ec="lightgray",
            )
        )

    def onclick(self, event):
        """Method that gets called whenever the user mouse-clicks onto the plot.

        In our case we change a slider value in case a mouse-click event is
        detected. The updating is then handled by ipywidets.

        event : matplotlib event
        """
        x = event.xdata  # this axis -> _
        y = event.ydata  # this axis -> |
        axlabel = event.inaxes.get_label()

        if axlabel == "main":
            self.read_slider.value = self.pos2idx(y, 0)
            self.phase_slider.value = self.pos2idx(x, 1)
        elif axlabel == "proj1":
            self.phase_slider.value = self.pos2idx(y, 1)
            self.slice_slider.value = self.pos2idx(x, 2)
        elif axlabel == "proj2":
            self.read_slider.value = self.pos2idx(y, 0)
            self.slice_slider.value = self.pos2idx(x, 2)

        # for debugging:
        # tx = f"{str(event.inaxes.get_label())}{event.xdata:.0f} {event.ydata:.0f}"
        # event.inaxes.set_title(tx)
        # event.canvas.draw()

    def draw_simulated_from_MEpHISDO_fit(self, ax, index):
        """Draw linear phase fit through echoes for ax1."""
        ax.clear()
        r_idx, p_idx, s_idx = index

        time_axis = np.linspace(0, (self.uMGE.TEs[-1] + 1) / 1000.0, 1000)

        measured_signal = self.uMGE._interp_magnitude[:, r_idx, p_idx, s_idx]

        fitted_freq = self.uMGE._multi_comp_fmaps[:, r_idx, p_idx, s_idx]
        fitted_freq = (
            fitted_freq[
                np.logical_not(
                    np.isnan(fitted_freq[: self.uMGE._max_num_comps_MEpHISDO])
                )
            ]
            - self.uMGE._f_offset
        )

        # add c1 contribution
        # fitted_freq =np.append(fitted_freq, -500)

        fitted_ampl = self.uMGE._multi_comp_amplitudes[:, r_idx, p_idx, s_idx]
        fitted_ampl = fitted_ampl[
            np.logical_not(np.isnan(fitted_ampl[: self.uMGE._max_num_comps_MEpHISDO]))
        ]

        # add c1 contribution
        # fitted_ampl = np.append(fitted_ampl, self.uMGE._c1_ampl_map[r_idx, p_idx, s_idx])

        t2_star = self.uMGE._T2_star_map[r_idx, p_idx, s_idx]

        signal_based_on_fit = create_test_signal(
            fitted_ampl, np.real(fitted_freq), time_axis, t2_star_decay=t2_star
        ).squeeze()

        TEs = self.uMGE.TEs
        ax.scatter(TEs, np.abs(measured_signal), color="black", marker="s")

        ax.plot(time_axis * 1000, np.abs(signal_based_on_fit), color="gray")

        # add some labelling and a title
        ax.set_xlim(0, TEs[-1] + 1)
        ax.set_xlabel("TE [ms]")
        ax.set_ylabel("Signal [a.u.]")
        ax.set_title(
            r"Per Pixel Signal Fit (Cost: "
            + f"{self.uMGE._fit_cost_map[r_idx,p_idx, s_idx]:.6f})"
        )

    def draw_phasefit(self, ax, index):
        """Draw linear phase fit through echoes for ax1."""
        ax.clear()
        r_idx, p_idx, s_idx = index

        TEs = self.uMGE.TEs
        y_phase = self.uMGE.unwrapped_phase[:, r_idx, p_idx, s_idx]

        ax.scatter(TEs, y_phase, color="black", marker="s")

        # additionally, use the stored fitting results to plot linear graph
        # indicating the found fit function
        m, m_err, t, t_err = self.uMGE._fit_results[r_idx, p_idx, s_idx]
        y_fit = TEs * m + t

        ax.plot(TEs, y_fit, color="gray", linestyle="dashed")

        # add some labelling and a title
        ax.set_xlabel("TE [ms]")
        ax.set_ylabel("Phase [rad]")
        # self.ax1.set_title("Single Voxel Phase Fit")
        ax.set_title(
            r"Per Pixel: $\quad\frac{d\phi}{dt}=$"
            + f"{m * 1e3 / (2 * np.pi):.1f} ± {m_err * 1e3 / (2 * np.pi):.1f} Hz"
        )

    def draw_magnitude(self, ax, index):
        """Draw linear phase fit through echoes for ax1."""
        ax.clear()

        r_idx, p_idx, s_idx = index

        TEs = self.uMGE.TEs[self.uMGE.useNechoes]

        y_magnitude = self.uMGE.magnitude[self.uMGE.useNechoes, r_idx, p_idx, s_idx]

        if np.ma.is_masked(y_magnitude):
            y_magnitude = y_magnitude.data
            color = "lightgray"
        else:
            color = "black"

        ax.plot(TEs, y_magnitude, color=color, marker="s")

        # add some labelling and a title
        ax.set_xlabel("TE [ms]")
        ax.set_ylabel("Mag [SNR]")
        ax.set_title("Per Pixel: Magnitude")

    def draw_T2_star_fit(self, ax, index):
        """Draw linear phase fit through echoes for ax1."""

        ax.clear()
        r_idx, p_idx, s_idx = index

        TEs = self.uMGE.TEs
        signal_evolution = self.uMGE.magnitude[:, r_idx, p_idx, s_idx]

        if np.ma.is_masked(signal_evolution):
            signal_evolution = signal_evolution.data
            color = "lightgray"
        else:
            color = "black"

        ax.scatter(TEs, signal_evolution, color=color, marker="s")

        # additionally, use the stored fitting results to plot linear graph
        # indicating the found fit function
        T2, M_0, T2_s_error, M_0_error = self.uMGE._T2_star_fit_results[
            :, r_idx, p_idx, s_idx
        ]
        T2_fit = M_0 * np.exp(-TEs / T2)

        ax.plot(TEs, T2_fit, color=color, linestyle="dashed")

        # add some labelling and a title
        ax.set_ylabel(r"signal magnitude [a. u.]")
        ax.set_xlabel(r"echo time [ms]")

        ax.set_title(r"Per Pixel: T2* = " + f"{T2:.1f} ± {T2_s_error:.1f} ms")

    def create_position_slider(self):
        """Create navigation slider for every array dimension."""
        read_dim, phase_dim, slice_dim = self.pvm_matrix
        # create interactive slider for x position
        self.read_slider = widgets.IntSlider(
            value=read_dim // 2,
            min=0,
            max=read_dim - 1,
            description=f"{self.labels[0]}: ",
        )
        self.read_slider.style.handle_color = "red"

        # create interactive slider for y position
        self.phase_slider = widgets.IntSlider(
            value=phase_dim // 2,
            min=0,
            max=phase_dim - 1,
            description=f"{self.labels[1]}: ",
        )
        self.phase_slider.style.handle_color = "green"

        # create interactive slider for slices
        self.slice_slider = widgets.IntSlider(
            value=slice_dim // 2,
            min=0,
            max=slice_dim - 1,
            description=f"{self.labels[2]}: ",
        )
        self.slice_slider.style.handle_color = "blue"

        # create interactive slider for echoes
        echo_dim = len(self.uMGE.TEs)
        self.echo_slider = widgets.IntSlider(
            value=0, min=0, max=echo_dim - 1, description="echo: "
        )

        # create interactive slider for multi-comp frequency and pH maps
        multi_comp_frequencies = len(self.uMGE._multi_comp_fmaps)
        self.multi_comp_map_slider = widgets.IntSlider(
            value=0, min=0, max=multi_comp_frequencies - 1, description="map: "
        )

    def draw_img(
        self, ax, data, drange, index, view, img_only=False, cmap=None, alpha=1
    ):
        dmin, dmax = drange if drange is not None else (None, None)
        r_idx, p_idx, s_idx = index

        r_pos = self.idx2pos(r_idx, 0)
        p_pos = self.idx2pos(p_idx, 1)
        s_pos = self.idx2pos(s_idx, 2)

        if view == "main":
            img = data[..., s_idx]
            x_tuple, y_tuple = (p_pos, 1), (r_pos, 0)
        elif view == "proj1":
            img = data[r_idx]
            x_tuple, y_tuple = (s_pos, 2), (p_pos, 1)
        elif view == "proj2":
            img = data[:, p_idx, :]
            x_tuple, y_tuple = (s_pos, 2), (r_pos, 0)

        ax.imshow(
            img,
            extent=self.get_extent(view),
            vmin=dmin,
            vmax=dmax,
            cmap=self.mode.cmap if cmap is None else cmap,
            origin="lower",
            alpha=alpha,
            **self.params,
        )

        if not img_only:
            self.add_title_and_label(ax, view)
            self.draw_cursor(ax, x_tuple, y_tuple)

    def draw_anat(self, ax, index, view):
        res_in = self.pvm_res  # could be interpolated resolution
        res_out = self.anatref.method["PVM_SpatResol"]
        if len(res_out) == 2:
            res_out = np.append(res_out, self.anatref.method["PVM_SliceThick"])

        def find_closest_index(idx_in, res_in, res_out):
            """Find closest index of an array of different resolution."""
            return int(np.round(idx_in * (res_in / res_out), decimals=0))

        r_idx = find_closest_index(index[0], res_in[0], res_out[0])
        p_idx = find_closest_index(index[1], res_in[1], res_out[1])
        s_idx = find_closest_index(index[2], res_in[2], res_out[2])

        if view == "main":
            img = self.anatref.data[..., s_idx]
        elif view == "proj1":
            img = self.anatref.data[r_idx]
        elif view == "proj2":
            img = self.anatref.data[:, p_idx, :]

        self.anatref.get_extent(view)

        ax.imshow(
            img,
            extent=self.anatref.get_extent(view),
            cmap=self.modes["magnitude"].cmap,
            origin="lower",
        )

    def update(self, **kwargs):
        """The plotting function called whenever the user interacts.

        'kwargs' are provided by the self.setup_interactive() method.
        """
        self.clear()

        self.set_mode(kwargs.get("mode_selection"))

        read_idx = kwargs.get("read_idx")
        phase_idx = kwargs.get("phase_idx")
        slice_idx = kwargs.get("slice_idx")
        index = read_idx, phase_idx, slice_idx

        data = self.mode.data
        dmin, dmax = kwargs.get("datarange_" + self.mode.name)

        if not self.mode.name in [
            "multi_comp_pHmaps",
            "multi_comp_mean_pHmap",
            "multi_comp_peak_pHmap",
        ]:
            data = np.ma.masked_where(data > dmax, data)
            data = np.ma.masked_where(data < dmin, data)

        if self.mode.name in ["magnitude", "phase"]:
            data = data[kwargs.get("echo_idx")]
            self.echo_slider.layout.visibility = "visible"
            self.multi_comp_map_slider.layout.visibility = "hidden"
        elif self.mode.name in ["multi_comp_fmaps", "multi_comp_pHmaps"]:
            data = data[kwargs.get("multi_comp_map_idx")]
            self.multi_comp_map_slider.layout.visibility = "visible"
            self.echo_slider.layout.visibility = "hidden"
        else:
            self.echo_slider.layout.visibility = "hidden"
            self.multi_comp_map_slider.layout.visibility = "hidden"

        # normalize internal data form 0 to 1
        norm = plt.Normalize(dmin, dmax)
        # plot colorbars on all caxes
        for cax in self.caxes:
            self.fig.colorbar(
                cm.ScalarMappable(norm=norm, cmap=self.mode.cmap),
                cax=cax,
                orientation="vertical",
                label=self.mode.label,
            )

        if self.mode.name in ["magnitude"]:
            self.draw_magnitude(self.ax1, index)
        elif self.mode.name in [
            "multi_comp_pHmaps",
            "multi_comp_mean_pH",
            "multi_comp_peak_pH",
        ]:
            self.draw_simulated_from_MEpHISDO_fit(self.ax1, index)
        else:
            self.draw_phasefit(self.ax1, index)

        for ax, v in zip([self.ax2, self.ax3, self.ax4], ["main", "proj1", "proj2"]):
            if kwargs.get("magnitude_background"):
                self.echo_slider.layout.visibility = "visible"
                self.opacity_slider.layout.visibility = "visible"

                if self.anatref is not None:
                    self.draw_anat(ax, index, view=v)
                else:
                    cmap = self.modes["magnitude"].cmap
                    mdata = self.uMGE.magnitude.data[
                        kwargs.get("echo_idx")
                    ]  # .data -> no mask
                    self.draw_img(
                        ax, mdata, None, index, view=v, img_only=True, cmap=cmap
                    )

                a = kwargs.get("alpha")
                self.draw_img(ax, data, (dmin, dmax), index, view=v, alpha=a)
            else:
                self.opacity_slider.layout.visibility = "hidden"
                self.draw_img(ax, data, (dmin, dmax), index, view=v, alpha=1)
        # draw title
        value = data[read_idx, phase_idx, slice_idx]
        value = f"{value:.2f}" if value else str(value)

        title = self.mode.title.replace("xXx", value)
        self.fig.suptitle(title, fontweight="heavy", fontsize="x-large")

        # show array index, incase you want to access individual voxels:
        self.ax2.set_title(f"{read_idx, phase_idx, slice_idx}", loc="right")

        # remove all axes ticks and labels from images
        for a in self.ax2, self.ax3, self.ax4:
            a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    def setup_interactive(self):
        """This method sets up the interactive ipywidget part of the plot.

        First we add a color range slider for every data mode. Then we add
        all modes to the dropdown menu where the user selects a mode. All range
        sliders are stored inside a widgets.stack.

        We align all widgets inside a css boxes for a nice layout and then link
        them with self.update().

        Then we link the 'click' event with the method onclick.
        """
        button = widgets.ToggleButton(
            value=False,
            description="Anatomical Ref.",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Magnitude Background",
            icon="",  # (FontAwesome names without the `fa-` prefix)
        )

        self.opacity_slider = widgets.FloatSlider(
            value=0.5, min=0, max=1, step=0.01, description="opacity: "
        )

        mode_dropdown = widgets.Dropdown(
            options=[(m.option_txt, m.name) for m in self.modes.values()],
            value=next(iter(self.modes.keys())),
            description="Map:",
        )

        range_stack = widgets.Stack(
            [s for s in self.range_sliders.values()],
            selected_index=0,
        )

        widgets.jslink((mode_dropdown, "index"), (range_stack, "selected_index"))

        self.create_position_slider()

        Hbox_layout = widgets.Layout(display="flex", justify_content="space-between")
        row1 = widgets.HBox(
            [self.read_slider, self.phase_slider, self.slice_slider], layout=Hbox_layout
        )
        row2 = widgets.HBox(
            [self.echo_slider, button, self.opacity_slider], layout=Hbox_layout
        )
        row3 = widgets.HBox(
            [self.multi_comp_map_slider, range_stack, mode_dropdown], layout=Hbox_layout
        )

        # put both sliders inside a HBox for nice alignment  etc.
        self.ui = widgets.VBox(
            [row1, row2, row3],
            layout=widgets.Layout(display="flex"),
        )

        std_controls = {
            "read_idx": self.read_slider,
            "phase_idx": self.phase_slider,
            "slice_idx": self.slice_slider,
            "echo_idx": self.echo_slider,
            "multi_comp_map_idx": self.multi_comp_map_slider,
            "mode_selection": mode_dropdown,
            "magnitude_background": button,
            "alpha": self.opacity_slider,
        }

        range_controls = {f"datarange_{k}": v for k, v in self.range_sliders.items()}

        controls = {**std_controls, **range_controls}

        # connect plotting function with sliders
        self.sliders = widgets.interactive_output(
            self.update,
            controls,
        )

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)


def plot_spectrum(spectra, frequ_axis):

    fig, ax = plt.subplots()

    def plot_spec(num_rep):
        [line.remove() for line in ax.lines]
        ax.clear()

        ax.plot(frequ_axis, np.abs(spectra[num_rep]))
        ax.set_ylabel("Amplitude [a.u.]")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_title("")

        fig.tight_layout()

    slider_rep = widgets.IntSlider(
        value=spectra.shape[0] // 2,
        min=0,
        max=spectra.shape[0] - 1,
        description="Spektrum: ",
    )

    out = widgets.interactive_output(
        plot_spec,
        {
            "num_rep": slider_rep,
        },
    )

    # This displays the Hbox containing the slider widgets.
    display(slider_rep, out)
