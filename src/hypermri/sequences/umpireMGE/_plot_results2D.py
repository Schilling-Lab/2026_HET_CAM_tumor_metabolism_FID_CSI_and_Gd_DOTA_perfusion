import ipywidgets as widgets

# colorbar
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ._plot_results3D import UmpirePlotter3D

DEFAULT_PARAMETER = {"interpolation": "none", "fmap": {"cmap": "plasma"}}


class UmpirePlotter2D(UmpirePlotter3D):
    """umpireMGE 2D plotter class

    This class is based on the more general 3D case.

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

    def setup_figure(self):
        """Setup the figure and axes layout."""

        with plt.ioff():
            fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(9, 4.5))

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

    def get_extent(self, view):
        """Calculate the image extent for a given view.

        In this 2D case, there is only a single view:

               b-----.
        main : |     |
               a,c---d

        Parameter
        ----------
        view : str {'main', 'proj1', 'proj2'}
            Returns extent of the image itself or one of its two projections.
        """
        read_ext, phase_ext, _ = self.pvm_fov

        a = -read_ext / 2  # - 0.5
        b = read_ext / 2  # - 0.5
        c = -phase_ext / 2  # - 0.5
        d = phase_ext / 2  # - 0.5

        if view == "main":
            return (c, d, a, b)

    def draw_ax1(self, r_idx, p_idx):
        """Draw linear phase fit through echoes for ax1."""
        TEs = self.uMGE.TEs
        y_phase = self.uMGE.unwrapped_phase[:, r_idx, p_idx]

        self.ax1.scatter(TEs, y_phase, color="black", marker="s")

        # additionally, use the stored fitting results to plot linear graph
        # indicating the found fit function
        m, m_err, t, t_err = self.uMGE._fit_results[r_idx, p_idx]
        y_fit = TEs * m + t

        self.ax1.plot(TEs, y_fit, color="gray", linestyle="dashed")

        # add some labelling and a title
        self.ax1.set_xlabel("TE [ms]")
        self.ax1.set_ylabel("Phase [rad]")
        # self.ax1.set_title("Single Voxel Phase Fit")
        self.ax1.set_title(
            r"Per Pixel: $\quad\frac{d\phi}{dt}=$"
            + f"{m * 1e3 / (2 * np.pi):.1f} ± {m_err * 1e3 / (2 * np.pi):.1f} Hz"
        )

    def create_position_slider(self):
        """Create navigation slider for every array dimension."""
        read_dim, phase_dim, _ = self.pvm_matrix
        # create interactive slider for x position
        self.read_slider = widgets.IntSlider(
            value=read_dim // 2,
            min=0,
            max=read_dim - 1,
            description=f"{self.labels[0]}: ",
        )
        # create interactive slider for y position
        self.phase_slider = widgets.IntSlider(
            value=phase_dim // 2,
            min=0,
            max=phase_dim - 1,
            description=f"{self.labels[1]}: ",
        )
        # create interactive slider for echoes
        echo_dim = len(self.uMGE.TEs)
        self.echo_slider = widgets.IntSlider(
            value=0, min=0, max=echo_dim - 1, description="echo: "
        )

        # create interactive slider for multi_comp frequency and pH maps
        multi_comp_frequencies = len(self.uMGE._multi_comp_fmaps)
        self.multi_comp_map_slider = widgets.IntSlider(
            value=0, min=0, max=multi_comp_frequencies - 1, description="map: "
        )

    def update(self, **kwargs):
        """The plotting function called whenever the user has interacted.

        'kwargs' are provided by the self.setup_interactive() function.
        """
        self.clear()

        self.set_mode(kwargs.get("mode_selection"))

        data = self.mode.data

        if self.mode.name in ["magnitude", "phase"]:
            data = data[kwargs.get("echo_idx")]
            self.echo_slider.layout.visibility = "visible"
            self.multi_comp_map_slider.layout.visibility = "hidden"
        elif self.mode.name in ["multi_comp_fmaps", "multi_comp_pHmap"]:
            data = data[kwargs.get("multi_comp_map_idx")]
            self.multi_comp_map_slider.layout.visibility = "visible"
            self.echo_slider.layout.visibility = "hidden"
        else:
            self.echo_slider.layout.visibility = "hidden"
            self.multi_comp_map_slider.layout.visibility = "hidden"

        read_idx = kwargs.get("read_idx")
        read_pos = self.idx2pos(read_idx, 0)

        phase_idx = kwargs.get("phase_idx")
        phase_pos = self.idx2pos(phase_idx, 1)

        dmin, dmax = kwargs.get("datarange_" + self.mode.name)

        data = np.ma.masked_where(data > dmax, data)
        data = np.ma.masked_where(data < dmin, data)

        # normalize internal data form 0 to 1
        norm = plt.Normalize(dmin, dmax)

        # plot the colorbar
        for cax in self.caxes:
            self.fig.colorbar(
                cm.ScalarMappable(norm=norm, cmap=self.mode.cmap),
                cax=cax,
                orientation="vertical",
                label=self.mode.label,
            )

        self.draw_ax1(read_idx, phase_idx)

        # plot image
        self.ax2.imshow(
            data,
            extent=self.get_extent("main"),
            vmin=dmin,
            vmax=dmax,
            cmap=self.mode.cmap,
            origin="lower",
            **self.params,
        )
        self.add_title_and_label(self.ax2, "main")
        self.draw_cursor(self.ax2, (phase_pos, 1), (read_pos, 0))

        value = data[read_idx, phase_idx]
        value = f"{value:.2f}" if value else str(value)

        title = self.mode.title.replace("xXx", value)
        self.fig.suptitle(title, fontweight="heavy", fontsize="x-large")

    def setup_interactive(self):
        """This method sets up the plot and then calls self.show()

        First we add a color range slider for every data mode. Then we add
        all modes to the dropdown menu where the user selects a mode. All range
        sliders are stored inside a widgets.stack.

        We align all widgets inside a css boxes for a nice layout and then link
        them with self.update().

        Then we link the 'click' event with the method onclick.
        """
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

        row1 = widgets.HBox([self.read_slider, self.phase_slider])
        row2 = widgets.HBox([self.echo_slider])
        row3 = widgets.HBox([self.multi_comp_map_slider, range_stack, mode_dropdown])

        # put both sliders inside a HBox for nice alignment  etc.
        self.ui = widgets.VBox(
            [row1, row2, row3],
            layout=widgets.Layout(display="flex"),
        )

        std_controls = {
            "read_idx": self.read_slider,
            "phase_idx": self.phase_slider,
            "echo_idx": self.echo_slider,
            "multi_comp_map_idx": self.multi_comp_map_slider,
            "mode_selection": mode_dropdown,
        }

        range_controls = {f"datarange_{k}": v for k, v in self.range_sliders.items()}

        controls = {**std_controls, **range_controls}

        # connect plotting function with sliders
        self.sliders = widgets.interactive_output(
            self.update,
            controls,
        )

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
