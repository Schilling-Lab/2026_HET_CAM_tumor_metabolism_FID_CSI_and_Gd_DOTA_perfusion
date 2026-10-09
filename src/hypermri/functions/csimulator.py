import ipywidgets as widgets
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..analysis.spectroscopy.core import get_freq_axis, norm


class CSImulator:
    def __init__(self, csi_object, voxel_selection=dict()):
        self.csi = csi_object
        self.fbins = get_freq_axis(data_obj=self, unit="Hz", cut_off=0, npoints=None)

        self.voxel_selection = voxel_selection

        self.magnitudes = None
        self.phases = None
        self.timepoints = None

    @staticmethod
    def tuple2idx(tup):
        return np.index_exp[:, tup[0], tup[1]]

    @property
    def labels(self):
        """Return labels from all voxel selections."""
        return list(self.voxel_selection.keys())

    @property
    def spectra(self):
        """Return spectra of all voxels from all voxel selections"""
        return [
            np.squeeze(self.csi.data[self.tuple2idx(tup)])
            for tup in list(self.voxel_selection.values())
        ]

    def propagate(self, A_spec, t):
        """Propagate all isochromates for a time t."""
        phase = 2 * np.pi * self.fbins * t

        return A_spec * np.exp(1j * phase), phase

    def spectrum2phase(self, A_spec, t, return_spin_vec_sum=False):
        """Propagate isochromates, afterwards collaps them and calculate effective total signal."""
        # let the spins propagate:
        spin_vecs, true_phase = self.propagate(A_spec, t)

        spin_vecs_sum = np.sum(spin_vecs)

        resulting_r = np.abs(spin_vecs_sum)
        resulting_theta = np.angle(spin_vecs_sum)

        if return_spin_vec_sum:
            return resulting_r, resulting_theta, spin_vecs_sum
        else:
            return resulting_r, resulting_theta

    def simulate(self, timepoints):
        """Generate simulated phase and magnitude images at given timepoints"""
        spatial_dims = self.csi.seq2d.shape[1:]
        phases = np.zeros((len(timepoints), *spatial_dims))
        magnitudes = np.zeros((len(timepoints), *spatial_dims))
        complex_data = np.zeros((len(timepoints), *spatial_dims), dtype="complex")

        for i, t in enumerate(timepoints):

            def func(spec):
                return self.spectrum2phase(spec, t, True)

            for idx, ph in self.csi.apply_to_every_spec(func):
                magnitudes[(i, *idx)] = ph[0]
                phases[(i, *idx)] = ph[1]
                complex_data[(i, *idx)] = ph[2]

        self.phases = np.squeeze(phases)
        self.magnitudes = np.squeeze(magnitudes)
        self.complex_data = complex_data
        self.timepoints = np.array(timepoints)

    def clocks(self, t_min=0, t_max=3e-3, t_step=0.5e-6):
        """Plot polar plots of all voxel selection spectra."""
        N = len(self.voxel_selection)

        assert N != 0, "No voxel selection provided!"

        fig, axes = plt.subplots(
            ncols=N, figsize=(9.5, 5.5), subplot_kw={"projection": "polar"}
        )

        r_total_t0 = [self.spectrum2phase(spec, t=0)[0] for spec in self.spectra]

        t_range = np.arange(t_min, t_max, t_step)

        self.__layout_adjusted = False  # ugly patch to fix fig.layout + widgets combo

        def plot_func(time_idx, show_total_vec):
            time = t_range[time_idx]
            for i in range(N):
                ax = axes[i]
                spec = self.spectra[i]
                label = self.labels[i]

                zs, true_phases = self.propagate(spec, time)

                rs = norm(np.abs(zs))
                phases = np.angle(zs)

                r_total = np.abs(np.sum(zs)) / r_total_t0[i] * 1.1
                phase_total = np.angle(np.sum(zs))

                ax.clear()
                ax.plot(phases, rs, color="blue")

                if show_total_vec:
                    ax.plot(
                        [r_total, phase_total],
                        [0, r_total],
                        linewidth=2.5,
                        color="orange",
                    )

                # some style improvements
                ax.set_rmax(1.5)
                ax.set_rticks([0.5, 1])  # Less radial ticks
                ax.set
                # Move radial labels away from plotted line
                ax.set_rlabel_position(90)
                ax.grid(True)

                title = r"$\bf{" + str(label) + "}$"
                title += "\n$\\phi_{tot} = $" + f"{phase_total:.3f} rad"
                ax.set_title(title)

            fig.suptitle(f"{time * 1e3:.2f} ms", fontweight=400, fontsize=20, y=0.99)

            if not self.__layout_adjusted:
                fig.tight_layout()
                self.__layout_adjusted = True

        time_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(t_range) - 1,
            step=1,
            readout=False,
            layout=widgets.Layout(width="500px"),
        )

        time_play = widgets.Play(
            value=0,
            min=0,
            max=len(t_range) - 1,
            step=1,
            interval=5,
            description="Press play",
            style={"description_width": "initial"},
        )

        play_speed_int = widgets.BoundedIntText(
            value=10,
            min=0,
            max=100,
            step=10,
            description="Speed:",
            layout=widgets.Layout(width="140px"),
        )

        widgets.jslink((time_play, "value"), (time_slider, "value"))
        widgets.jslink((time_play, "interval"), (play_speed_int, "value"))

        total_vec_button = widgets.ToggleButton(
            value=True,
            description="Total Vector",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        )
        # put both sliders inside a HBox for nice alignment  etc.
        row1 = widgets.HBox([time_play, time_slider, play_speed_int, total_vec_button])

        ui = widgets.VBox(
            [row1],
            layout=widgets.Layout(display="flex"),
        )

        sliders = widgets.interactive_output(
            plot_func,
            {
                "time_idx": time_slider,
                "show_total_vec": total_vec_button,
            },
        )

        display(ui, sliders)

    def plot(self, mag_cmap="gray", pha_cmap="bwr"):
        """Plot simulated magnitude and phase images."""
        assert self.timepoints is not None, "You have to call self.simulate() first."

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9.5, 5))

        # sort out arrangement of colorbar and plot
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)

        self.__layout_adjusted = False  # ugly patch for fig.tight_layout & widgets

        def plot_func(echo_nr):
            for ax, cax, cmap, data, title, label, ticks in zip(
                [ax1, ax2],
                [cax1, cax2],
                [mag_cmap, pha_cmap],
                [self.magnitudes, self.phases],
                ["magnitude", "phase"],
                ["a.u.", "rad"],
                [[], None],
            ):
                ax.clear()
                ax.imshow(data[echo_nr], cmap=cmap)
                ax.set_title(title)
                # normalize internal data form 0 to 1
                norm = plt.Normalize(np.nanmin(data), np.nanmax(data))
                # plot the colorbar
                fig.colorbar(
                    cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cax,
                    orientation="vertical",
                    label=label,
                    ticks=ticks,
                )

            fig.suptitle(
                f"{self.timepoints[echo_nr] * 1e3:.2f} ms", fontsize=15, fontweight=500
            )

            if not self.__layout_adjusted:
                fig.tight_layout()
                self.__layout_adjusted = True

        echo_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.timepoints) - 1,
            step=1,
            description="Echo Nr.:",
            style={"description_width": "initial"},
        )

        # put both sliders inside a HBox for nice alignment  etc.
        row1 = widgets.HBox([echo_slider])
        ui = widgets.VBox(
            [row1],
            layout=widgets.Layout(display="flex"),
        )
        sliders = widgets.interactive_output(
            plot_func,
            {
                "echo_nr": echo_slider,
            },
        )
        display(ui, sliders)

    def plot_selection_spectra(self, xmin=-1000, xmax=1000):
        """Plot spectra of selected voxel."""
        fig, ax = plt.subplots(figsize=(9, 4))

        N = len(self.voxel_selection)

        for i in range(N):
            ax.plot(self.fbins, self.spectra[i], label=self.labels[i])

        ax.set_xlim(xmin, xmax)
        ax.legend()
        ax.set_xlabel("Frequency [Hz]")
        ax.set_title("Spectra of Voxel Selection")

        fig.tight_layout()
        plt.show()

    def plot_selection_phase(self):
        """Plot phase evolution of selected voxel."""
        assert self.timepoints is not None, "You have to call self.simulate() first."

        fig, ax = plt.subplots()

        N = len(self.voxel_selection)

        for i in range(N):
            idx_tuple = list(self.voxel_selection.values())[i]
            ax.plot(
                self.timepoints * 1e3,
                self.phases[self.tuple2idx(idx_tuple)],
                marker="s",
                label=self.labels[i],
            )

        ax.set_xlabel("t [ms]")
        ax.set_ylabel("phase [rad]")
        ax.set_title("Phase Evolution of Voxel Selection")
        ax.legend()
        fig.tight_layout()
