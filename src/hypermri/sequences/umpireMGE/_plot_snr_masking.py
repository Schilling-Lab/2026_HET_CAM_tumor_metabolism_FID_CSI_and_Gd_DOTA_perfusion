import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from ...utils.plot.plotterABC import PlotterABC
from ._snr_masking import SNR, createMaskFromSliceIdx, createMaskFromThreshold


class SNRMaskingPlotter(PlotterABC):
    def __init__(self, umpireMGEobject, echoNr=2):
        print(f"EchoNr = {echoNr} [0...N]")
        self.umge = umpireMGEobject

        self.magnitude = self.umge.magnitude[echoNr]

        self.setup_figure()
        self.setup_interactive()
        self.show()

    def setup_figure(self):
        with plt.ioff():
            self.fig = plt.figure(figsize=(8, 6))

            gs = self.fig.add_gridspec(3, 2)
            self.ax1 = self.fig.add_subplot(gs[0:2, 0])
            self.ax2 = self.fig.add_subplot(gs[0:2, 1])
            self.ax3 = self.fig.add_subplot(gs[2, :])

    def reset(self):
        for a in [self.ax1, self.ax2, self.ax3]:
            a.clear()

        self.save_button.disabled = False
        self.save_button.button_style = "warning"
        self.save_button.description = " SAVE MASK"

    def save(self, mask_mag, stddev):
        """Store masking results and set save-button status accordingly."""
        if mask_mag is np.False_:
            self.save_button.value = False
            self.save_button.button_style = "danger"
            self.save_button.description = " FAILED"
            self.save_button.disabled = True
        else:
            self.umge.mask_dict["snr"] = np.squeeze(mask_mag)
            self.umge._noise_stddev = stddev

            self.save_button.value = False
            self.save_button.button_style = "success"
            self.save_button.description = " SAVED"
            self.save_button.disabled = True

    def update(self, **kwargs):
        """Variables defined outside this scope:
        fig, ax1, ax2, ax4, save_button, slice_slider, magnitude
        """
        self.reset()

        thr = kwargs.get("thr")
        save = kwargs.get("save")
        factor = kwargs.get("factor")
        slice_nr = kwargs.get("slice_nr")

        # SNR calculation part
        SNR_scaled_mag, noise_idx, stddev = SNR(self.magnitude, factor)

        mask_mag = createMaskFromThreshold(SNR_scaled_mag, thr)
        mask_noise = createMaskFromSliceIdx(SNR_scaled_mag, noise_idx)

        SignalArea = np.ma.masked_array(data=SNR_scaled_mag, mask=mask_mag)
        NoisyArea = np.ma.masked_array(data=SNR_scaled_mag, mask=mask_noise)

        if np.abs(self.thr_slider.max - np.ma.max(SignalArea)) > 1:
            self.thr_slider.max = np.ma.max(SignalArea)

        if self.magnitude.ndim == 2:

            def make3d(x):
                return np.expand_dims(x, axis=2)

            self.magnitude = make3d(self.magnitude)
            SignalArea = make3d(SignalArea)
            NoisyArea = make3d(NoisyArea)

            self.slice_slider.layout.visibility = "hidden"

        # SNR plotting part
        self.ax1.imshow(self.magnitude[..., slice_nr], alpha=0.6)
        self.ax1.imshow(NoisyArea[..., slice_nr])
        self.ax1.set_title("Noise Area Selection")
        self.ax1.tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )

        self.ax2.imshow(SignalArea[..., slice_nr])
        self.ax2.set_title("SNR masked image")
        self.ax2.tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )

        bin_heights, bin_borders, patches = self.ax3.hist(
            SNR_scaled_mag.flatten(), bins=80
        )
        self.ax3.set_yscale("log")
        self.ax3.grid()
        self.ax3.axvline(thr, color="orange", linewidth=2)
        self.ax3.set_xlabel("SNR")
        self.ax3.set_title("SNR Histogram")
        self.ax3.set_xlim(0, None)

        if save:
            self.save(mask_mag, stddev)

    def setup_interactive(self):
        Nslices = (
            self.umge.interp_matrix[2] if self.umge.interpolated else self.umge.Nslices
        )
        # create interactive slider for  slices
        self.slice_slider = widgets.IntSlider(
            value=Nslices // 2,
            min=0,
            max=Nslices - 1,
            description="Slice: ",
        )

        # create interactive slider for threshold
        self.thr_slider = widgets.FloatSlider(
            value=10,
            min=0,
            max=50,
            description="Thr. SNR: ",
        )

        # create interactive slider for threshold
        self.factor_slider = widgets.FloatSlider(
            value=0.05,
            min=0,
            max=0.25,
            step=0.001,
            description="Factor: ",
        )

        self.save_button = widgets.ToggleButton(
            value=False,
            description=" SAVE MASK",
            disabled=False,
            button_style="danger",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Click to save SNR mask.",
            icon="save",  # (FontAwesome names without the `fa-` prefix)
            layout=widgets.Layout(height="40px", width="200px"),
        )

        self.sliders = widgets.interactive_output(
            self.update,
            {
                "slice_nr": self.slice_slider,
                "thr": self.thr_slider,
                "factor": self.factor_slider,
                "save": self.save_button,
            },
        )

        row1 = widgets.HBox([self.slice_slider, self.thr_slider, self.factor_slider])
        row2 = widgets.HBox(
            [self.save_button],
            layout=widgets.Layout(
                display="flex", justify_content="flex-end", width="90%"
            ),
        )

        self.ui = widgets.VBox([row1, row2])

    def show(self):
        self.fig.tight_layout()
        display(self.ui, self.sliders)
        plt.show()
