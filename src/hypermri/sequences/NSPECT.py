"""Non-localised spectroscopy sequence utilities (NSPECT)."""

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from hypermri.core.experiment import MRIExperiment

from ..analysis.spectroscopy.core import get_freq_axis
from ..utils.system.logging import LOG_MODES, init_default_logger
from .base_spectroscopy import BaseSpectroscopy

# initialize logger
logger = init_default_logger(__name__)

logger.setLevel(LOG_MODES["Critical"])


class NSPECT(BaseSpectroscopy):
    """Wrapper which decides wether to load the single or dualchannel NSPECT class.

    Accepts directory path or MRIExperiment object as input.
    """

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        """Accepts a pre-loaded MRIExperiment object as input."""
        # Extract sequence-specific kwargs before calling parent constructor
        reco_options = kwargs.pop("reco_options", {})

        super().__init__(mri_exp_obj, *args, **kwargs)

        # --- This part should be moved somewhere else, maybe to the MRIExperiment class ---
        # try:
        #     self.reconstruct(reco_type="seq2d")
        # except:
        #     raise RuntimeError(
        #         "NSPECT Reconstruction of seq2d data failed."
        #         "Please check the input data or the reconstruction strategy."
        #     )
        # Raw data processing:
        try:
            self.reconstruct(reco_type="raw", **reco_options)
        except Exception as e:
            logger.error(f"NSPECT raw reconstruction failed: {e}")
            logger.error("NSPECT: Continuing with initialization but no reconstruction")
            # Don't raise - allow partial initialization like other sequences

        # ---------------------------------------------
        # This needs to brought to recon_strategy.py
        # ---------------------------------------------

        # self.n_repetitions = self.n_repetitions
        # self.spec, self.fids, self.complex_spec = self.get_spec()

        # # if we dont have rawdatajob0 file we load fid
        # if len(self.rawdatajob0) == 0:
        #     fid = self.fid
        # else:
        #     fid = self.rawdatajob0
        # # reshape rawdata:
        # fid = np.reshape(
        #     self.fid,
        #     (
        #         self.n_repetitions,
        #         self.n_averages,
        #         self.n_slices,
        #         1,
        #         1,
        #         self.n_receivers,
        #         self.n_acquisition,
        #     ),
        # )
        # fid = np.mean(fid, axis=1)
        # fid = np.reshape(
        #     fid,
        #     (
        #         self.n_repetitions,
        #         self.n_slices,
        #         1,
        #         1,
        #         self.n_receivers,
        #         self.n_acquisition,
        #     ),
        # )

        # fid = np.transpose(fid, (5, 0, 1, 2, 3, 4))
        # fid = fid[self.cut_off :, ...]
        # # self.fid_pp = fid
        # # self.kspace = self.fid_pp
        # # self.reco_rawdata(
        # #     postprocess_fid=False,
        # #     set_kspace_xarray=True,
        # #     shift_xarray=False,
        # #     set_reco_data=True,
        # # )

    def get_spec(self, LB=0, cut_off=0):
        """
        Calculate spectra and ppm axis for non localized spectroscopy measurements with repetitions.

        Parameters
        ----------
        LB : float, optional
            linebroadening applied in Hz, default is 0
        cut_off : float, optional
            number of first points the fid is cut off as these are only noise, default is 70

        Returns
        -------
        spec : array
            magnitude spectra in a n-dimensional array
        fids: array
            linebroadened fids in a n-dimensional array
        complex_spec: array
            complex spectra in n-dimensional array
        """

        # if we dont have rawdatajob0 file we load fid
        if len(self.rawdatajob0) == 0:
            fid = self.fid
        else:
            fid = self.rawdatajob0

        n_repetitions = self.n_repetitions
        n_acquisition = self.n_acquisition
        ac_time = self.method["PVM_SpecAcquisitionTime"]

        time_ax = np.linspace(0, ac_time, n_acquisition)[cut_off:] / 1000
        sigma = 2.0 * np.pi * LB

        fids = np.zeros((n_repetitions, n_acquisition - cut_off), dtype=complex)
        spec = np.zeros((n_repetitions, n_acquisition - cut_off))
        complex_spec = np.zeros((n_repetitions, n_acquisition - cut_off), dtype=complex)

        def ffts(x):
            return np.fft.fftshift(np.fft.fft(x))

        for rep in range(n_repetitions):
            tmp_spec = ffts(
                fid[cut_off + rep * n_acquisition : n_acquisition + rep * n_acquisition]
                * np.exp(-sigma * time_ax)
            )

            complex_spec[rep, :] = tmp_spec

            fids[rep, :] = fid[
                cut_off + rep * n_acquisition : n_acquisition + rep * n_acquisition
            ] * np.exp(-sigma * time_ax)

        spec = np.abs(complex_spec)

        s = np.squeeze
        return s(spec), s(fids), s(complex_spec)

    def phase_correct_fid(self, number=0):
        """
        Interactively allows phasing of fids.
        Parameters
        ----------
        number: int, number of the fid that should be phased, in case there are repetitons.
        """
        fid = self.fid[number, :]
        spec_complex = np.fft.fftshift(np.fft.fft(fid))
        # perform a baseline correction
        spec_base_line_corr = spec_complex - np.mean(spec_complex)
        # phase the real spectrum
        Integrals_th = []
        phases = np.linspace(0, 360, 50)
        for phase in phases:
            itgl = np.sum(
                np.real(spec_base_line_corr * np.exp(1j * (phase * np.pi) / 180.0))
            )
            Integrals_th.append(itgl)
        initial_guess_phase = phases[
            np.argmin(np.abs(Integrals_th - np.max(Integrals_th)))
        ]

        fig, ax = plt.subplots(1, figsize=(12, 4), tight_layout=True)

        (line_real,) = ax.plot(
            np.real(
                spec_base_line_corr * np.exp(1j * (initial_guess_phase * np.pi) / 180.0)
            ),
            label="Real-phased",
            color="k",
        )
        (line_abs,) = ax.plot(
            np.abs(spec_base_line_corr) - np.mean(np.abs(spec_base_line_corr)),
            label="Magnitude",
            color="r",
        )
        ax.hlines(
            0,
            0,
            len(spec_base_line_corr),
            linestyles="dashed",
            alpha=0.3,
            color="b",
            label="Baseline",
        )
        # ax.fill_between([8000, 9000],np.min(np.real(spec_base_line_corr)),np.max(np.real(spec_base_line_corr)), alpha=0.3, color='C2', label='Background')
        ax.set_xlabel("Points")
        ax.set_ylabel("I [a.u.]")
        ax.set_title("Phased with " + str(np.round(phase, 1)) + " °")
        ax.legend()

        @widgets.interact(phase=(0, 360, 0.1))
        def update(phase):
            line_real.set_ydata(
                np.real(spec_base_line_corr * np.exp(1j * (phase * np.pi) / 180.0))
            )
            line_abs.set_ydata(
                np.abs(spec_base_line_corr) - np.mean(np.abs(spec_base_line_corr))
            )
            ax.set_title("Phased with " + str(np.round(phase, 1)) + " °")

    def get_one_channel_fids(self, cut_off=70):
        """
        Splits recorded data into two channels for measurements with a 2-channel receiver coil.
        Returns
        -------
        fid_ch1: np.array, shape = (n_acquisition, 1, 1, 1, n_repetitions, 1)
        fid_ch2: np.array, shape = (n_acquisition, 1, 1, 1, n_repetitions, 1)
        """

        if len(self.rawdatajob0) > 0:
            data = self.rawdatajob0
        else:
            data = self.fid

        n_repetitions = self.n_repetitions
        n_acquisition = self.n_acquisition

        fid = np.zeros((n_repetitions, n_acquisition - cut_off), dtype=complex)
        for i in range(n_repetitions):
            fid[i, :] = data[
                i * n_acquisition + cut_off : i * n_acquisition + n_acquisition
            ]

        fid = np.reshape(
            np.transpose(fid, (1, 0)),
            (n_acquisition - cut_off, 1, 1, 1, n_repetitions, 1),
        )
        return fid

    def full_reco(self, cut_off=0):
        """
        Do the full reconstruction with one call.

        Parameters
        ----------
        cut_off: int
            where to start processing the fid. 0 for custom sequences, 70 for Bruker sequences

        """
        # get frequency range (in ppm)
        self.freq_range = get_freq_axis(data_obj=self, unit="ppm", cut_off=cut_off)

        # get frequency range (in ppm)
        ac_time = self.method["PVM_SpecAcquisitionTime"]
        n_acquisition = self.n_acquisition

        # define time axis:
        dt = ac_time / n_acquisition
        self.time_axis = (
            np.linspace(0, (n_acquisition - cut_off) * dt, (n_acquisition - cut_off))
            / 1000.0
        )

        # extract FIDs
        self.fid = self.get_one_channel_fids(cut_off=cut_off)

        # get spectra
        (
            self.freq_range,
            self.spec,
        ) = self.plot_spec_non_localized_spectroscopy_one_channel(
            cut_off=cut_off,
            fid=self.fid,
            return_complex=True,
        )

    def apply_phase_shift_dual_channel(
        self, phase_shift=None, cut_off=70, fid_ch1=None, fid_ch2=None
    ):
        """
        Applies phase correction by given value.
        Parameters
        ----------
        phase_shift: float, in degree, phase by which channel 1 is shifted against channel 2
        LB: float, linebroadening applied to spectra
        cut_off: int, number of points at beginning of fid that are left out.
        Returns
        -------
        phased_fid
        """
        n_repetitions = self.n_repetitions
        n_acquisition = self.n_acquisition

        if (fid_ch1 is None) and (fid_ch2 is None):
            ch_1 = self.fid_ch1
            ch_2 = self.fid_ch2
        else:
            ch_1 = fid_ch1
            ch_2 = fid_ch2
        if phase_shift is None:
            phase_shift = self.find_phase_shift_dual_channel(
                cut_off=cut_off, plot=False, fid_ch1=fid_ch1, fid_ch2=fid_ch2
            )
        else:
            pass

        phased_fid = np.zeros(
            (n_acquisition - cut_off, 1, 1, 1, n_repetitions, 1), dtype=complex
        )
        for n in np.arange(0, n_repetitions):
            phased_fid[:, :, :, :, n, :] = (
                ch_1[:, :, :, :, n, :] * np.exp(1j * (phase_shift * np.pi) / 180.0)
                + ch_2[:, :, :, :, n, :]
            )
        return phased_fid
