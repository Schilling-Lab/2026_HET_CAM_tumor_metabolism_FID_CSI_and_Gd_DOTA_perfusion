"""Multi-echo gradient echo (MGE) sequence utilities."""

import numpy as np

from hypermri.core.experiment import MRIExperiment
from hypermri.recon_strategy import BrukerMGERawRecoStrategy
from hypermri.reconstruction.registry import register_reconstruction_strategy

from .base_spectroscopy import BaseSpectroscopy


class MGE(BaseSpectroscopy):
    """Multi-echo gradient echo sequence encapsulating TE-dependent reconstructions."""

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        """Accepts a pre-loaded MRIExperiment object as input."""
        # Extract sequence-specific kwargs before calling parent constructor
        reco_options = kwargs.get("reco_options", {})
        minimal_loading = reco_options.get("minimal_loading", False)

        # Register the MGE-specific raw reconstruction strategy
        try:
            register_reconstruction_strategy(
                vendor="bruker",
                sequence="mge",
                reco_type="raw",
                strategy_class=BrukerMGERawRecoStrategy,
                options={},
            )
        except Exception as e:
            print(f"Warning: Failed to register MGE raw strategy: {e}")

        super().__init__(mri_exp_obj, *args, **kwargs)

        # Initialize MGE specific timing parameters
        self.TEs = self.get_TEs()
        if self.TEs is not None:
            # Ensure TEs is numpy array and convert ms to s if needed (EffectiveTE is usually ms)
            self.TEs = np.atleast_1d(self.TEs)

            # Check if TEs seem to be in ms (typical for Bruker) -> convert to s
            # If they are very small (e.g. < 0.5), they might already be in s, but typical TEs are > 1ms
            # Assuming ms as per standard Bruker parameter 'EffectiveTE'
            self.sampling_time_axis_s = self.TEs / 1000.0

            # Calculate dwell time from echoes
            if len(self.sampling_time_axis_s) > 1:
                # Assume uniform spacing for dwell time calculation
                self.dwell_time_s = (
                    self.sampling_time_axis_s[1] - self.sampling_time_axis_s[0]
                )
            elif len(self.sampling_time_axis_s) == 1:
                self.dwell_time_s = self.sampling_time_axis_s[0]
            else:
                self.dwell_time_s = 0.001  # Default to 1ms if unknown

            # Set bandwidth_hz for compatibility with spectroscopy tools
            if self.dwell_time_s > 0:
                self.bandwidth_hz = 1.0 / self.dwell_time_s
            else:
                self.bandwidth_hz = 0.0

            # Calculate bandwidth_ppm
            try:
                # Try to get reference frequency
                bf_mhz = None
                if hasattr(self, "method"):
                    bf_mhz = self.method.get("PVM_FrqRef", None)
                    if bf_mhz is None:
                        bf_mhz = self.method.get("PVM_FrqWork", None)

                # Handle list/array/tuple
                if isinstance(bf_mhz, (list, tuple, np.ndarray)):
                    bf_mhz = bf_mhz[0]

                if bf_mhz and float(bf_mhz) > 0:
                    self.bandwidth_ppm = self.bandwidth_hz / float(bf_mhz)
                else:
                    self.bandwidth_ppm = 0.0
            except Exception as e:
                print(f"Warning: Failed to calculate bandwidth_ppm: {e}")
                self.bandwidth_ppm = 0.0

            # Update handler to ensure base_spectroscopy picks it up
            if hasattr(self, "handler"):
                self.handler.sampling_time_axis_s = self.sampling_time_axis_s
                self.handler.dwell_time_s = self.dwell_time_s
                self.handler.bandwidth_hz = self.bandwidth_hz
                self.handler.bandwidth_ppm = self.bandwidth_ppm

            # Ensure n_acquisition is set to number of echoes for proper spectral axis
            if self.TEs is not None:
                self.n_acquisition = len(self.TEs)
                if hasattr(self, "handler"):
                    self.handler.n_acquisition = self.n_acquisition

        if not minimal_loading:
            # Ensure raw data is loaded before reconstruction
            if not hasattr(self.handler, "fid") or self.handler.fid is None:
                try:
                    self.handler.load_data(source=str(self.path))
                except Exception as e:
                    print(f"Warning: Failed to load raw data: {e}")

            # --- This part should be moved somewhere else, maybe to the MRIExperiment class ---
            try:
                self.reconstruct(reco_type="seq2d", **reco_options)
            except BaseException as e:
                print(f"MGE seq2d reconstruction failed: {e}")
            # Raw data processing:
            try:
                self.reconstruct(reco_type="raw", **reco_options)
            except BaseException as e:
                print(f"MGE raw reconstruction failed: {e}")

        # ---------------------------------------------
        # This needs to brought to recon_strategy.py
        # ---------------------------------------------

        # self.TEs = self.get_TEs()
        # self.reco_rawdata(
        #     postprocess_fid=True,
        #     reshape_fid=True,
        #     set_kspace_xarray=True,
        #     shift_xarray=True,
        #     set_reco_data=True,
        #     db=True,
        # )
        # if self.type == "<Bruker:MGE>":
        #     self.reco_seq2d()
        # elif self.type == "<Bruker:EPSI>":
        #     temp = self.seq2d.copy()
        #     # Desired initial axes permutation
        #     initial_axes = [0, 2, 1]
        #     specified_axes_set = set(initial_axes)
        #     remaining_axes = sorted(
        #         [i for i in range(temp.ndim) if i not in specified_axes_set]
        #     )
        #     final_axes = initial_axes + remaining_axes
        #     temp = np.transpose(temp, final_axes)

        #     self.reco_seq2d(input_data=temp)
        # else:
        #     pass
        # self.reco_all_old()

    def get_TEs(self):
        """Return the effective echo times reported by the protocol."""
        if self.type == "<Bruker:MGE>":
            return self.method["EffectiveTE"]
        elif self.type == "<Bruker:EPSI>":
            return self.method["PVM_EpiEchoSpacing"]
        elif "EffectiveTE" in self.method:
            return self.method["EffectiveTE"]
        else:
            pass
