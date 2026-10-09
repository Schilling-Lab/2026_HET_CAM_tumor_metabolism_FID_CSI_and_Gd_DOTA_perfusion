"""FAIR-RARE perfusion imaging sequence implementation."""

import traceback

from hypermri.core.experiment import MRIExperiment

from .base_anatomy import BaseAnatomy


class FAIR_RARE(BaseAnatomy):
    """Flow-sensitive alternating inversion recovery combined with RARE readout."""

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        """Initialise the FAIR-RARE wrapper and run the default reconstructions."""
        # Extract sequence-specific kwargs before calling parent constructor
        reco_options = kwargs.get("reco_options", {})
        minimal_loading = reco_options.get("minimal_loading", False)

        super().__init__(mri_exp_obj, *args, **kwargs)

        if not minimal_loading and not self._data_array_loaded:
            try:
                self.reconstruct(reco_type="seq2d", **reco_options)
            except Exception:
                traceback.print_exc()

        if not minimal_loading:
            try:
                self.reconstruct(reco_type="raw", **reco_options)
            except Exception:
                traceback.print_exc()
