"""B1 calibration sequence wrapper for acquiring power-response curves."""

from hypermri.core.experiment import MRIExperiment

from .base_spectroscopy import BaseSpectroscopy


class B1Calibration(BaseSpectroscopy):
    """Single-voxel spectroscopy sequence used to calibrate B1 flip angles."""

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        """Accepts directory path or BrukerExp object as input."""
        # Extract sequence-specific kwargs before calling parent constructor
        reco_options = kwargs.get("reco_options", {})
        minimal_loading = reco_options.get("minimal_loading", False)

        super().__init__(mri_exp_obj, *args, **kwargs)

        # --- This part should be moved somewhere else, maybe to the MRIExperiment class ---
        if not minimal_loading:
            try:
                self.reconstruct(reco_type="seq2d")
            except BaseException:
                pass
            # Raw data processing:
            try:
                self.reconstruct(reco_type="raw")
            except BaseException:
                pass
