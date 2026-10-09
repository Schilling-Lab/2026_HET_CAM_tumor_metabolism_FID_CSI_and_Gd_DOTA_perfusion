"""Point-resolved spectroscopy (PRESS) sequence utilities."""

from hypermri.core.experiment import MRIExperiment

from ..utils.system.logging import LOG_MODES, init_default_logger
from .base_spectroscopy import BaseSpectroscopy

# initialize logger
logger = init_default_logger(__name__)

logger.setLevel(LOG_MODES["Critical"])


class PRESS(BaseSpectroscopy):
    """Point-resolved spectroscopy with voxel localisation and review helpers."""

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        """Accepts a pre-loaded MRIExperiment object as input."""
        # Extract sequence-specific kwargs before calling parent constructor
        reco_options = kwargs.pop("reco_options", {})

        super().__init__(mri_exp_obj, *args, **kwargs)

        # Copy handler from the parent MRIExperiment object
        if hasattr(mri_exp_obj, "handler"):
            self.handler = mri_exp_obj.handler

        # Copy any existing reconstructed data
        if hasattr(mri_exp_obj, "data_array") and mri_exp_obj.data_array is not None:
            self.data_array = mri_exp_obj.data_array
        # kspace_array no longer stored permanently

        # Seq2d data processing:
        try:
            self.reconstruct(reco_type="seq2d", **reco_options)
        except:
            logger.error("PRESS seq2d reconstruction failed")
            # Don't raise - allow partial initialization

        # Raw data processing:
        try:
            self.reconstruct(reco_type="raw", **reco_options)
        except Exception as e:
            logger.error(f"PRESS raw reconstruction failed: {e}")
            # Don't raise - allow partial initialization
