from ..core import MRIExperiment
from ..utils.system.logging import LOG_MODES, init_default_logger
from .base_spectroscopy import BaseSpectroscopy

# initialize logger
logger = init_default_logger(__name__)
logger.setLevel(LOG_MODES["Warning"])


class STEAM(BaseSpectroscopy):
    """Stimulated echo acquisition mode sequence for localized spectroscopy."""

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        """Accepts a pre-loaded MRIExperiment object as input."""
        # Extract sequence-specific kwargs before calling parent constructor
        kwargs.pop("reco_options", {})

        super().__init__(mri_exp_obj, *args, **kwargs)

        # --- This part should be moved somewhere else, maybe to the MRIExperiment class ---
        try:
            self.reconstruct(reco_type="seq2d")
        except Exception as e:
            logger.error(f"STEAM seq2d reconstruction failed: {e}")
            # Don't raise - allow partial initialization

        # Raw data processing:
        try:
            self.reconstruct(reco_type="raw")
        except Exception as e:
            logger.error(f"STEAM raw reconstruction failed: {e}")
            logger.error("STEAM: Continuing with initialization but no reconstruction")
            # Don't raise - allow partial initialization like other sequences
