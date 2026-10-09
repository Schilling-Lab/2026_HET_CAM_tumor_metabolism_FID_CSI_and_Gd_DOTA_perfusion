from hypermri.core.experiment import MRIExperiment
from hypermri.utils.system.logging import init_default_logger

from .base_spectroscopy import BaseSpectroscopy

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


class Singlepulse(BaseSpectroscopy):
    """Wrapper which decides wether to load the single or dualchannel Singlepulse class.

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
        #         "Singlepulse Reconstruction of seq2d data failed."
        #         "Please check the input data or the reconstruction strategy."
        #     )
        # Raw data processing:
        try:
            self.reconstruct(reco_type="raw", **reco_options)
        except Exception as e:
            logger.error(f"Singlepulse raw reconstruction failed: {e}")
            logger.error(
                "Singlepulse: Continuing with initialization but no reconstruction"
            )
            # Don't raise - allow partial initialization like other sequences

    # ---------------------------------------------
    # This needs to brought to recon_strategy.py
    # ---------------------------------------------
