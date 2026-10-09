# hypermri/siemens_recon_strategy.py

import xarray as xr

from .recon_strategy import BaseRecoStrategy


class SiemensRawRecoStrategy(BaseRecoStrategy):
    """Reconstruction strategy for raw Siemens data."""

    def reconstruct(self, handler, **reco_options) -> xr.DataArray:
        """
        Perform the reconstruction for Siemens raw data.
        This is currently a placeholder and needs to be implemented.
        """
        print("SiemensRawRecoStrategy.reconstruct called")
        # Placeholder for Siemens reconstruction logic.
        # This will involve processing the data from handler.twix
        # and converting it into an xarray.DataArray.
        raise NotImplementedError(
            "Siemens raw data reconstruction is not yet implemented."
        )
