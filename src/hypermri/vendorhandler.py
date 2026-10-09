import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping

import numpy as np
import xarray as xr


class VendorHandler(ABC):
    """The 'blueprint' for how to load data from a specific vendor."""

    # Each subclass must define these properties and methods.
    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @property
    @abstractmethod
    def freq_axis_ppm(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def freq_axis_hz(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def dwell_times_s(self) -> float:
        pass

    def _parse_parameters(self, source) -> Mapping[str, Any]:
        """Return vendor-specific parameters mapped to canonical keys."""
        raise NotImplementedError

    def load_parameters(self, source) -> Dict[str, Any]:
        """Parse and persist standardized parameters for the handler instance."""
        parsed = dict(self._parse_parameters(source))
        if hasattr(self, "set_parameters"):
            self.set_parameters(parsed)
        else:
            self.params = parsed  # type: ignore[attr-defined]
        return parsed

    def get_parameters(self) -> Dict[str, Any]:
        """Return cached standardized parameters."""
        if hasattr(self, "params"):
            return dict(getattr(self, "params"))
        return {}

    @abstractmethod
    def load_data(self, source) -> xr.DataArray:
        """
        Loads the raw data from the specified source and returns it as an
        xarray.DataArray.
        """


class VendorFactory:
    """The 'factory' for creating a vendor-specific handler."""

    @staticmethod
    def get_handler(source):
        # This is a simplified example. In a real scenario, you'd inspect
        # the source (e.g., file extension, metadata file) to determine
        # the correct handler.
        if os.path.isdir(source) and "fid" in os.listdir(source):
            from hypermri.brukerexp import BrukerHandler

            return BrukerHandler(source)
        elif source.endswith(".dat"):
            from hypermri.siemensexp import SiemensHandler

            return SiemensHandler(source)
        # Add other vendors here
        else:
            raise ValueError(f"Unsupported data source: {source}")
