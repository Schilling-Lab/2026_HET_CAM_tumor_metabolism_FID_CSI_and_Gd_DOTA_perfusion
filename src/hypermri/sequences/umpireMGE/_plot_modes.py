from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class dataModeClass:
    """Container for all metadata regarding a data mode."""

    name: str
    data_attr: str
    option_txt: str
    label: str
    cmap: str
    title: str
    src: Any = None
    min_func: Any = np.nanmin
    max_func: Any = np.nanmax
    default_range: Any = None

    @property
    def data(self):
        return getattr(self.src, self.data_attr)

    @property
    def min(self):
        if (
            self.name == "multi_comp_mean_pH"
            or self.name == "multi_comp_pHmaps"
            or self.name == "pHmap"
            or self.name == "multi_comp_peak_pH"
        ):
            return 6.6
        elif self.name == "T2_star_map":
            return 0.0
        else:
            return self.min_func(self.data)

    @property
    def max(self):
        if (
            self.name == "multi_comp_mean_pH"
            or self.name == "multi_comp_pHmaps"
            or self.name == "pHmap"
            or self.name == "multi_comp_peak_pH"
        ):
            return 7.8
        elif self.name == "T2_star_map":
            return 40.0
        else:
            return self.max_func(self.data)


MODE_magnitude = dataModeClass(
    name="magnitude",
    data_attr="magnitude",
    option_txt="Magnitude",
    label="Magnitude [SNR]",
    cmap="gray",
    title="Magnitude [xXx SNR]",
)

MODE_phase = dataModeClass(
    name="phase",
    data_attr="phase",
    option_txt="Phase",
    label="Phase [rad]",
    cmap="seismic",
    title="Phase [xXx rad]",
)

MODE_fmap = dataModeClass(
    name="fmap",
    data_attr="fmap",
    option_txt="Frequency",
    label="Frequency [Hz]",
    cmap="plasma",
    title="Frequency Map [xXx Hz]",
)

MODE_multi_comp_fmaps = dataModeClass(
    name="multi_comp_fmaps",
    data_attr="_multi_comp_fmaps",
    option_txt="Frequency (Multi-Comp)",
    label="Frequency [Hz]",
    cmap="plasma",
    title="Frequency Map (Multi-Comp) [xXx Hz]",
)

MODE_fmap_err = dataModeClass(
    name="fmap_err",
    data_attr="fmap_err",
    option_txt="Fit-Error (Absolute)",
    label="Error [Hz]",
    cmap="Reds",
    title="Fit-Error Map [xXx Hz]",
)

MODE_fmap_err_rel = dataModeClass(
    name="fmap_err_rel",
    data_attr="fmap_err_rel",
    option_txt="Fit-Error (Relative)",
    label="Error [%]",
    cmap="Reds",
    title="Fit-Error Map [xXx %]",
    max_func=lambda x: min(np.max(x), 100),
)

MODE_pHmap = dataModeClass(
    name="pHmap",
    data_attr="pHmap",
    option_txt="pH",
    label="pH ",
    cmap="jet_r",
    title="pH-Map [pH = xXx]",
)

MODE_multi_comp_pHmaps = dataModeClass(
    name="multi_comp_pHmaps",
    data_attr="_multi_comp_pHmaps",
    option_txt="pH (Multi-Comp)",
    label="pH ",
    cmap="jet_r",
    title="pH-Map (Multi-Comp) [pH = xXx]",
)

MODE_multi_comp_mean_pH = dataModeClass(
    name="multi_comp_mean_pH",
    data_attr="_multi_comp_mean_pH",
    option_txt="Mean pH (Multi-Comp)",
    label="pH ",
    cmap="jet_r",
    title="Mean pH-Map (Multi-Comp) [pH = xXx]",
)

MODE_multi_comp_peak_pH = dataModeClass(
    name="multi_comp_peak_pH",
    data_attr="_multi_comp_peak_pH",
    option_txt="Peak pH (Multi-Comp)",
    label="pH ",
    cmap="jet_r",
    title="Peak pH-Map (Multi-Comp) [pH = xXx]",
)
