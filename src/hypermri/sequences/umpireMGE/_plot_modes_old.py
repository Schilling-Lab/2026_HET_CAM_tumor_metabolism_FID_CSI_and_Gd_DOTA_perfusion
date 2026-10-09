from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class dataModeClass_uMGE_Plot_Old:
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
            self.name == "mean_pHmap_pHIDEAL"
            or self.name == "pHmap_pHIDEAL"
            or self.name == "pHmap"
            or self.name == "peak_pHmap_pHIDEAL"
        ):
            return 6.6
        elif self.name == "T2_star_map":
            return 0.0
        else:
            return self.min_func(self.data)

    @property
    def max(self):
        if (
            self.name == "mean_pHmap_pHIDEAL"
            or self.name == "pHmap_pHIDEAL"
            or self.name == "pHmap"
            or self.name == "peak_pHmap_pHIDEAL"
        ):
            return 7.8
        elif self.name == "T2_star_map":
            return 40.0
        else:
            return self.max_func(self.data)


MODE_magnitude = dataModeClass_uMGE_Plot_Old(
    name="magnitude",
    data_attr="magnitude",
    option_txt="Magnitude",
    label="Magnitude [SNR]",
    cmap="gray",
    title="Magnitude [xXx SNR]",
)

MODE_phase = dataModeClass_uMGE_Plot_Old(
    name="phase",
    data_attr="phase",
    option_txt="Phase",
    label="Phase [rad]",
    cmap="seismic",
    title="Phase [xXx rad]",
)

MODE_fmap = dataModeClass_uMGE_Plot_Old(
    name="fmap",
    data_attr="fmap",
    option_txt="Frequency",
    label="Frequency [Hz]",
    cmap="plasma",
    title="Frequency Map [xXx Hz]",
)

MODE_fmaps_pHIDEAL = dataModeClass_uMGE_Plot_Old(
    name="fmaps_pHIDEAL",
    data_attr="_fmaps_pHIDEAL",
    option_txt="Frequency (pHIDEAL)",
    label="Frequency [Hz]",
    cmap="plasma",
    title="Frequency Map (pHIDEAL) [xXx Hz]",
)

MODE_fmap_err = dataModeClass_uMGE_Plot_Old(
    name="fmap_err",
    data_attr="fmap_err",
    option_txt="Fit-Error (Absolute)",
    label="Error [Hz]",
    cmap="Reds",
    title="Fit-Error Map [xXx Hz]",
)

MODE_fmap_err_rel = dataModeClass_uMGE_Plot_Old(
    name="fmap_err_rel",
    data_attr="fmap_err_rel",
    option_txt="Fit-Error (Relative)",
    label="Error [%]",
    cmap="Reds",
    title="Fit-Error Map [xXx %]",
    max_func=lambda x: min(np.max(x), 100),
)

MODE_pHmap = dataModeClass_uMGE_Plot_Old(
    name="pHmap",
    data_attr="pHmap",
    option_txt="pH",
    label="pH ",
    cmap="jet_r",
    title="pH-Map [pH = xXx]",
)

MODE_pHmap_pHIDEAL = dataModeClass_uMGE_Plot_Old(
    name="pHmap_pHIDEAL",
    data_attr="_pHmaps_pHIDEAL",
    option_txt="pH (pHIDEAL)",
    label="pH ",
    cmap="jet_r",
    title="pH-Map (pHIDEAL) [pH = xXx]",
)

MODE_mean_pHmap_pHIDEAL = dataModeClass_uMGE_Plot_Old(
    name="mean_pHmap_pHIDEAL",
    data_attr="_mean_pH_pHIDEAL",
    option_txt="Mean pH (pHIDEAL)",
    label="pH ",
    cmap="jet_r",
    title="Mean pH-Map (pHIDEAL) [pH = xXx]",
)

MODE_peak_pHmap_pHIDEAL = dataModeClass_uMGE_Plot_Old(
    name="peak_pHmap_pHIDEAL",
    data_attr="_peak_pH_pHIDEAL",
    option_txt="Peak pH (pHIDEAL)",
    label="pH ",
    cmap="jet_r",
    title="Peak pH-Map (pHIDEAL) [pH = xXx]",
)

MODE_T2_star = dataModeClass_uMGE_Plot_Old(
    name="_T2_star_map",
    data_attr="T2_star",
    option_txt="Fitted T2*",
    label="T2* [ms]",
    cmap="winter",
    title="Fitted T2* in ms [T2* = xXx]",
)
