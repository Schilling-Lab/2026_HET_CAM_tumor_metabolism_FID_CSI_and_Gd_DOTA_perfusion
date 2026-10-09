"""
NSA Simulation module for optimizing echo times in spectral imaging.
"""

from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import (
    HTML,
    Accordion,
    Checkbox,
    Dropdown,
    FloatSlider,
    HBox,
    IntSlider,
    Label,
    Layout,
    Output,
    VBox,
)

from hypermri.analysis.spectroscopy.core import METABOLITE_CS_PPM
from hypermri.utils.physics import get_gmr


def get_freq_hz(
    name: str, b0: float, use_reeder_phantom: bool, center_name: str
) -> float:
    """
    Returns the frequency in Hz relative to the center frequency.

    If 'use_reeder_phantom' is True, it returns the exact Hz shifts
    specified in Reeder et al. 2007 (Fig 1b) for 3T, scaled to the current B0.
    """
    # Exact shifts from Reeder 2007 Paper (Fig 1b) at 3.0T
    # Relative to Lactate (0 Hz)
    reeder_offsets_3T = {
        "lactate": 0.0,
        "alanine": -210.0,
        "pyruvate": -602.0,
        "pyruvatehydrate": -242.0,  # The "Ester" peak
        "none": 0.0,
    }

    # 1. Determine the absolute frequency/offset of the target
    if use_reeder_phantom and name in reeder_offsets_3T:
        # Scale Hz from 3T to current B0
        # freq = val_at_3T * (Current_B0 / 3.0)
        abs_freq = reeder_offsets_3T[name] * (b0 / 3.0)

        # Determine center offset
        if center_name in reeder_offsets_3T:
            center_freq = reeder_offsets_3T[center_name] * (b0 / 3.0)
        else:
            center_freq = 0.0

        return abs_freq - center_freq

    else:
        # Use PPM logic
        larmor = get_gmr(nucleus="13c", unit="MHz_T") * b0

        val_ppm = METABOLITE_CS_PPM.get(name, 0.0)
        center_ppm = METABOLITE_CS_PPM.get(center_name, 0.0)

        # ppm to Hz: (val_ppm - center_ppm) * Larmor_MHz
        # Larmor is in MHz/T * T = MHz
        # ppm diff * MHz = Hz
        return (val_ppm - center_ppm) * larmor


def calculate_nsa(
    metab_slots: List[str],
    center_name: str,
    b0: float,
    echo_times: np.ndarray,
    t2_stars: List[float],
    couple_pyr: bool,
    use_reeder: bool,
) -> Tuple[np.ndarray, List[str], List[float]]:

    # Filter out "none"
    active_indices = [i for i, m in enumerate(metab_slots) if m != "none"]
    selected_names = [metab_slots[i] for i in active_indices]
    selected_t2s = [t2_stars[i] for i in active_indices]

    cols = []
    out_names = []
    out_freqs = []  # For display

    t = echo_times.reshape(-1, 1)  # (N_echoes, 1)

    # Reeder Coupling Weights
    r_main = 0.61
    r_ester = 0.39

    # Track if we already processed coupled pyruvate to avoid duplicates
    processed_coupled = False

    for i, name in enumerate(selected_names):
        # Calculate Decay
        t2 = selected_t2s[i]
        decay = 1.0 if t2 <= 0 else np.exp(-t / t2)

        # Logic for Coupled Pyruvate
        if couple_pyr and name == "pyruvate":
            if processed_coupled:
                continue  # Skip if we already added the coupled column

            # Freq A: Pyruvate Main
            f_a = get_freq_hz("pyruvate", b0, use_reeder, center_name)
            term_a = r_main * np.exp(1j * 2 * np.pi * f_a * t)

            # Freq B: Pyruvate Hydrate (Implicit)
            f_b = get_freq_hz("pyruvatehydrate", b0, use_reeder, center_name)
            term_b = r_ester * np.exp(1j * 2 * np.pi * f_b * t)

            cols.append(decay * (term_a + term_b))
            out_names.append("Pyruvate (Cpl)")
            out_freqs.append(f"Main: {f_a:.1f}, Hyd: {f_b:.1f}")
            processed_coupled = True

        elif couple_pyr and name == "pyruvatehydrate":
            # If coupling is on, 'hydrate' is absorbed into 'pyruvate'.
            # If 'pyruvate' wasn't selected but 'hydrate' was, we technically
            # shouldn't simulate the couple unless the user explicitly wants Pyruvate.
            # For simplicity, we skip standalone hydrate if coupling is ON
            # (assuming the user selects Pyruvate to get the couple).
            pass

        else:
            # Standard Independent Species
            freq = get_freq_hz(name, b0, use_reeder, center_name)
            osc = np.exp(1j * 2 * np.pi * freq * t)
            cols.append(decay * osc)
            out_names.append(name.capitalize())
            out_freqs.append(f"{freq:.1f}")

    if not cols:
        return np.zeros(0), [], []

    A = np.hstack(cols)

    # NSA Calculation
    AtA = np.dot(A.conj().T, A)
    try:
        cov = np.linalg.inv(AtA)
        nsa = 1.0 / np.real(np.diag(cov))
        nsa[nsa < 0] = 0
        nsa[nsa > 1e6] = 0
    except np.linalg.LinAlgError:
        nsa = np.zeros(len(out_names))

    return nsa, out_names, out_freqs


def interactive_nsa_simulation(**kwargs):
    """
    Run the NSA Simulation Dashboard.

    Parameters
    ----------
    kwargs : dict, optional
        Initial values for the widgets.
        - b0: float (default 7.0)
        - echoes: int (default 3)
        - spacing: float (default 1.2)
        - flyback: float (default 0.0)
        - bw: float (default 25)
        - pts: int (default 64)
        - m1, m2, m3, m4: str (default 'pyruvate', 'lactate', 'pyruvatehydrate', 'bicarbonate')
        - center: str (default 'lactate')
        - couple: bool (default True)
        - reeder: bool (default False)
        - t2_1, t2_2, t2_3, t2_4: float (default -1)
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams["font.family"] = "sans-serif"

    # --- Defaults ---
    defaults = {
        "b0": 7.0,
        "echoes": 3,
        "spacing": 1.2,
        "flyback": 0.0,
        "bw": 25,
        "pts": 64,
        "m1": "pyruvate",
        "m2": "lactate",
        "m3": "pyruvatehydrate",
        "m4": "bicarbonate",
        "center": "lactate",
        "couple": True,
        "reeder": False,
        "t2_1": -1,
        "t2_2": -1,
        "t2_3": -1,
        "t2_4": -1,
    }

    # Update defaults with kwargs
    for k, v in kwargs.items():
        if k in defaults:
            defaults[k] = v

    # --- Widgets ---
    style = {"description_width": "initial"}
    layout_full = Layout(width="98%")

    # 1. Main Params
    w_b0 = FloatSlider(
        min=1.5,
        max=11.7,
        step=0.1,
        value=defaults["b0"],
        description="B0 Field (T)",
        style=style,
        layout=layout_full,
    )
    w_echoes = IntSlider(
        min=3,
        max=16,
        value=defaults["echoes"],
        description="Num Echoes",
        style=style,
        layout=layout_full,
    )
    w_spacing = FloatSlider(
        min=0.1,
        max=5.0,
        step=0.005,
        value=defaults["spacing"],
        description="Echo Spacing (ms)",
        style=style,
        layout=layout_full,
    )
    w_flyback = FloatSlider(
        min=0.0,
        max=2.0,
        step=0.1,
        value=defaults["flyback"],
        description="Flyback (ms)",
        style=style,
        layout=layout_full,
    )
    w_bw = FloatSlider(
        min=10,
        max=125,
        step=1,
        value=defaults["bw"],
        description="Bandwidth (kHz)",
        style=style,
        layout=layout_full,
    )
    w_pts = IntSlider(
        min=16,
        max=512,
        step=16,
        value=defaults["pts"],
        description="Read Points",
        style=style,
        layout=layout_full,
    )

    # 2. Metabolites
    # Add 'none' to options
    base_opts = list(METABOLITE_CS_PPM.keys())
    # remove duplicate none if exists
    if "none" in base_opts:
        base_opts.remove("none")
    metab_opts = ["none"] + base_opts

    w_m1 = Dropdown(
        options=metab_opts,
        value=defaults["m1"],
        description="Met 1",
        style=style,
        layout=layout_full,
    )
    w_m2 = Dropdown(
        options=metab_opts,
        value=defaults["m2"],
        description="Met 2",
        style=style,
        layout=layout_full,
    )
    w_m3 = Dropdown(
        options=metab_opts,
        value=defaults["m3"],
        description="Met 3",
        style=style,
        layout=layout_full,
    )
    w_m4 = Dropdown(
        options=metab_opts,
        value=defaults["m4"],
        description="Met 4",
        style=style,
        layout=layout_full,
    )
    w_center = Dropdown(
        options=base_opts,
        value=defaults["center"],
        description="Center Freq",
        style=style,
        layout=layout_full,
    )

    w_couple = Checkbox(
        value=defaults["couple"],
        description="Couple Pyruvate (Sums peaks)",
        style=style,
    )
    w_reeder = Checkbox(
        value=defaults["reeder"],
        description="Use Reeder Phantom Freqs (Fig 1b)",
        style=style,
    )

    # T2 Sliders
    def t2_slide(label, default_val):
        return FloatSlider(
            min=-1,
            max=100,
            step=1,
            value=default_val,
            description=f"{label} T2* (ms)",
            style=style,
            layout=layout_full,
        )

    w_t2_1 = t2_slide("Met 1", defaults["t2_1"])
    w_t2_2 = t2_slide("Met 2", defaults["t2_2"])
    w_t2_3 = t2_slide("Met 3", defaults["t2_3"])
    w_t2_4 = t2_slide("Met 4", defaults["t2_4"])

    # Output Widget for Plot
    plot_out = Output()

    # Freq Info Widget
    freq_info = HTML(value="Initializing...")
    min_tr_info = HTML(value="Min TR: -")

    plt.ioff()
    fig = plt.figure(figsize=(12, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
    ax_curve = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])
    plt.ion()

    with plot_out:
        display(fig.canvas)

    def update(change=None):
        # Gather inputs
        b0 = w_b0.value
        n_echoes = w_echoes.value
        te_val = w_spacing.value
        flyback = w_flyback.value
        bw = w_bw.value
        pts = w_pts.value

        # Gather Species
        mslots = [w_m1.value, w_m2.value, w_m3.value, w_m4.value]
        tslots_ms = [w_t2_1.value, w_t2_2.value, w_t2_3.value, w_t2_4.value]
        tslots_s = [t / 1000.0 if t > 0 else -1 for t in tslots_ms]

        # Calculate Unavailable Time
        min_te = (pts / (bw * 1000.0)) * 1000.0
        dead_zone = min_te + flyback

        # Calculate Min TR
        # Acq time per echo = pts / bw (ms)
        # Total per echo = acq_time + flyback
        # Total TR = total_per_echo * n_echoes
        acq_time_ms = (pts / (bw * 1000.0)) * 1000.0
        min_tr_val = (acq_time_ms + flyback) * n_echoes
        min_tr_info.value = f"<b>Min Repetition Time:</b> {min_tr_val:.2f} ms"

        ax_curve.clear()
        ax_bar.clear()

        # --- 1. Sweep Calculation ---
        spacings = np.linspace(0.1, 5.0, 250)

        # Run once to get output names and freqs
        _, names, freqs_hz = calculate_nsa(
            mslots,
            w_center.value,
            b0,
            np.zeros(n_echoes),
            tslots_s,
            w_couple.value,
            w_reeder.value,
        )

        # Update Info Box
        info_html = "<b>Calculated Frequencies (Hz):</b><br><ul>"
        for n, f in zip(names, freqs_hz):
            info_html += f"<li><b>{n}:</b> {f} Hz</li>"
        info_html += "</ul>"
        freq_info.value = info_html

        if not names:
            ax_curve.text(0.5, 0.5, "No Metabolites Selected", ha="center")
            fig.canvas.draw_idle()
            return

        # Calc Matrix
        nsa_mat = np.zeros((len(spacings), len(names)))

        for i, dte in enumerate(spacings):
            te_train = np.arange(n_echoes) * (dte / 1000.0)
            val, _, _ = calculate_nsa(
                mslots,
                w_center.value,
                b0,
                te_train,
                tslots_s,
                w_couple.value,
                w_reeder.value,
            )
            nsa_mat[i, :] = val

        # Plot Curves
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
        ]  # Blue, Orange, Green, Red
        for k in range(len(names)):
            ax_curve.plot(
                spacings,
                nsa_mat[:, k],
                label=names[k],
                color=colors[k % 4],
                linewidth=2,
            )

        # Formatting Curve Plot
        ax_curve.axvspan(
            0, dead_zone, color="gray", alpha=0.15, hatch="//", label="Unavailable"
        )
        ax_curve.axvline(dead_zone, color="gray", linestyle="--")
        ax_curve.axvline(dead_zone - flyback, color="gray", linestyle="--")
        ax_curve.axvline(te_val, color="black", linewidth=1.5)

        ax_curve.set_title(
            f"NSA Efficiency (Reeder Freqs: {w_reeder.value})",
            fontsize=11,
            fontweight="bold",
        )
        ax_curve.set_xlabel("Echo Spacing $\Delta$TE (ms)")
        ax_curve.set_ylabel("Effective NSA")
        ax_curve.set_xlim(0, 5.0)
        ax_curve.set_ylim(0, n_echoes * 1.05)
        ax_curve.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.9)

        # --- 2. Bar Plot (Current Selection) ---
        curr_te = np.arange(n_echoes) * (te_val / 1000.0)
        curr_nsa, _, _ = calculate_nsa(
            mslots,
            w_center.value,
            b0,
            curr_te,
            tslots_s,
            w_couple.value,
            w_reeder.value,
        )

        bars = ax_bar.bar(names, curr_nsa, color=colors[: len(names)], alpha=0.85)
        ax_bar.axhline(n_echoes, color="red", linestyle="--", linewidth=1)

        # Annotations
        for bar in bars:
            h = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.05,
                f"{h:.2f}",
                ha="center",
                fontweight="bold",
            )

        if te_val < dead_zone:
            ax_bar.text(
                0.5,
                0.5,
                "TIMING INVALID",
                transform=ax_bar.transAxes,
                ha="center",
                color="white",
                fontweight="bold",
                bbox=dict(facecolor="red", alpha=0.8),
            )

        ax_bar.set_title(f"Performance at {te_val} ms", fontsize=11)
        ax_bar.set_ylim(0, n_echoes * 1.1)

        fig.canvas.draw_idle()

    # --- Layout Assembly ---
    input_widgets = [
        w_b0,
        w_echoes,
        w_spacing,
        w_flyback,
        w_bw,
        w_pts,
        w_m1,
        w_m2,
        w_m3,
        w_m4,
        w_center,
        w_couple,
        w_reeder,
        w_t2_1,
        w_t2_2,
        w_t2_3,
        w_t2_4,
    ]

    for w in input_widgets:
        w.observe(update, names="value")

    # Groups
    c_acq = VBox(
        [
            HTML("<b>Sequence</b>"),
            w_b0,
            w_echoes,
            w_spacing,
            HTML("<hr><b>Timing</b>"),
            w_bw,
            w_pts,
            w_flyback,
            HTML("<hr>"),
            min_tr_info,
        ],
        layout=Layout(padding="8px"),
    )

    c_met = VBox(
        [
            HTML("<b>Configuration</b>"),
            w_couple,
            w_reeder,
            w_center,
            HTML("<hr><b>Species Selection</b>"),
            w_m1,
            w_t2_1,
            w_m2,
            w_t2_2,
            w_m3,
            w_t2_3,
            w_m4,
            w_t2_4,
        ],
        layout=Layout(padding="8px"),
    )

    acc = Accordion(children=[c_acq, c_met])
    acc.set_title(0, "Acquisition Parameters")
    acc.set_title(1, "Metabolites & T2*")

    # Left Sidebar
    sidebar = VBox(
        [
            Label(
                "LSCSI Simulation", style={"font_weight": "bold", "font_size": "14px"}
            ),
            acc,
            HTML("<hr>"),
            freq_info,
        ],
        layout=Layout(width="320px", min_width="320px", margin="0 20px 0 0"),
    )

    ui = HBox([sidebar, plot_out])

    # Init
    update()
    display(ui)
