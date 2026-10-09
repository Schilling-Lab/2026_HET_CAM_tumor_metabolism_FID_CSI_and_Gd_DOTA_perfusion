"""
Template for creating interactive dashboards with ipywidgets and matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import (
    HTML,
    Checkbox,
    FloatSlider,
    HBox,
    Label,
    Layout,
    Output,
    VBox,
)


def run_template_dashboard(initial_amplitude=1.0, initial_freq=1.0):
    """
    A template function for an interactive dashboard.

    Demonstrates:
    - Left sidebar for controls.
    - Right side for matplotlib figure.
    - Efficient in-place figure updating (no flickering).
    - Separation of calculation and plotting logic.
    """

    # 1. Setup Layout & Style
    style = {"description_width": "initial"}
    layout_full = Layout(width="98%")

    # 2. Define Widgets
    w_amp = FloatSlider(
        min=0.1,
        max=5.0,
        step=0.1,
        value=initial_amplitude,
        description="Amplitude",
        style=style,
        layout=layout_full,
    )
    w_freq = FloatSlider(
        min=0.1,
        max=10.0,
        step=0.1,
        value=initial_freq,
        description="Frequency (Hz)",
        style=style,
        layout=layout_full,
    )
    w_phase = FloatSlider(
        min=0,
        max=2 * np.pi,
        step=0.1,
        value=0,
        description="Phase (rad)",
        style=style,
        layout=layout_full,
    )
    w_noise = Checkbox(value=False, description="Add Noise", style=style)

    # Information Output (optional)
    info_html = HTML(value="Initializing...")

    # 3. Initialize Figure
    # Use plt.ioff() to prevent double display in notebooks
    plt.ioff()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    plt.ion()

    # 4. Define Update Logic
    def update(change=None):
        # Gather inputs
        amp = w_amp.value
        freq = w_freq.value
        phase = w_phase.value
        add_noise = w_noise.value

        # Perform Calculations
        t = np.linspace(0, 1, 500)
        y = amp * np.sin(2 * np.pi * freq * t + phase)

        if add_noise:
            y += np.random.normal(0, 0.1, size=t.shape)

        # Update Info
        info_html.value = f"<b>Peak Value:</b> {np.max(np.abs(y)):.2f}"

        # Clear and Replot
        ax.clear()
        ax.plot(t, y, label="Signal", color="#1f77b4", linewidth=2)
        ax.set_title(f"Sine Wave: {freq} Hz")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-6, 6)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Redraw efficiently
        fig.canvas.draw_idle()

    # 5. Bind Observers
    widgets_list = [w_amp, w_freq, w_phase, w_noise]
    for w in widgets_list:
        w.observe(update, names="value")

    # 6. Assemble UI

    # Output widget for the figure
    plot_out = Output()
    with plot_out:
        display(fig.canvas)

    # Sidebar layout
    sidebar = VBox(
        [
            Label(
                "Dashboard Controls", style={"font_weight": "bold", "font_size": "14px"}
            ),
            HTML("<hr>"),
            w_amp,
            w_freq,
            w_phase,
            HTML("<br>"),
            w_noise,
            HTML("<hr>"),
            info_html,
        ],
        layout=Layout(width="300px", margin="0 20px 0 0"),
    )  # Fixed width sidebar

    # Main container
    ui = HBox([sidebar, plot_out])

    # 7. Initial Update & Display
    update()
    display(ui)
