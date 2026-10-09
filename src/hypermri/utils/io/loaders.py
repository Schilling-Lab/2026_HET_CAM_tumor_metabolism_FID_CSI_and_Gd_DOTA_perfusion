"""
Specialized data loaders.

Contains loaders for physiological data (PC-SAM) and scan timing metadata.
"""

import datetime
import os
from typing import Optional

import pandas as pd

from ..system.logging import init_default_logger

logger = init_default_logger(__name__)


def load_phys_data(pc_sam_txt_file: str) -> pd.DataFrame:
    """
    Load physiological measurement data from a text file.

    Parameters
    ----------
    pc_sam_txt_file : str
        The file path to the physiological data text file.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the physiological data with additional timestamp columns.
    """

    # Load the main physiological data.
    df = pd.read_csv(
        pc_sam_txt_file,
        encoding="ISO-8859-1",
        decimal=",",
        sep=", ",
        engine="python",
        skiprows=3,
    )
    # Read the header row (first row) for timing information.
    header = pd.read_csv(
        pc_sam_txt_file,
        sep=", ",
        engine="python",
        nrows=1,
        header=None,
        encoding="ISO-8859-1",
    )

    # Remove leading/trailing double quotes from column names.
    df.columns = df.columns.str.strip('"')

    # Convert relevant columns to numeric.
    df["Seconds"] = pd.to_numeric(df["Seconds"], errors="coerce")
    df["°C"] = pd.to_numeric(df["°C"], errors="coerce")
    df["Resp/m"] = pd.to_numeric(df["Resp/m"], errors="coerce")

    # Extract breathing start and end times from the header.
    breathing_start_date_str = header[3][0][1:-1]

    breathing_start_time_str = header[7][0][1:-2]
    breathing_start_str = breathing_start_date_str + " " + breathing_start_time_str
    breathing_start_obj = datetime.datetime.strptime(
        breathing_start_str, "%m/%d/%y %H:%M:%S"
    )
    breathing_end_time_str = header[9][0][1:-1]
    breathing_end_str = breathing_start_date_str + " " + breathing_end_time_str
    breathing_end_obj = datetime.datetime.strptime(
        breathing_end_str, "%m/%d/%y %H:%M:%S"
    )

    # Compute the Timestamp column by adding the 'Seconds' offset to the
    # breathing start.
    df["Timestamp"] = df["Seconds"].apply(
        lambda s: breathing_start_obj + pd.Timedelta(seconds=s)
    )
    # Compute Timestamp_end by subtracting the remaining seconds from the
    # breathing end time.
    largest_s = df["Seconds"].max()
    df["Timestamp_end"] = breathing_end_obj - pd.to_timedelta(
        (largest_s - df["Seconds"]), unit="s"
    )
    return df


def load_scan_timing(data_obj=None, scans=None):
    """
    Load scan metadata from provided data objects.

    Parameters
    ----------
    data_obj : object, optional
        A single data object containing scan metadata. This is used if ``scans`` is None.
    scans : list, optional
        A list of data objects representing individual scans.

    Returns
    -------
    scans_df : pandas.DataFrame
        DataFrame containing scan metadata.
    """

    def parse_duration(duration_str):
        import re

        duration_str = duration_str.strip("<>")
        pattern = r"(?P<hours>\d+)h(?P<minutes>\d+)m(?P<seconds>\d+)s(?P<millis>\d+)ms"
        match = re.match(pattern, duration_str)
        if not match:
            raise ValueError(f"Invalid duration format: {duration_str}")
        hours = int(match.group("hours"))
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        millis = int(match.group("millis"))
        return datetime.timedelta(
            hours=hours, minutes=minutes, seconds=seconds, milliseconds=millis
        )

    def process_data_obj(data_obj):
        if data_obj is not None:
            duration_str = data_obj.method["PVM_ScanTimeStr"]
            duration_obj = parse_duration(duration_str)
            raw_str = data_obj.reco["RECO_time"]
            raw_str = raw_str.strip("<>")
            raw_str = raw_str.replace(",", ".")
            raw_str = raw_str[:-5] + ":" + raw_str[-2:]
            measurement_end_time_obj = datetime.datetime.strptime(
                raw_str[:-7], "%Y-%m-%dT%H:%M:%S"
            )
            measurement_start_time_obj = measurement_end_time_obj - duration_obj
            name = data_obj.__str__()
            nucleus = data_obj.method["PVM_Nucleus1"]
            return {
                "name": name,
                "nucleus": nucleus,
                "measurement_start_time_obj": measurement_start_time_obj,
                "measurement_end_time_obj": measurement_end_time_obj,
            }

    def process_scans(scans):
        rows = []
        for data_obj in scans:
            try:
                row = process_data_obj(data_obj)
                rows.append(row)
            except BaseException:
                pass
        return pd.DataFrame(rows)

    if scans is None:
        scans_df = process_data_obj(data_obj=data_obj)
        scans_df = pd.DataFrame([scans_df])
    else:
        scans_df = process_scans(scans=scans)
    return scans_df


def load_pc_sam_temp_file(
    filepath: str, day: str, time_offset: float, plot: bool = False
) -> Optional[pd.DataFrame]:
    """Load PC-SAM temperature export into a dataframe.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Location of the PC-SAM .txt file to load.
    day : str
        Acquisition day in YYYY-MM-DD format used to build full timestamps.
    time_offset : float
        Offset in seconds between the PC-SAM and MRI system clocks.
    plot : bool, optional
        If True plot the temperature trace after loading.

    Returns
    -------
    pandas.DataFrame or None
        Dataframe with DateTime, Timestamp and Temperature columns.
    """
    try:
        temperature_dataframe = pd.read_csv(
            filepath, encoding="ISO-8859-1", decimal=",", sep=", ", engine="python"
        )
    except UnicodeDecodeError:
        logger.error(
            "UTF8 codec error when reading %s, try removing the degree sign in the txt file."
            % filepath
        )
        return None

    day = day + " "
    temperature_dataframe.columns = temperature_dataframe.iloc[1]
    temperature_dataframe = temperature_dataframe.drop([0, 1, 2])
    temperature_dataframe = temperature_dataframe.reset_index()

    # Process temperature data
    try:
        # Convert timestamps and apply time offset
        temperature_dataframe["DateTime"] = pd.to_datetime(
            day + temperature_dataframe.iloc[:, 1], format="%Y-%m-%d %H:%M:%S"
        )
        temperature_dataframe["Timestamp"] = (
            temperature_dataframe["DateTime"].astype(int) / 10**9
        )
        temperature_dataframe["Timestamp"] = (
            temperature_dataframe["Timestamp"] + time_offset
        )

        # Extract temperature values
        temp_column = temperature_dataframe.columns[2]  # Temperature column
        temperature_dataframe["Temperature"] = pd.to_numeric(
            temperature_dataframe[temp_column], errors="coerce"
        )

    except Exception as e:
        logger.error(f"Error processing temperature data: {e}")
        return None

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(
            temperature_dataframe["DateTime"], temperature_dataframe["Temperature"]
        )
        plt.xlabel("Time")
        plt.ylabel("Temperature (°C)")
        plt.title("PC-SAM Temperature Data")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return temperature_dataframe


def load_pc_sam_phys_file(pc_sam_txt_file: str = None, data_obj=None, scans=None):
    """
    Load both physiological data and scan metadata.

    Wrapper for load_phys_data and load_scan_timing.
    """
    df = load_phys_data(pc_sam_txt_file)
    scans_df = load_scan_timing(data_obj, scans)
    return df, scans_df


def plot_pc_sam_phys_data(pc_sam_txt_file=None, data_obj=None, scans=None, **kwargs):
    """
    Create an interactive plotting and export interface for physiological data.
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display
    from matplotlib.dates import num2date

    # Load physiological and scan data.
    df, scans_df = load_pc_sam_phys_file(
        pc_sam_txt_file=pc_sam_txt_file, data_obj=data_obj, scans=scans
    )

    if "savepath" in kwargs:
        savepath = kwargs["savepath"]
    else:
        savepath = "path/to/export"

    if "savename" in kwargs:
        savename = kwargs["savename"]
    else:
        savename = "export_name"

    # Ensure scan times are in datetime format.
    scans_df["measurement_start_time_obj"] = pd.to_datetime(
        scans_df["measurement_start_time_obj"]
    )
    scans_df["measurement_end_time_obj"] = pd.to_datetime(
        scans_df["measurement_end_time_obj"]
    )

    # Set default display time values from scan metadata.
    default_start = (
        scans_df["measurement_start_time_obj"].min().strftime("%Y-%m-%d %H:%M:%S")
    )
    default_end = (
        scans_df["measurement_start_time_obj"].max().strftime("%Y-%m-%d %H:%M:%S")
    )

    # --- Create Display Widgets ---
    nucleus_w = widgets.Dropdown(
        options=["All", "1H", "13C"], value="All", description="Nucleus:"
    )
    start_time_w = widgets.Text(
        value=default_start,
        description="Start Time:",
        layout=widgets.Layout(width="300px"),
    )
    end_time_w = widgets.Text(
        value=default_end, description="End Time:", layout=widgets.Layout(width="300px")
    )
    measurement_w = widgets.SelectMultiple(
        options=list(scans_df["name"].unique()),
        value=tuple(scans_df["name"].unique()),
        description="Measurements:",
        layout=widgets.Layout(width="500px", height="150px"),
    )
    y_min_w = widgets.FloatText(
        value=0,
        description="Breath Y min:",
        layout=widgets.Layout(width="150px"),
    )
    y_max_w = widgets.FloatText(
        value=100,
        description="Breath Y max:",
        layout=widgets.Layout(width="150px"),
    )
    t_min_w = widgets.FloatText(
        value=30,
        description="Temp Y min:",
        layout=widgets.Layout(width="150px"),
    )
    t_max_w = widgets.FloatText(
        value=40,
        description="Temp Y max:",
        layout=widgets.Layout(width="150px"),
    )
    display_mode_w = widgets.ToggleButtons(
        options=["Breathing", "Temperature", "Both"],
        value="Breathing",
        description="Display:",
    )

    # --- Create Export Widgets ---
    # Export mode: choose among exporting all (displayed), selected scans, or
    # a selected time range.
    export_mode_w = widgets.ToggleButtons(
        options=["Export All", "Export Selected Scans", "Export Selected Time Range"],
        value="Export All",
        description="Export Mode:",
    )
    # When exporting a selected time range, these fields are used.
    export_start_time_w = widgets.Text(
        value=default_start,
        description="Export Start:",
        layout=widgets.Layout(width="300px"),
    )
    export_end_time_w = widgets.Text(
        value=default_end,
        description="Export End:",
        layout=widgets.Layout(width="300px"),
    )
    export_path_w = widgets.Text(
        value=savepath,
        description="Export Folder:",
        placeholder=savepath,
        layout=widgets.Layout(width="300px"),
    )
    export_basename_w = widgets.Text(
        value=savename,
        description="Base Name:",
        placeholder=savename,
        layout=widgets.Layout(width="200px"),
    )

    export_button = widgets.Button(description="Export Data", button_style="success")
    save_button = widgets.Button(description="Save Plot", button_style="info")

    # Global variables to hold the current figure and the data to be exported.
    current_fig = None
    exported_data = None

    # --- Main Plot Callback ---
    def interactive_plot(
        nucleus_filter,
        start_time,
        end_time,
        selected_names,
        y_min,
        y_max,
        display_mode,
        t_min,
        t_max,
    ):
        nonlocal current_fig
        try:
            # Parse the displayed time range.
            t0 = pd.to_datetime(start_time)
            t1 = pd.to_datetime(end_time)
        except Exception as e:
            print("Error parsing times:", e)
            return

        # Filter scans for plotting based on display time and nucleus.
        filtered_scans = scans_df.copy()
        if nucleus_filter != "All":
            filtered_scans = filtered_scans[
                filtered_scans["nucleus"].str.contains(nucleus_filter, case=False)
            ]
        filtered_scans = filtered_scans[
            (filtered_scans["measurement_start_time_obj"] >= t0)
            & (filtered_scans["measurement_start_time_obj"] <= t1)
        ]
        new_options = list(filtered_scans["name"].unique())
        measurement_w.options = new_options
        if selected_names:
            filtered_scans = filtered_scans[filtered_scans["name"].isin(selected_names)]

        fig, ax = plt.subplots(figsize=(10, 6))
        if display_mode == "Breathing":
            ax.plot(df["Timestamp_end"], df["Resp/m"], label="Resp/m", color="blue")
            ax.set_ylim(y_min, y_max)
        elif display_mode == "Temperature":
            ax.plot(
                df["Timestamp_end"], df["°C"], label="Temperature (°C)", color="orange"
            )
            ax.set_ylim(t_min, t_max)
        elif display_mode == "Both":
            ax.plot(df["Timestamp_end"], df["Resp/m"], label="Resp/m", color="blue")
            ax.set_ylim(y_min, y_max)
            ax2 = ax.twinx()
            ax2.plot(
                df["Timestamp_end"], df["°C"], label="Temperature (°C)", color="orange"
            )
            ax2.set_ylim(t_min, t_max)
            ax2.set_ylabel("Temperature (°C)")
        ax.set_xlim(t0, t1)
        ax.set_xlabel("Time")
        ax.set_ylabel("Respiration [1/min]")
        ax.set_title("Physiological Data")
        ax.legend(loc="upper left")
        if display_mode == "Both":
            ax2.legend(loc="upper right")
        # Plot scan windows.
        for idx, row in filtered_scans.iterrows():
            scan_start = row["measurement_start_time_obj"]
            scan_end = row["measurement_end_time_obj"]
            color = "blue" if row["nucleus"] == "13C" else "red"
            ax.axvspan(scan_start, scan_end, color=color, alpha=0.3)
            center = scan_start + (scan_end - scan_start) / 2
            ypos = y_min + 0.05 * (y_max - y_min)
            ax.text(
                center,
                ypos,
                row["name"],
                rotation=90,
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=8,
                color="black",
            )
        plt.tight_layout()
        current_fig = fig

        # --- Add Drag-and-Drop for Export Time Range ---
        # This callback is active only when the export mode is "Export Selected
        # Time Range".
        click_start = {"x": None}
        # Use a mutable container to store the current range indicator.
        range_indicator = {"span": None}

        def on_press(event):
            if event.inaxes != ax:
                return
            click_start["x"] = event.xdata

        def on_release(event):
            if event.inaxes != ax or click_start["x"] is None:
                return
            x0 = click_start["x"]
            x1 = event.xdata
            # Convert x-axis positions to datetime.
            dt0 = num2date(min(x0, x1)).replace(tzinfo=None)
            dt1 = num2date(max(x0, x1)).replace(tzinfo=None)
            # If export mode is "Export Selected Time Range", update the export
            # time widgets.
            if export_mode_w.value == "Export Selected Time Range":
                export_start_time_w.value = dt0.strftime("%Y-%m-%d %H:%M:%S")
                export_end_time_w.value = dt1.strftime("%Y-%m-%d %H:%M:%S")
                # Remove the old range indicator if it exists.
                if range_indicator["span"] is not None:
                    range_indicator["span"].remove()
                # Draw a new range indicator and store it.
                range_indicator["span"] = ax.axvspan(dt0, dt1, color="gray", alpha=0.2)
                fig.canvas.draw()
            click_start["x"] = None

        # Connect the mouse event handlers.
        fig.canvas.mpl_connect("button_press_event", on_press)
        fig.canvas.mpl_connect("button_release_event", on_release)

        plt.show()

    out = widgets.interactive_output(
        interactive_plot,
        {
            "nucleus_filter": nucleus_w,
            "start_time": start_time_w,
            "end_time": end_time_w,
            "selected_names": measurement_w,
            "y_min": y_min_w,
            "y_max": y_max_w,
            "display_mode": display_mode_w,
            "t_min": t_min_w,
            "t_max": t_max_w,
        },
    )

    # --- Export Data Callback ---
    def export_data_callback(b):
        nonlocal exported_data
        # --- Determine export mode and corresponding time range ---
        if export_mode_w.value == "Export All":
            try:
                t0 = pd.to_datetime(start_time_w.value)
                t1 = pd.to_datetime(end_time_w.value)
            except Exception as e:
                print("Error parsing display times for export:", e)
                return
            # Export all data within the displayed time range.
            exported_data = df[
                (df["Timestamp_end"] >= t0) & (df["Timestamp_end"] <= t1)
            ].copy()
            exported_data["scan_name"] = "no_scan"

        elif export_mode_w.value == "Export Selected Scans":
            try:
                t0 = pd.to_datetime(start_time_w.value)
                t1 = pd.to_datetime(end_time_w.value)
            except Exception as e:
                print("Error parsing display times for export:", e)
                return
            filtered_scans = scans_df.copy()
            if nucleus_w.value != "All":
                filtered_scans = filtered_scans[
                    filtered_scans["nucleus"].str.contains(nucleus_w.value, case=False)
                ]
            filtered_scans = filtered_scans[
                (filtered_scans["measurement_start_time_obj"] >= t0)
                & (filtered_scans["measurement_start_time_obj"] <= t1)
            ]
            new_options = list(filtered_scans["name"].unique())
            measurement_w.options = new_options
            if measurement_w.value:
                filtered_scans = filtered_scans[
                    filtered_scans["name"].isin(measurement_w.value)
                ]
            export_list = []
            if not filtered_scans.empty:
                for _, scan_row in filtered_scans.iterrows():
                    mask = (
                        df["Timestamp_end"] >= scan_row["measurement_start_time_obj"]
                    ) & (df["Timestamp_end"] <= scan_row["measurement_end_time_obj"])
                    sub_df = df[mask].copy()
                    sub_df["scan_name"] = scan_row["name"]
                    export_list.append(sub_df)
                if export_list:
                    exported_data = (
                        pd.concat(export_list).drop_duplicates().reset_index(drop=True)
                    )
                else:
                    exported_data = df[
                        (df["Timestamp_end"] >= t0) & (df["Timestamp_end"] <= t1)
                    ]
            else:
                exported_data = df[
                    (df["Timestamp_end"] >= t0) & (df["Timestamp_end"] <= t1)
                ]

        elif export_mode_w.value == "Export Selected Time Range":
            try:
                t0 = pd.to_datetime(export_start_time_w.value)
                t1 = pd.to_datetime(export_end_time_w.value)
            except Exception as e:
                print("Error parsing export time range:", e)
                return
            exported_data = df[
                (df["Timestamp_end"] >= t0) & (df["Timestamp_end"] <= t1)
            ].copy()
            # Initialize scan_name to "no_scan"
            exported_data["scan_name"] = "no_scan"
            # Now update rows that fall within any scan’s measurement window.
            for _, scan_row in scans_df.iterrows():
                mask = (
                    exported_data["Timestamp_end"]
                    >= scan_row["measurement_start_time_obj"]
                ) & (
                    exported_data["Timestamp_end"]
                    <= scan_row["measurement_end_time_obj"]
                )
                exported_data.loc[mask, "scan_name"] = scan_row["name"]

        # --- Save the exported data ---
        if export_path_w.value.strip() and export_basename_w.value.strip():
            save_dir = export_path_w.value.strip()
            base_name = export_basename_w.value.strip()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            export_filename = os.path.join(save_dir, f"{base_name}_physiological.csv")
            exported_data.to_csv(export_filename, index=False)
            print(f"Exported data saved to {export_filename}")
        else:
            print(
                "Export path or base name not provided. Exported data (first 5 rows):"
            )
            display(exported_data.head())

    export_button.on_click(export_data_callback)

    # --- Save Plot Callback ---
    def save_plot_callback(b):
        nonlocal current_fig
        if current_fig is not None:
            if export_path_w.value.strip() and export_basename_w.value.strip():
                save_dir = export_path_w.value.strip()
                base_name = export_basename_w.value.strip()
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plot_filename = os.path.join(save_dir, f"{base_name}_plot.png")
            else:
                plot_filename = "export_plot.png"
            current_fig.savefig(plot_filename, bbox_inches="tight")
            print(f"Plot saved to {plot_filename}")
        else:
            print("No figure to save yet.")

    save_button.on_click(save_plot_callback)

    # --- Layout the UI ---
    top_row = widgets.HBox([nucleus_w, start_time_w, end_time_w, display_mode_w])
    second_row = widgets.HBox([y_min_w, y_max_w, t_min_w, t_max_w])
    third_row = measurement_w

    export_ui_row1 = widgets.HBox([export_mode_w, export_path_w, export_basename_w])
    export_ui_row2 = widgets.HBox([export_start_time_w, export_end_time_w])
    export_ui = widgets.VBox([export_ui_row1, export_ui_row2])

    button_row = widgets.HBox([export_button, save_button])
    ui = widgets.VBox([top_row, second_row, third_row, export_ui, button_row])
    display(ui, out)
