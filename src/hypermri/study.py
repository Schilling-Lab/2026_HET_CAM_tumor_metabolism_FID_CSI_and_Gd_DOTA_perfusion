import json
import sys
import time
import traceback
import warnings
from collections.abc import Mapping
from datetime import datetime
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from tqdm.auto import tqdm

from .core.experiment import MRIExperiment
from .mixins import StandardizedPathMixin
from .utils.system.logging import LOG_MODES, init_default_logger, set_hypermri_log_level
from .vendors import detect_vendor, get_vendor_registry

logger = init_default_logger(__name__)


class StudyError(Exception):
    pass


class Study(Mapping, StandardizedPathMixin):
    """Dictionary-like class to handle scan collections from given directory.

    Parameters
    ----------
    path : string
        Path to the folder containing all scans of the study to be loaded in.

    keywords : string or tuple or list, optional
        Add additional columns to the internal dataframe. See
        Study.add_column and Study.add_columns for more information.
         - 'keyword' is string or tuple --> Study.add_column
         - 'keyword' is list --> Study.add_columns
        Default is None.

    sort : str, optional
        Sorting method for scan objects. Options are:
        - "time": Sort by timestamp (default)
        - "e_number": Sort by experiment number
        - "sequence_class": Sort by sequence class name
        - "scan_name": Sort by scan display name
        - None: No sorting (keep original order)

    verbose : bool, optional
        Wether or not Study.display is called to generate a nice scan over-
        view. Default is True.

    smart_load : bool, optional
        When set to true scans are not read in as MRIExperiment classes but as
        custom sequence classes whereever possible. The configuration which scan
        is loaded in as which class is specified in the config file:
        'config/SmartLoadMeasurements.json'. Default is True.

    load_data : bool or str, optional
        Default is "minimal". Options:
        - True: Full data loading (original behavior) - loads all raw data
        - False: Only parameter files (requires smart_load=False) - fastest, metadata only
        - "minimal": Load metadata + seq2d only (no raw data loading) - balanced approach
        - "lazy": Parameters only, data loaded on-demand - truly lazy loading

    log_mode : string
        Default is set to 'error'. Options are 'critical', 'error', 'warning', 'info',
        'debug', or 'notset'.
    """

    def __init__(
        self,
        path,
        keywords=None,
        verbose=True,
        smart_load=True,
        load_data="minimal",  # Changed default to minimal loading
        log_mode="error",
        colorize=True,
        sort="time",
    ):
        """
        Initialize the Study object.

        Parameters
        ----------
        path : str or Path
            Path to the study directory containing scans.
        keywords : str, tuple, list, optional
            Keywords for additional columns in the DataFrame.
        verbose : bool, optional
            Whether to display the DataFrame after loading. Default is True.
        smart_load : bool, optional
            Whether to use smart loading (only parameters) of scans. Default is True.
        load_data : bool or str, optional
            How to load data:
            - True: Load all data (default)
            - False: Load only parameters (requires smart_load=False)
            - "minimal": Load parameters + seq2d only (no raw data)
            - "lazy": Load parameters only, data loaded on-demand
        log_mode : str, optional
            Logging mode for the Study. Default is 'error'.
            Options are: 'critical', 'error', 'warning', 'info', 'debug', or 'notset'.
        colorize : bool, optional
            Whether to colorize the DataFrame display. Default is True.
        sort : str, optional
            Sorting method for the DataFrame. Options are:
            - "time": Sort by timestamp (default)
            - "e_number": Sort by experiment number
            - "sequence_class": Sort by sequence class name
            - "scan_name": Sort by scan display name
            - None: No sorting (keep original order)

        Returns
        -------
        None

        Raises
        ------
        StudyError
            If the provided path is not a valid directory or if the load_data
            parameter is invalid during smart loading.

        Notes
        -----
        This class initializes a study by reading in all scans from the specified path.
        The vendor-agnostic approach allows for flexible handling of scan data.
        """

        # Initialize mixin first
        super().__init__()

        start_time = time.time()

        self.log_mode = log_mode  # Store log_mode as instance variable

        logger.setLevel(LOG_MODES[log_mode])
        # Set log level for all hypermri loggers
        set_hypermri_log_level(log_mode)

        # Use standardized path handling
        self.path = path

        if smart_load:
            assert load_data in [
                True,
                "minimal",
                "lazy",
            ], "load_data must be True, 'minimal', or 'lazy' during smart load!"
            # preliminary read in of scans, we want the param files only
            prelim_start = time.time()
            # print(f"[TIMING] Starting preliminary parameter loading (load_data=False)")
            self.__scans = self.__ReadInMeasurements(load_data=False)
            time.time() - prelim_start
            # print(f"[TIMING] Preliminary parameter loading took {prelim_time:.3f}s")

            # do a smart reload of the scans
            smart_start = time.time()
            # print(f"[TIMING] Starting SmartLoadMeasurements with load_data={load_data}")
            self.__SmartLoadMeasurements(load_data=load_data)
            time.time() - smart_start
            # print(f"[TIMING] SmartLoadMeasurements took {smart_time:.3f}s")
        else:
            # read in all scans as MRIExperiment objects
            basic_start = time.time()
            # print(f"[TIMING] Starting basic measurements loading (smart_load=False)")
            self.__scans = self.__ReadInMeasurements(load_data)
            time.time() - basic_start
            # print(f"[TIMING] Basic measurements loading took {basic_time:.3f}s")

        self.new_column_from_scanobj(
            "Reconstructions", lambda x: x.recon_names if x.recon_names else "MISSING"
        )
        self.new_column_from_scanobj(
            "DICOM",
            lambda x: "Yes" if getattr(x, "has_dicom_reco", False) else "No",
        )
        self.new_column_from_scanobj("Scan Name", lambda x: x.display_name)

        self.date = self.__GetSubjectDate()
        self.subject = {}
        self.subject["subject_name"] = self.__GetSubjectName()
        self.subject["subject_dob"] = self.__GetSubjectDOB()
        self.subject["subject_weight"] = self.__GetSubjectWeight()
        self.subject["subject_type"] = self.__GetSubjectType()
        self.subject["rf_coils"] = self.__GetUsedCoils()

        if isinstance(keywords, (str, tuple)):
            self.add_column(keywords)

        if isinstance(keywords, list):
            self.add_columns(keywords)

        if sort:
            if sort == "time":
                # Sort by timestamp using the __lt__ and __gt__ attributes of the MRIExperiment
                self.__scans.sort_values(axis=0, by="ScanObject", inplace=True)
            elif sort == "e_number":
                # Sort by experiment number (index)
                self.__scans.sort_index(inplace=True)
            elif sort == "sequence_class":
                # Sort by sequence class name if available
                if "SequenceClass" in self.__scans.columns:
                    self.__scans.sort_values(axis=0, by="SequenceClass", inplace=True)
            elif sort == "scan_name":
                # Sort by scan display name
                if "Scan Name" in self.__scans.columns:
                    self.__scans.sort_values(axis=0, by="Scan Name", inplace=True)

        if verbose:
            display_start = time.time()
            self.display(colorize=colorize)
            time.time() - display_start
            # print(f"[TIMING] Display took {display_time:.3f}s")

        time.time() - start_time
        # print(f"[TIMING] Total Study initialization took {total_time:.3f}s")

    def __getitem__(self, key):
        if key in self.__scans.index:
            return self.__scans.loc[(key, "ScanObject")]
        else:
            raise StudyError(f"{key} is not a valid key.")

    def __iter__(self):
        return iter(self.__scans["ScanObject"])

    def __len__(self):
        return len(self.__scans)

    def __str__(self):
        return str(self.__scans)

    @property
    def dataframe(self):
        return self.__scans

    @dataframe.setter
    def dataframe(self, df):
        if isinstance(df, pd.DataFrame):
            self.__scans = df
        else:
            raise StudyError("Study.dataframe must be a pandas DataFrame object.")

    def colorize_col(self, col, df_styler=None, opacity=0.5):
        assert opacity <= 1 and opacity >= 0, "Opacity must be in range [0, 1]"

        if df_styler is None:
            df_styler = self.__scans.style

        # if the column item is not hashable, abort
        try:
            values = set(self.__scans[col].values)
        except BaseException:
            return

        # Remove 'MISSING' from values for color assignment
        values = values - {"MISSING"}
        if not values:
            return df_styler

        # Create rainbow colormap with consistent ordering

        try:
            from matplotlib.pyplot import get_cmap
        except ImportError:
            from matplotlib.cm import get_cmap

        # Sort values to ensure consistent color assignment regardless of DataFrame order
        # Use a predefined order for common sequence types to create a logical color progression
        sequence_priority = {
            "RARE": 0,  # Anatomical - violet/blue
            "FLASH": 1,  # Anatomical - blue
            "PRESS": 2,  # Spectroscopy - green/cyan
            "CSI": 3,  # Spectroscopy - yellow/orange
            "Fieldmap": 4,  # Utility - orange
            "Singlepulse": 5,  # Utility - red
        }

        # Sort values using priority order, with alphabetical fallback for unknown sequences
        def get_sort_key(value):
            return (sequence_priority.get(value, 1000), value)

        sorted_values = sorted(values, key=get_sort_key)
        n_colors = len(sorted_values)

        if n_colors == 0:
            return df_styler

        # Generate rainbow colors spanning full spectrum (violet to red)
        rainbow = get_cmap("rainbow")
        # Always use full rainbow range (0.0 to 1.0) regardless of number of colors
        if n_colors == 1:
            colors = [rainbow(0.5)]  # Use middle color for single item
        else:
            # Distribute colors evenly across the full rainbow spectrum
            colors = [rainbow(i / (n_colors - 1)) for i in range(n_colors)]

        # Convert to hex with opacity
        hex_colors = []
        for color in colors:
            # Convert RGBA to hex with opacity
            r, g, b, _ = color
            hex_color = "#{:02x}{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255), int(opacity * 255)
            )
            hex_colors.append(hex_color)

        # generate color look up table with consistent ordering
        color_lut = {k: v for k, v in zip(sorted_values, hex_colors)}
        color_lut["MISSING"] = ""

        def f(v):
            return f"background-color: {color_lut.get(v, '')};"

        try:
            return df_styler.map(f, subset=[col])
        except BaseException:
            return df_styler.applymap(f, subset=[col])

    def remove(self, data_obj=None):
        """
        Removes a scan with number scan_num from the object

        Parameters
        ----------
        scan: int
            Experiment number of the scan to be removed

        Returns
        -------
        None
        """
        if data_obj is None:
            return None

        assert isinstance(data_obj, int), "scan_num has to be an integer"

        # check if desired scan for removal is in Study
        if data_obj in self.__scans.index:
            # remove scan from scans:
            self.__scans.drop(index=data_obj, inplace=True)
        else:
            warnings.warn(f"scan {data_obj} does not exist in scans.")

    def display(self, colorize=True):
        """Display all scan-objects using pandas style functionalities."""
        styles = [
            # tabular headers
            {"selector": "th", "props": [("font-size", "109%")]},
            {"selector": "td", "props": [("font-size", "107%")]},  # tabular data
        ]

        # Reset index to make "Key" a regular column, then hide the new numeric index
        df_styler = (
            self.__scans.reset_index().style.set_table_styles(styles).hide(axis="index")
        )

        if self.date:
            date_string = self.date.strftime("%A %d. %B %Y  %H:%M:%S")

            # Add subject information if available
            subject_info = []
            if self.subject.get("subject_name"):
                subject_info.append(f"Subject: {self.subject['subject_name']}")
            if self.subject.get("subject_type"):
                subject_info.append(f"Type: {self.subject['subject_type']}")
            if self.subject.get("subject_weight"):
                subject_info.append(f"Weight: {self.subject['subject_weight']} g")
            if self.subject.get("subject_dob"):
                subject_info.append(f"DOB: {self.subject['subject_dob']}")

            caption = date_string
            if subject_info:
                caption += "<br>" + " | ".join(subject_info)

            df_styler = df_styler.set_caption(caption)

        if colorize:
            # skip first three columns, try to colorize all others
            for col in self.__scans.columns[3:]:
                self.colorize_col(col, df_styler, opacity=0.3)

        def color_MISSING(cell_value):
            # cell_value == "MISSING" raises a warning if cell_value is an
            # array
            if isinstance(cell_value, str) and cell_value == "MISSING":
                return "color: red"
            return ""

        try:
            df_styler = df_styler.map(color_MISSING)
        except BaseException:
            df_styler = df_styler.applymap(color_MISSING)

        # Format ScanObject to show visible class name instead of hidden <Object> repr
        df_styler = df_styler.format({"ScanObject": lambda x: x.__class__.__name__})

        # Hide ScanObject column as it is redundant with SequenceClass
        try:
            df_styler.hide(axis="columns", subset=["ScanObject"])
        except AttributeError:
            # Fallback for older pandas versions
            df_styler.hide_columns(["ScanObject"])

        display(df_styler)

    def get_available_keys(self):
        keys = []
        for key in self.__scans.index:
            keys.append(key)
        return keys

    def new_column_from_scanobj(self, colname, func):
        self.__scans[colname] = self.__scans["ScanObject"].apply(func)

    def add_column(self, keyword, paramfile="method", column_name=None):
        """Adds column with vendor-agnostic keyword extraction from scan objects.

        This method tries to extract a parameter using a vendor-agnostic approach.
        It can handle multiple formats and provides fallback mechanisms across vendors.

        Parameters
        ----------
        keyword: string, tuple, dict, or callable
            The parameter to extract. Supports multiple formats:

            - string: Simple key lookup (e.g., "TR", "TE")
            - tuple: (key, index) for array parameters (e.g., ("EchoTime", 0))
            - dict: Vendor-specific mapping (e.g., {"bruker": "PVM_RepetitionTime", "siemens": "TR"})
            - callable: Custom function that takes scan_obj and returns value

        paramfile: string, list, or dict, optional
            Parameter file(s) to search in. Supports:
            - string: Single paramfile (e.g., "method")
            - list: Multiple paramfiles to search in order (e.g., ["method", "acqp"])
            - dict: Vendor-specific paramfiles (e.g., {"bruker": "method", "siemens": "header"})
            Default is "method".

        column_name: string, optional
            Name for the resulting column. If None, uses the keyword as column name.

        Examples
        --------
        # Simple keyword
        study.add_column("TR")

        # Vendor-specific mapping
        study.add_column({"bruker": "PVM_RepetitionTime", "siemens": "TR"}, column_name="TR")

        # Multiple paramfiles
        study.add_column("EchoTime", paramfile=["method", "acqp"])

        # Custom function
        study.add_column(lambda scan: scan.get_vendor_parameter("TR"), column_name="TR")
        """

        def get_value(scan_obj):
            """Vendor-agnostic parameter extraction"""
            # Handle callable keyword
            if callable(keyword):
                try:
                    return keyword(scan_obj)
                except Exception:
                    return "MISSING"

            # 1. Try direct attribute access (for standardized parameters like 'repetition_time_ms')
            if isinstance(keyword, str) and hasattr(scan_obj, keyword):
                val = getattr(scan_obj, keyword)
                if val is not None:
                    return val

            # Determine vendor if available
            vendor = getattr(scan_obj, "vendor", None) or getattr(
                scan_obj, "handler_name", "unknown"
            )
            if hasattr(vendor, "__class__"):
                vendor = vendor.__class__.__name__.lower().replace("handler", "")

            # Handle vendor-specific keyword mapping
            if isinstance(keyword, dict):
                actual_keyword = keyword.get(
                    vendor, keyword.get("default", list(keyword.values())[0])
                )
            else:
                actual_keyword = keyword

            # Handle vendor-specific paramfile mapping
            if isinstance(paramfile, dict):
                actual_paramfiles = [
                    paramfile.get(vendor, paramfile.get("default", "method"))
                ]
            elif isinstance(paramfile, list):
                actual_paramfiles = paramfile
            else:
                actual_paramfiles = [paramfile]

            # Try to extract value from paramfiles in order
            for pfile in actual_paramfiles:
                if not hasattr(scan_obj, pfile):
                    continue

                lookup = getattr(scan_obj, pfile)
                if lookup is None:
                    continue

                try:
                    if isinstance(actual_keyword, tuple):
                        kw, idx = actual_keyword
                        if kw in lookup and hasattr(lookup[kw], "__getitem__"):
                            return lookup[kw][idx]
                    else:
                        if actual_keyword in lookup:
                            return lookup[actual_keyword]
                except (KeyError, IndexError, TypeError):
                    continue

            return "MISSING"

        # Determine column name
        if column_name is None:
            if isinstance(keyword, str):
                column_name = keyword
            elif isinstance(keyword, dict):
                column_name = keyword.get("column_name", "Parameter")
            else:
                column_name = "Parameter"

        self.new_column_from_scanobj(colname=column_name, func=get_value)

    def add_columns(self, keyword_list, paramfile="method", column_names=None):
        """A vendor-agnostic version of add_column that takes in a list of keywords.

        Parameters
        ----------
        keyword_list: list
            List of keywords in any supported format (string, tuple, dict, callable)
        paramfile: string, list, or dict, optional
            Parameter file(s) to search in (same as add_column)
        column_names: list, optional
            List of column names corresponding to keyword_list. If None, auto-generated.
        """
        if not isinstance(keyword_list, list):
            raise StudyError("keyword_list must be of type 'list'.")

        for i, keyword in enumerate(keyword_list):
            column_name = None
            if column_names and i < len(column_names):
                column_name = column_names[i]
            self.add_column(keyword, paramfile, column_name)

    def __GetSubjectName(self):
        """Retrieve the SubjectName of our bruker directory."""
        subject_fname = self.path / "subject"
        if subject_fname.is_file():
            with open(subject_fname, "r") as f:
                for line in f:
                    if line.startswith("##$SUBJECT_name_string"):
                        # Read the next line
                        subject_name = next(f).strip()[1:-1]
                        return subject_name
        return None

    def __GetSubjectDOB(self):
        """Retrieve the Subject's date of birth our bruker directory."""
        subject_fname = self.path / "subject"
        if subject_fname.is_file():
            with open(subject_fname, "r") as f:
                for line in f:
                    if line.startswith("##$SUBJECT_dbirth"):
                        # Read the next line
                        subject_dob = next(f).strip()[1:-1]
                        return subject_dob
        return None

    def __GetSubjectWeight(self):
        """Retrieve the Subject's weight of our bruker directory."""
        subject_fname = self.path / "subject"
        if subject_fname.is_file():
            with open(subject_fname, "r") as f:
                for line in f:
                    if line.startswith("##$SUBJECT_weight"):
                        # Read the next line
                        subject_weight = line.split("=")[1].strip()
                        # convert to gram:
                        subject_weight = 1000 * float(subject_weight)
                        return subject_weight
        return None

    def __GetSubjectDate(self):
        """Retrieve the SubjectDate of our bruker directory."""
        subject_fname = self.path / "subject"
        if subject_fname.is_file():
            with open(subject_fname, "r") as f:
                for line in f:
                    if line.startswith("##$SUBJECT_date"):
                        date = line.split("=")[1].strip()
            return datetime.strptime(date, "<%Y-%m-%dT%H:%M:%S,%f%z>")
        return None

    def __GetSubjectType(self):
        """Retrieve the Subject's date of birth of our bruker directory."""
        subject_fname = self.path / "subject"
        if subject_fname.is_file():
            with open(subject_fname, "r") as f:
                for line in f:
                    if line.startswith("##$SUBJECT_type"):
                        # Read the next line
                        subject_weight = line.split("=")[1].strip()
                        return subject_weight
        return None

    def __GetUsedCoils(self):
        """Aggregate unique receive/transmit coils from all scans."""
        unique_coils = set()

        if self.__scans is None or self.__scans.empty:
            return None

        for scan in self.__scans["ScanObject"]:
            # Get receive coils
            rx = getattr(scan, "coil_receive", None)
            if rx:
                if isinstance(rx, (list, np.ndarray)):
                    for c in rx:
                        unique_coils.add(str(c))
                else:
                    unique_coils.add(str(rx))

            # Get transmit coils
            tx = getattr(scan, "coil_transmit", None)
            if tx:
                if isinstance(tx, (list, np.ndarray)):
                    for c in tx:
                        unique_coils.add(str(c))
                else:
                    unique_coils.add(str(tx))

        # Filter out empty strings or None-like strings
        unique_coils = {c for c in unique_coils if c and c != "None" and c != "MISSING"}

        if not unique_coils:
            return None

        return ", ".join(sorted(list(unique_coils)))

    def __ReadInMeasurements(self, load_data=True, db=False) -> pd.DataFrame:
        """Collect all scan folders and read each scan in as MRIExperiment object.

        Parameters
        ----------
        load_data: bool
            When set to False only loads the parameter files, not the data files.
            Default is True, which loads in everything.

        Returns
        -------
        out: pandas.DataFrame
            Returns a DataFrame containing all scans. The scans are saved in the
            row 'ScanObject' with their corresponding E-Number as index.
        """
        # import scan-dir-validation method from MRIExperiment class
        # is_valid_scan_dir = BrukerHandler.is_valid_scan_dir

        # find all folders in given directory that are a valid scan dir
        dir_scan_start = time.time()
        candidate_paths = [p for p in self.path.iterdir() if p.is_dir()]
        time.time() - dir_scan_start
        # print(
        #           f"[TIMING] Directory scanning took {dir_scan_time:.3f}s, found {len(candidate_paths)} folders"
        # )

        registry = get_vendor_registry()
        valid_scan_entries = []

        for scan_path in candidate_paths:
            vendor_name = detect_vendor(scan_path)
            if not vendor_name:
                logger.debug("Skipping %s: no vendor detected", scan_path)
                continue

            handler_info = registry.get_vendor_info(vendor_name)
            handler_class = handler_info.get("handler_class") if handler_info else None

            if handler_class and hasattr(handler_class, "is_valid_scan_dir"):
                try:
                    if not handler_class.is_valid_scan_dir(scan_path):
                        logger.debug(
                            "Skipping %s: %s handler rejected directory",
                            scan_path,
                            vendor_name,
                        )
                        continue
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning(
                        "Skipping %s: %s validation raised %s",
                        scan_path,
                        vendor_name,
                        exc,
                    )
                    continue

            valid_scan_entries.append((scan_path, vendor_name))

        if not valid_scan_entries:
            logger.warning(
                "No valid scans detected in %s. Check directory structure or vendor support.",
                self.path,
            )
            return pd.DataFrame(columns=["ScanObject"])

        # read in all found scan-folders as vendor-agnostic MRIExperiment objects
        mriexp_start = time.time()
        all_scans = []
        for scan_path, vendor_name in valid_scan_entries:
            try:
                experiment = MRIExperiment(
                    scan_path,
                    load_data=load_data,
                    log_mode=getattr(self, "log_mode", "error"),
                    force_vendor=vendor_name,
                )
            except Exception as exc:
                logger.error(
                    "Failed to initialise experiment for %s (%s): %s",
                    scan_path,
                    vendor_name,
                    exc,
                )
                continue

            if experiment.handler is None:
                logger.debug("Skipping %s: handler creation returned None", scan_path)
                continue

            all_scans.append(experiment)

        time.time() - mriexp_start
        # print(
        #           f"[TIMING] MRIExperiment creation took {mriexp_time:.3f}s for {len(valid_scan_entries)} folders"
        # )

        # use the scan experiment number as index/key for our scans. In case
        # none was found, we assign a unique negative number.
        # Additionally, we check wether the key is already assigned (e.g. when
        # the scan folder has been duplicated). In that case we assign a unique
        # negative number, too.
        if not all_scans:
            logger.warning(
                "No valid experiments could be initialised from %s", self.path
            )
            return pd.DataFrame(columns=["ScanObject"])

        keys = []
        key_types = set()
        cntr = -1
        for scan in all_scans:
            if db:
                logger.info(f"-> Scan.handler: {scan.handler}")

            raw_id = getattr(scan, "id", None)
            normalized_key = None

            if isinstance(raw_id, (int, float)) and raw_id == raw_id:
                try:
                    normalized_int = int(raw_id)
                except (TypeError, ValueError):
                    normalized_int = None
                if normalized_int is not None and normalized_int > 0:
                    normalized_key = normalized_int
            elif isinstance(raw_id, str):
                stripped = raw_id.strip()
                if stripped.isdigit():
                    normalized_key = int(stripped)
                elif stripped:
                    normalized_key = stripped

            if normalized_key is None or normalized_key in keys:
                if normalized_key in keys:
                    msg = f"Scan '{scan.path}' has duplicate key {normalized_key}."
                else:
                    msg = f"Scan '{scan.path}' has no valid key."
                logger.warning(msg + f" -> Assign {cntr} as key instead.")
                normalized_key = cntr
                cntr -= 1

            keys.append(normalized_key)
            key_types.add(type(normalized_key))

        # create a DataFrame with 'keys' as index and scan objects as first entry
        df = pd.DataFrame(
            all_scans, index=pd.Index(keys, name="Key"), columns=["ScanObject"]
        )

        # Sort only when pandas can compare all index values safely
        if len(key_types) == 1:
            try:
                df = df.sort_index()
            except TypeError:
                logger.debug(
                    "Index contains values that cannot be compared reliably; skipping sort."
                )
        else:
            logger.debug(
                "Detected heterogeneous index types %s; keeping insertion order.",
                key_types,
            )

        return df

    def __get_config_content(self):
        """Reads in config file for SmartLoaMeasurement method as json object."""
        # read in the configuration file for our smart load
        module_path = Path(__file__).parents[0]
        config_path = module_path / "config/SmartLoadMeasurements.json"

        with open(config_path, "r") as openfile:
            # Reading from json file
            out = json.load(openfile)

        return out

    def __SmartLoadMeasurements(self, load_data=True):
        """Collect all scans and read them in as custom sequence objects.

        Instead of loading in each scan as a MRIExperiment object (like
        self.__ReadInMeasurements() does), this method detects whether a scan
        has a corresponding sequence-class and if so loads it in as such. The
        appropriate sequence-class is specified in the configuration file:

           '/hypermri/config/SmartLoadMeasurements.json'

        In case no matching sequence-class is found the method falls back to
        reading the scan in as a MRIExperiment object.

        How it works:
            1. Load in each scans metadata only. This is done by using the
               self.__ReadInMeasurements() method with 'load_data=False'.
            2. Use the 'SmartLoadMeasurements.json' config file and try to match
               a corresponding sequence class to each MRIExperiment.type
            3. Load in each scan (with data). Either as a custom sequence object
               - if one was specified in the config file - or as MRIExperiment class.

        Returns
        -------
        out: pandas.DataFrame
            Returns a DataFrame containing all scans. The scans are saved in the
            row 'ScanObject' with their corresponding E-Number as index.
            Additionally, a row 'SequenceClass' is added. Here, the name of the
            class used to load in the scan is listed.
        """

        def lookup_sequence_class_name(scan) -> str:
            """Returns name of sequence class for given scan obj using the config file."""
            if not scan.type:
                return "MRIExperiment"

            scan_type_lower = scan.type.lower()

            try:
                # In a dict of lists, finds the key corresponding to the list
                # containing a pattern that matches the scan type.
                for k, v in sequence_lookup_dict.items():
                    # Check each pattern in the list
                    for pattern in v:
                        # Extract the actual sequence name from patterns like '<Bruker:NSPECT>' or '<User:gNSPECT>'
                        # Remove vendor prefix and angle brackets, then compare case-insensitively
                        if (
                            ":" in pattern
                            and pattern.startswith("<")
                            and pattern.endswith(">")
                        ):
                            # Extract sequence name after the colon
                            sequence_name = pattern.split(":", 1)[1].rstrip(">")
                        else:
                            # Fallback for patterns without vendor prefix
                            sequence_name = pattern.strip("<>")

                        # Case-insensitive comparison
                        if scan_type_lower == sequence_name.lower():
                            return k

                # If no exact match, try substring matching (original behavior as fallback)
                return next(
                    k
                    for k, v in sequence_lookup_dict.items()
                    if scan.type and (scan.type == v or scan.type in v)
                )
            except StopIteration:
                return "MRIExperiment"

        def string2class(classname) -> type:
            """Clean way to convert a string of a class-name into a usable class."""
            try:
                # Try modern import system first
                import hypermri.sequences

                return getattr(hypermri.sequences, classname)
            except (ImportError, AttributeError) as e:
                logger.warning(
                    f"Failed to import {classname} from hypermri.sequences: {e}"
                )

                try:
                    # Try using sys.modules as fallback if __package__ is set
                    if __package__:
                        module_name = __package__ + ".sequences"
                    else:
                        module_name = "hypermri.sequences"
                    return getattr(sys.modules[module_name], classname)
                except KeyError:
                    # If sequences module not in sys.modules, import it
                    import hypermri.sequences

                    module_name = (
                        __package__ + ".sequences"
                        if __package__
                        else "hypermri.sequences"
                    )
                    sys.modules[module_name] = hypermri.sequences
                    return getattr(hypermri.sequences, classname)
                except AttributeError as e:
                    logger.error(
                        f"Class {classname} not found in sequences module: {e}"
                    )
                    raise

        # read in the configuration file for our smart load (as a dictionary)
        sequence_lookup_dict = self.__get_config_content()

        # Use the lookup_sequence_class_name function  to create a column
        # containing the assigned sequence class names for each scan object.
        self.new_column_from_scanobj("SequenceClass", lookup_sequence_class_name)

        # Add a column for the raw sequence type string to make it easier to identify unknown sequences
        # We strip angle brackets to avoid display issues and make it cleaner
        def get_clean_sequence_type(x):
            t = x.type
            if isinstance(t, str):
                return t.replace("<", "").replace(">", "")
            return t

        self.new_column_from_scanobj("Sequence Type", get_clean_sequence_type)

        # Go through every row of the DataFrame, and reload the 'hollow' scan
        # object using the sequence class specifed in the "SequenceClass"
        # column.
        def log_exceptions(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    logger.exception(f"Exception in {func.__name__}:")
                    raise

            return wrapper

        # @log_exceptions
        def load_single_scan(scan_hollow_mri_exp, new_class_name, load_data=True):
            # 'scan_hollow_mri_exp' is an MRIExperiment object with only param files loaded (load_data=False).

            if load_data == "minimal":
                # For minimal loading: only load 2dseq data, no raw data
                # loading
                mri_load_start = time.time()
                # Create a new MRIExperiment with parameters only, then selectively
                # load 2dseq
                fully_loaded_mri_exp = MRIExperiment(
                    scan_hollow_mri_exp.path,
                    load_data=False,
                    log_mode=getattr(self, "log_mode", "error"),
                )
                mri_load_time = time.time() - mri_load_start
                logger.info(
                    f"[TIMING]  -> MRIExperiment creation (params only) took {mri_load_time:.3f}s"
                )
                # Only load and reconstruct seq2d data - no raw data loading
                # Skip seq2d reconstruction for spectroscopy sequences
                spectroscopy_sequences = [
                    "CSI",
                    "PRESS",
                    "STEAM",
                    "NSPECT",
                    "semiLASER",
                    "Singlepulse",
                ]

                if new_class_name not in spectroscopy_sequences:
                    try:
                        reco_start = time.time()
                        # Load 2dseq file directly without loading raw data
                        seq2d_data = fully_loaded_mri_exp.handler.load_2dseq_file()
                        if len(seq2d_data) > 0:
                            fully_loaded_mri_exp.reconstruct("seq2d")
                        reco_time = time.time() - reco_start
                        logger.info(
                            f"[TIMING]-> seq2d-only reconstruction took {reco_time:.3f}s"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to reconstruct seq2d for scan {scan_hollow_mri_exp.path}: {e}"
                        )
                else:
                    logger.info(
                        f"[TIMING]-> Skipping seq2d reconstruction for spectroscopy sequence: {new_class_name}"
                    )
                    handler = fully_loaded_mri_exp.handler
                    if handler is not None:
                        try:
                            raw_load_start = time.time()
                            handler_path = Path(
                                getattr(handler, "path", fully_loaded_mri_exp.path)
                            )

                            def _file_has_data(file_path):
                                try:
                                    return (
                                        file_path.exists()
                                        and file_path.stat().st_size > 0
                                    )
                                except OSError:
                                    return False

                            fid_exists = _file_has_data(
                                handler_path / "fid"
                            ) or _file_has_data(handler_path / "ser")
                            if fid_exists:
                                handler.fid = handler.load_fid_file()
                            else:
                                logger.debug(
                                    f"[TIMING]-> Skipping fid/ser loading for scan {scan_hollow_mri_exp.path}: files missing."
                                )

                            rawdata_path = handler_path / "rawdata.job0"
                            if _file_has_data(rawdata_path):
                                handler.rawdatajob0 = handler.load_rawdatajob0_file()
                            else:
                                logger.debug(
                                    f"[TIMING]-> Skipping rawdata.job0 loading for scan {scan_hollow_mri_exp.path}: file missing."
                                )
                            raw_load_time = time.time() - raw_load_start
                            logger.info(
                                f"[TIMING]-> Minimal raw spectroscopy loading took {raw_load_time:.3f}s"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to minimally load spectroscopy raw data for scan {scan_hollow_mri_exp.path}: {e}"
                            )
            elif load_data == "lazy":
                # For truly lazy loading: parameters only, no data loading at
                # all
                mri_load_start = time.time()
                fully_loaded_mri_exp = MRIExperiment(
                    scan_hollow_mri_exp.path,
                    load_data=False,
                    log_mode=getattr(self, "log_mode", "error"),
                )
                mri_load_time = time.time() - mri_load_start
                logger.info(
                    f"[TIMING]-> Lazy MRIExperiment creation (params only) took {mri_load_time:.3f}s"
                )
                # No data loading or reconstruction - completely lazy
            else:
                # Full loading (original behavior) - loads all data including
                # raw files
                mri_load_start = time.time()
                try:
                    fully_loaded_mri_exp = MRIExperiment(
                        scan_hollow_mri_exp.path,
                        load_data=load_data,
                        log_mode=getattr(self, "log_mode", "error"),
                    )
                    mri_load_time = time.time() - mri_load_start
                    logger.info(
                        f"[TIMING]-> Full MRIExperiment loading took {mri_load_time:.3f}s\n---------------------------\n"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to load full MRIExperiment for scan {scan_hollow_mri_exp.path}: {e}\n---------------------------\n"
                    )

            if new_class_name == "MRIExperiment":
                # If it's a generic MRIExperiment, return the fully loaded MRIExperiment.
                # This MRIExperiment object now contains the loaded data.
                return fully_loaded_mri_exp
            else:
                # If it's a specific sequence (e.g., FLASH), instantiate it.
                # The sequence class's __init__ should be able to take the fully loaded MRIExperiment object
                # and then perform its specific processing on that data.
                seq_start = time.time()
                new_class = string2class(new_class_name)
                # Pass the fully loaded MRIExperiment object to the sequence class's constructor.
                # Pass load_data parameter to control reconstruction level
                reco_options = {"minimal_loading": load_data == "minimal"}
                result = new_class(
                    fully_loaded_mri_exp,
                    load_data=load_data,
                    reco_options=reco_options,
                    log_mode=getattr(self, "log_mode", "error"),
                )
                seq_time = time.time() - seq_start
                logger.debug(
                    f"[TIMING]-> {new_class_name} instantiation took {seq_time:.3f}s"
                )
                return result

        # Show timing information for each scan instead of just tqdm progress
        scan_indices = list(self.__scans.index)
        # print(f"[TIMING] Loading {len(scan_indices)} scans with SmartLoad...")

        for i in tqdm(scan_indices, leave=False, desc="Loading scans"):
            scan_start_time = time.time()
            scan = self.__scans.loc[(i, "ScanObject")]
            new_class_name = self.__scans.loc[(i, "SequenceClass")]
            # print(f"[TIMING] Loading scan {i} as '{new_class_name}'...")

            try:
                self.__scans.loc[(i, "ScanObject")] = load_single_scan(
                    scan,
                    new_class_name,
                    load_data=load_data,
                )
                scan_time = time.time() - scan_start_time
                logger.debug(
                    f"[TIMING] -> Scan {i} ({new_class_name}) took {scan_time:.3f}s to load"
                )

                # Also log at info level for consistency
                logger.info(
                    f"Scan {i} took {scan_time: .3f}s to load as {new_class_name}"
                )

            except Exception as e:
                scan_time = time.time() - scan_start_time
                err_msg, err_type = str(e), e.__class__.__name__
                msg = f"Error loading scan {i} as '{new_class_name}' (took {scan_time: .3f}s): \n"
                msg += f"    {err_type}: {err_msg}\n"
                msg += f"    Traceback:\n{traceback.format_exc()}"
                msg += "\n------------------------------------------------------------------\n"
                logger.error(msg)
                # print(f"[TIMING] -> Scan {i} FAILED after {scan_time:.3f}s")
