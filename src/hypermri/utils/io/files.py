"""
General file input/output operations.

Contains functions for saving and loading data in various formats,
and managing file system structures.
"""

import os
import pickle
from datetime import datetime
from pathlib import Path, PosixPath, WindowsPath

import numpy as np
import scipy.io as sio

from ..system.logging import init_default_logger

logger = init_default_logger(__name__)


def create_folder_structure(save_path=None, structure=None):
    """
    Creates a folder structure within a specified base directory.

    Parameters
    ----------
    save_path : str
        The base directory where the folder structure will be created.
    structure : dict, optional
        A dictionary defining the folder structure.
        The keys represent main folder names, and the values are lists of subfolder names.

    Returns
    -------
    None
    """
    if save_path is None:
        logger.error("savepath has to be passed!")
        return

    # Define the folder structure to use
    if structure is None:
        structure = {
            "01_figures": [
                "01_anatomical_images",
                "02_metabolic_images",
                "03_misc_images",
            ],
            "02_data": ["01_fit_results", "02_masks"],
        }

    # Create the base savepath if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Created base path: {save_path}")

    # Create folders and subfolders
    for main_folder, subfolders in structure.items():
        main_folder_path = os.path.join(save_path, main_folder)
        os.makedirs(main_folder_path, exist_ok=True)
        for subfolder in subfolders:
            subfolder_path = os.path.join(main_folder_path, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)


def save_data(
    file_path: str | None = None,
    file_name: str | None = None,
    file: object | None = None,
    file_keys: list[str] | None = None,
    use_timestamp: bool = True,
    print_file_path: bool = True,
    save_as_mat: bool = False,
) -> None:
    """
    Saves a given file, which can be a Python object or a NumPy array, to the specified directory.

    Parameters
    ----------
    file_path : str | None
        The directory path where the file will be saved.
    file_name : str | None
        The base name of the file.
    file : object | np.ndarray | None
        The data to be saved.
    file_keys : list[str] | None, optional
        Keys to use when saving a list of objects as a dictionary.
    use_timestamp : bool, optional
        If True, appends a timestamp to the file_name. Default is True.
    print_file_path : bool, optional
        If True, prints the full path of the saved file. Default is True.
    save_as_mat: bool, optional
        If True, data is saved as a .mat-file (experimental).

    Returns
    -------
    None
    """
    if file_path is None:
        file_path = "."
    os.makedirs(file_path, exist_ok=True)

    if use_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = f"_{timestamp}"
    else:
        suffix = ""

    if save_as_mat:
        full_path = os.path.join(file_path, f"{file_name}{suffix}.mat")
    else:
        # Determine extension later based on content
        base_path = os.path.join(file_path, f"{file_name}{suffix}")

    # Helper function to ensure keys are a list
    def ensure_keys_list(keys):
        if keys is None:
            return keys
        if isinstance(keys, list):
            return keys
        try:
            return list(keys)
        except TypeError:
            raise ValueError("keys cannot be converted to a list")

    file_keys = ensure_keys_list(keys=file_keys)

    if save_as_mat:
        if isinstance(file, dict):
            data_to_save = {
                k.replace(" ", "_"): np.atleast_2d(np.asarray(v))
                for k, v in file.items()
            }
        elif isinstance(file, np.ndarray) and file_keys is None:
            data_to_save = {"data": np.atleast_2d(file)}
        elif isinstance(file, list) and file_keys is not None:
            data_to_save = {
                key.replace(" ", "_"): np.atleast_2d(np.asarray(file[k]))
                for k, key in enumerate(file_keys)
            }
        else:
            raise ValueError("Unsupported file type or missing file_keys for lists")

        sio.savemat(full_path, data_to_save)
        if print_file_path:
            logger.info(f"Data saved as {full_path}")

    elif isinstance(file, np.ndarray) and file_keys is None:
        full_path = f"{base_path}.npz"
        np.savez_compressed(full_path, file)
        if print_file_path:
            logger.info(f"Data saved as {full_path}")

    else:
        # Pickling logic
        if isinstance(file, list) and file_keys is not None:
            dict_to_save = {key: file[k] for k, key in enumerate(file_keys)}
            file = dict_to_save
        elif isinstance(file, np.ndarray) and file_keys is not None:
            dict_to_save = {key: file[k] for k, key in enumerate(file_keys)}
            file = dict_to_save

        full_path = f"{base_path}.pkl"
        with open(full_path, "wb") as save_file:
            pickle.dump(file, save_file, protocol=pickle.HIGHEST_PROTOCOL)
        if print_file_path:
            logger.info(f"Data saved as {full_path}")


def load_data(
    file_path: str | None = None,
    file_name: str | None = None,
    full_name: str | None = None,
    global_vars: dict = None,
) -> object:
    """
    Loads data from a file in the specified directory. Can handle both `.npz` and `.pkl` files.

    Parameters
    ----------
    file_path : str
        The directory path from where the file will be loaded.
    file_name : str
        The base name of the file to be loaded.
    full_name : str, optional
        The file name and path in one string.
    global_vars : dict, optional
        If provided, updates globals with the loaded dictionary.

    Returns
    -------
    object
        The data loaded from the file.
    """

    # Custom Unpickler to replace WindowsPath with PosixPath
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "pathlib" and name == "WindowsPath":
                return PosixPath
            return super().find_class(module, name)

    def convert_paths(obj):
        """Recursively convert WindowsPath to PosixPath."""
        if isinstance(obj, WindowsPath):
            return PosixPath(obj.as_posix())
        elif isinstance(obj, dict):
            return {key: convert_paths(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_paths(item) for item in obj)
        return obj

    if full_name is not None:
        file_path = os.path.dirname(full_name)
        file_name = os.path.basename(full_name)
        logger.info(f"Loading file: {file_name} from {file_path}")

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Directory not found: {file_path}")

    # List all files that start with the filename and have the proper extensions
    files = [
        f
        for f in file_path.iterdir()
        if f.name.startswith(file_name) and f.suffix in {".pkl", ".npz"}
    ]
    if not files:
        raise FileNotFoundError(
            f"No files found starting with {file_name} in {file_path}"
        )

    if len(files) > 1:
        # Sort by modification time to get the latest
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        # For now, just pick the newest one or ask user?
        # The original code asked for input, which is bad for automation.
        # We will default to the newest one and log it.
        logger.info(f"Multiple files found. Selecting newest: {files[0].name}")
        selected_file = files[0]
    else:
        selected_file = files[0]

    # Load the file based on its extension
    if selected_file.suffix == ".npz":
        data = np.load(selected_file, allow_pickle=True)
        key = next(iter(data))
        loaded_data = data[key]
    elif selected_file.suffix == ".pkl":
        with open(selected_file, "rb") as file:
            loaded_data = CustomUnpickler(file).load()
            loaded_data = convert_paths(loaded_data)

    if global_vars is not None and isinstance(loaded_data, dict):
        global_vars.update(loaded_data)
    elif global_vars is not None:
        global_vars[file_name] = loaded_data

    return loaded_data


def load_plot_params(param_file=None, path_to_param_file=None, data_obj=None):
    """
    Load plotting parameters from a JSON file.

    Parameters
    ----------
    param_file : str
        Name of the parameter file.
    path_to_param_file : str, optional
        Path to the parameter file.
    data_obj : object, optional
        Data object to derive path from if not provided.

    Returns
    -------
    dict
        Loaded parameters or empty dict.
    """
    import json

    if param_file is None:
        logger.warning("param_file has to be defined")
        return {}

    # Determine the directory of the parameters file
    if path_to_param_file is None:
        if data_obj is not None:
            if hasattr(data_obj, "savepath"):
                path_parentfolder = data_obj.savepath
            elif hasattr(data_obj, "path"):
                import sys

                path_str = str(data_obj.path)
                if sys.platform == "win32":
                    path_parentfolder = path_str[: path_str.rfind("\\")]
                else:
                    path_parentfolder = path_str[: path_str.rfind("/")]
            else:
                return {}
        else:
            return {}
    else:
        path_parentfolder = path_to_param_file

    # Ensure the file has a '.json' extension
    if not param_file.endswith(".json"):
        param_file = param_file + ".json"

    # Attempt to load the JSON file
    try:
        with open(os.path.join(path_parentfolder, param_file)) as f:
            plot_params = json.load(f)
    except BaseException:
        try:
            with open(param_file) as f:
                plot_params = json.load(f)
        except BaseException:
            logger.warning("loading plot parameters did not work")
            return {}

    # Try saving the loaded parameters to the object's attribute
    if data_obj is not None:
        if not hasattr(data_obj, "plot_params"):
            setattr(data_obj, "plot_params", "")
        try:
            data_obj.plot_params = plot_params
        except BaseException:
            logger.warning("Saving plot parameters did not work")

    return plot_params
