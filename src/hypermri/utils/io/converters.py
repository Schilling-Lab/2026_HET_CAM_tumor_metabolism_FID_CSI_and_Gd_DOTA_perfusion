"""
File format conversion utilities.

Contains functions for exporting data to various formats like Word, Excel, CSV, MATLAB, and NetCDF.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr

from ..system.logging import init_default_logger

logger = init_default_logger(__name__)


def save_markdown_as_word(
    input_md: str | None = None,
    file_path: str | None = None,
    file_name: str | None = None,
    print_file_path: bool = True,
) -> None:
    """
    Convert a Markdown string to a Word (.docx) file using Pandoc via subprocess.

    Parameters
    ----------
    input_md : str or None
        The protocol content in Markdown format.
    file_path : str or None
        The directory path where the resulting .docx file should be saved.
    file_name : str or None
        The name of the output Word document.
    print_file_path : bool, optional
        If True, prints the final file path upon successful creation.

    Notes
    -----
    Pandoc must be installed on your system.
    """
    # Write the Markdown text to a temporary .md file
    temp_md = "temp_protocol.md"
    with open(temp_md, "w", encoding="utf-8") as f:
        f.write(input_md if input_md else "")

    # Construct the full output path
    if file_path is None:
        file_path = os.getcwd()
    output_docx_path = os.path.join(file_path, file_name) if file_name else None

    try:
        # Invoke Pandoc to convert Markdown to .docx
        subprocess.run(["pandoc", temp_md, "-o", output_docx_path], check=True)
        if print_file_path:
            logger.info(f"Saved Word file to: {output_docx_path}")
    except FileNotFoundError:
        logger.error("Pandoc not found. Please install pandoc to use this feature.")
    except Exception as e:
        logger.error(f"Error converting markdown to word: {e}")
    finally:
        # Remove the temporary file
        if os.path.exists(temp_md):
            os.remove(temp_md)


def save_dict_to_excel(
    data_dict, savepath, filename, sheet_name=None, header=None, index=False
):
    """
    Saves a dictionary to an Excel file.

    Parameters
    ----------
    data_dict : dict
        Dictionary to save.
    savepath : str
        Directory path.
    filename : str
        Name of the Excel file.
    sheet_name : str, optional
        Name of the sheet.
    header : list, optional
        Column names.
    index : bool, default False
        Whether to write row indices.
    """
    os.makedirs(savepath, exist_ok=True)
    file_path = os.path.join(savepath, f"{filename}.xlsx")
    sheet_name = sheet_name if sheet_name else "Sheet1"

    df = pd.DataFrame(
        list(data_dict.items()), columns=header if header else ["Key", "Value"]
    )

    if os.path.exists(file_path):
        with pd.ExcelWriter(
            file_path, engine="openpyxl", mode="a", if_sheet_exists="new"
        ) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index)
    else:
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index)

    logger.info(f"Data saved to: {file_path}, sheet: {sheet_name}")


def export_to_csv(
    data: Union[pd.DataFrame, np.ndarray, dict], filepath: Union[str, Path], **kwargs
) -> bool:
    """
    Export data to CSV format.

    Parameters
    ----------
    data : DataFrame, ndarray, or dict
        Data to export.
    filepath : str or Path
        Output file path.
    **kwargs
        Additional arguments for pandas.to_csv().

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    filepath = Path(filepath)

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, **kwargs)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                df = pd.DataFrame(data, columns=["Value"])
            elif data.ndim == 2:
                df = pd.DataFrame(data)
            else:
                logger.error("Cannot export arrays with >2 dimensions to CSV")
                return False
            df.to_csv(filepath, **kwargs)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
            df.to_csv(filepath, **kwargs)
        else:
            logger.error(f"Unsupported data type for CSV export: {type(data)}")
            return False

        logger.info(f"Data exported to CSV: {filepath}")
        return True

    except Exception as e:
        logger.error(f"Error exporting to CSV {filepath}: {e}")
        return False


def export_to_matlab(data: Dict[str, Any], filepath: Union[str, Path]) -> bool:
    """
    Export data to MATLAB .mat format.

    Parameters
    ----------
    data : dict
        Data dictionary to export.
    filepath : str or Path
        Output file path.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    filepath = Path(filepath)

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.suffix != ".mat":
            filepath = filepath.with_suffix(".mat")

        sio.savemat(str(filepath), data)
        logger.info(f"Data exported to MATLAB format: {filepath}")
        return True

    except Exception as e:
        logger.error(f"Error exporting to MATLAB {filepath}: {e}")
        return False


def load_from_matlab(filepath: Union[str, Path]) -> Union[Dict[str, Any], None]:
    """
    Load data from MATLAB .mat format.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    dict or None
        Loaded data dictionary, or None if failed.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return None

    try:
        data = sio.loadmat(str(filepath))
        # Remove MATLAB metadata
        cleaned_data = {k: v for k, v in data.items() if not k.startswith("__")}
        logger.info(f"Data loaded from MATLAB format: {filepath}")
        return cleaned_data

    except Exception as e:
        logger.error(f"Error loading from MATLAB {filepath}: {e}")
        return None


def export_xarray_to_netcdf(data_array, filepath: Union[str, Path], **kwargs) -> bool:
    """
    Export xarray DataArray/Dataset to NetCDF format.

    Parameters
    ----------
    data_array : xarray.DataArray or xarray.Dataset
        Data to export.
    filepath : str or Path
        Output file path.
    **kwargs
        Additional arguments for to_netcdf().

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    filepath = Path(filepath)

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.suffix != ".nc":
            filepath = filepath.with_suffix(".nc")

        data_array.to_netcdf(str(filepath), **kwargs)
        logger.info(f"xarray data exported to NetCDF: {filepath}")
        return True

    except Exception as e:
        logger.error(f"Error exporting xarray to NetCDF {filepath}: {e}")
        return False


def load_xarray_from_netcdf(filepath: Union[str, Path]):
    """
    Load xarray data from NetCDF format.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    xarray.DataArray or xarray.Dataset or None
        Loaded data, or None if failed.
    """
    import xarray as xr

    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return None

    try:
        data = xr.open_dataset(str(filepath))
        logger.info(f"xarray data loaded from NetCDF: {filepath}")
        return data

    except Exception as e:
        logger.error(f"Error loading xarray from NetCDF {filepath}: {e}")
        return None


def export_to_dicom(
    data: Union["xr.DataArray", np.ndarray],
    output_dir: Union[str, Path],
    reference_dicoms: list = None,
    filename_prefix: str = "MRIm",
    series_description: str = "Exported Data",
    series_number: int = 999,
    unit: str = None,
) -> bool:
    """
    Export data to DICOM files.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        The image data to export. Should be 3D (rows, cols, slices).
    output_dir : str or Path
        Directory to save the DICOM files.
    reference_dicoms : list, optional
        List of pydicom datasets to use as templates.
        If provided, metadata will be copied from these.
        Must match the number of slices in data.
    filename_prefix : str, optional
        Prefix for the output filenames. Default is "MRIm".
    series_description : str, optional
        Description for the DICOM series.
    series_number : int, optional
        Series number for the exported data.
    unit : str, optional
        Unit string to set in the DICOM RescaleType tag (0028,1054).

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        import pydicom
        from pydicom.dataset import FileDataset
        from pydicom.uid import generate_uid
    except ImportError:
        logger.error("pydicom is required for DICOM export.")
        return False

    import copy

    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        return False

    # Ensure data is numpy array
    if hasattr(data, "values"):
        img_data = data.values
    else:
        img_data = data

    # Handle data shape
    # Squeeze singleton dimensions to handle typical Bruker high-dim arrays
    img_data = np.squeeze(img_data)

    # Expecting (rows, cols, slices)
    if img_data.ndim == 2:
        img_data = img_data[..., np.newaxis]

    if img_data.ndim != 3:
        logger.error(
            f"Data must be 2D or 3D after squeezing, got {img_data.ndim}D. Shape: {img_data.shape}"
        )
        return False

    rows, cols, n_slices = img_data.shape

    # If reference dicoms provided, check count
    if reference_dicoms:
        if len(reference_dicoms) == n_slices:
            # Perfect match
            pass
        else:
            # Mismatch. Try to resolve.
            logger.info(
                f"Reference DICOM count ({len(reference_dicoms)}) != data slices ({n_slices}). "
                "Attempting to match..."
            )

            matched_refs = None

            # Strategy 1: Group by Spatial Position (Robust for Channel/Time reduction)
            try:
                # Ensure sorted by InstanceNumber for consistent ordering
                # We sort the ORIGINAL list to scan through it in order
                sorted_refs = sorted(
                    reference_dicoms, key=lambda x: getattr(x, "InstanceNumber", 0)
                )

                unique_refs = []
                seen_positions = set()

                for ds in sorted_refs:
                    # Use rounded tuple for position identity
                    # 3 decimal places = 1 micron, sufficient for MRI
                    if hasattr(ds, "ImagePositionPatient"):
                        pos = tuple(round(float(c), 3) for c in ds.ImagePositionPatient)
                        if pos not in seen_positions:
                            seen_positions.add(pos)
                            unique_refs.append(ds)

                if len(unique_refs) == n_slices:
                    logger.warning(
                        f"Found {len(unique_refs)} unique slice positions in {len(reference_dicoms)} reference DICOMs. "
                        f"Matching them to the {n_slices} data slices."
                    )
                    matched_refs = unique_refs
            except Exception as e:
                logger.debug(f"Position matching failed: {e}")

            if matched_refs:
                reference_dicoms = matched_refs
            else:
                # Strategy 2: Fallback to First N (Sequential assumption)
                # Only if simple multiple?
                if (
                    len(reference_dicoms) > n_slices
                    and len(reference_dicoms) % n_slices == 0
                ):
                    logger.warning(
                        f"Could not match by position. Falling back to using the first {n_slices} reference DICOMs "
                        "(assuming sequential ordering)."
                    )
                    # Sort references just in case
                    try:
                        reference_dicoms.sort(
                            key=lambda x: getattr(x, "InstanceNumber", 0)
                        )
                    except Exception:
                        pass
                    reference_dicoms = reference_dicoms[:n_slices]
                else:
                    logger.warning(
                        f"Could not reconcile reference DICOM count ({len(reference_dicoms)}) with data slices ({n_slices}). "
                        "Ignoring reference DICOMs."
                    )
                    reference_dicoms = None

        if reference_dicoms:
            # Sort references just in case
            try:
                reference_dicoms.sort(key=lambda x: getattr(x, "InstanceNumber", 0))
            except Exception:
                pass

    # Generate Series UID
    series_instance_uid = generate_uid()
    study_instance_uid = generate_uid()  # Default new study

    if reference_dicoms:
        study_instance_uid = reference_dicoms[0].StudyInstanceUID

    logger.info(f"Exporting {n_slices} slices to {output_dir}")

    for i in range(n_slices):
        slice_data = img_data[:, :, i]

        # Use reference if available
        if reference_dicoms:
            try:
                # Deep copy to avoid modifying original
                ds = copy.deepcopy(reference_dicoms[i])
            except Exception as e:
                logger.warning(
                    f"Failed to copy reference DICOM {i}: {e}. Creating new."
                )
                ds = None
        else:
            ds = None

        if ds is None:
            # Create fresh dataset
            file_meta = pydicom.dataset.FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

            # Preamble is required for valid DICOM file
            ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

            # Set minimal required tags
            ds.SOPClassUID = pydicom.uid.MRImageStorage
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID = study_instance_uid
            ds.SeriesInstanceUID = series_instance_uid
            ds.Modality = "MR"
            ds.SeriesNumber = series_number
            ds.InstanceNumber = i + 1

            ds.Rows = rows
            ds.Columns = cols
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"

        # Update metadata common to both paths
        ds.SeriesDescription = series_description
        ds.SeriesNumber = series_number
        ds.SeriesInstanceUID = series_instance_uid

        # Always generate new SOP Instance UID for the exported file
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

        # Update Image Type to indicate DERIVED if using reference?
        # ds.ImageType = ['DERIVED', 'SECONDARY']

        # Handle Pixel Data
        # Ensure data is in appropriate format (int16 for standard MRI)

        # Reset scaling tags by default to avoid carrying over incorrect reference scaling
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0

        # Set unit if provided
        if unit is not None:
            ds.RescaleType = unit

        if slice_data.dtype in [np.float32, np.float64]:
            # For float data, we should scale it to fit the int16 range to preserve precision
            # and avoid truncation/overflow
            min_val = np.nanmin(slice_data)
            max_val = np.nanmax(slice_data)

            # Handle constant or empty image
            if min_val == max_val:
                # Constant image
                # Check if it fits in int16
                if -32768 <= min_val <= 32767:
                    pixel_data = np.full_like(slice_data, int(min_val), dtype=np.int16)
                    ds.RescaleSlope = 1.0
                    ds.RescaleIntercept = 0.0
                else:
                    # Constant value outside int16, use intercept
                    pixel_data = np.zeros_like(slice_data, dtype=np.int16)
                    ds.RescaleSlope = 1.0
                    ds.RescaleIntercept = min_val
            else:
                # Map range [min_val, max_val] to [-32767, 32767] (keep -32768 reserve?)
                # We use int16 range.
                target_min = -32767.0
                target_max = 32767.0
                target_range = target_max - target_min
                data_range = max_val - min_val

                slope = data_range / target_range
                intercept = min_val - (target_min * slope)

                # Apply scaling: SV = (PV - b) / m
                scaled_data = (slice_data - intercept) / slope

                # Round and cast
                pixel_data = np.round(scaled_data).astype(np.int16)

                ds.RescaleSlope = slope
                ds.RescaleIntercept = intercept
        else:
            # Assume integer data provided by user fits in int16 or is already prepared
            # We enforce Slope 1, Intercept 0 for integer data to avoid "hidden" scaling from reference
            pixel_data = slice_data.astype(np.int16)
            ds.RescaleSlope = 1.0
            ds.RescaleIntercept = 0.0

        ds.PixelData = pixel_data.tobytes()
        ds.Rows, ds.Columns = pixel_data.shape

        # Save
        filename = f"{filename_prefix}{i + 1:02d}.dcm"
        try:
            ds.save_as(output_dir / filename)
        except Exception as e:
            logger.error(f"Failed to save DICOM {filename}: {e}")
            return False

    logger.info(f"Successfully exported DICOM files to {output_dir}")
    return True
