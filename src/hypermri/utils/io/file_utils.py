"""
File system utilities.

This module contains file system helper functions for managing files and directories.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..system.logging import init_default_logger

logger = init_default_logger(__name__)


def ensure_directory(path: Union[str, Path], mode: int = 0o755) -> bool:
    """
    Ensure that a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str or Path
        Directory path to ensure exists
    mode : int, optional
        Directory permissions mode. Default is 0o755.

    Returns
    -------
    bool
        True if directory exists or was created successfully
    """
    path = Path(path)

    try:
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        return False


def get_file_info(filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive information about a file.

    Parameters
    ----------
    filepath : str or Path
        Path to the file

    Returns
    -------
    dict or None
        File information dictionary, or None if file doesn't exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return None

    try:
        stat_info = filepath.stat()

        file_info = {
            "name": filepath.name,
            "path": str(filepath.absolute()),
            "size_bytes": stat_info.st_size,
            "size_mb": stat_info.st_size / (1024 * 1024),
            "created": datetime.fromtimestamp(stat_info.st_ctime),
            "modified": datetime.fromtimestamp(stat_info.st_mtime),
            "accessed": datetime.fromtimestamp(stat_info.st_atime),
            "is_file": filepath.is_file(),
            "is_dir": filepath.is_dir(),
            "suffix": filepath.suffix,
            "stem": filepath.stem,
            "parent": str(filepath.parent),
        }

        return file_info

    except Exception as e:
        logger.error(f"Error getting file info for {filepath}: {e}")
        return None


def find_files_by_pattern(
    directory: Union[str, Path], pattern: str, recursive: bool = True
) -> List[Path]:
    """
    Find files matching a pattern in a directory.

    Parameters
    ----------
    directory : str or Path
        Directory to search in
    pattern : str
        Glob pattern to match (e.g., "*.txt", "data_*.csv")
    recursive : bool, optional
        Whether to search recursively in subdirectories. Default is True.

    Returns
    -------
    list of Path
        List of matching file paths
    """
    directory = Path(directory)

    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        return []

    try:
        if recursive:
            pattern_path = directory / "**" / pattern
            files = list(Path().glob(str(pattern_path)))
        else:
            pattern_path = directory / pattern
            files = list(Path().glob(str(pattern_path)))

        # Sort by modification time (newest first)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return files

    except Exception as e:
        logger.error(f"Error finding files with pattern {pattern} in {directory}: {e}")
        return []


def backup_file(
    filepath: Union[str, Path],
    backup_dir: Optional[Union[str, Path]] = None,
    suffix: str = "_backup",
) -> Optional[Path]:
    """
    Create a backup copy of a file.

    Parameters
    ----------
    filepath : str or Path
        File to backup
    backup_dir : str or Path, optional
        Directory for backup. If None, uses same directory as original.
    suffix : str, optional
        Suffix to append to backup filename. Default is "_backup".

    Returns
    -------
    Path or None
        Path to backup file if successful, None otherwise
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return None

    try:
        if backup_dir is None:
            backup_dir = filepath.parent
        else:
            backup_dir = Path(backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)

        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{filepath.stem}{suffix}_{timestamp}{filepath.suffix}"
        backup_path = backup_dir / backup_name

        # Copy file
        shutil.copy2(filepath, backup_path)
        logger.info(f"Backup created: {backup_path}")

        return backup_path

    except Exception as e:
        logger.error(f"Error creating backup for {filepath}: {e}")
        return None


def cleanup_old_files(
    directory: Union[str, Path],
    pattern: str = "*",
    days_old: int = 30,
    dry_run: bool = True,
) -> List[Path]:
    """
    Clean up files older than specified number of days.

    Parameters
    ----------
    directory : str or Path
        Directory to clean up
    pattern : str, optional
        File pattern to match. Default is "*" (all files).
    days_old : int, optional
        Files older than this many days will be deleted. Default is 30.
    dry_run : bool, optional
        If True, only list files that would be deleted. Default is True.

    Returns
    -------
    list of Path
        List of files that were (or would be) deleted
    """
    directory = Path(directory)

    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        return []

    try:
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
        files_to_delete = []

        for filepath in directory.glob(pattern):
            if filepath.is_file():
                if filepath.stat().st_mtime < cutoff_time:
                    files_to_delete.append(filepath)

                    if not dry_run:
                        filepath.unlink()
                        logger.info(f"Deleted old file: {filepath}")

        if dry_run and files_to_delete:
            logger.info(
                f"Dry run: would delete {len(files_to_delete)} files older than {days_old} days"
            )
            for filepath in files_to_delete:
                logger.info(f"  - {filepath}")

        return files_to_delete

    except Exception as e:
        logger.error(f"Error during cleanup in {directory}: {e}")
        return []


def get_directory_size(directory: Union[str, Path]) -> Dict[str, Union[int, float]]:
    """
    Calculate total size of all files in a directory.

    Parameters
    ----------
    directory : str or Path
        Directory to analyze

    Returns
    -------
    dict
        Dictionary with size information (bytes, MB, GB, file_count)
    """
    directory = Path(directory)

    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        return {"bytes": 0, "MB": 0, "GB": 0, "file_count": 0}

    try:
        total_size = 0
        file_count = 0

        for filepath in directory.rglob("*"):
            if filepath.is_file():
                total_size += filepath.stat().st_size
                file_count += 1

        return {
            "bytes": total_size,
            "MB": total_size / (1024 * 1024),
            "GB": total_size / (1024 * 1024 * 1024),
            "file_count": file_count,
        }

    except Exception as e:
        logger.error(f"Error calculating directory size for {directory}: {e}")
        return {"bytes": 0, "MB": 0, "GB": 0, "file_count": 0}


def create_safe_filename(filename: str, replacement_char: str = "_") -> str:
    """
    Create a filesystem-safe filename by replacing problematic characters.

    Parameters
    ----------
    filename : str
        Original filename
    replacement_char : str, optional
        Character to replace problematic characters with. Default is "_".

    Returns
    -------
    str
        Safe filename
    """
    # Characters that are problematic on various filesystems
    problematic_chars = r'<>:"/\|?*'

    safe_filename = filename
    for char in problematic_chars:
        safe_filename = safe_filename.replace(char, replacement_char)

    # Remove leading/trailing spaces and dots
    safe_filename = safe_filename.strip(" .")

    # Limit length (255 is common filesystem limit)
    if len(safe_filename) > 255:
        name, ext = os.path.splitext(safe_filename)
        max_name_length = 255 - len(ext)
        safe_filename = name[:max_name_length] + ext

    return safe_filename


def sync_directories(
    source_dir: Union[str, Path], target_dir: Union[str, Path], dry_run: bool = True
) -> Dict[str, List[Path]]:
    """
    Synchronize files from source to target directory.

    Parameters
    ----------
    source_dir : str or Path
        Source directory
    target_dir : str or Path
        Target directory
    dry_run : bool, optional
        If True, only report what would be done. Default is True.

    Returns
    -------
    dict
        Dictionary with lists of files that were/would be copied, updated, deleted
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    result = {"copied": [], "updated": [], "deleted": [], "errors": []}

    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return result

    try:
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)

        # Get all files in source
        source_files = {}
        for filepath in source_dir.rglob("*"):
            if filepath.is_file():
                rel_path = filepath.relative_to(source_dir)
                source_files[rel_path] = filepath

        # Get all files in target
        target_files = set()
        if target_dir.exists():
            for filepath in target_dir.rglob("*"):
                if filepath.is_file():
                    rel_path = filepath.relative_to(target_dir)
                    target_files.add(rel_path)

        # Process each source file
        for rel_path, source_path in source_files.items():
            target_path = target_dir / rel_path

            if rel_path not in target_files:
                # New file - copy
                result["copied"].append(rel_path)
                if not dry_run:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, target_path)

            elif source_path.stat().st_mtime > target_path.stat().st_mtime:
                # Existing file is older - update
                result["updated"].append(rel_path)
                if not dry_run:
                    shutil.copy2(source_path, target_path)

        if dry_run:
            logger.info(f"Sync preview (source: {source_dir} -> target: {target_dir}):")
            logger.info(f"  Would copy {len(result['copied'])} new files")
            logger.info(f"  Would update {len(result['updated'])} existing files")

        return result

    except Exception as e:
        logger.error(f"Error syncing directories: {e}")
        result["errors"].append(str(e))
        return result
