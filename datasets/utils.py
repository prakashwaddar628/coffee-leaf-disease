"""
Utility functions for file management, downloading, and dataset operations.
"""

import os
import math
import hashlib
import zipfile
import tarfile
import requests
import json
import csv
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Optional, Dict, Any

try:
    from PIL import Image
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    pass # handled during setup

try:
    from datasets.logger import download_logger as logger
except ImportError:
    from logger import download_logger as logger

def create_directory(path: Path) -> None:
    """
    Creates a directory if it does not exist.

    Args:
        path (Path): The directory path to create.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {path}")
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise

def download_file(url: str, dest: Path, resume: bool = True) -> bool:
    """
    Downloads a file from a URL with a progress bar and resume capability.

    Args:
        url (str): The URL of the file to download.
        dest (Path): The destination path.
        resume (bool): Whether to attempt resuming a partial download.

    Returns:
        bool: True if the download is successful, False otherwise.
    """
    try:
        dest_dir = dest.parent
        create_directory(dest_dir)

        headers = {}
        file_mode = "wb"
        downloaded_size = 0

        if resume and dest.exists():
            downloaded_size = dest.stat().st_size
            headers["Range"] = f"bytes={downloaded_size}-"
            file_mode = "ab"

        response = requests.get(url, headers=headers, stream=True)
        
        # 416 Range Not Satisfiable means we probably already downloaded the whole file
        if response.status_code == 416:
            logger.info(f"File already fully downloaded: {dest}")
            return True
            
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0)) + downloaded_size

        logger.info(f"Downloading {url} to {dest}...")
        
        with open(dest, file_mode) as f, tqdm(
            desc=dest.name,
            total=total_size,
            initial=downloaded_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024 * 8):
                size = f.write(data)
                bar.update(size)
                
        logger.info(f"Download complete: {dest}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """
    Extracts a zip or tar archive.

    Args:
        archive_path (Path): Path to the archive.
        extract_to (Path): Destination directory for extraction.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        create_directory(extract_to)
        
        if str(archive_path).endswith('.zip'):
            logger.info(f"Extracting ZIP {archive_path} to {extract_to}...")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif str(archive_path).endswith(('.tar.gz', '.tgz', '.tar')):
            logger.info(f"Extracting TAR {archive_path} to {extract_to}...")
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            logger.error(f"Unsupported archive format: {archive_path}")
            return False
            
        logger.info(f"Extraction complete: {extract_to}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract {archive_path}: {e}")
        return False

def delete_file(path: Path) -> None:
    """
    Deletes a file if it exists.

    Args:
        path (Path): Path to the file to delete.
    """
    try:
        if path.exists() and path.is_file():
            path.unlink()
            logger.info(f"Deleted file: {path}")
    except Exception as e:
        logger.error(f"Failed to delete {path}: {e}")

def calculate_hash(file_path: Path, hash_algo: str = 'md5') -> str:
    """
    Calculates the hash of a file.

    Args:
        file_path (Path): Path to the file.
        hash_algo (str): Hash algorithm to use (default: 'md5').

    Returns:
        str: Hexadecimal hash string.
    """
    try:
        hash_func = getattr(hashlib, hash_algo)()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        return ""

def verify_download(file_path: Path, expected_hash: str, hash_algo: str = 'md5') -> bool:
    """
    Verifies a downloaded file against an expected hash.

    Args:
        file_path (Path): Path to the file.
        expected_hash (str): The expected hash string.
        hash_algo (str): Hash algorithm (default: 'md5').

    Returns:
        bool: True if hashes match, False otherwise.
    """
    if not expected_hash:
        logger.warning(f"No expected hash provided for {file_path}. Skipping verification.")
        return True
        
    logger.info(f"Verifying {file_path}...")
    actual_hash = calculate_hash(file_path, hash_algo)
    if actual_hash == expected_hash:
        logger.info("Verification passed.")
        return True
    else:
        logger.error(f"Verification failed! Expected {expected_hash}, got {actual_hash}")
        return False

def human_readable_size(size_bytes: int) -> str:
    """
    Converts bytes to a human-readable string.

    Args:
        size_bytes (int): Size in bytes.

    Returns:
        str: Human-readable size string.
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024))) if size_bytes > 0 else 0
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def verify_and_analyze_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """
    Thoroughly verifies dataset integrity and extracts rich metadata.
    """
    logger.info(f"Starting deep verification and analysis of {dataset_dir}...")
    
    stats = {
        "image_count": 0,
        "classes": {},
        "duplicate_images": 0,
        "corrupted_images": 0,
        "zero_byte_images": 0,
        "invalid_extensions": 0,
        "resolutions": [],
        "formats": defaultdict(int),
        "color_modes": defaultdict(int),
        "empty_folders": [],
        "duplicates_list": [],
        "corrupted_list": [],
        "verification": "Passed",
        "ready_for_training": True
    }
    
    if not dataset_dir.exists():
        stats["verification"] = "Failed (Directory not found)"
        stats["ready_for_training"] = False
        return stats

    # Valid image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_hashes = {}

    # Walk through the dataset directory
    for root, dirs, files in os.walk(dataset_dir):
        root_path = Path(root)
        
        # Check for empty folders
        if not dirs and not files:
            stats["empty_folders"].append(str(root_path))
            continue
            
        # Identify class (assuming subfolders are classes)
        if root_path != dataset_dir:
            class_name = root_path.name
            if class_name not in stats["classes"]:
                stats["classes"][class_name] = 0
                
        for file in files:
            file_path = root_path / file
            stats["image_count"] += 1
            
            if root_path != dataset_dir:
                stats["classes"][root_path.name] += 1
                
            # Check extension
            if file_path.suffix.lower() not in valid_extensions:
                stats["invalid_extensions"] += 1
                continue
                
            # Check zero byte
            if file_path.stat().st_size == 0:
                stats["zero_byte_images"] += 1
                stats["corrupted_list"].append(str(file_path))
                continue
                
            # Check duplicates via hash
            file_hash = calculate_hash(file_path)
            if file_hash in file_hashes:
                stats["duplicate_images"] += 1
                stats["duplicates_list"].append(str(file_path))
                continue
            file_hashes[file_hash] = str(file_path)
            
            # Verify image integrity and collect metadata
            try:
                with Image.open(file_path) as img:
                    img.verify() # Verify file is intact
                    
                # Re-open to get info (verify() requires re-opening for some formats)
                with Image.open(file_path) as img:
                    width, height = img.size
                    stats["resolutions"].append((width, height))
                    stats["formats"][img.format] += 1
                    stats["color_modes"][img.mode] += 1
                    
            except Exception as e:
                stats["corrupted_images"] += 1
                stats["corrupted_list"].append(str(file_path))

    # Compute resolution statistics
    if stats["resolutions"]:
        widths = [r[0] for r in stats["resolutions"]]
        heights = [r[1] for r in stats["resolutions"]]
        stats["min_resolution"] = f"{min(widths)}x{min(heights)}"
        stats["max_resolution"] = f"{max(widths)}x{max(heights)}"
        stats["average_resolution"] = f"{int(sum(widths)/len(widths))}x{int(sum(heights)/len(heights))}"
    else:
        stats["min_resolution"] = "N/A"
        stats["max_resolution"] = "N/A"
        stats["average_resolution"] = "N/A"
        
    # Determine readiness
    if stats["corrupted_images"] > 0 or stats["zero_byte_images"] > 0 or len(stats["classes"]) == 0:
        stats["verification"] = "Failed (Corruptions/Empty found)"
        stats["ready_for_training"] = False
    elif stats["invalid_extensions"] > 0:
        stats["verification"] = "Warning (Invalid extensions found)"
        
    # Convert defaultdicts to dicts for JSON serialization
    stats["formats"] = dict(stats["formats"])
    stats["color_modes"] = dict(stats["color_modes"])
    
    logger.info("Analysis complete.")
    return stats


def generate_dataset_tree(dataset_dir: Path, output_file: Path) -> None:
    """
    Generates a tree representation of the dataset directory.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{dataset_dir.name}/\n")
            for root, dirs, files in os.walk(dataset_dir):
                level = root.replace(str(dataset_dir), '').count(os.sep)
                indent = ' ' * 4 * (level)
                f.write(f"{indent}{os.path.basename(root)}/\n")
                subindent = ' ' * 4 * (level + 1)
                # Show up to 5 files per dir to keep it concise
                for file in files[:5]:
                    f.write(f"{subindent}{file}\n")
                if len(files) > 5:
                    f.write(f"{subindent}... and {len(files) - 5} more files\n")
    except Exception as e:
        logger.error(f"Failed to generate dataset tree: {e}")


def generate_visual_reports(stats: Dict[str, Any], reports_dir: Path) -> None:
    """
    Generates visual charts for the dataset analysis.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 1. Class Distribution
        if stats["classes"]:
            plt.figure(figsize=(10, 6))
            classes = list(stats["classes"].keys())
            counts = list(stats["classes"].values())
            # Use 'hue' instead of 'palette' directly to avoid FutureWarnings without hue
            sns.barplot(x=counts, y=classes, hue=classes, palette='viridis', legend=False)
            plt.title('Class Distribution')
            plt.xlabel('Number of Images')
            plt.ylabel('Classes')
            plt.tight_layout()
            plt.savefig(reports_dir / 'class_distribution.png')
            plt.close()
            
        # 2. Resolution Distribution (Scatter plot of Width vs Height)
        if stats["resolutions"]:
            plt.figure(figsize=(8, 8))
            widths = [r[0] for r in stats["resolutions"]]
            heights = [r[1] for r in stats["resolutions"]]
            sns.scatterplot(x=widths, y=heights, alpha=0.5, color='b')
            plt.title('Image Resolution Distribution')
            plt.xlabel('Width (pixels)')
            plt.ylabel('Height (pixels)')
            plt.tight_layout()
            plt.savefig(reports_dir / 'resolution_distribution.png')
            plt.close()
            
        logger.info("Generated visual reports.")
    except Exception as e:
        logger.error(f"Failed to generate visual reports: {e}")