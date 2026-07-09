"""
Main entry point for downloading datasets.
Provides CLI interface to list and download datasets.
"""

import argparse
import sys
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

try:
    from datasets.config import CONFIG, PROJECT_ROOT, DATA_DIR
    from datasets.logger import download_logger as logger, dataset_logger
    from datasets.utils import (
        create_directory, download_file, extract_archive,
        delete_file, verify_download, calculate_hash,
        verify_and_analyze_dataset, generate_dataset_tree, generate_visual_reports
    )
except ImportError:
    from config import CONFIG, PROJECT_ROOT, DATA_DIR
    from logger import download_logger as logger, dataset_logger
    from utils import (
        create_directory, download_file, extract_archive,
        delete_file, verify_download, calculate_hash,
        verify_and_analyze_dataset, generate_dataset_tree, generate_visual_reports
    )

def ensure_folder_structure() -> None:
    """
    Ensures all required data folders exist.
    """
    folders = ["raw", "processed", "train", "validation", "test", "reports"]
    for folder in folders:
        create_directory(DATA_DIR / folder)

def list_datasets() -> None:
    """
    Lists all available datasets from the configuration.
    """
    print("\nAvailable Datasets:\n")
    datasets = CONFIG.get("datasets", {})
    for i, (key, info) in enumerate(datasets.items(), 1):
        status = "Enabled" if info.get("enabled", False) else "Disabled"
        print(f"[{i}] {info.get('name', key)} ({key}) - {status}")
    print()

def generate_reports() -> None:
    """
    Generates dataset reports (JSON, CSV, MD) with rich metadata.
    """
    reports_dir = DATA_DIR / "reports"
    create_directory(reports_dir)
    
    datasets = CONFIG.get("datasets", {})
    report_data = []
    
    for key, info in datasets.items():
        if info.get("enabled", False):
            output_dir = PROJECT_ROOT / info.get("output_dir", f"data/raw/{key}")
            
            if output_dir.exists():
                logger.info(f"Analyzing dataset: {key}")
                stats = verify_and_analyze_dataset(output_dir)
                
                # Generate tree and visual reports
                generate_dataset_tree(output_dir, reports_dir / "dataset_tree.txt")
                generate_visual_reports(stats, reports_dir)
                
                # Calculate size
                total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
                
                # Build research summary dictionary
                research_report = {
                    "dataset_name": info.get("name", key),
                    "version": info.get("version", "1.0"),
                    "source": info.get("url", "Unknown"),
                    "license": info.get("license", "Unknown"),
                    "citation": info.get("citation", "Unknown"),
                    "download_method": "Automated CLI",
                    "download_date": datetime.now().isoformat(),
                    "dataset_size": f"{total_size} bytes",
                    "image_count": stats.get("image_count", 0),
                }
                
                for cls_name, count in stats.get("classes", {}).items():
                    research_report[f"{cls_name}_images"] = count
                    
                research_report.update({
                    "average_resolution": stats.get("average_resolution", "N/A"),
                    "duplicate_images": stats.get("duplicate_images", 0),
                    "corrupted_images": stats.get("corrupted_images", 0),
                    "verification": stats.get("verification", "Unknown"),
                    "ready_for_training": stats.get("ready_for_training", False)
                })
                
                report_data.append(research_report)
                
                # Verification Report
                verification_report = {
                    "corrupted_files": stats.get("corrupted_list", []),
                    "duplicate_files": stats.get("duplicates_list", []),
                    "empty_folders": stats.get("empty_folders", [])
                }
                with open(reports_dir / "verification_report.json", "w", encoding="utf-8") as f:
                    json.dump(verification_report, f, indent=4)
                    
                # Class Distribution CSV
                with open(reports_dir / "class_distribution.csv", "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Class", "Count"])
                    for cls_name, count in stats.get("classes", {}).items():
                        writer.writerow([cls_name, count])
                        
                # Image Formats CSV
                with open(reports_dir / "image_formats.csv", "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Format", "Count"])
                    for fmt, count in stats.get("formats", {}).items():
                        writer.writerow([fmt, count])

                # Image Resolution CSV
                with open(reports_dir / "image_resolution.csv", "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Metric", "Resolution"])
                    writer.writerow(["Min", stats.get("min_resolution")])
                    writer.writerow(["Max", stats.get("max_resolution")])
                    writer.writerow(["Average", stats.get("average_resolution")])

    # If no datasets were processed, return
    if not report_data:
        return

    # Combined dataset_report.json
    with open(reports_dir / "dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    # Manifest JSON (using the first dataset as primary if multiple, or combined)
    primary = report_data[0]
    manifest = {
        "dataset": primary.get("dataset_name"),
        "classes": len([k for k in primary.keys() if k.endswith("_images") and k not in ["image_count", "duplicate_images", "corrupted_images"]]),
        "images": primary.get("image_count"),
        "verified": primary.get("verification") == "Passed",
        "downloaded": True,
        "ready": primary.get("ready_for_training", False)
    }
    with open(reports_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
        
    # Status JSON
    status_file = reports_dir / "status.json"
    status = {
        "download": True,
        "verification": primary.get("verification") == "Passed",
        "eda": False,
        "preprocessing": False,
        "augmentation": False,
        "training": False,
        "comparison": False,
        "deployment": False
    }
    
    # Don't overwrite EDA/training if they are already true (in case script is re-run)
    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                existing_status = json.load(f)
                status.update({k: v for k, v in existing_status.items() if k not in ["download", "verification"]})
        except Exception:
            pass

    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=4)

    # Dataset Statistics CSV
    keys = set()
    for d in report_data:
        keys.update(d.keys())
    with open(reports_dir / "dataset_statistics.csv", "w", newline="", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, fieldnames=list(keys))
        dict_writer.writeheader()
        dict_writer.writerows(report_data)

    # MD Download Summary
    with open(reports_dir / "download_summary.md", "w", encoding="utf-8") as f:
        f.write("# Dataset Download Summary\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for d in report_data:
            f.write(f"## Dataset Name: {d['dataset_name']}\n\n")
            f.write(f"- **Total Images:** {d['image_count']}\n")
            for k, v in d.items():
                if k.endswith("_images") and k not in ["image_count", "duplicate_images", "corrupted_images"]:
                    clean_name = k.replace("_images", "").replace("coffee___", "").title()
                    f.write(f"- **{clean_name}:** {v}\n")
                    
            f.write(f"- **Average Resolution:** {d['average_resolution']}\n")
            f.write(f"- **Downloaded:** {d['download_date']}\n")
            f.write(f"- **Verified:** {d['verification']}\n")
            f.write(f"- **Status:** {'Ready for EDA' if d['ready_for_training'] else 'Needs Attention'}\n\n")

    logger.info("Generated comprehensive dataset reports in data/reports/")

def process_dataset(dataset_id: str, info: Dict[str, Any]) -> None:
    """
    Processes a single dataset (download, extract, verify).
    
    Args:
        dataset_id (str): The dataset key.
        info (Dict[str, Any]): The dataset configuration dictionary.
    """
    logger.info(f"--- Processing {info.get('name', dataset_id)} ---")
    
    url = info.get("url")
    if not url:
        logger.error(f"No URL provided for {dataset_id}")
        return
        
    # Path resolution
    output_dir_str = info.get("output_dir", f"data/raw/{dataset_id}")
    output_dir = PROJECT_ROOT / output_dir_str
    archive_name = info.get("archive_name", f"{dataset_id}.zip")
    archive_path = output_dir / archive_name
    
    # Check if already exists (naively checking if output dir has files other than archive)
    if output_dir.exists() and any(f.is_file() and f.name != archive_name for f in output_dir.iterdir()):
        logger.info(f"Dataset {dataset_id} already exists at {output_dir}. Skipping download.")
        return

    create_directory(output_dir)
    
    # Check for Kaggle URL
    if url.startswith("kaggle://"):
        kaggle_id = url[9:]
        dataset_logger.info(f"{info.get('name', dataset_id)} Kaggle Download Started for {kaggle_id}")
        import subprocess
        import sys
        import os
        kaggle_exe = os.path.join(os.path.dirname(sys.executable), "kaggle")
        if os.name == 'nt':
            kaggle_exe += ".exe"
            
        try:
            # Check if kaggle is installed
            subprocess.run([kaggle_exe, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Kaggle CLI is not installed or configured. Please install it with `pip install kaggle` and ensure kaggle.json is present in your user directory (~/.kaggle/kaggle.json).")
            return
            
        cmd = [kaggle_exe, "datasets", "download", "-d", kaggle_id, "-p", str(output_dir), "--unzip"]
        try:
            subprocess.run(cmd, check=True)
            dataset_logger.info("Kaggle Download and Extraction Complete")
        except subprocess.CalledProcessError as e:
            logger.error(f"Kaggle download failed: {e}")
        return

    # Regular Download
    dataset_logger.info(f"{info.get('name', dataset_id)} Download Started")
    success = download_file(url, archive_path)
    
    if not success:
        logger.error(f"Download failed for {dataset_id}")
        return
        
    dataset_logger.info("Download Complete")
    
    # Verify
    expected_hash = info.get("expected_hash")
    if expected_hash:
        if verify_download(archive_path, expected_hash):
            dataset_logger.info("Verification Passed")
        else:
            dataset_logger.error("Verification Failed")
            return
            
    # Extract
    if info.get("extract", False):
        if extract_archive(archive_path, output_dir):
            dataset_logger.info("Extraction Complete")
        else:
            dataset_logger.error("Extraction Failed")
            
    # Cleanup
    if info.get("delete_archive", False):
        delete_file(archive_path)

def download_datasets(target: str) -> None:
    """
    Downloads the specified dataset(s).
    
    Args:
        target (str): The dataset id, or 'all'.
    """
    datasets = CONFIG.get("datasets", {})
    
    if target == "all":
        for key, info in datasets.items():
            if info.get("enabled", False):
                process_dataset(key, info)
            else:
                logger.warning(f"Skipping {key} because it is disabled.")
    else:
        info = datasets.get(target)
        if info:
            if info.get("enabled", False):
                process_dataset(target, info)
            else:
                logger.error(f"Dataset {target} is disabled in config.")
        else:
            logger.error(f"Dataset {target} not found in configuration.")
            
    generate_reports()

def main():
    parser = argparse.ArgumentParser(description="Dataset Manager for Coffee Leaf Disease Research")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--download", type=str, metavar="DATASET", help="Download a specific dataset or 'all'")
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
    elif args.download:
        ensure_folder_structure()
        download_datasets(args.download)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()