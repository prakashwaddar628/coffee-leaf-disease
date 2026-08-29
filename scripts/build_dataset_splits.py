"""Create verified, deterministic and disjoint train/validation/test splits.

This is intentionally a standalone script rather than notebook-only logic so
the dataset used by experiments can be recreated and audited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR = DATA_DIR / "processed" / "final"
SPLITS = ("train", "validation", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_samples(source_dir: Path) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    labels: list[str] = []
    for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        for image in sorted(class_dir.glob("*.jpg")):
            files.append(image)
            labels.append(class_dir.name)
    if not files:
        raise ValueError(f"No JPG images found in {source_dir}")
    if min(Counter(labels).values()) < 10:
        raise ValueError("Every class needs at least 10 images for an 80/10/10 stratified split.")
    return files, labels


def prepare_output(clean: bool) -> None:
    existing = [DATA_DIR / split for split in SPLITS if (DATA_DIR / split).exists()]
    if existing and not clean:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Existing split directories: {names}. Re-run with --clean to replace them.")
    for path in existing:
        shutil.rmtree(path)


def validate_disjoint(split_files: dict[str, list[Path]]) -> None:
    seen_paths: set[str] = set()
    seen_hashes: dict[str, str] = {}
    for split, files in split_files.items():
        for source in files:
            source_key = str(source.resolve())
            if source_key in seen_paths:
                raise RuntimeError(f"Source image appears in more than one split: {source}")
            seen_paths.add(source_key)
            content_hash = sha256(source)
            prior_split = seen_hashes.get(content_hash)
            if prior_split is not None:
                raise RuntimeError(
                    f"Duplicate image content would span {prior_split} and {split}: {source}"
                )
            seen_hashes[content_hash] = split


def main(clean: bool) -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found: {SOURCE_DIR}")
    with (PROJECT_ROOT / "datasets" / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["training"]

    files, labels = collect_samples(SOURCE_DIR)
    train_files, temporary_files, train_labels, temporary_labels = train_test_split(
        files, labels, test_size=1 - config["train_split"], stratify=labels,
        random_state=config["random_seed"],
    )
    test_fraction_of_temp = config["test_split"] / (config["validation_split"] + config["test_split"])
    validation_files, test_files, _, _ = train_test_split(
        temporary_files, temporary_labels, test_size=test_fraction_of_temp,
        stratify=temporary_labels, random_state=config["random_seed"],
    )
    split_files = {"train": train_files, "validation": validation_files, "test": test_files}
    validate_disjoint(split_files)
    prepare_output(clean)

    manifest: list[dict[str, str]] = []
    for split, files_for_split in split_files.items():
        for source in files_for_split:
            destination = DATA_DIR / split / source.parent.name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest.append({
                "split": split,
                "class": source.parent.name,
                "relative_path": str(source.relative_to(SOURCE_DIR)).replace("\\", "/"),
                "sha256": sha256(source),
            })

    reports_dir = DATA_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with (reports_dir / "split_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "images"])
        writer.writerows((split, len(files_for_split)) for split, files_for_split in split_files.items())
    class_counts = Counter(train_labels)
    raw_weights = {label: len(train_labels) / count for label, count in class_counts.items()}
    weight_total = sum(raw_weights.values())
    with (reports_dir / "class_weights.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Class", "Weight"])
        writer.writerows((label, raw_weights[label] / weight_total) for label in sorted(raw_weights))
    status_path = reports_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status.update({
        "split_verified": True,
        "metrics_valid": False,
        "training": False,
        "mobilenet": False,
        "resnet": False,
    })
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("Created verified disjoint splits:", ", ".join(f"{split}={len(items)}" for split, items in split_files.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build verified train/validation/test splits.")
    parser.add_argument("--clean", action="store_true", help="Replace existing split directories.")
    arguments = parser.parse_args()
    main(clean=arguments.clean)
