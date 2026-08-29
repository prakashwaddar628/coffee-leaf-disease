import argparse
import hashlib
import shutil
import random
from pathlib import Path
from collections import defaultdict

def normalize_class_name(name):
    name = name.lower()
    if 'healthy' in name:
        return 'coffee___healthy'
    elif 'rust' in name or 'leaf rust' in name:
        return 'coffee___rust'
    elif 'miner' in name:
        return 'coffee___miner'
    elif 'cerc' in name or 'cersc' in name:
        return 'coffee___cercospora'
    elif 'phoma' in name:
        return 'coffee___phoma'
    elif 'spider' in name or 'mite' in name:
        return 'coffee___red_spider_mite'
    else:
        return 'coffee___' + name.lower().replace(' ', '_')

def content_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def unique_files(files, seen_hashes):
    """Return files with unique content, preserving deterministic input order."""
    unique = []
    for file_path in files:
        digest = content_hash(file_path)
        if digest not in seen_hashes:
            seen_hashes.add(digest)
            unique.append(file_path)
    return unique

def main(clean: bool = False):
    rocole_dir = Path('data/raw/rocole')
    jmuben_dir = Path('data/raw/jmuben')
    harmonized_dir = Path('data/raw/harmonized')

    # Never merge a new build into a prior one: that silently changes sample
    # counts and can later leak files across dataset splits.
    if harmonized_dir.exists():
        if not clean:
            raise FileExistsError(
                f"{harmonized_dir} already exists. Re-run with --clean to replace it."
            )
        shutil.rmtree(harmonized_dir)

    # ── Step 1: Collect all RoCoLe images ──────────────────────────────────
    print("Step 1: Collecting RoCoLe images...")
    rocole_files = defaultdict(list)
    for class_dir in rocole_dir.iterdir():
        if class_dir.is_dir():
            cls = normalize_class_name(class_dir.name)
            for img in class_dir.glob('*.jpg'):
                rocole_files[cls].append(img)

    seen_hashes = set()
    for cls in rocole_files:
        rocole_files[cls] = unique_files(rocole_files[cls], seen_hashes)

    rocole_count = sum(len(v) for v in rocole_files.values())
    print(f"  RoCoLe total: {rocole_count} images across {len(rocole_files)} classes")
    for cls, files in rocole_files.items():
        print(f"    {cls}: {len(files)}")

    # ── Step 2: Collect JMuBEN images (no hashing – different source) ──────
    print("\nStep 2: Collecting JMuBEN images...")
    jmuben_files = defaultdict(list)
    # Walk all subdirectories (JMuBEN has a nested structure: jmuben/JMuBEN/<Class>/)
    for img in jmuben_dir.rglob('*.jpg'):
        cls = normalize_class_name(img.parent.name)
        jmuben_files[cls].append(img)

    for cls in jmuben_files:
        jmuben_files[cls] = unique_files(jmuben_files[cls], seen_hashes)

    print(f"  JMuBEN total: {sum(len(v) for v in jmuben_files.values())} images across {len(jmuben_files)} classes")
    for cls, files in jmuben_files.items():
        print(f"    {cls}: {len(files)}")

    # ── Step 3: Build the unified class list & sample to ~3000 ─────────────
    print("\nStep 3: Sampling to 3,000 images...")
    TARGET = 3000
    needed_from_jmuben = TARGET - rocole_count

    # All unique classes across both datasets
    all_classes = set(rocole_files.keys()) | set(jmuben_files.keys())
    # JMuBEN-only classes (need sampling)
    jmuben_only = [c for c in jmuben_files if c not in rocole_files]

    # Distribute needed images evenly across JMuBEN classes
    per_class = needed_from_jmuben // len(jmuben_only) if jmuben_only else 0
    remainder = needed_from_jmuben - (per_class * len(jmuben_only))

    selected = defaultdict(list)
    # Keep all RoCoLe images
    for cls, files in rocole_files.items():
        selected[cls].extend(files)

    # Sample from JMuBEN
    random.seed(42)
    for i, cls in enumerate(sorted(jmuben_only)):
        n = per_class + (1 if i < remainder else 0)
        pool = jmuben_files[cls]
        if len(pool) < n:
            raise ValueError(f"Not enough unique JMuBEN images for {cls}: need {n}, found {len(pool)}")
        sampled = random.sample(pool, n)
        selected[cls].extend(sampled)

    total_selected = sum(len(v) for v in selected.values())
    print(f"  Final selection: {total_selected} images across {len(selected)} classes")
    for cls, files in sorted(selected.items()):
        print(f"    {cls}: {len(files)}")

    # ── Step 4: Copy files to harmonized directory ──────────────────────────
    print("\nStep 4: Copying files...")
    for cls, files in selected.items():
        out_dir = harmonized_dir / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(files):
            dst = out_dir / f"{cls}_{i:04d}.jpg"
            shutil.copy2(src, dst)

    print(f"\nDone! Harmonized dataset ready: {total_selected} images in data/raw/harmonized/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build the reproducible harmonized dataset.")
    parser.add_argument(
        "--clean", action="store_true",
        help="Replace the existing harmonized dataset (required for a rebuild).",
    )
    args = parser.parse_args()
    main(clean=args.clean)
