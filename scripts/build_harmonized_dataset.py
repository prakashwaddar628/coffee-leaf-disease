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
    elif 'cerc' in name:
        return 'coffee___cercospora'
    elif 'phoma' in name:
        return 'coffee___phoma'
    elif 'spider' in name or 'mite' in name:
        return 'coffee___red_spider_mite'
    else:
        return 'coffee___' + name.lower().replace(' ', '_')

def main():
    rocole_dir = Path('data/raw/rocole')
    jmuben_dir = Path('data/raw/jmuben')
    harmonized_dir = Path('data/raw/harmonized')

    # Clean output
    if harmonized_dir.exists():
        shutil.rmtree(harmonized_dir)

    # ── Step 1: Collect all RoCoLe images ──────────────────────────────────
    print("Step 1: Collecting RoCoLe images...")
    rocole_files = defaultdict(list)
    for class_dir in rocole_dir.iterdir():
        if class_dir.is_dir():
            cls = normalize_class_name(class_dir.name)
            for img in class_dir.glob('*.jpg'):
                rocole_files[cls].append(img)

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
        sampled = random.sample(pool, min(n, len(pool)))
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
    main()
