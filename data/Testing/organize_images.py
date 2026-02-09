#!/usr/bin/env python3
"""
Script to organize chest X-ray images into pneumonia/normal directories based on CSV labels.
Reads stage2_train_metadata.csv and copies images to organized_images/{pneumonia,normal}/
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, cast


def organize_images(base_dir: str = "."):
    """
    Organize images into pneumonia/normal directories based on Target column.

    Args:
        base_dir: Base directory containing Images/ and stage2_train_metadata.csv
    """
    base_path = Path(base_dir)
    csv_path = base_path / "stage2_train_metadata.csv"
    images_dir = base_path / "Images"
    output_dir = base_path / "organized_images"

    # Create output directories
    pneumonia_dir = output_dir / "pneumonia"
    normal_dir = output_dir / "normal"

    print(f"📁 Creating output directories...")
    pneumonia_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    print(f"📊 Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Group by patientId and take the max Target (handles duplicate rows)
    # If any row for a patient has Target=1, classify as pneumonia
    grouped = df.groupby("patientId")["Target"].max()
    patient_labels: Dict[str, int] = cast(Dict[str, int], grouped.to_dict())  # type: ignore

    print(f"📋 Found {len(patient_labels)} unique patients")

    # Copy images to appropriate directories
    copied_count = {"pneumonia": 0, "normal": 0}
    missing_images = []

    for patient_id, target in patient_labels.items():
        # Image filename is patientId.png
        image_filename = f"{patient_id}.png"
        source_path = images_dir / image_filename

        if not source_path.exists():
            missing_images.append(patient_id)
            continue

        # Determine destination directory and create new filename with label
        if target == 1:
            dest_dir = pneumonia_dir
            label = "pneumonia"
            copied_count["pneumonia"] += 1
        else:
            dest_dir = normal_dir
            label = "normal"
            copied_count["normal"] += 1

        # New filename format: label_patientId.png
        new_filename = f"{label}_{patient_id}.png"
        dest_path = dest_dir / new_filename

        # Copy image
        shutil.copy2(source_path, dest_path)

    # Print summary
    print("\n" + "=" * 60)
    print("✅ ORGANIZATION COMPLETE")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    print(f"   ├── pneumonia/  : {copied_count['pneumonia']} images")
    print(f"   └── normal/      : {copied_count['normal']} images")
    print(f"\n📊 Total images copied: {sum(copied_count.values())}")

    if missing_images:
        print(
            f"\n⚠️  Warning: {len(missing_images)} images not found in Images/ directory"
        )
        print(f"   First 5 missing: {missing_images[:5]}")

    # Calculate class distribution
    total = sum(copied_count.values())
    if total > 0:
        pneumonia_pct = (copied_count["pneumonia"] / total) * 100
        normal_pct = (copied_count["normal"] / total) * 100
        print(f"\n📈 Class distribution:")
        print(f"   Pneumonia: {pneumonia_pct:.1f}%")
        print(f"   Normal:    {normal_pct:.1f}%")


if __name__ == "__main__":
    import sys

    # Allow optional command-line argument for base directory
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print("🏥 Chest X-ray Image Organizer")
    print("=" * 60)
    organize_images(base_dir)
