"""
Training Time Analysis for Centralized vs Federated Learning
=============================================================
Analyzes and compares training times for both paradigms.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent

# Load centralized data
cent_df = pd.read_csv(BASE_DIR / "centralized" / "experiment_results.csv")

# Filter for seeds 44-48 (5 runs)
seeds = [44, 45, 46, 47, 48]
cent_filtered = cent_df[cent_df['seed'].isin(seeds)].copy()

print("=" * 70)
print("TRAINING TIME ANALYSIS: Centralized vs Federated")
print("=" * 70)

# ============================================================================
# CENTRALIZED TRAINING TIMES
# ============================================================================
print("\n" + "=" * 70)
print("CENTRALIZED LEARNING (Seeds 44-48)")
print("=" * 70)

cent_times = cent_filtered['training_time_seconds'].values
cent_epochs = cent_filtered['epochs_completed'].values

print("\nPer-Run Training Times:")
print("-" * 70)
for seed, time_sec, epochs in zip(seeds, cent_times, cent_epochs):
    hours = int(time_sec // 3600)
    minutes = int((time_sec % 3600) // 60)
    seconds = int(time_sec % 60)
    print(f"  Seed {seed}: {time_sec:7.1f}s ({hours}h {minutes:02d}m {seconds:02d}s) - {int(epochs)} epochs")

print("\nCentralized Statistics:")
print("-" * 70)
print(f"  Mean:        {cent_times.mean():7.1f}s ({cent_times.mean()/60:.1f} min)")
print(f"  Std Dev:     {cent_times.std():7.1f}s")
print(f"  Min:         {cent_times.min():7.1f}s ({cent_times.min()/60:.1f} min)")
print(f"  Max:         {cent_times.max():7.1f}s ({cent_times.max()/60:.1f} min)")
print(f"  Total (5 runs): {cent_times.sum():7.1f}s ({cent_times.sum()/3600:.2f} hours)")

# ============================================================================
# FEDERATED TRAINING TIMES
# ============================================================================
print("\n" + "=" * 70)
print("FEDERATED LEARNING (Seeds 44-48)")
print("=" * 70)

# Load federated training times from metadata files
fed_times = []
fed_times_formatted = []

# Check partial training archive for federated times
metadata_files = [
    "partial_training_archive/federated_pneumonia_detection_metadata_20251228_223921.json",
    "partial_training_archive/federated_pneumonia_detection_metadata_20251228_223939.json",
    "partial_training_archive/federated_pneumonia_detection_metadata_20251228_234849.json",
    "partial_training_archive/federated_pneumonia_detection_metadata_20251228_234851.json",
]

print("\nPer-Run Training Times (from available metadata):")
print("-" * 70)

for i, meta_file in enumerate(metadata_files):
    try:
        with open(BASE_DIR / "federated" / meta_file) as f:
            meta = json.load(f)
            time_sec = meta.get('training_duration_seconds', 0)
            time_fmt = meta.get('training_duration_formatted', 'N/A')
            fed_times.append(time_sec)
            fed_times_formatted.append(time_fmt)
            print(f"  Run {i+1}: {time_sec:7.1f}s ({time_fmt})")
    except FileNotFoundError:
        pass

# Also check federated_summary.json
try:
    with open(BASE_DIR / "federated" / "federated_summary.json") as f:
        summary = json.load(f)
        time_sec = summary['metrics_metadata']['training_duration_seconds']
        time_fmt = summary['metrics_metadata']['training_duration_formatted']
        fed_times.append(time_sec)
        fed_times_formatted.append(time_fmt)
        print(f"  Run {len(fed_times)}: {time_sec:7.1f}s ({time_fmt})")
except (FileNotFoundError, KeyError):
    pass

if fed_times:
    fed_times_arr = np.array(fed_times)

    print("\nFederated Statistics:")
    print("-" * 70)
    print(f"  Mean:        {fed_times_arr.mean():7.1f}s ({fed_times_arr.mean()/60:.1f} min)")
    print(f"  Std Dev:     {fed_times_arr.std():7.1f}s")
    print(f"  Min:         {fed_times_arr.min():7.1f}s ({fed_times_arr.min()/60:.1f} min)")
    print(f"  Max:         {fed_times_arr.max():7.1f}s ({fed_times_arr.max()/60:.1f} min)")
    print(f"  Total ({len(fed_times)} runs): {fed_times_arr.sum():7.1f}s ({fed_times_arr.sum()/3600:.2f} hours)")

    # Note about federated configuration
    print("\nFederated Configuration:")
    print("-" * 70)
    print("  • 5 rounds of federated aggregation")
    print("  • 2 local epochs per round per client")
    print("  • 2 clients participating")
    print("  • Total local training: 5 rounds × 2 epochs × 2 clients = 20 local epochs")
    print("  • Hardware: CPU (32-bit precision)")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("CENTRALIZED vs FEDERATED COMPARISON")
print("=" * 70)

print("\nCentralized:")
print(f"  • Average per run: {cent_times.mean():7.1f}s ({cent_times.mean()/60:.1f} min)")
print(f"  • Configuration: 10 epochs, GPU (16-bit precision)")
print(f"  • Total for 5 runs: {cent_times.sum()/3600:.2f} hours")

if fed_times:
    print("\nFederated:")
    print(f"  • Average per run: {fed_times_arr.mean():7.1f}s ({fed_times_arr.mean()/60:.1f} min)")
    print(f"  • Configuration: 5 rounds × 2 epochs, CPU (32-bit precision)")
    print(f"  • Total for {len(fed_times)} runs: {fed_times_arr.sum()/3600:.2f} hours")

    print("\nTime Difference:")
    print(f"  • Centralized is {fed_times_arr.mean()/cent_times.mean():.2f}x the time of Federated per run")
    print(f"  • Difference: {abs(cent_times.mean() - fed_times_arr.mean())/60:.1f} minutes per run")

    print("\nKey Observations:")
    print("-" * 70)
    if cent_times.mean() > fed_times_arr.mean():
        print("  ✓ Centralized takes LONGER per run due to:")
        print("    - More epochs (10 vs 5 rounds × 2 local epochs)")
        print("    - GPU overhead for smaller datasets")
        print("    - 16-bit precision training")
    else:
        print("  ✓ Federated takes LONGER per run due to:")
        print("    - Communication overhead between server and clients")
        print("    - CPU training (slower than GPU)")
        print("    - Multiple local training sessions")
else:
    print("\nFederated: Training time data not available")

# ============================================================================
# TIME PER EPOCH ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("TIME PER EPOCH ANALYSIS")
print("=" * 70)

cent_time_per_epoch = cent_times / cent_epochs
print(f"\nCentralized (10 epochs, GPU):")
print(f"  • Mean time per epoch: {cent_time_per_epoch.mean():.1f}s ({cent_time_per_epoch.mean()/60:.2f} min)")
print(f"  • Range: {cent_time_per_epoch.min():.1f}s - {cent_time_per_epoch.max():.1f}s")

if fed_times:
    # Federated: 5 rounds × 2 epochs per client × 2 clients = 20 total local epochs
    fed_total_epochs = 5 * 2  # Per client, per round
    fed_time_per_round = fed_times_arr / 5  # 5 rounds

    print(f"\nFederated (5 rounds, CPU):")
    print(f"  • Mean time per round: {fed_time_per_round.mean():.1f}s ({fed_time_per_round.mean()/60:.2f} min)")
    print(f"  • Each round = 2 local epochs per client")
    print(f"  • Estimated time per local epoch: {fed_time_per_round.mean()/2:.1f}s")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nFor 5 complete runs (seeds 44-48):")
print(f"  • Centralized: {cent_times.sum()/3600:.2f} hours total")
if fed_times:
    print(f"  • Federated: {fed_times_arr.sum()/3600:.2f} hours total")
    print(f"  • Difference: {abs(cent_times.sum() - fed_times_arr.sum())/3600:.2f} hours")

print("\n" + "=" * 70)
