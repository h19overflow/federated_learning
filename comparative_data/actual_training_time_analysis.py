"""
Actual Wall-Clock Training Time Analysis
=========================================
Calculates the real elapsed time from start to finish timestamps.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Load centralized data
cent_df = pd.read_csv(BASE_DIR / "centralized" / "experiment_results.csv")

print("=" * 80)
print("ACTUAL WALL-CLOCK TIME ANALYSIS")
print("=" * 80)

# ============================================================================
# CENTRALIZED - Calculate from timestamps
# ============================================================================
print("\n" + "=" * 80)
print("CENTRALIZED LEARNING - Wall-Clock Time")
print("=" * 80)

# Parse timestamps
cent_df['timestamp_dt'] = pd.to_datetime(cent_df['timestamp'])

# Get start and end times
start_time = cent_df['timestamp_dt'].min()
end_time = cent_df['timestamp_dt'].max()
total_elapsed = (end_time - start_time).total_seconds()

print(f"\nTimestamps for all 10 runs (seeds 42-51):")
print(f"  Start: {start_time}")
print(f"  End:   {end_time}")
print(f"  Total elapsed: {total_elapsed:.1f}s ({total_elapsed/3600:.2f} hours)")

# For seeds 44-48 specifically
seeds_5 = [44, 45, 46, 47, 48]
cent_filtered = cent_df[cent_df['seed'].isin(seeds_5)].copy()
start_time_5 = cent_filtered['timestamp_dt'].min()
end_time_5 = cent_filtered['timestamp_dt'].max()
elapsed_5 = (end_time_5 - start_time_5).total_seconds()

print(f"\nTimestamps for 5 runs (seeds 44-48):")
print(f"  Start: {start_time_5}")
print(f"  End:   {end_time_5}")
print(f"  Total elapsed: {elapsed_5:.1f}s ({elapsed_5/3600:.2f} hours)")

print("\nPer-run details (seeds 44-48):")
print("-" * 80)
for _, row in cent_filtered.iterrows():
    training_sec = row['training_time_seconds']
    print(f"  Seed {int(row['seed'])}: Finished at {row['timestamp_dt']} "
          f"(training time: {training_sec/60:.1f} min, {int(row['epochs_completed'])} epochs)")

# ============================================================================
# FEDERATED - Check if we have sequential timestamps
# ============================================================================
print("\n" + "=" * 80)
print("FEDERATED LEARNING - Wall-Clock Time")
print("=" * 80)

# Check log file for federated timing
try:
    with open(BASE_DIR / "metadata" / "experiment_run.log", 'r') as f:
        lines = f.readlines()

    # Find federated start and end times
    fed_start = None
    fed_end = None
    fed_runs_completed = 0

    for line in lines:
        if "PHASE 2: Federated Learning" in line:
            # Extract timestamp from log line
            timestamp_str = line.split(' - ')[0]
            fed_start = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
        if "[Federated Run" in line and "Starting with seed=" in line:
            fed_runs_completed += 1
        if "Experiment orchestrator completed" in line or "All experiments completed" in line:
            timestamp_str = line.split(' - ')[0]
            fed_end = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')

    if fed_start:
        print(f"\nFederated experiment start: {fed_start}")
        print(f"Federated runs detected in log: {fed_runs_completed}")

        if fed_end:
            print(f"Federated experiment end: {fed_end}")
            fed_elapsed = (fed_end - fed_start).total_seconds()
            print(f"Total elapsed: {fed_elapsed:.1f}s ({fed_elapsed/3600:.2f} hours)")
        else:
            print("End time not found in log (experiment might still be running or log incomplete)")
    else:
        print("Federated timing not found in experiment log")

except FileNotFoundError:
    print("Experiment log file not found")

# ============================================================================
# Training time from metadata
# ============================================================================
print("\n" + "=" * 80)
print("INDIVIDUAL RUN TIMES (from training_time_seconds)")
print("=" * 80)

print("\nCentralized (seeds 44-48):")
cent_times = cent_filtered['training_time_seconds'].values
print(f"  Total training time (sum): {cent_times.sum()/3600:.2f} hours")
print(f"  Average per run: {cent_times.mean()/60:.1f} minutes")

print("\nFederated (from metadata files):")
import json
fed_times = []
metadata_files = [
    "partial_training_archive/federated_pneumonia_detection_metadata_20251228_223921.json",
    "partial_training_archive/federated_pneumonia_detection_metadata_20251228_223939.json",
    "partial_training_archive/federated_pneumonia_detection_metadata_20251228_234849.json",
    "partial_training_archive/federated_pneumonia_detection_metadata_20251228_234851.json",
]

for meta_file in metadata_files:
    try:
        with open(BASE_DIR / "federated" / meta_file) as f:
            meta = json.load(f)
            time_sec = meta.get('training_duration_seconds', 0)
            fed_times.append(time_sec)
    except FileNotFoundError:
        pass

try:
    with open(BASE_DIR / "federated" / "federated_summary.json") as f:
        summary = json.load(f)
        time_sec = summary['metrics_metadata']['training_duration_seconds']
        fed_times.append(time_sec)
except (FileNotFoundError, KeyError):
    pass

if fed_times:
    import numpy as np
    fed_times_arr = np.array(fed_times)
    print(f"  Total training time ({len(fed_times)} runs, sum): {fed_times_arr.sum()/3600:.2f} hours")
    print(f"  Average per run: {fed_times_arr.mean()/60:.1f} minutes")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\n📊 CENTRALIZED (10 runs total, seeds 42-51):")
print(f"   Wall-clock time: {total_elapsed/3600:.2f} hours")
print(f"   Individual training times sum: {cent_df['training_time_seconds'].sum()/3600:.2f} hours")
print(f"   Overhead (setup, data loading, etc.): {(total_elapsed - cent_df['training_time_seconds'].sum())/3600:.2f} hours")

print("\n📊 CENTRALIZED (5 runs, seeds 44-48 only):")
print(f"   Wall-clock time: {elapsed_5/3600:.2f} hours")
print(f"   Individual training times sum: {cent_times.sum()/3600:.2f} hours")

print("\n" + "=" * 80)
print("\n💡 KEY INSIGHT:")
print("   The 'training_time_seconds' only captures actual training loop time.")
print("   Wall-clock time includes:")
print("   - Data loading and preprocessing")
print("   - Model initialization")
print("   - Validation runs")
print("   - Checkpoint saving")
print("   - System overhead")
print("=" * 80)
