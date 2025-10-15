# Running Experiments - Quick Guide

## Setup Validation

First, validate your setup:
```bash
python validate_setup.py
```

This will verify all imports and dependencies are working.

## Running Experiments

### Option 1: Complete Comparison (Recommended)
Runs both centralized and federated learning, then compares results:

```bash
python run_experiments.py --data-path path/to/your/data.zip --mode comparison
```

### Option 2: Centralized Only
```bash
python run_experiments.py --data-path path/to/your/data.zip --mode centralized
```

### Option 3: Federated Only
```bash
python run_experiments.py --data-path path/to/your/data.zip --mode federated --partition-strategy iid
```

## Partition Strategies

- `iid` - Random equal distribution (default)
- `non-iid` - Patient-based partitioning (realistic for medical data)
- `stratified` - Class-balanced distribution

Example:
```bash
python run_experiments.py --data-path data.zip --mode federated --partition-strategy non-iid
```

## Custom Configuration

Use a custom config file:
```bash
python run_experiments.py --data-path data.zip --config config/custom.yaml --mode comparison
```

## Output Location

By default, results are saved to `experiment_results/experiment_TIMESTAMP/`

Custom output directory:
```bash
python run_experiments.py --data-path data.zip --output-dir my_results --mode comparison
```

## Analyzing Results

After training completes, analyze the results:
```bash
# Analyze latest experiment
python analyze_results.py

# Analyze specific experiment
python analyze_results.py --experiment-dir experiment_results/experiment_20250115_123456
```

## Complete Example

```bash
# 1. Validate setup
python validate_setup.py

# 2. Run comparison with custom config
python run_experiments.py \
    --data-path /path/to/chest_xray_data.zip \
    --mode comparison \
    --config config/my_config.yaml \
    --partition-strategy non-iid \
    --output-dir pneumonia_experiments

# 3. Analyze results
python analyze_results.py
```

## What Gets Logged

All experiments create:
- `experiment.log` - Complete training log
- `centralized/results.json` - Centralized training results
- `centralized/checkpoints/` - Model checkpoints
- `centralized/logs/` - Training logs
- `federated/results.json` - Federated training results
- `federated/checkpoints/` - Federated model checkpoints
- `federated/logs/` - Federated training logs
- `comparison_report.json` - Comparison summary (if mode=comparison)
- `analysis_report.json` - Analysis results (after running analyze_results.py)

## Troubleshooting

**Import errors?**
Run `python validate_setup.py` to check dependencies.

**Out of memory?**
Reduce `batch_size` in config or use fewer federated clients.

**Data not found?**
Ensure your data path is correct and accessible.

## Help

```bash
python run_experiments.py --help
python analyze_results.py --help
```
