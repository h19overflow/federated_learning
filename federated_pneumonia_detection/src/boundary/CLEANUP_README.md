# Database Cleanup Utility

## Overview
This utility provides safe database cleanup operations for development and testing.

## Location
`federated_pneumonia_detection/src/boundary/cleanup_database.py`

---

## Usage

### Interactive Mode (Recommended)
Run the script without arguments to get an interactive menu:
```bash
python -m federated_pneumonia_detection.src.boundary.cleanup_database
```

**Menu Options:**
1. **Show current record counts** - View how many records exist in each table
2. **Delete all records** - Remove all data but keep table structure
3. **Delete only centralized runs** - Remove centralized training runs and their metrics
4. **Delete only federated runs** - Remove federated training runs and their metrics
5. **Reset database** - Drop and recreate all tables (complete reset)
6. **Exit** - Close the utility

---

### Command-Line Mode
Run specific operations directly:

```bash
# Show record counts
python -m federated_pneumonia_detection.src.boundary.cleanup_database show

# Delete all records (keeps schema)
python -m federated_pneumonia_detection.src.boundary.cleanup_database delete-all

# Delete only centralized runs
python -m federated_pneumonia_detection.src.boundary.cleanup_database delete-centralized

# Delete only federated runs
python -m federated_pneumonia_detection.src.boundary.cleanup_database delete-federated

# Reset database (drop and recreate tables)
python -m federated_pneumonia_detection.src.boundary.cleanup_database reset
```

---

## Safety Features

### Confirmation Prompts
All destructive operations require explicit confirmation by typing `yes`:
```
⚠️  WARNING: This will DELETE ALL RECORDS from the database.
Type 'yes' to confirm: yes
```

### Deletion Order
Records are deleted in the correct order to respect foreign key constraints:
1. `run_metrics` (depends on runs, clients, rounds)
2. `server_evaluations` (depends on runs)
3. `rounds` (depends on clients)
4. `clients` (depends on runs)
5. `runs` (no dependencies)

---

## Use Cases

### Clean Up Test Data
After running experiments, remove all records:
```bash
python -m federated_pneumonia_detection.src.boundary.cleanup_database delete-all
```

### Remove Failed Runs
Delete only centralized runs if they're cluttering the database:
```bash
python -m federated_pneumonia_detection.src.boundary.cleanup_database delete-centralized
```

### Fresh Start
Reset the database completely before production deployment:
```bash
python -m federated_pneumonia_detection.src.boundary.cleanup_database reset
```

### Check Database State
See current record counts:
```bash
python -m federated_pneumonia_detection.src.boundary.cleanup_database show
```

Output:
```
--- Current Record Counts ---
runs: 42
clients: 15
rounds: 225
run_metrics: 1,250
server_evaluations: 75
```

---

## Functions Available for Scripting

You can also import functions directly in your Python scripts:

```python
from federated_pneumonia_detection.src.boundary.cleanup_database import (
    delete_all_records,
    delete_runs_by_mode,
    reset_database,
    show_record_counts
)

# Delete all records
delete_all_records()

# Delete only federated runs
delete_runs_by_mode('federated')

# Show counts
show_record_counts()

# Reset database
reset_database()
```

---

## Warning

**This script performs IRREVERSIBLE operations!**

- Always confirm you're targeting the correct database (check your `.env` file)
- Never run in production without a backup
- Test on development database first

---

## Logging

All operations are logged to the application logger:
```
INFO - Starting database cleanup - deleting all records...
INFO - Deleting run_metrics...
INFO - Deleting server_evaluations...
INFO - Deleting rounds...
INFO - Deleting clients...
INFO - Deleting runs...
INFO - ✓ All records deleted successfully
```

---

## Error Handling

If an error occurs during deletion:
- Transaction is rolled back
- Database remains in consistent state
- Error is logged with full traceback
- Script exits with error message

---

## Related Files

- `engine.py` - Database connection and session management
- `models/` - Database model definitions
- `CRUD/` - Data access operations
