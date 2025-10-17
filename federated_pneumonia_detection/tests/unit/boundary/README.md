# Unit Tests for Boundary Layer

Comprehensive test suite for the database boundary layer (SQLAlchemy ORM models and database operations).

## Overview

This directory contains unit tests for the boundary layer which handles:
- SQLAlchemy ORM models (Experiment, Run, RunConfiguration, RunMetric, RunArtifact)
- Database session management
- Database table creation and initialization

## Quick Start

### Run all boundary layer tests:
```bash
pytest federated_pneumonia_detection/tests/unit/boundary/ -v
```

### Run specific test file:
```bash
pytest federated_pneumonia_detection/tests/unit/boundary/test_engine.py -v
```

### Run specific test class:
```bash
pytest federated_pneumonia_detection/tests/unit/boundary/test_engine.py::TestExperimentModel -v
pytest federated_pneumonia_detection/tests/unit/boundary/test_engine.py::TestRunModel -v
```

### Run with minimal output:
```bash
pytest federated_pneumonia_detection/tests/unit/boundary/ -q
```

## Test Files

### test_engine.py (72 tests)
Comprehensive tests for the database engine module:
- Experiment model (7 tests)
- Run model (12 tests)
- RunConfiguration model (5 tests)
- RunMetric model (7 tests)
- RunArtifact model (6 tests)
- Model relationships (5 tests)
- Base metadata (3 tests)
- create_tables function (6 tests)
- get_session function (7 tests)
- Settings integration (2 tests)
- Column constraints (5 tests)
- Foreign key constraints (4 tests)

## Test Statistics

| Test Class | # Tests | Status |
|-----------|---------|--------|
| TestExperimentModel | 7 | ✅ PASS |
| TestRunModel | 12 | ✅ PASS |
| TestRunConfigurationModel | 5 | ✅ PASS |
| TestRunMetricModel | 7 | ✅ PASS |
| TestRunArtifactModel | 6 | ✅ PASS |
| TestModelRelationships | 5 | ✅ PASS |
| TestBaseMetadata | 3 | ✅ PASS |
| TestCreateTablesFunction | 6 | ✅ PASS |
| TestGetSessionFunction | 7 | ✅ PASS |
| TestSettingsIntegration | 2 | ✅ PASS |
| TestColumnConstraints | 5 | ✅ PASS |
| TestForeignKeyConstraints | 4 | ✅ PASS |
| **TOTAL** | **72** | **✅ ALL PASS** |

## Coverage Details

### Experiment Model
- ✅ Table name validation
- ✅ Required columns existence
- ✅ Primary key definition
- ✅ Default values (created_at)
- ✅ Relationships (runs)
- ✅ Column types (String, Text, TIMESTAMP)

### Run Model
- ✅ Table name and columns
- ✅ Primary and foreign keys
- ✅ All relationships (experiment, configuration, metrics, artifacts)
- ✅ Column types validation
- ✅ String field lengths

### RunConfiguration Model
- ✅ Table structure and columns
- ✅ Foreign key to Run
- ✅ Bidirectional relationship
- ✅ Numeric columns (Float, Integer types)
- ✅ All hyperparameter fields

### RunMetric Model
- ✅ Metrics storage structure
- ✅ Foreign key relationship
- ✅ Column type validation
- ✅ Metric metadata (name, value, step, dataset_type)

### RunArtifact Model
- ✅ Artifact storage structure
- ✅ Foreign key constraint
- ✅ Column types and lengths
- ✅ Artifact metadata fields

### Relationships
- ✅ Experiment ↔ Run (one-to-many)
- ✅ Run ↔ RunConfiguration (one-to-one)
- ✅ Run ↔ RunMetric (one-to-many)
- ✅ Run ↔ RunArtifact (one-to-many)
- ✅ Bidirectional consistency

### Database Functions
- ✅ create_tables() with mocked engine
- ✅ get_session() session creation
- ✅ Settings integration
- ✅ PostgreSQL URI handling

## Key Testing Patterns

### Model Structure Tests
- Verify table names match conventions
- Ensure all required columns exist
- Validate column types and constraints
- Check primary key definitions

### Relationship Tests
- Validate relationship definitions
- Check bidirectional relationships
- Verify one-to-one vs one-to-many

### Function Tests
- Mock external dependencies (database engine)
- Validate function calls and sequences
- Check parameter passing
- Verify return values

### Foreign Key Tests
- Ensure proper foreign key definitions
- Check referential integrity
- Validate relationship targets

## Dependencies

- pytest >= 8.4.2
- sqlalchemy >= 2.0.0
- sqlalchemy.orm
- unittest.mock (built-in)

All dependencies already available in project environment.

## Implementation Notes

### Mock Strategy
- Database engine is mocked to avoid actual database connections
- Settings are mocked to provide test database URIs
- sessionmaker is mocked for session creation tests

### Model Validation
- Column existence checked via `__table__.columns`
- Column types verified with isinstance checks
- Relationships checked via `__mapper__.relationships`
- Foreign keys verified via `.foreign_keys` attribute

### Test Organization
- Tests grouped by model class
- Relationship tests in separate class
- Function tests for create_tables and get_session
- Integration tests for Settings usage

## Running Tests in CI/CD

```yaml
# Example GitHub Actions workflow
- name: Run boundary layer tests
  run: |
    pytest federated_pneumonia_detection/tests/unit/boundary/ \
      -v \
      --tb=short \
      --junitxml=test-results.xml
```

## Troubleshooting

### Import Errors
Ensure the project is installed in editable mode:
```bash
pip install -e .
```

### SQLAlchemy Warnings
Tests may show deprecation warnings about declarative_base(). These are warnings from the boundary layer code, not test failures.

### Mock Issues
If mocking fails, ensure unittest.mock is available (part of Python standard library in 3.3+).

## Contributing

When adding new tests:
1. Follow the existing naming convention: `test_<functionality>`
2. Group related tests in classes: `Test<ModelName>`
3. Use descriptive docstrings
4. Mock external dependencies appropriately
5. Test both presence and type of attributes
6. Verify relationships are bidirectional where applicable
7. Run all tests before submitting: `pytest federated_pneumonia_detection/tests/unit/boundary/ -v`

## Model Reference

### Experiment
Represents a machine learning experiment.
- **Fields**: id, name, description, created_at
- **Relationships**: runs (one-to-many)

### Run
Represents a single training run within an experiment.
- **Fields**: id, experiment_id, training_mode, status, start_time, end_time, wandb_id, source_path
- **Relationships**: experiment, configuration, metrics, artifacts

### RunConfiguration
Stores configuration parameters for a run.
- **Fields**: learning_rate, epochs, batch_size, num_rounds, num_clients, seed, etc.
- **Relationships**: run (one-to-one)

### RunMetric
Stores metrics collected during training.
- **Fields**: metric_name, metric_value, step, dataset_type
- **Relationships**: run (many-to-one)

### RunArtifact
Stores artifacts (models, checkpoints, etc.) from a run.
- **Fields**: artifact_name, artifact_path, artifact_type
- **Relationships**: run (many-to-one)

## Notes

- All models use SQLAlchemy ORM with declarative base
- Foreign keys ensure referential integrity
- Timestamps use UTC by default
- String fields have defined lengths for database optimization
- One-to-many relationships use lists, one-to-one uses uselist=False

## See Also

- [Control Layer Tests](../control/README.md) - Tests for control layer
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Database Schema](../../../documentation/)
