# Boundary Models

This module contains the SQLAlchemy ORM models for the federated pneumonia detection system. These models define the database schema and relationships.

## Entities

### 1. Run (`run.py`)
Represents a complete training execution, which can be either centralized or federated.
- **Attributes**: `id`, `run_description`, `training_mode`, `status`, `start_time`, `end_time`, `wandb_id`, `source_path`.
- **Relationships**:
  - `metrics`: One-to-Many with `RunMetric`
  - `clients`: One-to-Many with `Client`
  - `server_evaluations`: One-to-Many with `ServerEvaluation`

### 2. Client (`client.py`)
Represents a participant in a federated learning run.
- **Attributes**: `id`, `run_id`, `client_identifier`, `created_at`, `client_config`.
- **Relationships**:
  - `run`: Many-to-One with `Run`
  - `rounds`: One-to-Many with `Round`

### 3. Round (`round.py`)
Tracks communication rounds for each client in federated learning.
- **Attributes**: `id`, `client_id`, `round_number`, `start_time`, `end_time`, `round_metadata`.
- **Relationships**:
  - `client`: Many-to-One with `Client`

### 4. RunMetric (`run_metric.py`)
Stores training metrics (loss, accuracy, etc.) per epoch or round.
- **Attributes**: `id`, `run_id`, `client_id`, `round_id`, `metric_name`, `metric_value`, `step`, `dataset_type`, `context`.
- **Relationships**:
  - `run`: Many-to-One with `Run`
  - `client`: Many-to-One with `Client` (optional)
  - `round`: Many-to-One with `Round` (optional)

### 5. ServerEvaluation (`server_evaluation.py`)
Stores server-side evaluation metrics for the global model in federated learning.
- **Attributes**: `id`, `run_id`, `round_number`, `loss`, `accuracy`, `precision`, `recall`, `f1_score`, `auroc`, `true_positives`, `true_negatives`, `false_positives`, `false_negatives`, `num_samples`, `evaluation_time`, `additional_metrics`.
- **Relationships**:
  - `run`: Many-to-One with `Run`

### 6. ChatSession (`chat_session.py`)
Stores metadata for research chat sessions.
- **Attributes**: `id` (UUID), `title`, `created_at`, `updated_at`.

## Base Configuration (`base.py`)
All models inherit from a common `Base` class created using SQLAlchemy's `declarative_base()`.

## Pydantic Models
While the boundary layer primarily uses SQLAlchemy models for database interaction, corresponding Pydantic schemas for API requests and responses are located in `src/api/endpoints/schema/`. These schemas provide validation and serialization for the entities defined here.
