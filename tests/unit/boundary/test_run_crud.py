import pytest
from datetime import datetime
from sqlalchemy.orm import Session
from federated_pneumonia_detection.src.boundary.models import Run
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud


@pytest.mark.unit
def test_create_run(db_session: Session):
    """
    Test creating a run and verify all fields are saved correctly.
    """
    start_time = datetime(2025, 1, 1, 12, 0, 0)
    run_data = {
        "run_description": "Test Run Description",
        "training_mode": "centralized",
        "status": "in_progress",
        "start_time": start_time,
        "wandb_id": "test_wandb_123",
        "source_path": "/data/test/path",
    }

    # Create run
    run = run_crud.create(db_session, **run_data)

    # Assertions
    assert run.id is not None
    assert run.run_description == run_data["run_description"]
    assert run.training_mode == run_data["training_mode"]
    assert run.status == run_data["status"]
    assert run.start_time == start_time
    assert run.wandb_id == run_data["wandb_id"]
    assert run.source_path == run_data["source_path"]


@pytest.mark.unit
def test_get_run_by_id_success(db_session: Session):
    """
    Test retrieving an existing run by its ID.
    """
    # Setup: create a run
    run = run_crud.create(
        db_session,
        run_description="Searchable Run",
        training_mode="federated",
        status="in_progress",
        start_time=datetime.now(),
    )

    # Execution
    fetched_run = run_crud.get(db_session, run.id)

    # Assertions
    assert fetched_run is not None
    assert fetched_run.id == run.id
    assert fetched_run.run_description == "Searchable Run"
    assert fetched_run.training_mode == "federated"


@pytest.mark.unit
def test_get_run_by_id_not_found(db_session: Session):
    """
    Test that retrieving a non-existent run returns None.
    """
    # Execution
    fetched_run = run_crud.get(db_session, 9999)

    # Assertions
    assert fetched_run is None


@pytest.mark.unit
def test_get_runs_pagination(db_session: Session):
    """
    Test pagination using skip and limit parameters.
    """
    # Setup: create 5 runs
    for i in range(5):
        run_crud.create(
            db_session,
            run_description=f"Pagination Run {i}",
            training_mode="centralized",
            status="completed",
            start_time=datetime(2025, 1, 1, 12, i, 0),
        )

    # Execution: skip 2, limit 2
    runs = run_crud.get_multi(db_session, skip=2, limit=2)

    # Assertions
    assert len(runs) == 2
    # Assuming insertion order for SQLite without explicit order_by
    assert runs[0].run_description == "Pagination Run 2"
    assert runs[1].run_description == "Pagination Run 3"


@pytest.mark.unit
def test_update_run_status(db_session: Session):
    """
    Test updating the status of an existing run.
    """
    # Setup: create a run with 'in_progress' status
    run = run_crud.create(
        db_session,
        run_description="Status Update Run",
        training_mode="centralized",
        status="in_progress",
        start_time=datetime.now(),
    )

    # Execution: update status to 'completed'
    updated_run = run_crud.update_status(db_session, run.id, "completed")

    # Assertions
    assert updated_run is not None
    assert updated_run.status == "completed"

    # Verify persistence in DB
    db_run = run_crud.get(db_session, run.id)
    assert db_run.status == "completed"
