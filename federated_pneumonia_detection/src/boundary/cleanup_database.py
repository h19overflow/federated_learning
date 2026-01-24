"""
Database cleanup script for development/testing.

This script provides utilities to clean up database records or reset the database schema.
Use with caution - this will delete data permanently!
"""

import sys
from logging import getLogger
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.boundary.engine import (
    Base,
    create_tables,
    get_engine,
    get_session,
)

logger = getLogger(__name__)


def confirm_action(message: str) -> bool:
    """
    Prompt user for confirmation before destructive actions.

    Args:
        message: Confirmation message to display

    Returns:
        True if user confirms, False otherwise
    """
    print(f"\n WARNING: {message}")
    response = input("Type 'yes' to confirm: ").strip().lower()
    return response == "yes"


def delete_all_records(db: Optional[Session] = None) -> None:
    """
    Delete all records from all tables in the correct order (respecting foreign keys).

    This preserves the table schema but removes all data.

    Args:
        db: Optional database session. If None, creates a new session.
    """
    if not confirm_action("This will DELETE ALL RECORDS from the database."):
        logger.info("Operation cancelled by user")
        return

    close_session = False
    if db is None:
        db = get_session()
        close_session = True

    try:
        logger.info("Starting database cleanup - deleting all records...")

        # Delete in reverse order of dependencies to avoid foreign key violations

        # 1. Delete run_metrics (depends on runs, clients, rounds)
        logger.info("Deleting run_metrics...")
        db.execute(text("DELETE FROM run_metrics"))

        # 2. Delete server_evaluations (depends on runs)
        logger.info("Deleting server_evaluations...")
        db.execute(text("DELETE FROM server_evaluations"))

        # 3. Delete rounds (depends on clients)
        logger.info("Deleting rounds...")
        db.execute(text("DELETE FROM rounds"))

        # 4. Delete clients (depends on runs)
        logger.info("Deleting clients...")
        db.execute(text("DELETE FROM clients"))

        # 5. Delete runs (no dependencies)
        logger.info("Deleting runs...")
        db.execute(text("DELETE FROM runs"))

        # Commit all deletions
        db.commit()
        logger.info("✓ All records deleted successfully")

        # Show record counts (should all be 0)
        print("\n--- Record Counts After Cleanup ---")
        for table in ["runs", "clients", "rounds", "run_metrics", "server_evaluations"]:
            # Table names are from hardcoded whitelist, not user input
            result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))  # nosec B608
            count = result.scalar()
            print(f"{table}: {count}")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        db.rollback()
        raise
    finally:
        if close_session:
            db.close()


def reset_database() -> None:
    """
    Drop all tables and recreate them from scratch.

    This completely resets the database schema and all data.
    """
    if not confirm_action(
        "This will DROP ALL TABLES and RECREATE them. All data will be permanently lost.",
    ):
        logger.info("Operation cancelled by user")
        return

    try:
        logger.info("Dropping all tables...")
        engine = get_engine()
        Base.metadata.drop_all(engine)
        logger.info("✓ All tables dropped")

        logger.info("Recreating tables...")
        create_tables()
        logger.info("✓ All tables recreated")

        logger.info("✓ Database reset complete")

    except Exception as e:
        logger.error(f"Error during database reset: {e}")
        raise


def show_record_counts() -> None:
    """
    Display current record counts for all tables.
    """
    db = get_session()

    try:
        print("\n--- Current Record Counts ---")
        tables = ["runs", "clients", "rounds", "run_metrics", "server_evaluations"]

        for table in tables:
            # Table names are from hardcoded whitelist, not user input
            result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))  # nosec B608
            count = result.scalar()
            print(f"{table}: {count:,}")

    except Exception as e:
        logger.error(f"Error retrieving record counts: {e}")
        raise
    finally:
        db.close()


def delete_runs_by_mode(training_mode: str) -> None:
    """
    Delete all runs of a specific training mode and their related records.

    Args:
        training_mode: Either 'centralized' or 'federated'
    """
    if training_mode not in ["centralized", "federated"]:
        logger.error(
            f"Invalid training_mode: {training_mode}. Must be 'centralized' or 'federated'",
        )
        return

    if not confirm_action(
        f"This will DELETE ALL {training_mode.upper()} RUNS and their related records.",
    ):
        logger.info("Operation cancelled by user")
        return

    db = get_session()

    try:
        logger.info(f"Deleting all {training_mode} runs...")

        # Get run IDs to delete
        result = db.execute(
            text("SELECT id FROM runs WHERE training_mode = :mode"),
            {"mode": training_mode},
        )
        run_ids = [row[0] for row in result]

        if not run_ids:
            logger.info(f"No {training_mode} runs found")
            return

        logger.info(f"Found {len(run_ids)} {training_mode} runs to delete")

        # Delete related records first (foreign key constraints)
        # Use parameterized query with tuple binding for IN clause (defense-in-depth)
        run_ids_tuple = tuple(run_ids) if len(run_ids) > 1 else (run_ids[0],)

        logger.info("Deleting run_metrics...")
        db.execute(
            text("DELETE FROM run_metrics WHERE run_id IN :run_ids"),
            {"run_ids": run_ids_tuple},
        )

        logger.info("Deleting server_evaluations...")
        db.execute(
            text("DELETE FROM server_evaluations WHERE run_id IN :run_ids"),
            {"run_ids": run_ids_tuple},
        )

        if training_mode == "federated":
            # Delete federated-specific records
            logger.info("Deleting rounds...")
            db.execute(
                text("""
                DELETE FROM rounds
                WHERE client_id IN (
                    SELECT id FROM clients WHERE run_id IN :run_ids
                )
            """),
                {"run_ids": run_ids_tuple},
            )

            logger.info("Deleting clients...")
            db.execute(
                text("DELETE FROM clients WHERE run_id IN :run_ids"),
                {"run_ids": run_ids_tuple},
            )

        logger.info(f"Deleting {training_mode} runs...")
        db.execute(
            text("DELETE FROM runs WHERE id IN :run_ids"),
            {"run_ids": run_ids_tuple},
        )

        db.commit()
        logger.info(f"✓ Deleted {len(run_ids)} {training_mode} runs successfully")

    except Exception as e:
        logger.error(f"Error deleting {training_mode} runs: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def interactive_menu() -> None:
    """
    Interactive menu for database cleanup operations.
    """
    while True:
        print("\n" + "=" * 50)
        print("DATABASE CLEANUP UTILITY")
        print("=" * 50)
        print("\n1. Show current record counts")
        print("2. Delete all records (preserve schema)")
        print("3. Delete only centralized runs")
        print("4. Delete only federated runs")
        print("5. Reset database (drop and recreate tables)")
        print("6. Exit")
        print()

        choice = input("Enter your choice (1-6): ").strip()

        if choice == "1":
            show_record_counts()
        elif choice == "2":
            delete_all_records()
        elif choice == "3":
            delete_runs_by_mode("centralized")
        elif choice == "4":
            delete_runs_by_mode("federated")
        elif choice == "5":
            reset_database()
        elif choice == "6":
            print("\nExiting...")
            break
        else:
            print("\n❌ Invalid choice. Please enter a number between 1 and 6.")


if __name__ == "__main__":
    # If script is run with arguments, execute directly
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "show":
            show_record_counts()
        elif command == "delete-all":
            delete_all_records()
        elif command == "delete-centralized":
            delete_runs_by_mode("centralized")
        elif command == "delete-federated":
            delete_runs_by_mode("federated")
        elif command == "reset":
            reset_database()
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  show                 - Show record counts")
            print("  delete-all          - Delete all records")
            print("  delete-centralized  - Delete centralized runs only")
            print("  delete-federated    - Delete federated runs only")
            print("  reset               - Drop and recreate all tables")
    else:
        # No arguments - run interactive menu
        interactive_menu()
