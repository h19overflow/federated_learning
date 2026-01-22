import logging
from datetime import datetime

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.control.federated_new_version.core.server_evaluation import (
    create_central_evaluate_fn,
)
from federated_pneumonia_detection.src.control.federated_new_version.core.utils import (
    _build_global_model,
    _build_training_configs,
    _convert_metric_record_to_dict,
    _initialize_database_run,
    _initialize_strategy,
    _initialize_websocket_sender,
    _persist_server_evaluations,
    _setup_config_manager,
    read_configs_to_toml,
)
from federated_pneumonia_detection.src.internals.loggers.logger import setup_logger

# Setup logger for server app
logger = setup_logger(__name__)
logger.setLevel(logging.ERROR)

app = ServerApp()


@app.lifespan()
def lifespan(app: ServerApp):
    """Lifecycle management for ServerApp."""
    logger.info("=" * 80)
    logger.info("SERVER LIFESPAN: Starting up...")
    logger.info("=" * 80)

    logger.info("Verifying configuration synchronization...")
    flwr_configs = read_configs_to_toml()

    if flwr_configs:
        logger.info(f"[OK] Configuration verified: {flwr_configs}")
    else:
        logger.warning("[WARN] No federated configs found in default_config.yaml")

    yield

    logger.info("=" * 80)
    logger.info("SERVER LIFESPAN: Shutting down...")
    logger.info("=" * 80)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    num_rounds: int = context.run_config["num-server-rounds"]
    num_clients: int = len(list(grid.get_node_ids()))

    logger.info("=" * 80)
    logger.info("FEDERATED LEARNING SESSION STARTING")
    logger.info("=" * 80)
    logger.info(f"Configuration: {num_clients} clients, {num_rounds} rounds")
    logger.info(f"Context run_config: {context.run_config}")

    run_id, _ = _initialize_database_run()
    config_manager, experiment_seed, analysis_run_id = _setup_config_manager()
    train_config, eval_config = _build_training_configs(
        config_manager, num_clients, run_id, experiment_seed
    )
    global_model, arrays = _build_global_model(config_manager)
    ws_sender = _initialize_websocket_sender(num_clients, num_rounds)

    logger.info("Creating server-side evaluation function...")
    central_evaluate_fn = create_central_evaluate_fn(
        config_manager=config_manager,
        csv_path=config_manager.get("experiment.file-path"),
        image_dir=config_manager.get("experiment.image-dir"),
    )
    logger.info("[OK] Server evaluation function created")

    strategy = _initialize_strategy(train_config, eval_config, run_id, num_rounds)

    logger.info("Starting federated learning...")
    try:
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=num_rounds,
            evaluate_fn=central_evaluate_fn,
        )
        logger.info("Federated learning completed.")

        if run_id:
            logger.info("Marking federated run as completed in database...")
            db = get_session()
            try:
                run_crud.complete_run(db, run_id=run_id, status="completed")
                db.commit()
                logger.info(f"[OK] Run {run_id} marked as completed with end_time")
            except Exception as e:
                logger.error(f"[ERROR] Failed to update run completion: {e}")
                db.rollback()
            finally:
                db.close()

    except Exception as e:
        logger.error(f"Federated learning failed: {str(e)}", exc_info=True)

        if run_id:
            logger.error(f"Marking run {run_id} as failed in database...")
            db = get_session()
            try:
                run_crud.complete_run(db, run_id=run_id, status="failed")
                db.commit()
                logger.error(f"[OK] Run {run_id} marked as failed")
            except Exception as db_error:
                logger.error(f"[ERROR] Failed to mark run as failed: {db_error}")
                db.rollback()
            finally:
                db.close()

        raise

    all_results = {}

    if result.train_metrics_clientapp:
        all_results["train_metrics_clientapp"] = _convert_metric_record_to_dict(
            result.train_metrics_clientapp,
        )

    if result.evaluate_metrics_clientapp:
        all_results["evaluate_metrics_clientapp"] = _convert_metric_record_to_dict(
            result.evaluate_metrics_clientapp,
        )

    if result.evaluate_metrics_serverapp:
        all_results["evaluate_metrics_serverapp"] = _convert_metric_record_to_dict(
            result.evaluate_metrics_serverapp,
        )

    result_file_id = analysis_run_id if analysis_run_id else run_id
    with open(f"results_{result_file_id}.json", "w") as f:
        import json

        json.dump(all_results, f, indent=2)
    logger.info(f"[OK] Results saved to results_{result_file_id}.json")

    if result.evaluate_metrics_serverapp and run_id:
        logger.info("Persisting server evaluation metrics to database...")
        _persist_server_evaluations(run_id, result.evaluate_metrics_serverapp)
    else:
        logger.error(
            f"[WARN] Skipping server evaluation persistence: "
            f"evaluate_metrics_serverapp={bool(result.evaluate_metrics_serverapp)}, "
            f"run_id={run_id}",
        )

    logger.info(
        "All federated rounds complete. Sending training_end event to frontend...",
    )

    best_epoch = 1
    best_val_recall = 0.0
    if result.evaluate_metrics_serverapp:
        server_metrics = _convert_metric_record_to_dict(
            result.evaluate_metrics_serverapp,
        )
        if "loss" in server_metrics and "recall" in server_metrics:
            recalls = server_metrics["recall"]
            if recalls:
                best_epoch = recalls.index(max(recalls)) + 1
                best_val_recall = max(recalls)

    training_duration = None
    if run_id:
        db = get_session()
        try:
            run = db.query(run_crud.model).filter(run_crud.model.id == run_id).first()
            if run and run.start_time and run.end_time:
                duration_seconds = (run.end_time - run.start_time).total_seconds()
                duration_minutes = duration_seconds / 60
                training_duration = f"{duration_minutes:.2f}m"
        except Exception as e:
            logger.warning(f"Failed to calculate training duration: {e}")
        finally:
            db.close()

    ws_sender.send_metrics(
        {
            "run_id": run_id,
            "status": "completed",
            "experiment_name": f"federated_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "total_epochs": num_rounds,
            "training_mode": "federated",
            "best_epoch": best_epoch,
            "best_val_recall": best_val_recall,
            "training_duration": training_duration,
        },
        "training_end",
    )
    logger.info(f"[OK] Training complete notification sent (run_id={run_id})")
