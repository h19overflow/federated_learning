from flwr.app import ArrayRecord, Context
from flwr.serverapp import ServerApp, Grid
from pathlib import Path
from datetime import datetime
from federated_pneumonia_detection.src.control.federated_new_version.core.custom_strategy import (
    ConfigurableFedAvg,
)
from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet import (
    LitResNet,
)
from federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment import (
    update_flwr_config,
)
from federated_pneumonia_detection.src.control.federated_new_version.core.utils import (
    read_configs_to_toml,
)
from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.control.dl_model.utils.data.websocket_metrics_sender import (
    MetricsWebSocketSender,
)
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.control.federated_new_version.core.server_evaluation import (
    create_central_evaluate_fn,
)

app = ServerApp()


# TODO: When invoking make sure to add an update to the file path configuration.#
# TODO: Context objects in the client and server classes contains run id , node id , so adjust it such that we persist them to the db.
@app.lifespan()
def lifespan(app: ServerApp):
    print("Server is starting...")
    flwr_configs = read_configs_to_toml()
    if flwr_configs:
        update_flwr_config(**flwr_configs)
    yield
    print("Server is stopping...")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    num_rounds: int = context.run_config["num-server-rounds"]  # Read run config
    num_clients: int = len(list(grid.get_node_ids()))

    # Create the run in database BEFORE training starts
    print("[Server] Creating federated training run in database...")
    db = get_session()
    try:
        run_data = {
            "training_mode": "federated",
            "status": "in_progress",
            "start_time": datetime.now(),
            "wandb_id": f"federated_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "source_path": "federated_training",
        }
        new_run = run_crud.create(db, **run_data)
        db.commit()
        run_id = new_run.id
        print(f"[Server] ✅ Created run with id={run_id}")
    except Exception as e:
        print(f"[Server] ❌ Failed to create run: {e}")
        db.rollback()
        run_id = None
    finally:
        db.close()

    # Create training configuration to send to clients
    train_config = {
        "file_path": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\stage2_train_metadata.csv",
        "image_dir": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\Images",
        "num_partitions": num_clients,
        "run_id": run_id,  # Pass run_id to clients so they don't create their own
    }

    # Create evaluation configuration to send to clients
    eval_config = {
        "csv_path": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\stage2_train_metadata.csv",
        "image_dir": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\Images",
    }

    # Load global model with ConfigManager
    config_manager = ConfigManager(
        config_path=str(
            Path(__file__).parent.parent.parent.parent.parent
            / "config"
            / "default_config.yaml"
        )
    )
    global_model = LitResNet(config=config_manager)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize WebSocket sender to broadcast training mode
    ws_sender = MetricsWebSocketSender("ws://localhost:8765")

    # Signal to frontend that this is federated training
    print(
        f"[Server] Starting federated training with {num_clients} clients for {num_rounds} rounds"
    )
    ws_sender.send_training_mode(
        is_federated=True, num_rounds=num_rounds, num_clients=num_clients
    )

    # Initialize ConfigurableFedAvg strategy with configs
    strategy = ConfigurableFedAvg(
        fraction_train=1,
        fraction_evaluate=1,
        train_config=train_config,
        eval_config=eval_config,
        websocket_uri="ws://localhost:8765",
        run_id=run_id,  # Pass run_id for database persistence
    )

    # Set total rounds for progress tracking in strategy
    strategy.set_total_rounds(num_rounds)

    # Create centralized evaluation function for server-side evaluation
    # This evaluates the global model on a held-out test set after each round
    central_evaluate_fn = create_central_evaluate_fn(
        config_manager=config_manager,
        csv_path=train_config["file_path"],
        image_dir=train_config["image_dir"],
    )

    # Start strategy, run FedAvg for `num_rounds`
    # The strategy will automatically broadcast metrics after each round via aggregate_evaluate
    # Server-side evaluation will run after each round using central_evaluate_fn
    print(f"[Server] Starting federated learning for {num_rounds} rounds")
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=central_evaluate_fn,  # Enable server-side centralized evaluation
    )
    print("[Server] Federated learning completed successfully")

    # Persist all federated metrics from Result object to database
    if run_id:
        print("[Server] Persisting federated metrics from Result object...")
        db = get_session()
        try:
            # Persist client-side aggregated evaluation metrics
            if result.evaluate_metrics_clientapp:
                for (
                    round_num,
                    metric_record,
                ) in result.evaluate_metrics_clientapp.items():
                    # Use strategy's extraction method to normalize metric names
                    normalized_metrics = strategy._extract_round_metrics(
                        dict(metric_record)
                    )

                    # Convert to flattened dict with global_ prefix
                    flattened_metrics = {"epoch": round_num}
                    for metric_name, metric_value in normalized_metrics.items():
                        flattened_metrics[f"global_{metric_name}"] = float(metric_value)

                    print(
                        f"[Server] Round {round_num} client-aggregated metrics: {flattened_metrics}"
                    )

                    # Persist metrics for this round
                    run_crud.persist_metrics(
                        db=db,
                        run_id=run_id,
                        epoch_metrics=[flattened_metrics],
                        federated_context={
                            "is_global": True,
                            "round": round_num,
                            "source": "client_aggregated",
                        },
                    )

            # Persist server-side centralized evaluation metrics
            if result.evaluate_metrics_serverapp:
                for (
                    round_num,
                    metric_record,
                ) in result.evaluate_metrics_serverapp.items():
                    # Server metrics already have server_ prefix
                    flattened_metrics = {"epoch": round_num}
                    for metric_name, metric_value in metric_record.items():
                        flattened_metrics[metric_name] = float(metric_value)

                    print(
                        f"[Server] Round {round_num} server-evaluated metrics: {flattened_metrics}"
                    )

                    # Persist server-side metrics
                    run_crud.persist_metrics(
                        db=db,
                        run_id=run_id,
                        epoch_metrics=[flattened_metrics],
                        federated_context={
                            "is_global": True,
                            "round": round_num,
                            "source": "server_centralized",
                        },
                    )

            db.commit()
            print(
                f"[Server] ✅ Persisted metrics for client-side: {len(result.evaluate_metrics_clientapp)}, "
                f"server-side: {len(result.evaluate_metrics_serverapp)} rounds"
            )
        except Exception as e:
            print(f"[Server] ❌ Error persisting metrics: {e}")
            db.rollback()
        finally:
            db.close()

    # Save final model to disk
    print("\nSaving final model to disk...")
    _ = result.arrays.to_torch_state_dict()  # Future: Save model checkpoint
