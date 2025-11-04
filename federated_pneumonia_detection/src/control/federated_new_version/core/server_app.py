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

    # Start strategy, run FedAvg for `num_rounds`
    # The strategy will automatically broadcast metrics after each round via aggregate_evaluate
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    _ = result.arrays.to_torch_state_dict()  # Future: Save model checkpoint
