from flwr.app import ArrayRecord, Context
from flwr.serverapp import ServerApp, Grid
from pathlib import Path
from federated_pneumonia_detection.src.control.federated_new_version.core.custom_strategy import ConfigurableFedAvg
from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet import (
    LitResNet,
)
from federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment import (
    update_flwr_config,
)
from federated_pneumonia_detection.src.control.federated_new_version.core.utils import read_configs_to_toml
from federated_pneumonia_detection.config.config_manager import ConfigManager

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
    
    # Create training configuration to send to clients
    train_config = {
        "file_path": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\stage2_train_metadata.csv",
        "image_dir": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\Images",
        "num_partitions": len(list(grid.get_node_ids())),
    }
    
    # Create evaluation configuration to send to clients
    eval_config = {
        "csv_path": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\stage2_train_metadata.csv",
        "image_dir": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\Images",
    }
    
    # Load global model with ConfigManager
    config_manager = ConfigManager(config_path=str(Path(__file__).parent.parent.parent.parent.parent / "config" / "default_config.yaml"))
    global_model = LitResNet(config=config_manager)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize ConfigurableFedAvg strategy with configs
    strategy = ConfigurableFedAvg(
        fraction_train=1,
        fraction_evaluate=1,
        train_config=train_config,
        eval_config=eval_config,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()


