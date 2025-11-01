from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import ServerApp, Grid
from federated_pneumonia_detection.src.control.federated_new_version.core.custom_strategy import ConfigurableFedAvg
from torch.compiler import config
from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet import (
    LitResNet,
)
from federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment import (
    update_flwr_config,
)
from federated_pneumonia_detection.config.config_manager import ConfigManager

app = ServerApp()

# TODO: File "C:\Users\User\Projects\FYP2\federated_pneumonia_detection\src\utils\config_loader.py", line 144, in create_experiment_config
#     return ExperimentConfig(
#            ^^^^^^^^^^^^^^^^^
#   File "<string>", line 45, in __init__
#   File "C:\Users\User\Projects\FYP2\federated_pneumonia_detection\models\experiment_config.py", line 81, in __post_init__
#     self._validate_parameters()
#   File "C:\Users\User\Projects\FYP2\federated_pneumonia_detection\models\experiment_config.py", line 107, in _validate_parameters
#     raise ValueError("Clients per round cannot exceed total number of clients")
# ValueError: Clients per round cannot exceed total number of clients
# 2025-11-01 15:29:07 - datasets - INFO - PyTorch version 2.8.0+cu128 available. - config.py - 54
# TODO: Make sure frontend sends the correct updates to default-yaml
# since the keys have been adjusted to refelect the convention
# TODO: When invoking make sure to add an update to the file path configuration.
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
    config_manager = ConfigManager(
        config_path=r"federated_pneumonia_detection\config\default_config.yaml"
    )

    num_rounds: int = context.run_config["num-server-rounds"]  # Read run config
    
    # Create training configuration to send to clients
    train_config = {
        "file_path": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\stage2_train_metadata.csv",
        "image_dir": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\Images",
        "num_partitions": 4,
        "partition_id": 0,
        "run_id": 5,
    }
    
    # Create evaluation configuration to send to clients
    eval_config = {
        "csv_path": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\stage2_train_metadata.csv",
        "image_dir": r"C:\Users\User\Projects\FYP2\Training_Sample_5pct\Images",
        "run_id": 5,
    }
    
    # Load global model with required arguments
    from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader
    config_loader = ConfigLoader()
    consifgs = config_loader.load_config()
    constants = config_loader.create_system_constants(consifgs)
    config = config_loader.create_experiment_config(consifgs)
    
    global_model = LitResNet(constants=constants, config=config)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize ConfigurableFedAvg strategy with configs
    strategy = ConfigurableFedAvg(
        fraction_train=0.5,
        fraction_evaluate=0.5,
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


# Helper function
def read_configs_to_toml() -> dict:
    config_manager = ConfigManager(
        config_path=r"federated_pneumonia_detection\config\default_config.yaml"
    )
    flwr_configs = {}
    if config_manager.has_key("experiment.num-server-rounds"):
        flwr_configs["num_server_rounds"] = config_manager.get(
            "experiment.num-server-rounds"
        )
    if config_manager.has_key("experiment.max-epochs"):
        flwr_configs["max_epochs"] = config_manager.get("experiment.max-epochs")
    if config_manager.has_key("experiment.options.num-supernodes"):
        flwr_configs["options_num_supernodes"] = config_manager.get(
            "experiment.options.num-supernodes"
        )
    else:
        print("No num-supernodes found in config")
    print(flwr_configs)
    return flwr_configs
