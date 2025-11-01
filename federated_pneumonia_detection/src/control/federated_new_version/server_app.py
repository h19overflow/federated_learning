from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg
from torch.compiler import config
from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet import (
    LitResNet,
)
from federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment import (
    update_flwr_config,
)
from federated_pneumonia_detection.config.config_manager import ConfigManager

app = ServerApp()

# TODO: Make sure frontend sends the correct updates to default-yaml 
# since the keys have been adjusted to refelect the convention

@app.lifespan()
def lifespan(app: ServerApp):
    print("Server is starting...")
    flwr_configs = read_configs_to_toml()
    update_flwr_config(**flwr_configs)
    yield
    print("Server is stopping...")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    training_configs = ConfigRecord({})
    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]

    # Load global model
    global_model = LitResNet()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=0.5, fraction_evaluate=0.5)

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
        flwr_configs["num-server-rounds"] = config_manager.get(
            "experiment.num-server-rounds"
        )
    if config_manager.has_key("experiment.max-epochs"):
        flwr_configs["max-epochs"] = config_manager.get("experiment.max-epochs")
    if config_manager.has_key("experiment.options.num-supernodes"):
        flwr_configs["options.num-supernodes"] = config_manager.get(
            "experiment.options.num-supernodes"
        )
    else:
        print("No num-supernodes found in config")
    print(flwr_configs)
    update_flwr_config(
        num_server_rounds=flwr_configs["num-server-rounds"],
        max_epochs=flwr_configs["max-epochs"],
        options_num_supernodes=flwr_configs["options.num-supernodes"],
    )
