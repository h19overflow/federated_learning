import tomli
import tomli_w

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python < 3.11


def update_flwr_config(
    pyproject_path: str = r"federated_pneumonia_detection\src\control\federated_new_version\pyproject.toml",
    **kwargs,
):
    """
    Update Flwr configuration in pyproject.toml

    Example usage:
    update_flwr_config(
        "pyproject.toml",
        num_server_rounds=5,
        max_epochs=2,
        num_supernodes=8
    )
    """
    # Read existing config
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    # Update nested config values
    for key, value in kwargs.items():
        if key == "num_server_rounds":
            data["tool"]["flwr"]["app"]["config"]["num-server-rounds"] = value
        elif key == "max_epochs":
            data["tool"]["flwr"]["app"]["config"]["max-epochs"] = value
        elif key == "num_supernodes":
            data["tool"]["flwr"]["federations"]["local-simulation"]["options"][
                "num-supernodes"
            ] = value

    # Write back to file
    with open(pyproject_path, "wb") as f:
        tomli_w.dump(data, f)


if __name__ == "__main__":
    update_flwr_config(
        num_server_rounds=5,
        max_epochs=1,
    )
