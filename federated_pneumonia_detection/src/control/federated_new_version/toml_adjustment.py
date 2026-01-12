import tomli_w

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python < 3.11


def update_flwr_config(
    pyproject_path: str = None,
    **kwargs,
):
    """
    Update Flower configuration in pyproject.toml from default_config.yaml values.

    This ensures that the Flower framework uses the correct configuration values
    specified in the project's default_config.yaml file.

    Args:
        pyproject_path: Path to pyproject.toml file. If None, uses the default location.
        **kwargs: Configuration key-value pairs to update. Supported keys:
            - num_server_rounds: Number of federated learning rounds
            - max_epochs: Number of local training epochs per round
            - num_supernodes: Number of client nodes in simulation

    Example usage:
        update_flwr_config(
            "pyproject.toml",
            num_server_rounds=5,
            max_epochs=2,
            num_supernodes=8
        )

    Note:
        This function is called during server startup (lifespan) to sync configs.
    """
    if pyproject_path is None:
        from pathlib import Path

        pyproject_path = str(Path(__file__).parent / "pyproject.toml")

    print(f"[TOML Update] Updating pyproject.toml at: {pyproject_path}")

    try:
        # Read existing config
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        print(f"[TOML Update] [ERROR] pyproject.toml not found at {pyproject_path}")
        return
    except Exception as e:
        print(f"[TOML Update] [ERROR] Failed to read pyproject.toml: {e}")
        return

    # Track what was updated
    updates_made = []

    # Update nested config values
    for key, value in kwargs.items():
        try:
            if key == "num_server_rounds":
                data["tool"]["flwr"]["app"]["config"]["num-server-rounds"] = value
                updates_made.append(f"num-server-rounds={value}")
            elif key == "max_epochs":
                data["tool"]["flwr"]["app"]["config"]["max-epochs"] = value
                updates_made.append(f"max-epochs={value}")
            elif key == "num_supernodes":
                data["tool"]["flwr"]["federations"]["local-simulation"]["options"][
                    "num-supernodes"
                ] = value
                updates_made.append(f"num-supernodes={value}")
            else:
                print(f"[TOML Update] [WARN] Unknown config key: {key}")
        except KeyError as e:
            print(f"[TOML Update] [ERROR] Failed to update {key}: Missing key path {e}")
            continue

    # Write back to file
    try:
        with open(pyproject_path, "wb") as f:
            tomli_w.dump(data, f)

        if updates_made:
            print(f"[TOML Update] [OK] Successfully updated: {', '.join(updates_made)}")
        else:
            print("[TOML Update] [WARN] No updates were made")
    except Exception as e:
        print(f"[TOML Update] [ERROR] Failed to write pyproject.toml: {e}")


if __name__ == "__main__":
    update_flwr_config(
        num_server_rounds=5,
        max_epochs=1,
    )
