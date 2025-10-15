"""
Example usage of ConfigManager for the federated pneumonia detection system.

This file demonstrates how to use the ConfigManager class to read and modify
configuration values in a centralized way.
"""

from config_manager import ConfigManager, quick_get, quick_set

def demonstrate_config_manager():
    """Demonstrate various ConfigManager features."""
    
    print("=== ConfigManager Example Usage ===\n")
    
    # Create a config manager instance
    config = ConfigManager()
    print(f"Loaded configuration from: {config.config_path}")
    
    # Reading configuration values
    print("\n--- Reading Configuration Values ---")
    learning_rate = config.get('experiment.learning_rate')
    print(f"Current learning rate: {learning_rate}")
    
    img_size = config.get('system.img_size')
    print(f"Image size: {img_size}")
    
    batch_size = config.get('system.batch_size')
    print(f"Batch size: {batch_size}")
    
    # Using dictionary-style access
    print(f"Epochs: {config['experiment.epochs']}")
    
    # Check if key exists
    if 'experiment.learning_rate' in config:
        print("Learning rate configuration exists")
    
    # Get entire sections
    print("\n--- Getting Configuration Sections ---")
    experiment_config = config.get_section('experiment')
    print(f"Experiment section keys: {list(experiment_config.keys())}")
    
    system_config = config.get_section('system')
    print(f"System section keys: {list(system_config.keys())}")
    
    # Modify configuration values
    print("\n--- Modifying Configuration Values ---")
    original_lr = config.get('experiment.learning_rate')
    print(f"Original learning rate: {original_lr}")
    
    # Set new learning rate
    config.set('experiment.learning_rate', 0.002)
    new_lr = config.get('experiment.learning_rate')
    print(f"New learning rate: {new_lr}")
    
    # Dictionary-style assignment
    config['system.batch_size'] = 128
    print(f"New batch size: {config['system.batch_size']}")
    
    # Update multiple values at once
    config.update({
        'experiment.epochs': 25,
        'experiment.early_stopping_patience': 10,
        'system.num_workers': 8
    })
    
    print(f"Updated epochs: {config.get('experiment.epochs')}")
    print(f"Updated patience: {config.get('experiment.early_stopping_patience')}")
    print(f"Updated workers: {config.get('system.num_workers')}")
    
    # List all available keys
    print("\n--- Available Configuration Keys ---")
    all_keys = config.list_keys()
    print(f"Total configuration keys: {len(all_keys)}")
    
    # Show experiment keys only
    experiment_keys = config.list_keys('experiment')
    print(f"Experiment keys: {experiment_keys[:5]}...")  # Show first 5
    
    # Create backup before saving
    print("\n--- Backup and Save ---")
    backup_path = config.backup()
    print(f"Created backup at: {backup_path}")
    
    # Note: Uncomment the next line to actually save changes
    # config.save()
    # print("Configuration saved successfully!")
    
    # Reset to original state (since we don't want to modify the actual config)
    config.reset()
    print(f"Reset learning rate back to: {config.get('experiment.learning_rate')}")
    
    # Demonstrate convenience functions
    print("\n--- Convenience Functions ---")
    lr = quick_get('experiment.learning_rate')
    print(f"Quick get learning rate: {lr}")
    
    # Note: Uncomment to actually modify and save
    # quick_set('experiment.learning_rate', 0.001)
    # print("Used quick_set to update learning rate")
    
    print("\n=== Example Complete ===")


def show_current_config():
    """Display current configuration in a readable format."""
    config = ConfigManager()
    
    print("=== Current Configuration ===")
    print(config)


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_config_manager()
    
    # Optionally show the full config
    print("\n" + "="*50)
    show_current_config()