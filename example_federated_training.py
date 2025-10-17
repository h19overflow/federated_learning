"""
Example script demonstrating the Flower ClientApp federated learning architecture.

This shows the complete flow:
1. FlowerClient extends NumPyClient
2. client_fn creates FlowerClient and converts to Client via .to_client()
3. ClientApp wraps the client_fn
4. start_simulation runs federated learning
"""

from federated_pneumonia_detection.src.control.federated_learning.federated_trainer import FederatedTrainer

def main():
    """
    Run federated learning with the ClientApp pattern.
    
    Architecture:
    - FlowerClient (NumPyClient): Implements fit() and evaluate()
    - client_fn: Factory that creates FlowerClient and calls .to_client()
    - ClientApp: Wraps client_fn for Flower framework
    - start_simulation: Runs federated learning locally
    """
    
    print("=" * 80)
    print("Federated Learning with Flower ClientApp Pattern")
    print("=" * 80)
    
    # Initialize the federated trainer
    trainer = FederatedTrainer(
        checkpoint_dir="fed_results/checkpoints",
        logs_dir="fed_results/logs",
        partition_strategy="stratified"
    )
    
    print("\nArchitecture Components:")
    print("1. FlowerClient (fed_client.py)")
    print("   - Extends NumPyClient")
    print("   - Implements: fit(), evaluate(), get_parameters(), set_parameters()")
    print("   - Inherits: .to_client() method from NumPyClient")
    print()
    print("2. client_fn (in federated_trainer.py)")
    print("   - Factory function that takes Context")
    print("   - Creates FlowerClient with appropriate data loaders")
    print("   - Returns flower_client.to_client() (converts NumPyClient → Client)")
    print()
    print("3. ClientApp (modern Flower pattern)")
    print("   - Created with: ClientApp(client_fn=client_fn)")
    print("   - Wraps the client factory for Flower framework")
    print()
    print("4. start_simulation()")
    print("   - Runs federated learning locally")
    print("   - Uses client_fn to create clients for each round")
    print()
    
    # Run federated training
    # NOTE: Replace with your actual dataset path
    dataset_path = "path/to/your/chest_xray_dataset.zip"
    
    print(f"\nStarting federated training on: {dataset_path}")
    print("This will:")
    print("- Partition data across clients (stratified)")
    print("- Create data loaders for each client")
    print("- Run federated learning simulation")
    print("- Aggregate models using FedAvg")
    print("- Save final model")
    print()
    
    try:
        results = trainer.train(
            source_path=dataset_path,
            experiment_name="example_federated_run"
        )
        
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        print(f"\nResults:")
        print(f"- Experiment: {results['experiment_name']}")
        print(f"- Clients: {results['num_clients']}")
        print(f"- Rounds: {results['num_rounds']}")
        print(f"- Status: {results['status']}")
        print(f"- Checkpoint dir: {results['checkpoint_dir']}")
        
        # Print final metrics if available
        if 'metrics' in results:
            metrics = results['metrics']
            if metrics.get('losses_distributed'):
                final_loss = metrics['losses_distributed'][-1][1]
                print(f"- Final distributed loss: {final_loss:.4f}")
        
    except FileNotFoundError:
        print(f"\nERROR: Dataset not found at {dataset_path}")
        print("Please update the dataset_path variable with your actual dataset location.")
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
