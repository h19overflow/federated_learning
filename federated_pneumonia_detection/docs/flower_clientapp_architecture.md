# Flower ClientApp Architecture

## Overview

This document explains how the federated learning implementation is compatible with Flower's modern ClientApp API while using simulation for development and testing.

## Architecture Components

### 1. FlowerClient (fed_client.py)

The core client implementation that handles local training:

```python
class FlowerClient(NumPyClient):
    """Flower NumPy client for local training."""
    
    def get_parameters(self, config):
        """Return current model parameters as NumPy arrays."""
        
    def set_parameters(self, parameters):
        """Update model with parameters from server."""
        
    def fit(self, parameters, config):
        """Train model locally and return updated parameters."""
        
    def evaluate(self, parameters, config):
        """Evaluate model and return metrics."""
```

**Key Points:**
- Extends `NumPyClient` for automatic serialization
- Self-contained: manages its own model, optimizer, and training loop
- Can be converted to `Client` using `.to_client()` method
- Compatible with both simulation and deployment

### 2. SimulationRunner (simulation_runner.py)

Orchestrates local simulation for development:

```python
def client_fn(context: Context) -> Client:
    """Factory function that creates clients for simulation."""
    client_id = int(context.node_id)
    train_loader, val_loader = client_dataloaders[client_id]
    
    flower_client = FlowerClient(
        client_id=client_id,
        train_loader=train_loader,
        val_loader=val_loader,
        constants=constants,
        config=config,
        logger=logger
    )
    
    # Convert NumPyClient to Client
    return flower_client.to_client()

# Run simulation
history = start_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    config=server_config,
    strategy=strategy
)
```

**Key Points:**
- Uses `start_simulation()` for local testing
- Creates clients dynamically from pre-partitioned data
- Suitable for development and experimentation
- Runs all clients in same process (simulated)

### 3. ClientApp (client_app.py)

Modern deployment-ready client application:

```python
from flwr.client import ClientApp

def client_fn(context: Context):
    """Create client instance for deployment."""
    client_id = int(context.node_id)
    
    # Load client-specific data from context or local storage
    train_loader, val_loader = load_client_data(client_id)
    
    flower_client = FlowerClient(
        client_id=client_id,
        train_loader=train_loader,
        val_loader=val_loader,
        constants=constants,
        config=config,
        logger=logger
    )
    
    return flower_client.to_client()

# Create ClientApp
app = ClientApp(client_fn=client_fn)
```

**Key Points:**
- Uses modern `ClientApp` API (Flower 1.0+)
- Designed for real distributed deployment
- Each client runs in separate process/machine
- Data loading happens per-client at runtime

### 4. ServerApp (server_app.py)

Server-side configuration and strategy:

```python
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

def server_fn(context: Context):
    """Create server configuration."""
    # Create global model
    model = create_global_model(constants, config)
    
    # Initialize strategy
    strategy = FedAvg(
        initial_parameters=get_model_parameters(model),
        # ... other strategy parameters
    )
    
    server_config = ServerConfig(num_rounds=config.num_rounds)
    return server_config, strategy

app = ServerApp(server_fn=server_fn)
```

## Usage Patterns

### Pattern 1: Simulation (Current Implementation)

**Use Case:** Development, testing, experimentation

```python
from federated_trainer import FederatedTrainer

trainer = FederatedTrainer()
results = trainer.train(source_path="data.zip")
```

**Flow:**
1. FederatedTrainer partitions data across clients
2. SimulationRunner creates client_fn factory
3. start_simulation() runs all clients locally
4. Results returned to trainer

**Pros:**
- Fast iteration during development
- Easy debugging (all in one process)
- No infrastructure needed

**Cons:**
- Not representative of real deployment
- Limited by single machine resources

### Pattern 2: Deployment (ClientApp/ServerApp)

**Use Case:** Production deployment, real federated learning

```bash
# Start server
flower-server-app server_app:app

# Start clients (on different machines)
flower-client-app client_app:app --server-address=server:9092
```

**Flow:**
1. Server starts and waits for clients
2. Each client connects independently
3. Clients load their own local data
4. Training happens across distributed nodes

**Pros:**
- Real federated learning across devices
- Privacy-preserving (data stays local)
- Scalable to many clients

**Cons:**
- Requires infrastructure setup
- More complex debugging
- Need real data distribution

## Migration Path

### From Simulation to Deployment

1. **Use the same FlowerClient:**
   - No changes needed to core client logic
   - Already compatible with both patterns

2. **Adapt data loading:**
   ```python
   # Simulation: Pre-partition and pass to client
   client_dataloaders = [...]
   
   # Deployment: Each client loads its own data
   def client_fn(context):
       data = load_local_data(context.node_id)
       train_loader, val_loader = create_dataloaders(data)
       # ... rest is same
   ```

3. **Update configuration:**
   - Simulation: All config in SimulationRunner
   - Deployment: Config via Context or environment

4. **Deploy:**
   ```bash
   # Instead of trainer.train()
   flower-client-app client_app:app
   ```

## Compatibility Matrix

| Component | Simulation | Deployment | Status |
|-----------|-----------|------------|--------|
| FlowerClient | ✅ | ✅ | Compatible |
| client_fn pattern | ✅ | ✅ | Compatible |
| .to_client() | ✅ | ✅ | Required |
| start_simulation() | ✅ | ❌ | Sim only |
| ClientApp | ❌ | ✅ | Deploy only |
| ServerApp | ❌ | ✅ | Deploy only |

## Best Practices

### 1. Keep FlowerClient Pure
- Don't mix simulation-specific logic in FlowerClient
- FlowerClient should work with any data loaders
- Use dependency injection for configuration

### 2. Separate Data Loading
- Simulation: Pre-partition in SimulationRunner
- Deployment: Client-side loading in client_fn
- Keep data loading logic separate from client logic

### 3. Test Both Patterns
- Develop with simulation for speed
- Test with ClientApp before deployment
- Ensure FlowerClient works in both contexts

### 4. Configuration Management
- Use same config classes (SystemConstants, ExperimentConfig)
- Make configs loadable from files or environment
- Support different sources in different contexts

## References

- [Flower Documentation](https://flower.ai/docs/)
- [ClientApp API Reference](https://flower.ai/docs/framework/ref-api/flwr.client.ClientApp.html)
- [Migration Guide](https://flower.ai/docs/framework/how-to-upgrade-to-flower-1.0.html)
- [Simulation Guide](https://flower.ai/docs/framework/how-to-run-simulations.html)
