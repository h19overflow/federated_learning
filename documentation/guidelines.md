### Phase 1: Foundation & Core Infrastructure (Weeks 1-2)

#### Week 1 Focus: Project Structure & Entity Classes
**Primary Goals:**
- Establish clean, modular project structure
- Implement all Entity classes with proper validation
- Set up development environment and configuration management

**AI Assistant Specific Tasks:**
1. **Project Structure Creation:**
   ```
   federated_pneumonia_detection/
   ├── src/
   │   ├── entities/
   │   ├── control/
   │   ├── boundary/
   │   └── utils/
   ├── tests/
   ├── config/
   ├── data/
   ├── models/
   ├── logs/
   └── docs/
   ```

2. **Entity Classes to Implement:**
   - `CustomImageDataset`: Handle X-ray image paths, transformations, labels
   - `ResNetWithCustomHead`: ResNet50 V2 backbone + custom binary classifier
   - `ExperimentConfig`: Store all experiment parameters (learning rate, epochs, FL settings)
   - `SessionResultsCache`: Temporary storage for experiment artifacts
   - `SystemConstants`: Global configuration values

**Code Quality Requirements:**
- Add comprehensive docstrings for all classes
- Implement proper input validation and error handling
- Use type hints throughout
- Create unit tests for each entity class

#### Week 2 Focus: Data Handling & Validation
**Primary Goals:**
- Implement robust data loading and preprocessing
- Create dataset validation systems
- Establish train/validation splitting functionality

**AI Assistant Specific Tasks:**
1. **Data Pipeline Implementation:**
   - Image loading with proper error handling
   - Preprocessing: resize to 224x224, normalization using ImageNet stats
   - Data augmentation pipeline (rotation, flip, brightness adjustment)
   - Train/validation split with configurable ratios

2. **Validation Systems:**
   - Dataset structure validation (check for required folders/files)
   - Image format validation (JPEG, PNG support)
   - Label consistency checks
   - Memory usage monitoring for large datasets

### Phase 2: Core ML Components (Weeks 3-4)

#### Week 3 Focus: ResNet Model & Training Infrastructure
**Primary Goals:**
- Implement ResNet50 V2 with custom classification head
- Create PyTorch Lightning module for training
- Set up model checkpointing and parameter management

**AI Assistant Specific Tasks:**
1. **Model Architecture:**
   ```python
   # Example structure for ResNetWithCustomHead
   class ResNetWithCustomHead(nn.Module):
       def __init__(self, num_classes=2, freeze_backbone=True):
           # ResNet50 V2 pretrained on ImageNet
           # Custom head: Global Average Pool -> Dropout -> Linear -> Binary Classification
   ```

2. **PyTorch Lightning Integration:**
   - `LitResNetFixedModel` class with training_step, validation_step, test_step
   - Optimizer configuration (Adam with weight decay)
   - Learning rate scheduling (ReduceLROnPlateau)
   - Metrics logging (accuracy, precision, recall, F1, AUC)

**Technical Requirements:**
- Use transfer learning best practices
- Implement gradient flow management for frozen/unfrozen layers
- Add model summary utilities
- Ensure reproducible training with proper seed setting

#### Week 4 Focus: Data Module & Training Pipeline
**Primary Goals:**
- Create PyTorch Lightning DataModule
- Implement training pipeline with proper validation
- Add comprehensive metrics calculation and logging

**AI Assistant Specific Tasks:**
1. **Data Module Implementation:**
   ```python
   class XRayDataModuleFixedPipeline(pl.LightningDataModule):
       # Setup train/val/test dataloaders
       # Handle data augmentation
       # Memory-efficient batch processing
   ```

2. **Training Pipeline:**
   - Early stopping with patience
   - Model checkpointing (save best model based on validation metric)
   - Training progress monitoring
   - GPU memory optimization

### Phase 3: Centralized Training System (Weeks 5-6)

#### Week 5 Focus: Centralized Trainer & Evaluation
**Primary Goals:**
- Implement complete centralized training workflow
- Create comprehensive model evaluation system
- Generate performance metrics and reports

**AI Assistant Specific Tasks:**
1. **Centralized Trainer:**
   ```python
   class CentralizedTrainerFixed:
       # Orchestrate complete training workflow
       # Handle hyperparameter management
       # Integrate with PyTorch Lightning Trainer
       # Progress tracking and reporting
   ```

2. **Evaluation System:**
   - `ModelEvaluatorFixed`: Post-training comprehensive evaluation
   - `MetricsCalculator`: Standard ML metrics (accuracy, precision, recall, F1, AUC)
   - Confusion matrix generation and analysis
   - ROC curve calculation and visualization

#### Week 6 Focus: Reporting & Visualization
**Primary Goals:**
- Create report generation system
- Implement visualization components for training progress and results
- Add export functionality for results and visualizations

**AI Assistant Specific Tasks:**
1. **Reporting System:**
   - `ReportGenerator`: Generate textual classification reports
   - Export to multiple formats (CSV, JSON, PDF)
   - Performance summary generation
   - Training history documentation

2. **Visualization Components:**
   - `PlotGenerator`: Create confusion matrices, training curves, ROC curves
   - Use matplotlib/seaborn for static plots
   - Plotly integration for interactive visualizations
   - Consistent styling and branding across all plots

### Phase 4: Dashboard Foundation (Weeks 7-8)

#### Week 7 Focus: Basic Streamlit Dashboard
**Primary Goals:**
- Create main Streamlit application structure
- Implement file upload functionality
- Establish session management system

**AI Assistant Specific Tasks:**
1. **Dashboard Structure:**
   ```python
   class DashboardApp:
       # Multi-page Streamlit application
       # Navigation and routing
       # Session state management
       # Error handling and user feedback
   ```

2. **Core UI Components:**
   - `DataUploadComponent`: Dataset upload with validation
   - File drag-and-drop interface
   - Upload progress indicators
   - Dataset preview functionality

**Streamlit Best Practices:**
- Use st.cache_data for expensive operations
- Implement proper session state management
- Add loading spinners for long operations
- Create responsive layouts with columns

#### Week 8 Focus: Configuration & Controls Interface
**Primary Goals:**
- Create experiment configuration UI
- Implement training control interface
- Add parameter validation and user input handling

**AI Assistant Specific Tasks:**
1. **Configuration Interface:**
   - `ExperimentConfigComponent`: Dynamic form generation
   - Separate forms for Centralized vs Federated Learning modes
   - Parameter validation with helpful error messages
   - Default value management and preset configurations

2. **Control Interface:**
   - `RunControlComponent`: Start/stop/pause experiment functionality
   - Real-time progress monitoring
   - Log streaming interface
   - Resource utilization display

### Phase 5: Federated Learning Implementation (Weeks 9-10)

#### Week 9 Focus: Flower Client & Communication
**Primary Goals:**
- Implement Flower FL client for local training
- Set up secure client-server communication
- Handle parameter synchronization

**AI Assistant Specific Tasks:**
1. **Flower Client Implementation:**
   ```python
   class FlowerClientFixedModel(fl.client.Client):
       # Implement get_parameters, set_parameters methods
       # Local training with fit method
       # Model evaluation with evaluate method
       # Error handling for network issues
   ```

2. **Communication Setup:**
   - Secure parameter serialization/deserialization
   - Network error recovery mechanisms
   - Client authentication and security
   - Comprehensive logging for debugging

#### Week 10 Focus: FedAvg Strategy & Simulation
**Primary Goals:**
- Implement Federated Averaging algorithm
- Create federated learning simulation environment
- Handle multi-client coordination and aggregation

**AI Assistant Specific Tasks:**
1. **FedAvg Implementation:**
   ```python
   class FedAvgStrategy(fl.server.strategy.Strategy):
       # Weighted parameter averaging
       # Client selection strategies
       # Convergence monitoring
       # Round management
   ```

2. **Simulation Framework:**
   - `FederatedSimulatorFixed`: Multi-client simulation management
   - Data partitioning across clients (IID and non-IID)
   - Performance tracking across federated rounds
   - Client dropout handling

### Phase 6: Advanced Dashboard Features (Weeks 11-12)

#### Week 11 Focus: Results Visualization & Display
**Primary Goals:**
- Create comprehensive results display system
- Implement interactive visualizations
- Add performance comparison tools

**AI Assistant Specific Tasks:**
1. **Results Display System:**
   - `ResultsDisplayComponent`: Comprehensive result presentation
   - Real-time updates during training
   - Historical results comparison
   - Interactive charts using Plotly

2. **Advanced Visualizations:**
   - Interactive confusion matrices with hover details
   - Training progress with zoom/pan capabilities
   - Federated learning round-by-round performance
   - Client-specific performance tracking

#### Week 12 Focus: Progress Tracking & Export
**Primary Goals:**
- Implement real-time progress monitoring
- Create comprehensive export system
- Add automated report generation

**AI Assistant Specific Tasks:**
1. **Progress Monitoring:**
   - Real-time metric streaming
   - Training ETA predictions
   - Resource utilization monitoring
   - Performance trend analysis

2. **Export & Reporting:**
   - Multi-format export (PDF, CSV, JSON, PNG)
   - Custom report templates
   - Automated report generation
   - Results archiving and retrieval

### Phase 7: Integration & Testing (Week 13)

#### Week 13 Focus: Full System Integration
**Primary Goals:**
- Integrate all components into unified system
- Comprehensive testing and optimization
- Error handling improvements

**AI Assistant Specific Tasks:**
1. **System Integration:**
   - End-to-end workflow testing
   - Cross-component communication validation
   - Memory and resource optimization
   - Performance bottleneck identification

2. **Testing & Quality Assurance:**
   - Unit tests for all components
   - Integration tests for complete workflows
   - Performance benchmarking
   - User acceptance test scenarios

### Phase 8: Final Polish & Documentation (Week 14)

#### Week 14 Focus: Production Ready System
**Primary Goals:**
- Final UI/UX improvements
- Complete documentation
- Deployment preparation

**AI Assistant Specific Tasks:**
1. **Final Polish:**
   - UI/UX refinements based on testing feedback
   - Performance optimizations
   - Code cleanup and refactoring
   - Security audit and improvements

2. **Documentation & Deployment:**
   - API documentation generation
   - User manual and tutorials
   - Installation and deployment guides
   - Troubleshooting documentation

## General AI Assistant Guidelines

### Code Quality Standards
- **PEP 8 Compliance**: Follow Python style guidelines
- **Type Hints**: Use throughout for better maintainability
- **Error Handling**: Implement comprehensive try-catch blocks
- **Logging**: Add detailed logging for debugging and monitoring
- **Documentation**: Include docstrings for all functions and classes

### Testing Requirements
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark critical operations
- **Mock Objects**: Use for external dependencies

### Session Management Best Practices
- **Focus**: Keep sessions focused on specific milestones
- **Context**: Always reference current phase and week
- **Consistency**: Maintain consistent code style and patterns
- **Version Control**: Use meaningful commit messages and regular commits

### Common Patterns to Follow

#### Error Handling Pattern:
```python
import logging
logger = logging.getLogger(__name__)

def example_function():
    try:
        # Operation that might fail
        result = risky_operation()
        return result
    except SpecificException as e:
        logger.error(f"Specific error occurred: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in {__name__}: {e}")
        raise RuntimeError(f"Operation failed: {e}")
```

#### Configuration Loading Pattern:
```python
from typing import Dict, Any
import yaml

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            raise
```

#### Streamlit Component Pattern:
```python
import streamlit as st
from typing import Optional

class BaseComponent:
    def __init__(self, title: str):
        self.title = title
    
    def render(self) -> None:
        st.subheader(self.title)
        try:
            self._render_content()
        except Exception as e:
            st.error(f"Error rendering {self.title}: {e}")
            logger.error(f"Component {self.title} render error: {e}")
    
    def _render_content(self) -> None:
        # Override in subclasses
        pass
```

## Technical Specifications

### Dependencies Management
- **PyTorch**: Latest stable version with CUDA support
- **PyTorch Lightning**: For organized training workflows
- **Flower**: For federated learning implementation
- **Streamlit**: For web dashboard interface
- **scikit-learn**: For metrics and evaluation
- **matplotlib/seaborn**: For static visualizations
- **Plotly**: For interactive visualizations
- **Pillow**: For image processing
- **PyYAML**: For configuration management

### Hardware Considerations
- **GPU Support**: Ensure CUDA compatibility for training
- **Memory Management**: Handle large datasets efficiently
- **CPU Utilization**: Optimize for multi-core processing
- **Storage**: Efficient temporary file management

### Security Requirements
- **Data Privacy**: Implement federated learning privacy principles
- **Secure Communication**: Use TLS/SSL for client-server communication
- **Input Validation**: Sanitize all user inputs
- **Access Control**: Implement proper authentication (if required)

This comprehensive guide ensures that AI coding assistants have clear, specific instructions for each phase of the project implementation, maintaining consistency and quality throughout the development process.