"""
Verification script for WebSocket Metrics Streaming implementation.

This script verifies that all required components for WebSocket metrics streaming
are correctly implemented in the federated learning system.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple, Dict

class ImplementationVerifier:
    """Verify WebSocket metrics streaming implementation."""
    
    def __init__(self):
        self.base_path = Path(__file__).parent / "federated_pneumonia_detection" / "src" / "control" / "federated_learning"
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []
        
    def verify_all(self) -> bool:
        """Run all verification checks."""
        print("=" * 80)
        print("VERIFYING WEBSOCKET METRICS STREAMING IMPLEMENTATION")
        print("=" * 80)
        print()
        
        # Step 1: Verify FederatedMetricsCollector
        print("[STEP 1] Verifying FederatedMetricsCollector...")
        self.verify_federated_metrics_collector()
        
        # Step 2: Verify FederatedTrainer
        print("\n[STEP 2] Verifying FederatedTrainer...")
        self.verify_federated_trainer()
        
        # Step 3: Verify FlowerClient
        print("\n[STEP 3] Verifying FlowerClient...")
        self.verify_flower_client()
        
        # Print results
        self.print_results()
        
        return len(self.errors) == 0
    
    def verify_federated_metrics_collector(self):
        """Verify FederatedMetricsCollector implementation."""
        file_path = self.base_path / "federated_metrics_collector.py"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content)
        
        # Find FederatedMetricsCollector class
        collector_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "FederatedMetricsCollector":
                collector_class = node
                break
        
        if not collector_class:
            self.errors.append("FederatedMetricsCollector class not found")
            return
        
        # Check __init__ method
        init_method = None
        for item in collector_class.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                init_method = item
                break
        
        if init_method:
            # Check for websocket_uri parameter
            params = [arg.arg for arg in init_method.args.args]
            if "websocket_uri" in params:
                self.passed.append("✓ websocket_uri parameter in __init__")
            else:
                self.errors.append("✗ Missing websocket_uri parameter in __init__")
            
            # Check for run_id parameter
            if "run_id" in params:
                self.passed.append("✓ run_id parameter in __init__")
            else:
                self.errors.append("✗ Missing run_id parameter in __init__")
        
        # Check for required methods
        methods = [item.name for item in collector_class.body if isinstance(item, ast.FunctionDef)]
        
        required_methods = [
            "record_round_start",
            "record_local_epoch",
            "record_eval_metrics",
            "end_training"
        ]
        
        for method in required_methods:
            if method in methods:
                self.passed.append(f"✓ {method}() method exists")
            else:
                self.errors.append(f"✗ Missing {method}() method")
        
        # Check for WebSocket sender in content
        if "MetricsWebSocketSender" in content:
            self.passed.append("✓ MetricsWebSocketSender imported and used")
        else:
            self.errors.append("✗ MetricsWebSocketSender not found")
        
        # Check for WebSocket event sending
        if "send_metrics" in content or "send_round_end" in content:
            self.passed.append("✓ WebSocket event sending implemented")
        else:
            self.errors.append("✗ WebSocket event sending not implemented")
    
    def verify_federated_trainer(self):
        """Verify FederatedTrainer implementation."""
        file_path = self.base_path / "trainer.py"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content)
        
        # Find FederatedTrainer class
        trainer_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "FederatedTrainer":
                trainer_class = node
                break
        
        if not trainer_class:
            self.errors.append("FederatedTrainer class not found")
            return
        
        # Check __init__ method
        init_method = None
        for item in trainer_class.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                init_method = item
                break
        
        if init_method:
            params = [arg.arg for arg in init_method.args.args]
            if "websocket_uri" in params:
                self.passed.append("✓ websocket_uri parameter in __init__")
            else:
                self.errors.append("✗ Missing websocket_uri parameter in __init__")
        
        # Check for _create_run method
        methods = [item.name for item in trainer_class.body if isinstance(item, ast.FunctionDef)]
        
        if "_create_run" in methods:
            self.passed.append("✓ _create_run() method exists")
        else:
            self.errors.append("✗ Missing _create_run() method")
        
        # Check train method
        if "train" in methods:
            self.passed.append("✓ train() method exists")
        else:
            self.errors.append("✗ Missing train() method")
        
        # Check for required imports
        if "MetricsWebSocketSender" in content:
            self.passed.append("✓ MetricsWebSocketSender imported")
        else:
            self.errors.append("✗ MetricsWebSocketSender not imported")
        
        if "from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud" in content:
            self.passed.append("✓ run_crud imported")
        else:
            self.errors.append("✗ run_crud not imported")
        
        # Check for training_start event
        if "training_start" in content:
            self.passed.append("✓ training_start event sending")
        else:
            self.errors.append("✗ training_start event not sent")
        
        # Check for training_end event
        if "training_end" in content or "send_training_end" in content:
            self.passed.append("✓ training_end event sending")
        else:
            self.errors.append("✗ training_end event not sent")
        
        # Check for run_id creation
        if "self.run_id = self._create_run" in content:
            self.passed.append("✓ run_id creation in train()")
        else:
            self.errors.append("✗ run_id not created in train()")
    
    def verify_flower_client(self):
        """Verify FlowerClient implementation."""
        file_path = self.base_path / "client.py"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content)
        
        # Find FlowerClient class
        client_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "FlowerClient":
                client_class = node
                break
        
        if not client_class:
            self.errors.append("FlowerClient class not found")
            return
        
        # Check __init__ method
        init_method = None
        for item in client_class.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                init_method = item
                break
        
        if init_method:
            params = [arg.arg for arg in init_method.args.args]
            
            if "run_id" in params:
                self.passed.append("✓ run_id parameter in __init__")
            else:
                self.errors.append("✗ Missing run_id parameter in __init__")
            
            if "websocket_uri" in params:
                self.passed.append("✓ websocket_uri parameter in __init__")
            else:
                self.errors.append("✗ Missing websocket_uri parameter in __init__")
        
        # Check that run_id is passed to FederatedMetricsCollector
        if "run_id=run_id" in content and "FederatedMetricsCollector" in content:
            self.passed.append("✓ run_id passed to FederatedMetricsCollector")
        else:
            self.errors.append("✗ run_id not passed to FederatedMetricsCollector")
        
        # Check that websocket_uri is passed to FederatedMetricsCollector
        if "websocket_uri=websocket_uri" in content:
            self.passed.append("✓ websocket_uri passed to FederatedMetricsCollector")
        else:
            self.errors.append("✗ websocket_uri not passed to FederatedMetricsCollector")
    
    def print_results(self):
        """Print verification results."""
        print("\n" + "=" * 80)
        print("VERIFICATION RESULTS")
        print("=" * 80)
        
        if self.passed:
            print(f"\n✓ PASSED ({len(self.passed)}):")
            for item in self.passed:
                print(f"  {item}")
        
        if self.warnings:
            print(f"\n⚠ WARNINGS ({len(self.warnings)}):")
            for item in self.warnings:
                print(f"  {item}")
        
        if self.errors:
            print(f"\n✗ ERRORS ({len(self.errors)}):")
            for item in self.errors:
                print(f"  {item}")
        
        print("\n" + "=" * 80)
        if len(self.errors) == 0:
            print("✓ ALL CHECKS PASSED - Implementation is valid!")
            print("=" * 80)
        else:
            print(f"✗ VERIFICATION FAILED - {len(self.errors)} error(s) found")
            print("=" * 80)
        print()

def main():
    """Main verification function."""
    verifier = ImplementationVerifier()
    success = verifier.verify_all()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
