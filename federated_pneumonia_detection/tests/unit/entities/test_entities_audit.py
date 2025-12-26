import ast
import os
from pathlib import Path

def get_imports(file_path):
    """Parses a python file and returns a list of imported module names."""
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=str(file_path))
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports

def test_entities_isolation():
    """
    Audit Entities for strict isolation (ECB Pattern).
    Entities should NOT import from:
    - src/boundary
    - src/api
    - src/control (Circular dependency risk)
    """
    
    entities_dir = Path("federated_pneumonia_detection/src/entities")
    forbidden_prefixes = [
        "federated_pneumonia_detection.src.boundary",
        "federated_pneumonia_detection.src.api",
        "federated_pneumonia_detection.src.control",
    ]
    
    files_to_check = [f for f in entities_dir.glob("*.py") if f.name != "__init__.py"]
    
    violations = []
    
    for file_path in files_to_check:
        imports = get_imports(file_path)
        for imp in imports:
            for forbidden in forbidden_prefixes:
                if imp.startswith(forbidden):
                    violations.append(f"File '{file_path.name}' imports '{imp}' which violates ECB pattern (Forbidden: {forbidden})")

    assert not violations, "\n".join(violations)

def test_entities_are_pure_classes():
    """
    Audit Entities to ensure they are primarily data structures or business logic.
    Ideally, they should be simple classes, dataclasses, or Pydantic models.
    """
    # This is a heuristic check. We are looking for suspicious keywords that might indicate
    # side effects like database calls (cursor, commit) or API calls (request, get, post).
    
    entities_dir = Path("federated_pneumonia_detection/src/entities")
    suspicious_keywords = ["cursor.execute", "requests.get", "requests.post", "fastapi"]
    
    files_to_check = [f for f in entities_dir.glob("*.py") if f.name != "__init__.py"]
    
    violations = []
    
    for file_path in files_to_check:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            for keyword in suspicious_keywords:
                if keyword in content:
                     violations.append(f"File '{file_path.name}' contains suspicious keyword '{keyword}' which might indicate external side effects.")

    assert not violations, "\n".join(violations)
