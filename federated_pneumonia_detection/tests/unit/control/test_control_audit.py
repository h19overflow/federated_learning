import ast
from pathlib import Path

def get_imports(file_path):
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

def test_control_isolation():
    """
    Audit Control for orchestration logic (ECB Pattern).
    Control should be the bridge between API/Boundary and Entities.
    It should NOT import from:
    - src/api
    """
    control_dir = Path("federated_pneumonia_detection/src/control")
    forbidden_prefixes = [
        "federated_pneumonia_detection.src.api",
    ]
    
    # Check all .py files recursively
    files_to_check = list(control_dir.rglob("*.py"))
    files_to_check = [f for f in files_to_check if f.name != "__init__.py"]
    
    violations = []
    
    for file_path in files_to_check:
        imports = get_imports(file_path)
        for imp in imports:
            for forbidden in forbidden_prefixes:
                if imp.startswith(forbidden):
                    violations.append(f"File '{file_path}' imports '{imp}' which violates ECB pattern (Forbidden: {forbidden})")

    assert not violations, "\n".join(violations)

def test_control_imports_entities_and_boundary():
    """
    Control layer should typically import from both entities and boundary
    to perform its orchestration role.
    """
    # This is an informational check.
    pass
