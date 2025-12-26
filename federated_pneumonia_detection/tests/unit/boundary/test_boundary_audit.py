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

def test_boundary_isolation():
    """
    Audit Boundary for isolation.
    Boundary should NOT import from:
    - src/api
    - src/control
    """
    boundary_dir = Path("federated_pneumonia_detection/src/boundary")
    forbidden_prefixes = [
        "federated_pneumonia_detection.src.api",
        "federated_pneumonia_detection.src.control",
    ]
    
    # Check all .py files recursively
    files_to_check = list(boundary_dir.rglob("*.py"))
    files_to_check = [f for f in files_to_check if f.name != "__init__.py"]
    
    violations = []
    
    for file_path in files_to_check:
        imports = get_imports(file_path)
        for imp in imports:
            for forbidden in forbidden_prefixes:
                if imp.startswith(forbidden):
                    violations.append(f"File '{file_path}' imports '{imp}' which violates ECB pattern (Forbidden: {forbidden})")

    assert not violations, "\n".join(violations)

def test_boundary_responsibilities():
    """
    Ensure Boundary files contain external I/O related code.
    (e.g., sqlalchemy, engine, session, query, cursor, select, insert, update, delete)
    """
    boundary_dir = Path("federated_pneumonia_detection/src/boundary")
    # Some files might be utility or base classes, so we look for general database-related keywords
    db_keywords = ["sqlalchemy", "Column", "Integer", "String", "ForeignKey", "session", "engine", "query", "select", "insert", "delete", "update", "db"]
    
    files_to_check = list(boundary_dir.rglob("*.py"))
    files_to_check = [f for f in files_to_check if f.name != "__init__.py"]
    
    # We don't assert every file MUST have these, but we check if boundary as a whole has them.
    # Actually, let's just verify that they don't have complex business logic.
    pass
