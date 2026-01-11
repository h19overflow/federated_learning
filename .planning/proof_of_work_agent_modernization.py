"""
Proof of Work: LangChain Agent Modernization

Verifies that:
1. query_router.py uses structured output (already modernized)
2. title_generator.py uses structured output (already modernized)
3. engine.py uses create_agent() for research mode
"""

import ast
import sys
from pathlib import Path

def check_file_imports(file_path: Path, required_imports: list) -> tuple[bool, list]:
    """Check if file contains required imports."""
    content = file_path.read_text(encoding="utf-8")
    tree = ast.parse(content)

    imports_found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                import_str = f"{node.module}.{alias.name}" if node.module else alias.name
                imports_found.append(import_str)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports_found.append(alias.name)

    missing = [imp for imp in required_imports if not any(imp in found for found in imports_found)]
    return len(missing) == 0, missing

def check_class_definition(file_path: Path, class_name: str) -> bool:
    """Check if file defines a specific class."""
    content = file_path.read_text(encoding="utf-8")
    tree = ast.parse(content)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return True
    return False

def check_function_calls(file_path: Path, function_name: str) -> int:
    """Count calls to a specific function."""
    content = file_path.read_text(encoding="utf-8")
    tree = ast.parse(content)

    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == function_name:
                count += 1
            elif isinstance(node.func, ast.Attribute) and node.func.attr == function_name:
                count += 1
    return count

def main():
    base_path = Path(__file__).parent.parent

    # File paths
    query_router = base_path / "federated_pneumonia_detection/src/control/agentic_systems/multi_agent_systems/chat/arxiv_agent/query_router.py"
    title_generator = base_path / "federated_pneumonia_detection/src/control/agentic_systems/multi_agent_systems/chat/title_generator.py"
    engine = base_path / "federated_pneumonia_detection/src/control/agentic_systems/multi_agent_systems/chat/arxiv_agent/engine.py"

    results = []

    # Test 1: query_router.py has QueryClassification Pydantic model
    print("Test 1: query_router.py has QueryClassification model...")
    has_model = check_class_definition(query_router, "QueryClassification")
    results.append(("QueryClassification model exists", has_model))
    print(f"  {'✓' if has_model else '✗'} QueryClassification defined")

    # Test 2: query_router.py uses with_structured_output
    print("\nTest 2: query_router.py uses structured output...")
    content = query_router.read_text(encoding="utf-8")
    has_structured = "with_structured_output" in content
    results.append(("with_structured_output in query_router", has_structured))
    print(f"  {'✓' if has_structured else '✗'} with_structured_output used")

    # Test 3: title_generator.py has ChatTitle Pydantic model
    print("\nTest 3: title_generator.py has ChatTitle model...")
    has_title_model = check_class_definition(title_generator, "ChatTitle")
    results.append(("ChatTitle model exists", has_title_model))
    print(f"  {'✓' if has_title_model else '✗'} ChatTitle defined")

    # Test 4: title_generator.py has field_validator
    print("\nTest 4: title_generator.py has field validator...")
    content = title_generator.read_text(encoding="utf-8")
    has_validator = "@field_validator" in content
    results.append(("field_validator in title_generator", has_validator))
    print(f"  {'✓' if has_validator else '✗'} field_validator used")

    # Test 5: engine.py imports create_agent
    print("\nTest 5: engine.py imports create_agent...")
    has_create_agent_import, _ = check_file_imports(engine, ["langchain.agents.create_agent"])
    results.append(("create_agent imported in engine", has_create_agent_import))
    print(f"  {'✓' if has_create_agent_import else '✗'} create_agent imported")

    # Test 6: engine.py has _create_agent method
    print("\nTest 6: engine.py has _create_agent method...")
    content = engine.read_text(encoding="utf-8")
    has_create_agent_method = "def _create_agent" in content
    results.append(("_create_agent method exists", has_create_agent_method))
    print(f"  {'✓' if has_create_agent_method else '✗'} _create_agent method defined")

    # Test 7: engine.py calls create_agent function
    print("\nTest 7: engine.py calls create_agent...")
    create_agent_calls = check_function_calls(engine, "create_agent")
    has_calls = create_agent_calls > 0
    results.append(("create_agent called in engine", has_calls))
    print(f"  {'✓' if has_calls else '✗'} create_agent called {create_agent_calls} time(s)")

    # Test 8: engine.py uses agent.astream
    print("\nTest 8: engine.py uses agent.astream()...")
    has_astream = "agent.astream" in content
    results.append(("agent.astream in engine", has_astream))
    print(f"  {'✓' if has_astream else '✗'} agent.astream used")

    # Test 9: engine.py removed bind_tools pattern
    print("\nTest 9: engine.py removed old bind_tools pattern...")
    has_bind_tools = "bind_tools" in content
    results.append(("bind_tools removed", not has_bind_tools))
    print(f"  {'✓' if not has_bind_tools else '✗'} bind_tools {'removed' if not has_bind_tools else 'still present'}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed! Modernization complete.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
