"""
Quick syntax check for inference test files.
Run this to verify all test files have valid syntax.
"""

import ast
import sys
from pathlib import Path

test_files = [
    "tests/unit/control/test_image_validator.py",
    "tests/unit/control/test_image_processor.py",
    "tests/unit/control/test_inference_engine.py",
    "tests/unit/control/test_batch_statistics.py",
    "tests/unit/control/test_clinical_interpreter.py",
    "tests/unit/control/test_observability_logger.py",
    "tests/unit/control/test_gradcam.py",
    "tests/unit/control/test_inference_service.py",
    "tests/unit/control/conftest_inference.py",
]


def check_syntax(filepath):
    """Check if Python file has valid syntax."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)


if __name__ == "__main__":
    print("Checking syntax for inference test files...")
    print("=" * 60)

    all_valid = True
    for test_file in test_files:
        path = Path(test_file)
        if not path.exists():
            print(f"❌ {test_file}: File not found")
            all_valid = False
            continue

        valid, error = check_syntax(path)
        if valid:
            print(f"✅ {test_file}: Valid syntax")
        else:
            print(f"❌ {test_file}: Syntax error\n   {error}")
            all_valid = False

    print("=" * 60)
    if all_valid:
        print("✅ All test files have valid syntax!")
        sys.exit(0)
    else:
        print("❌ Some test files have syntax errors")
        sys.exit(1)
