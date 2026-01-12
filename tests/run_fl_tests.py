"""
Test runner script for federated learning tests.
Provides convenient interface to run different test suites.
"""

import sys
import subprocess
import argparse


def run_command(cmd: list) -> int:
    """Run command and return exit code."""
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    result = subprocess.run(cmd)
    print("-" * 80)
    return result.returncode


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run federated learning tests")
    parser.add_argument(
        '--suite',
        choices=['all', 'unit', 'integration', 'coverage'],
        default='all',
        help='Test suite to run'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--markers',
        '-m',
        type=str,
        help='Run tests matching given mark expression'
    )
    parser.add_argument(
        '--file',
        '-f',
        type=str,
        help='Run specific test file'
    )

    args = parser.parse_args()

    # Base pytest command
    base_cmd = ['pytest']

    if args.verbose:
        base_cmd.append('-v')

    # Determine test path
    if args.file:
        test_path = args.file
    elif args.suite == 'unit':
        test_path = 'tests/unit/control/federated_learning/'
    elif args.suite == 'integration':
        test_path = 'tests/integration/federated_learning/'
    elif args.suite == 'all':
        test_path = 'tests/unit/control/federated_learning/ tests/integration/federated_learning/'
    elif args.suite == 'coverage':
        base_cmd.extend([
            'tests/unit/control/federated_learning/',
            'tests/integration/federated_learning/',
            '--cov=federated_pneumonia_detection.src.control.federated_learning',
            '--cov-report=html',
            '--cov-report=term'
        ])
        return run_command(base_cmd)

    # Add test path
    if isinstance(test_path, str):
        base_cmd.extend(test_path.split())

    # Add markers if specified
    if args.markers:
        base_cmd.extend(['-m', args.markers])

    # Run tests
    return run_command(base_cmd)


if __name__ == '__main__':
    sys.exit(main())



