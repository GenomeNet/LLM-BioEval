#!/usr/bin/env python3
"""
Test Runner for MicrobeLLM

This script provides an easy way to run the test suite with various options.
"""
import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_tests(args):
    """Run pytest with specified arguments"""
    cmd = [sys.executable, '-m', 'pytest']

    if args.verbose:
        cmd.append('-v')

    if args.coverage:
        cmd.extend(['--cov=microbellm', '--cov-report=html', '--cov-report=term'])

    if args.parallel:
        cmd.extend(['-n', str(args.parallel)])

    if args.fail_fast:
        cmd.append('-x')

    if args.test_file:
        cmd.append(f'tests/{args.test_file}')

    if args.test_function:
        cmd.append(f'--k={args.test_function}')

    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def setup_environment():
    """Setup test environment"""
    print("Setting up test environment...")

    # Install test dependencies if requirements-test.txt exists
    if Path('requirements-test.txt').exists():
        print("Installing test dependencies...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-test.txt'])

    print("Test environment ready!")


def main():
    parser = argparse.ArgumentParser(description='Run MicrobeLLM tests')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-c', '--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('-p', '--parallel', type=int, help='Run tests in parallel (requires pytest-xdist)')
    parser.add_argument('-x', '--fail-fast', action='store_true', help='Stop on first failure')
    parser.add_argument('-f', '--test-file', help='Run specific test file (e.g., test_api.py)')
    parser.add_argument('-k', '--test-function', help='Run tests matching keyword')
    parser.add_argument('--setup', action='store_true', help='Setup test environment')

    args = parser.parse_args()

    if args.setup:
        setup_environment()
        return 0

    if args.coverage and not args.parallel:
        print("Tip: Use --parallel 4 with --coverage for faster execution")

    return_code = run_tests(args)

    if return_code == 0:
        print("\n✅ All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/")
    else:
        print(f"\n❌ Tests failed with exit code: {return_code}")

    return return_code


if __name__ == '__main__':
    sys.exit(main())
