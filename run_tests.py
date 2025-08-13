#!/usr/bin/env python3
"""
Test runner script for executing all tests in the GPU-Based Hierarchical PSO project.
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_all_tests():
    """Run all test suites."""
    print("=== Running All Test Suites ===\n")
    
    # Import and run each test suite
    try:
        from tests.test_logic import main as run_logic_tests
        print("Running Logic Tests...")
        run_logic_tests()
        print("Logic Tests completed successfully!\n")
    except Exception as e:
        print(f"Logic Tests failed: {e}\n")
    
    try:
        from tests.test_validation import main as run_validation_tests
        print("Running Validation Tests...")
        run_validation_tests()
        print("Validation Tests completed successfully!\n")
    except Exception as e:
        print(f"Validation Tests failed: {e}\n")
    
    try:
        from tests.test_autobounds import main as run_autobounds_tests
        print("Running Automatic Bounds Tests...")
        run_autobounds_tests()
        print("Automatic Bounds Tests completed successfully!\n")
    except Exception as e:
        print(f"Automatic Bounds Tests failed: {e}\n")
    
    print("=== All Test Suites Completed ===")


def main():
    """Main entry point."""
    run_all_tests()


if __name__ == "__main__":
    main()
