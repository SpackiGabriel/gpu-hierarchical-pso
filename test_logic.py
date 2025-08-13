"""
Simplified test file for validating the objective functions and error metrics logic without GPU.
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from registries import objective_registry, error_registry


def test_objective_registry():
    """Test the objective registry functionality."""
    print("Testing objective registry...")
    
    # Test listing objectives
    objectives = objective_registry.list_objectives()
    print(f"Registered objectives: {list(objectives.keys())}")
    
    # Test getting objective info
    langmuir_info = objective_registry.get("langmuir")
    print(f"Langmuir info: {langmuir_info}")
    
    # Test getting dimension
    langmuir_dim = objective_registry.get_dimension("langmuir")
    print(f"Langmuir dimension: {langmuir_dim}")
    
    # Test getting model ID
    langmuir_id = objective_registry.get_model_id("langmuir")
    print(f"Langmuir model ID: {langmuir_id}")
    
    print("Objective registry test passed!\n")


def test_error_registry():
    """Test the error registry functionality."""
    print("Testing error registry...")
    
    # Test listing errors
    errors = error_registry.list_errors()
    print(f"Registered errors: {list(errors.keys())}")
    
    # Test getting error info
    sse_info = error_registry.get("sse")
    print(f"SSE info: {sse_info}")
    
    # Test getting error ID
    sse_id = error_registry.get_error_id("sse")
    print(f"SSE error ID: {sse_id}")
    
    print("Error registry test passed!\n")


def test_dimension_validation():
    """Test dimension validation logic."""
    print("Testing dimension validation...")
    
    # Test with correct dimensions
    try:
        # Langmuir requires 2 parameters
        dim = objective_registry.get_dimension("langmuir")
        print(f"Langmuir requires {dim} parameters")
        
        # Sips requires 3 parameters
        dim = objective_registry.get_dimension("sips")
        print(f"Sips requires {dim} parameters")
        
        print("Dimension validation passed")
    except Exception as e:
        print(f"Dimension validation failed: {e}")
    
    print("Dimension validation test completed!\n")


def test_all_objectives():
    """Test all registered objectives."""
    print("Testing all registered objectives...")
    
    objectives = objective_registry.list_objectives()
    
    for obj_name, obj_info in objectives.items():
        dim = obj_info["dim"]
        model_id = obj_info["model_id"]
        print(f"{obj_name}: {dim} parameters, model_id={model_id}")
    
    print("All objectives test completed!\n")


def test_all_errors():
    """Test all registered error metrics."""
    print("Testing all registered error metrics...")
    
    errors = error_registry.list_errors()
    
    for error_name, error_info in errors.items():
        error_id = error_info["error_id"]
        print(f"{error_name}: error_id={error_id}")
    
    print("All errors test completed!\n")


def main():
    """Run all tests."""
    print("=== Running Logic Tests ===\n")
    
    test_objective_registry()
    test_error_registry()
    test_dimension_validation()
    test_all_objectives()
    test_all_errors()
    
    print("=== All Tests Completed ===")


if __name__ == "__main__":
    main()
