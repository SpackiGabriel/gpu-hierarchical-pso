"""
Test file for validating the dimension validation in hierarchical PSO without GPU.
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from registries import objective_registry, error_registry


def test_dimension_validation():
    """Test dimension validation for different objectives."""
    print("Testing dimension validation logic...")
    
    # Test with correct dimensions
    try:
        # Langmuir requires 2 parameters
        bounds_langmuir = [[0, 10], [0, 1]]
        dim = objective_registry.get_dimension("langmuir")
        
        if len(bounds_langmuir) != dim:
            raise ValueError(f"Objective 'langmuir' requires {dim} parameters, but got {len(bounds_langmuir)} bounds")
        
        print("Langmuir dimension validation passed")
    except Exception as e:
        print(f"Langmuir dimension validation failed: {e}")
    
    # Test with incorrect dimensions
    try:
        # Langmuir requires 2 parameters, but we provide 3
        bounds_wrong = [[0, 10], [0, 1], [0, 1]]
        dim = objective_registry.get_dimension("langmuir")
        
        if len(bounds_wrong) != dim:
            raise ValueError(f"Objective 'langmuir' requires {dim} parameters, but got {len(bounds_wrong)} bounds")
        
        print("ERROR: Langmuir dimension validation should have failed")
    except ValueError as e:
        print(f"Langmuir dimension validation correctly failed: {e}")
    
    # Test Sips model (requires 3 parameters)
    try:
        bounds_sips = [[0, 10], [0, 1], [0, 2]]
        dim = objective_registry.get_dimension("sips")
        
        if len(bounds_sips) != dim:
            raise ValueError(f"Objective 'sips' requires {dim} parameters, but got {len(bounds_sips)} bounds")
        
        print("Sips dimension validation passed")
    except Exception as e:
        print(f"Sips dimension validation failed: {e}")
    
    print("Dimension validation test completed!\n")


def test_objective_and_error_validation():
    """Test validation of objective and error names."""
    print("Testing objective and error validation...")
    
    # Test valid objective
    objective = "langmuir"
    if objective not in objective_registry.list_objectives():
        print(f"ERROR: Objective '{objective}' not registered")
    else:
        print(f"Objective '{objective}' validation passed")
    
    # Test invalid objective
    objective = "invalid_model"
    if objective not in objective_registry.list_objectives():
        print(f"Objective '{objective}' correctly identified as unregistered")
    else:
        print(f"ERROR: Objective '{objective}' should not be registered")
    
    # Test valid error
    error = "sse"
    if error not in error_registry.list_errors():
        print(f"ERROR: Error metric '{error}' not registered")
    else:
        print(f"Error metric '{error}' validation passed")
    
    # Test invalid error
    error = "invalid_error"
    if error not in error_registry.list_errors():
        print(f"Error metric '{error}' correctly identified as unregistered")
    else:
        print(f"ERROR: Error metric '{error}' should not be registered")
    
    print("Objective and error validation test completed!\n")


def main():
    """Run all tests."""
    print("=== Running Validation Tests ===\n")
    
    test_dimension_validation()
    test_objective_and_error_validation()
    
    print("=== All Tests Completed ===")


if __name__ == "__main__":
    main()
