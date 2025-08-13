"""
Test file for validating the objective functions and error metrics implementation.
"""

import numpy as np
import cupy as cp
from gpu_pso import GPU_PSO
from hierarchical_pso import hierarchical_pso
from kernel import pso_kernel
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


def test_synthetic_data():
    """Test with synthetic data where we know the true parameters."""
    print("Testing with synthetic data...")
    
    # Generate synthetic data for Langmuir model
    # True parameters: qmax=3.5, b=0.02
    p_true = np.linspace(0.1, 50, 20)
    qmax_true = 3.5
    b_true = 0.02
    q_true = (qmax_true * b_true * p_true) / (1.0 + b_true * p_true) + np.random.normal(0, 0.01, len(p_true))
    
    print(f"True parameters - qmax: {qmax_true}, b: {b_true}")
    
    # Test Langmuir + SSE
    print("\nTesting Langmuir model with SSE error:")
    result = hierarchical_pso(
        p_true, q_true,
        part_n=5000,
        iter_n=50,
        objective="langmuir",
        error="sse",
        divisions=2,
        w=0.8,
        c1=1.8,
        c2=1.8
    )
    print(f"Estimated parameters: {result[0]}")
    print(f"Fitness: {result[1]}")
    
    # Test Sips + RMSE
    print("\nTesting Sips model with RMSE error:")
    result = hierarchical_pso(
        p_true, q_true,
        part_n=5000,
        iter_n=50,
        objective="sips",
        error="rmse",
        divisions=2,
        w=0.8,
        c1=1.8,
        c2=1.8
    )
    print(f"Estimated parameters: {result[0]}")
    print(f"Fitness: {result[1]}")
    
    print("Synthetic data test completed!\n")


def test_dimension_validation():
    """Test dimension validation for different objectives."""
    print("Testing dimension validation...")
    
    # Test with correct dimensions
    try:
        # Langmuir requires 2 parameters
        bounds_langmuir = [[0, 10], [0, 1]]
        result = hierarchical_pso(
            [1, 2, 3], [1, 2, 3],
            part_n=100,
            iter_n=10,
            param_bounds=bounds_langmuir,
            objective="langmuir",
            error="sse",
            divisions=1
        )
        print("Langmuir dimension validation passed")
    except Exception as e:
        print(f"Langmuir dimension validation failed: {e}")
    
    # Test with incorrect dimensions
    try:
        # Langmuir requires 2 parameters, but we provide 3
        bounds_wrong = [[0, 10], [0, 1], [0, 1]]
        result = hierarchical_pso(
            [1, 2, 3], [1, 2, 3],
            part_n=100,
            iter_n=10,
            param_bounds=bounds_wrong,
            objective="langmuir",
            error="sse",
            divisions=1
        )
        print("ERROR: Langmuir dimension validation should have failed")
    except ValueError as e:
        print(f"Langmuir dimension validation correctly failed: {e}")
    
    print("Dimension validation test completed!\n")


def test_all_objectives():
    """Test all registered objectives with sample data."""
    print("Testing all registered objectives...")
    
    # Sample data
    p = [0.271, 1.448, 2.705, 3.948, 5.131]
    q = [0.905, 1.983, 2.358, 2.548, 2.673]
    
    objectives = objective_registry.list_objectives()
    errors = error_registry.list_errors()
    
    for obj_name in objectives.keys():
        dim = objective_registry.get_dimension(obj_name)
        bounds = [[0, 100] for _ in range(dim)]
        
        print(f"\nTesting {obj_name} model:")
        try:
            result = hierarchical_pso(
                p, q,
                part_n=1000,
                iter_n=20,
                param_bounds=bounds,
                objective=obj_name,
                error="sse",
                divisions=1
            )
            print(f"  Success - Parameters: {result[0]}, Fitness: {result[1]}")
        except Exception as e:
            print(f"  Failed: {e}")
    
    print("All objectives test completed!\n")


def test_all_errors():
    """Test all registered error metrics with sample data."""
    print("Testing all registered error metrics...")
    
    # Sample data
    p = [0.271, 1.448, 2.705, 3.948, 5.131]
    q = [0.905, 1.983, 2.358, 2.548, 2.673]
    
    errors = error_registry.list_errors()
    
    for error_name in errors.keys():
        print(f"\nTesting with {error_name} error:")
        try:
            result = hierarchical_pso(
                p, q,
                part_n=1000,
                iter_n=20,
                objective="langmuir",
                error=error_name,
                divisions=1
            )
            print(f"  Success - Parameters: {result[0]}, Fitness: {result[1]}")
        except Exception as e:
            print(f"  Failed: {e}")
    
    print("All errors test completed!\n")


def main():
    """Run all tests."""
    print("=== Running Objective Functions and Error Metrics Tests ===\n")
    
    test_objective_registry()
    test_error_registry()
    test_synthetic_data()
    test_dimension_validation()
    test_all_objectives()
    test_all_errors()
    
    print("=== All Tests Completed ===")


if __name__ == "__main__":
    main()
