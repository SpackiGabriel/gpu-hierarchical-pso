import time
from hierarchical_pso import hierarchical_pso


if __name__ == "__main__":
    print("=== Demonstrating different objective functions and error metrics ===")

    # Test 1: Default Langmuir + MAE
    print("\n1. Sphere model with MAE error:")
    start_time = time.time()
    result = hierarchical_pso(
        [], [],
        part_n=100,
        iter_n=100,
        objective="sphere",
        error="sse",
        divisions=8,
        w=0.8,
        c1=1.8,
        c2=2.3
    )

    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.5f} seconds")
    print("Result:", result)
