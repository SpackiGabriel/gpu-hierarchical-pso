import time
from hierarchical_pso import hierarchical_pso

if __name__ == "__main__":
    p = [0.271004, 1.44862, 2.70512, 3.94841, 5.13112, 6.61931, 8.60419, 11.0863, 13.5677, 16.068,
            18.5552, 43.8868, 44.409]
    q = [0.905796, 1.98353, 2.35874, 2.5484, 2.67333, 2.7765, 2.88334, 2.9774, 3.04683, 3.09766,
            3.14708, 3.30683, 3.30725]

    print("=== Demonstrating different objective functions and error metrics ===")
    
    # Test 1: Default Langmuir + SSE
    print("\n1. Langmuir model with SSE error:")
    start_time = time.time()
    result = hierarchical_pso(
        p, q,
        part_n=10000,
        iter_n=100,
        objective="langmuir",
        error="sse",
        divisions=3,
        w=0.8,
        c1=1.8,
        c2=1.8
    )
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.5f} seconds")
    print("Result:", result)
    
    # Test 2: Sips model with RMSE error
    print("\n2. Sips model with RMSE error:")
    start_time = time.time()
    result = hierarchical_pso(
        p, q,
        part_n=10000,
        iter_n=100,
        objective="sips",
        error="rmse",
        divisions=3,
        w=0.8,
        c1=1.8,
        c2=1.8
    )
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.5f} seconds")
    print("Result:", result)
    
    # Test 3: Toth model with MAE error
    print("\n3. Toth model with MAE error:")
    start_time = time.time()
    result = hierarchical_pso(
        p, q,
        part_n=10000,
        iter_n=100,
        objective="toth",
        error="mae",
        divisions=3,
        w=0.8,
        c1=1.8,
        c2=1.8
    )
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.5f} seconds")
    print("Result:", result)
    
    # Test 4: BET model with R2 error
    print("\n4. BET model with R2 error:")
    start_time = time.time()
    result = hierarchical_pso(
        p, q,
        part_n=10000,
        iter_n=100,
        objective="bet",
        error="r2",
        divisions=3,
        w=0.8,
        c1=1.8,
        c2=1.8
    )
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.5f} seconds")
    print("Result:", result)
    
    print("\n=== Demonstration complete ===")
