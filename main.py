import time
import os
import csv
import json
import numpy as np
from datetime import datetime
from hierarchical_pso import hierarchical_pso


def run_cross_validation_grid():
    """Run cross-validation grid with early stopping and export results to CSV."""
    # Constants for cross-validation
    DIVISIONS = [1, 2, 4, 8]
    PARTICLES = [10, 100, 1000]
    # Test all isotherm objective functions
    OBJECTIVES = ["langmuir", "sips", "toth", "bet", "gab", "newton"]
    ITER_N = 10000
    THRESHOLD = 1e-5
    W = 0.8
    C1 = 1.8
    C2 = 2.3
    ERROR = "mae"  # Use MAE for all objectives
    
    # Sample data for isotherm fitting
    p = [0.271, 1.448, 2.705, 3.948, 5.131]
    q = [0.905, 1.983, 2.358, 2.548, 2.673]
    
    print(f"\n=== Cross-Validation Grid ===\n")
    print(f"Divisions: {DIVISIONS}")
    print(f"Particles: {PARTICLES}")
    print(f"Objectives: {OBJECTIVES}")
    print(f"Max iterations: {ITER_N}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Error metric: {ERROR}")
    print(f"Total runs: {len(DIVISIONS) * len(PARTICLES) * len(OBJECTIVES)}\n")
    
    # Initialize CSV file with headers
    csv_filename = 'cv_results.csv'
    fieldnames = [
        'divisions', 'particles', 'iterations_executed', 'early_stop_reason',
        'best_fitness', 'best_position', 'elapsed_seconds', 'objective',
        'error', 'bounds_strategy', 'timestamp_iso'
    ]
    
    # Write CSV header
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    results = []
    
    for divisions in DIVISIONS:
        for particles in PARTICLES:
            for objective in OBJECTIVES:
                print(f"Running: divisions={divisions}, particles={particles}, objective={objective}")
                
                # Seed for determinism
                np.random.seed(42)
                
                start_time = time.time()
                
                try:
                    best_pos, best_fit, iterations_executed, early_stop_reason = hierarchical_pso(
                        p, q,
                        part_n=particles,
                        iter_n=ITER_N,
                        divisions=divisions,
                        w=W,
                        c1=C1,
                        c2=C2,
                        objective=objective,
                        error=ERROR,
                        threshold_fitness=THRESHOLD
                    )
                    
                    elapsed_seconds = time.time() - start_time
                    
                    # Create result row
                    result_row = {
                        'divisions': divisions,
                        'particles': particles,
                        'iterations_executed': iterations_executed,
                        'early_stop_reason': early_stop_reason,
                        'best_fitness': float(best_fit),
                        'best_position': json.dumps(best_pos.tolist()),
                        'elapsed_seconds': elapsed_seconds,
                        'objective': objective,
                        'error': ERROR,
                        'bounds_strategy': 'auto_by_objective_dim',
                        'timestamp_iso': datetime.now().isoformat()
                    }
                    
                    print(f"  Result: fitness={best_fit:.6e}, iterations={iterations_executed}, "
                          f"stop_reason={early_stop_reason}, time={elapsed_seconds:.2f}s")
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    # Add error result
                    result_row = {
                        'divisions': divisions,
                        'particles': particles,
                        'iterations_executed': 0,
                        'early_stop_reason': 'error',
                        'best_fitness': float('inf'),
                        'best_position': '[]',
                        'elapsed_seconds': time.time() - start_time,
                        'objective': objective,
                        'error': ERROR,
                        'bounds_strategy': 'auto_by_objective_dim',
                        'timestamp_iso': datetime.now().isoformat()
                    }
                
                # Immediately append result to CSV file
                with open(csv_filename, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(result_row)
                
                results.append(result_row)
                print(f"  → Saved to {csv_filename}")
    
    print(f"\n=== Cross-validation completed! ===")
    print(f"Results exported to: {csv_filename}")
    print(f"Total runs: {len(results)}")
    
    # Print summary statistics
    successful_runs = [r for r in results if r['early_stop_reason'] != 'error']
    if successful_runs:
        fitness_values = [r['best_fitness'] for r in successful_runs]
        early_stops = [r for r in successful_runs if r['early_stop_reason'] == 'fitness']
        
        print(f"Successful runs: {len(successful_runs)}/{len(results)}")
        print(f"Early stops (fitness): {len(early_stops)}")
        print(f"Best fitness overall: {min(fitness_values):.6e}")
        print(f"Average fitness: {np.mean(fitness_values):.6e}")


def run_smoke_tests():
    """Run smoke tests for early stopping validation."""
    print("\n=== Early Stopping Smoke Tests ===")
    
    # Sample data
    p = [0.271, 1.448, 2.705]
    q = [0.905, 1.983, 2.358]
    
    # Test 1: Loose threshold (should trigger early stop)
    print("\n1. Testing loose threshold (should trigger early stop):")
    np.random.seed(42)
    
    start_time = time.time()
    result = hierarchical_pso(
        p, q,
        part_n=50,
        iter_n=1000,
        divisions=2,
        objective="langmuir",
        error="mae",
        threshold_fitness=1e-1,  # Loose threshold
        w=0.8,
        c1=1.8,
        c2=2.3
    )
    elapsed = time.time() - start_time
    
    best_pos, best_fit, iterations, stop_reason = result
    print(f"  Result: fitness={best_fit:.6e}, iterations={iterations}, stop_reason={stop_reason}, time={elapsed:.2f}s")
    
    # Test 2: Strict threshold (should hit max iterations)
    print("\n2. Testing strict threshold (should hit max iterations):")
    np.random.seed(42)
    
    start_time = time.time()
    result = hierarchical_pso(
        p, q,
        part_n=20,
        iter_n=100,
        divisions=1,
        objective="langmuir",
        error="mae",
        threshold_fitness=1e-12,  # Very strict threshold
        w=0.8,
        c1=1.8,
        c2=2.3
    )
    elapsed = time.time() - start_time
    
    best_pos, best_fit, iterations, stop_reason = result
    print(f"  Result: fitness={best_fit:.6e}, iterations={iterations}, stop_reason={stop_reason}, time={elapsed:.2f}s")
    
    # Test 3: R2 maximization test
    print("\n3. Testing R2 maximization (should maximize):")
    np.random.seed(42)
    
    start_time = time.time()
    result = hierarchical_pso(
        p, q,
        part_n=30,
        iter_n=200,
        divisions=2,
        objective="langmuir",
        error="r2",
        threshold_fitness=0.8,  # R2 threshold (maximize)
        w=0.8,
        c1=1.8,
        c2=2.3
    )
    elapsed = time.time() - start_time
    
    best_pos, best_fit, iterations, stop_reason = result
    print(f"  Result: fitness={best_fit:.6f}, iterations={iterations}, stop_reason={stop_reason}, time={elapsed:.2f}s")


if __name__ == "__main__":
    print("=== GPU-Based Hierarchical PSO with Early Stopping ===")
    
    # Demo run (keep existing behavior)
    print("\n=== Demo Run ===")
    print("\nSphere model with SSE error:")
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
        c2=2.3,
        threshold_fitness=1e-5
    )

    end_time = time.time()
    best_pos, best_fit, iterations, stop_reason = result

    print(f"Execution time: {end_time - start_time:.5f} seconds")
    print(f"Result: position={best_pos}, fitness={best_fit}, iterations={iterations}, stop_reason={stop_reason}")
    
    # Run smoke tests
    run_smoke_tests()
    
    # Run cross-validation grid (controlled by environment variable)
    if os.getenv("RUN_CV", "1") == "1":
        run_cross_validation_grid()
    else:
        print("\nSkipping cross-validation grid (set RUN_CV=1 to enable)")
