from itertools import product
from gpu_pso import GPU_PSO
from utils import divide_intervals
from kernel import pso_kernel
from registries import objective_registry, error_registry


def hierarchical_pso(
    p,
    q,
    part_n: int = 100,
    iter_n: int = 100,
    param_bounds=None,
    divisions: int = 4,
    w: float = 0.8,
    c1: float = 1.8,
    c2: float = 1.8,
    objective: str = "langmuir",
    error: str = "sse",
    threshold_fitness: float = 1e-5,
):
    """
    Execute the hierarchical PSO algorithm.

    This function:
      1. Divides the parameter space into subintervals.
      2. Runs the PSO for each combination of subintervals.
      3. Aggregates the best results from each subinterval and performs a final global search.

    Args:
        p: Input vector p.
        q: Input vector q.
        part_n (int, optional): Number of particles for each subinterval PSO. Defaults to 100.
        iter_n (int, optional): Number of iterations to run each PSO. Defaults to 100.
        param_bounds (list, optional): List of tuples (lower, upper) for each dimension. Defaults to [[0, 1000000], [0, 1000000]].
        divisions (int, optional): Number of divisions for each parameter bound. Defaults to 4.
        w (float, optional): Inertia weight. Defaults to 0.8.
        c1 (float, optional): Cognitive acceleration coefficient. Defaults to 1.8.
        c2 (float, optional): Social acceleration coefficient. Defaults to 1.8.
        objective (str): Name of the objective function. Defaults to "langmuir".
        error (str): Name of the error metric. Defaults to "sse".

    Returns:
        tuple: (best_position, best_fitness, total_iterations, early_stop_reason)
    """
    # Validate objective and error
    if objective not in objective_registry.list_objectives():
        raise ValueError(f"Objective '{objective}' not registered")

    if error not in error_registry.list_errors():
        raise ValueError(f"Error metric '{error}' not registered")

    # Get required dimension for the objective
    required_dim = objective_registry.get_dimension(objective)

    if param_bounds is None:
        # Set default bounds based on the required dimension
        param_bounds = [[0, 1000000] for _ in range(required_dim)]

    # Validate dimension
    if len(param_bounds) != required_dim:
        raise ValueError(f"Objective '{objective}' requires {required_dim} parameters, but got {len(param_bounds)} bounds")

    divided_intervals = divide_intervals(param_bounds, divisions)
    subinterval_combinations = list(product(*divided_intervals))

    best_particles = []
    total_iterations = 0
    combined_early_stop_reason = "max_iter"

    for subinterval in subinterval_combinations:
        pso_instance = GPU_PSO(p, q, part_n, len(param_bounds), subinterval, pso_kernel, objective, error, threshold_fitness=threshold_fitness)
        best_pos, best_fit, sub_iterations, sub_early_stop = pso_instance.optimize(iter_n, w, c1, c2)
        best_particles.append((best_pos, best_fit))
        total_iterations += sub_iterations
        
        # Track if any subinterval stopped early due to fitness
        if sub_early_stop == "fitness":
            combined_early_stop_reason = "fitness"

    global_positions = [pos for pos, _ in best_particles]
    global_pso = GPU_PSO(p, q, len(global_positions), len(param_bounds), param_bounds, pso_kernel, objective, error, 
                        initial_positions=global_positions, threshold_fitness=threshold_fitness)
    global_best, global_best_fitness, global_iterations, global_early_stop = global_pso.optimize(iter_n, w, c1, c2)
    total_iterations += global_iterations
    
    # Final early stop reason is global phase result
    final_early_stop_reason = global_early_stop

    print(f"Final global best particle: {global_best}, Fitness: {global_best_fitness}")
    return global_best, global_best_fitness, total_iterations, final_early_stop_reason
