from itertools import product
from gpu_pso import GPU_PSO
from utils import divide_intervals
from kernel import pso_kernel


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

    Returns:
        tuple: Best global position (as a NumPy array) and the corresponding fitness value.
    """
    if param_bounds is None:
        param_bounds = [[0, 1000000], [0, 1000000]]
    divided_intervals = divide_intervals(param_bounds, divisions)
    subinterval_combinations = list(product(*divided_intervals))

    best_particles = []

    for subinterval in subinterval_combinations:
        pso_instance = GPU_PSO(p, q, part_n, len(param_bounds), subinterval, pso_kernel)
        best_pos, best_fit = pso_instance.optimize(iter_n, w, c1, c2)
        best_particles.append((best_pos, best_fit))

    global_positions = [pos for pos, _ in best_particles]
    global_pso = GPU_PSO(p, q, len(global_positions), len(param_bounds), param_bounds, pso_kernel, initial_positions=global_positions)
    global_best, global_best_fitness = global_pso.optimize(iter_n, w, c1, c2)

    print(f"Final global best particle: {global_best}, Fitness: {global_best_fitness}")
    return global_best, global_best_fitness
