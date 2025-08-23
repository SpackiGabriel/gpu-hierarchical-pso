import cupy as cp
from registries import objective_registry, error_registry


class GPU_PSO:
    """
    Encapsulates the Particle Swarm Optimization (PSO) algorithm executed on the GPU.

    Attributes:
        p (cupy.ndarray): Input vector p.
        q (cupy.ndarray): Input vector q.
        n_particles (int): Number of particles.
        dim (int): Dimensionality of each particle.
        param_bounds (list): List of tuples defining (lower, upper) bounds for each dimension.
        kernel: Compiled CUDA kernel used for updating particles.
        threads_per_block (int): Number of threads per block.
        blocks_per_grid (int): Number of blocks per grid.
        position (cupy.ndarray): Current positions of the particles.
        velocity (cupy.ndarray): Current velocities of the particles.
        personal_best (cupy.ndarray): Best known positions for each particle.
        fitness (cupy.ndarray): Current fitness values for the particles.
        personal_best_fitness (cupy.ndarray): Best fitness values per particle.
        global_best (cupy.ndarray): Best global position among all particles.
        global_best_fitness (float): Best global fitness value.
        objective (str): Name of the objective function.
        error (str): Name of the error metric.
        model_id (int): Device-side identifier for the objective function.
        error_id (int): Device-side identifier for the error metric.
        sst (float): Total sum of squares for R2 calculation.
    """

    def __init__(
        self,
        p,
        q,
        n_particles: int,
        dim: int,
        param_bounds,
        kernel,
        objective: str = "langmuir",
        error: str = "sse",
        threads_per_block: int = 256,
        initial_positions=None,
        threshold_fitness: float = 1e-5,
    ) -> None:
        """
        Initialize the GPU_PSO instance.

        Args:
            p: Input vector p.
            q: Input vector q.
            n_particles (int): Number of particles.
            dim (int): Dimensionality of each particle.
            param_bounds: List of tuples (lower, upper) for each dimension.
            kernel: Compiled CUDA kernel for updating positions and velocities.
            objective (str): Name of the objective function.
            error (str): Name of the error metric.
            threads_per_block (int, optional): Number of threads per block. Defaults to 256.
            initial_positions (optional): Predefined initial positions for particles.
        """
        self.p = cp.asarray(p, dtype=cp.float64)
        self.q = cp.asarray(q, dtype=cp.float64)
        self.n_particles: int = n_particles

        # Validate and set objective function
        self.objective = objective
        self.model_id = objective_registry.get_model_id(objective)
        required_dim = objective_registry.get_dimension(objective)
        if dim != required_dim:
            raise ValueError(f"Objective '{objective}' requires dimension {required_dim}, but got {dim}")
        self.dim: int = dim

        # Validate and set error metric
        self.error = error
        self.error_id = error_registry.get_error_id(error)

        self.param_bounds = param_bounds
        self.kernel = kernel
        self.threads_per_block: int = threads_per_block
        self.blocks_per_grid: int = (n_particles * dim + threads_per_block - 1) // threads_per_block

        # Calculate SST for R2 calculation if needed
        self.sst = 0.0
        if self.error == "r2":
            q_mean = cp.mean(self.q)
            self.sst = cp.sum((self.q - q_mean) ** 2)

        # Early stopping parameters
        self.threshold_fitness = threshold_fitness
        self.is_maximize = 1 if self.error == "r2" else 0

        self.initialize_particles(initial_positions)

    def initialize_particles(self, initial_positions=None) -> None:
        """
        Initialize particle positions, velocities, and best known values.

        If initial_positions is not provided, positions are generated randomly within the
        bounds specified by param_bounds.

        Args:
            initial_positions (optional): Predefined initial positions for particles.
        """
        if initial_positions is None:
            self.position = cp.empty((self.n_particles, self.dim), dtype=cp.float64)
            for dim_idx in range(self.dim):
                lower, upper = self.param_bounds[dim_idx]
                self.position[:, dim_idx] = cp.random.uniform(lower, upper, size=self.n_particles)
        else:
            self.position = cp.asarray(initial_positions, dtype=cp.float64)
        self.velocity = cp.zeros_like(self.position)
        self.personal_best = self.position.copy()

        # Initialize fitness values based on error metric
        if self.error == "r2":
            self.fitness = cp.full(self.n_particles, -cp.inf, dtype=cp.float64)
            self.personal_best_fitness = cp.full(self.n_particles, -cp.inf, dtype=cp.float64)
        else:
            # Other error metrics are minimized (0 is perfect)
            self.fitness = cp.full(self.n_particles, cp.inf, dtype=cp.float64)
            self.personal_best_fitness = cp.full(self.n_particles, cp.inf, dtype=cp.float64)

        self.global_best = cp.zeros(self.dim, dtype=cp.float64)
        if self.error == "r2":
            self.global_best_fitness = -cp.inf
        else:
            self.global_best_fitness = cp.inf

    def run_iteration(self, w: float, c1: float, c2: float, stop_flag_dev, gbest_fit_dev) -> None:
        """
        Run one iteration of the PSO algorithm, updating particle positions, velocities,
        and fitness values.

        Args:
            w (float): Inertia weight.
            c1 (float): Cognitive acceleration coefficient.
            c2 (float): Social acceleration coefficient.
            stop_flag_dev: Device scalar for early stopping flag.
            gbest_fit_dev: Device scalar for global best fitness.
        """
        r1 = cp.random.rand(self.n_particles, self.dim).astype(cp.float64)
        r2 = cp.random.rand(self.n_particles, self.dim).astype(cp.float64)
        self.kernel(
            (self.blocks_per_grid,),
            (self.threads_per_block,),
            (
                self.position.ravel(),
                self.velocity.ravel(),
                self.personal_best.ravel(),
                self.global_best,
                self.fitness,
                self.personal_best_fitness,
                r1.ravel(),
                r2.ravel(),
                w,
                c1,
                c2,
                self.n_particles,
                self.dim,
                len(self.p),
                self.p,
                self.q,
                self.model_id,
                self.error_id,
                self.sst,
                self.threshold_fitness,
                self.is_maximize,
                stop_flag_dev,
                gbest_fit_dev,
            )
        )

        # Update global best based on error metric
        if self.error == "r2":
            # For R2, higher values are better
            best_idx = cp.argmax(self.personal_best_fitness)
            if self.personal_best_fitness[best_idx] > self.global_best_fitness:
                self.global_best_fitness = self.personal_best_fitness[best_idx]
                self.global_best = self.personal_best[best_idx]
        else:
            # For other metrics, lower values are better
            best_idx = cp.argmin(self.personal_best_fitness)
            if self.personal_best_fitness[best_idx] < self.global_best_fitness:
                self.global_best_fitness = self.personal_best_fitness[best_idx]
                self.global_best = self.personal_best[best_idx]

    def optimize(self, n_iterations: int, w: float, c1: float, c2: float):
        """
        Execute the PSO algorithm for a given number of iterations and return the best
        position and its fitness.

        Args:
            n_iterations (int): Number of iterations.
            w (float): Inertia weight.
            c1 (float): Cognitive acceleration coefficient.
            c2 (float): Social acceleration coefficient.

        Returns:
            tuple: A tuple containing (best_position, best_fitness, iterations_executed, early_stop_reason).
        """
        # Allocate device scalars for early stopping
        stop_flag_dev = cp.zeros((), dtype=cp.int32)
        if self.is_maximize:
            init_val = -cp.inf
        else:
            init_val = cp.inf
        gbest_fit_dev = cp.full((), init_val, dtype=cp.float64)
        
        iterations_executed = 0
        early_stop_reason = "max_iter"
        
        for iteration in range(n_iterations):
            self.run_iteration(w, c1, c2, stop_flag_dev, gbest_fit_dev)
            iterations_executed = iteration + 1
            
            # Check stop flag (minimal host-device transfer)
            flag = int(stop_flag_dev.get())
            if flag == 1:
                early_stop_reason = "fitness"
                break
                
            # Update host-side global best tracking
            self.update_global_best_from_device()
        
        return cp.asnumpy(self.global_best), self.global_best_fitness, iterations_executed, early_stop_reason
    
    def update_global_best_from_device(self):
        """
        Update host-side global best tracking from personal best values.
        """
        if self.error == "r2":
            # For R2, higher values are better
            best_idx = cp.argmax(self.personal_best_fitness)
            if self.personal_best_fitness[best_idx] > self.global_best_fitness:
                self.global_best_fitness = self.personal_best_fitness[best_idx]
                self.global_best = self.personal_best[best_idx]
        else:
            # For other metrics, lower values are better
            best_idx = cp.argmin(self.personal_best_fitness)
            if self.personal_best_fitness[best_idx] < self.global_best_fitness:
                self.global_best_fitness = self.personal_best_fitness[best_idx]
                self.global_best = self.personal_best[best_idx]
