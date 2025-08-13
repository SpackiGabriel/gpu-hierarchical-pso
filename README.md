# GPU-Based Hierarchical PSO

This project implements a hierarchical Particle Swarm Optimization (PSO) algorithm that leverages GPU acceleration using CuPy and custom CUDA kernels. The algorithm divides the parameter space into subintervals, performs localized PSO searches in each subinterval, and then combines the best solutions from each region to execute a final global search.

The implementation now supports multiple objective functions (isotherm models) and error metrics, allowing flexible model selection for fitting experimental data.

## Project Structure

```bash
gpu-hierarchical-pso/ 
├── kernel.py          # Contains the CUDA kernel code and compiles the kernel using CuPy
├── pso_gpu.py         # Implements the GPU_PSO class that encapsulates the GPU-based PSO algorithm
├── hierarchical_pso.py # Orchestrates the hierarchical PSO execution over divided parameter spaces
├── utils.py           # Provides helper functions, such as dividing intervals into subintervals
├── registries.py      # Defines registries for objective functions and error metrics
├── main.py            # Main entry point to run the hierarchical PSO optimizer
└── test_objectives.py # Test suite for validating the implementation
```

## Prerequisites

- **Python 3.7+**
- **CUDA Toolkit** (compatible with your GPU)
- **CuPy** – a NumPy-like API accelerated with CUDA.

## Installation

First, ensure that you have CUDA installed on your system.

Install the necessary Python package using pip:

```bash
pip install cupy
```

If you prefer conda, you can install CuPy via:
```bash
conda install -c conda-forge cupy
```

## Usage

To run the hierarchical PSO optimizer, execute the main.py script. For example:

```bash
python main.py
```

When executed, the script will:
- Define the input vectors p and q.
- Set the PSO parameters such as the number of particles, iterations, and parameter bounds.
- Run the hierarchical PSO algorithm with different objective functions and error metrics.
- Print the final best solution and the execution time for each test case.

## Supported Objective Functions

The implementation supports the following isotherm models:

| Model | Parameters | Description |
|-------|------------|-------------|
| `langmuir` | 2 ([qmax, b]) | Langmuir isotherm model: q = (qmax * b * p) / (1 + b * p) |
| `sips` | 3 ([qmax, b, n]) | Sips (Langmuir-Freundlich) model: q = (qmax * (b * p)^n) / (1 + (b * p)^n) |
| `toth` | 3 ([qmax, b, t]) | Toth isotherm model: q = (qmax * b * p) / (1 + (b * p)^t)^(1/t) |
| `bet` | 3 ([qm, c, k]) | BET isotherm model |
| `gab` | 3 ([qm, c, k]) | GAB isotherm model |
| `newton` | 2 ([a, b]) | Linear Newton model: q = a + b * p |

## Supported Error Metrics

The implementation supports the following error metrics:

| Metric | Description |
|--------|-------------|
| `sse` | Sum of Squared Errors: Σ(q_obs - q_calc)² |
| `mse` | Mean Squared Error: SSE / n |
| `rmse` | Root Mean Squared Error: √MSE |
| `mae` | Mean Absolute Error: Σ|q_obs - q_calc| / n |
| `mape` | Mean Absolute Percentage Error: Σ|(q_obs - q_calc) / q_obs| / n |
| `r2` | R-squared: 1 - (SSE / SST) |

## API Changes

The `hierarchical_pso` function now accepts two new optional parameters:

- `objective` (str): Name of the objective function (default: "langmuir")
- `error` (str): Name of the error metric (default: "sse")

Example usage:
```python
# Using Sips model with RMSE error
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
```

## File Overview

### kernel.py

Contains the CUDA kernel code for the PSO algorithm and compiles the kernel using CuPy's `RawKernel`. This kernel now supports:
- Multiple objective functions via switch-based selection (Strategy A)
- Multiple error metrics via switch-based selection
- Template-based kernel generation for branch-free specialized kernels (Strategy B)
- Numerical stability improvements and parameter validation

### pso_gpu.py

Implements the `GPU_PSO` class which:
- Initializes particle positions and velocities within user-specified bounds.
- Executes PSO iterations by invoking the CUDA kernel.
- Updates each particle's personal best as well as the global best solution.
- Validates objective function dimensions and error metric selection.
- Handles fitness initialization and comparison based on the selected error metric.

### hierarchical_pso.py

Defines the `hierarchical_pso` function that:
- Divides the overall parameter space into smaller subintervals.
- Executes a localized PSO on each subinterval.
- Aggregates the best solutions from each subinterval to perform a final global optimization.
- Validates objective and error parameters before execution.
- Automatically sets parameter bounds based on the selected objective function.

### registries.py

Defines registries for objective functions and error metrics:
- `objective_registry`: Maps objective function names to required dimensions and device IDs
- `error_registry`: Maps error metric names to device IDs
- Provides validation and lookup functions for registered models and metrics

### utils.py

Provides helper functions used throughout the project, such as `divide_intervals` which splits the parameter bounds into subintervals based on a specified number of divisions.

### main.py

Serves as the main entry point for the application. It demonstrates:
- Different objective functions with various error metrics
- Proper parameter configuration for each model
- Execution time and result reporting

### test_objectives.py

Provides a comprehensive test suite that validates:
- Registry functionality
- Dimension validation
- Synthetic data fitting
- All registered objective functions
- All registered error metrics

## Performance Considerations

Two kernel strategies are implemented:

1. **Switch-based kernel (Strategy A, default)**: A single compiled kernel with switch statements to select the model and error metric. This approach avoids recompilation and is robust for rapid model swapping.

2. **Template-based kernel (Strategy B)**: Dynamic kernel generation that assembles source code by injecting the chosen objective implementation and error function, then compiles a specialized `RawKernel`. This approach eliminates branches in device code but has compilation overhead for new model combinations.

For most use cases, Strategy A is recommended as it provides immediate robustness without compilation overhead after the initial kernel load.

## Numerical Stability

The implementation includes several numerical stability improvements:

- Overflow protection for exponential calculations
- Division by zero prevention with epsilon values
- Parameter validation to ensure physically meaningful ranges
- Penalty functions for invalid parameter combinations
- Proper handling of edge cases in isotherm equations

## Testing

Run the test suite to validate the implementation:

```bash
python test_objectives.py
```

The test suite verifies:
- Registry functionality
- Dimension validation
- Synthetic data fitting with known parameters
- All registered objective functions
- All registered error metrics
- Error handling for invalid configurations

## Extending the Implementation

To add new objective functions:
1. Register the function in `registries.py` with required dimension and device ID
2. Add the model implementation in `kernel.py` within the switch statement
3. Update the template-based generation functions if using Strategy B

To add new error metrics:
1. Register the metric in `registries.py` with device ID
2. Add the error calculation in `kernel.py` within the error switch statement
3. Add finalization code if needed (e.g., for R2 calculation)

