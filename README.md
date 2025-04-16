# GPU-Based Hierarchical PSO

This project implements a hierarchical Particle Swarm Optimization (PSO) algorithm that leverages GPU acceleration using CuPy and custom CUDA kernels. The algorithm divides the parameter space into subintervals, performs localized PSO searches in each subinterval, and then combines the best solutions from each region to execute a final global search.

## Project Structure

```bash
pso_project/ 
├── kernel.py # Contains the CUDA kernel code and compiles the kernel using CuPy. 
├── pso_gpu.py # Implements the GPU_PSO class that encapsulates the GPU-based PSO algorithm. 
├── hierarchical_pso.py # Orchestrates the hierarchical PSO execution over divided parameter spaces. 
├── utils.py # Provides helper functions, such as dividing intervals into subintervals. └── main.py # Main entry point to run the hierarchical PSO optimizer.
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
conda install -c conda-forge cupy

## Usage

To run the hierarchical PSO optimizer, execute the main.py script. For example:

```bash
python main.py
```

When executed, the script will:
- Define the input vectors p and q.
- Set the PSO parameters such as the number of particles, iterations, and parameter bounds.
- Run the hierarchical PSO algorithm.
- Print the final best solution and the execution time.

## File Overview

### kernel.py

Contains the CUDA kernel code for the PSO algorithm and compiles the kernel using CuPy's `RawKernel`. This kernel updates particle positions, velocities, and computes the fitness based on the model's parameters.

### pso_gpu.py
Implements the `GPU_PSO` class which:
- Initializes particle positions and velocities within user-specified bounds.
- Executes PSO iterations by invoking the CUDA kernel.
- Updates each particle's personal best as well as the global best solution.

### hierarchical_pso.py
Defines the `hierarchical_pso` function that:
- Divides the overall parameter space into smaller subintervals.
- Executes a localized PSO on each subinterval.
- Aggregates the best solutions from each subinterval to perform a final global optimization.

### utils.py
Provides helper functions used throughout the project, such as `divide_intervals` which splits the parameter bounds into subintervals based on a specified number of divisions.

### main.py
Serves as the main entry point for the application. It sets up the input data and PSO parameters, invokes the hierarchical PSO solver, and prints the output (best solution and execution time).
