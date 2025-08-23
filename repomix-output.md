This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where content has been compressed (code blocks are separated by ⋮---- delimiter).

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Content has been compressed - code blocks are separated by ⋮---- delimiter
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
tests/
  test_autobounds.py
  test_logic.py
  test_objectives.py
  test_validation.py
.gitignore
CHANGELOG.md
gpu_pso.py
hierarchical_pso.py
kernel.py
main.py
README.md
registries.py
requirements.txt
run_tests.py
todo.MD
utils.py
```

# Files

## File: tests/test_autobounds.py
````python
"""
Test file for demonstrating automatic parameter bounds setting based on objective function.
"""
⋮----
# Add the current directory to the path so we can import our modules
⋮----
def demonstrate_automatic_bounds()
⋮----
"""Demonstrate how parameter bounds are automatically set based on objective function."""
⋮----
# Default bounds when none are provided
default_bound = [0, 1000000]
⋮----
# Test with different objectives
objectives = ["langmuir", "sips", "toth", "bet", "gab", "newton"]
⋮----
dim = objective_registry.get_dimension(objective)
# This is how the bounds would be automatically set in hierarchical_pso
auto_bounds = [default_bound for _ in range(dim)]
⋮----
def main()
⋮----
"""Run the demonstration."""
````

## File: tests/test_logic.py
````python
"""
Simplified test file for validating the objective functions and error metrics logic without GPU.
"""
⋮----
# Add the current directory to the path so we can import our modules
⋮----
def test_objective_registry()
⋮----
"""Test the objective registry functionality."""
⋮----
# Test listing objectives
objectives = objective_registry.list_objectives()
⋮----
# Test getting objective info
langmuir_info = objective_registry.get("langmuir")
⋮----
# Test getting dimension
langmuir_dim = objective_registry.get_dimension("langmuir")
⋮----
# Test getting model ID
langmuir_id = objective_registry.get_model_id("langmuir")
⋮----
def test_error_registry()
⋮----
"""Test the error registry functionality."""
⋮----
# Test listing errors
errors = error_registry.list_errors()
⋮----
# Test getting error info
sse_info = error_registry.get("sse")
⋮----
# Test getting error ID
sse_id = error_registry.get_error_id("sse")
⋮----
def test_dimension_validation()
⋮----
"""Test dimension validation logic."""
⋮----
# Test with correct dimensions
⋮----
# Langmuir requires 2 parameters
dim = objective_registry.get_dimension("langmuir")
⋮----
# Sips requires 3 parameters
dim = objective_registry.get_dimension("sips")
⋮----
def test_all_objectives()
⋮----
"""Test all registered objectives."""
⋮----
dim = obj_info["dim"]
model_id = obj_info["model_id"]
⋮----
def test_all_errors()
⋮----
"""Test all registered error metrics."""
⋮----
error_id = error_info["error_id"]
⋮----
def main()
⋮----
"""Run all tests."""
````

## File: tests/test_objectives.py
````python
"""
Test file for validating the objective functions and error metrics implementation.
"""
⋮----
def test_objective_registry()
⋮----
"""Test the objective registry functionality."""
⋮----
# Test listing objectives
objectives = objective_registry.list_objectives()
⋮----
# Test getting objective info
langmuir_info = objective_registry.get("langmuir")
⋮----
# Test getting dimension
langmuir_dim = objective_registry.get_dimension("langmuir")
⋮----
# Test getting model ID
langmuir_id = objective_registry.get_model_id("langmuir")
⋮----
def test_error_registry()
⋮----
"""Test the error registry functionality."""
⋮----
# Test listing errors
errors = error_registry.list_errors()
⋮----
# Test getting error info
sse_info = error_registry.get("sse")
⋮----
# Test getting error ID
sse_id = error_registry.get_error_id("sse")
⋮----
def test_synthetic_data()
⋮----
"""Test with synthetic data where we know the true parameters."""
⋮----
# Generate synthetic data for Langmuir model
# True parameters: qmax=3.5, b=0.02
p_true = np.linspace(0.1, 50, 20)
qmax_true = 3.5
b_true = 0.02
q_true = (qmax_true * b_true * p_true) / (1.0 + b_true * p_true) + np.random.normal(0, 0.01, len(p_true))
⋮----
# Test Langmuir + SSE
⋮----
result = hierarchical_pso(
⋮----
# Test Sips + RMSE
⋮----
def test_dimension_validation()
⋮----
"""Test dimension validation for different objectives."""
⋮----
# Test with correct dimensions
⋮----
# Langmuir requires 2 parameters
bounds_langmuir = [[0, 10], [0, 1]]
⋮----
# Test with incorrect dimensions
⋮----
# Langmuir requires 2 parameters, but we provide 3
bounds_wrong = [[0, 10], [0, 1], [0, 1]]
⋮----
def test_all_objectives()
⋮----
"""Test all registered objectives with sample data."""
⋮----
# Sample data
p = [0.271, 1.448, 2.705, 3.948, 5.131]
q = [0.905, 1.983, 2.358, 2.548, 2.673]
⋮----
dim = objective_registry.get_dimension(obj_name)
bounds = [[0, 100] for _ in range(dim)]
⋮----
def test_all_errors()
⋮----
"""Test all registered error metrics with sample data."""
⋮----
def main()
⋮----
"""Run all tests."""
````

## File: tests/test_validation.py
````python
"""
Test file for validating the dimension validation in hierarchical PSO without GPU.
"""
⋮----
# Add the current directory to the path so we can import our modules
⋮----
def test_dimension_validation()
⋮----
"""Test dimension validation for different objectives."""
⋮----
# Test with correct dimensions
⋮----
# Langmuir requires 2 parameters
bounds_langmuir = [[0, 10], [0, 1]]
dim = objective_registry.get_dimension("langmuir")
⋮----
# Test with incorrect dimensions
⋮----
# Langmuir requires 2 parameters, but we provide 3
bounds_wrong = [[0, 10], [0, 1], [0, 1]]
⋮----
# Test Sips model (requires 3 parameters)
⋮----
bounds_sips = [[0, 10], [0, 1], [0, 2]]
dim = objective_registry.get_dimension("sips")
⋮----
def test_objective_and_error_validation()
⋮----
"""Test validation of objective and error names."""
⋮----
# Test valid objective
objective = "langmuir"
⋮----
# Test invalid objective
objective = "invalid_model"
⋮----
# Test valid error
error = "sse"
⋮----
# Test invalid error
error = "invalid_error"
⋮----
def main()
⋮----
"""Run all tests."""
````

## File: .gitignore
````
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[codz]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py.cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# UV
#   Similar to Pipfile.lock, it is generally recommended to include uv.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#uv.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock
#poetry.toml

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#   pdm recommends including project-wide configuration in pdm.toml, but excluding .pdm-python.
#   https://pdm-project.org/en/latest/usage/project/#working-with-version-control
#pdm.lock
#pdm.toml
.pdm-python
.pdm-build/

# pixi
#   Similar to Pipfile.lock, it is generally recommended to include pixi.lock in version control.
#pixi.lock
#   Pixi creates a virtual environment in the .pixi directory, just like venv module creates one
#   in the .venv directory. It is recommended not to include this directory in version control.
.pixi

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.envrc
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
#  and can be added to the global gitignore or merged into this file.  For a more nuclear
#  option (not recommended) you can uncomment the following to ignore the entire idea folder.
#.idea/

# Abstra
# Abstra is an AI-powered process automation framework.
# Ignore directories containing user credentials, local state, and settings.
# Learn more at https://abstra.io/docs
.abstra/

# Visual Studio Code
#  Visual Studio Code specific template is maintained in a separate VisualStudioCode.gitignore 
#  that can be found at https://github.com/github/gitignore/blob/main/Global/VisualStudioCode.gitignore
#  and can be added to the global gitignore or merged into this file. However, if you prefer, 
#  you could uncomment the following to ignore the entire vscode folder
# .vscode/

# Ruff stuff:
.ruff_cache/

# PyPI configuration file
.pypirc

# Marimo
marimo/_static/
marimo/_lsp/
__marimo__/

# Streamlit
.streamlit/secrets.toml
````

## File: requirements.txt
````
cupy-cuda12x==13.6.0
fastrlock==0.8.3
numpy==2.3.2
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-curand-cu12==10.3.10.19
nvidia-pyindex==1.0.9
````

## File: run_tests.py
````python
#!/usr/bin/env python3
"""
Test runner script for executing all tests in the GPU-Based Hierarchical PSO project.
"""
⋮----
# Add the current directory to the path so we can import our modules
⋮----
def run_all_tests()
⋮----
"""Run all test suites."""
⋮----
# Import and run each test suite
⋮----
def main()
⋮----
"""Main entry point."""
````

## File: todo.MD
````markdown
# A FAZER

- **Alterar os invervalos de todas as funções**
    - [-10E-10, 10E10]

- **Fazer um for para exportar os resultados**
    - Divisões:
        - 1, 2, 4, 8
    - Partículas:
        - 10, 100, 1000

- **Fixar o número de iterações para 10000**
    - Exportar o número de iterações quando a fitness chegar em 10^-5

- **Alterar número de variáveis no register para 3**

C1: 1.8
C2: 2.3
W: 0.8
````

## File: CHANGELOG.md
````markdown
# Changelog

All notable changes to the GPU-Based Hierarchical PSO project will be documented in this file.

## [1.1.0] - 2025-08-13

### Added
- Pluggable objective functions and error metrics system
- Support for multiple isotherm models: Langmuir, Sips, Toth, BET, GAB, Newton
- Support for multiple error metrics: SSE, MSE, RMSE, MAE, MAPE, R2
- Registry system for managing objective functions and error metrics
- Comprehensive test suite for validating the implementation
- Detailed documentation in README.md
- CHANGELOG.md to track changes

### Changed

#### kernel.py
- Added `model_id` and `error_id` parameters to the CUDA kernel
- Implemented switch-case logic for all objective functions (Langmuir, Sips, Toth, BET, GAB, Newton)
- Implemented switch-case logic for all error metrics (SSE, MSE, RMSE, MAE, MAPE, R2)
- Added numerical stability improvements (overflow protection, division by zero prevention)
- Added parameter validation with penalty functions for invalid combinations
- Added template-based kernel generation functions for Strategy B
- Added support for R2 error metric requiring SST (Sum of Squares Total) calculation

#### gpu_pso.py
- Added `objective` and `error` parameters to GPU_PSO constructor
- Implemented validation of objective and error parameters against registries
- Added dimension checking to ensure particle dimension matches objective parameter dimension
- Updated fitness initialization and comparison logic to handle R2 (maximization) vs other errors (minimization)
- Added SST calculation for R2 error metric
- Modified kernel invocation to pass model_id, error_id, and SST parameters

#### hierarchical_pso.py
- Added `objective` and `error` parameters to hierarchical_pso function
- Implemented validation of objective and error parameters against registries
- Added automatic parameter bounds setting based on objective function dimension
- Updated GPU_PSO instantiation to pass objective and error parameters
- Added dimension validation to ensure parameter bounds match objective requirements

#### registries.py
- Created ObjectiveRegistry class for managing objective functions
- Created ErrorRegistry class for managing error metrics
- Registered all isotherm models with names, dimensions, and device-side IDs
- Registered all error metrics with names and device-side IDs
- Added methods for listing, getting, and validating registered functions and metrics

#### README.md
- Added documentation for all supported objective functions with parameters and descriptions
- Added documentation for all supported error metrics with descriptions
- Updated project structure to include new files
- Added API changes section documenting new parameters
- Added file overview for all components
- Added performance considerations section
- Added numerical stability improvements section
- Added testing section with instructions
- Added extensibility section for adding new models and metrics

#### main.py
- Updated to demonstrate different objective functions with various error metrics
- Added multiple test cases showing usage with Langmuir+SSE, Sips+RMSE, Toth+MAE, and BET+R2
- Added execution time and result reporting for each test case

#### utils.py
- No changes required

#### tests/
- Organized all test files into a dedicated tests/ directory
- Created __init__.py to make tests a Python package

##### test_logic.py
- Created test file for validating registry and basic logic without GPU dependencies
- Added tests for objective and error registry functionality
- Added tests for dimension validation
- Added tests for all registered objectives and errors

##### test_validation.py
- Created test file for validating parameter bounds and objective/error name validation
- Added tests for dimension validation with correct and incorrect parameter counts
- Added tests for objective and error name validation

##### test_autobounds.py
- Created test file for demonstrating automatic parameter bounds setting
- Added demonstration of how bounds are automatically set based on objective function dimension

##### test_objectives.py
- Created comprehensive test suite for validating the full implementation
- Added tests for registry functionality
- Added tests for synthetic data fitting with known parameters
- Added tests for all registered objective functions
- Added tests for all registered error metrics
- Added tests for error handling with invalid configurations

## [1.0.0] - 2025-08-13

### Added
- Initial implementation of GPU-Based Hierarchical PSO
- Basic Langmuir isotherm model with SSE error metric
- Core PSO algorithm with GPU acceleration using CuPy
- Hierarchical optimization approach with parameter space division
````

## File: registries.py
````python
"""
Registries for objective models and error metrics in the PSO implementation.
"""
⋮----
class ObjectiveRegistry
⋮----
"""Registry for objective models."""
⋮----
def __init__(self)
⋮----
def register(self, name: str, dim: int, model_id: int, description: str = "")
⋮----
"""Register a new objective function.

        Args:
            name: Name of the objective function
            dim: Required parameter dimension
            model_id: Device-side identifier
            description: Optional description
        """
⋮----
def get(self, name: str) -> Dict
⋮----
"""Get objective function info by name.

        Args:
            name: Name of the objective function

        Returns:
            Dictionary with objective function info

        Raises:
            ValueError: If objective function is not registered
        """
⋮----
def get_dimension(self, name: str) -> int
⋮----
"""Get required dimension for an objective function.

        Args:
            name: Name of the objective function

        Returns:
            Required parameter dimension
        """
⋮----
def get_model_id(self, name: str) -> int
⋮----
"""Get device-side identifier for an objective function.

        Args:
            name: Name of the objective function

        Returns:
            Device-side identifier
        """
⋮----
def list_objectives(self) -> Dict[str, Dict]
⋮----
"""List all registered objective functions.

        Returns:
            Dictionary of registered objective functions
        """
⋮----
class ErrorRegistry
⋮----
"""Registry for error metrics."""
⋮----
def register(self, name: str, error_id: int, description: str = "")
⋮----
"""Register a new error metric.

        Args:
            name: Name of the error metric
            error_id: Device-side identifier
            description: Optional description
        """
⋮----
"""Get error metric info by name.

        Args:
            name: Name of the error metric

        Returns:
            Dictionary with error metric info

        Raises:
            ValueError: If error metric is not registered
        """
⋮----
def get_error_id(self, name: str) -> int
⋮----
"""Get device-side identifier for an error metric.

        Args:
            name: Name of the error metric

        Returns:
            Device-side identifier
        """
⋮----
def list_errors(self) -> Dict[str, Dict]
⋮----
"""List all registered error metrics.

        Returns:
            Dictionary of registered error metrics
        """
⋮----
# Global registries
objective_registry = ObjectiveRegistry()
error_registry = ErrorRegistry()
⋮----
# Register objective functions
⋮----
# registries.py
⋮----
# Register error metrics
````

## File: hierarchical_pso.py
````python
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
        tuple: Best global position (as a NumPy array) and the corresponding fitness value.
    """
# Validate objective and error
⋮----
# Get required dimension for the objective
required_dim = objective_registry.get_dimension(objective)
⋮----
# Set default bounds based on the required dimension
param_bounds = [[0, 1000000] for _ in range(required_dim)]
⋮----
# Validate dimension
⋮----
divided_intervals = divide_intervals(param_bounds, divisions)
subinterval_combinations = list(product(*divided_intervals))
⋮----
best_particles = []
⋮----
pso_instance = GPU_PSO(p, q, part_n, len(param_bounds), subinterval, pso_kernel, objective, error)
⋮----
global_positions = [pos for pos, _ in best_particles]
global_pso = GPU_PSO(p, q, len(global_positions), len(param_bounds), param_bounds, pso_kernel, objective, error, initial_positions=global_positions)
````

## File: gpu_pso.py
````python
class GPU_PSO
⋮----
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
⋮----
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
⋮----
# Validate and set objective function
⋮----
required_dim = objective_registry.get_dimension(objective)
⋮----
# Validate and set error metric
⋮----
# Calculate SST for R2 calculation if needed
⋮----
q_mean = cp.mean(self.q)
⋮----
def initialize_particles(self, initial_positions=None) -> None
⋮----
"""
        Initialize particle positions, velocities, and best known values.

        If initial_positions is not provided, positions are generated randomly within the
        bounds specified by param_bounds.

        Args:
            initial_positions (optional): Predefined initial positions for particles.
        """
⋮----
# Initialize fitness values based on error metric
⋮----
# Other error metrics are minimized (0 is perfect)
⋮----
def run_iteration(self, w: float, c1: float, c2: float) -> None
⋮----
"""
        Run one iteration of the PSO algorithm, updating particle positions, velocities,
        and fitness values.

        Args:
            w (float): Inertia weight.
            c1 (float): Cognitive acceleration coefficient.
            c2 (float): Social acceleration coefficient.
        """
r1 = cp.random.rand(self.n_particles, self.dim).astype(cp.float64)
r2 = cp.random.rand(self.n_particles, self.dim).astype(cp.float64)
⋮----
# Update global best based on error metric
⋮----
# For R2, higher values are better
best_idx = cp.argmax(self.personal_best_fitness)
⋮----
# For other metrics, lower values are better
best_idx = cp.argmin(self.personal_best_fitness)
⋮----
def optimize(self, n_iterations: int, w: float, c1: float, c2: float)
⋮----
"""
        Execute the PSO algorithm for a given number of iterations and return the best
        position and its fitness.

        Args:
            n_iterations (int): Number of iterations.
            w (float): Inertia weight.
            c1 (float): Cognitive acceleration coefficient.
            c2 (float): Social acceleration coefficient.

        Returns:
            tuple: A tuple containing the best position as a NumPy array and the best fitness value.
        """
````

## File: README.md
````markdown
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
├── setup_env.sh       # Script to set up the virtual environment and install dependencies
├── main.py            # Main entry point to run the hierarchical PSO optimizer
├── run_tests.py       # Script to run all test suites
└── tests/             # Directory containing test suites for validating the implementation
    ├── test_logic.py      # Tests registry and basic logic without GPU dependencies
    ├── test_validation.py # Tests parameter bounds and objective/error name validation
    ├── test_autobounds.py # Demonstrates automatic parameter bounds setting
    └── test_objectives.py # Comprehensive test suite for the full implementation
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

### run_tests.py

A convenience script to run all test suites at once. This script imports and executes all individual test modules.

### tests/

Contains test suites for validating the implementation:

#### test_logic.py
- Tests registry and basic logic without GPU dependencies
- Validates objective and error registry functionality
- Tests dimension validation
- Tests all registered objectives and errors

#### test_validation.py
- Tests parameter bounds and objective/error name validation
- Validates dimension checking with correct and incorrect parameter counts
- Tests objective and error name validation

#### test_autobounds.py
- Demonstrates automatic parameter bounds setting
- Shows how bounds are automatically set based on objective function dimension

#### test_objectives.py
- Comprehensive test suite for the full implementation
- Tests registry functionality
- Tests synthetic data fitting with known parameters
- Tests all registered objective functions
- Tests all registered error metrics
- Tests error handling with invalid configurations

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

To run the tests, first activate your virtual environment:

```bash
source pso_venv/bin/activate  # On Windows: pso_venv\Scripts\activate
```

Then run all test suites at once using the test runner:

```bash
python run_tests.py
```

Or run individual test suites:

```bash
python -m tests.test_objectives
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
````

## File: utils.py
````python
def divide_intervals(param_bounds: List[Tuple[float, float]], divisions: int) -> List[List[Tuple[float, float]]]
⋮----
"""
    Divide each interval defined by (lower, upper) bounds into a specified number of subintervals.

    Args:
        param_bounds (List[Tuple[float, float]]): A list of tuples representing the lower and upper bounds of each parameter.
        divisions (int): The number of subintervals to create for each interval.

    Returns:
        List[List[Tuple[float, float]]]: A list where each element is a list of subintervals for the corresponding parameter.
    """
divided_intervals = []
⋮----
step = (upper - lower) / divisions
intervals = [(lower + i * step, lower + (i + 1) * step) for i in range(divisions)]
⋮----
def default_bounds_for(objective: str, dim: int)
⋮----
m = {
⋮----
# Schwefel: não informado no artigo; use padrão classico apenas se aceitar referência externa
````

## File: main.py
````python
# Test 1: Default Langmuir + MAE
⋮----
start_time = time.time()
result = hierarchical_pso(
⋮----
end_time = time.time()
````

## File: kernel.py
````python
"""
This module defines the CUDA kernel for the PSO algorithm and compiles it using CuPy.
Supports multiple objective functions and error metrics via switch-based selection.
"""
⋮----
# Define constants for model and error IDs
MODEL_LANGMUIR = 0
MODEL_SIPS = 1
MODEL_TOTH = 2
MODEL_BET = 3
MODEL_GAB = 4
MODEL_NEWTON = 5
MODEL_SPHERE = 100
MODEL_ROSENBROCK = 101
MODEL_QUARTIC = 102
MODEL_SCHWEFEL = 103
MODEL_RASTRIGIN = 104
MODEL_ACKLEY = 105
⋮----
ERROR_SSE = 0
ERROR_MSE = 1
ERROR_RMSE = 2
ERROR_MAE = 3
ERROR_MAPE = 4
ERROR_R2 = 5
⋮----
# Kernel code with support for multiple objective functions and error metrics
kernel_code = """
⋮----
pso_kernel = cp.RawKernel(kernel_code, 'update_velocity_position')
⋮----
# Template-based kernel generation (Strategy B)
kernel_template = """
⋮----
def get_model_function(model_name: str) -> str
⋮----
"""Get the CUDA code for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        CUDA code for the model
    """
model_functions = {
⋮----
def get_error_function(error_name: str) -> str
⋮----
"""Get the CUDA code for a specific error function.

    Args:
        error_name: Name of the error function

    Returns:
        CUDA code for the error function
    """
error_functions = {
⋮----
def get_error_finalization(error_name: str) -> str
⋮----
"""Get the CUDA code for finalizing the error calculation.

    Args:
        error_name: Name of the error function

    Returns:
        CUDA code for finalizing the error calculation
    """
error_finalizations = {
⋮----
def generate_specialized_kernel(model_name: str, error_name: str) -> cp.RawKernel
⋮----
"""Generate a specialized kernel for a specific model and error function.

    Args:
        model_name: Name of the model
        error_name: Name of the error function

    Returns:
        Compiled RawKernel
    """
# Get the model function
model_function = get_model_function(model_name)
⋮----
# Get the error accumulation code
error_accumulation = get_error_function(error_name)
⋮----
# Replace the placeholder in the model function
model_function = model_function.replace("// ERROR_ACCUMULATION_PLACEHOLDER", error_accumulation)
⋮----
# Replace the model function placeholder in the template
kernel_source = kernel_template.replace("// MODEL_FUNCTION_PLACEHOLDER", model_function)
⋮----
# Get the error finalization code
error_finalization = get_error_finalization(error_name)
⋮----
# Create the complete error function
error_function = error_accumulation + error_finalization
⋮----
# Replace the error function placeholder in the template
kernel_source = kernel_source.replace("// ERROR_FUNCTION_PLACEHOLDER", error_function)
⋮----
# Compile and return the kernel
````
