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
