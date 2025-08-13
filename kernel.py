"""
This module defines the CUDA kernel for the PSO algorithm and compiles it using CuPy.
Supports multiple objective functions and error metrics via switch-based selection.
"""

import cupy as cp
from registries import objective_registry, error_registry

# Define constants for model and error IDs
MODEL_LANGMUIR = 0
MODEL_SIPS = 1
MODEL_TOTH = 2
MODEL_BET = 3
MODEL_GAB = 4
MODEL_NEWTON = 5

ERROR_SSE = 0
ERROR_MSE = 1
ERROR_RMSE = 2
ERROR_MAE = 3
ERROR_MAPE = 4
ERROR_R2 = 5

# Kernel code with support for multiple objective functions and error metrics
kernel_code = """
extern "C" __global__
void update_velocity_position(
    double *position, 
    double *velocity, 
    double *personal_best, 
    double *global_best, 
    double *fitness, 
    double *personal_best_fitness, 
    double *r1, 
    double *r2, 
    double w, 
    double c1, 
    double c2, 
    int n_particles, 
    int dim, 
    int n_points, 
    double *p, 
    double *q,
    int model_id,
    int error_id,
    double sst  // Total sum of squares for R2 calculation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_particles * dim) {
        int particle_idx = idx / dim;
        int dim_idx = idx % dim;
        int pos = particle_idx * dim + dim_idx;
        
        // Update velocity and position
        velocity[pos] = w * velocity[pos] + c1 * r1[pos] * (personal_best[pos] - position[pos]) + c2 * r2[pos] * (global_best[dim_idx] - position[pos]);
        position[pos] += velocity[pos];
        
        // Compute fitness only once per particle (when dim_idx == 0)
        if (dim_idx == 0) {
            double fit = 0.0;
            double q_calc;
            
            // Compute q_calc based on selected model
            switch (model_id) {
                case 0: { // Langmuir
                    if (dim < 2) {
                        fit = 1e10; // Penalty for incorrect dimension
                        break;
                    }
                    double qmax = position[particle_idx * dim];
                    double b = position[particle_idx * dim + 1];
                    
                    // Check parameter validity
                    if (qmax <= 0 || b <= 0) {
                        fit = 1e10; // Penalty for invalid parameters
                        break;
                    }
                    
                    for (int i = 0; i < n_points; ++i) {
                        double bp = b * p[i];
                        if (bp > 700) { // Prevent overflow
                            q_calc = qmax;
                        } else {
                            q_calc = (qmax * bp) / (1.0 + bp);
                        }
                        
                        // Compute error based on selected error metric
                        switch (error_id) {
                            case 0: // SSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 1: // MSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 2: // RMSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 3: // MAE
                                fit += fabs(q[i] - q_calc);
                                break;
                            case 4: // MAPE
                                if (fabs(q[i]) > 1e-10) {
                                    fit += fabs((q[i] - q_calc) / q[i]);
                                } else {
                                    fit += fabs(q[i] - q_calc) / 1e-10;
                                }
                                break;
                            case 5: // R2
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                        }
                    }
                    break;
                }
                case 1: { // Sips
                    if (dim < 3) {
                        fit = 1e10; // Penalty for incorrect dimension
                        break;
                    }
                    double qmax = position[particle_idx * dim];
                    double b = position[particle_idx * dim + 1];
                    double n = position[particle_idx * dim + 2];
                    
                    // Check parameter validity
                    if (qmax <= 0 || b <= 0 || n <= 0) {
                        fit = 1e10; // Penalty for invalid parameters
                        break;
                    }
                    
                    for (int i = 0; i < n_points; ++i) {
                        double bp = b * p[i];
                        if (bp > 700 || n > 100) { // Prevent overflow
                            q_calc = qmax;
                        } else {
                            double bp_n = pow(bp, n);
                            q_calc = (qmax * bp_n) / (1.0 + bp_n);
                        }
                        
                        // Compute error based on selected error metric
                        switch (error_id) {
                            case 0: // SSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 1: // MSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 2: // RMSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 3: // MAE
                                fit += fabs(q[i] - q_calc);
                                break;
                            case 4: // MAPE
                                if (fabs(q[i]) > 1e-10) {
                                    fit += fabs((q[i] - q_calc) / q[i]);
                                } else {
                                    fit += fabs(q[i] - q_calc) / 1e-10;
                                }
                                break;
                            case 5: // R2
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                        }
                    }
                    break;
                }
                case 2: { // Toth
                    if (dim < 3) {
                        fit = 1e10; // Penalty for incorrect dimension
                        break;
                    }
                    double qmax = position[particle_idx * dim];
                    double b = position[particle_idx * dim + 1];
                    double t = position[particle_idx * dim + 2];
                    
                    // Check parameter validity
                    if (qmax <= 0 || b <= 0 || t <= 0) {
                        fit = 1e10; // Penalty for invalid parameters
                        break;
                    }
                    
                    for (int i = 0; i < n_points; ++i) {
                        double bp = b * p[i];
                        if (bp > 700 || t > 100) { // Prevent overflow
                            q_calc = qmax;
                        } else {
                            double bp_t = pow(bp, t);
                            double denom = 1.0 + bp_t;
                            if (denom < 1e-10) denom = 1e-10; // Prevent division by zero
                            double denom_t = pow(denom, 1.0/t);
                            if (denom_t < 1e-10) denom_t = 1e-10; // Prevent division by zero
                            q_calc = (qmax * bp) / denom_t;
                        }
                        
                        // Compute error based on selected error metric
                        switch (error_id) {
                            case 0: // SSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 1: // MSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 2: // RMSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 3: // MAE
                                fit += fabs(q[i] - q_calc);
                                break;
                            case 4: // MAPE
                                if (fabs(q[i]) > 1e-10) {
                                    fit += fabs((q[i] - q_calc) / q[i]);
                                } else {
                                    fit += fabs(q[i] - q_calc) / 1e-10;
                                }
                                break;
                            case 5: // R2
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                        }
                    }
                    break;
                }
                case 3: { // BET
                    if (dim < 3) {
                        fit = 1e10; // Penalty for incorrect dimension
                        break;
                    }
                    double qm = position[particle_idx * dim];
                    double c = position[particle_idx * dim + 1];
                    double k = position[particle_idx * dim + 2];
                    
                    // Check parameter validity
                    if (qm <= 0 || c <= 0 || k <= 0) {
                        fit = 1e10; // Penalty for invalid parameters
                        break;
                    }
                    
                    for (int i = 0; i < n_points; ++i) {
                        double p_val = p[i];
                        if (p_val >= 1.0/k) { // Prevent division by zero or negative values
                            q_calc = 0;
                        } else {
                            double denom1 = 1.0 - k * p_val;
                            if (fabs(denom1) < 1e-10) denom1 = 1e-10; // Prevent division by zero
                            double denom2 = 1.0 + (c - 1.0) * k * p_val;
                            if (fabs(denom2) < 1e-10) denom2 = 1e-10; // Prevent division by zero
                            q_calc = (qm * c * k * p_val) / (denom1 * denom2);
                        }
                        
                        // Compute error based on selected error metric
                        switch (error_id) {
                            case 0: // SSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 1: // MSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 2: // RMSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 3: // MAE
                                fit += fabs(q[i] - q_calc);
                                break;
                            case 4: // MAPE
                                if (fabs(q[i]) > 1e-10) {
                                    fit += fabs((q[i] - q_calc) / q[i]);
                                } else {
                                    fit += fabs(q[i] - q_calc) / 1e-10;
                                }
                                break;
                            case 5: // R2
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                        }
                    }
                    break;
                }
                case 4: { // GAB
                    if (dim < 3) {
                        fit = 1e10; // Penalty for incorrect dimension
                        break;
                    }
                    double qm = position[particle_idx * dim];
                    double c = position[particle_idx * dim + 1];
                    double k = position[particle_idx * dim + 2];
                    
                    // Check parameter validity
                    if (qm <= 0 || c <= 0 || k <= 0) {
                        fit = 1e10; // Penalty for invalid parameters
                        break;
                    }
                    
                    for (int i = 0; i < n_points; ++i) {
                        double p_val = p[i];
                        if (p_val >= 1.0/k) { // Prevent division by zero or negative values
                            q_calc = 0;
                        } else {
                            double denom1 = 1.0 - k * p_val;
                            if (fabs(denom1) < 1e-10) denom1 = 1e-10; // Prevent division by zero
                            double denom2 = 1.0 + (c - 1.0) * k * p_val;
                            if (fabs(denom2) < 1e-10) denom2 = 1e-10; // Prevent division by zero
                            q_calc = (qm * c * k * p_val) / (denom1 * denom2);
                        }
                        
                        // Compute error based on selected error metric
                        switch (error_id) {
                            case 0: // SSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 1: // MSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 2: // RMSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 3: // MAE
                                fit += fabs(q[i] - q_calc);
                                break;
                            case 4: // MAPE
                                if (fabs(q[i]) > 1e-10) {
                                    fit += fabs((q[i] - q_calc) / q[i]);
                                } else {
                                    fit += fabs(q[i] - q_calc) / 1e-10;
                                }
                                break;
                            case 5: // R2
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                        }
                    }
                    break;
                }
                case 5: { // Newton (linear)
                    if (dim < 2) {
                        fit = 1e10; // Penalty for incorrect dimension
                        break;
                    }
                    double a = position[particle_idx * dim];
                    double b = position[particle_idx * dim + 1];
                    
                    for (int i = 0; i < n_points; ++i) {
                        q_calc = a + b * p[i];
                        
                        // Compute error based on selected error metric
                        switch (error_id) {
                            case 0: // SSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 1: // MSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 2: // RMSE
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 3: // MAE
                                fit += fabs(q[i] - q_calc);
                                break;
                            case 4: // MAPE
                                if (fabs(q[i]) > 1e-10) {
                                    fit += fabs((q[i] - q_calc) / q[i]);
                                } else {
                                    fit += fabs(q[i] - q_calc) / 1e-10;
                                }
                                break;
                            case 5: // R2
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                        }
                    }
                    break;
                }
                default:
                    fit = 1e10; // Unknown model
                    break;
            }
            
            // Apply final transformation based on error metric
            switch (error_id) {
                case 1: // MSE
                    fit = fit / n_points;
                    break;
                case 2: // RMSE
                    fit = fit / n_points;
                    fit = sqrt(fit);
                    break;
                case 5: // R2
                    if (sst > 1e-10) {
                        fit = 1.0 - (fit / sst);
                    } else {
                        fit = -1e10; // Invalid R2
                    }
                    break;
            }
            
            fitness[particle_idx] = fit;
            if (fit < personal_best_fitness[particle_idx]) {
                personal_best_fitness[particle_idx] = fit;
                for (int d = 0; d < dim; ++d) {
                    personal_best[particle_idx * dim + d] = position[particle_idx * dim + d];
                }
            }
        }
    }
}
"""

pso_kernel = cp.RawKernel(kernel_code, 'update_velocity_position')

# Template-based kernel generation (Strategy B)
kernel_template = """
extern "C" __global__
void update_velocity_position(
    double *position, 
    double *velocity, 
    double *personal_best, 
    double *global_best, 
    double *fitness, 
    double *personal_best_fitness, 
    double *r1, 
    double *r2, 
    double w, 
    double c1, 
    double c2, 
    int n_particles, 
    int dim, 
    int n_points, 
    double *p, 
    double *q,
    double sst
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_particles * dim) {
        int particle_idx = idx / dim;
        int dim_idx = idx % dim;
        int pos = particle_idx * dim + dim_idx;
        
        // Update velocity and position
        velocity[pos] = w * velocity[pos] + c1 * r1[pos] * (personal_best[pos] - position[pos]) + c2 * r2[pos] * (global_best[dim_idx] - position[pos]);
        position[pos] += velocity[pos];
        
        // Compute fitness only once per particle (when dim_idx == 0)
        if (dim_idx == 0) {
            double fit = 0.0;
            double q_calc;
            
            // MODEL_FUNCTION_PLACEHOLDER
            
            // ERROR_FUNCTION_PLACEHOLDER
            
            fitness[particle_idx] = fit;
            if (fit < personal_best_fitness[particle_idx]) {
                personal_best_fitness[particle_idx] = fit;
                for (int d = 0; d < dim; ++d) {
                    personal_best[particle_idx * dim + d] = position[particle_idx * dim + d];
                }
            }
        }
    }
}
"""

def get_model_function(model_name: str) -> str:
    """Get the CUDA code for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        CUDA code for the model
    """
    model_functions = {
        "langmuir": """
            if (dim < 2) {
                fit = 1e10; // Penalty for incorrect dimension
            } else {
                double qmax = position[particle_idx * dim];
                double b = position[particle_idx * dim + 1];
                
                // Check parameter validity
                if (qmax <= 0 || b <= 0) {
                    fit = 1e10; // Penalty for invalid parameters
                } else {
                    for (int i = 0; i < n_points; ++i) {
                        double bp = b * p[i];
                        if (bp > 700) { // Prevent overflow
                            q_calc = qmax;
                        } else {
                            q_calc = (qmax * bp) / (1.0 + bp);
                        }
                        // ERROR_ACCUMULATION_PLACEHOLDER
                    }
                }
            }
        """,
        "sips": """
            if (dim < 3) {
                fit = 1e10; // Penalty for incorrect dimension
            } else {
                double qmax = position[particle_idx * dim];
                double b = position[particle_idx * dim + 1];
                double n = position[particle_idx * dim + 2];
                
                // Check parameter validity
                if (qmax <= 0 || b <= 0 || n <= 0) {
                    fit = 1e10; // Penalty for invalid parameters
                } else {
                    for (int i = 0; i < n_points; ++i) {
                        double bp = b * p[i];
                        if (bp > 700 || n > 100) { // Prevent overflow
                            q_calc = qmax;
                        } else {
                            double bp_n = pow(bp, n);
                            q_calc = (qmax * bp_n) / (1.0 + bp_n);
                        }
                        // ERROR_ACCUMULATION_PLACEHOLDER
                    }
                }
            }
        """,
        "toth": """
            if (dim < 3) {
                fit = 1e10; // Penalty for incorrect dimension
            } else {
                double qmax = position[particle_idx * dim];
                double b = position[particle_idx * dim + 1];
                double t = position[particle_idx * dim + 2];
                
                // Check parameter validity
                if (qmax <= 0 || b <= 0 || t <= 0) {
                    fit = 1e10; // Penalty for invalid parameters
                } else {
                    for (int i = 0; i < n_points; ++i) {
                        double bp = b * p[i];
                        if (bp > 700 || t > 100) { // Prevent overflow
                            q_calc = qmax;
                        } else {
                            double bp_t = pow(bp, t);
                            double denom = 1.0 + bp_t;
                            if (denom < 1e-10) denom = 1e-10; // Prevent division by zero
                            double denom_t = pow(denom, 1.0/t);
                            if (denom_t < 1e-10) denom_t = 1e-10; // Prevent division by zero
                            q_calc = (qmax * bp) / denom_t;
                        }
                        // ERROR_ACCUMULATION_PLACEHOLDER
                    }
                }
            }
        """,
        "bet": """
            if (dim < 3) {
                fit = 1e10; // Penalty for incorrect dimension
            } else {
                double qm = position[particle_idx * dim];
                double c = position[particle_idx * dim + 1];
                double k = position[particle_idx * dim + 2];
                
                // Check parameter validity
                if (qm <= 0 || c <= 0 || k <= 0) {
                    fit = 1e10; // Penalty for invalid parameters
                } else {
                    for (int i = 0; i < n_points; ++i) {
                        double p_val = p[i];
                        if (p_val >= 1.0/k) { // Prevent division by zero or negative values
                            q_calc = 0;
                        } else {
                            double denom1 = 1.0 - k * p_val;
                            if (fabs(denom1) < 1e-10) denom1 = 1e-10; // Prevent division by zero
                            double denom2 = 1.0 + (c - 1.0) * k * p_val;
                            if (fabs(denom2) < 1e-10) denom2 = 1e-10; // Prevent division by zero
                            q_calc = (qm * c * k * p_val) / (denom1 * denom2);
                        }
                        // ERROR_ACCUMULATION_PLACEHOLDER
                    }
                }
            }
        """,
        "gab": """
            if (dim < 3) {
                fit = 1e10; // Penalty for incorrect dimension
            } else {
                double qm = position[particle_idx * dim];
                double c = position[particle_idx * dim + 1];
                double k = position[particle_idx * dim + 2];
                
                // Check parameter validity
                if (qm <= 0 || c <= 0 || k <= 0) {
                    fit = 1e10; // Penalty for invalid parameters
                } else {
                    for (int i = 0; i < n_points; ++i) {
                        double p_val = p[i];
                        if (p_val >= 1.0/k) { // Prevent division by zero or negative values
                            q_calc = 0;
                        } else {
                            double denom1 = 1.0 - k * p_val;
                            if (fabs(denom1) < 1e-10) denom1 = 1e-10; // Prevent division by zero
                            double denom2 = 1.0 + (c - 1.0) * k * p_val;
                            if (fabs(denom2) < 1e-10) denom2 = 1e-10; // Prevent division by zero
                            q_calc = (qm * c * k * p_val) / (denom1 * denom2);
                        }
                        // ERROR_ACCUMULATION_PLACEHOLDER
                    }
                }
            }
        """,
        "newton": """
            if (dim < 2) {
                fit = 1e10; // Penalty for incorrect dimension
            } else {
                double a = position[particle_idx * dim];
                double b = position[particle_idx * dim + 1];
                
                for (int i = 0; i < n_points; ++i) {
                    q_calc = a + b * p[i];
                    // ERROR_ACCUMULATION_PLACEHOLDER
                }
            }
        """
    }
    
    return model_functions.get(model_name, """
        fit = 1e10; // Unknown model
    """)

def get_error_function(error_name: str) -> str:
    """Get the CUDA code for a specific error function.
    
    Args:
        error_name: Name of the error function
        
    Returns:
        CUDA code for the error function
    """
    error_functions = {
        "sse": """
            fit += (q[i] - q_calc) * (q[i] - q_calc);
        """,
        "mse": """
            fit += (q[i] - q_calc) * (q[i] - q_calc);
        """,
        "rmse": """
            fit += (q[i] - q_calc) * (q[i] - q_calc);
        """,
        "mae": """
            fit += fabs(q[i] - q_calc);
        """,
        "mape": """
            if (fabs(q[i]) > 1e-10) {
                fit += fabs((q[i] - q_calc) / q[i]);
            } else {
                fit += fabs(q[i] - q_calc) / 1e-10;
            }
        """,
        "r2": """
            fit += (q[i] - q_calc) * (q[i] - q_calc);
        """
    }
    
    return error_functions.get(error_name, """
        fit += (q[i] - q_calc) * (q[i] - q_calc);
    """)

def get_error_finalization(error_name: str) -> str:
    """Get the CUDA code for finalizing the error calculation.
    
    Args:
        error_name: Name of the error function
        
    Returns:
        CUDA code for finalizing the error calculation
    """
    error_finalizations = {
        "mse": """
            fit = fit / n_points;
        """,
        "rmse": """
            fit = fit / n_points;
            fit = sqrt(fit);
        """,
        "r2": """
            if (sst > 1e-10) {
                fit = 1.0 - (fit / sst);
            } else {
                fit = -1e10; // Invalid R2
            }
        """
    }
    
    return error_finalizations.get(error_name, "")

def generate_specialized_kernel(model_name: str, error_name: str) -> cp.RawKernel:
    """Generate a specialized kernel for a specific model and error function.
    
    Args:
        model_name: Name of the model
        error_name: Name of the error function
        
    Returns:
        Compiled RawKernel
    """
    # Get the model function
    model_function = get_model_function(model_name)
    
    # Get the error accumulation code
    error_accumulation = get_error_function(error_name)
    
    # Replace the placeholder in the model function
    model_function = model_function.replace("// ERROR_ACCUMULATION_PLACEHOLDER", error_accumulation)
    
    # Replace the model function placeholder in the template
    kernel_source = kernel_template.replace("// MODEL_FUNCTION_PLACEHOLDER", model_function)
    
    # Get the error finalization code
    error_finalization = get_error_finalization(error_name)
    
    # Create the complete error function
    error_function = error_accumulation + error_finalization
    
    # Replace the error function placeholder in the template
    kernel_source = kernel_source.replace("// ERROR_FUNCTION_PLACEHOLDER", error_function)
    
    # Compile and return the kernel
    return cp.RawKernel(kernel_source, 'update_velocity_position')

