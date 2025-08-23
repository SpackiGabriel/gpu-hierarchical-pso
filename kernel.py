"""
This module defines the CUDA kernel for the PSO algorithm and compiles it using CuPy.
Supports multiple objective functions and error metrics via switch-based selection.
"""

import cupy as cp

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


ERROR_SSE = 0
ERROR_MSE = 1
ERROR_RMSE = 2
ERROR_MAE = 3
ERROR_MAPE = 4
ERROR_R2 = 5

# Kernel code with support for multiple objective functions and error metrics
kernel_code = """
// Atomic operations for double precision
__device__ double atomicMinDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

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
    double sst,
    double threshold_fitness,
    int is_maximize,
    int* stop_flag,
    double* gbest_fitness
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_particles * dim) {
        int particle_idx = idx / dim;
        int dim_idx = idx % dim;
        int pos = particle_idx * dim + dim_idx;


        velocity[pos] = w * velocity[pos] + c1 * r1[pos] * (personal_best[pos] - position[pos]) + c2 * r2[pos] * (global_best[dim_idx] - position[pos]);
        position[pos] += velocity[pos];


        if (dim_idx == 0) {
            double fit = 0.0;
            double q_calc;


            switch (model_id) {
              case 100: {
    // f(x) = sum(x_j^2)
    fit = 0.0;
    for (int j=0; j<dim; ++j) {
        double xj = position[particle_idx * dim + j];
        fit += xj * xj;
    }
    break;
}
                case 101: {
                    fit = 0.0;
                    for (int j=0; j<dim-1; ++j) {
                        double xj = position[particle_idx * dim + j];
                        double xk = position[particle_idx * dim + j + 1];
                        double t1 = (1.0 - xj);
                        double t2 = (xk - xj*xj);
                        fit += 100.0 * t2*t2 + t1*t1;
                    }
                    break;
                }
                case 102: {
                    // f(x) = sum( j * x_j^4 ) + noise(0,1)
                    fit = 0.0;
                    for (int j=0; j<dim; ++j) {
                        double xj = position[particle_idx * dim + j];
                        fit += (j+1) * xj*xj*xj*xj;
                    }
                    // opcional: ruido pequeno para emular artigo (se necessario)
                    break;
                }
                case 103: {
                    // f(x) = 418.9829*dim - sum( x_j * sin(sqrt(|x_j|)) )
                    fit = 418.9829 * dim;
                    for (int j=0; j<dim; ++j) {
                        double xj = position[particle_idx * dim + j];
                        fit -= xj * sin(sqrt(fabs(xj)));
                    }
                    break;
                }
                case 104: {
                    // f(x) = 10*dim + sum( x_j^2 - 10*cos(2*pi*x_j) )
                    fit = 10.0 * dim;
                    for (int j=0; j<dim; ++j) {
                        double xj = position[particle_idx * dim + j];
                        fit += xj*xj - 10.0 * cos(6.2831853071795864769 * xj);
                    }
                    break;
                }
                case 105: {
                    // f(x) = -20 exp(-0.2 sqrt( (1/dim) sum x_j^2 ))
                    //        - exp( (1/dim) sum cos(2*pi*x_j) ) + 20 + e
                    double s1 = 0.0, s2 = 0.0;
                    for (int j=0; j<dim; ++j) {
                        double xj = position[particle_idx * dim + j];
                        s1 += xj*xj;
                        s2 += cos(6.2831853071795864769 * xj);  // 2 * pi
                    }
                    s1 /= (double)dim;
                    s2 /= (double)dim;
                    fit = fit = -20.0 * exp(-0.2 * sqrt(s1)) - exp(s2) + 20.0 + 2.71828182845904523536;
                    break;
                }

                case 0: {
                    if (dim < 2) {
                        fit = 1e10;
                        break;
                    }
                    double qmax = position[particle_idx * dim];
                    double b = position[particle_idx * dim + 1];


                    if (qmax <= 0 || b <= 0) {
                        fit = 1e10;
                        break;
                    }

                    for (int i = 0; i < n_points; ++i) {
                        double bp = b * p[i];
                        if (bp > 700) {
                            q_calc = qmax;
                        } else {
                            q_calc = (qmax * bp) / (1.0 + bp);
                        }


                        switch (error_id) {
                            case 0:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 1:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 2:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 3:
                                fit += fabs(q[i] - q_calc);
                                break;
                            case 4:
                                if (fabs(q[i]) > 1e-10) {
                                    fit += fabs((q[i] - q_calc) / q[i]);
                                } else {
                                    fit += fabs(q[i] - q_calc) / 1e-10;
                                }
                                break;
                            case 5:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                        }
                    }
                    break;
                }
                case 1: {
                    if (dim < 3) {
                        fit = 1e10;
                        break;
                    }
                    double qmax = position[particle_idx * dim];
                    double b = position[particle_idx * dim + 1];
                    double n = position[particle_idx * dim + 2];


                    if (qmax <= 0 || b <= 0 || n <= 0) {
                        fit = 1e10;
                        break;
                    }

                    for (int i = 0; i < n_points; ++i) {
                        double bp = b * p[i];
                        if (bp > 700 || n > 100) {
                            q_calc = qmax;
                        } else {
                            double bp_n = pow(bp, n);
                            q_calc = (qmax * bp_n) / (1.0 + bp_n);
                        }


                        switch (error_id) {
                            case 0:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 1:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 2:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 3:
                                fit += fabs(q[i] - q_calc);
                                break;
                            case 4:
                                if (fabs(q[i]) > 1e-10) {
                                    fit += fabs((q[i] - q_calc) / q[i]);
                                } else {
                                    fit += fabs(q[i] - q_calc) / 1e-10;
                                }
                                break;
                            case 5:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                        }
                    }
                    break;
                }
                case 2: {
                    if (dim < 3) {
                        fit = 1e10;
                        break;
                    }
                    double qmax = position[particle_idx * dim];
                    double b = position[particle_idx * dim + 1];
                    double t = position[particle_idx * dim + 2];


                    if (qmax <= 0 || b <= 0 || t <= 0) {
                        fit = 1e10;
                        break;
                    }

                    for (int i = 0; i < n_points; ++i) {
                        double bp = b * p[i];
                        if (bp > 700 || t > 100) {
                            q_calc = qmax;
                        } else {
                            double bp_t = pow(bp, t);
                            double denom = 1.0 + bp_t;
                            if (denom < 1e-10) denom = 1e-10;
                            double denom_t = pow(denom, 1.0/t);
                            if (denom_t < 1e-10) denom_t = 1e-10;
                            q_calc = (qmax * bp) / denom_t;
                        }


                        switch (error_id) {
                            case 0:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 1:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 2:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 3:
                                fit += fabs(q[i] - q_calc);
                                break;
                            case 4:
                                if (fabs(q[i]) > 1e-10) {
                                    fit += fabs((q[i] - q_calc) / q[i]);
                                } else {
                                    fit += fabs(q[i] - q_calc) / 1e-10;
                                }
                                break;
                            case 5:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                        }
                    }
                    break;
                }
                case 3: {
                    if (dim < 3) {
                        fit = 1e10;
                        break;
                    }
                    double qm = position[particle_idx * dim];
                    double c = position[particle_idx * dim + 1];
                    double k = position[particle_idx * dim + 2];


                    if (qm <= 0 || c <= 0 || k <= 0) {
                        fit = 1e10;
                        break;
                    }

                    for (int i = 0; i < n_points; ++i) {
                        double p_val = p[i];
                        if (p_val >= 1.0/k) {
                            q_calc = 0;
                        } else {
                            double denom1 = 1.0 - k * p_val;
                            if (fabs(denom1) < 1e-10) denom1 = 1e-10;
                            double denom2 = 1.0 + (c - 1.0) * k * p_val;
                            if (fabs(denom2) < 1e-10) denom2 = 1e-10;
                            q_calc = (qm * c * k * p_val) / (denom1 * denom2);
                        }


                        switch (error_id) {
                            case 0:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 1:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 2:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 3:
                                fit += fabs(q[i] - q_calc);
                                break;
                            case 4:
                                if (fabs(q[i]) > 1e-10) {
                                    fit += fabs((q[i] - q_calc) / q[i]);
                                } else {
                                    fit += fabs(q[i] - q_calc) / 1e-10;
                                }
                                break;
                            case 5:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                        }
                    }
                    break;
                }
                case 4: {
                    if (dim < 3) {
                        fit = 1e10;
                        break;
                    }
                    double qm = position[particle_idx * dim];
                    double c = position[particle_idx * dim + 1];
                    double k = position[particle_idx * dim + 2];


                    if (qm <= 0 || c <= 0 || k <= 0) {
                        fit = 1e10;
                        break;
                    }

                    for (int i = 0; i < n_points; ++i) {
                        double p_val = p[i];
                        if (p_val >= 1.0/k) {
                            q_calc = 0;
                        } else {
                            double denom1 = 1.0 - k * p_val;
                            if (fabs(denom1) < 1e-10) denom1 = 1e-10;
                            double denom2 = 1.0 + (c - 1.0) * k * p_val;
                            if (fabs(denom2) < 1e-10) denom2 = 1e-10;
                            q_calc = (qm * c * k * p_val) / (denom1 * denom2);
                        }


                        switch (error_id) {
                            case 0:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 1:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 2:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 3:
                                fit += fabs(q[i] - q_calc);
                                break;
                            case 4:
                                if (fabs(q[i]) > 1e-10) {
                                    fit += fabs((q[i] - q_calc) / q[i]);
                                } else {
                                    fit += fabs(q[i] - q_calc) / 1e-10;
                                }
                                break;
                            case 5:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                        }
                    }
                    break;
                }
                case 5: {
                    if (dim < 2) {
                        fit = 1e10;
                        break;
                    }
                    double a = position[particle_idx * dim];
                    double b = position[particle_idx * dim + 1];

                    for (int i = 0; i < n_points; ++i) {
                        q_calc = a + b * p[i];


                        switch (error_id) {
                            case 0:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 1:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 2:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                            case 3:
                                fit += fabs(q[i] - q_calc);
                                break;
                            case 4:
                                if (fabs(q[i]) > 1e-10) {
                                    fit += fabs((q[i] - q_calc) / q[i]);
                                } else {
                                    fit += fabs(q[i] - q_calc) / 1e-10;
                                }
                                break;
                            case 5:
                                fit += (q[i] - q_calc) * (q[i] - q_calc);
                                break;
                        }
                    }
                    break;
                }
                default:
                    fit = 1e10;
                    break;
            }


            switch (error_id) {
                case 1:
                    fit = fit / n_points;
                    break;
                case 2:
                    fit = fit / n_points;
                    fit = sqrt(fit);
                    break;
                case 5:
                    if (sst > 1e-10) {
                        fit = 1.0 - (fit / sst);
                    } else {
                        fit = -1e10;
                    }
                    break;
            }

            fitness[particle_idx] = fit;
            
            // Update personal best based on error metric
            bool is_better = false;
            if (error_id == 5) { // R2 - higher is better
                if (fit > personal_best_fitness[particle_idx]) {
                    is_better = true;
                }
            } else { // Other metrics - lower is better
                if (fit < personal_best_fitness[particle_idx]) {
                    is_better = true;
                }
            }
            
            if (is_better) {
                personal_best_fitness[particle_idx] = fit;
                for (int d = 0; d < dim; ++d) {
                    personal_best[particle_idx * dim + d] = position[particle_idx * dim + d];
                }
                
                // Update global best using atomic operations
                if (error_id == 5) { // R2 - maximize
                    atomicMaxDouble(gbest_fitness, fit);
                } else { // Other metrics - minimize
                    atomicMinDouble(gbest_fitness, fit);
                }
            }
            
            // Check for early stopping (only one thread per grid)
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                double current_best = *gbest_fitness;
                if (is_maximize == 1) {
                    // For maximizing metrics (R2)
                    if (current_best >= threshold_fitness) {
                        *stop_flag = 1;
                    }
                } else {
                    // For minimizing metrics (SSE, MSE, etc.)
                    if (current_best <= threshold_fitness) {
                        *stop_flag = 1;
                    }
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


        velocity[pos] = w * velocity[pos] + c1 * r1[pos] * (personal_best[pos] - position[pos]) + c2 * r2[pos] * (global_best[dim_idx] - position[pos]);
        position[pos] += velocity[pos];


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
                fit = 1e10;
            } else {
                double qmax = position[particle_idx * dim];
                double b = position[particle_idx * dim + 1];


                if (qmax <= 0 || b <= 0) {
                    fit = 1e10;
                } else {
                    for (int i = 0; i < n_points; ++i) {
                        double bp = b * p[i];
                        if (bp > 700) {
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
                fit = 1e10;
            } else {
                double qmax = position[particle_idx * dim];
                double b = position[particle_idx * dim + 1];
                double n = position[particle_idx * dim + 2];


                if (qmax <= 0 || b <= 0 || n <= 0) {
                    fit = 1e10;
                } else {
                    for (int i = 0; i < n_points; ++i) {
                        double bp = b * p[i];
                        if (bp > 700 || n > 100) {
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
                fit = 1e10;
            } else {
                double qmax = position[particle_idx * dim];
                double b = position[particle_idx * dim + 1];
                double t = position[particle_idx * dim + 2];


                if (qmax <= 0 || b <= 0 || t <= 0) {
                    fit = 1e10;
                } else {
                    for (int i = 0; i < n_points; ++i) {
                        double bp = b * p[i];
                        if (bp > 700 || t > 100) {
                            q_calc = qmax;
                        } else {
                            double bp_t = pow(bp, t);
                            double denom = 1.0 + bp_t;
                            if (denom < 1e-10) denom = 1e-10;
                            double denom_t = pow(denom, 1.0/t);
                            if (denom_t < 1e-10) denom_t = 1e-10;
                            q_calc = (qmax * bp) / denom_t;
                        }
                        // ERROR_ACCUMULATION_PLACEHOLDER
                    }
                }
            }
        """,
        "bet": """
            if (dim < 3) {
                fit = 1e10;
            } else {
                double qm = position[particle_idx * dim];
                double c = position[particle_idx * dim + 1];
                double k = position[particle_idx * dim + 2];


                if (qm <= 0 || c <= 0 || k <= 0) {
                    fit = 1e10;
                } else {
                    for (int i = 0; i < n_points; ++i) {
                        double p_val = p[i];
                        if (p_val >= 1.0/k) {
                            q_calc = 0;
                        } else {
                            double denom1 = 1.0 - k * p_val;
                            if (fabs(denom1) < 1e-10) denom1 = 1e-10;
                            double denom2 = 1.0 + (c - 1.0) * k * p_val;
                            if (fabs(denom2) < 1e-10) denom2 = 1e-10;
                            q_calc = (qm * c * k * p_val) / (denom1 * denom2);
                        }
                        // ERROR_ACCUMULATION_PLACEHOLDER
                    }
                }
            }
        """,
        "gab": """
            if (dim < 3) {
                fit = 1e10;
            } else {
                double qm = position[particle_idx * dim];
                double c = position[particle_idx * dim + 1];
                double k = position[particle_idx * dim + 2];


                if (qm <= 0 || c <= 0 || k <= 0) {
                    fit = 1e10;
                } else {
                    for (int i = 0; i < n_points; ++i) {
                        double p_val = p[i];
                        if (p_val >= 1.0/k) {
                            q_calc = 0;
                        } else {
                            double denom1 = 1.0 - k * p_val;
                            if (fabs(denom1) < 1e-10) denom1 = 1e-10;
                            double denom2 = 1.0 + (c - 1.0) * k * p_val;
                            if (fabs(denom2) < 1e-10) denom2 = 1e-10;
                            q_calc = (qm * c * k * p_val) / (denom1 * denom2);
                        }
                        // ERROR_ACCUMULATION_PLACEHOLDER
                    }
                }
            }
        """,
        "newton": """
            if (dim < 2) {
                fit = 1e10;
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
        fit = 1e10;
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
            fit += fabs(q[i] - q_calc) * sizeof(q[i]) / sizeof(q);
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
                fit = -1e10;
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
