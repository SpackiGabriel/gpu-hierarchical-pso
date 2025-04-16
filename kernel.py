"""
This module defines the CUDA kernel for the PSO algorithm and compiles it using CuPy.
"""

import cupy as cp

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
    double *q
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_particles * dim) {
        int particle_idx = idx / dim;
        int dim_idx = idx % dim;
        int pos = particle_idx * dim + dim_idx;
        velocity[pos] = w * velocity[pos] + c1 * r1[pos] * (personal_best[pos] - position[pos]) + c2 * r2[pos] * (global_best[dim_idx] - position[pos]);
        position[pos] += velocity[pos];
        if (dim_idx == 0) {
            double qmax = position[particle_idx * dim];
            double b = position[particle_idx * dim + 1];
            double fit = 0.0;
            for (int i = 0; i < n_points; ++i) {
                double q_calc = (qmax * b * p[i]) / (1.0 + b * p[i]);
                fit += (q[i] - q_calc) * (q[i] - q_calc);
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
