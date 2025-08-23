You are a senior CUDA/CuPy engineer working inside this repository. Implement **early stopping in the CUDA kernel path**, then orchestrate a **cross-validation grid** in `main.py`. Keep existing behavior intact except for the addition of early-stop signaling and result logging.

## Objectives
1) **Kernel-level early stop**: add device-side detection of convergence each iteration and signal back to host.
2) **Orchestration in `main.py`**: run all combinations of:
   - `divisions = [1, 2, 4, 8]`
   - `particles = [10, 100, 1000]`
3) **Fixed max iterations**: `iter_n = 10000` for every run.
4) **Stop criteria (run terminates when either is met)**:
   - Fitness threshold reached, or
   - Max iterations reached.
5) **CSV export**: one consolidated `cv_results.csv` with all runs and metadata.

## Kernel / GPU path changes
Work in `kernel.py` and the GPU loop in `gpu_pso.py`.

### A) Add early-stop signaling to the kernel
- Accept new kernel parameters:
  - `double threshold_fitness`
  - `int is_maximize` (0 for metrics minimized, 1 for metrics maximized like R2)
  - `int* stop_flag` (device scalar; 0 = keep going, 1 = stop)
  - `double* gbest_fitness` (device scalar storing the best fitness of the current iteration or running best; pick the simpler route for reliable signaling)
- Compute each particle’s fitness as currently done and update a **global best** value inside the kernel using an atomic operation on `gbest_fitness`.
  - Implement `atomicMinDouble` / `atomicMaxDouble` via `atomicCAS` on `unsigned long long` to support doubles.
  - For `is_maximize==1`, compare with `>=` and use atomic **max**; else use atomic **min**.
- After all threads update fitness, have a grid-wide decision (coarse approach):
  - Let **one designated thread** (e.g., `if (blockIdx.x==0 && threadIdx.x==0)`) read `*gbest_fitness` and set `*stop_flag = 1` if:
    - `is_maximize==0 && (*gbest_fitness <= threshold_fitness)`, or
    - `is_maximize==1 && (*gbest_fitness >= threshold_fitness)`.
- Keep the kernel **single-iteration**. The host loop still launches one kernel per iteration, but the **decision to stop is made in-kernel** and propagated via `stop_flag`.

### B) Host loop integration (CuPy)
- In `GPU_PSO.optimize`:
  - Allocate device scalars once per run:
    - `stop_flag_dev = cp.zeros((), dtype=cp.int32)`
    - `gbest_fit_dev = cp.full((), init_val, dtype=cp.float64)` where `init_val` is `+inf` for minimizing or `-inf` for maximizing.
  - Before each kernel launch, pass pointers to `stop_flag_dev` and `gbest_fit_dev`, plus `threshold_fitness` and `is_maximize`.
  - After each launch, read only the tiny `stop_flag_dev` scalar on host (`flag = int(stop_flag_dev.get())`).
    - If `flag == 1`, set `early_stop_reason = "fitness"` and break.
  - Count `iterations_executed`. If loop finishes because `iterations_executed == iter_n`, set `early_stop_reason = "max_iter"`.
- Preserve existing best-position/fitness bookkeeping. For R², remember it is maximized; for others, minimized.

### C) Return API extension
- Extend the return of `optimize` (and thread it back through `hierarchical_pso`) to include:
  - `iterations_executed: int`
  - `early_stop_reason: "fitness" | "max_iter"`
- If altering function signatures is too invasive, return a small dataclass-like tuple or a dict in addition to the current tuple, and update the callers in `hierarchical_pso`.

## Orchestration in `main.py`
- Keep existing demo runs intact.
- After demos, add a guarded block (e.g., `if os.getenv("RUN_CV", "1") == "1":`) to run the grid.
- Constants:
  - `DIVISIONS = [1, 2, 4, 8]`
  - `PARTICLES = [10, 100, 1000]`
  - `ITER_N = 10000`
  - `THRESHOLD = 1e-5`
  - `W = 0.8`, `C1 = 1.8`, `C2 = 2.3`
  - Choose a single objective+error for the grid (e.g., `objective="langmuir"`, `error="mae"`). Note: R² is maximized; most others minimize.
- Prepare `p` and `q` once and reuse for all runs.
- For each `(d, n)`:
  - Start timer.
  - Call `hierarchical_pso` with `part_n=n`, `divisions=d`, `iter_n=ITER_N`, `w=W`, `c1=C1`, `c2=C2`, and pass `threshold_fitness=THRESHOLD` through to the GPU path.
  - Collect:
    - `divisions`, `particles`
    - `iterations_executed`
    - `early_stop_reason`
    - `best_fitness`
    - `best_position` (JSON string)
    - `elapsed_seconds`
    - `objective`, `error`
    - `bounds_strategy="auto_by_objective_dim"`
    - `timestamp_iso`
- Append rows to `cv_results.csv` with header:

```csv
divisions,particles,iterations_executed,early_stop_reason,best_fitness,best_position,elapsed_seconds,objective,error,bounds_strategy,timestamp_iso
```

## Determinism and performance
- Seed `numpy` and `cupy` before each run. Log that GPU nondeterminism may remain due to parallel reductions.
- Reuse preallocated device buffers when possible.

## Tests and validation
- Add a small smoke test that sets a **looser** threshold (e.g., `1e-1`) to ensure the early stop path triggers quickly and `early_stop_reason == "fitness"`.
- Add a path where `threshold` is **very strict** (e.g., `1e-12`) so early stop falls back to `max_iter`.

## Acceptance criteria
- Running `python main.py` produces or updates `cv_results.csv` with 12 rows for the 4×3 grid.
- For each row:
- `iterations_executed` ≤ `ITER_N` and `early_stop_reason` is set.
- `elapsed_seconds` > 0 and `best_fitness` is numeric.
- The kernel receives and sets `stop_flag` correctly. Host loop terminates early on fitness, otherwise on `ITER_N`.

## Implementation hints
- Prefer explicit `cp.cuda.runtime.deviceSynchronize()` only when necessary; keep the host-device traffic to the scalar `stop_flag`.
- Implement `atomicCAS`-based double min/max helpers in the kernel to avoid race conditions on `gbest_fitness`.
- Keep R² handling consistent with existing maximize logic; for minimizing metrics, use standard ≤ threshold check.
