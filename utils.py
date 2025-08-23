from typing import List, Tuple

def divide_intervals(param_bounds: List[Tuple[float, float]], divisions: int) -> List[List[Tuple[float, float]]]:
    """
    Divide each interval defined by (lower, upper) bounds into a specified number of subintervals.

    Args:
        param_bounds (List[Tuple[float, float]]): A list of tuples representing the lower and upper bounds of each parameter.
        divisions (int): The number of subintervals to create for each interval.

    Returns:
        List[List[Tuple[float, float]]]: A list where each element is a list of subintervals for the corresponding parameter.
    """
    divided_intervals = []
    for lower, upper in param_bounds:
        step = (upper - lower) / divisions
        intervals = [(lower + i * step, lower + (i + 1) * step) for i in range(divisions)]
        divided_intervals.append(intervals)
    return divided_intervals

def default_bounds_for(objective: str, dim: int):
    m = {
        "sphere":     (-10E-10, 10E10),
        "rosenbrock": (-30.0, 30.0),
        "quartic":    (-1.28, 1.28),
        # Schwefel: não informado no artigo; use padrão clássico apenas se aceitar referência externa
        "schwefel":   (-500.0, 500.0),
        "rastrigin":  (-5.12, 5.12),
        "ackley":     (-32.0, 32.0),
    }
    lo, hi = m.get(objective, (-1.0, 1.0))
    return [(lo, hi) for _ in range(dim)]