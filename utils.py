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
