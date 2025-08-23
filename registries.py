"""
Registries for objective models and error metrics in the PSO implementation.
"""

from typing import Dict


class ObjectiveRegistry:
    """Registry for objective models."""

    def __init__(self):
        self._objectives: Dict[str, Dict] = {}

    def register(self, name: str, dim: int, model_id: int, description: str = ""):
        """Register a new objective function.

        Args:
            name: Name of the objective function
            dim: Required parameter dimension
            model_id: Device-side identifier
            description: Optional description
        """
        self._objectives[name] = {
            'dim': dim,
            'model_id': model_id,
            'description': description
        }

    def get(self, name: str) -> Dict:
        """Get objective function info by name.

        Args:
            name: Name of the objective function

        Returns:
            Dictionary with objective function info

        Raises:
            ValueError: If objective function is not registered
        """
        if name not in self._objectives:
            raise ValueError(f"Objective function '{name}' not registered")
        return self._objectives[name]

    def get_dimension(self, name: str) -> int:
        """Get required dimension for an objective function.

        Args:
            name: Name of the objective function

        Returns:
            Required parameter dimension
        """
        return self.get(name)['dim']

    def get_model_id(self, name: str) -> int:
        """Get device-side identifier for an objective function.

        Args:
            name: Name of the objective function

        Returns:
            Device-side identifier
        """
        return self.get(name)['model_id']

    def list_objectives(self) -> Dict[str, Dict]:
        """List all registered objective functions.

        Returns:
            Dictionary of registered objective functions
        """
        return self._objectives.copy()


class ErrorRegistry:
    """Registry for error metrics."""

    def __init__(self):
        self._errors: Dict[str, Dict] = {}

    def register(self, name: str, error_id: int, description: str = ""):
        """Register a new error metric.

        Args:
            name: Name of the error metric
            error_id: Device-side identifier
            description: Optional description
        """
        self._errors[name] = {
            'error_id': error_id,
            'description': description
        }

    def get(self, name: str) -> Dict:
        """Get error metric info by name.

        Args:
            name: Name of the error metric

        Returns:
            Dictionary with error metric info

        Raises:
            ValueError: If error metric is not registered
        """
        if name not in self._errors:
            raise ValueError(f"Error metric '{name}' not registered")
        return self._errors[name]

    def get_error_id(self, name: str) -> int:
        """Get device-side identifier for an error metric.

        Args:
            name: Name of the error metric

        Returns:
            Device-side identifier
        """
        return self.get(name)['error_id']

    def list_errors(self) -> Dict[str, Dict]:
        """List all registered error metrics.

        Returns:
            Dictionary of registered error metrics
        """
        return self._errors.copy()


# Global registries
objective_registry = ObjectiveRegistry()
error_registry = ErrorRegistry()

# Register objective functions
objective_registry.register("langmuir", 2, 0, "Langmuir isotherm model")
objective_registry.register("sips", 3, 1, "Sips (Langmuir-Freundlich) isotherm model")
objective_registry.register("toth", 3, 2, "Toth isotherm model")
objective_registry.register("bet", 3, 3, "BET isotherm model")
objective_registry.register("gab", 3, 4, "GAB isotherm model")
objective_registry.register("newton", 2, 5, "Linear Newton model")
# registries.py
objective_registry.register("sphere",     3, 100, "Sphere benchmark (De Jong)")
objective_registry.register("rosenbrock", 5, 101, "Rosenbrock benchmark")
objective_registry.register("quartic",    5, 102, "Quartic with noise")
objective_registry.register("schwefel",   5, 103, "Schwefel 2.26 benchmark")
objective_registry.register("rastrigin",  5, 104, "Rastrigin benchmark")
objective_registry.register("ackley",     5, 105, "Ackley benchmark")


# Register error metrics
error_registry.register("sse", 0, "Sum of Squared Errors")
error_registry.register("mse", 1, "Mean Squared Error")
error_registry.register("rmse", 2, "Root Mean Squared Error")
error_registry.register("mae", 3, "Mean Absolute Error")
error_registry.register("mape", 4, "Mean Absolute Percentage Error")
error_registry.register("r2", 5, "R-squared")
