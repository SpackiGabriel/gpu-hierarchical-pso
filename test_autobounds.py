"""
Test file for demonstrating automatic parameter bounds setting based on objective function.
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from registries import objective_registry


def demonstrate_automatic_bounds():
    """Demonstrate how parameter bounds are automatically set based on objective function."""
    print("Demonstrating automatic parameter bounds setting...")
    
    # Default bounds when none are provided
    default_bound = [0, 1000000]
    
    # Test with different objectives
    objectives = ["langmuir", "sips", "toth", "bet", "gab", "newton"]
    
    for objective in objectives:
        dim = objective_registry.get_dimension(objective)
        # This is how the bounds would be automatically set in hierarchical_pso
        auto_bounds = [default_bound for _ in range(dim)]
        print(f"{objective}: {dim} parameters -> bounds = {auto_bounds}")
    
    print("\nAutomatic bounds demonstration completed!\n")


def main():
    """Run the demonstration."""
    print("=== Automatic Parameter Bounds Demonstration ===\n")
    
    demonstrate_automatic_bounds()
    
    print("=== Demonstration Completed ===")


if __name__ == "__main__":
    main()
