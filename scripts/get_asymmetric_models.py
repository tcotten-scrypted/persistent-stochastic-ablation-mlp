#!/usr/bin/env python3
"""
Utility module to analyze parameter count matching and identify asymmetric configurations.

This module reads the configurations file and analyzes parameter counts to identify
which architectures are intentionally designed to match parameter counts (circles)
versus those that are asymmetric (triangles) in the parameter matching design.
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict, Set
import numpy as np


def calculate_parameter_count(depth: int, width: int) -> int:
    """
    Calculate the total number of parameters for a given architecture.
    
    Args:
        depth: Number of hidden layers
        width: Number of neurons per hidden layer
        
    Returns:
        Total parameter count
    """
    input_size = 784  # MNIST flattened
    output_size = 10  # MNIST digits
    
    if depth == 0:  # No hidden layers
        return input_size * output_size + output_size  # weights + bias
    else:
        # Input to first hidden layer
        param_count = input_size * width + width  # weights + bias
        # Hidden to hidden layers
        for _ in range(depth - 1):
            param_count += width * width + width  # weights + bias
        # Last hidden to output
        param_count += width * output_size + output_size  # weights + bias
        return param_count


def parse_configurations(config_file: Path) -> List[Tuple[int, int]]:
    """
    Parse the configurations file to extract depth and width values.
    
    Args:
        config_file: Path to the configurations file
        
    Returns:
        List of (depth, width) tuples
    """
    architectures = []
    if not config_file.is_file():
        print(f"Warning: Config file '{config_file}' not found.")
        return architectures
        
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '*' in line:
                try:
                    depth, width = map(int, line.split('*'))
                    architectures.append((depth, width))
                except ValueError:
                    continue
                    
    return architectures


def find_parameter_classes(architectures: List[Tuple[int, int]], tolerance: float = 0.1) -> Dict[int, List[Tuple[int, int]]]:
    """
    Group architectures by parameter count with a tolerance for matching.
    
    Args:
        architectures: List of (depth, width) tuples
        tolerance: Fractional tolerance for parameter count matching (default 10%)
        
    Returns:
        Dictionary mapping representative parameter count to list of matching architectures
    """
    # Calculate parameter counts for all architectures
    param_counts = {}
    for depth, width in architectures:
        param_count = calculate_parameter_count(depth, width)
        param_counts[(depth, width)] = param_count
    
    # Group by parameter count with tolerance
    parameter_classes = {}
    processed = set()
    
    for (depth, width), param_count in param_counts.items():
        if (depth, width) in processed:
            continue
            
        # Find all architectures within tolerance
        matching_archs = []
        for (d, w), pc in param_counts.items():
            if (d, w) not in processed:
                # Check if parameter counts are within tolerance
                if abs(pc - param_count) / param_count <= tolerance:
                    matching_archs.append((d, w))
                    processed.add((d, w))
        
        if matching_archs:
            # Use the median parameter count as the representative
            median_param = int(np.median([param_counts[arch] for arch in matching_archs]))
            parameter_classes[median_param] = matching_archs
    
    return parameter_classes


def parse_asymmetric_configurations(asymmetric_file: Path = Path('reproduction/asymmetric.txt')) -> Set[Tuple[int, int]]:
    """
    Parse the asymmetric configurations file to get the hardcoded asymmetric designs.
    
    Args:
        asymmetric_file: Path to the asymmetric configurations file
        
    Returns:
        Set of asymmetric (depth, width) tuples
    """
    asymmetric = set()
    if not asymmetric_file.is_file():
        print(f"Warning: Asymmetric file '{asymmetric_file}' not found.")
        return asymmetric
        
    with open(asymmetric_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '*' in line and not line.startswith('#'):
                try:
                    depth, width = map(int, line.split('*'))
                    asymmetric.add((depth, width))
                except ValueError:
                    continue
                    
    return asymmetric


def identify_asymmetric_configurations(architectures: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """
    Identify configurations that are asymmetric in the parameter matching design.
    
    Simple approach: Read the hardcoded asymmetric list and mark everything else as symmetric.
    
    Args:
        architectures: List of (depth, width) tuples
        
    Returns:
        Set of asymmetric (depth, width) tuples
    """
    # Get the hardcoded asymmetric configurations
    asymmetric = parse_asymmetric_configurations()
    
    return asymmetric


def get_asymmetric_models(config_file: Path = Path('reproduction/configurations.txt')) -> Set[str]:
    """
    Main function to get asymmetric model configurations.
    
    Args:
        config_file: Path to the configurations file
        
    Returns:
        Set of asymmetric configuration strings (e.g., "939*2")
    """
    architectures = parse_configurations(config_file)
    asymmetric_configs = identify_asymmetric_configurations(architectures)
    
    # Convert to string format
    asymmetric_strings = {f"{depth}*{width}" for depth, width in asymmetric_configs}
    
    return asymmetric_strings


def get_symmetric_models(config_file: Path = Path('reproduction/configurations.txt')) -> Set[str]:
    """
    Get symmetric model configurations (the complement of asymmetric).
    
    Args:
        config_file: Path to the configurations file
        
    Returns:
        Set of symmetric configuration strings
    """
    architectures = parse_configurations(config_file)
    asymmetric_configs = identify_asymmetric_configurations(architectures)
    
    # All architectures minus asymmetric ones
    all_configs = {f"{depth}*{width}" for depth, width in architectures}
    asymmetric_strings = {f"{depth}*{width}" for depth, width in asymmetric_configs}
    
    return all_configs - asymmetric_strings


def analyze_parameter_matching(config_file: Path = Path('reproduction/configurations.txt')) -> Dict:
    """
    Analyze parameter matching and return detailed information.
    
    Args:
        config_file: Path to the configurations file
        
    Returns:
        Dictionary with analysis results
    """
    architectures = parse_configurations(config_file)
    param_classes = find_parameter_classes(architectures)
    asymmetric_configs = identify_asymmetric_configurations(architectures)
    
    # Calculate parameter counts
    param_counts = {f"{depth}*{width}": calculate_parameter_count(depth, width) 
                   for depth, width in architectures}
    
    return {
        'total_architectures': len(architectures),
        'parameter_classes': len(param_classes),
        'asymmetric_count': len(asymmetric_configs),
        'symmetric_count': len(architectures) - len(asymmetric_configs),
        'parameter_classes_detail': param_classes,
        'asymmetric_configs': {f"{depth}*{width}" for depth, width in asymmetric_configs},
        'parameter_counts': param_counts
    }


if __name__ == "__main__":
    # Example usage
    config_file = Path('reproduction/configurations.txt')
    
    print("=== Parameter Matching Analysis ===")
    analysis = analyze_parameter_matching(config_file)
    
    print(f"Total architectures: {analysis['total_architectures']}")
    print(f"Parameter classes: {analysis['parameter_classes']}")
    print(f"Asymmetric configurations: {analysis['asymmetric_count']}")
    print(f"Symmetric configurations: {analysis['symmetric_count']}")
    
    print("\n=== Parameter Classes ===")
    for param_count, archs in analysis['parameter_classes_detail'].items():
        print(f"{param_count:,} params: {[f'{d}*{w}' for d, w in archs]}")
    
    print("\n=== Asymmetric Configurations ===")
    asymmetric = get_asymmetric_models(config_file)
    for config in sorted(asymmetric):
        param_count = analysis['parameter_counts'][config]
        print(f"{config} ({param_count:,} params)")
    
    print("\n=== Symmetric Configurations ===")
    symmetric = get_symmetric_models(config_file)
    for config in sorted(symmetric):
        param_count = analysis['parameter_counts'][config]
        print(f"{config} ({param_count:,} params)") 