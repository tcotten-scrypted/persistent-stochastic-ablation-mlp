#!/usr/bin/env python3
"""
Generate LaTeX table of network architectures and their parameter counts.

This script reads configurations from reproduction/configurations.txt and generates
a LaTeX table showing parameter counts for each architecture, categorized by
whether they are matched or asymmetric configurations.

Author: Tim Cotten @cottenio <tcotten@scrypted.ai, tcotten2@gmu.edu>
"""

def count_parameters(input_size, hidden_layers, output_size):
    """Count parameters according to the provided formula."""
    total_params = 0

    if not hidden_layers:
        # No hidden layers, direct input to output
        total_params = output_size * (input_size + 1)
    else:
        # Input to first hidden
        total_params += hidden_layers[0] * (input_size + 1)
        # Hidden to hidden
        for i in range(len(hidden_layers) - 1):
            total_params += hidden_layers[i + 1] * (hidden_layers[i] + 1)
        # Last hidden to output
        total_params += output_size * (hidden_layers[-1] + 1)

    return total_params


def main():
    """Generate LaTeX table from configurations."""
    # Read configurations from file
    config_file = "reproduction/configurations.txt"
    try:
        with open(config_file, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        print("Please ensure the file exists in the reproduction/ directory.")
        return

    # Constants
    N_in = 784  # MNIST input
    N_out = 10  # MNIST output

    # Import asymmetric configurations from utility
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from get_asymmetric_models import get_asymmetric_models
    
    asymmetric_configs = get_asymmetric_models()

    # Generate LaTeX table
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Parameters for Network Architectures}")
    print("\\label{tab:architecture_parameters}")
    print("\\begin{tabular}{lcl}")
    print("\\toprule")
    print("\\textbf{Architecture} & \\textbf{Parameters} & \\textbf{Parameter Matching}\\\\")
    print("\\midrule")

    for line in lines:
        L, H = map(int, line.split('*'))
        hidden_layers = [H] * L
        param_count = count_parameters(N_in, hidden_layers, N_out)
        architecture = f"{L}*{H}"
        formatted_params = f"{param_count:,}"
        class_type = "Matched" if (architecture not in asymmetric_configs) else "Asymmetric"
        print(f"{architecture:<10} & {formatted_params:<15} & {class_type} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    main() 