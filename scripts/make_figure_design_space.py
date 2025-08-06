#!/usr/bin/env python3
"""
Generate design space visualization for SimpleMLP architectures.

This script reads configurations from reproduction/configurations.txt and generates
a logarithmic scatter plot showing the design space of all tested architectures.

Author: Tim Cotten @cottenio <tcotten@scrypted.ai, tcotten2@gmu.edu>
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse


def parse_configurations(config_file):
    """Parse architecture configurations from the file."""
    architectures = []
    
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Parse format like "1*2048" into (depth, width)
                    depth, width = map(int, line.split('*'))
                    architectures.append((depth, width))
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        print("Please ensure the file exists in the reproduction/ directory.")
        return []
    except ValueError as e:
        print(f"Error parsing configuration file: {e}")
        return []
    
    return architectures


def generate_design_space_plot(architectures, output_file):
    """Generate the design space visualization."""
    if not architectures:
        print("No architectures to plot.")
        return False
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set logarithmic scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Axis limits for clarity
    ax.set_xlim(0.8, 3000)
    ax.set_ylim(0.8, 3000)
    
    # Axis labels and title
    ax.set_xlabel('Depth (# of layers)', fontsize=14)
    ax.set_ylabel('Width (neurons per layer)', fontsize=14)
    ax.set_title('SimpleMLP Design Space (Logarithmic Scale)', fontsize=16)
    
    # Grid lines for easier reading
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    
    # Separate architectures by type for proper layering
    non_squares = []
    squares = []
    
    for depth, width in architectures:
        if depth == width:
            squares.append((depth, width))
        else:
            non_squares.append((depth, width))
    
    # Plot non-squares first (so they appear behind)
    for depth, width in non_squares:
        rect = patches.Rectangle((1, 1), depth, width, linewidth=1,
                                 edgecolor='blue', facecolor='cyan', alpha=0.3)
        ax.add_patch(rect)
        # Annotate each rectangle
        ax.text(depth * 1.1, width * 1.1, f'{depth}×{width}', fontsize=8, 
                verticalalignment='bottom')
    
    # Plot squares last (so they appear on top)
    for depth, width in squares:
        rect = patches.Rectangle((1, 1), depth, width, linewidth=1,
                                 edgecolor='blue', facecolor='cyan', alpha=0.3)
        ax.add_patch(rect)
        # Annotate each rectangle
        ax.text(depth * 1.1, width * 1.1, f'{depth}×{width}', fontsize=8, 
                verticalalignment='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Design space plot saved to: {output_file}")
    
    # Close the plot to free memory
    plt.close()
    
    return True


def main():
    """Main function to generate the design space visualization."""
    parser = argparse.ArgumentParser(description="Generate design space visualization for SimpleMLP architectures.")
    parser.add_argument('--config-file', type=str, default='reproduction/configurations.txt',
                       help="Path to configurations file")
    parser.add_argument('--output-file', type=str, default='results/SimpleMLP_Testing_Design_Space.png',
                       help="Path for output PNG file")
    args = parser.parse_args()
    
    # Parse configurations
    architectures = parse_configurations(args.config_file)
    
    if architectures:
        print(f"Loaded {len(architectures)} architectures from {args.config_file}")
        
        # Generate the plot
        success = generate_design_space_plot(architectures, args.output_file)
        
        if success:
            print("✅ Design space visualization generated successfully!")
        else:
            print("❌ Failed to generate design space visualization.")
    else:
        print("❌ No architectures loaded. Check the configuration file.")


if __name__ == "__main__":
    main() 