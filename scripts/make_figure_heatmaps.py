#!/usr/bin/env python3
"""
Generates a suite of data-driven heat map visualizations for the SimpleMLP
design space, based on the results of Persistent Stochastic Ablation (PSA) trials.

This script produces five distinct plots to tell a comprehensive story:
1.  Baseline Performance: The raw performance of the control model.
2.  Ablation Effects: The quantitative benefit or harm of applying PSA.
3.  Instability/Chaos: The trial-to-trial variance of each architecture.
4.  Winning Strategy: The single best-performing mode for each architecture.
5.  Parameter Matching: Parameter counts across architectures to show design equivalences.

Author: Tim Cotten @cottenio <tcotten@scrypted.ai, tcotten2@gmu.edu>
Assisted by: Google's Gemini
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import re
import math
from collections import defaultdict
import sys
sys.path.append(str(Path(__file__).parent))
from regime_classifier import classify_regime

# Color constants for consistent visualization
COLORS = {
    'bright_green': '#4CAF50',    # Beneficial effects
    'deep_green': '#1B5E20',      # Stable/Good performance
    'light_green': '#7CB342',     # Moderate stability
    'red': '#E53935',             # Harmful effects/Unstable
    'brown': '#A1887F',           # Insignificant/Neutral
    'black': '#212121',           # Untrainable
    'blue': '#1565C0',            # Lower performance
    'yellow': '#FFEB3B',          # Moderate instability
    'orange': '#FF8A65',          # High instability
    'purple': '#651FFF',          # Full ablation
    'pink': '#E040FB',            # Hidden ablation
    'cyan': '#00B0FF',            # Output ablation
}

def parse_configurations(config_file: Path) -> list[tuple[int, int]]:
    """
    Parses architecture configurations from a file.
    Expects format like "1*2048" on each line.
    
    Args:
        config_file: Path to the configuration file.
    
    Returns:
        A list of tuples, where each tuple is (depth, width).
    """
    architectures = []
    if not config_file.is_file():
        print(f"Error: Configuration file '{config_file}' not found.")
        return []
    
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '*' in line:
                    depth, width = map(int, line.split('*'))
                    architectures.append((depth, width))
    except ValueError as e:
        print(f"Error parsing configuration file: {e}")
        return []
        
    return architectures

def parse_trial_data(trials_file: Path) -> dict:
    """
    Parses raw trial data from the markdown results file.
    
    Args:
        trials_file: Path to the markdown file containing trial data.
        
    Returns:
        A dictionary mapping architecture keys (e.g., "1x2048") to a list
        of trial result dictionaries.
    """
    trial_results = defaultdict(list)
    if not trials_file.is_file():
        print(f"Error: Trials file '{trials_file}' not found.")
        return {}
        
    with open(trials_file, 'r') as f:
        content = f.read()

    # Regex to find each architecture block and its content
    arch_blocks = re.findall(r'### Architecture: ([\d\*]+)\n\n(.*?)(?=\n### Architecture:|$)', content, re.DOTALL)
    
    for arch_str, block in arch_blocks:
        arch_key = arch_str.replace('*', 'x')
        
        # Find the raw data table within the block
        table_match = re.search(r'\| Trial \|.*?\|(.*)', block, re.DOTALL)
        if not table_match:
            continue
        
        table_content = table_match.group(1)
        lines = table_content.strip().split('\n')
        
        for line in lines:
            if '|' not in line or '---' in line:
                continue

            parts = [p.strip().replace('%', '') for p in line.split('|')]
            if len(parts) >= 8:
                try:
                    trial_data = {
                        'none': float(parts[2]),
                        'decay': float(parts[3]),
                        'dropout': float(parts[4]),
                        'full': float(parts[5]),
                        'hidden': float(parts[6]),
                        'output': float(parts[7])
                    }
                    trial_results[arch_key].append(trial_data)
                except (ValueError, IndexError):
                    continue
                    
    return dict(trial_results)

def calculate_summary_metrics(trial_results: dict) -> dict:
    """
    Calculates summary metrics from trial data.
    
    Args:
        trial_results: Dictionary mapping architecture keys to trial data.
        
    Returns:
        Dictionary mapping architecture keys to summary metrics.
    """
    metrics = {}
    for arch, trials in trial_results.items():
        if not trials:
            continue
            
        # Calculate metrics for each mode
        mode_metrics = {}
        for mode in ['none', 'decay', 'dropout', 'full', 'hidden', 'output']:
            if mode in trials[0]:  # Check if mode exists in first trial
                values = [trial[mode] for trial in trials if mode in trial and trial[mode] is not None]
                if values:
                    mode_metrics[mode] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        if mode_metrics:
            metrics[arch] = mode_metrics
    
    return metrics

def setup_plot(title: str) -> tuple[plt.Figure, plt.Axes]:
    """Creates a standard figure and axis with common styling."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.8, 4000)
    ax.set_ylim(0.8, 4000)
    ax.set_xlabel('Depth (# of layers)', fontsize=14)
    ax.set_ylabel('Width (neurons per layer)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, which="both", linestyle='--', alpha=0.4)
    return fig, ax

def save_and_close(fig: plt.Figure, output_file: Path):
    """Saves a figure to a file and closes it to free memory."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Plot saved successfully to: {output_file}")

def plot_baseline_performance(architectures: list, metrics: dict, output_file: Path):
    """Plot baseline performance (control model accuracy)."""
    fig, ax = setup_plot("Baseline Performance (Control Model)")
    
    depths, widths = zip(*architectures)
    
    # Create custom colormap for baseline accuracy
    from matplotlib.colors import LinearSegmentedColormap
    colors = [COLORS['blue'], COLORS['bright_green']]  # Blue -> green
    cmap = LinearSegmentedColormap.from_list('baseline_gradient', colors, N=100)
    norm = plt.Normalize(vmin=11.35, vmax=100)  # Start from untrainable threshold
    
    # Collect all baseline accuracies for colorbar
    baseline_accuracies = []
    scatter_objects = []
    
    for depth, width in architectures:
        arch_key = f"{depth}x{width}"
        
        if arch_key in metrics:
            m = metrics[arch_key]
            # Check if architecture is untrainable (all modes <= 11.35%)
            available_modes = [mode for mode in ['none', 'decay', 'dropout', 'full', 'hidden', 'output'] if mode in m]
            all_untrainable = all(m[mode]['mean'] <= 11.35 for mode in available_modes)
            
            if all_untrainable:
                color = COLORS['black']
                scatter_obj = ax.scatter(depth, width, c=[color], s=50, alpha=0.7)
            else:
                # Use the 'none' mode for baseline performance
                baseline_acc = m.get('none', {}).get('mean', 0)
                baseline_accuracies.append(baseline_acc)
                scatter_obj = ax.scatter(depth, width, c=[baseline_acc], cmap=cmap, norm=norm, s=50, alpha=0.7)
        else:
            # Default for missing data
            scatter_obj = ax.scatter(depth, width, c=[COLORS['brown']], s=50, alpha=0.7)
        
        scatter_objects.append(scatter_obj)
    
    # Add colorbar for baseline accuracy
    if baseline_accuracies:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Baseline Accuracy (%)', fontsize=12)
    
    # Add legend for untrainable
    ax.scatter([], [], c=[COLORS['black']], s=50, alpha=0.7, label='Untrainable')
    ax.legend()
    
    ax.set_xlabel('Depth (# of layers)')
    ax.set_ylabel('Width (neurons per layer)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    save_and_close(fig, output_file)

def plot_ablation_effects(architectures: list, metrics: dict, trial_results: dict, output_file: Path):
    """Plot ablation effects with statistical significance."""
    fig, ax = setup_plot("Ablation Effects (Statistical Significance)")
    
    for depth, width in architectures:
        arch_key = f"{depth}x{width}"
        
        if arch_key in metrics and arch_key in trial_results:
            m = metrics[arch_key]
            trials = trial_results[arch_key]
            
            # Check if architecture is untrainable (all modes <= 11.35%)
            available_modes = [mode for mode in ['none', 'decay', 'dropout', 'full', 'hidden', 'output'] if mode in m]
            all_untrainable = all(m[mode]['mean'] <= 11.35 for mode in available_modes)
            
            if all_untrainable:
                color = COLORS['black']
                size = 50
            else:
                # Calculate uplift for each trial (comparing full ablation to none)
                uplift_values = []
                for trial in trials:
                    if 'none' in trial and 'full' in trial:
                        uplift = trial['full'] - trial['none']
                        uplift_values.append(uplift)
                
                if uplift_values:
                    # Perform t-test
                    from scipy import stats
                    t_stat, p_value = stats.ttest_1samp(uplift_values, 0)
                    mean_uplift = np.mean(uplift_values)
                    
                    # Determine color based on significance and direction
                    if p_value < 0.05:  # Significant
                        if mean_uplift > 0:
                            color = COLORS['bright_green']  # Beneficial
                        else:
                            color = COLORS['red']  # Harmful
                    else:
                        color = COLORS['brown']  # Insignificant
                    
                    # Size based on effect magnitude (Cohen's d)
                    effect_size = abs(mean_uplift) / np.std(uplift_values) if np.std(uplift_values) > 0 else 0
                    size = 7.5 + (effect_size * 461.5)  # Scale to 7.5-469 range
                else:
                    color = COLORS['brown']
                    size = 50
            
            ax.scatter(depth, width, c=[color], s=size, alpha=0.7)
        else:
            # Default for missing data
            ax.scatter(depth, width, c=[COLORS['brown']], s=50, alpha=0.7)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['bright_green'], markersize=8, label='Beneficial'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['red'], markersize=8, label='Harmful'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['brown'], markersize=8, label='Neutral'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['black'], markersize=8, label='Untrainable')
    ]
    ax.legend(handles=legend_elements)
    
    ax.set_xlabel('Depth (# of layers)')
    ax.set_ylabel('Width (neurons per layer)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    save_and_close(fig, output_file)

def plot_instability(architectures: list, metrics: dict, output_file: Path):
    """Plot instability (max standard deviation across modes)."""
    fig, ax = setup_plot("Instability (Max Std across Six Ablation Modes)")
    
    # Create custom colormap for instability
    from matplotlib.colors import LinearSegmentedColormap
    colors = [COLORS['deep_green'], COLORS['light_green'], COLORS['yellow'], COLORS['orange'], COLORS['red']]
    cmap = LinearSegmentedColormap.from_list('instability_gradient', colors, N=100)
    norm = plt.Normalize(vmin=0, vmax=5)  # Assuming max std is around 5%
    
    # Collect all standard deviations for colorbar
    std_values = []
    scatter_objects = []
    
    for depth, width in architectures:
        arch_key = f"{depth}x{width}"
        
        if arch_key in metrics:
            m = metrics[arch_key]
            # Check if architecture is untrainable (all modes <= 11.35%)
            available_modes = [mode for mode in ['none', 'decay', 'dropout', 'full', 'hidden', 'output'] if mode in m]
            all_untrainable = all(m[mode]['mean'] <= 11.35 for mode in available_modes)
            
            if all_untrainable:
                color = COLORS['black']
                scatter_obj = ax.scatter(depth, width, c=[color], s=50, alpha=0.7)
            else:
                # Calculate maximum standard deviation across all modes
                max_std = max(m[mode]['std'] for mode in available_modes)
                std_values.append(max_std)
                scatter_obj = ax.scatter(depth, width, c=[max_std], cmap=cmap, norm=norm, s=50, alpha=0.7)
        else:
            # Default for missing data
            scatter_obj = ax.scatter(depth, width, c=[COLORS['brown']], s=50, alpha=0.7)
        
        scatter_objects.append(scatter_obj)
    
    # Add colorbar for instability
    if std_values:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Max Standard Deviation (%)', fontsize=12)
    
    # Add legend for untrainable
    ax.scatter([], [], c=[COLORS['black']], s=50, alpha=0.7, label='Untrainable')
    ax.legend()
    
    ax.set_xlabel('Depth (# of layers)')
    ax.set_ylabel('Width (neurons per layer)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    save_and_close(fig, output_file)

def plot_winning_strategy(architectures: list, metrics: dict, output_file: Path):
    """Plots a categorical map showing which mode had the highest mean accuracy."""
    title = 'SimpleMLP Design Space: Winning Strategy Map'
    fig, ax = setup_plot(title)
    
    depths = np.array([arch[0] for arch in architectures])
    widths = np.array([arch[1] for arch in architectures])
    
    # Updated color scheme with distinct colors for each mode
    mode_colors = {
        'none': COLORS['brown'],      # Brown
        'decay': COLORS['purple'],     # Purple
        'dropout': COLORS['pink'],     # Pink
        'full': COLORS['orange'],      # Orange
        'hidden': COLORS['yellow'],    # Yellow
        'output': COLORS['cyan'],      # Cyan
        'untrainable': COLORS['black'] # Dark black
    }
    mode_labels = {
        'none': 'Control (None)', 
        'decay': 'Weight Decay', 
        'dropout': 'Dropout', 
        'full': 'Full Ablation', 
        'hidden': 'Hidden Ablation', 
        'output': 'Output Ablation',
        'untrainable': 'Untrainable'
    }
    
    winners = []
    for d, w in zip(depths, widths):
        arch_key = f"{d}x{w}"
        winner = 'none' # Default
        if arch_key in metrics:
            m = metrics[arch_key]
            # Check if all available modes achieve ≤11.35% (untrainable)
            available_modes = [mode for mode in ['none', 'decay', 'dropout', 'full', 'hidden', 'output'] if mode in m]
            if available_modes:
                all_untrainable = all(m[mode]['mean'] <= 11.35 for mode in available_modes)
                if all_untrainable:
                    winner = 'untrainable'
                else:
                    # Find the mode with the highest mean accuracy
                    winner = max(available_modes, key=lambda mode: m[mode]['mean'])
            else:
                winner = 'untrainable'
        winners.append(winner)
    winners = np.array(winners)

    for mode_name, label in mode_labels.items():
        mask = winners == mode_name
        if np.any(mask):
            ax.scatter(depths[mask], widths[mask], c=mode_colors[mode_name], s=120, edgecolors='black', linewidth=0.5, label=label)

    ax.legend(title='Winning Mode', loc='upper right', fontsize=10)
    
    save_and_close(fig, output_file)

def plot_parameter_matching(architectures: list, output_file: Path):
    """Plots parameter counts across architectures to show parameter matching between different designs."""
    title = 'SimpleMLP Design Space: Parameter Matching & Asymmetries'
    fig, ax = setup_plot(title)

    depths = [arch[0] for arch in architectures]
    widths = [arch[1] for arch in architectures]
    
    # Import the asymmetric models utility
    import sys
    sys.path.append(str(Path(__file__).parent))
    from get_asymmetric_models import get_asymmetric_models
    asymmetric_configs = get_asymmetric_models()
    
    # Calculate parameter counts for each architecture
    parameter_counts = []
    for d, w in zip(depths, widths):
        # Parameter count formula: input_size * hidden_layers + hidden_layers * output_size + biases
        input_size = 784  # MNIST flattened
        output_size = 10  # MNIST digits
        
        if d == 0:  # No hidden layers
            param_count = input_size * output_size + output_size  # weights + bias
        else:
            # Input to first hidden layer
            param_count = input_size * w + w  # weights + bias
            # Hidden to hidden layers
            for _ in range(d - 1):
                param_count += w * w + w  # weights + bias
            # Last hidden to output
            param_count += w * output_size + output_size  # weights + bias
        
        parameter_counts.append(param_count)
    
    # Find min and max for color scaling
    min_params = min(parameter_counts)
    max_params = max(parameter_counts)
    
    # Create logarithmic size scaling
    log_params = np.log10(parameter_counts)
    min_log = np.log10(min_params)
    max_log = np.log10(max_params)
    
    # Normalize sizes (7.5 to 469)
    sizes = 7.5 + ((log_params - min_log) / (max_log - min_log)) * 461.5
    
    # Create custom colormap with distinct ranges
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colors as mcolors
    
    # Define parameter count ranges and colors (rainbow gradient)
    ranges = [0, 1000, 10000, 100000, 1000000, max_params]
    colors = ['#000000', '#FF0000', '#FF8000', '#FFFF00', '#00FF00', '#00FFFF']  # Black, Red, Orange, Yellow, Green, Cyan
    
    # Create custom colormap with logarithmic spacing
    norm = mcolors.LogNorm(vmin=min_params, vmax=max_params)
    cmap = LinearSegmentedColormap.from_list('parameter_gradient', colors, N=100)
    
    # Create scatter plots with different markers for symmetric vs asymmetric
    symmetric_depths = []
    symmetric_widths = []
    symmetric_params = []
    symmetric_sizes = []
    
    asymmetric_depths = []
    asymmetric_widths = []
    asymmetric_params = []
    asymmetric_sizes = []
    
    for i, (depth, width) in enumerate(zip(depths, widths)):
        config = f"{depth}*{width}"
        param_count = parameter_counts[i]
        size = sizes[i]
        
        if config in asymmetric_configs:
            asymmetric_depths.append(depth)
            asymmetric_widths.append(width)
            asymmetric_params.append(param_count)
            asymmetric_sizes.append(size)
        else:
            symmetric_depths.append(depth)
            symmetric_widths.append(width)
            symmetric_params.append(param_count)
            symmetric_sizes.append(size)
    
    # Plot symmetric configurations as circles
    if symmetric_depths:
        sc1 = ax.scatter(symmetric_depths, symmetric_widths, c=symmetric_params, cmap=cmap, norm=norm, 
                        s=symmetric_sizes, edgecolors='black', linewidth=0.5, marker='o', label='Symmetric')
    
    # Plot asymmetric configurations as triangles
    if asymmetric_depths:
        sc2 = ax.scatter(asymmetric_depths, asymmetric_widths, c=asymmetric_params, cmap=cmap, norm=norm, 
                        s=asymmetric_sizes, edgecolors='black', linewidth=0.5, marker='^', label='Asymmetric')
    
    # Create "virtual boxes" (rectangular buffers) for all points to ensure adjustText
    # correctly avoids their actual visual areas on the log-scaled plot.
    # Create rectangles in DATA coordinates with explicit transform for adjustText
    import matplotlib.patches as patches
    virtual_boxes_display = []
    
    # Process symmetric points (circles)
    if 'sc1' in locals() and sc1:
        # Get sizes and offsets in data coordinates
        sizes_data = sc1.get_sizes()
        offsets_data = sc1.get_offsets()
        
        for offset, size in zip(offsets_data, sizes_data):
            # Calculate marker radius in points and convert to pixels
            radius_points = np.sqrt(size) / 2
            radius_pixels = radius_points * (fig.dpi / 72)
            
            # Convert pixel radius back to data coordinates
            center_display = ax.transData.transform(offset)
            bottom_left_display = center_display - radius_pixels
            top_right_display = center_display + radius_pixels
            
            # Convert these points back to data coordinates for rectangle creation
            bottom_left_data = ax.transData.inverted().transform(bottom_left_display)
            top_right_data = ax.transData.inverted().transform(top_right_display)
            width_data, height_data = top_right_data - bottom_left_data
            
            rect_patch_data = patches.Rectangle(
                bottom_left_data, width_data, height_data,
                transform=ax.transData  # critical to explicitly set data transform
            )
            virtual_boxes_display.append(rect_patch_data)

    # Process asymmetric points (triangles)
    if 'sc2' in locals() and sc2:
        sizes_data = sc2.get_sizes()
        offsets_data = sc2.get_offsets()
        
        for offset, size in zip(offsets_data, sizes_data):
            # Calculate marker radius in points and convert to pixels
            radius_points = np.sqrt(size) / 2
            radius_pixels = radius_points * (fig.dpi / 72)
            
            # Convert pixel radius back to data coordinates
            center_display = ax.transData.transform(offset)
            bottom_left_display = center_display - radius_pixels
            top_right_display = center_display + radius_pixels
            
            # Convert these points back to data coordinates for rectangle creation
            bottom_left_data = ax.transData.inverted().transform(bottom_left_display)
            top_right_data = ax.transData.inverted().transform(top_right_display)
            width_data, height_data = top_right_data - bottom_left_data
            
            rect_patch_data = patches.Rectangle(
                bottom_left_data, width_data, height_data,
                transform=ax.transData  # critical to explicitly set data transform
            )
            virtual_boxes_display.append(rect_patch_data)
    
    # Add diagonal dashed line from bottom-left to top-right
    # Use axis limits to extend to the edges
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.plot([x_min, x_max], [y_min, y_max], 'k--', alpha=0.3, linewidth=1, zorder=0)
    
    from adjustText import adjust_text
    
    # Add labels for each point, with logic to prevent overlap
    texts_to_adjust = [] # For the square series
    for i, (depth, width) in enumerate(zip(depths, widths)):
        config = f"{depth}*{width}"
        size = sizes[i] # This is the marker size in points^2

        # Classify architecture and apply a tailored labeling scheme
        if width > depth:
            # A) Shallow-and-Wide (k*W): Label above with dynamic, log-aware offset
            # (This logic is already working well, so we draw it directly)
            x_pos, ha, va = depth, 'center', 'bottom'
            radii = np.sqrt(sizes)
            radius = np.sqrt(size)
            min_radius, max_radius = min(radii), max(radii)
            normalized_radius = (radius - min_radius) / (max_radius - min_radius) if max_radius > min_radius else 0
            offset_factor = 1.025 + (normalized_radius * 0.2)
            y_pos = width * offset_factor
            fontsize = 7
            ax.text(x_pos, y_pos, config,
                fontsize=fontsize, ha=ha, va=va, color='#212121',
                fontfamily='monospace', alpha=0.8, zorder=5)

        elif width == depth:
            # B) Square (k*k): Collect these labels for adjustText to handle.
            # Give labels an initial nudge to the right to encourage fanning.
            x_pos, y_pos = depth + 0.1, width 
            fontsize = 6
            texts_to_adjust.append(ax.text(x_pos, y_pos, config,
                fontsize=fontsize, ha='left', va='center', color='#212121',
                fontfamily='monospace', alpha=0.8, zorder=5))

        else: # width < depth
            # C) Deep-and-Narrow (L*k): Label below with dynamic, log-aware offset
            # (This logic is also working, so we draw it directly)
            x_pos, ha, va = depth, 'center', 'top'
            radii = np.sqrt(sizes)
            radius = np.sqrt(size)
            min_radius, max_radius = min(radii), max(radii)
            normalized_radius = (radius - min_radius) / (max_radius - min_radius) if max_radius > min_radius else 0
            offset_factor = 0.975 - (normalized_radius * 0.25)
            y_pos = width * offset_factor
            fontsize = 6
            ax.text(x_pos, y_pos, config,
                fontsize=fontsize, ha=ha, va=va, color='#212121',
                fontfamily='monospace', alpha=0.8, zorder=5)

    # Mathematical fan-out algorithm for square labels with clustering analysis
    if texts_to_adjust:
        # Collect square label positions and sort by depth
        square_labels = []
        for text in texts_to_adjust:
            depth = float(text.get_text().split('*')[0])
            width = depth  # squares have depth == width
            square_labels.append((depth, width, text))
        
        square_labels.sort(key=lambda x: x[0])  # Sort by depth
        
        # Analyze the distribution using a simple, tunable factor
        depths = [label[0] for label in square_labels]
        
        # Calculate distances between adjacent points
        adjacent_distances = []
        for i in range(len(depths) - 1):
            log_distance = abs(np.log10(depths[i+1]) - np.log10(depths[i]))
            adjacent_distances.append(log_distance)
        
        # Calculate average distance between adjacent points
        avg_distance = np.mean(adjacent_distances)
        
        # Use a tunable factor to define "close" points
        cluster_factor = 1.5  # Points within 1x the average distance are considered clustered
        distance_threshold = avg_distance * cluster_factor
        
        # Find all points that are close to their neighbors
        clustered_indices = []
        for i in range(len(depths) - 1):
            log_distance = abs(np.log10(depths[i+1]) - np.log10(depths[i]))
            if log_distance <= distance_threshold:
                clustered_indices.extend([i, i+1])
        
        # Remove duplicates and sort
        clustered_indices = sorted(list(set(clustered_indices)))
        
        # Find the largest continuous segment
        if clustered_indices:
            cluster_start_idx = clustered_indices[0]
            cluster_end_idx = clustered_indices[-1]
        else:
            # Fallback: use middle third of squares
            cluster_start_idx = len(square_labels) // 3
            cluster_end_idx = 2 * len(square_labels) // 3
        
        cluster_squares = square_labels[cluster_start_idx:cluster_end_idx + 1]
        cluster_depths = [sq[0] for sq in cluster_squares]
        tightest_center = np.mean(cluster_depths)
        
        print(f"DEBUG: Cluster factor: {cluster_factor}x average distance")
        print(f"DEBUG: Average distance: {avg_distance:.3f}, Threshold: {distance_threshold:.3f}")
        print(f"DEBUG: Clustered range from {cluster_squares[0][0]}×{cluster_squares[0][0]} to {cluster_squares[-1][0]}×{cluster_squares[-1][0]}")
        print(f"DEBUG: Cluster contains {len(cluster_squares)} squares")
        
        # Create Bezier curve for clustered squares
        if cluster_squares:
            # Get the clustered squares in order
            clustered_depths = [sq[0] for sq in cluster_squares]
            clustered_texts = [sq[2] for sq in cluster_squares]
            
            # Calculate Bezier curve control points
            # Start point: leftmost square position
            start_x = clustered_depths[0]
            start_y = clustered_depths[0]  # squares have depth == width
            
            # End point: rightmost square position  
            end_x = clustered_depths[-1]
            end_y = clustered_depths[-1]
            
            # Control point: middle square pushed furthest out (bottom-right)
            mid_idx = len(clustered_depths) // 2
            mid_depth = clustered_depths[mid_idx]
            
            # Push the middle point furthest out (bottom-right)
            # Account for logarithmic scaling: use multiplicative offsets like the other groups
            # Convert to points for consistent scaling
            control_x = mid_depth * 4.0  # 30% further right on log scale
            control_y = mid_depth * 0.2  # 30% further down on log scale
            
            # Generate Bezier curve points
            t_values = np.linspace(0, 1, len(clustered_depths))
            bezier_x = []
            bezier_y = []
            
            for t in t_values:
                # Quadratic Bezier curve formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
                x = (1-t)**2 * start_x + 2*(1-t)*t * control_x + t**2 * end_x
                y = (1-t)**2 * start_y + 2*(1-t)*t * control_y + t**2 * end_y
                bezier_x.append(x)
                bezier_y.append(y)
            
            # Position each text along the Bezier curve
            for i, (depth, width, text) in enumerate(cluster_squares):
                new_x = bezier_x[i]
                new_y = bezier_y[i]
                text.set_position((new_x, new_y))
                
                # Draw connecting line from circle edge to text
                # Calculate circle radius in data coordinates based on marker size
                # The marker size is in points^2, so radius = sqrt(size)/2 in points
                # Convert to data coordinates for the current point
                marker_size = sizes[depths.index(depth)]  # Get the marker size for this depth
                radius_points = np.sqrt(marker_size) / 2
                
                # Convert points to data coordinates (approximate)
                # On log scale, we need to account for the scaling factor
                circle_radius_data = depth * (radius_points / 100)  # Approximate conversion
                circle_edge_x = depth + circle_radius_data
                circle_edge_y = width
                
                # Draw line from circle edge to text (behind everything)
                ax.plot([circle_edge_x, new_x], [circle_edge_y, new_y], 
                       color='gray', linewidth=0.5, alpha=0.6, zorder=0)
        
        # Handle non-clustered squares with dynamic, log-aware right offset
        for i, (depth, width, text) in enumerate(square_labels):
            is_in_dense_cluster = cluster_start_idx <= i <= cluster_end_idx
            if not is_in_dense_cluster:
                # Calculate marker size to position label just outside the circle
                marker_size = sizes[depths.index(depth)]
                radius_points = np.sqrt(marker_size) / 2
                circle_radius_data = depth * (radius_points / 100)  # Same conversion as above
                
                # Apply dynamic, log-aware offset based on shape sizes (like Group A vertical positioning)
                radii = np.sqrt(sizes)
                radius = np.sqrt(marker_size)
                min_radius, max_radius = min(radii), max(radii)
                normalized_radius = (radius - min_radius) / (max_radius - min_radius) if max_radius > min_radius else 0
                offset_factor = 1.025 + (normalized_radius * 0.2)  # Same logic as Group A but for horizontal
                
                new_x = depth * offset_factor + circle_radius_data
                new_y = width
                text.set_position((new_x, new_y))
                text.set_horizontalalignment('left')  # Left-align the text
    
    # Use the first scatter plot for colorbar
    sc = sc1 if symmetric_depths else sc2
    
    # Add colorbar with logarithmic ticks
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Parameter Count (Logarithmic)', fontsize=12)
    
    # Set custom tick marks for the colorbar
    tick_values = [1000, 10000, 100000, 1000000]
    tick_labels = ['1K', '10K', '100K', '1M']
    
    # Add min and max if they're not already included
    if min_params < 1000:
        tick_values.insert(0, min_params)
        tick_labels.insert(0, f'{min_params:,}')
    if max_params > 1000000:
        tick_values.append(max_params)
        tick_labels.append(f'{max_params/1000000:.1f}M')
    
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(tick_labels)
    
    # Add legend for shapes only using Line2D for proper marker display
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Matched (Circle)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Asymmetric (Triangle)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    save_and_close(fig, output_file)

def plot_regimes(architectures: list, metrics: dict, output_file: Path):
    """Plots training regimes based on performance patterns."""
    title = 'SimpleMLP Design Space: Training Regimes'
    fig, ax = setup_plot(title)
    
    depths = [arch[0] for arch in architectures]
    widths = [arch[1] for arch in architectures]
    
    # Regime colors
    regime_colors = {
        'beneficial-regularization': '#4CAF50',  # Green
        'optimally-sized': '#E53935',      # Red
        'chaotic-optimization': '#1565C0',          # Blue
        'untrainable': '#212121',      # Black
        'unknown': '#A1887F'           # Brown for unknown
    }
    
    regime_labels = {
        'beneficial-regularization': 'I. Beneficial Regularization',
        'optimally-sized': 'II. Optimally Sized',
        'chaotic-optimization': 'III. Chaotic Optimization',
        'untrainable': 'IV. Untrainable',
        'unknown': 'Unknown'
    }
    
    # Classify each architecture
    regimes = []
    for depth, width in zip(depths, widths):
        arch_key = f"{depth}x{width}"
        regime = classify_regime(arch_key, metrics)
        regimes.append(regime)
    
    # Plot each regime with appropriate color
    for regime_name in ['beneficial-regularization', 'optimally-sized', 'chaotic-optimization', 'untrainable', 'unknown']:
        mask = [r == regime_name for r in regimes]
        if any(mask):
            regime_depths = [d for d, m in zip(depths, mask) if m]
            regime_widths = [w for w, m in zip(widths, mask) if m]
            ax.scatter(regime_depths, regime_widths, c=regime_colors[regime_name], 
                      s=120, edgecolors='black', linewidth=0.5, 
                      label=regime_labels[regime_name])
    
    # Add labels using the same system as parameter matching
    from adjustText import adjust_text
    
    # Create "virtual boxes" for all points
    import matplotlib.patches as patches
    virtual_boxes_display = []
    
    # Process all points
    for depth, width in zip(depths, widths):
        # Calculate marker radius in points and convert to pixels
        radius_points = np.sqrt(120) / 2  # Fixed size for regime plot
        radius_pixels = radius_points * (fig.dpi / 72)
        
        # Convert pixel radius back to data coordinates
        center_display = ax.transData.transform((depth, width))
        bottom_left_display = center_display - radius_pixels
        top_right_display = center_display + radius_pixels
        
        # Convert these points back to data coordinates for rectangle creation
        bottom_left_data = ax.transData.inverted().transform(bottom_left_display)
        top_right_data = ax.transData.inverted().transform(top_right_display)
        width_data, height_data = top_right_data - bottom_left_data
        
        rect_patch_data = patches.Rectangle(
            bottom_left_data, width_data, height_data,
            transform=ax.transData
        )
        virtual_boxes_display.append(rect_patch_data)
    
    # Add diagonal dashed line
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.plot([x_min, x_max], [y_min, y_max], 'k--', alpha=0.3, linewidth=1, zorder=0)
    
    # Add labels for each point with logic to prevent overlap
    texts_to_adjust = []
    for i, (depth, width) in enumerate(zip(depths, widths)):
        config = f"{depth}*{width}"
        
        # Classify architecture and apply a tailored labeling scheme
        if width > depth:
            # A) Shallow-and-Wide (k*W): Label above with dynamic, log-aware offset
            x_pos, ha, va = depth, 'center', 'bottom'
            offset_factor = 1.15  # Increased from 1.025
            y_pos = width * offset_factor
            fontsize = 7
            ax.text(x_pos, y_pos, config,
                fontsize=fontsize, ha=ha, va=va, color='#212121',
                fontfamily='monospace', alpha=0.8, zorder=5)

        elif width == depth:
            # B) Square (k*k): Collect these labels for adjustText to handle
            x_pos, y_pos = depth + 0.2, width  # Increased from 0.1
            fontsize = 6
            texts_to_adjust.append(ax.text(x_pos, y_pos, config,
                fontsize=fontsize, ha='left', va='center', color='#212121',
                fontfamily='monospace', alpha=0.8, zorder=5))

        else: # width < depth
            # C) Deep-and-Narrow (L*k): Label below with dynamic, log-aware offset
            x_pos, ha, va = depth, 'center', 'top'
            offset_factor = 0.85  # Decreased from 0.975 (further down)
            y_pos = width * offset_factor
            fontsize = 6
            ax.text(x_pos, y_pos, config,
                fontsize=fontsize, ha=ha, va=va, color='#212121',
                fontfamily='monospace', alpha=0.8, zorder=5)

    # Handle square labels with clustering analysis (simplified version)
    if texts_to_adjust:
        # Collect square label positions and sort by depth
        square_labels = []
        for text in texts_to_adjust:
            depth = float(text.get_text().split('*')[0])
            width = depth  # squares have depth == width
            square_labels.append((depth, width, text))
        
        square_labels.sort(key=lambda x: x[0])  # Sort by depth
        
        # Analyze the distribution
        depths = [label[0] for label in square_labels]
        
        # Calculate distances between adjacent points
        adjacent_distances = []
        for i in range(len(depths) - 1):
            log_distance = abs(np.log10(depths[i+1]) - np.log10(depths[i]))
            adjacent_distances.append(log_distance)
        
        # Calculate average distance between adjacent points
        avg_distance = np.mean(adjacent_distances) if adjacent_distances else 1.0
        
        # Use a tunable factor to define "close" points
        cluster_factor = 1.5
        distance_threshold = avg_distance * cluster_factor
        
        # Find all points that are close to their neighbors
        clustered_indices = []
        for i in range(len(depths) - 1):
            log_distance = abs(np.log10(depths[i+1]) - np.log10(depths[i]))
            if log_distance <= distance_threshold:
                clustered_indices.extend([i, i+1])
        
        # Remove duplicates and sort
        clustered_indices = sorted(list(set(clustered_indices)))
        
        # Find the largest continuous segment
        if clustered_indices:
            cluster_start_idx = clustered_indices[0]
            cluster_end_idx = clustered_indices[-1]
        else:
            # Fallback: use middle third of squares
            cluster_start_idx = len(square_labels) // 3
            cluster_end_idx = 2 * len(square_labels) // 3
        
        cluster_squares = square_labels[cluster_start_idx:cluster_end_idx + 1]
        
        # Create Bezier curve for clustered squares
        if cluster_squares:
            # Get the clustered squares in order
            clustered_depths = [sq[0] for sq in cluster_squares]
            clustered_texts = [sq[2] for sq in cluster_squares]
            
            # Calculate Bezier curve control points
            start_x = clustered_depths[0]
            start_y = clustered_depths[0]
            end_x = clustered_depths[-1]
            end_y = clustered_depths[-1]
            
            # Control point: middle square pushed furthest out
            mid_idx = len(clustered_depths) // 2
            mid_depth = clustered_depths[mid_idx]
            control_x = mid_depth * 6.0  # Increased from 4.0
            control_y = mid_depth * 0.15  # Decreased from 0.2 (further down)
            
            # Generate Bezier curve points
            t_values = np.linspace(0, 1, len(clustered_depths))
            bezier_x = []
            bezier_y = []
            
            for t in t_values:
                # Quadratic Bezier curve formula
                x = (1-t)**2 * start_x + 2*(1-t)*t * control_x + t**2 * end_x
                y = (1-t)**2 * start_y + 2*(1-t)*t * control_y + t**2 * end_y
                bezier_x.append(x)
                bezier_y.append(y)
            
            # Position each text along the Bezier curve
            for i, (depth, width, text) in enumerate(cluster_squares):
                new_x = bezier_x[i]
                new_y = bezier_y[i]
                text.set_position((new_x, new_y))
                
                # Draw connecting line from circle edge to text
                circle_radius_data = depth * (radius_points / 100)
                circle_edge_x = depth + circle_radius_data
                circle_edge_y = width
                
                ax.plot([circle_edge_x, new_x], [circle_edge_y, new_y], 
                       color='gray', linewidth=0.5, alpha=0.6, zorder=0)
        
        # Handle non-clustered squares with dynamic offset
        for i, (depth, width, text) in enumerate(square_labels):
            is_in_dense_cluster = cluster_start_idx <= i <= cluster_end_idx
            if not is_in_dense_cluster:
                offset_factor = 1.15  # Increased from 1.025
                new_x = depth * offset_factor
                new_y = width
                text.set_position((new_x, new_y))
                text.set_horizontalalignment('left')
    
    # Add legend
    ax.legend(title='Training Regime', loc='upper right', fontsize=10)
    
    save_and_close(fig, output_file)

def main():
    """Main function to parse arguments and generate all heat map visualizations."""
    parser = argparse.ArgumentParser(description="Generate a suite of heat map visualizations for SimpleMLP PSA results.")
    parser.add_argument('--config-file', type=Path, default=Path('reproduction/configurations.txt'),
                        help="Path to the architecture configurations file.")
    parser.add_argument('--trials-file', type=Path, default=Path('results/psa_simplemlp_trials.md'),
                        help="Path to the markdown file with raw trial data.")
    parser.add_argument('--output-dir', type=Path, default=Path('results/'),
                        help="Directory to save the output PNG files.")
    args = parser.parse_args()

    # --- Load and Process Data ---
    print("--- Starting Visualization Generation ---")
    architectures = parse_configurations(args.config_file)
    if not architectures:
        print("❌ Aborting: No architectures loaded. Check the configuration file path.")
        return

    trial_results = parse_trial_data(args.trials_file)
    if not trial_results:
        print("❌ Aborting: No trial data loaded. Check the trials file path and format.")
        return
        
    metrics = calculate_summary_metrics(trial_results)
    
    print(f"Loaded {len(architectures)} architectures.")
    print(f"Loaded trial data for {len(trial_results)} architectures.")
    print("--- Generating Plots ---")

    # --- Generate and Save Plots ---
    plot_baseline_performance(architectures, metrics, args.output_dir / 'SimpleMLP_Heatmap_Baseline_Performance.png')
    plot_ablation_effects(architectures, metrics, trial_results, args.output_dir / 'SimpleMLP_Heatmap_Ablation_Effects.png')
    plot_instability(architectures, metrics, args.output_dir / 'SimpleMLP_Heatmap_Instability.png')
    plot_winning_strategy(architectures, metrics, args.output_dir / 'SimpleMLP_Heatmap_Winning_Strategy.png')
    plot_parameter_matching(architectures, args.output_dir / 'SimpleMLP_Heatmap_Parameter_Matching.png')
    plot_regimes(architectures, metrics, args.output_dir / 'SimpleMLP_Heatmap_Regimes.png')

    print("--- Visualization Generation Complete ---")

if __name__ == "__main__":
    main() 