#!/usr/bin/env python3
"""
Generate convergence plots for PSA experiments.

This script creates Figure 4 from the paper, showing validation accuracy
progression across meta-loops for different architectures and training modes.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# --- Configuration ---
CONSOLE = Console()

# Color scheme for training modes (matching make_figure_heatmaps.py)
MODE_COLORS = {
    'none': '#A1887F',      # Brown
    'decay': '#651FFF',     # Purple  
    'dropout': '#E040FB',   # Pink
    'hidden': '#FFEB3B',    # Yellow
    'output': '#00B0FF',    # Cyan
    'full': '#FF8A65'       # Orange
}

MODE_LABELS = {
    'none': 'Control (None)',
    'decay': 'Weight Decay', 
    'dropout': 'Dropout',
    'hidden': 'Hidden Ablation',
    'output': 'Output Ablation', 
    'full': 'Full Ablation'
}

def parse_convergence_data(logs_dir: Path, target_architectures: List[str]) -> Dict:
    """
    Parse convergence data from CloudWatch logs for target architectures.
    
    Returns:
        Dictionary mapping architecture -> mode -> trials -> meta_loops -> accuracies
    """
    convergence_data = {}
    
    # Find all job directories
    job_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=CONSOLE
    ) as progress:
        task = progress.add_task("Parsing convergence data...", total=len(job_dirs))
        
        for job_dir in job_dirs:
            job_name = job_dir.name
            
            # Extract architecture info from job name
            # Format: psa-{width}x{depth}-{mode}-{timestamp}
            arch_match = re.match(r'psa-(\d+)x(\d+)-(\w+)-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3}', job_name)
            if not arch_match:
                progress.advance(task)
                continue
                
            width = int(arch_match.group(1))
            depth = int(arch_match.group(2))
            mode = arch_match.group(3)
            architecture = f"{width}*{depth}"
            
            # Skip if not in target architectures
            if architecture not in target_architectures:
                progress.advance(task)
                continue
            
            # Initialize data structure
            if architecture not in convergence_data:
                convergence_data[architecture] = {}
            if mode not in convergence_data[architecture]:
                convergence_data[architecture][mode] = {}
            
            # Parse log files for this job
            log_files = list(job_dir.rglob("*.log"))
            if not log_files:
                progress.advance(task)
                continue
            
            # Each log file contains all 10 trials for this job
            all_trials_data = {}
            
            for log_file in log_files:
                # Parse log file for all trials' convergence data
                current_trial = 1
                current_trial_data = {}
                
                with open(log_file, 'r') as f:
                    for line in f:
                        # Look for validation accuracy lines
                        # Pattern: "[18:39:19] INFO     Loop 1/100 (Global: 1) | Current: 90.23% | LKG: 90.23% |"
                        match = re.search(r'Loop (\d+)/100.*Current:\s*([\d.]+)%', line)
                        if match:
                            meta_loop = int(match.group(1))
                            accuracy = float(match.group(2))
                            
                            # If we see Loop 1, it's a new trial
                            if meta_loop == 1:
                                # Save previous trial if it exists
                                if current_trial_data:
                                    all_trials_data[current_trial] = current_trial_data
                                    current_trial += 1
                                # Start new trial
                                current_trial_data = {}
                            
                            current_trial_data[meta_loop] = accuracy
                
                # Save the last trial
                if current_trial_data:
                    all_trials_data[current_trial] = current_trial_data
            
            if all_trials_data:
                convergence_data[architecture][mode] = all_trials_data
            
            progress.advance(task)
    
    return convergence_data

def create_convergence_plot(convergence_data: Dict, target_architectures: List[str], 
                          output_file: Path) -> None:
    """
    Create convergence plots showing validation accuracy over meta-loops.
    
    Creates a figure with subplots stacked vertically, one for each target architecture.
    Each subplot shows all trials for all training modes.
    """
    n_architectures = len(target_architectures)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_architectures, 1, figsize=(12, 6 * n_architectures))
    if n_architectures == 1:
        axes = [axes]
    
    # Plot each architecture
    for i, architecture in enumerate(target_architectures):
        ax = axes[i]
        
        if architecture not in convergence_data:
            CONSOLE.print(f"⚠️  No data found for architecture {architecture}", style="yellow")
            continue
        
        # Plot each mode
        for mode in ['none', 'decay', 'dropout', 'hidden', 'output', 'full']:
            if mode not in convergence_data[architecture]:
                continue
            
            color = MODE_COLORS[mode]
            label = MODE_LABELS[mode]
            
            # Find the best performing trial for this mode
            best_trial = None
            best_peak_accuracy = -1
            
            for trial_num, trial_data in convergence_data[architecture][mode].items():
                peak_accuracy = max(trial_data.values())
                if peak_accuracy > best_peak_accuracy:
                    best_peak_accuracy = peak_accuracy
                    best_trial = trial_data
            

            
            # Plot only the best trial
            if best_trial:
                meta_loops = sorted(best_trial.keys())
                accuracies = [best_trial[ml] for ml in meta_loops]
                ax.plot(meta_loops, accuracies, color=color, alpha=0.8, linewidth=2, label=label)
        
        # Calculate dynamic y-axis range based on actual data
        all_accuracies = []
        for mode_data in convergence_data[architecture].values():
            for trial_data in mode_data.values():
                all_accuracies.extend(trial_data.values())
        
        if all_accuracies:
            min_acc = min(all_accuracies)
            max_acc = max(all_accuracies)
            range_acc = max_acc - min_acc
            
            # Set y-axis limits to 5% above and below the data range, but start at 1
            y_min = max(1, min_acc - range_acc * 0.05)
            y_max = min(100, max_acc + range_acc * 0.05)
            
            # Use logarithmic scale for y-axis (but keep percentage formatting)
            ax.set_yscale('log')
            ax.set_ylim(y_min, y_max)
            
            # Format y-axis as percentages and ensure proper tick spacing
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))
            
            # Force more tick marks for better readability
            if y_max > 50:
                # For high-performance ranges, use more tick marks
                ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=8))
            else:
                # For lower ranges, use standard spacing
                ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=6))
        else:
            # Fallback to linear scale if no data, but start at 1
            ax.set_ylim(1, 100)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))
        
        # Customize subplot
        ax.set_xlabel('Meta-Loop')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.set_title(f'Architecture {architecture}')
        ax.set_xlim(0.5, 100.5)  # Start slightly before 1 and end slightly after 100 for better visibility
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        # Add ZeroR baseline line
        ax.axhline(y=11.02, color='red', linestyle='--', alpha=0.5, label='ZeroR Baseline')
        
        # Add text annotation for architecture type
        if architecture == "1*1024":
            ax.text(0.02, 0.98, 'Over-parameterized (Stable)', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        elif architecture == "18*18":
            ax.text(0.02, 0.98, 'Chaotic Optimization', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    CONSOLE.print(f"✅ Convergence plot saved to {output_file}", style="green")
    
    plt.close()

def main():
    """Main function to generate convergence plots."""
    parser = argparse.ArgumentParser(description="Generate convergence plots for PSA experiments")
    parser.add_argument("--logs-dir", type=str, default="results/logs",
                       help="Directory containing downloaded logs (default: results/logs)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for plots (default: results)")
    parser.add_argument("--targets", type=str, default="1*1024,18*18",
                       help="Comma-separated list of target architectures (default: 1*1024,18*18)")
    args = parser.parse_args()
    
    # Parse target architectures
    target_architectures = [arch.strip() for arch in args.targets.split(',')]
    
    CONSOLE.print("📊 PSA Convergence Plot Generator", style="bold blue")
    CONSOLE.print(f"   Logs directory: {args.logs_dir}", style="dim")
    CONSOLE.print(f"   Target architectures: {target_architectures}", style="dim")
    
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    
    if not logs_dir.exists():
        CONSOLE.print(f"🔴 Logs directory not found: {logs_dir}", style="red")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse convergence data
    CONSOLE.print("\n🔍 Parsing convergence data from logs...", style="blue")
    convergence_data = parse_convergence_data(logs_dir, target_architectures)
    
    # Display summary
    CONSOLE.print("\n📊 Data Summary", style="bold green")
    for arch in target_architectures:
        if arch in convergence_data:
            modes = list(convergence_data[arch].keys())
            total_trials = sum(len(trials) for trials in convergence_data[arch].values())
            CONSOLE.print(f"   {arch}: {len(modes)} modes, {total_trials} total trials")
        else:
            CONSOLE.print(f"   {arch}: No data found", style="red")
    
    # Generate plot
    CONSOLE.print("\n🎨 Generating convergence plot...", style="blue")
    output_file = output_dir / "SimpleMLP_Plot_Convergence.png"
    create_convergence_plot(convergence_data, target_architectures, output_file)
    
    CONSOLE.print(f"\n✅ Convergence plot generated successfully!", style="bold green")
    CONSOLE.print(f"   Output: {output_file}", style="dim")
    
    return 0

if __name__ == "__main__":
    exit(main()) 