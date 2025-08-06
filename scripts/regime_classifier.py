#!/usr/bin/env python3
"""
Regime Classifier Module for SimpleMLP PSA Analysis

This module provides a clean interface for determining the training regime
of any SimpleMLP configuration based on Persistent Stochastic Ablation (PSA) trial data.

Regimes:
- Untrainable: All 6 modes <= 11.35% accuracy
- Chaotic Optimization: Baselines ineffective (<= 11.35%) but ablative modes show results > 11.35%
- Over-parameterized: Parameter count > 500,000
- At-capacity: Well-sized models (10,000 <= parameter count <= 500,000)

Author: Tim Cotten @cottenio <tcotten@scrypted.ai, tcotten2@gmu.edu>
"""

from pathlib import Path
import re
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def parse_trial_data(trials_file: Path) -> Dict[str, List[Dict]]:
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


def calculate_summary_metrics(trial_results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
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


def calculate_parameter_count(depth: int, width: int) -> int:
    """
    Calculates the parameter count for a SimpleMLP architecture.
    
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


def classify_regime(arch_key: str, metrics: Dict[str, Dict]) -> str:
    """
    Classifies the training regime of an architecture based on its metrics.
    
    Args:
        arch_key: Architecture key (e.g., "1x2048")
        metrics: Summary metrics for the architecture
        
    Returns:
        Regime classification: 'untrainable', 'chaotic-optimization', 'beneficial-regularization', or 'at-capacity'
    """
    if arch_key not in metrics:
        return 'unknown'
    
    m = metrics[arch_key]
    
    # Check if architecture is untrainable (all modes <= 11.35%)
    available_modes = [mode for mode in ['none', 'decay', 'dropout', 'full', 'hidden', 'output'] if mode in m]
    if available_modes:
        all_untrainable = all(m[mode]['mean'] <= 11.35 for mode in available_modes)
        if all_untrainable:
            return 'untrainable'
    
    # Check for Chaotic Optimization: baseline modes ineffective but ablative modes show results > 11.35%
    baseline_modes = ['none', 'decay', 'dropout']
    ablative_modes = ['full', 'hidden', 'output']
    
    # Check if all baseline modes are ineffective (<= 11.35%)
    baseline_available = [mode for mode in baseline_modes if mode in m]
    if baseline_available:
        all_baseline_ineffective = all(m[mode]['mean'] <= 11.35 for mode in baseline_available)
        
        # Check if any ablative mode shows results > 11.35%
        ablative_available = [mode for mode in ablative_modes if mode in m]
        if ablative_available:
            any_ablative_effective = any(m[mode]['mean'] > 11.35 for mode in ablative_available)
            
            if all_baseline_ineffective and any_ablative_effective:
                return 'chaotic-optimization'
    
    # For remaining trainable architectures, determine regime based on performance patterns
    baseline_available = [mode for mode in baseline_modes if mode in m]
    ablative_available = [mode for mode in ablative_modes if mode in m]
    
    if len(baseline_available) >= 2 and len(ablative_available) >= 2:
        # Calculate baseline variance
        baseline_performances = [m[mode]['mean'] for mode in baseline_available]
        baseline_variance = np.var(baseline_performances)
        
        # Calculate ablative variance
        ablative_performances = [m[mode]['mean'] for mode in ablative_available]
        ablative_variance = np.var(ablative_performances)
        
        # Calculate overall variance (all modes)
        all_performances = baseline_performances + ablative_performances
        overall_variance = np.var(all_performances)
        
        # Define thresholds
        baseline_threshold = 5.0  # Baseline modes should be within ~5 percentage points
        ablation_variance_threshold = 10.0  # Ablative modes should show more variance
        
        # Over-parameterized: All modes perform similarly (low overall variance)
        if overall_variance < baseline_threshold:
            return 'beneficial-regularization'
        
        # At-capacity: Baselines similar, ablations show larger variance
        elif baseline_variance < baseline_threshold and ablative_variance > ablation_variance_threshold:
            return 'at-capacity'
        
        # Default to at-capacity for other cases
        else:
            return 'at-capacity'
    
    # Fallback to parameter count if we don't have enough data
    depth, width = map(int, arch_key.split('x'))
    param_count = calculate_parameter_count(depth, width)
    
    if param_count > 500000:
        return 'over-parameterized'
    else:
        return 'at-capacity'


def get_regime_from_data(configuration: str, metrics: Dict[str, Dict]) -> str:
    """
    Main function to get regime classification for a configuration.
    
    Args:
        configuration: Configuration string (e.g., "1*2048" or "1x2048")
        metrics: Summary metrics dictionary
        
    Returns:
        Regime classification string
    """
    # Convert configuration format if needed
    arch_key = configuration.replace('*', 'x')
    
    return classify_regime(arch_key, metrics)


def get_all_regime_classifications(configurations_file: Path, trials_file: Path) -> Dict[str, str]:
    """
    Get regime classifications for all configurations.
    
    Args:
        configurations_file: Path to configurations file
        trials_file: Path to trials data file
        
    Returns:
        Dictionary mapping configuration strings to regime classifications
    """
    # Load data
    trial_results = parse_trial_data(trials_file)
    metrics = calculate_summary_metrics(trial_results)
    
    # Load configurations
    configurations = []
    with open(configurations_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '*' in line:
                configurations.append(line)
    
    # Classify each configuration
    classifications = {}
    for config in configurations:
        regime = get_regime_from_data(config, metrics)
        classifications[config] = regime
    
    return classifications


if __name__ == "__main__":
    # Example usage
    configs_file = Path('reproduction/configurations.txt')
    trials_file = Path('results/psa_simplemlp_trials.md')
    
    if configs_file.exists() and trials_file.exists():
        # Load data
        trial_results = parse_trial_data(trials_file)
        metrics = calculate_summary_metrics(trial_results)
        
        # Load configurations
        configurations = []
        with open(configs_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '*' in line:
                    configurations.append(line)
        
        print("Regime Classifications with Performance Analysis:")
        print("=" * 60)
        
        # Group by regime
        regime_counts = defaultdict(list)
        regime_details = defaultdict(list)
        
        for config in configurations:
            arch_key = config.replace('*', 'x')
            regime = get_regime_from_data(config, metrics)
            regime_counts[regime].append(config)
            
            # Store performance details for analysis
            if arch_key in metrics:
                m = metrics[arch_key]
                baseline_modes = ['none', 'decay', 'dropout']
                ablative_modes = ['full', 'hidden', 'output']
                
                baseline_available = [mode for mode in baseline_modes if mode in m]
                ablative_available = [mode for mode in ablative_modes if mode in m]
                
                if len(baseline_available) >= 2 and len(ablative_available) >= 2:
                    baseline_performances = [m[mode]['mean'] for mode in baseline_available]
                    ablative_performances = [m[mode]['mean'] for mode in ablative_available]
                    all_performances = baseline_performances + ablative_performances
                    
                    baseline_variance = np.var(baseline_performances)
                    ablative_variance = np.var(ablative_performances)
                    overall_variance = np.var(all_performances)
                    
                    regime_details[regime].append({
                        'config': config,
                        'baseline_variance': baseline_variance,
                        'ablative_variance': ablative_variance,
                        'overall_variance': overall_variance,
                        'baseline_performances': baseline_performances,
                        'ablative_performances': ablative_performances
                    })
        
        # Print results with performance analysis
        for regime in ['untrainable', 'chaotic', 'over-parameterized', 'at-capacity']:
            if regime in regime_counts:
                print(f"\n{regime.replace('-', ' ').title()} ({len(regime_counts[regime])}):")
                for config in sorted(regime_counts[regime]):
                    print(f"  {config}")
                
                # Show performance analysis for trainable regimes
                if regime in ['over-parameterized', 'at-capacity'] and regime in regime_details:
                    print(f"\n  Performance Analysis for {regime}:")
                    for detail in regime_details[regime][:5]:  # Show first 5 examples
                        print(f"    {detail['config']}: baseline_var={detail['baseline_variance']:.2f}, "
                              f"ablative_var={detail['ablative_variance']:.2f}, "
                              f"overall_var={detail['overall_variance']:.2f}")
                        print(f"      Baselines: {[f'{p:.1f}%' for p in detail['baseline_performances']]}")
                        print(f"      Ablations: {[f'{p:.1f}%' for p in detail['ablative_performances']]}")
    else:
        print("Configuration or trials file not found.") 