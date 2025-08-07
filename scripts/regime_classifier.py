#!/usr/bin/env python3
"""
Regime Classifier Module for SimpleMLP PSA Analysis

This module provides a clean interface for determining the training regime
of any SimpleMLP configuration based on Persistent Stochastic Ablation (PSA) trial data.
This version uses rule-based classification based on the performance patterns
observed in the comprehensive trial data.

Regimes:
- Untrainable: Vanishing Gradient Problem - no mode exceeds validation ZeroR baseline accuracy
- Chaotic Optimization: Baseline modes are ineffective (<= validation ZeroR), but at least one ablative mode shows significant learning (> validation ZeroR).
- Beneficial Regularization: High-performing models where traditional regularizers (Dropout, Decay) often outperform the baseline, and the performance gap between the best and worst modes is relatively small.
- Optimally Sized: High-performing models where the baseline ('none') is consistently the best performer, and any intervention (regularization or ablation) is detrimental.

Author: Tim Cotten @cottenio <tcotten@scrypted.ai, tcotten2@gmu.edu>
"""

from pathlib import Path
import re
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add the project root to the path to import analyze_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_dataset import analyze_mnist

# Cache for the validation ZeroR baseline to avoid recomputing
_VALIDATION_ZEROR_CACHE = None

def get_validation_zeror_baseline() -> float:
    """
    Get the validation set ZeroR baseline dynamically from the dataset analysis.
    
    Returns:
        The validation set ZeroR accuracy as a percentage (e.g., 11.02)
    """
    global _VALIDATION_ZEROR_CACHE
    
    if _VALIDATION_ZEROR_CACHE is None:
        print("🔍 Computing validation ZeroR baseline from MNIST dataset analysis...")
        analysis = analyze_mnist()
        _VALIDATION_ZEROR_CACHE = analysis.split_validation.zeror_accuracy
        print(f"📊 Validation ZeroR baseline: {_VALIDATION_ZEROR_CACHE:.2f}%")
    
    return _VALIDATION_ZEROR_CACHE


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
            if mode in trials[0]:
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
    input_size = 784
    output_size = 10
    
    if depth == 0:
        return input_size * output_size + output_size
    else:
        param_count = input_size * width + width
        param_count += (depth - 1) * (width * width + width)
        param_count += width * output_size + output_size
        return param_count


def classify_regime(arch_key: str, metrics: Dict[str, Dict]) -> str:
    """
    Classifies the training regime of an architecture based on its performance metrics.
    This version uses rule-based logic derived from empirical data patterns.
    
    Args:
        arch_key: Architecture key (e.g., "1x2048").
        metrics: Summary metrics for the architecture.
        
    Returns:
        Regime classification string.
    """
    if arch_key not in metrics:
        return 'Unknown'
    
    # Get the dynamic validation ZeroR baseline
    validation_zeror = get_validation_zeror_baseline()
    
    m = metrics[arch_key]
    
    all_modes = ['none', 'decay', 'dropout', 'full', 'hidden', 'output']
    baseline_modes = ['none', 'decay', 'dropout']
    ablative_modes = ['full', 'hidden', 'output']
    
    available_modes = [mode for mode in all_modes if mode in m]
    available_baselines = [mode for mode in baseline_modes if mode in m]
    available_ablatives = [mode for mode in ablative_modes if mode in m]

    if not available_modes or len(available_modes) < 6:
        return 'unknown'

    # --- Rule 1, Regime IV: Vanishing Gradient Problem (Untrainable) ---
    
    # Condition A: Untrainable
    # Check if NO mode's MAX performance exceeds validation ZeroR baseline - all modes' best trials are at or below this threshold
    # This indicates a fundamental architectural flaw where gradients vanish
    # and the network cannot learn meaningful representations
    max_perf_across_modes = max(m[mode]['max'] for mode in available_modes)
    if max_perf_across_modes <= validation_zeror:
        return 'untrainable'
    
    # Condition B: Unrescuable
    # Check if the means are all below validation ZeroR baseline, and if there is no MAX in the
    # ablative modes that exceeds validation ZeroR baseline
    if all(m[mode]['mean'] <= validation_zeror for mode in available_modes) and \
       all(m[mode]['max'] <= validation_zeror for mode in available_ablatives):
        return 'untrainable'
    
    # --- Rule 2, Regime I: Beneficial Regularization ---
    # Check if baseline means are within 1 std of each other, and each ablation mode is within 1 std of baseline
    # This indicates the model is robust and performs consistently across all modes
    # Calculate baseline statistics
    baseline_means = [m[mode]['mean'] for mode in available_baselines]
    baseline_mean = np.mean(baseline_means)
    baseline_std = np.std(baseline_means)
    
    # Cap std at 0.5% if it's too small (for high-performing models)
    effective_std = max(baseline_std, 0.5)
    
    # Check if all baseline means are within 1 std of baseline mean
    baselines_within_std = all(abs(mean - baseline_mean) <= effective_std for mean in baseline_means)
    
    # Check if each ablation mode is within 1 std of baseline mean
    ablations_within_std = all(abs(m[mode]['mean'] - baseline_mean) <= effective_std for mode in available_ablatives)
    
    # Check if ablation modes are consistent among themselves (within 1 std of ablation mean)
    ablative_means = [m[mode]['mean'] for mode in available_ablatives]
    ablative_mean = np.mean(ablative_means)
    ablative_std = np.std(ablative_means)
    effective_ablative_std = max(ablative_std, 0.5)
    ablations_consistent = all(abs(m[mode]['mean'] - ablative_mean) <= effective_ablative_std for mode in available_ablatives)
    
    # Beneficial Regularization: baselines are consistent AND (ablations are within baseline std OR there's significant overlap)
    if baselines_within_std and ablations_within_std:
        return 'beneficial-regularization'
    
    # Also check for significant overlap between baseline and ablation ranges
    if baselines_within_std:
        max_ablation = max(ablative_means)
        min_baseline = min(baseline_means)
        overlap = max_ablation - min_baseline
        
        # If there's any overlap (>0.01%), it's beneficial regularization
        if overlap >= 0.01:
            return 'beneficial-regularization'
    
    # Debug Rule 2 if it doesn't match
    print(f"\n=== Rule 2 Debug: {arch_key} ===")
    print(f"  Baseline means: {[f'{mean:.2f}%' for mean in baseline_means]}")
    print(f"  Baseline mean: {baseline_mean:.2f}%, Baseline std: {baseline_std:.2f}%, Effective std: {effective_std:.2f}%")
    print(f"  Baseline within std: {baselines_within_std}")
    ablation_means_debug = [f"{m[mode]['mean']:.2f}%" for mode in available_ablatives]
    print(f"  Ablation means: {ablation_means_debug}")
    print(f"  Ablations within std: {ablations_within_std}")
    print(f"  Rule 2 result: {baselines_within_std and ablations_within_std}")
    
    # --- Rule 3, Regime II: Optimally Sized ---
    # Check if ablation actively harms and underperforms baseline by 1 standard deviation
    # This indicates the model is optimally sized - any intervention hurts performance
    baseline_means = [m[mode]['mean'] for mode in available_baselines]
    ablative_means = [m[mode]['mean'] for mode in available_ablatives]
    baseline_mean = np.mean(baseline_means)
    ablative_mean = np.mean(ablative_means)
    
    # Calculate the standard deviation of baseline performance
    baseline_std = np.std(baseline_means)
    effective_std = max(baseline_std, 0.5)  # Apply same cap as Rule 2
    
    # Check if ablative mean underperforms baseline by at least 1 standard deviation
    # This indicates ablation actively harms the model's performance
    if ablative_mean < (baseline_mean - effective_std):
        return 'optimally-sized'
    
    # Check if best baseline max outperforms best ablative max
    # This captures cases where baseline modes can rescue better than ablative modes
    # BUT only when ablatives aren't helping on average (to avoid catching chaotic cases)
    baseline_maxes = [m[mode]['max'] for mode in available_baselines]
    ablative_maxes = [m[mode]['max'] for mode in available_ablatives]
    best_baseline_max = max(baseline_maxes)
    best_ablative_max = max(ablative_maxes)
    
    if best_baseline_max > best_ablative_max and ablative_mean <= baseline_mean:
        return 'optimally-sized'
    
    # Also check for consistent baselines + consistent ablations that are harmful
    # This captures cases where ablation is systematically harmful but consistent
    # BUT only if the harm is significant (no overlap and meaningful difference)
    if baselines_within_std and ablations_consistent and ablative_mean < baseline_mean:
        # Check if there's significant overlap or if the difference is too small
        max_ablation = max(ablative_means)
        min_baseline = min(baseline_means)
        overlap = max_ablation - min_baseline
        difference = baseline_mean - ablative_mean
        
        # Only classify as optimally-sized if there's no overlap AND meaningful difference
        if overlap <= 0 and difference > 0.1:  # No overlap and >0.1% difference
            return 'optimally-sized'
    
    # --- Rule 4, Regime III: Chaotic Optimization ---
    # Condition A: Optimizer Rescue (Baselines fail, Ablatives succeed on MAX)
    all_baselines_fail = all(m[mode]['max'] <= validation_zeror for mode in available_baselines)
    any_ablative_succeeds = any(m[mode]['max'] > validation_zeror for mode in available_ablatives)
    if all_baselines_fail and any_ablative_succeeds:
        return 'chaotic-optimization'
    
    # Condition B: Stochastic Overachievement (Ablation outperforms baseline)
    # This is the opposite of optimally-sized: ablation helps rather than hurts
    # Use a more lenient threshold for chaotic detection (0.5% instead of 1 std)
    chaotic_threshold = 0.5  # Fixed 0.5% threshold for chaotic detection
    if ablative_mean > (baseline_mean + chaotic_threshold):
        return 'chaotic-optimization'
    
    print(f"Regime: {arch_key} - #debugging data here")

    return 'unknown' 

def get_all_regime_classifications(configurations_file: Path, trials_file: Path) -> Dict[str, str]:
    """
    Get regime classifications for all configurations.
    
    Args:
        configurations_file: Path to configurations file
        trials_file: Path to trials data file
        
    Returns:
        Dictionary mapping configuration strings to regime classifications
    """
    trial_results = parse_trial_data(trials_file)
    metrics = calculate_summary_metrics(trial_results)
    
    configurations = []
    with open(configurations_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '*' in line:
                configurations.append(line)
    
    classifications = {config: classify_regime(config.replace('*', 'x'), metrics) for config in configurations}
    return classifications


if __name__ == "__main__":
    configs_file = Path('reproduction/configurations.txt')
    trials_file = Path('results/psa_simplemlp_trials.md')
    
    if configs_file.exists() and trials_file.exists():
        classifications = get_all_regime_classifications(configs_file, trials_file)
        
        print("Regime Classifications (Rule-Based):")
        print("=" * 60)
        
        regime_groups = defaultdict(list)
        for config, regime in classifications.items():
            regime_groups[regime].append(config)
            
        # Define the desired order for printing
        regime_order = ['Beneficial Regularization', 'Optimally Sized', 'Chaotic Optimization', 'Architecturally Flawed', 'Unknown']
        
        for regime in regime_order:
            if regime in regime_groups:
                configs = sorted(regime_groups[regime])
                print(f"\n--- {regime} ({len(configs)}) ---")
                # Print in a more compact format, e.g., 4 columns
                for i in range(0, len(configs), 4):
                    print("  ".join(f"{c:<10}" for c in configs[i:i+4]))
    else:
        print("Error: Configuration or trials file not found. Ensure 'reproduction/configurations.txt' and 'results/psa_simplemlp_trials.md' exist.")