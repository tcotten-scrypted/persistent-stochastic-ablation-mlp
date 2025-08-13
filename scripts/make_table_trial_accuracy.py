#!/usr/bin/env python3
"""
Generate LaTeX table from trial accuracy results data with statistical information.

This script reads results from results/psa_simplemlp_summary.md and generates
a LaTeX table showing mean accuracy with standard deviation for each architecture
and ablation mode. It dynamically detects the number of trials from the results.

Author: Tim Cotten @cottenio <tcotten@scrypted.ai, tcotten2@gmu.edu>
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path


def parse_results_file(filepath):
    """Parses the Markdown file with statistical results."""
    with open(filepath, 'r') as f:
        content = f.read()

    results = defaultdict(dict)
    
    # Regex to capture the architecture and the block of statistical data
    pattern = re.compile(r"Results Shape \{([\d\*]+)\} Parameters \{[\d,]+\}\n(.*?)\n\n", re.DOTALL)
    
    # Regex to parse each line of statistical data (Mean, Std, Min, Max)
    line_pattern = re.compile(r"\* (None|Decay|Dropout|Full|Hidden|Output): Mean=([\d\.]+)% \| Std=([\d\.]+)% \| Min=([\d\.]+)% \| Max=([\d\.]+)% \(n=\d+\)")

    for match in pattern.finditer(content):
        arch, data_block = match.groups()
        arch = arch.replace('*', 'x') # Standardize internally

        for line in data_block.strip().split('\n'):
            line_match = line_pattern.match(line.strip())
            if line_match:
                mode, mean, std, v_min, v_max = line_match.groups()
                results[arch][mode.lower()] = {
                    'mean': float(mean),
                    'std': float(std),
                    'min': float(v_min),
                    'max': float(v_max)
                }
    return results


def generate_latex_table(results, order, caption, label):
    """Generates the LaTeX code for a professional-looking table."""
    header = r"""
% \setcounter{LTchunksize}{50}
\tiny 
\begin{longtable}{@{}lccccccl@{}}
\caption{Mean of Validation Peak Accuracy (\%) with Standard Deviation over 10 Trials of 100 Meta-Loops for SimpleMLP Architectures}
\label{tab:results_summary_stats} \\
\toprule
\textbf{Architecture} & \textbf{none} & \textbf{decay} & \textbf{dropout} & \textbf{full} & \textbf{hidden} & \textbf{output} & \textbf{Winner} \\
 & Mean $\pm$ Std (\%) & Mean $\pm$ Std (\%) & Mean $\pm$ Std (\%) & Mean $\pm$ Std (\%) & Mean $\pm$ Std (\%) & Mean $\pm$ Std (\%) & \\
\midrule
\endfirsthead

\multicolumn{8}{c}%
{{\bfseries Table \thetable\ continued from previous page}} \\
\toprule
\textbf{Architecture} & \textbf{none} & \textbf{decay} & \textbf{dropout} & \textbf{full} & \textbf{hidden} & \textbf{output} & \textbf{Winner} \\
 & Mean $\pm$ Std (\%) & Mean $\pm$ Std (\%) & Mean $\pm$ Std (\%) & Mean $\pm$ Std (\%) & Mean $\pm$ Std (\%) & Mean $\pm$ Std (\%) & \\
\midrule
\endhead

\midrule
\multicolumn{8}{r}{{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot
"""
    
    print(header)

    for arch in order:
        if arch in results:
            modes = ['none', 'decay', 'dropout', 'full', 'hidden', 'output']
            arch_data = results[arch]
            
            mean_values = [arch_data.get(mode, {'mean': -1})['mean'] for mode in modes]
            max_mean = max(mean_values) if mean_values else -1

            tolerance = 0.01
            winner_modes = []
            for i, mode in enumerate(modes):
                if max_mean - mean_values[i] <= tolerance:
                    winner_modes.append(f"\\texttt{{{mode}}}")

            winner_str = "/".join(winner_modes)
            if not winner_str:
                winner_str = "N/A"
            elif len(winner_modes) > 1:
                winner_str = f"Tie: {winner_str}"

            cells = []
            for i, mode in enumerate(modes):
                if mode in arch_data:
                    mean = arch_data[mode]['mean']
                    std = arch_data[mode]['std']
                    # Wrap the mathematical expression in $...$
                    cell_str = f"${mean:.2f} \\pm {std:.2f}$"
                    if max_mean - mean <= tolerance:
                        cell_str = f"\\bfseries {cell_str}"
                    cells.append(cell_str)
                else:
                    cells.append("N/A")
            
            arch_display = arch.replace('x', '*')
            print(f"${arch_display}$ & {' & '.join(cells)} & {winner_str} \\\\")

    footer = r"""\end{longtable}"""

    print(footer)


def main():
    """Generate LaTeX table from results data."""
    parser = argparse.ArgumentParser(description="Generate LaTeX table from statistical results data.")
    parser.add_argument('--datafile', type=str, default='results/psa_simplemlp_summary.md',
                       help="Path to results summary file (Markdown)")
    parser.add_argument('--order', type=str, default='reproduction/configurations.txt',
                       help="Path to file with desired architecture order.")
    parser.add_argument('--caption', type=str, 
                       default="Mean Peak Accuracy (\\%) with Standard Deviation over 10 Trials for SimpleMLP Architectures", 
                       help="Table caption.")
    parser.add_argument('--label', type=str, default="results_summary_stats", 
                       help="LaTeX label for the table.")
    args = parser.parse_args()

    # Check if data file exists
    if not Path(args.datafile).exists():
        print(f"Error: Results file '{args.datafile}' not found.")
        print("Please ensure the file exists. You may need to run 'poetry run sagemaker-results-parser' first.")
        return

    # Check if order file exists
    if not Path(args.order).exists():
        print(f"Error: Order file '{args.order}' not found.")
        return

    try:
        with open(args.order, 'r') as f:
            desired_order = [line.strip().replace('*', 'x') for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: Order file not found at '{args.order}'. Please provide a valid path.")
        return

    results_data = parse_results_file(args.datafile)
    generate_latex_table(results_data, desired_order, args.caption, args.label)


if __name__ == "__main__":
    main() 
