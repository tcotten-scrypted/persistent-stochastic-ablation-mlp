# Utility Tools

This document describes the utility scripts available in the `scripts/` folder for various analysis and automation tasks.

## Dataset Analysis

### `analyze_dataset.py`

Analyze MNIST dataset using the exact same methodology as PSA SimpleMLP training.

**Purpose**: Provides comprehensive statistics about MNIST class distribution and ZeroR baselines using the identical data loading and splitting methodology as the training code, ensuring consistency between analysis and experimental results.

**Usage**:
```bash
# Direct Python
python analyze_dataset.py

# Poetry command
poetry run analyze-dataset
```

**Output**: Rich console tables showing:
- **Class Distribution**: Count and percentage for each digit (0-9) in each dataset split
- **ZeroR Baselines**: Most frequent class and its accuracy for each split
- **Summary Comparison**: Side-by-side comparison of all dataset splits

**Features**:
- **Exact Methodology**: Uses identical data loading, transforms, and splitting as `train_psa_simplemlp.py`
- **Reproducible Splits**: Same random seed (1337) for train/validation splitting
- **Structured Output**: Returns `MNISTAnalysis` object for programmatic access
- **ZeroR Calculation**: Baseline accuracy using most frequent class strategy

**Dataset Splits Analyzed**:
- **Full Training (60k)**: Original MNIST training set
- **Full Test (10k)**: Original MNIST test set  
- **Split Training (50k)**: Training portion after validation split
- **Split Validation (10k)**: Validation portion for model selection

**Key Findings**:
- Class 1 is consistently the most frequent across all splits
- ZeroR baselines: ~11.24% (full train), ~11.35% (test), ~11.28% (split train), ~11.02% (validation)
- Class distribution is well-balanced with slight variations

## Regime Classification

### `regime_classifier.py`

Rule-based classification system for determining training regimes based on PSA trial data.

**Purpose**: Provides a clean, modular interface for classifying SimpleMLP configurations into four distinct training regimes based on performance patterns observed in comprehensive trial data. Uses data-driven rules to categorize architectures by their response to different ablation strategies.

**Usage**:
```python
from scripts.regime_classifier import classify_regime, get_all_regime_classifications
from pathlib import Path

# Classify a single architecture
metrics = calculate_summary_metrics(trial_results)
regime = classify_regime('1x512', metrics)
print(f"1x512 is: {regime}")

# Classify all architectures
classifications = get_all_regime_classifications(
    Path('reproduction/configurations.txt'), 
    Path('results/psa_simplemlp_trials.md')
)
```

**Training Regimes**:
- **🔴 Untrainable**: Vanishing Gradient Problem - no mode exceeds validation ZeroR baseline (11.02%)
- **🔵 Chaotic Optimization**: Baseline modes ineffective (≤11.02%), but ablative modes show learning (>11.02%)
- **🟢 Beneficial Regularization**: High-performing models where regularizers often help, performance gaps are small
- **🟠 Optimally Sized**: High-performing models where baseline is best, any intervention is detrimental

**Classification Rules**:
1. **Rule 1 (Untrainable)**: `max_performance_across_all_modes <= validation_zeror` (11.02%)
2. **Rule 2 (Beneficial Regularization)**: Baselines consistent AND (ablations within baseline std OR meaningful overlap exists)
3. **Rule 3 (Optimally Sized)**: Ablation harms performance by >1 std OR (baseline max > ablative max AND ablative mean ≤ baseline mean)
4. **Rule 4 (Chaotic Optimization)**: Ablative mean > baseline mean + 0.5% (fixed threshold for sensitivity)

**Features**:
- **Data-Driven**: Uses actual trial results rather than heuristics
- **Robust Thresholds**: Applies 0.5% minimum standard deviation cap to prevent overly strict conditions
- **Comprehensive Coverage**: Achieves 100% classification rate with refined rules
- **Validation**: Includes detailed debugging output for rule verification

**Key Insights**:
- 11.02% threshold corresponds to ZeroR baseline (most frequent class in validation set)
- Fixed 0.5% threshold for chaotic detection provides better sensitivity than variable std
- Overlap analysis distinguishes beneficial regularization from optimal sizing
- Max-based rules capture baseline rescue superiority patterns

## Architecture Analysis

### `make_table_architectures.py`

Generate LaTeX tables showing parameter counts for network architectures.

**Purpose**: Creates publication-ready LaTeX tables from the experimental configurations, categorizing architectures as "Matched" or "Asymmetric" based on the paper's classification.

**Usage**:
```bash
# Direct Python
python scripts/make_table_architectures.py

# Poetry command
poetry run make-architecture-table
```

**Output**: LaTeX table with columns:
- **Architecture**: The network configuration (e.g., "1*2048")
- **Parameters**: Total parameter count with comma formatting
- **Parameter Matching**: "Matched" or "Asymmetric" classification

**Input**: Reads from `reproduction/configurations.txt`

**Example Output**:
```latex
\begin{table}[ht]
\centering
\caption{Parameters for Network Architectures}
\label{tab:architecture_parameters}
\begin{tabular}{lcl}
\toprule
\textbf{Architecture} & \textbf{Parameters} & \textbf{Parameter Matching}\\
\midrule
1*2048      & 1,628,170       & Matched \\
2*939       & 1,629,175       & Matched \\
939*2       & 1,629,175       & Asymmetric \\
...
\bottomrule
\end{tabular}
\end{table}
```

**Features**:
- Automatic parameter counting using the same formula as the training script
- Categorization based on the paper's asymmetric configuration list
- Proper LaTeX formatting with booktabs-style tables
- Error handling for missing configuration files

## Results Analysis

### `make_table_trial_accuracy.py`

Generate LaTeX tables from trial accuracy results with statistical information.

**Purpose**: Creates publication-ready LaTeX tables from experimental results, showing mean accuracy with standard deviation for each architecture and ablation mode. Includes sophisticated winner detection with tolerance for statistical ties.

**Usage**:
```bash
# Direct Python
python scripts/make_table_trial_accuracy.py

# Poetry command
poetry run make-trial-accuracy-table

# With custom files and options
python scripts/make_table_trial_accuracy.py --datafile results/my_results.md --order reproduction/my_configs.txt --caption "Custom Caption" --label "custom_label"
```

**Output**: LaTeX table with columns:
- **Architecture**: The network configuration (e.g., "1*2048")
- **`none`**: Mean ± Std accuracy with no ablation
- **`full`**: Mean ± Std accuracy with full ablation
- **`hidden`**: Mean ± Std accuracy with hidden layer ablation
- **`output`**: Mean ± Std accuracy with output layer ablation
- **Winner**: The ablation mode(s) with highest accuracy (handles ties)

**Input**: 
- Reads from `results/psa_simplemlp_summary.md` (default)
- Uses `reproduction/configurations.txt` for row ordering

**Example Output**:
```latex
\begin{table}[ht]
\centering
\caption{Mean Peak Accuracy (\%) with Standard Deviation over 10 Trials for SimpleMLP Architectures}
\label{tab:results_summary_stats}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l l l l l l}
\toprule
\textbf{Architecture} & {\textbf{`none`}} & {\textbf{`full`}} & {\textbf{`hidden`}} & {\textbf{`output`}} & \textbf{Winner} \\
 & {Mean $\pm$ Std (\%)} & {Mean $\pm$ Std (\%)} & {Mean $\pm$ Std (\%)} & {Mean $\pm$ Std (\%)} & \\
\midrule
1*2048 & \bfseries $98.33 \pm 0.04$ & $98.30 \pm 0.04$ & \bfseries $98.33 \pm 0.05$ & $98.17 \pm 0.07$ & Tie: \texttt{none}/\texttt{hidden} \\
1*1024 & \bfseries $98.26 \pm 0.05$ & $98.19 \pm 0.07$ & $98.25 \pm 0.07$ & $98.03 \pm 0.08$ & \texttt{none} \\
...
\bottomrule
\end{tabular}
}
\end{table}
```

**Features**:
- **Statistical Information**: Shows mean ± standard deviation for all results
- **Tie Detection**: Identifies statistical ties within 0.01% tolerance
- **Professional Formatting**: Uses `\resizebox` for table fitting and mathematical notation
- **Winner Highlighting**: Bold formatting for winning modes
- **Configurable Options**: Custom captions, labels, and file paths
- **Error Handling**: Graceful handling of missing files
- **Maintains Ordering**: Uses configuration file for consistent row ordering

## Visualization

### `make_figure_design_space.py`

Generate design space visualization for SimpleMLP architectures.

**Purpose**: Creates a logarithmic scatter plot showing the design space of all tested architectures, visualizing the relationship between network depth and width.

**Usage**:
```bash
# Direct Python
python scripts/make_figure_design_space.py

# Poetry command
poetry run make-design-space-figure

# With custom files
python scripts/make_figure_design_space.py --config-file reproduction/my_configs.txt --output-file results/my_plot.png
```

**Output**: High-resolution PNG file showing:
- **Logarithmic Scale**: Both depth and width axes use log scale for better visualization
- **Architecture Points**: Each tested architecture plotted as a rectangle
- **Labels**: Architecture notation (e.g., "1×2048") displayed for each point
- **Grid Lines**: Reference grid for easier reading of values
- **Professional Formatting**: Publication-ready figure with proper labels and title

**Input**: 
- Reads from `reproduction/configurations.txt` (default)
- Outputs to `results/SimpleMLP_Testing_Design_Space.png` (default)

**Features**:
- **Dynamic Configuration Loading**: Reads actual experimental configurations
- **High-Resolution Output**: 300 DPI PNG suitable for publications
- **Logarithmic Visualization**: Optimal for wide-ranging architectural parameters
- **Error Handling**: Graceful handling of missing files and parsing errors
- **Configurable Paths**: Custom input and output file locations
- **Memory Management**: Proper cleanup of matplotlib resources

### `make_figure_heatmaps.py`

Generate comprehensive heat map visualizations for SimpleMLP design space.

**Purpose**: Creates a suite of six data-driven heat map visualizations that tell a comprehensive story about the SimpleMLP design space under Persistent Stochastic Ablation (PSA).

**Usage**:
```bash
# Direct Python
python scripts/make_figure_heatmaps.py

# Poetry command
poetry run make-figure-heatmaps

# With custom files
python scripts/make_figure_heatmaps.py --config-file reproduction/my_configs.txt --trials-file results/my_trials.md --output-dir results/
```

**Output**: Six high-resolution PNG files:
1. **Baseline Performance**: Raw performance of control models with colorbar (viridis colormap)
2. **Ablation Effects**: Quantitative benefit/harm of PSA (green=beneficial, red=harmful, brown=neutral)
3. **Instability**: Max standard deviation across six ablation modes (magma colormap)
4. **Winning Strategy**: Best-performing ablation mode per architecture (categorical with distinct colors)
5. **Parameter Matching**: Architectural complexity visualization with dynamic labeling system
6. **Regimes**: Training regime classification using rule-based system (four regime colors)

**Input**: 
- Reads from `reproduction/configurations.txt` (default)
- Reads from `results/psa_simplemlp_trials.md` (default)
- Outputs to `results/` directory (default)

**Features**:
- **Comprehensive Analysis**: Six complementary visualizations for complete understanding
- **Data-Driven**: Based on actual experimental trial data and regime classification
- **Statistical Rigor**: Uses mean and standard deviation calculations with robust thresholds
- **Advanced Labeling**: Dynamic Bezier curve positioning for non-overlapping architecture labels
- **Professional Visualization**: High-resolution (300 DPI) publication-ready figures
- **Logarithmic Scale**: Optimal for wide-ranging architectural parameters
- **Regime Integration**: Uses `regime_classifier.py` for data-driven regime classification
- **Memory Efficient**: Proper cleanup of matplotlib resources
- **Configurable**: Custom input files and output directory
- **Error Handling**: Graceful handling of missing files and parsing errors 