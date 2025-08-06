# Changelog

All notable changes to the PSA SimpleMLP project are documented in this file.

## [Unreleased] - 2024-12-19

### Added

#### New Analysis Tools
- **`analyze_dataset.py`**: Comprehensive MNIST dataset analysis tool
  - Uses exact same methodology as PSA SimpleMLP training code
  - Provides class distribution statistics for all dataset splits
  - Calculates ZeroR baselines for each split (11.24% train, 11.35% test, 11.28% split-train, 11.02% validation)
  - Available as `poetry run analyze-dataset` command
  - Returns structured `MNISTAnalysis` object for programmatic access

- **`scripts/regime_classifier.py`**: Rule-based training regime classification system
  - Data-driven classification using actual trial results
  - Four distinct training regimes: Untrainable, Chaotic Optimization, Beneficial Regularization, Optimally Sized
  - Robust classification rules with 100% coverage (no "unknown" cases)
  - Applies 0.5% minimum standard deviation cap to prevent overly strict conditions
  - Uses 11.35% threshold corresponding to ZeroR baseline from test set

#### Enhanced Visualizations
- **Sixth Heatmap**: "Regimes" visualization added to `make_figure_heatmaps.py`
  - Uses regime classifier for data-driven regime assignment
  - Four distinct colors: black (untrainable), red (optimally-sized), green (beneficial-regularization), blue (chaotic-optimization)
  - Uniform circle sizes with increased label offsets for clarity
  - Copies advanced labeling system from Parameter Matching heatmap

#### Classification Rules
- **Rule 1 (Untrainable)**: No mode exceeds 11.35% accuracy (Vanishing Gradient Problem)
- **Rule 2 (Beneficial Regularization)**: Baselines consistent AND (ablations within baseline std OR meaningful overlap >0.01%)
- **Rule 3 (Optimally Sized)**: Ablation harms by >1 effective std OR (baseline max > ablative max AND ablative mean ≤ baseline mean)
- **Rule 4 (Chaotic Optimization)**: Ablative mean > baseline mean + 0.5% (fixed threshold for better sensitivity)

### Enhanced

#### Visualization Improvements
- **Baseline Performance Heatmap**: Added colorbar for baseline accuracy visualization
- **Winning Strategy Heatmap**: Updated `mode_colors` dictionary to assign distinct colors to all six ablation modes
- **Parameter Matching Heatmap**: Refined Bezier curve control points and circle radius calculations for logarithmic axes
- **Design Space Plot**: Modified plotting order so square architectures appear on top of non-square ones

#### Advanced Labeling System
- **Dynamic Positioning**: Bezier curves for non-overlapping architecture labels
- **Logarithmic Scaling**: Proper point-based sizing accounting for logarithmic axes
- **Three Label Categories**: 
  - Shallow-and-wide: Curved labels above points
  - Square: Straight offset labels to the right
  - Deep-and-narrow: Curved labels below points

#### Data Processing
- **Six Ablation Modes**: Updated trial data parsing to handle all six modes (none, decay, dropout, full, hidden, output)
- **Enhanced Metrics**: Comprehensive statistics calculation for all modes
- **Robust Error Handling**: Graceful handling of missing data and parsing errors

### Changed

#### Regime Classification Methodology
- **From Heuristic to Data-Driven**: Replaced parameter-count-based heuristics with performance-pattern analysis
- **Refined Thresholds**: 
  - Fixed 0.5% threshold for chaotic detection (more sensitive than variable std)
  - 0.01% overlap threshold for beneficial regularization
  - 0.1% difference threshold for optimally-sized classification
- **Max-Based Rules**: Added rules comparing best baseline vs best ablative performance

#### Documentation Updates
- **README.md**: Added dataset analysis and regime classification sections
- **TOOLS.md**: Comprehensive documentation for new tools and enhanced features
- **CHANGELOG.md**: Created detailed changelog for tracking all changes

#### Poetry Configuration
- **New Script**: Added `analyze-dataset = "analyze_dataset:main"` to `pyproject.toml`
- **Dependencies**: Added `shapely = "<2.0"` for geometric operations (later removed when regimes visualization was simplified)

### Technical Details

#### Key Insights from Analysis
- **ZeroR Correspondence**: 11.35% untrainable threshold aligns with test set ZeroR baseline
- **Class 1 Dominance**: Most frequent class across all MNIST splits
- **Regime Distribution**: Final classification achieves:
  - Beneficial Regularization: 13 architectures (13%)
  - Optimally Sized: 23 architectures (23%)
  - Untrainable: 45 architectures (46%)
  - Chaotic Optimization: 17 architectures (17%)

#### Performance Optimizations
- **Memory Management**: Proper cleanup of matplotlib resources in all visualization scripts
- **Batch Processing**: Efficient handling of large datasets with robust error handling
- **Statistical Robustness**: Effective standard deviation capping prevents edge cases

#### Code Quality
- **Modular Design**: Clean separation between data processing, classification, and visualization
- **Type Hints**: Comprehensive type annotations for better code maintainability
- **Error Handling**: Graceful degradation with informative error messages
- **Documentation**: Extensive docstrings and inline comments

### Files Modified

#### New Files
- `analyze_dataset.py` - MNIST dataset analysis utility
- `scripts/regime_classifier.py` - Training regime classification module
- `CHANGELOG.md` - This changelog file

#### Modified Files
- `scripts/make_figure_heatmaps.py` - Added sixth heatmap and enhanced existing visualizations
- `scripts/make_figure_design_space.py` - Improved shape layering for better visibility
- `TOOLS.md` - Added documentation for new analysis tools
- `README.md` - Updated with new tools and enhanced feature descriptions
- `pyproject.toml` - Added new script entry point

#### Generated/Updated Files
- `results/SimpleMLP_Heatmap_Regimes.png` - New regime classification heatmap
- All existing heatmap PNG files - Regenerated with enhanced visualizations
- `results/psa_simplemlp_summary.md` - Updated with latest trial results
- `results/psa_simplemlp_trials.md` - Updated with comprehensive trial data

### Notes

This release represents a significant enhancement to the PSA SimpleMLP analysis toolkit, transitioning from heuristic-based classification to a comprehensive, data-driven approach. The new regime classification system provides robust, reproducible categorization of training behaviors, while the enhanced visualization suite offers deeper insights into the experimental results.

The addition of dataset analysis tools ensures consistency between experimental setup and analysis methodology, providing important baseline references for interpreting results.