# Persistent Stochastic Ablation (PSA) - SimpleMLP

Source code for the mini-paper "Beyond Pruning and Dropout: Evolving Robust Networks via Persistent Stochastic Ablation"

## Prerequisites

### Poetry Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. Install Poetry using one of these methods:

**Official Installer (Recommended):**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Alternative methods:**
```bash
pip install poetry
# or
brew install poetry  # macOS
```

After installation, ensure Poetry is in your PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Project Setup

1. **Clone and setup:**
   ```bash
   git clone git@github.com:tcotten-scrypted/persistent-stochastic-ablation-mlp.git
   cd persistent-stochastic-ablation-mlp
   poetry install
   ```

2. **Activate environment:**
   ```bash
   poetry shell
   # or run commands directly: poetry run train
   ```

## Project Structure

```
├── src/                    # Source code
│   └── train_psa_simplemlp.py
├── scripts/                # Utility and execution scripts
│   ├── clean.py
│   ├── clean_all.py
│   ├── generate_reproduction_tests.py
│   ├── make_table_architectures.py
│   ├── make_table_trial_accuracy.py
│   ├── make_figure_design_space.py
│   └── make_figure_heatmaps.py
├── aws/                    # AWS SageMaker integration
│   └── sagemaker/
│       ├── train.py
│       ├── runner.py
│       ├── results_parser.py
│       ├── current_batch_configurations.txt
│       └── requirements.txt
├── models/                 # User-generated model files
├── dataset/                # Downloaded datasets (MNIST)
├── results/                # Experimental results and figures
├── reproduction/           # Reproduction configurations
│   └── configurations.txt
├── .venv/                  # Poetry virtual environment
├── pyproject.toml          # Poetry configuration and dependencies
├── poetry.lock            # Locked dependencies
├── poetry.toml            # Poetry settings
├── run_reproduction.sh    # Generated reproduction script
├── train.py               # Legacy training script
├── README.md              # This file
├── TOOLS.md               # Utility scripts documentation
├── REPRODUCTION.md        # Reproduction guide
├── LICENSE                # Apache 2.0 license
└── .gitignore            # Git ignore rules
```

## Usage

### Basic Training

Train with default settings:
```bash
poetry run train
```

### Training with Custom Parameters

Train with custom architecture and ablation mode:
```bash
poetry run train -- --arch "[4*4, 2*8]" --ablation-mode hidden --lr 1e-3
```

### Available Parameters

- `--arch`: Define architecture with string, e.g., `"[4*4, 2*8, 1*16]"`
- `--ablation-mode`: Set ablation mode (`none`, `decay`, `dropout`, `full`, `hidden`, `output`)
- `--lr`: Learning rate (default: 1e-4)
- `--batch-size`: Batch size for training (default: 256)
- `--meta-loops`: Total meta-loops to run (default: 100)
- `--model-dir`: Path to store/retrieve models (default: "models/")
- `--debug`: Enable verbose debug logging
- `--num-workers`: DataLoader workers (default: 4)
- `--device`: Override device detection (`cpu`, `cuda`, `mps`)
- `--weight-decay`: Weight decay rate (default: 1e-4, only used with `decay` mode)
- `--dropout`: Dropout rate (default: 0.1, only used with `dropout`)

### Device Detection

The script automatically detects the best available device:
- **CUDA** (highest priority): NVIDIA GPUs
- **Metal** (MPS): Apple Silicon (M1/M2/M3) chips
- **CPU** (fallback): When no GPU acceleration is available

Override with `--device`:
```bash
poetry run train -- --device cpu      # Force CPU
poetry run train -- --device cuda     # Force CUDA
poetry run train -- --device mps      # Force Metal (Apple Silicon)
```

### Examples

**Control experiment (no ablation):**
```bash
poetry run train -- --ablation-mode none
```

**Weight decay baseline:**
```bash
poetry run train -- --ablation-mode decay --weight-decay 1e-4
```

**Dropout baseline:**
```bash
poetry run train -- --ablation-mode dropout --dropout 0.2
```

**Hidden layer ablation:**
```bash
poetry run train -- --ablation-mode hidden --arch "[2*512, 1*256]"
```

**Output layer ablation:**
```bash
poetry run train -- --ablation-mode output --lr 5e-4
```

**Full ablation with custom settings:**
```bash
poetry run train -- --ablation-mode full --batch-size 128 --meta-loops 150 --debug
```

### Resuming Training

The training script automatically resumes from the last known good (LKG) checkpoint if one exists:

**Continue training on the same architecture:**
```bash
# First run
poetry run train -- --arch "[1*1024]" --ablation-mode hidden

# Continue training (resumes from checkpoint)
poetry run train -- --arch "[1*1024]" --ablation-mode hidden

# Switch ablation mode (resumes from same checkpoint)
poetry run train -- --arch "[1*1024]" --ablation-mode output
```

**Important Notes:**
- **Same Architecture**: You can resume training and even switch ablation modes on the same architecture
- **Different Architecture**: If you change `--arch`, you must clean the checkpoint first:
  ```bash
  poetry run clean  # Remove old checkpoint
  poetry run train -- --arch "[2*512]" --ablation-mode hidden  # Start fresh
  ```

### Cleaning Model Files

Remove the default checkpoint:
```bash
poetry run clean
```

Remove all files in the models directory:
```bash
poetry run clean --all
# or
poetry run clean-all
```

Remove specific files:
```bash
poetry run clean file1.safetensors file2.safetensors
```

### Analysis and Visualization Tools

Generate LaTeX tables and figures for publication:

**Dataset Analysis:**
```bash
poetry run analyze-dataset              # MNIST dataset statistics and ZeroR baselines
```

**Architecture Analysis:**
```bash
poetry run make-architecture-table      # Parameter counts and classifications
poetry run make-trial-accuracy-table    # Experimental results with statistics
```

**Visualization:**
```bash
poetry run make-design-space-figure     # Design space scatter plot
poetry run make-figure-heatmaps         # Six comprehensive heatmap visualizations
poetry run make-convergence-plots       # Convergence plots showing validation accuracy over meta-loops (SimpleMLP_Plot_Convergence.png) - use 1*1024,18*18 format
```

**Regime Classification:**
```bash
python -c "from scripts.regime_classifier import get_all_regime_classifications; print(get_all_regime_classifications('reproduction/configurations.txt', 'results/psa_simplemlp_trials.md'))"
```

**Reproduction Tools:**
```bash
poetry run generate-reproduction-tests  # Generate commands for all configurations
```

### AWS SageMaker Integration

Parse and analyze SageMaker training results:
```bash
poetry run sagemaker-results-parser
```

**Requirements:**
- AWS credentials configured
- `.env` file with AWS configuration in `aws/sagemaker/` (copy from `.env.example`)

**Output Files:**
- `results/psa_simplemlp_summary.md` - Statistical summary of all experiments
- `results/psa_simplemlp_trials.md` - Raw trial data in markdown tables

## Documentation

This repository includes comprehensive documentation for different aspects of the project:

### [CHANGELOG.md](CHANGELOG.md) - Project Changes

Detailed changelog tracking all notable changes, enhancements, and additions to the project:

- **Version History**: Chronological record of all significant updates
- **Feature Additions**: New analysis tools, visualization enhancements, and regime classification system
- **Technical Details**: Key insights, performance optimizations, and code quality improvements
- **File Modifications**: Complete list of new, modified, and generated files

### [REPRODUCTION.md](REPRODUCTION.md) - Research Reproduction Guide

A detailed guide for reproducing the experimental results from the mini-paper:

- **Background & Methodology**: Explanation of Persistent Stochastic Ablation (PSA) and the Frustration Engine meta-learning framework
- **Experimental Design**: Details on the six ablation modes (none, decay, dropout, full, hidden, output) and behavioral regimes
- **Reproduction Steps**: Step-by-step instructions for validating all 98 architectural configurations
- **Multi-Trial Validation**: Instructions for running 10 trials per configuration for statistical significance
- **AWS SageMaker Automation**: Documentation of cloud-based automation scripts for large-scale experimentation
- **Expected Results**: Summary of the four behavioral regimes and their characteristics

### [TOOLS.md](TOOLS.md) - Utility Scripts Reference

A comprehensive reference for all utility scripts and tools:

- **Dataset Analysis**: `analyze_dataset.py` - Analyze MNIST dataset using exact PSA methodology, providing class distributions and ZeroR baselines
- **Regime Classification**: `regime_classifier.py` - Rule-based classification system for determining training regimes (Untrainable, Chaotic, Beneficial Regularization, Optimally Sized)
- **Architecture Analysis**: `make_table_architectures.py` - Generate LaTeX tables of network architectures and parameter counts
- **Results Analysis**: `make_table_trial_accuracy.py` - Create publication-ready LaTeX tables from experimental results with statistical information
- **Visualization**: `make_figure_design_space.py` and `make_figure_heatmaps.py` - Generate design space plots and six comprehensive heatmap visualizations
- **Usage Examples**: Both direct Python and Poetry command examples for each tool
- **Output Formats**: Sample LaTeX output and formatting details
- **Features**: Detailed descriptions of each tool's capabilities and options

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

Copyright 2025 Tim Cotten <tcotten@scrypted.ai> <tcotten2@gmu.edu>
