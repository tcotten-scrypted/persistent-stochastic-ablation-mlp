# Reproduction Guide: Persistent Stochastic Ablation (PSA) Mini-Paper

This document provides a comprehensive guide for reproducing the experimental results from the mini-paper "Beyond Pruning and Dropout: Evolving Robust Networks via Persistent Stochastic Ablation."

## Background

### Research Question
This study investigates whether **Persistent Stochastic Ablation (PSA)** can act as beneficial "evolutionary pressure" on neural networks by introducing permanent, blind damage to a network's best-performing state between training cycles. The hypothesis is that this "frustration" can drive networks toward more robust and generalizable solutions.

### Key Concepts

**Persistent Stochastic Ablation (PSA):** A novel training paradigm that mimics environmental evolutionary pressure by permanently ablating neurons in the network's Last Known Good (LKG) state between training cycles.

**The Frustration Engine:** A meta-learning framework that implements PSA through iterative cycles of:
1. Start with Last Known Good (LKG) model state
2. Create a fresh copy of LKG state
3. For ablative modes (full/hidden/output), apply ONE ablation to the copy
4. Train the copy for one epoch
5. Evaluate on validation set
6. If improved, update LKG state (unablated version)
7. Repeat with fresh copy of LKG

Each meta-loop resets the optimizer to ensure fair comparison between ablation strategies.

**Six Ablation Modes:**
- **`none`**: Control group (no regularization)
- **`decay`**: Traditional weight decay regularization
- **`dropout`**: Traditional dropout regularization
- **`full`**: Partially ablates a neuron in ANY linear layer (hidden or output)
- **`hidden`**: Fully ablates a neuron in a HIDDEN layer only
- **`output`**: Partially ablates a neuron in the OUTPUT layer only

## Methodology

### SimpleMLP Architecture
The study uses a flexible Multi-Layer Perceptron (MLP) architecture with:
- Fully-connected linear layers with ReLU activations
- Dynamic architecture specification via `--arch` parameter
- Input: 784 dimensions (MNIST flattened)
- Output: 10 dimensions (MNIST digits 0-9)

**Architecture Format:** `[L*W]` where:
- `L` = number of layers
- `W` = width (neurons per layer)
- Example: `[2*939]` = 2 layers of 939 neurons each

### Experimental Design
The study explores 98 unique homogeneous architectures across three categories:
1. **Shallow-and-wide**: `1*W` and `2*W` configurations
2. **Deeper-and-square**: `L*L` configurations  
3. **Deep-and-narrow**: `L*2` and `L*1` configurations

Each architecture was tested with all six ablation modes over 10 independent trials with different random seeds.

### Four Behavioral Regimes Identified

1. **Beneficial Regularization**: Large, over-parameterized models where ablation improves performance
2. **Detrimental Damage**: Well-sized models where ablation harms performance
3. **Architectural Failure**: Deep models that fail to train due to vanishing gradients
4. **Chaotic Optimization**: Small/pathological models where ablation acts as random search

## Reproduction Steps

### Prerequisites
- Python 3.9+
- Poetry package manager
- CUDA-compatible GPU or Metal (optional, supports CPU fallback)

### Setup
```bash
# Clone and setup
git clone git@github.com:tcotten-scrypted/persistent-stochastic-ablation-mlp.git
cd persistent-stochastic-ablation-mlp
poetry install
```

### Validation Configurations

The `reproduction/configurations.txt` file contains all 98 architectural configurations validated in the paper. This file will be used to systematically validate each configuration with all ablation modes.

**Configuration Categories:**
- **Shallow-and-wide**: Lines 1-22 (`1*2048` through `2*4`)
- **Deeper-and-square**: Lines 23-39 (`115*115` through `*1`)  
- **Deep-and-narrow**: Lines 40-98 (`939*2` through `2*1`)

### Step 1: Validate Individual Configurations

For each architecture in `configurations.txt`, run all six ablation modes:

```bash
# Example: Validate 1*2048 architecture
poetry run train -- --arch "[1*2048]" --ablation-mode none --meta-loops 100
poetry run train -- --arch "[1*2048]" --ablation-mode decay --weight-decay 1e-4 --meta-loops 100
poetry run train -- --arch "[1*2048]" --ablation-mode dropout --dropout 0.1 --meta-loops 100
poetry run train -- --arch "[1*2048]" --ablation-mode full --meta-loops 100
poetry run train -- --arch "[1*2048]" --ablation-mode hidden --meta-loops 100
poetry run train -- --arch "[1*2048]" --ablation-mode output --meta-loops 100
```

### Step 2: Multi-Trial Validation

For statistical significance, each configuration should be run 10 times. The training harness uses random hardware seeds for each trial:

```bash
# Example: 10 trials for 1*2048 with 'none' ablation
for _ in {1..10}; do
    poetry run train -- --arch "[1*2048]" --ablation-mode none --meta-loops 100
done
```

### Step 3: AWS SageMaker Automation (Optional)

For large-scale experimentation, we used AWS SageMaker to automate the running of multiple trials. The scripts in the `aws/sagemaker/` folder provide a primitive but effective way to run 10 trials per configuration using spot or on-demand instances.

**Available Scripts:**
- `runner.py` - Main orchestration script for launching SageMaker training jobs
- `train.py` - Training script adapted for SageMaker environment
- `current_batch_configurations.txt` - Configuration list (copy from `reproduction/configurations.txt`)
- `requested-jobs.txt` - Generated file containing job IDs for CloudWatch log access

**Usage:**
1. Copy desired configurations from `reproduction/configurations.txt` to `aws/sagemaker/current_batch_configurations.txt`
2. Copy `aws/sagemaker/.env.example` to `aws/sagemaker/.env` and configure your AWS settings
3. Run the automation: `python aws/sagemaker/runner.py`

**Important Notes:**
- **Primitive Implementation**: These scripts are basic and not very fault-tolerant
- **Manual Cleanup**: S3 bucket cleanup between full runs is currently manual
- **Customization Required**: Configure S3 bucket, instance type, and spot instances in `.env`
- **Cost Considerations**: Set `USE_SPOT_INSTANCES=true` for cost efficiency
- **Results Parsing**: Use `poetry run sagemaker-results-parser` to analyze results
- **Job Tracking**: Job IDs are saved to `aws/sagemaker/requested-jobs.txt` for easy CloudWatch log access

**Advantages:**
- **Automated Scaling**: Run multiple trials in parallel
- **Cost Effective**: Spot instances reduce compute costs
- **Reproducible**: Consistent environment across all trials
- **Batch Processing**: Process multiple configurations automatically

**Limitations:**
- **Manual Bucket Management**: Requires manual S3 cleanup between runs
- **Basic Error Handling**: Limited fault tolerance for job failures
- **Configuration Overhead**: Requires AWS setup and credential management

**⚠️ CRITICAL WARNING: Configuration Conflicts**
- **Rerunning Configurations**: The runner script is MEANT to check for existing completed jobs and skip them by default, but likely DOESN'T due to a bug. Use with caution!
- **Data Overwrite Risk**: Using `--force-rerun` will launch new jobs that may overwrite existing results
- **S3 Bucket Conflicts**: Multiple jobs with the same configuration can overwrite each other's results
- **Safe Testing**: Use unique configurations (like `7*3`, `9*1`) for testing to avoid conflicts with real data
- **Manual Verification**: Always verify existing results before rerunning configurations
- **Bug Warning**: The job status checking logic may not work correctly - assume it will launch duplicate jobs

### Expected Results

**Regime 1 - Beneficial Regularization:**
- Large models (e.g., `1*2048`, `2*939`) should show ablation benefits
- `full` or `hidden` show potential to improve on `none`
- `output` is generally harmful here

**Regime 2 - Detrimental Damage:**
- Medium models (e.g., `1*512`, `2*115`) should show ablation harms
- `none` mode should consistently win

**Regime 3 - Architectural Failure:**
- Deep models (e.g., `115*115`, `91*91`) should fail to train
- All modes should achieve ~11.02% accuracy (validation ZeroR baseline)

**Regime 4 - Chaotic Optimization:**
- Small models (e.g., `8*8`, `2*1`) should show unpredictable results
- `output` mode often performs best in this regime

### Key Metrics to Validate

1. **Peak Accuracy**: Highest accuracy achieved during training
2. **Statistical Significance**: Compare means across 10 trials
3. **Regime Classification**: Verify each architecture falls into expected regime
4. **Ablation Mode Hierarchy**: Confirm relative performance of ablation modes

### Validation Checklist

- [ ] All 98 configurations from `configurations.txt` tested
- [ ] Each configuration tested with all 6 ablation modes
- [ ] 10 independent trials per configuration-mode combination
- [ ] Results match expected regime classifications
- [ ] Statistical significance calculations performed
- [ ] Peak accuracy values within expected ranges

### Troubleshooting

**Common Issues:**
- **Memory Errors**: Reduce batch size for large architectures
- **Slow Training**: Use GPU acceleration when available
- **Checkpoint Conflicts**: Use `poetry run clean` between different architectures or specify different paths with --model-dir

**Device Selection:**
```bash
# Force specific device if needed
poetry run train -- --device cuda    # NVIDIA GPU
poetry run train -- --device mps     # Apple Silicon
poetry run train -- --device cpu     # CPU fallback
```

## Expected Outcomes

### Statistical Validation
The reproduction should confirm:
- No statistically significant benefits in over-parameterized regime over `none`, but isolated improvements for `full` and `hidden` when `none` becomes stuck
- Clear detrimental effects in well-sized regime  
- Universal failure in deep architectures
- Chaotic but sometimes beneficial effects in small models

### Key Findings to Replicate
1. **Regime Boundaries**: Verify the transition points between regimes
2. **Ablation Mode Effectiveness**: Confirm relative performance of `none`, `full`, `hidden`, `output`
3. **Vanishing Gradient Problem**: Validate the hard boundary at ~20 layers
4. **Parameter Matching**: Verify that matched-parameter architectures show similar trends

### Success Criteria
- [ ] All 98 configurations reproduce within ±2% of published results
- [ ] Regime classifications match published findings
- [ ] Statistical significance aligns with paper conclusions
- [ ] Ablation mode hierarchy confirmed across all valid architectures

## Future Work

This reproduction establishes the foundation for:
1. **ResNet-based experiments** to overcome vanishing gradient limitations
2. **Multi-dataset validation** beyond MNIST
3. **Advanced ablation strategies** (percentage-based, targeted ablation)
4. **Patience mechanisms** in the Frustration Engine

## References

For complete methodology and results, refer to the original mini-paper: "Beyond Pruning and Dropout: Evolving Robust Networks via Persistent Stochastic Ablation" by Tim Cotten. 