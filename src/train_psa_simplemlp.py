# train_psa_simplemlp.py
#
# Author: Tim Cotten @cottenio <tcotten@scrypted.ai, tcotten2@gmu.edu> 
#
# Description:
# A comprehensive training harness for testing architectural variations and four
# distinct live ablation strategies on the MNIST dataset.
#
# Core Concepts:
# 1. Six Ablation Modes:
#    - 'none': Control group.
#    - 'full': Partially ablates a neuron in ANY linear layer (hidden or output).
#    - 'hidden': Fully ablates a neuron in a HIDDEN layer only.
#    - 'output': Partially ablates a neuron in the OUTPUT layer only.
#
# 2. Each meta-loop starts with a fresh copy of the Last Known Good (LKG) model.
# This copy is ablated once (for ablative modes) to create the active model state
# for the current loop. The LKG state itself is only updated when the validation
# accuracy improves, preserving the best model found so far. A "bounty" tracks
# the highest validation accuracy achieved across all meta-loops.
#
# 3. Ablation strategy is always ONE neuron randomly selected from the
# LKG model at the start of each meta-loop. For the 'full' mode this means randomly
# selecting a layer first (including the output layer), then a random neuron from
# within the layer. For the 'hidden' mode this means randomly selecting from
# a list of all available neurons from the hidden layers only. Each meta-loop
# starts fresh with a single ablation - ablations do not accumulate across loops.
#
# 4. Each meta-loop resets the optimizer to ensure fair comparison between
# ablation strategies, preventing momentum or adaptive state from carrying over
# between loops.
#
# 5. Dynamic Architecture: Use --arch "[4*4, 2*8]" to define complex models.
# 6. Frustration Engine: The training orchestration driving the experiments.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import argparse
import logging
from pathlib import Path
import random
from dataclasses import dataclass
import os
import copy
import re
import platform

# --- Dependency Imports ---
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler
from rich.progress import (
    Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn,
)
from rich.table import Table
from safetensors.torch import save_file, load_file

# --- 1. Configuration & Argument Parsing ---

def detect_best_device() -> str:
    """Detect the best available device, prioritizing CUDA > Metal > CPU."""
    # Check for CUDA first (highest priority)
    if torch.cuda.is_available():
        return "cuda"
    
    # Check for Metal (Apple Silicon)
    if platform.system() == "Darwin":  # macOS
        try:
            # Check if MPS (Metal Performance Shaders) is available
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except (AttributeError, RuntimeError):
            pass
    
    # Fallback to CPU
    return "cpu"

@dataclass
class Config:
    """Configuration class for all hyperparameters and settings."""
    MODEL_DIR: str = "models/"
    CHECKPOINT_NAME: str = "mnist_lkg.safetensors"
    INPUT_SIZE: int = 28 * 28
    HIDDEN_LAYERS: list[int] = None
    ARCH_STRING: str = ""
    OUTPUT_SIZE: int = 10
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 256
    NUM_META_LOOPS: int = 100
    DEVICE: str = detect_best_device()
    ABLATION_MODE: str = "none"
    LOG_INTERVAL: int = 20
    DEBUG: bool = False
    NUM_WORKERS: int = 4
    WEIGHT_DECAY: float = 1e-4  # Default weight decay rate
    DROPOUT_RATE: float = 0.1   # Default dropout rate
    GLOBAL_META_LOOPS: int = 0  # Total meta-loops trained across all sessions
    BOUNTY_META_LOOP: int = 0   # Global meta-loop where bounty was last improved

    def getCheckpointPath(self) -> Path:
        return Path(self.MODEL_DIR) / self.CHECKPOINT_NAME

def parse_arch_string(arch_str: str) -> list[int]:
    """Parse architecture string '[4*4, 2*8]' into list [4, 4, 8, 8]."""
    if not re.match(r'^\[[\d\s,\*]+\]$', arch_str):
        raise ValueError(f"Invalid architecture string format: {arch_str}")

    content = arch_str.strip()[1:-1] # Remove brackets
    if not content:
        return []

    hidden_dims = []
    parts = [part.strip() for part in content.split(',')]
    for part in parts:
        try:
            count, size = [int(p.strip()) for p in part.split('*')]
            hidden_dims.extend([size] * count)
        except ValueError:
            raise ValueError(f"Malformed segment in architecture string: '{part}'")
    return hidden_dims

def get_config() -> Config:
    """Parse command-line arguments and return a Config dataclass instance."""
    parser = argparse.ArgumentParser(description="Persistent Stochastic Ablation Trainer for MNIST")
    parser.add_argument("--model-dir", type=str, default="models/", help="Path to store/retrieve models.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--meta-loops", type=int, default=100, help="Total meta-loops to run.")
    parser.add_argument(
        "--arch", type=str, default=None,
        help="Define architecture with a string, e.g., '[4*4, 2*8, 1*16]'."
    )
    parser.add_argument(
        "--hidden-layers", type=int, nargs='+', default=[1024],
        help="A list of hidden layer sizes (used if --arch is not provided)."
    )
    parser.add_argument(
        "--ablation-mode", type=str, default="none", 
        choices=["none", "decay", "dropout", "full", "hidden", "output"],
        help="Set the ablation mode."
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], 
                       help="Override device detection (cpu, cuda, mps)")
    parser.add_argument("--weight-decay", type=float, default=Config.WEIGHT_DECAY, 
                       help="Weight decay rate (only used with --ablation-mode decay)")
    parser.add_argument("--dropout", type=float, default=Config.DROPOUT_RATE, 
                       help="Dropout rate (only used with --ablation-mode dropout)")
    args = parser.parse_args()

    if args.arch:
        hidden_layers = parse_arch_string(args.arch)
        arch_string_for_display = args.arch
    else:
        hidden_layers = args.hidden_layers
        arch_string_for_display = f"[{'*'.join(map(str, [len(hidden_layers), hidden_layers[0]])) if len(set(hidden_layers)) == 1 else 'Custom'}]"


    # Check for SageMaker environment variable
    sagemaker_model_dir = os.environ.get("SAGEMAKER_MODEL_DIR")
    if sagemaker_model_dir:
        model_dir = Path(sagemaker_model_dir)
    else:
        model_dir = Path(args.model_dir)
    
    model_dir.mkdir(parents=True, exist_ok=True)

    # Handle device override
    device = args.device if args.device else detect_best_device()

    config = Config(
        MODEL_DIR=str(model_dir),
        LEARNING_RATE=args.lr,
        BATCH_SIZE=args.batch_size,
        NUM_META_LOOPS=args.meta_loops,
        HIDDEN_LAYERS=hidden_layers,
        ARCH_STRING=arch_string_for_display,
        ABLATION_MODE=args.ablation_mode,
        DEBUG=args.debug,
        NUM_WORKERS=args.num_workers,
        WEIGHT_DECAY=args.weight_decay,
        DROPOUT_RATE=args.dropout,
    )
    # Override device if specified
    if args.device:
        config.DEVICE = device
    return config

def setup_logging(is_debug: bool, console: Console) -> logging.Logger:
    """Configure logging with rich formatting."""
    log_level = "DEBUG" if is_debug else "INFO"
    logging.basicConfig(level=log_level, format="%(message)s", datefmt="[%X]",
                        handlers=[RichHandler(rich_tracebacks=True, show_path=is_debug, console=console)])
    return logging.getLogger("rich")

# --- 2. Core Model and Ablator ---

class SimpleMLP(nn.Module):
    """MLP built from a list of hidden layer sizes."""
    def __init__(self, config: Config):
        super().__init__()
        self.flatten = nn.Flatten()
        self.config = config
        layers = []
        input_size = config.INPUT_SIZE
        if config.HIDDEN_LAYERS:
            for hidden_size in config.HIDDEN_LAYERS:
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(nn.ReLU())
                # Always add dropout layers for consistent architecture
                # They will be activated/deactivated during forward pass based on ablation mode
                layers.append(nn.Dropout(config.DROPOUT_RATE))
                input_size = hidden_size
        layers.append(nn.Linear(input_size, config.OUTPUT_SIZE))
        self.layer_stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Set dropout layers to eval mode if not using dropout ablation mode
        if self.config.ABLATION_MODE != "dropout":
            # Temporarily set all dropout layers to eval mode
            dropout_layers = [layer for layer in self.layer_stack if isinstance(layer, nn.Dropout)]
            original_modes = [layer.training for layer in dropout_layers]
            for layer in dropout_layers:
                layer.eval()
        
        result = self.layer_stack(self.flatten(x))
        
        # Restore original training modes
        if self.config.ABLATION_MODE != "dropout":
            for layer, original_mode in zip(dropout_layers, original_modes):
                layer.train(original_mode)
        
        return result

class Ablator:
    """Handles different modes of neuron ablation for deep networks."""
    def __init__(self, model: nn.Module, mode: str, log: logging.Logger):
        self.log = log
        self.mode = mode
        self.ablatable_targets = []
        # Only set up ablation targets for actual ablation modes
        if self.mode == "hidden": 
            self._index_hidden_neurons(model)
        elif self.mode == "full": 
            self._index_full_layers(model)
        elif self.mode == "output": 
            self._index_output_layer(model)
        # baseline modes (none, decay, dropout) don't need ablation targets

    def _index_full_layers(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear): self.ablatable_targets.append({'name': name, 'module': module})
        self.log.info(f"Ablator (full mode) indexed {len(self.ablatable_targets)} total linear layers.")

    def _index_hidden_neurons(self, model: nn.Module):
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if len(linear_layers) <= 1:
            self.log.warning("Ablator (hidden mode) found no hidden layers to index.")
            return
        hidden_layers, output_layer = linear_layers[:-1], linear_layers[-1]
        module_to_name = {v: k for k, v in dict(model.named_modules()).items()}
        self.log.info(f"Ablator (hidden mode) identified {len(hidden_layers)} hidden layer(s).")
        for i, layer in enumerate(hidden_layers):
            layer_name = module_to_name[layer]
            next_layer = hidden_layers[i + 1] if i + 1 < len(hidden_layers) else output_layer
            next_layer_name = module_to_name[next_layer]
            for neuron_idx in range(layer.out_features):
                self.ablatable_targets.append({
                    "layer_name": layer_name, "neuron_idx": neuron_idx,
                    "incoming_weight_key": f"{layer_name}.weight",
                    "incoming_bias_key": f"{layer_name}.bias",
                    "outgoing_weight_key": f"{next_layer_name}.weight"
                })
        if self.ablatable_targets: self.log.info(f"Ablator (hidden mode) indexed {len(self.ablatable_targets)} hidden neurons.")

    def _index_output_layer(self, model: nn.Module):
        """Indexes only the final linear layer for the 'output' ablation mode."""
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            self.log.warning("Ablator (output mode) found no linear layers.")
            return
        output_layer_module = linear_layers[-1]
        module_to_name = {v: k for k, v in dict(model.named_modules()).items()}
        output_layer_name = module_to_name[output_layer_module]
        self.ablatable_targets.append({'name': output_layer_name, 'module': output_layer_module})
        self.log.info(f"Ablator (output mode) indexed the final output layer: '{output_layer_name}'.")

    def ablate(self, model: nn.Module) -> dict:
        if self.mode == "none" or not self.ablatable_targets: return model.state_dict()
        state_dict = copy.deepcopy(model.state_dict())

        if self.mode == "output":
            target = self.ablatable_targets[0] 
            name, module = target['name'], target['module']
            idx = random.randint(0, module.out_features - 1)
            self.log.info(f"🧠 (Output Mode) Partially ablating neuron {idx} in output layer '{name}'.")
            state_dict[f"{name}.weight"][idx, :] = 0.0
            if module.bias is not None: state_dict[f"{name}.bias"][idx] = 0.0
        elif self.mode == "full":
            target = random.choice(self.ablatable_targets)
            name, module = target['name'], target['module']
            idx = random.randint(0, module.out_features - 1)
            self.log.info(f"🧠 (Full Mode) Partially ablating neuron {idx} in layer '{name}'.")
            state_dict[f"{name}.weight"][idx, :] = 0.0
            if module.bias is not None: state_dict[f"{name}.bias"][idx] = 0.0
        elif self.mode == "hidden":
            target = random.choice(self.ablatable_targets)
            name, idx = target['layer_name'], target['neuron_idx']
            self.log.info(f"🧠 (Hidden Mode) Fully ablating neuron {idx} in hidden layer '{name}'.")
            state_dict[target['incoming_weight_key']][idx, :] = 0.0
            state_dict[target['incoming_bias_key']][idx] = 0.0
            state_dict[target['outgoing_weight_key']][:, idx] = 0.0
        return state_dict

# --- 3. UI and Helper Functions ---

def display_architecture_summary(model: SimpleMLP, config: Config, console: Console):
    """Create and print a summary table of the model's architecture."""
    table = Table(title=f"Model Architecture Summary: {config.ARCH_STRING}", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    
    groups = []
    if config.HIDDEN_LAYERS:
        i = 0
        while i < len(config.HIDDEN_LAYERS):
            size = config.HIDDEN_LAYERS[i]
            count = 1
            while i + count < len(config.HIDDEN_LAYERS) and config.HIDDEN_LAYERS[i + count] == size:
                count += 1
            groups.append({'start_idx': i, 'count': count, 'size': size})
            i += count

    table.add_column("Input (i)", justify="right")
    for group in groups:
        header = f"h_{group['start_idx']}" if group['count'] == 1 else f"h_{group['start_idx']}-{group['start_idx'] + group['count'] - 1}"
        table.add_column(header, justify="right")
    table.add_column("Output (o)", justify="right")

    shapes, params = ["Shape"], ["Parameters"]
    shapes.append(f"{config.INPUT_SIZE}")
    params.append("-")

    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    
    group_idx = 0
    layer_idx_in_group = 0
    for i, linear_layer in enumerate(linear_layers[:-1]):
        if layer_idx_in_group == 0:
            shapes.append(f"{linear_layer.in_features} → {linear_layer.out_features}")
            group_params = sum(
                sum(p.numel() for p in layer.parameters()) 
                for layer in linear_layers[i : i + groups[group_idx]['count']]
            )
            params.append(f"{group_params:,}")

        layer_idx_in_group += 1
        if layer_idx_in_group >= groups[group_idx]['count']:
            group_idx += 1
            layer_idx_in_group = 0
            
    output_layer = linear_layers[-1]
    shapes.append(f"{output_layer.in_features} → {output_layer.out_features}")
    p = sum(p.numel() for p in output_layer.parameters())
    params.append(f"{p:,}")

    table.add_row(*shapes)
    table.add_row(*params)
    console.print(table)


def get_mnist_loaders(config: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test splits for MNIST with optimized data loading:
    - Training: 50,000 images (from train=True)
    - Validation: 10,000 images (from train=True) 
    - Test: 10,000 images (from train=False)
    
    Optimized to load data into RAM once to eliminate I/O bottlenecks.
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    # Load full datasets from disk ONCE
    full_train_ds = datasets.MNIST("dataset", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("dataset", train=False, download=True, transform=transform)
    
    # Split training data into train (50k) and validation (10k)
    train_size = 50000
    val_size = 10000
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_train_ds)), [train_size, val_size],
        generator=torch.Generator().manual_seed(1337)  # Fixed seed for reproducibility
    )
    
    # Load all data into RAM as tensors (eliminates I/O bottlenecks)
    train_images = torch.stack([full_train_ds[idx][0] for idx in train_indices.indices])
    train_labels = torch.tensor([full_train_ds[idx][1] for idx in train_indices.indices])
    
    val_images = torch.stack([full_train_ds[idx][0] for idx in val_indices.indices])
    val_labels = torch.tensor([full_train_ds[idx][1] for idx in val_indices.indices])
    
    test_images = torch.stack([img for img, _ in test_ds])
    test_labels = torch.tensor([label for _, label in test_ds])
    
    # Create in-memory TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_images, val_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    
    # Create data loaders with optimized settings
    # Use num_workers=0 for in-memory data to avoid overhead
    # Disable pin_memory on MPS (not supported) and use smaller batch for validation
    pin_memory = config.DEVICE == "cuda"  # Only use pin_memory for CUDA
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                             num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, 
                           num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, 
                            num_workers=0, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader

# --- 4. Training & Evaluation Functions ---

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, config: Config, progress: Progress, task_id: TaskID):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(config.DEVICE), target.to(config.DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        progress.update(task_id, advance=1)
        if batch_idx % config.LOG_INTERVAL == 0:
            progress.update(task_id, description=f"Training... Loss: {loss.item():.4f}")

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, config: Config, progress: Progress) -> float:
    model.eval()
    correct, total = 0, 0
    eval_task = progress.add_task("[cyan]Evaluating...", total=len(loader))
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            progress.update(eval_task, advance=1)
    progress.remove_task(eval_task)
    return 100. * correct / total

# --- 5. Main Orchestration ---

def main():
    config = get_config()
    console = Console()
    log = setup_logging(config.DEBUG, console)

    model = SimpleMLP(config).to(config.DEVICE)
    ablator = Ablator(model, config.ABLATION_MODE, log)
    
    criterion = nn.CrossEntropyLoss()
    
    console.print(Panel.fit(
        "[bold magenta]Frustration Engine: MNIST Ablation Test[/bold magenta]",
        subtitle=f"Ablation Mode: [yellow]{config.ABLATION_MODE}[/yellow] | Device: {config.DEVICE}"
    ))
    display_architecture_summary(model, config, console)

    train_loader, val_loader, test_loader = get_mnist_loaders(config)
    log.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
    log.info(f"Training for {config.NUM_META_LOOPS} meta-loops.")
    log.info(f"📊 Dataset splits: 50k train, 10k validation, 10k test")
    log.info(f"🎯 Meta-loops use validation accuracy for LKG decisions")
    log.info(f"🧪 Final test accuracy reported at completion")

    lkg_score, bounty = -1.0, -1.0
    lkg_model_state = None
    checkpoint_path = config.getCheckpointPath()

    if checkpoint_path.exists():
        log.info(f"Loading LKG checkpoint from {checkpoint_path}")
        
        # Try to load as torch.save format first (new format with metadata)
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=config.DEVICE)
            if isinstance(checkpoint_data, dict) and 'model_state' in checkpoint_data:
                # New format with metadata
                lkg_model_state = checkpoint_data['model_state']
                config.GLOBAL_META_LOOPS = checkpoint_data.get('global_meta_loops', 0)
                config.BOUNTY_META_LOOP = checkpoint_data.get('bounty_meta_loop', 0)
                log.info(f"Resuming from global meta-loop {config.GLOBAL_META_LOOPS}, bounty achieved at loop {config.BOUNTY_META_LOOP}")
            else:
                # Legacy torch.save format (just model state)
                lkg_model_state = checkpoint_data
                log.info("Legacy torch.save format detected - starting global meta-loop tracking from 0")
        except Exception as e:
            # Try safetensors format as fallback
            try:
                checkpoint_data = load_file(checkpoint_path, device=config.DEVICE)
                # Legacy safetensors format (just model state)
                lkg_model_state = checkpoint_data
                log.info("Legacy safetensors format detected - starting global meta-loop tracking from 0")
            except Exception as e2:
                log.error(f"Failed to load checkpoint: {e2}")
                log.warning(f"No valid checkpoint found at {checkpoint_path}. Starting from scratch.")
                lkg_model_state = model.state_dict()
        
        model.load_state_dict(lkg_model_state)
        with Progress(transient=True, console=console) as progress:
           lkg_score = evaluate(model, val_loader, criterion, config, progress)
        bounty = lkg_score
        log.info(f"Resuming with LKG validation accuracy: {lkg_score:.2f}%")
    else:
        log.warning(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        lkg_model_state = model.state_dict()

    active_model_state = copy.deepcopy(lkg_model_state)
    
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"), BarColumn(),
            MofNCompleteColumn(), TimeRemainingColumn(), console=console
        ) as progress:
            meta_loop_task = progress.add_task("[bold]Meta-Loops[/bold]", total=config.NUM_META_LOOPS)
            for loop in range(config.NUM_META_LOOPS):
                current_global_loop = config.GLOBAL_META_LOOPS + loop + 1
                model.load_state_dict(active_model_state)
                
                # Reset optimizer each meta-loop for fair comparison
                # This ensures each meta-loop starts with a fresh optimizer state,
                # providing a fair comparison between ablation strategies without
                # momentum carryover from previous attempts.
                if config.ABLATION_MODE == "decay":
                    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
                    if loop == 0:  # Log once at the start
                        log.info(f"🔧 Using weight decay: {config.WEIGHT_DECAY}")
                elif config.ABLATION_MODE == "dropout":
                    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
                    if loop == 0:  # Log once at the start
                        log.info(f"🔧 Using dropout rate: {config.DROPOUT_RATE}")
                else:
                    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
                
                train_task = progress.add_task("Training...", total=len(train_loader))
                train_one_epoch(model, train_loader, optimizer, criterion, config, progress, train_task)
                progress.remove_task(train_task)

                new_score = evaluate(model, val_loader, criterion, config, progress)
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_row("Previous LKG Validation Accuracy", f"[cyan]{lkg_score:.2f}%[/cyan]")
                table.add_row("Current Loop Validation Accuracy", f"[bold yellow]{new_score:.2f}%[/bold yellow]")
                table.add_row("Global Meta-Loop", f"[bold blue]{current_global_loop}[/bold blue]")

                if new_score > lkg_score:
                    lkg_score, lkg_model_state = new_score, copy.deepcopy(model.state_dict())
                    # Save checkpoint with metadata
                    # Use torch.save for the full checkpoint with metadata
                    checkpoint_data = {
                        'model_state': lkg_model_state,
                        'global_meta_loops': config.GLOBAL_META_LOOPS + loop + 1,
                        'bounty_meta_loop': config.BOUNTY_META_LOOP
                    }
                    torch.save(checkpoint_data, checkpoint_path)
                    status, message = "[bold green]IMPROVEMENT[/bold green]", "New LKG. Checkpoint saved."
                    if lkg_score > bounty:
                        bounty = lkg_score
                        config.BOUNTY_META_LOOP = current_global_loop
                        message += f" 🏆 New Bounty: {bounty:.2f}% @ {current_global_loop}"
                else:
                    status, message = "[bold red]NO IMPROVEMENT[/bold red]", "Discarding weights."
                console.print(Panel(table, title=f"Meta-Loop {loop + 1}/{config.NUM_META_LOOPS} (Global: {current_global_loop})", subtitle=status, border_style="blue"))
                log.info(message)
                
                if config.ABLATION_MODE not in ['none', 'decay', 'dropout']:
                    log.info(f"Ablating LKG model for next loop (mode: {config.ABLATION_MODE})...")
                    temp_model = SimpleMLP(config)
                    temp_model.load_state_dict(lkg_model_state)
                    active_model_state = ablator.ablate(temp_model)
                else:
                    active_model_state = copy.deepcopy(lkg_model_state)
                progress.update(meta_loop_task, advance=1)
                console.print("")
    except KeyboardInterrupt:
        log.warning("\nTraining interrupted by user.")
    finally:
        # Update global meta-loop count
        config.GLOBAL_META_LOOPS += config.NUM_META_LOOPS
        
        # Final evaluation on test set
        log.info(f"Final LKG model stored at: {checkpoint_path}")
        log.info(f"🏆 Final Bounty (best validation accuracy achieved): {bounty:.2f}% @ {config.BOUNTY_META_LOOP}/{config.GLOBAL_META_LOOPS}")
        
        # Load the best model and evaluate on test set
        if lkg_model_state:
            model.load_state_dict(lkg_model_state)
            with Progress(transient=True, console=console) as progress:
                final_test_accuracy = evaluate(model, test_loader, criterion, config, progress)
            log.info(f"🧪 Final Test Accuracy: {final_test_accuracy:.2f}%")
            console.print(Panel.fit(f"[bold green]✅ Training Finished. Final Bounty (Validation): {bounty:.2f}% @ {config.BOUNTY_META_LOOP}/{config.GLOBAL_META_LOOPS} | Test: {final_test_accuracy:.2f}%[/bold green]"))
        else:
            console.print(Panel.fit(f"[bold green]✅ Training Finished. Final Bounty (Validation): {bounty:.2f}% @ {config.BOUNTY_META_LOOP}/{config.GLOBAL_META_LOOPS}[/bold green]"))

if __name__ == "__main__":
    main()
