#!/usr/bin/env python3
"""
Dataset Analysis Utility for MNIST using PSA SimpleMLP methodology.

This script analyzes the MNIST dataset using the exact same data loading and splitting
methodology as the PSA SimpleMLP training code, providing detailed statistics about
class distribution and ZeroR baselines.

Author: Tim Cotten @cottenio <tcotten@scrypted.ai, tcotten2@gmu.edu>
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


@dataclass
class DatasetStats:
    """Structured container for dataset statistics."""
    name: str
    total_samples: int
    class_counts: Dict[int, int]
    class_percentages: Dict[int, float]
    zeror_accuracy: float
    zeror_class: int


@dataclass
class MNISTAnalysis:
    """Complete MNIST dataset analysis results."""
    full_train: DatasetStats
    full_test: DatasetStats
    split_train: DatasetStats
    split_validation: DatasetStats


def analyze_dataset(labels: torch.Tensor, name: str) -> DatasetStats:
    """Analyze a dataset and return structured statistics."""
    labels_np = labels.numpy()
    total_samples = len(labels_np)
    
    # Count occurrences of each class (0-9)
    class_counts = {}
    for i in range(10):
        class_counts[i] = int(np.sum(labels_np == i))
    
    # Calculate percentages
    class_percentages = {}
    for i in range(10):
        class_percentages[i] = (class_counts[i] / total_samples) * 100
    
    # ZeroR baseline (most frequent class)
    zeror_class = max(class_counts, key=class_counts.get)
    zeror_accuracy = (class_counts[zeror_class] / total_samples) * 100
    
    return DatasetStats(
        name=name,
        total_samples=total_samples,
        class_counts=class_counts,
        class_percentages=class_percentages,
        zeror_accuracy=zeror_accuracy,
        zeror_class=zeror_class
    )


def get_mnist_data_splits() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and split MNIST data using the exact same methodology as PSA SimpleMLP.
    Returns: (full_train_labels, full_test_labels, split_train_labels, split_val_labels)
    """
    # Use the same transform as the training code
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full datasets from disk (same as training code)
    full_train_ds = datasets.MNIST("dataset", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("dataset", train=False, download=True, transform=transform)
    
    # Extract labels for full datasets
    full_train_labels = torch.tensor([label for _, label in full_train_ds])
    full_test_labels = torch.tensor([label for _, label in test_ds])
    
    # Split training data into train (50k) and validation (10k) using SAME methodology
    train_size = 50000
    val_size = 10000
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_train_ds)), [train_size, val_size],
        generator=torch.Generator().manual_seed(1337)  # Same seed as training code
    )
    
    # Extract labels for split datasets
    split_train_labels = torch.tensor([full_train_ds[idx][1] for idx in train_indices.indices])
    split_val_labels = torch.tensor([full_train_ds[idx][1] for idx in val_indices.indices])
    
    return full_train_labels, full_test_labels, split_train_labels, split_val_labels


def analyze_mnist() -> MNISTAnalysis:
    """
    Perform complete MNIST analysis using PSA SimpleMLP methodology.
    Returns structured analysis object.
    """
    # Get data splits using exact same methodology as training
    full_train_labels, full_test_labels, split_train_labels, split_val_labels = get_mnist_data_splits()
    
    # Analyze each dataset
    full_train_stats = analyze_dataset(full_train_labels, "Full Training Set")
    full_test_stats = analyze_dataset(full_test_labels, "Full Test Set")
    split_train_stats = analyze_dataset(split_train_labels, "Split Training Set (50k)")
    split_val_stats = analyze_dataset(split_val_labels, "Split Validation Set (10k)")
    
    return MNISTAnalysis(
        full_train=full_train_stats,
        full_test=full_test_stats,
        split_train=split_train_stats,
        split_validation=split_val_stats
    )


def print_dataset_table(stats: DatasetStats, console: Console):
    """Print a formatted table for a single dataset."""
    table = Table(title=f"{stats.name} - Class Distribution")
    table.add_column("Class", justify="center", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Percentage", justify="right", style="green")
    
    for i in range(10):
        count = stats.class_counts[i]
        percentage = stats.class_percentages[i]
        # Highlight the ZeroR class
        style = "bold yellow" if i == stats.zeror_class else None
        table.add_row(str(i), str(count), f"{percentage:.2f}%", style=style)
    
    # Add summary row
    table.add_row("", "", "", style="dim")
    table.add_row("Total", str(stats.total_samples), "100.00%", style="bold")
    
    console.print(table)
    console.print(f"[bold green]ZeroR Baseline:[/bold green] Class {stats.zeror_class} → {stats.zeror_accuracy:.2f}% accuracy")
    console.print()


def print_analysis_summary(analysis: MNISTAnalysis, console: Console):
    """Print a summary comparison table."""
    summary_table = Table(title="MNIST Dataset Analysis Summary")
    summary_table.add_column("Dataset", style="cyan")
    summary_table.add_column("Total Samples", justify="right", style="magenta")
    summary_table.add_column("ZeroR Class", justify="center", style="yellow")
    summary_table.add_column("ZeroR Accuracy", justify="right", style="green")
    
    datasets = [
        ("Full Training (60k)", analysis.full_train),
        ("Full Test (10k)", analysis.full_test),
        ("Split Training (50k)", analysis.split_train),
        ("Split Validation (10k)", analysis.split_validation)
    ]
    
    for name, stats in datasets:
        summary_table.add_row(
            name,
            str(stats.total_samples),
            str(stats.zeror_class),
            f"{stats.zeror_accuracy:.2f}%"
        )
    
    console.print(summary_table)


def main():
    """Main function to run the analysis and display results."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold magenta]MNIST Dataset Analysis[/bold magenta]",
        subtitle="Using PSA SimpleMLP Data Loading Methodology"
    ))
    
    console.print("[bold blue]Analyzing MNIST dataset...[/bold blue]")
    analysis = analyze_mnist()
    
    console.print("\n[bold]📊 Detailed Class Distributions[/bold]\n")
    
    # Print detailed tables for each dataset
    datasets = [
        analysis.full_train,
        analysis.full_test,
        analysis.split_train,
        analysis.split_validation
    ]
    
    for stats in datasets:
        print_dataset_table(stats, console)
    
    console.print("[bold]📋 Summary Comparison[/bold]\n")
    print_analysis_summary(analysis, console)
    
    console.print(Panel.fit(
        "[bold green]✅ Analysis Complete[/bold green]",
        subtitle="All statistics computed using exact PSA SimpleMLP methodology"
    ))


if __name__ == "__main__":
    main()