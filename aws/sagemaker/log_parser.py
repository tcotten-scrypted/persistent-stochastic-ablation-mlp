#!/usr/bin/env python3
"""
AWS SageMaker Log Parser for PSA Experiments

This script parses CloudWatch training logs to extract LKG growth and convergence data,
generating markdown tables for analysis of training progression across meta-loops.

Author: Tim Cotten @cottenio <tcotten@scrypted.ai, tcotten2@gmu.edu>
"""

import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import pandas as pd

# --- Configuration ---
CONSOLE = Console()

class LogParser:
    """Parser for PSA training logs to extract LKG and convergence data."""
    
    def __init__(self, logs_dir: Path):
        self.logs_dir = Path(logs_dir)
        self.lkg_growth_data = []
        self.convergence_data = []
        
    def parse_job_logs(self, job_name: str) -> Tuple[Dict, Dict]:
        """
        Parse logs for a single job to extract LKG growth and convergence data.
        
        Returns:
            Tuple of (lkg_data, convergence_data) dictionaries
        """
        job_dir = self.logs_dir / job_name
        if not job_dir.exists():
            return {}, {}
        
        # Find all log files for this job
        log_files = list(job_dir.rglob("*.log"))
        if not log_files:
            return {}, {}
        
        # Parse all log files for this job
        lkg_history = []
        convergence_history = []
        
        for log_file in log_files:
            with open(log_file, 'r') as f:
                for line in f:
                    # Parse LKG updates
                    lkg_match = self._parse_lkg_line(line)
                    if lkg_match:
                        lkg_history.append(lkg_match)
                    
                    # Parse validation accuracy
                    val_match = self._parse_validation_line(line)
                    if val_match:
                        convergence_history.append(val_match)
        
        # Sort by meta-loop and trial
        lkg_history.sort(key=lambda x: (x['meta_loop'], x['trial']))
        convergence_history.sort(key=lambda x: (x['meta_loop'], x['trial']))
        
        return self._process_lkg_data(lkg_history), self._process_convergence_data(convergence_history)
    
    def _parse_lkg_line(self, line: str) -> Optional[Dict]:
        """Parse LKG update lines from training logs."""
        # Look for patterns like:
        # "[18:39:19] INFO     Loop 1/100 (Global: 1) | Current: 90.23% | LKG: 90.23% |"
        # Extract meta-loop number and LKG accuracy
        pattern = r'Loop (\d+)/100.*LKG:\s*([\d.]+)%'
        
        match = re.search(pattern, line)
        if match:
            return {
                'accuracy': float(match.group(2)),
                'meta_loop': int(match.group(1)),
                'trial': 0  # Default trial number since not specified in logs
            }
        
        return None
    
    def _parse_validation_line(self, line: str) -> Optional[Dict]:
        """Parse validation accuracy lines from training logs."""
        # Look for patterns like:
        # "[18:39:19] INFO     Loop 1/100 (Global: 1) | Current: 90.23% | LKG: 90.23% |"
        # Extract meta-loop number and current validation accuracy
        pattern = r'Loop (\d+)/100.*Current:\s*([\d.]+)%'
        
        match = re.search(pattern, line)
        if match:
            return {
                'accuracy': float(match.group(2)),
                'meta_loop': int(match.group(1)),
                'trial': 0  # Default trial number since not specified in logs
            }
        
        return None
    
    def _process_lkg_data(self, lkg_history: List[Dict]) -> Dict:
        """Process LKG history to calculate growth per meta-loop."""
        if not lkg_history:
            return {}
        
        # Group by meta-loop and find the best LKG for each
        meta_loop_lkgs = {}
        for entry in lkg_history:
            meta_loop = entry['meta_loop']
            if meta_loop not in meta_loop_lkgs or entry['accuracy'] > meta_loop_lkgs[meta_loop]['accuracy']:
                meta_loop_lkgs[meta_loop] = entry
        
        # Calculate growth (difference from previous LKG)
        growth_data = {}
        prev_lkg = None
        
        for meta_loop in sorted(meta_loop_lkgs.keys()):
            current_lkg = meta_loop_lkgs[meta_loop]['accuracy']
            
            if prev_lkg is not None:
                growth = current_lkg - prev_lkg
            else:
                growth = 0.0  # First LKG, no growth
            
            growth_data[meta_loop] = growth
            prev_lkg = current_lkg
        
        return growth_data
    
    def _process_convergence_data(self, convergence_history: List[Dict]) -> Dict:
        """Process convergence history to get validation accuracy per meta-loop."""
        if not convergence_history:
            return {}
        
        # Group by meta-loop and find the best validation accuracy for each
        meta_loop_accuracies = {}
        for entry in convergence_history:
            meta_loop = entry['meta_loop']
            if meta_loop not in meta_loop_accuracies or entry['accuracy'] > meta_loop_accuracies[meta_loop]:
                meta_loop_accuracies[meta_loop] = entry['accuracy']
        
        return meta_loop_accuracies
    
    def parse_all_jobs(self) -> None:
        """Parse all job logs and collect data."""
        job_dirs = [d for d in self.logs_dir.iterdir() if d.is_dir()]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=CONSOLE
        ) as progress:
            task = progress.add_task("Parsing job logs...", total=len(job_dirs))
            
            for job_dir in job_dirs:
                job_name = job_dir.name
                progress.update(task, description=f"Parsing {job_name}...")
                
                lkg_data, convergence_data = self.parse_job_logs(job_name)
                
                if lkg_data or convergence_data:
                    # Extract architecture info from job name
                    # Format: psa-{width}x{depth}-{mode}-{timestamp}
                    arch_match = re.match(r'psa-(\d+)x(\d+)-(\w+)-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3}', job_name)
                    if arch_match:
                        width = int(arch_match.group(1))
                        depth = int(arch_match.group(2))
                        mode = arch_match.group(3)
                        
                        # Add to data collections
                        if lkg_data:
                            self.lkg_growth_data.append({
                                'architecture': f"{width}x{depth}",
                                'mode': mode,
                                'job_name': job_name,
                                'lkg_growth': lkg_data
                            })
                        
                        if convergence_data:
                            self.convergence_data.append({
                                'architecture': f"{width}x{depth}",
                                'mode': mode,
                                'job_name': job_name,
                                'convergence': convergence_data
                            })
                
                progress.advance(task)
    
    def generate_lkg_growth_markdown(self, output_file: Path) -> None:
        """Generate LKG growth markdown table."""
        if not self.lkg_growth_data:
            CONSOLE.print("⚠️  No LKG growth data found", style="yellow")
            return
        
        # Find all unique meta-loops
        all_meta_loops = set()
        for entry in self.lkg_growth_data:
            all_meta_loops.update(entry['lkg_growth'].keys())
        
        meta_loops = sorted(all_meta_loops)
        
        # Create markdown table
        lines = [
            "# PSA SimpleMLP Trials - LKG Growth Analysis",
            "",
            "This table shows the Last Known Good (LKG) growth per meta-loop for each architecture and ablation mode.",
            "Values represent the improvement in validation accuracy from the previous LKG (0.0 if no improvement).",
            "",
            "| Architecture | Mode | " + " | ".join([f"Meta-Loop {ml}" for ml in meta_loops]) + " | Final Test |",
            "|-------------|------|" + "|".join(["---" for _ in meta_loops]) + "|------------|"
        ]
        
        # Sort data by architecture and mode
        sorted_data = sorted(self.lkg_growth_data, key=lambda x: (x['architecture'], x['mode']))
        
        for entry in sorted_data:
            arch = entry['architecture']
            mode = entry['mode']
            
            # Get growth values for each meta-loop
            growth_values = []
            for ml in meta_loops:
                growth = entry['lkg_growth'].get(ml, 0.0)
                growth_values.append(f"{growth:.2f}")
            
            # TODO: Extract final test score from logs or summary files
            final_test = "N/A"  # Placeholder
            
            row = f"| {arch} | {mode} | " + " | ".join(growth_values) + f" | {final_test} |"
            lines.append(row)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        
        CONSOLE.print(f"✅ LKG growth data written to {output_file}", style="green")
    
    def generate_convergence_markdown(self, output_file: Path) -> None:
        """Generate convergence markdown table."""
        if not self.convergence_data:
            CONSOLE.print("⚠️  No convergence data found", style="yellow")
            return
        
        # Find all unique meta-loops
        all_meta_loops = set()
        for entry in self.convergence_data:
            all_meta_loops.update(entry['convergence'].keys())
        
        meta_loops = sorted(all_meta_loops)
        
        # Create markdown table
        lines = [
            "# PSA SimpleMLP Trials - Convergence Analysis",
            "",
            "This table shows the validation accuracy per meta-loop for each architecture and ablation mode.",
            "",
            "| Architecture | Mode | " + " | ".join([f"Meta-Loop {ml}" for ml in meta_loops]) + " | Final Test |",
            "|-------------|------|" + "|".join(["---" for _ in meta_loops]) + "|------------|"
        ]
        
        # Sort data by architecture and mode
        sorted_data = sorted(self.convergence_data, key=lambda x: (x['architecture'], x['mode']))
        
        for entry in sorted_data:
            arch = entry['architecture']
            mode = entry['mode']
            
            # Get validation accuracy for each meta-loop
            acc_values = []
            for ml in meta_loops:
                acc = entry['convergence'].get(ml, "N/A")
                if acc != "N/A":
                    acc_values.append(f"{acc:.2f}")
                else:
                    acc_values.append("N/A")
            
            # TODO: Extract final test score from logs or summary files
            final_test = "N/A"  # Placeholder
            
            row = f"| {arch} | {mode} | " + " | ".join(acc_values) + f" | {final_test} |"
            lines.append(row)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        
        CONSOLE.print(f"✅ Convergence data written to {output_file}", style="green")

def main():
    """Main function to parse logs and generate markdown tables."""
    parser = argparse.ArgumentParser(description="Parse PSA training logs for LKG growth and convergence analysis")
    parser.add_argument("--logs-dir", type=str, default="results/logs",
                       help="Directory containing downloaded logs (default: results/logs)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for markdown files (default: results)")
    args = parser.parse_args()
    
    CONSOLE.print("🔍 PSA Log Parser", style="bold blue")
    CONSOLE.print(f"   Logs directory: {args.logs_dir}", style="dim")
    CONSOLE.print(f"   Output directory: {args.output_dir}", style="dim")
    
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    
    if not logs_dir.exists():
        CONSOLE.print(f"🔴 Logs directory not found: {logs_dir}", style="red")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize parser and parse all jobs
    parser = LogParser(logs_dir)
    parser.parse_all_jobs()
    
    # Generate markdown files
    lkg_output = output_dir / "psa_simplemlp_trials_lkg_growth.md"
    convergence_output = output_dir / "psa_simplemlp_trials_convergence.md"
    
    parser.generate_lkg_growth_markdown(lkg_output)
    parser.generate_convergence_markdown(convergence_output)
    
    # Display summary
    CONSOLE.print("\n📊 Parsing Summary", style="bold green")
    
    summary_table = Table(title="Parsing Results")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Jobs with LKG Data", str(len(parser.lkg_growth_data)))
    summary_table.add_row("Jobs with Convergence Data", str(len(parser.convergence_data)))
    summary_table.add_row("LKG Growth File", str(lkg_output))
    summary_table.add_row("Convergence File", str(convergence_output))
    
    CONSOLE.print(summary_table)
    
    return 0

if __name__ == "__main__":
    exit(main()) 