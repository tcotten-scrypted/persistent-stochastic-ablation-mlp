#!/usr/bin/env python3
"""
AWS SageMaker Cost Estimator for PSA Experiments

This script queries SageMaker training jobs to calculate total costs based on
billable time and instance pricing.

Author: Tim Cotten @cottenio <tcotten@scrypted.ai, tcotten2@gmu.edu>
"""

import os
import boto3
import json
from pathlib import Path
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
script_dir = Path(__file__).resolve().parent
dotenv_path = script_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "ml.g4dn.xlarge")
INSTANCE_COST_PER_HOUR = 0.7364  # Fixed cost for ml.g4dn.xlarge in us-east-1
CONSOLE = Console()

def get_sagemaker_client():
    """Initialize and return SageMaker client."""
    try:
        session = boto3.Session(region_name=AWS_REGION)
        return session.client('sagemaker')
    except Exception as e:
        CONSOLE.print(f"🔴 Error initializing SageMaker client: {e}", style="red")
        return None

def load_requested_jobs(jobs_file_path="aws/sagemaker/requested-jobs.txt"):
    """
    Load job names from the requested-jobs.txt file.
    Returns a list of job names.
    """
    try:
        with open(jobs_file_path, 'r') as f:
            job_names = [line.strip() for line in f if line.strip()]
        CONSOLE.print(f"📄 Loaded {len(job_names)} jobs from {jobs_file_path}", style="green")
        return job_names
    except FileNotFoundError:
        CONSOLE.print(f"🔴 Jobs file not found: {jobs_file_path}", style="red")
        CONSOLE.print("   Run 'poetry run sagemaker-estimate-storage' first to generate the jobs file", style="yellow")
        return []
    except Exception as e:
        CONSOLE.print(f"🔴 Error reading jobs file: {e}", style="red")
        return []

def get_job_details(sagemaker_client, job_names):
    """
    Get detailed information for specific training jobs.
    Returns a list of job details including billable time.
    """
    CONSOLE.print(f"🔍 Querying SageMaker for {len(job_names)} specific training jobs...", style="blue")
    
    jobs = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=CONSOLE
    ) as progress:
        task = progress.add_task("Fetching job details...", total=len(job_names))
        
        for job_name in job_names:
            try:
                # Get detailed job information
                job_details = sagemaker_client.describe_training_job(
                    TrainingJobName=job_name
                )
                
                # Extract billable time
                billable_time = job_details.get('BillableTimeInSeconds', 0)
                
                jobs.append({
                    'name': job_name,
                    'status': job_details.get('TrainingJobStatus', 'Unknown'),
                    'creation_time': job_details.get('CreationTime'),
                    'end_time': job_details.get('TrainingEndTime'),
                    'billable_seconds': billable_time,
                    'billable_hours': billable_time / 3600,
                    'cost': (billable_time / 3600) * INSTANCE_COST_PER_HOUR
                })
                
            except ClientError as e:
                CONSOLE.print(f"⚠️  Error getting details for {job_name}: {e}", style="yellow")
                continue
            
            progress.update(task, advance=1, description=f"Processed {len(jobs)}/{len(job_names)} jobs...")
    
    CONSOLE.print(f"✅ Retrieved details for {len(jobs)} training jobs", style="green")
    return jobs

def parse_job_name(job_name):
    """
    Parse job name to extract architecture and mode.
    Format: psa-{depth}x{width}-{mode}-{timestamp}
    """
    try:
        # Remove timestamp and split
        parts = job_name.split('-')
        if len(parts) >= 3:
            arch_part = parts[1]  # e.g., "1x2048"
            mode = parts[2]       # e.g., "none"
            
            # Parse architecture
            if 'x' in arch_part:
                depth, width = arch_part.split('x')
                return {
                    'architecture': f"{depth}*{width}",
                    'depth': int(depth),
                    'width': int(width),
                    'mode': mode
                }
    except Exception:
        pass
    
    return {
        'architecture': 'unknown',
        'depth': 0,
        'width': 0,
        'mode': 'unknown'
    }

def calculate_cost_breakdown(jobs):
    """
    Calculate cost breakdown by various categories.
    """
    breakdown = {
        'total': {
            'jobs': 0,
            'billable_hours': 0,
            'cost': 0
        },
        'concurrency': {
            'max_wall_clock_hours': 0,
            'max_wall_clock_job': None,
            'max_wall_clock_architecture': None,
            'max_wall_clock_mode': None
        },
        'by_status': {},
        'by_architecture': {},
        'by_mode': {},
        'by_architecture_mode': {}
    }
    
    for job in jobs:
        # Total
        breakdown['total']['jobs'] += 1
        breakdown['total']['billable_hours'] += job['billable_hours']
        breakdown['total']['cost'] += job['cost']
        
        # Track maximum wall-clock time (for concurrent execution)
        if job['billable_hours'] > breakdown['concurrency']['max_wall_clock_hours']:
            breakdown['concurrency']['max_wall_clock_hours'] = job['billable_hours']
            breakdown['concurrency']['max_wall_clock_job'] = job['name']
            
            # Parse job name for architecture and mode
            job_info = parse_job_name(job['name'])
            breakdown['concurrency']['max_wall_clock_architecture'] = job_info['architecture']
            breakdown['concurrency']['max_wall_clock_mode'] = job_info['mode']
        
        # By status
        status = job['status']
        if status not in breakdown['by_status']:
            breakdown['by_status'][status] = {'jobs': 0, 'billable_hours': 0, 'cost': 0}
        breakdown['by_status'][status]['jobs'] += 1
        breakdown['by_status'][status]['billable_hours'] += job['billable_hours']
        breakdown['by_status'][status]['cost'] += job['cost']
        
        # Parse job name
        job_info = parse_job_name(job['name'])
        arch = job_info['architecture']
        mode = job_info['mode']
        
        # By architecture
        if arch not in breakdown['by_architecture']:
            breakdown['by_architecture'][arch] = {'jobs': 0, 'billable_hours': 0, 'cost': 0}
        breakdown['by_architecture'][arch]['jobs'] += 1
        breakdown['by_architecture'][arch]['billable_hours'] += job['billable_hours']
        breakdown['by_architecture'][arch]['cost'] += job['cost']
        
        # By mode
        if mode not in breakdown['by_mode']:
            breakdown['by_mode'][mode] = {'jobs': 0, 'billable_hours': 0, 'cost': 0}
        breakdown['by_mode'][mode]['jobs'] += 1
        breakdown['by_mode'][mode]['billable_hours'] += job['billable_hours']
        breakdown['by_mode'][mode]['cost'] += job['cost']
        
        # By architecture + mode
        arch_mode = f"{arch}-{mode}"
        if arch_mode not in breakdown['by_architecture_mode']:
            breakdown['by_architecture_mode'][arch_mode] = {'jobs': 0, 'billable_hours': 0, 'cost': 0}
        breakdown['by_architecture_mode'][arch_mode]['jobs'] += 1
        breakdown['by_architecture_mode'][arch_mode]['billable_hours'] += job['billable_hours']
        breakdown['by_architecture_mode'][arch_mode]['cost'] += job['cost']
    
    return breakdown

def display_cost_summary(breakdown, jobs=None):
    """Display cost summary in a formatted table."""
    CONSOLE.print("\n💰 PSA Experiment Cost Summary", style="bold green")
    CONSOLE.print(f"   Instance: {INSTANCE_TYPE} @ ${INSTANCE_COST_PER_HOUR}/hr", style="dim")
    CONSOLE.print(f"   Region: {AWS_REGION}", style="dim")
    
    # Total summary
    total = breakdown['total']
    concurrency = breakdown['concurrency']
    
    CONSOLE.print(f"\n📊 Total: {total['jobs']} jobs, {total['billable_hours']:.1f} hours, ${total['cost']:.2f}", style="bold")
    
    # Concurrency information
    CONSOLE.print(f"\n⏱️  Concurrency Analysis:", style="bold blue")
    CONSOLE.print(f"   Wall-clock time: {concurrency['max_wall_clock_hours']:.1f} hours", style="dim")
    CONSOLE.print(f"   Longest job: {concurrency['max_wall_clock_job']}", style="dim")
    CONSOLE.print(f"   Architecture: {concurrency['max_wall_clock_architecture']}", style="dim")
    CONSOLE.print(f"   Mode: {concurrency['max_wall_clock_mode']}", style="dim")
    
    # By status
    if breakdown['by_status']:
        table = Table(title="Cost Breakdown by Job Status")
        table.add_column("Status", style="cyan")
        table.add_column("Jobs", justify="right")
        table.add_column("Hours", justify="right")
        table.add_column("Cost", justify="right")
        
        for status, data in sorted(breakdown['by_status'].items()):
            table.add_row(
                status,
                str(data['jobs']),
                f"{data['billable_hours']:.1f}",
                f"${data['cost']:.2f}"
            )
        
        CONSOLE.print(table)
    
    # By mode
    if breakdown['by_mode']:
        table = Table(title="Cost Breakdown by Ablation Mode")
        table.add_column("Mode", style="cyan")
        table.add_column("Jobs", justify="right")
        table.add_column("Hours", justify="right")
        table.add_column("Cost", justify="right")
        
        for mode, data in sorted(breakdown['by_mode'].items()):
            table.add_row(
                mode,
                str(data['jobs']),
                f"{data['billable_hours']:.1f}",
                f"${data['cost']:.2f}"
            )
        
        CONSOLE.print(table)
    
    # Top 5 longest-running jobs
    if jobs:
        # Sort jobs by billable hours (descending)
        sorted_jobs = sorted(jobs, key=lambda x: x['billable_hours'], reverse=True)
        
        table = Table(title="Top 5 Longest-Running Jobs")
        table.add_column("Rank", style="cyan", justify="right")
        table.add_column("Job Name", style="cyan")
        table.add_column("Architecture", style="cyan")
        table.add_column("Mode", style="cyan")
        table.add_column("Hours", justify="right")
        table.add_column("Cost", justify="right")
        
        for i, job in enumerate(sorted_jobs[:5], 1):
            job_info = parse_job_name(job['name'])
            table.add_row(
                str(i),
                job['name'][:30] + "..." if len(job['name']) > 30 else job['name'],
                job_info['architecture'],
                job_info['mode'],
                f"{job['billable_hours']:.1f}",
                f"${job['cost']:.2f}"
            )
        
        CONSOLE.print(table)

def save_detailed_report(jobs, breakdown, output_file="cost_report.json"):
    """Save detailed cost report to JSON file."""
    report = {
        'metadata': {
            'instance_type': INSTANCE_TYPE,
            'cost_per_hour': INSTANCE_COST_PER_HOUR,
            'region': AWS_REGION,
            'generated_at': datetime.now().isoformat(),
            'total_jobs': len(jobs),
            'total_cost': breakdown['total']['cost'],
            'total_hours': breakdown['total']['billable_hours']
        },
        'breakdown': breakdown,
        'jobs': jobs
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    CONSOLE.print(f"📄 Detailed report saved to {output_file}", style="green")

def main():
    """Main function to estimate SageMaker costs."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Estimate SageMaker training costs for PSA experiments")
    parser.add_argument("--jobs-file", type=str, default="aws/sagemaker/requested-jobs.txt",
                       help="Path to jobs file (default: aws/sagemaker/requested-jobs.txt)")
    parser.add_argument("--output-file", type=str, default="results/sagemaker_cost_report.json",
                       help="Output file for detailed report (default: results/sagemaker_cost_report.json)")
    
    args = parser.parse_args()
    
    CONSOLE.print("💰 SageMaker Cost Estimator for PSA Experiments", style="bold blue")
    CONSOLE.print(f"   Instance: {INSTANCE_TYPE} @ ${INSTANCE_COST_PER_HOUR}/hr", style="dim")
    CONSOLE.print(f"   Region: {AWS_REGION}", style="dim")
    CONSOLE.print(f"   Jobs file: {args.jobs_file}", style="dim")
    
    # Initialize SageMaker client
    sagemaker_client = get_sagemaker_client()
    if not sagemaker_client:
        return
    
    # Load requested jobs from file
    job_names = load_requested_jobs(args.jobs_file)
    if not job_names:
        return
    
    # Get detailed job information
    jobs = get_job_details(sagemaker_client, job_names)
    if not jobs:
        CONSOLE.print("🔴 No job details retrieved", style="red")
        return
    
    # Calculate cost breakdown
    breakdown = calculate_cost_breakdown(jobs)
    
    # Display summary
    display_cost_summary(breakdown, jobs)
    
    # Save detailed report
    save_detailed_report(jobs, breakdown, args.output_file)
    
    CONSOLE.print("\n✅ Cost estimation completed!", style="green")

if __name__ == "__main__":
    main() 