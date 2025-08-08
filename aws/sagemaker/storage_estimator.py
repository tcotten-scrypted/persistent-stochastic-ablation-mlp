#!/usr/bin/env python3
"""
AWS SageMaker Storage Estimator for PSA Experiments

This script analyzes S3 storage usage for all PSA training jobs and reconstructs
the requested-jobs.txt file by scanning the S3 bucket structure.

Author: Tim Cotten @cottenio <tcotten@scrypted.ai, tcotten2@gmu.edu>
"""

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from pathlib import Path
import os
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from datetime import datetime
import argparse

# --- Load environment variables ---
script_dir = Path(__file__).resolve().parent
dotenv_path = script_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)

# --- Configuration ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "scrypted-ai-training")
S3_PREFIX = os.getenv("S3_PREFIX", "psa-experiment")
CONSOLE = Console()

def get_s3_client():
    """Initialize and return S3 client."""
    try:
        session = boto3.Session(region_name=AWS_REGION)
        return session.client('s3')
    except Exception as e:
        CONSOLE.print(f"🔴 Error initializing S3 client: {e}", style="red")
        return None

def list_all_jobs(s3_client):
    """
    Scan S3 bucket and reconstruct the complete list of job names.
    Returns a list of job names found in the bucket.
    """
    CONSOLE.print("🔍 Scanning S3 bucket for job directories...", style="blue")
    
    job_names = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=CONSOLE
    ) as progress:
        task = progress.add_task("Scanning S3 bucket...", total=None)
        
        try:
            pages = paginator.paginate(
                Bucket=S3_BUCKET_NAME,
                Prefix=f"{S3_PREFIX}/",
                Delimiter='/'
            )
            
            for page in pages:
                if 'CommonPrefixes' in page:
                    for prefix in page['CommonPrefixes']:
                        job_path = prefix['Prefix']
                        # Extract job name from path (e.g., "psa-experiment/psa-1x2048-none-2025-08-04-19-53-35-910/")
                        job_name = job_path.rstrip('/').split('/')[-1]
                        if job_name.startswith('psa-'):
                            job_names.append(job_name)
                
                progress.update(task, description=f"Found {len(job_names)} jobs so far...")
        
        except ClientError as e:
            CONSOLE.print(f"🔴 Error scanning S3 bucket: {e}", style="red")
            return []
    
    CONSOLE.print(f"✅ Found {len(job_names)} jobs in S3 bucket", style="green")
    return sorted(job_names)

def estimate_job_storage(s3_client, job_name):
    """
    Estimate storage usage for a single job.
    Returns a dictionary with storage details.
    """
    job_prefix = f"{S3_PREFIX}/{job_name}/"
    
    total_size = 0
    file_count = 0
    file_types = {}
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=job_prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    size = obj['Size']
                    key = obj['Key']
                    total_size += size
                    file_count += 1
                    
                    # Categorize file types
                    if key.endswith('.tar.gz'):
                        file_types['model_archives'] = file_types.get('model_archives', 0) + size
                    elif key.endswith('.json'):
                        file_types['json_files'] = file_types.get('json_files', 0) + size
                    elif key.endswith('.txt'):
                        file_types['text_files'] = file_types.get('text_files', 0) + size
                    elif key.endswith('.log'):
                        file_types['log_files'] = file_types.get('log_files', 0) + size
                    else:
                        file_types['other'] = file_types.get('other', 0) + size
    
    except ClientError as e:
        CONSOLE.print(f"⚠️  Error estimating storage for {job_name}: {e}", style="yellow")
        return None
    
    return {
        'job_name': job_name,
        'total_size': total_size,
        'file_count': file_count,
        'file_types': file_types
    }

def format_size(size_bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def main():
    """Main function to estimate storage for PSA experiments."""
    parser = argparse.ArgumentParser(description="Estimate S3 storage for PSA experiments")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed storage breakdown per job")
    parser.add_argument("--jobs-file", type=str, default=None,
                       help="Custom jobs file path (default: aws/sagemaker/requested-jobs.txt)")
    args = parser.parse_args()
    
    CONSOLE.print("🚀 PSA S3 Storage Estimator", style="bold blue")
    CONSOLE.print(f"   Bucket: {S3_BUCKET_NAME}", style="dim")
    CONSOLE.print(f"   Prefix: {S3_PREFIX}", style="dim")
    CONSOLE.print(f"   Region: {AWS_REGION}", style="dim")
    
    # Initialize S3 client
    s3_client = get_s3_client()
    if not s3_client:
        return 1
    
    # Get job names
    if args.jobs_file:
        jobs_file = Path(args.jobs_file)
    else:
        jobs_file = script_dir / "requested-jobs.txt"
    
    if not jobs_file.exists():
        CONSOLE.print(f"🔴 {jobs_file} not found. Run reconstruct_jobs.py first.", style="red")
        return 1
    
    with open(jobs_file, 'r') as f:
        job_names = [line.strip() for line in f if line.strip()]
    
    CONSOLE.print(f"📋 Loaded {len(job_names)} jobs from {jobs_file}", style="green")
    
    # Estimate storage for all jobs
    CONSOLE.print("📊 Estimating storage usage...", style="blue")
    
    total_storage = 0
    total_files = 0
    job_details = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=CONSOLE
    ) as progress:
        task = progress.add_task("Analyzing storage...", total=len(job_names))
        
        for job_name in job_names:
            progress.update(task, description=f"Analyzing {job_name}...")
            
            storage_info = estimate_job_storage(s3_client, job_name)
            if storage_info:
                job_details.append(storage_info)
                total_storage += storage_info['total_size']
                total_files += storage_info['file_count']
            
            progress.advance(task)
    
    # Display results
    CONSOLE.print("\n📈 Storage Analysis Results", style="bold green")
    
    # Summary table
    summary_table = Table(title="Storage Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total Jobs", str(len(job_names)))
    summary_table.add_row("Total Storage", format_size(total_storage))
    summary_table.add_row("Total Files", f"{total_files:,}")
    summary_table.add_row("Average per Job", format_size(total_storage / len(job_names)))
    
    CONSOLE.print(summary_table)
    
    # Detailed breakdown if requested
    if args.detailed:
        CONSOLE.print("\n📋 Detailed Storage Breakdown", style="bold blue")
        
        detail_table = Table(title="Job Storage Details")
        detail_table.add_column("Job Name", style="cyan")
        detail_table.add_column("Size", style="green")
        detail_table.add_column("Files", style="yellow")
        detail_table.add_column("Types", style="magenta")
        
        for job_info in sorted(job_details, key=lambda x: x['total_size'], reverse=True):
            types_str = ", ".join([f"{k}: {format_size(v)}" for k, v in job_info['file_types'].items()])
            detail_table.add_row(
                job_info['job_name'],
                format_size(job_info['total_size']),
                str(job_info['file_count']),
                types_str
            )
        
        CONSOLE.print(detail_table)
    
    # File type breakdown
    CONSOLE.print("\n📁 File Type Breakdown", style="bold blue")
    
    type_totals = {}
    for job_info in job_details:
        for file_type, size in job_info['file_types'].items():
            type_totals[file_type] = type_totals.get(file_type, 0) + size
    
    type_table = Table(title="File Types")
    type_table.add_column("Type", style="cyan")
    type_table.add_column("Total Size", style="green")
    type_table.add_column("Percentage", style="yellow")
    
    for file_type, total_size in sorted(type_totals.items(), key=lambda x: x[1], reverse=True):
        percentage = (total_size / total_storage) * 100
        type_table.add_row(
            file_type,
            format_size(total_size),
            f"{percentage:.1f}%"
        )
    
    CONSOLE.print(type_table)
    
    # Save detailed report
    report_file = script_dir / "storage_report.json"
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "bucket": S3_BUCKET_NAME,
        "prefix": S3_PREFIX,
        "total_jobs": len(job_names),
        "total_storage_bytes": total_storage,
        "total_storage_human": format_size(total_storage),
        "total_files": total_files,
        "average_per_job": format_size(total_storage / len(job_names)),
        "file_types": type_totals,
        "job_details": job_details
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    CONSOLE.print(f"\n💾 Detailed report saved to {report_file}", style="green")
    
    return 0

if __name__ == "__main__":
    exit(main()) 