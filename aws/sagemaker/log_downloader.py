#!/usr/bin/env python3
"""
AWS SageMaker CloudWatch Log Downloader for PSA Experiments

This script downloads all CloudWatch training logs for PSA experiments and
organizes them in a structured directory for further analysis.

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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from datetime import datetime, timedelta
import argparse
import time

# --- Load environment variables ---
script_dir = Path(__file__).resolve().parent
dotenv_path = script_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)

# --- Configuration ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "scrypted-ai-training")
S3_PREFIX = os.getenv("S3_PREFIX", "psa-experiment")
CONSOLE = Console()

def get_aws_clients():
    """Initialize and return AWS clients."""
    try:
        session = boto3.Session(region_name=AWS_REGION)
        return {
            'logs': session.client('logs'),
            'sagemaker': session.client('sagemaker'),
            's3': session.client('s3')
        }
    except Exception as e:
        CONSOLE.print(f"🔴 Error initializing AWS clients: {e}", style="red")
        return None

def get_job_names_from_file():
    """Read job names from requested-jobs.txt file."""
    jobs_file = script_dir / "requested-jobs.txt"
    if not jobs_file.exists():
        CONSOLE.print(f"🔴 {jobs_file} not found. Run with --reconstruct-jobs first.", style="red")
        return []
    
    with open(jobs_file, 'r') as f:
        job_names = [line.strip() for line in f if line.strip()]
    
    CONSOLE.print(f"📋 Loaded {len(job_names)} jobs from {jobs_file}", style="green")
    return job_names

def get_job_names_from_s3(s3_client):
    """Get job names by scanning S3 bucket."""
    CONSOLE.print("🔍 Scanning S3 bucket for job directories...", style="blue")
    
    job_names = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
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
                    job_name = job_path.rstrip('/').split('/')[-1]
                    if job_name.startswith('psa-'):
                        job_names.append(job_name)
    
    except ClientError as e:
        CONSOLE.print(f"🔴 Error scanning S3 bucket: {e}", style="red")
        return []
    
    CONSOLE.print(f"✅ Found {len(job_names)} jobs in S3 bucket", style="green")
    return sorted(job_names)

def get_log_group_name(job_name):
    """Return the CloudWatch Logs group used by SageMaker Training Jobs (constant)."""
    # SageMaker Training Jobs write all logs under this single group, with per-job stream prefixes
    return "/aws/sagemaker/TrainingJobs"

def download_job_logs(logs_client, job_name, output_dir):
    """
    Download all log streams for a specific job by looking up streams under the
    SageMaker TrainingJobs log group with the job name as a prefix.
    Returns (success, [downloaded_file_paths]).
    """
    log_group_name = get_log_group_name(job_name)
    job_output_dir = output_dir / job_name
    job_output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = []

    try:
        # Discover all streams that begin with this job name
        paginator = logs_client.get_paginator('describe_log_streams')
        # When using logStreamNamePrefix, CloudWatch Logs API does not allow orderBy/descending
        pages = paginator.paginate(
            logGroupName=log_group_name,
            logStreamNamePrefix=f"{job_name}/"
        )

        log_streams = []
        for page in pages:
            if 'logStreams' in page:
                log_streams.extend(page['logStreams'])

        # Sort locally by lastEventTimestamp if available (desc)
        try:
            log_streams.sort(key=lambda s: s.get('lastEventTimestamp', 0), reverse=True)
        except Exception:
            pass

        if not log_streams:
            CONSOLE.print(f"⚠️  No log streams found for {job_name}", style="yellow")
            return False, []

        # Download each log stream with full pagination of events
        for stream in log_streams:
            stream_name = stream['logStreamName']  # e.g., psa-.../algo-1-12345

            # Preserve nested structure under the job directory
            stream_rel_path = Path(stream_name)  # contains a '/'
            stream_path = job_output_dir / stream_rel_path
            stream_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = stream_path.with_suffix('.log')

            try:
                next_token = None
                with open(output_file, 'w') as f:
                    while True:
                        kwargs = {
                            'logGroupName': log_group_name,
                            'logStreamName': stream_name,
                            'startFromHead': True,
                        }
                        if next_token:
                            kwargs['nextToken'] = next_token

                        resp = logs_client.get_log_events(**kwargs)
                        events = resp.get('events', [])

                        for event in events:
                            timestamp = datetime.fromtimestamp(event['timestamp'] / 1000).isoformat()
                            message = event.get('message', '').rstrip('\n')
                            f.write(f"[{timestamp}] {message}\n")

                        prev_token = next_token
                        next_token = resp.get('nextForwardToken')
                        # Stop when tokens stop advancing or no events
                        if not events or next_token == prev_token:
                            break

                downloaded_files.append(output_file)

            except ClientError as e:
                CONSOLE.print(f"⚠️  Error downloading stream {stream_name} for {job_name}: {e}", style="yellow")
                continue

        return True, downloaded_files

    except ClientError as e:
        if e.response['Error'].get('Code') == 'ResourceNotFoundException':
            CONSOLE.print(f"⚠️  Log group not found for {job_name} (group: {log_group_name})", style="yellow")
            return False, []
        CONSOLE.print(f"🔴 Error downloading logs for {job_name}: {e}", style="red")
        return False, []

def create_log_summary(job_name, log_files):
    """Create a summary of downloaded logs for a job."""
    summary = {
        'job_name': job_name,
        'timestamp': datetime.now().isoformat(),
        'log_files': [str(f) for f in log_files],
        'total_files': len(log_files),
        'total_size': sum(f.stat().st_size for f in log_files if f.exists())
    }
    
    summary_file = Path(f"results/logs/{job_name}/summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    """Main function to download all CloudWatch logs."""
    parser = argparse.ArgumentParser(description="Download CloudWatch logs for PSA experiments")
    parser.add_argument("--reconstruct-jobs", action="store_true",
                       help="Reconstruct job list from S3 bucket first")
    parser.add_argument("--output-dir", type=str, default="results/logs",
                       help="Output directory for logs (default: results/logs)")
    parser.add_argument("--jobs-file", type=str, default=None,
                       help="Custom jobs file path")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download of existing logs")
    args = parser.parse_args()
    
    CONSOLE.print("🚀 PSA CloudWatch Log Downloader", style="bold blue")
    CONSOLE.print(f"   Region: {AWS_REGION}", style="dim")
    CONSOLE.print(f"   Output: {args.output_dir}", style="dim")
    
    # Initialize AWS clients
    clients = get_aws_clients()
    if not clients:
        return 1
    
    # Get job names
    if args.reconstruct_jobs:
        job_names = get_job_names_from_s3(clients['s3'])
        if job_names:
            jobs_file = script_dir / "requested-jobs.txt"
            with open(jobs_file, 'w') as f:
                for job_name in job_names:
                    f.write(f"{job_name}\n")
            CONSOLE.print(f"✅ Reconstructed {jobs_file} with {len(job_names)} jobs", style="green")
    elif args.jobs_file:
        with open(args.jobs_file, 'r') as f:
            job_names = [line.strip() for line in f if line.strip()]
    else:
        job_names = get_job_names_from_file()
    
    if not job_names:
        CONSOLE.print("🔴 No jobs found", style="red")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    CONSOLE.print(f"📁 Output directory: {output_dir.absolute()}", style="green")
    
    # Download logs for each job
    successful_downloads = 0
    failed_downloads = 0
    total_files = 0
    total_size = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=CONSOLE
    ) as progress:
        task = progress.add_task("Downloading logs...", total=len(job_names))
        
        for job_name in job_names:
            progress.update(task, description=f"Downloading {job_name}...")
            
            # Check if already downloaded
            job_dir = output_dir / job_name
            if job_dir.exists() and not args.force:
                CONSOLE.print(f"⏭️  Skipping {job_name} (already exists)", style="yellow")
                progress.advance(task)
                continue
            
            success, log_files = download_job_logs(clients['logs'], job_name, output_dir)
            
            if success:
                successful_downloads += 1
                total_files += len(log_files)
                total_size += sum(f.stat().st_size for f in log_files if f.exists())
                
                # Create summary
                summary = create_log_summary(job_name, log_files)
                
                progress.update(task, description=f"✅ {job_name} ({len(log_files)} files)")
            else:
                failed_downloads += 1
                progress.update(task, description=f"❌ {job_name}")
            
            progress.advance(task)
    
    # Display results
    CONSOLE.print("\n📊 Download Summary", style="bold green")
    
    summary_table = Table(title="Download Results")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total Jobs", str(len(job_names)))
    summary_table.add_row("Successful Downloads", str(successful_downloads))
    summary_table.add_row("Failed Downloads", str(failed_downloads))
    summary_table.add_row("Total Log Files", f"{total_files:,}")
    summary_table.add_row("Total Size", f"{total_size / (1024*1024):.2f} MB")
    
    CONSOLE.print(summary_table)
    
    # Create overall summary
    overall_summary = {
        "timestamp": datetime.now().isoformat(),
        "region": AWS_REGION,
        "output_directory": str(output_dir.absolute()),
        "total_jobs": len(job_names),
        "successful_downloads": successful_downloads,
        "failed_downloads": failed_downloads,
        "total_log_files": total_files,
        "total_size_bytes": total_size,
        "total_size_mb": total_size / (1024*1024)
    }
    
    summary_file = output_dir / "download_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    CONSOLE.print(f"\n💾 Overall summary saved to {summary_file}", style="green")
    
    if failed_downloads > 0:
        CONSOLE.print(f"\n⚠️  {failed_downloads} jobs failed to download. Check the logs above for details.", style="yellow")
    
    return 0

if __name__ == "__main__":
    exit(main()) 