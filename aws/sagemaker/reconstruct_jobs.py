#!/usr/bin/env python3
"""
Simple script to reconstruct requested-jobs.txt from S3 bucket scanning.

This script scans the S3 bucket for PSA job directories and reconstructs
the requested-jobs.txt file with all found job names.

Usage: python reconstruct_jobs.py
"""

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from pathlib import Path
import os

# Load environment variables
script_dir = Path(__file__).resolve().parent
dotenv_path = script_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "scrypted-ai-training")
S3_PREFIX = os.getenv("S3_PREFIX", "psa-experiment")

def main():
    """Reconstruct requested-jobs.txt from S3 bucket."""
    print(f"🔍 Scanning S3 bucket {S3_BUCKET_NAME} for PSA jobs...")
    
    try:
        # Initialize S3 client
        session = boto3.Session(region_name=AWS_REGION)
        s3_client = session.client('s3')
        
        # Scan for job directories
        job_names = []
        paginator = s3_client.get_paginator('list_objects_v2')
        
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
        
        # Sort job names
        job_names.sort()
        
        print(f"✅ Found {len(job_names)} jobs in S3 bucket")
        
        # Write to requested-jobs.txt
        jobs_file = script_dir / "requested-jobs.txt"
        with open(jobs_file, 'w') as f:
            for job_name in job_names:
                f.write(f"{job_name}\n")
        
        print(f"📝 Reconstructed {jobs_file} with {len(job_names)} jobs")
        
        # Show first few jobs as preview
        if job_names:
            print("\nPreview of first 5 jobs:")
            for job in job_names[:5]:
                print(f"  {job}")
            if len(job_names) > 5:
                print(f"  ... and {len(job_names) - 5} more")
        
        return 0
        
    except ClientError as e:
        print(f"🔴 Error scanning S3 bucket: {e}")
        return 1
    except Exception as e:
        print(f"🔴 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 