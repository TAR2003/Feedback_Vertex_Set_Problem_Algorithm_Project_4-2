#!/usr/bin/env python3
"""
Download and extract PACE 2022 Challenge datasets (Directed graphs only).

Usage:
    python download_pace_data.py

This script:
1. Downloads the PACE 2022 heuristic track instances with progress animation
2. Extracts them directly to data/pace2022/ with extraction progress
3. Automatically removes the .tar.gz file after extraction to save space
4. Flattens all .gr files into a single directory
"""

import os
import sys
import tarfile
import urllib.request
import shutil
import ssl
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "pace2022"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Disable SSL verification (for HTTPS downloads that may have cert issues)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# PACE 2022 heuristic track download link
PACE_URL = "https://heibox.uni-heidelberg.de/f/97634323e3cb4aab8291/?dl=1"
DOWNLOAD_PATH = DATA_DIR / "heuristic_track_final_instances_all.tar.gz"


def download_progress_hook(block_num, block_size, total_size):
    """Display download progress with animation."""
    downloaded = block_num * block_size
    
    if total_size <= 0:
        percent = 0
        remaining = "Unknown"
    else:
        percent = min(100, int(100 * downloaded / total_size))
        remaining_bytes = max(0, total_size - downloaded)
        
        # Convert bytes to human-readable format
        if remaining_bytes < 1024:
            remaining = f"{remaining_bytes}B"
        elif remaining_bytes < 1024 * 1024:
            remaining = f"{remaining_bytes / 1024:.1f}KB"
        elif remaining_bytes < 1024 * 1024 * 1024:
            remaining = f"{remaining_bytes / (1024 * 1024):.1f}MB"
        else:
            remaining = f"{remaining_bytes / (1024 * 1024 * 1024):.1f}GB"

    # Calculate total for display
    if total_size > 0:
        total_mb = total_size / (1024 * 1024)
        downloaded_mb = downloaded / (1024 * 1024)
        size_str = f"{downloaded_mb:.1f}MB / {total_mb:.1f}MB"
    else:
        size_str = f"{downloaded / (1024 * 1024):.1f}MB"

    # Progress bar animation
    bar_length = 40
    filled = int(bar_length * percent / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    sys.stdout.write(f"\r[{bar}] {percent}% | {size_str} | Remaining: {remaining}")
    sys.stdout.flush()


def extraction_progress_hook(members):
    """Wrapper to show extraction progress."""
    total = len(members)
    for i, member in enumerate(members):
        percent = int(100 * i / total) if total > 0 else 0
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        sys.stdout.write(f"\r[{bar}] {percent}% | Extracting: {i}/{total} files")
        sys.stdout.flush()
        yield member
    print()  # Newline after extraction


print("Downloading PACE 2022 heuristic track data...")
print(f"URL: {PACE_URL}")
print(f"Destination: {DOWNLOAD_PATH}\n")

try:
    # Create a custom opener with SSL context
    https_handler = urllib.request.HTTPSHandler(context=ssl_context)
    opener = urllib.request.build_opener(https_handler)
    urllib.request.install_opener(opener)
    
    urllib.request.urlretrieve(PACE_URL, DOWNLOAD_PATH, download_progress_hook)
    print("\n✓ Download complete!")
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    exit(1)

print("\nExtracting archive with progress...")
try:
    with tarfile.open(DOWNLOAD_PATH, "r:gz") as tar:
        members = tar.getmembers()
        for member in extraction_progress_hook(members):
            tar.extract(member, path=DATA_DIR)
    print("✓ Extraction complete!")
except Exception as e:
    print(f"✗ Extraction failed: {e}")
    exit(1)

# Clean up and organize files
print("\nOrganizing files...")

# The extracted folder structure
extracted_dir = DATA_DIR / "heuristic_track_final_instances_all"

# The PACE files are in extracted_dir named as h_001, h_002, etc. (no extension)
graph_files = []

# Look for h_NNN files directly in extracted_dir
if extracted_dir.exists():
    for item in extracted_dir.iterdir():
        # Check if it's a file starting with 'h_' and is a graph file
        if item.is_file() and item.name.startswith('h_'):
            graph_files.append(item)

if not graph_files:
    print("✗ No graph files found in extracted archive!")
    print(f"Debug: Checking {extracted_dir}")
    if extracted_dir.exists():
        items = list(extracted_dir.iterdir())
        print(f"  Found {len(items)} items:")
        for item in items[:5]:
            print(f"    {item} (dir={item.is_dir()})")
    exit(1)

# Move all graph files directly to DATA_DIR (flatten structure)
files_collected = 0
for graph_file in graph_files:
    dest = DATA_DIR / graph_file.name
    # Handle duplicate names by renaming
    if dest.exists():
        base = dest.stem
        ext = dest.suffix if dest.suffix else ""
        counter = 1
        while dest.exists():
            if ext:
                dest = DATA_DIR / f"{base}_{counter}{ext}"
            else:
                dest = DATA_DIR / f"{base}_{counter}"
            counter += 1
    shutil.move(str(graph_file), str(dest))
    files_collected += 1

print(f"✓ Moved {files_collected} graph files to pace2022/")

# Delete the extraction folder and all subdirectories
print("\nCleaning up temporary files...")
if extracted_dir.exists():
    shutil.rmtree(extracted_dir)
    print(f"✓ Deleted heuristic_track_final_instances_all/ folder")

# Delete tar.gz file
if DOWNLOAD_PATH.exists():
    size_mb = DOWNLOAD_PATH.stat().st_size / (1024 * 1024)
    DOWNLOAD_PATH.unlink()
    print(f"✓ Deleted {DOWNLOAD_PATH.name} ({size_mb:.1f}MB freed)")

# Count final files (look for h_* files without extension)
final_files = [f for f in DATA_DIR.glob("h_*") if f.is_file()]

print("\n" + "="*70)
print("✓ PACE 2022 data setup complete!")
print("="*70)
print(f"\nData location: {DATA_DIR}")
print(f"Total graph files: {len(final_files)}")
print(f"\nUsage examples:")
print(f"  python experiments/benchmark_directed.py --algo ALL --test data/pace2022/")
print(f"  python experiments/benchmark_directed.py --algo MA --test data/pace2022/ --output pace_results.csv")
print(f"  python experiments/benchmark_directed.py --algo IC --test data/pace2022/ --output pace_results.csv")
print(f"  python experiments/benchmark_directed.py --algo BST --test data/pace2022/ --output pace_results.csv")
