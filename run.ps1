#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

$DatasetSlug = 'tawkirazizrahman/fvs-synthetic-dataset-20k'
$TargetDir = Join-Path $ScriptDir 'data\synthetic'

$PaceUrl = 'https://heibox.uni-heidelberg.de/f/97634323e3cb4aab8291/?dl=1'
$PaceTar = Join-Path $ScriptDir 'pace_temp.tar.gz'
$PaceTargetDir = Join-Path $ScriptDir 'data\pace2022'
$PaceExtractedFolder = Join-Path $ScriptDir 'data\heuristic_track_final_instances_all'
$PaceTimeout = 60

function Invoke-Python {
    param (
        [Parameter(Mandatory=$true)]
        [string[]]$Args
    )

    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) { $python = Get-Command python3 -ErrorAction SilentlyContinue }
    if (-not $python) { throw 'Python is not found on PATH. Please install Python and add it to PATH.' }

    & $python @Args
}

Write-Host '--- [1/5] Installing dependencies and building the C++ engine ---'
Invoke-Python -Args @('build_engine.py')

Write-Host '--- [2/5] Downloading Kaggle Dataset (Public) ---'
Invoke-Python -Args @('-m', 'pip', 'install', '-q', 'kagglehub')

$pythonCode = @'
import kagglehub
import shutil
import os

print(f'Downloading {os.environ.get("DATASET_SLUG", "' + $DatasetSlug + '")}...')
path = kagglehub.dataset_download("' + $DatasetSlug + '")

target = r"' + $TargetDir + '"
if os.path.exists(target):
    shutil.rmtree(target)
os.makedirs(os.path.dirname(target), exist_ok=True)

shutil.copytree(path, target)
print(f'✅ Kaggle Dataset moved to {target}')
'@
Invoke-Python -Args @('-c', $pythonCode)

Write-Host '--- [3/5] Downloading and Extracting PACE 2022 Dataset ---'
New-Item -ItemType Directory -Path (Join-Path $ScriptDir 'data') -Force | Out-Null

Write-Host 'Downloading PACE 2022 instances...'
Invoke-WebRequest -Uri $PaceUrl -OutFile $PaceTar -UseBasicParsing

if (Test-Path $PaceTargetDir) {
    Remove-Item -Recurse -Force $PaceTargetDir
}

Write-Host 'Extracting archive...'
& tar -xzf $PaceTar -C (Join-Path $ScriptDir 'data')

if (Test-Path $PaceExtractedFolder) {
    Move-Item -Path $PaceExtractedFolder -Destination $PaceTargetDir -Force
    Write-Host "✅ PACE 2022 Dataset moved to $PaceTargetDir"
} else {
    Write-Warning "Expected folder $PaceExtractedFolder not found after extraction. Please verify the archive contents."
}

Remove-Item -Force -ErrorAction SilentlyContinue $PaceTar

Write-Host '--- Remove all previous csv files from the results folder ---'
Get-ChildItem -Path (Join-Path $ScriptDir 'results\*.csv') -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

Write-Host '--- [4/5] Running the default pipeline ---'
Invoke-Python -Args @('experiments/run_pipeline.py', '--mode', 'all', '--algo', 'ALL', '--timeout', '30')

Write-Host '--- [5/5] check the fvs ---'
Invoke-Python -Args @('experiments/brute_force.py')
Invoke-Python -Args @('experiments/fvs_checker.py')

New-Item -ItemType Directory -Path (Join-Path $ScriptDir 'directed_results') -Force | Out-Null
Get-ChildItem -Path (Join-Path $ScriptDir 'directed_results\*.csv') -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path (Join-Path $ScriptDir 'results\directed_*') -File -ErrorAction SilentlyContinue | Move-Item -Destination (Join-Path $ScriptDir 'directed_results') -Force
Invoke-Python -Args @('directed_results/evaluate_fvs_scores.py')

New-Item -ItemType Directory -Path (Join-Path $ScriptDir 'undirected_results') -Force | Out-Null
Get-ChildItem -Path (Join-Path $ScriptDir 'undirected_results\*.csv') -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path (Join-Path $ScriptDir 'results\undirected_*') -File -ErrorAction SilentlyContinue | Move-Item -Destination (Join-Path $ScriptDir 'undirected_results') -Force
Invoke-Python -Args @('undirected_results/evaluate_fvs_scores.py')

Write-Host '--- running the pace pipeline ---'
Invoke-Python -Args @('experiments/benchmark_directed.py', '--algo', 'MA', '--test', 'data/pace2022/', '--pop', '20', '--gens', '100', '--timeout', $PaceTimeout.ToString())
Invoke-Python -Args @('experiments/benchmark_directed.py', '--algo', 'KMA', '--test', 'data/pace2022/', '--pop', '20', '--gens', '100', '--timeout', $PaceTimeout.ToString())
Invoke-Python -Args @('experiments/benchmark_directed.py', '--algo', 'GNN-KMA', '--test', 'data/pace2022/', '--pop', '20', '--gens', '100', '--timeout', $PaceTimeout.ToString())
Invoke-Python -Args @('experiments/benchmark_directed.py', '--algo', 'GNN-KMA-2', '--test', 'data/pace2022/', '--pop', '20', '--gens', '100', '--timeout', $PaceTimeout.ToString())

New-Item -ItemType Directory -Path (Join-Path $ScriptDir 'paceresults') -Force | Out-Null
Get-ChildItem -Path (Join-Path $ScriptDir 'paceresults\*.csv') -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Copy-Item -Path (Join-Path $ScriptDir 'pace2022_winner.csv') -Destination (Join-Path $ScriptDir 'paceresults') -Force
Get-ChildItem -Path (Join-Path $ScriptDir 'results\*') -File -ErrorAction SilentlyContinue | Move-Item -Destination (Join-Path $ScriptDir 'paceresults') -Force
Invoke-Python -Args @('paceresults/evaluate_fvs_scores.py')

Write-Host '--- [5/5] Pipeline finished ---'
Write-Host 'To customize the run, invoke Python directly:'
Write-Host '  python experiments/run_pipeline.py --mode directed --algo MA --include-pace --total-directed 50'
Write-Host '  python experiments/run_pipeline.py --mode undirected --algo IC --prepare-only --total-undirected 100'
