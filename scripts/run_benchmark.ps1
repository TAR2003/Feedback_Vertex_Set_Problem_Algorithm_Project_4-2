# PowerShell Benchmark Script for FVS Algorithms
# Windows-compatible version of run_benchmark.sh

$OUTPUT_DIR = "benchmark_results"
$GRAPH_DIR = "data/graphs"
$RESULTS_CSV = "$OUTPUT_DIR/all_results.csv"
$FVS_EXE = "build/Release/fvs.exe"
$GENERATE_EXE = "build/Release/generate_graphs.exe"

# Create output directory
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

# Clear previous results
if (Test-Path $RESULTS_CSV) {
    Remove-Item $RESULTS_CSV
}

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "FVS Algorithm Benchmark Suite" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if graphs exist, if not generate them
if (-not (Test-Path "$GRAPH_DIR/*.txt")) {
    Write-Host "Graphs not found. Generating benchmark graphs..." -ForegroundColor Yellow
    if (Test-Path $GENERATE_EXE) {
        & $GENERATE_EXE $GRAPH_DIR
    } else {
        Write-Host "Error: generate_graphs.exe not found. Please build the project first." -ForegroundColor Red
        exit 1
    }
}

# Array of algorithms to test
$ALGORITHMS = @("twoapprox", "greedy", "ic", "kernelbst", "ga", "memetic")
$K_VALUES = @(5, 10, 15, 20)

# Find all graph files
$GRAPH_FILES = Get-ChildItem -Path $GRAPH_DIR -Filter "*.txt" | Sort-Object Name

if ($GRAPH_FILES.Count -eq 0) {
    Write-Host "Error: No graph files found in $GRAPH_DIR" -ForegroundColor Red
    exit 1
}

$TOTAL_GRAPHS = $GRAPH_FILES.Count
Write-Host "Found $TOTAL_GRAPHS graph files" -ForegroundColor Green
Write-Host ""

$counter = 0

# Run benchmarks
foreach ($graph_file in $GRAPH_FILES) {
    $counter++
    $graph_name = $graph_file.Name
    $graph_path = $graph_file.FullName
    
    Write-Host "[$counter/$TOTAL_GRAPHS] Testing $graph_name..." -ForegroundColor Cyan
    
    # Test approximation and heuristic algorithms (no k parameter)
    foreach ($alg in @("twoapprox", "greedy", "ga", "memetic")) {
        Write-Host "  Running $alg..." -ForegroundColor Gray
        
        if ($alg -eq "ga" -or $alg -eq "memetic") {
            # Run with reasonable population/generation for benchmarking
            & $FVS_EXE -i $graph_path -a $alg -o $RESULTS_CSV --ga-pop 50 --ga-gen 100 2>$null
        } else {
            & $FVS_EXE -i $graph_path -a $alg -o $RESULTS_CSV 2>$null
        }
    }
    
    # Test exact algorithms (with k parameter)
    # Only test on smaller graphs to avoid long runtimes
    $lines = Get-Content $graph_path
    $edge_count = ($lines | Where-Object { $_ -notmatch '^#' }).Count
    $graph_size = [Math]::Sqrt($edge_count * 2) + 10  # Rough estimate
    
    if ($graph_size -le 50) {
        foreach ($alg in @("ic", "kernelbst")) {
            foreach ($k in $K_VALUES) {
                Write-Host "  Running $alg with k=$k..." -ForegroundColor Gray
                # Use timeout (requires Windows 10+)
                $job = Start-Job -ScriptBlock {
                    param($exe, $graph, $alg, $k, $csv)
                    & $exe -i $graph -a $alg -k $k -o $csv 2>$null
                } -ArgumentList $FVS_EXE, $graph_path, $alg, $k, $RESULTS_CSV
                
                $job | Wait-Job -Timeout 60 | Out-Null
                if ($job.State -eq "Running") {
                    $job | Stop-Job
                    Write-Host "    (timeout)" -ForegroundColor Yellow
                }
                $job | Remove-Job -Force
            }
        }
    } else {
        Write-Host "  Skipping exact algorithms (graph too large)" -ForegroundColor Yellow
    }
    
    Write-Host ""
}

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Benchmark Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Results saved to: $RESULTS_CSV" -ForegroundColor Green
Write-Host ""

if (Test-Path $RESULTS_CSV) {
    $line_count = (Get-Content $RESULTS_CSV).Count
    Write-Host "Summary: $line_count lines in results file" -ForegroundColor Green
} else {
    Write-Host "Warning: No results file generated" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "You can analyze results using:" -ForegroundColor Cyan
Write-Host "  - Excel: Open $RESULTS_CSV" -ForegroundColor Gray
Write-Host "  - Python: pandas.read_csv('$RESULTS_CSV')" -ForegroundColor Gray
Write-Host "  - R: read.csv('$RESULTS_CSV')" -ForegroundColor Gray
Write-Host ""
