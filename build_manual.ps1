# Manual build script for FVS project (without CMake)
# Uses g++ directly

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Building FVS Project (Manual Compilation)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$CXX = "g++"
$CXXFLAGS = "-std=c++17 -O3 -Wall -Wextra -march=native -g"
$LDFLAGS = "-lpsapi"  # Link Windows psapi library for memory tracking
$SRC_DIR = "src"
$BUILD_DIR = "build"

# Create build directory if it doesn't exist
if (-not (Test-Path $BUILD_DIR)) {
    New-Item -ItemType Directory -Path $BUILD_DIR | Out-Null
}

# Create Release subdirectory
$RELEASE_DIR = "$BUILD_DIR\Release"
if (-not (Test-Path $RELEASE_DIR)) {
    New-Item -ItemType Directory -Path $RELEASE_DIR | Out-Null
}

Write-Host "`nCompiling source files..." -ForegroundColor Yellow

# Source files for fvs executable
$FVS_SOURCES = @(
    "$SRC_DIR\main.cpp",
    "$SRC_DIR\graph.cpp",
    "$SRC_DIR\utils.cpp",
    "$SRC_DIR\alg_exact.cpp",
    "$SRC_DIR\alg_approx.cpp",
    "$SRC_DIR\genetic.cpp",
    "$SRC_DIR\alg_iterative_compression.cpp",
    "$SRC_DIR\alg_kernelization.cpp",
    "$SRC_DIR\alg_bounded_search_tree.cpp",
    "$SRC_DIR\alg_memetic.cpp"
)

# Source files for generate_graphs executable
$GEN_SOURCES = @(
    "$SRC_DIR\generate_graphs.cpp",
    "$SRC_DIR\graph_generators.cpp",
    "$SRC_DIR\graph.cpp"
)

# Compile fvs executable
Write-Host "Building fvs.exe..." -ForegroundColor Green
$fvs_cmd = "$CXX $CXXFLAGS $($FVS_SOURCES -join ' ') $LDFLAGS -o $RELEASE_DIR\fvs.exe"
Write-Host "  $fvs_cmd" -ForegroundColor DarkGray
Invoke-Expression $fvs_cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ fvs.exe built successfully" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to build fvs.exe" -ForegroundColor Red
    exit 1
}

# Compile generate_graphs executable
Write-Host "`nBuilding generate_graphs.exe..." -ForegroundColor Green
$gen_cmd = "$CXX $CXXFLAGS $($GEN_SOURCES -join ' ') -o $RELEASE_DIR\generate_graphs.exe"
Write-Host "  $gen_cmd" -ForegroundColor DarkGray
Invoke-Expression $gen_cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ generate_graphs.exe built successfully" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to build generate_graphs.exe" -ForegroundColor Red
    exit 1
}

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Build Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "`nExecutables created:"
Write-Host "  - $RELEASE_DIR\fvs.exe"
Write-Host "  - $RELEASE_DIR\generate_graphs.exe"
Write-Host "`nNext steps:"
Write-Host "  1. Generate graphs: .\$RELEASE_DIR\generate_graphs.exe .\data\graphs"
Write-Host "  2. Test: .\$RELEASE_DIR\fvs.exe -i .\data\graphs\sample_triangle.txt -a ic -k 5"
Write-Host "  3. Benchmark: powershell .\scripts\run_benchmark.ps1"
