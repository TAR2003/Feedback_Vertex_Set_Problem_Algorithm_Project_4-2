[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Install,
    [ValidateSet('Release','Debug','RelWithDebInfo','MinSizeRel')]
    [string]$BuildType = 'Release',
    [int]$Jobs = 0,
    [Alias('?')]
    [switch]$Help
)

if ($Help) {
    Write-Host 'Usage: .\build_cpp.ps1 [-Clean] [-Install] [-BuildType Release|Debug|RelWithDebInfo|MinSizeRel] [-Jobs N]'
    Write-Host 'Examples:'
    Write-Host '  .\build_cpp.ps1'
    Write-Host '  .\build_cpp.ps1 -Clean -Jobs 8'
    Write-Host '  .\build_cpp.ps1 -Install'
    exit 0
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$args = @('build_engine.py', '--build-type', $BuildType)
if ($Clean) { $args += '--clean' }
if ($Install) { $args += '--install' }
if ($Jobs -gt 0) { $args += @('--jobs', $Jobs) }

python @args
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
