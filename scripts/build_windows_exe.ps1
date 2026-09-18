# Windows PowerShell build script for Traffic Annotator standalone executable
[CmdletBinding()]
param (
    [switch]$Clean = $true,
    [switch]$Zip = $true,
    [switch]$Console = $false
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = Split-Path -Parent $ScriptDir

Set-Location $ProjectDir

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "   Traffic Annotator - Windows Executable Builder" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Check for Python
$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCmd) {
    Write-Error "Python is required but was not found in PATH. Please install Python 3.12+."
}

# Run build script via uv or python
$UvCmd = Get-Command uv -ErrorAction SilentlyContinue
$BuildArgs = @("scripts/build_windows_exe.py")
if ($Clean) { $BuildArgs += "--clean" }
if ($Zip) { $BuildArgs += "--zip" }
if ($Console) { $BuildArgs += "--console" }

if ($UvCmd) {
    Write-Host "[*] Building with uv..." -ForegroundColor Green
    & uv run python @BuildArgs
} else {
    Write-Host "[*] Building with python..." -ForegroundColor Green
    & python @BuildArgs
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[SUCCESS] Standalone distribution created in dist\TrafficAnnotator" -ForegroundColor Green
    Write-Host "Zip package created in dist\TrafficAnnotator-windows-x64.zip" -ForegroundColor Green
} else {
    Write-Error "Build failed with exit code $LASTEXITCODE"
}
