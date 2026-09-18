# PowerShell uninstaller for Traffic Annotator on Windows
# Counterpart to uninstall.sh on Linux
[CmdletBinding()]
param (
    [string]$InstallDir = "$env:LOCALAPPDATA\TrafficAnnotator\app"
)

$ErrorActionPreference = "Continue"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "     Traffic Annotator - Windows Uninstaller" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Remove Shortcuts
$ProgramsDir = [Environment]::GetFolderPath([Environment+SpecialFolder]::Programs)
$StartMenuShortcut = Join-Path $ProgramsDir "Traffic Annotator.lnk"
if (Test-Path $StartMenuShortcut) {
    Remove-Item -Path $StartMenuShortcut -Force
    Write-Host "Removed Start Menu shortcut: $StartMenuShortcut" -ForegroundColor Green
}

$DesktopDir = [Environment]::GetFolderPath([Environment+SpecialFolder]::Desktop)
$DesktopShortcut = Join-Path $DesktopDir "Traffic Annotator.lnk"
if (Test-Path $DesktopShortcut) {
    Remove-Item -Path $DesktopShortcut -Force
    Write-Host "Removed Desktop shortcut: $DesktopShortcut" -ForegroundColor Green
}

# 2. Remove Command-line launcher
$BinLauncher = "$env:USERPROFILE\.local\bin\traffic-annotator.cmd"
if (Test-Path $BinLauncher) {
    Remove-Item -Path $BinLauncher -Force
    Write-Host "Removed CLI launcher: $BinLauncher" -ForegroundColor Green
}

# 3. Remove application installation directory
if (Test-Path $InstallDir) {
    Remove-Item -Path $InstallDir -Recurse -Force
    Write-Host "Removed application files: $InstallDir" -ForegroundColor Green
}

Write-Host ""
Write-Host "Traffic Annotator uninstalled successfully." -ForegroundColor Green
Write-Host "Note: User datasets (%USERPROFILE%\TrafficAnnotator\datasets), logs, and cache were preserved." -ForegroundColor Yellow
Write-Host ""
