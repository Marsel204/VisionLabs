# PowerShell installer for Traffic Annotator on Windows
# Counterpart to install.sh on Linux
[CmdletBinding()]
param (
    [switch]$CpuOnly = $false,
    [string]$InstallDir = "$env:LOCALAPPDATA\TrafficAnnotator\app",
    [switch]$NoShortcuts = $false,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host @"
Usage: .\install.ps1 [options]

Installs Traffic Annotator for the current Windows user.

Options:
  -CpuOnly          Force standard CPU-only installation even if CUDA GPU is present.
  -InstallDir PATH  Override installation directory (default: %LOCALAPPDATA%\TrafficAnnotator\app).
  -NoShortcuts      Do not create Start Menu or Desktop shortcuts.
  -Help             Show this help message.
"@
    exit 0
}

$ErrorActionPreference = "Stop"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "     Traffic Annotator - Windows User Installer" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

$SourceDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if ($SourceDir -eq $InstallDir) {
    Write-Error "Please run install.ps1 from the source checkout, not inside the target installation directory."
}

# 1. Check Python or uv
Write-Host "[1/5] Checking Python environment..." -ForegroundColor Yellow
$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
$HasValidPython = $false

if ($PythonCmd) {
    $PyVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    $Major, $Minor = $PyVersion.Split(".")
    if ([int]$Major -gt 3 -or ([int]$Major -eq 3 -and [int]$Minor -ge 12)) {
        $HasValidPython = $true
        Write-Host "      Found Python $PyVersion ($($PythonCmd.Source))" -ForegroundColor Green
    } else {
        Write-Host "      Found Python $PyVersion, but Python 3.12 or newer is required." -ForegroundColor Yellow
    }
}

# 2. Check or install uv
Write-Host "[2/5] Checking uv package manager..." -ForegroundColor Yellow
$UvCmd = Get-Command uv -ErrorAction SilentlyContinue

if (-not $UvCmd) {
    Write-Host "      uv is not installed. Installing uv for Windows..." -ForegroundColor Cyan
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $Env:PATH = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$Env:PATH"
        $UvCmd = Get-Command uv -ErrorAction SilentlyContinue
    } catch {
        Write-Host "      Could not automatically download uv via web script." -ForegroundColor Yellow
    }
}

if (-not $UvCmd -and $HasValidPython) {
    Write-Host "      Installing uv via pip..." -ForegroundColor Cyan
    & python -m pip install --user uv
    $UvCmd = Get-Command uv -ErrorAction SilentlyContinue
}

if (-not $UvCmd) {
    Write-Error "uv could not be found or installed. Please install uv from https://astral.sh/uv or install Python 3.12+."
}
Write-Host "      Using uv: $(& uv --version)" -ForegroundColor Green

# 3. Copy source files to installation directory
Write-Host "[3/5] Installing files to $InstallDir..." -ForegroundColor Yellow
if (Test-Path $InstallDir) {
    Write-Host "      Updating existing installation..." -ForegroundColor Cyan
} else {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

$ExcludeItems = @(".venv", ".git", "__pycache__", ".pytest_cache", "dist", "build")
Get-ChildItem -Path $SourceDir -Force | Where-Object { $_.Name -notin $ExcludeItems } | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $InstallDir -Recurse -Force
}

# 4. Synchronize dependencies using uv
Write-Host "[4/5] Installing application dependencies..." -ForegroundColor Yellow
Set-Location $InstallDir

if ($CpuOnly) {
    Write-Host "      Syncing locked dependencies in CPU mode..." -ForegroundColor Cyan
    & uv sync --directory $InstallDir --locked --no-dev
} else {
    Write-Host "      Syncing locked dependencies..." -ForegroundColor Cyan
    & uv sync --directory $InstallDir --locked --no-dev
}

# Verify imports
Write-Host "      Verifying PySide6 and PyTorch installation..." -ForegroundColor Cyan
& uv run --directory $InstallDir python -c "import PySide6, torch; print(f'PySide6 {PySide6.__version__}; PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}')"

# 5. Create Launchers and Shortcuts
Write-Host "[5/5] Configuring Windows launchers and shortcuts..." -ForegroundColor Yellow

$BinDir = "$env:USERPROFILE\.local\bin"
if (-not (Test-Path $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
}

# Command-line batch launcher: traffic-annotator.cmd
$CmdLauncher = Join-Path $BinDir "traffic-annotator.cmd"
@"
@echo off
uv run --directory "$InstallDir" traffic-annotator %*
"@ | Set-Content -Path $CmdLauncher -Encoding Ascii

# Desktop runner script inside InstallDir
$LocalRunner = Join-Path $InstallDir "run.bat"
@"
@echo off
cd /d "%~dp0"
start "" uv run traffic-annotator %*
"@ | Set-Content -Path $LocalRunner -Encoding Ascii

# Create Windows Shortcuts (.lnk) via WScript.Shell
if (-not $NoShortcuts) {
    try {
        $WshShell = New-Object -ComObject WScript.Shell
        
        # Start Menu Shortcut
        $ProgramsDir = [Environment]::GetFolderPath([Environment+SpecialFolder]::Programs)
        $StartMenuShortcut = Join-Path $ProgramsDir "Traffic Annotator.lnk"
        $Shortcut = $WshShell.CreateShortcut($StartMenuShortcut)
        $Shortcut.TargetPath = "cmd.exe"
        $Shortcut.Arguments = "/c start `"`" `"$LocalRunner`""
        $Shortcut.WorkingDirectory = $InstallDir
        $Shortcut.Description = "AI-assisted traffic annotation application"
        $Shortcut.WindowStyle = 7 # Minimized
        $Shortcut.Save()
        Write-Host "      Created Start Menu shortcut: $StartMenuShortcut" -ForegroundColor Green

        # Desktop Shortcut
        $DesktopDir = [Environment]::GetFolderPath([Environment+SpecialFolder]::Desktop)
        $DesktopShortcut = Join-Path $DesktopDir "Traffic Annotator.lnk"
        $DShortcut = $WshShell.CreateShortcut($DesktopShortcut)
        $DShortcut.TargetPath = "cmd.exe"
        $DShortcut.Arguments = "/c start `"`" `"$LocalRunner`""
        $DShortcut.WorkingDirectory = $InstallDir
        $DShortcut.Description = "AI-assisted traffic annotation application"
        $DShortcut.WindowStyle = 7 # Minimized
        $DShortcut.Save()
        Write-Host "      Created Desktop shortcut: $DesktopShortcut" -ForegroundColor Green
    } catch {
        Write-Host "      Notice: Could not automatically create .lnk shortcuts: $_" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "  Traffic Annotator installed successfully on Windows! " -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "Launch with:"
Write-Host "  - Double-click the Desktop or Start Menu shortcut 'Traffic Annotator'"
Write-Host "  - Or run: & `"$LocalRunner`""
Write-Host "  - Or run command: traffic-annotator (if ~/.local/bin is in PATH)"
Write-Host ""
Write-Host "To uninstall:"
Write-Host "  - Run: powershell -ExecutionPolicy Bypass -File `"$InstallDir\uninstall.ps1`""
Write-Host ""
