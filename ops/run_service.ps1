# Supervisor loop for the Vision Service prototype (main.py / uvicorn).
#
# Starts uvicorn, waits for it to exit (crash, kill, oneDNN error, etc.),
# then restarts it after a short delay - forever. Writes the child PID to
# logs/uvicorn.pid so health_check.ps1 can target the exact process instead
# of pattern-matching python.exe (see CLAUDE.md gotcha #5 - "ghost process
# on port 8000").
#
# Run this directly for a foreground session, or register it as a Scheduled
# Task with an "At startup" trigger for unattended pilot use (see
# register_tasks.ps1).

param(
    [int]$Port = 8000,
    [int]$RestartDelaySeconds = 5
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$logDir = Join-Path $root "logs"
$logFile = Join-Path $logDir "service_supervisor.log"
$pidFile = Join-Path $logDir "uvicorn.pid"
$pythonExe = Join-Path $root "venv\Scripts\python.exe"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logFile -Value "[$ts] $Message"
}

if (-not (Test-Path $pythonExe)) {
    Write-Log "FATAL: venv python not found at $pythonExe - is the venv set up?"
    exit 1
}

Write-Log "=== Supervisor started (port=$Port) ==="

while ($true) {
    Write-Log "Starting uvicorn..."
    $proc = Start-Process -FilePath $pythonExe `
        -ArgumentList @("-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$Port") `
        -WorkingDirectory $root `
        -PassThru -NoNewWindow

    $proc.Id | Set-Content -Path $pidFile
    Write-Log "uvicorn started, PID=$($proc.Id)"

    $proc.WaitForExit()
    $exitCode = $proc.ExitCode
    Remove-Item $pidFile -ErrorAction SilentlyContinue

    Write-Log "uvicorn exited (PID=$($proc.Id), exit code=$exitCode). Restarting in $RestartDelaySeconds s..."
    Start-Sleep -Seconds $RestartDelaySeconds
}
