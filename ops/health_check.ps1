# Health monitor for the Vision Service prototype.
#
# Pings GET /api/health. If it fails (timeout, connection refused, non-200),
# logs the failure and force-kills the uvicorn process tracked in
# logs/uvicorn.pid - run_service.ps1's supervisor loop will notice the exit
# and restart it within RestartDelaySeconds.
#
# This targets the exact PID the supervisor started, not a name-based match,
# to avoid the "ghost process on port 8000" ambiguity (CLAUDE.md gotcha #5).
#
# Intended to run on a schedule (e.g. every 2-5 min via Scheduled Task), not
# as a long-running process - each invocation checks once and exits.

param(
    [string]$Url = "http://127.0.0.1:8000/api/health",
    [int]$TimeoutSeconds = 5
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$logDir = Join-Path $root "logs"
$logFile = Join-Path $logDir "health_monitor.log"
$pidFile = Join-Path $logDir "uvicorn.pid"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logFile -Value "[$ts] $Message"
}

try {
    $resp = Invoke-WebRequest -Uri $Url -TimeoutSec $TimeoutSeconds -UseBasicParsing
    if ($resp.StatusCode -eq 200) {
        exit 0
    }
    Write-Log "Health check returned unexpected status $($resp.StatusCode)"
} catch {
    Write-Log "Health check FAILED: $($_.Exception.Message)"
}

if (-not (Test-Path $pidFile)) {
    Write-Log "No uvicorn.pid found - service may not be running under the supervisor. Nothing to kill."
    exit 1
}

$stuckPid = Get-Content $pidFile -ErrorAction SilentlyContinue
if (-not $stuckPid) {
    Write-Log "uvicorn.pid was empty. Nothing to kill."
    exit 1
}

$proc = Get-Process -Id $stuckPid -ErrorAction SilentlyContinue
if (-not $proc) {
    Write-Log "PID=$stuckPid in uvicorn.pid is not a running process (already exited). Supervisor should restart it on its own."
    exit 0
}

Write-Log "Killing unresponsive uvicorn process PID=$stuckPid so the supervisor restarts it..."
try {
    Stop-Process -Id $stuckPid -Force -ErrorAction Stop
    Write-Log "Killed PID=$stuckPid."
} catch {
    Write-Log "Could not kill PID=$stuckPid: $($_.Exception.Message)"
    exit 1
}
