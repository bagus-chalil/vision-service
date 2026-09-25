# Registers the Vision Service supervisor + health monitor as Windows
# Scheduled Tasks, so the pilot service survives reboots and crashes without
# anyone remoting in to restart it by hand.
#
# Run this ONCE on the machine that will actually host the pilot service
# (e.g. the .24 box), from an elevated PowerShell prompt:
#
#   powershell -ExecutionPolicy Bypass -File .\ops\register_tasks.ps1
#
# Re-running it is safe - it unregisters and re-creates both tasks.

#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$supervisorScript = Join-Path $root "ops\run_service.ps1"
$healthCheckScript = Join-Path $root "ops\health_check.ps1"

$supervisorTaskName = "VisionServiceSupervisor"
$healthCheckTaskName = "VisionServiceHealthCheck"

# --- Supervisor task: starts at boot, keeps uvicorn alive forever ---

Unregister-ScheduledTask -TaskName $supervisorTaskName -Confirm:$false -ErrorAction SilentlyContinue

$supervisorAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$supervisorScript`""

$supervisorTrigger = New-ScheduledTaskTrigger -AtStartup

$supervisorSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0)

$supervisorPrincipal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName $supervisorTaskName `
    -Action $supervisorAction -Trigger $supervisorTrigger `
    -Settings $supervisorSettings -Principal $supervisorPrincipal `
    -Description "Runs the Vision Service (uvicorn) under a crash-restart supervisor loop." `
    | Out-Null

Write-Host "Registered task: $supervisorTaskName (starts at boot)"

# --- Health check task: pings /api/health every 5 min, kills a hung process ---

Unregister-ScheduledTask -TaskName $healthCheckTaskName -Confirm:$false -ErrorAction SilentlyContinue

$healthAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$healthCheckScript`""

$healthTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)

$healthSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

$healthPrincipal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName $healthCheckTaskName `
    -Action $healthAction -Trigger $healthTrigger `
    -Settings $healthSettings -Principal $healthPrincipal `
    -Description "Pings /api/health every 5 min and kills the uvicorn process if it stops responding." `
    | Out-Null

Write-Host "Registered task: $healthCheckTaskName (every 5 min)"

Write-Host ""
Write-Host "Starting supervisor task now (instead of waiting for next reboot)..."
Start-ScheduledTask -TaskName $supervisorTaskName
Write-Host "Done. Check logs\service_supervisor.log and logs\health_monitor.log for activity."
