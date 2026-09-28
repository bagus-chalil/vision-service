# Restricts inbound access to the Vision Service port to a specific allowlist
# of IPs (e.g. the Laravel backend at 100.100.160.23, and whatever machines
# QC will capture from during the pilot), instead of leaving it open to the
# whole network. This is a stopgap for pilot deployment - the service itself
# still has no auth/API key/HTTPS (see CLAUDE.md), so restricting who can
# even reach the port matters.
#
# Run ONCE on the host machine (e.g. the .24 box) from an elevated
# PowerShell prompt, after deciding which IPs actually need access:
#
#   powershell -ExecutionPolicy Bypass -File .\ops\configure_firewall.ps1 `
#       -AllowedIPs "100.100.160.23","100.100.160.50"
#
# Re-running it is safe - it removes and re-creates the rule each time, so
# updating the allowlist later is just re-running with the new -AllowedIPs.
#
# To remove the restriction entirely (open to everyone again - NOT
# recommended for pilot), run:
#   powershell -ExecutionPolicy Bypass -File .\ops\configure_firewall.ps1 -Remove

#Requires -RunAsAdministrator

param(
    [int]$Port = 8000,
    [string[]]$AllowedIPs = @(),
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$ruleName = "VisionServicePilotAccess"

Remove-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue

if ($Remove) {
    Write-Host "Removed firewall restriction '$ruleName' (port $Port is no longer filtered by this script)."
    exit 0
}

if ($AllowedIPs.Count -eq 0) {
    Write-Host "ERROR: pass -AllowedIPs with at least one IP address, e.g.:" -ForegroundColor Red
    Write-Host '  .\configure_firewall.ps1 -AllowedIPs "100.100.160.23","100.100.160.50"'
    exit 1
}

New-NetFirewallRule `
    -DisplayName $ruleName `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort $Port `
    -RemoteAddress $AllowedIPs `
    -Action Allow | Out-Null

Write-Host "Firewall rule '$ruleName' created: inbound TCP $Port allowed only from:"
$AllowedIPs | ForEach-Object { Write-Host "  - $_" }
Write-Host ""
Write-Host "NOTE: Windows Firewall's default inbound policy on most profiles is" -ForegroundColor Yellow
Write-Host "'Block' unless a matching Allow rule exists, but double check the" -ForegroundColor Yellow
Write-Host "active profile's default action if this port was reachable from" -ForegroundColor Yellow
Write-Host "elsewhere before running this script." -ForegroundColor Yellow
