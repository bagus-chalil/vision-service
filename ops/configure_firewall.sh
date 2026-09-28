#!/usr/bin/env bash
# Linux/ufw equivalent of ops/configure_firewall.ps1 - restricts inbound
# access to the service port to a specific IP allowlist instead of leaving
# it open to the whole network. Stopgap for pilot deployment: the service
# itself still has no auth/API key/HTTPS (see CLAUDE.md), so restricting who
# can even reach the port matters.
#
# NOTE (Proxmox users): this VM's NIC was created with firewall=1, meaning
# Proxmox's OWN per-VM firewall also sits in front of this. ufw rules here
# are necessary but may not be sufficient - if traffic still doesn't get
# through after running this script, check Datacenter/VM > Firewall rules
# in the Proxmox UI too; both layers can independently block the same port.
#
# Usage (run once, as root, after deciding which IPs actually need access):
#   ./configure_firewall.sh 100.100.160.23 100.100.160.50
#
# Re-running it is safe - it resets the app-specific rules each time, so
# updating the allowlist later is just re-running with new IPs.

set -euo pipefail

PORT="${PORT:-8000}"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <allowed-ip> [<allowed-ip> ...]" >&2
    echo "Example: $0 100.100.160.23" >&2
    exit 1
fi

if ! command -v ufw > /dev/null 2>&1; then
    echo "ufw not found - install it first: apt install ufw" >&2
    exit 1
fi

echo "==> Ensuring SSH (22) stays reachable before enabling ufw"
ufw allow 22/tcp comment "SSH"

echo "==> Removing any previous rules for port $PORT (idempotent re-run)"
# Delete existing numbered rules that mention this port; ignore errors if none exist.
while ufw status numbered | grep -q "$PORT"; do
    rule_num=$(ufw status numbered | grep "$PORT" | head -1 | sed -E 's/^\[([0-9]+)\].*/\1/')
    [ -n "$rule_num" ] || break
    yes | ufw delete "$rule_num" > /dev/null
done

echo "==> Allowing port $PORT only from: $*"
for ip in "$@"; do
    ufw allow from "$ip" to any port "$PORT" proto tcp comment "vision-service pilot"
done

echo "==> Default deny for anything else on $PORT"
ufw deny "$PORT"/tcp

echo "==> Enabling ufw (no-op if already enabled)"
ufw --force enable

echo "==> Done. Current rules:"
ufw status verbose
