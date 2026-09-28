#!/usr/bin/env bash
# One-time bootstrap for a fresh Ubuntu VPS to host the Vision Service pilot
# + a GitLab Runner that deploys to it (VPS spec assumed: 4 vCPU / 4GB RAM /
# 32GB disk, Ubuntu 26.04 - see README.md "Deploy checklist" for why this
# picks native venv+systemd instead of Docker on this spec).
#
# Run ONCE, as root, from a clone of this repo sitting on the VPS (e.g.
# cloned temporarily from GitHub just for this bootstrap - ongoing deploys
# come from the GitLab-registered runner's own checkout afterwards, this
# script does not need to be re-run for every deploy):
#
#   git clone https://github.com/bagus-chalil/vision-service.git /root/bootstrap
#   cd /root/bootstrap
#   sudo ./ops/provision_vps.sh
#
# Safe to re-run - every step checks for existing state first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_DIR="/opt/vision-service"
DEPLOY_USER="visionsvc"

if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root: sudo ./ops/provision_vps.sh" >&2
    exit 1
fi

echo "==> [1/8] apt update + base packages"
apt-get update -qq
apt-get install -y --no-install-recommends \
    git rsync curl ufw software-properties-common ca-certificates gnupg

echo "==> [2/8] Python 3.11 (deadsnakes PPA)"
# Matches the tested venv (CLAUDE.md: "Python 3.11 venv"). Ubuntu 26.04's
# default python3 is likely newer and may not have paddleocr/paddlepaddle
# wheels published for it yet - don't risk it, install 3.11 explicitly.
if ! command -v python3.11 > /dev/null 2>&1; then
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y python3.11 python3.11-venv python3.11-dev
else
    echo "    python3.11 already present, skipping"
fi

echo "==> [3/8] Swap file (4GB)"
# This VM only has 4GB RAM; installing torch/paddlepaddle and loading
# PaddleOCR + YOLO models at startup can spike memory. Swap is a safety net
# against an OOM-killed service, not a substitute for enough RAM long-term.
if [ ! -f /swapfile ]; then
    fallocate -l 4G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo "/swapfile none swap sw 0 0" >> /etc/fstab
    echo "    created 4GB /swapfile"
else
    echo "    /swapfile already exists, skipping"
fi

echo "==> [4/8] Dedicated service user ($DEPLOY_USER, no login)"
if ! id "$DEPLOY_USER" > /dev/null 2>&1; then
    useradd --system --create-home --home-dir "$DEPLOY_DIR" --shell /usr/sbin/nologin "$DEPLOY_USER"
else
    echo "    user $DEPLOY_USER already exists, skipping"
fi
mkdir -p "$DEPLOY_DIR/logs" "$DEPLOY_DIR/debug_output"
chown -R "$DEPLOY_USER:$DEPLOY_USER" "$DEPLOY_DIR"

echo "==> [5/8] GitLab Runner (official package)"
if ! command -v gitlab-runner > /dev/null 2>&1; then
    curl -L "https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh" | bash
    apt-get install -y gitlab-runner
else
    echo "    gitlab-runner already installed, skipping"
fi

echo "==> [6/8] Installing privileged deploy script + systemd unit"
install -o root -g root -m 750 "$REPO_ROOT/ops/deploy.sh" /usr/local/bin/vision-service-deploy.sh
install -o root -g root -m 644 "$REPO_ROOT/ops/vision-service.service" /etc/systemd/system/vision-service.service
systemctl daemon-reload
systemctl enable vision-service || true
# Not starting it yet - /opt/vision-service is still empty until the first
# deploy (CI or manual) actually populates it.

echo "==> [7/8] Sudoers: gitlab-runner may run ONLY the deploy script, nothing else"
# Everything privileged lives inside that one root-owned script (see its
# header comment) - this is the entire attack surface a CI job gets, on
# purpose, instead of a pile of wildcard sudo rules.
cat > /etc/sudoers.d/vision-service-deploy <<'EOF'
gitlab-runner ALL=(root) NOPASSWD: /usr/local/bin/vision-service-deploy.sh
EOF
chmod 440 /etc/sudoers.d/vision-service-deploy
visudo -c -f /etc/sudoers.d/vision-service-deploy

echo "==> [8/8] Done."
cat <<EOF

Remaining MANUAL steps (need values only GitLab's web UI can give you):

  1. On GitLab: open the mirror project ->
     Settings > CI/CD > Runners > "New project runner" -> copy the
     registration token (and confirm the GitLab URL, e.g. https://gitlab.com).

  2. On this VPS, register the runner (tag it vps-aiocr - .gitlab-ci.yml
     expects that exact tag):
       sudo gitlab-runner register --url https://gitlab.com \\
           --tag-list vps-aiocr --executor shell

  3. Restrict port 8000 to whoever actually needs it:
       sudo $SCRIPT_DIR/configure_firewall.sh <allowed-ip> [<allowed-ip> ...]

  4. This VM's NIC has Proxmox's own per-VM firewall enabled (firewall=1) -
     ufw rules above are necessary but not sufficient by themselves. Also
     add matching allow rules under Datacenter/VM > Firewall in the Proxmox
     UI, or traffic can still get blocked there even with ufw configured.

  5. Seed the first deploy manually (before any pipeline has run yet):
       sudo /usr/local/bin/vision-service-deploy.sh $REPO_ROOT
     After this, every push through the GitHub -> GitLab mirror triggers
     .gitlab-ci.yml, which calls this same script automatically.
EOF
