#!/usr/bin/env bash
# Privileged deploy script - installed to /usr/local/bin/vision-service-deploy.sh
# (root-owned, mode 750) by ops/provision_vps.sh. The GitLab CI "deploy" job
# is only allowed to run this ONE script via a narrow NOPASSWD sudoers rule
# (see provision_vps.sh) - it cannot sudo to anything else. Keeping every
# privileged step inside a single root-controlled script (instead of several
# wildcard sudo rules) means a compromised/malicious CI job can only ever
# trigger this exact deploy flow, nothing else.
#
# Usage: sudo /usr/local/bin/vision-service-deploy.sh <source checkout dir>
#   <source checkout dir> is $CI_PROJECT_DIR - the fresh git checkout GitLab's
#   shell executor already made for this job.
#
# Design: install deps + run the smoke test INSIDE the new checkout's venv
# BEFORE touching the running service. If the smoke test fails, this script
# exits non-zero, the pipeline goes red, and the previous (working) service
# instance is never restarted - a broken deploy never takes down a working
# one. This mirrors the project's own "never silently guess, flag instead"
# principle (see CLAUDE.md) applied to the deploy process itself.

set -euo pipefail

DEPLOY_DIR="/opt/vision-service"
DEPLOY_USER="visionsvc"
SOURCE_DIR="${1:-}"

if [ -z "$SOURCE_DIR" ] || [ ! -d "$SOURCE_DIR" ]; then
    echo "Usage: $0 <source checkout dir>" >&2
    exit 1
fi

echo "==> Syncing $SOURCE_DIR -> $DEPLOY_DIR"
# venv/logs/debug_output/.env are excluded so they persist across deploys
# instead of getting wiped and rebuilt every time.
rsync -a --delete \
    --exclude 'venv/' \
    --exclude 'logs/' \
    --exclude 'debug_output/' \
    --exclude '.git/' \
    --exclude '.env' \
    "$SOURCE_DIR/" "$DEPLOY_DIR/"
chown -R "$DEPLOY_USER:$DEPLOY_USER" "$DEPLOY_DIR"

cd "$DEPLOY_DIR"

if [ ! -d venv ]; then
    echo "==> No venv found, creating one (python3.11)"
    runuser -u "$DEPLOY_USER" -- python3.11 -m venv venv
fi

echo "==> Installing/updating dependencies"
runuser -u "$DEPLOY_USER" -- venv/bin/pip install --upgrade pip
runuser -u "$DEPLOY_USER" -- venv/bin/pip install -r requirements.txt

echo "==> Smoke test (tests/test_ocr.py) - must pass before the running service is touched"
runuser -u "$DEPLOY_USER" -- venv/bin/python tests/test_ocr.py

echo "==> Smoke test passed - restarting vision-service"
systemctl restart vision-service

echo "==> Waiting for /api/health to come back up..."
for _ in $(seq 1 15); do
    if curl -fsS http://127.0.0.1:8000/api/health > /dev/null 2>&1; then
        echo "==> Deploy OK - /api/health responding."
        exit 0
    fi
    sleep 2
done

echo "==> ERROR: service did not become healthy within ~30s after restart." >&2
systemctl status vision-service --no-pager || true
exit 1
