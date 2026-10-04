#!/usr/bin/env bash
# One-time setup inside the Ubuntu VM. Run from the repo root:  bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt update
sudo apt install -y python3-venv python3-pip time fio procps
python3 -m venv .venv                       # avoids the "externally-managed-environment" error
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
echo "Setup OK. Python: $(.venv/bin/python --version)"
