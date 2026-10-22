#!/usr/bin/env bash
set -euo pipefail
# Generic EGA entry. Supply --settings FILE, or use a per-task launcher.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="/data0/pretrained/tmm_phi4_deps:${PYTHONPATH:-}"
cd "$PROJECT_DIR"
exec "${TMM_PYTHON:-/data0/pretrained/env_humanomniv2/bin/python}" -B scripts/ega_run.py "$@"
