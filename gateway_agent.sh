#!/usr/bin/env bash
set -euo pipefail
echo "[agent] starting gateway agent (notebook export)"
python - << 'PY'
from pathlib import Path
import os
# assume we are running inside the same env
from time import sleep
try:
    from __main__ import gateway_agent_loop
except ImportError:
    # if we ran this as a plain file
    from gateway_notebook import gateway_agent_loop  # adjust if needed
gateway_agent_loop(profile="daily", interval_s=300, jitter_s=30, max_loops=0)
PY
