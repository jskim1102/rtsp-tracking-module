"""Legacy no-op threshold env must not affect tracker module imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def test_import_manager_ignores_removed_conf_threshold_surface():
    env = {
        **os.environ,
        "YOLO_CONF_THRESHOLD": "high",
        "PYTHONPATH": str(BACKEND) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    r = subprocess.run(
        [sys.executable, "-c", "import app.streaming.manager; print('OK')"],
        cwd=str(BACKEND),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "OK" in r.stdout
