"""Process-wide test safety defaults."""

import os


# Ultralytics reads this at import time and otherwise defaults to installing
# missing packages into the active environment.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
