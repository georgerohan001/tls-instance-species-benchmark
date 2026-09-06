#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
runpy.run_path(str(REPO / "src" / "seg" / "stage_gt_layers.py"), run_name="__main__")
