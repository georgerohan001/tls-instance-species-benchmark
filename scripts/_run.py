#!/usr/bin/env python3
"""CLI wrappers — run from repository root."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(rel: str) -> None:
    sys.path.insert(0, str(REPO))
    runpy.run_path(str(REPO / rel), run_name="__main__")


if __name__ == "__main__":
    print("Use one of: stage_gt_layers, build_aligned, inject_sat, run_metrics,")
    print("analyze_per_tree, analyze_detailview, prepare_figures")
    sys.exit(2)
