"""Build the committed tiny_site fixture arrays (synthetic, not thesis data).

Run: python tests/build_tiny_site_fixture.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "tiny_site"
ALIGNED = FIX / "aligned"
TABLES = REPO / "outputs" / "tiny_site" / "tables"


def _disk(cx: float, cy: float, n: int, z0: float, gid: int, rng: np.random.Generator):
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = rng.uniform(0, 0.8, n)
    x = cx + rad * np.cos(ang)
    y = cy + rad * np.sin(ang)
    z = z0 + rng.uniform(0, 8, n)
    return x, y, z, np.full(n, gid, dtype=np.int32)


def main() -> None:
    rng = np.random.default_rng(0)
    ALIGNED.mkdir(parents=True, exist_ok=True)
    (FIX / "gt_layers" / "tile_001").mkdir(parents=True, exist_ok=True)

    # Interior stem near centre; two fragment stems (distinct gt_ids) near each other;
    # one stem near the western edge for edge_censored_5m.
    parts = [
        _disk(10.0, 10.0, 40, 0.0, 1000, rng),  # tree_00001
        _disk(12.0, 10.5, 40, 0.0, 2001, rng),  # tree_00002_1
        _disk(12.5, 10.0, 40, 0.0, 2002, rng),  # tree_00002_2
        _disk(2.0, 10.0, 40, 0.0, 3000, rng),  # tree_00003 (edge)
    ]
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    z = np.concatenate([p[2] for p in parts])
    gt = np.concatenate([p[3] for p in parts])
    n = len(gt)
    # Perfect FM match; SAT merges the two fragments into one id (still valid for smoke).
    mamba = gt.copy()
    sat = gt.copy()
    sat[gt == 2002] = 2001

    np.savez_compressed(
        ALIGNED / "x_eval.npz",
        x=x.astype(np.float64),
        y=y.astype(np.float64),
        z=z.astype(np.float64),
        gt_id=gt.astype(np.int32),
        tile_id=np.ones(n, dtype=np.int32),
        mamba_id=mamba.astype(np.int32),
        mamba_score=np.ones(n, dtype=np.float32),
        sat_id=sat.astype(np.int32),
        kind=np.zeros(n, dtype=np.int8),
    )

    cat = pd.DataFrame(
        [
            {"gt_id": 1000, "tile": 1, "n_points": 40, "layer": "tree_00001.laz"},
            {"gt_id": 2001, "tile": 1, "n_points": 40, "layer": "tree_00002_1.laz"},
            {"gt_id": 2002, "tile": 1, "n_points": 40, "layer": "tree_00002_2.laz"},
            {"gt_id": 3000, "tile": 1, "n_points": 40, "layer": "tree_00003.laz"},
        ]
    )
    cat.to_csv(ALIGNED / "gt_instances.csv", index=False)

    # Placeholder layer names (extension-less markers; points live in the npz).
    layer_dir = FIX / "gt_layers" / "tile_001"
    layer_dir.mkdir(parents=True, exist_ok=True)
    for name in cat["layer"]:
        (layer_dir / f"{name}.placeholder").write_text(
            "Synthetic name marker only; XYZ lives in aligned/x_eval.npz.\n",
            encoding="utf-8",
        )
    print(f"Wrote fixture under {FIX}")
    print(f"N_GT layers={len(cat)} distinct gt_ids={sorted(cat['gt_id'].tolist())}")


if __name__ == "__main__":
    main()
