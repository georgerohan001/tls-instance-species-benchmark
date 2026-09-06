"""Build X_eval from staged GT layers and attach ForestMamba instance IDs.

SAT IDs are left as -1 here; use inject_galaxy_sat.py with site.sat_laz afterwards.
Mamba IDs come from PredInstance_FM on exported GT layers when present.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import laspy
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.paths import SitePaths, load_site_config

ALIGNED_NPZ: Path | None = None
ALIGNED_DIR: Path | None = None
GT_DIR: Path | None = None
FM_LAZ: Path | None = None
SAT_LAZ: Path | None = None
TILES: tuple[int, ...] = ()
NN_M = 0.05
N_MIN = 100
SITE_TITLE = ""

QUANTIZE_MM = 1


def configure(site: SitePaths) -> None:
    global ALIGNED_NPZ, ALIGNED_DIR, GT_DIR, FM_LAZ, SAT_LAZ
    global TILES, NN_M, N_MIN, SITE_TITLE
    ALIGNED_NPZ = site.aligned
    ALIGNED_DIR = site.aligned.parent
    GT_DIR = site.gt_layers_dir
    FM_LAZ = site.fm_laz
    SAT_LAZ = site.sat_laz
    TILES = site.tiles
    NN_M = site.nn_m
    N_MIN = site.n_min
    SITE_TITLE = site.title
    site.ensure_output_dirs()


def parse_gt_id(name: str) -> int:
    m = re.search(r"tree_(\d+)", name, re.I)
    return int(m.group(1)) if m else -1


def layer_kind(name: str) -> str:
    low = name.lower()
    if "misc_inst" in low:
        return "misc"
    if "ground" in low:
        return "ground"
    if "tree_" in low:
        return "tree"
    return "other"


def load_gt_layers():
    xs, ys, zs = [], [], []
    gt_ids, tile_ids, mamba_ids, scores, kinds = [], [], [], [], []
    catalogue = []
    counts = {"tree": 0, "misc": 0, "ground": 0}

    for tid in TILES:
        tdir = GT_DIR / f"tile_{tid:03d}"
        if not tdir.is_dir():
            print(f"Warning: missing tile dir {tdir}", flush=True)
            continue
        for path in sorted(tdir.glob("*.las*")):
            kind = layer_kind(path.name)
            if kind == "other":
                continue
            las = laspy.read(path)
            n = len(las.x)
            if n == 0:
                continue
            x = np.asarray(las.x, dtype=np.float64)
            y = np.asarray(las.y, dtype=np.float64)
            z = np.asarray(las.z, dtype=np.float64)
            gid = parse_gt_id(path.name) if kind == "tree" else -1
            mid = (
                np.asarray(las.PredInstance_FM, dtype=np.int32)
                if hasattr(las, "PredInstance_FM")
                else np.full(n, -1, dtype=np.int32)
            )
            sc = (
                np.asarray(las.PredScore_FM, dtype=np.float32)
                if hasattr(las, "PredScore_FM")
                else np.full(n, np.nan, dtype=np.float32)
            )
            xs.append(x)
            ys.append(y)
            zs.append(z)
            gt_ids.append(np.full(n, gid, dtype=np.int32))
            tile_ids.append(np.full(n, tid, dtype=np.int16))
            mamba_ids.append(mid)
            scores.append(sc)
            kinds.append(np.full(n, {"tree": 0, "misc": 1, "ground": 2}[kind], dtype=np.int8))
            counts[kind] += n
            if kind == "tree":
                catalogue.append(
                    {
                        "gt_id": gid,
                        "tile": tid,
                        "n_points": int(n),
                        "centroid_x": float(x.mean()),
                        "centroid_y": float(y.mean()),
                        "z_min": float(z.min()),
                        "z_max": float(z.max()),
                        "layer": path.name,
                    }
                )

    if not xs:
        raise SystemExit(f"No GT layers found under {GT_DIR} for tiles {TILES}")

    arrays = {
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        "z": np.concatenate(zs),
        "gt_id": np.concatenate(gt_ids),
        "tile_id": np.concatenate(tile_ids),
        "mamba_id": np.concatenate(mamba_ids),
        "mamba_score": np.concatenate(scores),
        "kind": np.concatenate(kinds),
    }
    print(
        f"X_eval n={len(arrays['x']):,} tree={counts['tree']:,} "
        f"misc={counts['misc']:,} ground={counts['ground']:,} n_gt={len(catalogue)}"
    )
    return arrays, catalogue, counts


def main() -> None:
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    arrays, catalogue, counts = load_gt_layers()
    n = len(arrays["x"])
    # SAT filled later by inject_galaxy_sat.py
    arrays["sat_id"] = np.full(n, -1, dtype=np.int32)

    n_tree = int(np.sum(arrays["gt_id"] > 0))
    tree_mamba_pos = int(np.sum((arrays["gt_id"] > 0) & (arrays["mamba_id"] > 0)))
    mamba_info = {
        "source": "PredInstance_FM_on_exported_LAS",
        "match_rate_mamba_tree_positive": tree_mamba_pos / max(n_tree, 1),
        "n_mamba_unassigned": int(np.sum(arrays["mamba_id"] < 0)),
        "fm_laz": str(FM_LAZ) if FM_LAZ else None,
    }
    print("Mamba:", mamba_info)

    out_npz = ALIGNED_NPZ
    np.savez_compressed(
        out_npz,
        x=arrays["x"].astype(np.float64),
        y=arrays["y"].astype(np.float64),
        z=arrays["z"].astype(np.float64),
        gt_id=arrays["gt_id"],
        tile_id=arrays["tile_id"],
        mamba_id=arrays["mamba_id"],
        sat_id=arrays["sat_id"],
        mamba_score=arrays["mamba_score"],
        kind=arrays["kind"],
    )
    print(f"Wrote {out_npz} ({out_npz.stat().st_size / 1e9:.2f} GB)")

    cat_path = ALIGNED_DIR / "gt_instances.csv"
    with cat_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "gt_id",
                "tile",
                "n_points",
                "centroid_x",
                "centroid_y",
                "z_min",
                "z_max",
                "layer",
            ],
        )
        w.writeheader()
        w.writerows(catalogue)

    sat_info = {
        "sat_id": "unassigned (-1)",
        "note": "Run inject_galaxy_sat.py --config <site.yaml> to fill sat_id from sat_laz",
        "sat_laz": str(SAT_LAZ) if SAT_LAZ else None,
    }

    manifest = {
        "status": "aligned",
        "site_title": SITE_TITLE,
        "n_points": int(n),
        "n_gt_trees": len(catalogue),
        "n_tree_points": counts["tree"],
        "n_misc_points": counts["misc"],
        "n_ground_points": counts["ground"],
        "n_background_points": counts["misc"] + counts["ground"],
        "tau_default": 0.5,
        "n_min": N_MIN,
        "quantize_mm": QUANTIZE_MM,
        "nn_cm": NN_M * 100,
        "tiles": list(TILES),
        "paths": {
            "x_eval": str(out_npz),
            "gt_instances": str(cat_path),
            "gt_layers": str(GT_DIR),
            "fm_laz": str(FM_LAZ) if FM_LAZ else None,
            "sat_laz": str(SAT_LAZ) if SAT_LAZ else None,
        },
        "mamba": mamba_info,
        "sat": sat_info,
        "skip_rules": {
            "gt_background_id": -1,
            "pred_unassigned_mamba": "id < 0",
            "pred_unassigned_sat": "id <= 0",
            "n_min_pred_points": N_MIN,
        },
    }
    man_path = ALIGNED_DIR / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {man_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to site YAML config")
    args = parser.parse_args()
    configure(load_site_config(args.config))
    main()
