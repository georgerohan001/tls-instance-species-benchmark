"""Inject PredInstance_SAT from site.sat_laz into existing x_eval.npz."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.paths import SitePaths, load_site_config

ALIGNED_NPZ: Path | None = None
ALIGNED_DIR: Path | None = None
SAT_LAZ: Path | None = None
NN_M = 0.15
SITE_TITLE = ""

CHUNK = 3_000_000
VOXEL_M = 0.05
QUANT_MM = 1


def configure(site: SitePaths) -> None:
    global ALIGNED_NPZ, ALIGNED_DIR, SAT_LAZ, NN_M, SITE_TITLE
    ALIGNED_NPZ = site.aligned
    ALIGNED_DIR = site.aligned.parent
    SAT_LAZ = site.sat_laz
    # Prefer a slightly larger NN for SAT voxel fill; fall back to site.nn_m * 3
    NN_M = max(site.nn_m * 3, 0.15)
    SITE_TITLE = site.title
    site.ensure_output_dirs()


def pack_xyz_mm(x, y, z, ox, oy, oz):
    xi = np.floor((x - ox) * 1000.0 / QUANT_MM + 0.5).astype(np.int64)
    yi = np.floor((y - oy) * 1000.0 / QUANT_MM + 0.5).astype(np.int64)
    zi = np.floor((z - oz) * 1000.0 / QUANT_MM + 0.5).astype(np.int64)
    return xi + yi * 2_000_000 + zi * 2_000_000 * 2_000_000


def main() -> None:
    if SAT_LAZ is None or not SAT_LAZ.exists():
        raise SystemExit(f"Missing sat_laz (configure sat_laz in site YAML): {SAT_LAZ}")
    if ALIGNED_NPZ is None or not ALIGNED_NPZ.exists():
        raise SystemExit(f"Missing aligned npz: {ALIGNED_NPZ}")

    z = np.load(ALIGNED_NPZ)
    x, y, zz = z["x"], z["y"], z["z"]
    n = len(x)
    sat_id = np.full(n, -1, dtype=np.int32)
    print(f"[{SITE_TITLE}] Loaded X_eval n={n:,}", flush=True)

    ox, oy, oz = float(x.min()), float(y.min()), float(zz.min())
    eval_keys = pack_xyz_mm(x, y, zz, ox, oy, oz)
    order = np.argsort(eval_keys)
    keys_sorted = eval_keys[order]

    exact = 0
    scanned = 0
    mamba_agree = 0
    mamba_cmp = 0
    print(f"Streaming PredInstance_SAT from {SAT_LAZ} ...", flush=True)
    with laspy.open(SAT_LAZ) as reader:
        extras = {d.name for d in reader.header.point_format.extra_dimensions}
        if "PredInstance_SAT" not in extras:
            # Fall back to common instance field names
            field = None
            for cand in ("PredInstance_SAT", "PredInstance", "pred_instance"):
                if cand in extras:
                    field = cand
                    break
            if field is None:
                for d in extras:
                    if "instance" in d.lower() and "sat" in d.lower():
                        field = d
                        break
            if field is None:
                raise SystemExit(f"PredInstance_SAT missing; extras={sorted(extras)}")
        else:
            field = "PredInstance_SAT"
        has_fm = "PredInstance" in extras
        total = reader.header.point_count
        done = 0
        while done < total:
            pts = reader.read_points(min(CHUNK, total - done))
            done += len(pts)
            scanned += len(pts)
            sx = np.asarray(pts.x, dtype=np.float64)
            sy = np.asarray(pts.y, dtype=np.float64)
            sz = np.asarray(pts.z, dtype=np.float64)
            sid = np.asarray(getattr(pts, field), dtype=np.int32)
            sk = pack_xyz_mm(sx, sy, sz, ox, oy, oz)
            loc = np.searchsorted(keys_sorted, sk)
            ok = loc < n
            ok[ok] &= keys_sorted[loc[ok]] == sk[ok]
            if np.any(ok):
                idx = order[loc[ok]]
                sat_id[idx] = sid[ok]
                exact += int(ok.sum())
                if has_fm and "mamba_id" in z.files:
                    fm = np.asarray(pts.PredInstance, dtype=np.int32)[ok]
                    mid = z["mamba_id"][idx]
                    mamba_cmp += int(ok.sum())
                    mamba_agree += int(np.sum(fm == mid))
            if done % (CHUNK * 5) < CHUNK or done >= total:
                print(
                    f"  scan {done:,}/{total:,} exact_hits={exact:,} "
                    f"filled={int(np.mean(sat_id >= 0)*n):,}",
                    flush=True,
                )

    need = sat_id < 0
    n_need = int(need.sum())
    print(f"Exact filled={exact:,}; remaining={n_need:,} -> voxel reverse NN", flush=True)

    margin = 0.5
    xmin, xmax = float(x.min()) - margin, float(x.max()) + margin
    ymin, ymax = float(y.min()) - margin, float(y.max()) + margin
    zmin, zmax = float(zz.min()) - margin, float(zz.max()) + margin
    inv = 1.0 / VOXEL_M
    vox_map: dict[tuple[int, int, int], int] = {}

    with laspy.open(SAT_LAZ) as reader:
        extras = {d.name for d in reader.header.point_format.extra_dimensions}
        field = "PredInstance_SAT" if "PredInstance_SAT" in extras else None
        if field is None:
            for cand in ("PredInstance", "pred_instance"):
                if cand in extras:
                    field = cand
                    break
        if field is None:
            raise SystemExit(f"No SAT instance field; extras={sorted(extras)}")
        total = reader.header.point_count
        done = 0
        while done < total:
            pts = reader.read_points(min(CHUNK, total - done))
            done += len(pts)
            sx = np.asarray(pts.x, dtype=np.float64)
            sy = np.asarray(pts.y, dtype=np.float64)
            sz = np.asarray(pts.z, dtype=np.float64)
            sid = np.asarray(getattr(pts, field), dtype=np.int32)
            m = (
                (sx >= xmin)
                & (sx <= xmax)
                & (sy >= ymin)
                & (sy <= ymax)
                & (sz >= zmin)
                & (sz <= zmax)
                & (sid > 0)
            )
            if not np.any(m):
                continue
            sx, sy, sz, sid = sx[m], sy[m], sz[m], sid[m]
            vx = np.floor(sx * inv).astype(np.int64)
            vy = np.floor(sy * inv).astype(np.int64)
            vz = np.floor(sz * inv).astype(np.int64)
            rx = int(vx.max() - vx.min()) + 2
            ry = int(vy.max() - vy.min()) + 2
            oxv, oyv, ozv = int(vx.min()), int(vy.min()), int(vz.min())
            packed = (vx - oxv) + (vy - oyv) * rx + (vz - ozv) * (rx * ry)
            _, first = np.unique(packed, return_index=True)
            for j in first:
                key = (int(vx[j]), int(vy[j]), int(vz[j]))
                if key not in vox_map:
                    vox_map[key] = int(sid[j])
            if done % (CHUNK * 10) < CHUNK or done >= total:
                print(f"  voxels {done:,}/{total:,} n={len(vox_map):,}", flush=True)

    nn_filled = 0
    if vox_map and n_need:
        keys = np.array(list(vox_map.keys()), dtype=np.int64)
        sat_xyz = (keys.astype(np.float64) + 0.5) * VOXEL_M
        sat_ids = np.array(list(vox_map.values()), dtype=np.int32)
        tree = cKDTree(sat_xyz)
        q_idx = np.where(need)[0]
        batch = 500_000
        for start in range(0, len(q_idx), batch):
            end = min(start + batch, len(q_idx))
            ii = q_idx[start:end]
            q = np.column_stack([x[ii], y[ii], zz[ii]])
            d, ix = tree.query(q, k=1, workers=-1, distance_upper_bound=NN_M)
            good = np.isfinite(d) & (d <= NN_M)
            sat_id[ii[good]] = sat_ids[ix[good]]
            nn_filled += int(good.sum())
            print(f"  nn {end:,}/{len(q_idx):,} filled={nn_filled:,}", flush=True)

    rate = float(np.mean(sat_id >= 0))
    tree_rate = float(np.mean(sat_id > 0))
    print(
        f"SAT match_rate(id>=0)={rate:.4f} tree_rate(id>0)={tree_rate:.4f} "
        f"exact={exact:,} nn={nn_filled:,}",
        flush=True,
    )
    if mamba_cmp:
        print(f"Mamba PredInstance agreement on exact hits: {mamba_agree/mamba_cmp:.4f}", flush=True)

    out = {k: z[k] for k in z.files}
    out["sat_id"] = sat_id
    np.savez_compressed(ALIGNED_NPZ, **out)

    man_path = ALIGNED_DIR / "manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
    else:
        man = {}
    man["paths"] = {**man.get("paths", {}), "sat_laz": str(SAT_LAZ), "x_eval": str(ALIGNED_NPZ)}
    man["sat"] = {
        "sat_field": field,
        "method": "galaxy_sat_inject",
        "source_laz": str(SAT_LAZ),
        "quantize_mm": QUANT_MM,
        "voxel_m": VOXEL_M,
        "nn_m": NN_M,
        "sat_match_rate": rate,
        "sat_tree_rate": tree_rate,
        "exact_hits": int(exact),
        "nn_filled": int(nn_filled),
        "n_sat_voxels": len(vox_map),
        "mamba_agree_frac": (mamba_agree / mamba_cmp) if mamba_cmp else None,
    }
    man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"Updated {ALIGNED_NPZ} and {man_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to site YAML config")
    args = parser.parse_args()
    configure(load_site_config(args.config))
    main()
