"""Portable pathway smoke tests (no thesis data required).

Run from repo root:
  python tests/build_tiny_site_fixture.py
  python tests/smoke_tiny_site.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from shapely.geometry import box

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "seg"))

from src.paths import load_site_config, resolve_path  # noqa: E402
from src.seg.build_aligned import parse_gt_id  # noqa: E402
import analyze_per_tree as apt  # noqa: E402
from run_metrics import configure  # noqa: E402


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}: {a!r} != {b!r}")


def smoke_parse_gt_id() -> None:
    assert_eq(parse_gt_id("tree_01102.laz"), 1102000)
    assert_eq(parse_gt_id("tree_00998_3.laz"), 998003)
    assert_eq(parse_gt_id(r"foo/bar/tree_00002_1.laz"), 2001)
    assert_eq(parse_gt_id("tree_00001.laz (C:/somewhere/else)"), 1000)
    assert_eq(parse_gt_id("misc_inst-1.laz"), -1)
    try:
        parse_gt_id("tree_1_1000.laz")
    except SystemExit:
        pass
    else:
        raise AssertionError("frag>=1000 should SystemExit")
    # Distinct fragments must not glue
    assert_eq(len({parse_gt_id("tree_00002_1.laz"), parse_gt_id("tree_00002_2.laz")}), 2)
    print("OK parse_gt_id")


def smoke_paths() -> None:
    site = load_site_config(REPO / "configs" / "tiny_site.example.yaml")
    assert_eq(site.site_id, "tiny_site")
    assert 1 in site.tile_xy_bounds
    assert_eq(len(site.tile_xy_bounds[1]), 4)
    try:
        resolve_path(r"C:\abs\path.laz", base=REPO)
    except ValueError:
        pass
    else:
        raise AssertionError("absolute paths must be rejected")
    # Missing bounds → empty dict, not crash
    raw = yaml.safe_load((REPO / "configs" / "site.example.yaml").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        data = td_path / "data" / "tmp"
        data.mkdir(parents=True)
        cfg = {
            "site_id": "tmp",
            "data_root": str(data.relative_to(REPO)) if False else None,
        }
        # Write a minimal config under repo-relative temp is hard; use fixture-adjacent.
        cfg_path = REPO / "tests" / "fixtures" / "_tmp_no_bounds.yaml"
        cfg_path.write_text(
            "\n".join(
                [
                    "site_id: no_bounds_site",
                    "title: No bounds",
                    "data_root: tests/fixtures/tiny_site",
                    "tiles: [1]",
                    "aligned: aligned/x_eval.npz",
                    "gt_layers_dir: gt_layers",
                    "nn_m: 0.05",
                    "n_min: 5",
                    "outputs:",
                    "  tables: outputs/tiny_site/tables",
                    "  figures: outputs/tiny_site/figures",
                    "  dv_results: outputs/tiny_site/detailview",
                ]
            ),
            encoding="utf-8",
        )
        try:
            site2 = load_site_config(cfg_path)
            assert_eq(site2.tile_xy_bounds, {})
        finally:
            cfg_path.unlink(missing_ok=True)
    print("OK paths / tile_xy_bounds")


def smoke_neighbour_and_hull() -> None:
    # Hand calculation on toy centroids inside [0,20]^2
    xmin, ymin, xmax, ymax = 0.0, 0.0, 20.0, 20.0
    assert apt.edge_censored_for_radius(10.0, 10.0, xmin, ymin, xmax, ymax, 5.0) is False
    assert apt.edge_censored_for_radius(2.0, 10.0, xmin, ymin, xmax, ymax, 5.0) is True

    a = box(0, 0, 2, 2)
    b = box(1, 0, 3, 2)  # 50% of a overlaps b
    frac = float(a.intersection(b).area / a.area)
    if abs(frac - 0.5) > 1e-9:
        raise AssertionError(f"expected overlap 0.5 got {frac}")
    print("OK neighbour edge + hull overlap arithmetic")


def smoke_missing_bounds_errors() -> None:
    apt.TILE_XY_BOUNDS = {}
    try:
        apt.require_tile_xy_bounds({1})
    except SystemExit as e:
        msg = str(e)
        if "tile_xy_bounds" not in msg:
            raise AssertionError(f"message not actionable: {msg}")
    else:
        raise AssertionError("missing bounds must SystemExit")
    print("OK missing bounds error")


def smoke_end_to_end() -> None:
    cfg = REPO / "configs" / "tiny_site.example.yaml"
    site = load_site_config(cfg)
    apt.configure_site(site)

    # Minimal T1 so analyze main can proceed
    tables = site.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    site.figures_dir.mkdir(parents=True, exist_ok=True)
    t1 = pd.DataFrame(
        [
            {"Method": "ForestMamba", "Prec": 100.0, "Rec": 100.0, "F1": 100.0, "Cov": 100.0},
            {"Method": "SegmentAnyTree", "Prec": 50.0, "Rec": 75.0, "F1": 60.0, "Cov": 75.0},
        ]
    )
    t1.to_csv(tables / "T1_overall.csv", index=False)

    apt.main()

    cat = pd.read_csv(site.aligned.parent / "gt_instances.csv")
    ids = set(int(x) for x in cat["gt_id"])
    if ids != {1000, 2001, 2002, 3000}:
        raise AssertionError(f"unexpected gt_ids {ids} (glued?)")
    if len(ids) != 4:
        raise AssertionError("fragments must remain distinct")

    per = pd.read_csv(tables / "per_tree_scores.csv")
    for col in ("n_gt_within_5m", "edge_censored_5m", "hull_overlap_frac"):
        if col not in per.columns:
            raise AssertionError(f"missing column {col}")
    if not (tables / "T10b_neighbour_count_collinearity.csv").is_file():
        raise AssertionError("missing T10b")
    if not (tables / "T10c_hull_overlap_collinearity.csv").is_file():
        raise AssertionError("missing T10c")

    # Edge stem (gt_id 3000) should be edge-censored
    edge_rows = per[(per["gt_id"] == 3000) & (per["Method"] == per["Method"].iloc[0])]
    if not bool(edge_rows["edge_censored_5m"].iloc[0]):
        raise AssertionError("gt_id 3000 should be edge_censored_5m")

    print("OK end-to-end analyze_per_tree on tiny_site")


def smoke_rerun() -> None:
    apt.configure_site(load_site_config(REPO / "configs" / "tiny_site.example.yaml"))
    apt.main()
    print("OK idempotent re-run")


def main() -> None:
    smoke_parse_gt_id()
    smoke_paths()
    smoke_neighbour_and_hull()
    smoke_missing_bounds_errors()
    smoke_end_to_end()
    smoke_rerun()
    print("ALL SMOKES PASSED")


if __name__ == "__main__":
    main()
