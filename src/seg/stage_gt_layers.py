"""Stage GT LAS layers into site.gt_layers_dir.

CloudCompare BIN export is external — run that yourself (or reuse an existing
tile_XXX/tree_*.las tree) and point this script at the folder.

Behaviour:
  1. If gt_layers_dir already has tile_*/tree_*.las files for configured tiles,
     print a short inventory and exit 0.
  2. Otherwise copy from --source (relative to data_root) or from
     raw config key manual_gt_source (also relative to data_root).

Expected layout under source / destination:
  tile_001/tree_*.las
  tile_002/...
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.paths import SitePaths, load_site_config, resolve_path

GT_DIR: Path | None = None
TILES: tuple[int, ...] = ()
DATA_ROOT: Path | None = None
SITE_TITLE = ""


def configure(site: SitePaths) -> None:
    global GT_DIR, TILES, DATA_ROOT, SITE_TITLE
    GT_DIR = site.gt_layers_dir
    TILES = site.tiles
    DATA_ROOT = site.data_root
    SITE_TITLE = site.title


def _tree_las_count(tile_dir: Path) -> int:
    if not tile_dir.is_dir():
        return 0
    return sum(1 for _ in tile_dir.glob("tree_*.las*"))


def gt_layers_populated() -> bool:
    """True if every configured tile has at least one tree_*.las* file."""
    if not TILES:
        return False
    for tid in TILES:
        if _tree_las_count(GT_DIR / f"tile_{tid:03d}") < 1:
            return False
    return True


def inventory() -> list[dict]:
    rows = []
    for tid in TILES:
        tdir = GT_DIR / f"tile_{tid:03d}"
        n_tree = _tree_las_count(tdir)
        n_all = len(list(tdir.glob("*.las*"))) if tdir.is_dir() else 0
        rows.append({"tile": tid, "tree_las": n_tree, "all_las": n_all, "dir": str(tdir)})
    return rows


def copy_from_source(source: Path) -> int:
    """Copy tile_XXX/* into GT_DIR. Returns number of files copied."""
    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")
    copied = 0
    for tid in TILES:
        src_tile = source / f"tile_{tid:03d}"
        if not src_tile.is_dir():
            # Also accept bare tile_XXX.las layout? stick to tile folders.
            print(f"  skip missing source tile: {src_tile}", flush=True)
            continue
        dest_tile = GT_DIR / f"tile_{tid:03d}"
        dest_tile.mkdir(parents=True, exist_ok=True)
        for path in sorted(src_tile.glob("*.las*")):
            if not path.is_file():
                continue
            dest = dest_tile / path.name
            shutil.copy2(path, dest)
            copied += 1
        print(
            f"  tile {tid:03d}: copied from {src_tile} -> {dest_tile} "
            f"({_tree_las_count(dest_tile)} tree files)",
            flush=True,
        )
    return copied


def main(source_rel: str | None = None, site: SitePaths | None = None) -> None:
    GT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{SITE_TITLE}] gt_layers_dir={GT_DIR}", flush=True)

    if gt_layers_populated():
        print("gt_layers already populated; nothing to do.", flush=True)
        for row in inventory():
            print(
                f"  tile {row['tile']:03d}: tree_las={row['tree_las']} "
                f"all_las={row['all_las']}",
                flush=True,
            )
        return

    raw = (site.raw if site is not None else {}) or {}
    rel = source_rel or raw.get("manual_gt_source")
    if not rel:
        raise SystemExit(
            "gt_layers is empty and no source given. Pass --source "
            "(relative to data_root) or set manual_gt_source in the site YAML.\n"
            "CloudCompare BIN → LAS export is external; stage the resulting "
            "tile_XXX/tree_*.las tree first."
        )

    source = resolve_path(rel, base=DATA_ROOT)
    print(f"Copying GT layers from {source} ...", flush=True)
    n = copy_from_source(source)
    if n == 0:
        raise SystemExit(f"No LAS files copied from {source}")
    if not gt_layers_populated():
        print("Warning: after copy, some tiles still lack tree_*.las files:", flush=True)
        for row in inventory():
            print(
                f"  tile {row['tile']:03d}: tree_las={row['tree_las']} "
                f"all_las={row['all_las']}",
                flush=True,
            )
        raise SystemExit(1)
    print(f"Staged {n} files into {GT_DIR}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to site YAML config")
    parser.add_argument(
        "--source",
        default=None,
        help="Folder with tile_XXX/tree_*.las relative to data_root "
        "(overrides manual_gt_source)",
    )
    args = parser.parse_args()
    site = load_site_config(args.config)
    configure(site)
    main(source_rel=args.source, site=site)
