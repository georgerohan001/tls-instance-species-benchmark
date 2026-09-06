"""Methods figure: side-by-side 25 m subplot layouts (one panel per site)."""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import laspy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from shapely import concave_hull
from shapely.geometry import MultiPoint, Polygon, box

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paths import SitePaths, load_site_config, resolve_path  # noqa: E402

warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)

INK = "#1a1f18"
MUTED = "#5c6658"
LINE = "#d7d0c3"
CLIP = "#2f5d3a"
CELL_EDGE = "#ffffff"
CELL_FACE = "#d7e3cf"
CELL_FACE_FOCUS = "#a8c49a"
CLOUD = "#9aa39a"
EXCLUDED = "#c46b5a"

CHUNK = 2_000_000
TARGET_PTS = 40_000
CELL_SIZE = 25.0
MARGIN_M = 8.0
POINT_SIZE = 0.45
DPI = 220


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _under_data(site: SitePaths, key: str, default: str | None = None) -> Path | None:
    val = site.raw.get(key, default)
    if val is None or val == "":
        return None
    return resolve_path(val, base=site.data_root)


def shapely_to_mpl(poly: Polygon, **kwargs) -> MplPolygon:
    return MplPolygon(list(poly.exterior.coords), closed=True, **kwargs)


def make_hull(points: MultiPoint, *, ratio: float | None, max_vertices: int | None) -> Polygon:
    if max_vertices is not None:
        for r in (0.2, 0.15, 0.25, 0.3, 0.1, 0.35, 0.4, 0.5, 1.0):
            hull = concave_hull(points, ratio=r, allow_holes=False)
            if len(hull.exterior.coords) - 1 <= max_vertices:
                return hull
        return points.convex_hull
    return concave_hull(points, ratio=ratio if ratio is not None else 0.05, allow_holes=False)


def load_inventory(path: Path, layer: str | None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    return gdf.loc[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()


def build_grid_cells(clip_poly: Polygon) -> dict[tuple[int, int], Polygon]:
    cx, cy = clip_poly.centroid.x, clip_poly.centroid.y
    half = CELL_SIZE / 2.0
    minx, miny, maxx, maxy = clip_poly.bounds
    i_min = math.floor((minx - cx - half) / CELL_SIZE)
    i_max = math.ceil((maxx - cx + half) / CELL_SIZE)
    j_min = math.floor((miny - cy - half) / CELL_SIZE)
    j_max = math.ceil((maxy - cy + half) / CELL_SIZE)
    cells: dict[tuple[int, int], Polygon] = {}
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            cell = box(
                cx + i * CELL_SIZE - half,
                cy + j * CELL_SIZE - half,
                cx + i * CELL_SIZE + half,
                cy + j * CELL_SIZE + half,
            )
            clipped = cell.intersection(clip_poly)
            if not clipped.is_empty and clipped.area > 1.0:
                cells[(i, j)] = cell
    return cells


def sample_points(
    laz_path: Path,
    bounds: tuple[float, float, float, float],
    *,
    instance_field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minx, miny, maxx, maxy = bounds
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    insts: list[np.ndarray] = []
    with laspy.open(laz_path) as reader:
        extras = {d.name for d in reader.header.point_format.extra_dimensions}
        has_inst = instance_field in extras
        for chunk in reader.chunk_iterator(CHUNK):
            cx = np.asarray(chunk.x)
            cy = np.asarray(chunk.y)
            mask = (cx >= minx) & (cx <= maxx) & (cy >= miny) & (cy <= maxy)
            if not np.any(mask):
                continue
            xs.append(cx[mask])
            ys.append(cy[mask])
            if has_inst:
                insts.append(np.asarray(getattr(chunk, instance_field))[mask])
            else:
                insts.append(np.zeros(int(mask.sum()), dtype=np.int32))
    if not xs:
        return np.array([]), np.array([]), np.array([])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    inst = np.concatenate(insts)
    if len(x) > TARGET_PTS:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(x), size=TARGET_PTS, replace=False)
        x, y, inst = x[pick], y[pick], inst[pick]
    return x, y, inst


def draw_panel(
    ax,
    *,
    title: str,
    clip_poly: Polygon,
    cells: dict,
    cell_to_tile: dict,
    focus_tiles: set[int],
    excluded_ids: set[int],
    xs,
    ys,
    inst,
) -> None:
    cx = clip_poly.centroid.x
    cy = clip_poly.centroid.y

    def shift_xy(x, y):
        return x - cx, y - cy

    def shift_poly(poly: Polygon) -> Polygon:
        return Polygon([(px - cx, py - cy) for px, py in poly.exterior.coords])

    clip_l = shift_poly(clip_poly)
    labels = []
    for (i, j), cell in cells.items():
        tile = cell_to_tile.get((i, j))
        face = CELL_FACE_FOCUS if tile in focus_tiles else CELL_FACE
        ax.add_patch(
            shapely_to_mpl(
                shift_poly(cell),
                facecolor=face,
                edgecolor=CELL_EDGE,
                linewidth=1.1,
                alpha=0.95,
                zorder=2,
            )
        )
        if tile is not None:
            cc = cell.centroid
            labels.append((cc.x - cx, cc.y - cy, f"{int(tile):03d}"))

    if len(xs):
        x_l, y_l = shift_xy(xs, ys)
        excl = np.isin(inst, list(excluded_ids)) if excluded_ids else np.zeros(len(inst), dtype=bool)
        keep = ~excl
        if np.any(keep):
            ax.scatter(
                x_l[keep],
                y_l[keep],
                c=CLOUD,
                s=POINT_SIZE,
                linewidths=0,
                rasterized=True,
                zorder=3,
                alpha=0.5,
            )
        if np.any(excl):
            ax.scatter(
                x_l[excl],
                y_l[excl],
                c=EXCLUDED,
                s=POINT_SIZE * 1.35,
                linewidths=0,
                rasterized=True,
                zorder=4,
                alpha=0.7,
            )

    ax.add_patch(shapely_to_mpl(clip_l, fill=False, edgecolor=CLIP, linewidth=2.0, zorder=6))
    fontsize = 6.5 if len(labels) > 50 else 8.0
    for lx, ly, lab in labels:
        ax.text(
            lx,
            ly,
            lab,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=650,
            color=INK,
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.82, edgecolor="none"),
        )

    b = clip_l.bounds
    ax.set_xlim(b[0] - MARGIN_M, b[2] + MARGIN_M)
    ax.set_ylim(b[1] - MARGIN_M, b[3] + MARGIN_M)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight=650, color=INK, pad=8)
    ax.set_xlabel("Relative easting (m)", fontsize=9, color=MUTED)
    ax.set_ylabel("Relative northing (m)", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.grid(True, color=LINE, linewidth=0.55, alpha=0.65)
    ax.set_facecolor("#fbfaf7")


def prepare_site(site: SitePaths) -> dict | None:
    if site.inventory is None or not site.inventory.is_file():
        print(f"skip {site.site_id}: missing inventory")
        return None
    manifest_path = _under_data(site, "subplot_manifest") or _under_data(site, "manifest_path")
    if manifest_path is None or not manifest_path.is_file():
        print(f"skip {site.site_id}: missing subplot_manifest")
        return None
    laz = (
        _under_data(site, "subplot_laz")
        or _under_data(site, "clip_cloud")
        or site.fm_laz
    )
    if laz is None or not laz.is_file():
        print(f"skip {site.site_id}: missing subplot_laz/clip_cloud")
        return None

    buffer_m = float(site.raw.get("buffer_m", 10.0))
    ratio = site.raw.get("hull_ratio")
    max_vertices = site.raw.get("max_hull_vertices")
    instance_field = str(site.raw.get("instance_field", "PredInstance_FM"))
    focus_tiles = set(int(t) for t in site.raw.get("focus_tiles", list(site.tiles)))

    gdf = load_inventory(site.inventory, site.inventory_layer)
    hull = make_hull(
        MultiPoint(gdf.geometry.tolist()),
        ratio=float(ratio) if ratio is not None else None,
        max_vertices=int(max_vertices) if max_vertices is not None else None,
    )
    clip_poly = hull.buffer(buffer_m)
    cells = build_grid_cells(clip_poly)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cell_to_tile = {
        tuple(map(int, k.split(","))): int(v) for k, v in manifest.get("cell_to_tile", {}).items()
    }
    excluded = set(int(x) for x in manifest.get("excluded_trees", []))
    b = clip_poly.bounds
    roi = (b[0] - MARGIN_M, b[1] - MARGIN_M, b[2] + MARGIN_M, b[3] + MARGIN_M)
    print(f"Sampling {site.title}…")
    xs, ys, inst = sample_points(laz, roi, instance_field=instance_field)
    return {
        "title": site.title,
        "clip_poly": clip_poly,
        "cells": cells,
        "cell_to_tile": cell_to_tile,
        "focus_tiles": focus_tiles,
        "excluded_ids": excluded,
        "xs": xs,
        "ys": ys,
        "inst": inst,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", nargs="+", required=True, help="Site YAML path(s).")
    ap.add_argument(
        "--output",
        default="outputs/figures/subplots_side_by_side.png",
        help="Output PNG relative to repo root.",
    )
    args = ap.parse_args()
    panels = []
    for cfg in args.config:
        panel = prepare_site(load_site_config(cfg))
        if panel is not None:
            panels.append(panel)
    if not panels:
        print("skip make_subplots_figure: no usable sites")
        raise SystemExit(0)

    out = resolve_path(args.output, base=REPO_ROOT)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "axes.unicode_minus": False,
        }
    )
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.9), dpi=DPI)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    for ax, site in zip(axes, panels):
        draw_panel(ax, **site)

    legend_handles = [
        Line2D([0], [0], color=CLIP, linewidth=2.0, label="Clip boundary"),
        MplPolygon([[0, 0]], closed=True, facecolor=CELL_FACE, edgecolor=CELL_EDGE, label="25 m subplot"),
        MplPolygon([[0, 0]], closed=True, facecolor=CELL_FACE_FOCUS, edgecolor=CELL_EDGE, label="Analysed subplot"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLOUD, markersize=5, label="TLS points"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=EXCLUDED, markersize=5, label="Excluded edge trees"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {_rel(out)}")


if __name__ == "__main__":
    main()
