"""Methods figure: inventory concave-hull clip verification (one panel per site)."""
from __future__ import annotations

import argparse
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
from shapely.geometry import MultiPoint, Polygon

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paths import SitePaths, load_site_config, resolve_path  # noqa: E402

warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)

INK = "#1a1f18"
MUTED = "#5c6658"
LINE = "#d7d0c3"
CLIP = "#2f5d3a"
HULL = "#4a5560"
BUFFER_FACE = "#c8d9b8"
CLOUD = "#9aa39a"
INV = "#111111"

CHUNK = 2_000_000
TARGET_PTS = 45_000
MARGIN_M = 25.0
POINT_SIZE = 0.55
INV_SIZE = 10.0
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


def sample_points(laz_path: Path, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    minx, miny, maxx, maxy = bounds
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    total_in = 0
    with laspy.open(laz_path) as reader:
        for chunk in reader.chunk_iterator(CHUNK):
            cx = np.asarray(chunk.x)
            cy = np.asarray(chunk.y)
            mask = (cx >= minx) & (cx <= maxx) & (cy >= miny) & (cy <= maxy)
            if not np.any(mask):
                continue
            xs.append(cx[mask])
            ys.append(cy[mask])
            total_in += int(mask.sum())
    if not xs:
        return np.array([]), np.array([])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if len(x) > TARGET_PTS:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(x), size=TARGET_PTS, replace=False)
        x, y = x[pick], y[pick]
    print(f"  {_rel(laz_path)}: kept {len(x):,} of {total_in:,} ROI points")
    return x, y


def draw_panel(ax, *, title: str, gdf: gpd.GeoDataFrame, hull: Polygon, buffered: Polygon, xs, ys) -> None:
    cx = buffered.centroid.x
    cy = buffered.centroid.y

    def shift_poly(poly: Polygon) -> Polygon:
        return Polygon([(x - cx, y - cy) for x, y in poly.exterior.coords])

    hull_l = shift_poly(hull)
    buf_l = shift_poly(buffered)
    buffer_band = buf_l.difference(hull_l)
    geoms = [buffer_band] if buffer_band.geom_type == "Polygon" else list(buffer_band.geoms)
    ax.add_collection(
        PatchCollection(
            [shapely_to_mpl(g) for g in geoms],
            facecolor=BUFFER_FACE,
            edgecolor="none",
            alpha=0.55,
            zorder=1,
        )
    )
    if len(xs):
        ax.scatter(xs - cx, ys - cy, c=CLOUD, s=POINT_SIZE, linewidths=0, rasterized=True, zorder=2, alpha=0.55)
    ax.add_patch(shapely_to_mpl(hull_l, fill=False, edgecolor=HULL, linewidth=1.4, linestyle=(0, (4, 2.5)), zorder=4))
    ax.add_patch(shapely_to_mpl(buf_l, fill=False, edgecolor=CLIP, linewidth=2.0, zorder=5))
    ax.scatter(
        gdf.geometry.x.to_numpy() - cx,
        gdf.geometry.y.to_numpy() - cy,
        s=INV_SIZE,
        c=INV,
        edgecolors="white",
        linewidths=0.35,
        zorder=6,
    )
    b = buf_l.bounds
    ax.set_xlim(b[0] - MARGIN_M, b[2] + MARGIN_M)
    ax.set_ylim(b[1] - MARGIN_M, b[3] + MARGIN_M)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight=650, color=INK, pad=8)
    ax.set_xlabel("Relative easting (m)", fontsize=9, color=MUTED)
    ax.set_ylabel("Relative northing (m)", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.grid(True, color=LINE, linewidth=0.6, alpha=0.7)
    ax.set_facecolor("#fbfaf7")


def prepare_site(site: SitePaths) -> dict | None:
    if site.inventory is None or not site.inventory.is_file():
        print(f"skip {site.site_id}: missing inventory")
        return None
    laz = _under_data(site, "clip_cloud") or _under_data(site, "sample_laz") or site.fm_laz
    if laz is None or not laz.is_file():
        print(f"skip {site.site_id}: missing clip_cloud/sample_laz (or fm_laz)")
        return None
    buffer_m = float(site.raw.get("buffer_m", 10.0))
    ratio = site.raw.get("hull_ratio")
    max_vertices = site.raw.get("max_hull_vertices")
    gdf = load_inventory(site.inventory, site.inventory_layer)
    if gdf.empty:
        print(f"skip {site.site_id}: empty inventory")
        return None
    hull = make_hull(
        MultiPoint(gdf.geometry.tolist()),
        ratio=float(ratio) if ratio is not None else None,
        max_vertices=int(max_vertices) if max_vertices is not None else None,
    )
    buffered = hull.buffer(buffer_m)
    b = buffered.bounds
    roi = (b[0] - MARGIN_M, b[1] - MARGIN_M, b[2] + MARGIN_M, b[3] + MARGIN_M)
    print(f"Sampling {site.title}…")
    xs, ys = sample_points(laz, roi)
    return {
        "title": site.title,
        "gdf": gdf,
        "hull": hull,
        "buffered": buffered,
        "xs": xs,
        "ys": ys,
        "site": site,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", nargs="+", required=True, help="Site YAML path(s).")
    ap.add_argument(
        "--output",
        default="outputs/figures/clip_hulls_side_by_side.png",
        help="Output PNG relative to repo root.",
    )
    args = ap.parse_args()
    sites_cfg = [load_site_config(c) for c in args.config]
    panels = []
    for site in sites_cfg:
        panel = prepare_site(site)
        if panel is not None:
            panels.append(panel)
    if not panels:
        print("skip make_clip_hull_figure: no usable sites")
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
    for ax, panel in zip(axes, panels):
        draw_panel(
            ax,
            title=panel["title"],
            gdf=panel["gdf"],
            hull=panel["hull"],
            buffered=panel["buffered"],
            xs=panel["xs"],
            ys=panel["ys"],
        )

    legend_handles = [
        Line2D([0], [0], color=CLIP, linewidth=2.0, label="Clip boundary (hull + buffer)"),
        Line2D([0], [0], color=HULL, linewidth=1.4, linestyle=(0, (4, 2.5)), label="Inventory concave hull"),
        MplPolygon([[0, 0]], closed=True, facecolor=BUFFER_FACE, edgecolor="none", alpha=0.8, label="Buffer"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLOUD, markersize=5, label="TLS points"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=INV,
            markeredgecolor="white",
            markersize=6,
            label="Inventory stems",
        ),
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
