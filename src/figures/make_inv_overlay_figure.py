"""Methods figure: breast-height inventory overlays (one panel per site).

Coloured points are manual tree layers in a 1.3 m +/- 0.05 m band along
the local ground normal; circles mark inventory DBH.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import laspy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paths import SitePaths, load_site_config, resolve_path  # noqa: E402

warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)

INK = "#1a1f18"
MUTED = "#5c6658"
LINE = "#d7d0c3"
GROUND = "#9aa39a"
INV_HALO = "#ffffff"
INV_RING = "#c2185b"
INV_FILL = "#c2185b"
PANEL_BG = "#fbfaf7"

SLICE_HEIGHT_M = 1.3
SLICE_TOLERANCE_M = 0.05
SKIP_SUBSTRINGS = ("misc_inst", "inst0_other")
POINT_SIZE = 1.4
MIN_INV_RADIUS_M = 0.32
INV_FILL_ALPHA = 0.22
INV_HALO_LW = 0.66
INV_RING_LW = 0.35
DPI = 220
MARGIN_M = 2.5


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


def should_skip(name: str) -> bool:
    low = name.lower()
    return any(s in low for s in SKIP_SUBSTRINGS)


def eval_surface_and_slope(x: np.ndarray, y: np.ndarray, model: dict):
    c0, c1, c2 = model["c0"], model["c1"], model["c2"]
    c3, c4, c5 = model["c3"], model["c4"], model["c5"]
    x0, y0 = model["x0"], model["y0"]
    X, Y = x - x0, y - y0
    z = c0 + c1 * X + c2 * Y + c3 * X * X + c4 * X * Y + c5 * Y * Y
    fx = c1 + 2.0 * c3 * X + c4 * Y
    fy = c2 + c4 * X + 2.0 * c5 * Y
    return z, fx, fy


def height_along_normal(x: np.ndarray, y: np.ndarray, z: np.ndarray, model: dict) -> np.ndarray:
    z_s, fx, fy = eval_surface_and_slope(x, y, model)
    return (z - z_s) / np.sqrt(1.0 + fx * fx + fy * fy)


def tree_color(idx: int) -> tuple[float, float, float, float]:
    hue = (idx * 0.6180339887) % 1.0
    r = 0.22 + 0.68 * hue
    g = 0.22 + 0.68 * ((hue * 2.3) % 1.0)
    b = 0.22 + 0.68 * ((hue * 4.7) % 1.0)
    return (r, g, b, 0.88)


def load_surface_model(manifest: Path) -> dict:
    return json.loads(manifest.read_text(encoding="utf-8"))["model"]


def collect_tile_slice(
    layers_dir: Path, model: dict
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    lo = SLICE_HEIGHT_M - SLICE_TOLERANCE_M
    hi = SLICE_HEIGHT_M + SLICE_TOLERANCE_M
    trees: list[tuple[np.ndarray, np.ndarray]] = []
    ground: list[tuple[np.ndarray, np.ndarray]] = []
    for path in sorted(layers_dir.glob("*.las*")):
        if should_skip(path.name):
            continue
        las = laspy.read(path)
        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        z = np.asarray(las.z, dtype=np.float64)
        h = height_along_normal(x, y, z, model)
        mask = (h >= lo) & (h <= hi)
        if not np.any(mask):
            continue
        xy = (x[mask], y[mask])
        if "ground" in path.name.lower():
            ground.append(xy)
        else:
            trees.append(xy)
    return trees, ground


def diameter_column(gdf: gpd.GeoDataFrame) -> str:
    if "diameter_m" in gdf.columns:
        return "diameter_m"
    if "DBH" in gdf.columns:
        return "DBH"
    raise KeyError("No diameter column found")


def ground_xy_bounds(manifests: list[Path]) -> tuple[float, float, float, float]:
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for path in manifests:
        b = json.loads(path.read_text(encoding="utf-8"))["xy_bounds"]
        xmin = min(xmin, b[0])
        ymin = min(ymin, b[1])
        xmax = max(xmax, b[2])
        ymax = max(ymax, b[3])
    return xmin, ymin, xmax, ymax


def load_inventory_for_extent(
    path: Path,
    layer: str | None,
    bounds: tuple[float, float, float, float],
    *,
    shift_xy: tuple[float, float] | None = None,
) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    gdf = gdf.loc[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    if shift_xy is not None:
        dx, dy = shift_xy
        gdf.geometry = gdf.geometry.translate(xoff=dx, yoff=dy)
    diam = diameter_column(gdf)
    xmin, ymin, xmax, ymax = bounds
    mask = (
        gdf.geometry.x.between(xmin, xmax)
        & gdf.geometry.y.between(ymin, ymax)
        & gdf[diam].notna()
        & (gdf[diam] > 0)
    )
    out = gdf.loc[mask].copy()
    out.attrs["diameter_col"] = diam
    return out


def slice_extent(trees, ground):
    chunks_x = [c[0] for c in trees + ground]
    chunks_y = [c[1] for c in trees + ground]
    if not chunks_x:
        raise RuntimeError("No slice points")
    x = np.concatenate(chunks_x)
    y = np.concatenate(chunks_y)
    return float(x.min()), float(y.min()), float(x.max()), float(y.max())


def prepare_site(site: SitePaths) -> dict | None:
    if site.inventory is None or not site.inventory.is_file():
        print(f"skip {site.site_id}: missing inventory")
        return None

    ground_root = _under_data(site, "ground_surface_dir", "inputs/ground_surface")
    slice_root = _under_data(site, "height_slice_dir", "inputs/height_slice_13")
    if ground_root is None or slice_root is None:
        print(f"skip {site.site_id}: missing ground_surface_dir / height_slice_dir")
        return None

    tile_ids = tuple(int(t) for t in site.tiles)
    trees_all: list[tuple[np.ndarray, np.ndarray]] = []
    ground_all: list[tuple[np.ndarray, np.ndarray]] = []
    tile_boxes: list[tuple[float, float, float, float]] = []
    manifests: list[Path] = []

    for tid in tile_ids:
        manifest = ground_root / f"tile_{tid:03d}" / "surface_manifest.json"
        layers = slice_root / f"tile_{tid:03d}" / "layers_las"
        if not manifest.is_file() or not layers.is_dir():
            print(f"skip {site.site_id} tile {tid:03d}: missing {_rel(manifest)} or {_rel(layers)}")
            continue
        model = load_surface_model(manifest)
        trees, ground = collect_tile_slice(layers, model)
        trees_all.extend(trees)
        ground_all.extend(ground)
        manifests.append(manifest)
        if trees or ground:
            tile_boxes.append(slice_extent(trees, ground))

    if not manifests or not (trees_all or ground_all):
        print(f"skip {site.site_id}: no ground/slice inputs")
        return None

    bounds = ground_xy_bounds(manifests)
    shift_xy = None
    shift_path = _under_data(site, "inventory_shift_json")
    if shift_path is not None and shift_path.is_file():
        shift = json.loads(shift_path.read_text(encoding="utf-8"))["estimated_shift_m"]
        shift_xy = (float(shift["dx"]), float(shift["dy"]))

    try:
        inv = load_inventory_for_extent(
            site.inventory,
            site.inventory_layer,
            bounds,
            shift_xy=shift_xy,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"skip {site.site_id}: inventory load failed ({exc})")
        return None

    return {
        "title": site.title,
        "trees": trees_all,
        "ground": ground_all,
        "inventory": inv,
        "tile_boxes": tile_boxes if len(tile_ids) > 1 else [],
    }


def draw_panel(ax, *, title: str, trees, ground, inventory, tile_boxes) -> None:
    all_x = np.concatenate([c[0] for c in trees + ground]) if (trees or ground) else np.array([0.0])
    all_y = np.concatenate([c[1] for c in trees + ground]) if (trees or ground) else np.array([0.0])
    x0 = float(all_x.min())
    y0 = float(all_y.min())

    for xs, ys in ground:
        ax.scatter(
            xs - x0,
            ys - y0,
            s=POINT_SIZE * 0.7,
            c=[GROUND],
            linewidths=0,
            rasterized=True,
            zorder=2,
            alpha=0.55,
        )
    for i, (xs, ys) in enumerate(trees):
        ax.scatter(
            xs - x0,
            ys - y0,
            s=POINT_SIZE,
            c=[tree_color(i)],
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
    for xmin, ymin, xmax, ymax in tile_boxes:
        ax.plot(
            [xmin - x0, xmax - x0, xmax - x0, xmin - x0, xmin - x0],
            [ymin - y0, ymin - y0, ymax - y0, ymax - y0, ymin - y0],
            color="0.45",
            linewidth=0.7,
            linestyle="--",
            zorder=4,
            alpha=0.7,
        )

    diam = inventory.attrs["diameter_col"]
    if len(inventory):
        centres = [
            (
                float(row.geometry.x) - x0,
                float(row.geometry.y) - y0,
                max(float(row[diam]) / 2.0, MIN_INV_RADIUS_M),
            )
            for _, row in inventory.iterrows()
        ]
        discs = [Circle((cx, cy), r) for cx, cy, r in centres]
        ax.add_collection(
            PatchCollection(discs, facecolor=INV_FILL, edgecolor="none", alpha=INV_FILL_ALPHA, zorder=5)
        )
        ax.add_collection(
            PatchCollection(
                [Circle((cx, cy), r) for cx, cy, r in centres],
                facecolor="none",
                edgecolor=INV_HALO,
                linewidths=INV_HALO_LW,
                zorder=6,
            )
        )
        ax.add_collection(
            PatchCollection(
                [Circle((cx, cy), r) for cx, cy, r in centres],
                facecolor="none",
                edgecolor=INV_RING,
                linewidths=INV_RING_LW,
                zorder=7,
            )
        )

    xmax = float(all_x.max()) - x0
    ymax = float(all_y.max()) - y0
    ax.set_xlim(-MARGIN_M, xmax + MARGIN_M)
    ax.set_ylim(-MARGIN_M, ymax + MARGIN_M)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight=650, color=INK, pad=8)
    ax.set_xlabel("Relative easting (m)", fontsize=9, color=MUTED)
    ax.set_ylabel("Relative northing (m)", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.grid(True, color=LINE, linewidth=0.55, alpha=0.65)
    ax.set_facecolor(PANEL_BG)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", nargs="+", required=True, help="Site YAML path(s).")
    ap.add_argument(
        "--output",
        default="outputs/figures/inventory_overlays_side_by_side.png",
        help="Output PNG relative to repo root.",
    )
    args = ap.parse_args()
    panels = []
    for cfg in args.config:
        site = load_site_config(cfg)
        print(f"Preparing {site.title}…")
        panel = prepare_site(site)
        if panel is not None:
            print(f"  trees={len(panel['trees'])} inventory={len(panel['inventory'])}")
            panels.append(panel)
    if not panels:
        print("skip make_inv_overlay_figure: no usable sites")
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
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.2), dpi=DPI)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    for ax, site in zip(axes, panels):
        draw_panel(ax, **site)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4a8f5c", markersize=6, label="Manual tree slice"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUND, markersize=6, label="Ground slice"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=INV_FILL,
            markeredgecolor=INV_RING,
            markeredgewidth=1.0,
            markersize=9,
            alpha=0.55,
            label="Inventory DBH",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {_rel(out)}")


if __name__ == "__main__":
    main()
