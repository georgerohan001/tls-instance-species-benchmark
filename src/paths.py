"""Repo-relative path resolution for site configs.

All study data paths come from YAML. Nothing here embeds absolute machine paths.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def resolve_path(raw: str | Path, *, base: Path) -> Path:
    """Resolve a path relative to *base* (usually data_root or REPO_ROOT)."""
    p = Path(raw)
    if p.is_absolute():
        raise ValueError(
            f"Absolute paths are not allowed in configs (got {p}). "
            "Use a path relative to the repo or to data_root."
        )
    out = (base / p).resolve()
    return out


@dataclass(frozen=True)
class SitePaths:
    """Resolved paths for one evaluation site."""

    site_id: str
    repo_root: Path
    data_root: Path
    tiles: tuple[int, ...]
    aligned: Path
    gt_layers_dir: Path
    fm_laz: Path | None
    sat_laz: Path | None
    detailview_laz: Path | None
    match_csv: Path | None
    enriched_match_csv: Path | None
    inventory: Path | None
    inventory_layer: str | None
    inventory_id_col: str
    inventory_species_col: str
    species_lookup: Path
    tables_dir: Path
    figures_dir: Path
    dv_results_dir: Path
    nn_m: float
    n_min: int
    title: str
    raw: dict[str, Any]

    def ensure_output_dirs(self) -> None:
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.dv_results_dir.mkdir(parents=True, exist_ok=True)
        self.aligned.parent.mkdir(parents=True, exist_ok=True)


def _env_data_root() -> Path | None:
    env = os.environ.get("TLS_DATA_ROOT", "").strip()
    if not env:
        return None
    p = Path(env)
    if p.is_absolute():
        # Allowed only via env so large data can live outside the clone.
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def load_site_config(config_path: str | Path) -> SitePaths:
    """Load a site YAML and resolve all paths."""
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = (REPO_ROOT / cfg_file).resolve()
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_file}")

    with cfg_file.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    site_id = str(raw["site_id"])
    title = str(raw.get("title", site_id))

    env_root = _env_data_root()
    if env_root is not None:
        data_root = env_root / site_id if (env_root / site_id).is_dir() else env_root
    else:
        dr = raw.get("data_root", f"data/{site_id}")
        data_root = resolve_path(dr, base=REPO_ROOT)

    # data_root may be outside the repo when TLS_DATA_ROOT is set.
    if env_root is None and not _is_under(data_root, REPO_ROOT):
        raise ValueError(
            f"data_root must stay under the repository ({REPO_ROOT}), got {data_root}. "
            "Or set TLS_DATA_ROOT to an external data directory."
        )

    def under_data(key: str, default: str | None = None, required: bool = False) -> Path | None:
        val = raw.get(key, default)
        if val is None or val == "":
            if required:
                raise KeyError(f"Missing required config key: {key}")
            return None
        return resolve_path(val, base=data_root)

    outputs = raw.get("outputs") or {}
    tables = outputs.get("tables", f"outputs/{site_id}/tables")
    figures = outputs.get("figures", f"outputs/{site_id}/figures")
    dv_results = outputs.get("dv_results", f"outputs/{site_id}/detailview")

    species_map = raw.get("species_map", "contracts/detailview_lookup.csv")
    species_lookup = resolve_path(species_map, base=REPO_ROOT)

    tiles = tuple(int(t) for t in raw.get("tiles", [1]))

    return SitePaths(
        site_id=site_id,
        repo_root=REPO_ROOT,
        data_root=data_root,
        tiles=tiles,
        aligned=under_data("aligned", "aligned/x_eval.npz", required=True),  # type: ignore[arg-type]
        gt_layers_dir=under_data("gt_layers_dir", "gt_layers", required=True),  # type: ignore[arg-type]
        fm_laz=under_data("fm_laz"),
        sat_laz=under_data("sat_laz"),
        detailview_laz=under_data("detailview_laz"),
        match_csv=under_data("match_csv"),
        enriched_match_csv=under_data("enriched_match_csv"),
        inventory=under_data("inventory"),
        inventory_layer=raw.get("inventory_layer"),
        inventory_id_col=str(raw.get("inventory_id_col", "inventory_id")),
        inventory_species_col=str(raw.get("inventory_species_col", "species")),
        species_lookup=species_lookup,
        tables_dir=resolve_path(tables, base=REPO_ROOT),
        figures_dir=resolve_path(figures, base=REPO_ROOT),
        dv_results_dir=resolve_path(dv_results, base=REPO_ROOT),
        nn_m=float(raw.get("nn_m", 0.05)),
        n_min=int(raw.get("n_min", 100)),
        title=title,
        raw=raw,
    )


def load_figures_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Multi-site figure config (lists site YAMLs to combine)."""
    if config_path is None:
        config_path = REPO_ROOT / "configs" / "figures.example.yaml"
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = (REPO_ROOT / cfg_file).resolve()
    with cfg_file.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    out_dir = resolve_path(raw.get("output_dir", "outputs/figures"), base=REPO_ROOT)
    sites = []
    for entry in raw.get("sites", []):
        site = load_site_config(entry["config"])
        sites.append({"site": site, "label": entry.get("label", site.title)})
    return {"output_dir": out_dir, "sites": sites, "raw": raw}
