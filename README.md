# TLS instance segmentation and DetailView species pipeline

Private reproducibility package for an MSc thesis workflow that benchmarks **ForestMamba** and **SegmentAnyTree (SAT)** instance segmentation against **manual** tree instances, then evaluates **DetailView** species classification on multiple instance sources (manual, SAT, FM, FM+BluePoint).

This repository ships **code, configs, and contracts only**. Point clouds, inventories, and result tables are not included. Obtain study data separately if you need to reproduce thesis numbers, or point the configs at your own plots.

## What this is / is not

| In scope | Out of scope |
|----------|----------------|
| Align GT + FM (+ inject SAT) into `x_eval.npz` | Running ForestMamba / SAT / DetailView inference |
| ForestFormer3D-style metrics (T1–T7, T8–T11) | CloudCompare BIN editing |
| DetailView FOR-species20K-style tables (T20–T26) | Shipping multi-GB LAZ / BIN files |
| Paper-style Results figures from CSV tables | Thesis LaTeX / HTML preview |

Inference of SAT, ForestMamba, and DetailView is assumed to happen externally (e.g. Galaxy Europe). This repo evaluates outputs you already have.

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`
- Optional: set `TLS_DATA_ROOT` to an external folder if large inputs should not live under the clone

## Quick start

```powershell
cd tls-instance-species-benchmark
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Copy an example config and edit relative paths
copy configs\site.example.yaml configs\my_site.yaml

# 2. Place inputs under data/my_site/ (see data/README.md)

# 3. Instance segmentation track
python scripts/stage_gt_layers.py --config configs/my_site.yaml
python scripts/build_aligned.py --config configs/my_site.yaml
python scripts/inject_sat.py --config configs/my_site.yaml
python scripts/run_metrics.py --config configs/my_site.yaml
python scripts/analyze_per_tree.py --config configs/my_site.yaml

# 4. DetailView species track (after multi-track LAZ exists)
python scripts/analyze_detailview.py --config configs/my_site.yaml

# 5. Multi-site Results figures (after tables exist for each site)
copy configs\figures.example.yaml configs\figures.yaml
# edit figures.yaml to list your site configs
python scripts/prepare_figures.py --config configs/figures.yaml
```

## Pipeline

```text
manual GT LAS (tile_*/tree_*.las)
        + ForestMamba LAZ
        + Galaxy SAT LAZ
              │
              ▼
     build_aligned → x_eval.npz
              │
     inject_sat (PredInstance_SAT)
              │
              ├─► run_metrics      → T1–T7, density curves, seg figures
              └─► analyze_per_tree → T8–T11, per_tree_scores.csv

multi-track DetailView LAZ + inventory match CSV
              │
              ▼
     analyze_detailview → T20–T26, per-stem species tables

tables from one or more sites
              │
              ▼
     prepare_figures → combined Results PNGs
```

## Adding a new site

1. Copy `configs/site.example.yaml` → `configs/<site_id>.yaml`.
2. Set `site_id`, `data_root: data/<site_id>`, and `tiles`.
3. Fill `data/<site_id>/` using the layout in [`data/README.md`](data/README.md).
4. Map inventory columns (`inventory_id_col`, `inventory_species_col`) to your GeoPackage.
5. Run the commands above.

Paths in YAML must be **relative** (to `data_root` or the repo root). Absolute paths are rejected so the package stays portable and does not embed machine-specific locations.

## Reproducing thesis sites

Example configs: `configs/ettenheim.example.yaml`, `configs/mathislewald.example.yaml`.  
They describe the expected layout only. Place data you obtain separately under `data/ettenheim/` and `data/mathislewald/`, then copy the example YAMLs to non-`.example` names (or pass the example paths if your layout matches).

## Documentation

- [`docs/protocol.md`](docs/protocol.md) — metrics definitions and evaluation rules
- [`docs/input_contracts.md`](docs/input_contracts.md) — required LAZ fields and CSV columns
- [`docs/galaxy_field_maps.md`](docs/galaxy_field_maps.md) — Galaxy / DetailView field naming
- [`contracts/`](contracts/) — species id lookup and multi-track dimension map

## Privacy

- Do not commit anything under `data/` (gitignored).
- Do not put absolute paths in configs or code.
- Large data may live outside the repo via `TLS_DATA_ROOT`.

## License / access

Private repository. Invite collaborators from the GitHub settings page.
