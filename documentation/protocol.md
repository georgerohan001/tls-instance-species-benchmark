# Evaluation protocol

Instance metrics follow the ForestFormer3D individual-tree protocol (Xiang et al., ICCV 2025). Species metrics follow a FOR-species20K-style stem-level evaluation on DetailView outputs.

This protocol is **site-agnostic**: any plot can be evaluated by pointing a YAML config at relative data paths (see `configs/site.example.yaml`). Thesis example YAMLs are illustrative only.

## Evaluation footprint

1. Collect all points from manual GT layers for the configured tiles (`tree_*`, optional `misc_*`, optional `ground_*`).
2. That XYZ set is \(X_{\mathrm{eval}}\).
3. ForestMamba and SAT labels are transferred onto the same points (exact key match and/or nearest neighbour within `nn_m`).

Background / non-tree GT points use `gt_id = -1`. Tree instances use positive integer `gt_id` parsed from layer filenames.

### GT ID encoding (one layer = one tree)

`build_aligned.parse_gt_id` encodes **one unique id per CloudCompare tree layer file**:

- `tree_01102.laz` → `1102000` (`base * 1000 + 0`)
- `tree_00998_3.laz` → `998003` (`base * 1000 + fragment`)

Fragment suffixes must be `< 1000`. Layers such as `tree_00998_1` / `_2` / `_3` are **separate trees**, not fragments of one stem. Do **not** collapse on `tree_(\d+)` alone.

## Matching and metrics (τ = 0.5 by default)

For predicted instances with at least `n_min` points (default 100):

1. Build pairwise IoU between each GT tree and each prediction from point-set intersection/union.
2. Greedy one-to-one matching by maximum IoU.
3. A match with IoU ≥ τ counts as a true positive (TP).
4. Unmatched predictions → FP; unmatched GT → FN.

\[
\mathrm{Prec}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\quad
\mathrm{Rec}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\quad
\mathrm{F1}=\frac{2\,\mathrm{Prec}\,\mathrm{Rec}}{\mathrm{Prec}+\mathrm{Rec}}
\]

**Coverage (Cov):** fraction of GT trees that receive any matched prediction above a minimum IoU (reported alongside F1; Cov ≥ F1 highlights over-segmentation / soft matches).

Default τ sweep: 0.3, 0.4, 0.5, 0.6, 0.7.

**Hard vs soft per-tree F1** (`analyze_per_tree`): hard F1 uses the τ-matched partner; soft F1 uses the best-IoU partner even when below τ. Macro hard F1 is the mean of per-tree hard F1 scores (T8).

## Structure covariates and packing side checks

Primary packing covariate for interpretation remains **nearest-neighbour distance** among GT centroids (`nn_dist_m`), with height and hull volume as additional structure fields.

Optional density / packing **side checks** (written by `analyze_per_tree` when `tile_xy_bounds` is set):

| Output | Content |
|--------|---------|
| `n_gt_within_{3,5,7}m` | Fixed-radius neighbour counts among GT centroids |
| `edge_censored_5m` | Focal within 5 m of the tile XY footprint edge |
| `hull_overlap_frac` | Fraction of the focal 2D convex hull overlapping other GT hulls |
| T10b | Neighbour-count collinearity / sensitivity (interior focals) |
| T10c | Hull-overlap collinearity / residual-after-NN (interior focals) |

### `tile_xy_bounds` (required for T10b/T10c)

In the site YAML, supply each tile’s footprint as `[xmin, ymin, xmax, ymax]` in the same CRS as the point cloud:

```yaml
tile_xy_bounds:
  1: [0.0, 0.0, 25.0, 25.0]
```

If neighbour-count / overlap covariates are computed and a tile’s bounds are missing, the script exits with an actionable error (it does not silently treat all stems as interior).

## Main instance tables

| Table | Content |
|-------|---------|
| T1 | Overall Prec/Rec/F1/Cov per method |
| T2 | Per-tile breakdown |
| T3 | Error taxonomy (over-/under-seg, missed, leakage) |
| T5 | Size strata |
| T7 | τ sweep |
| density_keep_rates | Subsample robustness curves |
| per_tree_scores | Per-GT-tree IoU / F1 + structure covariates |
| T8–T11 | Macro vs plot F1, covariate strata, correlations, height by size |
| T10b / T10c | Neighbour-count and hull-overlap side tables |

## DetailView / species track

For each inventory-matched stem and each instance source (GT, SAT, FM, FM_BP1):

1. Read `species_id_*` / `species_prob_*` on the multi-track LAZ (see `contracts/layer_map_manual_SAT_FM_BP1_v2.json`).
2. Map field inventory species strings to DetailView ids (`INV_TO_DV` in `src/detailview/constants.py`).
3. Score accuracy / macro-F1 / per-class F1; ignore species id 255 (unclassified).

| Table | Content |
|-------|---------|
| T20 | Leaderboard by site × method |
| T21 | Per-class F1 |
| T23–T25 | Height / lifeform / mature-domain splits |
| T26 | Gap vs segmentation quality |
| detailview_per_tree_v2 | Stem-level predictions |

## Practical checklist

1. Stage or copy manual `gt_layers/tile_XXX/*.las`.
2. `build_aligned` → `aligned/x_eval.npz` with `gt_id`, `mamba_id`, …
3. `inject_sat` from Galaxy final LAZ with `PredInstance_SAT`.
4. `run_metrics` + `analyze_per_tree` (set `tile_xy_bounds` for packing side checks).
5. Prepare multi-track DetailView LAZ; `analyze_detailview`.
6. `prepare_figures` with a multi-site figures YAML.

## Synthetic smoke (portable)

Without study data you can still verify pathways:

```text
python tests/build_tiny_site_fixture.py
python tests/smoke_tiny_site.py
```

This uses `configs/tiny_site.example.yaml` and toy coordinates under `tests/fixtures/tiny_site/` (not thesis plots).
