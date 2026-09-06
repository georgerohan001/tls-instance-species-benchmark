# Evaluation protocol

Instance metrics follow the ForestFormer3D individual-tree protocol (Xiang et al., ICCV 2025). Species metrics follow a FOR-species20K-style stem-level evaluation on DetailView outputs.

## Evaluation footprint

1. Collect all points from manual GT layers for the configured tiles (`tree_*`, optional `misc_*`, optional `ground_*`).
2. That XYZ set is \(X_{\mathrm{eval}}\).
3. ForestMamba and SAT labels are transferred onto the same points (exact key match and/or nearest neighbour within `nn_m`).

Background / non-tree GT points use `gt_id = -1`. Tree instances use positive integer `gt_id` parsed from `tree_#####` filenames (or your staging convention).

## Matching and metrics (¤ä = 0.5 by default)

For predicted instances with at least `n_min` points (default 100):

1. Build pairwise IoU between each GT tree and each prediction from point-set intersection/union.
2. Greedy one-to-one matching by maximum IoU.
3. A match with IoU ÔëÑ ¤ä counts as a true positive (TP).
4. Unmatched predictions ÔåÆ FP; unmatched GT ÔåÆ FN.

\[
\mathrm{Prec}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\quad
\mathrm{Rec}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\quad
\mathrm{F1}=\frac{2\,\mathrm{Prec}\,\mathrm{Rec}}{\mathrm{Prec}+\mathrm{Rec}}
\]

**Coverage (Cov):** fraction of GT trees that receive any matched prediction above a minimum IoU (reported alongside F1; Cov ÔêÆ F1 highlights over-segmentation / soft matches).

Default ¤ä sweep: 0.3, 0.4, 0.5, 0.6, 0.7.

## Main instance tables

| Table | Content |
|-------|---------|
| T1 | Overall Prec/Rec/F1/Cov per method |
| T2 | Per-tile breakdown |
| T3 | Error taxonomy (over-/under-seg, missed, leakage) |
| T5 | Size strata |
| T7 | ¤ä sweep |
| density_keep_rates | Subsample robustness curves |
| per_tree_scores | Per-GT-tree IoU / F1 + structure covariates |
| T8ÔÇôT11 | Macro vs plot F1, covariate strata, correlations, height by size |

## DetailView / species track

For each inventory-matched stem and each instance source (GT, SAT, FM, FM_BP1):

1. Read `species_id_*` / `species_prob_*` on the multi-track LAZ (see `contracts/layer_map_manual_SAT_FM_BP1_v2.json`).
2. Map field inventory species strings to DetailView ids (`INV_TO_DV` in `src/detailview/constants.py`).
3. Score accuracy / macro-F1 / per-class F1; ignore species id 255 (unclassified).

| Table | Content |
|-------|---------|
| T20 | Leaderboard by site ├ù method |
| T21 | Per-class F1 |
| T23ÔÇôT25 | Height / lifeform / mature-domain splits |
| T26 | Gap vs segmentation quality |
| detailview_per_tree_v2 | Stem-level predictions |

## Practical checklist

1. Stage or copy manual `gt_layers/tile_XXX/*.las`.
2. `build_aligned` ÔåÆ `aligned/x_eval.npz` with `gt_id`, `mamba_id`, ÔÇª
3. `inject_sat` from Galaxy final LAZ with `PredInstance_SAT`.
4. `run_metrics` + `analyze_per_tree`.
5. Prepare multi-track DetailView LAZ; `analyze_detailview`.
6. `prepare_figures` with a multi-site figures YAML.
