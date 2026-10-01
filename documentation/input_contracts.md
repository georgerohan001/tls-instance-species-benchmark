# Input contracts

## Manual GT layers

Directory: `gt_layers/tile_XXX/` (XXX = zero-padded tile id from config `tiles`).

| Pattern | Role |
|---------|------|
| `tree_*.las` / `.laz` | One manual tree instance; `gt_id` from fragment-safe encoding (see below) |
| `misc_inst*.las` | Non-tree / leftover points (`gt_id = -1`) |
| `ground*.las` | Ground points (`gt_id = -1`) |

Prefer attaching ForestMamba fields on the same points when staging (`PredInstance_FM`, optional `PredScore_FM`). Otherwise `build_aligned` nearest-neighbour transfers from `fm_laz`.

### GT ID encoding

One layer file = one GT tree. Encoding is `base * 1000 + fragment` where `tree_#####.laz` has fragment `0` and `tree_#####_N.laz` uses suffix `N` (`N < 1000`). Do not collapse multiple `_N` layers onto the same base id.

## Tile XY bounds (`tile_xy_bounds`)

Optional mapping in the site YAML; **required** to run neighbour-count edge censoring and T10b/T10c:

```yaml
tile_xy_bounds:
  1: [xmin, ymin, xmax, ymax]
```

Coordinates must match the point-cloud CRS. Example thesis footprints appear only in `configs/*.example.yaml` as illustrations — any site can supply its own values.

## ForestMamba LAZ (`fm_laz`)

Required extra dimension: `PredInstance_FM` or `PredInstance` (script accepts FM-specific name when present).

Same absolute XYZ frame as the manual GT.

## Galaxy SAT final LAZ (`sat_laz`)

Required: `PredInstance_SAT`.

Used by `inject_sat` to fill `sat_id` in `x_eval.npz`.

## Multi-track DetailView product (`detailview_laz`)

See `contracts/layer_map_manual_SAT_FM_BP1_v2.json`. Minimum useful set:

- `PredInstance` ÔÇö manual inventory-matched stem id
- `species_id_GT`, `species_prob_GT`
- `species_id_SAT`, `species_prob_SAT`
- `species_id_FM`, `species_prob_FM`
- `species_id_FM_BP1`, `species_prob_FM_BP1`

`species_id == 255` means unclassified.

### PredInstance encoding

Use a stable integer per inventory stem (one PredInstance Ôåö one stem). Example schemes:

- Composite plot/tree codes (e.g. plot ├ù 1000 + tree)
- Native inventory `TreeID`

The match CSV must use the same ids.

## Match CSV (`match_csv`)

Inventory-matched stems used for DetailView evaluation. Expected columns (names may vary; `pred_instance` / path columns are typical):

| Column | Meaning |
|--------|---------|
| `inventory_id` | Stem id in the inventory |
| `pred_instance` | Integer PredInstance on the LAZ |
| `cleaned_las` | Optional path to per-stem cloud (not required for metrics if LAZ is complete) |
| `n_points` | Optional |

## Enriched match CSV (`enriched_match_csv`, optional)

Links manual layer names to inventory ids for structure covariates in Results figures. Useful columns: `layer`, `inventory_id`, plus any size/height fields you already computed.

## Inventory GeoPackage (`inventory`)

| Config key | Purpose |
|------------|---------|
| `inventory_layer` | Layer name inside the GPKG (or `null`) |
| `inventory_id_col` | Stem id column |
| `inventory_species_col` | Species string / code column |

Species strings are mapped through `INV_TO_DV` in `src/detailview/constants.py` (extend that dict for new taxa).

## Aligned array (`aligned/x_eval.npz`)

Written by `build_aligned` / updated by `inject_sat`. Typical arrays:

`x, y, z, gt_id, tile_id, mamba_id, mamba_score, sat_id, kind`

## Species lookup

`contracts/detailview_lookup.csv`: `species,species_id` for FOR-species20K-style class names.
