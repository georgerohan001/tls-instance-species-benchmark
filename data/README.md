# Data directory (not committed)

Place site inputs here. Nothing under `data/` except this README is tracked by git.

## Expected layout per site

```text
data/<site_id>/
├── inputs/
│   ├── forestmamba.laz              # ForestMamba PredInstance (+ optional PredScore)
│   ├── galaxy_sat_final.laz         # Galaxy SAT with PredInstance_SAT
│   ├── manual_SAT_FM_BP1_v2_final.laz   # multi-track DetailView product
│   ├── matched_trees_cleaned_las_index.csv
│   ├── matched_layers_enriched_las_index.csv   # optional; structure covariates
│   └── inventory.gpkg
├── gt_layers/
│   └── tile_001/
│       ├── tree_00001.las
│       ├── misc_inst-1.las          # optional background
│       └── ground_sem0.las          # optional
└── aligned/                         # written by build_aligned / inject_sat
    └── x_eval.npz
```

Outputs (tables, figures) go under `outputs/<site_id>/` by default (see site YAML).

Copy `configs/site.example.yaml` to `configs/<site_id>.yaml` and point relative paths at this tree.
