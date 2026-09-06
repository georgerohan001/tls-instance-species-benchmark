# Galaxy and DetailView field maps

SAT and DetailView are run **outside** this repository (for example on [Galaxy Europe](https://usegalaxy.eu/)). This document records the field naming used by the evaluation scripts.

## Recommended uploads

| Tool | Input cloud | Notes |
|------|-------------|--------|
| SegmentAnyTree | XYZ-only clip of the evaluation footprint | No instance labels |
| DetailView | Cloud with an instance column | Tree ID column = that instance field; model region as appropriate (e.g. Europe) |

Clip the upload to the same XY footprint as your manual GT tiles (small margin is fine).

## After SegmentAnyTree

Expect instance (and optionally semantic) fields. This pipeline looks for:

| Field | Role |
|-------|------|
| `PredInstance_SAT` | SAT instance id on the Galaxy final product |
| `PredSemantic_SAT` | Optional semantic class |

`scripts/inject_sat.py` streams `PredInstance_SAT` onto `x_eval.npz`.

## After DetailView on each instance source

DetailView does not segment; it classifies existing instances. Run it separately (or in a merged product) for:

| Instance source | Species fields consumed here |
|-----------------|------------------------------|
| Manual GT | `species_id_GT`, `species_prob_GT` |
| SAT | `species_id_SAT`, `species_prob_SAT` |
| ForestMamba | `species_id_FM`, `species_prob_FM` |
| ForestMamba + BluePoint | `species_id_FM_BP1`, `species_prob_FM_BP1` |

A convenient product keeps **all** of the above on one LAZ together with `PredInstance` (manual) and the various `PredInstance_*` fields. The dimension list is in `contracts/layer_map_manual_SAT_FM_BP1_v2.json`.

## Suggested DetailView settings

- Tree ID column: the instance field you want classified (`PredInstance`, or the SAT/FM instance column for that run)
- Projection backend: PyTorch (or NumPy)
- Unclassified species id: **255** (ignored by metrics)

## Wiring outputs into this repo

1. Place the Galaxy final SAT LAZ at the path given by `sat_laz` in your site YAML (under `data/<site>/inputs/ÔÇª`).
2. Place the multi-track DetailView LAZ at `detailview_laz`.
3. Run `inject_sat` then `analyze_detailview`.
