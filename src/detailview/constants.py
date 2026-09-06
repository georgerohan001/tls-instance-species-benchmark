"""Shared constants for DetailView FOR-species20K-style benchmark."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOOKUP = REPO_ROOT / "contracts" / "detailview_lookup.csv"
LOOKUP = DEFAULT_LOOKUP

METHODS = ("GT", "SAT", "FM", "FM_BP1")
METHOD_SPECIES = {
    "GT": "species_id_GT",
    "SAT": "species_id_SAT",
    "FM": "species_id_FM",
    "FM_BP1": "species_id_FM_BP1",
}
METHOD_PROB = {
    "GT": "species_prob_GT",
    "SAT": "species_prob_SAT",
    "FM": "species_prob_FM",
    "FM_BP1": "species_prob_FM_BP1",
}
METHOD_LABEL = {
    "GT": "DetailView_manual",
    "SAT": "DetailView_SAT",
    "FM": "DetailView_FM",
    "FM_BP1": "DetailView_FM_BP1",
}

INV_TO_DV = {
    "Beech": 10,
    "BE": 10,
    "Silver Fir": 0,
    "ESF": 0,
    "Spruce": 14,
    "NS": 14,
    "Douglas Fir": 25,
    "DF": 25,
    "Larch": 13,
}

# FOR-species20K training counts (paper text / Fig 3 approximate)
FORSPECIES_TRAIN_N = {
    0: 119,  # Abies_alba
    10: 2482,  # Fagus_sylvatica
    13: 94,  # Larix_decidua
    14: 1983,  # Picea_abies
    25: 350,  # Pseudotsuga_menziesii (approx temperate share)
}

CONIFER_IDS = {0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25}
BROADLEAF_IDS = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32
}
LOCAL_TRUE_IDS = {0, 10, 13, 14, 25}

CHUNK = 2_000_000
# DetailView uses 255 as nodata; species_id 0 is Abies_alba (valid class).
IGNORE_SPECIES = {255}

# Deprecated: site paths come from YAML via src.paths.load_site_config.
SITE_CFG: dict = {}


def lifeform(dv_id: int) -> str:
    if int(dv_id) in CONIFER_IDS:
        return "conifer"
    if int(dv_id) in BROADLEAF_IDS:
        return "broadleaf"
    return "other"
