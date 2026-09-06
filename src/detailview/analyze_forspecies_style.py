"""FOR-species20K-style DetailView benchmark driven by site YAML configs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detailview.constants import (  # noqa: E402
    CHUNK,
    DEFAULT_LOOKUP,
    FORSPECIES_TRAIN_N,
    IGNORE_SPECIES,
    INV_TO_DV,
    LOCAL_TRUE_IDS,
    METHOD_LABEL,
    METHOD_PROB,
    METHOD_SPECIES,
    METHODS,
    lifeform,
)
from src.detailview.metrics import (  # noqa: E402
    bootstrap_oa_ci,
    confusion_counts,
    height_bin_label,
    overall_metrics,
    per_class_rows,
)
from src.detailview import plotting as plots  # noqa: E402
from src.paths import SitePaths, load_site_config, resolve_path  # noqa: E402


def load_lookup(path: Path | None = None) -> dict[int, str]:
    df = pd.read_csv(path or DEFAULT_LOOKUP)
    return dict(zip(df["species_id"].astype(int), df["species"].astype(str)))


def _pick_col(gdf, names: tuple[str, ...]) -> str | None:
    for n in names:
        if n in gdf.columns:
            return n
    return None


def load_stem_meta(site: SitePaths) -> tuple[pd.DataFrame, int]:
    if site.match_csv is None or not site.match_csv.is_file():
        raise SystemExit(f"Missing match_csv for {site.site_id}: {site.match_csv}")
    match = pd.read_csv(site.match_csv)
    match = match.sort_values("n_points", ascending=False).drop_duplicates(
        "inventory_id", keep="first"
    )
    match["inventory_id"] = match["inventory_id"].astype(str)
    match["pred_instance"] = match["pred_instance"].astype(int)

    import geopandas as gpd

    if site.inventory is None or not site.inventory.is_file():
        raise SystemExit(f"Missing inventory for {site.site_id}: {site.inventory}")
    layer = site.inventory_layer
    gdf = gpd.read_file(site.inventory, layer=layer) if layer else gpd.read_file(site.inventory)

    id_col = site.inventory_id_col
    sp_col = site.inventory_species_col
    if id_col not in gdf.columns and id_col == "full_id" and {"plot_id", "tree_id"} <= set(gdf.columns):
        gdf["full_id"] = gdf["plot_id"].astype(str) + "_" + gdf["tree_id"].astype(str)
    if id_col not in gdf.columns:
        raise SystemExit(f"{site.site_id}: inventory missing id column {id_col!r}")
    if sp_col not in gdf.columns:
        raise SystemExit(f"{site.site_id}: inventory missing species column {sp_col!r}")

    # Drop non-numeric TreeID-style rows when the id column is numeric-ish.
    if id_col in ("TreeID", "tree_id"):
        gdf = gdf[pd.to_numeric(gdf[id_col], errors="coerce").notna()].copy()

    height_col = site.raw.get("inventory_height_col") or _pick_col(
        gdf, ("tls_treeheight", "height_m", "Height")
    )
    dbh_col = site.raw.get("inventory_dbh_col") or _pick_col(
        gdf, ("diameter_m", "DBH", "dbh_m")
    )
    height_pref = str(site.raw.get("height_source_pref") or ("inventory" if height_col else "cloud"))

    inv_ids = gdf[id_col]
    if id_col in ("TreeID", "tree_id"):
        inv_ids = pd.to_numeric(inv_ids, errors="coerce").astype("Int64").astype(str)
    else:
        inv_ids = inv_ids.astype(str)

    inv = pd.DataFrame(
        {
            "inventory_id": inv_ids,
            "species": gdf[sp_col].astype(str),
            "height_inv": (
                pd.to_numeric(gdf[height_col], errors="coerce")
                if height_col
                else pd.Series(np.nan, index=gdf.index, dtype=float)
            ),
            "dbh_m": (
                pd.to_numeric(gdf[dbh_col], errors="coerce")
                if dbh_col
                else pd.Series(np.nan, index=gdf.index, dtype=float)
            ),
            "height_source_pref": height_pref,
        }
    )

    stems = match.merge(inv, on="inventory_id", how="left")
    stems["true_dv_id"] = stems["species"].map(INV_TO_DV)
    n_excl = int(stems["true_dv_id"].isna().sum())
    stems["site"] = site.site_id
    return stems, n_excl


def accumulate_by_manual_instance(
    laz_path: Path,
    want_ids: set[int],
    *,
    collect_xyz_for: set[int] | None = None,
    xyz_cap: int = 8000,
) -> tuple[dict, dict, dict]:
    """Mode species + mean prob + height stats per PredInstance; optional XYZ samples."""
    want_ids = set(int(x) for x in want_ids)
    collect_xyz_for = set(int(x) for x in (collect_xyz_for or set()))

    counts: dict[str, dict[int, Counter]] = {m: defaultdict(Counter) for m in METHODS}
    psum: dict[str, dict[int, float]] = {m: defaultdict(float) for m in METHODS}
    pn: dict[str, dict[int, int]] = {m: defaultdict(int) for m in METHODS}
    n_pts: dict[int, int] = defaultdict(int)
    z_vals: dict[int, list[np.ndarray]] = defaultdict(list)
    xyz_buf: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    xyz_n: dict[int, int] = defaultdict(int)

    print(f"Scanning {laz_path.name} for {len(want_ids)} PredInstance ids …", flush=True)
    with laspy.open(laz_path) as reader:
        extras = {d.name for d in reader.header.point_format.extra_dimensions}
        if "PredInstance" not in extras:
            raise SystemExit("Missing PredInstance")
        for m, dim in METHOD_SPECIES.items():
            if dim not in extras:
                raise SystemExit(f"Missing {dim}")
        has_prob = {m: METHOD_PROB[m] in extras for m in METHODS}
        total = reader.header.point_count
        done = 0
        while done < total:
            pts = reader.read_points(min(CHUNK, total - done))
            done += len(pts)
            pid = np.asarray(pts.PredInstance, dtype=np.int32)
            # Only process wanted IDs present in chunk
            uniq = np.unique(pid)
            hit = [int(u) for u in uniq if int(u) in want_ids]
            if not hit:
                if done % (CHUNK * 4) < CHUNK or done >= total:
                    print(f"  scan {done:,}/{total:,}", flush=True)
                continue

            x = np.asarray(pts.x, dtype=np.float64)
            y = np.asarray(pts.y, dtype=np.float64)
            z = np.asarray(pts.z, dtype=np.float64)
            sp = {
                m: np.asarray(getattr(pts, METHOD_SPECIES[m]), dtype=np.int16) for m in METHODS
            }
            pr = {
                m: (
                    np.asarray(getattr(pts, METHOD_PROB[m]), dtype=np.float32)
                    if has_prob[m]
                    else None
                )
                for m in METHODS
            }

            for k in hit:
                sel = pid == k
                n_sel = int(sel.sum())
                n_pts[k] += n_sel
                z_vals[k].append(z[sel])
                if k in collect_xyz_for and xyz_n[k] < xyz_cap:
                    take = min(xyz_cap - xyz_n[k], n_sel)
                    idx = np.flatnonzero(sel)
                    if n_sel > take:
                        idx = idx[:: max(1, n_sel // take)][:take]
                    xyz_buf[k].append((x[idx], y[idx], z[idx]))
                    xyz_n[k] += len(idx)

                for m in METHODS:
                    spm = sp[m]
                    valid = sel & ~np.isin(spm, list(IGNORE_SPECIES))
                    if not np.any(valid):
                        continue
                    vals, cnts = np.unique(spm[valid], return_counts=True)
                    for v, c in zip(vals, cnts):
                        counts[m][k][int(v)] += int(c)
                    if pr[m] is not None:
                        psum[m][k] += float(pr[m][valid].sum())
                        pn[m][k] += int(valid.sum())

            if done % (CHUNK * 4) < CHUNK or done >= total:
                print(f"  scan {done:,}/{total:,}", flush=True)

    modes: dict[str, dict[int, int]] = {}
    confs: dict[str, dict[int, float]] = {}
    labeled_n: dict[str, dict[int, int]] = {}
    for m in METHODS:
        modes[m], confs[m], labeled_n[m] = {}, {}, {}
        for k, ctr in counts[m].items():
            if not ctr:
                continue
            modes[m][k] = int(ctr.most_common(1)[0][0])
            labeled_n[m][k] = int(sum(ctr.values()))
            confs[m][k] = (psum[m][k] / pn[m][k]) if pn[m][k] else float("nan")

    height_cloud: dict[int, float] = {}
    for k, chunks in z_vals.items():
        zz = np.concatenate(chunks)
        if len(zz) < 10:
            continue
        height_cloud[k] = float(np.percentile(zz, 99) - np.percentile(zz, 5))

    xyz_samples: dict[int, dict] = {}
    for k, parts in xyz_buf.items():
        xs = np.concatenate([p[0] for p in parts])
        ys = np.concatenate([p[1] for p in parts])
        zs = np.concatenate([p[2] for p in parts])
        if len(xs) > xyz_cap:
            step = max(1, len(xs) // xyz_cap)
            xs, ys, zs = xs[::step][:xyz_cap], ys[::step][:xyz_cap], zs[::step][:xyz_cap]
        xyz_samples[k] = {"x": xs, "y": ys, "z": zs}

    meta = {"n_pts": n_pts, "height_cloud": height_cloud, "labeled_n": labeled_n}
    return modes, confs, {"meta": meta, "xyz": xyz_samples}


def build_per_tree(
    stems: pd.DataFrame,
    n_excl: int,
    modes: dict,
    confs: dict,
    meta: dict,
    id_to_name: dict,
) -> pd.DataFrame:
    df = stems.copy()
    df["n_excl_site"] = n_excl
    hc = meta["height_cloud"]
    npt = meta["n_pts"]
    labn = meta["labeled_n"]

    df["n_points_cloud"] = df["pred_instance"].map(lambda x: npt.get(int(x), 0))
    df["height_cloud"] = df["pred_instance"].map(lambda x: hc.get(int(x), np.nan))

    h_inv = pd.to_numeric(df.get("height_inv"), errors="coerce")
    use_inv = h_inv.notna() & (h_inv > 0.5) & (df.get("height_source_pref", "inventory") != "cloud")
    # Ettenheim prefers inventory; MW always cloud (height_inv NaN)
    prefer_cloud = df.get("height_source_pref", pd.Series(["inventory"] * len(df))) == "cloud"
    df["height_m"] = np.where(prefer_cloud | ~use_inv, df["height_cloud"], h_inv)
    df["height_source"] = np.where(
        prefer_cloud | ~use_inv,
        "cloud_z99_z05",
        "inventory",
    )
    df["height_bin"] = df["height_m"].map(lambda h: height_bin_label(float(h)) if pd.notna(h) else "NA")

    df["true_name"] = df["true_dv_id"].map(
        lambda x: id_to_name.get(int(x), None) if pd.notna(x) else None
    )
    df["true_lifeform"] = df["true_dv_id"].map(
        lambda x: lifeform(int(x)) if pd.notna(x) else None
    )

    for m in METHODS:
        df[f"pred_id_{m}"] = df["pred_instance"].map(lambda x, mm=m: modes[mm].get(int(x)))
        df[f"pred_prob_{m}"] = df["pred_instance"].map(lambda x, mm=m: confs[mm].get(int(x)))
        df[f"n_labeled_{m}"] = df["pred_instance"].map(
            lambda x, mm=m: labn[mm].get(int(x), 0)
        )
        df[f"pred_name_{m}"] = df[f"pred_id_{m}"].map(
            lambda x: id_to_name.get(int(x), str(int(x))) if pd.notna(x) else None
        )
        df[f"ok_{m}"] = df[f"pred_id_{m}"].notna()
        df[f"correct_{m}"] = (
            df[f"ok_{m}"]
            & df["true_dv_id"].notna()
            & (df[f"pred_id_{m}"] == df["true_dv_id"])
        )
    return df


def evaluate_site(site: SitePaths, id_to_name: dict) -> dict:
    site_key = site.site_id
    if site.detailview_laz is None or not site.detailview_laz.is_file():
        raise SystemExit(f"Missing DetailView LAZ for {site_key}: {site.detailview_laz}")
    laz = site.detailview_laz

    stems, n_excl = load_stem_meta(site)
    want = set(int(x) for x in stems["pred_instance"].tolist())

    # Sanity: PredInstance coverage
    print(f"[{site_key}] stems in match index: {len(stems)}; excluded unmappable: {n_excl}", flush=True)

    # First pass without XYZ; gallery IDs chosen after we know errors
    modes, confs, pack = accumulate_by_manual_instance(laz, want, collect_xyz_for=set())
    found = set(pack["meta"]["n_pts"].keys())
    missing_ids = sorted(want - found)
    if missing_ids:
        print(f"[{site_key}] WARNING missing PredInstance in LAZ: {missing_ids[:20]}…", flush=True)
    else:
        print(f"[{site_key}] All {len(want)} PredInstance IDs present in LAZ", flush=True)

    per = build_per_tree(stems, n_excl, modes, confs, pack["meta"], id_to_name)
    eval_df = per.dropna(subset=["true_dv_id"]).copy()
    eval_df["true_dv_id"] = eval_df["true_dv_id"].astype(int)

    # Collect XYZ for high-confidence GT errors
    err = eval_df[(eval_df["ok_GT"]) & (~eval_df["correct_GT"])].copy()
    err = err.sort_values("pred_prob_GT", ascending=False)
    gallery_ids = set(int(x) for x in err.head(8)["pred_instance"].tolist())
    xyz_samples: dict[int, dict] = {}
    if gallery_ids:
        print(f"[{site_key}] Rescan for misclass gallery ({len(gallery_ids)} trees) …", flush=True)
        _, _, pack2 = accumulate_by_manual_instance(laz, gallery_ids, collect_xyz_for=gallery_ids)
        xyz_samples = pack2["xyz"]

    site.ensure_output_dirs()
    tables = site.tables_dir
    figures = site.figures_dir
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    # Per-tree CSV
    out_cols = [
        "site",
        "inventory_id",
        "pred_instance",
        "species",
        "true_dv_id",
        "true_name",
        "true_lifeform",
        "height_m",
        "height_source",
        "height_bin",
        "dbh_m",
        "n_points",
        "n_points_cloud",
    ]
    for m in METHODS:
        out_cols += [
            f"pred_id_{m}",
            f"pred_name_{m}",
            f"pred_prob_{m}",
            f"n_labeled_{m}",
            f"ok_{m}",
            f"correct_{m}",
        ]
    per_out = eval_df[[c for c in out_cols if c in eval_df.columns]].copy()
    per_out.to_csv(tables / "detailview_per_tree_v2.csv", index=False)

    true_labels = sorted(eval_df["true_dv_id"].unique().tolist())
    leaderboard_rows = []
    per_class_all = []
    conf_payload = {}

    for m in METHODS:
        ok = eval_df[eval_df[f"ok_{m}"]].copy()
        n_miss = int((~eval_df[f"ok_{m}"]).sum())
        if len(ok) == 0:
            leaderboard_rows.append(
                {
                    "site": site_key,
                    "method": m,
                    "method_label": METHOD_LABEL[m],
                    "N": 0,
                    "N_excluded_unmappable": n_excl,
                    "N_missing_pred": n_miss,
                    "accuracy": float("nan"),
                    "balanced_accuracy": float("nan"),
                    "macro_precision": float("nan"),
                    "macro_recall": float("nan"),
                    "macro_f1": float("nan"),
                    "weighted_f1": float("nan"),
                    "oa_ci_lo": float("nan"),
                    "oa_ci_hi": float("nan"),
                }
            )
            continue
        y_true = ok["true_dv_id"].to_numpy(dtype=int)
        y_pred = ok[f"pred_id_{m}"].astype(int).to_numpy()
        # Include predicted open-set labels in confusion axis
        pred_labs = sorted(set(y_pred.tolist()) | set(true_labels))
        met = overall_metrics(y_true, y_pred, true_labels)
        lo, hi = bootstrap_oa_ci(y_true, y_pred, n_boot=1000, seed=0)
        leaderboard_rows.append(
            {
                "site": site_key,
                "method": m,
                "method_label": METHOD_LABEL[m],
                "N_excluded_unmappable": n_excl,
                "N_missing_pred": n_miss,
                "oa_ci_lo": lo,
                "oa_ci_hi": hi,
                **met,
            }
        )
        per_class_all.extend(
            per_class_rows(
                y_true,
                y_pred,
                true_labels,
                id_to_name,
                site=site_key,
                method=m,
                train_n=FORSPECIES_TRAIN_N,
            )
        )
        cm = confusion_counts(y_true, y_pred, pred_labs)
        names = [id_to_name.get(i, str(i)) for i in pred_labs]
        pd.DataFrame(cm, index=names, columns=names).to_csv(
            tables / f"T22_detailview_confusion_{site_key}_{m}.csv"
        )
        # Also percent
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        pct = 100.0 * cm / row_sum
        pd.DataFrame(pct, index=names, columns=names).to_csv(
            tables / f"T22_detailview_confusion_{site_key}_{m}_pct.csv"
        )
        plots.plot_confusion(
            cm,
            names,
            f"{site_key.capitalize()} · {METHOD_LABEL[m].replace('DetailView_', 'DetailView ')}",
            figures / f"dv_confusion_{site_key}_{m}.png",
        )
        conf_payload[m] = {"labels": pred_labs, "names": names, "cm": cm}

    t20 = pd.DataFrame(leaderboard_rows)
    t21 = pd.DataFrame(per_class_all)

    # Height strata
    height_rows = []
    hist_rows = []
    for m in METHODS:
        for b, g in eval_df.groupby("height_bin"):
            if b == "NA":
                continue
            ok = g[g[f"ok_{m}"]]
            if len(ok) == 0:
                continue
            acc = 100.0 * float((ok[f"pred_id_{m}"] == ok["true_dv_id"]).mean())
            height_rows.append(
                {
                    "site": site_key,
                    "method": m,
                    "height_bin": b,
                    "N": int(len(ok)),
                    "accuracy": acc,
                }
            )
    # histogram of stems by height (once)
    for b, g in eval_df.groupby("height_bin"):
        if b == "NA":
            continue
        hist_rows.append({"site": site_key, "height_bin": b, "n": int(len(g))})

    t23 = pd.DataFrame(height_rows)
    height_hist = pd.DataFrame(hist_rows)

    # Lifeform
    life_rows = []
    for m in METHODS:
        for lf, g in eval_df.groupby("true_lifeform"):
            ok = g[g[f"ok_{m}"]]
            if len(ok) == 0:
                continue
            life_rows.append(
                {
                    "site": site_key,
                    "method": m,
                    "lifeform": lf,
                    "N": int(len(ok)),
                    "accuracy": 100.0
                    * float((ok[f"pred_id_{m}"] == ok["true_dv_id"]).mean()),
                }
            )
    t24 = pd.DataFrame(life_rows)

    # Mature domain height > 5
    mature_rows = []
    mature = eval_df[eval_df["height_m"] > 5]
    for m in METHODS:
        ok = mature[mature[f"ok_{m}"]]
        if len(ok) == 0:
            continue
        y_true = ok["true_dv_id"].to_numpy(dtype=int)
        y_pred = ok[f"pred_id_{m}"].astype(int).to_numpy()
        met = overall_metrics(y_true, y_pred, true_labels)
        mature_rows.append(
            {
                "site": site_key,
                "method": m,
                "filter": "height_gt_5m",
                "species_present": ",".join(id_to_name.get(i, str(i)) for i in true_labels),
                **met,
            }
        )
    t25 = pd.DataFrame(mature_rows)

    # Open-set preds
    open_rows = []
    for m in METHODS:
        ok = eval_df[eval_df[f"ok_{m}"]]
        for _, r in ok.iterrows():
            pid = int(r[f"pred_id_{m}"])
            if pid not in LOCAL_TRUE_IDS:
                open_rows.append(
                    {
                        "site": site_key,
                        "method": m,
                        "inventory_id": r["inventory_id"],
                        "true_name": r["true_name"],
                        "pred_id": pid,
                        "pred_name": id_to_name.get(pid, str(pid)),
                        "pred_prob": r[f"pred_prob_{m}"],
                    }
                )
    open_df = pd.DataFrame(open_rows)

    # Misclass table (GT track)
    mis = eval_df[(eval_df["ok_GT"]) & (~eval_df["correct_GT"])].copy()
    mis = mis.sort_values("pred_prob_GT", ascending=False)
    mis_out = mis[
        [
            c
            for c in [
                "inventory_id",
                "pred_instance",
                "true_name",
                "pred_name_GT",
                "pred_prob_GT",
                "height_m",
                "n_points_cloud",
            ]
            if c in mis.columns
        ]
    ]
    mis_out.to_csv(tables / f"detailview_misclass_GT_{site_key}.csv", index=False)
    if len(mis_out):
        plots.plot_misclass_gallery(
            xyz_samples,
            mis_out.rename(columns={"pred_name_GT": "pred_name_GT"}),
            figures / f"dv_misclass_gallery_{site_key}.png",
            title=f"{site_key}: high-confidence DetailView_manual errors",
        )

    # Species counts note
    counts = eval_df["true_name"].value_counts().to_dict()
    print(f"[{site_key}] species counts: {counts}", flush=True)
    print(t20.to_string(index=False), flush=True)

    # Write site tables
    t20.to_csv(tables / "T20_detailview_leaderboard.csv", index=False)
    t21.to_csv(tables / "T21_detailview_per_class.csv", index=False)
    t23.to_csv(tables / "T23_detailview_by_height.csv", index=False)
    t24.to_csv(tables / "T24_detailview_by_lifeform.csv", index=False)
    t25.to_csv(tables / "T25_detailview_mature_domain.csv", index=False)
    if len(open_df):
        open_df.to_csv(tables / "open_set_preds.csv", index=False)
    else:
        pd.DataFrame(columns=["site", "method", "inventory_id"]).to_csv(
            tables / "open_set_preds.csv", index=False
        )

    # stem_meta
    stem_meta = eval_df[
        [
            c
            for c in [
                "site",
                "inventory_id",
                "pred_instance",
                "species",
                "true_dv_id",
                "true_name",
                "true_lifeform",
                "height_m",
                "height_source",
                "dbh_m",
                "n_points",
                "n_points_cloud",
            ]
            if c in eval_df.columns
        ]
    ]
    stem_meta.to_csv(tables / "stem_meta.csv", index=False)

    return {
        "site": site_key,
        "site_paths": site,
        "per_tree": per_out,
        "t20": t20,
        "t21": t21,
        "t23": t23,
        "t24": t24,
        "t25": t25,
        "height_hist": height_hist,
        "open_df": open_df,
        "n_excl": n_excl,
        "species_counts": counts,
        "eval_df": eval_df,
    }


def seg_quality_gaps(t20_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site in sorted(t20_all["site"].unique()):
        if site == "both":
            continue
        sub = t20_all[t20_all["site"] == site].set_index("method")
        if "GT" not in sub.index:
            continue
        gt_oa = float(sub.loc["GT", "accuracy"])
        gt_f1 = float(sub.loc["GT", "macro_f1"])
        for m in ("SAT", "FM", "FM_BP1"):
            if m not in sub.index:
                continue
            rows.append(
                {
                    "site": site,
                    "contrast": f"GT_minus_{m}",
                    "delta_oa": gt_oa - float(sub.loc[m, "accuracy"]),
                    "delta_macro_f1": gt_f1 - float(sub.loc[m, "macro_f1"]),
                    "oa_GT": gt_oa,
                    f"oa_{m}": float(sub.loc[m, "accuracy"]),
                }
            )
        if "FM" in sub.index and "FM_BP1" in sub.index:
            rows.append(
                {
                    "site": site,
                    "contrast": "FM_BP1_minus_FM",
                    "delta_oa": float(sub.loc["FM_BP1", "accuracy"])
                    - float(sub.loc["FM", "accuracy"]),
                    "delta_macro_f1": float(sub.loc["FM_BP1", "macro_f1"])
                    - float(sub.loc["FM", "macro_f1"]),
                    "oa_GT": gt_oa,
                    "oa_FM": float(sub.loc["FM", "accuracy"]),
                    "oa_FM_BP1": float(sub.loc["FM_BP1", "accuracy"]),
                }
            )
    return pd.DataFrame(rows)


def agreement_rows(eval_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in eval_frames:
        site = df["site"].iloc[0]
        base = df.dropna(subset=["true_dv_id"])
        pairs = [("GT", "FM"), ("GT", "SAT"), ("GT", "FM_BP1"), ("FM", "FM_BP1")]
        for a, b in pairs:
            both = base[base[f"ok_{a}"] & base[f"ok_{b}"]]
            if len(both) == 0:
                continue
            agree = float((both[f"pred_id_{a}"] == both[f"pred_id_{b}"]).mean())
            rows.append(
                {
                    "site": site,
                    "pair": f"{a}_vs_{b}",
                    "N": int(len(both)),
                    "agreement": 100.0 * agree,
                }
            )
    return pd.DataFrame(rows)


def write_combined_figures(results: list[dict], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    t20 = pd.concat([r["t20"] for r in results], ignore_index=True)
    t21 = pd.concat([r["t21"] for r in results], ignore_index=True)
    t23 = pd.concat([r["t23"] for r in results], ignore_index=True)
    t24 = pd.concat([r["t24"] for r in results], ignore_index=True)
    height_hist = pd.concat([r["height_hist"] for r in results], ignore_index=True)
    per_tree = pd.concat([r["per_tree"] for r in results], ignore_index=True)

    # Pooled both
    pool_rows = []
    for m in METHODS:
        chunks = []
        for r in results:
            d = r["eval_df"]
            ok = d[d[f"ok_{m}"]]
            chunks.append(ok)
        ok = pd.concat(chunks, ignore_index=True)
        if len(ok) == 0:
            continue
        labels = sorted(ok["true_dv_id"].unique().tolist())
        met = overall_metrics(
            ok["true_dv_id"].to_numpy(dtype=int),
            ok[f"pred_id_{m}"].astype(int).to_numpy(),
            labels,
        )
        pool_rows.append(
            {
                "site": "both",
                "method": m,
                "method_label": METHOD_LABEL[m],
                "N_excluded_unmappable": sum(r["n_excl"] for r in results),
                "N_missing_pred": 0,
                "oa_ci_lo": float("nan"),
                "oa_ci_hi": float("nan"),
                **met,
            }
        )
    t20 = pd.concat([t20, pd.DataFrame(pool_rows)], ignore_index=True)

    t26 = seg_quality_gaps(t20)
    agree = agreement_rows([r["eval_df"] for r in results])
    t26_full = t26.copy()
    if len(agree):
        agree.to_csv(results_dir / "T26b_pred_agreement.csv", index=False)

    plots.plot_leaderboard(t20, fig_dir / "dv_leaderboard_bars.png")
    plots.plot_f1_vs_support(t21, fig_dir / "dv_f1_vs_support.png")
    plots.plot_oa_by_height(t23, height_hist, fig_dir / "dv_oa_by_height.png")
    plots.plot_lifeform(t24, fig_dir / "dv_lifeform_bars.png")
    plots.plot_seg_quality_delta(t20, fig_dir / "dv_seg_quality_delta.png")
    plots.plot_confidence(per_tree, fig_dir / "dv_confidence_correct_vs_err.png")

    # Copy combined figs into each site figures too
    for r in results:
        site_fig: Path = r["site_paths"].figures_dir
        site_fig.mkdir(parents=True, exist_ok=True)
        for fn in fig_dir.glob("dv_*.png"):
            dest = site_fig / fn.name
            dest.write_bytes(fn.read_bytes())

    t20.to_csv(results_dir / "T20_detailview_leaderboard.csv", index=False)
    t21.to_csv(results_dir / "T21_detailview_per_class.csv", index=False)
    t23.to_csv(results_dir / "T23_detailview_by_height.csv", index=False)
    t24.to_csv(results_dir / "T24_detailview_by_lifeform.csv", index=False)
    pd.concat([r["t25"] for r in results], ignore_index=True).to_csv(
        results_dir / "T25_detailview_mature_domain.csv", index=False
    )
    t26_full.to_csv(results_dir / "T26_seg_quality_gap.csv", index=False)
    per_tree.to_csv(results_dir / "detailview_per_tree_v2.csv", index=False)
    open_all = pd.concat(
        [r["open_df"] for r in results if len(r["open_df"])], ignore_index=True
    ) if any(len(r["open_df"]) for r in results) else pd.DataFrame()
    open_all.to_csv(results_dir / "open_set_preds.csv", index=False)

    # Also write combined slices into each site tables
    for r in results:
        site_tab: Path = r["site_paths"].tables_dir
        site_tab.mkdir(parents=True, exist_ok=True)
        t20[t20["site"].isin([r["site"], "both"])].to_csv(
            site_tab / "T20_detailview_leaderboard.csv", index=False
        )
        t26_full[t26_full["site"] == r["site"]].to_csv(
            site_tab / "T26_seg_quality_gap.csv", index=False
        )

    summary = {
        "sites": {r["site"]: r["species_counts"] for r in results},
        "leaderboard": t20.to_dict(orient="records"),
        "seg_quality_gaps": t26_full.to_dict(orient="records"),
        "agreement": agree.to_dict(orient="records") if len(agree) else [],
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        rel = results_dir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        rel = results_dir
    print("Wrote combined results to", rel, flush=True)
    print(t20.to_string(index=False), flush=True)
    if len(t26_full):
        print(t26_full.to_string(index=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        nargs="+",
        required=True,
                        help="One or more site YAML configs (paths relative to repo root).",
    )
    ap.add_argument(
        "--results-dir",
        default="outputs/detailview",
        help="Combined DetailView outputs directory (relative to repo root).",
    )
    args = ap.parse_args()

    sites = [load_site_config(p) for p in args.config]
    results_dir = resolve_path(args.results_dir, base=REPO_ROOT)

    results = []
    for site in sites:
        id_to_name = load_lookup(site.species_lookup)
        results.append(evaluate_site(site, id_to_name))
    if len(results) >= 1:
        write_combined_figures(results, results_dir)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()

