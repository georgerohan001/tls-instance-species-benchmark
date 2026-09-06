"""Compute ForestFormer3D-style metrics, tables, and figures from x_eval.npz."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.paths import SitePaths, load_site_config

ALIGNED_NPZ: Path | None = None
ALIGNED_DIR: Path | None = None
TABLES: Path | None = None
FIGURES: Path | None = None
SITE_TITLE = ""
N_MIN = 100
NN_M = 0.05
TILES: tuple[int, ...] = ()
GT_DIR: Path | None = None
FM_LAZ: Path | None = None
SAT_LAZ: Path | None = None

TAUS = [0.3, 0.4, 0.5, 0.6, 0.7]


def configure(site: SitePaths) -> None:
    global ALIGNED_NPZ, ALIGNED_DIR, TABLES, FIGURES, SITE_TITLE
    global N_MIN, NN_M, TILES, GT_DIR, FM_LAZ, SAT_LAZ
    ALIGNED_NPZ = site.aligned
    ALIGNED_DIR = site.aligned.parent
    TABLES = site.tables_dir
    FIGURES = site.figures_dir
    SITE_TITLE = site.title
    N_MIN = site.n_min
    NN_M = site.nn_m
    TILES = site.tiles
    GT_DIR = site.gt_layers_dir
    FM_LAZ = site.fm_laz
    SAT_LAZ = site.sat_laz
    site.ensure_output_dirs()


def load_data():
    z = np.load(ALIGNED_NPZ)
    return {k: z[k] for k in z.files}


def metrics_from_counts(tp, fp, fn, cov):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "Prec": 100 * prec,
        "Rec": 100 * rec,
        "F1": 100 * f1,
        "Cov": 100 * cov,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


def build_iou_and_match(gt_id, pred_id, tau=0.5, n_min=None, pred_positive_min=1):
    """Greedy max-IoU matching. pred_positive_min: 1 for mamba (>0), 1 for sat (>0)."""
    if n_min is None:
        n_min = N_MIN
    gt_mask = gt_id > 0
    gt_labels = np.unique(gt_id[gt_mask])
    pred_all = pred_id.copy()
    pred_all[pred_all < pred_positive_min] = -1

    pred_labels, pred_counts = np.unique(pred_all[pred_all > 0], return_counts=True)
    keep_pred = set(pred_labels[pred_counts >= n_min].tolist())
    if not keep_pred:
        empty = metrics_from_counts(0, 0, len(gt_labels), 0.0)
        empty.update(
            {
                "N_pred": 0,
                "N_GT": len(gt_labels),
                "max_iou": np.zeros(len(gt_labels)),
                "matches": [],
                "gt_labels": gt_labels,
                "pred_labels": np.array([], dtype=np.int32),
                "iou_pairs": {},
            }
        )
        return empty

    gt_labels = np.asarray(sorted(gt_labels), dtype=np.int32)
    pred_labels = np.asarray(sorted(keep_pred), dtype=np.int32)
    g_index = {int(g): i for i, g in enumerate(gt_labels)}
    p_index = {int(p): i for i, p in enumerate(pred_labels)}
    n_g, n_p = len(gt_labels), len(pred_labels)

    df = pd.DataFrame({"g": gt_id, "p": pred_all})
    df = df[(df["g"] > 0) & (df["p"].isin(keep_pred))]
    if len(df) == 0:
        pair_counts = {}
    else:
        vc = df.groupby(["g", "p"], sort=False).size()
        pair_counts = {(int(g), int(p)): int(c) for (g, p), c in vc.items()}

    gt_sizes = {int(g): int(np.sum(gt_id == g)) for g in gt_labels}
    pred_sizes = {int(p): int(np.sum(pred_all == p)) for p in pred_labels}

    triples = []
    iou_mat = np.zeros((n_g, n_p), dtype=np.float32)
    for (g, p), inter in pair_counts.items():
        if g not in g_index or p not in p_index:
            continue
        union = gt_sizes[g] + pred_sizes[p] - inter
        iou = inter / union if union else 0.0
        gi, pj = g_index[g], p_index[p]
        iou_mat[gi, pj] = iou
        triples.append((iou, gi, pj, g, p))

    triples.sort(reverse=True, key=lambda t: t[0])
    matched_g, matched_p = set(), set()
    matches = []
    for iou, gi, pj, g, p in triples:
        if iou < tau:
            break
        if gi in matched_g or pj in matched_p:
            continue
        matched_g.add(gi)
        matched_p.add(pj)
        matches.append((g, p, float(iou)))

    tp = len(matches)
    fp = n_p - tp
    fn = n_g - tp
    max_iou = iou_mat.max(axis=1) if n_p else np.zeros(n_g)
    cov = float(max_iou.mean()) if n_g else 0.0
    out = metrics_from_counts(tp, fp, fn, cov)
    out.update(
        {
            "N_pred": n_p,
            "N_GT": n_g,
            "max_iou": max_iou,
            "matches": matches,
            "gt_labels": gt_labels,
            "pred_labels": pred_labels,
            "iou_mat": iou_mat,
            "g_index": g_index,
            "p_index": p_index,
            "iou_pairs": pair_counts,
            "gt_sizes": gt_sizes,
            "pred_sizes": pred_sizes,
            "pred_all": pred_all,
        }
    )
    return out


def error_taxonomy(result, gt_id, pred_all, tau=0.5):
    gt_labels = result["gt_labels"]
    pred_labels = result["pred_labels"]
    iou_mat = result["iou_mat"]
    g_index = result["g_index"]
    p_index = result["p_index"]
    matches = {g: (p, iou) for g, p, iou in result["matches"]}

    over = 0
    under = 0
    missed = 0
    leakage = 0

    for g in gt_labels:
        gi = g_index[int(g)]
        strong = int(np.sum(iou_mat[gi] >= 0.25))
        if strong >= 2:
            over += 1
        if int(g) not in matches:
            missed += 1

    for p in pred_labels:
        pj = p_index[int(p)]
        strong = int(np.sum(iou_mat[:, pj] >= 0.25))
        if strong >= 2:
            under += 1

    for g, (p, iou) in matches.items():
        if 0.5 <= iou < 0.7:
            leakage += 1
            continue
        mask = gt_id == g
        leak_frac = float(np.mean((pred_all[mask] != p) & (pred_all[mask] > 0)))
        if leak_frac > 0.1:
            leakage += 1

    return {
        "over_seg": over,
        "under_seg": under,
        "missed": missed,
        "leakage_flagged_tp": leakage,
    }


def run_for_model(data, model: str, pred_key: str, mask=None, tau=0.5):
    gt = data["gt_id"] if mask is None else data["gt_id"][mask]
    pred = data[pred_key] if mask is None else data[pred_key][mask]
    res = build_iou_and_match(gt, pred, tau=tau, n_min=N_MIN, pred_positive_min=1)
    tax = error_taxonomy(res, gt, res["pred_all"], tau=tau)
    row = {
        "Method": model,
        "Prec": res["Prec"],
        "Rec": res["Rec"],
        "F1": res["F1"],
        "Cov": res["Cov"],
        "N_pred": res["N_pred"],
        "N_GT": res["N_GT"],
        "TP": res["TP"],
        "FP": res["FP"],
        "FN": res["FN"],
        **tax,
    }
    return row, res


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_data()
    models = [("ForestMamba", "mamba_id"), ("SegmentAnyTree", "sat_id")]
    tile_labels = [f"{t:03d}" for t in TILES]

    t1_rows, full_res = [], {}
    for name, key in models:
        row, res = run_for_model(data, name, key, tau=0.5)
        t1_rows.append(row)
        full_res[name] = res
        print(f"T1 {name}: F1={row['F1']:.1f} Cov={row['Cov']:.1f} TP={row['TP']} FP={row['FP']} FN={row['FN']}")
    t1 = pd.DataFrame(t1_rows)
    t1.to_csv(TABLES / "T1_overall.csv", index=False)

    t2_rows = []
    for tid in TILES:
        mask = data["tile_id"] == tid
        for name, key in models:
            row, _ = run_for_model(data, name, key, mask=mask, tau=0.5)
            row["Tile"] = f"{tid:03d}"
            t2_rows.append(row)
            print(f"T2 tile {tid} {name}: F1={row['F1']:.1f}")
    t2 = pd.DataFrame(t2_rows)
    t2.to_csv(TABLES / "T2_per_tile.csv", index=False)

    t3 = t1[
        [
            "Method",
            "TP",
            "FP",
            "FN",
            "over_seg",
            "under_seg",
            "missed",
            "leakage_flagged_tp",
        ]
    ].copy()
    t3.to_csv(TABLES / "T3_error_taxonomy.csv", index=False)

    cat = pd.read_csv(ALIGNED_DIR / "gt_instances.csv")
    qs = cat["n_points"].quantile([1 / 3, 2 / 3]).to_list()

    def stratum(n):
        if n <= qs[0]:
            return "small"
        if n <= qs[1]:
            return "medium"
        return "large"

    cat["stratum"] = cat["n_points"].map(stratum)
    t5_rows = []
    for stratum_name, sub in cat.groupby("stratum"):
        gids = set(sub["gt_id"].astype(int))
        is_gt = np.isin(data["gt_id"], list(gids))
        is_bg = data["gt_id"] < 0
        mask = is_gt | is_bg
        gt_s = data["gt_id"][mask].copy()
        gt_s[~np.isin(gt_s, list(gids)) & (gt_s > 0)] = -1
        for name, key in models:
            pred_s = data[key][mask]
            res = build_iou_and_match(gt_s, pred_s, tau=0.5)
            tax = error_taxonomy(res, gt_s, res["pred_all"])
            t5_rows.append(
                {
                    "Stratum": stratum_name,
                    "Rule": f"n_points terciles q33={qs[0]:.0f} q66={qs[1]:.0f}",
                    "Method": name,
                    "Prec": res["Prec"],
                    "Rec": res["Rec"],
                    "F1": res["F1"],
                    "Cov": res["Cov"],
                    "N_GT": res["N_GT"],
                    **tax,
                }
            )
            print(f"T5 {stratum_name} {name}: F1={res['F1']:.1f}")
    pd.DataFrame(t5_rows).to_csv(TABLES / "T5_size_strata.csv", index=False)

    t7_rows = []
    for tau in TAUS:
        for name, key in models:
            row, res = run_for_model(data, name, key, tau=tau)
            t7_rows.append(
                {
                    "Method": name,
                    "tau": tau,
                    "Prec": row["Prec"],
                    "Rec": row["Rec"],
                    "F1": row["F1"],
                    "Cov": row["Cov"],
                    "TP": row["TP"],
                    "FP": row["FP"],
                    "FN": row["FN"],
                }
            )
            print(f"T7 tau={tau} {name}: F1={row['F1']:.1f}")
    t7 = pd.DataFrame(t7_rows)
    t7.to_csv(TABLES / "T7_tau_sweep.csv", index=False)

    dens_rows = []
    rng = np.random.default_rng(42)
    n = len(data["gt_id"])
    for keep in (0.1, 0.25, 0.5, 1.0):
        if keep < 1.0:
            sel = rng.random(n) < keep
        else:
            sel = np.ones(n, dtype=bool)
        for name, key in models:
            row, _ = run_for_model(data, name, key, mask=sel, tau=0.5)
            dens_rows.append(
                {
                    "keep_rate": keep,
                    "Method": name,
                    "Prec": row["Prec"],
                    "Rec": row["Rec"],
                    "F1": row["F1"],
                    "Cov": row["Cov"],
                }
            )
            print(f"Density keep={keep} {name}: F1={row['F1']:.1f}")
    dens = pd.DataFrame(dens_rows)
    dens.to_csv(TABLES / "density_keep_rates.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax, metric in zip(axes.ravel(), ["Cov", "Prec", "Rec", "F1"]):
        for name, color in [("ForestMamba", "#1b9e77"), ("SegmentAnyTree", "#d95f02")]:
            sub = t7[t7["Method"] == name]
            ax.plot(sub["tau"], sub[metric], "o-", label=name, color=color)
        ax.set_ylabel(f"{metric} (%)")
        ax.set_xlabel(r"IoU threshold $\tau$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{SITE_TITLE}: metrics vs IoU threshold")
    fig.tight_layout()
    fig.savefig(FIGURES / "tau_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["F1", "Cov"]):
        x = np.arange(len(TILES))
        w = 0.35
        for i, (name, color) in enumerate(
            [("ForestMamba", "#1b9e77"), ("SegmentAnyTree", "#d95f02")]
        ):
            vals = [
                t2[(t2["Tile"] == lab) & (t2["Method"] == name)][metric].values[0]
                for lab in tile_labels
            ]
            ax.bar(x + i * w, vals, w, label=name, color=color)
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(tile_labels)
        ax.set_ylabel(f"{metric} (%)")
        ax.set_xlabel("Tile")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"{SITE_TITLE}: per-tile metrics at τ=0.5")
    fig.tight_layout()
    fig.savefig(FIGURES / "per_tile_bars.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, color in [("ForestMamba", "#1b9e77"), ("SegmentAnyTree", "#d95f02")]:
        ax.hist(
            full_res[name]["max_iou"],
            bins=20,
            range=(0, 1),
            alpha=0.5,
            label=name,
            color=color,
        )
    ax.axvline(0.5, color="k", ls="--", lw=1, label=r"$\tau=0.5$")
    ax.set_xlabel("Max IoU per GT tree")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title(f"{SITE_TITLE}: per-GT best IoU")
    fig.tight_layout()
    fig.savefig(FIGURES / "iou_histogram.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    cat = pd.read_csv(ALIGNED_DIR / "gt_instances.csv")
    for ax, name in zip(axes, ["ForestMamba", "SegmentAnyTree"]):
        res = full_res[name]
        matched_g = {g for g, p, iou in res["matches"]}
        matched_p = {p for g, p, iou in res["matches"]}
        for _, r in cat.iterrows():
            gid = int(r["gt_id"])
            if gid in matched_g:
                c, m = "#2ca02c", "o"
            else:
                c, m = "#1f77b4", "x"
            ax.scatter(r["centroid_x"], r["centroid_y"], c=c, marker=m, s=40, zorder=3)
        pred = data["mamba_id" if name == "ForestMamba" else "sat_id"]
        for p in res["pred_labels"]:
            if int(p) in matched_p:
                continue
            msk = pred == int(p)
            if not np.any(msk):
                continue
            ax.scatter(
                data["x"][msk].mean(),
                data["y"][msk].mean(),
                c="#ff7f0e",
                marker="^",
                s=28,
                alpha=0.8,
                zorder=2,
            )
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", label="TP GT", markersize=8),
        Line2D([0], [0], marker="x", color="#1f77b4", label="FN GT", markersize=8),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#ff7f0e", label="FP pred", markersize=8),
    ]
    axes[1].legend(handles=legend, loc="best", fontsize=8)
    fig.suptitle(rf"{SITE_TITLE}: spatial TP / FN / FP ($\tau=0.5$)")
    fig.tight_layout()
    fig.savefig(FIGURES / "spatial_tp_fp_fn.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    first_tile = TILES[0] if TILES else 1
    m = data["tile_id"] == first_tile
    idx = np.where(m)[0]
    if len(idx) > 80_000:
        idx = np.random.default_rng(0).choice(idx, 80_000, replace=False)
    titles = ["GT", "ForestMamba", "SegmentAnyTree"]
    fields = [data["gt_id"], data["mamba_id"], data["sat_id"]]
    for ax, title, field in zip(axes, titles, fields):
        labs = field[idx].astype(np.int32)
        uniq = np.unique(labs)
        rng = np.random.default_rng(1)
        cmap = {int(u): rng.random(3) if u > 0 else np.array([0.7, 0.7, 0.7]) for u in uniq}
        cols = np.array([cmap[int(u)] for u in labs])
        ax.scatter(data["x"][idx], data["y"][idx], c=cols, s=0.2, linewidths=0)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    fig.suptitle(f"{SITE_TITLE}: tile {first_tile:03d} instance colours (subsampled)")
    fig.tight_layout()
    fig.savefig(FIGURES / f"qualitative_instances_tile{first_tile:03d}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(t1))
    ax.bar(x - 0.2, t1["over_seg"], 0.2, label="over-seg")
    ax.bar(x, t1["under_seg"], 0.2, label="under-seg")
    ax.bar(x + 0.2, t1["missed"], 0.2, label="missed")
    ax.set_xticks(x)
    ax.set_xticklabels(t1["Method"])
    ax.legend()
    ax.set_ylabel("Count")
    ax.set_title(f"{SITE_TITLE}: error taxonomy")
    fig.tight_layout()
    fig.savefig(FIGURES / "error_rates.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for _, r in t1.iterrows():
        ax.scatter(r["Cov"], r["F1"], s=80, label=r["Method"])
        ax.annotate(r["Method"], (r["Cov"], r["F1"]), xytext=(5, 5), textcoords="offset points")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3)
    ax.set_xlabel("Cov (%)")
    ax.set_ylabel("F1 (%)")
    ax.set_title(f"{SITE_TITLE}: Cov vs F1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "cov_f1_gap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, color in [("ForestMamba", "#1b9e77"), ("SegmentAnyTree", "#d95f02")]:
        sub = dens[dens["Method"] == name]
        ax.plot(sub["keep_rate"] * 100, sub["F1"], "o-", label=f"{name} F1", color=color)
        ax.plot(sub["keep_rate"] * 100, sub["Cov"], "s--", label=f"{name} Cov", color=color, alpha=0.7)
    ax.set_xlabel("Keep rate (%)")
    ax.set_ylabel("Metric (%)")
    ax.set_title(f"{SITE_TITLE}: density robustness")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "density_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = ["Prec", "Rec", "F1", "Cov"]
    x = np.arange(len(metrics))
    w = 0.35
    for i, (name, color) in enumerate(
        [("ForestMamba", "#1b9e77"), ("SegmentAnyTree", "#d95f02")]
    ):
        vals = [t1[t1["Method"] == name][m].values[0] for m in metrics]
        ax.bar(x + i * w, vals, w, label=name, color=color)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("%")
    ax.legend()
    ax.set_title(rf"{SITE_TITLE}: overall metrics ($\tau=0.5$)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "T1_bars.png", dpi=150)
    plt.close(fig)

    summary = {
        "T1": t1.to_dict(orient="records"),
        "cov_f1_gap": {
            r["Method"]: float(r["Cov"] - r["F1"]) for _, r in t1.iterrows()
        },
    }
    (TABLES / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done metrics + figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to site YAML config")
    args = parser.parse_args()
    configure(load_site_config(args.config))
    main()
