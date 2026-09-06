"""Per-tree pairwise F1 + structure covariates for a site benchmark."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.paths import load_site_config

import run_metrics as rm  # noqa: E402
from run_metrics import build_iou_and_match, configure, load_data  # noqa: E402

SEED = 0
N_BOOT = 1000
HULL_MAX_PTS = 50_000
MODELS = [("ForestMamba", "mamba_id"), ("SegmentAnyTree", "sat_id")]


def pairwise_f1(inter: int, g_size: int, p_size: int) -> tuple[float, float, float]:
    if g_size <= 0 or p_size <= 0 or inter <= 0:
        return 0.0, 0.0, 0.0
    prec = inter / p_size
    rec = inter / g_size
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return float(prec), float(rec), float(f1)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_boot, dtype=np.float64)
    n = len(values)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def tercile_labels(series: pd.Series) -> tuple[pd.Series, list[float]]:
    qs = series.quantile([1 / 3, 2 / 3]).tolist()

    def lab(v):
        if not np.isfinite(v):
            return "na"
        if v <= qs[0]:
            return "low"
        if v <= qs[1]:
            return "mid"
        return "high"

    return series.map(lab), qs


def size_tercile(n_points: pd.Series) -> tuple[pd.Series, list[float]]:
    qs = n_points.quantile([1 / 3, 2 / 3]).tolist()

    def lab(n):
        if n <= qs[0]:
            return "small"
        if n <= qs[1]:
            return "medium"
        return "large"

    return n_points.map(lab), qs


def rank_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residual of rank(y) after linear fit on rank(x)."""
    yr = pd.Series(y).rank().to_numpy(dtype=np.float64)
    xr = pd.Series(x).rank().to_numpy(dtype=np.float64)
    mask = np.isfinite(yr) & np.isfinite(xr)
    out = np.full_like(yr, np.nan)
    if mask.sum() < 3:
        return out
    coef = np.polyfit(xr[mask], yr[mask], 1)
    out[mask] = yr[mask] - (coef[0] * xr[mask] + coef[1])
    return out


def compute_tree_covariates(data: dict, cat: pd.DataFrame) -> pd.DataFrame:
    """Height, NN distance, hull volume per gt_id."""
    rng = np.random.default_rng(SEED)
    gt = data["gt_id"]
    x, y, z = data["x"], data["y"], data["z"]

    rows = []
    for _, r in cat.iterrows():
        gid = int(r["gt_id"])
        mask = gt == gid
        if not np.any(mask):
            continue
        zz = z[mask]
        xx = x[mask]
        yy = y[mask]
        z05, z99 = np.percentile(zz, [5, 99])
        height = float(z99 - z05)
        cx, cy = float(xx.mean()), float(yy.mean())

        n = len(xx)
        if n >= 4:
            if n > HULL_MAX_PTS:
                idx = rng.choice(n, HULL_MAX_PTS, replace=False)
                pts = np.column_stack([xx[idx], yy[idx], zz[idx]])
            else:
                pts = np.column_stack([xx, yy, zz])
            try:
                hull = ConvexHull(pts)
                vol = float(hull.volume)
            except (QhullError, ValueError):
                vol = float("nan")
        else:
            vol = float("nan")

        rows.append(
            {
                "gt_id": gid,
                "tile": int(r["tile"]),
                "n_points": int(r["n_points"]),
                "centroid_x": cx,
                "centroid_y": cy,
                "height_m": height,
                "hull_volume_m3": vol,
                "z05": float(z05),
                "z99": float(z99),
            }
        )

    cov = pd.DataFrame(rows)
    ids = cov["gt_id"].to_numpy()
    xy = np.column_stack([cov["centroid_x"], cov["centroid_y"]])
    tree = cKDTree(xy)
    d, ix = tree.query(xy, k=2)
    cov["nn_dist_m"] = d[:, 1]
    cov["nn_gt_id"] = ids[ix[:, 1]]
    size_lab, size_qs = size_tercile(cov["n_points"])
    cov["size_tercile"] = size_lab
    cov.attrs["size_qs"] = size_qs
    return cov


def per_tree_scores_for_model(data: dict, cov: pd.DataFrame, method: str, pred_key: str):
    res = build_iou_and_match(data["gt_id"], data[pred_key], tau=0.5, n_min=rm.N_MIN)
    match_map = {int(g): (int(p), float(iou)) for g, p, iou in res["matches"]}
    gt_labels = [int(g) for g in res["gt_labels"]]
    max_iou = {gid: float(res["max_iou"][res["g_index"][gid]]) for gid in gt_labels}
    gt_sizes = res["gt_sizes"]
    pred_sizes = res.get("pred_sizes", {})
    iou_pairs = res.get("iou_pairs", {})

    best_by_iou: dict[int, tuple[int, float, int]] = {}
    for (g, p), inter in iou_pairs.items():
        gs = gt_sizes.get(g, 0)
        ps = pred_sizes.get(p, 0)
        union = gs + ps - inter
        iou = inter / union if union else 0.0
        cur = best_by_iou.get(g)
        if cur is None or iou > cur[1]:
            best_by_iou[g] = (p, iou, inter)

    rows = []
    cov_idx = cov.set_index("gt_id")
    for gid in gt_labels:
        gs = int(gt_sizes.get(gid, 0))
        success = 1 if gid in match_map else 0
        if success:
            pid, miou = match_map[gid]
            inter = int(iou_pairs.get((gid, pid), 0))
            ps = int(pred_sizes.get(pid, 0))
            prec_h, rec_h, f1_h = pairwise_f1(inter, gs, ps)
            matched_pred = pid
            matched_iou = miou
            inter_hard = inter
            pred_size_hard = ps
        else:
            f1_h = prec_h = rec_h = 0.0
            matched_pred = -1
            matched_iou = 0.0
            inter_hard = 0
            pred_size_hard = 0

        if gid in best_by_iou:
            pid_s, iou_s, inter_s = best_by_iou[gid]
            ps_s = int(pred_sizes.get(pid_s, 0))
            prec_s, rec_s, f1_s = pairwise_f1(inter_s, gs, ps_s)
            soft_pred = pid_s
            inter_soft = inter_s
            pred_size_soft = ps_s
        else:
            f1_s = prec_s = rec_s = 0.0
            soft_pred = -1
            inter_soft = 0
            pred_size_soft = 0

        c = cov_idx.loc[gid]
        rows.append(
            {
                "Method": method,
                "gt_id": gid,
                "tile": int(c["tile"]),
                "n_points": int(c["n_points"]),
                "size_tercile": c["size_tercile"],
                "centroid_x": float(c["centroid_x"]),
                "centroid_y": float(c["centroid_y"]),
                "height_m": float(c["height_m"]),
                "nn_dist_m": float(c["nn_dist_m"]),
                "hull_volume_m3": float(c["hull_volume_m3"])
                if np.isfinite(c["hull_volume_m3"])
                else np.nan,
                "max_iou": max_iou.get(gid, 0.0),
                "success_05": success,
                "matched_pred": matched_pred,
                "matched_iou": matched_iou,
                "inter_hard": inter_hard,
                "pred_size_hard": pred_size_hard,
                "prec_hard": prec_h,
                "rec_hard": rec_h,
                "f1_hard": f1_h,
                "soft_pred": soft_pred,
                "inter_soft": inter_soft,
                "pred_size_soft": pred_size_soft,
                "prec_soft": prec_s,
                "rec_soft": rec_s,
                "f1_soft": f1_s,
            }
        )
    return pd.DataFrame(rows), res


def build_tables(per: pd.DataFrame, plot_t1: pd.DataFrame, size_qs: list[float], rng: np.random.Generator):
    t8_rows = []
    for method in per["Method"].unique():
        sub = per[per["Method"] == method]
        f1_mean, f1_lo, f1_hi = bootstrap_mean_ci(sub["f1_hard"].to_numpy(), rng)
        f1s_mean, f1s_lo, f1s_hi = bootstrap_mean_ci(sub["f1_soft"].to_numpy(), rng)
        mi_mean, mi_lo, mi_hi = bootstrap_mean_ci(sub["max_iou"].to_numpy(), rng)
        plot = plot_t1[plot_t1["Method"] == method].iloc[0]
        t8_rows.append(
            {
                "Method": method,
                "N_GT": int(len(sub)),
                "plot_Prec": float(plot["Prec"]),
                "plot_Rec": float(plot["Rec"]),
                "plot_F1": float(plot["F1"]),
                "plot_Cov": float(plot["Cov"]),
                "macro_f1_hard": 100 * f1_mean,
                "macro_f1_hard_ci_lo": 100 * f1_lo,
                "macro_f1_hard_ci_hi": 100 * f1_hi,
                "macro_f1_soft": 100 * f1s_mean,
                "macro_f1_soft_ci_lo": 100 * f1s_lo,
                "macro_f1_soft_ci_hi": 100 * f1s_hi,
                "mean_max_iou": 100 * mi_mean,
                "mean_max_iou_ci_lo": 100 * mi_lo,
                "mean_max_iou_ci_hi": 100 * mi_hi,
                "mean_success_05": 100 * float(sub["success_05"].mean()),
            }
        )
    t8 = pd.DataFrame(t8_rows)

    t9_rows = []
    covariates = [
        ("height_m", "height"),
        ("nn_dist_m", "nn_dist"),
        ("hull_volume_m3", "hull_volume"),
        ("n_points", "n_points"),
    ]
    base = per[per["Method"] == per["Method"].iloc[0]].copy()
    for col, name in covariates:
        labs, qs = tercile_labels(base[col])
        cut_map = dict(zip(base["gt_id"], labs))
        for method, sub in per.groupby("Method"):
            sub = sub.copy()
            sub["stratum"] = sub["gt_id"].map(cut_map)
            for stratum, g in sub.groupby("stratum"):
                if stratum == "na":
                    continue
                m, lo, hi = bootstrap_mean_ci(g["f1_hard"].to_numpy(), rng)
                t9_rows.append(
                    {
                        "Method": method,
                        "Covariate": name,
                        "Stratum": stratum,
                        "q33": qs[0],
                        "q66": qs[1],
                        "N": int(len(g)),
                        "mean_f1_hard": 100 * m,
                        "mean_f1_hard_ci_lo": 100 * lo,
                        "mean_f1_hard_ci_hi": 100 * hi,
                        "mean_max_iou": 100 * float(g["max_iou"].mean()),
                        "detection_rate": 100 * float(g["success_05"].mean()),
                    }
                )
    t9 = pd.DataFrame(t9_rows)

    t10_rows = []
    for method, sub in per.groupby("Method"):
        for score in ("f1_hard", "f1_soft"):
            for col, label in [
                ("height_m", "height_m"),
                ("nn_dist_m", "nn_dist_m"),
                ("hull_volume_m3", "log1p_hull_volume"),
                ("n_points", "log1p_n_points"),
            ]:
                y = sub[score].to_numpy(dtype=np.float64)
                if col.startswith("hull") or col == "n_points":
                    x = np.log1p(sub[col].to_numpy(dtype=np.float64))
                else:
                    x = sub[col].to_numpy(dtype=np.float64)
                mask = np.isfinite(y) & np.isfinite(x)
                n = int(mask.sum())
                if n < 5:
                    rho, p = float("nan"), float("nan")
                else:
                    rho, p = spearmanr(x[mask], y[mask])
                t10_rows.append(
                    {
                        "Method": method,
                        "Score": score,
                        "Covariate": label,
                        "Spearman_rho": float(rho),
                        "p_value": float(p),
                        "N": n,
                        "partial": False,
                    }
                )
        npts = np.log1p(sub["n_points"].to_numpy(dtype=np.float64))
        y = sub["f1_hard"].to_numpy(dtype=np.float64)
        y_res = rank_residual(y, npts)
        for col, label in [("height_m", "height_m"), ("nn_dist_m", "nn_dist_m")]:
            x = sub[col].to_numpy(dtype=np.float64)
            x_res = rank_residual(x, npts)
            mask = np.isfinite(y_res) & np.isfinite(x_res)
            n = int(mask.sum())
            if n < 5:
                rho, p = float("nan"), float("nan")
            else:
                rho, p = spearmanr(x_res[mask], y_res[mask])
            t10_rows.append(
                {
                    "Method": method,
                    "Score": "f1_hard",
                    "Covariate": f"{label}_partial_log_n_points",
                    "Spearman_rho": float(rho),
                    "p_value": float(p),
                    "N": n,
                    "partial": True,
                }
            )
    t10 = pd.DataFrame(t10_rows)

    base = per[per["Method"] == per["Method"].iloc[0]]
    t11_rows = []
    for stratum, g in base.groupby("size_tercile"):
        h = g["height_m"]
        t11_rows.append(
            {
                "size_tercile": stratum,
                "N": int(len(g)),
                "height_min": float(h.min()),
                "height_p25": float(h.quantile(0.25)),
                "height_p50": float(h.median()),
                "height_p75": float(h.quantile(0.75)),
                "height_max": float(h.max()),
                "n_points_q33": size_qs[0],
                "n_points_q66": size_qs[1],
            }
        )
    t11 = pd.DataFrame(t11_rows)
    return t8, t9, t10, t11


def make_figures(per: pd.DataFrame, t8: pd.DataFrame, t9: pd.DataFrame, t11: pd.DataFrame):
    rm.FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"ForestMamba": "#1b9e77", "SegmentAnyTree": "#d95f02"}

    base = per[per["Method"] == per["Method"].iloc[0]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for stratum, color in [("small", "#66c2a5"), ("medium", "#fc8d62"), ("large", "#8da0cb")]:
        h = base.loc[base["size_tercile"] == stratum, "height_m"]
        ax.hist(h, bins=15, alpha=0.55, label=f"{stratum} (n={len(h)})", color=color)
    ax.set_xlabel("Tree height (m) — z99 − z05")
    ax.set_ylabel("Count")
    ax.set_title("Height distribution by point-count size tercile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(rm.FIGURES / "height_hist_by_size.png", dpi=150)
    plt.close(fig)

    def scatter_f1(xcol, xlabel, fname, logx=False):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, method in zip(axes, ["ForestMamba", "SegmentAnyTree"]):
            sub = per[per["Method"] == method]
            x = sub[xcol].to_numpy()
            y = 100 * sub["f1_hard"].to_numpy()
            ax.scatter(x, y, s=28, alpha=0.75, c=colors[method], edgecolors="none")
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Per-tree F1 hard (%)")
            ax.set_title(method)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-5, 105)
        fig.suptitle("Per-tree pairwise F1 (hard) vs structure")
        fig.tight_layout()
        fig.savefig(rm.FIGURES / fname, dpi=150)
        plt.close(fig)

    scatter_f1("height_m", "Height (m)", "f1_vs_height.png")
    scatter_f1("nn_dist_m", "Nearest GT neighbour distance (m)", "f1_vs_nn_dist.png")
    scatter_f1("hull_volume_m3", "Convex-hull volume (m³)", "f1_vs_hull_volume.png", logx=True)

    cov_order = ["height", "nn_dist", "hull_volume", "n_points"]
    stratum_order = ["low", "mid", "high"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, cov in zip(axes.ravel(), cov_order):
        x = np.arange(len(stratum_order))
        w = 0.35
        for i, method in enumerate(["ForestMamba", "SegmentAnyTree"]):
            vals = []
            for s in stratum_order:
                row = t9[(t9["Method"] == method) & (t9["Covariate"] == cov) & (t9["Stratum"] == s)]
                vals.append(float(row["mean_f1_hard"].iloc[0]) if len(row) else np.nan)
            ax.bar(x + i * w, vals, w, label=method, color=colors[method])
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(stratum_order)
        ax.set_title(cov)
        ax.set_ylabel("Mean per-tree F1 hard (%)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Mean per-tree F1 by covariate tercile")
    fig.tight_layout()
    fig.savefig(rm.FIGURES / "macro_strata_bars.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = ["ForestMamba", "SegmentAnyTree"]
    x = np.arange(len(methods))
    w = 0.35
    plot_f1 = [float(t8[t8["Method"] == m]["plot_F1"].iloc[0]) for m in methods]
    macro_f1 = [float(t8[t8["Method"] == m]["macro_f1_hard"].iloc[0]) for m in methods]
    ax.bar(x - w / 2, plot_f1, w, label="Plot F1 (T1)", color="#6a3d9a")
    ax.bar(x + w / 2, macro_f1, w, label="Macro F1 (mean per-tree)", color="#33a02c")
    for i, m in enumerate(methods):
        row = t8[t8["Method"] == m].iloc[0]
        ax.errorbar(
            i + w / 2,
            row["macro_f1_hard"],
            yerr=[
                [row["macro_f1_hard"] - row["macro_f1_hard_ci_lo"]],
                [row["macro_f1_hard_ci_hi"] - row["macro_f1_hard"]],
            ],
            fmt="none",
            ecolor="k",
            capsize=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("F1 (%)")
    ax.set_title(f"{rm.SITE_TITLE}: plot-level F1 vs macro hard F1")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(rm.FIGURES / "plot_vs_macro_f1.png", dpi=150)
    plt.close(fig)


def main() -> None:
    rm.TABLES.mkdir(parents=True, exist_ok=True)
    rm.FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Loading data ...", flush=True)
    data = load_data()
    cat = pd.read_csv(rm.ALIGNED_DIR / "gt_instances.csv")
    plot_t1 = pd.read_csv(rm.TABLES / "T1_overall.csv")

    print("Computing tree covariates (height / NN / hull) ...", flush=True)
    cov = compute_tree_covariates(data, cat)
    size_qs = cov.attrs.get("size_qs", cov["n_points"].quantile([1 / 3, 2 / 3]).tolist())

    frames = []
    for method, key in MODELS:
        print(f"Per-tree F1 for {method} ...", flush=True)
        df, _ = per_tree_scores_for_model(data, cov, method, key)
        frames.append(df)
        print(
            f"  macro f1_hard={100*df['f1_hard'].mean():.1f}%  "
            f"macro f1_soft={100*df['f1_soft'].mean():.1f}%  "
            f"detection={100*df['success_05'].mean():.1f}%",
            flush=True,
        )
    per = pd.concat(frames, ignore_index=True)
    per_path = rm.TABLES / "per_tree_scores.csv"
    per.to_csv(per_path, index=False)
    print(f"Wrote {per_path}", flush=True)

    print("Building T8–T11 ...", flush=True)
    t8, t9, t10, t11 = build_tables(per, plot_t1, size_qs, rng)
    t8.to_csv(rm.TABLES / "T8_macro_vs_plot.csv", index=False)
    t9.to_csv(rm.TABLES / "T9_score_by_covariate.csv", index=False)
    t10.to_csv(rm.TABLES / "T10_correlations.csv", index=False)
    t11.to_csv(rm.TABLES / "T11_height_by_size.csv", index=False)
    print("Wrote T8–T11", flush=True)

    print("Figures ...", flush=True)
    make_figures(per, t8, t9, t11)
    print("Done per-tree analysis.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to site YAML config")
    args = parser.parse_args()
    configure(load_site_config(args.config))
    main()
