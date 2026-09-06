"""Figures for DetailView FOR-species20K-style benchmark."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .constants import METHODS, METHOD_LABEL


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_leaderboard(t20: pd.DataFrame, out: Path) -> None:
    sub = t20[t20["site"] != "both"].copy()
    if sub.empty:
        return
    sites = sorted(sub["site"].unique())
    methods = [m for m in METHODS if m in set(sub["method"])]
    x = np.arange(len(methods))
    width = 0.35 if len(sites) == 2 else 0.6
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, metric, title in zip(axes, ["accuracy", "macro_f1"], ["Overall accuracy", "Macro F1"]):
        for i, site in enumerate(sites):
            vals = []
            for m in methods:
                row = sub[(sub["site"] == site) & (sub["method"] == m)]
                vals.append(float(row[metric].iloc[0]) if len(row) else np.nan)
            offset = (i - (len(sites) - 1) / 2) * width
            ax.bar(x + offset, vals, width=width, label=site)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABEL[m].replace("DetailView_", "") for m in methods], rotation=20)
        ax.set_ylabel("%")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("DetailView leaderboard (FOR-species20K-style)", fontsize=12)
    _save(fig, out)


def plot_confusion(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    out: Path,
    *,
    normalize: bool = True,
) -> None:
    mat = cm.astype(float)
    if normalize and mat.sum(axis=1).any():
        row_sum = mat.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        mat = 100.0 * mat / row_sum

    pretty = [lab.replace("_", " ") for lab in labels]
    n = len(labels)
    fig_w = max(6.5, 0.62 * n + 2.8)
    fig_h = max(5.4, 0.55 * n + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fffdf8")

    cmap = LinearSegmentedColormap.from_list(
        "paper_green",
        ["#ffffff", "#e7efe4", "#9db89a", "#4a7c59", "#1f3d28"],
    )
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100 if normalize else None, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(pretty, rotation=40, ha="right", fontsize=8, color="#5c6658")
    ax.set_yticklabels(pretty, fontsize=8, color="#5c6658")
    ax.set_xlabel("Predicted species", fontsize=10, color="#1a1f18", labelpad=8)
    ax.set_ylabel("True species", fontsize=10, color="#1a1f18", labelpad=8)
    ax.set_title(title, fontsize=12, fontweight=650, color="#1a1f18", pad=12)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_color("#d7d0c3")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if normalize:
                if val < 0.5:
                    continue
                txt = f"{val:.0f}"
            else:
                if cm[i, j] == 0:
                    continue
                txt = str(int(cm[i, j]))
            dark = val >= 55 if normalize else False
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=8,
                fontweight=600,
                color="#ffffff" if dark else "#1a1f18",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("% of true class" if normalize else "Count", color="#5c6658")
    cbar.ax.tick_params(colors="#5c6658", labelsize=8)
    cbar.outline.set_edgecolor("#d7d0c3")
    _save(fig, out)


def plot_f1_vs_support(t21: pd.DataFrame, out: Path) -> None:
    sub = t21[(t21["site"] != "both") & (t21["support"] > 0)].copy()
    if sub.empty:
        return
    methods = [m for m in METHODS if m in set(sub["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Local support
    ax = axes[0]
    for m in methods:
        d = sub[sub["method"] == m]
        ax.scatter(d["support"], d["f1"] / 100.0, s=40 + 8 * d["support"], alpha=0.75, label=m)
        for _, r in d.iterrows():
            ax.annotate(
                str(r["species"]).replace("_", " ")[:18],
                (r["support"], r["f1"] / 100.0),
                fontsize=6,
                alpha=0.8,
            )
    ax.set_xlabel("Local support (N stems)")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1 vs local class support")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    # FOR-species training N
    ax = axes[1]
    d0 = sub[sub["method"] == "GT"] if "GT" in methods else sub[sub["method"] == methods[0]]
    d0 = d0[d0["forspecies_train_n"] > 0]
    if len(d0):
        ax.scatter(d0["forspecies_train_n"], d0["f1"] / 100.0, s=50, c="#1b6b4a")
        for _, r in d0.iterrows():
            ax.annotate(
                str(r["species"]).replace("_", " ")[:18],
                (r["forspecies_train_n"], r["f1"] / 100.0),
                fontsize=6,
            )
    ax.set_xlabel("FOR-species20K training N (approx)")
    ax.set_ylabel("F1 (manual GT track)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Context: F1 vs FOR-species20K train size")
    ax.grid(alpha=0.3)
    _save(fig, out)


def plot_oa_by_height(t23: pd.DataFrame, height_hist: pd.DataFrame, out: Path) -> None:
    sites = sorted(s for s in t23["site"].unique() if s != "both")
    if not sites:
        return
    fig, axes = plt.subplots(len(sites), 2, figsize=(11, 3.6 * len(sites)), squeeze=False)
    for row, site in enumerate(sites):
        ax = axes[row][0]
        d = t23[(t23["site"] == site) & (t23["method"].isin(["GT", "FM"]))].copy()
        bins = sorted(d["height_bin"].unique(), key=lambda s: float(s.split(",")[0].strip("([]")))
        x = np.arange(len(bins))
        for i, m in enumerate(["GT", "FM"]):
            vals = []
            for b in bins:
                r = d[(d["height_bin"] == b) & (d["method"] == m)]
                vals.append(float(r["accuracy"].iloc[0]) if len(r) else np.nan)
            ax.bar(x + (i - 0.5) * 0.4, vals, width=0.38, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("OA (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"{site}: OA by height bin")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        ax2 = axes[row][1]
        hh = height_hist[height_hist["site"] == site].copy()
        if len(hh):
            hh = hh.sort_values(
                "height_bin",
                key=lambda s: s.map(lambda b: float(b.split(",")[0].strip("([]"))),
            )
            ax2.bar(np.arange(len(hh)), hh["n"], color="#c97b84")
            ax2.set_xticks(np.arange(len(hh)))
            ax2.set_xticklabels(hh["height_bin"], rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"{site}: height distribution")
    _save(fig, out)


def plot_lifeform(t24: pd.DataFrame, out: Path) -> None:
    sub = t24[t24["site"] != "both"].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sites = sorted(sub["site"].unique())
    forms = ["conifer", "broadleaf"]
    methods = [m for m in METHODS if m in set(sub["method"])]
    xpos = np.arange(len(methods))
    width = 0.18
    k = 0
    for site in sites:
        for form in forms:
            vals = []
            for m in methods:
                r = sub[(sub["site"] == site) & (sub["method"] == m) & (sub["lifeform"] == form)]
                vals.append(float(r["accuracy"].iloc[0]) if len(r) else np.nan)
            ax.bar(xpos + (k - 1.5) * width, vals, width=width, label=f"{site[:3]}-{form[:3]}")
            k += 1
    ax.set_xticks(xpos)
    ax.set_xticklabels([METHOD_LABEL[m].replace("DetailView_", "") for m in methods], rotation=20)
    ax.set_ylabel("OA (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Conifer vs broadleaf OA by method")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def plot_seg_quality_delta(t20: pd.DataFrame, out: Path) -> None:
    sub = t20[t20["site"] != "both"].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sites = sorted(sub["site"].unique())
    order = ["GT", "FM", "FM_BP1", "SAT"]
    for site in sites:
        ys = []
        for m in order:
            r = sub[(sub["site"] == site) & (sub["method"] == m)]
            ys.append(float(r["accuracy"].iloc[0]) if len(r) else np.nan)
        ax.plot(range(len(order)), ys, marker="o", label=site)
        for i, v in enumerate(ys):
            if np.isfinite(v):
                ax.annotate(f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, 6), fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("OA (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Species OA vs instance source (segmentation quality)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out)


def plot_confidence(per_tree: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, site in zip(axes, sorted(per_tree["site"].unique())):
        d = per_tree[per_tree["site"] == site]
        data, labels = [], []
        for m in METHODS:
            col_c = f"correct_{m}"
            col_p = f"pred_prob_{m}"
            if col_c not in d.columns:
                continue
            ok = d[d[col_c] == True][col_p].dropna()  # noqa: E712
            bad = d[d[col_c] == False][col_p].dropna()  # noqa: E712
            if len(ok):
                data.append(ok)
                labels.append(f"{m}\nOK")
            if len(bad):
                data.append(bad)
                labels.append(f"{m}\nerr")
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(site)
        ax.set_ylabel("species_prob")
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("DetailView confidence: correct vs incorrect")
    _save(fig, out)


def plot_misclass_gallery(
    samples: dict[int, dict],
    rows: pd.DataFrame,
    out: Path,
    *,
    title: str,
) -> None:
    if rows.empty:
        return
    n = min(8, len(rows))
    rows = rows.head(n)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)
    for i, (_, r) in enumerate(rows.iterrows()):
        ax = axes[i // ncols][i % ncols]
        pid = int(r["pred_instance"])
        samp = samples.get(pid)
        if samp is None:
            ax.set_title("no xyz", fontsize=8)
            ax.axis("off")
            continue
        x, z = samp["x"], samp["z"]
        c = samp["z"] - samp["z"].min()
        ax.scatter(x - x.mean(), z - z.min(), c=c, s=0.4, cmap="viridis", linewidths=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        true_n = str(r["true_name"]).replace("_", " ")
        pred_n = str(r.get("pred_name_GT", r.get("pred_GT_name", ""))).replace("_", " ")
        prob = r.get("pred_prob_GT", np.nan)
        ax.set_title(f"GT:{true_n}\nPred:{pred_n}\np={prob:.2f}" if pd.notna(prob) else f"GT:{true_n}", fontsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title, fontsize=11)
    _save(fig, out)
