"""Copy and restyle key Results figures into outputs/figures (from figures YAML)."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paths import load_figures_config, resolve_path  # noqa: E402

# Module globals — set by configure_figures()
PAPER_FIG: Path = REPO_ROOT / "outputs" / "figures"
SITES: list[dict] = []
DV_TAB: Path = REPO_ROOT / "outputs" / "detailview"
DV_FIG: Path = DV_TAB / "figures"


def configure_figures(fig_cfg: dict) -> None:
    """Bind output dirs and site entries from load_figures_config()."""
    global PAPER_FIG, SITES, DV_TAB, DV_FIG
    PAPER_FIG = Path(fig_cfg["output_dir"])
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    SITES = list(fig_cfg["sites"])
    raw = fig_cfg.get("raw") or {}
    dv_raw = raw.get("dv_results_dir", "outputs/detailview")
    DV_TAB = resolve_path(dv_raw, base=REPO_ROOT)
    DV_FIG = DV_TAB / "figures"


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _sid(entry: dict) -> str:
    return entry["site"].site_id


def _label(entry: dict) -> str:
    return entry["label"]


def _tables(entry: dict) -> Path:
    return entry["site"].tables_dir


def _figures(entry: dict) -> Path:
    return entry["site"].figures_dir


def _matched_n(entry: dict) -> int | None:
    csv = entry["site"].match_csv
    if csv is None or not Path(csv).is_file():
        return None
    return int(len(pd.read_csv(csv)))


def _unmatched_n(entry: dict) -> int:
    raw = entry["site"].raw or {}
    val = raw.get("unmatched_count", 0)
    try:
        return int(val or 0)
    except (TypeError, ValueError):
        return 0


def _require_sites(n: int, name: str) -> bool:
    if len(SITES) < n:
        print(f"skip {name}: need >= {n} sites, have {len(SITES)}")
        return False
    return True


# Paper palette (matches HTML preview accents without purple defaults)
INK = "#1a1f18"
MUTED = "#5c6658"
ACCENT = "#2f5d3a"
FM = "#2f5d3a"
SAT = "#1d4f91"
CARD = "#fffdf8"
LINE = "#d7d0c3"
GT = "#2f5d3a"
FM_C = "#4a7c59"
BP = "#8aa67a"
SAT_C = "#1d4f91"

_SITE_COLORS = (ACCENT, SAT_C, FM_C, BP)


def style_ax(ax):
    ax.set_facecolor(CARD)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)


def copy_named(pairs: list[tuple[Path, str]]) -> None:
    for src, name in pairs:
        if not src.exists():
            print(f"MISSING {src}")
            continue
        dst = PAPER_FIG / name
        shutil.copy2(src, dst)
        print(f"copied {name}")


def fig_seg_overview() -> None:
    if not _require_sites(1, "fig_seg_overview"):
        return
    n = len(SITES)
    fig, axes = plt.subplots(1, n, figsize=(5.1 * n, 3.8), sharey=True)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    metrics = ["Prec", "Rec", "F1", "Cov"]
    x = np.arange(len(metrics))
    w = 0.36
    for ax, entry in zip(axes, SITES):
        path = _tables(entry) / "T1_overall.csv"
        if not path.is_file():
            print(f"skip seg overview panel {_label(entry)}: missing {_rel(path)}")
            continue
        df = pd.read_csv(path)
        style_ax(ax)
        fm = df.loc[df["Method"] == "ForestMamba", metrics].iloc[0].to_numpy(float)
        sat = df.loc[df["Method"] == "SegmentAnyTree", metrics].iloc[0].to_numpy(float)
        ax.bar(x - w / 2, fm, w, color=FM, label="ForestMamba", zorder=2)
        ax.bar(x + w / 2, sat, w, color=SAT, label="SegmentAnyTree", zorder=2)
        ax.set_xticks(x, metrics)
        ax.set_ylim(0, 100)
        ax.set_title(_label(entry), fontsize=12, fontweight=600, pad=8)
        ax.grid(axis="y", color=LINE, linewidth=0.8, zorder=0)
        ax.set_ylabel("Score (%)" if ax is axes[0] else "")
    axes[-1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Instance segmentation at IoU threshold τ = 0.5", fontsize=13, fontweight=650, color=INK, y=1.02)
    fig.tight_layout()
    out = PAPER_FIG / "res_seg_overview.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.name}")


def fig_dv_leaderboard() -> None:
    path = DV_TAB / "T20_detailview_leaderboard.csv"
    if not path.is_file():
        print(f"skip fig_dv_leaderboard: missing {_rel(path)}")
        return
    if not _require_sites(1, "fig_dv_leaderboard"):
        return
    df = pd.read_csv(path)
    site_ids = [_sid(e) for e in SITES]
    df = df[df["site"].isin(site_ids)].copy()
    order = ["GT", "SAT", "FM", "FM_BP1"]
    labels = {
        "GT": "Manual",
        "SAT": "SegmentAnyTree",
        "FM": "ForestMamba",
        "FM_BP1": "ForestMamba+BluePoint",
    }
    colors = {"GT": GT, "SAT": SAT_C, "FM": FM_C, "FM_BP1": BP}

    n = len(SITES)
    fig, axes = plt.subplots(1, n, figsize=(5.1 * n, 3.9), sharey=True)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    for ax, entry in zip(axes, SITES):
        style_ax(ax)
        site = _sid(entry)
        sub = df[df["site"] == site].set_index("method")
        if sub.empty or "GT" not in sub.index:
            print(f"skip leaderboard panel {_label(entry)}: no rows")
            continue
        xs = np.arange(len(order))
        oa = [float(sub.loc[m, "accuracy"]) if m in sub.index else np.nan for m in order]
        f1 = [float(sub.loc[m, "macro_f1"]) if m in sub.index else np.nan for m in order]
        w = 0.36
        ax.bar(xs - w / 2, oa, w, color=[colors[m] for m in order], label="Overall accuracy", zorder=2)
        ax.bar(xs + w / 2, f1, w, color=[colors[m] for m in order], alpha=0.45, label="Macro F1", zorder=2)
        ax.set_xticks(xs, [labels[m] for m in order], rotation=18, ha="right")
        ax.set_ylim(0, 100)
        ax.set_title(_label(entry), fontsize=12, fontweight=600, pad=8)
        ax.grid(axis="y", color=LINE, linewidth=0.8, zorder=0)
        ax.set_ylabel("Score (%)" if ax is axes[0] else "")
        n_stems = int(sub.loc["GT", "N"])
        ax.text(0.98, 0.95, f"N = {n_stems}", transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=9)
    axes[-1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("DetailView stem-level scores by instance source", fontsize=13, fontweight=650, color=INK, y=1.02)
    fig.tight_layout()
    out = PAPER_FIG / "res_dv_leaderboard.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.name}")


def fig_matching_summary() -> None:
    if not _require_sites(1, "fig_matching_summary"):
        return
    sites_labels = []
    matched_list = []
    unmatched_list = []
    for entry in SITES:
        n = _matched_n(entry)
        if n is None:
            print(f"skip fig_matching_summary: missing match_csv for {_label(entry)}")
            return
        sites_labels.append(_label(entry))
        matched_list.append(float(n))
        unmatched_list.append(float(_unmatched_n(entry)))

    matched = np.array(matched_list, dtype=float)
    unmatched = np.array(unmatched_list, dtype=float)
    totals = matched + unmatched
    matched_c = ACCENT
    unmatched_c = "#9aab96"
    show_unmatched = bool(np.any(unmatched > 0))

    fig, ax = plt.subplots(figsize=(max(6.6, 2.2 * len(sites_labels)), 3.9))
    fig.patch.set_facecolor("white")
    style_ax(ax)

    x = np.arange(len(sites_labels))
    w = 0.34
    if show_unmatched:
        ax.bar(
            x - w / 2,
            matched,
            w,
            color=matched_c,
            label="Matched",
            zorder=2,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.bar(
            x + w / 2,
            unmatched,
            w,
            color=unmatched_c,
            label="Unmatched",
            zorder=2,
            edgecolor="white",
            linewidth=0.6,
        )
        annotate_pairs = list(zip(matched, unmatched))
        xpos_pairs = [(i - w / 2, i + w / 2) for i in range(len(sites_labels))]
    else:
        ax.bar(
            x,
            matched,
            w * 1.4,
            color=matched_c,
            label="Matched",
            zorder=2,
            edgecolor="white",
            linewidth=0.6,
        )
        annotate_pairs = [(m,) for m in matched]
        xpos_pairs = [(i,) for i in range(len(sites_labels))]

    for i, vals in enumerate(annotate_pairs):
        for xpos, val in zip(xpos_pairs[i], vals):
            if val < 10:
                xytext = (9, 8)
                ha = "left"
            else:
                xytext = (0, 8)
                ha = "center"
            ax.annotate(
                f"{int(val)}",
                xy=(xpos, val),
                xytext=xytext,
                textcoords="offset points",
                ha=ha,
                va="bottom",
                color=INK,
                fontweight=700,
                fontsize=11,
                clip_on=False,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\nn = {int(t)}" for s, t in zip(sites_labels, totals)], color=MUTED)
    ax.set_ylabel("Inventory stems")
    ax.set_ylim(0, float(np.max(matched)) * 1.22 if matched.size else 1.0)
    ax.tick_params(axis="x", pad=2)
    ax.yaxis.grid(True, color=LINE, linewidth=0.75, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=2, columnspacing=1.2)
    ax.set_title(
        "Exclusive assignment of inventory stems to trees",
        fontsize=12,
        fontweight=650,
        color=INK,
        pad=10,
    )
    fig.tight_layout()
    out = PAPER_FIG / "res_inventory_match_counts.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.name}")


def _short_species(name: str) -> str:
    return str(name).replace("_", " ")


def _confusion_pct(counts: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    row_ok = counts.sum(axis=1) > 0
    col_ok = counts.sum(axis=0) > 0
    keep_rows = counts.index[row_ok]
    keep_cols = counts.columns[col_ok | counts.columns.isin(keep_rows)]
    sub = counts.loc[keep_rows, keep_cols].astype(float)
    row_sum = sub.sum(axis=1).replace(0, np.nan)
    pct = (100.0 * sub.div(row_sum, axis=0)).fillna(0.0).to_numpy()
    return (
        pct,
        [_short_species(x) for x in sub.columns],
        [_short_species(x) for x in sub.index],
    )


def _draw_confusion_ax(
    ax,
    pct: np.ndarray,
    col_labels: list[str],
    row_labels: list[str],
    *,
    cmap,
    title: str,
    show_cbar: bool = False,
    fontsize: float = 7.5,
    aspect: str = "equal",
) -> object:
    n_r, n_c = pct.shape
    im = ax.imshow(pct, cmap=cmap, vmin=0, vmax=100, aspect=aspect)
    ax.set_xticks(range(n_c))
    ax.set_yticks(range(n_r))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=fontsize - 0.5, color=MUTED)
    ax.set_yticklabels(row_labels, fontsize=fontsize - 0.5, color=MUTED)
    if title:
        ax.set_title(title, fontsize=10, fontweight=650, color=INK, pad=6)
    ax.set_xticks(np.arange(-0.5, n_c, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_r, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    for i in range(n_r):
        for j in range(n_c):
            val = pct[i, j]
            if val < 0.5:
                continue
            ax.text(
                j,
                i,
                f"{val:.0f}",
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight=600,
                color="#ffffff" if val >= 55 else INK,
            )
    if show_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("% of true class", color=MUTED, fontsize=9)
        cbar.ax.tick_params(colors=MUTED, labelsize=8)
        cbar.outline.set_edgecolor(LINE)
    return im


def fig_confusion_matrices() -> None:
    """Paper-styled DetailView confusion matrices: GT singles + RQ3 multi-track grids."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "paper_green",
        ["#ffffff", "#e7efe4", "#9db89a", "#4a7c59", "#1f3d28"],
    )
    track_meta = (
        ("GT", "Manual"),
        ("SAT", "SegmentAnyTree"),
        ("FM", "ForestMamba"),
        ("FM_BP1", "ForestMamba+BluePoint"),
    )
    if not _require_sites(1, "fig_confusion_matrices"):
        return

    gt_figsize = (10.4, 5.6)
    gt_cell_fs = 8.5
    for entry in SITES:
        site = _label(entry)
        key = _sid(entry)
        tab = _tables(entry)
        n_stems = _matched_n(entry)
        if n_stems is None:
            t20 = tab / "T20_detailview_leaderboard.csv"
            if t20.is_file():
                tdf = pd.read_csv(t20)
                gt = tdf[(tdf["site"] == key) & (tdf["method"] == "GT")]
                n_stems = int(gt["N"].iloc[0]) if len(gt) else 0
            else:
                n_stems = 0
        csv_path = tab / f"T22_detailview_confusion_{key}_GT.csv"
        if not csv_path.exists():
            print(f"skip confusion ({site} GT): missing {csv_path.name}")
            continue
        counts = pd.read_csv(csv_path, index_col=0)
        pct, col_labels, row_labels = _confusion_pct(counts)
        fig, ax = plt.subplots(figsize=gt_figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor(CARD)
        _draw_confusion_ax(
            ax,
            pct,
            col_labels,
            row_labels,
            cmap=cmap,
            title="",
            show_cbar=True,
            fontsize=gt_cell_fs,
            aspect="equal",
        )
        ax.set_xlabel("Predicted species", fontsize=10, color=INK, labelpad=8)
        ax.set_ylabel("True species", fontsize=10, color=INK, labelpad=8)
        fig.suptitle(site, fontsize=13, fontweight=650, color=INK, y=0.995)
        ax.set_title(
            f"DetailView on manual instances · row-normalized · N = {n_stems}",
            fontsize=9.5,
            fontweight=400,
            color=MUTED,
            pad=10,
        )
        out = PAPER_FIG / f"res_cm_{key}_gt.png"
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.90))
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.18)
        plt.close(fig)
        print(f"wrote {out.name}")
    # Per-site 2x2 grids across instance tracks (RQ3).
    for entry in SITES:
        site = _label(entry)
        key = _sid(entry)
        tab = _tables(entry)
        n_stems = _matched_n(entry) or 0
        panels = []
        for method, label in track_meta:
            csv_path = tab / f"T22_detailview_confusion_{key}_{method}.csv"
            if not csv_path.exists():
                print(f"skip grid panel {site}/{method}")
                continue
            counts = pd.read_csv(csv_path, index_col=0)
            pct, col_labels, row_labels = _confusion_pct(counts)
            panels.append((label, pct, col_labels, row_labels))
        if len(panels) < 4:
            print(f"skip RQ3 grid ({site}): only {len(panels)} panels")
            continue

        fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6))
        fig.patch.set_facecolor("white")
        last_im = None
        for ax, (label, pct, col_labels, row_labels) in zip(axes.ravel(), panels):
            ax.set_facecolor(CARD)
            last_im = _draw_confusion_ax(
                ax,
                pct,
                col_labels,
                row_labels,
                cmap=cmap,
                title=label,
                show_cbar=False,
                fontsize=7.0,
            )
            ax.set_xlabel("Predicted", fontsize=8, color=MUTED)
            ax.set_ylabel("True", fontsize=8, color=MUTED)

        fig.suptitle(
            f"{site}: DetailView confusion by instance source (N = {n_stems})",
            fontsize=13,
            fontweight=650,
            color=INK,
            y=0.98,
        )
        fig.text(
            0.5,
            0.935,
            "Row-normalized percentages on the same inventory-matched stems",
            ha="center",
            va="top",
            fontsize=9,
            color=MUTED,
        )
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("% of true class", color=MUTED)
        cbar.ax.tick_params(colors=MUTED, labelsize=8)
        cbar.outline.set_edgecolor(LINE)
        fig.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.12, wspace=0.28, hspace=0.32)
        out = PAPER_FIG / f"res_cm_{key}_tracks.png"
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out.name}")


def fig_rq3_agreement() -> None:
    """Stem-level species-label agreement between instance tracks."""
    path = DV_TAB / "T26b_pred_agreement.csv"
    if not path.exists():
        print("skip RQ3 agreement: missing T26b")
        return
    if not _require_sites(1, "fig_rq3_agreement"):
        return
    df = pd.read_csv(path)
    pair_order = ["GT_vs_SAT", "GT_vs_FM", "GT_vs_FM_BP1", "FM_vs_FM_BP1"]
    pair_labels = {
        "GT_vs_SAT": "Manual vs SAT",
        "GT_vs_FM": "Manual vs FM",
        "GT_vs_FM_BP1": "Manual vs FM+BP",
        "FM_vs_FM_BP1": "FM vs FM+BP",
    }
    sites = [_sid(e) for e in SITES]
    site_labels = {_sid(e): _label(e) for e in SITES}
    colors = {_sid(e): _SITE_COLORS[i % len(_SITE_COLORS)] for i, e in enumerate(SITES)}

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    fig.patch.set_facecolor("white")
    style_ax(ax)
    x = np.arange(len(pair_order))
    n = len(sites)
    w = 0.8 / max(n, 1)
    for i, site in enumerate(sites):
        vals = []
        for pair in pair_order:
            row = df[(df["site"] == site) & (df["pair"] == pair)]
            vals.append(float(row["agreement"].iloc[0]) if len(row) else np.nan)
        ax.bar(
            x + (i - (n - 1) / 2) * w,
            vals,
            w * 0.9,
            color=colors[site],
            label=site_labels[site],
            zorder=2,
            edgecolor="white",
            linewidth=0.5,
        )
        for xi, v in zip(x + (i - (n - 1) / 2) * w, vals):
            if np.isfinite(v):
                ax.annotate(
                    f"{v:.0f}",
                    xy=(xi, v),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight=650,
                    color=INK,
                    clip_on=False,
                )
    ax.set_xticks(x, [pair_labels[p] for p in pair_order], rotation=15, ha="right")
    ax.set_ylabel("Stem-level label agreement (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=LINE, linewidth=0.75, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(
        "Agreement of DetailView species labels across instance tracks",
        fontsize=12,
        fontweight=650,
        color=INK,
        pad=10,
    )
    fig.tight_layout()
    out = PAPER_FIG / "res_dv_pred_agreement.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.name}")


def fig_rq3_perclass_f1() -> None:
    """Per-class F1 under each instance track for classes with inventory support."""
    path = DV_TAB / "T21_detailview_per_class.csv"
    if not path.exists():
        print("skip RQ3 per-class F1: missing T21")
        return
    if not _require_sites(1, "fig_rq3_perclass_f1"):
        return
    df = pd.read_csv(path)
    methods = ["GT", "SAT", "FM", "FM_BP1"]
    method_labels = {
        "GT": "Manual",
        "SAT": "SAT",
        "FM": "FM",
        "FM_BP1": "FM+BP",
    }
    method_colors = {"GT": GT, "SAT": SAT_C, "FM": FM_C, "FM_BP1": BP}

    n = len(SITES)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    for ax, entry in zip(axes, SITES):
        site = _sid(entry)
        title = _label(entry)
        style_ax(ax)
        gt = df[(df["site"] == site) & (df["method"] == "GT") & (df["support"] > 0)]
        species = gt.sort_values("support", ascending=False)["species"].tolist()
        x = np.arange(len(species))
        w = 0.18
        for i, m in enumerate(methods):
            vals = []
            for sp in species:
                row = df[(df["site"] == site) & (df["method"] == m) & (df["species"] == sp)]
                vals.append(float(row["f1"].iloc[0]) if len(row) else np.nan)
            ax.bar(
                x + (i - 1.5) * w,
                vals,
                w,
                color=method_colors[m],
                label=method_labels[m],
                zorder=2,
                edgecolor="white",
                linewidth=0.4,
            )
        labels = [
            f"{_short_species(sp)} (n={int(gt.loc[gt['species'] == sp, 'support'].iloc[0])})"
            for sp in species
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8, color=MUTED)
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=12, fontweight=600, pad=8)
        ax.yaxis.grid(True, color=LINE, linewidth=0.75, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylabel("Per-class F1 (%)" if ax is axes[0] else "")
    axes[-1].legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    fig.suptitle(
        "DetailView per-class F1 by instance source",
        fontsize=13,
        fontweight=650,
        color=INK,
        y=1.02,
    )
    fig.tight_layout()
    out = PAPER_FIG / "res_dv_perclass_f1_tracks.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.name}")


def _parse_layer_id(layer: object) -> int | None:
    import re

    m = re.search(r"(\d+)", str(layer))
    return int(m.group(1)) if m else None


def _load_seg_per_tree() -> pd.DataFrame:
    frames = []
    for entry in SITES:
        path = _tables(entry) / "per_tree_scores.csv"
        if not path.is_file():
            print(f"skip per_tree_scores for {_label(entry)}: missing {_rel(path)}")
            continue
        df = pd.read_csv(path)
        df["site"] = _sid(entry)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_dv_with_structure() -> pd.DataFrame:
    """Join DetailView stems to GT nearest-neighbour and hull-volume covariates."""
    dv_path = DV_TAB / "detailview_per_tree_v2.csv"
    if not dv_path.is_file():
        raise FileNotFoundError(f"missing {_rel(dv_path)}")
    dv = pd.read_csv(dv_path)
    seg = _load_seg_per_tree()
    if seg.empty:
        return dv
    cov = (
        seg[seg["Method"] == "ForestMamba"][
            ["site", "gt_id", "nn_dist_m", "hull_volume_m3", "height_m"]
        ]
        .drop_duplicates(["site", "gt_id"])
        .rename(columns={"height_m": "gt_height_m"})
    )

    links = []
    for entry in SITES:
        idx_path = entry["site"].enriched_match_csv
        if idx_path is None or not Path(idx_path).is_file():
            print(f"skip enriched index for {_label(entry)}")
            continue
        idx = pd.read_csv(idx_path)
        idx = idx.dropna(subset=["inventory_id"]).copy()
        idx["inventory_id"] = idx["inventory_id"].astype(str)
        idx["gt_id"] = idx["layer"].map(_parse_layer_id)
        idx["site"] = _sid(entry)
        links.append(idx[["site", "inventory_id", "gt_id"]].drop_duplicates(["site", "inventory_id"]))
    if not links:
        return dv
    link = pd.concat(links, ignore_index=True)

    out = dv.merge(link, on=["site", "inventory_id"], how="left")
    out = out.merge(cov, on=["site", "gt_id"], how="left")
    return out


def _spearman_ann(x: np.ndarray, y: np.ndarray) -> str:
    from scipy.stats import spearmanr

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return "ρ = n/a"
    rho, p = spearmanr(x[mask], y[mask])
    if p < 0.001:
        ps = "p<0.001"
    elif p < 0.01:
        ps = f"p={p:.3f}"
    else:
        ps = f"p={p:.2f}"
    return f"ρ={rho:.2f}, {ps}"


def fig_seg_covariates() -> None:
    """Per-tree hard F1 vs height, NN distance and hull volume for both sites."""
    df = _load_seg_per_tree()
    if df.empty:
        print("skip fig_seg_covariates: no per_tree_scores")
        return
    methods = [("ForestMamba", FM), ("SegmentAnyTree", SAT)]
    sites = [(_sid(e), _label(e)) for e in SITES]
    if len(sites) != 2:
        print(f"skip fig_seg_covariates: needs exactly 2 sites, have {len(sites)}")
        return
    specs = [
        (
            "height_m",
            False,
            "res_f1_height_both.png",
            "Per-tree hard F1 versus tree height",
        ),
        (
            "nn_dist_m",
            False,
            "res_f1_nn_both.png",
            "Per-tree hard F1 versus planimetric nearest-neighbour distance",
        ),
        (
            "hull_volume_m3",
            True,
            "res_f1_hull_both.png",
            "Per-tree hard F1 versus convex-hull volume",
        ),
    ]

    for col, log_x, out_name, title in specs:
        fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), sharey=True)
        fig.patch.set_facecolor("white")
        for r, (site, site_lab) in enumerate(sites):
            for c, (method, color) in enumerate(methods):
                ax = axes[r, c]
                style_ax(ax)
                sub = df[(df["site"] == site) & (df["Method"] == method)].copy()
                x = sub[col].to_numpy(float)
                y = (sub["f1_hard"] * 100.0).to_numpy(float)
                ax.scatter(
                    x,
                    y,
                    s=22,
                    alpha=0.55,
                    color=color,
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=3,
                )
                # Tercile mean markers
                q33, q66 = np.nanpercentile(x, [33.333, 66.667])
                for lo, hi, xpos in (
                    (np.nanmin(x) - 1e-9, q33, np.nanmedian(x[x <= q33])),
                    (q33, q66, np.nanmedian(x[(x > q33) & (x <= q66)])),
                    (q66, np.nanmax(x) + 1e-9, np.nanmedian(x[x > q66])),
                ):
                    m = (x > lo) & (x <= hi)
                    if m.sum() == 0:
                        continue
                    ax.scatter(
                        [xpos],
                        [np.nanmean(y[m])],
                        s=70,
                        marker="D",
                        color=INK,
                        zorder=4,
                        edgecolors="white",
                        linewidths=0.6,
                    )
                ann = _spearman_ann(x, y)
                n = int(np.isfinite(x).sum())
                ax.text(
                    0.03,
                    0.97,
                    f"{ann}\nN={n}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                    color=MUTED,
                    linespacing=1.25,
                )
                if log_x:
                    ax.set_xscale("log")
                ax.set_ylim(-3, 105)
                ax.yaxis.grid(True, color=LINE, linewidth=0.7, zorder=0)
                ax.set_axisbelow(True)
                if r == 0:
                    ax.set_title(method, fontsize=11, fontweight=600, pad=6)
                if c == 0:
                    ax.set_ylabel(f"{site_lab}\nHard F1 (%)", fontsize=10)
                else:
                    ax.set_ylabel("")
                if r == 1:
                    if col == "height_m":
                        ax.set_xlabel(r"Tree height $z_{99}-z_{05}$ (m)", fontsize=9.5)
                    elif col == "nn_dist_m":
                        ax.set_xlabel("Nearest-neighbour distance (m)", fontsize=9.5)
                    else:
                        ax.set_xlabel(r"Convex-hull volume (m$^3$)", fontsize=9.5)
        # Shared legend for diamonds
        axes[0, 1].scatter([], [], s=70, marker="D", color=INK, label="Tercile mean")
        axes[0, 1].legend(frameon=False, fontsize=8.5, loc="lower right")
        fig.suptitle(title, fontsize=13, fontweight=650, color=INK, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = PAPER_FIG / out_name
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out.name}")


def fig_dv_structure_covariates() -> None:
    """DetailView OA by NN and hull-volume terciles; strip plots for manual track."""
    from scipy.stats import spearmanr

    dv = _load_dv_with_structure()
    # Persist joined table for Results numbers / reuse
    out_csv = DV_TAB / "detailview_with_structure.csv"
    dv.to_csv(out_csv, index=False)
    print(f"wrote {_rel(out_csv)}")

    method_map = [
        ("GT", "correct_GT", "Manual", GT),
        ("SAT", "correct_SAT", "SegmentAnyTree", SAT_C),
        ("FM", "correct_FM", "ForestMamba", FM_C),
        ("FM_BP1", "correct_FM_BP1", "FM+BluePoint", BP),
    ]
    sites = [(_sid(e), _label(e)) for e in SITES]
    if len(sites) != 2:
        print(f"skip fig_dv_structure_covariates: needs exactly 2 sites, have {len(sites)}")
        return

    rows = []
    for cov_col, cov_label, out_bar, out_strip, title_bar, title_strip, log_x in (
        (
            "nn_dist_m",
            "Nearest-neighbour distance",
            "res_dv_oa_nn.png",
            "res_dv_correct_nn.png",
            "DetailView overall accuracy by nearest-neighbour tercile",
            "Manual-track DetailView correctness versus nearest-neighbour distance",
            False,
        ),
        (
            "hull_volume_m3",
            "Convex-hull volume",
            "res_dv_oa_hull.png",
            "res_dv_correct_hull.png",
            "DetailView overall accuracy by convex-hull volume tercile",
            "Manual-track DetailView correctness versus convex-hull volume",
            True,
        ),
    ):
        # --- Bar figure: OA by site-specific tercile ---
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=True)
        fig.patch.set_facecolor("white")
        for ax, (site, site_lab) in zip(axes, sites):
            style_ax(ax)
            sub = dv[dv["site"] == site].dropna(subset=[cov_col]).copy()
            q33, q66 = np.nanpercentile(sub[cov_col], [33.333, 66.667])

            if cov_col == "nn_dist_m":
                order = ["Low NN\n(crowded)", "Mid NN", "High NN\n(open)"]

                def stratum(v: float) -> str:
                    if v <= q33:
                        return "Low NN\n(crowded)"
                    if v <= q66:
                        return "Mid NN"
                    return "High NN\n(open)"

            else:
                order = ["Low\nvolume", "Mid\nvolume", "High\nvolume"]

                def stratum(v: float) -> str:
                    if v <= q33:
                        return "Low\nvolume"
                    if v <= q66:
                        return "Mid\nvolume"
                    return "High\nvolume"

            sub["stratum"] = sub[cov_col].map(stratum)
            x = np.arange(len(order))
            w = 0.18
            for i, (mkey, col, lab, color) in enumerate(method_map):
                means = []
                ns = []
                for st in order:
                    s = sub[sub["stratum"] == st]
                    ns.append(len(s))
                    means.append(100.0 * float(s[col].mean()) if len(s) else np.nan)
                    rows.append(
                        {
                            "site": site,
                            "covariate": cov_col,
                            "stratum": st.replace("\n", " "),
                            "method": mkey,
                            "N": len(s),
                            "OA": means[-1],
                            "q33": q33,
                            "q66": q66,
                        }
                    )
                ax.bar(
                    x + (i - 1.5) * w,
                    means,
                    w,
                    color=color,
                    label=lab,
                    zorder=2,
                    edgecolor="white",
                    linewidth=0.35,
                )
            ax.set_xticks(x, order, fontsize=8.5)
            ax.set_ylim(0, 105)
            ax.set_title(site_lab, fontsize=12, fontweight=600, pad=8)
            ax.yaxis.grid(True, color=LINE, linewidth=0.75, zorder=0)
            ax.set_axisbelow(True)
            ax.set_ylabel("Overall accuracy (%)" if ax is axes[0] else "")
            # N under first method's crowded bin as guide
            n_tot = len(sub)
            ax.text(
                0.98,
                0.97,
                f"N={n_tot}\nterciles @ {q33:.2g} / {q66:.2g}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color=MUTED,
            )
        axes[1].legend(frameon=False, fontsize=8, loc="lower right", ncol=1)
        fig.suptitle(title_bar, fontsize=13, fontweight=650, color=INK, y=1.02)
        fig.tight_layout()
        out = PAPER_FIG / out_bar
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out.name}")

        # --- Strip figure: manual correctness vs covariate ---
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), sharey=True)
        fig.patch.set_facecolor("white")
        for ax, (site, site_lab) in zip(axes, sites):
            style_ax(ax)
            sub = dv[dv["site"] == site].dropna(subset=[cov_col, "correct_GT"]).copy()
            x = sub[cov_col].to_numpy(float)
            y = sub["correct_GT"].astype(float).to_numpy()
            # jitter
            rng = np.random.default_rng(0)
            yj = y + rng.uniform(-0.08, 0.08, size=len(y))
            ok = y > 0.5
            ax.scatter(
                x[ok],
                yj[ok],
                s=28,
                alpha=0.65,
                color=ACCENT,
                edgecolors="white",
                linewidths=0.35,
                label="Correct",
                zorder=3,
            )
            ax.scatter(
                x[~ok],
                yj[~ok],
                s=28,
                alpha=0.55,
                color="#b85c38",
                edgecolors="white",
                linewidths=0.35,
                label="Incorrect",
                zorder=3,
            )
            # Binned OA line
            qs = np.nanpercentile(x, [0, 33.333, 66.667, 100])
            xs_line, ys_line = [], []
            for a, b in zip(qs[:-1], qs[1:]):
                m = (x >= a) & (x <= b) if b == qs[-1] else (x >= a) & (x < b)
                if m.sum() == 0:
                    continue
                xs_line.append(np.nanmedian(x[m]))
                ys_line.append(float(np.nanmean(y[m])))
            ax.plot(xs_line, ys_line, color=INK, linewidth=1.6, marker="D", markersize=6, zorder=4, label="Tercile OA")
            ann = _spearman_ann(x, y)
            ax.text(
                0.03,
                0.97,
                ann,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                color=MUTED,
            )
            if log_x:
                ax.set_xscale("log")
            ax.set_yticks([0, 1], ["Incorrect", "Correct"])
            ax.set_ylim(-0.25, 1.25)
            ax.set_title(site_lab, fontsize=12, fontweight=600, pad=8)
            ax.xaxis.grid(True, color=LINE, linewidth=0.7, zorder=0)
            ax.set_axisbelow(True)
            if cov_col == "nn_dist_m":
                ax.set_xlabel("Nearest-neighbour distance (m)", fontsize=9.5)
            else:
                ax.set_xlabel(r"Convex-hull volume (m$^3$)", fontsize=9.5)
        axes[1].legend(frameon=False, fontsize=8.5, loc="lower right")
        fig.suptitle(title_strip, fontsize=13, fontweight=650, color=INK, y=1.03)
        fig.tight_layout()
        out = PAPER_FIG / out_strip
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out.name}")

    # Spearman summary table for species
    spear_rows = []
    for site, _ in sites:
        sub = dv[dv["site"] == site]
        for cov_col in ("nn_dist_m", "hull_volume_m3", "height_m"):
            for mkey, col, lab, _ in method_map:
                x = sub[cov_col].to_numpy(float)
                y = sub[col].astype(float).to_numpy()
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 5:
                    continue
                rho, p = spearmanr(x[mask], y[mask])
                spear_rows.append(
                    {
                        "site": site,
                        "method": mkey,
                        "method_label": lab,
                        "covariate": cov_col,
                        "Spearman_rho": rho,
                        "p_value": p,
                        "N": int(mask.sum()),
                        "OA": 100.0 * float(y[mask].mean()),
                    }
                )
    spear_df = pd.DataFrame(spear_rows)
    spear_path = DV_TAB / "T27_detailview_structure_spearman.csv"
    spear_df.to_csv(spear_path, index=False)
    print(f"wrote {_rel(spear_path)}")

    terc_df = pd.DataFrame(rows)
    terc_path = DV_TAB / "T27_detailview_by_structure.csv"
    terc_df.to_csv(terc_path, index=False)
    print(f"wrote {_rel(terc_path)}")

    # Print key numbers for Results prose
    print("\n--- Species structure summary (manual track) ---")
    for site, _ in sites:
        s = spear_df[(spear_df.site == site) & (spear_df.method == "GT")]
        print(site)
        print(s[["covariate", "Spearman_rho", "p_value", "N"]].to_string(index=False))
        t = terc_df[(terc_df.site == site) & (terc_df.method == "GT")]
        print(t[["covariate", "stratum", "N", "OA"]].to_string(index=False))


def fig_tau_curves() -> None:
    """IoU-threshold sweeps with matching titles for configured sites."""
    if not _require_sites(1, "fig_tau_curves"):
        return
    for entry in SITES:
        site = _label(entry)
        key = _sid(entry)
        tab = _tables(entry) / "T7_tau_sweep.csv"
        out_name = f"res_tau_{key}.png"
        bench_fig = _figures(entry) / "tau_curves.png"
        if not tab.is_file():
            print(f"skip tau curves ({site}): missing {_rel(tab)}")
            continue
        df = pd.read_csv(tab)
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6), sharex=True)
        fig.patch.set_facecolor("white")
        for ax, metric in zip(axes.ravel(), ["Cov", "Prec", "Rec", "F1"]):
            style_ax(ax)
            for method, color in (("ForestMamba", FM), ("SegmentAnyTree", SAT)):
                sub = df[df["Method"] == method].sort_values("tau")
                ax.plot(
                    sub["tau"],
                    sub[metric],
                    "o-",
                    color=color,
                    label=method,
                    linewidth=1.8,
                    markersize=5.5,
                    zorder=3,
                )
            ax.set_ylabel(f"{metric} (%)")
            ax.set_xlabel(r"IoU threshold $\tau$")
            ax.set_ylim(0, 105)
            ax.yaxis.grid(True, color=LINE, linewidth=0.7, zorder=0)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=8.5, loc="best")
        fig.suptitle(f"{site}: metrics vs IoU threshold", fontsize=13, fontweight=650, color=INK, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = PAPER_FIG / out_name
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        bench_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(bench_fig, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out.name}")


def _retitle_existing_png(src: Path, dst: Path, title: str, *, crop_frac: float = 0.075) -> None:
    """Crop an old embedded title and redraw with a consistent site heading."""
    if not src.exists():
        print(f"MISSING {src}")
        return
    img = plt.imread(src)
    h = img.shape[0]
    cut = max(1, int(round(h * crop_frac)))
    crop = img[cut:, ...]
    aspect = crop.shape[1] / max(crop.shape[0], 1)
    fig_w = 10.2
    fig_h = max(3.2, fig_w / aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.imshow(crop)
    ax.set_axis_off()
    fig.suptitle(title, fontsize=13, fontweight=650, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(dst, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"retitled {dst.name}")


def fig_paired_site_diagnostics() -> None:
    """Regenerate site figures with uniform '{Site}: …' titles and paper styling."""
    err_colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    err_keys = [
        ("over_seg", "over-seg"),
        ("under_seg", "under-seg"),
        ("missed", "missed"),
        ("leakage_flagged_tp", "leakage"),
    ]
    if not _require_sites(1, "fig_paired_site_diagnostics"):
        return

    for entry in SITES:
        site = _label(entry)
        key = _sid(entry)
        tab = _tables(entry) / "T1_overall.csv"
        pts = _tables(entry) / "per_tree_scores.csv"
        t8 = _tables(entry) / "T8_macro_vs_plot.csv"
        out_iou = f"res_iou_{key}.png"
        out_err = f"res_errors_{key}.png"
        out_macro = f"res_macro_{key}.png"
        bench_iou = _figures(entry) / "iou_histogram.png"
        bench_err = _figures(entry) / "error_rates.png"
        bench_macro = _figures(entry) / "plot_vs_macro_f1.png"
        if not pts.is_file() or not tab.is_file() or not t8.is_file():
            print(f"skip diagnostics ({site}): missing tables under {_rel(_tables(entry))}")
            continue

        pdf = pd.read_csv(pts)
        fig, ax = plt.subplots(figsize=(9.2, 3.9))
        fig.patch.set_facecolor("white")
        style_ax(ax)
        for method, color in (("ForestMamba", FM), ("SegmentAnyTree", SAT)):
            vals = pdf.loc[pdf["Method"] == method, "max_iou"].to_numpy(float)
            ax.hist(vals, bins=20, range=(0, 1), alpha=0.55, label=method, color=color, zorder=2)
        ax.axvline(0.5, color=INK, ls="--", lw=1.1, label=r"$\tau=0.5$", zorder=3)
        ax.set_xlabel("Max IoU per GT tree")
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        ax.yaxis.grid(True, color=LINE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f"{site}: per-GT best IoU", fontsize=13, fontweight=650, color=INK, pad=8)
        fig.tight_layout()
        out = PAPER_FIG / out_iou
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        _figures(entry).mkdir(parents=True, exist_ok=True)
        fig.savefig(bench_iou, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out.name}")

        t1 = pd.read_csv(tab)
        methods = t1["Method"].tolist()
        x = np.arange(len(methods))
        w = 0.18
        fig, ax = plt.subplots(figsize=(8.4, 4.0))
        fig.patch.set_facecolor("white")
        style_ax(ax)
        for i, (ekey, lab) in enumerate(err_keys):
            vals = [float(t1.loc[t1["Method"] == m, ekey].iloc[0]) for m in methods]
            ax.bar(
                x + (i - 1.5) * w,
                vals,
                w,
                label=lab,
                color=err_colors[i],
                zorder=2,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks(x, methods)
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
        ax.yaxis.grid(True, color=LINE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f"{site}: error taxonomy", fontsize=13, fontweight=650, color=INK, pad=8)
        fig.tight_layout()
        out = PAPER_FIG / out_err
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        fig.savefig(bench_err, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out.name}")

        mdf = pd.read_csv(t8)
        fig, ax = plt.subplots(figsize=(7.6, 4.0))
        fig.patch.set_facecolor("white")
        style_ax(ax)
        xs = np.arange(len(methods))
        w = 0.34
        plot_vals = [float(mdf.loc[mdf["Method"] == m, "plot_F1"].iloc[0]) for m in methods]
        macro_vals = [float(mdf.loc[mdf["Method"] == m, "macro_f1_hard"].iloc[0]) for m in methods]
        lo = [float(mdf.loc[mdf["Method"] == m, "macro_f1_hard_ci_lo"].iloc[0]) for m in methods]
        hi = [float(mdf.loc[mdf["Method"] == m, "macro_f1_hard_ci_hi"].iloc[0]) for m in methods]
        ax.bar(xs - w / 2, plot_vals, w, color=FM, label="Plot F1", zorder=2)
        ax.bar(xs + w / 2, macro_vals, w, color=SAT, label="Macro hard F1", zorder=2)
        ax.errorbar(
            xs + w / 2,
            macro_vals,
            yerr=np.vstack([np.array(macro_vals) - np.array(lo), np.array(hi) - np.array(macro_vals)]),
            fmt="none",
            ecolor=INK,
            capsize=3.5,
            zorder=3,
        )
        ax.set_xticks(xs, methods)
        ax.set_ylabel("F1 (%)")
        ax.set_ylim(0, 105)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        ax.yaxis.grid(True, color=LINE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f"{site}: plot-level F1 vs macro hard F1", fontsize=13, fontweight=650, color=INK, pad=8)
        fig.tight_layout()
        out = PAPER_FIG / out_macro
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        fig.savefig(bench_macro, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out.name}")

    if len(SITES) >= 1:
        methods = ["ForestMamba", "SegmentAnyTree"]
        n = len(SITES)
        fig, axes = plt.subplots(1, n, figsize=(5.1 * n, 3.9), sharey=True)
        if n == 1:
            axes = [axes]
        fig.patch.set_facecolor("white")
        for ax, entry in zip(axes, SITES):
            t8path = _tables(entry) / "T8_macro_vs_plot.csv"
            if not t8path.is_file():
                print(f"skip macro both panel {_label(entry)}")
                continue
            style_ax(ax)
            mdf = pd.read_csv(t8path)
            xs = np.arange(len(methods))
            w = 0.34
            plot_vals = [float(mdf.loc[mdf["Method"] == m, "plot_F1"].iloc[0]) for m in methods]
            macro_vals = [float(mdf.loc[mdf["Method"] == m, "macro_f1_hard"].iloc[0]) for m in methods]
            lo = [float(mdf.loc[mdf["Method"] == m, "macro_f1_hard_ci_lo"].iloc[0]) for m in methods]
            hi = [float(mdf.loc[mdf["Method"] == m, "macro_f1_hard_ci_hi"].iloc[0]) for m in methods]
            ax.bar(xs - w / 2, plot_vals, w, color=FM, label="Plot F1", zorder=2)
            ax.bar(xs + w / 2, macro_vals, w, color=SAT, label="Macro hard F1", zorder=2)
            ax.errorbar(
                xs + w / 2,
                macro_vals,
                yerr=np.vstack([np.array(macro_vals) - np.array(lo), np.array(hi) - np.array(macro_vals)]),
                fmt="none",
                ecolor=INK,
                capsize=3.5,
                zorder=3,
            )
            ax.set_xticks(xs, methods, rotation=12, ha="right")
            ax.set_ylim(0, 105)
            ax.set_title(_label(entry), fontsize=12, fontweight=600, pad=8)
            ax.yaxis.grid(True, color=LINE, linewidth=0.7, zorder=0)
            ax.set_axisbelow(True)
            ax.set_ylabel("F1 (%)" if ax is axes[0] else "")
        axes[-1].legend(frameon=False, fontsize=8.5, loc="upper right")
        fig.suptitle("Plot-level F1 versus macro hard F1", fontsize=13, fontweight=650, color=INK, y=1.02)
        fig.tight_layout()
        out = PAPER_FIG / "res_macro_both.png"
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out.name}")

    for entry in SITES:
        key = _sid(entry)
        site = _label(entry)
        fig_dir = _figures(entry)
        spatial = fig_dir / "spatial_tp_fp_fn.png"
        qual = fig_dir / "qualitative_instances_tile001.png"
        if spatial.is_file():
            _retitle_existing_png(
                spatial,
                PAPER_FIG / f"res_spatial_{key}.png",
                rf"{site}: spatial TP / FN / FP ($\tau=0.5$)",
                crop_frac=0.08,
            )
        else:
            print(f"skip spatial retitle ({site}): missing {_rel(spatial)}")
        if qual.is_file():
            _retitle_existing_png(
                qual,
                PAPER_FIG / f"res_qual_{key}.png",
                f"{site}: tile 001 instance colours (subsampled)",
                crop_frac=0.09,
            )
        else:
            print(f"skip qual retitle ({site}): missing {_rel(qual)}")

    for entry in SITES:
        t2_path = _tables(entry) / "T2_per_tile.csv"
        if not t2_path.is_file():
            continue
        site = _label(entry)
        key = _sid(entry)
        t2 = pd.read_csv(t2_path)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True)
        fig.patch.set_facecolor("white")
        tile_ids = sorted(t2["Tile"].unique().tolist())
        tile_labels = [f"{int(t):03d}" for t in tile_ids]
        for ax, metric in zip(axes, ["F1", "Cov"]):
            style_ax(ax)
            x = np.arange(len(tile_ids))
            w = 0.36
            for i, (method, color) in enumerate((("ForestMamba", FM), ("SegmentAnyTree", SAT))):
                vals = [
                    float(t2[(t2["Tile"] == t) & (t2["Method"] == method)][metric].iloc[0])
                    for t in tile_ids
                ]
                ax.bar(x + (i - 0.5) * w, vals, w, label=method, color=color, zorder=2)
            ax.set_xticks(x, tile_labels)
            ax.set_xlabel("Tile")
            ax.set_ylabel(f"{metric} (%)")
            ax.set_ylim(0, 105)
            ax.yaxis.grid(True, color=LINE, linewidth=0.7, zorder=0)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        fig.suptitle(rf"{site}: per-tile metrics at $\tau=0.5$", fontsize=13, fontweight=650, color=INK, y=1.02)
        fig.tight_layout()
        out = PAPER_FIG / f"res_pertile_{key}.png"
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        _figures(entry).mkdir(parents=True, exist_ok=True)
        fig.savefig(_figures(entry) / "per_tile_bars.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out.name}")


def _height_bin_key(b: object) -> float:
    s = str(b)
    return float(s.split(",")[0].strip("([]"))


def fig_dv_oa_height() -> None:
    """Paper-styled DetailView OA by height bin + height histograms for configured sites."""
    t23_path = DV_TAB / "T23_detailview_by_height.csv"
    dv_path = DV_TAB / "detailview_per_tree_v2.csv"
    if not t23_path.is_file() or not dv_path.is_file():
        print(f"skip fig_dv_oa_height: missing DV tables under {_rel(DV_TAB)}")
        return
    t23 = pd.read_csv(t23_path)
    dv = pd.read_csv(dv_path)
    site_labels = {_sid(e): _label(e) for e in SITES}
    method_meta = (
        ("GT", "Manual", GT),
        ("SAT", "SegmentAnyTree", SAT_C),
        ("FM", "ForestMamba", FM_C),
        ("FM_BP1", "FM+BluePoint", BP),
    )
    configured = [_sid(e) for e in SITES]
    sites = [s for s in configured if s in set(t23["site"])]
    if not sites:
        print("skip fig_dv_oa_height: no matching sites in T23")
        return

    fig, axes = plt.subplots(len(sites), 2, figsize=(11.4, 3.85 * len(sites)), squeeze=False)
    fig.patch.set_facecolor("white")

    for row, site in enumerate(sites):
        pretty = site_labels.get(site, site.title())
        d = t23[t23["site"] == site].copy()
        bins = sorted(d["height_bin"].unique(), key=_height_bin_key)
        x = np.arange(len(bins))

        ax = axes[row][0]
        style_ax(ax)
        for method, label, color in method_meta:
            vals = []
            for b in bins:
                r = d[(d["height_bin"] == b) & (d["method"] == method)]
                vals.append(float(r["accuracy"].iloc[0]) if len(r) else np.nan)
            ax.plot(
                x,
                vals,
                "o-",
                color=color,
                label=label,
                linewidth=1.7,
                markersize=4.8,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in bins], rotation=40, ha="right", fontsize=7.5, color=MUTED)
        ax.set_xlabel("Height bin (m)", fontsize=9.5, color=INK)
        ax.set_ylabel("Overall accuracy (%)")
        ax.set_ylim(-3, 108)
        ax.set_xlim(-0.6, len(bins) - 0.4)
        ax.yaxis.grid(True, color=LINE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f"{pretty}: overall accuracy by height", fontsize=11, fontweight=600, color=INK, pad=8)

        ax2 = axes[row][1]
        style_ax(ax2)
        hh = (
            dv[dv["site"] == site]
            .groupby("height_bin", dropna=False)
            .size()
            .reset_index(name="n")
        )
        if len(hh):
            hh = hh.sort_values("height_bin", key=lambda s: s.map(_height_bin_key))
            xpos = np.arange(len(hh))
            ax2.bar(
                xpos,
                hh["n"],
                width=0.55,
                color="#8aa67a",
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
            )
            ax2.set_xticks(xpos)
            ax2.set_xticklabels(hh["height_bin"].astype(str), rotation=40, ha="right", fontsize=7.5, color=MUTED)
            ax2.set_xlim(-0.8, max(len(hh) - 0.2, 12))
        ax2.set_xlabel("Height bin (m)", fontsize=9.5, color=INK)
        ax2.set_ylabel("Stem count")
        ax2.yaxis.grid(True, color=LINE, linewidth=0.7, zorder=0)
        ax2.set_axisbelow(True)
        ax2.set_title(f"{pretty}: height distribution", fontsize=11, fontweight=600, color=INK, pad=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle(
        r"DetailView overall accuracy by $2\,\mathrm{m}$ height strata",
        fontsize=13,
        fontweight=650,
        color=INK,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = PAPER_FIG / "res_dv_oa_height.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    DV_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(DV_FIG / "dv_oa_by_height.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="configs/figures.example.yaml",
        help="Figures YAML (relative to repo root).",
    )
    args = ap.parse_args()
    fig_cfg = load_figures_config(args.config)
    configure_figures(fig_cfg)
    print(f"output_dir={_rel(PAPER_FIG)} sites={[ _label(e) for e in SITES]} dv={_rel(DV_TAB)}")

    copy_named(
        [
            (DV_FIG / "dv_lifeform_bars.png", "res_dv_lifeform.png"),
            (DV_FIG / "dv_seg_quality_delta.png", "res_dv_seg_gap.png"),
        ]
    )
    overlay = PAPER_FIG / "inventory_overlays_side_by_side.png"
    if overlay.exists():
        shutil.copy2(overlay, PAPER_FIG / "res_inventory_overlays_side_by_side.png")
        print("copied res_inventory_overlays_side_by_side.png")

    fig_tau_curves()
    fig_paired_site_diagnostics()
    fig_dv_oa_height()
    fig_seg_overview()
    fig_dv_leaderboard()
    fig_matching_summary()
    fig_confusion_matrices()
    fig_rq3_agreement()
    fig_rq3_perclass_f1()
    fig_seg_covariates()
    fig_dv_structure_covariates()


if __name__ == "__main__":
    main()
