"""
plot.py
-------
Generate figures for the FVS research project report.

Reads ONLY from results/report.csv and results/*.json.
Never re-runs any algorithms.

Run as:  python plot.py
Saves:   figures/fig{N}_*.{png,pdf}
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR  = Path(__file__).parent
RESULTS_DIR  = PROJECT_DIR / "results"
FIGURES_DIR  = PROJECT_DIR / "figures"
REPORT_CSV   = RESULTS_DIR / "report.csv"

# ---------------------------------------------------------------------------
# Consistent color/marker scheme for all figures
# ---------------------------------------------------------------------------
ALGO_STYLE = {
    "IC":         {"color": "#2196F3", "marker": "o", "ls": "-",  "label": "IC"},
    "KBST":       {"color": "#4CAF50", "marker": "s", "ls": "--", "label": "KBST"},
    "MEMETIC":    {"color": "#FF9800", "marker": "^", "ls": "-.", "label": "MEMETIC"},
    "OPTIMAL":    {"color": "#9C27B0", "marker": "*", "ls": ":",  "label": "Optimal"},
    "BRUTE_FORCE":{"color": "#9C27B0", "marker": "*", "ls": ":",  "label": "BruteForce"},
}
DPI = 300


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure as PNG only."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    try:
        fig.savefig(str(path), dpi=DPI, bbox_inches="tight")
    except Exception as exc:
        logger.error("Failed to save %s: %s", path, exc)
    plt.close(fig)
    logger.info("Saved: %s", name)


def _load_report() -> pd.DataFrame:
    """Load report.csv; return empty DataFrame if missing."""
    if not REPORT_CSV.exists():
        logger.warning("report.csv not found at %s", REPORT_CSV)
        return pd.DataFrame()
    try:
        df = pd.read_csv(REPORT_CSV)
        # Coerce numeric columns
        for col in ["fvs_size", "n_vertices", "n_edges", "wall_time_sec",
                    "cpu_time_sec", "peak_memory_mb", "approximation_ratio",
                    "optimality_gap_pct", "optimal_fvs_size"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as exc:
        logger.error("Failed to load report.csv: %s", exc)
        return pd.DataFrame()


def _load_json(name: str) -> dict:
    """Load a JSON file; return empty dict if missing."""
    path = RESULTS_DIR / name
    if not path.exists():
        logger.warning("JSON not found: %s", path)
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return {}


# ---------------------------------------------------------------------------
# Figure 1 — FVS size box plot by algorithm and graph type
# ---------------------------------------------------------------------------

def fig1_fvs_size_boxplot(df: pd.DataFrame) -> None:
    if df.empty:
        logger.warning("[FIG1] No data; skipping")
        return
    exp2 = df[(df["experiment_id"] == "EXP2") & (df["fvs_size"] >= 0)].copy()
    if exp2.empty:
        logger.warning("[FIG1] No EXP2 data; skipping")
        return

    gtypes = sorted(exp2["graph_type"].dropna().unique())
    algos  = ["IC", "KBST", "MEMETIC"]
    n_sub  = len(gtypes)
    if n_sub == 0:
        return

    fig, axes = plt.subplots(1, n_sub, figsize=(4 * n_sub, 5), squeeze=False)
    fig.suptitle("FVS Solution Size Distribution by Algorithm and Graph Type", fontsize=13)

    for ax, gtype in zip(axes[0], gtypes):
        data_by_algo = [
            exp2[(exp2["graph_type"] == gtype) & (exp2["algorithm"] == a)]["fvs_size"].dropna()
            for a in algos
        ]
        bp = ax.boxplot(data_by_algo, tick_labels=algos, patch_artist=True)
        for patch, algo in zip(bp["boxes"], algos):
            patch.set_facecolor(ALGO_STYLE.get(algo, {}).get("color", "grey"))
            patch.set_alpha(0.7)
        ax.set_title(gtype, fontsize=10)
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("FVS Size")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    _save(fig, "fig1_fvs_size_boxplot")


# ---------------------------------------------------------------------------
# Figure 2 — Runtime scaling (log-log)
# ---------------------------------------------------------------------------

def fig2_runtime_scaling(df: pd.DataFrame) -> None:
    if df.empty:
        logger.warning("[FIG2] No data; skipping")
        return
    exp3 = df[(df["experiment_id"] == "EXP3") & (df["wall_time_sec"] >= 0)].copy()
    if exp3.empty:
        logger.warning("[FIG2] No EXP3 data; skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Runtime Scaling with Graph Size (Log-Log Scale)", fontsize=13)

    for algo in ["IC", "KBST", "MEMETIC"]:
        sub = exp3[exp3["algorithm"] == algo].dropna(subset=["n_vertices", "wall_time_sec"])
        if sub.empty:
            continue
        grp = sub.groupby("n_vertices")["wall_time_sec"]
        ns   = np.array(sorted(grp.groups.keys()), dtype=float)
        means = np.array([grp.get_group(n).mean() for n in ns])
        stds  = np.array([grp.get_group(n).std(ddof=0) if len(grp.get_group(n)) > 1 else 0
                          for n in ns])
        style = ALGO_STYLE.get(algo, {})
        ax.plot(ns, means, label=algo, color=style.get("color"),
                marker=style.get("marker"), linestyle=style.get("ls"))
        ax.fill_between(ns, np.maximum(means - stds, 1e-6), means + stds,
                        alpha=0.15, color=style.get("color"))

    # Reference complexity lines
    n_ref = np.logspace(1, 3, 50)
    scale = 1e-4
    ax.plot(n_ref, scale * n_ref,       "k:", lw=0.8, label="O(n)")
    ax.plot(n_ref, scale * n_ref**2,    "k--", lw=0.8, label="O(n²)")
    ax.plot(n_ref, scale * n_ref**3,    "k-.", lw=0.8, label="O(n³)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n (vertices)"); ax.set_ylabel("Wall Time (s)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig2_runtime_scaling")


# ---------------------------------------------------------------------------
# Figure 3 — Pareto frontier
# ---------------------------------------------------------------------------

def fig3_pareto_frontier(df: pd.DataFrame) -> None:
    data = _load_json("exp4_pareto.json").get("pareto_points", [])
    if not data:
        logger.warning("[FIG3] No EXP4 pareto data; skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Quality-Runtime Pareto Frontier", fontsize=13)

    by_algo: dict = {}
    for pt in data:
        algo = pt["algorithm"]
        by_algo.setdefault(algo, []).append((pt["wall_time"], pt["fvs_size"]))

    for algo, pts in by_algo.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        style = ALGO_STYLE.get(algo, {})
        ax.scatter(xs, ys, label=algo, color=style.get("color"),
                   marker=style.get("marker"), s=50, alpha=0.7)
        # Draw Pareto frontier
        pareto = _pareto_frontier(pts)
        px, py = zip(*pareto) if pareto else ([], [])
        ax.plot(px, py, color=style.get("color"), linestyle=style.get("ls"), lw=1.5)

    ax.set_xlabel("Wall Time (s)"); ax.set_ylabel("FVS Size")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig3_pareto_frontier")


def _pareto_frontier(points: list) -> list:
    """Return Pareto-optimal (x, y) pairs where lower x AND lower y is better."""
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    pareto = []
    best_y = float("inf")
    for x, y in sorted_pts:
        if y < best_y:
            pareto.append((x, y))
            best_y = y
    return pareto


# ---------------------------------------------------------------------------
# Figure 4 — Structure heatmap (FVS size)
# ---------------------------------------------------------------------------

def fig4_structure_heatmap(df: pd.DataFrame) -> None:
    data = _load_json("exp5_heatmap_data.json")
    if not data:
        logger.warning("[FIG4] No EXP5 heatmap data; skipping")
        return

    algos  = sorted(data.keys())
    gtypes = sorted({g for a in data.values() for g in a.keys()})
    matrix = np.full((len(algos), len(gtypes)), np.nan)

    for i, algo in enumerate(algos):
        for j, gt in enumerate(gtypes):
            val = data.get(algo, {}).get(gt, {}).get("median_fvs")
            if val is not None:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(max(6, len(gtypes) * 1.4), max(4, len(algos) * 1.2)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(gtypes))); ax.set_xticklabels(gtypes, rotation=40, ha="right")
    ax.set_yticks(range(len(algos)));  ax.set_yticklabels(algos)
    plt.colorbar(im, ax=ax, label="Median FVS Size")
    for i in range(len(algos)):
        for j in range(len(gtypes)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                        fontsize=8, color="black")
    ax.set_title("Algorithm Performance Across Graph Types (Median FVS Size)", fontsize=12)
    plt.tight_layout()
    _save(fig, "fig4_structure_heatmap")


# ---------------------------------------------------------------------------
# Figure 5 — Runtime heatmap
# ---------------------------------------------------------------------------

def fig5_runtime_heatmap(df: pd.DataFrame) -> None:
    data = _load_json("exp5_heatmap_data.json")
    if not data:
        logger.warning("[FIG5] No EXP5 heatmap data; skipping")
        return

    algos  = sorted(data.keys())
    gtypes = sorted({g for a in data.values() for g in a.keys()})
    matrix = np.full((len(algos), len(gtypes)), np.nan)

    for i, algo in enumerate(algos):
        for j, gt in enumerate(gtypes):
            val = data.get(algo, {}).get(gt, {}).get("median_time")
            if val is not None:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(max(6, len(gtypes) * 1.4), max(4, len(algos) * 1.2)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(gtypes))); ax.set_xticklabels(gtypes, rotation=40, ha="right")
    ax.set_yticks(range(len(algos)));  ax.set_yticklabels(algos)
    plt.colorbar(im, ax=ax, label="Median Runtime (s)")
    for i in range(len(algos)):
        for j in range(len(gtypes)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color="black")
    ax.set_title("Algorithm Runtime Across Graph Types (Median Seconds)", fontsize=12)
    plt.tight_layout()
    _save(fig, "fig5_runtime_heatmap")


# ---------------------------------------------------------------------------
# Figure 6 — GA parameter sensitivity
# ---------------------------------------------------------------------------

def fig6_ga_parameter_sensitivity(df: pd.DataFrame) -> None:
    data = _load_json("exp6_grid.json").get("grid_results", [])
    if not data:
        logger.warning("[FIG6] No EXP6 grid data; skipping")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("GA Hyperparameter Sensitivity Analysis", fontsize=13)

    params = [
        ("pop",   "Population Size",  axes[0]),
        ("mut",   "Mutation Rate",    axes[1]),
        ("cross", "Crossover Rate",   axes[2]),
    ]
    instances = sorted({d["base_id"] for d in data})
    colors    = plt.cm.tab10(np.linspace(0, 1, max(len(instances), 1)))

    for param_key, xlabel, ax in params:
        for inst, col in zip(instances, colors):
            sub = [d for d in data if d["base_id"] == inst and d["fvs_size"] > 0]
            vals = sorted({d[param_key] for d in sub})
            for v in vals:
                pass  # unique values
            means = []
            xs    = sorted(set(d[param_key] for d in sub))
            for x in xs:
                pts = [d["fvs_size"] for d in sub if d[param_key] == x]
                means.append(np.mean(pts) if pts else np.nan)
            short = inst.split("_")[0]
            ax.plot(xs, means, marker="o", label=short, color=col, alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean FVS Size")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig6_ga_parameter_sensitivity")


# ---------------------------------------------------------------------------
# Figure 7 — Convergence curves
# ---------------------------------------------------------------------------

def fig7_convergence_curves(df: pd.DataFrame) -> None:
    data = _load_json("exp7_convergence.json")
    if not data:
        logger.warning("[FIG7] No EXP7 convergence data; skipping")
        return

    instances = list(data.keys())
    n_sub = len(instances)
    if n_sub == 0:
        return

    fig, axes = plt.subplots(1, n_sub, figsize=(5 * n_sub, 5), squeeze=False)
    fig.suptitle("GA Convergence Over Generations", fontsize=13)

    for ax, inst in zip(axes[0], instances):
        curves = data[inst]  # list of list-of-[gen, size] pairs
        if not curves:
            continue
        # Align by generation index
        max_gen = max((len(c) for c in curves), default=0)
        sizes_per_gen = []
        for c in curves:
            gen_sizes = [s for _, s in c] if c and isinstance(c[0], list) else [s for _, s in c]
            sizes_per_gen.append(gen_sizes)

        # Compute mean and std per generation
        min_len = min(len(s) for s in sizes_per_gen)
        arr = np.array([s[:min_len] for s in sizes_per_gen])
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        gens = np.arange(min_len)

        ax.plot(gens, mean, color="#FF9800", lw=2, label="Mean")
        ax.fill_between(gens, mean - std, mean + std, alpha=0.3, color="#FF9800")
        ax.set_title(inst[:20], fontsize=9)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best FVS Size")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig7_convergence_curves")


# ---------------------------------------------------------------------------
# Figure 8 — Optimality gap
# ---------------------------------------------------------------------------

def fig8_optimality_gap(df: pd.DataFrame) -> None:
    if df.empty:
        return
    exp8 = df[(df["experiment_id"] == "EXP8") &
              (df["algorithm"] != "BRUTE_FORCE") &
              (df["fvs_size"] >= 0)].copy()
    if exp8.empty:
        logger.warning("[FIG8] No EXP8 data; skipping")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Optimality Gap Assessment on Small Instances (n ≤ 20)", fontsize=13)

    # Left: mean optimality gap per algorithm
    for algo in ["IC", "KBST", "MEMETIC"]:
        sub = exp8[(exp8["algorithm"] == algo)]["optimality_gap_pct"].dropna()
        mean_gap = sub.mean() if len(sub) > 0 else 0
        ax1.bar(algo, mean_gap, color=ALGO_STYLE[algo]["color"], alpha=0.8)
    ax1.set_ylabel("Mean Optimality Gap (%)")
    ax1.set_title("Mean Optimality Gap per Algorithm")
    ax1.grid(True, axis="y", alpha=0.3)

    # Right: scatter fvs_size vs optimal
    exp8_with_opt = exp8.dropna(subset=["optimal_fvs_size"])
    if not exp8_with_opt.empty:
        max_val = max(exp8_with_opt["fvs_size"].max(),
                      exp8_with_opt["optimal_fvs_size"].max(), 1)
        ax2.plot([0, max_val], [0, max_val], "k--", lw=1, label="Optimal line")
        for algo in ["IC", "KBST", "MEMETIC"]:
            sub = exp8_with_opt[exp8_with_opt["algorithm"] == algo]
            if sub.empty:
                continue
            style = ALGO_STYLE[algo]
            ax2.scatter(sub["optimal_fvs_size"], sub["fvs_size"],
                        color=style["color"], marker=style["marker"],
                        label=algo, alpha=0.7, s=40)
        ax2.set_xlabel("Optimal FVS Size (BruteForce)")
        ax2.set_ylabel("Algorithm FVS Size")
        ax2.set_title("FVS Size vs Optimal")
        ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig8_optimality_gap")


# ---------------------------------------------------------------------------
# Figure 9 — Real-world performance
# ---------------------------------------------------------------------------

def fig9_realworld_performance(df: pd.DataFrame) -> None:
    if df.empty:
        return
    exp9 = df[(df["experiment_id"] == "EXP9") & (df["fvs_size"] >= 0)].copy()
    if exp9.empty:
        logger.warning("[FIG9] No EXP9 data; skipping")
        return

    datasets = sorted(exp9["instance_id"].unique())
    algos    = ["IC", "KBST", "MEMETIC"]
    x        = np.arange(len(datasets))
    width    = 0.25

    fig, ax1 = plt.subplots(figsize=(max(8, len(datasets) * 1.8), 6))
    ax2 = ax1.twinx()

    fig.suptitle("Performance on Real-World Networks", fontsize=13)

    for i, algo in enumerate(algos):
        sizes = []
        times = []
        for ds in datasets:
            sub = exp9[(exp9["instance_id"] == ds) & (exp9["algorithm"] == algo)]
            sizes.append(sub["fvs_size"].mean() if not sub.empty else 0)
            times.append(sub["wall_time_sec"].mean() if not sub.empty else 0)
        style = ALGO_STYLE[algo]
        ax1.bar(x + i * width, sizes, width, label=f"{algo} FVS",
                color=style["color"], alpha=0.8)
        ax2.plot(x + i * width + width / 2, times,
                 color=style["color"], marker=style["marker"],
                 linestyle=style["ls"], label=f"{algo} time")

    ax1.set_xticks(x + width); ax1.set_xticklabels(
        [d.replace("realworld_", "") for d in datasets], rotation=30, ha="right")
    ax1.set_ylabel("FVS Size")
    ax2.set_ylabel("Runtime (s)")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    _save(fig, "fig9_realworld_performance")


# ---------------------------------------------------------------------------
# Figure 10 — Robustness violin plot
# ---------------------------------------------------------------------------

def fig10_robustness_violin(df: pd.DataFrame) -> None:
    if df.empty:
        return
    exp10 = df[(df["experiment_id"] == "EXP10") & (df["fvs_size"] >= 0)].copy()
    exp10_stats = _load_json("exp10_stats.json")
    if exp10.empty:
        logger.warning("[FIG10] No EXP10 data; skipping")
        return

    # Group by base instance (strip run suffix)
    exp10["base_id"] = exp10["instance_id"].str.replace(r"_run\d+$", "", regex=True)
    base_ids = sorted(exp10["base_id"].unique())

    if not base_ids:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(base_ids) * 1.5), 6))
    ax.set_title("GA Solution Stability Across 10 Independent Runs", fontsize=13)

    violin_data = [exp10[exp10["base_id"] == bid]["fvs_size"].dropna().tolist()
                   for bid in base_ids]
    violin_data = [d for d in violin_data if d]  # remove empty

    if violin_data:
        vp = ax.violinplot(violin_data, positions=range(len(violin_data)), showmedians=True)
        for body in vp["bodies"]:
            body.set_facecolor("#FF9800")
            body.set_alpha(0.7)

        # Annotate CV
        for i, bid in enumerate(base_ids[:len(violin_data)]):
            cv = exp10_stats.get(bid, {}).get("coefficient_of_variation")
            if cv is not None:
                data_i = exp10[exp10["base_id"] == bid]["fvs_size"].dropna()
                top = data_i.max() if len(data_i) > 0 else 0
                ax.text(i, top + 0.2, f"CV={cv:.1f}%", ha="center", fontsize=7)

    ax.set_xticks(range(len(base_ids)))
    ax.set_xticklabels([b[:20] for b in base_ids], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("FVS Size")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig10_robustness_violin")


# ---------------------------------------------------------------------------
# Figure 11 — Win/tie/loss matrix
# ---------------------------------------------------------------------------

def fig11_win_matrix(df: pd.DataFrame) -> None:
    if df.empty:
        return
    exp2 = df[(df["experiment_id"] == "EXP2") & (df["fvs_size"] >= 0)].copy()
    if exp2.empty:
        logger.warning("[FIG11] No EXP2 data; skipping")
        return

    algos  = ["IC", "KBST", "MEMETIC"]
    pairs  = [("IC", "KBST"), ("IC", "MEMETIC"), ("KBST", "MEMETIC")]
    labels = [f"{a} vs {b}" for a, b in pairs]

    # Win/tie/loss per pair per instance
    matrix = np.zeros((len(pairs), 3))  # [wins, ties, losses] per pair

    instances = exp2["instance_id"].unique()
    for inst in instances:
        sub = exp2[exp2["instance_id"] == inst].set_index("algorithm")["fvs_size"]
        for pi, (a, b) in enumerate(pairs):
            if a in sub and b in sub:
                sa, sb = sub[a], sub[b]
                if sa < sb:
                    matrix[pi, 0] += 1  # win
                elif sa == sb:
                    matrix[pi, 1] += 1  # tie
                else:
                    matrix[pi, 2] += 1  # loss

    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "wtl", ["#F44336", "white", "#4CAF50"]
    )
    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                   vmin=0, vmax=max(matrix.max(), 1))

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Wins (row)", "Ties", "Losses (row)"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(3):
            ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center",
                    fontsize=10, fontweight="bold")

    ax.set_title("Pairwise Algorithm Comparison (Win/Tie/Loss)", fontsize=12)
    plt.tight_layout()
    _save(fig, "fig11_win_matrix")


# ---------------------------------------------------------------------------
# Figure 12 — Summary table
# ---------------------------------------------------------------------------

def fig12_summary_table(df: pd.DataFrame) -> None:
    if df.empty:
        return

    valid = df[df["fvs_size"] >= 0].copy()
    if valid.empty:
        logger.warning("[FIG12] No valid data; skipping")
        return

    algos  = [a for a in ["IC", "KBST", "MEMETIC"] if a in valid["algorithm"].unique()]
    rows   = []
    for algo in algos:
        sub = valid[valid["algorithm"] == algo]
        rows.append([
            algo,
            f"{sub['fvs_size'].mean():.2f}" if len(sub) > 0 else "N/A",
            f"{sub['fvs_size'].median():.1f}" if len(sub) > 0 else "N/A",
            f"{sub['wall_time_sec'].mean():.4f}s" if len(sub) > 0 else "N/A",
            f"{sub['is_valid_solution'].mean() * 100:.1f}%" if len(sub) > 0 else "N/A",
        ])

    col_labels = ["Algorithm", "Mean FVS", "Median FVS", "Mean Runtime", "Success Rate"]
    fig, ax = plt.subplots(figsize=(10, 2 + len(rows)))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(True)
    tbl.scale(1, 1.5)
    ax.set_title("Comprehensive Performance Summary", fontsize=13, pad=20)
    plt.tight_layout()
    _save(fig, "fig12_summary_table")


# ---------------------------------------------------------------------------
# Figure 13-16 — Runtime points + linear regression by algorithm
# ---------------------------------------------------------------------------

def _runtime_points_from_exp3(df: pd.DataFrame, algo: str) -> pd.DataFrame:
    """Return cleaned EXP3 runtime points for one algorithm."""
    sub = df[
        (df["experiment_id"] == "EXP3")
        & (df["algorithm"] == algo)
        & (df["is_valid_solution"] == True)
    ].copy()
    if sub.empty:
        return sub

    sub = sub.dropna(subset=["n_vertices", "wall_time_sec"])
    sub = sub[sub["wall_time_sec"] >= 0]
    return sub


def _draw_runtime_regression(ax: plt.Axes, points: pd.DataFrame, algo: str, title: str) -> bool:
    """Draw scatter + linear fit for one algorithm; returns True if data exists."""
    if points.empty:
        return False

    style = ALGO_STYLE.get(algo, {"color": "#333333", "marker": "o"})
    x = points["n_vertices"].to_numpy(dtype=float)
    y = points["wall_time_sec"].to_numpy(dtype=float)

    ax.scatter(x, y, color=style["color"], marker=style["marker"], alpha=0.65, s=36)

    # Linear regression on raw scale: runtime = a*n + b
    if len(x) >= 2 and len(np.unique(x)) >= 2:
        a, b = np.polyfit(x, y, deg=1)
        xr = np.linspace(x.min(), x.max(), 100)
        yr = a * xr + b
        ax.plot(xr, yr, color=style["color"], lw=2.0, linestyle="--",
                label=f"fit: y={a:.4f}n+{b:.3f}")
        ax.legend(fontsize=8)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("n (vertices)")
    ax.set_ylabel("Wall Time (s)")
    ax.grid(True, alpha=0.3)
    return True


def fig13_ic_runtime_points(df: pd.DataFrame) -> None:
    pts = _runtime_points_from_exp3(df, "IC")
    if pts.empty:
        logger.warning("[FIG13] No EXP3 IC data; skipping")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_runtime_regression(ax, pts, "IC", "IC Runtime Points vs n (Linear Fit)")
    plt.tight_layout()
    _save(fig, "fig13_ic_runtime_points")


def fig14_kbst_runtime_points(df: pd.DataFrame) -> None:
    pts = _runtime_points_from_exp3(df, "KBST")
    if pts.empty:
        logger.warning("[FIG14] No EXP3 KBST data; skipping")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_runtime_regression(ax, pts, "KBST", "KBST Runtime Points vs n (Linear Fit)")
    plt.tight_layout()
    _save(fig, "fig14_kbst_runtime_points")


def fig15_memetic_runtime_points(df: pd.DataFrame) -> None:
    pts = _runtime_points_from_exp3(df, "MEMETIC")
    if pts.empty:
        logger.warning("[FIG15] No EXP3 MEMETIC data; skipping")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_runtime_regression(ax, pts, "MEMETIC", "MEMETIC Runtime Points vs n (Linear Fit)")
    plt.tight_layout()
    _save(fig, "fig15_memetic_runtime_points")


def fig16_runtime_points_comparison(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Runtime Points vs n: IC vs KBST vs MEMETIC (Linear Fits)", fontsize=12)

    any_data = False
    for algo in ["IC", "KBST", "MEMETIC"]:
        pts = _runtime_points_from_exp3(df, algo)
        if pts.empty:
            continue
        any_data = True

        style = ALGO_STYLE.get(algo, {"color": "#333333", "marker": "o"})
        x = pts["n_vertices"].to_numpy(dtype=float)
        y = pts["wall_time_sec"].to_numpy(dtype=float)

        ax.scatter(x, y, color=style["color"], marker=style["marker"], alpha=0.5,
                   s=30, label=f"{algo} points")

        if len(x) >= 2 and len(np.unique(x)) >= 2:
            a, b = np.polyfit(x, y, deg=1)
            xr = np.linspace(x.min(), x.max(), 100)
            ax.plot(xr, a * xr + b, color=style["color"], linestyle="--", lw=2,
                    label=f"{algo} fit")

    if not any_data:
        logger.warning("[FIG16] No EXP3 data; skipping")
        plt.close(fig)
        return

    ax.set_xlabel("n (vertices)")
    ax.set_ylabel("Wall Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    _save(fig, "fig16_runtime_points_comparison")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate figures from report.csv and JSON files."""
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] [%(levelname)s] %(message)s",
                        stream=sys.stdout)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_report()
    logger.info("Loaded report.csv: %d rows", len(df))

    generators = [
        ("fig1",  "fig1_fvs_size_boxplot",       fig1_fvs_size_boxplot),
        ("fig2",  "fig2_runtime_scaling",         fig2_runtime_scaling),
        ("fig3",  "fig3_pareto_frontier",         fig3_pareto_frontier),
        ("fig4",  "fig4_structure_heatmap",       fig4_structure_heatmap),
        ("fig5",  "fig5_runtime_heatmap",         fig5_runtime_heatmap),
        ("fig6",  "fig6_ga_parameter_sensitivity", fig6_ga_parameter_sensitivity),
        ("fig7",  "fig7_convergence_curves",      fig7_convergence_curves),
        ("fig8",  "fig8_optimality_gap",          fig8_optimality_gap),
        ("fig9",  "fig9_realworld_performance",   fig9_realworld_performance),
        ("fig10", "fig10_robustness_violin",      fig10_robustness_violin),
        ("fig11", "fig11_win_matrix",             fig11_win_matrix),
        ("fig12", "fig12_summary_table",          fig12_summary_table),
        ("fig13", "fig13_ic_runtime_points",      fig13_ic_runtime_points),
        ("fig14", "fig14_kbst_runtime_points",    fig14_kbst_runtime_points),
        ("fig15", "fig15_memetic_runtime_points", fig15_memetic_runtime_points),
        ("fig16", "fig16_runtime_points_comparison", fig16_runtime_points_comparison),
    ]

    saved = 0
    for name, file_stem, fn in generators:
        # Remove stale files first so counting reflects this run only.
        target = FIGURES_DIR / f"{file_stem}.png"
        if target.exists():
            target.unlink()

        try:
            fn(df)
            png_ok = (FIGURES_DIR / f"{file_stem}.png").exists()
            if png_ok:
                saved += 1
        except Exception as exc:
            logger.error("Failed to generate %s: %s", name, exc)

    logger.info("Figures generated: %d/%d  → %s/", saved, len(generators), FIGURES_DIR)


if __name__ == "__main__":
    main()
