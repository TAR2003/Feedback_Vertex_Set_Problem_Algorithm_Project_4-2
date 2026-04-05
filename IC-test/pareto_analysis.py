import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from analyze_ic_test import collect_test_data


def gather_points(method_rows, only_success=True):
    points_by_method = {}
    for method, rows in method_rows.items():
        pts = []
        for r in rows:
            if r.get("k", 0) <= 0:
                continue
            if only_success and not r.get("success", False):
                continue
            pts.append((r["k"], r["runtime"]))
        if pts:
            points_by_method[method] = pts
    return points_by_method


def pareto_front(points):
    # points: list of (k, runtime) where smaller is better for both
    pts = sorted(points, key=lambda x: (x[0], x[1]))
    front = []
    best_runtime = float("inf")
    for k, rt in pts:
        if rt < best_runtime:
            front.append((k, rt))
            best_runtime = rt
    return front


def plot_pareto(points_by_method, output_path, title="Pareto analysis (k vs runtime)"):
    plt.figure(figsize=(10, 6))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    method_colors = {}
    for i, method in enumerate(sorted(points_by_method.keys())):
        method_colors[method] = colors[i % len(colors)]

    # scatter all points per method
    for method, pts in points_by_method.items():
        ks = [p[0] for p in pts]
        rts = [p[1] for p in pts]
        plt.scatter(ks, rts, label=method, color=method_colors.get(method), alpha=0.6, s=50)

    # compute and plot per-method Pareto fronts
    combined_points = []
    for method, pts in points_by_method.items():
        front = pareto_front(pts)
        if front:
            ks_f = [p[0] for p in front]
            rts_f = [p[1] for p in front]
            plt.plot(ks_f, rts_f, linestyle="-", marker="o", color=method_colors.get(method), linewidth=2)
            combined_points.extend(front)

    # combined Pareto front
    if combined_points:
        combined_front = pareto_front(combined_points)
        if combined_front:
            ks_c = [p[0] for p in combined_front]
            rts_c = [p[1] for p in combined_front]
            plt.plot(ks_c, rts_c, linestyle="--", color="black", linewidth=2.5, label="Combined Pareto")

    plt.xlabel("FVS size (k)")
    plt.ylabel("Runtime (seconds)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Pareto plot: {output_path}")


def analyze_ic_folder(ic_subfolder: Path, show_plots: bool = False):
    data = collect_test_data(ic_subfolder)
    # flatten rows per method
    method_rows = {}
    for method, files in data.items():
        rows = []
        for file_rows in files.values():
            rows.extend(file_rows)
        if rows:
            method_rows[method] = rows

    if not method_rows:
        print(f"No method rows found in {ic_subfolder}")
        return

    points_by_method = gather_points(method_rows, only_success=True)
    if not points_by_method:
        print(f"No successful points found in {ic_subfolder} to analyze Pareto front.")
        return

    plot_dir = Path(ic_subfolder) / "plots"
    output_path = plot_dir / "pareto_analysis.png"
    plot_pareto(points_by_method, output_path, title=f"Pareto analysis for {ic_subfolder.name}")
    if show_plots:
        import matplotlib.image as mpimg

        img = mpimg.imread(output_path)
        plt.imshow(img)
        plt.axis("off")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Pareto analysis for IC-test FVS results")
    parser.add_argument("--ic-folder", default="IC-test", help="Root IC-test folder")
    parser.add_argument("--mode", choices=["directed", "undirected", "all"], default="all")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    root = Path(args.ic_folder)
    folders = []
    if args.mode in ("directed", "all"):
        folders.append(root / "directed")
    if args.mode in ("undirected", "all"):
        folders.append(root / "undirected")

    for f in folders:
        if f.exists():
            analyze_ic_folder(f, show_plots=args.show)
        else:
            print(f"Skipping missing folder: {f}")


if __name__ == "__main__":
    main()
