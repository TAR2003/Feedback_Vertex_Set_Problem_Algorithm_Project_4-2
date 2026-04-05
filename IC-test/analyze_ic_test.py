import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv_data(csv_path):
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            try:
                n = int(row.get("n", row.get("N", "0")))
                m = int(row.get("m", row.get("M", "0")))
                # Handle FVS_size which might be a number or "TIMEOUT"
                fvs_str = row.get("FVS_size", row.get("k", row.get("K", "0"))).strip()
                if fvs_str.upper() == "TIMEOUT":
                    k = -1  # Sentinel value indicating FVS_size is unknown due to timeout
                else:
                    k = int(fvs_str)
                runtime_raw = row.get("runtime", row.get("time", "0"))
                if runtime_raw is None:
                    runtime_raw = ""
                runtime_raw = runtime_raw.strip()
                if runtime_raw.upper() == "TIMEOUT":
                    runtime = float("inf")
                else:
                    try:
                        runtime = float(runtime_raw)
                    except Exception:
                        # non-numeric runtime -> treat as infinite timeout
                        runtime = float("inf")
            except ValueError:
                continue
            if k == 0:
                continue
            # treat runtimes > 10 or non-numeric/TIMEOUT as timeout for plotting and success calculations
            # also treat k == -1 (FVS_size was TIMEOUT) as timed_out
            timed_out = runtime > 10.0 or runtime == float("inf") or k == -1
            plot_runtime = min(runtime, 10.0) if runtime != float("inf") else 10.0
            validity = row.get("validity", "True").strip().lower() in {"true", "1", "yes"}
            success = not timed_out and validity
            rows.append({
                "file": row.get("file", ""),
                "n": n,
                "m": m,
                "k": k,
                "runtime": plot_runtime,
                "timed_out": timed_out,
                "validity": validity,
                "success": success,
            })
    return rows


def collect_test_data(ic_folder):
    ic_path = Path(ic_folder)
    data = defaultdict(lambda: defaultdict(list))
    for csv_path in sorted(ic_path.glob("*.csv")):
        method_name = csv_path.stem
        if method_name.startswith("directed_"):
            method = method_name[len("directed_"):]
        elif method_name.startswith("undirected_"):
            method = method_name[len("undirected_"):]
        else:
            method = method_name
        data[method][csv_path.name] = load_csv_data(csv_path)
    return data


def choose_best_n(rows):
    distinct_k_by_n = defaultdict(set)
    for row in rows:
        distinct_k_by_n[row["n"]].add(row["k"])
    if not distinct_k_by_n:
        return None, 0, {}
    best_n, best_count = max(distinct_k_by_n.items(), key=lambda item: (len(item[1]), -item[0]))
    return best_n, len(distinct_k_by_n[best_n]), {n: len(ks) for n, ks in distinct_k_by_n.items()}


def choose_best_k(rows):
    distinct_n_by_k = defaultdict(set)
    for row in rows:
        distinct_n_by_k[row["k"]].add(row["n"])
    if not distinct_n_by_k:
        return None, 0, {}
    best_k, best_count = max(distinct_n_by_k.items(), key=lambda item: (len(item[1]), -item[0]))
    return best_k, len(distinct_n_by_k[best_k]), {k: len(ns) for k, ns in distinct_n_by_k.items()}


METHOD_STYLE = {
    "BST_exact": {"label": "BST", "color": "#1f77b4", "marker": "o"},
    "IC_exact": {"label": "IC", "color": "#d62728", "marker": "s"},
    "brute_force": {"label": "Brute Force", "color": "#9467bd", "marker": "^"},
}


def plot_runtime_scatter(output_path, title, data_by_method, x_label):
    plt.figure(figsize=(10, 6))
    # Determine which x positions have at least one successful (non-timeout) solution
    x_has_solution = {}
    for method, points in data_by_method.items():
        for x, runtime, timed_out in points:
            if x not in x_has_solution:
                x_has_solution[x] = False
            if not timed_out:
                x_has_solution[x] = True

    for method, points in data_by_method.items():
        style = METHOD_STYLE.get(method, {"label": method, "color": None, "marker": "o"})
        pts = sorted(points, key=lambda item: item[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        timed_outs = [p[2] for p in pts]
        # plot measured runtimes (timed-out runs are capped at 10.0)
        if xs and ys:
            plt.scatter(xs, ys, label=style["label"], color=style["color"], marker=style["marker"], s=60, alpha=0.8)
        # draw colored crosses at y=10.0 for timed-out runs where another method solved the same x
        cross_xs = [x for x, to in zip(xs, timed_outs) if to and x_has_solution.get(x, False)]
        if cross_xs:
            cross_ys = [10.0] * len(cross_xs)
            plt.scatter(cross_xs, cross_ys, marker="x", color=style.get("color", None), s=100, linewidths=2)

    plt.xlabel(x_label)
    plt.ylabel("Runtime (seconds)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")


def build_method_column(method):
    if method == "BST_exact":
        return "BST"
    if method == "IC_exact":
        return "IC"
    if method == "brute_force":
        return "Brute Force"
    return method


def calculate_success_percentages(rows, group_key):
    counts = defaultdict(lambda: {"total": 0, "success": 0})
    for row in rows:
        group_value = row.get(group_key)
        if group_value is None:
            continue
        counts[group_value]["total"] += 1
        if row.get("success", False):
            counts[group_value]["success"] += 1

    return {
        value: (counts[value]["success"] / counts[value]["total"] * 100 if counts[value]["total"] else 0.0)
        for value in counts
    }


def plot_percentage_bar_chart(value_map, x_label, title, output_path):
    plt.figure(figsize=(10, 6))
    keys = sorted(value_map)
    values = [value_map[k] for k in keys]
    plt.bar(keys, values, color="#2ca02c")
    plt.xlabel(x_label)
    plt.ylabel("Success percentage (%)")
    plt.ylim(0, 100)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")


def generate_method_success_plots(method_rows, method_name, method_label, plot_dir, summary_output, show_plots=False):
    rows = method_rows.get(method_name, [])
    if not rows:
        print(f"No rows found for {method_label}")
        return

    success_by_n = calculate_success_percentages(rows, "n")
    if success_by_n:
        output_path = plot_dir / f"{method_name}_success_percentage_by_n.png"
        plot_percentage_bar_chart(
            success_by_n,
            "n",
            f"{method_label} success percentage by n (10-second timeout)",
            output_path,
        )
        if show_plots:
            plt.show()

        with summary_output.open("a", encoding="utf-8") as out:
            out.write(f"\n{method_label} success percentage by n (10-second timeout):\n")
            for value in sorted(success_by_n):
                out.write(f"n={value}: {success_by_n[value]:.1f}%\n")

    success_by_k = calculate_success_percentages(rows, "k")
    if success_by_k:
        output_path = plot_dir / f"{method_name}_success_percentage_by_k.png"
        plot_percentage_bar_chart(
            success_by_k,
            "k",
            f"{method_label} success percentage by k (10-second timeout)",
            output_path,
        )
        if show_plots:
            plt.show()

        with summary_output.open("a", encoding="utf-8") as out:
            out.write(f"\n{method_label} success percentage by k (10-second timeout):\n")
            for value in sorted(success_by_k):
                out.write(f"k={value}: {success_by_k[value]:.1f}%\n")


def analyze_folder(folder_path, show_plots=False):
    folder = Path(folder_path)
    mode = folder.name
    print(f"\nAnalyzing {mode} tests in: {folder}")
    data = collect_test_data(folder)
    method_rows = {}
    for method, files in data.items():
        rows = []
        for file_rows in files.values():
            rows.extend(file_rows)
        method_rows[method] = rows

    all_rows = []
    for rows in method_rows.values():
        all_rows.extend(rows)

    if not all_rows:
        print(f"No valid rows found in {folder}")
        return

    best_n, distinct_k_count, n_to_distinct_k = choose_best_n(all_rows)
    best_k, distinct_n_count, k_to_distinct_n = choose_best_k(all_rows)

    print(f"Best n by distinct k values: n={best_n}, distinct k count={distinct_k_count}")
    print(f"Best k by distinct n values: k={best_k}, distinct n count={distinct_n_count}")

    summary_output = folder / f"{mode}_summary.txt"
    with summary_output.open("w", encoding="utf-8") as out:
        out.write(f"Analysis summary for {mode}\n")
        out.write(f"Best n by distinct k values: n={best_n}, distinct k count={distinct_k_count}\n")
        out.write(f"Best k by distinct n values: k={best_k}, distinct n count={distinct_n_count}\n")
        out.write("\nDistinct k counts by n:\n")
        for n, count in sorted(n_to_distinct_k.items(), key=lambda item: (item[1], item[0]), reverse=True):
            out.write(f"n={n}: distinct k count={count}\n")
        out.write("\nDistinct n counts by k:\n")
        for k, count in sorted(k_to_distinct_n.items(), key=lambda item: (item[1], item[0]), reverse=True):
            out.write(f"k={k}: distinct n count={count}\n")
    print(f"Saved summary: {summary_output}")

    def gather_runtime_points(filter_key, filter_value, x_key, runtime_key="runtime", timed_out_key="timed_out"):
        result = {}
        for method, rows in method_rows.items():
            points = [
                (r[x_key], r[runtime_key], r.get(timed_out_key, False))
                for r in rows
                if r[filter_key] == filter_value
            ]
            if points:
                result[method] = points
        return result

    plot_dir = folder / "plots"
    plot_dir.mkdir(exist_ok=True)

    if best_n is not None:
        runtime_points = gather_runtime_points("n", best_n, "k")
        if runtime_points:
            output_path = plot_dir / f"{mode}_runtime_vs_k_for_n_{best_n}.png"
            plot_runtime_scatter(
                output_path,
                f"This is the plot for n={best_n}: runtime vs k",
                runtime_points,
                "k",
            )
            if show_plots:
                plt.show()
        else:
            print(f"No rows to plot for n={best_n} in {mode}")

    if best_k is not None:
        runtime_points = gather_runtime_points("k", best_k, "n")
        if runtime_points:
            output_path = plot_dir / f"{mode}_runtime_vs_n_for_k_{best_k}.png"
            plot_runtime_scatter(
                output_path,
                f"This is the plot for k={best_k}: runtime vs n",
                runtime_points,
                "n",
            )
            if show_plots:
                plt.show()
        else:
            print(f"No rows to plot for k={best_k} in {mode}")

    def plot_value_counts(value_map, label, filename):
        plt.figure(figsize=(10, 6))
        keys = sorted(value_map)
        values = [value_map[k] for k in keys]
        plt.bar(keys, values, color="#4C72B0")
        plt.xlabel(label)
        plt.ylabel(f"Distinct count")
        plt.title(f"Distinct count by {label} for {mode}")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        out_path = plot_dir / filename
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot: {out_path}")

    plot_value_counts(n_to_distinct_k, "n", f"{mode}_distinct_k_count_by_n.png")
    plot_value_counts(k_to_distinct_n, "k", f"{mode}_distinct_n_count_by_k.png")

    generate_method_success_plots(method_rows, "IC_exact", "IC", plot_dir, summary_output, show_plots=show_plots)
    generate_method_success_plots(method_rows, "BST_exact", "BST", plot_dir, summary_output, show_plots=show_plots)
    generate_method_success_plots(method_rows, "brute_force", "Brute Force", plot_dir, summary_output, show_plots=show_plots)


def main():
    parser = argparse.ArgumentParser(description="Analyze IC-test CSV files for directed and undirected datasets.")
    parser.add_argument("--ic-folder", default="IC-test", help="Root folder containing directed/ and undirected/ subfolders")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    root = Path(args.ic_folder)
    if not root.exists():
        raise FileNotFoundError(f"Root IC-test folder not found: {root}")

    for subfolder in [root / "directed", root / "undirected"]:
        if subfolder.exists():
            analyze_folder(subfolder, show_plots=args.show)
        else:
            print(f"Skipping missing folder: {subfolder}")


if __name__ == "__main__":
    main()
