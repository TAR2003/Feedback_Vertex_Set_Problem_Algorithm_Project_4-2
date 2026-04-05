from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PACER_DIR = ROOT / 'paceresults'
OUT_DIR = PACER_DIR / 'figure'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_summary():
    sf = PACER_DIR / 'summary_scores.csv'
    if not sf.exists():
        raise FileNotFoundError(f"Expected {sf} to exist. Run evaluate_fvs_scores.py first.")
    df = pd.read_csv(sf)
    return df


def save_bar(df, xcol, ycol, title, out_name, ylabel='Mean score'):
    # defensive: drop rows with missing y values
    tdf = df.copy()
    tdf = tdf.dropna(subset=[ycol])
    if tdf.empty:
        print(f"No data to plot for {out_name}")
        return

    # sort for clarity (descending)
    tdf = tdf.sort_values(by=ycol, ascending=False)

    plt.figure(figsize=(max(6, 0.5*len(tdf)), 5))
    x = np.arange(len(tdf))
    y = tdf[ycol].astype(float).values

    bars = plt.bar(x, y, color='C0')
    plt.xticks(x, tdf[xcol], rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(title)

    # smart y-limits: zoom into range if values are clustered
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    y_range = y_max - y_min
    if np.isnan(y_min) or np.isnan(y_max):
        y_min, y_max = 0, 100
    else:
        if y_range < 10:
            pad = max(1.0, 0.05 * max(1.0, y_max))
            lower = max(0.0, y_min - pad)
            upper = min(100.0, y_max + pad)
            plt.ylim(lower, upper)
        else:
            # default to 0-100 for percentage-like scores but allow slight padding
            lower = max(0.0, y_min - 0.05 * y_range)
            upper = min(100.0, y_max + 0.05 * y_range)
            plt.ylim(lower, upper)

    # annotate values above bars
    for rect, val in zip(bars, y):
        height = rect.get_height()
        offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.01
        plt.text(rect.get_x() + rect.get_width() / 2, height + offset, f"{val:.2f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = OUT_DIR / out_name
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def main():
    df = read_summary()

    # 1) All solvers together
    save_bar(df, 'solver', 'mean_score', 'PACE 2022 — all solvers mean normalized score', 'all_solvers_mean_score.png')

    # 2) Files with 15,30,60,90 in name
    for suffix in ['15', '30', '60', '90']:
        subset = df[df['solver'].str.contains(suffix, case=False, na=False)]
        if not subset.empty:
            save_bar(subset, 'solver', 'mean_score', f'PACE 2022 — solvers with {suffix}', f'group_{suffix}_mean_score.png')

    # 3) Groups: MA, KMA, DKMA
    # 3) Dynamic groups discovered from solver name tokens (auto-detect common model prefixes)
    def discover_groups(df):
        import re
        names = df['solver'].fillna('').astype(str).tolist()
        token_counts: dict[str,int] = {}
        name_tokens: list[list[str]] = []
        for n in names:
            tokens = [t for t in re.findall(r"[A-Za-z0-9]+", n) if t]
            name_tokens.append(tokens)
            for t in tokens:
                token_counts[t.upper()] = token_counts.get(t.upper(), 0) + 1

        # exclude trivial tokens and numeric suffixes handled earlier
        exclude = {"SOLVER", "MODEL", "HEURISTIC", "DIRECTED", "UNDIRECTED", "SUMMARY"}
        numeric_exclude = {"15", "30", "60", "90"}

        # prefer longer tokens first to avoid splitting KMA->MA
        tokens_sorted = sorted([t for t in token_counts.keys() if t not in exclude and t not in numeric_exclude], key=lambda x: (-len(x), -token_counts[x]))

        groups: dict[str, list[bool]] = {}
        for tok in tokens_sorted:
            # require token to appear at least twice to be considered a group
            if token_counts.get(tok, 0) < 2:
                continue
            mask = [tok.lower() in " ".join(ts).lower() for ts in name_tokens]
            if any(mask):
                groups[tok] = mask
        return groups

    groups = discover_groups(df)
    for name, mask in groups.items():
        subset = df[[m for m in mask]]
        if not subset.empty:
            save_bar(subset, 'solver', 'mean_score', f'PACE 2022 — {name} solvers', f'models_{name}_mean_score.png')

    # 4) GNN models (try to find GNN entries in summary); otherwise use directed_comparison.csv
    gnn_subset = df[df['solver'].str.contains('GNN', case=False, na=False)]
    if not gnn_subset.empty:
        save_bar(gnn_subset, 'solver', 'mean_score', 'GNN models (summary_scores)', 'gnn_models_mean_score.png')
    else:
        comp = ROOT / 'directed_comparison.csv'
        if comp.exists():
            cdf = pd.read_csv(comp)
            # look for GNN related columns
            targets = [c for c in cdf.columns if 'GNN' in c or 'KME' in c and 'GNN' in c]
            # specific fallback columns
            fallback = ['GNN-KME_size', 'MA_size', 'KME_size']
            present = [c for c in fallback if c in cdf.columns]
            if present:
                rows = []
                for c in present:
                    rows.append({'solver': c, 'mean_score': pd.to_numeric(cdf[c], errors='coerce').mean()})
                tdf = pd.DataFrame(rows)
                save_bar(tdf, 'solver', 'mean_score', 'GNN comparison from directed_comparison', 'gnn_comparison_directed.csv.png')


if __name__ == '__main__':
    main()
