#!/usr/bin/env python3
"""
Generate PT datasets for GNN training from exact and/or heuristic tracks.

Track behavior:
- exact_track: label with PACE22 winner solver
- heuristic_track: label with PACE22 winner solver

Output layout:
        gnn_model/datasets/pt/<family>/<track>/<category>/*.pt

Each .pt file is a torch_geometric Data object with:
    data.x, data.edge_index, data.y, data.fvs_size
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import random
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import networkx as nx
from feature_engineering_v2 import (
    compute_node_features_directed_v2,
    compute_node_features_undirected_v2,
)
try:
    from feature_engineering_v3 import (
        compute_node_features_directed_v3,
        compute_node_features_undirected_v3,
    )
    HAS_FEAT_V3 = True
except ImportError:
    HAS_FEAT_V3 = False
    compute_node_features_directed_v3 = None
    compute_node_features_undirected_v3 = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for candidate in ("build-linux", "build-macos", "build-win", "build"):
    sys.path.insert(0, str(PROJECT_ROOT / "cpp_engine" / candidate))

try:
    import cpp_engine

    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    print("WARNING: cpp_engine not found. Using Python fallback solver (slow).")

try:
    import torch
    from torch_geometric.data import Data

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _log(msg: str) -> None:
    """Always flush output so progress is visible during long runs."""
    print(msg, flush=True)


OUTPUT_ROOT = PROJECT_ROOT / "gnn_model" / "datasets" / "pt"
SYNTHETIC_ROOT = PROJECT_ROOT / "data" / "synthetic"
CSV_ROOT = PROJECT_ROOT / "gnn_model" / "datasets"

EXACT_TRACK = "exact_track"
HEURISTIC_TRACK = "heuristic_track"
TRACK_CHOICES = (EXACT_TRACK, HEURISTIC_TRACK)
UNDIRECTED_WEIGHTS: Dict[str, float] = {
    "real_world": 0.20,
    "scale_free": 0.20,
    "small_world": 0.20,
    "random_er": 0.20,
    "grids_trees": 0.20,
}

DIRECTED_WEIGHTS: Dict[str, float] = {
    "real_world_ego": 0.30,
    "scale_free": 0.20,
    "random_er": 0.20,
    "directed_grids": 0.15,
    "dags": 0.15,
}

SOLVER_TIMEOUT_SECONDS = 20
KMA_POP_SIZE = 20
KMA_MAX_GENS = 100
KMA_EARLY_STOP = 20

PACE_SOLVER_DIR = PROJECT_ROOT / "fvs_solver_pace22_winner"
PACE_SOLVER_BIN = PACE_SOLVER_DIR / "fvs_solver"


class SolverTimeoutError(RuntimeError):
    """Raised when a single graph solve exceeds the configured timeout."""


class InvalidFVSResultError(RuntimeError):
    """Raised when a solver returns an invalid FVS set."""


class SolverExecutionError(RuntimeError):
    """Raised when the primary and fallback solvers both fail."""


class AggregateProgress:
    """Track-level progress with a spinner-like status line."""

    _frames = ("|", "/", "-", "\\")

    def __init__(self, totals_by_track: Dict[str, int]):
        self.totals_by_track = {
            EXACT_TRACK: int(totals_by_track.get(EXACT_TRACK, 0)),
            HEURISTIC_TRACK: int(totals_by_track.get(HEURISTIC_TRACK, 0)),
        }
        self.done_by_track = {EXACT_TRACK: 0, HEURISTIC_TRACK: 0}
        self._frame_idx = 0

    def advance(self, track: str, count: int = 1) -> None:
        if track not in self.done_by_track:
            return
        self.done_by_track[track] += int(count)
        self.print_status()

    def print_status(self) -> None:
        frame = self._frames[self._frame_idx % len(self._frames)]
        self._frame_idx += 1

        exact_done = self.done_by_track[EXACT_TRACK]
        exact_total = self.totals_by_track[EXACT_TRACK]
        heuristic_done = self.done_by_track[HEURISTIC_TRACK]
        heuristic_total = self.totals_by_track[HEURISTIC_TRACK]
        remaining = max((exact_total + heuristic_total) - (exact_done + heuristic_done), 0)

        print(
            f"[{frame}] Progress | Exact {exact_done}/{exact_total} | "
            f"Heuristic {heuristic_done}/{heuristic_total} | Remaining {remaining}",
            flush=True,
        )


def _to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    as_posix = resolved.as_posix()
    if ":/" not in as_posix:
        return as_posix
    drive, tail = as_posix.split(":/", 1)
    return f"/mnt/{drive.lower()}/{tail}"


def _pace_solver_command() -> List[str]:
    if not PACE_SOLVER_BIN.exists():
        raise FileNotFoundError(f"PACE22 solver binary not found: {PACE_SOLVER_BIN}")
    if sys.platform.startswith("win"):
        return ["wsl", _to_wsl_path(PACE_SOLVER_BIN)]
    return [str(PACE_SOLVER_BIN)]


def _format_solver_input(graph_type: str, n: int, edges: List[Tuple[int, int]]) -> str:
    directed = 1 if graph_type == "directed" else 0
    lines = [
        "# format: edge_list_v1",
        f"# directed: {directed}",
        f"p edge {n} {len(edges)}",
    ]
    lines.extend(f"{u} {v}" for u, v in edges)
    return "\n".join(lines) + "\n"


def _parse_solver_output(stdout: str) -> List[int]:
    fvs: List[int] = []
    seen: Set[int] = set()
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        tok = s.split()[0]
        if tok.lstrip("-").isdigit():
            v = int(tok)
            if v not in seen:
                seen.add(v)
                fvs.append(v)
    return fvs


def _solve_with_timeout(
    graph_type: str,
    n: int,
    edges: List[Tuple[int, int]],
    timeout_seconds: int,
) -> List[int]:
    def _hard_kill(proc: subprocess.Popen[str]) -> None:
        # Bound all cleanup operations so timeout handling itself cannot hang.
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5,
                )
            except Exception:
                pass
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass

        try:
            proc.kill()
        except Exception:
            pass

        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

        try:
            proc.wait(timeout=2)
        except Exception:
            pass

    def _fallback_solve() -> List[int]:
        # Fallback path prevents one crashing instance from aborting the full dataset build.
        if graph_type == "undirected":
            return solve_undirected(n, edges)
        return solve_directed(n, edges)

    solver_input = _format_solver_input(graph_type, n, edges)
    cmd = _pace_solver_command()
    popen_kwargs = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if os.name == "nt":
        # Start in a separate process group so we can hard-kill the full tree.
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
        try:
            stdout, stderr = proc.communicate(input=solver_input, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            # Strict timeout: immediately kill process tree and move on.
            _hard_kill(proc)
            raise SolverTimeoutError(f"solver exceeded {timeout_seconds}s") from exc
        proc_result_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        raise SolverTimeoutError(f"solver exceeded {timeout_seconds}s") from exc
    except FileNotFoundError as exc:
        try:
            return _fallback_solve()
        except Exception as fb_exc:  # pragma: no cover - defensive fallback path
            raise SolverExecutionError(
                "Failed to launch PACE22 solver (is WSL installed on Windows?) and fallback failed"
            ) from fb_exc

    if proc_result_code != 0:
        stderr = (stderr or "").strip()
        tail = "\n".join(stderr.splitlines()[-10:]) if stderr else "<no stderr output>"
        try:
            return _fallback_solve()
        except Exception as fb_exc:  # pragma: no cover - defensive fallback path
            raise SolverExecutionError(
                f"PACE22 solver failed with exit code {proc_result_code}: {tail}; fallback failed"
            ) from fb_exc

    return _parse_solver_output(stdout or "")


def _normalize_edges(edges: Iterable[Tuple[int, int]], directed: bool) -> List[Tuple[int, int]]:
    seen: Set[Tuple[int, int]] = set()
    for u_raw, v_raw in edges:
        u = int(u_raw)
        v = int(v_raw)
        if u == v:
            continue
        if directed:
            seen.add((u, v))
        else:
            seen.add((u, v) if u <= v else (v, u))
    return sorted(seen)


def _randint(rng: random.Random, lo: int, hi: int) -> int:
    return lo if lo == hi else rng.randint(lo, hi)


def _list_existing_txt(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.txt") if p.is_file())


def _parse_edge_list_txt(path: Path) -> Tuple[int, List[Tuple[int, int]]]:
    edges: List[Tuple[int, int]] = []
    n_hint: int | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("#", "%", "c ")):
                continue

            parts = s.split()
            if not parts:
                continue

            if parts[0].lower() == "p" and len(parts) >= 4:
                try:
                    n_hint = int(parts[2])
                except ValueError:
                    pass
                continue

            if len(parts) < 2 or not parts[0].lstrip("-").isdigit() or not parts[1].lstrip("-").isdigit():
                continue

            edges.append((int(parts[0]), int(parts[1])))

    if not edges:
        raise ValueError(f"No edges found in {path}")

    verts = {v for e in edges for v in e}
    min_v = min(verts)
    max_v = max(verts)

    # Existing corpora may be 1-based. Normalize to 0-based for engine/model inputs.
    if min_v == 1:
        edges = [(u - 1, v - 1) for u, v in edges]
        max_v -= 1

    n = (n_hint if n_hint is not None else (max_v + 1))
    n = max(n, max_v + 1)
    return n, edges


def _random_tree(n: int, seed: int) -> nx.Graph:
    if hasattr(nx, "random_labeled_tree"):
        return nx.random_labeled_tree(n, seed=seed)
    return nx.random_tree(n, seed=seed)


def _molecule_like_graph(n: int, rng: random.Random) -> nx.Graph:
    g = _random_tree(n, seed=rng.randint(0, 10**9))
    candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if not g.has_edge(u, v)]
    rng.shuffle(candidates)
    target_extra = max(1, n // 8)
    added = 0
    for u, v in candidates:
        if added >= target_extra:
            break
        if g.degree(u) >= 4 or g.degree(v) >= 4:
            continue
        g.add_edge(u, v)
        added += 1
    return g


def _directed_scale_free(n: int, rng: random.Random) -> nx.DiGraph:
    base = nx.scale_free_graph(n, seed=rng.randint(0, 10**9))
    d = nx.DiGraph()
    d.add_nodes_from(range(n))
    for u, v in base.edges():
        if u != v:
            d.add_edge(int(u), int(v))
    return d


def _directed_grid(rows: int, cols: int, rng: random.Random) -> nx.DiGraph:
    g = nx.grid_2d_graph(rows, cols)
    g = nx.convert_node_labels_to_integers(g)
    d = nx.DiGraph()
    d.add_nodes_from(g.nodes())
    for u, v in g.edges():
        if rng.random() < 0.5:
            d.add_edge(u, v)
        else:
            d.add_edge(v, u)
        if rng.random() < 0.08:
            d.add_edge(v, u)
    return d


def _directed_real_world_proxy(n: int, rng: random.Random, dense: bool) -> nx.DiGraph:
    g = nx.gn_graph(n, seed=rng.randint(0, 10**9)).to_directed()
    add_budget = max(1, n // (12 if dense else 25))
    for _ in range(add_budget):
        u = rng.randrange(0, n)
        v = rng.randrange(0, n)
        if u != v:
            g.add_edge(v, u)
    return g


def _build_undirected(category: str, track: str, rng: random.Random, idx: int) -> nx.Graph:
    if category == "real_world":
        if track == EXACT_TRACK:
            n = _randint(rng, 10, 35)
            return _molecule_like_graph(n, rng)
        n = _randint(rng, 100, 5000)
        m = _randint(rng, 2, 6)
        return nx.barabasi_albert_graph(n, m, seed=rng.randint(0, 10**9))

    if category == "scale_free":
        if track == EXACT_TRACK:
            n = _randint(rng, 15, 35)
            m = _randint(rng, 2, 3)
        else:
            n = _randint(rng, 100, 2000)
            m = _randint(rng, 2, 5)
        return nx.barabasi_albert_graph(n, m, seed=rng.randint(0, 10**9))

    if category == "small_world":
        if track == EXACT_TRACK:
            n = _randint(rng, 15, 35)
            k, p = 4, 0.1
        else:
            n = _randint(rng, 100, 1000)
            k, p = 6, 0.2
        return nx.watts_strogatz_graph(n, k, p, seed=rng.randint(0, 10**9))

    if category == "random_er":
        if track == EXACT_TRACK:
            n = _randint(rng, 10, 30)
            p = rng.uniform(0.1, 0.4)
        else:
            n = _randint(rng, 50, 1000)
            p = rng.uniform(0.05, 0.2)
        return nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 10**9))

    if category == "grids_trees":
        subtype = "grid" if idx % 2 == 0 else "tree"
        if subtype == "grid":
            if track == EXACT_TRACK:
                rows = _randint(rng, 3, 6)
                cols = _randint(rng, 3, 5)
            else:
                rows = _randint(rng, 10, 50)
                cols = _randint(rng, 10, 50)
            g = nx.grid_2d_graph(rows, cols)
            return nx.convert_node_labels_to_integers(g)

        if track == EXACT_TRACK:
            n = _randint(rng, 10, 35)
        else:
            n = _randint(rng, 100, 5000)
        return _random_tree(n, seed=rng.randint(0, 10**9))

    raise ValueError(f"Unknown undirected category: {category}")


def _build_directed(category: str, track: str, rng: random.Random) -> nx.DiGraph:
    if category == "real_world_ego":
        if track == EXACT_TRACK:
            n = _randint(rng, 10, 35)
            return _directed_real_world_proxy(n, rng, dense=False)
        n = _randint(rng, 100, 5000)
        return _directed_real_world_proxy(n, rng, dense=True)

    if category == "scale_free":
        if track == EXACT_TRACK:
            n = _randint(rng, 15, 35)
        else:
            n = _randint(rng, 100, 2000)
        return _directed_scale_free(n, rng)

    if category == "random_er":
        if track == EXACT_TRACK:
            n = _randint(rng, 10, 30)
            p = rng.uniform(0.1, 0.4)
        else:
            n = _randint(rng, 50, 1000)
            p = rng.uniform(0.05, 0.2)
        return nx.erdos_renyi_graph(n, p, directed=True, seed=rng.randint(0, 10**9))

    if category == "directed_grids":
        if track == EXACT_TRACK:
            side = _randint(rng, 3, 6)
            rows, cols = side, side
        else:
            rows = _randint(rng, 10, 50)
            cols = _randint(rng, 10, 50)
        return _directed_grid(rows, cols, rng)

    if category == "dags":
        if track == EXACT_TRACK:
            n = _randint(rng, 10, 35)
        else:
            n = _randint(rng, 100, 5000)
        return nx.gn_graph(n, seed=rng.randint(0, 10**9))

    raise ValueError(f"Unknown directed category: {category}")


def _graph_to_edge_list(g: nx.Graph, directed: bool) -> Tuple[int, List[Tuple[int, int]]]:
    node_map = {node: idx for idx, node in enumerate(sorted(g.nodes(), key=str))}
    edges = [(node_map[u], node_map[v]) for u, v in g.edges()]
    n = len(node_map)
    return n, _normalize_edges(edges, directed=directed)


def _cap_graph_size(g: nx.Graph, max_nodes: int, seed: int) -> nx.Graph:
    """Downsample very large graphs to keep PT label generation practical."""
    if max_nodes <= 0 or g.number_of_nodes() <= max_nodes:
        return g

    rng = random.Random(seed)
    nodes = list(g.nodes())
    rng.shuffle(nodes)
    keep = set(nodes[:max_nodes])
    sub = g.subgraph(keep).copy()
    return nx.convert_node_labels_to_integers(sub)


def compute_node_features_undirected(n: int, edges: List[Tuple[int, int]]) -> List[List[float]]:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)

    degrees = dict(g.degree())
    clust = nx.clustering(g)
    feats: List[List[float]] = []
    for v in range(n):
        deg = degrees.get(v, 0)
        feats.append(
            [
                deg / max(n - 1, 1),
                clust.get(v, 0.0),
                math.log(deg + 1) / math.log(n + 1),
            ]
        )
    return feats


def compute_node_features_directed(n: int, edges: List[Tuple[int, int]]) -> List[List[float]]:
    in_deg = [0] * n
    out_deg = [0] * n
    for u, v in edges:
        out_deg[u] += 1
        in_deg[v] += 1

    feats: List[List[float]] = []
    for v in range(n):
        ind = in_deg[v]
        outd = out_deg[v]
        feats.append(
            [
                ind / max(n - 1, 1),
                outd / max(n - 1, 1),
                min(ind, outd) / max(n - 1, 1),
            ]
        )
    return feats


def solve_undirected(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    if HAS_ENGINE:
        return cpp_engine.solve_undirected_IC(n, edges)

    adj = {v: set() for v in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    fvs: List[int] = []
    removed: Set[int] = set()

    def has_cycle() -> bool:
        visited: Set[int] = set()

        def dfs(cur: int, parent: int) -> bool:
            visited.add(cur)
            for nb in adj[cur]:
                if nb in removed:
                    continue
                if nb == parent:
                    continue
                if nb in visited:
                    return True
                if dfs(nb, cur):
                    return True
            return False

        for node in range(n):
            if node not in removed and node not in visited and dfs(node, -1):
                return True
        return False

    while has_cycle():
        best = max((v for v in range(n) if v not in removed), key=lambda v: len(adj[v] - removed))
        removed.add(best)
        fvs.append(best)
    return fvs


def solve_directed(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    if HAS_ENGINE:
        return cpp_engine.solve_directed_IC(n, edges)

    out_adj = {v: set() for v in range(n)}
    for u, v in edges:
        out_adj[u].add(v)

    fvs: List[int] = []
    removed: Set[int] = set()

    def has_dcycle() -> bool:
        color: Dict[int, int] = {}

        def dfs(cur: int) -> bool:
            color[cur] = 1
            for nb in out_adj[cur]:
                if nb in removed:
                    continue
                c = color.get(nb, 0)
                if c == 1:
                    return True
                if c == 0 and dfs(nb):
                    return True
            color[cur] = 2
            return False

        for node in range(n):
            if node not in removed and color.get(node, 0) == 0 and dfs(node):
                return True
        return False

    while has_dcycle():
        best = max((v for v in range(n) if v not in removed), key=lambda v: len(out_adj[v] - removed))
        removed.add(best)
        fvs.append(best)
    return fvs


def solve_undirected_kma(
    n: int,
    edges: List[Tuple[int, int]],
    timeout_seconds: int = SOLVER_TIMEOUT_SECONDS,
    pop_size: int = KMA_POP_SIZE,
    max_gens: int = KMA_MAX_GENS,
    early_stop: int = KMA_EARLY_STOP,
) -> List[int]:
    if HAS_ENGINE:
        if hasattr(cpp_engine, "solve_undirected_KMA"):
            return cpp_engine.solve_undirected_KMA(
                n, edges, pop_size, max_gens, early_stop, timeout_seconds
            )
        if hasattr(cpp_engine, "solve_undirected_KME"):
            return cpp_engine.solve_undirected_KME(
                n, edges, pop_size, max_gens, early_stop, timeout_seconds
            )
        return cpp_engine.solve_undirected_MA(
            n, edges, pop_size, max_gens, early_stop, timeout_seconds
        )

    # Python fallback (no cpp_engine): use the exact fallback heuristic.
    return solve_undirected(n, edges)


def solve_directed_kma(
    n: int,
    edges: List[Tuple[int, int]],
    timeout_seconds: int = SOLVER_TIMEOUT_SECONDS,
    pop_size: int = KMA_POP_SIZE,
    max_gens: int = KMA_MAX_GENS,
    early_stop: int = KMA_EARLY_STOP,
) -> List[int]:
    if HAS_ENGINE:
        if hasattr(cpp_engine, "solve_directed_KMA"):
            return cpp_engine.solve_directed_KMA(
                n, edges, pop_size, max_gens, early_stop, timeout_seconds
            )
        if hasattr(cpp_engine, "solve_directed_KME"):
            return cpp_engine.solve_directed_KME(
                n, edges, pop_size, max_gens, early_stop, timeout_seconds
            )
        return cpp_engine.solve_directed_MA(
            n, edges, pop_size, max_gens, early_stop, timeout_seconds
        )

    # Python fallback (no cpp_engine): use the exact fallback heuristic.
    return solve_directed(n, edges)


def _is_acyclic_undirected_after_removal(
    n: int,
    edges: List[Tuple[int, int]],
    removed: Set[int],
) -> bool:
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        if u in removed or v in removed or u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)

    visited = [False] * n

    for start in range(n):
        if start in removed or visited[start]:
            continue

        stack: List[Tuple[int, int]] = [(start, -1)]
        visited[start] = True

        while stack:
            cur, parent = stack.pop()
            for nb in adj[cur]:
                if nb == parent:
                    continue
                if visited[nb]:
                    return False
                visited[nb] = True
                stack.append((nb, cur))

    return True


def _is_acyclic_directed_after_removal(
    n: int,
    edges: List[Tuple[int, int]],
    removed: Set[int],
) -> bool:
    indeg = [0] * n
    out_adj: List[List[int]] = [[] for _ in range(n)]
    active_count = 0

    for v in range(n):
        if v not in removed:
            active_count += 1

    for u, v in edges:
        if u in removed or v in removed:
            continue
        out_adj[u].append(v)
        indeg[v] += 1

    q = [v for v in range(n) if v not in removed and indeg[v] == 0]
    head = 0
    seen = 0
    while head < len(q):
        cur = q[head]
        head += 1
        seen += 1
        for nb in out_adj[cur]:
            indeg[nb] -= 1
            if indeg[nb] == 0:
                q.append(nb)

    return seen == active_count


def validate_fvs_solution(
    graph_type: str,
    n: int,
    edges: List[Tuple[int, int]],
    fvs: List[int],
) -> bool:
    removed = set(int(v) for v in fvs)
    if len(removed) != len(fvs):
        return False
    if any(v < 0 or v >= n for v in removed):
        return False

    if graph_type == "undirected":
        return _is_acyclic_undirected_after_removal(n, edges, removed)
    return _is_acyclic_directed_after_removal(n, edges, removed)


def _build_pt_sample(
    graph_type: str,
    n: int,
    edges: List[Tuple[int, int]],
    variant: str = "v1",
    solver_timeout_seconds: int = SOLVER_TIMEOUT_SECONDS,
    track: str = EXACT_TRACK,
    kma_pop_size: int = KMA_POP_SIZE,
    kma_max_gens: int = KMA_MAX_GENS,
    kma_early_stop: int = KMA_EARLY_STOP,
    family: str = "unknown",
    category: str = "unknown",
) -> Data:
    if not HAS_TORCH:
        raise RuntimeError("torch and torch_geometric are required for PT generation")

    # Run the timeout-guarded solver first so hard time limits apply immediately
    # for each graph before any potentially expensive feature engineering.
    fvs = _solve_with_timeout(graph_type, n, edges, solver_timeout_seconds)

    if variant == "v3":
        if not HAS_FEAT_V3:
            raise ImportError("feature_engineering_v3.py not found; run from gnn_model/ directory")
        if graph_type == "undirected":
            feats = compute_node_features_undirected_v3(n, edges)
        else:
            feats = compute_node_features_directed_v3(n, edges)
    elif variant == "v2":
        if graph_type == "undirected":
            feats = compute_node_features_undirected_v2(n, edges)
        else:
            feats = compute_node_features_directed_v2(n, edges)
    else:
        if graph_type == "undirected":
            feats = compute_node_features_undirected(n, edges)
        else:
            feats = compute_node_features_directed(n, edges)

    if not validate_fvs_solution(graph_type, n, edges, fvs):
        raise InvalidFVSResultError(
            "invalid FVS produced by PACE22 winner solver"
        )

    x = torch.tensor(feats, dtype=torch.float)
    y = torch.zeros(n, dtype=torch.long)
    for v in fvs:
        if 0 <= v < n:
            y[v] = 1

    if edges:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        if graph_type == "undirected":
            ei = torch.cat([ei, ei.flip(0)], dim=1)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=ei, y=y)
    data.fvs_size = len(fvs)
    # Store family/category for stratified train/val splitting in train.py
    data.family = family
    data.category = category
    return data


def _list_existing_pt(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.pt") if p.is_file())


def _get_csv_path(variant: str) -> Path:
    """Get the path to the CSV tracking file for a variant."""
    CSV_ROOT.mkdir(parents=True, exist_ok=True)
    return CSV_ROOT / f"dataset_gen_{variant}.csv"


def _csv_headers() -> List[str]:
    """Return the CSV column headers."""
    return [
        "source_file",
        "family",
        "track",
        "category",
        "solver",
        "status",  # 'completed' or 'timeout' or 'invalid' or 'solver_error'
        "fvs_size",
        "timestamp",
        "feature_set",
    ]


def _load_csv_records(variant: str) -> Dict[str, Dict[str, str]]:
    """Load existing CSV records keyed by family/track/category/source_file."""
    csv_path = _get_csv_path(variant)
    records: Dict[str, Dict[str, str]] = {}
    
    if not csv_path.exists():
        return records
    
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader is None or reader.fieldnames is None:
                return records
            for row in reader:
                if row:
                    # Track-aware key prevents collisions across exact/heuristic data.
                    key = (
                        f"{row.get('family', '')}/"
                        f"{row.get('track', EXACT_TRACK)}/"
                        f"{row.get('category', '')}/"
                        f"{row.get('source_file', '')}"
                    )
                    records[key] = row
    except Exception as e:
        _log(f"[WARN] Failed to read CSV {csv_path}: {e}")
    
    return records


def _record_exists(
    records: Dict[str, Dict[str, str]],
    family: str,
    track: str,
    category: str,
    source_file: str,
) -> bool:
    """Check if a record exists for this file."""
    key = f"{family}/{track}/{category}/{source_file}"
    return key in records


def _save_csv_record(
    variant: str,
    family: str,
    track: str,
    category: str,
    solver: str,
    source_file: str,
    status: str,
    fvs_size: int = 0,
) -> None:
    """Append a record to the CSV file."""
    csv_path = _get_csv_path(variant)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().isoformat()
    
    # Check if file exists; if not, write headers
    file_exists = csv_path.exists()
    
    try:
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_csv_headers())
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "source_file": source_file,
                "family": family,
                "track": track,
                "category": category,
                "solver": solver,
                "status": status,
                "fvs_size": fvs_size if status == "completed" else "",
                "timestamp": timestamp,
                "feature_set": variant,
            })
    except Exception as e:
        _log(f"[WARN] Failed to write CSV record: {e}")


def _trim_non_source_pt(folder: Path, source_stems: Set[str]) -> int:
    removed = 0
    for p in _list_existing_pt(folder):
        if p.stem not in source_stems:
            p.unlink()
            removed += 1
    return removed


def _generate_bucket_from_sources(
    family: str,
    track: str,
    category: str,
    output_root: Path,
    variant: str,
    force: bool,
    progress_every: int,
    solver_timeout_seconds: int,
    kma_pop_size: int,
    kma_max_gens: int,
    kma_early_stop: int,
    progress: AggregateProgress,
) -> Tuple[int, int]:
    src_dir = SYNTHETIC_ROOT / family / track / category
    src_files = _list_existing_txt(src_dir)
    if not src_files:
        _log(f"[WARN] No source graphs found in {src_dir}")
        return 0, 0

    out_dir = output_root / family / track / category
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV records for this variant
    csv_records = _load_csv_records(variant)

    if force:
        removed = _list_existing_pt(out_dir)
        for p in removed:
            p.unlink()
        if removed:
            _log(f"[CLEAN] {family}/{track}/{category}: removed {len(removed)} old .pt files")

    source_stems = {p.stem for p in src_files}
    trimmed = _trim_non_source_pt(out_dir, source_stems)
    if trimmed:
        _log(f"[TRIM] {family}/{track}/{category}: removed {trimmed} non-source .pt files")

    created = 0
    existing_used = 0
    total = len(src_files)
    graph_type = "directed" if family == "directed" else "undirected"

    for idx, src_path in enumerate(src_files, start=1):
        out_path = out_dir / f"{src_path.stem}.pt"
        
        # Check CSV first: if record exists, skip
        if _record_exists(csv_records, family, track, category, src_path.name):
            record = csv_records[f"{family}/{track}/{category}/{src_path.name}"]
            status = record.get("status", "unknown")
            _log(
                f"  [{family}/{track}/{category}] graph {idx}/{total} "
                f"skip | source={src_path.name} (already {status})"
            )
            existing_used += 1
            progress.advance(track)
            continue

        # Fallback: if .pt file exists but not in CSV, use it
        if out_path.exists():
            existing_used += 1
            progress.advance(track)
            continue

        n, edges = _parse_edge_list_txt(src_path)
        edges = _normalize_edges(edges, directed=(graph_type == "directed"))

        solver_name = "pace22_winner"
        _log(
            f"  [{family}/{track}/{category}] graph {idx}/{total} "
            f"start | n={n} m={len(edges)} solver={solver_name} source={src_path.name}"
        )
        t0 = time.perf_counter()
        try:
            data = _build_pt_sample(
                graph_type,
                n,
                edges,
                variant=variant,
                solver_timeout_seconds=solver_timeout_seconds,
                track=track,
                kma_pop_size=kma_pop_size,
                kma_max_gens=kma_max_gens,
                kma_early_stop=kma_early_stop,
                family=family,
                category=category,
            )
        except SolverTimeoutError:
            dt = time.perf_counter() - t0
            _log(
                f"    [TIMEOUT] {dt:.2f}s "
                f"(limit={solver_timeout_seconds}s) | source={src_path.name}"
            )
            # Record the timeout in CSV
            _save_csv_record(
                variant=variant,
                family=family,
                track=track,
                category=category,
                solver=solver_name,
                source_file=src_path.name,
                status="timeout",
                fvs_size=0,
            )
            existing_used += 1
            progress.advance(track)
            continue
        except InvalidFVSResultError as exc:
            dt = time.perf_counter() - t0
            _log(f"    [INVALID] {dt:.2f}s | source={src_path.name} | reason={exc}")
            _save_csv_record(
                variant=variant,
                family=family,
                track=track,
                category=category,
                solver=solver_name,
                source_file=src_path.name,
                status="invalid",
                fvs_size=0,
            )
            existing_used += 1
            progress.advance(track)
            continue
        except SolverExecutionError as exc:
            dt = time.perf_counter() - t0
            _log(f"    [SOLVER-ERROR] {dt:.2f}s | source={src_path.name} | reason={exc}")
            _save_csv_record(
                variant=variant,
                family=family,
                track=track,
                category=category,
                solver=solver_name,
                source_file=src_path.name,
                status="solver_error",
                fvs_size=0,
            )
            existing_used += 1
            progress.advance(track)
            continue
        except RuntimeError as exc:
            dt = time.perf_counter() - t0
            _log(f"    [SOLVER-ERROR] {dt:.2f}s | source={src_path.name} | reason={exc}")
            _save_csv_record(
                variant=variant,
                family=family,
                track=track,
                category=category,
                solver=solver_name,
                source_file=src_path.name,
                status="solver_error",
                fvs_size=0,
            )
            existing_used += 1
            progress.advance(track)
            continue
        dt = time.perf_counter() - t0
        data.family = family
        data.track = track
        data.category = category
        data.source_file = src_path.name
        data.feature_set = variant
        torch.save(data, out_path)
        created += 1

        # Record completion in CSV
        _save_csv_record(
            variant=variant,
            family=family,
            track=track,
            category=category,
            solver=solver_name,
            source_file=src_path.name,
            status="completed",
            fvs_size=int(data.fvs_size),
        )

        progress.advance(track)

        _log(f"    done in {dt:.2f}s | fvs_size={int(data.fvs_size)} | saved={out_path.name}")
        if (created % progress_every) == 0 or (created + existing_used) == total:
            done = created + existing_used
            pct = 100.0 * done / max(total, 1)
            _log(f"  [{family}/{track}/{category}] ready {done}/{total} ({pct:.1f}%)")

    return created, existing_used


def _run_family(
    family: str,
    output_root: Path,
    variant: str,
    tracks: Sequence[str],
    force: bool,
    progress_every: int,
    solver_timeout_seconds: int,
    kma_pop_size: int,
    kma_max_gens: int,
    kma_early_stop: int,
    progress: AggregateProgress,
) -> Tuple[int, int]:
    weights = UNDIRECTED_WEIGHTS if family == "undirected" else DIRECTED_WEIGHTS
    categories = list(weights.keys())

    _log(f"\n{family.upper()} PT plan")
    _log("-" * 76)
    for track in tracks:
        for category in categories:
            src_dir = SYNTHETIC_ROOT / family / track / category
            count = len(_list_existing_txt(src_dir))
            _log(f"{track:<16} {category:<18} source-files={count:>8}")

    created = 0
    skipped = 0
    for track in tracks:
        for category in categories:
            c, s = _generate_bucket_from_sources(
                family=family,
                track=track,
                category=category,
                output_root=output_root,
                variant=variant,
                force=force,
                progress_every=progress_every,
                solver_timeout_seconds=solver_timeout_seconds,
                kma_pop_size=kma_pop_size,
                kma_max_gens=kma_max_gens,
                kma_early_stop=kma_early_stop,
                progress=progress,
            )
            created += c
            skipped += s
            _log(f"[DONE] {family}/{track}/{category}: created={c}, existing-used={s}")
    return created, skipped


def _build_track_totals(
    families: Sequence[str],
    tracks: Sequence[str],
) -> Dict[str, int]:
    totals = {EXACT_TRACK: 0, HEURISTIC_TRACK: 0}
    for family in families:
        categories = (
            list(UNDIRECTED_WEIGHTS.keys()) if family == "undirected" else list(DIRECTED_WEIGHTS.keys())
        )
        for track in tracks:
            for category in categories:
                src_dir = SYNTHETIC_ROOT / family / track / category
                totals[track] += len(_list_existing_txt(src_dir))
    return totals


def _get_output_root(variant: str) -> Path:
    if variant == "v1":
        return OUTPUT_ROOT
    if variant == "v2":
        return PROJECT_ROOT / "gnn_model" / "datasets" / "pt_v2"
    if variant == "v3":
        return PROJECT_ROOT / "gnn_model" / "datasets" / "pt_v3"
    raise ValueError(f"Unsupported variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exact/heuristic-track PT datasets for GNN training")
    parser.add_argument("--family", choices=["all", "undirected", "directed"], default="all")
    parser.add_argument(
        "--track",
        choices=["exact", "heuristic", "both"],
        default="both",
        help="Source tracks to generate labels for",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate selected track buckets by deleting existing PT files before generation",
    )
    parser.add_argument(
        "--clean-root",
        action="store_true",
        help="Delete the full PT output root before generation",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N created files per bucket (default: 10)",
    )
    parser.add_argument(
        "--variant",
        choices=["v1", "v2", "v3"],
        default="v1",
        help="Feature pipeline variant: v1 (legacy), v2 (RWSE+motifs+coreness), or v3",
    )
    parser.add_argument(
        "--solver-timeout",
        type=int,
        default=SOLVER_TIMEOUT_SECONDS,
        help="PACE22 winner solver timeout in seconds (used for both exact and heuristic tracks).",
    )
    # Backward-compatible no-op flags kept so older scripts do not break.
    parser.add_argument("--kma-pop", type=int, default=KMA_POP_SIZE, help=argparse.SUPPRESS)
    parser.add_argument("--kma-gens", type=int, default=KMA_MAX_GENS, help=argparse.SUPPRESS)
    parser.add_argument("--kma-early-stop", type=int, default=KMA_EARLY_STOP, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not HAS_TORCH:
        _log("ERROR: torch and torch_geometric are required for PT generation.")
        _log("Install with: pip install torch torch-geometric")
        sys.exit(1)

    output_root = _get_output_root(args.variant)

    if args.clean_root and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    progress_every = max(1, args.progress_every)
    solver_timeout_seconds = max(1, args.solver_timeout)

    if args.track == "exact":
        tracks: Sequence[str] = (EXACT_TRACK,)
    elif args.track == "heuristic":
        tracks = (HEURISTIC_TRACK,)
    else:
        tracks = TRACK_CHOICES

    families: Sequence[str] = ("undirected", "directed") if args.family == "all" else (args.family,)
    totals = _build_track_totals(families, tracks)
    progress = AggregateProgress(totals)
    progress.print_status()

    total_created = 0
    total_existing_used = 0
    for family in families:
        c, s = _run_family(
            family,
            output_root,
            args.variant,
            tracks,
            args.force,
            progress_every,
            solver_timeout_seconds,
            args.kma_pop,
            args.kma_gens,
            args.kma_early_stop,
            progress,
        )
        total_created += c
        total_existing_used += s

    _log("\nSummary")
    _log("-------")
    _log(f"Track(s):      {', '.join(tracks)}")
    _log(f"Created:       {total_created}")
    _log(f"Existing-used: {total_existing_used}")
    _log(f"Output root:   {output_root}")


if __name__ == "__main__":
    main()
