"""
test_gnn_components.py
======================
Unit tests for the research-grade GNN-KMA components.

Tests:
  1. AsymmetricFVSLoss — verifies FP loss > FN loss
  2. compute_topk_precision — verifies metric on known inputs
  3. compute_rwse_fast — verifies diagonal probability on simple cycle
  4. compute_scc_features — verifies SCC sizes on known directed graph
  5. _pick_gnn_candidates_precision_first — verifies empty return when no
     vertex exceeds threshold (no topk-fallback)

Run with:
  python -m pytest tests/test_gnn_components.py -v

Or directly:
  python tests/test_gnn_components.py
"""

import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gnn_model"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
#  Test: AsymmetricFVSLoss
# ═══════════════════════════════════════════════════════════════════════════════

def test_asymmetric_fvs_loss_fp_vs_fn():
    """
    Verify that FP loss (y=0, pred=1) is higher than FN loss (y=1, pred=0)
    for equivalent prediction confidence.

    With fp_gamma=2.0, fn_gamma=0.5:
      FP: high-confidence wrong prediction → large focal weight
      FN: high-confidence wrong prediction → small focal weight
    """
    try:
        import torch
        from gnn_model.train import AsymmetricFVSLoss
    except ImportError:
        pytest.skip("torch or train.py not available")

    loss_fn = AsymmetricFVSLoss(fp_gamma=2.0, fn_gamma=0.5)

    # Both predictions equally wrong at confidence 0.9
    # FP case: model predicts 1 (logit=2.2) but y=0
    logit_fp = torch.tensor([2.2])
    y_fp = torch.tensor([0], dtype=torch.long)

    # FN case: model predicts 0 (logit=-2.2) but y=1
    logit_fn = torch.tensor([-2.2])
    y_fn = torch.tensor([1], dtype=torch.long)

    loss_fp = loss_fn(logit_fp, y_fp).item()
    loss_fn_ = loss_fn(logit_fn, y_fn).item()

    print(f"  FP loss: {loss_fp:.6f}")
    print(f"  FN loss: {loss_fn_:.6f}")

    assert loss_fp > loss_fn_, (
        f"Expected FP loss ({loss_fp:.6f}) > FN loss ({loss_fn_:.6f}). "
        "Asymmetric loss should penalize false positives more."
    )


def test_asymmetric_fvs_loss_correct_predictions_near_zero():
    """Correct predictions should contribute near-zero loss."""
    try:
        import torch
        from gnn_model.train import AsymmetricFVSLoss
    except ImportError:
        pytest.skip("torch or train.py not available")

    loss_fn = AsymmetricFVSLoss(fp_gamma=2.0, fn_gamma=0.5)

    # Very confident correct predictions
    logits  = torch.tensor([5.0, -5.0, 5.0, -5.0])
    targets = torch.tensor([1,    0,    1,    0], dtype=torch.long)
    loss = loss_fn(logits, targets).item()
    assert loss < 0.01, f"Correct predictions should have near-zero loss, got {loss:.6f}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Test: compute_topk_precision
# ═══════════════════════════════════════════════════════════════════════════════

def test_compute_topk_precision_perfect():
    """Perfect predictions → precision@k = 1.0."""
    try:
        import torch
        from gnn_model.train import compute_topk_precision
    except ImportError:
        pytest.skip("torch or train.py not available")

    # 10 vertices, 2 in FVS (20%)
    logits  = torch.tensor([5.0, 5.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    labels  = torch.tensor([1,   1,    0,    0,    0,    0,    0,    0,    0,    0])

    # With k_fraction=0.2: k=2, top-2 should be the FVS vertices
    prec = compute_topk_precision(logits, labels, k_fraction=0.2)
    assert prec == 1.0, f"Expected precision=1.0, got {prec}"


def test_compute_topk_precision_zero():
    """All wrong → precision@k = 0.0."""
    try:
        import torch
        from gnn_model.train import compute_topk_precision
    except ImportError:
        pytest.skip("torch or train.py not available")

    logits  = torch.tensor([5.0, 5.0, -1.0, -1.0, -1.0])
    labels  = torch.tensor([0,   0,    1,    1,    1])

    prec = compute_topk_precision(logits, labels, k_fraction=0.4)
    assert prec == 0.0, f"Expected precision=0.0 (all wrong), got {prec}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Test: compute_rwse_fast
# ═══════════════════════════════════════════════════════════════════════════════

def test_compute_rwse_on_triangle():
    """
    For a directed triangle (0→1, 1→2, 2→0), each node has return-probability
    of exactly 1/3 at step=3 (exactly one cycle of length 3).

    RWSE step=3 diagonal should be 1/3 for all nodes.
    """
    try:
        from gnn_model.feature_engineering_v3 import compute_rwse_fast
    except ImportError:
        pytest.skip("feature_engineering_v3.py not available")

    n = 3
    edges = [(0, 1), (1, 2), (2, 0)]
    rwse = compute_rwse_fast(n, edges, steps=[2, 3], directed=True)

    assert rwse.shape == (3, 2), f"Expected shape (3, 2), got {rwse.shape}"

    # Step=2: 0→1→2 (no return), so diagonal should be 0
    for v in range(n):
        assert abs(rwse[v, 0]) < 1e-4, f"RWSE step=2 for v={v} should be ~0, got {rwse[v,0]}"

    # Step=3: 0→1→2→0 (return), diagonal should be ~1/3 * (1/3) = ...
    # Actually (P^3)_{vv}: each node has exactly one outgoing edge (det. walk)
    # so P is a permutation matrix for this triangle, P^3 = I → diagonal = 1
    for v in range(n):
        assert abs(rwse[v, 1] - 1.0) < 1e-4, \
            f"RWSE step=3 for v={v} should be ~1.0, got {rwse[v,1]}"


def test_compute_rwse_isolated_node():
    """Isolated nodes (no outgoing edges) should have all-zero RWSE."""
    try:
        from gnn_model.feature_engineering_v3 import compute_rwse_fast
    except ImportError:
        pytest.skip("feature_engineering_v3.py not available")

    n = 4
    edges = [(0, 1), (1, 0)]  # Only nodes 0 and 1 connected; 2, 3 isolated
    rwse = compute_rwse_fast(n, edges, steps=[2, 4], directed=True)

    # Isolated nodes 2, 3 should have zero RWSE
    for v in [2, 3]:
        for k_idx in range(2):
            assert abs(rwse[v, k_idx]) < 1e-6, \
                f"Isolated node v={v} RWSE step={[2,4][k_idx]} should be 0, got {rwse[v,k_idx]}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Test: compute_scc_features
# ═══════════════════════════════════════════════════════════════════════════════

def test_compute_scc_features_known_graph():
    """
    Graph: 0→1, 1→0 (SCC={0,1} size=2), 2→3 (SCCs={2},{3} size=1 each)

    Expected:
      - Nodes 0,1: in_nontrivial_scc=1, scc_size_raw=2
      - Nodes 2,3: in_nontrivial_scc=0, scc_size_raw=1
    """
    try:
        from gnn_model.feature_engineering_v3 import compute_scc_features
    except ImportError:
        pytest.skip("feature_engineering_v3.py not available")

    n = 4
    edges = [(0, 1), (1, 0), (2, 3)]
    scc_norm, in_nontriv, scc_raw = compute_scc_features(n, edges)

    assert in_nontriv[0] == 1.0, f"Node 0 should be in nontrivial SCC, got {in_nontriv[0]}"
    assert in_nontriv[1] == 1.0, f"Node 1 should be in nontrivial SCC, got {in_nontriv[1]}"
    assert in_nontriv[2] == 0.0, f"Node 2 should NOT be in nontrivial SCC, got {in_nontriv[2]}"
    assert in_nontriv[3] == 0.0, f"Node 3 should NOT be in nontrivial SCC, got {in_nontriv[3]}"

    assert scc_raw[0] == 2.0, f"SCC size for node 0 should be 2, got {scc_raw[0]}"
    assert scc_raw[2] == 1.0, f"SCC size for node 2 should be 1, got {scc_raw[2]}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Test: _pick_gnn_candidates_precision_first
# ═══════════════════════════════════════════════════════════════════════════════

def test_pick_gnn_candidates_returns_empty_when_below_threshold():
    """
    When all probabilities are below threshold (0.65), _pick_gnn_candidates_from_probs
    must return an EMPTY set — no topk-fallback.

    This is the critical invariant: never hard-fix uncertain predictions.
    """
    try:
        import torch
        sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
        from run_hybrid import _pick_gnn_candidates_from_probs
    except ImportError:
        pytest.skip("run_hybrid.py not importable (cpp_engine required)")

    # All probabilities below threshold=0.65
    probs = torch.tensor([0.1, 0.3, 0.4, 0.55, 0.64])
    candidates, mode = _pick_gnn_candidates_from_probs(probs, threshold=0.65)

    assert len(candidates) == 0, (
        f"Expected empty candidate set when all probs < 0.65, "
        f"got {candidates} (mode={mode}). "
        "The precision-first design must NOT use topk-fallback."
    )
    assert mode == "no_fix_insufficient_confidence", \
        f"Expected mode='no_fix_insufficient_confidence', got '{mode}'"


def test_pick_gnn_candidates_selects_high_confidence_only():
    """High-confidence vertices (>= 0.65) should be selected."""
    try:
        import torch
        from run_hybrid import _pick_gnn_candidates_from_probs
    except ImportError:
        pytest.skip("run_hybrid.py not importable (cpp_engine required)")

    probs = torch.tensor([0.9, 0.8, 0.1, 0.05, 0.75, 0.3])
    candidates, mode = _pick_gnn_candidates_from_probs(
        probs, threshold=0.65, min_fraction=0.0, max_fraction=1.0
    )

    expected = {0, 1, 4}   # vertices with prob >= 0.65
    assert candidates == expected, f"Expected {expected}, got {candidates}"


def test_pick_gnn_candidates_caps_at_max_fraction():
    """When too many pass threshold, cap at max_fraction of n."""
    try:
        import torch
        from run_hybrid import _pick_gnn_candidates_from_probs
    except ImportError:
        pytest.skip("run_hybrid.py not importable (cpp_engine required)")

    # 10 vertices all with high probability
    probs = torch.linspace(0.7, 0.95, 10)
    # max_fraction=0.2 → k=2 (ceil(10*0.2))
    candidates, mode = _pick_gnn_candidates_from_probs(
        probs, threshold=0.65, min_fraction=0.0, max_fraction=0.2
    )

    assert len(candidates) == 2, f"Expected 2 candidates (capped), got {len(candidates)}"
    assert mode == "high_conf_capped", f"Expected mode='high_conf_capped', got '{mode}'"

    # Should be the top-2 highest prob (indices 8 and 9)
    assert 9 in candidates and 8 in candidates, \
        f"Expected top-2 (indices 8,9), got {candidates}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Test: stratified_split
# ═══════════════════════════════════════════════════════════════════════════════

def test_stratified_split_respects_families():
    """Each family should have representatives in both train and val sets."""
    try:
        from gnn_model.train import stratified_split
    except ImportError:
        pytest.skip("train.py not available")

    # Create mock Data-like objects
    class MockData:
        def __init__(self, family):
            self.family = family

    dataset = (
        [MockData("er")] * 20 +
        [MockData("ba")] * 20 +
        [MockData("small_world")] * 10
    )

    train_set, val_set = stratified_split(dataset, val_ratio=0.2, seed=42)

    train_families = {g.family for g in train_set}
    val_families   = {g.family for g in val_set}

    assert train_families == {"er", "ba", "small_world"}, \
        f"All families should appear in train set, got {train_families}"
    assert val_families == {"er", "ba", "small_world"}, \
        f"All families should appear in val set, got {val_families}"

    # Verify proportions approximately
    assert abs(len(val_set) / len(dataset) - 0.2) < 0.05, \
        f"Val fraction should be ~20%, got {len(val_set)/len(dataset):.2%}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Test: warmup_cosine_scheduler
# ═══════════════════════════════════════════════════════════════════════════════

def test_warmup_cosine_scheduler_monotonic_warmup():
    """LR should increase monotonically during warmup phase."""
    try:
        import torch
        from gnn_model.train import get_warmup_cosine_scheduler
    except ImportError:
        pytest.skip("train.py not available")

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=20)

    lrs = []
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    # Each LR in warmup should be >= previous (monotonic increase)
    for i in range(1, len(lrs)):
        assert lrs[i] >= lrs[i-1] - 1e-7, \
            f"LR should increase during warmup, but got {lrs}"


if __name__ == "__main__":
    print("Running unit tests for GNN-KMA components...")
    print()

    tests = [
        test_asymmetric_fvs_loss_fp_vs_fn,
        test_asymmetric_fvs_loss_correct_predictions_near_zero,
        test_compute_topk_precision_perfect,
        test_compute_topk_precision_zero,
        test_compute_rwse_on_triangle,
        test_compute_rwse_isolated_node,
        test_compute_scc_features_known_graph,
        test_pick_gnn_candidates_returns_empty_when_below_threshold,
        test_pick_gnn_candidates_selects_high_confidence_only,
        test_pick_gnn_candidates_caps_at_max_fraction,
        test_stratified_split_respects_families,
        test_warmup_cosine_scheduler_monotonic_warmup,
    ]

    passed = failed = skipped = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ {test_fn.__name__}")
            passed += 1
        except pytest.skip.Exception as e:
            print(f"  ⊘ {test_fn.__name__} [SKIP: {e}]")
            skipped += 1
        except Exception as e:
            print(f"  ✗ {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n  Passed: {passed}  Skipped: {skipped}  Failed: {failed}")
    if failed > 0:
        sys.exit(1)
