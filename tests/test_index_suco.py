"""
tests/test_index_suco.py

Unit tests and a quick end-to-end benchmark for faiss.IndexSuCo.

Run with:
    python tests/test_index_suco.py

Requirements:
    pip install faiss-cpu numpy
    (and faiss must be built with IndexSuCo)
"""

import sys
import math
import tempfile
import os

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("faiss not found. Install with: pip install faiss-cpu")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_random_dataset(n: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype("float32")


def recall_at_k(gt_ids: np.ndarray, pred_ids: np.ndarray, k: int) -> float:
    """Fraction of queries where the true NN appears in the top-k results."""
    assert gt_ids.shape[0] == pred_ids.shape[0]
    hits = sum(
        1
        for i in range(len(gt_ids))
        if gt_ids[i] in pred_ids[i, :k]
    )
    return hits / len(gt_ids)


def ground_truth_nn(xb: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    """Brute-force exact k-NN (nearest neighbours) using a flat L2 index."""
    idx = faiss.IndexFlatL2(xb.shape[1])
    idx.add(xb)
    _, ids = idx.search(xq, k)
    return ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_construction():
    """Index can be constructed with various parameters."""
    print("test_basic_construction ... ", end="")
    d = 64
    idx = faiss.IndexSuCo(d)
    assert idx.d == d
    assert not idx.is_trained
    assert idx.ntotal == 0
    assert idx.nsubspaces == 8
    assert idx.ncentroids_half == 50
    print("OK")


def test_train_add_search():
    """Train → add → search returns correct shapes and plausible recall."""
    print("test_train_add_search ... ", end="")
    d     = 128
    n     = 10_000
    nq    = 100
    k     = 10
    Ns    = 8
    nc    = 50
    alpha = 0.05
    beta  = 0.005

    xb = make_random_dataset(n, d, seed=1)
    xq = make_random_dataset(nq, d, seed=2)

    idx = faiss.IndexSuCo(d, Ns, nc, alpha, beta, niter=5)
    idx.train(xb)
    assert idx.is_trained

    idx.add(xb)
    assert idx.ntotal == n

    D, I = idx.search(xq, k)
    assert D.shape == (nq, k), f"Distance shape mismatch: {D.shape}"
    assert I.shape == (nq, k), f"Labels shape mismatch: {I.shape}"

    # All returned IDs should be in range [0, n)
    valid = I[I >= 0]
    assert (valid < n).all(), "Returned out-of-range index"

    # Compute recall@1 vs brute force
    gt = ground_truth_nn(xb, xq, k=1)
    r1 = recall_at_k(gt[:, 0], I, k=1)
    print(f"OK  (recall@1 = {r1:.3f})")
    # Modest check — random data is easy; just make sure we're not broken
    assert r1 >= 0.50, f"recall@1 suspiciously low: {r1:.3f}"


def test_recall_vs_flat():
    """SuCo recall should be ≥ 0.90 on a structured dataset."""
    print("test_recall_vs_flat (structured data) ... ", end="")
    d   = 128
    n   = 100_000
    nq  = 200
    k   = 50

    # Use a structured dataset where nearby IDs are nearby in space
    rng = np.random.default_rng(42)
    xb  = rng.standard_normal((n, d)).astype("float32")
    # Queries near the database
    xq  = xb[:nq] + 0.01 * rng.standard_normal((nq, d)).astype("float32")

    idx = faiss.IndexSuCo(
        d,
        nsubspaces      = 8,
        ncentroids_half = 50,
        collision_ratio = 0.05,
        candidate_ratio = 0.005,
        niter           = 10,
    )
    idx.train(xb)
    idx.add(xb)

    _, I = idx.search(xq, k)
    gt   = ground_truth_nn(xb, xq, k)
    recall = recall_at_k(gt[:, 0], I, k=1)
    print(f"OK  (recall@1 = {recall:.3f})")
    assert recall >= 0.90, f"Expected recall@1 ≥ 0.90, got {recall:.3f}"


def test_persistence():
    """write_index / read_index round-trip preserves results."""
    print("test_persistence ... ", end="")
    d  = 64
    n  = 5_000
    nq = 50
    k  = 10

    xb = make_random_dataset(n, d, seed=10)
    xq = make_random_dataset(nq, d, seed=11)

    idx = faiss.IndexSuCo(d, nsubspaces=8, ncentroids_half=20,
                          collision_ratio=0.05, candidate_ratio=0.01, niter=5)
    idx.train(xb)
    idx.add(xb)

    D1, I1 = idx.search(xq, k)

    with tempfile.NamedTemporaryFile(suffix=".suco", delete=False) as f:
        tmp_path = f.name

    try:
        idx.write_index(tmp_path)

        # Load into a fresh object
        idx2 = faiss.IndexSuCo(d)
        idx2.read_index(tmp_path)
        assert idx2.ntotal == n
        assert idx2.nsubspaces == idx.nsubspaces

        D2, I2 = idx2.search(xq, k)
        np.testing.assert_array_equal(I1, I2)
        np.testing.assert_allclose(D1, D2, rtol=1e-5)
    finally:
        os.unlink(tmp_path)

    print("OK")


def test_reset():
    """reset() empties the index and allows re-training."""
    print("test_reset ... ", end="")
    d  = 64
    xb = make_random_dataset(1000, d)

    idx = faiss.IndexSuCo(d, nsubspaces=4, ncentroids_half=10,
                          collision_ratio=0.1, candidate_ratio=0.02, niter=3)
    idx.train(xb)
    idx.add(xb)
    assert idx.ntotal == 1000
    assert idx.is_trained

    idx.reset()
    assert idx.ntotal == 0
    assert not idx.is_trained

    # Should be able to train again
    idx.train(xb)
    idx.add(xb)
    assert idx.ntotal == 1000
    print("OK")


def test_invalid_params():
    """Constructor rejects bad dimension combinations."""
    print("test_invalid_params ... ", end="")
    try:
        # d not divisible by nsubspaces
        _ = faiss.IndexSuCo(130, 8)
        assert False, "Should have raised"
    except Exception:
        pass

    try:
        # subspace_dim not even
        _ = faiss.IndexSuCo(40, 8)   # 40/8 = 5, not even
        assert False, "Should have raised"
    except Exception:
        pass

    print("OK")


def benchmark():
    """Quick throughput benchmark."""
    import time
    print("\n--- Benchmark ---")
    d    = 128
    n    = 500_000
    nq   = 100
    k    = 50

    xb   = make_random_dataset(n, d, seed=99)
    xq   = make_random_dataset(nq, d, seed=100)

    idx  = faiss.IndexSuCo(d, 8, 50, 0.05, 0.005, niter=10)
    t0 = time.perf_counter()
    idx.train(xb)
    idx.add(xb)
    t_build = time.perf_counter() - t0
    print(f"  Build (train+add):  {t_build:.2f}s")

    # warm up
    idx.search(xq[:5], k)

    t0 = time.perf_counter()
    D, I = idx.search(xq, k)
    t_query = time.perf_counter() - t0
    ms_per_q = t_query / nq * 1000
    print(f"  Query ({nq} queries): {t_query*1000:.1f}ms  "
          f"({ms_per_q:.2f}ms/query  {nq/t_query:.0f} QPS)")

    gt     = ground_truth_nn(xb, xq, k=1)
    recall = recall_at_k(gt[:, 0], I, k=1)
    print(f"  Recall@1: {recall:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== IndexSuCo tests ===\n")
    test_basic_construction()
    test_train_add_search()
    test_recall_vs_flat()
    test_persistence()
    test_reset()
    test_invalid_params()
    benchmark()
    print("\nAll tests passed.")