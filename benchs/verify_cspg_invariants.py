#!/usr/bin/env python3
"""
Verify CSPG partition/routing invariants.

Checks:
  1. Every partition shares the same routing prefix (first `num_routing`
     local IDs reference identical global IDs across partitions).
  2. Non-routing vectors are assigned to exactly one partition.
  3. Coverage: every global ID in [0, n) is reachable through some partition.
  4. Routing vectors are RANDOMLY sampled (not a deterministic prefix).
     The mean of the sampled global IDs should be ~ n/2 with small relative
     error, and they should not be the first `num_routing` integers.
  5. I/O round-trip: save then load an index; search results must match.
  6. Query sanity: recall@10 must be non-trivial on a tiny synthetic set.
"""

import numpy as np
import sys
import tempfile
import os

import faiss


def fail(msg):
    print(f"  FAIL: {msg}")
    sys.exit(1)


def ok(msg):
    print(f"  OK:   {msg}")


def build_small_index(n=5000, d=32, m=2, lam=0.5, M=16, efC=64, seed=0):
    rng = np.random.default_rng(seed)
    xb = rng.standard_normal((n, d), dtype=np.float32)

    idx = faiss.IndexCSPG(d, M, m, float(lam))
    idx.efConstruction = efC
    idx.add(xb)
    return idx, xb


def collect_partitions(idx):
    """Return list of np.arrays, one per partition, of global IDs."""
    parts = []
    for p in range(idx.num_partitions):
        row = idx.refunction.at(p)
        arr = np.array([row.at(i) for i in range(row.size())], dtype=np.int64)
        parts.append(arr)
    return parts


def check_routing_prefix(idx, parts):
    num_routing = int(idx.num_routing)
    if num_routing <= 0:
        fail(f"num_routing={num_routing} should be positive")

    ref0 = parts[0][:num_routing]
    for p in range(1, len(parts)):
        if not np.array_equal(parts[p][:num_routing], ref0):
            fail(f"partition {p} routing prefix differs from partition 0")
    ok(f"all {len(parts)} partitions share identical routing prefix "
       f"(num_routing={num_routing})")
    return ref0


def check_unique_assignment(idx, parts, routing_ids):
    n = int(idx.ntotal)
    routing_set = set(routing_ids.tolist())

    seen_count = np.zeros(n, dtype=np.int32)
    for p_ids in parts:
        for gid in p_ids:
            seen_count[gid] += 1

    # Routing IDs should appear in every partition.
    bad_routing = [gid for gid in routing_set if seen_count[gid] != len(parts)]
    if bad_routing:
        fail(f"routing IDs not present in all partitions: "
             f"{bad_routing[:5]}... ({len(bad_routing)} total)")
    ok(f"every routing ID appears in all {len(parts)} partitions")

    # Non-routing IDs: exactly one partition.
    non_routing_mask = np.ones(n, dtype=bool)
    non_routing_mask[list(routing_set)] = False
    counts_non_routing = seen_count[non_routing_mask]
    if (counts_non_routing != 1).any():
        bad = np.where(counts_non_routing != 1)[0][:5]
        fail(f"non-routing IDs with count!=1: indices {bad.tolist()}, "
             f"counts {counts_non_routing[bad].tolist()}")
    ok(f"every non-routing ID assigned to exactly one partition "
       f"({int(non_routing_mask.sum())} vectors)")


def check_coverage(idx, parts):
    n = int(idx.ntotal)
    covered = np.zeros(n, dtype=bool)
    for p_ids in parts:
        covered[p_ids] = True
    if not covered.all():
        missing = np.where(~covered)[0]
        fail(f"{len(missing)} global IDs are uncovered (e.g. {missing[:5]})")
    ok(f"all n={n} global IDs covered by partitions")


def check_random_sampling(idx, routing_ids):
    n = int(idx.ntotal)
    num_routing = len(routing_ids)

    # Property 1: mean should be ~ (n-1)/2 if truly random.
    mean_gid = float(routing_ids.mean())
    expected_mean = (n - 1) / 2.0
    rel_err = abs(mean_gid - expected_mean) / expected_mean
    if rel_err > 0.05:
        fail(f"routing-ID mean {mean_gid:.1f} differs from expected "
             f"{expected_mean:.1f} by {rel_err * 100:.1f}% (>5%); "
             f"possibly deterministic prefix?")
    ok(f"routing-ID mean={mean_gid:.1f} matches expected~{expected_mean:.1f} "
       f"(rel err {rel_err * 100:.2f}%)")

    # Property 2: not the first num_routing IDs.
    prefix = np.arange(num_routing, dtype=np.int64)
    if np.array_equal(routing_ids, prefix):
        fail("routing IDs equal [0..num_routing) — deterministic prefix bug")
    # Max should exceed num_routing-1 for sure.
    if routing_ids.max() < n - 1 - num_routing:
        fail(f"routing max={routing_ids.max()} too small; range looks "
             f"clustered at the start")
    ok(f"routing IDs span the full range (max={int(routing_ids.max())}, "
       f"n-1={n - 1})")


def check_io_roundtrip(idx, xb, k=10):
    nq = 50
    xq = xb[:nq].copy()

    D1, I1 = idx.search(xq, k)

    with tempfile.NamedTemporaryFile(suffix=".idx", delete=False) as f:
        path = f.name
    try:
        faiss.write_index(idx, path)
        idx2 = faiss.read_index(path)
        D2, I2 = idx2.search(xq, k)
    finally:
        os.unlink(path)

    if not np.array_equal(I1, I2):
        fail(f"I/O round-trip changed neighbor IDs "
             f"({(I1 != I2).sum()} mismatches)")
    if not np.allclose(D1, D2, rtol=1e-5, atol=1e-5):
        fail("I/O round-trip changed distances")
    ok(f"I/O round-trip preserves search results exactly (nq={nq}, k={k})")


def check_search_sanity(idx, xb, k=10):
    """Recall@k against ground truth on a random query subset."""
    nq = 200
    rng = np.random.default_rng(7)
    qidx = rng.choice(xb.shape[0], size=nq, replace=False)
    xq = xb[qidx].copy()

    # Ground truth via brute force.
    flat = faiss.IndexFlatL2(xb.shape[1])
    flat.add(xb)
    _, I_gt = flat.search(xq, k)

    _, I_cspg = idx.search(xq, k)

    hits = 0
    for i in range(nq):
        hits += len(set(I_cspg[i].tolist()) & set(I_gt[i].tolist()))
    recall = hits / (nq * k)
    if recall < 0.80:
        fail(f"recall@{k}={recall:.3f} below sanity threshold (0.80)")
    ok(f"recall@{k}={recall:.3f} on synthetic data (sanity threshold 0.80)")


def main():
    print("========================================================")
    print("CSPG invariant verification")
    print("========================================================")
    print()

    for m in [2, 3]:
        for lam in [0.3, 0.5]:
            print(f"--- m={m}, lambda={lam} ---")
            idx, xb = build_small_index(m=m, lam=lam)
            parts = collect_partitions(idx)

            print(f"  num_partitions={idx.num_partitions}, "
                  f"num_routing={idx.num_routing}, ntotal={idx.ntotal}")
            for p, ids in enumerate(parts):
                print(f"  partition {p}: size={len(ids)}")

            routing_ids = check_routing_prefix(idx, parts)
            check_unique_assignment(idx, parts, routing_ids)
            check_coverage(idx, parts)
            check_random_sampling(idx, routing_ids)
            check_io_roundtrip(idx, xb)
            check_search_sanity(idx, xb)
            print()

    print("========================================================")
    print("ALL INVARIANT CHECKS PASSED")
    print("========================================================")


if __name__ == "__main__":
    main()
