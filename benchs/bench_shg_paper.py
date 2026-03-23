#!/usr/bin/env python3
"""
benchs/bench_shg_paper.py

Comprehensive benchmark suite reproducing experiments from:
  "Accelerating Approximate Nearest Neighbor Search in Hierarchical Graphs:
   Efficient Level Navigation with Shortcuts" (PVLDB 18(10), 2025)

Benchmarks:
  1. Construction time & memory cost
  2. Recall vs time (k=20, k=50)
  3. Ablation: SHG with/without shortcuts, with/without LB pruning
  4. Robustness with unseen query vectors

Indices tested:
  - IndexSHG (SHG implementation)
  - IndexHNSWFlat (baseline HNSW)
  - IndexIVFFlat
  - IndexIVFPQ
  - IndexHNSWFlatPanorama

Datasets: OpenAI (d=1536), Enron (d=1369), GIST1M (d=960),
          Msong (d=420), UQ-V (d=256), MsTuring10M (d=100)

Usage:
  python benchs/bench_shg_paper.py --dataset gist1m --benchmark all
  python benchs/bench_shg_paper.py --dataset all --benchmark all
  python benchs/bench_shg_paper.py --dataset enron --benchmark recall_k20

All indices (including IndexSHG) are saved to --index-dir via
faiss.write_index / faiss.read_index for reuse across runs.
"""

import argparse
import json
import os
import struct
import sys
import time
import traceback

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("Cannot import faiss. Build FAISS with IndexSHG support first.")


# ---------------------------------------------------------------------------
# Dataset I/O helpers
# ---------------------------------------------------------------------------

def read_fvecs(fname):
    """Read .fvecs file -> np.ndarray of float32."""
    with open(fname, "rb") as f:
        d = struct.unpack("i", f.read(4))[0]
        f.seek(0)
        n = os.path.getsize(fname) // (4 + d * 4)
        data = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            dim = struct.unpack("i", f.read(4))[0]
            assert dim == d, f"Dim mismatch at row {i}: {dim} vs {d}"
            data[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return data


def read_ivecs(fname):
    """Read .ivecs file -> np.ndarray of int32."""
    with open(fname, "rb") as f:
        d = struct.unpack("i", f.read(4))[0]
        f.seek(0)
        n = os.path.getsize(fname) // (4 + d * 4)
        data = np.zeros((n, d), dtype=np.int32)
        for i in range(n):
            dim = struct.unpack("i", f.read(4))[0]
            assert dim == d
            data[i] = np.frombuffer(f.read(d * 4), dtype=np.int32)
    return data


def read_fbin(fname, dtype=np.float32):
    """Read .fbin file: [n, d] int32 header, then n*d dtype values.
    For cropped files where the header n exceeds actual data, compute n
    from the file size."""
    with open(fname, "rb") as f:
        n, d = struct.unpack("ii", f.read(8))
        file_size = os.path.getsize(fname)
        actual_n = (file_size - 8) // (d * np.dtype(dtype).itemsize)
        if actual_n < n:
            n = actual_n
        data = np.fromfile(f, dtype=dtype, count=n * d).reshape(n, d)
    return data


def read_ibin(fname):
    """Read .bin ground truth: [n, k] int32 header, then n*k int32."""
    with open(fname, "rb") as f:
        n, k = struct.unpack("ii", f.read(8))
        file_size = os.path.getsize(fname)
        actual_n = (file_size - 8) // (k * 4)
        if actual_n < n:
            n = actual_n
        data = np.fromfile(f, dtype=np.int32, count=n * k).reshape(n, k)
    return data


def read_enron_data(fname):
    """Read enron.data_new: 3-int32 header (version, n, d), then n*d float32."""
    with open(fname, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=3)
        _version, n, d = int(header[0]), int(header[1]), int(header[2])
        data = np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)
    return data


def read_openai_parquet(data_dir, max_vectors=1_000_000):
    """Read OpenAI embeddings from parquet files."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow required for OpenAI dataset: pip install pyarrow")

    parquet_dir = os.path.join(data_dir, "openai1m")
    files = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])

    all_vecs = []
    total = 0
    for fname in files:
        if total >= max_vectors:
            break
        table = pq.read_table(os.path.join(parquet_dir, fname))
        for col_name in ["emb", "embedding", "vector", "values"]:
            if col_name in table.column_names:
                break
        else:
            col_name = table.column_names[-1]

        col = table[col_name]
        for row in col:
            if total >= max_vectors:
                break
            vec = np.array(row.as_py(), dtype=np.float32)
            all_vecs.append(vec)
            total += 1

    return np.array(all_vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(name, data_dir):
    """Returns (xb, xq, gt)."""
    name = name.lower()

    if name == "openai":
        gt_cache = os.path.join(data_dir, "openai1m", "openai_gt100.npy")
        xb_cache = os.path.join(data_dir, "openai1m", "openai_xb.npy")
        xq_cache = os.path.join(data_dir, "openai1m", "openai_xq.npy")

        if os.path.exists(xb_cache) and os.path.exists(xq_cache):
            xb = np.load(xb_cache)
            xq = np.load(xq_cache)
        else:
            all_vecs = read_openai_parquet(data_dir, max_vectors=1_000_000)
            nq = 10_000
            xq = all_vecs[-nq:].copy()
            xb = all_vecs[:-nq].copy()
            del all_vecs
            np.save(xb_cache, xb)
            np.save(xq_cache, xq)

        if os.path.exists(gt_cache):
            gt = np.load(gt_cache)
        else:
            gt = compute_ground_truth(xb, xq, k=100)
            np.save(gt_cache, gt)
        return xb, xq, gt

    elif name == "enron":
        p = os.path.join(data_dir, "enron")
        xb = read_enron_data(os.path.join(p, "enron.data_new"))
        xq = read_fvecs(os.path.join(p, "enron_query.fvecs"))
        gt = read_ivecs(os.path.join(p, "enron_groundtruth.ivecs"))
        return xb, xq, gt

    elif name == "gist1m":
        p = os.path.join(data_dir, "gist1M")
        xb = read_fvecs(os.path.join(p, "gist_base.fvecs"))
        xq = read_fvecs(os.path.join(p, "gist_query.fvecs"))
        gt = read_ivecs(os.path.join(p, "gist_groundtruth.ivecs"))
        return xb, xq, gt

    elif name == "msong":
        p = os.path.join(data_dir, "msong")
        xb = read_fvecs(os.path.join(p, "msong_base.fvecs"))
        xq = read_fvecs(os.path.join(p, "msong_query.fvecs"))
        gt = read_ivecs(os.path.join(p, "msong_groundtruth.ivecs"))
        return xb, xq, gt

    elif name == "uqv":
        p = os.path.join(data_dir, "uqv")
        xb = read_fvecs(os.path.join(p, "uqv_base.fvecs"))
        xq = read_fvecs(os.path.join(p, "uqv_query.fvecs"))
        gt = read_ivecs(os.path.join(p, "uqv_groundtruth.ivecs"))
        return xb, xq, gt

    elif name == "msturing10m":
        p = os.path.join(data_dir, "msturing10m")
        xb = read_fbin(os.path.join(p, "base1b.fbin.crop_nb_10000000"))
        xq = read_fbin(os.path.join(p, "testQuery10K.fbin"))
        gt = read_ibin(os.path.join(p, "msturing-gt-10M"))
        # GT may have more rows than queries; truncate to match
        if gt.shape[0] > xq.shape[0]:
            gt = gt[: xq.shape[0]]
        return xb, xq, gt

    else:
        raise ValueError(f"Unknown dataset: {name}")


def compute_ground_truth(xb, xq, k=100):
    """Compute exact k-NN ground truth using brute force.
    Uses faiss.knn to avoid duplicating xb into an IndexFlatL2."""
    print(f"  Computing ground truth (n={xb.shape[0]}, nq={xq.shape[0]}, k={k})...")
    # faiss.knn computes distances directly without copying xb into an index
    D, I = faiss.knn(xq, xb, k, metric=faiss.METRIC_L2)
    return I


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

# Paper uses M=48, efConstruction=80 for HNSW and SHG
DEFAULT_M = 48
DEFAULT_EFC = 80


def build_index_shg(xb, d):
    idx = faiss.IndexSHG(d, DEFAULT_M)
    idx.hnsw.efConstruction = DEFAULT_EFC
    t0 = time.time()
    idx.add(xb)
    t_add = time.time() - t0
    t1 = time.time()
    idx.build_shortcut()
    t_sc = time.time() - t1
    t_total = t_add + t_sc
    print(f"  SHG: add={t_add:.2f}s, shortcut={t_sc:.2f}s, total={t_total:.2f}s")
    return idx, t_total


def build_index_hnsw(xb, d):
    idx = faiss.IndexHNSWFlat(d, DEFAULT_M)
    idx.hnsw.efConstruction = DEFAULT_EFC
    t0 = time.time()
    idx.add(xb)
    t_total = time.time() - t0
    print(f"  HNSW: build={t_total:.2f}s")
    return idx, t_total


def build_index_panorama(xb, d):
    idx = faiss.IndexHNSWFlatPanorama(d, DEFAULT_M, faiss.METRIC_L2)
    idx.hnsw.efConstruction = DEFAULT_EFC
    t0 = time.time()
    idx.add(xb)
    t_total = time.time() - t0
    print(f"  Panorama: build={t_total:.2f}s")
    return idx, t_total


def build_index_ivfflat(xb, d):
    n = xb.shape[0]
    nlist = int(np.sqrt(n))
    quantizer = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFFlat(quantizer, d, nlist)
    t0 = time.time()
    idx.train(xb)
    idx.add(xb)
    t_total = time.time() - t0
    print(f"  IVFFlat (nlist={nlist}): build={t_total:.2f}s")
    return idx, t_total


def build_index_ivfpq(xb, d):
    n = xb.shape[0]
    nlist = int(np.sqrt(n))
    # Choose m_pq as a divisor of d (target ~d/8 sub-vectors)
    target = max(1, d // 8)
    # Find all divisors of d, pick the one closest to target
    divisors = [i for i in range(1, d + 1) if d % i == 0]
    m_pq = min(divisors, key=lambda x: abs(x - target))
    quantizer = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFPQ(quantizer, d, nlist, m_pq, 8)
    t0 = time.time()
    idx.train(xb)
    idx.add(xb)
    t_total = time.time() - t0
    print(f"  IVFPQ (nlist={nlist}, m={m_pq}): build={t_total:.2f}s")
    return idx, t_total


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def index_size_bytes(idx):
    """Estimate index memory via serialization size."""
    try:
        buf = faiss.serialize_index(idx)
        return len(buf)
    except Exception:
        return -1


def compute_recall_at_k(I, gt, k):
    """Recall@k: |A ∩ A*| / k, averaged over queries."""
    nq = I.shape[0]
    k_gt = min(k, gt.shape[1])
    k_ret = min(k, I.shape[1])
    hits = 0
    for i in range(nq):
        hits += len(set(gt[i, :k_gt].tolist()) & set(I[i, :k_ret].tolist()))
    return hits / (nq * k_gt)


def per_query_recall(I, gt, k):
    """Per-query recall array."""
    nq = I.shape[0]
    k_gt = min(k, gt.shape[1])
    k_ret = min(k, I.shape[1])
    recalls = np.zeros(nq)
    for i in range(nq):
        recalls[i] = len(set(gt[i, :k_gt].tolist()) & set(I[i, :k_ret].tolist())) / k_gt
    return recalls


def search_hnsw(idx, xq, k, efSearch):
    sp = faiss.SearchParametersHNSW()
    sp.efSearch = efSearch
    t0 = time.time()
    D, I = idx.search(xq, k, params=sp)
    return D, I, time.time() - t0


def search_shg(idx, xq, k, efSearch, use_shortcut=True, use_lb=True):
    sp = faiss.SearchParametersSHG()
    sp.efSearch = efSearch
    sp.use_shortcut = use_shortcut
    sp.use_lb_pruning = use_lb
    t0 = time.time()
    D, I = idx.search(xq, k, params=sp)
    return D, I, time.time() - t0


def search_ivf(idx, xq, k, nprobe):
    idx.nprobe = nprobe
    t0 = time.time()
    D, I = idx.search(xq, k)
    return D, I, time.time() - t0


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

EF_VALUES = [10, 16, 20, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 500]
NPROBE_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def sweep_hnsw(idx, label, xq, gt, k):
    """Sweep efSearch for HNSW-type index, return list of {efSearch, recall, ms_per_query}."""
    results = []
    for ef in EF_VALUES:
        if ef < k:
            continue
        try:
            D, I, t = search_hnsw(idx, xq, k, ef)
            r = compute_recall_at_k(I, gt, k)
            ms = t * 1000 / xq.shape[0]
            results.append({"efSearch": ef, "recall": round(r, 6), "ms_per_query": round(ms, 6)})
            print(f"  {label} ef={ef:4d}: recall={r:.4f}, time={ms:.4f} ms/q")
        except Exception as e:
            print(f"  {label} ef={ef}: ERROR - {e}")
    return results


def sweep_shg(idx, label, xq, gt, k, use_sc=True, use_lb=True):
    results = []
    for ef in EF_VALUES:
        if ef < k:
            continue
        try:
            D, I, t = search_shg(idx, xq, k, ef, use_sc, use_lb)
            r = compute_recall_at_k(I, gt, k)
            ms = t * 1000 / xq.shape[0]
            results.append({"efSearch": ef, "recall": round(r, 6), "ms_per_query": round(ms, 6)})
            print(f"  {label} ef={ef:4d}: recall={r:.4f}, time={ms:.4f} ms/q")
        except Exception as e:
            print(f"  {label} ef={ef}: ERROR - {e}")
    return results


def sweep_ivf(idx, label, xq, gt, k):
    results = []
    for nprobe in NPROBE_VALUES:
        try:
            D, I, t = search_ivf(idx, xq, k, nprobe)
            r = compute_recall_at_k(I, gt, k)
            ms = t * 1000 / xq.shape[0]
            results.append({"nprobe": nprobe, "recall": round(r, 6), "ms_per_query": round(ms, 6)})
            print(f"  {label} nprobe={nprobe:4d}: recall={r:.4f}, time={ms:.4f} ms/q")
        except Exception as e:
            print(f"  {label} nprobe={nprobe}: ERROR - {e}")
    return results


# ---------------------------------------------------------------------------
# Run all benchmarks for one dataset
# ---------------------------------------------------------------------------

ALL_DATASETS = ["openai", "enron", "gist1m", "msong", "uqv", "msturing10m"]
ALL_BENCHMARKS = ["construction", "recall_k20", "recall_k50", "robustness"]


def run_benchmarks(dataset_name, benchmarks, data_dir, index_dir, output_dir):
    import gc

    print(f"\n{'#'*70}")
    print(f"# Dataset: {dataset_name.upper()}")
    print(f"{'#'*70}")

    # Load data
    print(f"\nLoading dataset {dataset_name}...")
    t0 = time.time()
    xb, xq, gt = load_dataset(dataset_name, data_dir)
    print(f"  Loaded in {time.time()-t0:.1f}s: xb={xb.shape}, xq={xq.shape}, gt={gt.shape}")

    # Recompute GT if precomputed IDs reference vectors beyond the base size
    if gt.max() >= xb.shape[0]:
        print(f"  GT has IDs up to {gt.max()} but base has only {xb.shape[0]} vectors — recomputing GT...")
        gt = compute_ground_truth(xb, xq, k=gt.shape[1])
    d = xb.shape[1]
    n = int(xb.shape[0])

    all_results = {
        "dataset": dataset_name,
        "n": n,
        "d": d,
        "nq": int(xq.shape[0]),
    }

    # -----------------------------------------------------------------------
    # Process indices ONE AT A TIME to stay within memory limits.
    # Each index is loaded (or built), benchmarked, then released.
    # -----------------------------------------------------------------------
    builders = [
        ("SHG",       build_index_shg),
        ("HNSW",      build_index_hnsw),
        ("Panorama",  build_index_panorama),
        ("IVFFlat",   build_index_ivfflat),
        ("IVFPQ",     build_index_ivfpq),
    ]

    # Prepare robustness queries once (shared across all indices).
    # Must be done while xb is still in memory (needed for noise generation
    # and GT computation on first run).
    unseen_q = None
    gt_unseen = None
    if "robustness" in benchmarks:
        rob_q_cache = os.path.join(index_dir, f"{dataset_name}_robustness_q.npy")
        rob_gt_cache = os.path.join(index_dir, f"{dataset_name}_robustness_gt.npy")

        if os.path.exists(rob_q_cache) and os.path.exists(rob_gt_cache):
            unseen_q = np.load(rob_q_cache)
            gt_unseen = np.load(rob_gt_cache)
            print(f"  Loaded robustness queries from cache")
        else:
            n_unseen = min(1000, xq.shape[0])
            rng = np.random.RandomState(42)
            sample_ids = rng.choice(xb.shape[0], size=n_unseen, replace=False)
            noise_scale = float(np.std(xb)) * 0.1
            unseen_q = xb[sample_ids].copy() + \
                rng.randn(n_unseen, d).astype(np.float32) * noise_scale
            gt_unseen = compute_ground_truth(xb, unseen_q, k=50)
            np.save(rob_q_cache, unseen_q)
            np.save(rob_gt_cache, gt_unseen)

    # Check which indices still need building (need xb for those)
    needs_build = False
    for name, _ in builders:
        idx_path = os.path.join(index_dir, f"{dataset_name}_{name.lower()}.idx")
        if not os.path.exists(idx_path):
            needs_build = True
            break

    # Free xb if all indices are already built — saves ~6 GB for OpenAI
    if not needs_build:
        print(f"  All indices cached, freeing base vectors ({xb.nbytes / 1e9:.1f} GB)")
        del xb
        gc.collect()
        xb = None

    # Load previous results to carry forward build times for cached indices
    prev_construction = {}
    prev_results_path = os.path.join(output_dir, f"results_{dataset_name}.json")
    if os.path.exists(prev_results_path):
        try:
            with open(prev_results_path) as f:
                prev_construction = json.load(f).get("construction", {})
        except Exception:
            pass

    # Result accumulators
    construction_results = {}
    rt_k20_results = {}
    rt_k50_results = {}
    rob_results = {}

    for name, builder in builders:
        idx_path = os.path.join(index_dir, f"{dataset_name}_{name.lower()}.idx")
        idx = None
        build_time = -1
        mem_mb = -1

        # Try loading a previously saved index
        if os.path.exists(idx_path):
            print(f"\n--- Loading {name} from {idx_path} ---")
            try:
                idx = faiss.read_index(idx_path)
                mem = index_size_bytes(idx)
                mem_mb = mem / (1024 * 1024) if mem > 0 else -1
                print(f"  {name}: loaded, memory={mem_mb:.2f}MB")
            except Exception as e:
                print(f"  Failed to load, rebuilding: {e}")
                idx = None

        if idx is None:
            if xb is None:
                print(f"  {name}: skipping (cached file corrupt/missing, "
                      f"re-run to rebuild)")
                construction_results[name] = {
                    "build_time_s": -1, "memory_mb": -1}
                continue
            print(f"\n--- Building {name} ---")
            try:
                idx, build_time = builder(xb, d)
                mem = index_size_bytes(idx)
                mem_mb = mem / (1024 * 1024) if mem > 0 else -1
                print(f"  {name}: time={build_time:.2f}s, memory={mem_mb:.2f}MB")

                # Save index to disk for reuse
                try:
                    faiss.write_index(idx, idx_path)
                    print(f"  Saved to {idx_path}")
                except Exception as e:
                    print(f"  Could not save: {e}")
            except Exception as e:
                print(f"  {name}: FAILED - {e}")
                traceback.print_exc()

        if idx is None:
            construction_results[name] = {
                "build_time_s": -1, "memory_mb": -1}
            continue

        # -- Construction stats --
        # Carry forward build_time from previous run if index was loaded from cache
        recorded_build_time = round(build_time, 2) if build_time >= 0 else -1
        if recorded_build_time < 0 and name in prev_construction:
            prev_bt = prev_construction[name].get("build_time_s", -1)
            if prev_bt >= 0:
                recorded_build_time = prev_bt
        construction_results[name] = {
            "build_time_s": recorded_build_time,
            "memory_mb": round(mem_mb, 2) if mem_mb >= 0 else -1,
        }

        # -- Recall vs time k=20 --
        if "recall_k20" in benchmarks:
            k = 20
            if name == "SHG":
                rt_k20_results["SHG"] = sweep_shg(
                    idx, "SHG", xq, gt, k)
                rt_k20_results["SHG-no-shortcut"] = sweep_shg(
                    idx, "SHG-no-shortcut", xq, gt, k,
                    use_sc=False, use_lb=True)
                rt_k20_results["SHG-no-lb"] = sweep_shg(
                    idx, "SHG-no-lb", xq, gt, k,
                    use_sc=True, use_lb=False)
                rt_k20_results["SHG-no-both"] = sweep_shg(
                    idx, "SHG-no-both", xq, gt, k,
                    use_sc=False, use_lb=False)
            elif name in ("HNSW", "Panorama"):
                rt_k20_results[name] = sweep_hnsw(idx, name, xq, gt, k)
            elif name in ("IVFFlat", "IVFPQ"):
                rt_k20_results[name] = sweep_ivf(idx, name, xq, gt, k)

        # -- Recall vs time k=50 --
        if "recall_k50" in benchmarks:
            k = 50
            if name == "SHG":
                rt_k50_results["SHG"] = sweep_shg(
                    idx, "SHG", xq, gt, k)
            elif name in ("HNSW", "Panorama"):
                rt_k50_results[name] = sweep_hnsw(idx, name, xq, gt, k)
            elif name in ("IVFFlat", "IVFPQ"):
                rt_k50_results[name] = sweep_ivf(idx, name, xq, gt, k)

        # -- Robustness --
        if "robustness" in benchmarks:
            k_rob = 20
            ef_test = max(k_rob * 4, 100)
            nprobe_test = 64
            try:
                if name == "SHG":
                    D, I, t = search_shg(idx, unseen_q, k_rob, ef_test)
                elif name in ("HNSW", "Panorama"):
                    D, I, t = search_hnsw(idx, unseen_q, k_rob, ef_test)
                elif name in ("IVFFlat", "IVFPQ"):
                    D, I, t = search_ivf(idx, unseen_q, k_rob, nprobe_test)
                else:
                    D, I, t = None, None, None

                if I is not None:
                    pqr = per_query_recall(I, gt_unseen, k_rob)
                    rob_results[name] = {
                        "mean_recall": round(float(pqr.mean()), 4),
                        "median_recall": round(float(np.median(pqr)), 4),
                        "min_recall": round(float(pqr.min()), 4),
                        "max_recall": round(float(pqr.max()), 4),
                        "q25_recall": round(float(np.percentile(pqr, 25)), 4),
                        "q75_recall": round(float(np.percentile(pqr, 75)), 4),
                        "ms_per_query": round(t * 1000 / unseen_q.shape[0], 4),
                    }
            except Exception as e:
                print(f"  {name} robustness: ERROR - {e}")
                traceback.print_exc()

        # Release index memory before loading the next one
        del idx
        gc.collect()

    # -----------------------------------------------------------------------
    # Print summaries
    # -----------------------------------------------------------------------
    if "construction" in benchmarks:
        print(f"\n{'='*70}")
        print(f"BENCHMARK: Construction - {dataset_name}")
        print(f"{'='*70}")
        for name, stats in construction_results.items():
            bt = stats["build_time_s"]
            mem = stats["memory_mb"]
            if bt >= 0:
                print(f"  {name}: time={bt:.2f}s, memory={mem:.2f}MB")
            elif mem >= 0:
                print(f"  {name}: loaded from disk, memory={mem:.2f}MB")
            else:
                print(f"  {name}: FAILED")
        all_results["construction"] = construction_results

    if "recall_k20" in benchmarks:
        print(f"\n{'='*70}")
        print(f"BENCHMARK: Recall vs Time (k=20) - {dataset_name}")
        print(f"{'='*70}")
        all_results["recall_k20"] = rt_k20_results

    if "recall_k50" in benchmarks:
        print(f"\n{'='*70}")
        print(f"BENCHMARK: Recall vs Time (k=50) - {dataset_name}")
        print(f"{'='*70}")
        all_results["recall_k50"] = rt_k50_results

    if "robustness" in benchmarks:
        print(f"\n{'='*70}")
        print(f"BENCHMARK: Robustness (unseen queries, k=20) - {dataset_name}")
        print(f"{'='*70}")
        for name, stats in rob_results.items():
            print(f"  {name}: mean={stats['mean_recall']:.4f}, "
                  f"median={stats['median_recall']:.4f}, "
                  f"min={stats['min_recall']:.4f}, "
                  f"max={stats['max_recall']:.4f}")
        all_results["robustness"] = rob_results

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"results_{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SHG Paper Benchmark Suite")
    parser.add_argument("--data-dir", default="/Users/dhm/Documents/data")
    parser.add_argument("--index-dir", default="/Users/dhm/Documents/indices")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset", default="all",
                        choices=ALL_DATASETS + ["all"])
    parser.add_argument("--benchmark", nargs="+", default=["all"],
                        choices=ALL_BENCHMARKS + ["all"])
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results")

    benchmarks = ALL_BENCHMARKS if "all" in args.benchmark else args.benchmark
    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    os.makedirs(args.index_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Data dir:   {args.data_dir}")
    print(f"Index dir:  {args.index_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Datasets:   {datasets}")
    print(f"Benchmarks: {benchmarks}")

    for ds in datasets:
        try:
            run_benchmarks(ds, benchmarks, args.data_dir, args.index_dir, args.output_dir)
        except Exception as e:
            print(f"\nERROR processing {ds}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
