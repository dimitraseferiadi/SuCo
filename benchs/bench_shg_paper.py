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
    """Read .fbin file: [n, d] int32 header, then n*d float32."""
    with open(fname, "rb") as f:
        n, d = struct.unpack("ii", f.read(8))
        data = np.fromfile(f, dtype=dtype, count=n * d).reshape(n, d)
    return data


def read_ibin(fname):
    """Read .bin ground truth: [n, k] int32 header, then n*k int32."""
    with open(fname, "rb") as f:
        n, k = struct.unpack("ii", f.read(8))
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
        xb = read_openai_parquet(data_dir, max_vectors=1_000_000)
        nq = 10_000
        xq = xb[-nq:]
        xb = xb[:-nq]
        gt = compute_ground_truth(xb, xq, k=100)
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
        return xb, xq, gt

    else:
        raise ValueError(f"Unknown dataset: {name}")


def compute_ground_truth(xb, xq, k=100):
    """Compute exact k-NN ground truth using brute force."""
    print(f"  Computing ground truth (n={xb.shape[0]}, nq={xq.shape[0]}, k={k})...")
    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)
    _, I = index.search(xq, k)
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
    # Choose m_pq as a divisor of d
    m_pq = None
    for candidate in [d // 8, d // 4, d // 16, d // 2, 8, 16, 32]:
        if candidate > 0 and d % candidate == 0:
            m_pq = candidate
            break
    if m_pq is None:
        m_pq = 8
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
    print(f"\n{'#'*70}")
    print(f"# Dataset: {dataset_name.upper()}")
    print(f"{'#'*70}")

    # Load data
    print(f"\nLoading dataset {dataset_name}...")
    t0 = time.time()
    xb, xq, gt = load_dataset(dataset_name, data_dir)
    print(f"  Loaded in {time.time()-t0:.1f}s: xb={xb.shape}, xq={xq.shape}, gt={gt.shape}")
    d = xb.shape[1]

    all_results = {
        "dataset": dataset_name,
        "n": int(xb.shape[0]),
        "d": d,
        "nq": int(xq.shape[0]),
    }

    # -----------------------------------------------------------------------
    # Build all indices (kept in memory for search benchmarks)
    # -----------------------------------------------------------------------
    indices = {}  # name -> (index, build_time, mem_mb)

    builders = [
        ("SHG",       build_index_shg),
        ("HNSW",      build_index_hnsw),
        ("Panorama",  build_index_panorama),
        ("IVFFlat",   build_index_ivfflat),
        ("IVFPQ",     build_index_ivfpq),
    ]

    for name, builder in builders:
        print(f"\n--- Building {name} ---")
        try:
            idx, build_time = builder(xb, d)
            mem = index_size_bytes(idx)
            mem_mb = mem / (1024 * 1024) if mem > 0 else -1
            indices[name] = (idx, build_time, mem_mb)
            print(f"  {name}: time={build_time:.2f}s, memory={mem_mb:.2f}MB")

            # Save index to disk for reuse
            idx_path = os.path.join(index_dir, f"{dataset_name}_{name.lower()}.idx")
            try:
                faiss.write_index(idx, idx_path)
                print(f"  Saved to {idx_path}")
            except Exception as e:
                print(f"  Could not save: {e}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            traceback.print_exc()
            indices[name] = (None, -1, -1)

    # -----------------------------------------------------------------------
    # Benchmark 1: Construction
    # -----------------------------------------------------------------------
    if "construction" in benchmarks:
        print(f"\n{'='*70}")
        print(f"BENCHMARK: Construction - {dataset_name}")
        print(f"{'='*70}")
        construction_results = {}
        for name, (idx, bt, mem) in indices.items():
            construction_results[name] = {
                "build_time_s": round(bt, 2) if bt >= 0 else -1,
                "memory_mb": round(mem, 2) if mem >= 0 else -1,
            }
            if bt >= 0:
                print(f"  {name}: time={bt:.2f}s, memory={mem:.2f}MB")
            else:
                print(f"  {name}: FAILED")
        all_results["construction"] = construction_results

    # -----------------------------------------------------------------------
    # Benchmark 2: Recall vs time k=20
    # -----------------------------------------------------------------------
    if "recall_k20" in benchmarks:
        k = 20
        print(f"\n{'='*70}")
        print(f"BENCHMARK: Recall vs Time (k={k}) - {dataset_name}")
        print(f"{'='*70}")

        rt_results = {}

        for name, (idx, _, _) in indices.items():
            if idx is None:
                continue
            if name == "SHG":
                rt_results["SHG"] = sweep_shg(idx, "SHG", xq, gt, k)
                # Ablations
                rt_results["SHG-no-shortcut"] = sweep_shg(
                    idx, "SHG-no-shortcut", xq, gt, k, use_sc=False, use_lb=True)
                rt_results["SHG-no-lb"] = sweep_shg(
                    idx, "SHG-no-lb", xq, gt, k, use_sc=True, use_lb=False)
                rt_results["SHG-no-both"] = sweep_shg(
                    idx, "SHG-no-both", xq, gt, k, use_sc=False, use_lb=False)
            elif name in ("HNSW", "Panorama"):
                rt_results[name] = sweep_hnsw(idx, name, xq, gt, k)
            elif name in ("IVFFlat", "IVFPQ"):
                rt_results[name] = sweep_ivf(idx, name, xq, gt, k)

        all_results["recall_k20"] = rt_results

    # -----------------------------------------------------------------------
    # Benchmark 3: Recall vs time k=50
    # -----------------------------------------------------------------------
    if "recall_k50" in benchmarks:
        k = 50
        print(f"\n{'='*70}")
        print(f"BENCHMARK: Recall vs Time (k={k}) - {dataset_name}")
        print(f"{'='*70}")

        rt_results = {}

        for name, (idx, _, _) in indices.items():
            if idx is None:
                continue
            if name == "SHG":
                rt_results["SHG"] = sweep_shg(idx, "SHG", xq, gt, k)
            elif name in ("HNSW", "Panorama"):
                rt_results[name] = sweep_hnsw(idx, name, xq, gt, k)
            elif name in ("IVFFlat", "IVFPQ"):
                rt_results[name] = sweep_ivf(idx, name, xq, gt, k)

        all_results["recall_k50"] = rt_results

    # -----------------------------------------------------------------------
    # Benchmark 4: Robustness with unseen queries
    # -----------------------------------------------------------------------
    if "robustness" in benchmarks:
        k = 20
        print(f"\n{'='*70}")
        print(f"BENCHMARK: Robustness (unseen queries, k={k}) - {dataset_name}")
        print(f"{'='*70}")

        # Generate unseen queries: sample base vectors + gaussian noise
        n_unseen = min(1000, xq.shape[0])
        rng = np.random.RandomState(42)
        sample_ids = rng.choice(xb.shape[0], size=n_unseen, replace=False)
        noise_scale = float(np.std(xb)) * 0.1
        unseen_q = xb[sample_ids].copy() + \
            rng.randn(n_unseen, d).astype(np.float32) * noise_scale

        gt_unseen = compute_ground_truth(xb, unseen_q, k=max(k, 50))

        ef_test = max(k * 4, 100)
        nprobe_test = 64

        rob_results = {}

        for name, (idx, _, _) in indices.items():
            if idx is None:
                continue
            try:
                if name == "SHG":
                    D, I, t = search_shg(idx, unseen_q, k, ef_test)
                elif name in ("HNSW", "Panorama"):
                    D, I, t = search_hnsw(idx, unseen_q, k, ef_test)
                elif name in ("IVFFlat", "IVFPQ"):
                    D, I, t = search_ivf(idx, unseen_q, k, nprobe_test)
                else:
                    continue

                pqr = per_query_recall(I, gt_unseen, k)
                rob_results[name] = {
                    "mean_recall": round(float(pqr.mean()), 4),
                    "median_recall": round(float(np.median(pqr)), 4),
                    "min_recall": round(float(pqr.min()), 4),
                    "max_recall": round(float(pqr.max()), 4),
                    "q25_recall": round(float(np.percentile(pqr, 25)), 4),
                    "q75_recall": round(float(np.percentile(pqr, 75)), 4),
                    "ms_per_query": round(t * 1000 / n_unseen, 4),
                }
                print(f"  {name}: mean={pqr.mean():.4f}, "
                      f"median={np.median(pqr):.4f}, "
                      f"min={pqr.min():.4f}, max={pqr.max():.4f}")
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
                traceback.print_exc()

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
