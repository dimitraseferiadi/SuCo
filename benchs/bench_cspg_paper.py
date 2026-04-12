#!/usr/bin/env python3
"""
benchs/bench_cspg_paper.py

Benchmark suite for IndexCSPG (Crossing Sparse Proximity Graph, NeurIPS 2024)
against FAISS baselines (HNSW, IVFFlat, IVFPQ, exact Flat).

Reproduces the experimental methodology from:
  Ming Yang, Yuzheng Cai, Weiguo Zheng.
  "CSPG: Crossing Sparse Proximity Graphs for Approximate Nearest Neighbor
   Search." NeurIPS 2024.

Benchmarks:
  construction  — Build time & memory (Table 1 / Figure 4 style)
  recall_k10    — Recall@10 vs query-time curves (Figure 3 style)
  recall_k20    — Recall@20 vs query-time curves
  recall_k50    — Recall@50 vs query-time curves
  ablation_m    — Vary num_partitions m ∈ {1,2,4,8} (Figure 5 style)
  ablation_lam  — Vary routing ratio λ ∈ {0.1,0.2,0.3,0.4,0.5} (Figure 6 style)
  ablation_ef1  — Vary first-stage ef1 ∈ {1,2,4,8,16} (Figure 7 style)
  robustness    — Per-query recall distribution on held-out queries (Figure 8 style)

Indices benchmarked:
    CSPG          — IndexCSPG (default: M=32, efConstruction=128, m=2, λ=0.5)
    HNSW          — IndexHNSWFlat (same M, efConstruction as CSPG)
  IVFFlat       — IndexIVFFlat (nlist = sqrt(n))
  IVFPQ         — IndexIVFPQ   (nlist = sqrt(n), m_pq ≈ d/8)
  Flat          — IndexFlatL2  (exact, skipped for n > 2M by default)

Default index set:
    cspg, hnsw

Datasets (paper set):
    sift1m, deep1m, gist1m, sift10m

Usage:
  python benchs/bench_cspg_paper.py --dataset gist1m --benchmark all
  python benchs/bench_cspg_paper.py --dataset all --benchmark construction recall_k10
    python benchs/bench_cspg_paper.py --dataset deep1m --benchmark ablation_m ablation_lam
"""

import argparse
import gc
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
    sys.exit("Cannot import faiss. Build FAISS with IndexCSPG support first.")

# Verify IndexCSPG is available
if not hasattr(faiss, "IndexCSPG"):
    sys.exit(
        "faiss.IndexCSPG not found. Recompile FAISS with IndexCSPG support "
        "and the corresponding Python bindings."
    )
if not hasattr(faiss, "SearchParametersCSPG"):
    print(
        "WARNING: faiss.SearchParametersCSPG not found in Python bindings. "
        "Search parameter sweeps will use index-level defaults (efSearch, ef1). "
        "Recompile with bindings to enable per-query parameter control."
    )
    _HAS_CSPG_PARAMS = False
else:
    _HAS_CSPG_PARAMS = True


# ---------------------------------------------------------------------------
# CSPG default construction parameters (from paper Appendix D)
# ---------------------------------------------------------------------------
DEFAULT_M = 32           # HNSW M (degree per node)
DEFAULT_EFC = 128        # efConstruction
DEFAULT_NUM_PARTITIONS = 2
DEFAULT_LAMBDA = 0.5     # routing vector ratio

# Search parameter sweep grids
EF2_SWEEP = [10, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1000, 1500]
NPROBE_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Ablation grids (matching paper Figures 5-7)
ABLATION_M_VALUES = [1, 2, 4, 8]
ABLATION_LAMBDA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]
ABLATION_EF1_VALUES = [1, 2, 4, 8, 16]

# Skip exact Flat search for datasets larger than this
FLAT_MAX_VECTORS = 2_000_000

ALL_DATASETS = ["sift1m", "deep1m", "gist1m", "uqv1m", "sift10m"]
ALL_BENCHMARKS = [
    "construction",
    "recall_k10",
    "recall_k20",
    "recall_k50",
    "ablation_m",
    "ablation_lam",
    "ablation_ef1",
    "robustness",
]

ALL_INDEX_TYPES = ["cspg", "hnsw", "ivfflat", "ivfpq", "flat"]
DEFAULT_INDEX_TYPES = ["cspg", "hnsw"]


# ===========================================================================
# Dataset I/O  (identical helpers to bench_shg_paper.py)
# ===========================================================================

def read_fvecs(fname):
    with open(fname, "rb") as f:
        d = struct.unpack("i", f.read(4))[0]
        f.seek(0)
        n = os.path.getsize(fname) // (4 + d * 4)
        data = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            dim = struct.unpack("i", f.read(4))[0]
            assert dim == d, f"Dim mismatch row {i}: {dim} vs {d}"
            data[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return data


def read_ivecs(fname):
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
    with open(fname, "rb") as f:
        n, d = struct.unpack("ii", f.read(8))
        file_size = os.path.getsize(fname)
        actual_n = (file_size - 8) // (d * np.dtype(dtype).itemsize)
        if actual_n < n:
            n = actual_n
        data = np.fromfile(f, dtype=dtype, count=n * d).reshape(n, d)
    return data


def read_ibin(fname):
    with open(fname, "rb") as f:
        n, k = struct.unpack("ii", f.read(8))
        file_size = os.path.getsize(fname)
        actual_n = (file_size - 8) // (k * 4)
        if actual_n < n:
            n = actual_n
        data = np.fromfile(f, dtype=np.int32, count=n * k).reshape(n, k)
    return data


def read_enron_data(fname):
    with open(fname, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=3)
        _version, n, d = int(header[0]), int(header[1]), int(header[2])
        data = np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)
    return data


def read_openai_parquet(data_dir, max_vectors=1_000_000):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow required for OpenAI dataset: pip install pyarrow")
    parquet_dir = os.path.join(data_dir, "openai1m")
    files = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])
    all_vecs, total = [], 0
    for fname in files:
        if total >= max_vectors:
            break
        table = pq.read_table(os.path.join(parquet_dir, fname))
        for col_name in ["emb", "embedding", "vector", "values"]:
            if col_name in table.column_names:
                break
        else:
            col_name = table.column_names[-1]
        for row in table[col_name]:
            if total >= max_vectors:
                break
            all_vecs.append(np.array(row.as_py(), dtype=np.float32))
            total += 1
    return np.array(all_vecs, dtype=np.float32)


def _first_existing_path(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _find_dataset_dir(data_dir, candidates, required_file):
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.exists(os.path.join(p, required_file)):
            return p
    return None


def _read_fvecs_mmap(path, n=None):
    """Read fvecs via FAISS mmap helper when available; fallback to read_fvecs."""
    try:
        vecs_io = __import__("faiss.contrib.vecs_io", fromlist=["fvecs_mmap"])
        fvecs_mmap = vecs_io.fvecs_mmap

        x = fvecs_mmap(path)
        if n is not None:
            x = x[:n]
        return np.ascontiguousarray(x.astype(np.float32, copy=False))
    except Exception:
        x = read_fvecs(path)
        if n is not None:
            x = x[:n]
        return np.ascontiguousarray(x.astype(np.float32, copy=False))


def _load_sift10m(data_dir, nb=10_000_000, nq=10_000):
    """Load SIFT10M vectors from MATLAB/HDF5 file and GT from .npy/.ivecs."""
    mat_path = _first_existing_path([
        os.path.join(data_dir, "SIFT10M", "SIFT10Mfeatures.mat"),
        os.path.join(data_dir, "sift10m", "SIFT10Mfeatures.mat"),
        os.path.join(data_dir, "SIFT10Mfeatures.mat"),
    ])
    if mat_path is None:
        raise FileNotFoundError(
            "Could not find SIFT10Mfeatures.mat. Expected one of: "
            "<data_dir>/SIFT10M/SIFT10Mfeatures.mat, "
            "<data_dir>/sift10m/SIFT10Mfeatures.mat, or <data_dir>/SIFT10Mfeatures.mat"
        )

    try:
        from scipy.io import loadmat

        data = loadmat(mat_path)
        key = next((k for k in data.keys() if not k.startswith("_")), None)
        if key is None:
            raise RuntimeError(f"No feature matrix found in {mat_path}")
        raw = np.asarray(data[key])
        del data
    except NotImplementedError:
        try:
            import h5py
        except ImportError as e:
            raise RuntimeError(
                "SIFT10M .mat appears to be MATLAB v7.3 (HDF5). Install h5py."
            ) from e

        with h5py.File(mat_path, "r") as f:
            preferred = ["fea", "features", "X", "data"]
            key = next((k for k in preferred if k in f), None)
            if key is None:
                key = next((k for k in f.keys() if getattr(f[k], "ndim", 0) == 2), None)
            if key is None:
                raise RuntimeError(f"Could not find 2D feature dataset in {mat_path}")
            dset = f[key]
            # Only read the rows we need (nb + nq) to avoid loading full dataset
            need = nb + nq
            if dset.ndim != 2:
                raise RuntimeError(f"Expected 2D feature dataset in {mat_path}, got shape {dset.shape}")
            if dset.shape[1] == 128:
                raw = np.empty((need, 128), dtype=np.float32)
                dset.read_direct(raw, np.s_[:need, :])
            elif dset.shape[0] == 128:
                raw = np.ascontiguousarray(dset[:, :need].T.astype(np.float32))
            else:
                raise RuntimeError(f"Cannot infer SIFT10M matrix layout from shape {dset.shape}")

    if raw.ndim != 2:
        raise RuntimeError(f"Expected 2D feature matrix in {mat_path}, got {raw.shape}")
    if raw.shape[1] == 128:
        x = np.ascontiguousarray(raw, dtype=np.float32)
    elif raw.shape[0] == 128:
        x = np.ascontiguousarray(raw.T, dtype=np.float32)
    else:
        raise RuntimeError(f"Cannot infer SIFT10M matrix layout from shape {raw.shape}")

    if x.shape[0] < nb + nq:
        raise ValueError(
            f"SIFT10M file has {x.shape[0]} vectors but requires at least {nb + nq}"
        )
    xb = x[:nb]
    xq = x[nb: nb + nq]

    gt_path = _first_existing_path([
        os.path.join(data_dir, "sift10m_gt.npy"),
        os.path.join(data_dir, "SIFT10M", "sift10m_gt.npy"),
        os.path.join(data_dir, "sift10m", "sift10m_gt.npy"),
        os.path.join(data_dir, "SIFT10M", "sift_groundtruth.ivecs"),
        os.path.join(data_dir, "sift10m", "sift_groundtruth.ivecs"),
    ])
    if gt_path is None:
        gt = compute_ground_truth(xb, xq, 100)
    elif gt_path.endswith(".npy"):
        gt = np.load(gt_path)
    else:
        gt = read_ivecs(gt_path)

    if gt.shape[0] > xq.shape[0]:
        gt = gt[:xq.shape[0]]
    return xb, xq, gt


def load_dataset(name, data_dir):
    """Returns (xb, xq, gt)."""
    name = name.lower()

    if name == "sift1m":
        p = _find_dataset_dir(data_dir, ["sift1M", "sift1m", "sift"], "sift_base.fvecs")
        if p is None:
            raise FileNotFoundError(
                "Could not find SIFT1M under data_dir. Expected one of "
                "sift1M/, sift1m/, or sift/ with sift_base.fvecs"
            )
        xb = _read_fvecs_mmap(os.path.join(p, "sift_base.fvecs"))
        xq = _read_fvecs_mmap(os.path.join(p, "sift_query.fvecs"))
        gt = read_ivecs(os.path.join(p, "sift_groundtruth.ivecs"))
        return xb, xq, gt

    elif name == "deep1m":
        p = os.path.join(data_dir, "deep1b")
        xb = _read_fvecs_mmap(os.path.join(p, "base.fvecs"), n=1_000_000)
        xq = _read_fvecs_mmap(os.path.join(p, "deep1B_queries.fvecs"), n=10_000)
        gt_path = _first_existing_path([
            os.path.join(p, "deep1M_groundtruth.ivecs"),
            os.path.join(p, "deep1M_groundtruth.npy"),
        ])
        if gt_path is None:
            gt = compute_ground_truth(xb, xq, 100)
        elif gt_path.endswith(".npy"):
            gt = np.load(gt_path)
        else:
            gt = read_ivecs(gt_path)
        if gt.shape[0] > xq.shape[0]:
            gt = gt[:xq.shape[0]]
        return xb, xq, gt

    elif name == "gist1m":
        p = _find_dataset_dir(data_dir, ["gist1M", "gist1m", "gist"], "gist_base.fvecs")
        if p is None:
            raise FileNotFoundError(
                "Could not find GIST1M under data_dir. Expected one of "
                "gist1M/, gist1m/, or gist/ with gist_base.fvecs"
            )
        xb = _read_fvecs_mmap(os.path.join(p, "gist_base.fvecs"))
        xq = _read_fvecs_mmap(os.path.join(p, "gist_query.fvecs"))
        gt = read_ivecs(os.path.join(p, "gist_groundtruth.ivecs"))
        return xb, xq, gt

    elif name == "uqv1m":
        p = _find_dataset_dir(data_dir, ["uqv", "uqv1m", "UQV"], "uqv_base.fvecs")
        if p is None:
            raise FileNotFoundError(
                "Could not find UQV under data_dir. Expected one of "
                "uqv/, uqv1m/, or UQV/ with uqv_base.fvecs"
            )
        xb = _read_fvecs_mmap(os.path.join(p, "uqv_base.fvecs"))
        xq = _read_fvecs_mmap(os.path.join(p, "uqv_query.fvecs"))
        gt = read_ivecs(os.path.join(p, "uqv_groundtruth.ivecs"))
        return xb, xq, gt

    elif name == "sift10m":
        return _load_sift10m(data_dir)

    else:
        raise ValueError(f"Unknown dataset: {name!r}")


def _get_dataset_size(name):
    """Return n (number of base vectors) without loading them."""
    sizes = {"sift1m": 1_000_000, "deep1m": 1_000_000, "gist1m": 1_000_000,
             "uqv1m": 1_000_000, "sift10m": 10_000_000}
    name = name.lower()
    if name in sizes:
        return sizes[name]
    raise ValueError(f"Unknown dataset size for {name!r}")


def load_dataset_queries_only(name, data_dir):
    """Load only xq and gt (no base vectors). Used when all indices are cached."""
    name = name.lower()

    if name == "sift1m":
        p = _find_dataset_dir(data_dir, ["sift1M", "sift1m", "sift"], "sift_base.fvecs")
        xq = _read_fvecs_mmap(os.path.join(p, "sift_query.fvecs"))
        gt = read_ivecs(os.path.join(p, "sift_groundtruth.ivecs"))

    elif name == "deep1m":
        p = os.path.join(data_dir, "deep1b")
        xq = _read_fvecs_mmap(os.path.join(p, "deep1B_queries.fvecs"), n=10_000)
        gt_path = _first_existing_path([
            os.path.join(p, "deep1M_groundtruth.ivecs"),
            os.path.join(p, "deep1M_groundtruth.npy"),
        ])
        if gt_path is None:
            raise FileNotFoundError("deep1m ground truth not found and xb not loaded")
        gt = np.load(gt_path) if gt_path.endswith(".npy") else read_ivecs(gt_path)
        if gt.shape[0] > xq.shape[0]:
            gt = gt[:xq.shape[0]]

    elif name == "gist1m":
        p = _find_dataset_dir(data_dir, ["gist1M", "gist1m", "gist"], "gist_base.fvecs")
        xq = _read_fvecs_mmap(os.path.join(p, "gist_query.fvecs"))
        gt = read_ivecs(os.path.join(p, "gist_groundtruth.ivecs"))

    elif name == "sift10m":
        # Queries were split from the .mat file at offset 10M; load from cache
        gt_path = _first_existing_path([
            os.path.join(data_dir, "sift10m_gt.npy"),
            os.path.join(data_dir, "SIFT10M", "sift10m_gt.npy"),
            os.path.join(data_dir, "sift10m", "sift10m_gt.npy"),
            os.path.join(data_dir, "SIFT10M", "sift_groundtruth.ivecs"),
            os.path.join(data_dir, "sift10m", "sift_groundtruth.ivecs"),
        ])
        if gt_path is None:
            raise FileNotFoundError("sift10m ground truth not found and xb not loaded")
        gt = np.load(gt_path) if gt_path.endswith(".npy") else read_ivecs(gt_path)

        # Load only query vectors from the .mat file (rows 10M..10M+10K)
        nb, nq = 10_000_000, 10_000
        mat_path = _first_existing_path([
            os.path.join(data_dir, "SIFT10M", "SIFT10Mfeatures.mat"),
            os.path.join(data_dir, "sift10m", "SIFT10Mfeatures.mat"),
            os.path.join(data_dir, "SIFT10Mfeatures.mat"),
        ])
        try:
            import h5py
            with h5py.File(mat_path, "r") as f:
                preferred = ["fea", "features", "X", "data"]
                key = next((k for k in preferred if k in f), None)
                if key is None:
                    key = next((k for k in f.keys() if getattr(f[k], "ndim", 0) == 2), None)
                dset = f[key]
                if dset.shape[1] == 128:
                    xq = np.ascontiguousarray(dset[nb:nb+nq, :], dtype=np.float32)
                else:
                    xq = np.ascontiguousarray(dset[:, nb:nb+nq].T, dtype=np.float32)
        except Exception:
            # Fallback: load full dataset (shouldn't normally happen)
            xb_full, xq, _ = _load_sift10m(data_dir)
            del xb_full

        if gt.shape[0] > xq.shape[0]:
            gt = gt[:xq.shape[0]]

    else:
        raise ValueError(f"Unknown dataset: {name!r}")

    return xq, gt


def compute_ground_truth(xb, xq, k=100):
    print(f"  Computing ground truth (n={xb.shape[0]}, nq={xq.shape[0]}, k={k})…")
    _, I = faiss.knn(xq, xb, k, metric=faiss.METRIC_L2)
    return I


# ===========================================================================
# Index builders
# ===========================================================================

def build_cspg(xb, d, M=DEFAULT_M, efc=DEFAULT_EFC,
               num_partitions=DEFAULT_NUM_PARTITIONS, lam=DEFAULT_LAMBDA):
    idx = faiss.IndexCSPG(d, M, num_partitions, lam)
    idx.efConstruction = efc
    t0 = time.time()
    idx.add(xb)
    elapsed = time.time() - t0
    tag = f"CSPG(m={num_partitions}, λ={lam})"
    print(f"  {tag}: build={elapsed:.2f}s")
    return idx, elapsed


def build_hnsw(xb, d, M=DEFAULT_M, efc=DEFAULT_EFC):
    idx = faiss.IndexHNSWFlat(d, M)
    idx.hnsw.efConstruction = efc
    t0 = time.time()
    idx.add(xb)
    elapsed = time.time() - t0
    print(f"  HNSW(M={M}): build={elapsed:.2f}s")
    return idx, elapsed


def build_ivfflat(xb, d):
    nlist = max(1, int(np.sqrt(xb.shape[0])))
    quantizer = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFFlat(quantizer, d, nlist)
    t0 = time.time()
    idx.train(xb); idx.add(xb)
    elapsed = time.time() - t0
    print(f"  IVFFlat(nlist={nlist}): build={elapsed:.2f}s")
    return idx, elapsed


def build_ivfpq(xb, d):
    nlist  = max(1, int(np.sqrt(xb.shape[0])))
    target = max(1, d // 8)
    divisors = [i for i in range(1, d + 1) if d % i == 0]
    m_pq = min(divisors, key=lambda x: abs(x - target))
    quantizer = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFPQ(quantizer, d, nlist, m_pq, 8)
    t0 = time.time()
    idx.train(xb); idx.add(xb)
    elapsed = time.time() - t0
    print(f"  IVFPQ(nlist={nlist}, m={m_pq}): build={elapsed:.2f}s")
    return idx, elapsed


def build_flat(xb, d):
    idx = faiss.IndexFlatL2(d)
    t0 = time.time()
    idx.add(xb)
    elapsed = time.time() - t0
    print(f"  Flat(exact): build={elapsed:.2f}s")
    return idx, elapsed


# ===========================================================================
# Index persistence helpers
# ===========================================================================

def try_save_index(idx, path, label):
    """Attempt faiss.write_index; skip gracefully if the type is unregistered."""
    try:
        faiss.write_index(idx, path)
        print(f"  Saved {label} → {path}")
        return True
    except Exception as e:
        print(f"  Could not save {label} (type not registered for FAISS I/O): {e}")
        return False


def try_load_index(path, label):
    """Attempt faiss.read_index; return None on failure."""
    if not os.path.exists(path):
        return None
    try:
        idx = faiss.read_index(path)
        print(f"  Loaded {label} from {path}")
        return idx
    except Exception as e:
        print(f"  Could not load {label}: {e} — will rebuild")
        return None


# ===========================================================================
# Measurement utilities
# ===========================================================================

def index_size_mb(idx):
    try:
        buf = faiss.serialize_index(idx)
        return len(buf) / (1024 * 1024)
    except Exception:
        return -1.0


def per_query_recall(I, gt, k):
    nq   = I.shape[0]
    k_gt = min(k, gt.shape[1])
    k_rt = min(k, I.shape[1])
    out  = np.zeros(nq, dtype=np.float64)
    for i in range(nq):
        gt_s  = set(gt[i, :k_gt].tolist()) - {-1}
        rt_s  = set(I[i, :k_rt].tolist()) - {-1}
        out[i] = len(gt_s & rt_s) / k_gt if k_gt else 0.0
    return out


# ===========================================================================
# Search factories
# ===========================================================================
#
# Each factory returns a callable  fn(idx, xq, k) -> (D, I)
# that is pre-configured for one operating point.

def _cspg_search_fn(ef2, ef1=1):
    """CSPG two-stage search with fixed ef1 and variable ef2."""
    if _HAS_CSPG_PARAMS:
        def fn(idx, xq, k):
            sp = faiss.SearchParametersCSPG()
            sp.ef1      = ef1
            sp.efSearch = ef2
            return idx.search(xq, k, params=sp)
    else:
        # Fall back: mutate index-level defaults (not thread-safe, but workable
        # for single-threaded timing loops)
        def fn(idx, xq, k):
            idx.ef1      = ef1
            idx.efSearch = ef2
            return idx.search(xq, k)
    return fn


def _hnsw_search_fn(ef_search):
    def fn(idx, xq, k):
        sp = faiss.SearchParametersHNSW()
        sp.efSearch = ef_search
        return idx.search(xq, k, params=sp)
    return fn


def _ivf_search_fn(nprobe):
    def fn(idx, xq, k):
        idx.nprobe = nprobe
        return idx.search(xq, k)
    return fn


def _flat_search_fn():
    def fn(idx, xq, k):
        return idx.search(xq, k)
    return fn


# ===========================================================================
# Recall-time curve sweep
# ===========================================================================

def recall_time_curve(idx, label, xq, gt, k, search_fn_factory,
                      param_values, n_warmup=3):
    """
    Sweep `param_values`, time each batch search, compute mean recall.

    Parameters
    ----------
    search_fn_factory : callable
        fn(param) -> search_fn, where search_fn(idx, xq, k) -> (D, I)
    param_values : list
        Values to sweep (ef2 for CSPG/HNSW, nprobe for IVF).

    Returns
    -------
    list of dict {param, recall, ms_per_query}
    """
    nq   = xq.shape[0]
    k_gt = min(k, gt.shape[1])

    prev_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)   # single-threaded timing matches the paper

    points = []
    for param in param_values:
        search_fn = search_fn_factory(param)

        # Warm-up
        for _ in range(n_warmup):
            search_fn(idx, xq[:1], k)

        # Timed batch search
        t0 = time.perf_counter()
        D, I = search_fn(idx, xq, k)
        elapsed = time.perf_counter() - t0

        # Recall
        recalls = np.zeros(nq)
        for i in range(nq):
            gt_s = set(gt[i, :k_gt].tolist()) - {-1}
            rt_s = set(I[i].tolist()) - {-1}
            recalls[i] = len(gt_s & rt_s) / k_gt if k_gt else 0.0

        mean_r = float(recalls.mean())
        ms_q   = (elapsed / nq) * 1000.0

        points.append({
            "param":        int(param),
            "recall":       round(mean_r, 6),
            "ms_per_query": round(ms_q,   6),
        })
        print(f"    {label} (param={param:5d}): "
              f"recall={mean_r:.4f}, time={ms_q:.4f} ms/q")

    faiss.omp_set_num_threads(prev_threads)
    points.sort(key=lambda x: x["recall"])
    return points


# ===========================================================================
# Main benchmark runner (one dataset)
# ===========================================================================

def run_benchmarks(dataset_name, benchmarks, index_types, data_dir, index_dir, output_dir):

    print(f"\n{'#'*72}")
    print(f"# Dataset: {dataset_name.upper()}")
    print(f"{'#'*72}")

    # ------------------------------------------------------------------
    # Load dataset (defer xb if all indices are already cached)
    # ------------------------------------------------------------------

    # Collect all cache keys we'll need so we can check if xb is required
    _needed_keys = set()
    if "cspg" in index_types:
        _needed_keys.add("cspg_default")
    if "hnsw" in index_types:
        _needed_keys.add("hnsw")
    if "ivfflat" in index_types:
        _needed_keys.add("ivfflat")
    if "ivfpq" in index_types:
        _needed_keys.add("ivfpq")
    if "ablation_m" in benchmarks:
        for _m in ABLATION_M_VALUES:
            _needed_keys.add(f"cspg_m{_m}_l{int(DEFAULT_LAMBDA*100)}")
    if "ablation_lam" in benchmarks:
        for _lam in ABLATION_LAMBDA_VALUES:
            _needed_keys.add(f"cspg_m{DEFAULT_NUM_PARTITIONS}_l{int(round(_lam*100))}")

    _all_cached = all(
        os.path.exists(os.path.join(index_dir, f"{dataset_name}_{ck}.idx"))
        for ck in _needed_keys
    )
    # Also check robustness cache
    _rob_cached = True
    if "robustness" in benchmarks:
        _rob_cached = (
            os.path.exists(os.path.join(index_dir, f"{dataset_name}_cspg_rob_q.npy")) and
            os.path.exists(os.path.join(index_dir, f"{dataset_name}_cspg_rob_gt.npy"))
        )

    _need_xb = not (_all_cached and _rob_cached)

    print(f"\nLoading {dataset_name}…")
    t0 = time.time()
    if _need_xb:
        xb, xq, gt = load_dataset(dataset_name, data_dir)
        print(f"  Done in {time.time()-t0:.1f}s | "
              f"xb={xb.shape}  xq={xq.shape}  gt={gt.shape}")
        if gt.max() >= xb.shape[0]:
            print(f"  GT IDs out of range — recomputing…")
            gt = compute_ground_truth(xb, xq, gt.shape[1])
        d = xb.shape[1]
        n = int(xb.shape[0])
    else:
        xb = None
        xq, gt = load_dataset_queries_only(dataset_name, data_dir)
        print(f"  Queries only in {time.time()-t0:.1f}s | "
              f"xq={xq.shape}  gt={gt.shape}  (xb skipped — all indices cached)")
        d = xq.shape[1]
        n = _get_dataset_size(dataset_name)

    all_results = {"dataset": dataset_name, "n": n, "d": d, "nq": int(xq.shape[0])}

    # ------------------------------------------------------------------
    # Pre-compute robustness queries once (before possibly freeing xb)
    # ------------------------------------------------------------------
    unseen_q, gt_unseen = None, None
    if "robustness" in benchmarks:
        rob_q_path  = os.path.join(index_dir, f"{dataset_name}_cspg_rob_q.npy")
        rob_gt_path = os.path.join(index_dir, f"{dataset_name}_cspg_rob_gt.npy")
        if os.path.exists(rob_q_path) and os.path.exists(rob_gt_path):
            unseen_q   = np.load(rob_q_path)
            gt_unseen  = np.load(rob_gt_path)
            print(f"  Robustness queries loaded from cache.")
        else:
            n_rob = min(1000, xq.shape[0])
            rng   = np.random.RandomState(42)
            ids   = rng.choice(xb.shape[0], size=n_rob, replace=False)
            noise = rng.randn(n_rob, d).astype(np.float32) * float(np.std(xb)) * 0.1
            unseen_q  = xb[ids].copy() + noise
            gt_unseen = compute_ground_truth(xb, unseen_q, k=50)
            np.save(rob_q_path,  unseen_q)
            np.save(rob_gt_path, gt_unseen)
            print(f"  Robustness queries generated and cached.")

    # ------------------------------------------------------------------
    # Result accumulators
    # ------------------------------------------------------------------
    construction_results = {}
    recall_results       = {f"recall_k{k}": {} for k in [10, 20, 50]}
    ablation_m_results   = {}
    ablation_lam_results = {}
    ablation_ef1_results = {}
    robustness_results   = {}

    # ==================================================================
    # Helper: build or load a single index variant
    # ==================================================================
    def _get_index(cache_key, builder_fn, label):
        """Return (idx, build_time_s, mem_mb) — loads from disk if cached."""
        idx_path = os.path.join(index_dir, f"{dataset_name}_{cache_key}.idx")

        cached = try_load_index(idx_path, label)
        if cached is not None:
            build_time = prebuild_times.get(cache_key, -1.0)
            mem_mb = index_size_mb(cached)
            return cached, build_time, mem_mb

        if xb is None:
            print(f"  {label}: no cache and xb was freed — skip")
            return None, -1.0, -1.0
        print(f"\n--- Building {label} ---")
        try:
            idx, build_time = builder_fn(xb, d)
            try_save_index(idx, idx_path, label)
        except Exception as e:
            print(f"  {label}: BUILD FAILED — {e}")
            traceback.print_exc()
            return None, -1.0, -1.0

        mem_mb = index_size_mb(idx)
        return idx, build_time, mem_mb

    # ------------------------------------------------------------------
    # Load cached construction times from a prior run (carry-forward)
    # ------------------------------------------------------------------
    prev_construction = {}
    prev_path = os.path.join(output_dir, f"results_cspg_{dataset_name}.json")
    if os.path.exists(prev_path):
        try:
            with open(prev_path) as fp:
                prev_construction = json.load(fp).get("construction", {})
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Pre-build pass: ensure every index variant is cached on disk so
    # that we can free xb before the memory-intensive benchmark loops.
    # Each index is built, saved, and immediately freed.
    # ------------------------------------------------------------------
    prebuild_times = {}  # cache_key -> build_time_s

    def _ensure_cached(cache_key, builder_fn, label):
        idx_path = os.path.join(index_dir, f"{dataset_name}_{cache_key}.idx")
        if os.path.exists(idx_path):
            return
        print(f"\n--- Pre-building {label} ---")
        idx, build_time = builder_fn(xb, d)
        try_save_index(idx, idx_path, label)
        prebuild_times[cache_key] = build_time
        del idx
        gc.collect()

    prebuild_plan = []
    if "cspg" in index_types:
        prebuild_plan.append(
            ("cspg_default", lambda xb, d: build_cspg(xb, d),
             f"CSPG(m={DEFAULT_NUM_PARTITIONS}, λ={DEFAULT_LAMBDA})"))
    if "hnsw" in index_types:
        prebuild_plan.append(
            ("hnsw", lambda xb, d: build_hnsw(xb, d), "HNSW"))
    if "ivfflat" in index_types:
        prebuild_plan.append(
            ("ivfflat", lambda xb, d: build_ivfflat(xb, d), "IVFFlat"))
    if "ivfpq" in index_types:
        prebuild_plan.append(
            ("ivfpq", lambda xb, d: build_ivfpq(xb, d), "IVFPQ"))
    if "flat" in index_types and n <= FLAT_MAX_VECTORS:
        prebuild_plan.append(
            ("flat", lambda xb, d: build_flat(xb, d), "Flat"))

    # Ablation m variants
    if "ablation_m" in benchmarks:
        for m in ABLATION_M_VALUES:
            ck = f"cspg_m{m}_l{int(DEFAULT_LAMBDA*100)}"
            prebuild_plan.append(
                (ck, lambda xb, d, _m=m: build_cspg(xb, d, num_partitions=_m, lam=DEFAULT_LAMBDA),
                 f"CSPG(m={m},λ={DEFAULT_LAMBDA})"))

    # Ablation lambda variants
    if "ablation_lam" in benchmarks:
        for lam in ABLATION_LAMBDA_VALUES:
            lam_pct = int(round(lam * 100))
            ck = f"cspg_m{DEFAULT_NUM_PARTITIONS}_l{lam_pct}"
            prebuild_plan.append(
                (ck, lambda xb, d, _lam=lam: build_cspg(xb, d, num_partitions=DEFAULT_NUM_PARTITIONS, lam=_lam),
                 f"CSPG(m={DEFAULT_NUM_PARTITIONS},λ={lam})"))

    seen_keys = set()
    for cache_key, builder, label in prebuild_plan:
        if cache_key in seen_keys:
            continue
        seen_keys.add(cache_key)
        try:
            _ensure_cached(cache_key, builder, label)
        except Exception as e:
            print(f"  Pre-build {label} FAILED: {e}")
            traceback.print_exc()

    # Free base vectors — all indices are now cached on disk
    del xb
    xb = None
    gc.collect()
    print(f"\n  Base vectors freed — all indices cached to disk.")

    # ==================================================================
    # 1. CONSTRUCTION benchmark
    # ==================================================================
    if "construction" in benchmarks:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Construction — {dataset_name}")
        print(f"{'='*60}")

        build_plan = []
        if "cspg" in index_types:
            build_plan.append(
                ("cspg_default",
                 f"CSPG(m={DEFAULT_NUM_PARTITIONS}, λ={DEFAULT_LAMBDA})",
                 lambda xb, d: build_cspg(xb, d))
            )
        if "hnsw" in index_types:
            build_plan.append(("hnsw", "HNSW", lambda xb, d: build_hnsw(xb, d)))
        if "ivfflat" in index_types:
            build_plan.append(("ivfflat", "IVFFlat", lambda xb, d: build_ivfflat(xb, d)))
        if "ivfpq" in index_types:
            build_plan.append(("ivfpq", "IVFPQ", lambda xb, d: build_ivfpq(xb, d)))
        if "flat" in index_types and n <= FLAT_MAX_VECTORS:
            build_plan.append(
                ("flat", "Flat", lambda xb, d: build_flat(xb, d))
            )

        for cache_key, label, builder in build_plan:
            idx, bt, mem = _get_index(cache_key, builder, label)
            if idx is None:
                construction_results[label] = {"build_time_s": -1, "memory_mb": -1}
                continue

            # Carry forward build_time if loaded from cache
            if bt < 0 and label in prev_construction:
                bt = prev_construction[label].get("build_time_s", -1)

            construction_results[label] = {
                "build_time_s": round(bt, 2),
                "memory_mb":    round(mem, 2),
            }
            print(f"  {label}: time={bt:.2f}s, memory={mem:.2f}MB")
            del idx
            gc.collect()

        all_results["construction"] = construction_results

    # ==================================================================
    # 2-4. RECALL vs TIME benchmarks  (k=10, 20, 50)
    # ==================================================================
    recall_k_list = []
    for bm in benchmarks:
        if bm.startswith("recall_k"):
            try:
                recall_k_list.append(int(bm[len("recall_k"):]))
            except ValueError:
                pass

    if recall_k_list:
        # Define index variants and their search factories + sweep grids
        recall_index_plan = []
        if "cspg" in index_types:
            recall_index_plan.append(
                (
                    "cspg_default",
                    f"CSPG(m={DEFAULT_NUM_PARTITIONS},λ={DEFAULT_LAMBDA})",
                    lambda xb, d: build_cspg(xb, d),
                    lambda ef2: _cspg_search_fn(ef2, ef1=1),
                    EF2_SWEEP,
                )
            )
        if "hnsw" in index_types:
            recall_index_plan.append(
                (
                    "hnsw",
                    "HNSW",
                    lambda xb, d: build_hnsw(xb, d),
                    _hnsw_search_fn,
                    EF2_SWEEP,
                )
            )
        if "ivfflat" in index_types:
            recall_index_plan.append(
                (
                    "ivfflat",
                    "IVFFlat",
                    lambda xb, d: build_ivfflat(xb, d),
                    _ivf_search_fn,
                    NPROBE_SWEEP,
                )
            )
        if "ivfpq" in index_types:
            recall_index_plan.append(
                (
                    "ivfpq",
                    "IVFPQ",
                    lambda xb, d: build_ivfpq(xb, d),
                    _ivf_search_fn,
                    NPROBE_SWEEP,
                )
            )
        if "flat" in index_types and n <= FLAT_MAX_VECTORS:
            recall_index_plan.append((
                "flat", "Flat",
                lambda xb, d: build_flat(xb, d),
                lambda _: _flat_search_fn(),
                [0],  # only one point (exact)
            ))

        for k in recall_k_list:
            bm_key = f"recall_k{k}"
            print(f"\n{'='*60}")
            print(f"BENCHMARK: Recall vs Time (k={k}) — {dataset_name}")
            print(f"{'='*60}")
            k_results = {}

            for cache_key, label, builder, sf_factory, sweep in recall_index_plan:
                idx, _, _ = _get_index(cache_key, builder, label)
                if idx is None:
                    continue
                print(f"  → {label}")
                try:
                    k_results[label] = recall_time_curve(
                        idx, label, xq, gt, k, sf_factory, sweep)
                except Exception as e:
                    print(f"    FAILED: {e}")
                    traceback.print_exc()
                del idx
                gc.collect()

            recall_results[bm_key] = k_results
            all_results[bm_key] = k_results

    # ==================================================================
    # 5. ABLATION: varying m (num_partitions)
    # ==================================================================
    if "ablation_m" in benchmarks:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Ablation — m (num_partitions) — {dataset_name}")
        print(f"{'='*60}")
        k_abl = 10

        for m in ABLATION_M_VALUES:
            label      = f"CSPG(m={m},λ={DEFAULT_LAMBDA})"
            cache_key  = f"cspg_m{m}_l{int(DEFAULT_LAMBDA*100)}"
            lam        = DEFAULT_LAMBDA

            def _builder(xb, d, _m=m, _lam=lam):
                return build_cspg(xb, d, num_partitions=_m, lam=_lam)

            idx, _, _ = _get_index(cache_key, _builder, label)
            if idx is None:
                continue
            print(f"  → {label}")
            try:
                ablation_m_results[label] = recall_time_curve(
                    idx, label, xq, gt, k_abl,
                    lambda ef2: _cspg_search_fn(ef2, ef1=1),
                    EF2_SWEEP)
            except Exception as e:
                print(f"    FAILED: {e}")
                traceback.print_exc()
            del idx
            gc.collect()

        all_results["ablation_m"] = ablation_m_results

    # ==================================================================
    # 6. ABLATION: varying λ (routing vector ratio)
    # ==================================================================
    if "ablation_lam" in benchmarks:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Ablation — λ (routing ratio) — {dataset_name}")
        print(f"{'='*60}")
        k_abl = 10

        for lam in ABLATION_LAMBDA_VALUES:
            lam_pct    = int(round(lam * 100))
            label      = f"CSPG(m={DEFAULT_NUM_PARTITIONS},λ={lam})"
            cache_key  = f"cspg_m{DEFAULT_NUM_PARTITIONS}_l{lam_pct}"

            def _builder(xb, d, _lam=lam):
                return build_cspg(xb, d, num_partitions=DEFAULT_NUM_PARTITIONS, lam=_lam)

            idx, _, _ = _get_index(cache_key, _builder, label)
            if idx is None:
                continue
            print(f"  → {label}")
            try:
                ablation_lam_results[label] = recall_time_curve(
                    idx, label, xq, gt, k_abl,
                    lambda ef2: _cspg_search_fn(ef2, ef1=1),
                    EF2_SWEEP)
            except Exception as e:
                print(f"    FAILED: {e}")
                traceback.print_exc()
            del idx
            gc.collect()

        all_results["ablation_lam"] = ablation_lam_results

    # ==================================================================
    # 7. ABLATION: varying ef1 (first-stage beam width)
    # ==================================================================
    if "ablation_ef1" in benchmarks:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Ablation — ef1 — {dataset_name}")
        print(f"{'='*60}")
        k_abl = 10

        # Single CSPG index — only built once
        idx, _, _ = _get_index(
            "cspg_default",
            lambda xb, d: build_cspg(xb, d),
            f"CSPG(m={DEFAULT_NUM_PARTITIONS},λ={DEFAULT_LAMBDA})",
        )
        if idx is not None:
            for ef1 in ABLATION_EF1_VALUES:
                label = f"CSPG(ef1={ef1})"
                print(f"  → {label}")
                try:
                    ablation_ef1_results[label] = recall_time_curve(
                        idx, label, xq, gt, k_abl,
                        lambda ef2, _ef1=ef1: _cspg_search_fn(ef2, ef1=_ef1),
                        EF2_SWEEP)
                except Exception as e:
                    print(f"    FAILED: {e}")
                    traceback.print_exc()
            del idx
            gc.collect()

        all_results["ablation_ef1"] = ablation_ef1_results

    # ==================================================================
    # 8. ROBUSTNESS: per-query recall distribution on unseen queries
    # ==================================================================
    if "robustness" in benchmarks and unseen_q is not None:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Robustness — {dataset_name}")
        print(f"{'='*60}")

        # Fixed operating point: ef2=200 (targeting ~90% recall region)
        ROB_EF2 = 200
        ROB_K   = 20

        rob_plan = []
        if "cspg" in index_types:
            rob_plan.append(
                (
                    "cspg_default",
                    f"CSPG(m={DEFAULT_NUM_PARTITIONS},λ={DEFAULT_LAMBDA})",
                    lambda xb, d: build_cspg(xb, d),
                    lambda idx: idx.search(unseen_q, ROB_K,
                        params=(faiss.SearchParametersCSPG().__setattr__("efSearch", ROB_EF2)
                                or faiss.SearchParametersCSPG()) if _HAS_CSPG_PARAMS else None),
                )
            )
        if "hnsw" in index_types:
            rob_plan.append(
                (
                    "hnsw",
                    "HNSW",
                    lambda xb, d: build_hnsw(xb, d),
                    lambda idx: idx.search(unseen_q, ROB_K,
                        params=_make_hnsw_params(ROB_EF2)),
                )
            )
        if "ivfflat" in index_types:
            rob_plan.append(
                (
                    "ivfflat",
                    "IVFFlat",
                    lambda xb, d: build_ivfflat(xb, d),
                    lambda idx: (setattr(idx, "nprobe", 64),
                                 idx.search(unseen_q, ROB_K))[1],
                )
            )
        if "ivfpq" in index_types:
            rob_plan.append(
                (
                    "ivfpq",
                    "IVFPQ",
                    lambda xb, d: build_ivfpq(xb, d),
                    lambda idx: (setattr(idx, "nprobe", 64),
                                 idx.search(unseen_q, ROB_K))[1],
                )
            )

        for cache_key, label, builder, searcher in rob_plan:
            idx, _, _ = _get_index(cache_key, builder, label)
            if idx is None:
                continue
            print(f"  → {label}")
            try:
                t0 = time.time()
                if label.startswith("CSPG"):
                    # Use direct search_fn so ef2 is applied properly
                    sf = _cspg_search_fn(ROB_EF2, ef1=1)
                    _, I = sf(idx, unseen_q, ROB_K)
                    t_s = time.time() - t0
                elif label == "HNSW":
                    sp = faiss.SearchParametersHNSW()
                    sp.efSearch = ROB_EF2
                    t0 = time.time()
                    _, I = idx.search(unseen_q, ROB_K, params=sp)
                    t_s = time.time() - t0
                else:
                    idx.nprobe = 64
                    t0 = time.time()
                    _, I = idx.search(unseen_q, ROB_K)
                    t_s = time.time() - t0

                pqr = per_query_recall(I, gt_unseen, ROB_K)
                robustness_results[label] = {
                    "mean_recall":   round(float(pqr.mean()), 4),
                    "median_recall": round(float(np.median(pqr)), 4),
                    "min_recall":    round(float(pqr.min()), 4),
                    "max_recall":    round(float(pqr.max()), 4),
                    "q25_recall":    round(float(np.percentile(pqr, 25)), 4),
                    "q75_recall":    round(float(np.percentile(pqr, 75)), 4),
                    "ms_per_query":  round(t_s * 1000 / unseen_q.shape[0], 4),
                }
                print(f"    mean={pqr.mean():.4f}  "
                      f"median={np.median(pqr):.4f}  "
                      f"[{pqr.min():.4f}, {pqr.max():.4f}]")
            except Exception as e:
                print(f"    FAILED: {e}")
                traceback.print_exc()
            del idx
            gc.collect()

        all_results["robustness"] = robustness_results

    # ==================================================================
    # Summary print
    # ==================================================================
    if "construction" in benchmarks and construction_results:
        print(f"\n{'='*60}")
        print(f"SUMMARY: Construction — {dataset_name}")
        print(f"{'='*60}")
        for lbl, stats in construction_results.items():
            bt  = stats["build_time_s"]
            mem = stats["memory_mb"]
            print(f"  {lbl:45s} time={bt:8.2f}s  mem={mem:8.1f}MB")

    # ==================================================================
    # Persist results
    # ==================================================================
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"results_cspg_{dataset_name}.json")
    with open(out_path, "w") as fp:
        json.dump(all_results, fp, indent=2, default=str)
    print(f"\nResults → {out_path}")
    return all_results


# ---------------------------------------------------------------------------
# Small helper needed inside robustness block
# ---------------------------------------------------------------------------
def _make_hnsw_params(ef):
    sp = faiss.SearchParametersHNSW()
    sp.efSearch = ef
    return sp


# ===========================================================================
# CLI
# ===========================================================================

def main():
    global DEFAULT_M, DEFAULT_EFC, DEFAULT_NUM_PARTITIONS, DEFAULT_LAMBDA

    parser = argparse.ArgumentParser(
        description="CSPG (NeurIPS 2024) benchmark suite vs FAISS baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="/Users/dhm/Documents/data",
        help="Root directory containing dataset sub-folders",
    )
    parser.add_argument(
        "--index-dir", default="/Users/dhm/Documents/indices",
        help="Directory for persisting built indices",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for JSON result files (default: <script_dir>/results)",
    )
    parser.add_argument(
        "--dataset", default="all",
        choices=ALL_DATASETS + ["all"],
        help="Dataset to benchmark (default: all)",
    )
    parser.add_argument(
        "--benchmark", nargs="+", default=["all"],
        choices=ALL_BENCHMARKS + ["all"],
        help="Which benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--index-type", nargs="+", default=DEFAULT_INDEX_TYPES,
        choices=ALL_INDEX_TYPES + ["all"],
        help=("Index families to benchmark (default: cspg hnsw). "
              "Use 'all' to include ivfflat ivfpq flat."),
    )
    # Override construction parameters
    parser.add_argument("--M",   type=int,   default=DEFAULT_M,
                        help=f"HNSW M parameter (default {DEFAULT_M})")
    parser.add_argument("--efc", type=int,   default=DEFAULT_EFC,
                        help=f"efConstruction (default {DEFAULT_EFC})")
    parser.add_argument("--m",   type=int,   default=DEFAULT_NUM_PARTITIONS,
                        help=f"CSPG num_partitions (default {DEFAULT_NUM_PARTITIONS})")
    parser.add_argument("--lam", type=float, default=DEFAULT_LAMBDA,
                        help=f"CSPG lambda routing ratio (default {DEFAULT_LAMBDA})")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results"
        )

    # Apply CLI overrides to module-level defaults
    DEFAULT_M              = args.M
    DEFAULT_EFC            = args.efc
    DEFAULT_NUM_PARTITIONS = args.m
    DEFAULT_LAMBDA         = args.lam

    benchmarks = ALL_BENCHMARKS if "all" in args.benchmark else args.benchmark
    index_types = ALL_INDEX_TYPES if "all" in args.index_type else args.index_type
    datasets   = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    os.makedirs(args.index_dir,  exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("CSPG Paper Benchmark Suite")
    print("=" * 72)
    print(f"  Data dir:         {args.data_dir}")
    print(f"  Index dir:        {args.index_dir}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Datasets:         {datasets}")
    print(f"  Benchmarks:       {benchmarks}")
    print(f"  Index types:      {index_types}")
    print(f"  CSPG params:      M={DEFAULT_M}, efC={DEFAULT_EFC}, "
          f"m={DEFAULT_NUM_PARTITIONS}, λ={DEFAULT_LAMBDA}")
    print(f"  SearchParamsCSPG: {'available' if _HAS_CSPG_PARAMS else 'NOT FOUND — using index defaults'}")
    print("=" * 72)

    for ds in datasets:
        try:
            run_benchmarks(ds, benchmarks, index_types,
                           args.data_dir, args.index_dir, args.output_dir)
        except Exception as e:
            print(f"\nERROR on dataset {ds!r}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()