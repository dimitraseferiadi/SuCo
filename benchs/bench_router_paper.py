 #!/usr/bin/env python3
"""
benchs/bench_router_paper.py

Default-parameter benchmark of IndexSuCo / IndexSHG / IndexCSPG across the
11 datasets used as training data for the index-selection router.

Indices and defaults (taken from each paper / report):
  - SuCo    (report 1, §3.4):           Ns=8, nc=50, α=0.05, β=0.005, niter=10
  - SHG     (report 2, §4.2):           M=48, efConstruction=128, η=2
  - CSPG    (report 3, §4.2):           M=32, efConstruction=128, m=2, λ=0.5, ef1=1
  - HNSW32  (CSPG substrate baseline):  M=32, efConstruction=128
  - HNSW48  (SHG substrate baseline):   M=48, efConstruction=128

Datasets:
  sift1m, sift10m, gist1m, deep1m, deep10m, spacev10m,
  msong, enron, openai1m, msturing10m, uqv

Benchmarks:
  construction        — build time + memory + index size
  features            — dataset features (n, d, LID, pdist moments, kmeans inertia)
  recall_k{1,10,20,50,100}  — QPS vs Recall@k curve, mean ± std over N_RUNS
  robustness          — per-query recall@20 distribution at fixed search budget
  unseen_robustness   — recall on held-out base vectors (rebuild w/o them)
  hard_robustness     — recall stratified by GT k-th distance (easy vs hard 10%)
  latency_tail        — p50/p95/p99/p99.9 per-query latency at recall {0.90,0.95,0.99}
  cold_warm           — first-query/cold-cache vs warm-cache latency
  mre                 — mean relative error at recall {0.90,0.95,0.99}
  pareto              — derived Pareto upper envelope from recall_k* curves
  time_at_recall      — interpolated ms/query at recall {0.80,0.90,0.95,0.99}
                        + speedup-at-recall vs HNSW32 baseline

All results land in {output_dir}/results_<dataset>.json.
Indices are persisted to {index_dir}/<dataset>_<index>.idx and reloaded if present.

Usage:
  python benchs/bench_router_paper.py --dataset sift1m --benchmark all
  python benchs/bench_router_paper.py --dataset all --benchmark all
  python benchs/bench_router_paper.py --dataset gist1m --benchmark recall_k10 latency_tail mre
"""

import argparse
import gc
import json
import os
import platform as _platform
import resource as _resource
import struct
import sys
import time
import traceback

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("Cannot import faiss. Build FAISS with custom index support first.")


# ---------------------------------------------------------------------------
# Defaults — straight from the three reports
# ---------------------------------------------------------------------------

# SuCo (report 1, §3.4)
SUCO_NSUBSPACES_PREFERRED = 8
SUCO_NCENTROIDS_HALF = 50
SUCO_COLLISION_RATIO = 0.05
SUCO_CANDIDATE_RATIO = 0.005
SUCO_NITER = 10

# SHG (report 2, §4.2 / SHG paper §5.1)
SHG_M = 48
SHG_EFC = 80

# CSPG (report 3, §4.2)
CSPG_M = 32
CSPG_EFC = 128
CSPG_NUM_PARTITIONS = 2
CSPG_LAMBDA = 0.5
CSPG_EF1 = 1

# HNSW reference baselines — same per-graph budget as the indices they sit under.
HNSW32_M = 32; HNSW32_EFC = 128   # matches CSPG substrate
HNSW48_M = 48; HNSW48_EFC = 80    # matches SHG paper baseline (§5.1)


# Search-parameter sweeps for QPS-recall curves.
EF_SEARCH_VALUES = [
    10, 15, 20, 30, 40, 60, 80, 100, 150,
    200, 300, 400, 600, 800, 1000, 1500, 2000,
]
SUCO_CANDIDATE_RATIO_VALUES = [
    0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2,
]

# Recall@k values to record.
RECALL_KS = (1, 10, 20, 50, 100)
SEARCH_K = max(RECALL_KS)

# Repetitions for QPS std reporting.
N_RUNS = 3

# Robustness experiment: fixed search budget per index family.
ROBUSTNESS_EFSEARCH = 200
ROBUSTNESS_CANDIDATE_RATIO = 0.005

# Tail-latency / MRE / cold-warm: recall targets at which to operate.
LATENCY_RECALL_TARGETS = (0.90, 0.95, 0.99)
LATENCY_NUM_QUERIES = 5000          # cap per-query timing loop
COLDWARM_RECALL_TARGET = 0.95
COLDWARM_NUM_COLD = 30
COLDWARM_NUM_WARM = 500
COLD_EVICT_MB = 2048                # heap-thrash buffer to evict caches

# Hard / unseen robustness configuration.
HARD_QUERY_PCTILE = 90              # top 10% hardest by GT k-th distance
UNSEEN_FRAC = 0.05                  # fraction of base held out as unseen queries
UNSEEN_K = 20
UNSEEN_GT_K = 100
UNSEEN_MAX_QUERIES = 5000

# Time-at-recall extraction (post-processing on recall_k* curves).
TIME_AT_RECALL_TARGETS = (0.80, 0.90, 0.95, 0.99)
SPEEDUP_BASELINE_LABEL = "HNSW32"   # denominator for speedup ratios

# MRE evaluation.
MRE_K = 20

ALL_DATASETS = [
    "sift1m", "sift10m", "gist1m",
    "deep1m", "deep10m",
    "spacev10m",
    "msong", "enron", "openai1m",
    "msturing10m", "uqv",
]
ALL_BENCHMARKS = [
    "construction",
    "features",
    "recall_k1", "recall_k10", "recall_k20", "recall_k50", "recall_k100",
    "robustness",
    "unseen_robustness",
    "hard_robustness",
    "latency_tail",
    "cold_warm",
    "mre",
    "pareto",
    "time_at_recall",
]
# Benchmarks computed per (dataset,index): driven inside the per-index loop.
PER_INDEX_BENCHMARKS = {
    "construction",
    "robustness", "hard_robustness",
    "latency_tail", "cold_warm", "mre",
    *(f"recall_k{k}" for k in RECALL_KS),
}
# Benchmarks computed once per dataset (need rebuild or aggregate over results).
DATASET_BENCHMARKS = {"features", "unseen_robustness", "pareto", "time_at_recall"}

ALL_INDEX_TYPES = ["suco", "shg", "cspg", "hnsw32", "hnsw48"]
DEFAULT_INDEX_TYPES = ["suco", "shg", "cspg", "hnsw32", "hnsw48"]


# ---------------------------------------------------------------------------
# Memory measurement (resource.getrusage tracks peak automatically)
# ---------------------------------------------------------------------------

_RUSAGE_DIVISOR = 1024 * 1024 if _platform.system() == "Darwin" else 1024


def _peak_rss_mb():
    return _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / _RUSAGE_DIVISOR


def index_size_mb(idx):
    try:
        return len(faiss.serialize_index(idx)) / (1024 * 1024)
    except Exception:
        return -1.0


# ===========================================================================
# Dataset I/O helpers
# ===========================================================================

def read_fvecs(path, n=None):
    with open(path, "rb") as f:
        d = struct.unpack("i", f.read(4))[0]
    row_bytes = 4 + d * 4
    total = os.path.getsize(path) // row_bytes
    if n is None or n > total:
        n = total
    arr = np.memmap(path, dtype=np.uint8, mode="r")[: n * row_bytes].reshape(n, row_bytes)
    return np.ascontiguousarray(arr[:, 4:].view(np.float32).reshape(n, d), dtype=np.float32)


def read_ivecs(path):
    with open(path, "rb") as f:
        d = struct.unpack("i", f.read(4))[0]
    row_bytes = 4 + d * 4
    n = os.path.getsize(path) // row_bytes
    arr = np.memmap(path, dtype=np.uint8, mode="r")[: n * row_bytes].reshape(n, row_bytes)
    return np.ascontiguousarray(arr[:, 4:].view(np.int32).reshape(n, d))


def read_fbin(path, dtype=np.float32):
    """Header (n,d) int32, then n*d values of dtype. Caps n by file size."""
    itemsize = np.dtype(dtype).itemsize
    with open(path, "rb") as f:
        n_hdr, d = struct.unpack("ii", f.read(8))
    actual_n = (os.path.getsize(path) - 8) // (d * itemsize)
    n = min(n_hdr, actual_n)
    return np.fromfile(path, dtype=dtype, count=n * d, offset=8).reshape(n, d)


def read_ibin(path):
    return read_fbin(path, dtype=np.int32)


def read_enron(path):
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=np.int32, count=3)
        _, n, d = int(hdr[0]), int(hdr[1]), int(hdr[2])
        data = np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)
    return data


def compute_ground_truth(xb, xq, k=100):
    print(f"  Computing ground truth (n={xb.shape[0]}, nq={xq.shape[0]}, k={k})...")
    _, I = faiss.knn(xq, xb, k, metric=faiss.METRIC_L2)
    return I.astype(np.int32)


# ===========================================================================
# Dataset loaders — one branch per dataset
# ===========================================================================

def load_dataset(name, data_dir):
    name = name.lower()

    if name == "sift1m":
        p = os.path.join(data_dir, "sift1M")
        return (read_fvecs(os.path.join(p, "sift_base.fvecs")),
                read_fvecs(os.path.join(p, "sift_query.fvecs")),
                read_ivecs(os.path.join(p, "sift_groundtruth.ivecs")))

    if name == "sift10m":
        return _load_sift10m(data_dir)

    if name == "gist1m":
        p = os.path.join(data_dir, "gist1M")
        return (read_fvecs(os.path.join(p, "gist_base.fvecs")),
                read_fvecs(os.path.join(p, "gist_query.fvecs")),
                read_ivecs(os.path.join(p, "gist_groundtruth.ivecs")))

    if name == "deep1m":
        p = os.path.join(data_dir, "deep1b")
        xb = read_fvecs(os.path.join(p, "base.fvecs"), n=1_000_000)
        xq = read_fvecs(os.path.join(p, "deep1B_queries.fvecs"), n=10_000)
        gt = read_ivecs(os.path.join(p, "deep1M_groundtruth.ivecs"))
        if gt.shape[0] > xq.shape[0]:
            gt = gt[: xq.shape[0]]
        return xb, xq, gt

    if name == "deep10m":
        p = os.path.join(data_dir, "deep1b")
        xb = read_fvecs(os.path.join(p, "base.fvecs"), n=10_000_000)
        xq = read_fvecs(os.path.join(p, "deep1B_queries.fvecs"), n=10_000)
        gt = read_ivecs(os.path.join(p, "deep10M_groundtruth.ivecs"))
        if gt.shape[0] > xq.shape[0]:
            gt = gt[: xq.shape[0]]
        return xb, xq, gt

    if name == "spacev10m":
        p = os.path.join(data_dir, "spacev10m")
        with open(os.path.join(p, "base.100M.i8bin"), "rb") as f:
            n_hdr, d = struct.unpack("ii", f.read(8))
        n_use = min(10_000_000, n_hdr)
        xb = np.fromfile(
            os.path.join(p, "base.100M.i8bin"),
            dtype=np.int8, count=n_use * d, offset=8,
        ).reshape(n_use, d).astype(np.float32)
        with open(os.path.join(p, "query.30K.i8bin"), "rb") as f:
            nq, dq = struct.unpack("ii", f.read(8))
        if dq != d:
            raise RuntimeError(f"SpaceV dim mismatch: base d={d}, query d={dq}")
        xq = np.fromfile(
            os.path.join(p, "query.30K.i8bin"),
            dtype=np.int8, count=nq * dq, offset=8,
        ).reshape(nq, dq).astype(np.float32)
        gt = read_ibin(os.path.join(p, "groundtruth.30K.i32bin"))
        if gt.shape[0] > xq.shape[0]:
            gt = gt[: xq.shape[0]]
        return xb, xq, gt

    if name == "msong":
        p = os.path.join(data_dir, "msong")
        return (read_fvecs(os.path.join(p, "msong_base.fvecs")),
                read_fvecs(os.path.join(p, "msong_query.fvecs")),
                read_ivecs(os.path.join(p, "msong_groundtruth.ivecs")))

    if name == "enron":
        p = os.path.join(data_dir, "enron")
        return (read_enron(os.path.join(p, "enron.data_new")),
                read_fvecs(os.path.join(p, "enron_query.fvecs")),
                read_ivecs(os.path.join(p, "enron_groundtruth.ivecs")))

    if name == "openai1m":
        p = os.path.join(data_dir, "openai1m")
        xb = np.ascontiguousarray(np.load(os.path.join(p, "openai_xb.npy")), dtype=np.float32)
        xq = np.ascontiguousarray(np.load(os.path.join(p, "openai_xq.npy")), dtype=np.float32)
        gt = np.load(os.path.join(p, "openai_gt100.npy")).astype(np.int32)
        return xb, xq, gt

    if name == "msturing10m":
        p = os.path.join(data_dir, "msturing10m")
        xb = read_fbin(os.path.join(p, "base1b.fbin.crop_nb_10000000"))
        xq = read_fbin(os.path.join(p, "testQuery10K.fbin"))
        gt = read_ibin(os.path.join(p, "msturing-gt-10M"))
        if gt.shape[0] > xq.shape[0]:
            gt = gt[: xq.shape[0]]
        return xb, xq, gt

    if name == "uqv":
        p = os.path.join(data_dir, "uqv")
        return (read_fvecs(os.path.join(p, "uqv_base.fvecs")),
                read_fvecs(os.path.join(p, "uqv_query.fvecs")),
                read_ivecs(os.path.join(p, "uqv_groundtruth.ivecs")))

    raise ValueError(f"Unknown dataset: {name!r}")


def _load_sift10m(data_dir, nb=10_000_000, nq=10_000):
    p = os.path.join(data_dir, "SIFT10M", "SIFT10Mfeatures.mat")
    if not os.path.exists(p):
        raise FileNotFoundError(f"SIFT10M features file not found: {p}")
    try:
        from scipy.io import loadmat
        data = loadmat(p)
        key = next((k for k in data.keys() if not k.startswith("_")), None)
        raw = np.asarray(data[key])
        del data
    except NotImplementedError:
        import h5py
        with h5py.File(p, "r") as f:
            key = next((k for k in ("fea", "features", "X", "data") if k in f), None)
            if key is None:
                key = next(k for k in f.keys() if getattr(f[k], "ndim", 0) == 2)
            dset = f[key]
            need = nb + nq
            if dset.shape[1] == 128:
                raw = np.empty((need, 128), dtype=np.float32)
                dset.read_direct(raw, np.s_[:need, :])
            else:
                raw = np.ascontiguousarray(dset[:, :need].T.astype(np.float32))
    if raw.shape[1] != 128:
        raw = raw.T
    x = np.ascontiguousarray(raw, dtype=np.float32)
    xb, xq = x[:nb], x[nb : nb + nq]

    gt_path = None
    for cand in [
        os.path.join(data_dir, "sift10m_gt.npy"),
        os.path.join(data_dir, "SIFT10M", "sift10m_gt.npy"),
    ]:
        if os.path.exists(cand):
            gt_path = cand
            break
    if gt_path:
        gt = np.load(gt_path).astype(np.int32)
    else:
        gt = compute_ground_truth(xb, xq, 100)
        cache = os.path.join(data_dir, "SIFT10M", "sift10m_gt.npy")
        try:
            np.save(cache, gt)
            print(f"  Cached GT to {cache}")
        except Exception as e:
            print(f"  Could not cache GT: {e}")
    return xb, xq, gt


# ===========================================================================
# Index builders
# ===========================================================================

def _pick_suco_nsubspaces(d, preferred=SUCO_NSUBSPACES_PREFERRED):
    candidates = [n for n in range(preferred, 0, -1) if d % n == 0 and (d // n) % 2 == 0]
    return candidates[0] if candidates else None


def build_index_suco(xb, d):
    n = _pick_suco_nsubspaces(d)
    if n is None:
        raise RuntimeError(
            f"SuCo: no valid nsubspaces ≤ {SUCO_NSUBSPACES_PREFERRED} for d={d} "
            f"(needs d%n==0 and (d/n)%2==0)"
        )
    if n != SUCO_NSUBSPACES_PREFERRED:
        print(f"  SuCo: using nsubspaces={n} (preferred {SUCO_NSUBSPACES_PREFERRED} invalid for d={d})")

    idx = faiss.IndexSuCo(
        d, n, SUCO_NCENTROIDS_HALF,
        SUCO_COLLISION_RATIO, SUCO_CANDIDATE_RATIO, SUCO_NITER,
    )
    idx.verbose = False
    t0 = time.time()
    idx.train(xb)
    idx.add(xb)
    t_total = time.time() - t0
    print(f"  SuCo (Ns={n}, nc={SUCO_NCENTROIDS_HALF}): build={t_total:.2f}s")
    return idx, t_total


def build_index_shg(xb, d):
    idx = faiss.IndexSHG(d, SHG_M)
    idx.hnsw.efConstruction = SHG_EFC
    t0 = time.time()
    idx.add(xb)
    t_add = time.time() - t0
    t1 = time.time()
    idx.build_shortcut()
    t_sc = time.time() - t1
    t_total = t_add + t_sc
    print(f"  SHG (M={SHG_M}, efC={SHG_EFC}): add={t_add:.2f}s, "
          f"shortcut={t_sc:.2f}s, total={t_total:.2f}s")
    return idx, t_total


def build_index_cspg(xb, d):
    idx = faiss.IndexCSPG(d, CSPG_M, CSPG_NUM_PARTITIONS, CSPG_LAMBDA)
    idx.efConstruction = CSPG_EFC
    try:
        idx.ef1 = CSPG_EF1
    except Exception:
        pass
    t0 = time.time()
    idx.add(xb)
    t_total = time.time() - t0
    print(f"  CSPG (M={CSPG_M}, efC={CSPG_EFC}, m={CSPG_NUM_PARTITIONS}, "
          f"λ={CSPG_LAMBDA}): build={t_total:.2f}s")
    return idx, t_total


def _build_hnsw(xb, d, M, efc, label):
    idx = faiss.IndexHNSWFlat(d, M)
    idx.hnsw.efConstruction = efc
    t0 = time.time()
    idx.add(xb)
    t_total = time.time() - t0
    print(f"  {label} (M={M}, efC={efc}): build={t_total:.2f}s")
    return idx, t_total


def build_index_hnsw32(xb, d):
    return _build_hnsw(xb, d, HNSW32_M, HNSW32_EFC, "HNSW32")


def build_index_hnsw48(xb, d):
    return _build_hnsw(xb, d, HNSW48_M, HNSW48_EFC, "HNSW48")


BUILDERS = {
    "suco":   ("SuCo",   build_index_suco),
    "shg":    ("SHG",    build_index_shg),
    "cspg":   ("CSPG",   build_index_cspg),
    "hnsw32": ("HNSW32", build_index_hnsw32),
    "hnsw48": ("HNSW48", build_index_hnsw48),
}


# ===========================================================================
# Recall / MRE metrics
# ===========================================================================

def compute_recall_at_k(I, gt, k):
    nq = I.shape[0]
    k_gt = min(k, gt.shape[1])
    k_ret = min(k, I.shape[1])
    hits = 0
    for i in range(nq):
        gt_set  = set(gt[i, :k_gt].tolist()) - {-1}
        ret_set = set(I[i, :k_ret].tolist()) - {-1}
        hits += len(gt_set & ret_set)
    return hits / (nq * k_gt) if k_gt > 0 else 0.0


def per_query_recall(I, gt, k):
    nq = I.shape[0]
    k_gt = min(k, gt.shape[1])
    k_ret = min(k, I.shape[1])
    out = np.zeros(nq)
    for i in range(nq):
        gt_set  = set(gt[i, :k_gt].tolist()) - {-1}
        ret_set = set(I[i, :k_ret].tolist()) - {-1}
        out[i] = len(gt_set & ret_set) / k_gt if k_gt > 0 else 0.0
    return out


def approx_ratio_at_k(D_ret, xb, xq, gt, k=10):
    """Mean over queries of mean(d_ret[:k] / d_true[:k]). Squared L2."""
    k_use = min(k, gt.shape[1], D_ret.shape[1])
    nq = xq.shape[0]
    ratios = []
    for i in range(nq):
        gt_ids = gt[i, :k_use]
        gt_ids = gt_ids[gt_ids >= 0]
        if len(gt_ids) == 0:
            continue
        true_d = ((xb[gt_ids] - xq[i]) ** 2).sum(axis=1)
        ret_d = D_ret[i, :k_use]
        ratios.append((np.sort(ret_d)[: len(gt_ids)] / np.maximum(true_d, 1e-12)).mean())
    return float(np.mean(ratios)) if ratios else -1.0


def compute_mre_at_k(I, xb, xq, gt, k=MRE_K):
    """SuCo-style MRE: (1/k) Σ ‖q,oᵢ‖ / ‖q,o*ᵢ‖. Real (non-squared) L2."""
    nq = xq.shape[0]
    k_use = min(k, gt.shape[1], I.shape[1])
    mres = []
    for i in range(nq):
        gt_ids = gt[i, :k_use]
        gt_ids = gt_ids[gt_ids >= 0]
        ret_ids = I[i, :k_use]
        ret_ids = ret_ids[ret_ids >= 0]
        if len(gt_ids) == 0 or len(ret_ids) == 0:
            continue
        q = xq[i]
        true_d = np.sqrt(np.maximum(((xb[gt_ids] - q) ** 2).sum(axis=1), 0))
        ret_d  = np.sqrt(np.maximum(((xb[ret_ids] - q) ** 2).sum(axis=1), 0))
        ret_sorted = np.sort(ret_d)[: len(true_d)]
        mres.append(float(np.mean(ret_sorted / np.maximum(true_d, 1e-12))))
    return float(np.mean(mres)) if mres else -1.0


# ===========================================================================
# Search-factory helpers
# ===========================================================================

def _make_shg_search_factory():
    def factory(ef_search):
        def fn(idx, xq, k):
            sp = faiss.SearchParametersSHG()
            sp.use_shortcut = True
            sp.use_lb_pruning = True
            sp.efSearch = int(ef_search)
            return idx.search(xq, k, params=sp)
        return fn
    return factory


def _make_cspg_search_factory():
    def factory(ef_search):
        def fn(idx, xq, k):
            try:
                sp = faiss.SearchParametersCSPG()
                sp.efSearch = int(ef_search)
                return idx.search(xq, k, params=sp)
            except Exception:
                try:
                    idx.efSearch = int(ef_search)
                except Exception:
                    idx.hnsw.efSearch = int(ef_search)
                return idx.search(xq, k)
        return fn
    return factory


def _make_suco_search_factory():
    def factory(candidate_ratio):
        def fn(idx, xq, k):
            idx.candidate_ratio = float(candidate_ratio)
            return idx.search(xq, k)
        return fn
    return factory


def _make_hnsw_search_factory():
    def factory(ef_search):
        def fn(idx, xq, k):
            sp = faiss.SearchParametersHNSW()
            sp.efSearch = int(ef_search)
            return idx.search(xq, k, params=sp)
        return fn
    return factory


SEARCH_FACTORY = {
    "suco":   (_make_suco_search_factory(), SUCO_CANDIDATE_RATIO_VALUES, "candidate_ratio"),
    "shg":    (_make_shg_search_factory(),  EF_SEARCH_VALUES,            "efSearch"),
    "cspg":   (_make_cspg_search_factory(), EF_SEARCH_VALUES,            "efSearch"),
    "hnsw32": (_make_hnsw_search_factory(), EF_SEARCH_VALUES,            "efSearch"),
    "hnsw48": (_make_hnsw_search_factory(), EF_SEARCH_VALUES,            "efSearch"),
}


# ===========================================================================
# Latency helpers
# ===========================================================================

def per_query_latencies(search_fn, idx, xq, k, max_n=LATENCY_NUM_QUERIES):
    """Run queries one-by-one, return array of per-query latencies in ms."""
    nq = min(int(xq.shape[0]), int(max_n))
    times = np.zeros(nq, dtype=np.float64)
    for i in range(nq):
        q = xq[i:i+1]
        t0 = time.perf_counter()
        search_fn(idx, q, k)
        times[i] = (time.perf_counter() - t0) * 1000.0
    return times


def latency_quantiles(times):
    return {
        "p50":   round(float(np.percentile(times, 50)),   6),
        "p95":   round(float(np.percentile(times, 95)),   6),
        "p99":   round(float(np.percentile(times, 99)),   6),
        "p999":  round(float(np.percentile(times, 99.9)), 6),
        "mean":  round(float(times.mean()),               6),
        "std":   round(float(times.std()),                6),
        "max":   round(float(times.max()),                6),
        "n":     int(len(times)),
    }


def evict_caches(size_mb=COLD_EVICT_MB):
    """Allocate + touch a large buffer to evict CPU caches and (partially) page cache."""
    try:
        n_floats = (int(size_mb) * 1024 * 1024) // 8
        arr = np.random.rand(n_floats).astype(np.float64)
        _ = float(arr.sum())  # force materialization
        del arr
        gc.collect()
        return True
    except Exception:
        return False


def flush_os_cache():
    """Best-effort page-cache flush (needs root). Returns True on success."""
    if _platform.system() == "Linux":
        return os.system("sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null") == 0
    if _platform.system() == "Darwin":
        return os.system("purge >/dev/null 2>&1") == 0
    return False


# ===========================================================================
# Pareto / time-at-recall (post-processing)
# ===========================================================================

def pareto_frontier(rows):
    """Upper envelope on (recall, qps): for each recall level, keep best qps."""
    if not rows:
        return []
    rows_sorted = sorted(rows, key=lambda r: r["recall"])
    out = []
    best = -1.0
    for r in reversed(rows_sorted):
        if r["qps"] > best:
            best = r["qps"]
            out.append(r)
    return list(reversed(out))


def pick_param_for_recall(curve, target):
    """Return the row whose recall is the smallest one that meets `target`."""
    above = [r for r in curve if r["recall"] >= target]
    if not above:
        return None
    return min(above, key=lambda r: r["recall"])


def time_at_recall(curve, target):
    """Linear interpolation of ms_per_query at a recall target. None if unreachable."""
    rows = sorted(curve, key=lambda r: r["recall"])
    above = [r for r in rows if r["recall"] >= target]
    below = [r for r in rows if r["recall"] < target]
    if not above:
        return None
    if not below:
        return float(above[0]["ms_per_query"])
    lo, hi = below[-1], above[0]
    if hi["recall"] == lo["recall"]:
        return float(hi["ms_per_query"])
    t = (target - lo["recall"]) / (hi["recall"] - lo["recall"])
    return float(lo["ms_per_query"] + t * (hi["ms_per_query"] - lo["ms_per_query"]))


# ===========================================================================
# Sweep — one curve per (index, k), with N_RUNS for std reporting
# ===========================================================================

def recall_time_curve(idx, label, xq, gt, k, factory, param_values,
                      n_warmup=3, n_runs=N_RUNS):
    """
    Sweep search parameter, batch-time the full xq across n_runs, report mean ± std.
    Returns list of dicts sorted by ascending recall.
    """
    nq = xq.shape[0]
    rows = []

    for param in param_values:
        search_fn = factory(param)
        for _ in range(n_warmup):
            search_fn(idx, xq[: min(5, nq)], k)

        run_times = []
        I_last = None
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _, I_last = search_fn(idx, xq, k)
            run_times.append(time.perf_counter() - t0)
        run_times = np.asarray(run_times)

        recall = compute_recall_at_k(I_last, gt, k)
        mean_t = float(run_times.mean())
        std_t  = float(run_times.std())
        ms_per_q = (mean_t / nq) * 1000.0
        ms_std   = (std_t  / nq) * 1000.0
        qps = nq / mean_t if mean_t > 0 else 0.0
        # qps std via 1st-order propagation: σ_qps ≈ qps * σ_t / t.
        qps_std = qps * (std_t / mean_t) if mean_t > 0 else 0.0

        rows.append({
            "param": float(param) if isinstance(param, float) else int(param),
            "recall": round(recall, 6),
            "ms_per_query":     round(ms_per_q, 6),
            "ms_per_query_std": round(ms_std,   6),
            "qps":              round(qps,      2),
            "qps_std":          round(qps_std,  2),
            "n_runs": int(n_runs),
        })
        print(f"  {label} ({param}): recall@{k}={recall:.4f}, "
              f"ms/q={ms_per_q:.4f}±{ms_std:.4f}, qps={qps:.0f}±{qps_std:.0f}")

    rows.sort(key=lambda r: r["recall"])
    return rows


# ===========================================================================
# Dataset features
# ===========================================================================

def compute_dataset_features(xb, xq, sample_n=10_000, k_lid=20):
    rng = np.random.default_rng(0)
    n, d = xb.shape
    sample_idx = rng.choice(n, size=min(sample_n, n), replace=False)
    sample = np.ascontiguousarray(xb[sample_idx], dtype=np.float32)

    D, _ = faiss.knn(sample, sample, k_lid + 1, metric=faiss.METRIC_L2)
    D = np.sqrt(np.maximum(D[:, 1 : k_lid + 1], 0))
    rk = D[:, -1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = D / rk
        log = np.log(np.clip(ratios, 1e-12, None))
    lid_per_pt = -1.0 / (log[:, :-1].mean(axis=1) + 1e-12)
    lid_mle = float(np.median(lid_per_pt[np.isfinite(lid_per_pt)]))

    pairs = rng.choice(sample.shape[0], size=(min(2000, sample.shape[0]), 2))
    diffs = sample[pairs[:, 0]] - sample[pairs[:, 1]]
    pdist = np.sqrt((diffs ** 2).sum(axis=1))
    pd_mean, pd_std = float(pdist.mean()), float(pdist.std())

    try:
        km = faiss.Kmeans(d, 16, niter=10, verbose=False, seed=0)
        km.train(sample)
        _, idx = km.index.search(sample, 1)
        c = km.centroids[idx.ravel()]
        inertia_16 = float(((sample - c) ** 2).sum())
        inertia_1  = float(((sample - sample.mean(axis=0)) ** 2).sum())
        clusterability = inertia_16 / max(inertia_1, 1e-12)
    except Exception:
        clusterability = -1.0

    return {
        "n": int(n),
        "d": int(d),
        "nq": int(xq.shape[0]),
        "lid_mle": lid_mle,
        "pdist_mean": pd_mean,
        "pdist_std": pd_std,
        "kmeans_inertia_ratio_16": clusterability,
    }


# ===========================================================================
# Per-index benchmarks
# ===========================================================================

def run_robustness(idx, kind, xq, gt, k=20):
    factory, _, _ = SEARCH_FACTORY[kind]
    if kind == "suco":
        search_fn = factory(ROBUSTNESS_CANDIDATE_RATIO)
    else:
        search_fn = factory(ROBUSTNESS_EFSEARCH)
    t0 = time.time()
    _, I = search_fn(idx, xq, k)
    elapsed = time.time() - t0
    pqr = per_query_recall(I, gt, k)
    return {
        "k": k,
        "param": ROBUSTNESS_CANDIDATE_RATIO if kind == "suco" else ROBUSTNESS_EFSEARCH,
        "param_name": "candidate_ratio" if kind == "suco" else "efSearch",
        "mean_recall":   round(float(pqr.mean()), 4),
        "median_recall": round(float(np.median(pqr)), 4),
        "min_recall":    round(float(pqr.min()), 4),
        "max_recall":    round(float(pqr.max()), 4),
        "q25_recall":    round(float(np.percentile(pqr, 25)), 4),
        "q75_recall":    round(float(np.percentile(pqr, 75)), 4),
        "ms_per_query":  round(elapsed * 1000 / xq.shape[0], 4),
    }


def run_hard_robustness(idx, kind, xb, xq, gt, k=20, recall_target=0.90, recall_curve=None):
    """Stratify queries by GT k-th distance; report recall on hard 10% vs easy 10%."""
    factory, params, _ = SEARCH_FACTORY[kind]
    if recall_curve is None:
        recall_curve = recall_time_curve(idx, kind, xq, gt, k, factory, params, n_runs=1)
    chosen = pick_param_for_recall(recall_curve, recall_target)
    if chosen is None:
        chosen = recall_curve[-1]
    search_fn = factory(chosen["param"])
    _, I = search_fn(idx, xq, k)
    pqr = per_query_recall(I, gt, k)

    nq = xq.shape[0]
    k_gt = min(k, gt.shape[1])
    hardness = np.zeros(nq)
    for i in range(nq):
        ids = gt[i, :k_gt]
        ids = ids[ids >= 0]
        if len(ids) == 0:
            continue
        d = ((xb[ids] - xq[i]) ** 2).sum(axis=1)
        hardness[i] = float(d.max())  # k-th nearest distance

    p_easy = np.percentile(hardness, 100 - HARD_QUERY_PCTILE)  # <= bottom 10%
    p_hard = np.percentile(hardness, HARD_QUERY_PCTILE)         # >= top 10%
    easy_mask = hardness <= p_easy
    hard_mask = hardness >= p_hard

    def _stats(mask):
        if not mask.any():
            return {"n": 0}
        sub = pqr[mask]
        return {
            "n": int(mask.sum()),
            "mean_recall":   round(float(sub.mean()), 4),
            "median_recall": round(float(np.median(sub)), 4),
            "min_recall":    round(float(sub.min()), 4),
            "p10_recall":    round(float(np.percentile(sub, 10)), 4),
            "p90_recall":    round(float(np.percentile(sub, 90)), 4),
        }

    return {
        "param": chosen["param"],
        "achieved_recall": chosen["recall"],
        "target_recall":   recall_target,
        "k": k,
        "overall_mean_recall": round(float(pqr.mean()), 4),
        "easy": _stats(easy_mask),
        "hard": _stats(hard_mask),
    }


def run_latency_tail(idx, kind, xq, gt, k=20, recall_curve=None):
    """Per-query p50/p95/p99/p99.9 at each recall target."""
    factory, params, _ = SEARCH_FACTORY[kind]
    if recall_curve is None:
        recall_curve = recall_time_curve(idx, kind, xq, gt, k, factory, params, n_runs=1)
    out = {}
    for target in LATENCY_RECALL_TARGETS:
        chosen = pick_param_for_recall(recall_curve, target)
        if chosen is None:
            continue
        search_fn = factory(chosen["param"])
        for _ in range(5):
            search_fn(idx, xq[:5], k)
        times = per_query_latencies(search_fn, idx, xq, k)
        q = latency_quantiles(times)
        q["param"] = chosen["param"]
        q["achieved_recall"] = chosen["recall"]
        q["target_recall"] = target
        out[f"r{int(round(target * 100))}"] = q
        print(f"  {kind} latency @recall≥{target}: p50={q['p50']:.3f} "
              f"p95={q['p95']:.3f} p99={q['p99']:.3f} p999={q['p999']:.3f} ms")
    return out


def run_cold_warm(idx, kind, xq, gt, k=20, recall_curve=None):
    """Cold-cache (evict between queries) vs steady-state warm latency at recall≥0.95."""
    factory, params, _ = SEARCH_FACTORY[kind]
    if recall_curve is None:
        recall_curve = recall_time_curve(idx, kind, xq, gt, k, factory, params, n_runs=1)
    chosen = pick_param_for_recall(recall_curve, COLDWARM_RECALL_TARGET)
    if chosen is None:
        chosen = recall_curve[-1]
    search_fn = factory(chosen["param"])

    page_flushed = flush_os_cache()
    cold_times = []
    for i in range(min(COLDWARM_NUM_COLD, xq.shape[0])):
        flush_os_cache()
        evict_caches()
        q = xq[i:i+1]
        t0 = time.perf_counter()
        search_fn(idx, q, k)
        cold_times.append((time.perf_counter() - t0) * 1000.0)

    for _ in range(20):
        search_fn(idx, xq[:50], k)
    warm_times = []
    n_warm = min(COLDWARM_NUM_WARM, xq.shape[0])
    for i in range(n_warm):
        q = xq[i:i+1]
        t0 = time.perf_counter()
        search_fn(idx, q, k)
        warm_times.append((time.perf_counter() - t0) * 1000.0)

    cold = np.asarray(cold_times)
    warm = np.asarray(warm_times)
    return {
        "param": chosen["param"],
        "achieved_recall": chosen["recall"],
        "target_recall":   COLDWARM_RECALL_TARGET,
        "page_cache_flushed": bool(page_flushed),
        "evict_buffer_mb": COLD_EVICT_MB,
        "cold_first_ms":  round(float(cold[0]), 6),
        "cold_mean_ms":   round(float(cold.mean()), 6),
        "cold_p95_ms":    round(float(np.percentile(cold, 95)), 6),
        "warm_mean_ms":   round(float(warm.mean()), 6),
        "warm_p95_ms":    round(float(np.percentile(warm, 95)), 6),
        "cold_warm_ratio": round(float(cold.mean() / max(warm.mean(), 1e-9)), 3),
        "n_cold": int(len(cold)),
        "n_warm": int(len(warm)),
    }


def run_mre(idx, kind, xb, xq, gt, k=MRE_K, recall_curve=None):
    """Mean Relative Error at each recall target (SuCo Fig 11/12 style)."""
    factory, params, _ = SEARCH_FACTORY[kind]
    if recall_curve is None:
        recall_curve = recall_time_curve(idx, kind, xq, gt, k, factory, params, n_runs=1)
    out = {}
    for target in LATENCY_RECALL_TARGETS:
        chosen = pick_param_for_recall(recall_curve, target)
        if chosen is None:
            continue
        search_fn = factory(chosen["param"])
        D, I = search_fn(idx, xq, k)
        mre = compute_mre_at_k(I, xb, xq, gt, k=k)
        ar  = approx_ratio_at_k(D, xb, xq, gt, k=k)
        out[f"r{int(round(target * 100))}"] = {
            "k": k,
            "param": chosen["param"],
            "achieved_recall": chosen["recall"],
            "target_recall":   target,
            "mre": round(float(mre), 6),
            "approx_ratio_sq_l2": round(float(ar), 6),
        }
        print(f"  {kind} MRE @recall≥{target}: {mre:.4f} (approx-ratio^2={ar:.4f})")
    return out


# ===========================================================================
# Dataset-level benchmarks
# ===========================================================================

def run_unseen_robustness(xb, xq, data_dir, dataset, index_types, k=UNSEEN_K):
    """
    SHG-style robustness: hold out UNSEEN_FRAC of base, rebuild each index without
    them, then query with the held-out vectors. Returns recall distribution per index.
    """
    rng = np.random.default_rng(42)
    n = xb.shape[0]
    n_held = int(round(n * UNSEEN_FRAC))
    n_held = min(n_held, UNSEEN_MAX_QUERIES)
    if n_held < 100:
        return {"skipped": "dataset too small for unseen split"}
    perm = rng.permutation(n)
    held_idx = perm[:n_held]
    keep_idx = np.setdiff1d(perm, held_idx, assume_unique=True)
    xb_keep = np.ascontiguousarray(xb[keep_idx])
    xq_held = np.ascontiguousarray(xb[held_idx])
    print(f"\n  Unseen split: keep={len(keep_idx)}, query/held={len(held_idx)}")

    print(f"  Computing brute-force GT for unseen queries (k={UNSEEN_GT_K})...")
    _, gt_held = faiss.knn(xq_held, xb_keep, UNSEEN_GT_K, metric=faiss.METRIC_L2)
    gt_held = gt_held.astype(np.int32)

    out = {"n_held": int(n_held), "n_keep": int(len(keep_idx)), "k": int(k), "per_index": {}}
    d = xb_keep.shape[1]
    for kind in index_types:
        if kind not in BUILDERS:
            continue
        label, builder = BUILDERS[kind]
        try:
            print(f"\n  --- Rebuilding {label} on kept set ---")
            idx, _ = builder(xb_keep, d)
            factory, _, _ = SEARCH_FACTORY[kind]
            if kind == "suco":
                search_fn = factory(ROBUSTNESS_CANDIDATE_RATIO)
            else:
                search_fn = factory(ROBUSTNESS_EFSEARCH)
            _, I = search_fn(idx, xq_held, k)
            pqr = per_query_recall(I, gt_held, k)
            out["per_index"][label] = {
                "param": ROBUSTNESS_CANDIDATE_RATIO if kind == "suco" else ROBUSTNESS_EFSEARCH,
                "mean_recall":   round(float(pqr.mean()),       4),
                "median_recall": round(float(np.median(pqr)),    4),
                "min_recall":    round(float(pqr.min()),         4),
                "p10_recall":    round(float(np.percentile(pqr, 10)), 4),
                "p25_recall":    round(float(np.percentile(pqr, 25)), 4),
                "p75_recall":    round(float(np.percentile(pqr, 75)), 4),
                "p90_recall":    round(float(np.percentile(pqr, 90)), 4),
            }
            del idx
            gc.collect()
        except Exception as e:
            print(f"  {label} unseen FAILED: {e}")
            traceback.print_exc()
            out["per_index"][label] = {"error": str(e)}
    return out


def derive_pareto(all_results):
    """Pareto upper envelope on each recall_k* curve, per index."""
    out = {}
    for k in RECALL_KS:
        curves = all_results.get(f"recall_k{k}", {}) or {}
        out[f"recall_k{k}"] = {
            label: pareto_frontier(rows) for label, rows in curves.items()
        }
    return out


def derive_time_at_recall(all_results):
    """Time-at-recall + speedup-vs-baseline tables for every k and target."""
    out = {}
    for k in RECALL_KS:
        curves = all_results.get(f"recall_k{k}", {}) or {}
        if not curves:
            continue
        baseline = curves.get(SPEEDUP_BASELINE_LABEL)
        per_k = {}
        for target in TIME_AT_RECALL_TARGETS:
            entry = {}
            base_ms = time_at_recall(baseline, target) if baseline else None
            for label, rows in curves.items():
                ms = time_at_recall(rows, target)
                if ms is None:
                    entry[label] = {"ms_per_query": None, "speedup_vs_" + SPEEDUP_BASELINE_LABEL: None,
                                    "qps": None}
                    continue
                speedup = (base_ms / ms) if (base_ms is not None and ms > 0) else None
                entry[label] = {
                    "ms_per_query": round(ms, 6),
                    "qps":          round(1000.0 / ms, 2) if ms > 0 else None,
                    f"speedup_vs_{SPEEDUP_BASELINE_LABEL}":
                        round(speedup, 3) if speedup is not None else None,
                }
            per_k[f"r{int(round(target * 100))}"] = entry
        out[f"recall_k{k}"] = per_k
    return out


# ===========================================================================
# Main per-dataset driver
# ===========================================================================

def run_benchmarks(dataset, benchmarks, index_types, data_dir, index_dir, output_dir):
    print(f"\n{'#'*70}\n# Dataset: {dataset.upper()}\n{'#'*70}")

    print(f"\nLoading {dataset}...")
    t0 = time.time()
    xb, xq, gt = load_dataset(dataset, data_dir)
    print(f"  Loaded in {time.time()-t0:.1f}s: xb={xb.shape}, xq={xq.shape}, gt={gt.shape}")

    if gt.max() >= xb.shape[0]:
        print(f"  GT has IDs up to {gt.max()} but base has only {xb.shape[0]} — recomputing")
        gt = compute_ground_truth(xb, xq, k=gt.shape[1])

    d, n = int(xb.shape[1]), int(xb.shape[0])

    out_path = os.path.join(output_dir, f"results_{dataset}.json")
    all_results = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}
    all_results.update({"dataset": dataset, "n": n, "d": d, "nq": int(xq.shape[0])})

    # ----- Dataset features -----
    if "features" in benchmarks:
        print(f"\n{'='*70}\nBENCHMARK: features - {dataset}\n{'='*70}")
        all_results["features"] = compute_dataset_features(xb, xq)
        print(f"  {all_results['features']}")

    prev_construction = all_results.get("construction", {}) or {}
    construction_results = dict(prev_construction)
    recall_curves = {
        f"recall_k{k}": dict(all_results.get(f"recall_k{k}", {}) or {})
        for k in RECALL_KS
    }
    robustness_results = dict(all_results.get("robustness", {}) or {})
    hard_results       = dict(all_results.get("hard_robustness", {}) or {})
    latency_results    = dict(all_results.get("latency_tail", {}) or {})
    coldwarm_results   = dict(all_results.get("cold_warm", {}) or {})
    mre_results        = dict(all_results.get("mre", {}) or {})

    for kind in index_types:
        if kind not in BUILDERS:
            print(f"  Unknown index type {kind!r}, skipping")
            continue
        label, builder = BUILDERS[kind]
        idx_path = os.path.join(index_dir, f"{dataset}_{kind}.idx")
        idx = None
        build_time = -1.0
        peak_rss_mb = -1.0

        if os.path.exists(idx_path):
            print(f"\n--- Loading {label} from {idx_path} ---")
            try:
                idx = faiss.read_index(idx_path)
                print(f"  {label}: loaded")
            except Exception as e:
                print(f"  Failed to load, will rebuild: {e}")
                idx = None

        if idx is None:
            print(f"\n--- Building {label} ---")
            peak_before = _peak_rss_mb()
            try:
                idx, build_time = builder(xb, d)
                peak_rss_mb = max(0.0, _peak_rss_mb() - peak_before)
                try:
                    faiss.write_index(idx, idx_path)
                    print(f"  Saved to {idx_path}")
                except Exception as e:
                    print(f"  Could not save: {e}")
            except Exception as e:
                print(f"  {label}: BUILD FAILED — {e}")
                traceback.print_exc()
                construction_results[label] = {
                    "build_time_s": -1, "memory_mb": -1, "size_mb": -1,
                    "build_failed": str(e),
                }
                continue

        size_mb = index_size_mb(idx)
        if build_time < 0 and label in prev_construction:
            build_time = prev_construction[label].get("build_time_s", -1)
        if peak_rss_mb < 0 and label in prev_construction:
            peak_rss_mb = prev_construction[label].get("memory_mb", -1)
        construction_results[label] = {
            "build_time_s": round(build_time, 2) if build_time >= 0 else -1,
            "memory_mb":    round(peak_rss_mb, 2) if peak_rss_mb >= 0 else -1,
            "size_mb":      round(size_mb, 2) if size_mb >= 0 else -1,
        }

        factory, params, _ = SEARCH_FACTORY[kind]

        # Recall curves (with std over N_RUNS)
        for k in RECALL_KS:
            bench_name = f"recall_k{k}"
            if bench_name in benchmarks:
                print(f"\n--- {label} recall@{k} curve (n_runs={N_RUNS}) ---")
                recall_curves[bench_name][label] = recall_time_curve(
                    idx, label, xq, gt, k, factory, params,
                )

        # Re-use the k=20 curve for downstream operating-point benchmarks if available.
        k20_curve = recall_curves["recall_k20"].get(label)

        if "robustness" in benchmarks:
            print(f"\n--- {label} robustness (k=20) ---")
            try:
                robustness_results[label] = run_robustness(idx, kind, xq, gt, k=20)
                print(f"  {label}: {robustness_results[label]}")
            except Exception as e:
                print(f"  {label} robustness FAILED: {e}")

        if "hard_robustness" in benchmarks:
            print(f"\n--- {label} hard-query robustness ---")
            try:
                hard_results[label] = run_hard_robustness(
                    idx, kind, xb, xq, gt, k=20, recall_curve=k20_curve,
                )
                print(f"  {label}: easy={hard_results[label]['easy']} "
                      f"hard={hard_results[label]['hard']}")
            except Exception as e:
                print(f"  {label} hard-robustness FAILED: {e}")
                traceback.print_exc()

        if "latency_tail" in benchmarks:
            print(f"\n--- {label} latency tail ---")
            try:
                latency_results[label] = run_latency_tail(
                    idx, kind, xq, gt, k=20, recall_curve=k20_curve,
                )
            except Exception as e:
                print(f"  {label} latency_tail FAILED: {e}")
                traceback.print_exc()

        if "cold_warm" in benchmarks:
            print(f"\n--- {label} cold/warm cache ---")
            try:
                coldwarm_results[label] = run_cold_warm(
                    idx, kind, xq, gt, k=20, recall_curve=k20_curve,
                )
                print(f"  {label}: {coldwarm_results[label]}")
            except Exception as e:
                print(f"  {label} cold_warm FAILED: {e}")
                traceback.print_exc()

        if "mre" in benchmarks:
            print(f"\n--- {label} MRE ---")
            try:
                mre_results[label] = run_mre(
                    idx, kind, xb, xq, gt, k=MRE_K, recall_curve=k20_curve,
                )
            except Exception as e:
                print(f"  {label} MRE FAILED: {e}")
                traceback.print_exc()

        del idx
        gc.collect()

    # ----- Print summaries + assemble JSON -----
    if "construction" in benchmarks:
        print(f"\n{'='*70}\nBENCHMARK: construction - {dataset}\n{'='*70}")
        for label, stats in construction_results.items():
            bt, mem, sz = stats["build_time_s"], stats["memory_mb"], stats["size_mb"]
            print(f"  {label}: build={bt}s, peak_rss_delta={mem}MB, serialized={sz}MB")
        all_results["construction"] = construction_results

    for k in RECALL_KS:
        bench_name = f"recall_k{k}"
        if bench_name in benchmarks:
            all_results[bench_name] = recall_curves[bench_name]

    if "robustness"        in benchmarks: all_results["robustness"]        = robustness_results
    if "hard_robustness"   in benchmarks: all_results["hard_robustness"]   = hard_results
    if "latency_tail"      in benchmarks: all_results["latency_tail"]      = latency_results
    if "cold_warm"         in benchmarks: all_results["cold_warm"]         = coldwarm_results
    if "mre"               in benchmarks: all_results["mre"]               = mre_results

    # Dataset-level benchmarks (last so curves are populated).
    if "unseen_robustness" in benchmarks:
        print(f"\n{'='*70}\nBENCHMARK: unseen_robustness - {dataset}\n{'='*70}")
        try:
            all_results["unseen_robustness"] = run_unseen_robustness(
                xb, xq, data_dir, dataset, index_types, k=UNSEEN_K,
            )
        except Exception as e:
            print(f"  unseen_robustness FAILED: {e}")
            traceback.print_exc()
            all_results["unseen_robustness"] = {"error": str(e)}

    if "pareto" in benchmarks:
        print(f"\n{'='*70}\nBENCHMARK: pareto - {dataset}\n{'='*70}")
        all_results["pareto"] = derive_pareto(all_results)

    if "time_at_recall" in benchmarks:
        print(f"\n{'='*70}\nBENCHMARK: time_at_recall - {dataset}\n{'='*70}")
        all_results["time_at_recall"] = derive_time_at_recall(all_results)

    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    del xb, xq, gt
    gc.collect()
    return all_results


# ===========================================================================
# CLI
# ===========================================================================

def main():
    global N_RUNS
    ap = argparse.ArgumentParser(description="Router-training benchmark suite")
    ap.add_argument("--data-dir",   default="/Users/dhm/Documents/data")
    ap.add_argument("--index-dir",  default="/Users/dhm/Documents/indices")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dataset",    default="all",
                    choices=ALL_DATASETS + ["all"])
    ap.add_argument("--benchmark",  nargs="+", default=["all"],
                    choices=ALL_BENCHMARKS + ["all"])
    ap.add_argument("--index-type", nargs="+", default=DEFAULT_INDEX_TYPES,
                    choices=ALL_INDEX_TYPES + ["all"])
    ap.add_argument("--n-runs", type=int, default=N_RUNS,
                    help="Repetitions for QPS/recall curves (std reporting). Default 3.")
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results_router")

    N_RUNS = max(1, int(args.n_runs))

    benchmarks  = ALL_BENCHMARKS    if "all" in args.benchmark  else args.benchmark
    index_types = ALL_INDEX_TYPES   if "all" in args.index_type else args.index_type
    datasets    = ALL_DATASETS      if args.dataset == "all"    else [args.dataset]

    os.makedirs(args.index_dir,  exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Data dir:   {args.data_dir}")
    print(f"Index dir:  {args.index_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Datasets:   {datasets}")
    print(f"Benchmarks: {benchmarks}")
    print(f"Indexes:    {index_types}")
    print(f"N runs:     {N_RUNS}")

    for ds in datasets:
        try:
            run_benchmarks(ds, benchmarks, index_types,
                           args.data_dir, args.index_dir, args.output_dir)
        except Exception as e:
            print(f"\nERROR processing {ds}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
