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

Benchmarks (paper-style, no per-paper ablations):
  construction   — build time + memory + index size
  recall_k1      — QPS vs Recall@1   curve (full efSearch / candidate_ratio sweep)
  recall_k10     — QPS vs Recall@10  curve
  recall_k20     — QPS vs Recall@20  curve
  recall_k50     — QPS vs Recall@50  curve
  recall_k100    — QPS vs Recall@100 curve
  robustness     — per-query recall@20 distribution at fixed search budget
  features       — dataset features for router (n, d, LID, pdist moments, kmeans inertia)

All results land in {output_dir}/results_<dataset>.json.
Indices are persisted to {index_dir}/<dataset>_<index>.idx and reloaded if present.

Usage:
  python benchs/bench_router_paper.py --dataset sift1m --benchmark all
  python benchs/bench_router_paper.py --dataset all --benchmark all
  python benchs/bench_router_paper.py --dataset gist1m --benchmark recall_k10 recall_k20
"""

import argparse
import gc
import json
import os
import resource as _resource
import struct
import subprocess
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

# SHG (report 2, §4.2)
SHG_M = 48
SHG_EFC = 128

# CSPG (report 3, §4.2)
CSPG_M = 32
CSPG_EFC = 128
CSPG_NUM_PARTITIONS = 2
CSPG_LAMBDA = 0.5
CSPG_EF1 = 1

# HNSW reference baselines — same per-graph budget as the indices they sit under.
HNSW32_M = 32; HNSW32_EFC = 128   # matches CSPG substrate
HNSW48_M = 48; HNSW48_EFC = 128   # matches SHG  substrate


# Search-parameter sweeps for QPS-recall curves.
# Graph-based (SHG, CSPG): efSearch sweep matching the SHG/CSPG papers.
EF_SEARCH_VALUES = [
    10, 15, 20, 30, 40, 60, 80, 100, 150,
    200, 300, 400, 600, 800, 1000, 1500, 2000,
]
# SuCo: sweep the candidate_ratio (β) — the dominant accuracy/speed knob.
SUCO_CANDIDATE_RATIO_VALUES = [
    0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2,
]

# Recall@k values to record for each operating point.
RECALL_KS = (1, 10, 20, 50, 100)
SEARCH_K = max(RECALL_KS)

# Robustness experiment: fixed search budget per index family.
ROBUSTNESS_EFSEARCH = 200
ROBUSTNESS_CANDIDATE_RATIO = 0.005  # SuCo paper default

ALL_DATASETS = [
    "sift1m", "sift10m", "gist1m",
    "deep1m", "deep10m",
    "spacev10m",
    "msong", "enron", "openai1m",
    "msturing10m", "uqv",
]
ALL_BENCHMARKS = [
    "construction",
    "recall_k1", "recall_k10", "recall_k20", "recall_k50", "recall_k100",
    "robustness",
    "features",
]
ALL_INDEX_TYPES = ["suco", "shg", "cspg", "hnsw32", "hnsw48"]
DEFAULT_INDEX_TYPES = ["suco", "shg", "cspg", "hnsw32", "hnsw48"]


# ---------------------------------------------------------------------------
# Memory measurement (resource.getrusage tracks peak automatically)
# ---------------------------------------------------------------------------

import platform as _platform
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


def _prepare_deep1b(data_dir, nb, nt):
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepare_deep1m.py")
    cmd = [
        sys.executable,
        script_path,
        "--data-dir", data_dir,
        "--nb", str(nb),
        "--nt", str(nt),
    ]
    print("  Preparing Deep1B data:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ===========================================================================
# Dataset loaders — one branch per dataset, paper-style
# ===========================================================================

def load_dataset(name, data_dir):
    """Returns (xb, xq, gt). xt (training set) defaults to xb when not separate."""
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
        _prepare_deep1b(data_dir, nb=1_000_000, nt=500_000)
        p = os.path.join(data_dir, "deep1b")
        xb = read_fvecs(os.path.join(p, "base.fvecs"), n=1_000_000)
        xq = read_fvecs(os.path.join(p, "deep1B_queries.fvecs"), n=10_000)
        gt = read_ivecs(os.path.join(p, "deep1M_groundtruth.ivecs"))
        if gt.shape[0] > xq.shape[0]:
            gt = gt[: xq.shape[0]]
        return xb, xq, gt

    if name == "deep10m":
        _prepare_deep1b(data_dir, nb=10_000_000, nt=1_000_000)
        p = os.path.join(data_dir, "deep1b")
        xb = read_fvecs(os.path.join(p, "deep10M.fvecs"))
        xq = read_fvecs(os.path.join(p, "deep1B_queries.fvecs"), n=10_000)
        gt = read_ivecs(os.path.join(p, "deep10M_groundtruth.ivecs"))
        if gt.shape[0] > xq.shape[0]:
            gt = gt[: xq.shape[0]]
        return xb, xq, gt

    if name == "spacev10m":
        # int8 base (truncated to 10M) + int8 queries; cast to float32 (d=100).
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
# Index builders — one per index family, paper-style
# ===========================================================================

def _pick_suco_nsubspaces(d, preferred=SUCO_NSUBSPACES_PREFERRED):
    """SuCo requires d % n == 0 AND (d/n) % 2 == 0. Pick largest valid n ≤ preferred."""
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
# Recall metrics
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


# ===========================================================================
# Search-factory helpers (paper-style)
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
                # No SearchParameters bindings — fall back to mutating the index.
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
# Sweep — one curve per (index, k)
# ===========================================================================

def recall_time_curve(idx, label, xq, gt, k, factory, param_values, n_warmup=3):
    """
    Sweep search parameter, batch-time the full xq, record (recall, ms/q, qps).
    Uses FAISS's default thread count (no omp_set_num_threads).
    Returns list of dicts sorted by ascending recall.
    """
    nq = xq.shape[0]
    rows = []

    for param in param_values:
        search_fn = factory(param)
        for _ in range(n_warmup):
            search_fn(idx, xq[: min(5, nq)], k)

        t0 = time.perf_counter()
        _, I = search_fn(idx, xq, k)
        total_time = time.perf_counter() - t0

        recall = compute_recall_at_k(I, gt, k)
        ms_per_q = (total_time / nq) * 1000.0
        rows.append({
            "param": float(param) if isinstance(param, float) else int(param),
            "recall": round(recall, 6),
            "ms_per_query": round(ms_per_q, 6),
            "qps": round(nq / total_time, 2),
        })
        print(f"  {label} ({param}): recall@{k}={recall:.4f}, "
              f"ms/q={ms_per_q:.4f}, qps={rows[-1]['qps']:.0f}")

    rows.sort(key=lambda r: r["recall"])
    return rows


# ===========================================================================
# Dataset features (router input)
# ===========================================================================

def compute_dataset_features(xb, xq, sample_n=10_000, k_lid=20):
    rng = np.random.default_rng(0)
    n, d = xb.shape
    sample_idx = rng.choice(n, size=min(sample_n, n), replace=False)
    sample = np.ascontiguousarray(xb[sample_idx], dtype=np.float32)

    # LID via MLE on k-NN distances within the sample.
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
# Robustness: per-query recall@20 distribution at fixed search budget
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
    all_results = {"dataset": dataset, "n": n, "d": d, "nq": int(xq.shape[0])}

    # ----- Dataset features -----
    if "features" in benchmarks:
        print(f"\n{'='*70}\nBENCHMARK: features - {dataset}\n{'='*70}")
        all_results["features"] = compute_dataset_features(xb, xq)
        print(f"  {all_results['features']}")

    # Carry forward construction stats from a prior run when reloading indices.
    prev_path = os.path.join(output_dir, f"results_{dataset}.json")
    prev_construction = {}
    if os.path.exists(prev_path):
        try:
            with open(prev_path) as f:
                prev_construction = json.load(f).get("construction", {})
        except Exception:
            pass

    construction_results = {}
    recall_curves = {f"recall_k{k}": {} for k in RECALL_KS}
    robustness_results = {}

    for kind in index_types:
        if kind not in BUILDERS:
            print(f"  Unknown index type {kind!r}, skipping")
            continue
        label, builder = BUILDERS[kind]
        idx_path = os.path.join(index_dir, f"{dataset}_{kind}.idx")
        idx = None
        build_time = -1.0
        peak_rss_mb = -1.0

        # Try to reload
        if os.path.exists(idx_path):
            print(f"\n--- Loading {label} from {idx_path} ---")
            try:
                idx = faiss.read_index(idx_path)
                print(f"  {label}: loaded")
            except Exception as e:
                print(f"  Failed to load, will rebuild: {e}")
                idx = None

        # Build if needed
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

        # Construction stats
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

        # Recall curves
        factory, params, _ = SEARCH_FACTORY[kind]
        for k in RECALL_KS:
            bench_name = f"recall_k{k}"
            if bench_name in benchmarks:
                print(f"\n--- {label} recall@{k} curve ---")
                recall_curves[bench_name][label] = recall_time_curve(
                    idx, label, xq, gt, k, factory, params,
                )

        # Robustness
        if "robustness" in benchmarks:
            print(f"\n--- {label} robustness (k=20) ---")
            try:
                robustness_results[label] = run_robustness(idx, kind, xq, gt, k=20)
                print(f"  {label}: {robustness_results[label]}")
            except Exception as e:
                print(f"  {label} robustness FAILED: {e}")

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

    if "robustness" in benchmarks:
        all_results["robustness"] = robustness_results

    # ----- Save -----
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"results_{dataset}.json")
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
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results_router")

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

    for ds in datasets:
        try:
            run_benchmarks(ds, benchmarks, index_types,
                           args.data_dir, args.index_dir, args.output_dir)
        except Exception as e:
            print(f"\nERROR processing {ds}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
