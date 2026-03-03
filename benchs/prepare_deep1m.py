#!/usr/bin/env python3
"""
benchs/prepare_deep1m.py

Extracts the vectors needed for the Deep1M benchmark from the raw Deep1B
shard files (base00, learn00) and writes properly-sized .fvecs files that
FAISS's DatasetDeep1B loader expects.

Files produced inside <data-dir>/deep1b/:
  base.fvecs   – first 1 000 000 vectors from base00   (~370 MB)
  learn.fvecs  – first   500 000 vectors from learn00  (~185 MB)

Queries and ground-truth (deep1B_queries.fvecs, deep1M_groundtruth.ivecs)
are kept as-is — they are already valid .fvecs / .ivecs files.

Usage:
    python benchs/prepare_deep1m.py --data-dir /home/dhm/data/
    python benchs/prepare_deep1m.py --data-dir /home/dhm/data/ --nb 1000000 --nt 500000
"""

import argparse
import os
import sys
import struct

# fvecs format: each vector is stored as
#   [int32: d] [float32 × d]
# Deep1B uses d=96, so each record is 4 + 96*4 = 388 bytes.

D             = 96
BYTES_PER_VEC = 4 + D * 4   # 388


def extract_fvecs(src_path: str, dst_path: str, n_vectors: int) -> None:
    """
    Read the first `n_vectors` vectors from `src_path` (fvecs format, d=96)
    using plain file I/O (no memmap, safe with partial-download files) and
    write them to `dst_path`.
    """
    src_size  = os.path.getsize(src_path)
    available = src_size // BYTES_PER_VEC

    if available < n_vectors:
        print(f"  WARNING: {src_path} only has {available:,} complete vectors "
              f"(requested {n_vectors:,}). Will extract {available:,}.",
              file=sys.stderr)
        n_vectors = available

    need_bytes = n_vectors * BYTES_PER_VEC
    print(f"  Reading {n_vectors:,} vectors ({need_bytes / 1024**3:.3f} GiB) "
          f"from {os.path.basename(src_path)} …")

    buf = bytearray(need_bytes)
    with open(src_path, "rb") as f:
        total_read = 0
        view = memoryview(buf)
        while total_read < need_bytes:
            chunk = min(256 * 1024 * 1024, need_bytes - total_read)  # 256 MiB at a time
            n = f.readinto(view[total_read: total_read + chunk])
            if n == 0:
                break
            total_read += n

    if total_read < need_bytes:
        raise IOError(f"Could only read {total_read:,} of {need_bytes:,} bytes")

    # Sanity-check: verify first vector's dimension header
    d_check = struct.unpack_from("<i", buf, 0)[0]
    if d_check != D:
        raise ValueError(
            f"Expected d={D} in first vector header, got {d_check}. "
            "File may not be in fvecs format."
        )

    print(f"  Writing {os.path.basename(dst_path)} …")
    with open(dst_path, "wb") as f:
        f.write(buf)

    written = os.path.getsize(dst_path)
    print(f"  Done → {dst_path}  ({written / 1024**2:.1f} MB, "
          f"{written // BYTES_PER_VEC:,} vectors)")


def main():
    p = argparse.ArgumentParser(
        description="Prepare Deep1M .fvecs files from raw Deep1B shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="/home/dhm/data/",
                   help="Root directory containing the deep1b/ sub-folder.")
    p.add_argument("--nb",  type=int, default=1_000_000,
                   help="Number of base (database) vectors to extract.")
    p.add_argument("--nt",  type=int, default=500_000,
                   help="Number of training vectors to extract.")
    args = p.parse_args()

    deep1b = os.path.join(args.data_dir, "deep1b")

    base_src  = os.path.join(deep1b, "base00")
    learn_src = os.path.join(deep1b, "learn00")
    base_dst  = os.path.join(deep1b, "base.fvecs")
    learn_dst = os.path.join(deep1b, "learn.fvecs")

    for src in (base_src, learn_src):
        if not os.path.exists(src):
            sys.exit(f"ERROR: not found: {src}")

    # Remove old symlinks if present so we can write real files
    for dst in (base_dst, learn_dst):
        if os.path.islink(dst):
            os.unlink(dst)
            print(f"  Removed symlink {dst}")

    print(f"\nExtracting base vectors  (nb={args.nb:,})")
    extract_fvecs(base_src, base_dst, args.nb)

    print(f"\nExtracting train vectors (nt={args.nt:,})")
    extract_fvecs(learn_src, learn_dst, args.nt)

    print("\nAll done. Verifying with FAISS loader …")
    try:
        import faiss
        from faiss.contrib.datasets import DatasetDeep1B, set_dataset_basedir
        set_dataset_basedir(args.data_dir)
        ds = DatasetDeep1B(nb=args.nb)
        xq = ds.get_queries()
        xb = ds.get_database()
        xt = ds.get_train()
        gt = ds.get_groundtruth(k=100)
        print(f"  queries : {xq.shape}  dtype={xq.dtype}")
        print(f"  base    : {xb.shape}  dtype={xb.dtype}")
        print(f"  train   : {xt.shape}  dtype={xt.dtype}")
        print(f"  gt      : {gt.shape}  dtype={gt.dtype}")
        print("\nDataset ready. Run the benchmark with:")
        print(f"  python benchs/bench_suco_deep1m.py --data-dir {args.data_dir}")
    except Exception as e:
        print(f"  Verification failed: {e}", file=sys.stderr)
        print("  Files were written but FAISS loader check failed.")


if __name__ == "__main__":
    main()
