#!/usr/bin/env python3
"""
benchs/prepare_deep1m.py

Extracts vectors from raw Deep1B shard files (base00, base01, …, learn00)
and writes a single large base.fvecs + learn.fvecs that FAISS's DatasetDeep1B
loader can read.  A single large file supports Deep1M, Deep10M and Deep100M
via DatasetDeep1B(nb=...) because the loader slices [:nb] from the mmap.

Chunk layout (each shard ≈ 9.9 GiB, 27.4 M vectors, d=96):
  base00          – already downloaded  → covers Deep1M + Deep10M
  base01/02/03    – download for 100M   → python benchs/downloadDeep1B.py
  learn00         – already downloaded  → training for all sizes

Files produced inside <data-dir>/deep1b/:
  base.fvecs  – first --nb vectors concatenated from base00, base01, …
  learn.fvecs – first --nt vectors from learn00

Queries and ground-truth are already valid; this script does not touch them.

Recommended invocations
-----------------------
# Deep1M only  (base00 already present → no extra downloads needed):
    python benchs/prepare_deep1m.py --data-dir /home/dhm/data/ \\
        --nb 1_000_000 --nt 500_000

# Deep10M  (base00 is sufficient):
    python benchs/prepare_deep1m.py --data-dir /home/dhm/data/ \\
        --nb 10_000_000 --nt 1_000_000

# Deep100M  (requires base00-base03; download base01-03 first):
#   python benchs/downloadDeep1B.py --data-dir /home/dhm/data/ --base-chunks 1 2 3
    python benchs/prepare_deep1m.py --data-dir /home/dhm/data/ \\
        --nb 100_000_000 --nt 5_000_000

# All-in-one (produces a 100M base.fvecs usable for 1M/10M/100M):
    python benchs/prepare_deep1m.py --data-dir /home/dhm/data/ \\
        --nb 100_000_000 --nt 5_000_000
"""

import argparse
import os
import struct
import sys

# ---------------------------------------------------------------------------
# fvecs constants for Deep1B (d=96)
# Each record: [int32 d][float32 × 96] = 4 + 384 = 388 bytes
# ---------------------------------------------------------------------------
D             = 96
BYTES_PER_VEC = 4 + D * 4   # 388

# Each raw base/learn shard contains this many complete vectors:
_VECTORS_PER_CHUNK = None   # computed lazily from actual file size


def _count_vectors_in_file(path: str) -> int:
    return os.path.getsize(path) // BYTES_PER_VEC


# ---------------------------------------------------------------------------
# Core: stream-copy vectors from src -> dst with range [start, end)
# ---------------------------------------------------------------------------

def _copy_vectors(src_path: str, dst_file, start: int, end: int) -> int:
    """
    Read vectors [start, end) from src_path (fvecs format) and write them
    to the already-open dst_file.  Returns number of vectors written.
    """
    n = end - start
    if n <= 0:
        return 0

    src_size  = os.path.getsize(src_path)
    available = src_size // BYTES_PER_VEC
    if start >= available:
        print(f"  WARNING: {os.path.basename(src_path)} has only {available:,} vectors; "
              f"start={start:,} is past end – skipping.", file=sys.stderr)
        return 0
    if start + n > available:
        old_n = n
        n = available - start
        print(f"  WARNING: only {n:,} vectors available (requested {old_n:,}).", file=sys.stderr)

    byte_start = start * BYTES_PER_VEC
    byte_count = n * BYTES_PER_VEC
    chunk_size = 256 * 1024 * 1024   # 256 MiB read buffer

    with open(src_path, "rb") as f:
        f.seek(byte_start)
        read_so_far = 0
        while read_so_far < byte_count:
            this_chunk = min(chunk_size, byte_count - read_so_far)
            buf = f.read(this_chunk)
            if not buf:
                break
            dst_file.write(buf)
            read_so_far += len(buf)

    if read_so_far < byte_count:
        raise IOError(f"Only read {read_so_far:,} of {byte_count:,} bytes from {src_path}")
    return n


# ---------------------------------------------------------------------------
# Build base.fvecs from one or more chunk files
# ---------------------------------------------------------------------------

def build_fvecs(
    chunk_paths: list,
    dst_path: str,
    n_total: int,
    label: str = "vectors",
) -> int:
    """
    Concatenate the first n_total vectors from the ordered list of chunk_paths
    into dst_path.  Returns the actual number of vectors written.
    """
    if os.path.islink(dst_path):
        os.unlink(dst_path)
        print(f"  Removed symlink {dst_path}")

    # Verify first vector header from the first chunk
    with open(chunk_paths[0], "rb") as f:
        d_check = struct.unpack_from("<i", f.read(4))[0]
    if d_check != D:
        raise ValueError(
            f"Expected d={D} in {chunk_paths[0]}, got {d_check}. "
            "File may not be fvecs format."
        )

    print(f"  Writing {n_total:,} {label} to {os.path.basename(dst_path)} …")
    written_total = 0
    remaining     = n_total

    with open(dst_path, "wb") as out:
        for path in chunk_paths:
            if remaining <= 0:
                break
            available = _count_vectors_in_file(path)
            to_copy   = min(remaining, available)
            print(f"    {os.path.basename(path):12s}  "
                  f"[0 … {to_copy:,})  of  {available:,}  available")
            written = _copy_vectors(path, out, start=0, end=to_copy)
            written_total += written
            remaining     -= written

    actual_bytes = os.path.getsize(dst_path)
    print(f"  Done → {dst_path}")
    print(f"         {actual_bytes / 1024**2:.1f} MB, {written_total:,} vectors")
    if written_total < n_total:
        print(f"  WARNING: requested {n_total:,} but only {written_total:,} available.",
              file=sys.stderr)
    return written_total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare Deep1M/10M/100M .fvecs files from raw Deep1B shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir", default="/Users/dhm/Documents/data/",
        help="Root directory containing the deep1b/ sub-folder.",
    )
    p.add_argument(
        "--nb", type=int, default=1_000_000,
        help=(
            "Number of base (database) vectors to extract. "
            "Use 1000000 (1M), 10000000 (10M), or 100000000 (100M). "
            "base00 alone covers up to 1M and 10M. "
            "100M requires base00–base03 (download with downloadDeep1B.py)."
        ),
    )
    p.add_argument(
        "--nt", type=int, default=500_000,
        help=(
            "Number of training vectors to extract from learn00. "
            "learn00 contains ~27.4M vectors. Recommended: "
            "500K for Deep1M, 1M for Deep10M, 5M for Deep100M."
        ),
    )
    p.add_argument(
        "--base-chunks", type=int, nargs="+", default=None,
        metavar="N",
        help=(
            "Explicit list of base chunk indices to use (e.g. 0 1 2 3). "
            "Default: auto-detect all base## files present in deep1b/."
        ),
    )
    args = p.parse_args()

    deep1b = os.path.join(args.data_dir, "deep1b")

    # ------------------------------------------------------------------
    # Locate base chunk files
    # ------------------------------------------------------------------
    if args.base_chunks is not None:
        base_chunk_paths = [os.path.join(deep1b, f"base{i:02d}")
                            for i in sorted(args.base_chunks)]
    else:
        # Auto-detect: base00, base01, base02, …
        base_chunk_paths = sorted(
            os.path.join(deep1b, f)
            for f in os.listdir(deep1b)
            if f.startswith("base") and f[4:].isdigit()
        )

    if not base_chunk_paths:
        sys.exit(
            f"ERROR: No base## chunk files found in {deep1b}/\n"
            "Download at least base00 with:\n"
            "  python benchs/downloadDeep1B.py --data-dir "
            f"{args.data_dir} --base-chunks 0"
        )

    # Check for missing chunks
    for path in base_chunk_paths:
        if not os.path.exists(path):
            sys.exit(
                f"ERROR: {path} not found.\n"
                "Download it with:\n"
                f"  python benchs/downloadDeep1B.py --data-dir {args.data_dir} "
                f"--base-chunks {' '.join(str(p.split('base')[-1]) for p in base_chunk_paths)}"
            )

    learn_src = os.path.join(deep1b, "learn00")
    if not os.path.exists(learn_src):
        sys.exit(
            f"ERROR: {learn_src} not found.\n"
            "Download it with:\n"
            f"  python benchs/downloadDeep1B.py --data-dir {args.data_dir} --learn-chunks 0"
        )

    # Total capacity check
    total_base_available = sum(_count_vectors_in_file(p) for p in base_chunk_paths)
    total_learn_available = _count_vectors_in_file(learn_src)

    print(f"\nChunk inventory:")
    for path in base_chunk_paths:
        n = _count_vectors_in_file(path)
        print(f"  {os.path.basename(path):12s}  {n:>12,} vectors  "
              f"({os.path.getsize(path)/1024**3:.2f} GiB)")
    print(f"  {'learn00':12s}  {total_learn_available:>12,} vectors  "
          f"({os.path.getsize(learn_src)/1024**3:.2f} GiB)")
    print(f"\nRequested:  nb={args.nb:,}  nt={args.nt:,}")

    if args.nb > total_base_available:
        sys.exit(
            f"\nERROR: Requested {args.nb:,} base vectors but only "
            f"{total_base_available:,} available across {len(base_chunk_paths)} chunk(s).\n"
            "Download more chunks with:\n"
            f"  python benchs/downloadDeep1B.py --data-dir {args.data_dir} "
            "--base-chunks 1 2 3"
        )
    if args.nt > total_learn_available:
        print(
            f"\nWARNING: Requested {args.nt:,} training vectors but learn00 only has "
            f"{total_learn_available:,}. Will use all available.",
            file=sys.stderr,
        )
        args.nt = total_learn_available

    base_dst  = os.path.join(deep1b, "base.fvecs")
    learn_dst = os.path.join(deep1b, "learn.fvecs")

    # ------------------------------------------------------------------
    # Build base.fvecs
    # ------------------------------------------------------------------
    print(f"\nExtracting {args.nb:,} base vectors …")
    build_fvecs(base_chunk_paths, base_dst, args.nb, label="base vectors")

    # ------------------------------------------------------------------
    # Build learn.fvecs
    # ------------------------------------------------------------------
    print(f"\nExtracting {args.nt:,} training vectors …")
    build_fvecs([learn_src], learn_dst, args.nt, label="training vectors")

    # ------------------------------------------------------------------
    # Verify with FAISS
    # ------------------------------------------------------------------
    print("\nVerifying with FAISS loader …")
    try:
        import faiss   # noqa: F401
        from faiss.contrib.datasets import DatasetDeep1B, set_dataset_basedir

        set_dataset_basedir(args.data_dir)

        # Test all supported sizes that fit within what we extracted
        for nb_test, name in [
            (1_000_000,  "Deep1M"),
            (10_000_000, "Deep10M"),
            (100_000_000,"Deep100M"),
        ]:
            if nb_test > args.nb:
                break
            ds = DatasetDeep1B(nb=nb_test)
            xb = ds.get_database()
            xq = ds.get_queries()
            xt = ds.get_train(maxtrain=min(args.nt, 500_000))
            gt = ds.get_groundtruth(k=100)
            print(f"  {name:10s}  base={xb.shape}  queries={xq.shape}  "
                  f"train={xt.shape}  gt={gt.shape}")

        print(f"\nDataset ready.  Run benchmarks with any of:")
        for nb_test, name in [
            (1_000_000,  "Deep1M"),
            (10_000_000, "Deep10M"),
            (100_000_000,"Deep100M"),
        ]:
            if nb_test <= args.nb:
                print(
                    f"  python benchs/bench_suco_deep1m.py "
                    f"--data-dir {args.data_dir} --nb {nb_test}"
                )
    except Exception as e:
        print(f"  Verification failed: {e}", file=sys.stderr)
        print("  Files were written but FAISS loader check failed.")


if __name__ == "__main__":
    main()
