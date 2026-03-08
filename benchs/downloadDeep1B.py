#!/usr/bin/env python3
"""
benchs/downloadDeep1B.py

Downloads specific raw chunk files from the Deep1B dataset hosted on Yandex Disk.

Dataset: https://yadi.sk/d/11eDCm7Dsn9GA

Chunk layout (each chunk ~9.9 GiB, ~27.4M vectors, d=96):
  base00  – base vectors chunk 0    (already needed for Deep1M / Deep10M)
  base01  – base vectors chunk 1  ┐
  base02  – base vectors chunk 2  ├─ additionally needed for Deep100M
  base03  – base vectors chunk 3  ┘
  learn00 – training vectors chunk 0  (already needed; covers all sizes)

What you need per benchmark size:
  Deep1M   : base00 (already have) + learn00 (already have)  → no downloads
  Deep10M  : base00 (already have) + learn00 (already have)  → no downloads
  Deep100M : base00–base03 + learn00 → download base01, base02, base03

Usage:
  # Download only the chunks needed for Deep100M (base01, base02, base03):
  python benchs/downloadDeep1B.py --data-dir /home/dhm/data/ --base-chunks 1 2 3

  # Download all four base chunks (0-3) and one learn chunk:
  python benchs/downloadDeep1B.py --data-dir /home/dhm/data/ --base-chunks 0 1 2 3 --learn-chunks 0

  # Skip already-downloaded files (default: skip if file size > 1 GB):
  python benchs/downloadDeep1B.py --data-dir /home/dhm/data/ --base-chunks 1 2 3 --skip-existing

Notes:
  - Each chunk is ~9.9 GiB. Ensure you have enough disk space (~30 GiB for 3 chunks).
  - Downloads use the Yandex Disk public API (no account required).
  - Requires 'requests' (pip install requests) and 'wget' on PATH, or falls back
    to Python urllib for the actual download.
  - Files are saved as base00, base01, … (no underscore) in <data-dir>/deep1b/.
"""

import argparse
import json
import os
import sys
import time
import urllib.request


YADISK_PUBLIC_KEY = "https://yadi.sk/d/11eDCm7Dsn9GA"
YADISK_API_URL    = "https://cloud-api.yandex.net/v1/disk/public/resources/download"

# Minimum size (bytes) to consider a file "already completely downloaded"
# (9 GiB threshold – any complete chunk is ~9.9 GiB)
MIN_COMPLETE_SIZE = 9 * 1024 ** 3


def get_download_url(public_key: str, path: str) -> str:
    """
    Ask the Yandex Disk API for a direct download URL for `path` inside
    the shared folder identified by `public_key`.
    """
    api = f"{YADISK_API_URL}?public_key={public_key}&path={path}"
    req = urllib.request.Request(api, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    href = data.get("href")
    if not href:
        raise RuntimeError(f"API response has no 'href': {data}")
    return href


def download_file(url: str, dest: str, expected_min_bytes: int = 0) -> None:
    """
    Download `url` to `dest` with a simple progress indicator.
    Resumes partial downloads if the server supports Range requests.
    """
    existing = os.path.getsize(dest) if os.path.exists(dest) else 0
    headers = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        print(f"  Resuming from {existing / 1024**3:.2f} GiB …")

    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing > 0 else "wb"

    with urllib.request.urlopen(req, timeout=60) as resp, \
         open(dest, mode) as out:
        content_length = resp.headers.get("Content-Length")
        total = int(content_length) if content_length else None
        downloaded = existing
        chunk_size = 8 * 1024 * 1024  # 8 MiB
        t0 = time.time()

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - t0
            speed   = (downloaded - existing) / max(elapsed, 0.001) / 1024 ** 2
            if total:
                pct = downloaded / (total + existing) * 100
                print(f"\r  {downloaded / 1024**3:.2f} GiB / "
                      f"{(total + existing) / 1024**3:.2f} GiB  "
                      f"({pct:.1f}%)  {speed:.1f} MiB/s   ", end="", flush=True)
            else:
                print(f"\r  {downloaded / 1024**3:.2f} GiB  "
                      f"{speed:.1f} MiB/s   ", end="", flush=True)

    print()  # newline after progress


def download_chunk(
    deep1b_dir: str,
    kind: str,
    chunk_idx: int,
    skip_existing: bool,
) -> None:
    """
    Download one chunk (base or learn) from Yandex Disk.

    kind        : "base" or "learn"
    chunk_idx   : 0, 1, 2, 3 …
    """
    # Remote path inside the shared Yandex Disk folder
    remote_path = f"/{kind}/{kind}_{chunk_idx:02d}"

    # Local filename (no underscore, matching prepare_deep1m.py convention)
    local_name = f"{kind}{chunk_idx:02d}"
    local_path = os.path.join(deep1b_dir, local_name)

    # Also check for the underscore variant in case it was previously downloaded
    alt_path = os.path.join(deep1b_dir, f"{kind}_{chunk_idx:02d}")
    if os.path.exists(alt_path) and not os.path.exists(local_path):
        print(f"  Found {alt_path}, symlinking to {local_path} …")
        os.symlink(alt_path, local_path)

    if skip_existing and os.path.exists(local_path):
        size = os.path.getsize(local_path)
        if size >= MIN_COMPLETE_SIZE:
            print(f"  SKIP {local_name}  ({size / 1024**3:.2f} GiB already present)")
            return
        print(f"  Partial file found ({size / 1024**3:.2f} GiB), will resume …")

    print(f"\n  Fetching download URL for {remote_path} …")
    try:
        url = get_download_url(YADISK_PUBLIC_KEY, remote_path)
    except Exception as e:
        print(f"  ERROR: Could not get download URL: {e}", file=sys.stderr)
        print("  If the Yandex Disk link is expired, check:", file=sys.stderr)
        print("    https://github.com/facebookresearch/faiss/tree/main/benchs#getting-deep1b",
              file=sys.stderr)
        return

    print(f"  Downloading {local_name} from Yandex Disk …")
    download_file(url, local_path)

    final_size = os.path.getsize(local_path)
    print(f"  Done → {local_path}  ({final_size / 1024**3:.2f} GiB, "
          f"{final_size // (4 + 96 * 4):,} vectors)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download Deep1B chunk files from Yandex Disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Minimum downloads per benchmark size:
  Deep1M   – nothing (base00 + learn00 already downloaded)
  Deep10M  – nothing (base00 has 27.4M vectors, enough for 10M)
  Deep100M – base01, base02, base03  (~30 GiB total)

Example for Deep100M:
  python benchs/downloadDeep1B.py --data-dir /home/dhm/data/ --base-chunks 1 2 3
""",
    )
    p.add_argument(
        "--data-dir", default="/home/dhm/data/",
        help="Root directory containing the deep1b/ sub-folder.",
    )
    p.add_argument(
        "--base-chunks", type=int, nargs="+", default=[],
        metavar="N",
        help="Which base chunk indices to download (e.g. 1 2 3).",
    )
    p.add_argument(
        "--learn-chunks", type=int, nargs="+", default=[],
        metavar="N",
        help="Which learn chunk indices to download (e.g. 0).",
    )
    p.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip files that are already fully downloaded (>= 9 GiB).",
    )
    p.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false",
        help="Re-download even if the file exists.",
    )
    args = p.parse_args()

    if not args.base_chunks and not args.learn_chunks:
        p.print_help()
        print("\nNo chunks specified. Nothing to do.")
        print("\nFor Deep100M, run:")
        print(f"  python {sys.argv[0]} --data-dir {args.data_dir} "
              f"--base-chunks 1 2 3")
        sys.exit(0)

    deep1b_dir = os.path.join(args.data_dir, "deep1b")
    os.makedirs(deep1b_dir, exist_ok=True)

    all_chunks = (
        [("base",  i) for i in sorted(args.base_chunks)] +
        [("learn", i) for i in sorted(args.learn_chunks)]
    )
    print(f"Will download {len(all_chunks)} chunk(s) into {deep1b_dir}/")

    for kind, idx in all_chunks:
        download_chunk(deep1b_dir, kind, idx, skip_existing=args.skip_existing)

    print("\nAll requested chunks downloaded.")
    print("Next step: run prepare_deep1m.py to build base.fvecs and learn.fvecs.")
    print("  python benchs/prepare_deep1m.py --data-dir", args.data_dir,
          "--nb 100000000 --nt 5000000")


if __name__ == "__main__":
    main()
