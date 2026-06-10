from __future__ import annotations

import logging
import tarfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING

import zstandard as zstd
from tqdm import tqdm

from molmospaces_resources.file_utils import _safe_extract, _complete_extract_flag
from molmospaces_resources.constants import TQDM_DISABLE_THRES

if TYPE_CHECKING:
    from molmospaces_resources.remote_storage import RemoteStorage

logger = logging.getLogger("molmospaces_resources")


def _download_and_extract(
    package: str,
    relative_path: Path,
    cache_dest: Path,
    source: RemoteStorage,
    *,
    read_only: bool = True,
) -> bool:
    """Download a single ``.tar.zst`` archive and extract it to *cache_dest*."""
    import time as _time

    if not package.endswith(".tar.zst"):
        logger.warning("Unknown archive extension: %s", package)
        return False
    # Retry with backoff: the R2 public endpoint rate-limits (429) and
    # occasionally 502s on bursts of many tiny archives. Transient -- retry.
    attempts = 6
    for attempt in range(attempts):
        try:
            raw_stream = source.stream_archive(relative_path, package)
            with zstd.ZstdDecompressor().stream_reader(raw_stream) as reader:
                with tarfile.open(fileobj=reader, mode="r|*") as tar:
                    _safe_extract(tar, cache_dest, read_only=read_only)
            flag = _complete_extract_flag(package, cache_dest)
            try:
                flag.touch(exist_ok=False)
            except FileExistsError:
                pass
            return True
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if attempt < attempts - 1:
                _time.sleep(min(2 ** attempt, 30))  # 1,2,4,8,16,30s backoff
                continue
            logger.warning("Download/extract failure for %s: %s: %s", package, type(exc).__name__, exc)
            return False


def _extract_worker(q_in: Queue, q_out: Queue) -> None:
    try:
        while True:
            item = q_in.get()
            if item is None:
                break
            package, relative_path, cache_dest, source, read_only = item
            ok = _download_and_extract(
                package, relative_path, cache_dest, source, read_only=read_only
            )
            q_out.put((package, ok))
    finally:
        q_out.put(None)


def _logging_worker(
    q_out: Queue,
    manifest: dict[str, float],
    num_workers: int,
    result: dict,
) -> None:
    pending = set(manifest)
    failed: set[str] = set()
    ended = 0
    pbar = tqdm(
        total=len(manifest),
        desc="Extracting",
        disable=len(manifest) < TQDM_DISABLE_THRES,
    )
    try:
        while True:
            msg = q_out.get()
            if msg is None:
                ended += 1
                if ended == num_workers:
                    break
                continue
            name, ok = msg
            pending.discard(name)
            if not ok:
                failed.add(name)
            pbar.update(1)
    finally:
        pbar.close()
    result["failed"] = pending | failed


def _parallel_extract(
    packages_to_size: dict[str, float],
    relative_path: Path,
    cache_dest: Path,
    source: RemoteStorage,
    *,
    read_only: bool = True,
    max_workers: int = int(__import__("os").environ.get("MLSPACES_EXTRACT_WORKERS", "48")),
    min_items_per_worker: int = 1,
) -> set[str]:
    """Download + extract *packages_to_size* across PROCESSES (one GIL each).

    The per-archive work is dominated by Python ``tarfile`` extraction of many
    tiny files, which holds the GIL -- so threads don't scale. Each archive is
    self-contained, so we fan out over a process pool to use all cores.
    Returns failed package names.
    """
    if not packages_to_size:
        return set()

    pkgs = list(packages_to_size)
    num_workers = max(min(max_workers, len(pkgs)), 1)

    def _run(pkg_list: list[str]) -> set[str]:
        failed_local: set[str] = set()
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futs = {
                pool.submit(
                    _download_and_extract,
                    pkg,
                    relative_path,
                    cache_dest,
                    source,
                    read_only=read_only,
                ): pkg
                for pkg in pkg_list
            }
            for fut in tqdm(
                as_completed(futs),
                total=len(pkg_list),
                desc="Extracting",
                disable=len(pkg_list) < TQDM_DISABLE_THRES,
            ):
                pkg = futs[fut]
                try:
                    ok = fut.result()
                except Exception as exc:  # BrokenProcessPool, pickling, etc.
                    logger.warning(
                        "Extract process failure for %s: %s: %s",
                        pkg, type(exc).__name__, exc,
                    )
                    ok = False
                if not ok:
                    failed_local.add(pkg)
        return failed_local

    if len(pkgs) > 1:
        logger.debug("Extracting %d packages across %d processes...", len(pkgs), num_workers)

    failed = _run(pkgs)
    if not failed or len(failed) > max(0.1 * len(pkgs), 1):
        return failed

    logger.debug("Retrying %d failures...", len(failed))
    return _run(list(failed))
