"""Resource manager for MolmoSpaces assets

Design tenets
=============
1. **Versioned, read-only cache.**  Downloaded data lives under ``cache_dir``
   in a ``<data_type>/<source>/<version>/`` tree.  Extracted files are set
   read-only so the cache is safe to share across workers.

2. **Install directory with symlinks.**  A separate ``symlink_dir`` (the
   user-facing assets directory) contains symlinks into the cache.  Two
   linking strategies are supported, configured per source:

     * ``PER_FILE`` -- ``symlink_dir/<type>/<source>/`` is a real directory;
       each file inside is an individual symlink into the cache.
     * ``GLOBAL`` -- ``symlink_dir/<type>/<source>`` is itself a single
       directory-level symlink pointing at the versioned cache directory.

3. **Optional cache lock.**  When ``cache_lock=False`` the manager assumes
   the cache is fully pre-populated (e.g. by a previous run) and only acquires
   the symlink-dir lock.  Useful for multi-worker setups.

4. **Eager vs on-demand extraction.**  Each source is configured as either
   ``EAGER`` (extract all archives at setup time) or ``ON_DEMAND`` (download
   manifests up front, extract individual archives on first use).

5. **Pluggable archive indexing.**  Looking up *which archive contains a
   given asset path* is delegated to an ``ArchiveIndex`` instance attached
   to each source.  Built-in indices: ``NumericIndex``, ``SubstringIndex``.

6. **Layered configuration.**  Behaviour is specified as per-``data_type``
   defaults (``DATA_TYPE_DEFAULTS``) with optional per-source partial
   overrides (``SOURCE_OVERRIDES``).  The version dict
   (``data_type_to_source_to_version``) stays simple and unchanged.

7. **Fork-resilient.**  Both ``ResourceManager`` and the module-level
   singleton cache detect ``fork()`` via PID checks and automatically
   reset LMDB-backed caches and indices in child processes, matching
   the same pattern used by the underlying ``GenericLMDBMap``.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from itertools import chain
from pathlib import Path
from typing import Any, TypedDict

from tqdm import tqdm

from molmospaces_resources.behaviors import (
    LinkStrategy,
    InstallMode,
    SourceBehavior,
    DATA_TYPE_DEFAULTS,
    SOURCE_OVERRIDES,
    _resolve_behavior,
)
from molmospaces_resources.compact_trie import CompactPathTrie
from molmospaces_resources.constants import (
    REMOTE_MANIFEST_NAME,
    COMBINED_TRIES_NAME,
    LOCAL_MANIFEST_NAME,
    TQDM_DISABLE_THRES,
)
from molmospaces_resources.remote_storage import RemoteStorage
from molmospaces_resources.file_utils import (
    _lock_context,
    _complete_extract_flag,
    _complete_link_flag,
    load_json_manifest,
    save_json_manifest,
)
from molmospaces_resources.lmdb_data import PickleLMDBMap
from molmospaces_resources.threading_utils import _parallel_extract


logger = logging.getLogger("molmospaces_resources")


class SourceInfo(TypedDict):
    root_dir: Path
    archive_to_relative_paths: dict[str, list[Path]]


class ResourceManager:
    """Configuration-driven resource manager.

    See module docstring for design tenets.
    """

    # ---- construction -------------------------------------------------------

    def __init__(
        self,
        remote_storage: RemoteStorage,
        data_type_to_source_to_version: dict[str, dict[str, str]],
        symlink_dir: Path,
        cache_dir: Path,
        *,
        data_type_defaults: dict[str, SourceBehavior] | None = None,
        source_overrides: dict[tuple[str, str], dict[str, Any]] | None = None,
        force_install: bool = False,
        max_priority_returns: int = 1,
        cache_lock: bool = True,
        symlink_lock: bool = True,
    ) -> None:
        # -- validate paths ---------------------------------------------------
        resolved_symlink = Path(symlink_dir).resolve()
        resolved_cache = Path(cache_dir).resolve()
        if resolved_symlink == resolved_cache:
            raise ValueError(
                f"symlink_dir and cache_dir resolve to the same path: {resolved_cache}\n"
                f"  symlink_dir={symlink_dir}\n"
                f"  cache_dir={cache_dir}"
            )
        if resolved_symlink.is_relative_to(resolved_cache):
            raise ValueError(
                f"symlink_dir is inside cache_dir -- this could cause cache corruption:\n"
                f"  symlink_dir={resolved_symlink}\n"
                f"  cache_dir={resolved_cache}"
            )
        if resolved_cache.is_relative_to(resolved_symlink):
            raise ValueError(
                f"cache_dir is inside symlink_dir -- this could cause cache corruption:\n"
                f"  cache_dir={resolved_cache}\n"
                f"  symlink_dir={resolved_symlink}"
            )

        # -- store config ------------------------------------------------------
        self.remote_storage = remote_storage
        self.versions = data_type_to_source_to_version
        self.symlink_dir = symlink_dir
        self.cache_dir = cache_dir
        self.force_install = force_install
        self.max_priority_returns = max_priority_returns
        self.cache_lock = cache_lock
        self.symlink_lock = symlink_lock

        self._defaults = data_type_defaults or DATA_TYPE_DEFAULTS
        self._overrides = (
            source_overrides if source_overrides is not None else SOURCE_OVERRIDES
        )

        # -- per-source caches (lazy, reset on fork) ---------------------------
        self._pid: int = os.getpid()
        self._behaviors: dict[tuple[str, str], SourceBehavior] = {}
        self._tries: dict[tuple[str, str], Mapping[str, CompactPathTrie]] = {}
        self._indices_built: set[tuple[str, str]] = set()

        # -- startup info -----------------------------------------------------
        if not cache_lock:
            logger.warning("Cache lock disabled -- assuming cache is pre-populated")
            self._check_cache_populated()
        if not symlink_lock:
            logger.warning("Symlink lock disabled")
        logger.info(
            "ResourceManager (%s) using:\n  symlink_dir=%s\n  cache_dir=%s\n  force_install=%s",
            remote_storage,
            symlink_dir,
            cache_dir,
            force_install,
        )

    # ---- fork safety --------------------------------------------------------

    def _check_pid(self) -> None:
        """Reset fork-unsafe caches (LMDB-backed tries, indices) after fork."""
        pid = os.getpid()
        if pid != self._pid:
            logger.debug(
                "Fork detected (pid %d -> %d), resetting caches", self._pid, pid
            )
            self._pid = pid
            self._tries.clear()
            self._indices_built.clear()
            self._behaviors.clear()

    # ---- behaviour resolution -----------------------------------------------

    def behavior(self, data_type: str, source: str) -> SourceBehavior:
        """Return the effective ``SourceBehavior`` for *(data_type, source)*.

        Resolved once and cached so each source gets a stable index instance.
        """
        self._check_pid()
        key = (data_type, source)
        if key not in self._behaviors:
            self._behaviors[key] = _resolve_behavior(
                data_type, source, self._defaults, self._overrides
            )
        return self._behaviors[key]

    # ---- path accessors -----------------------------------------------------

    def cache_path(self, data_type: str, source: str) -> Path:
        """``cache_dir / data_type / source / version`` -- always the versioned cache dir."""
        version = self.versions[data_type][source]
        return self.cache_dir / data_type / source / version

    def relative_path(self, data_type: str, source: str) -> Path:
        """``data_type / source / version`` -- relative fragment used in URLs and cache layout."""
        version = self.versions[data_type][source]
        return Path(data_type) / source / version

    def symlink_path(self, data_type: str, source: str) -> Path:
        """``symlink_dir / data_type / source`` -- the location in the symlink tree."""
        return self.symlink_dir / data_type / source

    def source_dir(self, data_type: str, source: str) -> Path:
        """The *canonical root directory* for a source.

        * ``GLOBAL``   -> ``cache_path`` (stable, versioned, independent of symlink state)
        * ``PER_FILE`` -> ``symlink_path`` (real dir containing per-file symlinks)
        """
        beh = self.behavior(data_type, source)
        if beh.link_strategy is LinkStrategy.GLOBAL:
            return self.cache_path(data_type, source)
        return self.symlink_path(data_type, source)

    # ---- trie loading (lazy) ------------------------------------------------

    def tries(self, data_type: str, source: str) -> Mapping[str, CompactPathTrie]:
        """Return ``{archive_name: CompactPathTrie}`` for a source, loading lazily."""
        self._check_pid()
        key = (data_type, source)
        if key not in self._tries:
            self._tries[key] = self._load_tries(data_type, source)
            self._ensure_index_built(data_type, source)
        return self._tries[key]

    def _load_tries(self, data_type: str, source: str) -> Mapping[str, CompactPathTrie]:
        src_dir = self.source_dir(data_type, source)
        lmdb_dir = self.symlink_dir / ".lmdb" / data_type / source
        try:
            if not PickleLMDBMap.database_exists(lmdb_dir):
                lmdb_dir.mkdir(parents=True, exist_ok=True)
                with _lock_context(lmdb_dir, None):
                    if not PickleLMDBMap.database_exists(lmdb_dir):
                        logger.debug(
                            "Building trie DB for %s/%s under %s",
                            data_type,
                            source,
                            lmdb_dir,
                        )
                        with gzip.open(src_dir / COMBINED_TRIES_NAME, "rb") as f:
                            archive_to_paths = json.load(f)
                        archive_to_trie = {
                            arc: CompactPathTrie.from_dict(pd)
                            for arc, pd in archive_to_paths.items()
                        }
                        PickleLMDBMap.from_dict(archive_to_trie, lmdb_dir)
            return PickleLMDBMap(lmdb_dir)
        except FileNotFoundError:
            return {}

    def _ensure_index_built(self, data_type: str, source: str) -> None:
        key = (data_type, source)
        if key in self._indices_built:
            return
        beh = self.behavior(data_type, source)
        if beh.archive_index is not None:
            archive_names = list(self._tries.get(key, {}).keys())
            beh.archive_index.build(archive_names)
        self._indices_built.add(key)

    # ---- archive lookup -----------------------------------------------------

    def find_archives(
        self,
        data_type: str,
        source: str,
        paths: Sequence[str | Path],
    ) -> list[str]:
        """Return the minimal set of archive names that contain *paths*.

        Uses the source's ``ArchiveIndex`` for a fast-path hint, then
        verifies against the trie.  Falls back to linear scan when the
        index has no opinion.
        """
        trie_map = self.tries(data_type, source)
        beh = self.behavior(data_type, source)
        src_dir = self.source_dir(data_type, source)
        index = beh.archive_index

        archives: set[str] = set()
        for raw_path in paths:
            p = Path(raw_path)
            if p.is_relative_to(src_dir):
                p = p.relative_to(src_dir)
            query = str(p)

            # Fast path: ask the index for candidates
            candidates: Sequence[str] = ()
            if index is not None:
                candidates = index.candidates(
                    query, max_returns=self.max_priority_returns
                )

            # Check candidates first, then fall back to full scan
            for arc in chain(candidates, trie_map.keys()):
                if trie_map[arc].exists(query):
                    archives.add(arc)
                    break
            else:
                raise ValueError(
                    f"No archive contains {query!r} (data_type={data_type}, source={source})"
                )

        return sorted(archives)

    def find_all_packages_for_data_type(self, data_type: str) -> dict[str, list[str]]:
        """Return ``{source: [package, ...]}`` for all sources of *data_type*."""
        result: dict[str, list[str]] = {}
        for source in self.versions.get(data_type, {}):
            pkgs = self.find_all_packages_for_source(data_type, source)
            if pkgs:
                result[source] = pkgs
        return result

    def find_all_packages_for_source(self, data_type: str, source: str) -> list[str]:
        """Return ``[package, ...]`` for the given source under *data_type*."""
        return list(self.tries(data_type, source).keys())

    # ---- top-level setup ----------------------------------------------------

    def setup(self) -> None:
        """Download manifests, extract eager sources, create symlinks.  Idempotent."""
        os.makedirs(self.symlink_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        with _lock_context(
            self.symlink_dir if self.symlink_lock else None,
            self.cache_dir if self.cache_lock else None,
        ):
            cache_manifest = load_json_manifest(self.cache_dir / LOCAL_MANIFEST_NAME)
            symlink_manifest = load_json_manifest(
                self.symlink_dir / LOCAL_MANIFEST_NAME
            )

            for data_type, source_to_version in self.versions.items():
                cache_manifest.setdefault(data_type, {})
                symlink_manifest.setdefault(data_type, {})

                for source, version in source_to_version.items():
                    cache_manifest[data_type].setdefault(source, [])
                    symlink_manifest[data_type].setdefault(source, None)

                    cache_tries_path = (
                        self.symlink_dir / data_type / source / COMBINED_TRIES_NAME
                    ).resolve()

                    cache_manifest_path = (
                        self.symlink_dir / data_type / source / REMOTE_MANIFEST_NAME
                    ).resolve()

                    if (
                        cache_tries_path.is_file()
                        and cache_manifest_path.is_file()
                        and cache_tries_path.parent == cache_manifest_path.parent
                        and cache_tries_path.parent.name == version
                        and symlink_manifest[data_type][source] == version
                    ):
                        logger.debug(
                            "Already installed: %s/%s %s", data_type, source, version
                        )
                        continue

                    self._setup_source(
                        data_type,
                        source,
                        version,
                        cache_manifest,
                        symlink_manifest,
                    )

                    if self.cache_lock:
                        save_json_manifest(
                            self.cache_dir / LOCAL_MANIFEST_NAME, cache_manifest
                        )
                    if self.symlink_lock:
                        save_json_manifest(
                            self.symlink_dir / LOCAL_MANIFEST_NAME, symlink_manifest
                        )

    def _setup_source(
        self,
        data_type: str,
        source: str,
        version: str,
        cache_manifest: dict,
        symlink_manifest: dict,
    ) -> None:
        """Set up a single (data_type, source) pair: download to cache + create links."""
        beh = self.behavior(data_type, source)
        rel = self.relative_path(data_type, source)
        cache_dest = self.cache_dir / rel
        inst = self.symlink_path(data_type, source)

        logger.debug("Setting up %s/%s version=%s", data_type, source, version)

        # -- handle version mismatch at symlink location ----------------------
        self._handle_version_mismatch(
            inst, data_type, source, version, symlink_manifest
        )

        # -- ensure cache is populated ----------------------------------------
        must_download = version not in cache_manifest[data_type][source]
        if must_download:
            if cache_dest.exists():
                raise RuntimeError(
                    f"Directory path exists on disk but is not recorded in the cache manifest:\n"
                    f"  path: {cache_dest}\n"
                    f"  data_type={data_type}, source={source}, version={version}\n"
                    f"  manifest: {self.cache_dir / LOCAL_MANIFEST_NAME}\n\n"
                    f"This can happen when:\n"
                    f"  1. The manifest is missing an entry — manually add "
                    f'"{version}" to manifest["{data_type}"]["{source}"].\n'
                    f"  2. The directory is a remnant from a previous broken "
                    f"extraction — manually remove {cache_dest} and re-run.\n\n"
                    f"Refusing to delete the directory automatically."
                )
            cache_dest.mkdir(parents=True, exist_ok=True)
            self.remote_storage.fetch_file(rel, REMOTE_MANIFEST_NAME, cache_dest)
            self.remote_storage.fetch_file(rel, COMBINED_TRIES_NAME, cache_dest)

        with open(cache_dest / REMOTE_MANIFEST_NAME) as f:
            remote_manifest: dict[str, float] = json.load(f)

        # -- extract archives (or skip for on-demand) -------------------------
        if must_download:
            if beh.install_mode is InstallMode.EAGER:
                if not self.cache_lock:
                    raise ValueError(
                        f"We cannot download and extract without using cache_lock"
                    )
                logger.debug(
                    "Extracting %.1f MB to cache...", sum(remote_manifest.values())
                )
                failed = _parallel_extract(
                    remote_manifest,
                    rel,
                    cache_dest,
                    self.remote_storage,
                    read_only=True,
                )
                if failed:
                    raise RuntimeError(
                        f"Failed to extract {len(failed)} packages for {data_type}/{source}: {failed}"
                    )
            else:
                logger.debug(
                    "On-demand: skipping extraction (%.1f MB available)",
                    sum(remote_manifest.values()),
                )

        # -- create symlink-side links ----------------------------------------
        self._create_install_links(data_type, source, beh, cache_dest, inst)

        # -- update manifests --------------------------------------------------
        if version not in cache_manifest[data_type][source]:
            cache_manifest[data_type][source].append(version)
        symlink_manifest[data_type][source] = version

    # ---- on-demand package installation -------------------------------------

    def install_packages(
        self,
        data_type: str,
        source_to_packages: dict[str, Sequence[str]],
        **kwargs: Any,
    ) -> None:
        """Install packages for multiple sources of the same *data_type*.

        Convenience wrapper matching the old ``install_scenes({src: pkgs})``
        call pattern.
        """
        target_dir = Path(self.symlink_dir).resolve()
        assert target_dir.is_dir()
        assert Path(self.cache_dir).is_dir()

        with _lock_context(
            target_dir if self.symlink_lock else None,
            self.cache_dir if self.cache_lock else None,
        ):
            for source, packages in source_to_packages.items():
                version = self.versions[data_type][source]
                logger.debug(
                    "Ensuring %d %s archive(s) for %s version=%s",
                    len(packages),
                    data_type,
                    source,
                    version,
                )
                rel = self.relative_path(data_type, source)
                inst = self.symlink_path(data_type, source)
                beh = self.behavior(data_type, source)
                cache_dest = self.cache_path(data_type, source)

                # 1. Extract to cache if needed
                if self.cache_lock:
                    with open(cache_dest / REMOTE_MANIFEST_NAME) as f:
                        remote_manifest: dict[str, float] = json.load(f)

                    to_download: dict[str, float] = {}
                    for pkg in tqdm(
                        packages,
                        desc=f"Checking {data_type}/{source}",
                        disable=len(packages) < TQDM_DISABLE_THRES,
                    ):
                        assert pkg in remote_manifest, f"Unknown package {pkg!r}"
                        if _complete_extract_flag(pkg, cache_dest).exists():
                            continue
                        trie = self.tries(data_type, source).get(pkg)
                        if trie is not None and all(
                            (cache_dest / p).exists() for p in trie.leaf_paths()
                        ):
                            _complete_extract_flag(pkg, cache_dest).touch(
                                exist_ok=False
                            )
                            continue
                        to_download[pkg] = remote_manifest[pkg]

                    if to_download:
                        failed = _parallel_extract(
                            to_download,
                            rel,
                            cache_dest,
                            self.remote_storage,
                            read_only=True,
                        )
                        if failed:
                            raise RuntimeError(
                                f"Failed to extract {len(failed)} packages for "
                                f"{data_type}/{source}: {failed}"
                            )

                # 2. Per-file linking
                skip_linking = kwargs.get("skip_linking", False)
                if beh.link_strategy is LinkStrategy.GLOBAL or skip_linking:
                    continue

                # Parallelize across threads: per-package symlink/mkdir/stat
                # are NFS-latency-bound and release the GIL, so threads scale.
                from concurrent.futures import ThreadPoolExecutor as _TPE

                trie_map = self.tries(data_type, source)
                pkgs_to_link = [p for p in packages if trie_map.get(p) is not None]
                link_workers = max(1, int(os.environ.get("MLSPACES_LINK_WORKERS", "32")))

                def _link_one(pkg: str) -> None:
                    self._ensure_package_linked(inst, pkg, trie_map[pkg], rel)

                with _TPE(max_workers=link_workers) as _lp:
                    list(
                        tqdm(
                            _lp.map(_link_one, pkgs_to_link),
                            total=len(pkgs_to_link),
                            desc=f"Linking {data_type}/{source}",
                            disable=len(pkgs_to_link) < TQDM_DISABLE_THRES,
                        )
                    )

    def install_all_for_source(
        self, data_type: str, source: str, **kwargs: Any
    ) -> None:
        """Install every known package for a source."""
        pkgs = list(self.tries(data_type, source).keys())
        if pkgs:
            self.install_packages(data_type, {source: pkgs}, **kwargs)

    def install_all_for_data_type(self, data_type: str, **kwargs: Any) -> None:
        """Install all packages for every source of *data_type*."""
        all_pkgs = self.find_all_packages_for_data_type(data_type)
        if all_pkgs:
            self.install_packages(data_type, all_pkgs, **kwargs)

    # ---- index query methods (for callers that need token-level lookups) ----

    def index_lookup(self, data_type: str, source: str, token: str) -> set[str]:
        """Look up archives matching a single index *token*.

        For ``NumericIndex``: *token* is a number string like ``"42"``.
        For ``SubstringIndex``: *token* is a name fragment like ``"abc123"``.
        Returns an empty set if no index is configured or token not found.
        """
        _ = self.tries(data_type, source)  # ensure index is built
        beh = self.behavior(data_type, source)
        if beh.archive_index is None:
            return set()
        return beh.archive_index.query_token(token)

    def unindexed_archives(self, data_type: str, source: str) -> list[str]:
        """Return archives that don't match any index token.

        E.g. for ``NumericIndex``, returns archives whose names contain no
        numbers -- useful for finding "base" / non-numbered archives.
        """
        trie_map = self.tries(data_type, source)  # ensure index is built
        beh = self.behavior(data_type, source)
        all_names = list(trie_map.keys())
        if beh.archive_index is None:
            return all_names
        return beh.archive_index.unindexed(all_names)

    # ---- root + archive path helpers ----------------------------------------

    def source_info(
        self,
        data_type: str,
        source: str,
        *,
        recursive: bool = True,
    ) -> SourceInfo:
        """Return ``{"root_dir": Path, "archive_to_relative_paths": {...}}``."""
        root = self.source_dir(data_type, source)
        trie_map = self.tries(data_type, source)

        def collect(trie: CompactPathTrie) -> list[Path]:
            paths = trie.all_paths() if recursive else list(trie.root.keys())
            return [Path(p) for p in paths]

        return {
            "root_dir": root,
            "archive_to_relative_paths": {
                arc: collect(trie) for arc, trie in trie_map.items()
            },
        }

    # ---- internal: symlink creation -----------------------------------------

    def _create_install_links(
        self,
        data_type: str,
        source: str,
        beh: SourceBehavior,
        cache_dest: Path,
        inst: Path,
    ) -> None:
        """Create the install-side symlink structure for one source."""
        inst.parent.mkdir(parents=True, exist_ok=True)

        # Clean up previous install
        if inst.is_symlink():
            inst.unlink()
        elif inst.is_dir():
            shutil.rmtree(inst)

        if beh.link_strategy is LinkStrategy.GLOBAL:
            logger.debug("Symlink (global): %s -> %s", inst, cache_dest)
            inst.symlink_to(cache_dest, target_is_directory=True)

        elif beh.link_strategy is LinkStrategy.PER_FILE:
            logger.debug("Directory (per-file): %s", inst)
            inst.mkdir(parents=True, exist_ok=True)
            # Symlink manifest files so they're accessible from symlink_dir
            for manifest in (REMOTE_MANIFEST_NAME, COMBINED_TRIES_NAME):
                src = cache_dest / manifest
                dst = inst / manifest
                if src.exists() and not dst.exists():
                    dst.symlink_to(src)

            # For EAGER per-file sources, link all packages now
            if beh.install_mode is InstallMode.EAGER:
                self._link_all_packages(data_type, source, cache_dest, inst)

    def _link_all_packages(
        self,
        data_type: str,
        source: str,
        cache_dest: Path,
        inst: Path,
    ) -> None:
        """Create per-file symlinks for every archive (used for EAGER + PER_FILE sources)."""
        rel = self.relative_path(data_type, source)
        with gzip.open(cache_dest / COMBINED_TRIES_NAME, "rb") as f:
            archive_to_paths = json.load(f)
        archive_to_trie = {
            arc: CompactPathTrie.from_dict(pd) for arc, pd in archive_to_paths.items()
        }
        for arc in tqdm(
            archive_to_trie,
            desc=f"Linking {data_type}/{source}",
            disable=len(archive_to_trie) < TQDM_DISABLE_THRES,
        ):
            self._ensure_package_linked(inst, arc, archive_to_trie[arc], rel)

    def _ensure_package_linked(
        self,
        link_dir: Path,
        package: str,
        trie: CompactPathTrie,
        relative_path: Path,
    ) -> None:
        """Create per-file symlinks from *link_dir* into the cache for one package."""
        if _complete_link_flag(package, link_dir).exists():
            return

        # Create directories first. link_dir is absolute, so skip .resolve()
        # (resolving stats every path component -- expensive over NFS).
        for dir_path in CompactPathTrie.from_paths(trie.non_leaf_paths()).leaf_paths():
            (link_dir / dir_path).mkdir(parents=True, exist_ok=True)

        # Symlink each leaf file. Avoid .resolve()/.exists() on the NFS source
        # path; os.symlink doesn't need a resolved target. Just attempt the
        # link and swallow "already exists" -- the only NFS-free fast path.
        for leaf in trie.leaf_paths():
            dst = link_dir / leaf
            src = self.cache_dir / relative_path / leaf
            try:
                os.symlink(src, dst, target_is_directory=False)
            except FileExistsError:
                pass

        _complete_link_flag(package, link_dir).touch(exist_ok=False)

    # ---- internal: version mismatch handling --------------------------------

    def _handle_version_mismatch(
        self,
        inst: Path,
        data_type: str,
        source: str,
        version: str,
        symlink_manifest: dict,
    ) -> None:
        """Detect and handle a version mismatch at the symlink location."""
        # Symlink whose target version differs (normal for GLOBAL on upgrade)
        if inst.is_symlink() and inst.readlink().name != version:
            if not self.force_install:
                raise ValueError(
                    f"Version mismatch for {data_type}/{source}: "
                    f"installed={inst.readlink().name}, requested={version}. "
                    f"Set force_install=True to overwrite."
                )
            inst.unlink()
            self._clear_lmdb(data_type, source)

        # Directory whose manifest version differs (normal for PER_FILE on upgrade)
        elif inst.is_dir() and symlink_manifest[data_type][source] != version:
            if not self.force_install:
                raise ValueError(
                    f"Version mismatch for {data_type}/{source}: "
                    f"installed={symlink_manifest[data_type][source]}, requested={version}. "
                    f"Set force_install=True to overwrite."
                )
            # Don't rmtree here -- _create_install_links handles cleanup safely
            self._clear_lmdb(data_type, source)

    def _clear_lmdb(self, data_type: str, source: str) -> None:
        lmdb_path = self.symlink_dir / ".lmdb" / data_type / source
        if lmdb_path.exists():
            shutil.rmtree(lmdb_path)
        # Also invalidate in-memory caches
        key = (data_type, source)
        self._tries.pop(key, None)
        self._indices_built.discard(key)
        self._behaviors.pop(key, None)

    # ---- internal: cache validation -----------------------------------------

    def _check_cache_populated(self) -> None:
        """Verify cache manifest covers all requested sources (used when cache_lock=False)."""
        manifest_path = self.cache_dir / LOCAL_MANIFEST_NAME
        if not manifest_path.exists():
            raise ValueError(f"Missing cache manifest: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)

        missing: dict[str, dict[str, str]] = {}
        for dt, sv in self.versions.items():
            for src, ver in sv.items():
                if dt not in manifest or src not in manifest[dt]:
                    missing.setdefault(dt, {})[src] = ver

        if missing:
            raise ValueError(
                f"Cache is missing sources:\n{json.dumps(missing, indent=2)}"
            )
