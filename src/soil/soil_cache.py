"""Thread-safe, lightweight TTL cache for soil data lookups.
Stores parsed soil records to respect source rate limits and avoid repeated network calls.
Distinguishes retrieval time from observation time and never caches failed requests as valid data.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SoilDataCache:
    """In-memory cache with optional disk fallback and configurable TTL."""

    def __init__(self, default_ttl_seconds: int = 3600, disk_cache_dir: str | Path = "outputs/cache") -> None:
        self.default_ttl = default_ttl_seconds
        self.disk_dir = Path(disk_cache_dir)
        self.disk_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._memory_cache: dict[str, dict[str, Any]] = {}

    def _normalize_key(self, source_name: str, location_key: str, params_key: str = "") -> str:
        """Construct normalized, lowercase cache key."""
        clean_loc = location_key.strip().lower().replace(" ", "_")
        clean_src = source_name.strip().lower().replace(" ", "_")
        clean_params = params_key.strip().lower().replace(" ", "_")
        return f"{clean_src}::{clean_loc}::{clean_params}".strip(":")

    def get(self, source_name: str, location_key: str, params_key: str = "") -> dict[str, Any] | None:
        """Retrieve unexpired entry from memory or disk cache."""
        key = self._normalize_key(source_name, location_key, params_key)
        now = time.time()

        with self._lock:
            # 1. Check memory cache
            entry = self._memory_cache.get(key)
            if entry:
                if now < entry["expires_at"]:
                    return entry["data"]
                else:
                    # Expired
                    del self._memory_cache[key]

        # 2. Check disk cache
        disk_file = self.disk_dir / f"{key.replace('::', '__')}.json"
        if disk_file.exists():
            try:
                raw = json.loads(disk_file.read_text(encoding="utf-8"))
                if now < raw.get("expires_at", 0):
                    with self._lock:
                        self._memory_cache[key] = raw
                    return raw.get("data")
                else:
                    # Stale file, remove
                    disk_file.unlink(missing_ok=True)
            except Exception:
                pass

        return None

    def set(
        self,
        source_name: str,
        location_key: str,
        data: dict[str, Any],
        params_key: str = "",
        ttl_seconds: int | None = None,
        observation_date: str | None = None,
    ) -> None:
        """Store valid soil payload in memory and disk cache. Rejects error/failure payloads."""
        if not data or not isinstance(data, dict):
            return

        # Do not cache failed or error payloads as valid soil measurements
        if data.get("error") or data.get("success") is False:
            return

        key = self._normalize_key(source_name, location_key, params_key)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        now = time.time()

        cache_record = {
            "key": key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": now + ttl,
            "observation_date": observation_date,
            "data": data,
        }

        with self._lock:
            self._memory_cache[key] = cache_record

        # Persist to disk
        try:
            disk_file = self.disk_dir / f"{key.replace('::', '__')}.json"
            disk_file.write_text(json.dumps(cache_record, indent=2), encoding="utf-8")
        except Exception:
            pass

    def clear(self) -> None:
        """Flush in-memory and disk cache entries."""
        with self._lock:
            self._memory_cache.clear()
        for f in self.disk_dir.glob("*.json"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass


_default_soil_cache: SoilDataCache | None = None


def get_soil_cache() -> SoilDataCache:
    """Singleton getter for SoilDataCache."""
    global _default_soil_cache
    if _default_soil_cache is None:
        _default_soil_cache = SoilDataCache()
    return _default_soil_cache
