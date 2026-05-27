"""
Catalog Sync
------------
Lightweight file-watcher that tracks whether the products CSV has been
modified since the last Pinecone index run and exposes a clean status
API for the Streamlit UI.

The sync itself (embedding + upsert) is triggered from the UI; this
module only tracks state and tells the UI whether a sync is due.

Usage
-----
    sync = CatalogSync()              # or CatalogSync("/path/to/products.csv")

    status = sync.get_status()        # dict for UI display
    if sync.needs_sync():
        # ... run the embed + upsert pipeline ...
        sync.mark_synced(n_vectors)   # record successful completion
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_SYNC_STATE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "sync_state.json"
)
_DEFAULT_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "products.csv"
)


class CatalogSync:
    def __init__(self, csv_path: str = _DEFAULT_CSV):
        self.csv_path = csv_path
        self._state   = self._load_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict:
        if os.path.exists(_SYNC_STATE_FILE):
            try:
                with open(_SYNC_STATE_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "last_synced_mtime"   : 0,
            "last_synced_at"      : None,
            "vectors_indexed"     : 0,
            "auto_sync_enabled"   : False,
        }

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(_SYNC_STATE_FILE), exist_ok=True)
        try:
            with open(_SYNC_STATE_FILE, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as exc:
            logger.warning("Could not save sync state: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def csv_mtime(self) -> float:
        """Return the file modification timestamp of the CSV, or 0.0."""
        try:
            return os.path.getmtime(self.csv_path)
        except FileNotFoundError:
            return 0.0

    def needs_sync(self) -> bool:
        """True when the CSV is newer than the last recorded sync."""
        return self.csv_mtime() > self._state.get("last_synced_mtime", 0)

    def is_auto_sync_enabled(self) -> bool:
        return self._state.get("auto_sync_enabled", False)

    def set_auto_sync(self, enabled: bool) -> None:
        self._state["auto_sync_enabled"] = enabled
        self._save_state()

    def mark_synced(self, vectors_indexed: int) -> None:
        """Call after a successful upload to update the sync ledger."""
        self._state.update({
            "last_synced_mtime" : self.csv_mtime(),
            "last_synced_at"    : datetime.now(timezone.utc).isoformat(),
            "vectors_indexed"   : vectors_indexed,
        })
        self._save_state()

    def get_status(self) -> dict:
        """Return a status dict ready for Streamlit display."""
        mtime = self.csv_mtime()
        raw_last = self._state.get("last_synced_at")
        if raw_last:
            try:
                dt = datetime.fromisoformat(raw_last)
                last_synced_str = dt.strftime("%d %b %Y, %H:%M UTC")
            except Exception:
                last_synced_str = raw_last
        else:
            last_synced_str = "Never"

        return {
            "csv_exists"        : mtime > 0,
            "needs_sync"        : self.needs_sync(),
            "auto_sync_enabled" : self.is_auto_sync_enabled(),
            "last_synced_at"    : last_synced_str,
            "vectors_indexed"   : self._state.get("vectors_indexed", 0),
            "csv_modified_at"   : (
                datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M")
                if mtime else "N/A"
            ),
        }
