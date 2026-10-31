"""SQLite cache. Every fetch is appended as a snapshot (kind, key, fetched_at, source, JSON payload),
every attempt — success or failure — is logged, so the app can always say what it is showing and why."""
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    kind TEXT NOT NULL, key TEXT NOT NULL, fetched_at TEXT NOT NULL, source TEXT NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY (kind, key, fetched_at));
CREATE TABLE IF NOT EXISTS fetch_log (
    kind TEXT NOT NULL, key TEXT NOT NULL, attempted_at TEXT NOT NULL, ok INTEGER NOT NULL, error TEXT);
CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS watchlist (ticker TEXT PRIMARY KEY, added_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_log ON fetch_log (kind, key, attempted_at);
"""


def utcnow():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


class Cache:
    def __init__(self, path="data/market.sqlite"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()

    def put(self, kind, key, payload, source, fetched_at=None):
        fetched_at = fetched_at or utcnow()
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?)",
                            (kind, key, iso(fetched_at), source, json.dumps(payload, allow_nan=True)))
            self.db.commit()
        return fetched_at

    def latest(self, kind, key):
        with self._lock:
            row = self.db.execute("SELECT payload, fetched_at, source FROM snapshots WHERE kind=? AND key=? "
                                  "ORDER BY fetched_at DESC LIMIT 1", (kind, key)).fetchone()
        if not row:
            return None
        return dict(payload=json.loads(row[0]), fetched_at=datetime.fromisoformat(row[1]), source=row[2])

    def log(self, kind, key, ok, error=None, at=None):
        with self._lock:
            self.db.execute("INSERT INTO fetch_log VALUES (?,?,?,?,?)",
                            (kind, key, iso(at or utcnow()), int(ok), None if error is None else str(error)[:500]))
            self.db.commit()

    def last_attempt(self, kind, key):
        with self._lock:
            row = self.db.execute("SELECT attempted_at, ok, error FROM fetch_log WHERE kind=? AND key=? "
                                  "ORDER BY attempted_at DESC LIMIT 1", (kind, key)).fetchone()
        return None if not row else dict(at=datetime.fromisoformat(row[0]), ok=bool(row[1]), error=row[2])

    def keys(self, kind):
        with self._lock:
            return [r[0] for r in self.db.execute("SELECT DISTINCT key FROM snapshots WHERE kind=? ORDER BY key",
                                                  (kind,))]

    def set_setting(self, name, value):
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO settings VALUES (?,?,?)", (name, json.dumps(value), iso(utcnow())))
            self.db.commit()

    def get_setting(self, name, default=None):
        with self._lock:
            row = self.db.execute("SELECT value, updated_at FROM settings WHERE name=?", (name,)).fetchone()
        return default if not row else json.loads(row[0])

    def setting_time(self, name):
        with self._lock:
            row = self.db.execute("SELECT updated_at FROM settings WHERE name=?", (name,)).fetchone()
        return None if not row else datetime.fromisoformat(row[0])

    def delete_setting(self, name):
        with self._lock:
            self.db.execute("DELETE FROM settings WHERE name=?", (name,))
            self.db.commit()

    def watch(self, ticker):
        with self._lock:
            self.db.execute("INSERT OR IGNORE INTO watchlist VALUES (?,?)", (ticker.upper(), iso(utcnow())))
            self.db.commit()

    def unwatch(self, ticker):
        with self._lock:
            self.db.execute("DELETE FROM watchlist WHERE ticker=?", (ticker.upper(),))
            self.db.commit()

    def watchlist(self):
        with self._lock:
            return [r[0] for r in self.db.execute("SELECT ticker FROM watchlist ORDER BY ticker")]

    def prune(self, keep_days=30, now=None):
        """Drop snapshots older than keep_days, but always keep the newest one per (kind, key)."""
        cut = iso((now or utcnow()) - timedelta(days=keep_days))
        with self._lock:
            self.db.execute("""DELETE FROM snapshots WHERE fetched_at < ? AND fetched_at <
                               (SELECT MAX(s2.fetched_at) FROM snapshots s2
                                WHERE s2.kind = snapshots.kind AND s2.key = snapshots.key)""", (cut,))
            self.db.execute("DELETE FROM fetch_log WHERE attempted_at < ?", (cut,))
            self.db.commit()
