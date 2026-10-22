"""SQLite persistence (strategies, agents, memory, events, counters) and the in-process event bus."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections import deque
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    uid TEXT PRIMARY KEY, created REAL, fitness REAL, ok INTEGER, author TEXT, origin TEXT,
    family TEXT, instrument TEXT, spec TEXT, result TEXT
);
CREATE INDEX IF NOT EXISTS idx_strat_fit ON strategies(fitness DESC);
CREATE TABLE IF NOT EXISTS agents (id TEXT PRIMARY KEY, updated REAL, data TEXT);
CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, author TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    # strategies
    def save_results(self, rows: list[tuple[dict, dict]], keep_equity_above: float = 0.3) -> None:
        data = []
        now = time.time()
        for spec, res in rows:
            res = dict(res)
            if res.get("ok") and res.get("fitness", 0) < keep_equity_above:
                res.pop("equity", None)  # recomputed on demand; keeps the database small
            data.append((
                spec["uid"], now, float(res.get("fitness", -1.0)) if res.get("ok") else -1.0, 1 if res.get("ok") else 0,
                spec.get("author", ""), spec.get("origin", ""), spec.get("family", ""), spec.get("instrument", ""),
                json.dumps(spec), json.dumps(res),
            ))
        self.db.executemany("INSERT OR REPLACE INTO strategies VALUES (?,?,?,?,?,?,?,?,?,?)", data)
        self.db.commit()

    def top(self, n: int) -> list[dict]:
        cur = self.db.execute("SELECT spec, result, created FROM strategies WHERE ok=1 ORDER BY fitness DESC LIMIT ?", (n,))
        return [{"spec": json.loads(s), "result": json.loads(r), "t": c} for s, r, c in cur.fetchall()]

    def get(self, uid: str) -> dict | None:
        row = self.db.execute("SELECT spec, result, created FROM strategies WHERE uid=?", (uid,)).fetchone()
        if not row:
            return None
        return {"spec": json.loads(row[0]), "result": json.loads(row[1]), "t": row[2]}

    def update_result(self, uid: str, result: dict) -> None:
        self.db.execute("UPDATE strategies SET result=? WHERE uid=?", (json.dumps(result), uid))
        self.db.commit()

    def prune(self, max_rows: int, max_events: int = 5000) -> None:
        """Keep only the best ``max_rows`` strategies and the latest ``max_events`` events."""
        (n,) = self.db.execute("SELECT COUNT(*) FROM strategies").fetchone()
        if n > max_rows:
            self.db.execute(
                "DELETE FROM strategies WHERE uid IN (SELECT uid FROM strategies ORDER BY fitness ASC LIMIT ?)",
                (n - max_rows,),
            )
        self.db.execute("DELETE FROM events WHERE id <= (SELECT MAX(id) FROM events) - ?", (max_events,))
        self.db.commit()

    def all_uids(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT uid FROM strategies")}

    def counts(self) -> tuple[int, int]:
        total, ok = self.db.execute("SELECT COUNT(*), COALESCE(SUM(ok),0) FROM strategies").fetchone()
        return int(total), int(ok)

    def niches(self) -> list[tuple[str, int, float]]:
        cur = self.db.execute(
            "SELECT family || ' / ' || instrument, COUNT(*), MAX(fitness) FROM strategies WHERE ok=1 GROUP BY family, instrument ORDER BY MAX(fitness) DESC"
        )
        return [(k, int(c), float(f or 0)) for k, c, f in cur.fetchall()]

    # agents
    def save_agent(self, agent: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO agents VALUES (?,?,?)", (agent["id"], time.time(), json.dumps(agent)))
        self.db.commit()

    def load_agents(self) -> list[dict]:
        return [json.loads(r[0]) for r in self.db.execute("SELECT data FROM agents ORDER BY updated")]

    # memory
    def add_memory(self, author: str, text: str) -> None:
        self.db.execute("INSERT INTO memory (ts, author, text) VALUES (?,?,?)", (time.time(), author, text))
        self.db.commit()

    def load_memory(self, n: int = 60) -> list[dict]:
        rows = self.db.execute("SELECT ts, author, text FROM memory ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [{"t": t, "author": a, "text": x} for t, a, x in reversed(rows)]

    # events
    def add_event(self, ev: dict) -> None:
        self.db.execute("INSERT INTO events (ts, kind, data) VALUES (?,?,?)", (ev["t"], ev["kind"], json.dumps(ev)))
        self.db.commit()

    def recent_events(self, n: int = 200) -> list[dict]:
        rows = self.db.execute("SELECT data FROM events ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [json.loads(r[0]) for r in reversed(rows)]

    # key/value
    def kv_get(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def kv_set(self, key: str, value) -> None:
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value)))
        self.db.commit()


class Bus:
    """Fan-out of live events to websocket subscribers, with a replay buffer."""

    def __init__(self, store: Store | None = None, keep: int = 400):
        self.store = store
        self.subscribers: set[asyncio.Queue] = set()
        self.events: deque = deque(maxlen=keep)
        self._n = 0
        if store is not None:
            for ev in store.recent_events(keep):
                self.events.append(ev)
                self._n = max(self._n, ev.get("id", 0))

    def emit(self, kind: str, msg: str, persist: bool = True, **data) -> dict:
        self._n += 1
        ev = {"id": self._n, "t": time.time(), "kind": kind, "msg": msg, **data}
        self.events.append(ev)
        if persist and self.store is not None:
            self.store.add_event(ev)
        for q in list(self.subscribers):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                pass
        return ev

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)
