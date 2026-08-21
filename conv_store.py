"""Where an anonymous conversation's only credential is kept.

Anonymous turns ARE stored by the vendor -- measured 2026-08-21 -- and the
device id is the entire credential:

    fresh client, no cookies, same device_id  -> 200, full mapping
    fresh client, any other device_id         -> 404 conversation_inaccessible
    after the original session was closed     -> 200, still readable

So an anonymous conversation is never lost upstream. It is lost HERE: the
session pool is in-memory with a 30 minute TTL, and when a session is evicted
the device id goes with it, leaving the conversation intact on the vendor's side
with nobody able to name the key. This table is that key, and nothing else.

What is deliberately NOT stored: the messages. The vendor holds them and stays
authoritative, so there is no copy to diverge, no chat text on this disk, and a
conversation deleted upstream does not live on in a mirror. Only the binding
(conversation_id -> device_id) plus what the listing needs to show a row.

The listing is the second half. /backend-anon/conversations answers total=0 even
for the device that owns two live conversations, so anonymously there is nothing
upstream to proxy -- the index has to be local or it does not exist.

Scope: rows are keyed by the caller's bearer token, the same namespace
`_files` and `_pools` already use. Callers who send no token all share the
"anonymous" namespace, exactly as they already share a session pool, so an
anonymous listing shows what that shared namespace created. Sending any bearer
string is what separates one caller's index from another's.
"""
import os
import pathlib
import sqlite3
import threading
import time
from typing import Optional

# Next to this module by default, like tokens.json. In a container this MUST
# point at a mounted volume: the whole point is surviving a restart, and a db
# inside the image layer is erased by the next deploy.
DB_PATH = pathlib.Path(os.environ.get(
    "CONV_DB_PATH", str(pathlib.Path(__file__).parent / "conversations.db")))

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    user_id         TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    device_id       TEXT NOT NULL,
    title           TEXT,
    create_time     REAL NOT NULL,
    update_time     REAL NOT NULL,
    PRIMARY KEY (user_id, conversation_id)
);
CREATE INDEX IF NOT EXISTS conversations_by_recency
    ON conversations (user_id, update_time DESC);
"""


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because the handlers run on the event loop's
        # worker threads; every access goes through _lock, so one connection is
        # enough and avoids a per-request open.
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        # WAL so a read never blocks behind the write of a turn in flight.
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(_SCHEMA)
        _conn.commit()
    return _conn


def reset(path: Optional[str] = None) -> None:
    """Point the store at a different file (tests) and drop the open handle."""
    global _conn, DB_PATH
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
        if path is not None:
            DB_PATH = pathlib.Path(path)


def record(user_id: str, conversation_id: str, device_id: str,
           title: Optional[str] = None, now: Optional[float] = None) -> None:
    """Remember which device can read this conversation.

    Called once per anonymous turn. The device id is overwritten on purpose:
    when a device exhausts its quota the pool swaps in a new one, and it is the
    CURRENT owner that can still read the thread. A title only ever replaces a
    missing one -- the vendor generates it on the first turn and does not resend
    it later, so a later turn must not blank it out.
    """
    if not conversation_id or not device_id:
        return
    now = time.time() if now is None else now
    with _lock:
        conn = _connect()
        conn.execute(
            """INSERT INTO conversations
                   (user_id, conversation_id, device_id, title, create_time, update_time)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id, conversation_id) DO UPDATE SET
                   device_id   = excluded.device_id,
                   title       = COALESCE(conversations.title, excluded.title),
                   update_time = excluded.update_time""",
            (user_id, conversation_id, device_id, title, now, now))
        conn.commit()


def lookup(user_id: str, conversation_id: str) -> Optional[dict]:
    """The device that can read this conversation, or None."""
    with _lock:
        row = _connect().execute(
            "SELECT * FROM conversations WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id)).fetchone()
    return dict(row) if row else None


def listing(user_id: str, limit: int = 28, offset: int = 0) -> tuple[list[dict], int]:
    """(rows, total) for this caller, most recently used first."""
    with _lock:
        conn = _connect()
        total = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE user_id = ?", (user_id,)).fetchone()[0]
        rows = conn.execute(
            """SELECT * FROM conversations WHERE user_id = ?
               ORDER BY update_time DESC LIMIT ? OFFSET ?""",
            (user_id, max(0, limit), max(0, offset))).fetchall()
    return [dict(r) for r in rows], total


def forget(user_id: str, conversation_id: str) -> None:
    """Drop a row the vendor no longer honours.

    How long the vendor keeps an anonymous conversation is not known -- it
    cannot be measured in one sitting. So the index is not trusted to stay
    true: a 404 on read is what prunes it, rather than listing a row that
    resolves to nothing forever.
    """
    with _lock:
        conn = _connect()
        conn.execute("DELETE FROM conversations WHERE user_id = ? AND conversation_id = ?",
                     (user_id, conversation_id))
        conn.commit()
