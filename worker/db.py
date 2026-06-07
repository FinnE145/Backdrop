import sqlite3
import numpy as np


def init_db(config: dict) -> sqlite3.Connection:
    db_path = config["paths"]["sqlite_path"]
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Combined catalog + vector store. Rejected rows only carry hash/status/
    # reject_reason/dates; all other columns are NULL for them.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            hash                TEXT PRIMARY KEY,
            status              TEXT NOT NULL,
            reject_reason       TEXT,
            orig_filename       TEXT,
            capture_date        TEXT,
            ingest_date         TEXT,
            stored_path         TEXT,
            width               INTEGER,
            height              INTEGER,
            native_orientation  TEXT,
            aesthetic_score     REAL,
            palette             TEXT,
            mean_value          REAL,
            contrast            REAL,
            vector              BLOB
        )
    """)

    # Import jobs posted by ingest clients (source='import') or the browser
    # upload page (source='manual'). The import worker polls this table.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            status      TEXT NOT NULL,
            batch_dir   TEXT NOT NULL,
            source      TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            finished_at TEXT
        )
    """)

    # Key/value store for process-wide state. Schema and catalog versions let
    # the web app know when to reload its in-memory vector matrix.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Seed meta on first init; INSERT OR IGNORE means re-opening is safe.
    model_sha = config["model"]["checkpoint_sha256"]
    model_name = config["model"]["name"]
    for key, value in [
        ("model_id",        model_sha),
        ("model_name",      model_name),
        ("schema_version",  "1"),
        ("catalog_version", "0"),
    ]:
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )

    # Every process asserts the stored model_id matches config on open.
    # A mismatch means image and text encoders are from different checkpoints,
    # which silently breaks genre search and aesthetic scores.
    stored = conn.execute(
        "SELECT value FROM meta WHERE key='model_id'"
    ).fetchone()[0]
    if stored != model_sha:
        raise ValueError(
            f"DB model_id mismatch: DB has {stored!r}, "
            f"config has {model_sha!r}. "
            "Re-embedding the library is required if the checkpoint changed."
        )

    conn.commit()
    return conn


def load_kept_vectors(conn: sqlite3.Connection) -> tuple[np.ndarray, list[str]]:
    rows = conn.execute(
        "SELECT hash, vector FROM photos WHERE status='kept'"
    ).fetchall()

    if not rows:
        return np.empty((0, 768), dtype=np.float32), []

    hashes = [row[0] for row in rows]
    # Each BLOB is 768 little-endian float32s written by the client at embed time.
    vectors = np.stack([
        np.frombuffer(row[1], dtype=np.float32) for row in rows
    ])
    return vectors, hashes
