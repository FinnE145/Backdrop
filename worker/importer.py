import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone

from worker.manifest import read_manifest

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_record(conn, record: dict, images_dir: str, batch_dir: str) -> None:
    hash_ = record["hash"]

    # Hash dedup — reruns are safe because kept/rejected records are idempotent.
    if conn.execute("SELECT 1 FROM photos WHERE hash=?", (hash_,)).fetchone():
        return

    if record["status"] == "kept":
        src = os.path.join(batch_dir, hash_)
        dst = os.path.join(images_dir, hash_)
        os.makedirs(images_dir, exist_ok=True)
        shutil.move(src, dst)

        palette = record.get("palette")
        conn.execute(
            """
            INSERT OR IGNORE INTO photos (
                hash, status, orig_filename, capture_date, ingest_date,
                stored_path, width, height, native_orientation,
                aesthetic_score, palette, mean_value, contrast, vector
            ) VALUES (?, 'kept', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hash_,
                record.get("orig_filename"),
                record.get("capture_date"),
                record.get("ingest_date", _now()),
                dst,
                record.get("width"),
                record.get("height"),
                record.get("native_orientation"),
                record.get("aesthetic_score"),
                json.dumps(palette) if palette is not None else None,
                record.get("mean_value"),
                record.get("contrast"),
                record.get("vector"),  # raw bytes from read_manifest
            ),
        )
    else:
        conn.execute(
            """
            INSERT OR IGNORE INTO photos (hash, status, reject_reason, ingest_date)
            VALUES (?, 'rejected', ?, ?)
            """,
            (
                hash_,
                record.get("reject_reason"),
                record.get("ingest_date", _now()),
            ),
        )

    conn.commit()


def apply_manifest(conn, config: dict, job: dict) -> None:
    images_dir = config["paths"]["images_dir"]
    batch_dir = job["batch_dir"]
    manifest_path = os.path.join(batch_dir, "manifest.jsonl")

    for record in read_manifest(manifest_path):
        _apply_record(conn, record, images_dir, batch_dir)

    # Bump catalog_version so the web app knows to reload its vector matrix.
    current = conn.execute(
        "SELECT value FROM meta WHERE key='catalog_version'"
    ).fetchone()[0]
    conn.execute(
        "UPDATE meta SET value=? WHERE key='catalog_version'",
        (str(int(current) + 1),),
    )
    conn.commit()

    # Batch dir was created solely for this job — clean it up.
    shutil.rmtree(batch_dir, ignore_errors=True)


def run(conn, config: dict) -> None:
    # Stuck-job recovery: if the worker died mid-import the job stays 'running'.
    # Reset to 'pending' — INSERT OR IGNORE in _apply_record makes reruns safe.
    conn.execute("UPDATE jobs SET status='pending' WHERE status='running'")
    conn.commit()
    logger.info("Import worker started")

    while True:
        row = conn.execute(
            """
            SELECT id, batch_dir, source, created_at
            FROM jobs WHERE status='pending'
            ORDER BY created_at ASC LIMIT 1
            """
        ).fetchone()

        if row is None:
            time.sleep(1)
            continue

        job = {"id": row[0], "batch_dir": row[1], "source": row[2], "created_at": row[3]}
        job_id = job["id"]

        conn.execute("UPDATE jobs SET status='running' WHERE id=?", (job_id,))
        conn.commit()
        logger.info("Processing job %d from %s", job_id, job["batch_dir"])

        try:
            apply_manifest(conn, config, job)
            conn.execute(
                "UPDATE jobs SET status='done', finished_at=? WHERE id=?",
                (_now(), job_id),
            )
            conn.commit()
            logger.info("Job %d done", job_id)
        except Exception:
            conn.execute(
                "UPDATE jobs SET status='failed', finished_at=? WHERE id=?",
                (_now(), job_id),
            )
            conn.commit()
            logger.exception("Job %d failed", job_id)


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open("config.json") as f:
        config = json.load(f)
    from worker.db import init_db
    conn = init_db(config)
    run(conn, config)
