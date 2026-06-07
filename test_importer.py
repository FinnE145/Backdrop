import json
import os
import shutil
import tempfile

import numpy as np

from worker.db import init_db, load_kept_vectors
from worker.importer import apply_manifest
from worker.manifest import write_record

with open("config.json") as f:
    config = json.load(f)

# All temp dirs — nothing written to real paths
tmp_root = tempfile.mkdtemp()
db_path = os.path.join(tmp_root, "test.db")
images_dir = os.path.join(tmp_root, "images")
batch_dir = os.path.join(tmp_root, "batch")
os.makedirs(images_dir)
os.makedirs(batch_dir)

config["paths"]["sqlite_path"] = db_path
config["paths"]["images_dir"] = images_dir

try:
    conn = init_db(config)

    # --- Build a synthetic manifest ---
    kept_hash = "aabbcc112233"
    rejected_hash = "ddeeff445566"
    fake_vector = np.random.rand(768).astype(np.float32)
    fake_palette = [[50.0, 10.0, -20.0, 0.3], [30.0, 5.0, 8.0, 0.2]]

    # Write manifest to a saved copy too, so the idempotency test can use it
    # after apply_manifest deletes batch_dir.
    saved_manifest = os.path.join(tmp_root, "manifest_saved.jsonl")
    manifest_path = os.path.join(batch_dir, "manifest.jsonl")
    kept_record = {
        "hash": kept_hash,
        "status": "kept",
        "orig_filename": "IMG_0001.HEIC",
        "capture_date": "2024-01-15T10:30:00",
        "ingest_date": "2024-01-16T08:00:00",
        "width": 4032,
        "height": 3024,
        "native_orientation": "landscape",
        "aesthetic_score": 5.7,
        "palette": fake_palette,
        "mean_value": 0.42,
        "contrast": 0.18,
        "vector": fake_vector.tolist(),
    }
    rejected_record = {
        "hash": rejected_hash,
        "status": "rejected",
        "reject_reason": "aesthetic",
        "ingest_date": "2024-01-16T08:00:00",
    }
    with open(manifest_path, "w") as f, open(saved_manifest, "w") as f2:
        for record in (kept_record, rejected_record):
            write_record(f, record)
            write_record(f2, record)

    # Place a fake image file in batch_dir named by hash (as the client would rsync it)
    fake_image_src = os.path.join(batch_dir, kept_hash)
    with open(fake_image_src, "wb") as f:
        f.write(b"fake image bytes")

    # Insert a job row and apply it
    conn.execute(
        "INSERT INTO jobs (status, batch_dir, source, created_at) VALUES ('pending', ?, 'import', '2024-01-16T08:00:00')",
        (batch_dir,),
    )
    conn.commit()
    job_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    job = {"id": job_id, "batch_dir": batch_dir, "source": "import", "created_at": "2024-01-16T08:00:00"}
    apply_manifest(conn, config, job)
    print("apply_manifest OK")

    # --- Assertions ---

    # Kept row is in DB with correct fields
    row = conn.execute("SELECT * FROM photos WHERE hash=?", (kept_hash,)).fetchone()
    assert row is not None, "Kept row missing from DB"
    col = {d[0]: row[i] for i, d in enumerate(conn.execute("SELECT * FROM photos WHERE hash=?", (kept_hash,)).description)}
    assert col["status"] == "kept"
    assert col["width"] == 4032
    assert col["height"] == 3024
    assert col["native_orientation"] == "landscape"
    assert abs(col["aesthetic_score"] - 5.7) < 1e-5
    assert abs(col["mean_value"] - 0.42) < 1e-5
    palette_stored = json.loads(col["palette"])
    assert palette_stored == fake_palette
    print("Kept row fields OK")

    # Vector round-trips correctly
    stored_vec = np.frombuffer(col["vector"], dtype=np.float32)
    assert np.allclose(stored_vec, fake_vector), "Vector mismatch"
    print("Vector round-trip OK")

    # Image was moved into images_dir
    dst = os.path.join(images_dir, kept_hash)
    assert os.path.exists(dst), f"Image not found at {dst}"
    assert not os.path.exists(fake_image_src), "Image still in batch_dir (not moved)"
    assert col["stored_path"] == dst
    print("Image move OK")

    # Rejected row is in DB
    rej = conn.execute(
        "SELECT status, reject_reason FROM photos WHERE hash=?", (rejected_hash,)
    ).fetchone()
    assert rej is not None, "Rejected row missing from DB"
    assert rej[0] == "rejected"
    assert rej[1] == "aesthetic"
    print("Rejected row OK")

    # catalog_version bumped to 1
    cat_ver = conn.execute("SELECT value FROM meta WHERE key='catalog_version'").fetchone()[0]
    assert cat_ver == "1", f"Expected catalog_version='1', got {cat_ver!r}"
    print("catalog_version OK:", cat_ver)

    # batch_dir cleaned up
    assert not os.path.exists(batch_dir), "batch_dir not cleaned up"
    print("batch_dir cleanup OK")

    # load_kept_vectors sees the new row
    vectors, hashes = load_kept_vectors(conn)
    assert vectors.shape == (1, 768)
    assert hashes == [kept_hash]
    assert np.allclose(vectors[0], fake_vector)
    print("load_kept_vectors OK:", vectors.shape)

    # Idempotency: re-applying the same manifest (with a new batch_dir copy) is a no-op
    batch_dir2 = os.path.join(tmp_root, "batch2")
    os.makedirs(batch_dir2)
    shutil.copy(saved_manifest, os.path.join(batch_dir2, "manifest.jsonl"))
    # No image file this time — if dedup works, it never tries to move it
    job2 = {"id": 2, "batch_dir": batch_dir2, "source": "import", "created_at": "2024-01-16T09:00:00"}
    apply_manifest(conn, config, job2)
    count = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    assert count == 2, f"Expected 2 rows after re-apply, got {count}"
    print("Idempotency OK")

    print("\nAll importer tests passed.")

finally:
    shutil.rmtree(tmp_root, ignore_errors=True)
