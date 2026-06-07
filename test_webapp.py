import json
import os
import shutil
import tempfile

from webapp.app import create_app

with open("config.json") as f:
    config = json.load(f)

tmp = tempfile.mkdtemp()
config["paths"]["sqlite_path"] = os.path.join(tmp, "test.db")

try:
    app, db = create_app(config)
    client = app.test_client()

    # --- GET /index on empty DB ---
    r = client.get("/index")
    assert r.status_code == 200
    data = r.get_json()
    assert data == {"hashes": {}}, f"Expected empty hashes, got {data}"
    print("GET /index (empty) OK")

    # --- GET /index with some rows ---
    db.execute("INSERT INTO photos (hash, status) VALUES ('aaa', 'kept')")
    db.execute("INSERT INTO photos (hash, status) VALUES ('bbb', 'rejected')")
    db.commit()

    r = client.get("/index")
    data = r.get_json()
    assert data == {"hashes": {"aaa": "kept", "bbb": "rejected"}}, f"Unexpected: {data}"
    print("GET /index (2 rows) OK")

    # --- POST /ingest/import happy path ---
    r = client.post("/ingest/import", json={"batch_dir": "/tmp/staging/abc123"})
    assert r.status_code == 202, f"Expected 202, got {r.status_code}"
    data = r.get_json()
    assert "job_id" in data, f"No job_id in response: {data}"
    job_id = data["job_id"]
    print(f"POST /ingest/import OK, job_id={job_id}")

    # Verify the job row was actually written
    row = db.execute("SELECT status, batch_dir, source FROM jobs WHERE id=?", (job_id,)).fetchone()
    assert row is not None, "Job row not found in DB"
    assert row[0] == "pending"
    assert row[1] == "/tmp/staging/abc123"
    assert row[2] == "import"
    print("Job row in DB OK")

    # --- POST /ingest/import missing batch_dir ---
    r = client.post("/ingest/import", json={})
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print("POST /ingest/import (missing batch_dir) OK")

    print("\nAll webapp tests passed.")

finally:
    shutil.rmtree(tmp, ignore_errors=True)
