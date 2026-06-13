import json
import os
import shutil
import sqlite3
import tempfile
from unittest.mock import patch

import numpy as np

from webapp.app import create_app

with open("config.json") as f:
    config = json.load(f)

tmp = tempfile.mkdtemp()
config["paths"]["sqlite_path"] = os.path.join(tmp, "test.db")

try:
    app = create_app(config)
    client = app.test_client()
    db = sqlite3.connect(config["paths"]["sqlite_path"])

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

    # --- GET /search missing q ---
    r = client.get("/search")
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print("GET /search (missing q) OK")

    # --- GET /search on empty library ---
    vec = np.ones(768, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    with patch("webapp.app.encode_text", return_value=vec):
        r = client.get("/search?q=mountain+lake")
    assert r.status_code == 200
    data = r.get_json()
    assert data == {"results": []}, f"Expected empty results, got {data}"
    print("GET /search (empty library) OK")

    # --- GET /search with results ---
    # Insert two photos with known vectors; query vector matches vec1 exactly.
    vec1 = np.zeros(768, dtype=np.float32); vec1[0] = 1.0
    vec2 = np.zeros(768, dtype=np.float32); vec2[1] = 1.0
    db.execute(
        "INSERT INTO photos (hash, status, vector) VALUES ('img1', 'kept', ?)", (vec1.tobytes(),)
    )
    db.execute(
        "INSERT INTO photos (hash, status, vector) VALUES ('img2', 'kept', ?)", (vec2.tobytes(),)
    )
    db.execute("UPDATE meta SET value='1' WHERE key='catalog_version'")
    db.commit()

    with patch("webapp.app.encode_text", return_value=vec1):
        r = client.get("/search?q=forest")
    assert r.status_code == 200
    data = r.get_json()
    results = data["results"]
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0]["hash"] == "img1", f"Expected img1 first, got {results[0]['hash']}"
    assert abs(results[0]["score"] - 1.0) < 1e-5, f"Unexpected score: {results[0]['score']}"
    assert results[1]["hash"] == "img2"
    assert abs(results[1]["score"]) < 1e-5
    print(f"GET /search OK — img1 score={results[0]['score']:.4f}, img2 score={results[1]['score']:.4f}")

    # --- limit param ---
    with patch("webapp.app.encode_text", return_value=vec1):
        r = client.get("/search?q=forest&limit=1")
    assert len(r.get_json()["results"]) == 1
    print("GET /search (limit=1) OK")

    print("\nAll webapp tests passed.")

finally:
    shutil.rmtree(tmp, ignore_errors=True)
