import json
import os
import tempfile

import numpy as np

from worker.db import init_db, load_kept_vectors

with open("config.json") as f:
    config = json.load(f)

with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
    tmp_path = tmp.name

config["paths"]["sqlite_path"] = tmp_path

try:
    conn = init_db(config)
    print("init_db OK")

    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {"photos", "jobs", "meta"} <= tables, f"Missing tables: {tables}"
    print("Tables OK:", tables)

    model_id = conn.execute("SELECT value FROM meta WHERE key='model_id'").fetchone()[0]
    assert model_id == config["model"]["checkpoint_sha256"]
    print("meta model_id OK:", model_id[:12] + "...")

    vectors, hashes = load_kept_vectors(conn)
    assert vectors.shape == (0, 768), f"Expected (0, 768), got {vectors.shape}"
    assert hashes == []
    print("load_kept_vectors (empty) OK:", vectors.shape)

    fake_vector = np.random.rand(768).astype(np.float32)
    conn.execute(
        "INSERT INTO photos (hash, status, vector) VALUES (?, 'kept', ?)",
        ("fakehash123", fake_vector.tobytes()),
    )
    conn.commit()

    vectors, hashes = load_kept_vectors(conn)
    assert vectors.shape == (1, 768), f"Expected (1, 768), got {vectors.shape}"
    assert hashes == ["fakehash123"]
    assert np.allclose(vectors[0], fake_vector)
    print("load_kept_vectors (1 row) OK:", vectors.shape)

finally:
    os.unlink(tmp_path)

print("\nAll DB tests passed.")
