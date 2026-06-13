import json
import os
from datetime import datetime, timezone

import numpy as np
from flask import Flask, jsonify, request, send_file

from worker.db import init_db, load_kept_vectors
from webapp.text_encoder import encode_text


def create_app(config: dict):
    app = Flask(__name__)
    db = init_db(config)
    cache = {"vectors": None, "hashes": None, "catalog_version": None}

    def _ensure_vectors():
        version = db.execute(
            "SELECT value FROM meta WHERE key='catalog_version'"
        ).fetchone()[0]
        if version != cache["catalog_version"]:
            cache["vectors"], cache["hashes"] = load_kept_vectors(db)
            cache["catalog_version"] = version
        return cache["vectors"], cache["hashes"]

    @app.get("/index")
    def index():
        rows = db.execute("SELECT hash, status FROM photos").fetchall()
        return jsonify({"hashes": {row[0]: row[1] for row in rows}})

    @app.post("/ingest/import")
    def ingest_import():
        body = request.get_json(force=True)
        batch_dir = body.get("batch_dir")
        if not batch_dir:
            return jsonify({"error": "batch_dir required"}), 400

        cur = db.execute(
            """
            INSERT INTO jobs (status, batch_dir, source, created_at)
            VALUES ('pending', ?, 'import', ?)
            """,
            (batch_dir, datetime.now(timezone.utc).isoformat()),
        )
        db.commit()
        return jsonify({"job_id": cur.lastrowid}), 202

    @app.get("/search")
    def search():
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify({"error": "q required"}), 400
        limit = min(int(request.args.get("limit", 50)), 200)

        vectors, hashes = _ensure_vectors()
        if vectors.shape[0] == 0:
            return jsonify({"results": []})

        model_cfg = config["model"]
        clip_path = os.path.join(
            config["paths"]["models_dir"], model_cfg["checkpoint_filename"]
        )
        query_vec = encode_text(q, model_cfg["name"], clip_path)
        scores = vectors @ query_vec
        top_idx = np.argsort(scores)[::-1][:limit]
        top_hashes = [hashes[i] for i in top_idx]

        placeholders = ",".join("?" * len(top_hashes))
        name_rows = db.execute(
            f"SELECT hash, orig_filename FROM photos WHERE hash IN ({placeholders})",
            top_hashes,
        ).fetchall()
        names = {row[0]: row[1] for row in name_rows}

        return jsonify({
            "results": [
                {"hash": hashes[i], "score": float(scores[i]), "orig_filename": names.get(hashes[i])}
                for i in top_idx
            ]
        })

    @app.get("/photos/<hash>")
    def serve_photo(hash):
        row = db.execute(
            "SELECT stored_path FROM photos WHERE hash=? AND status='kept'", (hash,)
        ).fetchone()
        if row is None:
            return jsonify({"error": "not found"}), 404
        return send_file(row[0])

    return app, db


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    app, _ = create_app(config)
    app.run(host=config["network"]["host"], port=config["network"]["port"])
