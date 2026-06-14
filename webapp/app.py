import json
import mimetypes
import os
import random
import sqlite3
import threading
from datetime import datetime, timezone

mimetypes.add_type("image/heic", ".heic")
mimetypes.add_type("image/heic", ".HEIC")

import numpy as np
import pillow_heif
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

from worker.db import init_db, load_kept_vectors
from webapp.text_encoder import encode_text


def create_app(config: dict):
    app = Flask(__name__)
    init_db(config)  # create schema, validate model id
    db_path = config["paths"]["sqlite_path"]
    cache = {"vectors": None, "hashes": None, "catalog_version": None}
    _local = threading.local()

    def _conn():
        if not hasattr(_local, "db"):
            c = sqlite3.connect(db_path, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            _local.db = c
        return _local.db

    def _ensure_vectors():
        version = _conn().execute(
            "SELECT value FROM meta WHERE key='catalog_version'"
        ).fetchone()[0]
        if version != cache["catalog_version"]:
            cache["vectors"], cache["hashes"] = load_kept_vectors(_conn())
            cache["catalog_version"] = version
        return cache["vectors"], cache["hashes"]

    @app.get("/index")
    def index():
        rows = _conn().execute("SELECT hash, status FROM photos").fetchall()
        return jsonify({"hashes": {row[0]: row[1] for row in rows}})

    @app.post("/ingest/import")
    def ingest_import():
        body = request.get_json(force=True)
        batch_dir = body.get("batch_dir")
        if not batch_dir:
            return jsonify({"error": "batch_dir required"}), 400

        cur = _conn().execute(
            """
            INSERT INTO jobs (status, batch_dir, source, created_at)
            VALUES ('pending', ?, 'import', ?)
            """,
            (batch_dir, datetime.now(timezone.utc).isoformat()),
        )
        _conn().commit()
        return jsonify({"job_id": cur.lastrowid}), 202

    def _fetch_meta(top_hashes):
        if not top_hashes:
            return {}
        placeholders = ",".join("?" * len(top_hashes))
        rows = _conn().execute(
            f"SELECT hash, orig_filename, width, height, aesthetic_score"
            f" FROM photos WHERE hash IN ({placeholders})",
            top_hashes,
        ).fetchall()
        return {
            row[0]: {"orig_filename": row[1], "width": row[2], "height": row[3], "aesthetic_score": row[4]}
            for row in rows
        }

    @app.get("/api/search")
    def api_search():
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
        meta = _fetch_meta(top_hashes)

        return jsonify({
            "results": [
                {
                    "hash": hashes[i],
                    "score": float(scores[i]),
                    **meta.get(hashes[i], {}),
                }
                for i in top_idx
            ]
        })

    @app.get("/api/match")
    def api_match():
        hash_ = request.args.get("hash", "").strip()
        if not hash_:
            return jsonify({"error": "hash required"}), 400
        limit = min(int(request.args.get("limit", 50)), 200)

        row = _conn().execute(
            "SELECT vector FROM photos WHERE hash=? AND status='kept'", (hash_,)
        ).fetchone()
        if row is None:
            return jsonify({"error": "not found"}), 404

        query_vec = np.frombuffer(row[0], dtype=np.float32)
        vectors, hashes = _ensure_vectors()
        if vectors.shape[0] == 0:
            return jsonify({"results": []})

        scores = vectors @ query_vec
        top_idx = np.argsort(scores)[::-1]
        top_idx = [i for i in top_idx if hashes[i] != hash_][:limit]
        top_hashes = [hashes[i] for i in top_idx]
        meta = _fetch_meta(top_hashes)

        return jsonify({
            "results": [
                {
                    "hash": hashes[i],
                    "score": float(scores[i]),
                    **meta.get(hashes[i], {}),
                }
                for i in top_idx
            ]
        })

    @app.get("/api/browse")
    def api_browse():
        limit = min(int(request.args.get("limit", 20)), 200)
        alpha = float(config.get("browse", {}).get("power", 1.5))

        rows = _conn().execute(
            "SELECT hash, orig_filename, aesthetic_score, width, height"
            " FROM photos WHERE status='kept' AND aesthetic_score IS NOT NULL"
        ).fetchall()
        if not rows:
            return jsonify({"results": []})

        hashes = [r[0] for r in rows]
        meta = {r[0]: {"orig_filename": r[1], "aesthetic_score": r[2], "width": r[3], "height": r[4]} for r in rows}
        scores = [r[2] for r in rows]

        min_s = min(scores)
        max_s = max(scores)
        span = max_s - min_s if max_s > min_s else 1.0
        weights = [((s - min_s) / span) ** alpha for s in scores]

        n = min(limit, len(rows))
        chosen = random.choices(range(len(rows)), weights=weights, k=n * 10)
        seen = set()
        picked = []
        for idx in chosen:
            if idx not in seen:
                seen.add(idx)
                picked.append(idx)
            if len(picked) == n:
                break
        if len(picked) < n:
            remaining = [i for i in range(len(rows)) if i not in seen]
            picked.extend(remaining[:n - len(picked)])

        picked.sort(key=lambda i: scores[i], reverse=True)

        return jsonify({
            "results": [
                {"hash": hashes[i], "score": float(scores[i]), **meta[hashes[i]]}
                for i in picked
            ]
        })

    @app.get("/photos/<hash>")
    def serve_photo(hash):
        row = _conn().execute(
            "SELECT stored_path, orig_filename FROM photos WHERE hash=? AND status='kept'", (hash,)
        ).fetchone()
        if row is None:
            return jsonify({"error": "not found"}), 404
        stored_path, orig_filename = row

        if request.args.get("thumbnail"):
            thumb_dir = config["paths"]["thumbnail_dir"]
            thumb_path = os.path.join(thumb_dir, hash + ".jpg")
            if not os.path.exists(thumb_path):
                os.makedirs(thumb_dir, exist_ok=True)
                img = ImageOps.exif_transpose(Image.open(stored_path))
                w, h = img.size
                if h > 400:
                    img = img.resize((int(w * 400 / h), 400), Image.LANCZOS)
                img.convert("RGB").save(thumb_path, "JPEG", quality=85)
            return send_file(thumb_path, mimetype="image/jpeg")

        mimetype, _ = mimetypes.guess_type(orig_filename or "")
        return send_file(stored_path, mimetype=mimetype or "application/octet-stream")

    @app.post("/photos/<hash>/delete")
    def delete_photo(hash):
        row = _conn().execute(
            "SELECT stored_path FROM photos WHERE hash=?", (hash,)
        ).fetchone()
        if row is None:
            return jsonify({"error": "not found"}), 404
        _conn().execute("DELETE FROM photos WHERE hash=?", (hash,))
        _conn().execute(
            "UPDATE meta SET value=CAST(CAST(value AS INTEGER)+1 AS TEXT) WHERE key='catalog_version'"
        )
        _conn().commit()
        try:
            os.remove(row[0])
        except FileNotFoundError:
            pass
        return jsonify({"deleted": hash})

    @app.get("/browse")
    def browse_page():
        return render_template("browse.html")

    @app.get("/search")
    def search_page():
        return render_template("search.html")

    @app.get("/match")
    def match_page():
        return render_template("match.html")

    @app.get("/cleanup")
    def cleanup():
        return render_template("cleanup.html")

    return app


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    app = create_app(config)
    app.run(host=config["network"]["host"], port=config["network"]["port"])
