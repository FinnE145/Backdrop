import json
from datetime import datetime, timezone

from flask import Flask, jsonify, request

from worker.db import init_db


def create_app(config: dict):
    app = Flask(__name__)
    db = init_db(config)

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

    return app, db


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    app, _ = create_app(config)
    app.run(host=config["network"]["host"], port=config["network"]["port"])
