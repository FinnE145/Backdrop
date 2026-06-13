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
from flask import Flask, jsonify, request, send_file

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

    def _fetch_names(top_hashes):
        if not top_hashes:
            return {}
        placeholders = ",".join("?" * len(top_hashes))
        rows = _conn().execute(
            f"SELECT hash, orig_filename FROM photos WHERE hash IN ({placeholders})",
            top_hashes,
        ).fetchall()
        return {row[0]: row[1] for row in rows}

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
        names = _fetch_names(top_hashes)

        return jsonify({
            "results": [
                {"hash": hashes[i], "score": float(scores[i]), "orig_filename": names.get(hashes[i])}
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
        names = _fetch_names(top_hashes)

        return jsonify({
            "results": [
                {"hash": hashes[i], "score": float(scores[i]), "orig_filename": names.get(hashes[i])}
                for i in top_idx
            ]
        })

    @app.get("/api/browse")
    def api_browse():
        limit = min(int(request.args.get("limit", 20)), 200)
        alpha = float(config.get("browse", {}).get("power", 1.5))

        rows = _conn().execute(
            "SELECT hash, orig_filename, aesthetic_score FROM photos WHERE status='kept' AND aesthetic_score IS NOT NULL"
        ).fetchall()
        if not rows:
            return jsonify({"results": []})

        hashes = [r[0] for r in rows]
        names = {r[0]: r[1] for r in rows}
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
                {"hash": hashes[i], "score": float(scores[i]), "orig_filename": names.get(hashes[i])}
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
        return """<!doctype html>
<html>
<head><title>Backdrop Browse</title></head>
<body>
<h2>Browse</h2>
<input id="limit" type="number" value="20" min="1" max="200">
<button onclick="doBrowse()">Regenerate</button>
<p id="status"></p>
<div id="results" style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px"></div>
<script>
function makeCard(item) {
    const wrap = document.createElement('div');
    wrap.style = 'display:flex;flex-direction:column;align-items:center;gap:4px';
    wrap.id = 'wrap-' + item.hash;

    const img = document.createElement('img');
    img.src = '/photos/' + item.hash;
    img.title = (item.orig_filename || item.hash) + ' (' + item.score.toFixed(3) + ')';
    img.style = 'height:200px;object-fit:cover;cursor:pointer';
    img.onclick = () => window.open(img.src);

    const removeBtn = document.createElement('button');
    removeBtn.textContent = 'Remove';
    removeBtn.onclick = async () => {
        if (!confirm('Remove ' + (item.orig_filename || item.hash) + '?')) return;
        const res = await fetch('/photos/' + item.hash + '/delete', {method:'POST'});
        if (res.ok) document.getElementById('wrap-' + item.hash).remove();
        else alert('Failed to remove');
    };

    const simBtn = document.createElement('button');
    simBtn.textContent = 'Similar...';
    simBtn.onclick = () => window.location.href = '/match?hash=' + item.hash;

    wrap.appendChild(img);
    wrap.appendChild(removeBtn);
    wrap.appendChild(simBtn);
    return wrap;
}

async function doBrowse() {
    const limit = document.getElementById('limit').value;
    document.getElementById('status').textContent = 'Loading...';
    document.getElementById('results').innerHTML = '';
    const r = await fetch('/api/browse?limit=' + limit);
    const data = await r.json();
    document.getElementById('status').textContent = data.results.length + ' results';
    for (const item of data.results)
        document.getElementById('results').appendChild(makeCard(item));
}

doBrowse();
</script>
</body>
</html>"""

    @app.get("/search")
    def search_page():
        return """<!doctype html>
<html>
<head><title>Backdrop Search</title></head>
<body>
<h2>Search</h2>
<input id="q" type="text" placeholder="e.g. mountain lake" size="40">
<input id="limit" type="number" value="20" min="1" max="200">
<button onclick="doSearch()">Search</button>
<button onclick="toCleanup()">Switch to Cleanup Mode</button>
<p id="status"></p>
<div id="results" style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px"></div>
<script>
function makeCard(item) {
    const wrap = document.createElement('div');
    wrap.style = 'display:flex;flex-direction:column;align-items:center;gap:4px';
    wrap.id = 'wrap-' + item.hash;

    const img = document.createElement('img');
    img.src = '/photos/' + item.hash;
    img.title = (item.orig_filename || item.hash) + ' (' + item.score.toFixed(3) + ')';
    img.style = 'height:200px;object-fit:cover;cursor:pointer';
    img.onclick = () => window.open(img.src);

    const removeBtn = document.createElement('button');
    removeBtn.textContent = 'Remove';
    removeBtn.onclick = async () => {
        if (!confirm('Remove ' + (item.orig_filename || item.hash) + '?')) return;
        const res = await fetch('/photos/' + item.hash + '/delete', {method:'POST'});
        if (res.ok) document.getElementById('wrap-' + item.hash).remove();
        else alert('Failed to remove');
    };

    const simBtn = document.createElement('button');
    simBtn.textContent = 'Similar...';
    simBtn.onclick = () => window.location.href = '/match?hash=' + item.hash;

    wrap.appendChild(img);
    wrap.appendChild(removeBtn);
    wrap.appendChild(simBtn);
    return wrap;
}

async function doSearch() {
    const q = document.getElementById('q').value.trim();
    const limit = document.getElementById('limit').value;
    if (!q) return;
    document.getElementById('status').textContent = 'Searching...';
    document.getElementById('results').innerHTML = '';
    const r = await fetch('/api/search?q=' + encodeURIComponent(q) + '&limit=' + limit);
    const data = await r.json();
    document.getElementById('status').textContent = data.results.length + ' results';
    for (const item of data.results)
        document.getElementById('results').appendChild(makeCard(item));
}

function toCleanup() {
    const q = document.getElementById('q').value.trim();
    const limit = document.getElementById('limit').value;
    window.location.href = '/cleanup?q=' + encodeURIComponent(q) + '&limit=' + limit;
}

const params = new URLSearchParams(window.location.search);
const initQ = params.get('q');
if (initQ) { document.getElementById('q').value = initQ; doSearch(); }
if (params.get('limit')) document.getElementById('limit').value = params.get('limit');
document.getElementById('q').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
</script>
</body>
</html>"""

    @app.get("/match")
    def match_page():
        return """<!doctype html>
<html>
<head><title>Backdrop Match</title></head>
<body>
<h2>Find Similar</h2>
<input id="hash" type="text" placeholder="image hash" size="70">
<input id="limit" type="number" value="20" min="1" max="200">
<button onclick="doSearch()">Find Similar</button>
<button onclick="toCleanup()">Switch to Cleanup Mode</button>
<p id="status"></p>
<div id="results" style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px"></div>
<script>
function makeCard(item) {
    const wrap = document.createElement('div');
    wrap.style = 'display:flex;flex-direction:column;align-items:center;gap:4px';
    wrap.id = 'wrap-' + item.hash;

    const img = document.createElement('img');
    img.src = '/photos/' + item.hash;
    img.title = (item.orig_filename || item.hash) + ' (' + item.score.toFixed(3) + ')';
    img.style = 'height:200px;object-fit:cover;cursor:pointer';
    img.onclick = () => window.open(img.src);

    const removeBtn = document.createElement('button');
    removeBtn.textContent = 'Remove';
    removeBtn.onclick = async () => {
        if (!confirm('Remove ' + (item.orig_filename || item.hash) + '?')) return;
        const res = await fetch('/photos/' + item.hash + '/delete', {method:'POST'});
        if (res.ok) document.getElementById('wrap-' + item.hash).remove();
        else alert('Failed to remove');
    };

    const simBtn = document.createElement('button');
    simBtn.textContent = 'Similar...';
    simBtn.onclick = () => window.location.href = '/match?hash=' + item.hash;

    wrap.appendChild(img);
    wrap.appendChild(removeBtn);
    wrap.appendChild(simBtn);
    return wrap;
}

async function doSearch() {
    const hash = document.getElementById('hash').value.trim();
    const limit = document.getElementById('limit').value;
    if (!hash) return;
    if (hash.length < 15 || hash.includes(' ')) {
        window.location.href = '/search?q=' + encodeURIComponent(hash) + '&limit=' + limit;
        return;
    }
    document.getElementById('status').textContent = 'Searching...';
    document.getElementById('results').innerHTML = '';
    const r = await fetch('/api/match?hash=' + encodeURIComponent(hash) + '&limit=' + limit);
    const data = await r.json();
    if (data.error) {
        const a = document.createElement('a');
        a.href = '/search?q=' + encodeURIComponent(hash) + '&limit=' + limit;
        a.textContent = 'search instead...';
        const status = document.getElementById('status');
        status.textContent = 'Not found. ';
        status.appendChild(a);
        return;
    }
    document.getElementById('status').textContent = data.results.length + ' results';
    for (const item of data.results)
        document.getElementById('results').appendChild(makeCard(item));
}

function toCleanup() {
    const hash = document.getElementById('hash').value.trim();
    const limit = document.getElementById('limit').value;
    window.location.href = '/cleanup?hash=' + encodeURIComponent(hash) + '&limit=' + limit;
}

const params = new URLSearchParams(window.location.search);
const h = params.get('hash');
if (h) { document.getElementById('hash').value = h; doSearch(); }
document.getElementById('hash').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
</script>
</body>
</html>"""

    @app.get("/cleanup")
    def cleanup():
        return """<!doctype html>
<html>
<head><title>Backdrop Cleanup</title></head>
<body>
<h2>Cleanup</h2>
<input id="q" type="text" placeholder="e.g. mountain lake" size="40">
<input id="hash-input" type="text" placeholder="image hash" size="70" style="display:none">
<input id="limit" type="number" value="20" min="1" max="200">
<button onclick="doSearch()">Search</button>
<button onclick="deleteAll()">Delete All</button>
<button onclick="toView()">Switch to View Mode</button>
<p id="status"></p>
<div id="results" style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px"></div>
<script>
const kept = new Set();
let currentResults = [];

const params = new URLSearchParams(window.location.search);
const initHash = params.get('hash');
const initQ = params.get('q');
if (params.get('limit')) document.getElementById('limit').value = params.get('limit');
if (initHash) {
    document.getElementById('q').style.display = 'none';
    document.getElementById('hash-input').style.display = '';
    document.getElementById('hash-input').value = initHash;
} else if (initQ) {
    document.getElementById('q').value = initQ;
}

async function doSearch() {
    const hash = document.getElementById('hash-input').value.trim();
    const q = document.getElementById('q').value.trim();
    const limit = document.getElementById('limit').value;
    if (!hash && !q) return;
    const url = hash ? '/api/match?hash=' + encodeURIComponent(hash) + '&limit=' + limit
                     : '/api/search?q=' + encodeURIComponent(q) + '&limit=' + limit;
    document.getElementById('status').textContent = 'Searching...';
    document.getElementById('results').innerHTML = '';
    kept.clear();
    currentResults = [];
    const r = await fetch(url);
    const data = await r.json();
    if (data.error) { document.getElementById('status').textContent = 'Error: ' + data.error; return; }
    currentResults = data.results;
    document.getElementById('status').textContent = data.results.length + ' results';
    for (const item of data.results) {
        const wrap = document.createElement('div');
        wrap.style = 'display:flex;flex-direction:column;align-items:center;gap:4px';
        wrap.id = 'wrap-' + item.hash;

        const img = document.createElement('img');
        img.src = '/photos/' + item.hash;
        img.title = (item.orig_filename || item.hash) + ' (' + item.score.toFixed(3) + ')';
        img.style = 'height:200px;object-fit:cover;cursor:pointer';
        img.onclick = () => window.open(img.src);

        const keepBtn = document.createElement('button');
        keepBtn.textContent = 'Keep';
        keepBtn.onclick = () => {
            kept.add(item.hash);
            document.getElementById('wrap-' + item.hash).style.display = 'none';
        };

        const simBtn = document.createElement('button');
        simBtn.textContent = 'Similar...';
        simBtn.onclick = () => window.location.href = '/cleanup?hash=' + item.hash;

        wrap.appendChild(img);
        wrap.appendChild(keepBtn);
        wrap.appendChild(simBtn);
        document.getElementById('results').appendChild(wrap);
    }
}

async function deleteAll() {
    const toDelete = currentResults.filter(item => !kept.has(item.hash));
    if (toDelete.length === 0) { alert('Nothing to delete.'); return; }
    if (!confirm('Delete ' + toDelete.length + ' photos?')) return;
    document.getElementById('status').textContent = 'Deleting...';
    let deleted = 0;
    for (const item of toDelete) {
        const res = await fetch('/photos/' + item.hash + '/delete', {method:'POST'});
        if (res.ok) {
            const el = document.getElementById('wrap-' + item.hash);
            if (el) el.remove();
            deleted++;
        }
    }
    currentResults = currentResults.filter(item => kept.has(item.hash));
    document.getElementById('status').textContent = 'Deleted ' + deleted + ' photos.';
}

function toView() {
    const hash = document.getElementById('hash-input').value.trim();
    const q = document.getElementById('q').value.trim();
    const limit = document.getElementById('limit').value;
    if (hash) window.location.href = '/match?hash=' + encodeURIComponent(hash) + '&limit=' + limit;
    else window.location.href = '/search?q=' + encodeURIComponent(q) + '&limit=' + limit;
}

document.getElementById('q').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
document.getElementById('hash-input').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
if (initHash || initQ) doSearch();
</script>
</body>
</html>"""

    return app


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    app = create_app(config)
    app.run(host=config["network"]["host"], port=config["network"]["port"])
