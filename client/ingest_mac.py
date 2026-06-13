"""Mac ingest client — exports from iCloud Photos, runs the pipeline, ships to server."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone

import osxphotos
import requests

from pipeline.processor import process_image
from worker.manifest import read_manifest, write_record

BATCH_DIR = "local/current_batch"


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint() -> tuple[set, str]:
    """Returns (already_processed_hashes, batch_uuid) from a previous partial run."""
    uuid_path = os.path.join(BATCH_DIR, ".batch_uuid")
    manifest_path = os.path.join(BATCH_DIR, "manifest.jsonl")

    # Generate a fresh UUID unless we're resuming
    if os.path.exists(uuid_path):
        with open(uuid_path) as f:
            batch_uuid = f.read().strip()
    else:
        batch_uuid = str(uuid.uuid4())

    hashes = set()
    if os.path.exists(manifest_path):
        try:
            for record in read_manifest(manifest_path):
                hashes.add(record["hash"])
        except Exception as e:
            print(f"  Warning: could not read checkpoint ({e}), starting fresh")
            hashes = set()
        if hashes:
            print(f"  Resuming from checkpoint: {len(hashes)} already processed")

    return hashes, batch_uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process at most N photos (for testing)")
    args = parser.parse_args()

    with open("config.json") as f:
        config = json.load(f)

    server_host = config["network"]["host"]
    server_port = config["network"]["port"]
    server_user = config["network"]["server_user"]
    rsync_host = config["network"].get("rsync_host", server_host)
    staging_dir = config["paths"]["staging_dir"]
    base_url = f"http://{server_host}:{server_port}"

    # 1. Fetch known hashes from server for dedup
    print("Fetching known hashes from server...")
    r = requests.get(f"{base_url}/index")
    r.raise_for_status()
    known_hashes = r.json()["hashes"]
    print(f"  {len(known_hashes)} known hashes")

    # 2. Open Photos library
    print("Opening Photos library...")
    photosdb = osxphotos.PhotosDB()
    photos = photosdb.photos(hidden=False)
    print(f"  {len(photos)} photos in library")
    if args.limit:
        photos = photos[:args.limit]
        print(f"  limiting to {args.limit} for this run")

    # 3. Set up persistent batch dir — survives crashes so progress is not lost
    os.makedirs(BATCH_DIR, exist_ok=True)
    checkpoint_hashes, batch_uuid = load_checkpoint()

    uuid_path = os.path.join(BATCH_DIR, ".batch_uuid")
    if not os.path.exists(uuid_path):
        with open(uuid_path, "w") as f:
            f.write(batch_uuid)

    manifest_path = os.path.join(BATCH_DIR, "manifest.jsonl")

    kept = 0
    rejected = 0
    skipped = 0
    ingest_date = datetime.now(timezone.utc).isoformat()

    # Append to manifest so a restart picks up where we left off
    with open(manifest_path, "a") as manifest_f:
        for photo in photos:
            path = photo.path
            if not path or not os.path.exists(path):
                skipped += 1
                continue

            try:
                file_hash = hash_file(path)
            except Exception as e:
                print(f"  {photo.original_filename} ... skipped (hash error: {e})")
                skipped += 1
                continue

            if file_hash in known_hashes or file_hash in checkpoint_hashes:
                skipped += 1
                continue

            # Add to checkpoint set immediately so a crash mid-photo doesn't
            # leave a partial record that gets re-processed on restart
            checkpoint_hashes.add(file_hash)

            print(f"  {photo.original_filename} ...", end=" ", flush=True)

            try:
                result = process_image(path, config)

                record = {"hash": file_hash, "ingest_date": ingest_date}

                if result.kept:
                    link_path = os.path.join(BATCH_DIR, file_hash)
                    if not os.path.exists(link_path):
                        os.symlink(os.path.abspath(path), link_path)
                    record.update({
                        "status": "kept",
                        "orig_filename": photo.original_filename,
                        "capture_date": photo.date.isoformat() if photo.date else None,
                        "width": result.width,
                        "height": result.height,
                        "native_orientation": result.native_orientation,
                        "aesthetic_score": result.aesthetic_score,
                        "palette": result.palette,
                        "mean_value": result.mean_value,
                        "contrast": result.contrast,
                        "vector": result.vector,
                    })
                    kept += 1
                    print(f"kept (aesthetic {result.aesthetic_score:.2f})")
                else:
                    record.update({
                        "status": "rejected",
                        "reject_reason": result.reject_reason,
                    })
                    rejected += 1
                    print(f"rejected ({result.reject_reason})")

                write_record(manifest_f, record)
                manifest_f.flush()

            except Exception as e:
                print(f"error ({e})")
                write_record(manifest_f, {
                    "hash": file_hash,
                    "status": "rejected",
                    "reject_reason": "decode_error",
                    "ingest_date": ingest_date,
                })
                manifest_f.flush()
                rejected += 1

    total_processed = kept + rejected
    if total_processed == 0 and not checkpoint_hashes:
        print("\nNothing new to process.")
        return

    if total_processed == 0:
        print(f"\nNo new photos this run (checkpoint has {len(checkpoint_hashes)} already processed).")
    else:
        print(f"\nResults this run: {kept} kept, {rejected} rejected, {skipped} skipped")

    # Count total kept in the full batch (including checkpoint)
    total_kept = sum(
        1 for record in read_manifest(manifest_path) if record.get("status") == "kept"
    )
    if total_kept == 0 and total_processed == 0:
        print("Nothing to send.")
        shutil.rmtree(BATCH_DIR, ignore_errors=True)
        return

    # 4. rsync full batch dir to server (dereference symlinks with -L)
    remote_batch = f"{staging_dir}/{batch_uuid}"
    rsync_dest = f"{server_user}@{rsync_host}:{remote_batch}/"
    print(f"\nRsyncing to {rsync_dest} ...")
    subprocess.run(
        ["/opt/homebrew/bin/rsync", "-avL", "--partial", f"{BATCH_DIR}/", rsync_dest],
        check=True,
    )

    # 5. Trigger import and clean up only on success
    print("Triggering import...")
    r = requests.post(f"{base_url}/ingest/import", json={"batch_dir": remote_batch})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"Import job queued (job_id={job_id}). Worker will process in the background.")

    shutil.rmtree(BATCH_DIR, ignore_errors=True)
    print("Checkpoint cleared.")


if __name__ == "__main__":
    main()
