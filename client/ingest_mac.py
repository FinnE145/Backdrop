"""Mac ingest client — exports from iCloud Photos, runs the pipeline, ships to server."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone

import argparse

import osxphotos
import requests

from pipeline.processor import process_image
from worker.manifest import write_record


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process at most N photos (for testing)")
    args = parser.parse_args()

    with open("config.json") as f:
        config = json.load(f)

    server_host = config["network"]["host"]
    server_port = config["network"]["port"]
    server_user = config["network"]["server_user"]
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
    photos = photosdb.photos()
    print(f"  {len(photos)} photos in library")
    if args.limit:
        photos = photos[:args.limit]
        print(f"  limiting to {args.limit} for this run")

    # 3. Process unknowns into a local staging batch
    batch_uuid = str(uuid.uuid4())
    local_batch_dir = tempfile.mkdtemp(prefix="backdrop_")
    manifest_path = os.path.join(local_batch_dir, "manifest.jsonl")

    kept = 0
    rejected = 0
    skipped = 0
    ingest_date = datetime.now(timezone.utc).isoformat()

    try:
        with open(manifest_path, "w") as manifest_f:
            for photo in photos:
                path = photo.path
                if not path or not os.path.exists(path):
                    # Original not downloaded locally — skip silently
                    skipped += 1
                    continue

                file_hash = hash_file(path)

                if file_hash in known_hashes:
                    skipped += 1
                    continue

                print(f"  {photo.original_filename} ...", end=" ", flush=True)
                result = process_image(path, config)

                record = {"hash": file_hash, "ingest_date": ingest_date}

                if result.kept:
                    # Symlink the original into the batch dir named by hash.
                    # rsync -L dereferences it so the server receives the full file.
                    os.symlink(os.path.abspath(path), os.path.join(local_batch_dir, file_hash))
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

        if kept == 0 and rejected == 0:
            print("\nNothing new to process.")
            return

        print(f"\nResults: {kept} kept, {rejected} rejected, {skipped} skipped")

        # 4. rsync batch dir to server staging (dereference symlinks with -L)
        remote_batch = f"{staging_dir}/{batch_uuid}"
        rsync_dest = f"{server_user}@{server_host}:{remote_batch}/"
        print(f"\nRsyncing to {rsync_dest} ...")
        subprocess.run(
            ["/opt/homebrew/bin/rsync", "-avL", "--partial", f"{local_batch_dir}/", rsync_dest],
            check=True,
        )

        # 5. Trigger import — server queues the job, worker picks it up
        print("Triggering import...")
        r = requests.post(f"{base_url}/ingest/import", json={"batch_dir": remote_batch})
        r.raise_for_status()
        job_id = r.json()["job_id"]
        print(f"Import job queued (job_id={job_id}). Worker will process in the background.")

    finally:
        shutil.rmtree(local_batch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
