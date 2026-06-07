# Backdrop — CLAUDE.md

## Overview

Backdrop is a self-hosted wallpaper library system. It indexes a personal photo library, filters it down to wallpaper-worthy images via ML, and serves them through a web portal with free-text genre search, colour filtering, and on-demand crop/download. The full technical spec is in `ai-background-picker-spec.md`.

The system has three main runtime processes:
- **Pipeline Module** — shared per-image logic (decode → person detect → embed → aesthetic gate). Runs on the client (Mac M1 or G14), never the server for normal ingest.
- **Import Worker** — server daemon that polls a `jobs` table and applies client manifests to the SQLite datastore. Also runs the full Pipeline Module on CPU for any images that arrive via a non-client path (e.g. files dropped directly into the staging folder or uploaded via the browser UI). This is slow but acceptable for small batches.
- **Web App** — Flask, always-on. Serves the ingest API and the user portal. Holds the CLIP text encoder for queries.

Hardware context:
- **Mac (M1)** — primary ingest client, runs Pipeline Module on MPS via osxphotos
- **G14 (RTX 5070)** — fallback ingest client, CUDA
- **fe-home (i5)** — server, import worker + web app. Embeds on CPU for any files that arrive outside the client manifest path (manual uploads, files dropped in staging folder). Slow but fine for small batches.

## Local Scratch Folder

`local/` is gitignored (except `.gitkeep`). Use it for:
- Test images (HEIC, JPEG, PNG) to run through the pipeline
- Sample output files
- Throwaway scripts and one-off experiments

Drop a `local/images/` subfolder in there for test photos.

## How to Work in This Repo

- Code is written in Python (3.11+).
- The spec (`ai-background-picker-spec.md`) is the source of truth. Follow it exactly. Do not invent behavior not defined there.
- Favor clear, correct implementations over unnecessary abstraction.
- **The user writes the code.** Claude's role is to help plan, review, answer questions, and point at the right approach — not to write implementations unless explicitly asked.
- When asked to write code, write only what was asked for. Do not refactor surroundings, add features, or introduce abstractions beyond the task.
- Development is incremental and immediately testable at each step. Each new gate or feature added to the pipeline should be runnable and visually/terminally verifiable before moving on — e.g. after decode, you can feed a HEIC in and see an image out; after colour stats, you see palette output in the terminal. Do not bundle steps together.

## Environment and Tooling

- This is a new machine. Common tools (brew, gh, etc.) may not be installed.
- **If the right tool for a task is not available, stop and ask — do not find a workaround or substitute.** The user wants to set things up properly, not accumulate hacks.
- Check that a CLI tool exists before using it in instructions (e.g. `which python3`, `which ruff`).

## Locked Decisions (beyond the spec)

| Decision | Choice |
|---|---|
| Repo layout | Top-level packages: `pipeline/`, `worker/`, `webapp/` |
| Config | `config.example.json` committed; `config.json` gitignored; each machine copies and edits |
| Frontend | Flask + Bootstrap + vanilla JS |
| YOLO version | v11n (`ultralytics`) |
| Aesthetic predictor | Simplest implementation that works cleanly with OpenAI L/14 |
| Python deps | Separate `requirements-client.txt` and `requirements-server.txt` |
| File transfer | `rsync` over LAN IP; web UI access via Tailscale |
| CLIP weights | Not yet downloaded; to be vendored in `models/` on each machine |
| Testing | Lightweight pytest for core pipeline logic and manifest format |
| osxphotos / USB walk | Not yet implemented; initial testing uses a local folder |
| Manual server ingest | The server runs the full Pipeline Module on CPU for any files dropped in staging |

## Spec Deviations and Open Items

- The spec has several explicitly open items (§13). Do not resolve them unilaterally — surface them and ask.
- Threshold values (YOLO person detection, LAION aesthetic gate) are starting points to be tuned, not hardcoded decisions.
- The model checkpoint (OpenAI CLIP ViT-L/14) is locked and non-negotiable. All other choices are open to discussion.

## Configuration

- A single `config.json` is the source of all tunable values. Nothing is hardcoded across modules. (§11.1)
- Paths, thresholds, model checkpoint info, negative prompts, colour params, and target presets all live here.

## When Unsure

- Ask questions (using the built-in question tool if available) instead of guessing.
- Clarify ambiguities, edge cases, or missing spec details before suggesting an approach.
- It is better to pause and confirm than to proceed with incorrect assumptions.

## Problem-Solving Limits

- If a clean solution isn't emerging after reasoning through it twice, stop. Explain the constraint and ask for direction.
- Do not layer fixes on top of fixes or reach for hacky workarounds. Surface the problem instead of digging deeper.
- The signal to stop is when you are thinking in circles or compounding complexity to force something to work.

## End of Session

When the user says a session is done or asks to finish up:
- Commit logically; separate features or fixes get their own commits.
- On average, 1–2 commits per session depending on scope.
- Do not push unless explicitly asked.
