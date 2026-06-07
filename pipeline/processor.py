import os
from dataclasses import dataclass

import pipeline.decode
import pipeline.person_detect
import pipeline.embed
import pipeline.aesthetic
import pipeline.metadata


@dataclass
class ProcessResult:
    kept: bool
    reject_reason: str | None  # None | "person" | "aesthetic" | "decode_error"
    vector: list[float] | None = None
    width: int | None = None
    height: int | None = None
    native_orientation: str | None = None
    aesthetic_score: float | None = None
    palette: list | None = None
    mean_value: float | None = None
    contrast: float | None = None


def process_image(path: str, config: dict) -> ProcessResult:
    paths = config["paths"]
    model_cfg = config["model"]
    thresh = config["thresholds"]
    colour = config["colour"]
    models_dir = paths["models_dir"]

    # 1. Decode
    try:
        image, width, height = pipeline.decode.decode_image(path)
    except Exception:
        return ProcessResult(kept=False, reject_reason="decode_error")

    # 2. Person detection — reject if a person dominates the frame
    yolo_path = os.path.join(models_dir, model_cfg["yolo_model_filename"])
    rejected, _ = pipeline.person_detect.detect_persons(
        image, width, height,
        model_path=yolo_path,
        conf_threshold=thresh["yolo_confidence"],
        area_threshold=thresh["yolo_bbox_area_fraction"],
    )
    if rejected:
        return ProcessResult(kept=False, reject_reason="person")

    # 3. Embed — CLIP ViT-L/14, L2-normalised 768-dim vector
    clip_path = os.path.join(models_dir, model_cfg["checkpoint_filename"])
    vector = pipeline.embed.embed_image(
        image,
        model_name=model_cfg["name"],
        checkpoint_path=clip_path,
    )

    # 4. Aesthetic gate — LAION predictor; reject below the ingest floor
    aesthetic_path = os.path.join(models_dir, model_cfg["aesthetic_checkpoint_filename"])
    score = pipeline.aesthetic.score_image(vector, aesthetic_path)
    if score < thresh["aesthetic_floor"]:
        return ProcessResult(kept=False, reject_reason="aesthetic")

    # 5. Colour stats — only for survivors; colour work never runs on rejects
    metadata = pipeline.metadata.extract_metadata(
        image, width, height,
        sample_size=colour["sample_size"],
        palette_k=colour["palette_k"],
    )

    return ProcessResult(
        kept=True,
        reject_reason=None,
        vector=vector,
        width=width,
        height=height,
        native_orientation=metadata["native_orientation"],
        aesthetic_score=score,
        palette=metadata["palette"],
        mean_value=metadata["mean_value"],
        contrast=metadata["contrast"],
    )
