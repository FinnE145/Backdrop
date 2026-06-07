from ultralytics import YOLO

_model = None
_model_path = None


def _get_model(model_path):
    global _model, _model_path
    if _model is None or _model_path != model_path:
        _model = YOLO(model_path)
        _model_path = model_path
    return _model


def detect_persons(image, width, height, model_path, conf_threshold, area_threshold):
    image_area = width * height

    model = _get_model(model_path)
    results = model(image, verbose=False)

    detections = []
    for cls, conf, box in zip(
        results[0].boxes.cls,
        results[0].boxes.conf,
        results[0].boxes.xyxy,
    ):
        if int(cls) != 0: # class 0 is "person" in COCO dataset
            continue
        bbox_area = float((box[2] - box[0]) * (box[3] - box[1]))
        fraction = bbox_area / image_area
        if float(conf) > conf_threshold and fraction > area_threshold:
            detections.append({"conf": float(conf), "bbox_area_fraction": fraction})

    return len(detections) > 0, detections  # if [0] is True, it should be rejected. [1] lists why.
