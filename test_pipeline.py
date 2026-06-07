import os
from iformat import iprint
from time import time


import numpy as np
from skimage.color import lab2rgb

import pipeline.decode
import pipeline.metadata
import pipeline.person_detect
import pipeline.embed
import pipeline.aesthetic

test_images = "local/images/"
css_output = "local/palette_preview.css"

palette_sample_size = 10_000

conf_threshold = 0.5
area_threshold = 0.01

model_name = "ViT-L-14-quickgelu"
checkpoint_path = "models/clip-vit-large-patch14.pt"
yolo_model_path = "models/yolo11n.pt"
aesthetic_checkpoint_path = "models/sac+logos+ava1-l14-linearMSE.pth"


def lab_to_hex(L, a, b):
    rgb = lab2rgb(np.array([[[L, a, b]]]))  # shape (1,1,3), values 0–1
    r, g, b = (np.clip(rgb[0, 0], 0, 1) * 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def image_to_css_block(filename, width, height, metadata):
    selector = "." + os.path.splitext(filename)[0].replace(" ", "-").replace(".", "-")
    lines = [
        f"/* {filename} */",
        f"{selector} {{",
        f"  width: {width};",
        f"  height: {height};",
        f"  native_orientation: {metadata['native_orientation']};",
        f"  value: {metadata['mean_value']:.3f};",
        f"  contrast: {metadata['contrast']:.3f};",
    ]
    for i, p in enumerate(metadata["palette"], 1):
        lines.append(f"  c{i}: {lab_to_hex(*p[:3])}; /* {p[3]:.1%} */")
    lines.append("}")
    return "\n".join(lines)


results = []
times = {}
for filename in sorted(os.listdir(test_images)):
    if filename.lower().endswith((".heic", ".heif", ".jpg", ".jpeg", ".png", ".tiff")):
        print(f"Processing {filename}...")
        start_time = time()
        path = os.path.join(test_images, filename)
        image, width, height = pipeline.decode.decode_image(path)
        done_decode = time()
        metadata = pipeline.metadata.extract_metadata(image, width, height, sample_size=palette_sample_size)
        done_metadata = time()
        person, detections = pipeline.person_detect.detect_persons(image, width, height, model_path=yolo_model_path, conf_threshold=conf_threshold, area_threshold=area_threshold)
        done_person = time()
        embedding = pipeline.embed.embed_image(image, model_name=model_name, checkpoint_path=checkpoint_path)
        done_embed = time()
        aesthetic_score = pipeline.aesthetic.score_image(embedding, aesthetic_checkpoint_path)
        print(f"Aesthetic score: {aesthetic_score:.3f}")
        done_aesthetic = time()
        end_time = time()
        times[filename] = {
            "width": width,
            "height": height,
            "total": end_time - start_time,
            "decode": done_decode - start_time,
            "metadata": done_metadata - done_decode,
            "person": done_person - done_metadata,
            "embed": done_embed - done_person,
            "aesthetic": done_aesthetic - done_embed
        }
        print("=======")
        #results.append(image_to_css_block(filename, width, height, metadata))

print("Timing summary:")
for filename, t in times.items():
    print(f"{filename} ({t['width']}x{t['height']}): total {t['total']:.2f}s "
          f"(decode {t['decode']:.2f}s, metadata {t['metadata']:.2f}s, "
          f"person {t['person']:.2f}s, embed {t['embed']:.2f}s, aesthetic {t['aesthetic']:.2f}s)")

if times:
    times.pop(next(iter(times)))  # drop the FIRST item (model-load warmup)
    num_images = len(times)
    keys = ['total', 'decode', 'metadata', 'person', 'embed', 'aesthetic']
    avgs = {k: sum(t[k] for t in times.values()) / num_images for k in keys}

    print(f"\nProcessed {num_images} images in {avgs['total']*num_images:.2f}s total")
    print(f"Average per image: {avgs['total']:.2f}s")
    for k in keys[1:]:
        print(f"  - {k}: {avgs[k]:.2f}s")
    

""" with open(css_output, "w") as f:
    f.write("\n\n".join(results) + "\n") """