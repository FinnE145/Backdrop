import numpy as np
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from time import time

SAMPLE_SIZE = 10_000


def _sample_lab_pixels(image, n=SAMPLE_SIZE, seed=527737):
    # Numpy array of shape (height, width, 3) with values in [0, 1]
    arr = np.array(image, dtype=np.float32) / 255.0
    # Flat list of RGB pixels (width * height, 3)
    rgb_pixels = arr.reshape(-1, 3)
    # To avoid converting the whole image, take a random sample of pixels first
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rgb_pixels), size=min(n, len(rgb_pixels)))
    sampled_rgb = rgb_pixels[idx]
    # Convert the sampled pixels to CIELAB format
    lab_sampled = rgb2lab(sampled_rgb)
    # Final shape (sample_size, 3) with values in CIELAB format
    return lab_sampled


def extract_metadata(image, width, height, sample_size=SAMPLE_SIZE, palette_k=6):
    if width > height:
        orientation = "landscape"
    elif height > width:
        orientation = "portrait"
    else:
        orientation = "square"

    #start_time = time()
    downsampled_LAB = _sample_lab_pixels(image, n=sample_size)
    #done_sample = time()

    L = downsampled_LAB[:, 0]
    value = L.mean() / 100.0
    contrast = L.std() / 100.0
    #done_value_contrast = time()
    kmeans = KMeans(n_clusters=palette_k).fit(downsampled_LAB)
    #done_kmeans_fit = time()
    centers = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    fractions = counts / counts.sum()
    palette = [list(center) + [float(frac)] for center, frac in zip(centers, fractions)]
    #done_palette_calcs = time()
    #print(f"Metadata extraction times: sample {done_sample - start_time:.2f}s, value/contrast {done_value_contrast - done_sample:.2f}s, kmeans fit {done_kmeans_fit - done_value_contrast:.2f}s, palette calcs {done_palette_calcs - done_kmeans_fit:.2f}s")

    return {
        "native_orientation": orientation,
        "mean_value": value,
        "contrast": contrast,
        "palette": palette,
    }