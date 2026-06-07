import pillow_heif
import PIL
pillow_heif.register_heif_opener()

def decode_image(path):
    heif_file = PIL.Image.open(path)
    image = heif_file.convert("RGB")
    return image, image.width, image.height