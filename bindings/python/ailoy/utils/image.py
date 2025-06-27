import base64
import io

from PIL.Image import Image


def pillow_image_to_base64(img: Image):
    buffered = io.BytesIO()
    img.save(buffered, format=img.format)
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{img.format.lower()};base64,{b64}"
