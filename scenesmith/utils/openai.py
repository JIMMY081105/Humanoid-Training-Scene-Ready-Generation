import base64

from io import BytesIO
from pathlib import Path

import numpy as np

from PIL import Image


def encode_image_to_base64(image: np.ndarray | str | Path) -> str:
    """Encode an image as PNG bytes and return their base64 payload.

    Every SceneSmith caller labels this payload ``data:image/png;base64``. The
    encoder must therefore produce actual PNG bytes; emitting JPEG bytes under
    that MIME type can be rejected by multimodal provider validation and leaves
    visual tool grounding undefined.

    Args:
        image: Either a numpy array of shape (H, W, 3) in RGB format, a path string,
            or a Path object to an image file.

    Returns:
        str: The base64 encoded image string.
    """
    if isinstance(image, (str, Path)):
        # Read image directly from path.
        with Image.open(image) as img:
            # Convert to RGB in case it's not.
            img = img.convert("RGB")
            # Save to bytes.
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        # Convert numpy array to PIL Image.
        img = Image.fromarray(image)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
