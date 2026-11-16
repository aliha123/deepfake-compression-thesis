"""Image compression utilities for robustness experiments.

The helpers in this module intentionally avoid model-specific assumptions so
they can be reused with different deepfake detectors and evaluation pipelines.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

from PIL import Image


SUPPORTED_FORMATS = {"JPEG", "WEBP"}


def compress_image(
    image: Image.Image,
    quality: int,
    image_format: str = "JPEG",
) -> Image.Image:
    """Return an in-memory compressed copy of an image.

    Args:
        image: Source PIL image.
        quality: Compression quality from 1 to 100.
        image_format: JPEG or WEBP.

    Returns:
        Decoded PIL image after compression.

    Raises:
        ValueError: If quality or format is unsupported.
    """
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")

    image_format = image_format.upper()
    if image_format not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported format: {image_format}")

    buffer = BytesIO()
    prepared = image.convert("RGB")
    prepared.save(buffer, format=image_format, quality=quality)
    buffer.seek(0)

    with Image.open(buffer) as compressed:
        return compressed.convert("RGB").copy()


def build_compression_grid(
    image: Image.Image,
    qualities: Iterable[int],
    image_format: str = "JPEG",
) -> dict[int, Image.Image]:
    """Compress one image at multiple quality levels."""
    return {
        quality: compress_image(image, quality, image_format)
        for quality in qualities
    }


def compress_file(
    input_path: str | Path,
    output_path: str | Path,
    quality: int,
    image_format: str = "JPEG",
) -> Path:
    """Compress an image from disk and write the result."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as image:
        compressed = compress_image(image, quality, image_format)
        compressed.save(output_path, format=image_format, quality=quality)

    return output_path
