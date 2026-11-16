from PIL import Image
import pytest

from src.compression import build_compression_grid, compress_image


def test_compress_image_preserves_size():
    image = Image.new("RGB", (32, 24), color="white")
    result = compress_image(image, quality=70)
    assert result.size == image.size
    assert result.mode == "RGB"


def test_compression_grid_uses_requested_qualities():
    image = Image.new("RGB", (16, 16), color="black")
    grid = build_compression_grid(image, qualities=[95, 75, 50])
    assert set(grid) == {95, 75, 50}


@pytest.mark.parametrize("quality", [0, 101])
def test_invalid_quality_raises(quality):
    image = Image.new("RGB", (8, 8), color="white")
    with pytest.raises(ValueError):
        compress_image(image, quality=quality)


def test_invalid_format_raises():
    image = Image.new("RGB", (8, 8), color="white")
    with pytest.raises(ValueError):
        compress_image(image, quality=80, image_format="PNG")
