from PIL import Image
import pytest

from src import image_analysis


def test_no_image():
    result = image_analysis.analyze_image(None)

    assert result["image_provided"] is False
    assert result["caption"] is None
    assert result["model_id"] is None

def test_valid_image(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.jpg"

    Image.new(
        "RGB",
        (400, 300),
        color="gray",
    ).save(image_path)

    monkeypatch.setattr(
        image_analysis,
        "caption_image",
        lambda image: "a car parked outside",
    )

    result = image_analysis.analyze_image(image_path)

    assert result["image_provided"] is True
    assert result["caption"] == "a car parked outside"

    assert result["model_id"] == image_analysis.MODEL_ID
    assert result["analysis_type"] == "image_caption"

def test_invalid_image(tmp_path):
    image_path = tmp_path / "invalid.jpg"
    image_path.write_text("not an image")

    with pytest.raises(ValueError):
        image_analysis.analyze_image(image_path)

def test_missing_image_path(tmp_path):
    image_path = tmp_path / "missing.jpg"

    with pytest.raises(FileNotFoundError):
        image_analysis.analyze_image(image_path)

def test_image_too_small(tmp_path):
    width, height = 50, 50
    color = (0, 102, 204)
    img = Image.new("RGB", (width, height), color)
    image_path = tmp_path / "small.jpg"
    img.save(image_path)

    with pytest.raises(ValueError):
        image_analysis.load_image(image_path)

def test_no_image_skips_model(monkeypatch):
    def unexpected_caption_call(image):
        raise AssertionError("Captioner must not run.")

    monkeypatch.setattr(
        image_analysis,
        "caption_image",
        unexpected_caption_call,
    )

    result = image_analysis.analyze_image(None)

    assert result["image_provided"] is False
