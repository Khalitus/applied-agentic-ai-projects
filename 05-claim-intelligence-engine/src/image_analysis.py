import argparse
import json
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


MODEL_ID = "Salesforce/blip-image-captioning-base"

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_SIZE = (1024, 1024)

VISUAL_LABELS = [
    "a car with a dented front bumper",
    "a car with rear bumper damage",
    "a car with cracked or broken windows",
    "a car with side panel damage",
    "a car with visible hail dents",
    "a car with no obvious exterior damage",
]

def load_image(image_path: str | Path) -> Image.Image:
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(
            "Image exceeds the 10 MB upload limit."
        )

    try:
        with Image.open(path) as raw:
            if raw.format not in SUPPORTED_FORMATS:
                raise ValueError(
                    "Supported formats: JPEG, PNG, WebP."
                )

            width, height = raw.size

            if width < 128 or height < 128:
                raise ValueError("Image's width or height is below 128 pixels.")

            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    "Image dimensions are too large."
                )

            image = ImageOps.exif_transpose(raw).convert("RGB")
            image.thumbnail(MAX_IMAGE_SIZE)

            return image

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            "Unable to read the uploaded image."
        ) from exc

@lru_cache(maxsize=1)
def load_captioner():
    from transformers import (
        BlipForConditionalGeneration,
        BlipProcessor,
    )

    processor = BlipProcessor.from_pretrained(MODEL_ID)

    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_ID
    )

    model.eval()

    return processor, model

def caption_image(image: Image.Image) -> str:
    import torch

    processor, model = load_captioner()

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=2,
        )

    caption = processor.decode(
        output_ids[0],
        skip_special_tokens=True,
    ).strip()

    if not caption:
        raise ValueError("Image caption generation returned empty text.")
       
       
    return caption

def analyze_image(
        image_path: str | Path | None,
        include_categories: bool = False,
    ) -> dict:
    if image_path is None:
        return {
            "image_provided": False,
            "caption": None,
            "model_id": None,
            "analysis_type": None,
            "limitations": "No photograph provided.",
        }

    image = load_image(image_path)

    caption = caption_image(image)

    return {
        "image_provided": True,
        "caption": caption,
        "model_id": MODEL_ID,
        "analysis_type": "image_caption",
        "limitations": "Automated visual description; not verified damage classification.",
    }

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


@lru_cache(maxsize=1)
def load_visual_classifier():
    from transformers import pipeline

    return pipeline(
        task="zero-shot-image-classification",
        model=CLIP_MODEL_ID,
        device=-1,
    )

def classify_image(image: Image.Image) -> list[dict]:
    classifier = load_visual_classifier()

    results = classifier(
        image,
        candidate_labels=VISUAL_LABELS,
    )

    return [
        {"label": result["label"], "score": round(result["score"], 4)}
        for result in results[:3]
    ]

def main():
    parser = argparse.ArgumentParser(
        description="Analyze an optional claim photograph."
    )

    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to a local JPG, PNG, or WebP image.",
    )

    args = parser.parse_args()

    result = analyze_image(args.image)

    print("\nImage analysis")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
