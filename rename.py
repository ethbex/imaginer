#!/usr/bin/env python3
import argparse
import os
import re
from typing import List, Callable, Dict

# --- Config ---
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}  # extend if you like
_MODEL_CACHE: Dict[str, Callable[[str], str]] = {}  # lazy, per-model cache for the captioner

# --- Captioning ---
def _get_captioner(model_size: str):
    """Lazy-load and cache an image->text caption function for the chosen model size."""
    if model_size in _MODEL_CACHE:
        return _MODEL_CACHE[model_size]

    # lazy imports to avoid heavy startup when not used
    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
    from PIL import Image

    model_map = {
        "small": "Salesforce/blip-image-captioning-base",
        "large": "Salesforce/blip-image-captioning-large",
    }
    model_id = model_map.get(model_size, model_map["small"])

    # load processor + model (these are the recommended objects for vision->text models)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id)

    # Some processors expose a tokenizer via `processor.tokenizer`; if not, fall back to AutoTokenizer.
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    def caption(image_path: str) -> str:
        """Return a generated caption string for image_path."""
        image = Image.open(image_path).convert("RGB")
        # processor will prepare pixel_values and any other required inputs
        inputs = processor(images=image, return_tensors="pt")

        # generate token ids
        generated_ids = model.generate(**inputs)

        # decode to text; prefer processor.decode if available, otherwise tokenizer.decode
        decode_fn = getattr(processor, "decode", None)
        if callable(decode_fn):
            text = decode_fn(generated_ids[0], skip_special_tokens=True)
        else:
            # tokenizer may require passing the ids as a list of ints
            text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # ensure string type
        return str(text)

    _MODEL_CACHE[model_size] = caption
    return caption

def generate_name(image_path: str, model_size: str = "small") -> str:
    """
    Generate a caption and return it as a string.
    Keeps the previous behavior (returning a plain string) so the rest of your pipeline can use it.
    """
    cap = _get_captioner(model_size)
    return cap(image_path)

# --- String transforms ---
def prefix(name: str, string: str) -> str:
    return string + name

def suffix(name: str, string: str) -> str:
    return name + string

def glue(name: str, string: str) -> str:
    return name.replace(" ", string)

def case(name: str, case_type: str) -> str:
    if case_type == "upper":
        return name.upper()
    if case_type == "lower":
        return name.lower()
    if case_type == "title":
        return name.title()
    if case_type == "sentence":
        return name.capitalize()
    return name

def normalise(name: str) -> str:
    # keep only letters, digits, spaces
    name = re.sub(r"[^A-Za-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    name = name.strip()
    # fall back if empty
    return name or "image"

# --- IO helpers ---
def _is_image_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTS

def _list_images_in_dir(folder: str) -> List[str]:
    try:
        entries = os.listdir(folder)
    except FileNotFoundError:
        return []
    files = []
    for name in entries:
        p = os.path.join(folder, name)
        if _is_image_file(p):
            files.append(p)
    return files

def rename_file(path: str, new_name_with_ext: str) -> None:
    dir_name = os.path.dirname(path)
    base, ext = os.path.splitext(new_name_with_ext)
    # ensure we keep the original file's extension if caller passed only name
    if not ext:
        ext = os.path.splitext(path)[1]
    new_path = os.path.join(dir_name, base + ext)

    # avoid collisions: append _1, _2, ...
    counter = 1
    while os.path.exists(new_path):
        new_path = os.path.join(dir_name, f"{base}_{counter}{ext}")
        counter += 1

    os.rename(path, new_path)

# --- CLI ---
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Path to an image file or a folder of images")
    p.add_argument("--prefix", type=str, default=None)
    p.add_argument("--suffix", type=str, default=None)
    p.add_argument("--glue", type=str, default=None)
    p.add_argument("--case", type=str, choices=["upper", "lower", "title", "sentence"], default=None)
    p.add_argument("--model", type=str, choices=["small", "large"], default="small", help="Caption model size")
    return p.parse_args()

# --- Main ---
if __name__ == "__main__":
    args = parse_args()

    # resolve input(s)
    if os.path.isdir(args.image):
        image_paths = _list_images_in_dir(args.image)
        if not image_paths:
            print("No supported images found in that folder.")
            raise SystemExit(0)
    else:
        if not _is_image_file(args.image):
            print("Provided path is not a supported image file.")
            raise SystemExit(1)
        image_paths = [args.image]

    # process
    for image_path in image_paths:
        try:
            name = generate_name(image_path, args.model)
        except Exception as e:
            print(f"Caption failed for {image_path}: {e}")
            continue

        name = normalise(name)

        if args.glue:
            name = glue(name, args.glue)
        if args.prefix:
            name = prefix(name, args.prefix)
        if args.suffix:
            name = suffix(name, args.suffix)
        if args.case:
            name = case(name, args.case)

        # ensure final filename keeps original extension
        ext = os.path.splitext(image_path)[1]
        try:
            rename_file(image_path, name + ext)
            print(f"Renamed: {os.path.basename(image_path)} → {name}{ext}")
        except Exception as e:
            print(f"Rename failed for {image_path}: {e}")
