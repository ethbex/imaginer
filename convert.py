from PIL import Image, ImageOps
import argparse
import os
from pathlib import Path

FORMAT_ALIASES = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}  # extend if you like

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def convert_image(image: str, fmt: str) -> str:
    """Convert to fmt and remove original. Returns the new path."""
    fmt = FORMAT_ALIASES.get(fmt.lower(), fmt.upper())
    base, _ = os.path.splitext(image)
    new_path = f"{base}.{fmt.lower()}"

    same_path = os.path.abspath(new_path) == os.path.abspath(image)

    with Image.open(image) as im:
        im = ImageOps.exif_transpose(im)
        # JPEG can't have alpha
        if fmt == "JPEG":
            if im.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            elif im.mode != "RGB":
                im = im.convert("RGB")
        im.save(new_path, format=fmt)

    if not same_path and os.path.exists(new_path) and os.path.exists(image):
        os.remove(image)

    return new_path

def max_size(image: str, max_width: int | None, max_height: int | None) -> None:
    with Image.open(image) as im:
        im = ImageOps.exif_transpose(im)
        w = max_width if max_width is not None else 10**9
        h = max_height if max_height is not None else 10**9
        im.thumbnail((w, h), Image.Resampling.LANCZOS)
        im.save(image)

def compress_image(image: str, quality: int) -> None:
    with Image.open(image) as im:
        fmt = (im.format or "").upper()
        save_kwargs = {}
        if fmt in ("JPG", "JPEG"):
            if im.mode != "RGB":
                im = im.convert("RGB")
            save_kwargs = dict(quality=quality, optimize=True, progressive=True, subsampling=2)
        elif fmt == "PNG":
            save_kwargs = dict(optimize=True)  # PNG: lossless; "quality" ignored
        elif fmt == "WEBP":
            save_kwargs = dict(quality=quality, method=6)  # needs Pillow with libwebp
        else:
            save_kwargs = dict(quality=quality)
        im.save(image, **save_kwargs)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('path', help='Path to an image file or a folder of images')
    p.add_argument('--format', choices=['jpeg', 'jpg', 'png', 'webp'], default=None)
    p.add_argument('--max-width', type=int, default=None)
    p.add_argument('--max-height', type=int, default=None)
    p.add_argument('--compress', type=int, default=None)
    return p.parse_args()

def process_one(file_path: Path, fmt, max_w, max_h, quality) -> None:
    current_path = str(file_path)

    if fmt is not None:
        current_path = convert_image(current_path, fmt)

    if (max_w is not None) or (max_h is not None):
        max_size(current_path, max_w, max_h)

    if quality is not None:
        compress_image(current_path, quality)

    print(f"Done: {current_path}")

if __name__ == "__main__":
    args = parse_args()
    p = Path(args.path)

    if p.is_dir():
        files = [f for f in p.iterdir() if is_image_file(f)]
        if not files:
            print("No image files found in that folder.")
        count = 0
        for f in files:
            try:
                process_one(f, args.format, args.max_width, args.max_height, args.compress)
                count += 1
            except Exception as e:
                print(f"Skipped {f}: {e}")
        print(f"Processed {count} file(s).")
    elif p.is_file():
        if not is_image_file(p):
            print("The provided file does not look like a supported image.")
        else:
            try:
                process_one(p, args.format, args.max_width, args.max_height, args.compress)
            except Exception as e:
                print(f"Error processing {p}: {e}")
    else:
        print("Path not found.")
