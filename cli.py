"""
imaginer: A CLI tool for simple image workflows

Subcommands:
  - convert: Convert/resize/compress images (single file or folder)
  - caption: Generate captions and rename images accordingly (file or folder)

Install (with uv):
  uv tools install .

Usage examples:
  imaginer convert ./image.jpg --format webp --max-width 1200 --compress 80
  imaginer convert ./images --max-width 1024
  imaginer caption ./image.jpg --case title --glue "-"
  imaginer caption ./images --prefix "IMG_" --case lower
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="imaginer", description="Image conversion and captioning CLI")
	subparsers = parser.add_subparsers(dest="command", required=True)

	# convert subcommand
	p_conv = subparsers.add_parser("convert", help="Convert/resize/compress images")
	p_conv.add_argument("path", help="Path to an image file or a folder of images")
	p_conv.add_argument("--format", choices=["jpeg", "jpg", "png", "webp"], default=None)
	p_conv.add_argument("--max-width", type=int, default=None)
	p_conv.add_argument("--max-height", type=int, default=None)
	p_conv.add_argument("--compress", type=int, default=None, help="Compression quality (e.g., 80)")

	# caption subcommand
	p_cap = subparsers.add_parser("caption", help="Generate captions and rename images")
	p_cap.add_argument("path", help="Path to an image file or a folder of images")
	p_cap.add_argument("--prefix", type=str, default=None)
	p_cap.add_argument("--suffix", type=str, default=None)
	p_cap.add_argument("--glue", type=str, default=None, help="Replace spaces with this string in caption")
	p_cap.add_argument(
		"--case",
		type=str,
		choices=["upper", "lower", "title", "sentence"],
		default=None,
		help="Change caption case",
	)

	return parser


def handle_convert(args: argparse.Namespace) -> int:
	# Lazy import to avoid requiring Pillow unless this command is used
	from convert import (
		IMAGE_EXTS,
		is_image_file,
		convert_image,
		max_size,
		compress_image,
	)

	p = Path(args.path)

	def process_one(file_path: Path):
		current_path = str(file_path)
		if args.format is not None:
			current_path = convert_image(current_path, args.format)
		if (args.max_width is not None) or (args.max_height is not None):
			max_size(current_path, args.max_width, args.max_height)
		if args.compress is not None:
			compress_image(current_path, args.compress)
		print(f"Done: {current_path}")

	if p.is_dir():
		files = [f for f in p.iterdir() if is_image_file(f)]
		if not files:
			print("No image files found in that folder.")
			return 0
		count = 0
		for f in files:
			try:
				process_one(f)
				count += 1
			except Exception as e:  # noqa: BLE001 - CLI surface prints errors
				print(f"Skipped {f}: {e}")
		print(f"Processed {count} file(s).")
		return 0
	elif p.is_file():
		if p.suffix.lower() not in IMAGE_EXTS:
			print("The provided file does not look like a supported image.")
			return 2
		try:
			process_one(p)
			return 0
		except Exception as e:
			print(f"Error processing {p}: {e}")
			return 1
	else:
		print("Path not found.")
		return 2


def handle_caption(args: argparse.Namespace) -> int:
	# Lazy import to avoid requiring Transformers/Torch unless this command is used
	from caption import (
		generate_caption,
		normalise,
		glue,
		prefix as add_prefix,
		suffix as add_suffix,
		case as apply_case,
		rename_file,
	)
	from convert import IMAGE_EXTS

	p = Path(args.path)

	def process_one(image_path: Path):
		cap = generate_caption(str(image_path))
		cap = normalise(cap)
		if args.glue:
			cap = glue(cap, args.glue)
		if args.prefix:
			cap = add_prefix(cap, args.prefix)
		if args.suffix:
			cap = add_suffix(cap, args.suffix)
		if args.case:
			cap = apply_case(cap, args.case)
		# keep original extension when renaming
		new_name = cap + image_path.suffix
		rename_file(str(image_path), new_name)
		print(cap)

	def image_files_in_dir(dir_path: Path):
		return [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]

	if p.is_dir():
		files = image_files_in_dir(p)
		if not files:
			print("No image files found in that folder.")
			return 0
		for f in files:
			try:
				process_one(f)
			except Exception as e:
				print(f"Skipped {f}: {e}")
		return 0
	elif p.is_file():
		if p.suffix.lower() not in IMAGE_EXTS:
			print("The provided file does not look like a supported image.")
			return 2
		try:
			process_one(p)
			return 0
		except Exception as e:
			print(f"Error processing {p}: {e}")
			return 1
	else:
		print("Path not found.")
		return 2


def main(argv: list[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	if args.command == "convert":
		return handle_convert(args)
	if args.command == "caption":
		return handle_caption(args)
	parser.print_help()
	return 2


if __name__ == "__main__":
	raise SystemExit(main())

