"""Generate visual previews of YOLO annotations for manual inspection."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .dataset_utils import (
    load_class_mapping,
    load_metadata_file,
    load_yaml_file,
    project_root,
    resolve_config_path,
    resolve_dataset_root,
    validate_label_file,
)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_image_file(image_dir: Path, image_id: str) -> Path | None:
    """Find an image file by image ID or stem."""

    for extension in sorted(SUPPORTED_IMAGE_EXTENSIONS):
        candidate = image_dir / f"{image_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def _text_box(draw: ImageDraw.ImageDraw, position: tuple[int, int], text: str) -> tuple[int, int, int, int]:
    """Measure a text box using a backward-compatible Pillow call."""

    try:
        left, top, right, bottom = draw.textbbox(position, text)
        return left, top, right, bottom
    except AttributeError:
        width, height = draw.textsize(text)
        x, y = position
        return x, y, x + width, y + height


def _draw_label(draw: ImageDraw.ImageDraw, position: tuple[int, int], text: str) -> None:
    """Draw a filled label badge behind annotation text."""

    left, top, right, bottom = _text_box(draw, position, text)
    padding = 4
    draw.rectangle(
        [left - padding, top - padding, right + padding, bottom + padding],
        fill=(0, 0, 0),
    )
    draw.text(position, text, fill=(255, 255, 255))


def generate_preview(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    class_names: dict[int, str],
    image_id: str,
    capture_session: str | None = None,
) -> list[str]:
    """Generate a preview image with boxes and labels, returning any issues found."""

    issues: list[str] = []
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        annotations, label_issues = validate_label_file(label_path, width, height, len(class_names))
        for label_issue in label_issues:
            issues.append(label_issue.message)

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        line_width = max(2, min(width, height) // 240)
        top_text = f"Image ID: {image_id}"
        if capture_session:
            top_text += f" | Session: {capture_session}"
        _draw_label(draw, (10, 10), top_text)

        for annotation in annotations:
            left = (annotation.x_center - annotation.width / 2.0) * width
            top = (annotation.y_center - annotation.height / 2.0) * height
            right = (annotation.x_center + annotation.width / 2.0) * width
            bottom = (annotation.y_center + annotation.height / 2.0) * height
            class_name = class_names.get(annotation.class_id, f"Class {annotation.class_id}")
            label_text = f"{class_name}"

            draw.rectangle([left, top, right, bottom], outline=(255, 215, 0), width=line_width)
            label_position = (max(10, int(left)), max(30, int(top) - 18))
            _draw_label(draw, label_position, label_text)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    return issues


def collect_preview_images(image_dir: Path) -> list[Path]:
    """Collect source images eligible for preview generation."""

    if not image_dir.exists():
        return []
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)


def main() -> int:
    """Generate annotation previews from the command line."""

    parser = argparse.ArgumentParser(description="Generate annotation previews for the Sardinella Lemuru dataset.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--image-id", type=str, help="Generate a preview for one image ID")
    selection.add_argument("--sample", type=int, help="Generate previews for a random sample of images")
    selection.add_argument("--all", action="store_true", help="Generate previews for all images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where previews will be written",
    )
    args = parser.parse_args()

    repo_root = project_root()
    dataset_config = load_yaml_file(repo_root / "configs" / "dataset.yaml")
    classes_config = repo_root / "configs" / "classes.yaml"
    class_names = load_class_mapping(classes_config, dataset_config)
    metadata = load_metadata_file(repo_root / "dataset" / "metadata.csv")
    dataset_root = resolve_dataset_root(dataset_config, repo_root)
    image_dir = resolve_config_path(dataset_root, "dataset/annotated/images")
    label_dir = resolve_config_path(dataset_root, "dataset/annotated/labels")
    output_dir = args.output_dir or (repo_root / "dataset" / "annotation_previews")

    if not class_names:
        print("ERROR: Class names are not configured.")
        return 1

    images = collect_preview_images(image_dir)
    if not images:
        print("ERROR: No annotated images are available for preview generation.")
        return 1

    if args.image_id:
        selected_images = [find_image_file(image_dir, args.image_id)]
    elif args.sample is not None:
        random_generator = random.Random(args.seed)
        sample_size = min(args.sample, len(images))
        selected_images = random_generator.sample(images, sample_size)
    else:
        selected_images = images

    selected_images = [image for image in selected_images if image is not None]
    if not selected_images:
        print("ERROR: The requested image could not be found.")
        return 1

    generated_count = 0
    warnings: list[str] = []
    for image_path in selected_images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            warnings.append(f"WARNING: {image_path.name} has no matching label file.")
            continue

        output_path = output_dir / f"{image_path.stem}_preview.png"
        capture_session = metadata.session_for(image_path.stem)
        issues = generate_preview(
            image_path=image_path,
            label_path=label_path,
            output_path=output_path,
            class_names=class_names,
            image_id=image_path.stem,
            capture_session=capture_session,
        )
        for issue in issues:
            warnings.append(f"WARNING: {image_path.name} - {issue}")
        generated_count += 1

    print(f"Generated {generated_count} annotation preview(s) in {output_dir}")
    for warning in warnings:
        print(warning)

    return 0 if generated_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
