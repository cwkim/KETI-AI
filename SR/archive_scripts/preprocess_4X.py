#!/usr/bin/env python3
"""
Preprocess 4X magnification images - crop microdisks
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def detect_and_crop_microdisks(image_path, output_dir, min_area=500, max_area=50000):
    """Detect and crop microdisks from 4X images"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read {image_path}")
        return 0

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and crop
    count = 0
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if min_area < area < max_area:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)

            # Crop
            cropped = img[y:y+h, x:x+w]

            # Save
            filename = image_path.stem
            output_path = output_dir / f'{filename}_md_{count+1:03d}.png'
            cv2.imwrite(str(output_path), cropped)
            count += 1

    return count


def main():
    base_dir = Path('/home/keti/cwkim/KETI-AI/SR')
    origin_dir = base_dir / 'images' / 'origin_codes'
    output_base = base_dir / 'preprocessing' / 'final'

    # Find all 4X images
    image_files = list(origin_dir.glob('*/*4X*.tif'))

    print(f"Found {len(image_files)} 4X images")
    print("="*80)

    total_cropped = 0

    for img_path in tqdm(image_files, desc="Processing 4X images"):
        # Extract code number
        code_name = img_path.parent.name  # e.g., "code#56"
        mag = '4X'

        # Create output directory
        output_dir = output_base / f'cropped_{code_name}_{mag}'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process
        count = detect_and_crop_microdisks(img_path, output_dir)
        total_cropped += count

        print(f"  {img_path.name}: {count} microdisks cropped")

    print("="*80)
    print(f"\nTotal: {total_cropped} microdisks cropped from {len(image_files)} images")
    print(f"Output directory: {output_base}")

    # Summary by code
    print("\nSummary by code:")
    for code_dir in sorted(output_base.glob('cropped_*4X')):
        count = len(list(code_dir.glob('*.png')))
        print(f"  {code_dir.name}: {count} images")


if __name__ == '__main__':
    main()
