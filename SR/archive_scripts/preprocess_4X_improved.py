#!/usr/bin/env python3
"""
Improved preprocessing for 4X images - detect and crop individual microdisks
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def detect_and_crop_microdisks_4X(image_path, output_dir,
                                   min_area=1000, max_area=20000,
                                   circularity_thresh=0.6):
    """
    Detect and crop individual microdisks from 4X images
    4X images have many small microdisks in a circular field
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read {image_path}")
        return 0

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to better separate microdisks
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)

    # Morphological operations to clean up
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and crop based on area and circularity
    count = 0
    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by area
        if min_area < area < max_area:
            # Calculate circularity (4π*area/perimeter²)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Filter by circularity (microdisks should be roughly circular)
            if circularity > circularity_thresh:
                valid_contours.append(contour)

    # Sort contours by area (largest first) and take reasonable ones
    valid_contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    # Crop each valid contour
    for idx, contour in enumerate(valid_contours):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Make square crop (use max of w, h)
        size = max(w, h)
        center_x = x + w // 2
        center_y = y + h // 2

        # Add padding
        padding = 10
        half_size = size // 2 + padding

        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(img.shape[1], center_x + half_size)
        y2 = min(img.shape[0], center_y + half_size)

        # Crop
        cropped = img[y1:y2, x1:x2]

        # Skip if too small
        if cropped.shape[0] < 20 or cropped.shape[1] < 20:
            continue

        # Save
        filename = image_path.stem
        output_path = output_dir / f'{filename}_md_{count+1:03d}.png'
        cv2.imwrite(str(output_path), cropped)
        count += 1

    return count


def main():
    base_dir = Path('/home/keti/cwkim/KETI-AI/SR')
    origin_dir = base_dir / 'images' / 'origin_codes'
    output_base = base_dir / 'preprocessing' / 'final_4X_improved'

    # Remove old directory if exists
    if output_base.exists():
        import shutil
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

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

        # Process with improved detection
        count = detect_and_crop_microdisks_4X(img_path, output_dir,
                                               min_area=1000, max_area=20000,
                                               circularity_thresh=0.6)
        total_cropped += count

        if count > 0:
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
