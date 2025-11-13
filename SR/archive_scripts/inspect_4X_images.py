#!/usr/bin/env python3
"""
Inspect 4X images to understand structure
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def inspect_image(image_path):
    """Inspect a 4X image"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read {image_path}")
        return

    print(f"\nImage: {image_path.name}")
    print(f"Size: {img.shape[1]}x{img.shape[0]}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try different thresholding methods
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_150 = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_otsu_morph = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel)
    binary_otsu_morph = cv2.morphologyEx(binary_otsu_morph, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours_otsu, _ = cv2.findContours(binary_otsu_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Total contours found: {len(contours_otsu)}")

    # Analyze contour sizes
    areas = [cv2.contourArea(c) for c in contours_otsu]
    if areas:
        areas_sorted = sorted(areas, reverse=True)
        print(f"Top 10 contour areas: {areas_sorted[:10]}")
        print(f"Min area: {min(areas)}, Max area: {max(areas)}")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(blurred, cmap='gray')
    axes[0, 2].set_title('Blurred')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(binary_otsu, cmap='gray')
    axes[1, 0].set_title('Binary (Otsu)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(binary_otsu_morph, cmap='gray')
    axes[1, 1].set_title(f'Morphology ({len(contours_otsu)} contours)')
    axes[1, 1].axis('off')

    # Draw contours with area filtering
    img_contours = img.copy()
    filtered_count = 0
    for contour in contours_otsu:
        area = cv2.contourArea(contour)
        if 500 < area < 50000:
            cv2.drawContours(img_contours, [contour], -1, (0, 255, 0), 2)
            filtered_count += 1

    axes[1, 2].imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'Contours (filtered: {filtered_count})')
    axes[1, 2].axis('off')

    plt.tight_layout()
    output_path = Path('inspection_results') / f'inspect_{image_path.stem}.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved inspection to: {output_path}")


def main():
    base_dir = Path('/home/keti/cwkim/KETI-AI/SR')
    origin_dir = base_dir / 'images' / 'origin_codes'

    # Select a few representative 4X images
    sample_images = [
        origin_dir / 'code#56' / '56_4X_10.tif',
        origin_dir / 'code#79' / '79_4X_3.tif',
        origin_dir / 'code#68' / '68_4X_8.tif',
    ]

    for img_path in sample_images:
        if img_path.exists():
            inspect_image(img_path)


if __name__ == '__main__':
    main()
