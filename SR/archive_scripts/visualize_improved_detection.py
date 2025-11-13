#!/usr/bin/env python3
"""
Visualize improved microdisk detection results for 4X images
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def visualize_detection_results(original_path, cropped_dir, output_path):
    """Visualize original image with detected microdisk bounding boxes"""
    # Read original
    img = cv2.imread(str(original_path))
    if img is None:
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Left: Original
    axes[0].imshow(img_rgb)
    axes[0].set_title(f'Original 4X Image\n{img.shape[1]}x{img.shape[0]}',
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Right: With detection boxes
    img_annotated = img_rgb.copy()

    # Find all cropped images from this original
    filename_base = original_path.stem
    cropped_files = list(cropped_dir.glob(f'{filename_base}_md_*.png'))

    # For each cropped file, we need to find its location
    # Since we don't store the original coordinates, we'll do a template matching
    count = 0
    for cropped_path in sorted(cropped_files):
        cropped = cv2.imread(str(cropped_path))
        if cropped is None:
            continue

        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # Simple template matching to find location
        h, w = cropped_rgb.shape[:2]
        if h < 10 or w < 10:
            continue

        # Use template matching
        result = cv2.matchTemplate(img_rgb, cropped_rgb, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # If good match, draw box
        if max_val > 0.8:
            x, y = max_loc
            cv2.rectangle(img_annotated, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img_annotated, f'{count+1}', (x+5, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            count += 1

    axes[1].imshow(img_annotated)
    axes[1].set_title(f'Detected Microdisks: {len(cropped_files)} found',
                     fontsize=14, fontweight='bold', color='green')
    axes[1].axis('off')

    plt.suptitle(f'Improved Detection Result: {original_path.name}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return True


def main():
    base_dir = Path('/home/keti/cwkim/KETI-AI/SR')
    origin_dir = base_dir / 'images' / 'origin_codes'
    cropped_base = base_dir / 'preprocessing' / 'final_4X_improved'
    output_dir = base_dir / 'detection_visualization'
    output_dir.mkdir(exist_ok=True)

    # Select representative images
    samples = [
        ('code#56', '56_4X_3.tif'),
        ('code#68', '68_4X_3.tif'),
        ('code#75', '75_4X_3.tif'),
        ('code#79', '79_4X_3.tif'),
    ]

    for code, filename in samples:
        original_path = origin_dir / code / filename
        cropped_dir = cropped_base / f'cropped_{code}_4X'

        if not original_path.exists():
            continue

        output_path = output_dir / f'detection_{filename.replace(".tif", ".png")}'

        print(f"Processing {filename}...")
        success = visualize_detection_results(original_path, cropped_dir, output_path)

        if success:
            print(f"  Saved to: {output_path}")

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
