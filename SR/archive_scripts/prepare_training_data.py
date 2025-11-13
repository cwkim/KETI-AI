#!/usr/bin/env python3
"""
Prepare training data for Real-ESRGAN fine-tuning
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def prepare_training_data(base_dir='/home/keti/cwkim/KETI-AI/SR'):
    """Prepare HR and LR image pairs for training"""
    base_dir = Path(base_dir)
    cropped_dir = base_dir / 'preprocessing' / 'final'
    output_dir = base_dir / 'results' / 'train'

    train_hr_dir = output_dir / 'hr'
    train_lr_dir = output_dir / 'lr'

    train_hr_dir.mkdir(parents=True, exist_ok=True)
    train_lr_dir.mkdir(parents=True, exist_ok=True)

    # Get all cropped images
    image_files = []
    for folder in cropped_dir.glob('cropped_*'):
        if folder.is_dir():
            image_files.extend(list(folder.glob('*.png')))
            image_files.extend(list(folder.glob('*.jpg')))

    print(f"Found {len(image_files)} cropped images")

    # Process images
    valid_count = 0
    min_size = 64  # Minimum size for training (reduced for small microdisk images)

    for img_path in tqdm(image_files, desc="Preparing training data"):
        try:
            # Read image
            hr_img = cv2.imread(str(img_path))
            if hr_img is None:
                continue

            h, w = hr_img.shape[:2]

            # Skip if too small
            if h < min_size or w < min_size:
                continue

            # Upscale small images to make them larger (for better training)
            if h < 128 or w < 128:
                scale_factor = max(128 / h, 128 / w)
                new_h = int(h * scale_factor)
                new_w = int(w * scale_factor)
                hr_img = cv2.resize(hr_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                h, w = new_h, new_w

            # Resize to multiple of 4 if needed
            new_h = (h // 4) * 4
            new_w = (w // 4) * 4

            if new_h != h or new_w != w:
                hr_img = cv2.resize(hr_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # Save HR image
            hr_path = train_hr_dir / f'{valid_count:05d}.png'
            cv2.imwrite(str(hr_path), hr_img)

            # Create LR image by downsampling
            lr_img = cv2.resize(hr_img, (new_w//4, new_h//4), interpolation=cv2.INTER_CUBIC)

            # Optional: Add slight blur to LR to simulate real degradation
            lr_img = cv2.GaussianBlur(lr_img, (3, 3), 0.5)

            # Save LR image
            lr_path = train_lr_dir / f'{valid_count:05d}.png'
            cv2.imwrite(str(lr_path), lr_img)

            valid_count += 1

            if valid_count >= 500:  # Limit to 500 training pairs
                break

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"\nSuccessfully prepared {valid_count} training pairs")
    print(f"HR images: {train_hr_dir}")
    print(f"LR images: {train_lr_dir}")

    return valid_count


if __name__ == '__main__':
    count = prepare_training_data()
    if count == 0:
        print("\nERROR: No training data prepared!")
    else:
        print(f"\nReady to train with {count} image pairs!")
