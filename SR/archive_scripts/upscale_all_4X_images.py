#!/usr/bin/env python3
"""
Upscale all 4X images from original_codes with fine-tuned model
"""

import cv2
import numpy as np
import torch
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append('AI_models/Real-ESRGAN')
from basicsr.archs.rrdbnet_arch import RRDBNet


def enhance_with_model(model, img, device, tile_size=400, overlap=20):
    """
    Enhance image with model using tiling for large images
    """
    h, w = img.shape[:2]
    output_h, output_w = h * 4, w * 4
    output = np.zeros((output_h, output_w, 3), dtype=np.float32)
    weight = np.zeros((output_h, output_w, 3), dtype=np.float32)

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            # Extract tile
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = img[y:y_end, x:x_end]

            # Convert to tensor
            tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
            tile_tensor = tile_tensor.unsqueeze(0).to(device)

            # Process tile
            with torch.no_grad():
                output_tensor = model(tile_tensor)

            # Convert back to numpy
            tile_output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Place in output with blending
            out_y = y * 4
            out_x = x * 4
            out_y_end = min(out_y + tile_output.shape[0], output_h)
            out_x_end = min(out_x + tile_output.shape[1], output_w)

            tile_h = out_y_end - out_y
            tile_w = out_x_end - out_x

            # Create weight map for blending (reduce edge artifacts)
            tile_weight = np.ones((tile_h, tile_w, 3), dtype=np.float32)

            # Feather edges if there's overlap
            if overlap > 0:
                fade = min(overlap * 2, tile_h // 4, tile_w // 4)
                for i in range(fade):
                    alpha = i / fade
                    if y > 0:  # Top edge
                        tile_weight[i, :, :] *= alpha
                    if x > 0:  # Left edge
                        tile_weight[:, i, :] *= alpha

            output[out_y:out_y_end, out_x:out_x_end] += tile_output[:tile_h, :tile_w] * tile_weight
            weight[out_y:out_y_end, out_x:out_x_end] += tile_weight

    # Normalize by weight
    output = output / np.maximum(weight, 1e-8)
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    return output


def main():
    base_dir = Path('/home/keti/cwkim/KETI-AI/SR')
    weights_dir = base_dir / 'AI_models' / 'Real-ESRGAN' / 'weights'
    origin_dir = base_dir / 'images' / 'origin_codes'
    output_dir = base_dir / 'upscaled_4X_images'

    # Create output directory structure
    output_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Find all 4X images
    image_files = list(origin_dir.glob('*/*4X*.tif'))
    print(f"Found {len(image_files)} 4X images\n")

    if len(image_files) == 0:
        print("No 4X images found!")
        return

    # Load fine-tuned model
    print("Loading Fine-tuned model...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                   num_block=23, num_grow_ch=32, scale=4)
    state_dict = torch.load(str(weights_dir / 'RealESRGAN_x4plus_finetuned.pth'),
                           map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print(f"\nUpscaling {len(image_files)} images...\n")

    success_count = 0

    for img_path in tqdm(image_files, desc="Processing 4X images"):
        try:
            # Read original
            original = cv2.imread(str(img_path))
            if original is None:
                print(f"\n  Failed to read: {img_path.name}")
                continue

            # Get code folder name
            code_name = img_path.parent.name  # e.g., "code#56"

            # Create code-specific output directory
            code_output_dir = output_dir / code_name
            code_output_dir.mkdir(exist_ok=True)

            # Output filename
            output_name = img_path.stem + '_upscaled_4x.png'
            output_path = code_output_dir / output_name

            # Upscale
            upscaled = enhance_with_model(model, original, device, tile_size=400, overlap=20)

            # Save
            cv2.imwrite(str(output_path), upscaled)
            success_count += 1

        except Exception as e:
            print(f"\n  Error processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("4X IMAGE UPSCALING COMPLETE")
    print("="*80)
    print(f"\nSuccessfully processed: {success_count}/{len(image_files)} images")
    print(f"Results saved to: {output_dir}")

    # Show summary by code
    print("\nSummary by code:")
    for code_dir in sorted(output_dir.glob('code#*')):
        count = len(list(code_dir.glob('*.png')))
        print(f"  {code_dir.name}: {count} images")


if __name__ == '__main__':
    main()
