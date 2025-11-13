#!/usr/bin/env python3
"""
Upscale full 4X images (not cropped) with fine-tuned model
"""

import cv2
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.append('AI_models/Real-ESRGAN')
from basicsr.archs.rrdbnet_arch import RRDBNet


def enhance_with_model(model, img, device, scale=4):
    """Directly enhance image with model"""
    # Convert to tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output_tensor = model(img_tensor)

    # Convert back to numpy
    output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    return output


def main():
    base_dir = Path('/home/keti/cwkim/KETI-AI/SR')
    weights_dir = base_dir / 'AI_models' / 'Real-ESRGAN' / 'weights'
    origin_dir = base_dir / 'images' / 'origin_codes'
    output_dir = base_dir / 'results_4X_full_upscale'

    output_dir.mkdir(exist_ok=True)
    (output_dir / 'originals').mkdir(exist_ok=True)
    (output_dir / 'finetuned').mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load fine-tuned model
    print("Loading Fine-tuned model...")
    finetuned_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                              num_block=23, num_grow_ch=32, scale=4)
    state_dict = torch.load(str(weights_dir / 'RealESRGAN_x4plus_finetuned.pth'), map_location='cpu')
    finetuned_model.load_state_dict(state_dict, strict=True)
    finetuned_model = finetuned_model.to(device)
    finetuned_model.eval()

    # Select a few representative 4X images
    samples = [
        origin_dir / 'code#56' / '56_4X_3.tif',
        origin_dir / 'code#68' / '68_4X_3.tif',
        origin_dir / 'code#75' / '75_4X_3.tif',
    ]

    for img_path in samples:
        if not img_path.exists():
            continue

        print(f"\nProcessing {img_path.name}...")

        # Read original
        original = cv2.imread(str(img_path))
        if original is None:
            continue

        print(f"  Original size: {original.shape[1]}x{original.shape[0]}")

        # Save original as PNG
        output_name = img_path.stem + '.png'
        cv2.imwrite(str(output_dir / 'originals' / output_name), original)

        # Upscale with fine-tuned (process in tiles due to memory)
        print(f"  Upscaling with fine-tuned model...")

        # For large images, process in tiles
        tile_size = 400
        overlap = 20
        h, w = original.shape[:2]
        output_h, output_w = h * 4, w * 4
        output = np.zeros((output_h, output_w, 3), dtype=np.uint8)

        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                # Extract tile
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                tile = original[y:y_end, x:x_end]

                # Process tile
                tile_output = enhance_with_model(finetuned_model, tile, device, scale=4)

                # Place in output
                out_y = y * 4
                out_x = x * 4
                out_y_end = min(out_y + tile_output.shape[0], output_h)
                out_x_end = min(out_x + tile_output.shape[1], output_w)

                output[out_y:out_y_end, out_x:out_x_end] = tile_output[:out_y_end-out_y, :out_x_end-out_x]

        # Save result
        cv2.imwrite(str(output_dir / 'finetuned' / output_name), output)
        print(f"  Output size: {output.shape[1]}x{output.shape[0]}")
        print(f"  Saved to: {output_dir / 'finetuned' / output_name}")

    print(f"\n{'='*80}")
    print("Full 4X image upscaling complete!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
