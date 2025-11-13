#!/usr/bin/env python3
"""
Compare 4X images: Direct upscaling without degradation
Pre-trained vs Fine-tuned Real-ESRGAN
"""

import os
import cv2
import numpy as np
import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append('AI_models/Real-ESRGAN')
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def create_upsampler(model_path, device):
    """Create Real-ESRGAN upsampler"""
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                   num_block=23, num_grow_ch=32, scale=4)

    upsampler = RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )
    return upsampler


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


def create_comparison_visualization(original, pretrained, finetuned,
                                   output_path, image_name):
    """Create side-by-side comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Convert BGR to RGB
    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    pre_rgb = cv2.cvtColor(pretrained, cv2.COLOR_BGR2RGB)
    fine_rgb = cv2.cvtColor(finetuned, cv2.COLOR_BGR2RGB)

    # Top row - Full images
    axes[0, 0].imshow(orig_rgb)
    axes[0, 0].set_title(f'Original 4X Image\n{original.shape[1]}x{original.shape[0]}',
                         fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pre_rgb)
    axes[0, 1].set_title(f'Pre-trained 4X Upscaled\n{pretrained.shape[1]}x{pretrained.shape[0]}',
                         fontsize=12, fontweight='bold', color='blue')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(fine_rgb)
    axes[0, 2].set_title(f'Fine-tuned 4X Upscaled\n{finetuned.shape[1]}x{finetuned.shape[0]}',
                         fontsize=12, fontweight='bold', color='green')
    axes[0, 2].axis('off')

    # Bottom row - Zoomed regions (center crop)
    zoom_size = min(orig_rgb.shape[0]//2, orig_rgb.shape[1]//2, 50)
    h, w = orig_rgb.shape[:2]
    y1, x1 = (h - zoom_size)//2, (w - zoom_size)//2
    y2, x2 = y1 + zoom_size, x1 + zoom_size

    orig_zoom = orig_rgb[y1:y2, x1:x2]

    # For upscaled images, zoom the corresponding region (4x larger)
    zoom_size_4x = zoom_size * 4
    h_4x, w_4x = pre_rgb.shape[:2]
    y1_4x, x1_4x = (h_4x - zoom_size_4x)//2, (w_4x - zoom_size_4x)//2
    y2_4x, x2_4x = y1_4x + zoom_size_4x, x1_4x + zoom_size_4x

    pre_zoom = pre_rgb[y1_4x:y2_4x, x1_4x:x2_4x]
    fine_zoom = fine_rgb[y1_4x:y2_4x, x1_4x:x2_4x]

    axes[1, 0].imshow(orig_zoom)
    axes[1, 0].set_title('Original (Zoomed)', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pre_zoom)
    axes[1, 1].set_title('Pre-trained (Zoomed)', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(fine_zoom)
    axes[1, 2].set_title('Fine-tuned (Zoomed)', fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle(f'4X Direct Upscaling: {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    base_dir = Path('/home/keti/cwkim/KETI-AI/SR')
    weights_dir = base_dir / 'AI_models' / 'Real-ESRGAN' / 'weights'
    cropped_dir = base_dir / 'preprocessing' / 'final_4X_improved'
    results_dir = base_dir / 'results_4X_direct_upscale'

    # Create output directories
    results_dir.mkdir(exist_ok=True)
    (results_dir / 'originals').mkdir(exist_ok=True)
    (results_dir / 'pretrained').mkdir(exist_ok=True)
    (results_dir / 'finetuned').mkdir(exist_ok=True)
    (results_dir / 'comparisons').mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Get 4X cropped images
    image_files = []
    for folder in cropped_dir.glob('cropped_*4X'):
        if folder.is_dir():
            image_files.extend(list(folder.glob('*.png')))

    image_files = sorted(image_files)
    print(f"Found {len(image_files)} 4X cropped images\n")

    if len(image_files) == 0:
        print("No 4X images found!")
        return

    # Create upsamplers
    print("Loading Pre-trained model...")
    pretrained_upsampler = create_upsampler(
        weights_dir / 'RealESRGAN_x4plus.pth', device
    )

    print("Loading Fine-tuned model...")
    finetuned_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                              num_block=23, num_grow_ch=32, scale=4)
    state_dict = torch.load(str(weights_dir / 'RealESRGAN_x4plus_finetuned.pth'), map_location='cpu')
    finetuned_model.load_state_dict(state_dict, strict=True)
    finetuned_model = finetuned_model.to(device)
    finetuned_model.eval()

    print(f"\nProcessing {len(image_files)} 4X images with direct 4X upscaling...\n")

    for idx, img_path in enumerate(tqdm(image_files, desc="Processing 4X")):
        try:
            # Read original image
            original = cv2.imread(str(img_path))
            if original is None:
                continue

            # Save original
            cv2.imwrite(str(results_dir / 'originals' / img_path.name), original)

            # Direct 4X upscale with pre-trained
            pretrained_output, _ = pretrained_upsampler.enhance(original, outscale=4)

            # Direct 4X upscale with fine-tuned
            finetuned_output = enhance_with_model(finetuned_model, original, device, scale=4)

            # Save outputs
            cv2.imwrite(str(results_dir / 'pretrained' / img_path.name), pretrained_output)
            cv2.imwrite(str(results_dir / 'finetuned' / img_path.name), finetuned_output)

            # Create visualization
            comparison_path = results_dir / 'comparisons' / f'4X_direct_{idx:03d}_{img_path.stem}.png'
            create_comparison_visualization(
                original, pretrained_output, finetuned_output,
                comparison_path, img_path.name
            )

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("4X DIRECT UPSCALING COMPLETE")
    print("="*80)
    print(f"\nProcessed {len(image_files)} images")
    print(f"Results saved to: {results_dir}")
    print(f"  - Originals: {results_dir / 'originals'}")
    print(f"  - Pre-trained outputs: {results_dir / 'pretrained'}")
    print(f"  - Fine-tuned outputs: {results_dir / 'finetuned'}")
    print(f"  - Comparison images: {results_dir / 'comparisons'}")
    print("\nNote: 4X images were directly upscaled 4x without degradation")
    print("      Visual comparison only (no GT available for metrics)")


if __name__ == '__main__':
    main()
