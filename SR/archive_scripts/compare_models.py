#!/usr/bin/env python3
"""
Compare Pre-trained vs Fine-tuned Real-ESRGAN models
"""

import os
import cv2
import numpy as np
import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
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


def calculate_metrics(img1, img2):
    """Calculate PSNR and SSIM"""
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1, gray2 = img1, img2

    psnr_value = psnr(gray1, gray2, data_range=255)
    ssim_value = ssim(gray1, gray2, data_range=255)

    return psnr_value, ssim_value


def create_comparison_visualization(original, pretrained, finetuned, metrics, output_path, image_name):
    """Create side-by-side comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Convert BGR to RGB
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    pretrained_rgb = cv2.cvtColor(pretrained, cv2.COLOR_BGR2RGB)
    finetuned_rgb = cv2.cvtColor(finetuned, cv2.COLOR_BGR2RGB)

    # Top row - Full images
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title(f'Original (Low Resolution)\n{original.shape[1]}x{original.shape[0]}',
                         fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pretrained_rgb)
    axes[0, 1].set_title(f'Pre-trained Real-ESRGAN\n{pretrained.shape[1]}x{pretrained.shape[0]}\nPSNR: {metrics["pretrained_psnr"]:.2f} dB | SSIM: {metrics["pretrained_ssim"]:.4f}',
                         fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(finetuned_rgb)
    axes[0, 2].set_title(f'Fine-tuned Real-ESRGAN\n{finetuned.shape[1]}x{finetuned.shape[0]}\nPSNR: {metrics["finetuned_psnr"]:.2f} dB | SSIM: {metrics["finetuned_ssim"]:.4f}',
                         fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Bottom row - Zoomed regions (center crop)
    zoom_size = min(original_rgb.shape[0]//2, original_rgb.shape[1]//2, 100)
    h, w = original_rgb.shape[:2]
    y1, x1 = (h - zoom_size)//2, (w - zoom_size)//2
    y2, x2 = y1 + zoom_size, x1 + zoom_size

    orig_zoom = original_rgb[y1:y2, x1:x2]

    # For upscaled images, crop from center with 4x size
    h_up, w_up = pretrained_rgb.shape[:2]
    y1_up, x1_up = (h_up - zoom_size*4)//2, (w_up - zoom_size*4)//2
    y2_up, x2_up = y1_up + zoom_size*4, x1_up + zoom_size*4

    pre_zoom = pretrained_rgb[y1_up:y2_up, x1_up:x2_up]
    fine_zoom = finetuned_rgb[y1_up:y2_up, x1_up:x2_up]

    axes[1, 0].imshow(orig_zoom)
    axes[1, 0].set_title('Original (Zoomed)', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pre_zoom)
    axes[1, 1].set_title('Pre-trained (Zoomed)', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(fine_zoom)
    axes[1, 2].set_title('Fine-tuned (Zoomed)', fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle(f'Super-Resolution Comparison: {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    base_dir = Path('/home/keti/cwkim/KETI-AI/SR')
    weights_dir = base_dir / 'AI_models' / 'Real-ESRGAN' / 'weights'
    cropped_dir = base_dir / 'preprocessing' / 'final'
    results_dir = base_dir / 'results_comparison'

    # Create output directories
    results_dir.mkdir(exist_ok=True)
    (results_dir / 'pretrained').mkdir(exist_ok=True)
    (results_dir / 'finetuned').mkdir(exist_ok=True)
    (results_dir / 'comparisons').mkdir(exist_ok=True)
    (results_dir / 'metrics').mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Get cropped images
    image_files = []
    for folder in cropped_dir.glob('cropped_*'):
        if folder.is_dir():
            image_files.extend(list(folder.glob('*.png')))
            image_files.extend(list(folder.glob('*.jpg')))

    image_files = sorted(image_files)[:30]  # Process 30 images
    print(f"Found {len(image_files)} images to process\n")

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

    print("\nProcessing images...\n")
    all_metrics = []

    for idx, img_path in enumerate(tqdm(image_files, desc="Upscaling")):
        try:
            # Read original
            original = cv2.imread(str(img_path))
            if original is None:
                continue

            # Upscale with pre-trained
            pretrained_output, _ = pretrained_upsampler.enhance(original, outscale=4)

            # Upscale with fine-tuned (direct inference)
            finetuned_output = enhance_with_model(finetuned_model, original, device, scale=4)

            # Save outputs
            cv2.imwrite(str(results_dir / 'pretrained' / img_path.name), pretrained_output)
            cv2.imwrite(str(results_dir / 'finetuned' / img_path.name), finetuned_output)

            # Resize original to match upscaled size for metric calculation
            h, w = pretrained_output.shape[:2]
            original_resized = cv2.resize(original, (w, h), interpolation=cv2.INTER_CUBIC)

            # Calculate metrics
            pre_psnr, pre_ssim = calculate_metrics(original_resized, pretrained_output)
            fine_psnr, fine_ssim = calculate_metrics(original_resized, finetuned_output)

            metrics = {
                'image': img_path.name,
                'pretrained_psnr': pre_psnr,
                'pretrained_ssim': pre_ssim,
                'finetuned_psnr': fine_psnr,
                'finetuned_ssim': fine_ssim,
                'improvement_psnr': fine_psnr - pre_psnr,
                'improvement_ssim': fine_ssim - pre_ssim
            }
            all_metrics.append(metrics)

            # Create visualization for first 10 images
            if idx < 10:
                comparison_path = results_dir / 'comparisons' / f'comparison_{idx:03d}_{img_path.stem}.png'
                create_comparison_visualization(
                    original, pretrained_output, finetuned_output,
                    metrics, comparison_path, img_path.name
                )

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Create summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    avg_pre_psnr = np.mean([m['pretrained_psnr'] for m in all_metrics])
    avg_pre_ssim = np.mean([m['pretrained_ssim'] for m in all_metrics])
    avg_fine_psnr = np.mean([m['finetuned_psnr'] for m in all_metrics])
    avg_fine_ssim = np.mean([m['finetuned_ssim'] for m in all_metrics])

    print(f"\nPre-trained Model:")
    print(f"  Average PSNR: {avg_pre_psnr:.4f} dB")
    print(f"  Average SSIM: {avg_pre_ssim:.4f}")

    print(f"\nFine-tuned Model:")
    print(f"  Average PSNR: {avg_fine_psnr:.4f} dB")
    print(f"  Average SSIM: {avg_fine_ssim:.4f}")

    print(f"\nImprovement:")
    print(f"  PSNR: {avg_fine_psnr - avg_pre_psnr:+.4f} dB ({((avg_fine_psnr/avg_pre_psnr - 1) * 100):+.2f}%)")
    print(f"  SSIM: {avg_fine_ssim - avg_pre_ssim:+.4f} ({((avg_fine_ssim/avg_pre_ssim - 1) * 100):+.2f}%)")

    # Create bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    models = ['Pre-trained', 'Fine-tuned']
    psnr_values = [avg_pre_psnr, avg_fine_psnr]
    ssim_values = [avg_pre_ssim, avg_fine_ssim]

    colors = ['#3498db', '#2ecc71']

    bars1 = ax1.bar(models, psnr_values, color=colors, alpha=0.8)
    ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('Average PSNR Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([min(psnr_values) * 0.95, max(psnr_values) * 1.05])

    for bar, val in zip(bars1, psnr_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    bars2 = ax2.bar(models, ssim_values, color=colors, alpha=0.8)
    ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
    ax2.set_title('Average SSIM Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([min(ssim_values) * 0.95, max(ssim_values) * 1.05])

    for bar, val in zip(bars2, ssim_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    summary_path = results_dir / 'metrics' / 'summary_comparison.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save detailed metrics
    import csv
    csv_path = results_dir / 'metrics' / 'detailed_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        if all_metrics:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            for metrics in all_metrics:
                writer.writerow(metrics)

    print(f"\n{'='*80}")
    print("Comparison complete!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {results_dir}")
    print(f"  - Summary chart: {summary_path}")
    print(f"  - Detailed metrics: {csv_path}")
    print(f"  - Comparison images: {results_dir / 'comparisons'}")


if __name__ == '__main__':
    main()
