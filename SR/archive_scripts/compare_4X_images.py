#!/usr/bin/env python3
"""
Compare 4X images: Pre-trained vs Fine-tuned Real-ESRGAN
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
    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1, gray2 = img1, img2

    psnr_value = psnr(gray1, gray2, data_range=255)
    ssim_value = ssim(gray1, gray2, data_range=255)

    return psnr_value, ssim_value


def create_comparison_visualization(hr_original, lr_input, pretrained, finetuned,
                                   metrics, output_path, image_name):
    """Create comprehensive comparison"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Convert BGR to RGB
    hr_rgb = cv2.cvtColor(hr_original, cv2.COLOR_BGR2RGB)
    lr_rgb = cv2.cvtColor(lr_input, cv2.COLOR_BGR2RGB)
    pre_rgb = cv2.cvtColor(pretrained, cv2.COLOR_BGR2RGB)
    fine_rgb = cv2.cvtColor(finetuned, cv2.COLOR_BGR2RGB)

    # Top row - Full images
    axes[0, 0].imshow(hr_rgb)
    axes[0, 0].set_title(f'Ground Truth (4X)\n{hr_original.shape[1]}x{hr_original.shape[0]}',
                         fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(lr_rgb)
    axes[0, 1].set_title(f'Low Resolution Input\n{lr_input.shape[1]}x{lr_input.shape[0]}',
                         fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(pre_rgb)
    axes[0, 2].set_title(f'Pre-trained Output\n{pretrained.shape[1]}x{pretrained.shape[0]}\nPSNR: {metrics["pretrained_psnr"]:.2f} | SSIM: {metrics["pretrained_ssim"]:.4f}',
                         fontsize=11, fontweight='bold', color='blue')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(fine_rgb)
    axes[0, 3].set_title(f'Fine-tuned Output\n{finetuned.shape[1]}x{finetuned.shape[0]}\nPSNR: {metrics["finetuned_psnr"]:.2f} | SSIM: {metrics["finetuned_ssim"]:.4f}',
                         fontsize=11, fontweight='bold', color='green')
    axes[0, 3].axis('off')

    # Bottom row - Zoomed regions
    zoom_size = min(hr_rgb.shape[0]//2, hr_rgb.shape[1]//2, 50)
    h, w = hr_rgb.shape[:2]
    y1, x1 = (h - zoom_size)//2, (w - zoom_size)//2
    y2, x2 = y1 + zoom_size, x1 + zoom_size

    hr_zoom = hr_rgb[y1:y2, x1:x2]
    lr_zoom = cv2.resize(lr_rgb, (zoom_size, zoom_size), interpolation=cv2.INTER_NEAREST)
    pre_zoom = pre_rgb[y1:y2, x1:x2]
    fine_zoom = fine_rgb[y1:y2, x1:x2]

    axes[1, 0].imshow(hr_zoom)
    axes[1, 0].set_title('GT (Zoomed)', fontsize=10, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(lr_zoom)
    axes[1, 1].set_title('LR (Zoomed)', fontsize=10, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(pre_zoom)
    axes[1, 2].set_title('Pre-trained (Zoomed)', fontsize=10, fontweight='bold')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(fine_zoom)
    axes[1, 3].set_title('Fine-tuned (Zoomed)', fontsize=10, fontweight='bold')
    axes[1, 3].axis('off')

    plt.suptitle(f'4X Image SR Comparison: {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    base_dir = Path('/home/keti/cwkim/KETI-AI/SR')
    weights_dir = base_dir / 'AI_models' / 'Real-ESRGAN' / 'weights'
    cropped_dir = base_dir / 'preprocessing' / 'final'
    results_dir = base_dir / 'results_4X_comparison'

    # Create output directories
    results_dir.mkdir(exist_ok=True)
    (results_dir / 'lr_inputs').mkdir(exist_ok=True)
    (results_dir / 'pretrained').mkdir(exist_ok=True)
    (results_dir / 'finetuned').mkdir(exist_ok=True)
    (results_dir / 'comparisons').mkdir(exist_ok=True)
    (results_dir / 'metrics').mkdir(exist_ok=True)

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

    print(f"\nProcessing {len(image_files)} 4X images...\n")
    all_metrics = []

    for idx, img_path in enumerate(tqdm(image_files, desc="Processing 4X")):
        try:
            # Read HR original (ground truth)
            hr_original = cv2.imread(str(img_path))
            if hr_original is None:
                continue

            # Create LR by downsampling
            h, w = hr_original.shape[:2]
            lr_input = cv2.resize(hr_original, (w//4, h//4), interpolation=cv2.INTER_CUBIC)
            lr_input = cv2.GaussianBlur(lr_input, (3, 3), 0.5)

            # Save LR input
            cv2.imwrite(str(results_dir / 'lr_inputs' / img_path.name), lr_input)

            # Upscale with pre-trained
            pretrained_output, _ = pretrained_upsampler.enhance(lr_input, outscale=4)

            # Upscale with fine-tuned
            finetuned_output = enhance_with_model(finetuned_model, lr_input, device, scale=4)

            # Save outputs
            cv2.imwrite(str(results_dir / 'pretrained' / img_path.name), pretrained_output)
            cv2.imwrite(str(results_dir / 'finetuned' / img_path.name), finetuned_output)

            # Calculate metrics
            pre_psnr, pre_ssim = calculate_metrics(hr_original, pretrained_output)
            fine_psnr, fine_ssim = calculate_metrics(hr_original, finetuned_output)

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

            # Create visualization for all images
            comparison_path = results_dir / 'comparisons' / f'4X_comparison_{idx:03d}_{img_path.stem}.png'
            create_comparison_visualization(
                hr_original, lr_input, pretrained_output, finetuned_output,
                metrics, comparison_path, img_path.name
            )

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_metrics:
        print("No images processed successfully!")
        return

    # Create summary
    print("\n" + "="*80)
    print("4X IMAGE COMPARISON RESULTS")
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
    ax1.set_title('4X Images: Average PSNR', fontsize=14, fontweight='bold')
    ax1.set_ylim([min(psnr_values) * 0.95, max(psnr_values) * 1.05])

    for bar, val in zip(bars1, psnr_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    bars2 = ax2.bar(models, ssim_values, color=colors, alpha=0.8)
    ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
    ax2.set_title('4X Images: Average SSIM', fontsize=14, fontweight='bold')
    ax2.set_ylim([min(ssim_values) * 0.95, max(ssim_values) * 1.05])

    for bar, val in zip(bars2, ssim_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    summary_path = results_dir / 'metrics' / 'summary_4X_comparison.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save detailed metrics
    import csv
    csv_path = results_dir / 'metrics' / 'detailed_4X_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        if all_metrics:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            for metrics in all_metrics:
                writer.writerow(metrics)

    print(f"\n{'='*80}")
    print("4X Comparison complete!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {results_dir}")
    print(f"  - Summary chart: {summary_path}")
    print(f"  - Detailed metrics: {csv_path}")
    print(f"  - Comparison images: {results_dir / 'comparisons'}")


if __name__ == '__main__':
    main()
