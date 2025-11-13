#!/usr/bin/env python3
"""
Evaluate 4X upscaling quality by comparing with original
Workflow: Original (GT) -> Downsample (LR) -> Upscale (SR) -> Compare with GT
"""

import cv2
import numpy as np
import torch
import sys
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('AI_models/Real-ESRGAN')
from basicsr.archs.rrdbnet_arch import RRDBNet


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


def enhance_with_model(model, img, device, tile_size=400, overlap=20):
    """Enhance image with model using tiling"""
    h, w = img.shape[:2]
    output_h, output_w = h * 4, w * 4
    output = np.zeros((output_h, output_w, 3), dtype=np.float32)
    weight = np.zeros((output_h, output_w, 3), dtype=np.float32)

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = img[y:y_end, x:x_end]

            tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
            tile_tensor = tile_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output_tensor = model(tile_tensor)

            tile_output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

            out_y = y * 4
            out_x = x * 4
            out_y_end = min(out_y + tile_output.shape[0], output_h)
            out_x_end = min(out_x + tile_output.shape[1], output_w)

            tile_h = out_y_end - out_y
            tile_w = out_x_end - out_x

            tile_weight = np.ones((tile_h, tile_w, 3), dtype=np.float32)

            if overlap > 0:
                fade = min(overlap * 2, tile_h // 4, tile_w // 4)
                for i in range(fade):
                    alpha = i / fade
                    if y > 0:
                        tile_weight[i, :, :] *= alpha
                    if x > 0:
                        tile_weight[:, i, :] *= alpha

            output[out_y:out_y_end, out_x:out_x_end] += tile_output[:tile_h, :tile_w] * tile_weight
            weight[out_y:out_y_end, out_x:out_x_end] += tile_weight

    output = output / np.maximum(weight, 1e-8)
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    return output


def main():
    base_dir = Path('/home/keti/cwkim/KETI-AI/SR')
    weights_dir = base_dir / 'AI_models' / 'Real-ESRGAN' / 'weights'
    origin_dir = base_dir / 'images' / 'origin_codes'
    output_dir = base_dir / 'evaluation_4X_results'

    output_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Find all 4X images
    image_files = list(origin_dir.glob('*/*4X*.tif'))
    print(f"Found {len(image_files)} 4X images\n")

    # Load fine-tuned model
    print("Loading Fine-tuned model...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                   num_block=23, num_grow_ch=32, scale=4)
    state_dict = torch.load(str(weights_dir / 'RealESRGAN_x4plus_finetuned.pth'),
                           map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print(f"\nEvaluating {len(image_files)} images...\n")

    all_metrics = []

    for img_path in tqdm(image_files, desc="Evaluating"):
        try:
            # Read original (Ground Truth)
            gt_original = cv2.imread(str(img_path))
            if gt_original is None:
                continue

            h, w = gt_original.shape[:2]

            # Create LR by downsampling (1/4 size)
            lr_input = cv2.resize(gt_original, (w//4, h//4), interpolation=cv2.INTER_CUBIC)
            lr_input = cv2.GaussianBlur(lr_input, (3, 3), 0.5)

            # Upscale with fine-tuned model
            sr_output = enhance_with_model(model, lr_input, device, tile_size=100, overlap=10)

            # Upscale with bicubic for comparison
            bicubic_output = cv2.resize(lr_input, (w, h), interpolation=cv2.INTER_CUBIC)

            # Calculate metrics
            ft_psnr, ft_ssim = calculate_metrics(gt_original, sr_output)
            bicubic_psnr, bicubic_ssim = calculate_metrics(gt_original, bicubic_output)

            all_metrics.append({
                'image': img_path.name,
                'bicubic_psnr': bicubic_psnr,
                'bicubic_ssim': bicubic_ssim,
                'finetuned_psnr': ft_psnr,
                'finetuned_ssim': ft_ssim,
                'improvement_psnr': ft_psnr - bicubic_psnr,
                'improvement_ssim': ft_ssim - bicubic_ssim
            })

        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            continue

    if not all_metrics:
        print("No images processed successfully!")
        return

    # Calculate averages
    avg_bicubic_psnr = np.mean([m['bicubic_psnr'] for m in all_metrics])
    avg_bicubic_ssim = np.mean([m['bicubic_ssim'] for m in all_metrics])
    avg_ft_psnr = np.mean([m['finetuned_psnr'] for m in all_metrics])
    avg_ft_ssim = np.mean([m['finetuned_ssim'] for m in all_metrics])

    # Print summary
    print("\n" + "="*80)
    print("4X UPSCALING EVALUATION RESULTS")
    print("="*80)

    print(f"\nBicubic Interpolation:")
    print(f"  Average PSNR: {avg_bicubic_psnr:.4f} dB")
    print(f"  Average SSIM: {avg_bicubic_ssim:.4f}")

    print(f"\nFine-tuned Real-ESRGAN:")
    print(f"  Average PSNR: {avg_ft_psnr:.4f} dB")
    print(f"  Average SSIM: {avg_ft_ssim:.4f}")

    print(f"\nImprovement:")
    psnr_diff = avg_ft_psnr - avg_bicubic_psnr
    ssim_diff = avg_ft_ssim - avg_bicubic_ssim
    print(f"  PSNR: {psnr_diff:+.4f} dB ({(psnr_diff/avg_bicubic_psnr * 100):+.2f}%)")
    print(f"  SSIM: {ssim_diff:+.4f} ({(ssim_diff/avg_bicubic_ssim * 100):+.2f}%)")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    methods = ['Bicubic', 'Fine-tuned']
    psnr_values = [avg_bicubic_psnr, avg_ft_psnr]
    ssim_values = [avg_bicubic_ssim, avg_ft_ssim]

    colors = ['#e74c3c', '#2ecc71']

    bars1 = ax1.bar(methods, psnr_values, color=colors, alpha=0.8)
    ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('4X Upscaling: Average PSNR', fontsize=14, fontweight='bold')
    ax1.set_ylim([min(psnr_values) * 0.95, max(psnr_values) * 1.05])

    for bar, val in zip(bars1, psnr_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    bars2 = ax2.bar(methods, ssim_values, color=colors, alpha=0.8)
    ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
    ax2.set_title('4X Upscaling: Average SSIM', fontsize=14, fontweight='bold')
    ax2.set_ylim([min(ssim_values) * 0.95, max(ssim_values) * 1.05])

    for bar, val in zip(bars2, ssim_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    summary_path = output_dir / 'evaluation_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save detailed metrics
    import csv
    csv_path = output_dir / 'detailed_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        if all_metrics:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            for metrics in all_metrics:
                writer.writerow(metrics)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"  - Summary chart: {summary_path}")
    print(f"  - Detailed metrics: {csv_path}")


if __name__ == '__main__':
    main()
