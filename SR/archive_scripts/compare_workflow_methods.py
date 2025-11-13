#!/usr/bin/env python3
"""
Compare two workflows:
1. Full image upscale -> crop
2. Crop -> individual upscale
"""

import cv2
import numpy as np
import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append('AI_models/Real-ESRGAN')
from basicsr.archs.rrdbnet_arch import RRDBNet


def calculate_brisque(image):
    """Calculate BRISQUE score (lower is better)"""
    try:
        import cv2
        # OpenCV's quality module
        if hasattr(cv2.quality, 'QualityBRISQUE_compute'):
            brisque = cv2.quality.QualityBRISQUE_create()
            score = brisque.compute(image)[0]
            return score
    except:
        pass
    return None


def calculate_sharpness(image):
    """Calculate image sharpness using Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def calculate_edge_strength(image):
    """Calculate average edge strength"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Sobel edges
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.sqrt(sobelx**2 + sobely**2)

    return edge_strength.mean()


def enhance_with_model(model, img, device):
    """Enhance image with model"""
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(img_tensor)

    output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    return output


def extract_microdisk_from_upscaled(upscaled_img, x, y, w, h, scale=4):
    """Extract microdisk region from upscaled full image"""
    x_scaled = x * scale
    y_scaled = y * scale
    w_scaled = w * scale
    h_scaled = h * scale

    cropped = upscaled_img[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled]
    return cropped


def detect_microdisk_locations(image_path):
    """Detect microdisk locations (similar to preprocessing)"""
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    locations = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 1000 < area < 20000:
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.6:
                x, y, w, h = cv2.boundingRect(contour)
                size = max(w, h)
                center_x = x + w // 2
                center_y = y + h // 2
                padding = 10
                half_size = size // 2 + padding

                x1 = max(0, center_x - half_size)
                y1 = max(0, center_y - half_size)
                x2 = min(img.shape[1], center_x + half_size)
                y2 = min(img.shape[0], center_y + half_size)

                locations.append((x1, y1, x2-x1, y2-y1))

    return locations[:10]  # Limit to 10 for comparison


def main():
    base_dir = Path('/home/keti/cwkim/KETI-AI/SR')
    weights_dir = base_dir / 'AI_models' / 'Real-ESRGAN' / 'weights'
    origin_dir = base_dir / 'images' / 'origin_codes'
    full_upscale_dir = base_dir / 'results_4X_full_upscale'
    output_dir = base_dir / 'workflow_comparison'

    output_dir.mkdir(exist_ok=True)
    (output_dir / 'method1_full_then_crop').mkdir(exist_ok=True)
    (output_dir / 'method2_crop_then_upscale').mkdir(exist_ok=True)
    (output_dir / 'comparisons').mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load fine-tuned model
    print("Loading Fine-tuned model...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                   num_block=23, num_grow_ch=32, scale=4)
    state_dict = torch.load(str(weights_dir / 'RealESRGAN_x4plus_finetuned.pth'), map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    # Test image
    test_image = origin_dir / 'code#56' / '56_4X_3.tif'

    if not test_image.exists():
        print(f"Test image not found: {test_image}")
        return

    print(f"\nAnalyzing: {test_image.name}")

    # Read original
    original = cv2.imread(str(test_image))

    # Method 1: Full upscale -> crop
    print("\nMethod 1: Full image upscale -> crop")
    upscaled_full_path = full_upscale_dir / 'finetuned' / f'{test_image.stem}.png'
    upscaled_full = cv2.imread(str(upscaled_full_path))

    # Detect microdisk locations
    locations = detect_microdisk_locations(test_image)
    print(f"  Found {len(locations)} microdisks to compare")

    results = []

    for idx, (x, y, w, h) in enumerate(tqdm(locations, desc="Comparing methods")):
        # Method 1: Extract from full upscaled
        method1_crop = extract_microdisk_from_upscaled(upscaled_full, x, y, w, h, scale=4)

        # Method 2: Crop then upscale
        original_crop = original[y:y+h, x:x+w]
        method2_upscaled = enhance_with_model(model, original_crop, device)

        # Ensure same size
        if method1_crop.shape != method2_upscaled.shape:
            min_h = min(method1_crop.shape[0], method2_upscaled.shape[0])
            min_w = min(method1_crop.shape[1], method2_upscaled.shape[1])
            method1_crop = method1_crop[:min_h, :min_w]
            method2_upscaled = method2_upscaled[:min_h, :min_w]

        # Save crops
        cv2.imwrite(str(output_dir / 'method1_full_then_crop' / f'md_{idx:03d}.png'), method1_crop)
        cv2.imwrite(str(output_dir / 'method2_crop_then_upscale' / f'md_{idx:03d}.png'), method2_upscaled)

        # Calculate metrics
        sharpness1 = calculate_sharpness(method1_crop)
        sharpness2 = calculate_sharpness(method2_upscaled)

        edge1 = calculate_edge_strength(method1_crop)
        edge2 = calculate_edge_strength(method2_upscaled)

        results.append({
            'index': idx,
            'method1_sharpness': sharpness1,
            'method2_sharpness': sharpness2,
            'method1_edge': edge1,
            'method2_edge': edge2,
        })

        # Create comparison visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        orig_rgb = cv2.cvtColor(original_crop, cv2.COLOR_BGR2RGB)
        m1_rgb = cv2.cvtColor(method1_crop, cv2.COLOR_BGR2RGB)
        m2_rgb = cv2.cvtColor(method2_upscaled, cv2.COLOR_BGR2RGB)

        axes[0].imshow(orig_rgb)
        axes[0].set_title(f'Original Crop\n{orig_rgb.shape[1]}x{orig_rgb.shape[0]}',
                         fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(m1_rgb)
        axes[1].set_title(f'Method 1: Full→Crop\nSharpness: {sharpness1:.2f}\nEdge: {edge1:.2f}',
                         fontsize=11, fontweight='bold', color='blue')
        axes[1].axis('off')

        axes[2].imshow(m2_rgb)
        axes[2].set_title(f'Method 2: Crop→Upscale\nSharpness: {sharpness2:.2f}\nEdge: {edge2:.2f}',
                         fontsize=11, fontweight='bold', color='green')
        axes[2].axis('off')

        plt.suptitle(f'Workflow Comparison - Microdisk #{idx+1}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'comparisons' / f'comparison_md_{idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Summary
    print("\n" + "="*80)
    print("WORKFLOW COMPARISON RESULTS")
    print("="*80)

    avg_sharp1 = np.mean([r['method1_sharpness'] for r in results])
    avg_sharp2 = np.mean([r['method2_sharpness'] for r in results])
    avg_edge1 = np.mean([r['method1_edge'] for r in results])
    avg_edge2 = np.mean([r['method2_edge'] for r in results])

    print(f"\nMethod 1 (Full Upscale → Crop):")
    print(f"  Average Sharpness: {avg_sharp1:.4f}")
    print(f"  Average Edge Strength: {avg_edge1:.4f}")

    print(f"\nMethod 2 (Crop → Individual Upscale):")
    print(f"  Average Sharpness: {avg_sharp2:.4f}")
    print(f"  Average Edge Strength: {avg_edge2:.4f}")

    print(f"\nComparison:")
    sharp_diff = avg_sharp2 - avg_sharp1
    edge_diff = avg_edge2 - avg_edge1

    print(f"  Sharpness difference: {sharp_diff:+.4f} ({(sharp_diff/avg_sharp1*100):+.2f}%)")
    print(f"  Edge strength difference: {edge_diff:+.4f} ({(edge_diff/avg_edge1*100):+.2f}%)")

    if sharp_diff > 0 and edge_diff > 0:
        print("\n✓ Method 2 (Crop → Upscale) produces sharper results")
    elif sharp_diff < 0 and edge_diff < 0:
        print("\n✓ Method 1 (Full → Crop) produces sharper results")
    else:
        print("\n≈ Results are mixed, visual inspection recommended")

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"  - Method 1 outputs: {output_dir / 'method1_full_then_crop'}")
    print(f"  - Method 2 outputs: {output_dir / 'method2_crop_then_upscale'}")
    print(f"  - Comparisons: {output_dir / 'comparisons'}")


if __name__ == '__main__':
    main()
