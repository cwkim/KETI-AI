# Microdisk Image Super-Resolution

Real-ESRGAN fine-tuned model for 4x super-resolution of microscope microdisk images.

## Quick Links

- **[Project Summary](PROJECT_SUMMARY.md)** - Full documentation
- **[Quick Start Guide](QUICK_START.md)** - Get started quickly
- **[Training Methodology](TRAINING_METHODOLOGY.md)** - Fine-tuning vs 재학습 설명

## Results

| Method | PSNR | SSIM |
|--------|------|------|
| Bicubic | 30.26 dB | 0.9196 |
| **Fine-tuned** | **30.95 dB** | **0.9323** |

**Improvement**: +0.69 dB PSNR, +0.0127 SSIM

## Final Outputs

- **Upscaled 4X images**: `upscaled_4X_images/` (40 images, 831MB)
- **Evaluation results**: `evaluation_4X_results/`
- **Fine-tuned model**: `AI_models/Real-ESRGAN/weights/RealESRGAN_x4plus_finetuned.pth`

## Usage

```bash
python archive_scripts/upscale_all_4X_images.py
```

See `QUICK_START.md` for more details.
