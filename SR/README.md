# Microdisk Image Super-Resolution

Real-ESRGAN fine-tuned model for 4x super-resolution of microscope microdisk images.

## Quick Links

- **[Project Summary](PROJECT_SUMMARY.md)** - Full documentation
- **[Quick Start Guide](QUICK_START.md)** - Get started quickly
- **[Training Methodology](TRAINING_METHODOLOGY.md)** - Fine-tuning vs 재학습 설명

## Results (10X Microdisk Images)

| Method | PSNR | SSIM |
|--------|------|------|
| Pretrained | 18.12 dB | 0.4787 |
| **Fine-tuned** | **24.93 dB** | **0.7869** |

**Improvement**: +6.81 dB PSNR (+37.55%), +0.3082 SSIM (+64.39%)

## Key Features

- **Training Data**: 10X magnification microdisk images
- **Preprocessing**: Traditional CV techniques + SAM2 for accurate microdisk detection
- **Fine-tuning Time**: ~15-18 minutes (50 epochs on Tesla V100)
- **Test Images**: 15 high-quality 10X microdisk samples

## Final Outputs

- **Fine-tuned model**: `AI_models/Real-ESRGAN/weights/RealESRGAN_x4plus_finetuned.pth`
- **Evaluation results**: `results_proper_comparison/metrics/detailed_metrics.csv`
- **Comparison images**: `results_proper_comparison/comparisons/`

## Usage

```bash
# See QUICK_START.md for detailed usage
python archive_scripts/upscale_all_4X_images.py
```

See `QUICK_START.md` for more details.
