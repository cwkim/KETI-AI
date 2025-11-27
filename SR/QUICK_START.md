# Real-ESRGAN Super-Resolution - Quick Start Guide

## 빠른 시작 (새로운 이미지 업스케일)

### 단일 4X 이미지 업스케일

```python
#!/usr/bin/env python3
import cv2
import torch
import sys
from pathlib import Path

sys.path.append('AI_models/Real-ESRGAN')
from basicsr.archs.rrdbnet_arch import RRDBNet

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
               num_block=23, num_grow_ch=32, scale=4)
state_dict = torch.load('AI_models/Real-ESRGAN/weights/RealESRGAN_x4plus_finetuned.pth',
                       map_location='cpu', weights_only=False)
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()

# 이미지 로드
img = cv2.imread('your_image.tif')

# 업스케일
img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
img_tensor = img_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)

output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
output = (output * 255.0).clip(0, 255).astype('uint8')

# 저장
cv2.imwrite('output_4x.png', output)
```

### 전체 폴더 업스케일

```bash
python archive_scripts/upscale_all_4X_images.py
```

---

## 모델 정보

### Fine-tuned 모델
- **경로**: `AI_models/Real-ESRGAN/weights/RealESRGAN_x4plus_finetuned.pth`
- **크기**: 64MB
- **입력**: RGB 이미지 (any size)
- **출력**: 4배 확대된 RGB 이미지
- **권장**: GPU 사용 (CPU는 느림)

### 성능
- **학습 데이터**: 10X 이미지 (SAM2 + 전통적 CV 기법으로 전처리)
- **학습 시간**: ~15-18분 (50 epochs, Tesla V100)
- **10X 이미지 평가**: PSNR 24.93 dB, SSIM 0.7869 (Pre-trained 대비 +37.55% PSNR)
- **4X 이미지 평가**: PSNR 30.95 dB, SSIM 0.9323 (Bicubic 대비 +2.27% PSNR)

---

## 결과 확인

### 업스케일된 이미지
```bash
ls upscaled_4X_images/code#56/
# 출력: 56_4X_1_upscaled_4x.png, 56_4X_2_upscaled_4x.png, ...
```

### 평가 결과
```bash
cat evaluation_4X_results/detailed_metrics.csv
# PSNR, SSIM 확인
```

### 비교 이미지
```bash
ls results_proper_comparison/comparisons/
# 시각적 비교 확인
```

---

## 문제 해결

### Out of Memory 에러
- `tile_size`를 작게 조정 (400 → 200)
- 또는 CPU 사용

### CUDA 에러
```bash
# CPU로 변경
device = torch.device('cpu')
```

### 느린 속도
- GPU 사용 권장
- Tile-based processing 활용

---

## 상세 문서

전체 프로젝트 정보는 `PROJECT_SUMMARY.md` 참조
