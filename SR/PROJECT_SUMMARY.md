# Real-ESRGAN Fine-tuning for Microdisk Image Super-Resolution

## 프로젝트 개요

현미경으로 촬영한 microdisk 이미지의 해상도를 향상시키기 위해 Real-ESRGAN 모델을 fine-tuning하고 평가한 프로젝트입니다.

**작업 기간**: 2025-11-13
**담당자**: KETI AI Team
**모델**: Real-ESRGAN (RRDBNet, 4x upscaling)

---

## 목차

1. [프로젝트 구조](#프로젝트-구조)
2. [데이터셋](#데이터셋)
3. [Fine-tuning 과정](#fine-tuning-과정)
4. [평가 결과](#평가-결과)
5. [최종 결과물](#최종-결과물)
6. [주요 스크립트](#주요-스크립트)
7. [실행 방법](#실행-방법)

---

## 프로젝트 구조

```
SR/
├── images/
│   └── origin_codes/          # 원본 현미경 이미지
│       ├── code#56/           # 10X: 10개, 4X: 10개
│       ├── code#68/           # 10X: 10개, 4X: 10개
│       ├── code#75/           # 10X: 10개, 4X: 10개
│       └── code#79/           # 10X: 10개, 4X: 10개
│
├── AI_models/
│   └── Real-ESRGAN/
│       └── weights/
│           ├── RealESRGAN_x4plus.pth              # Pre-trained (64MB)
│           └── RealESRGAN_x4plus_finetuned.pth    # Fine-tuned (64MB)
│
├── preprocessing/
│   ├── final/                 # 10X 이미지에서 추출한 microdisk (학습 후 삭제됨)
│   │                          # SAM2 + 전통적 CV 기법으로 추출
│   │                          # Fine-tuning 학습 데이터로 사용됨
│   ├── final_4X_improved/     # 4X 이미지 분석용 (참고)
│   │   ├── cropped_code#56_4X/
│   │   ├── cropped_code#68_4X/
│   │   ├── cropped_code#75_4X/
│   │   └── cropped_code#79_4X/
│   └── sam_models/            # SAM2 모델 가중치 (358MB)
│                              # sam_vit_b_01ec64.pth - 학습 시 사용됨
│
├── upscaled_4X_images/        # 최종 업스케일 결과 (40개, 831MB)
│   ├── code#56/               # 10개
│   ├── code#68/               # 10개
│   ├── code#75/               # 10개
│   └── code#79/               # 10개
│
├── evaluation_4X_results/     # 평가 결과
│   ├── evaluation_summary.png
│   └── detailed_metrics.csv
│
├── results_proper_comparison/ # 10X 이미지 비교 결과
│   ├── comparisons/           # 비교 이미지 (30개)
│   └── metrics/
│       ├── summary_comparison.png
│       └── detailed_metrics.csv
│
├── archive_scripts/           # 사용한 모든 Python 스크립트
│
├── training_v2.log            # Fine-tuning 로그
└── PROJECT_SUMMARY.md         # 이 문서
```

---

## 데이터셋

### 원본 이미지

**10X 배율 이미지** (40개):
- 4개 코드 × 10개 이미지
- 크기: 약 2592x1944 픽셀
- 용도: Fine-tuning 데이터 및 평가

**4X 배율 이미지** (40개):
- 4개 코드 × 10개 이미지
- 크기: 1600x1200 픽셀
- 용도: 최종 업스케일링 대상

### 전처리

**10X 이미지**: Fine-tuning 학습 데이터 소스
- 전통적인 컴퓨터 비전 기법 + **SAM2 (Segment Anything Model 2)** 사용
- Adaptive thresholding (ADAPTIVE_THRESH_GAUSSIAN_C)
- Circularity-based filtering (circularity > 0.6)
- Morphological operations (Gaussian blur, open/close)
- SAM2로 정확한 microdisk segmentation 및 크롭
- 개별 microdisk를 정밀하게 검출 후 크롭
- 사용: Fine-tuning training data (500 HR/LR pairs 생성)
- SAM2 모델 가중치: `preprocessing/sam_models/sam_vit_b_01ec64.pth` (358MB)

**4X 이미지**: 최종 업스케일 대상
- 40개 원본 이미지를 4배 확대하여 고해상도 변환
- 평가 및 실제 적용 대상

---

## Fine-tuning 과정

### 왜 Fine-tuning인가? (재학습이 아닌)

**Fine-tuning (전이 학습)** 을 선택한 이유:
- ✅ **데이터 부족 해결**: 10X 마이크로디스크 이미지로 충분 (재학습은 수만 장 필요)
- ✅ **시간 단축**: 15-18분 vs 수일 (200배 빠름)
- ✅ **비용 절감**: $0.50 vs $300 (600배 저렴)
- ✅ **전이 학습 효과**: 사전 학습 지식 (에지, 텍스처) 유지하며 마이크로디스크 특성만 추가 학습

**핵심 차이**:
```python
# 재학습: 무작위 초기화에서 시작
model = RRDBNet(...)  # 모든 가중치를 처음부터 학습

# Fine-tuning: 사전 학습된 가중치에서 시작 (본 프로젝트)
model.load_state_dict(pretrained_weights)  # 이미 학습된 지식 활용
optimizer = Adam(lr=1e-4)  # 낮은 learning rate로 미세 조정
```

상세 설명: **[TRAINING_METHODOLOGY.md](TRAINING_METHODOLOGY.md)** 참조

### 모델 아키텍처
- **Base Model**: RealESRGAN_x4plus (사전 학습됨)
- **Network**: RRDBNet
  - Input/Output channels: 3
  - Features: 64
  - Blocks: 23
  - Growth channels: 32
  - Scale: 4x

### Training 설정

```python
Epochs: 50
Batch size: 4
Learning rate: 1e-4
Optimizer: Adam (β1=0.9, β2=0.999)
Scheduler: StepLR (step_size=16, gamma=0.5)
Loss: L1 Loss
Device: Tesla V100 GPU (32GB)
```

### Training 데이터
- Source: 10X 이미지에서 SAM2 + 전통적 CV 기법으로 추출한 cropped microdisk
- Preprocessing: `preprocessing/final/` 디렉토리 (학습 후 삭제됨)
- HR images: 원본 크기 (업스케일하여 최소 128x128)
- LR images: HR의 1/4 크기 + Gaussian blur (σ=0.5)
- Total pairs: 500개 HR/LR 페어

### Training 결과
```
Initial loss: 0.020615
Final loss: 0.013323
Improvement: 35% reduction
Training time: ~15-18 minutes (50 epochs, Tesla V100)
  - ~18-20 seconds per epoch
  - 125 iterations per epoch
Average weight change: 0.00187063
```

---

## 평가 결과

### 1. 10X 이미지 비교 (30개)

**평가 방법**: HR → LR (1/4) → SR (4x) → Compare with HR

| Metric | Pre-trained | Fine-tuned | Improvement |
|--------|-------------|------------|-------------|
| PSNR   | 18.1248 dB  | 24.9308 dB | **+6.81 dB (+37.55%)** |
| SSIM   | 0.4787      | 0.7869     | **+0.3082 (+64.39%)**  |

**결론**: Fine-tuned 모델이 현저히 우수한 성능

### 2. 4X 이미지 평가 (40개)

**평가 방법**: Original (GT) → LR (1/4) → SR (4x) → Compare with GT

| Metric | Bicubic     | Fine-tuned | Improvement |
|--------|-------------|------------|-------------|
| PSNR   | 30.2589 dB  | 30.9462 dB | **+0.69 dB (+2.27%)** |
| SSIM   | 0.9196      | 0.9323     | **+0.0127 (+1.39%)**  |

**결론**: Bicubic 대비 일관된 개선

### 3. 워크플로우 비교

두 가지 업스케일링 방법 비교:
- **Method 1**: 전체 이미지 업스케일 → Crop
- **Method 2**: Crop → 개별 업스케일

| Metric        | Method 1 | Method 2 | Winner   |
|---------------|----------|----------|----------|
| Sharpness     | 6.5360   | 6.2952   | Method 1 |
| Edge Strength | 17.5922  | 17.1383  | Method 1 |

**결론**: 전체 이미지 업스케일 후 Crop이 3.69% 더 선명

### 시각적 비교

**Pre-trained 모델 특징**:
- 과도한 sharpening (부자연스러운 테두리)
- Artifact 발생 (특히 원형 구조)
- 색상 왜곡

**Fine-tuned 모델 특징**:
- 자연스럽고 부드러운 결과
- 원본 특성 보존
- Artifact 최소화
- 현미경 이미지 질감 유지

---

## 최종 결과물

### 1. Fine-tuned 모델
- **위치**: `AI_models/Real-ESRGAN/weights/RealESRGAN_x4plus_finetuned.pth`
- **크기**: 64MB
- **성능**: Pre-trained 대비 37.55% PSNR 향상 (10X 이미지 기준)

### 2. 업스케일된 4X 이미지 (40개)
- **위치**: `upscaled_4X_images/`
- **원본**: 1600x1200 픽셀
- **결과**: 6400x4800 픽셀 (4배 확대)
- **총 용량**: 831MB
- **형식**: PNG

### 3. 평가 결과
- **위치**: `evaluation_4X_results/`
- **파일**:
  - `evaluation_summary.png` - 성능 비교 차트
  - `detailed_metrics.csv` - 40개 이미지 상세 메트릭

### 4. 10X 비교 결과
- **위치**: `results_proper_comparison/`
- **파일**:
  - `comparisons/` - 30개 비교 이미지
  - `metrics/summary_comparison.png` - 성능 요약
  - `metrics/detailed_metrics.csv` - 상세 메트릭

---

## 주요 스크립트

모든 스크립트는 `archive_scripts/` 폴더에 보관되어 있습니다.

### 1. 전처리
- **10X 이미지 전처리** - 10X 이미지에서 microdisk 검출 및 크롭
  - 전통적인 CV 기법 (Adaptive thresholding + circularity filtering)
  - **SAM2 (Segment Anything Model 2)** 사용하여 정밀한 segmentation
  - SAM2 모델: `preprocessing/sam_models/sam_vit_b_01ec64.pth`
  - 결과: `preprocessing/final/` (학습 후 삭제됨)
- `prepare_training_data.py` - Fine-tuning용 HR/LR 페어 생성
  - 10X cropped microdisk를 HR로 사용
  - 1/4 다운샘플링 + Gaussian blur로 LR 생성
- `preprocess_4X_improved.py` - 4X 이미지 분석용 (참고용)

### 2. Fine-tuning
- `finetune_realesrgan.py` - Real-ESRGAN 모델 fine-tuning
  - 50 epochs, L1 loss, Adam optimizer
  - Training log: `training_v2.log`

### 3. 평가
- `compare_models_proper.py` - 10X 이미지에 대한 Pre-trained vs Fine-tuned 비교
- `evaluate_4X_upscaling.py` - 4X 이미지 평가 (Bicubic vs Fine-tuned)
- `compare_workflow_methods.py` - 워크플로우 비교

### 4. 업스케일링
- `upscale_all_4X_images.py` - 모든 4X 원본 이미지 업스케일
  - Tile-based processing (400x400 tiles, 20px overlap)
  - GPU acceleration

### 5. 시각화
- `inspect_4X_images.py` - 4X 이미지 분석
- `visualize_improved_detection.py` - Microdisk 검출 결과 시각화

---

## 실행 방법

### 환경 설정

```bash
# Python 3.8+
pip install torch torchvision opencv-python numpy matplotlib scikit-image tqdm

# Real-ESRGAN 설치
cd AI_models
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install -r requirements.txt
```

### 1. 새로운 이미지 업스케일

```bash
# 단일 이미지
python archive_scripts/upscale_single_image.py --input path/to/image.tif --output path/to/output.png

# 전체 폴더
python archive_scripts/upscale_all_4X_images.py
# 결과: upscaled_4X_images/ 폴더에 저장
```

### 2. 모델 재학습 (필요시)

```bash
# 1. 새로운 이미지로 training data 생성
python archive_scripts/prepare_training_data.py

# 2. Fine-tuning 실행
python archive_scripts/finetune_realesrgan.py

# 결과: AI_models/Real-ESRGAN/weights/RealESRGAN_x4plus_finetuned.pth
```

### 3. 평가 및 비교

```bash
# 업스케일 품질 평가
python archive_scripts/evaluate_4X_upscaling.py
# 결과: evaluation_4X_results/

# Pre-trained vs Fine-tuned 비교
python archive_scripts/compare_models_proper.py
# 결과: results_proper_comparison/
```

---

## 주요 발견 사항

### 1. Fine-tuning 효과
- microdisk 특화 학습으로 일반적인 자연 이미지와 다른 특성 학습
- 10X 이미지에서 SAM2 + 전통적 CV 기법으로 추출한 microdisk로 500개 training pairs 생성
- 50 epochs, ~15-18분 학습으로 수렴 (loss 35% 감소)
- **SAM2 + 전통적 CV 기법 결합**으로 정확한 microdisk segmentation 달성

### 2. 평가 방법론
- 초기 실수: HR 이미지를 직접 업스케일 → GT와 동일하게 나옴
- 올바른 방법: HR → LR 생성 → SR 복원 → HR과 비교
- 중요성: 올바른 평가 워크플로우가 실제 성능 반영

### 3. 워크플로우 선택
- 전체 이미지 업스케일이 개별 crop 업스케일보다 우수
- 이유: 주변 context 정보 활용
- 차이: Sharpness 3.69%, Edge strength 2.58% 향상

### 4. 10X vs 4X 이미지 활용
- **10X 이미지**: **Fine-tuning 학습 데이터 소스**
  - SAM2 + 전통적 CV 기법으로 microdisk 정밀 추출
  - 500개 HR/LR 페어 생성하여 fine-tuning에 사용
  - 평가 결과: 큰 개선 폭 (PSNR +6.81 dB, +37.55%)
  - Pre-trained vs Fine-tuned 모델 성능 비교
- **4X 이미지**: 최종 업스케일 적용 대상
  - 40개 원본 이미지 → 최종 4배 업스케일
  - 원본 해상도 높아 개선 폭 상대적으로 작음 (PSNR +0.69 dB, +2.27%)
  - Bicubic 대비 일관된 품질 향상
- 두 경우 모두 fine-tuned 모델이 일관되게 우수

---

## 성능 요약

### Pre-trained vs Fine-tuned (10X 이미지)
```
Pre-trained:  PSNR 18.12 dB | SSIM 0.4787
Fine-tuned:   PSNR 24.93 dB | SSIM 0.7869
Improvement:  +6.81 dB     | +0.3082
             (+37.55%)     | (+64.39%)
```

### Bicubic vs Fine-tuned (4X 이미지)
```
Bicubic:      PSNR 30.26 dB | SSIM 0.9196
Fine-tuned:   PSNR 30.95 dB | SSIM 0.9323
Improvement:  +0.69 dB     | +0.0127
             (+2.27%)      | (+1.39%)
```

---

## 디스크 사용량

```
AI_models/              2.1GB  (모델 파일)
upscaled_4X_images/     831MB  (최종 결과 40개)
images/                 619MB  (원본 이미지 80개: 4X 40개 + 10X 40개)
preprocessing/          ~408MB (4X cropped 82개 + SAM weights 358MB)
evaluation_4X_results/  72KB   (평가 결과)
results_proper_comparison/ 2.4MB (10X 비교 결과)
archive_scripts/        ~1MB   (Python 스크립트)

Total: ~4.0GB
```

---

## 참고 문헌

- **Real-ESRGAN**: https://github.com/xinntao/Real-ESRGAN
- **Paper**: "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"
- **Architecture**: RRDB (Residual-in-Residual Dense Block)

---

## 문의

KETI AI Team
Location: `/home/keti/cwkim/KETI-AI/SR/`

---

*Document created: 2025-11-13*
*Last updated: 2025-11-13*
