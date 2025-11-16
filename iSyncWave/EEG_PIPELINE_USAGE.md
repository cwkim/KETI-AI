# iSyncWave EEG 데이터 처리 파이프라인 사용법

## 개요

이 파이프라인은 iSyncWave 장비로 수집한 EEG 데이터를 Redis에서 읽어 Nature Communications 2025 논문의 전처리 방법을 적용하여 분석 및 학습용 데이터를 생성합니다.

**논문**: "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level"
**오픈소스 코드**: https://github.com/bfinl/Finger-BCI-Decoding

---

## 시스템 요구사항

### 필수 패키지

```bash
pip install redis numpy mne scipy matplotlib
```

### Redis 서버

```bash
# Redis 서버 실행 (이미 실행 중이면 생략)
redis-server
```

---

## 파일 구조

```
iSyncWave/
├── redis_eeg_pipeline.py          # 메인 파이프라인 (사용)
├── EEG_PIPELINE_USAGE.md          # 이 문서
├── lsl_to_redis.py                # LSL → Redis 데이터 수집
├── lsl_to_redis.sh                # 데이터 수집 스크립트
├── grafana/                        # Grafana 대시보드 설정
└── docs/                           # 참고 논문
```

---

## 사용 방법

### 1. 데이터 수집

iSyncWave 장비를 착용하고 LSL 스트림을 Redis로 저장:

```bash
# 백그라운드로 데이터 수집 시작
./lsl_to_redis.sh

# 또는 직접 실행
python3 lsl_to_redis.py
```

**Redis 저장 형식:**
- 메타데이터: `isyncwave:eeg:meta` (hash)
  - `channel_count`: 채널 수
  - `sampling_rate`: 샘플링 레이트 (Hz)
  - `channels`: 채널 이름 (쉼표 구분)
- 스트림 데이터: `isyncwave:eeg:stream` (stream)
  - 각 샘플마다 채널별 값 + LSL 타임스탬프

### 2. 데이터 전처리 및 분석

파이프라인 실행:

```bash
python3 redis_eeg_pipeline.py
```

**자동 실행되는 작업:**
1. Redis에서 EEG 데이터 로드
2. Nature 논문 전처리 적용 (CAR, 필터링, 다운샘플링)
3. 1초 윈도우로 세그먼트화 (125ms 스텝)
4. 알파/베타 밴드 특징 추출
5. 결과 시각화 (`redis_pipeline_results.png`)
6. 전처리 데이터 저장 (`.pkl` 파일)

### 3. Python API 사용

#### 기본 사용법

```python
from redis_eeg_pipeline import RedisEEGPipeline

# 파이프라인 초기화
pipeline = RedisEEGPipeline(
    redis_host='localhost',
    redis_port=6379,
    redis_db=0
)

# 1. Redis 연결
if not pipeline.connect():
    print("Redis 연결 실패!")
    exit()

# 2. 사용 가능한 데이터 시간 범위 확인
time_range = pipeline.get_available_time_range()
# 출력: 시작 시간, 종료 시간, 총 시간, 총 샘플 수

# 3. 데이터 로드 (여러 방법)

# 방법 1: 전체 데이터 로드
if not pipeline.load_data():
    print("데이터 로드 실패!")
    exit()

# 방법 2: 최근 N개 샘플만 로드
if not pipeline.load_data(count=100000):
    print("데이터 로드 실패!")
    exit()

# 방법 3: LSL 타임스탬프로 시간 범위 지정 (예: 100초~200초)
if not pipeline.load_data(start_time=100.0, end_time=200.0):
    print("데이터 로드 실패!")
    exit()

# 방법 4: datetime 객체로 시간 범위 지정
from datetime import datetime, timedelta
start_dt = datetime(2025, 11, 15, 14, 30, 0)  # 2025-11-15 14:30:00
end_dt = start_dt + timedelta(minutes=5)      # 5분간 데이터
if not pipeline.load_data(start_time=start_dt, end_time=end_dt):
    print("데이터 로드 실패!")
    exit()

# 3. 전처리 (ICA 아티팩트 제거 포함)
if not pipeline.preprocess(apply_ica=True):
    print("전처리 실패!")
    exit()

# 4. 세그먼트화
if not pipeline.segment_data(
    window_size=1.0,    # 1초 윈도우
    step_size=0.125     # 125ms 스텝
):
    print("세그먼트화 실패!")
    exit()

# 5. 특징 추출
if not pipeline.extract_features(
    alpha_band=(8, 13),   # 알파 밴드
    beta_band=(13, 30)    # 베타 밴드
):
    print("특징 추출 실패!")
    exit()

# 6. 결과 시각화
pipeline.visualize_results(save_path='my_results.png')

# 7. 데이터 저장
output_file = pipeline.save_processed_data(
    output_path='my_eeg_data.pkl'
)
print(f"저장 완료: {output_file}")
```

### 4. 저장된 데이터 사용

```python
import pickle
import numpy as np

# 저장된 데이터 로드
with open('processed_eeg_data_20251115_234510.pkl', 'rb') as f:
    data = pickle.load(f)

# 데이터 구조
print("원본 데이터:", data['original_data']['signals'].shape)
# (19, 61748) - 19채널 × 61748샘플

print("전처리 데이터:", data['processed_data']['signals'].shape)
# (19, 24699) - 19채널 × 24699샘플 (100 Hz)

print("세그먼트:", data['segments']['data'].shape)
# (2050, 19, 100) - 2050개 세그먼트 × 19채널 × 100샘플(1초)

print("알파 파워:", data['features']['alpha_power'].shape)
# (2050, 19) - 2050개 세그먼트 × 19채널

print("베타 파워:", data['features']['beta_power'].shape)
# (2050, 19) - 2050개 세그먼트 × 19채널

# 특징 데이터 추출
alpha_features = data['features']['alpha_power']
beta_features = data['features']['beta_power']

# 특징 결합 (딥러닝 모델 입력용)
features = np.concatenate([alpha_features, beta_features], axis=1)
# (2050, 38) - 2050개 세그먼트 × 38개 특징
```

---

## 전처리 파이프라인 상세

### Nature Communications 2025 논문 방법

```
┌─────────────────────────────────────────────────┐
│ 1. Common Average Reference (CAR)              │
│    모든 채널의 평균을 각 채널에서 빼서         │
│    공통 노이즈 제거                             │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 2. Notch Filter (50Hz, 60Hz)                   │
│    전원 라인 노이즈 제거                        │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 3. Downsample to 100 Hz                        │
│    250 Hz → 100 Hz로 다운샘플링                │
│    (계산 효율성 및 노이즈 감소)                 │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 4. Bandpass Filter (4-40 Hz)                   │
│    4차 Butterworth 필터                         │
│    EEG 주요 주파수 대역만 통과                  │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 5. ICA Artifact Removal (선택적)               │
│    눈 깜빡임(EOG) 등 아티팩트 자동 제거         │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 6. Segmentation (1s window, 125ms step)        │
│    연속 데이터를 1초 윈도우로 분할              │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 7. Z-score Normalization                       │
│    각 세그먼트마다 정규화                       │
└─────────────────────────────────────────────────┘
```

### 특징 추출

**Event-Related Desynchronization (ERD):**
- **알파 밴드 (8-13 Hz)**: 운동 상상/실행 시 파워 감소
- **베타 밴드 (13-30 Hz)**: 운동 준비/실행 시 변화

**계산 방법:**
- Welch 방법으로 Power Spectral Density (PSD) 계산
- 각 주파수 밴드의 평균 파워 추출

---

## 출력 파일

### 1. 시각화 이미지 (`redis_pipeline_results.png`)

- **전처리된 EEG 신호**: 첫 번째 채널의 시계열
- **알파 파워 시계열**: 시간에 따른 알파 밴드 변화
- **베타 파워 시계열**: 시간에 따른 베타 밴드 변화
- **채널별 평균 알파 파워**: 어느 채널이 가장 활성화되는지
- **채널별 평균 베타 파워**: 채널별 베타 활동 비교

### 2. 전처리 데이터 (`processed_eeg_data_YYYYMMDD_HHMMSS.pkl`)

```python
{
    'original_data': {
        'channels': ['Fp1', 'Fp2', ...],
        'signals': numpy.ndarray,      # (n_channels, n_samples)
        'sampling_rate': 250.0,
        'timestamps': numpy.ndarray,
        'duration': float
    },
    'processed_data': {
        'signals': numpy.ndarray,      # (n_channels, n_samples_100hz)
        'sampling_rate': 100.0,
        'channels': ['Fp1', 'Fp2', ...],
        'duration': float
    },
    'segments': {
        'data': numpy.ndarray,         # (n_segments, n_channels, 100)
        'times': numpy.ndarray,        # 각 세그먼트 시작 시간
        'window_size': 1.0,
        'step_size': 0.125,
        'sampling_rate': 100.0
    },
    'features': {
        'alpha_power': numpy.ndarray,  # (n_segments, n_channels)
        'beta_power': numpy.ndarray,   # (n_segments, n_channels)
        'alpha_band': (8, 13),
        'beta_band': (13, 30),
        'segment_times': numpy.ndarray
    },
    'pipeline_info': {
        'preprocessing': 'Nature Communications 2025',
        'timestamp': '2025-11-15T23:45:10'
    }
}
```

---

## 딥러닝 모델 학습 예제

### EEGNet 사용 예제

```python
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# 1. 데이터 로드
with open('processed_eeg_data_20251115_234510.pkl', 'rb') as f:
    data = pickle.load(f)

# 2. 세그먼트 데이터 추출
X = data['segments']['data']  # (n_segments, n_channels, n_samples)
# X shape: (2050, 19, 100)

# 레이블 추가 필요 (실험 프로토콜에 따라)
# y = ... (각 세그먼트에 대한 레이블)

# 3. 데이터 형태 변환 (EEGNet 입력 형식)
X = np.expand_dims(X, axis=-1)  # (n_segments, n_channels, n_samples, 1)

# 4. 간단한 CNN 모델
def create_model(n_channels=19, n_samples=100):
    model = keras.Sequential([
        layers.Input(shape=(n_channels, n_samples, 1)),

        # Conv2D
        layers.Conv2D(16, (1, 32), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),

        # DepthwiseConv2D
        layers.DepthwiseConv2D((n_channels, 1),
                               depth_multiplier=2,
                               depthwise_constraint=keras.constraints.max_norm(1.)),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(0.5),

        # SeparableConv2D
        layers.SeparableConv2D(32, (1, 16), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('elu'),
        layers.AveragePooling2D((1, 8)),
        layers.Dropout(0.5),

        layers.Flatten(),
        layers.Dense(10, activation='softmax')  # 10개 클래스 예시
    ])

    return model

# 5. 모델 컴파일 및 학습
model = create_model()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)
```

---

## FAQ

### Q1. Redis에 데이터가 없다고 나옵니다.

**A:** LSL 스트림을 먼저 시작하고 `lsl_to_redis.py`를 실행하세요.

```bash
# LSL 스트림 확인
python3 discover_lsl_streams.py

# Redis에 데이터 저장
python3 lsl_to_redis.py
```

### Q2. ICA가 실패합니다.

**A:** EOG 채널이 없으면 ICA가 실패할 수 있습니다. `apply_ica=False`로 설정하세요.

```python
pipeline.preprocess(apply_ica=False)
```

### Q3. 메모리 부족 오류가 발생합니다.

**A:** `load_data()`의 `count` 파라미터를 줄이세요.

```python
pipeline.load_data(count=10000)  # 기본값 100000에서 감소
```

### Q4. 타임스탬프 마커를 제거하고 싶습니다.

**A:** 이 파이프라인은 타임스탬프 마커 없이 연속 데이터를 처리합니다. 별도 작업 불필요합니다.

### Q5. 특정 이벤트 구간만 분석하고 싶습니다.

**A:** `get_available_time_range()`로 전체 범위를 확인한 후, `load_data()`의 `start_time`/`end_time` 파라미터로 원하는 구간만 로드하세요.

```python
# 1. 전체 시간 범위 확인
time_range = pipeline.get_available_time_range()
print(f"전체 범위: {time_range['start_timestamp']:.2f} ~ {time_range['end_timestamp']:.2f}")

# 2. 특정 이벤트 구간만 로드 (예: 100초~150초 구간)
pipeline.load_data(start_time=100.0, end_time=150.0)
```

### Q6. 여러 이벤트 구간을 순차적으로 분석하려면?

**A:** 각 구간마다 `load_data()` + `preprocess()` + ... 를 반복하면 됩니다.

```python
# 이벤트 구간 정의 (시작, 종료 시간)
events = [
    (100.0, 120.0),  # 이벤트 1
    (150.0, 170.0),  # 이벤트 2
    (200.0, 220.0),  # 이벤트 3
]

all_features = []

for start, end in events:
    # 구간별 데이터 로드
    pipeline.load_data(start_time=start, end_time=end)
    pipeline.preprocess(apply_ica=False)
    pipeline.segment_data()
    pipeline.extract_features()

    # 특징 저장
    all_features.append(pipeline.features['alpha_power'])

# 모든 특징 결합
import numpy as np
combined_features = np.concatenate(all_features, axis=0)
```

### Q7. 실시간 분석이 가능한가요?

**A:** 현재는 배치 처리만 지원합니다. 실시간 분석을 위해서는 Redis XREAD를 사용한 스트리밍 모드가 필요합니다.

---

## 참고 자료

### 논문
- Nature Communications 2025: "EEG-based brain-computer interface enables real-time robotic hand control"
- 논문 PDF: `docs/EEG_based_brain_computer_interface_enables_real_time_robotic_hand.pdf`

### 오픈소스 코드
- GitHub: https://github.com/bfinl/Finger-BCI-Decoding

### MNE-Python 문서
- https://mne.tools/stable/index.html

---

## 라이선스

MIT License

---

## 문의

문제가 발생하거나 질문이 있으면 이슈를 등록해주세요.
