# Redis와 Grafana를 이용한 실시간 EEG 데이터 모니터링

이 가이드는 iSyncWave에서 수신한 EEG 데이터를 Redis에 저장하고 Grafana에서 실시간으로 시각화하는 방법을 설명합니다.

## 시스템 구성

```
iSyncWave 장치 (LSL) → Python 스크립트 → Redis → Grafana 웹 대시보드
                              ↓
                          CSV 파일
```

## Redis 데이터 구조

스크립트는 Redis에 다음 3가지 키를 사용합니다:

1. **`isyncwave:eeg:stream`** (Redis Stream)
   - 모든 EEG 샘플 데이터 저장 (최근 10,000개)
   - 시계열 데이터로 히스토리 조회 가능

2. **`isyncwave:eeg:latest`** (Redis Hash)
   - 가장 최근의 EEG 샘플 데이터
   - 실시간 모니터링에 사용

3. **`isyncwave:eeg:meta`** (Redis Hash)
   - 스트림 메타데이터 (채널 정보, 샘플링 레이트 등)

## 1. 데이터 수신 및 저장

### 기본 사용법 (Redis 저장 활성화)

```bash
# CSV와 Redis에 동시 저장
python3 save_lsl_to_csv.py -d 60

# 무한정 수신 (Ctrl+C로 중지)
python3 save_lsl_to_csv.py
```

### Redis 없이 CSV만 저장

```bash
python3 save_lsl_to_csv.py --no-redis -d 60
```

### Redis 연결 설정 변경

```bash
# 다른 Redis 서버 사용
python3 save_lsl_to_csv.py --redis-host 192.168.0.10 --redis-port 6379

# 다른 Redis 데이터베이스 사용
python3 save_lsl_to_csv.py --redis-db 1
```

## 2. Redis 데이터 조회

`view_redis_data.py` 스크립트로 Redis에 저장된 데이터를 조회할 수 있습니다.

### 최신 데이터 조회

```bash
python3 view_redis_data.py latest
```

출력 예시:
```
================================================================================
Latest EEG Data from Redis
================================================================================

Stream Information:
  Name: iSyncWave_EEG
  Type: EEG
  Channels: 19
  Sampling Rate: 250.0 Hz
  Start Time: 2025-11-13T14:30:45.123456

Latest Sample:
  Timestamp: 1699876845.123
  DateTime: 2025-11-13T14:30:45.123456

  Channel Values:
    Fp1 : 0.00156726
    Fp2 : 0.00012409
    F7  : 0.00093045
    ...
```

### 최근 히스토리 조회

```bash
# 최근 10개 샘플 조회
python3 view_redis_data.py stream

# 최근 50개 샘플 조회
python3 view_redis_data.py stream -c 50
```

### 실시간 모니터링

```bash
# 1초마다 최신 데이터 출력
python3 view_redis_data.py monitor

# 0.5초마다 업데이트
python3 view_redis_data.py monitor -i 0.5
```

### Redis 통계 조회

```bash
python3 view_redis_data.py stats
```

출력 예시:
```
================================================================================
Redis Data Statistics
================================================================================

Metadata:
  stream_name: iSyncWave_EEG
  stream_type: EEG
  channel_count: 19
  sampling_rate: 250.0
  channels: Fp1,Fp2,F7,F3,Fz,F4,F8,T3,C3,Cz,C4,T4,T5,P3,Pz,P4,T6,O1,O2
  start_time: 2025-11-13T14:30:45.123456
  total_samples: 15000
  duration_seconds: 60.0

Stream Information:
  Total entries: 10000
  First entry ID: 1699876845123-0
  Last entry ID: 1699876905123-0

Redis Memory Usage: 12.45 MB
```

## 3. Grafana 설정

### 3.1. Redis Data Source 플러그인 설치

Grafana에서 Redis를 데이터소스로 사용하려면 플러그인 설치가 필요합니다.

```bash
# Grafana CLI로 설치
grafana-cli plugins install redis-datasource

# Grafana 재시작
sudo systemctl restart grafana-server
```

또는 Grafana UI에서:
1. Grafana 웹 접속: http://localhost:3000
2. 좌측 메뉴 → Administration → Plugins
3. "Redis" 검색
4. Redis Data Source 플러그인 설치

### 3.2. Grafana 로그인 설정

**계정을 keti / keti1234! 로 설정:**

```bash
cd /home/keti/cwkim/KETI-AI/iSyncWave
sudo ./reset_grafana_simple.sh
```

완료 후:
- URL: http://localhost:3000
- Username: `keti`
- Password: `keti1234!`

자세한 방법: [GRAFANA_PASSWORD_RESET.md](GRAFANA_PASSWORD_RESET.md) 또는 [GRAFANA_LOGIN_INSTRUCTIONS.txt](GRAFANA_LOGIN_INSTRUCTIONS.txt)

### 3.3. Redis 데이터소스 추가

1. Grafana 웹 접속: http://localhost:3000 (keti / keti1234! 로 로그인)

2. 좌측 메뉴 → Connections → Data sources → Add data source

3. Redis 선택

4. 설정:
   - **Name**: iSyncWave Redis
   - **Address**: `localhost:6379`
   - **Database**: `0`
   - **Save & test** 클릭

### 3.4. 대시보드 Import

준비된 대시보드 JSON 파일을 import합니다:

1. 좌측 메뉴 → Dashboards → Import

2. "Upload JSON file" 클릭

3. `grafana_dashboard.json` 파일 선택

4. Redis 데이터소스 선택: "iSyncWave Redis"

5. Import 클릭

### 3.5. 대시보드 구성

Import한 대시보드에는 다음 패널이 포함되어 있습니다:

1. **Stream Metadata** - 스트림 정보 표시
2. **Latest EEG Data Gauges** - 주요 채널의 최신 데이터 (Fp1, Fp2, Cz, O1)
3. **EEG Time Series** - 모든 채널의 시계열 그래프 (최근 100 샘플)
4. **Sampling Rate Monitor** - 샘플링 레이트 모니터
5. **Total Samples** - 총 수집 샘플 수
6. **Recording Duration** - 기록 시간

### 3.6. 대시보드 접속

```
http://localhost:3000/d/isyncwave-eeg
```

대시보드는 기본적으로 5초마다 자동 새로고침됩니다.

## 4. 수동으로 대시보드 패널 만들기

대시보드를 직접 만들고 싶다면:

### 4.1. 최신 데이터 표시 (Stat 패널)

1. Add panel → Visualization: Stat
2. Query:
   ```
   HGET isyncwave:eeg:latest Fp1
   ```
3. Panel title: "Fp1 Latest Value"

### 4.2. 시계열 그래프 (Time series 패널)

1. Add panel → Visualization: Time series
2. Query:
   ```
   XREVRANGE isyncwave:eeg:stream + - COUNT 100
   ```
3. Transform: "Extract fields" 선택
   - Source: Select field names matching pattern
   - Pattern: `Fp1|Fp2|Cz|O1` (원하는 채널)

### 4.3. 메타데이터 표시 (Table 패널)

1. Add panel → Visualization: Table
2. Query:
   ```
   HGETALL isyncwave:eeg:meta
   ```

## 5. Redis 직접 명령어

Redis CLI로 직접 데이터를 조회할 수도 있습니다:

```bash
# Redis CLI 접속
redis-cli

# 최신 데이터 조회
HGETALL isyncwave:eeg:latest

# 특정 채널 값 조회
HGET isyncwave:eeg:latest Fp1

# 메타데이터 조회
HGETALL isyncwave:eeg:meta

# 스트림 최근 10개 조회
XREVRANGE isyncwave:eeg:stream + - COUNT 10

# 스트림 총 개수
XLEN isyncwave:eeg:stream
```

## 6. 실시간 모니터링 워크플로우

### 전체 워크플로우

```bash
# 터미널 1: 데이터 수신 및 저장
python3 save_lsl_to_csv.py

# 터미널 2: Redis 실시간 모니터링 (선택사항)
python3 view_redis_data.py monitor

# 웹 브라우저: Grafana 대시보드 접속
# http://localhost:3000
```

### 출력 예시 (save_lsl_to_csv.py)

```
Searching for LSL streams...
======================================================================
Connecting to stream:
  Name: iSyncWave_EEG
  Type: EEG
  Channels: 19
  Sampling Rate: 250.0 Hz
======================================================================

Channel names: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2

✓ Connected to Redis at localhost:6379
✓ Redis metadata saved to 'isyncwave:eeg:meta'

📁 Saving data to: data/lsl_data_25-11-13_14_30_45.csv
📊 Streaming to Redis: isyncwave:eeg:stream
📍 Latest data in Redis: isyncwave:eeg:latest
Duration: infinite seconds
Press Ctrl+C to stop

✓ CSV file created with header
✓ Receiving data...

📊 2500 samples saved (Redis: 2500) | Rate: 250.00 Hz | Elapsed: 10.0s
```

## 7. 트러블슈팅

### Redis 연결 실패

```
⚠ Could not connect to Redis: Connection refused
```

**해결 방법**:
```bash
# Redis 실행 확인
redis-cli ping

# Redis 시작
redis-server
```

### Grafana에서 데이터가 보이지 않음

1. **Redis 데이터소스 연결 확인**
   - Grafana → Data sources → iSyncWave Redis → Test
   - "Data source is working" 메시지 확인

2. **데이터가 Redis에 있는지 확인**
   ```bash
   python3 view_redis_data.py latest
   ```

3. **쿼리 문법 확인**
   - Redis 명령어가 올바른지 확인
   - 대소문자 구분 (키 이름은 정확히 일치해야 함)

### 대시보드 새로고침이 안 됨

- 대시보드 상단의 Auto-refresh 설정 확인 (기본: 5s)
- 수동 새로고침: 우측 상단의 Refresh 버튼 클릭

### Redis 메모리 부족

Redis Stream은 최근 10,000개 샘플만 유지하도록 설정되어 있습니다 (약 2-3MB).

더 많은 히스토리가 필요하다면 `save_lsl_to_csv.py` 189번째 줄 수정:
```python
redis_client.xadd(redis_stream_key, data_dict, maxlen=10000)  # 원하는 숫자로 변경
```

## 8. 고급 설정

### 여러 장치 동시 모니터링

여러 iSyncWave 장치를 동시에 모니터링하려면:

```bash
# 장치 1
python3 save_lsl_to_csv.py -n "Device1" --redis-db 0

# 장치 2
python3 save_lsl_to_csv.py -n "Device2" --redis-db 1
```

각 장치마다 다른 Redis 데이터베이스를 사용하고, Grafana에서 여러 데이터소스를 추가합니다.

### 데이터 보존 정책

장기간 데이터 보존이 필요하다면:
- CSV 파일 사용 (영구 보존)
- Redis는 실시간 모니터링 용도로만 사용
- 필요시 Redis를 디스크에 저장 (RDB/AOF 설정)

## 9. 참고 자료

- Redis Streams: https://redis.io/docs/data-types/streams/
- Grafana Redis Data Source: https://grafana.com/grafana/plugins/redis-datasource/
- LSL Protocol: https://labstreaminglayer.readthedocs.io/

## 요약

1. **데이터 수신**: `python3 save_lsl_to_csv.py` → CSV + Redis
2. **데이터 조회**: `python3 view_redis_data.py [mode]`
3. **실시간 시각화**: Grafana 웹 (http://localhost:3000)

모든 데이터는 CSV 파일에도 저장되므로, Redis가 없어도 나중에 분석 가능합니다!
