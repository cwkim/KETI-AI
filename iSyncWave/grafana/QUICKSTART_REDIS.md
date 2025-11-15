# 빠른 시작: Redis & Grafana 실시간 모니터링

5분 안에 iSyncWave EEG 데이터를 실시간으로 웹에서 모니터링하는 방법

## 전제 조건

✅ Redis 실행 중 (`redis-cli ping` 으로 확인)
✅ Grafana 실행 중 (http://localhost:3000 접속 가능)
✅ iSyncWave 장치와 태블릿 앱 준비

## Step 1: 데이터 수신 시작 (30초)

터미널에서 실행:

```bash
cd /home/keti/cwkim/KETI-AI/iSyncWave
python3 save_lsl_to_csv.py
```

출력 확인:
```
✓ Connected to Redis at localhost:6379
✓ Redis metadata saved to 'isyncwave:eeg:meta'
📁 Saving data to: data/lsl_data_25-11-13_14_30_45.csv
📊 Streaming to Redis: isyncwave:eeg:stream
📍 Latest data in Redis: isyncwave:eeg:latest
```

이 스크립트는:
- LSL 스트림에서 데이터 수신
- CSV 파일로 저장
- Redis에 실시간 스트리밍

## Step 2: Redis 데이터 확인 (30초)

새 터미널을 열고:

```bash
# 최신 데이터 조회
python3 view_redis_data.py latest
```

또는 실시간 모니터링:

```bash
# 1초마다 업데이트
python3 view_redis_data.py monitor
```

데이터가 보이면 Redis 저장이 정상 작동하는 것입니다!

## Step 3: Grafana 설정 (3분)

### 3.1 Grafana 접속 및 계정 설정

**계정을 keti / keti1234! 로 설정:**

```bash
# 자동 스크립트 실행
sudo ./reset_grafana_simple.sh
```

완료 후 웹 브라우저에서 접속:
```
http://localhost:3000
```

- Username: `keti`
- Password: `keti1234!`

**수동 설정을 원하면:** [GRAFANA_PASSWORD_RESET.md](GRAFANA_PASSWORD_RESET.md) 참고

### 3.2 Redis Data Source 플러그인 설치

방법 1: CLI로 설치 (추천)
```bash
grafana-cli plugins install redis-datasource
sudo systemctl restart grafana-server
```

방법 2: UI에서 설치
1. 좌측 메뉴 → Administration → Plugins
2. "Redis" 검색
3. Redis Data Source 설치
4. Grafana 재시작

### 3.3 Redis 데이터소스 추가

1. 좌측 메뉴 → Connections → Data sources
2. "Add data source" 클릭
3. "Redis" 선택
4. 설정:
   - Name: `iSyncWave Redis`
   - Address: `localhost:6379`
   - Database: `0`
5. "Save & test" 클릭
6. ✅ "Data source is working" 확인

### 3.4 대시보드 Import

1. 좌측 메뉴 → Dashboards → Import
2. "Upload JSON file" 클릭
3. `grafana_dashboard.json` 파일 선택
4. Redis 데이터소스 선택: "iSyncWave Redis"
5. "Import" 클릭

## Step 4: 실시간 모니터링 확인! 🎉

대시보드가 열리면 다음을 볼 수 있습니다:

- 📊 **Stream Metadata**: 채널 수, 샘플링 레이트 등
- 🎚️ **Latest EEG Gauges**: 주요 채널의 실시간 값
- 📈 **Time Series Graph**: 모든 채널의 시계열 데이터
- 📉 **Statistics**: 총 샘플 수, 기록 시간 등

대시보드는 **5초마다 자동 새로고침**됩니다!

## 전체 워크플로우

```
터미널 1                      터미널 2 (선택)              웹 브라우저
─────────                    ───────────────            ──────────
python3 save_lsl_to_csv.py   python3 view_redis_data.py  http://localhost:3000
       ↓                            ↓                         ↓
   CSV + Redis                실시간 모니터링             Grafana 대시보드
```

## 트러블슈팅

### ❌ Redis 연결 실패

```bash
# Redis 실행 확인
redis-cli ping

# PONG 응답이 없으면 Redis 시작
redis-server
```

### ❌ Grafana에서 데이터가 안 보임

1. Redis에 데이터가 있는지 확인:
   ```bash
   python3 view_redis_data.py latest
   ```

2. Grafana Data Source 연결 테스트:
   - Connections → Data sources → iSyncWave Redis → Test

3. 대시보드 수동 새로고침 (우측 상단 Refresh 버튼)

### ❌ LSL 스트림을 찾을 수 없음

1. iSyncWave 장치 전원 확인
2. 태블릿 앱 실행 및 LSL 스트리밍 활성화 확인
3. 네트워크 연결 확인 (같은 Wi-Fi)

## 다음 단계

✅ **데이터 분석**: CSV 파일을 Python pandas로 분석
✅ **대시보드 커스터마이징**: Grafana에서 원하는 패널 추가
✅ **장기 모니터링**: 무한 수신 모드로 실행

자세한 내용은 [REDIS_GRAFANA_SETUP.md](REDIS_GRAFANA_SETUP.md) 참고!

## 요약 명령어

```bash
# 1. 데이터 수신 시작
python3 save_lsl_to_csv.py

# 2. Redis 데이터 확인
python3 view_redis_data.py latest
python3 view_redis_data.py monitor

# 3. Grafana 접속
# http://localhost:3000

# 4. 데이터 저장 위치
# CSV: data/lsl_data_*.csv
# Redis: isyncwave:eeg:* 키들
```

끝! 🎊
