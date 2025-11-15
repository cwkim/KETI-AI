# iSyncWave LSL to Redis Streamer

iSyncWave 장치에서 전송되는 LSL(Lab Streaming Layer) 뇌파 데이터를 실시간으로 수신하여 Redis에 저장하고, Grafana로 시각화하는 시스템입니다.

## 주요 기능

- ✅ **LSL 스트림 자동 검색 및 연결**
- ✅ **Redis 실시간 데이터 스트리밍** (주 기능)
- ✅ **19채널 표준 EEG 전극 이름 지원** (10-20 system)
- ✅ **Grafana 웹 대시보드 실시간 시각화**
- ✅ **선택적 CSV 파일 저장**
- ✅ **Redis 데이터 보관 정책 설정 가능**

## 빠른 시작

### 1. Redis로 데이터 스트리밍 (기본)

```bash
# 기본 사용 (무한정 수신, Redis에 무제한 저장)
python3 lsl_to_redis.py

# 또는 shell script 사용
./lsl_to_redis.sh
```

### 2. Redis 데이터 보관 정책 설정

```bash
# 최근 10,000개 샘플만 유지 (약 40초 분량, 250Hz 기준)
python3 lsl_to_redis.py --redis-maxlen 10000

# 최근 50,000개 샘플만 유지 (약 3분 20초 분량)
python3 lsl_to_redis.py --redis-maxlen 50000
```

### 3. CSV 파일도 함께 저장

```bash
# Redis + CSV 동시 저장
python3 lsl_to_redis.py --enable-csv

# 파일명 지정
python3 lsl_to_redis.py --enable-csv -o my_experiment.csv

# 특정 디렉토리에 저장
python3 lsl_to_redis.py --enable-csv -dir experiments
```

### 4. Grafana 대시보드로 실시간 모니터링

```bash
# Grafana 계정 설정 (최초 1회)
sudo ./grafana/reset_grafana_simple.sh
```

**웹 브라우저에서 접속:**
- URL: http://localhost:3000
- 계정: `keti` / 비밀번호: `keti1234!`
- 19개 EEG 채널 실시간 시각화
- 자동 새로고침 (5초 간격)

## EEG 채널 구성

데이터는 **19채널 표준 10-20 시스템** 전극 배치로 저장됩니다:

```
Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2
```

## Redis 데이터 구조

### 저장 키
- `isyncwave:eeg:stream` - Redis Stream (시계열 데이터)
- `isyncwave:eeg:meta` - 메타데이터 (채널 정보, 샘플링 레이트 등)

### 데이터 포맷
```json
{
  "timestamp": "1234567890.123",
  "datetime": "2025-11-14T12:34:56.123456",
  "Fp1": "-1514.08",
  "Fp2": "-1268.14",
  "F7": "-937.76",
  ...
  "O2": "-346.85"
}
```

### Redis 데이터 조회

```bash
# 최신 데이터 확인
python3 view_redis_data.py latest

# 최근 50개 샘플 히스토리 조회
python3 view_redis_data.py stream -c 50

# 실시간 모니터링 (1초마다 업데이트)
python3 view_redis_data.py monitor

# Redis 통계 조회
python3 view_redis_data.py stats
```

## 기타 LSL 도구

### LSL 스트림 검색

```bash
# 1회 검색
./lsl_discover.sh

# 또는 Python으로
python3 discover_lsl_streams.py
```

### LSL 데이터 수신 (Redis 없이)

```bash
# 10초 동안 데이터 수신
./lsl_receive.sh -d 10

# 무한정 수신 (Ctrl+C로 종료)
./lsl_receive.sh -d 0

# 조용한 모드 (통계만 표시)
./lsl_receive.sh -q
```

### LSL 연속 모니터링

```bash
# 스트림 자동 검색 및 연결
./lsl_monitor.sh
```

## 명령줄 옵션

### lsl_to_redis.py 옵션

```bash
python3 lsl_to_redis.py [OPTIONS]

옵션:
  -n, --name TEXT          특정 스트림 이름 지정
  -d, --duration INT       수신 시간(초), 0=무한 (기본: 0)

  --enable-csv             CSV 파일 저장 활성화 (기본: 비활성)
  -o, --output TEXT        CSV 파일명 (자동 생성)
  -dir, --directory TEXT   CSV 저장 디렉토리 (기본: data)

  --no-redis               Redis 저장 비활성화
  --redis-host TEXT        Redis 호스트 (기본: localhost)
  --redis-port INT         Redis 포트 (기본: 6379)
  --redis-db INT           Redis DB 번호 (기본: 0)
  --redis-maxlen INT       Redis Stream 최대 길이 (기본: None=무제한)
```

### 사용 예시

```bash
# 60초 동안 수신, 최근 20,000개만 유지
python3 lsl_to_redis.py -d 60 --redis-maxlen 20000

# 특정 스트림에서 CSV + Redis 동시 저장
python3 lsl_to_redis.py -n "iSyncWave-EEG" --enable-csv -o test.csv

# 원격 Redis 서버 사용
python3 lsl_to_redis.py --redis-host 192.168.0.10 --redis-port 6379
```

## 네트워크 설정

현재 장치 IP: **192.168.0.5** (wlan0)

### 연결 요구사항
1. iSyncWave 장치 전원 ON
2. 태블릿 앱 실행 및 LSL 스트리밍 활성화
3. 태블릿과 PC가 동일 Wi-Fi 네트워크에 연결 (192.168.0.x)

## 트러블슈팅

### "No LSL streams found" 오류

1. **태블릿 확인**
   - 앱이 실행 중인지 확인
   - LSL 스트리밍 활성화 확인

2. **네트워크 확인**
   ```bash
   # 태블릿 ping 테스트
   ping <태블릿_IP>
   ```

3. **방화벽 확인**
   - LSL은 UDP 멀티캐스트 사용
   - 필요시 방화벽 규칙 확인

### Redis 연결 오류

```bash
# Redis 서비스 상태 확인
sudo systemctl status redis-server

# Redis 재시작
sudo systemctl restart redis-server
```

### Grafana 접속 불가

```bash
# Grafana 서비스 상태 확인
sudo systemctl status grafana-server

# Grafana 재시작
sudo systemctl restart grafana-server
```

## 프로젝트 구조

```
iSyncWave/
├── README.md                    # 이 파일
│
├── 핵심 Python 스크립트
│   ├── lsl_to_redis.py         # ⭐ Redis 스트리밍 (메인)
│   ├── discover_lsl_streams.py # LSL 스트림 검색
│   ├── monitor_lsl.py          # LSL 연속 모니터링
│   ├── receive_lsl_data.py     # LSL 데이터 수신
│   └── view_redis_data.py      # Redis 데이터 조회
│
├── Shell 스크립트
│   ├── lsl_to_redis.sh         # ⭐ Redis 스트리밍 (메인)
│   ├── lsl_discover.sh         # LSL 검색
│   ├── lsl_monitor.sh          # LSL 모니터링
│   └── lsl_receive.sh          # LSL 수신
│
├── grafana/                    # Grafana & Redis 설정
│   ├── REDIS_GRAFANA_SETUP.md  # 상세 설정 가이드
│   ├── QUICKSTART_REDIS.md     # 빠른 시작 가이드
│   ├── grafana_dashboard.json  # 대시보드 템플릿
│   ├── install_redis_plugin.sh # Redis 플러그인 설치
│   ├── reset_grafana_password.sh
│   └── reset_grafana_simple.sh # 계정 설정 스크립트
│
└── test_edf/                   # 테스트 EDF 파일
```

## 기술 정보

- **LSL 프로토콜 버전**: 1.16.2
- **Python 버전**: 3.8.10
- **Redis 버전**: 6.x
- **Grafana 버전**: 9.x
- **플랫폼**: Linux ARM64 (aarch64)
- **샘플링 레이트**: 250 Hz (iSyncWave 기본값)
- **채널 수**: 19 (표준 10-20 system)

## 문서

### 데이터 사양
- `DATA_SPECIFICATION.md` - **EEG 데이터 수신 상세 사양** ⭐
  - 샘플링 레이트, 채널 정보, 수신 패턴 분석
  - Redis 저장 구조, 타임스탬프 설명
  - 성능 통계 및 데이터 품질 검증 방법

### Grafana 및 Redis 설정
- `grafana/REDIS_GRAFANA_SETUP.md` - Redis와 Grafana 완벽 설정 가이드
- `grafana/QUICKSTART_REDIS.md` - 5분 빠른 시작 가이드
- `grafana/GRAFANA_SETUP_STEP_BY_STEP.md` - Grafana 단계별 설정
- `grafana/GRAFANA_PASSWORD_RESET.md` - Grafana 비밀번호 재설정 가이드

## CSV 파일 포맷

`--enable-csv` 옵션 사용 시 생성되는 CSV 파일 형식:

```csv
Timestamp,Fp1,Fp2,F7,F3,Fz,F4,F8,T3,C3,Cz,C4,T4,T5,P3,Pz,P4,T6,O1,O2
1234567890.123,-1514.08,-1268.14,-937.76,-539.80,-526.67,...
1234567890.127,-1275.57,-1379.72,-807.26,-520.86,-330.59,...
```

## 라이선스

내부 연구용 프로젝트
