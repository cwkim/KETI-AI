# Grafana Redis 대시보드 설정 - 단계별 가이드

실시간 EEG 데이터를 Grafana에서 확인하는 전체 과정

## 전제 조건 확인

```bash
# 1. LSL 데이터 수신 중인지 확인
# 터미널에서 실행 (계속 실행 중이어야 함)
python3 save_lsl_to_csv.py

# 2. Redis에 데이터가 있는지 확인
python3 view_redis_data.py latest
```

---

## Step 1: Grafana 로그인

### 1.1 웹 브라우저 열기
```
http://localhost:3000
```

### 1.2 로그인
- **Username**: `keti`
- **Password**: `keti1234!`

> 만약 로그인이 안 되면:
> ```bash
> sudo ./reset_grafana_simple.sh
> ```

---

## Step 2: Redis Data Source 플러그인 설치

### 방법 A: Docker로 설치 (추천)

```bash
# Grafana 컨테이너 이름 확인
GRAFANA_CONTAINER=$(sudo docker ps --format "{{.Names}}" | grep -i grafana | head -1)
echo "Container: $GRAFANA_CONTAINER"

# Redis 플러그인 설치
sudo docker exec $GRAFANA_CONTAINER grafana-cli plugins install redis-datasource

# Grafana 재시작
sudo docker restart $GRAFANA_CONTAINER
```

재시작 후 약 10초 기다렸다가 다시 http://localhost:3000 접속

### 방법 B: Grafana UI에서 설치

1. 좌측 메뉴 → **Administration** (톱니바퀴 아이콘) → **Plugins**
2. 검색창에 **"Redis"** 입력
3. **"Redis Data Source"** 클릭
4. **Install** 버튼 클릭
5. 설치 완료 후 Grafana 재시작 필요

---

## Step 3: Redis 데이터소스 추가

### 3.1 Data Sources 메뉴 열기

1. 좌측 메뉴 → **Connections** (플러그 아이콘)
2. **Data sources** 클릭
3. 우측 상단 **"Add data source"** 버튼 클릭

### 3.2 Redis 선택

- 검색창에 **"Redis"** 입력
- **"Redis Data Source"** 선택 (빨간색 Redis 로고)

### 3.3 Redis 설정 입력

**기본 설정:**

| 항목 | 값 |
|------|-----|
| **Name** | `iSyncWave Redis` (또는 원하는 이름) |
| **Address** | `localhost:6379` |
| **Timeout** | `10` (기본값) |

**고급 설정 (선택사항):**

- **Database**: `0` (기본값)
- **Password**: (비워둠, Redis에 비밀번호가 없다면)
- **TLS/SSL**: 비활성화

### 3.4 연결 테스트

1. 페이지 하단의 **"Save & test"** 버튼 클릭
2. ✅ **"Data source is working"** 메시지 확인

> ⚠️ 만약 에러가 발생하면:
> - Redis가 실행 중인지 확인: `redis-cli ping`
> - Address를 `localhost:6379` 대신 `127.0.0.1:6379`로 시도
> - Docker 네트워크 이슈라면: `host.docker.internal:6379` 시도

---

## Step 4: 대시보드 Import

### 4.1 Dashboards 메뉴 열기

1. 좌측 메뉴 → **Dashboards** (네모 4개 아이콘)
2. 우측 상단 **"New"** 버튼 → **"Import"** 선택

### 4.2 JSON 파일 업로드

1. **"Upload JSON file"** 버튼 클릭
2. 파일 선택: `/home/keti/cwkim/KETI-AI/iSyncWave/grafana_dashboard.json`
3. **"Load"** 버튼 클릭

### 4.3 데이터소스 선택

- **"Select a data source"** 드롭다운에서
- **"iSyncWave Redis"** (Step 3에서 만든 이름) 선택

### 4.4 Import 완료

- 우측 하단 **"Import"** 버튼 클릭
- 자동으로 대시보드가 열림

---

## Step 5: 대시보드 확인

### 5.1 자동 새로고침 설정

- 우측 상단에서 **Auto-refresh** 설정
- 추천: **"5s"** (5초마다 자동 새로고침)

### 5.2 시간 범위 설정

- 우측 상단 시계 아이콘 클릭
- **"Last 5 minutes"** 또는 **"Last 15 minutes"** 선택

### 5.3 대시보드 패널 확인

대시보드에는 다음이 표시되어야 합니다:

1. **Stream Metadata** (상단)
   - Stream name: iSyncWave-Android-F6EE
   - Channels: 19
   - Sampling Rate: 250.0

2. **Latest EEG Gauges** (중간)
   - Fp1, Fp2, Cz, O1 채널의 실시간 값
   - 게이지 바로 표시

3. **EEG Time Series** (하단)
   - 모든 채널의 시계열 그래프
   - 최근 100개 샘플

4. **Statistics** (최하단)
   - Sampling Rate
   - Total Samples
   - Recording Duration

---

## 실시간 데이터 확인

### 데이터가 보이지 않으면?

#### 체크리스트:

```bash
# 1. LSL 데이터 수신 중인지 확인
ps aux | grep save_lsl_to_csv

# 2. Redis에 데이터가 있는지 확인
python3 view_redis_data.py latest

# 3. Redis 연결 확인
redis-cli ping
```

#### Grafana 대시보드에서:

1. **패널 Edit 하기**
   - 패널 제목 클릭 → **"Edit"**
   - Query 탭에서 쿼리 확인

2. **Redis 명령어 확인**
   - 예: `HGET isyncwave:eeg:latest Fp1`
   - Query Inspector로 실제 쿼리 확인

3. **데이터소스 재테스트**
   - Connections → Data sources → iSyncWave Redis
   - **"Save & test"** 다시 클릭

---

## 대시보드 커스터마이징

### 새 패널 추가하기

1. 대시보드 상단 **"Add"** → **"Visualization"**
2. Data source: **iSyncWave Redis** 선택
3. Query 입력 예시:

**최신 데이터 조회:**
```
HGET isyncwave:eeg:latest Channel_5
```

**히스토리 조회:**
```
XREVRANGE isyncwave:eeg:stream + - COUNT 50
```

**메타데이터 조회:**
```
HGETALL isyncwave:eeg:meta
```

### 패널 타입 변경

- **Stat**: 단일 숫자 값
- **Gauge**: 게이지 바
- **Time series**: 시계열 그래프
- **Table**: 테이블 형식

---

## 트러블슈팅

### ❌ "Data source is not working"

**원인**: Redis 연결 실패

**해결**:
```bash
# Redis 실행 확인
redis-cli ping

# Redis 재시작
redis-server --daemonize yes

# 데이터 확인
redis-cli
> KEYS isyncwave*
> HGETALL isyncwave:eeg:latest
> EXIT
```

### ❌ "No data" 또는 빈 패널

**원인**: 데이터가 Redis에 없음

**해결**:
```bash
# 데이터 수신 다시 시작
python3 save_lsl_to_csv.py -d 30

# 다른 터미널에서 실시간 확인
python3 view_redis_data.py monitor
```

### ❌ "Plugin not found"

**원인**: Redis 플러그인 미설치

**해결**:
```bash
GRAFANA_CONTAINER=$(sudo docker ps --format "{{.Names}}" | grep -i grafana)
sudo docker exec $GRAFANA_CONTAINER grafana-cli plugins install redis-datasource
sudo docker restart $GRAFANA_CONTAINER
```

### ❌ 패널이 "Pending" 상태로 계속 로딩

**원인**: Query 문법 오류 또는 타임아웃

**해결**:
1. 패널 Edit → Query Inspector
2. 에러 메시지 확인
3. Query 문법 수정
4. Timeout 값 증가

### ❌ 그래프에 데이터가 선으로 안 그려짐

**원인**: Time series 데이터 형식 문제

**해결**:
1. Transform 탭 → "Extract fields" 추가
2. Source: "Auto"
3. Format: "Time series"

---

## 빠른 명령어 모음

```bash
# 1. Redis 플러그인 설치 및 재시작
GRAFANA_CONTAINER=$(sudo docker ps --format "{{.Names}}" | grep -i grafana | head -1)
sudo docker exec $GRAFANA_CONTAINER grafana-cli plugins install redis-datasource
sudo docker restart $GRAFANA_CONTAINER

# 2. 데이터 수신 시작
python3 save_lsl_to_csv.py

# 3. 실시간 모니터링
python3 view_redis_data.py monitor

# 4. Redis 데이터 확인
redis-cli
> HGETALL isyncwave:eeg:latest
> XLEN isyncwave:eeg:stream
> EXIT
```

---

## 완성된 대시보드 예시

### URL:
```
http://localhost:3000/d/isyncwave-eeg
```

### 포함된 패널:
1. **Stream Metadata** - 스트림 정보
2. **Fp1 Gauge** - 전두엽 좌측 실시간 값
3. **Fp2 Gauge** - 전두엽 우측 실시간 값
4. **Cz Gauge** - 중앙 실시간 값
5. **O1 Gauge** - 후두엽 좌측 실시간 값
6. **EEG Time Series** - 모든 채널 그래프
7. **Sampling Rate** - 샘플링 레이트
8. **Total Samples** - 총 샘플 수
9. **Recording Duration** - 기록 시간

### 자동 새로고침:
- 5초마다 자동 업데이트
- 실시간 데이터 스트리밍

---

## 다음 단계

✅ 대시보드 커스터마이징
✅ 알림(Alert) 설정
✅ 여러 장치 모니터링
✅ 데이터 분석 및 시각화

**완료!** 🎉

이제 http://localhost:3000 에서 실시간 EEG 데이터를 확인하세요!
