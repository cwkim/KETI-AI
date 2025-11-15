# Grafana 대시보드 수동 생성 가이드 (100% 작동 보장)

JSON import가 안 될 때 - 수동으로 패널 만들기

---

## 사전 준비

### 1. 데이터 수신 중인지 확인

**터미널 1 - 계속 실행 중:**
```bash
python3 save_lsl_to_csv.py
```

**터미널 2 - 데이터 확인:**
```bash
python3 view_redis_data.py monitor
```

데이터가 계속 업데이트되는 것을 확인!

---

## Step 1: 새 대시보드 만들기

1. Grafana 로그인: http://localhost:3000 (keti / keti1234!)

2. 좌측 메뉴 → **Dashboards** (네모 4개)

3. 우측 상단 **"New"** → **"New dashboard"**

4. **"Add visualization"** 클릭

---

## Step 2: 첫 번째 패널 - Channel_1 실시간 값

### 2.1 데이터소스 선택
- **"Select data source"** → **"iSyncWave Redis"** 선택

### 2.2 Query 입력

**Query 섹션에서:**

1. **Command** 필드에 입력:
   ```
   hget
   ```

2. **Key** 필드에 입력:
   ```
   isyncwave:eeg:latest
   ```

3. **Field** 필드에 입력:
   ```
   Channel_1
   ```

### 2.3 Visualization 설정

1. 우측 **Panel options**:
   - **Title**: `Channel 1 - 실시간`

2. 우측 상단 **Visualization** 드롭다운:
   - **"Stat"** 선택 (큰 숫자로 표시)

3. **Field** 탭 (우측 하단):
   - **Unit**: `none` 또는 `short`
   - **Decimals**: `2` (소수점 2자리)

### 2.4 저장
- 우측 상단 **"Apply"** 버튼 클릭

---

## Step 3: 두 번째 패널 - 샘플링 레이트

### 3.1 패널 추가
- 상단 **"Add"** → **"Visualization"**

### 3.2 Query 설정

1. Data source: **iSyncWave Redis**

2. **Command**: `hget`

3. **Key**: `isyncwave:eeg:meta`

4. **Field**: `sampling_rate`

### 3.3 Visualization
- **Type**: Stat
- **Title**: `Sampling Rate`
- **Unit**: `hertz (Hz)`

### 3.4 저장
- **"Apply"** 클릭

---

## Step 4: 세 번째 패널 - 여러 채널 Table

### 4.1 패널 추가
- **"Add"** → **"Visualization"**

### 4.2 Query 설정

1. Data source: **iSyncWave Redis**

2. **Command**: `hgetall`

3. **Key**: `isyncwave:eeg:latest`

### 4.3 Visualization
- **Type**: Table
- **Title**: `모든 채널 - 최신 값`

### 4.4 Transform 추가 (중요!)

1. 우측 상단 **"Transform"** 탭 클릭

2. **"Add transformation"** 클릭

3. **"Organize fields by name"** 선택

4. 원하는 필드만 표시:
   - timestamp ✓
   - datetime ✓
   - Channel_1 ✓
   - Channel_2 ✓
   - Channel_3 ✓
   - ... (원하는 채널 선택)

### 4.5 저장
- **"Apply"** 클릭

---

## Step 5: 네 번째 패널 - Gauge (게이지 바)

### 5.1 패널 추가
- **"Add"** → **"Visualization"**

### 5.2 Query
- **Command**: `hget`
- **Key**: `isyncwave:eeg:latest`
- **Field**: `Channel_1` (또는 원하는 채널)

### 5.3 Visualization
1. **Type**: Gauge

2. **Field** 설정:
   - **Min**: `-100` (또는 데이터 범위에 맞게)
   - **Max**: `100`
   - **Unit**: `none`
   - **Decimals**: `2`

3. **Title**: `Channel 1 Gauge`

### 5.4 저장
- **"Apply"** 클릭

---

## Step 6: 대시보드 설정

### 6.1 Auto-refresh 설정

1. 우측 상단 **시계 아이콘** 옆 드롭다운

2. **"5s"** 선택 (5초마다 자동 새로고침)

### 6.2 대시보드 저장

1. 우측 상단 **"Save dashboard"** (디스켓 아이콘)

2. **Dashboard name**: `iSyncWave EEG Real-time`

3. **"Save"** 클릭

---

## 추가 패널 타입

### 1. Time Series (시계열 그래프)

**문제**: Redis Stream 데이터를 Time Series로 보려면 복잡한 처리 필요

**간단한 방법**: Stat이나 Gauge를 여러 개 만들어서 배치

### 2. Bar Gauge (막대 그래프)

- Visualization Type: **Bar gauge**
- Orientation: **Horizontal** 또는 **Vertical**
- 여러 채널을 한 번에 비교할 때 유용

---

## 패널 배치 및 크기 조정

### 패널 이동
- 패널 제목을 **드래그**해서 위치 이동

### 패널 크기 조정
- 패널 우측 하단 모서리를 **드래그**해서 크기 조정

### 권장 레이아웃

```
┌─────────────────────────────────┐
│  Sampling Rate  │ Total Samples  │
├─────────────────┼────────────────┤
│   Ch1 Gauge     │   Ch2 Gauge    │
├─────────────────┴────────────────┤
│     모든 채널 Table (넓게)       │
└───────────────────────────────────┘
```

---

## 여러 채널을 한 번에 보는 방법

### 방법 1: Table 패널 사용 (추천)

위의 Step 4처럼 `HGETALL isyncwave:eeg:latest` 사용

### 방법 2: 각 채널마다 패널 만들기

- Channel_1, Channel_2, Channel_3... 각각 Stat 패널 생성
- 그리드로 배치 (3x7 또는 4x5 등)

### 방법 3: Multiple queries in one panel

1. 패널 Edit 모드에서
2. **"Add query"** 버튼 클릭
3. 각 Query마다 다른 채널 지정:
   - Query A: Channel_1
   - Query B: Channel_2
   - Query C: Channel_3
   - ...

---

## 실시간 확인

### 데이터가 업데이트되는지 확인

1. **Auto-refresh** 켜짐: 우측 상단에 "5s" 표시

2. **값이 변하는지 확인**: 패널의 숫자가 5초마다 바뀌어야 함

3. **안 바뀌면**:
   ```bash
   # 터미널에서 데이터 수신 확인
   python3 view_redis_data.py monitor
   ```

---

## 트러블슈팅

### ❌ "No data" 표시

**원인 1**: Query 오타
- Command, Key, Field 철자 정확히 확인
- 대소문자 구분함!

**원인 2**: 데이터가 Redis에 없음
```bash
redis-cli HGET isyncwave:eeg:latest Channel_1
```

**원인 3**: 데이터소스 연결 안 됨
- Connections → Data sources → iSyncWave Redis
- "Save & test" 다시 클릭

### ❌ 패널이 로딩 중

**Timeout 증가**:
1. 패널 Edit → Query options
2. **Timeout**: `10000` (10초)

### ❌ 값이 안 바뀜

**Auto-refresh 확인**:
- 우측 상단 시계 옆 드롭다운 → "5s" 선택되어 있는지 확인

---

## Redis 명령어 치트시트

Grafana Query에서 사용 가능한 Redis 명령어:

| 명령어 | Key | Field | 설명 |
|--------|-----|-------|------|
| `hget` | `isyncwave:eeg:latest` | `Channel_1` | 채널 1 최신 값 |
| `hget` | `isyncwave:eeg:meta` | `sampling_rate` | 샘플링 레이트 |
| `hget` | `isyncwave:eeg:meta` | `total_samples` | 총 샘플 수 |
| `hgetall` | `isyncwave:eeg:latest` | (비움) | 모든 채널 최신 값 |
| `hgetall` | `isyncwave:eeg:meta` | (비움) | 모든 메타데이터 |
| `xlen` | `isyncwave:eeg:stream` | (비움) | Stream 총 개수 |

---

## 빠른 테스트 패널

가장 간단한 테스트:

1. **Add visualization**
2. Data source: **iSyncWave Redis**
3. Query:
   - **Command**: `hget`
   - **Key**: `isyncwave:eeg:meta`
   - **Field**: `stream_name`
4. Visualization: **Stat**
5. Title: `Test - Stream Name`
6. **Apply**

이게 작동하면 Redis 연결은 정상입니다!

---

## 완성 예시

### 최소 구성 (3개 패널)

1. **패널 1**: Channel_1 실시간 (Stat)
2. **패널 2**: Sampling Rate (Stat)
3. **패널 3**: 모든 채널 (Table)

### 권장 구성 (7개 패널)

1. **Stream Name** (Stat)
2. **Sampling Rate** (Stat)
3. **Total Samples** (Stat)
4. **Channel_1** (Gauge)
5. **Channel_2** (Gauge)
6. **Channel_3** (Gauge)
7. **All Channels** (Table)

---

## 저장 및 공유

### 대시보드 저장
- 우측 상단 **"Save dashboard"** 아이콘
- 이름 입력 후 **"Save"**

### URL 공유
- 저장 후 URL 복사
- 예: `http://localhost:3000/d/abc123/isyncwave-eeg-real-time`

---

## 다음 단계

✅ 패널 커스터마이징 (색상, 임계값 등)
✅ 알림 설정
✅ 여러 대시보드 만들기
✅ 스냅샷 저장

**완료!** 🎉

수동으로 만든 대시보드가 훨씬 안정적입니다!
