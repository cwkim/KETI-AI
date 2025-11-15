# Grafana 빠른 테스트 - 5분 안에 작동 확인

가장 간단한 테스트로 Redis 연결 확인하기

---

## 1분 테스트 - Redis 연결 확인

### Step 1: 데이터 수신 중인지 확인

```bash
# 데이터 수신 (백그라운드)
python3 save_lsl_to_csv.py &

# 5초 후 데이터 확인
sleep 5
redis-cli HGET isyncwave:eeg:meta stream_name
```

출력 예: `iSyncWave-Android-F6EE`

---

## 2분 테스트 - Grafana 첫 패널 만들기

### Step 1: Grafana 접속
```
http://localhost:3000
keti / keti1234!
```

### Step 2: 새 대시보드
1. 좌측 **Dashboards** 메뉴
2. **"New"** → **"New dashboard"**
3. **"Add visualization"**

### Step 3: 가장 간단한 패널
1. Data source: **iSyncWave Redis**

2. **Query 입력** (3개 필드):
   ```
   Command: hget
   Key: isyncwave:eeg:meta
   Field: stream_name
   ```

3. **Apply** 클릭

### 결과
✅ `iSyncWave-Android-F6EE` 값이 보이면 성공!

---

## 3분 테스트 - 실시간 데이터

### 다음 패널 추가

1. **"Add"** → **"Visualization"**

2. Data source: **iSyncWave Redis**

3. **Query**:
   ```
   Command: hget
   Key: isyncwave:eeg:latest
   Field: Channel_1
   ```

4. Visualization: **Stat**

5. **Apply**

### 자동 새로고침 설정
- 우측 상단 시계 옆 → **"5s"** 선택

### 결과
✅ 숫자가 5초마다 바뀌면 성공!

---

## 5분 테스트 - 완전한 대시보드

### 패널 3: 모든 채널 보기

1. **"Add"** → **"Visualization"**

2. Query:
   ```
   Command: hgetall
   Key: isyncwave:eeg:latest
   ```

3. Visualization: **Table**

4. **Apply**

### 패널 4: 샘플링 레이트

1. **"Add"** → **"Visualization"**

2. Query:
   ```
   Command: hget
   Key: isyncwave:eeg:meta
   Field: sampling_rate
   ```

3. Visualization: **Stat**
4. Unit: **hertz (Hz)**
5. **Apply**

### 대시보드 저장
- 우측 상단 **Save** 아이콘
- Name: `iSyncWave Test`
- **Save**

---

## 트러블슈팅 - 30초 진단

### ❌ "No data source found"

**해결**:
```bash
# Redis 플러그인 설치
sudo ./install_redis_plugin.sh
```

데이터소스 추가:
- Connections → Add data source → Redis
- Address: `localhost:6379`
- Save & test

### ❌ "No data"

**진단**:
```bash
# 1. LSL 수신 중?
ps aux | grep save_lsl_to_csv

# 2. Redis에 데이터?
redis-cli HGET isyncwave:eeg:latest Channel_1

# 3. 데이터 다시 수신
pkill -f save_lsl_to_csv
python3 save_lsl_to_csv.py -d 30
```

### ❌ 값이 안 바뀜

**Auto-refresh 켜기**:
- 우측 상단 → "5s" 선택

**데이터 업데이트 확인**:
```bash
# 터미널에서
watch -n 1 'redis-cli HGET isyncwave:eeg:latest Channel_1'
```

---

## 성공 체크리스트

| 항목 | 확인 |
|------|------|
| ✅ Redis 플러그인 설치됨 | `sudo ./install_redis_plugin.sh` |
| ✅ 데이터소스 추가됨 | "Save & test" 성공 |
| ✅ LSL 데이터 수신 중 | `python3 save_lsl_to_csv.py` 실행 중 |
| ✅ Redis에 데이터 있음 | `redis-cli HGET isyncwave:eeg:latest Channel_1` 값 나옴 |
| ✅ 패널에 값 표시됨 | 숫자 또는 텍스트 보임 |
| ✅ Auto-refresh 작동 | 5초마다 값 변함 |

---

## 다음 단계

✅ 더 많은 패널 추가 → **GRAFANA_MANUAL_SETUP.md** 참고

✅ 대시보드 커스터마이징

✅ 알림 설정

**성공!** 🎉 이제 실시간 모니터링이 가능합니다!
