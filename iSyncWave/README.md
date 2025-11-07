# iSyncWave LSL 데이터 수신 도구

iSyncWave 장치에서 전송되는 LSL(Lab Streaming Layer) 뇌파 데이터를 수신하고 확인하는 도구입니다.

## 설치 완료된 구성 요소

- ✅ pylsl Python 라이브러리
- ✅ liblsl 네이티브 라이브러리 (소스에서 빌드)
- ✅ LSL 스트림 검색 스크립트
- ✅ LSL 데이터 수신 스크립트
- ✅ LSL 연속 모니터링 스크립트

## 네트워크 정보

현재 장치는 다음 네트워크에 연결되어 있습니다:
- **wlan0**: 192.168.0.5 (주요 Wi-Fi 인터페이스)

## 사용 방법

### 1. LSL 스트림 검색 (1회 검색)

```bash
./lsl_discover.sh
```

네트워크에서 사용 가능한 모든 LSL 스트림을 검색합니다 (5초 대기).

### 2. LSL 데이터 수신

```bash
# 기본 사용 (10초 동안 데이터 수신)
./lsl_receive.sh

# 30초 동안 데이터 수신
./lsl_receive.sh -d 30

# 무한정 수신 (Ctrl+C로 종료)
./lsl_receive.sh -d 0

# 개별 샘플 출력 없이 통계만 표시
./lsl_receive.sh -q

# 특정 스트림 이름으로 연결
./lsl_receive.sh -n "iSyncWave"
```

### 3. LSL 연속 모니터링 (권장)

```bash
./lsl_monitor.sh
```

이 모드는:
- 네트워크에서 LSL 스트림을 계속 검색합니다
- 스트림이 발견되면 자동으로 연결하고 데이터를 수신합니다
- 태블릿 앱을 시작하기 전에 실행해두면 편리합니다
- Ctrl+C를 눌러 종료할 수 있습니다

## 태블릿 앱 확인 사항

LSL 데이터를 수신하려면 태블릿에서 다음을 확인하세요:

1. **iSyncWave 장치 전원 ON**: 장치가 켜져 있고 태블릿과 연결되어 있는지 확인
2. **태블릿 앱 실행**: iSyncWave 앱이 실행 중이어야 합니다
3. **LSL 스트리밍 활성화**: 앱에서 LSL 스트리밍 기능이 켜져 있는지 확인
4. **동일 네트워크**: 태블릿과 이 장치가 같은 Wi-Fi 네트워크에 연결되어 있는지 확인
   - 이 장치: 192.168.0.5
   - 태블릿도 192.168.0.x 대역에 있어야 합니다

## 트러블슈팅

### "No LSL streams found" 메시지가 나올 때

1. **태블릿 앱 확인**
   - 앱이 실행 중인지 확인
   - LSL 스트리밍이 활성화되어 있는지 확인

2. **네트워크 확인**
   ```bash
   # 태블릿 IP 주소 확인 (태블릿에서)
   # 설정 > Wi-Fi > 연결된 네트워크 정보

   # 이 장치에서 태블릿으로 ping 테스트
   ping <태블릿_IP>
   ```

3. **방화벽 확인**
   - LSL은 UDP 멀티캐스트를 사용합니다
   - 필요시 방화벽 규칙 확인

### 스트림은 보이지만 데이터가 안 올 때

- 앱을 재시작해보세요
- 장치를 재부팅해보세요
- 태블릿을 재부팅해보세요

## 출력 예시

### 스트림 발견 시

```
============================================================
LSL Stream Monitor for iSyncWave
============================================================

Continuously monitoring for LSL streams...

✓ Found 1 stream(s)!

  Stream 1:
    Name: iSyncWave_EEG
    Type: EEG
    Channels: 8
    Sampling Rate: 250.0 Hz
    Source: iSyncWave-12345

🔗 Connecting to 'iSyncWave_EEG'...
✓ Connected! Receiving data...

[18:00:01.123] [   123.45,   234.56,   345.67, ... ]
[18:00:01.127] [   123.78,   234.89,   346.01, ... ]
```

## 기술 정보

- **LSL 프로토콜 버전**: 1.16.2
- **Python 버전**: 3.8.10
- **플랫폼**: Linux ARM64 (aarch64)
- **네트워크 프로토콜**: UDP 멀티캐스트
- **기본 포트**: LSL 자동 할당

## 파일 설명

- `lsl_discover.sh` - LSL 스트림 검색 스크립트
- `lsl_receive.sh` - LSL 데이터 수신 스크립트
- `lsl_monitor.sh` - LSL 연속 모니터링 스크립트
- `discover_lsl_streams.py` - Python 스트림 검색 구현
- `receive_lsl_data.py` - Python 데이터 수신 구현
- `monitor_lsl.py` - Python 연속 모니터링 구현
