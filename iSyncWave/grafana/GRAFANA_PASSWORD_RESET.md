# Grafana 비밀번호 재설정 가이드

Grafana 계정을 keti / keti1234! 로 설정하는 방법

## 방법 1: 자동 스크립트 사용 (추천)

```bash
cd /home/keti/cwkim/KETI-AI/iSyncWave
sudo ./reset_grafana_password.sh
```

스크립트 실행 후:
1. 옵션 선택
   - 1번: admin 계정 비밀번호를 'keti1234!'로 변경
   - 2번: 새로운 'keti' 계정 생성 (비밀번호: keti1234!)
2. 완료!

## 방법 2: 수동 명령어 (Docker)

### Docker 컨테이너 이름 확인
```bash
sudo docker ps | grep grafana
```

### 옵션 A: admin 비밀번호 재설정
```bash
sudo docker exec -it <container_name> grafana-cli admin reset-admin-password 'keti1234!'
```

그런 다음 로그인:
- Username: `admin`
- Password: `keti1234!`

### 옵션 B: 새 keti 계정 생성
```bash
sudo docker exec -it <container_name> grafana-cli admin create --admin-user keti --admin-password 'keti1234!'
```

그런 다음 로그인:
- Username: `keti`
- Password: `keti1234!`

## 방법 3: Grafana API 사용 (현재 admin 계정에 접근 가능한 경우)

```bash
# 현재 admin 계정 비밀번호를 알고 있다면
curl -X PUT http://localhost:3000/api/user/password \
  -u admin:current_password \
  -H "Content-Type: application/json" \
  -d '{"oldPassword":"current_password","newPassword":"keti1234!","confirmNew":"keti1234!"}'
```

## 방법 4: Grafana 설정 파일로 기본 admin 계정 활성화

Grafana 설정 파일을 수정하여 기본 admin 계정을 다시 활성화:

```bash
# 설정 파일 편집 (Docker 컨테이너 내부)
sudo docker exec -it <container_name> sh -c "echo 'admin_user = keti' >> /etc/grafana/grafana.ini"
sudo docker exec -it <container_name> sh -c "echo 'admin_password = keti1234!' >> /etc/grafana/grafana.ini"

# Grafana 재시작
sudo docker restart <container_name>
```

## 빠른 명령어 (복사해서 실행)

### 1단계: Docker 컨테이너 이름 확인
```bash
GRAFANA_CONTAINER=$(sudo docker ps --format "{{.Names}}" | grep -i grafana | head -1)
echo "Grafana container: $GRAFANA_CONTAINER"
```

### 2단계: 새 keti 계정 생성
```bash
sudo docker exec -it $GRAFANA_CONTAINER grafana-cli admin create --admin-user keti --admin-password 'keti1234!'
```

또는 admin 비밀번호만 변경:
```bash
sudo docker exec -it $GRAFANA_CONTAINER grafana-cli admin reset-admin-password 'keti1234!'
```

### 3단계: Grafana 로그인
```
URL: http://localhost:3000
Username: keti (또는 admin)
Password: keti1234!
```

## 트러블슈팅

### ❌ "User already exists" 오류

이미 keti 계정이 있다면 비밀번호만 재설정:

```bash
# SQLite 데이터베이스에서 직접 수정 (고급)
sudo docker exec -it $GRAFANA_CONTAINER sqlite3 /var/lib/grafana/grafana.db "UPDATE user SET password = '', salt = '' WHERE login = 'keti';"
```

그런 다음 Grafana를 재시작하면 비밀번호가 초기화됩니다.

### ❌ Docker 권한 오류

```bash
# 현재 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER

# 로그아웃 후 다시 로그인하거나
newgrp docker
```

### ❌ Container를 찾을 수 없음

Grafana가 Docker가 아닌 다른 방식으로 실행 중일 수 있습니다:

```bash
# 시스템 서비스로 실행 중인 경우
sudo systemctl status grafana-server

# grafana-cli 직접 사용
grafana-cli admin reset-admin-password 'keti1234!'
```

## 성공 확인

비밀번호 변경 후 API로 확인:

```bash
curl -s http://localhost:3000/api/admin/settings -u keti:keti1234! | head -20
```

또는:

```bash
curl -s http://localhost:3000/api/admin/settings -u admin:keti1234! | head -20
```

정상 응답이 오면 성공입니다!

## 요약

**가장 쉬운 방법:**
```bash
sudo ./reset_grafana_password.sh
```

**수동으로 하려면:**
```bash
GRAFANA_CONTAINER=$(sudo docker ps --format "{{.Names}}" | grep -i grafana | head -1)
sudo docker exec -it $GRAFANA_CONTAINER grafana-cli admin create --admin-user keti --admin-password 'keti1234!'
```

**로그인:**
- http://localhost:3000
- keti / keti1234!
