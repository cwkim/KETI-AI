#!/bin/bash

# Grafana Redis Plugin 설치 스크립트

echo "=========================================="
echo "Grafana Redis Data Source 플러그인 설치"
echo "=========================================="
echo ""

# Find Grafana container
echo "Grafana 컨테이너 찾는 중..."
CONTAINER=$(sudo docker ps --format "{{.Names}}" | grep -i grafana | head -1)

if [ -z "$CONTAINER" ]; then
    echo "❌ Grafana 컨테이너를 찾을 수 없습니다."
    echo ""
    echo "수동으로 설치하려면:"
    echo "  1. Grafana 웹 접속: http://localhost:3000"
    echo "  2. Administration → Plugins"
    echo "  3. 'Redis' 검색 후 설치"
    exit 1
fi

echo "✓ 컨테이너 발견: $CONTAINER"
echo ""

# Check if plugin already installed
echo "기존 플러그인 확인 중..."
PLUGIN_CHECK=$(sudo docker exec $CONTAINER grafana-cli plugins ls 2>/dev/null | grep redis-datasource || echo "")

if [ ! -z "$PLUGIN_CHECK" ]; then
    echo "⚠ Redis 플러그인이 이미 설치되어 있습니다."
    echo ""
    read -p "재설치하시겠습니까? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "설치를 취소했습니다."
        exit 0
    fi
fi

# Install plugin
echo ""
echo "Redis Data Source 플러그인 설치 중..."
sudo docker exec $CONTAINER grafana-cli plugins install redis-datasource

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 플러그인 설치 실패"
    exit 1
fi

echo ""
echo "✓ 플러그인 설치 완료"
echo ""

# Restart Grafana
echo "Grafana 재시작 중..."
sudo docker restart $CONTAINER

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Grafana 재시작 실패"
    exit 1
fi

echo ""
echo "✓ Grafana 재시작 완료"
echo ""
echo "=========================================="
echo "설치 완료!"
echo "=========================================="
echo ""
echo "다음 단계:"
echo "  1. 10초 정도 기다린 후 Grafana 접속"
echo "  2. http://localhost:3000"
echo "  3. keti / keti1234! 로 로그인"
echo "  4. Connections → Data sources → Add data source"
echo "  5. Redis 선택 후 설정"
echo ""
echo "자세한 가이드: GRAFANA_SETUP_STEP_BY_STEP.md"
echo ""
