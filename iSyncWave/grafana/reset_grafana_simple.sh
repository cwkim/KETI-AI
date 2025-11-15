#!/bin/bash

# Simple Grafana Password Reset
# Sets credentials to keti / keti1234!

echo "=========================================="
echo "Grafana Password Reset"
echo "Setting up: keti / keti1234!"
echo "=========================================="
echo ""

# Find Grafana container
echo "Finding Grafana container..."
CONTAINER=$(sudo docker ps --format "{{.Names}}" | grep -i grafana | head -1)

if [ -z "$CONTAINER" ]; then
    echo "❌ Could not find Grafana container"
    echo ""
    echo "Run this manually:"
    echo "  sudo docker ps | grep grafana"
    echo "  sudo docker exec -it <container_name> grafana-cli admin create --admin-user keti --admin-password 'keti1234!'"
    exit 1
fi

echo "✓ Found: $CONTAINER"
echo ""
echo "Creating keti account..."

# Try to create keti user
sudo docker exec $CONTAINER grafana-cli admin create --admin-user keti --admin-password 'keti1234!'

if [ $? -eq 0 ]; then
    echo ""
    echo "✓✓✓ SUCCESS! ✓✓✓"
    echo ""
    echo "Grafana Login:"
    echo "  URL: http://localhost:3000"
    echo "  Username: keti"
    echo "  Password: keti1234!"
    echo ""
else
    echo ""
    echo "⚠ User might already exist. Trying password reset for admin..."
    sudo docker exec $CONTAINER grafana-cli admin reset-admin-password 'keti1234!'

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Admin password reset!"
        echo ""
        echo "Grafana Login:"
        echo "  URL: http://localhost:3000"
        echo "  Username: admin"
        echo "  Password: keti1234!"
        echo ""
        echo "Note: You can create 'keti' user from Grafana UI"
    else
        echo ""
        echo "❌ Failed. Please check the error message above."
    fi
fi
