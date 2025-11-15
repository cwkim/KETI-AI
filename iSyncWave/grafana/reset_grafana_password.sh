#!/bin/bash

# Grafana Password Reset Script
# Reset Grafana admin password or create new user

echo "===================================="
echo "Grafana Password Reset Tool"
echo "===================================="
echo ""

# Check if running as root/sudo
if [ "$EUID" -eq 0 ]; then
    SUDO_CMD=""
else
    SUDO_CMD="sudo"
fi

# Find Docker container
echo "Finding Grafana Docker container..."
CONTAINER=$($SUDO_CMD docker ps --format "{{.Names}}" 2>/dev/null | grep -i grafana | head -1)

if [ -z "$CONTAINER" ]; then
    echo "❌ Could not find Grafana Docker container"
    echo ""
    echo "Alternative methods:"
    echo "1. Use Grafana UI password reset (if you have access to the email)"
    echo "2. Manually reset via Docker:"
    echo "   sudo docker exec -it <container_name> grafana-cli admin reset-admin-password <new_password>"
    echo ""
    echo "3. Create new admin user via Docker:"
    echo "   sudo docker exec -it <container_name> grafana-cli admin create --admin-user keti --admin-password 'keti1234!'"
    exit 1
fi

echo "✓ Found container: $CONTAINER"
echo ""

# Menu
echo "Select an option:"
echo "1. Reset admin password to 'keti1234!'"
echo "2. Create new admin user 'keti' with password 'keti1234!'"
echo "3. Custom password reset"
echo ""
read -p "Enter option (1-3): " OPTION

case $OPTION in
    1)
        echo ""
        echo "Resetting admin password..."
        $SUDO_CMD docker exec -it $CONTAINER grafana-cli admin reset-admin-password 'keti1234!'
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Password reset successfully!"
            echo "  Username: admin"
            echo "  Password: keti1234!"
        fi
        ;;
    2)
        echo ""
        echo "Creating new admin user 'keti'..."
        $SUDO_CMD docker exec -it $CONTAINER grafana-cli admin create --admin-user keti --admin-password 'keti1234!'
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ User created successfully!"
            echo "  Username: keti"
            echo "  Password: keti1234!"
        fi
        ;;
    3)
        echo ""
        read -p "Enter new username: " NEW_USER
        read -s -p "Enter new password: " NEW_PASS
        echo ""
        echo "Creating user..."
        $SUDO_CMD docker exec -it $CONTAINER grafana-cli admin create --admin-user "$NEW_USER" --admin-password "$NEW_PASS"
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ User created successfully!"
            echo "  Username: $NEW_USER"
        fi
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "You can now login to Grafana at http://localhost:3000"
