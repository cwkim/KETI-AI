#!/bin/bash

# LSL to Redis Streamer Script
# Convenient wrapper for lsl_to_redis.py

echo "==================================="
echo "iSyncWave LSL to Redis Streamer"
echo "==================================="
echo ""

# Default values
DURATION=0  # 0 means infinite
OUTPUT_DIR="data"

# Show help
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    python3 lsl_to_redis.py --help
    exit 0
fi

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -dir|--directory)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--name)
            STREAM_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python3 lsl_to_redis.py"

if [ ! -z "$STREAM_NAME" ]; then
    CMD="$CMD -n \"$STREAM_NAME\""
fi

if [ "$DURATION" != "0" ]; then
    CMD="$CMD -d $DURATION"
fi

if [ ! -z "$OUTPUT" ]; then
    CMD="$CMD -o \"$OUTPUT\""
fi

CMD="$CMD -dir \"$OUTPUT_DIR\""

# Execute
echo "Executing: $CMD"
echo ""
eval $CMD
