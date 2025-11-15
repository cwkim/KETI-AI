#!/usr/bin/env python3
"""
Redis Data Viewer for iSyncWave EEG Data
View real-time and historical EEG data stored in Redis
"""

import redis
import json
import time
from datetime import datetime
import argparse

def view_latest_data(redis_host='localhost', redis_port=6379, redis_db=0):
    """
    View the latest EEG data from Redis
    """
    try:
        r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        r.ping()

        print("=" * 80)
        print("Latest EEG Data from Redis")
        print("=" * 80)

        # Get metadata
        meta = r.hgetall('isyncwave:eeg:meta')
        if meta:
            print("\nStream Information:")
            print(f"  Name: {meta.get('stream_name', 'N/A')}")
            print(f"  Type: {meta.get('stream_type', 'N/A')}")
            print(f"  Channels: {meta.get('channel_count', 'N/A')}")
            print(f"  Sampling Rate: {meta.get('sampling_rate', 'N/A')} Hz")
            print(f"  Start Time: {meta.get('start_time', 'N/A')}")

        # Get latest data
        latest = r.hgetall('isyncwave:eeg:latest')
        if latest:
            print("\nLatest Sample:")
            print(f"  Timestamp: {latest.get('timestamp', 'N/A')}")
            print(f"  DateTime: {latest.get('datetime', 'N/A')}")

            # Get channel names from metadata
            if meta and 'channels' in meta:
                channels = meta['channels'].split(',')
                print("\n  Channel Values:")
                for ch in channels:
                    value = latest.get(ch, 'N/A')
                    print(f"    {ch:4s}: {value}")
            else:
                print("\n  Raw Data:")
                for key, value in latest.items():
                    if key not in ['timestamp', 'datetime']:
                        print(f"    {key}: {value}")
        else:
            print("\n⚠ No latest data available")

    except redis.ConnectionError:
        print(f"❌ Could not connect to Redis at {redis_host}:{redis_port}")
    except Exception as e:
        print(f"❌ Error: {e}")

def view_stream_data(redis_host='localhost', redis_port=6379, redis_db=0, count=10):
    """
    View recent data from Redis Stream
    """
    try:
        r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        r.ping()

        print("=" * 80)
        print(f"Recent EEG Data from Redis Stream (last {count} samples)")
        print("=" * 80)

        # Get stream data
        stream_data = r.xrevrange('isyncwave:eeg:stream', count=count)

        if not stream_data:
            print("\n⚠ No stream data available")
            return

        # Get metadata for channel names
        meta = r.hgetall('isyncwave:eeg:meta')
        channels = []
        if meta and 'channels' in meta:
            channels = meta['channels'].split(',')

        print(f"\nShowing {len(stream_data)} most recent samples:\n")

        for stream_id, data in reversed(stream_data):
            timestamp = data.get('timestamp', 'N/A')
            dt = data.get('datetime', 'N/A')

            print(f"[{dt}] Timestamp: {timestamp}")

            if channels:
                values = [data.get(ch, 'N/A') for ch in channels]
                # Show first 4 and last 4 channels if more than 8
                if len(channels) > 8:
                    display = ', '.join([f"{v:>8}" for v in values[:4]])
                    display += ' ... '
                    display += ', '.join([f"{v:>8}" for v in values[-4:]])
                else:
                    display = ', '.join([f"{v:>8}" for v in values])
                print(f"  [{display}]")
            print()

    except redis.ConnectionError:
        print(f"❌ Could not connect to Redis at {redis_host}:{redis_port}")
    except Exception as e:
        print(f"❌ Error: {e}")

def monitor_realtime(redis_host='localhost', redis_port=6379, redis_db=0, interval=1):
    """
    Monitor real-time EEG data from Redis
    """
    try:
        r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        r.ping()

        print("=" * 80)
        print("Real-time EEG Data Monitor (Press Ctrl+C to stop)")
        print("=" * 80)

        # Get metadata
        meta = r.hgetall('isyncwave:eeg:meta')
        channels = []
        if meta and 'channels' in meta:
            channels = meta['channels'].split(',')
            print(f"\nChannels: {', '.join(channels)}\n")

        last_timestamp = None

        while True:
            latest = r.hgetall('isyncwave:eeg:latest')

            if latest:
                timestamp = latest.get('timestamp', 'N/A')

                # Only print if data has changed
                if timestamp != last_timestamp:
                    dt = latest.get('datetime', 'N/A')
                    print(f"[{dt}] ", end='')

                    if channels:
                        values = [float(latest.get(ch, 0)) for ch in channels]
                        # Show first 4 and last 4 channels
                        if len(channels) > 8:
                            display = ', '.join([f"{v:8.5f}" for v in values[:4]])
                            display += ' ... '
                            display += ', '.join([f"{v:8.5f}" for v in values[-4:]])
                        else:
                            display = ', '.join([f"{v:8.5f}" for v in values])
                        print(f"[{display}]")

                    last_timestamp = timestamp

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")
    except redis.ConnectionError:
        print(f"❌ Could not connect to Redis at {redis_host}:{redis_port}")
    except Exception as e:
        print(f"❌ Error: {e}")

def view_statistics(redis_host='localhost', redis_port=6379, redis_db=0):
    """
    View statistics about Redis data
    """
    try:
        r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        r.ping()

        print("=" * 80)
        print("Redis Data Statistics")
        print("=" * 80)

        # Metadata
        meta = r.hgetall('isyncwave:eeg:meta')
        if meta:
            print("\nMetadata:")
            for key, value in meta.items():
                print(f"  {key}: {value}")

        # Stream info
        stream_info = r.xinfo_stream('isyncwave:eeg:stream')
        print("\nStream Information:")
        print(f"  Total entries: {stream_info.get('length', 0)}")
        print(f"  First entry ID: {stream_info.get('first-entry', ['N/A'])[0]}")
        print(f"  Last entry ID: {stream_info.get('last-entry', ['N/A'])[0]}")

        # Redis memory
        memory = r.info('memory')
        used_memory_mb = memory.get('used_memory', 0) / (1024 * 1024)
        print(f"\nRedis Memory Usage: {used_memory_mb:.2f} MB")

    except redis.ConnectionError:
        print(f"❌ Could not connect to Redis at {redis_host}:{redis_port}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='View iSyncWave EEG data from Redis'
    )

    parser.add_argument('mode', choices=['latest', 'stream', 'monitor', 'stats'],
                       help='Viewing mode: latest (current data), stream (recent history), '
                            'monitor (real-time), stats (statistics)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Redis host (default: localhost)')
    parser.add_argument('--port', type=int, default=6379,
                       help='Redis port (default: 6379)')
    parser.add_argument('--db', type=int, default=0,
                       help='Redis database number (default: 0)')
    parser.add_argument('-c', '--count', type=int, default=10,
                       help='Number of samples to show (for stream mode, default: 10)')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                       help='Update interval in seconds (for monitor mode, default: 1.0)')

    args = parser.parse_args()

    if args.mode == 'latest':
        view_latest_data(args.host, args.port, args.db)
    elif args.mode == 'stream':
        view_stream_data(args.host, args.port, args.db, args.count)
    elif args.mode == 'monitor':
        monitor_realtime(args.host, args.port, args.db, args.interval)
    elif args.mode == 'stats':
        view_statistics(args.host, args.port, args.db)
