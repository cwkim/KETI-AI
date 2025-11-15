#!/usr/bin/env python3
"""
Continuous LSL Stream Monitor
Continuously monitors for LSL streams and automatically receives data when found
"""

import pylsl
import time
from datetime import datetime

def monitor_streams(check_interval=2):
    """
    Continuously monitor for LSL streams

    Args:
        check_interval: How often to check for streams (seconds)
    """
    print("=" * 60)
    print("LSL Stream Monitor for iSyncWave")
    print("=" * 60)
    print(f"\nContinuously monitoring for LSL streams...")
    print(f"Check interval: {check_interval} seconds")
    print("Press Ctrl+C to stop\n")

    last_stream_count = 0
    connected = False
    inlet = None
    stream_info = None

    try:
        while True:
            # Check for streams
            streams = pylsl.resolve_streams(timeout=1)

            # If stream count changed, print update
            if len(streams) != last_stream_count:
                timestamp = datetime.now().strftime('%H:%M:%S')

                if len(streams) == 0:
                    print(f"[{timestamp}] ⏳ No streams detected. Waiting...")
                    connected = False
                    inlet = None
                else:
                    print(f"\n[{timestamp}] ✓ Found {len(streams)} stream(s)!")
                    for i, stream in enumerate(streams, 1):
                        print(f"\n  Stream {i}:")
                        print(f"    Name: {stream.name()}")
                        print(f"    Type: {stream.type()}")
                        print(f"    Channels: {stream.channel_count()}")
                        print(f"    Sampling Rate: {stream.nominal_srate()} Hz")
                        print(f"    Source: {stream.source_id()}")

                    # Connect to first stream
                    if not connected and len(streams) > 0:
                        stream_info = streams[0]
                        print(f"\n🔗 Connecting to '{stream_info.name()}'...")
                        inlet = pylsl.StreamInlet(stream_info)
                        connected = True
                        print("✓ Connected! Receiving data...\n")

                last_stream_count = len(streams)

            # If connected, receive and display data
            if connected and inlet:
                sample, timestamp = inlet.pull_sample(timeout=0.1)

                if sample:
                    timestamp_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]

                    # Format sample data
                    if len(sample) <= 8:
                        channel_str = ", ".join([f"{v:8.2f}" for v in sample])
                    else:
                        channel_str = ", ".join([f"{v:8.2f}" for v in sample[:4]]) + \
                                    " ... " + \
                                    ", ".join([f"{v:8.2f}" for v in sample[-4:]])

                    print(f"[{timestamp_str}] [{channel_str}]")
            else:
                time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n\n✓ Monitor stopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Monitor for LSL streams from iSyncWave')
    parser.add_argument('-i', '--interval', type=int, default=2,
                       help='Check interval in seconds (default: 2)')

    args = parser.parse_args()

    monitor_streams(check_interval=args.interval)
