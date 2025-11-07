#!/usr/bin/env python3
"""
LSL Data Receiver for iSyncWave EEG Data
Receives and displays real-time EEG data from LSL streams
"""

import pylsl
import time
import numpy as np
from datetime import datetime

def receive_eeg_data(stream_name=None, duration=10, show_data=True):
    """
    Receive EEG data from LSL stream

    Args:
        stream_name: Specific stream name to connect to (None for first available)
        duration: How long to receive data (seconds, None for infinite)
        show_data: Whether to print received samples
    """
    print("Searching for LSL streams...")

    # Resolve streams
    if stream_name:
        print(f"Looking for stream: {stream_name}")
        streams = pylsl.resolve_byprop('name', stream_name, wait_time=5)
    else:
        print("Looking for any EEG stream...")
        streams = pylsl.resolve_stream('type', 'EEG', wait_time=5)

        # If no EEG type found, try to get any stream
        if not streams:
            print("No EEG streams found, trying all streams...")
            streams = pylsl.resolve_streams(wait_time=5)

    if not streams:
        print("\n❌ No LSL streams found!")
        print("\nTroubleshooting:")
        print("1. Check if the tablet app is running")
        print("2. Verify both devices are on the same network")
        print("3. Check firewall settings")
        return

    # Use the first stream found
    stream_info = streams[0]

    print("\n" + "=" * 60)
    print(f"Connecting to stream:")
    print(f"  Name: {stream_info.name()}")
    print(f"  Type: {stream_info.type()}")
    print(f"  Channels: {stream_info.channel_count()}")
    print(f"  Sampling Rate: {stream_info.nominal_srate()} Hz")
    print(f"  Source: {stream_info.source_id()}")
    print("=" * 60)

    # Create inlet to receive data
    inlet = pylsl.StreamInlet(stream_info)

    print(f"\n✓ Connected! Receiving data for {duration if duration else 'infinite'} seconds...")
    print("Press Ctrl+C to stop\n")

    start_time = time.time()
    sample_count = 0

    try:
        while True:
            # Check duration
            if duration and (time.time() - start_time) > duration:
                break

            # Pull a sample with timestamp
            sample, timestamp = inlet.pull_sample(timeout=1.0)

            if sample:
                sample_count += 1

                if show_data:
                    # Display sample data
                    timestamp_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]

                    # Format channel values
                    if len(sample) <= 8:
                        # Show all channels if 8 or fewer
                        channel_str = ", ".join([f"{v:8.2f}" for v in sample])
                    else:
                        # Show first 4 and last 4 if more than 8 channels
                        channel_str = ", ".join([f"{v:8.2f}" for v in sample[:4]]) + \
                                    " ... " + \
                                    ", ".join([f"{v:8.2f}" for v in sample[-4:]])

                    print(f"[{timestamp_str}] Sample {sample_count:5d}: [{channel_str}]")

                # Print statistics every 100 samples
                if sample_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = sample_count / elapsed
                    print(f"\n📊 Statistics: {sample_count} samples received, "
                          f"Rate: {rate:.2f} Hz, "
                          f"Elapsed: {elapsed:.1f}s\n")

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        elapsed = time.time() - start_time
        if sample_count > 0:
            rate = sample_count / elapsed
            print("\n" + "=" * 60)
            print(f"Reception Summary:")
            print(f"  Total Samples: {sample_count}")
            print(f"  Duration: {elapsed:.2f} seconds")
            print(f"  Average Rate: {rate:.2f} Hz")
            print(f"  Expected Rate: {stream_info.nominal_srate()} Hz")
            print("=" * 60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Receive LSL EEG data from iSyncWave')
    parser.add_argument('-n', '--name', type=str, help='Stream name to connect to')
    parser.add_argument('-d', '--duration', type=int, default=10,
                       help='Duration in seconds (default: 10, 0 for infinite)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Do not print individual samples')

    args = parser.parse_args()

    duration = None if args.duration == 0 else args.duration

    receive_eeg_data(
        stream_name=args.name,
        duration=duration,
        show_data=not args.quiet
    )
