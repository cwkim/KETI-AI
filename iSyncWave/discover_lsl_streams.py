#!/usr/bin/env python3
"""
LSL Stream Discovery Script
Discovers all available LSL streams on the local network
"""

import pylsl
import time

def discover_streams(timeout=5):
    """
    Discover all available LSL streams on the network

    Args:
        timeout: How long to wait for streams (seconds)
    """
    print(f"Searching for LSL streams for {timeout} seconds...")
    print("-" * 60)

    # Resolve all streams on the network
    streams = pylsl.resolve_streams(timeout=timeout)

    if not streams:
        print("No LSL streams found on the network.")
        print("\nMake sure:")
        print("1. The iSyncWave device is powered on")
        print("2. The tablet app is running and streaming data")
        print("3. Both devices are on the same network")
        return []

    print(f"Found {len(streams)} stream(s):\n")

    for i, stream in enumerate(streams, 1):
        print(f"Stream {i}:")
        print(f"  Name: {stream.name()}")
        print(f"  Type: {stream.type()}")
        print(f"  Channel Count: {stream.channel_count()}")
        print(f"  Sampling Rate: {stream.nominal_srate()} Hz")
        print(f"  Format: {stream.channel_format()}")
        print(f"  Source ID: {stream.source_id()}")
        print(f"  Hostname: {stream.hostname()}")
        print("-" * 60)

    return streams

if __name__ == "__main__":
    try:
        streams = discover_streams(timeout=5)

        if streams:
            print(f"\n✓ Successfully discovered {len(streams)} LSL stream(s)")
            print("\nYou can now use 'receive_lsl_data.py' to receive data from these streams.")

    except KeyboardInterrupt:
        print("\n\nStream discovery interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
