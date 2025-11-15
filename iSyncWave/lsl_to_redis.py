#!/usr/bin/env python3
"""
LSL to Redis Streamer
Receives real-time EEG data from iSyncWave LSL streams and saves to Redis (with optional CSV export)
"""

import pylsl
import time
import csv
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import redis

def stream_lsl_to_redis(stream_name=None, duration=None, output_file=None, output_dir='data',
                        enable_csv=False, enable_redis=True, redis_host='localhost', redis_port=6379, redis_db=0,
                        redis_maxlen=None):
    """
    Receive EEG data from LSL stream and save to Redis (and optionally CSV file)

    Args:
        stream_name: Specific stream name to connect to (None for first available)
        duration: How long to receive data (seconds, None for infinite)
        output_file: Output CSV filename (None for auto-generated)
        output_dir: Directory to save CSV files (default: 'data')
        enable_csv: Enable CSV file saving (default: False)
        enable_redis: Enable Redis saving (default: True)
        redis_host: Redis host (default: 'localhost')
        redis_port: Redis port (default: 6379)
        redis_db: Redis database number (default: 0)
        redis_maxlen: Maximum length of Redis stream (None for unlimited, default: None)
    """
    print("Searching for LSL streams...")

    # Resolve streams
    if stream_name:
        print(f"Looking for stream: {stream_name}")
        streams = pylsl.resolve_byprop('name', stream_name, timeout=5)
    else:
        print("Looking for any EEG stream...")
        streams = pylsl.resolve_byprop('type', 'EEG', timeout=5)

        # If no EEG type found, try to get any stream
        if not streams:
            print("No EEG streams found, trying all streams...")
            streams = pylsl.resolve_streams(wait_time=5)

    if not streams:
        print("\n❌ No LSL streams found!")
        print("\nTroubleshooting:")
        print("1. Check if the iSyncWave device is powered on")
        print("2. Check if the tablet app is running and streaming")
        print("3. Verify both devices are on the same network")
        print("4. Check firewall settings")
        return

    # Use the first stream found
    stream_info = streams[0]

    print("\n" + "=" * 70)
    print(f"Connecting to stream:")
    print(f"  Name: {stream_info.name()}")
    print(f"  Type: {stream_info.type()}")
    print(f"  Channels: {stream_info.channel_count()}")
    print(f"  Sampling Rate: {stream_info.nominal_srate()} Hz")
    print(f"  Source: {stream_info.source_id()}")
    print("=" * 70)

    # Create inlet to receive data
    inlet = pylsl.StreamInlet(stream_info)

    # Define standard 19-channel EEG electrode names (10-20 system)
    standard_channel_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]

    # Use standard channel names up to the number of channels in the stream
    channel_count = stream_info.channel_count()
    if channel_count <= len(standard_channel_names):
        channel_names = standard_channel_names[:channel_count]
    else:
        # If more channels than standard names, add numbered channels
        channel_names = standard_channel_names + [f"Channel_{i+1}" for i in range(len(standard_channel_names), channel_count)]

    print(f"\nChannel names: {', '.join(channel_names)}")

    # Connect to Redis if enabled
    redis_client = None
    redis_stream_key = "isyncwave:eeg:stream"
    redis_meta_key = "isyncwave:eeg:meta"

    if enable_redis:
        try:
            redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db,
                                      decode_responses=True)
            redis_client.ping()
            print(f"\n✓ Connected to Redis at {redis_host}:{redis_port}")

            # Store metadata
            metadata = {
                'stream_name': stream_info.name(),
                'stream_type': stream_info.type(),
                'channel_count': stream_info.channel_count(),
                'sampling_rate': stream_info.nominal_srate(),
                'channels': ','.join(channel_names),
                'start_time': datetime.now().isoformat()
            }
            redis_client.hset(redis_meta_key, mapping=metadata)
            print(f"✓ Redis metadata saved to '{redis_meta_key}'")

        except redis.ConnectionError as e:
            print(f"\n⚠ Could not connect to Redis: {e}")
            print("Continuing without Redis...")
            redis_client = None
        except Exception as e:
            print(f"\n⚠ Redis error: {e}")
            print("Continuing without Redis...")
            redis_client = None

    # Prepare CSV output if enabled
    csv_path = None
    if enable_csv:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime('%y-%m-%d_%H_%M_%S')
            output_file = f"lsl_data_{timestamp}.csv"

        csv_path = output_path / output_file
        print(f"\n📁 CSV saving enabled: {csv_path}")
    else:
        print(f"\n📁 CSV saving disabled")

    if redis_client:
        print(f"📊 Streaming to Redis: {redis_stream_key}")
    print(f"Duration: {duration if duration else 'infinite'} seconds")
    print("Press Ctrl+C to stop\n")

    start_time = time.time()
    sample_count = 0
    redis_save_count = 0
    no_data_count = 0
    waiting_for_data = True
    last_sample_time = None

    # Open CSV file for writing if enabled
    csvfile = None
    csv_writer = None

    if enable_csv:
        csvfile = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csvfile)

        # Write header row
        header = ['Timestamp'] + channel_names
        csv_writer.writerow(header)
        print("✓ CSV file created with header")

    print(f"✓ Waiting for data...\n")

    try:
        while True:
            # Check duration (only count time when receiving data)
            if duration and last_sample_time and (time.time() - start_time) > duration:
                break

            # Pull a sample with timeout
            sample, timestamp = inlet.pull_sample(timeout=1.0)

            if sample:
                # Reset no data counter
                if no_data_count > 0:
                    if waiting_for_data:
                        print(f"\n✓ Data reception started!\n")
                        waiting_for_data = False
                    else:
                        print(f"\n✓ Data reception resumed after {no_data_count}s wait!\n")
                    no_data_count = 0

                last_sample_time = time.time()
                sample_count += 1

                # Write to CSV if enabled
                if csv_writer:
                    row = [timestamp] + sample
                    csv_writer.writerow(row)

                # Save to Redis if connected
                if redis_client:
                    try:
                        # Prepare data for Redis
                        data_dict = {
                            'lsl_timestamp': timestamp,  # LSL timestamp (seconds since system boot)
                            'timestamp': time.time(),    # Unix timestamp (for proper datetime)
                            'datetime': datetime.now().isoformat()  # Current system time
                        }

                        # Add channel data
                        for i, ch_name in enumerate(channel_names):
                            data_dict[ch_name] = sample[i]

                        # Save to Redis Stream
                        if redis_maxlen:
                            redis_client.xadd(redis_stream_key, data_dict, maxlen=redis_maxlen)
                        else:
                            redis_client.xadd(redis_stream_key, data_dict)

                        redis_save_count += 1

                    except Exception as e:
                        if sample_count % 1000 == 0:
                            print(f"\n⚠ Redis save error: {e}")

                # Show progress every 100 samples
                if sample_count % 100 == 0:
                    if csvfile:
                        csvfile.flush()

                    elapsed = time.time() - start_time
                    rate = sample_count / elapsed

                    status = f"📊 {sample_count} samples"
                    if redis_client:
                        status += f" (Redis: {redis_save_count})"
                    if csv_writer:
                        status += f" (CSV: {sample_count})"
                    status += f" | Rate: {rate:.2f} Hz | Elapsed: {elapsed:.1f}s"
                    print(status, end='\r')

            else:
                # No data received
                no_data_count += 1

                # Show waiting message every 5 seconds
                if no_data_count % 5 == 1:
                    if waiting_for_data:
                        print(f"⏳ Waiting for data... ({no_data_count}s)", end='\r')
                    else:
                        print(f"\n⚠ No data for {no_data_count}s, waiting for stream to resume...", end='\r')

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        if csvfile:
            csvfile.flush()
            csvfile.close()

    # Print summary
    elapsed = time.time() - start_time
    if sample_count > 0:
        rate = sample_count / elapsed

        print("\n" + "=" * 70)
        print(f"Reception Summary:")
        print(f"  Total Samples: {sample_count}")
        print(f"  Duration: {elapsed:.2f} seconds")
        print(f"  Average Rate: {rate:.2f} Hz")
        print(f"  Expected Rate: {stream_info.nominal_srate()} Hz")

        if csv_path and csv_path.exists():
            file_size = csv_path.stat().st_size / (1024 * 1024)  # MB
            print(f"\n  CSV Summary:")
            print(f"    File Size: {file_size:.2f} MB")
            print(f"    Saved to: {csv_path}")

        if redis_client:
            print(f"\n  Redis Summary:")
            print(f"    Samples saved to Redis: {redis_save_count}")
            print(f"    Stream key: {redis_stream_key}")
            print(f"    Metadata key: {redis_meta_key}")

            # Update metadata with final stats
            try:
                redis_client.hset(redis_meta_key, mapping={
                    'end_time': datetime.now().isoformat(),
                    'total_samples': sample_count,
                    'duration_seconds': elapsed
                })
            except:
                pass

        print("=" * 70)
    else:
        print("\n⚠ No samples were received.")
        print("Check if the tablet app is streaming EEG data.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Receive LSL EEG data from iSyncWave and save to Redis (and optionally CSV)'
    )
    parser.add_argument('-n', '--name', type=str,
                       help='Stream name to connect to')
    parser.add_argument('-d', '--duration', type=int, default=0,
                       help='Duration in seconds (default: 0 for infinite)')
    parser.add_argument('--enable-csv', action='store_true',
                       help='Enable CSV file saving (disabled by default)')
    parser.add_argument('-o', '--output', type=str,
                       help='Output CSV filename (auto-generated if not specified)')
    parser.add_argument('-dir', '--directory', type=str, default='data',
                       help='Output directory (default: data)')
    parser.add_argument('--no-redis', action='store_true',
                       help='Disable Redis saving')
    parser.add_argument('--redis-host', type=str, default='localhost',
                       help='Redis host (default: localhost)')
    parser.add_argument('--redis-port', type=int, default=6379,
                       help='Redis port (default: 6379)')
    parser.add_argument('--redis-db', type=int, default=0,
                       help='Redis database number (default: 0)')
    parser.add_argument('--redis-maxlen', type=int, default=None,
                       help='Maximum length of Redis stream (default: None for unlimited)')

    args = parser.parse_args()

    duration = None if args.duration == 0 else args.duration

    stream_lsl_to_redis(
        stream_name=args.name,
        duration=duration,
        output_file=args.output,
        output_dir=args.directory,
        enable_csv=args.enable_csv,
        enable_redis=not args.no_redis,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        redis_maxlen=args.redis_maxlen
    )
