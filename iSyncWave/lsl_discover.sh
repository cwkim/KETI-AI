#!/bin/bash
# LSL Stream Discovery Wrapper Script
export PYLSL_LIB=/tmp/liblsl/build/liblsl.so
/usr/bin/python3 discover_lsl_streams.py "$@"
