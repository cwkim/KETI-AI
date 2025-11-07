#!/bin/bash
# LSL Continuous Monitor Wrapper Script
export PYLSL_LIB=/tmp/liblsl/build/liblsl.so
/usr/bin/python3 -u monitor_lsl.py "$@"
