#!/bin/bash
# LSL Data Receiver Wrapper Script
export PYLSL_LIB=/tmp/liblsl/build/liblsl.so
/usr/bin/python3 receive_lsl_data.py "$@"
