#!/bin/bash
# This script demonstrates how to extract frames from a video using the frame labeler tool (mode 0).
# It will work only after properly compiling the tool.
# Usage: ./01_frame_extraction.sh
../build/framelabeler 0 -i ./video_list.txt -f ./frames
