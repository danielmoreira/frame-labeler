#!/bin/bash
# This script demonstrates how to label frames from a video using the frame labeler tool (mode 1).
# It will work only after properly compiling the tool.
# Usage: ./02_frame_labeling.sh

# creates a list with the video frames previously extracted
ls -d -1 ./frames/*.jpg > ./frame_list.txt

# runs the frame labeler; labels will be saved in "./labels.txt"
../build/framelabeler 1 -i ./frame_list.txt -e arc -o ./labels.txt