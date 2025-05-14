#!/bin/bash
# This script demonstrates how to label an entire video's frames as negative with the frame labeler tool (mode 2).
# It will work only after properly compiling the tool.
# Usage: ./04_label_fully_negative.sh

# runs the frame labeler; labels will be saved inside the "./neg_labels" folder
../build/framelabeler 2 -i ./video_list.txt -e arc -o ./neg_labels
