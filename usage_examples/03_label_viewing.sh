#!/bin/bash
# This script demonstrates how to load frame labels previously annotated with the frame labeler tool (mode 1).
# It will work only after properly compiling the tool.
# Usage: ./03_label_viewing.sh

# runs the frame labeler; previous labels are loaded from "./labels.txt"; new labels will be saved in "./new_labels.txt"
../build/framelabeler 1 -i ./frame_list.txt -e arc -g ./labels.txt -o ./new_labels.txt
