#! /bin/bash
MAIN_PATH="$PWD"
INPUT_DIR="$1"
OUTPUT_DIR="$2"

find . -type d -name $INPUT_DIR -exec rm -fr {} \;
find . -type d -name $OUTPUT_DIR -exec rm -fr {} \;
find . -name "*.log*" -type f -print0 | xargs -0 /bin/rm -f
find . -name "*.out" -type f -print0 | xargs -0 /bin/rm -f
find . -type d -name "sim_*" -exec rm -fr {} \;