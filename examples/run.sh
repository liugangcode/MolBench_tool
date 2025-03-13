#!/bin/bash

# Define log file name with timestamp
LOG_FILE="logs/training.log"

# Ensure the logs directory exists
mkdir -p logs

# Run the script in the background with nohup
nohup python 2_test_bench.py > "$LOG_FILE" 2>&1 &

# Print the process ID
# 30542
echo "Training started with PID: $!"

# ps -u liugang -o pid,cmd