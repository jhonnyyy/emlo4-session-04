#!/bin/bash
set -e

echo "Starting training..."
python /app/src/train.py

echo "Starting evaluation..."
python /app/src/eval.py

echo "Starting inference..."
python /app/src/infer.py

echo "All tasks completed."