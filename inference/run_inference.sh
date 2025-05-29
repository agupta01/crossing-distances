#!/bin/bash

# Configuration
MODEL_PATH="${1:-train_10}"
MODE="${2:-full}"

# List of cities
CITIES="mke ict ral lnk buf int"

# Run inference for all cities in parallel using GNU Parallel
parallel --verbose \
  "CITY_CODE={} MODAL_ENVIRONMENT={} modal run -d sam_inference.py --model $MODEL_PATH --mode $MODE" \
  ::: $CITIES

echo "All processes completed."