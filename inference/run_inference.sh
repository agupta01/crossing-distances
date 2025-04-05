#!/bin/bash

# Configuration
MODEL_PATH="${1:-train_9}"  # First argument or default to "train_1"
MODE="${2:-test}"           # Second argument or default to "test"
FORCE_FLAG="${3:-}"         # Third argument, empty by default

# List of cities
CITIES="abq ana anc apa atl"

# Run inference for all cities in parallel using GNU Parallel
parallel --verbose \
  "CITY_CODE={} MODAL_ENVIRONMENT={} modal run -d sam_inference.py --model $MODEL_PATH --mode $MODE $FORCE_FLAG" \
  ::: $CITIES

echo "All processes completed."