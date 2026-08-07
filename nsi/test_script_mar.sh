#!/bin/bash

# Test script for large recons: MAR with num_metal=2
# Uses the mbirtorch default partition sequence.

DATA_PATH=/depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal/

mkdir -p ~/mbirtorch_notes/
python Lilly_recon.py \
  --data_path "$DATA_PATH" \
  --downsampling 1 \
  --subsample_view_factor 2 \
  --num_metal 2 \
  --sino_cropping 1 \
  --max_iterations 15 \
  2>&1 | tee ~/mbirtorch_notes/Lilly_mar_ds1_run.log
