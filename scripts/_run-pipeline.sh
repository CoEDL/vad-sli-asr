#!/usr/bin/bash

# Usage:
# ./scripts/_run-pipeline.sh data/raw/MATHEWS_J33-002163A-S1.wav

python scripts/run_vad-by-silero.py $1 --overwrite

python scripts/run_sli-by-sblr.py models/zmu-eng_sli_k10.pkl $1 --overwrite --rm_vad_tier

python scripts/run_asr-by-w2v2.py /projects/muruwari/data/checkpoints/no-lm_b10 $1 --cuda --overwrite
