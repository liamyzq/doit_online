#!/usr/bin/env bash

# Shared environment for Orion runs.
export DOIT_REPO=/home/mlw0719/doit_correction
export DOIT_STORAGE=/home/mlw0719/doit_correction_storage
export HF_HOME="$DOIT_STORAGE/huggingface"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
