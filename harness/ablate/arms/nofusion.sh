#!/bin/sh
# DMK ablation arm nofusion: set the variables, run the command
export GGML_CUDA_DISABLE_FUSION=1
exec "$@"
