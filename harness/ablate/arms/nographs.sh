#!/bin/sh
# DMK ablation arm nographs: set the variables, run the command
export GGML_CUDA_DISABLE_GRAPHS=1
exec "$@"
