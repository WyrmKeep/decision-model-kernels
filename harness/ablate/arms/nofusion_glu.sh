#!/bin/sh
# DMK ablation arm nofusion_glu: set the variables, run the command
export GGML_CUDA_DISABLE_FUSION=1
export DMK_ABLATE=glu
exec "$@"
