#!/bin/sh
# DMK ablation arm matmul_q: set the variables, run the command
export DMK_ABLATE=matmul,quantize
exec "$@"
