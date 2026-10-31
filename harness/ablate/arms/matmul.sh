#!/bin/sh
# DMK ablation arm matmul: set the variables, run the command
export DMK_ABLATE=matmul
exec "$@"
