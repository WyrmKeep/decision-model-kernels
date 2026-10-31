#!/bin/sh
# DMK ablation arm all_timing: set the variables, run the command
export DMK_ABLATE=all
export DMK_TIMING=1
exec "$@"
