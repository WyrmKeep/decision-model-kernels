#!/bin/sh
# DMK ablation arm mmq_quantize: set the variables, run the command
export DMK_ABLATE=mmq,quantize
exec "$@"
