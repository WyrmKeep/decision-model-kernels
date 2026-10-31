#!/bin/sh
# DMK ablation arm quantize: set the variables, run the command
export DMK_ABLATE=quantize
exec "$@"
