#!/bin/sh
# DMK ablation arm l2norm: set the variables, run the command
export DMK_ABLATE=l2norm
exec "$@"
