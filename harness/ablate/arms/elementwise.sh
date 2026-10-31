#!/bin/sh
# DMK ablation arm elementwise: set the variables, run the command
export DMK_ABLATE=elementwise
exec "$@"
