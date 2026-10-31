#!/bin/sh
# DMK ablation arm sync_all: set the variables, run the command
export DMK_ABLATE=all
export DMK_ABLATE_SYNC=1
exec "$@"
