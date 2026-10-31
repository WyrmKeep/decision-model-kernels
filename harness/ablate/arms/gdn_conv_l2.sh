#!/bin/sh
# DMK ablation arm gdn_conv_l2: set the variables, run the command
export DMK_ABLATE=gdn,ssm_conv,l2norm
exec "$@"
