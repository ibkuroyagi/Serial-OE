#!/bin/bash

# Copyright 2022 Ibuki Kuroyanagi
# shellcheck disable=SC1091
. ./cmd.sh || exit 1
. ./path.sh || exit 1
# basic settings
conf=
tag=

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y/%m/%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
set -euo pipefail

for pos_machine in "fan" "pump" "slider" "ToyCar" "ToyConveyor" "valve"; do
    outdir="exp/${pos_machine}/${tag}"
    for feature in "_embed" "_prediction"; do
        # feature="_embed"
        log "Inference start. See the progress via ${outdir}/infer_${pos_machine}${feature}_${tag}.log."
        # shellcheck disable=SC2154,SC2086
        ${cuda_cmd} "${outdir}/infer_${pos_machine}${feature}_${tag}.log" \
            python -m asd_tools.bin.infer \
            --pos_machine "${pos_machine}" \
            --checkpoints "exp/${pos_machine}/${tag}/checkpoint-100epochs/checkpoint-100epochs.pkl" \
            --config "${conf}" \
            --feature "${feature}"
        log "Successfully finished Inference."
    done
done

for feature in "_embed" "_prediction"; do
    ./local/scoring.sh \
        --no "${tag}" \
        --feature "${feature}" \
        --epochs "100"
done
