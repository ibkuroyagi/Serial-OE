#!/bin/bash

stage=1
start_stage=3

# training related
valid_ratio=0.15
model_name=serial_oe.normal
seed=0

# anomaly related
n_anomaly=-1 # If you train the *.normal.yaml, you should set n_anomaly=<-1.
# If you train the *.{utilize,contaminated}.yaml, you should set n_anomaly=>0.
max_anomaly_pow=6

# inference related
feature=_embed

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

set -euo pipefail

machines=("fan" "pump" "slider" "ToyCar" "ToyConveyor" "valve")
epochs="50 100"
resume=""
tag="${model_name}"

if [ "${stage}" -le 1 ] && [ "${stage}" -ge 1 ]; then
    for machine in "${machines[@]}"; do
        ./run.sh \
            --stage "${start_stage}" \
            --stop_stage "5" \
            --conf "conf/tuning/${model_name}.yaml" \
            --pos_machine "${machine}" \
            --resume "${resume}" \
            --tag "${tag}" \
            --feature "${feature}" \
            --epochs "${epochs}" \
            --valid_ratio "${valid_ratio}" \
            --n_anomaly "${n_anomaly}" \
            --max_anomaly_pow "${max_anomaly_pow}" \
            --seed "${seed}"
    done
fi

if [ "${stage}" -le 2 ] && [ "${stage}" -ge 2 ]; then
    tag="${model_name}_${valid_ratio}"
    if [ "${seed}" -ge 0 ] && [ "${n_anomaly}" -le -1 ]; then
        tag+="_seed${seed}"
    fi
    if [ "${n_anomaly}" -ge 0 ]; then
        tag+="_anomaly${n_anomaly}_max${max_anomaly_pow}_seed${seed}"
    fi
    ./local/scoring.sh \
        --tag "${tag}" \
        --feature "${feature}" \
        --epochs "${epochs}"
fi
