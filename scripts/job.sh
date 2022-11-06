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
# shellcheck disable=SC1091
. utils/original_funcs.sh || exit 1
available_gpus=15
epochs="50 100"

set -euo pipefail

machines=("fan" "pump" "slider" "ToyCar" "ToyConveyor" "valve")

resume=""
tag=${model_name}
if [ "${stage}" -le 1 ] && [ "${stage}" -ge 1 ]; then
    for machine in "${machines[@]}"; do
        slurm_gpu_scheduler "${available_gpus}"
        log "Start model training ${machine}.${model_name}."
        sbatch --mail-type=END --mail-user=kuroyanagi.ibuki@g.sp.m.is.nagoya-u.ac.jp -J "${machine}.${model_name}" ./run.sh \
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
    tag=${model_name}_${valid_ratio}
    if [ ${seed} -ge 0 ] && [ "${n_anomaly}" -le -1 ]; then
        tag+="_seed${seed}"
    fi
    if [ ${n_anomaly} -ge 0 ]; then
        tag+="_anomaly${n_anomaly}_max${max_anomaly_pow}_seed${seed}"
    fi
    ./local/scoring.sh \
        --tag "${tag}" \
        --feature "${feature}" \
        --epochs "${epochs}"
fi
