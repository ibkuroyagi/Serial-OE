#!/bin/bash

stage=1
start_stage=3
no=v020
valid_ratio=0.15
# anomaly related
max_anomaly_pow=6
n_anomaly=-1
seed=0
# inference related
feature=_embed
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
# shellcheck disable=SC1091
. utils/original_funcs.sh || exit 1
available_gpus=15
epochs="100"

set -euo pipefail
machines=("fan" "pump" "slider" "ToyCar" "ToyConveyor" "valve")
# machines=("valve")
# machines=("pump" "slider" "ToyCar" "ToyConveyor" "valve")
resume=""
tag=${no}
if [ "${stage}" -le 1 ] && [ "${stage}" -ge 1 ]; then
    for machine in "${machines[@]}"; do
        slurm_gpu_scheduler "${available_gpus}"
        log "Start model training ${machine}/${no}."
        sbatch --mail-type=END --mail-user=kuroyanagi.ibuki@g.sp.m.is.nagoya-u.ac.jp -J "${machine}${no}" ./run.sh \
            --stage "${start_stage}" \
            --stop_stage "5" \
            --conf "conf/tuning/asd_model.${no}.yaml" \
            --pos_machine "${machine}" \
            --resume "${resume}" \
            --tag "${tag}" \
            --feature "${feature}" \
            --epochs "${epochs}" \
            --valid_ratio "${valid_ratio}" \
            --n_anomaly "${n_anomaly}" \
            --seed "${seed}"
    done
fi

if [ "${stage}" -le 2 ] && [ "${stage}" -ge 2 ]; then
    tag=${no}_${valid_ratio}
    if [ ${seed} -ge 0 ] && [ "${n_anomaly}" -le -1 ]; then
        tag+="_seed${seed}"
    fi
    if [ ${n_anomaly} -ge 0 ]; then
        tag+="_anomaly${n_anomaly}_max${max_anomaly_pow}_seed${seed}"
    fi
    ./local/scoring.sh \
        --no "${tag}" \
        --feature "${feature}" \
        --epochs "${epochs}"
fi
