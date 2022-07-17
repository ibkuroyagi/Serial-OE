#!/bin/bash

stage=1
start_stage=3
no=audioset_v000
feature=_embed
use_10sec=false
audioset_pow=0
valid_ratio=0.15
use_uav=false
use_idmt=false
# anomaly related
max_anomaly_pow=6
n_anomaly=-1
seed=0
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

epochs="100"

set -euo pipefail
machines=("fan" "pump" "slider" "ToyCar" "ToyConveyor" "valve")
# machines=("fan")
# machines=("pump" "slider" "ToyCar" "ToyConveyor" "valve")
resume=""
tag=${no}
if [ "${stage}" -le 1 ] && [ "${stage}" -ge 1 ]; then
    for machine in "${machines[@]}"; do
        echo "Start model training ${machine}/${no}. resume:${resume}"
        sbatch --mail-type=END --mail-user=kuroyanagi.ibuki@g.sp.m.is.nagoya-u.ac.jp -J "${machine}${no}" ./run.sh \
            --stage "${start_stage}" \
            --stop_stage "5" \
            --conf "conf/tuning/asd_model.${no}.yaml" \
            --pos_machine "${machine}" \
            --resume "${resume}" \
            --tag "${tag}" \
            --feature "${feature}" \
            --use_10sec "${use_10sec}" \
            --epochs "${epochs}" \
            --valid_ratio "${valid_ratio}" \
            --audioset_pow "${audioset_pow}" \
            --use_uav "${use_uav}" \
            --use_idmt "${use_idmt}" \
            --n_anomaly "${n_anomaly}" \
            --max_anomaly_pow "${max_anomaly_pow}" \
            --seed "${seed}"
    done
fi

if [ "${stage}" -le 2 ] && [ "${stage}" -ge 2 ]; then
    if [ "${use_10sec}" = "true" ]; then
        feature="_10sec${feature}"
    fi
    tag=${no}_${valid_ratio}
    if [ ${audioset_pow} -gt 0 ]; then
        tag+=_p${audioset_pow}
    fi
    if "${use_uav}"; then
        tag+="_uav"
    fi
    if "${use_idmt}"; then
        tag+="_idmt"
    fi
    if [ ${n_anomaly} -ge 0 ]; then
        tag+="_anomaly${n_anomaly}_max${max_anomaly_pow}_seed${seed}"
    fi
    ./local/scoring.sh \
        --no "${tag}" \
        --feature "${feature}" \
        --epochs "${epochs}"
fi
