#!/bin/bash

stage=1
start_stage=3
no=audioset_v020
valid_ratio=0.15
# outlier related
audioset_pow=0
use_uav=false
use_idmt=false
# anomaly related
max_anomaly_pow=6
n_anomaly=-1
seed=0
# inference related
use_dev=false
feature=_embed
use_10sec=false
machine=fan
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
# shellcheck disable=SC1091
. utils/original_funcs.sh || exit 1
available_gpus=15
epochs="100"

set -euo pipefail

resume=""
tag=${no}
if [ "${stage}" -le 1 ] && [ "${stage}" -ge 1 ]; then
    inlier_scp=""
    if "${use_dev}"; then
        inlier_scp="dump/dev/${machine}/train/dev.scp"
    fi
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
        --use_10sec "${use_10sec}" \
        --epochs "${epochs}" \
        --valid_ratio "${valid_ratio}" \
        --audioset_pow "${audioset_pow}" \
        --use_uav "${use_uav}" \
        --use_idmt "${use_idmt}" \
        --n_anomaly "${n_anomaly}" \
        --max_anomaly_pow "${max_anomaly_pow}" \
        --seed "${seed}" \
        --inlier_scp "${inlier_scp}"
fi

if [ "${stage}" -le 2 ] && [ "${stage}" -ge 2 ]; then
    if [ "${use_10sec}" = "true" ]; then
        feature="_10sec${feature}"
    fi
    if "${use_dev}"; then
        feature="_dev${feature}"
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
