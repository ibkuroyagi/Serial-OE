#!/bin/bash

stage=1          # job
start_stage=3    # scp
stop_stage=4     # scp
run_stage=3      # training
col_name=machine # outlier, section, machine, outlier2, section2, or machine2
threshold=1
seed=0
no=audioset_v000
n_anomaly=-1
feature=_embed
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
# shellcheck disable=SC1091
. utils/original_funcs.sh || exit 1
available_gpus=16
set -euo pipefail

valid_ratio=0.15
epochs="100"
machines=("fan" "pump" "slider" "ToyCar" "ToyConveyor" "valve")

# machines=("ToyConveyor")
# machines=("pump" "slider" "fan" "ToyConveyor" "valve")
if [ "${stage}" -le 1 ] && [ "${stage}" -ge 1 ]; then
    for machine in "${machines[@]}"; do
        slurm_gpu_scheduler "${available_gpus}"
        log "Start model training ${machine}/${no}_${col_name}${threshold}_${valid_ratio}."
        sbatch --mail-type=END --mail-user=kuroyanagi.ibuki@g.sp.m.is.nagoya-u.ac.jp -J "${machine}_${col_name}${threshold}_seed${seed}_${valid_ratio}" ./local/use_outlier.sh \
            --stage "${start_stage}" \
            --run_stage "${run_stage}" \
            --stop_stage "${stop_stage}" \
            --pos_machine "${machine}" \
            --no "${no}" \
            --epochs "${epochs}" \
            --valid_ratio "${valid_ratio}" \
            --col_name "${col_name}" \
            --threshold "${threshold}" \
            --seed "${seed}" \
            --max_anomaly_pow "6" \
            --n_anomaly "${n_anomaly}" \
            --feature "${feature}"
    done
fi

if [ "${stage}" -le 2 ] && [ "${stage}" -ge 2 ]; then
    ./local/scoring.sh \
        --no "${no}_${col_name}${threshold}_${valid_ratio}_seed${seed}" \
        --feature "${feature}" \
        --epochs "${epochs}"
fi
