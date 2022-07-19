#!/bin/bash

stage=1       # job
start_stage=3 # scp
run_stage=3   # training

threshold=0.999
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

no=audioset_v000
valid_ratio=0.15
epochs="100"

set -euo pipefail
machines=("fan" "pump" "slider" "ToyCar" "ToyConveyor" "valve")
# machines=("fan")
# machines=("pump" "slider" "fan" "ToyConveyor" "valve")
if [ "${stage}" -le 1 ] && [ "${stage}" -ge 1 ]; then
    for machine in "${machines[@]}"; do
        echo "Start model training ${machine}/${no}_outlier${threshold}_${valid_ratio}."
        sbatch --mail-type=END --mail-user=kuroyanagi.ibuki@g.sp.m.is.nagoya-u.ac.jp -J "${no}_outlier${threshold}_${valid_ratio}" ./run.sh \
            --stage "${start_stage}" \
            --run_stage "${run_stage}" \
            --stop_stage "3" \
            --conf "conf/tuning/asd_model.${no}.yaml" \
            --pos_machine "${machine}" \
            --epochs "${epochs}" \
            --valid_ratio "${valid_ratio}" \
            --threshold "${threshold}"
    done
fi

if [ "${stage}" -le 2 ] && [ "${stage}" -ge 2 ]; then
    ./local/scoring.sh \
        --no "${no}_outlier${threshold}_${valid_ratio}" \
        --feature "_embed" \
        --epochs "${epochs}"
fi
