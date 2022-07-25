#!/bin/bash

stage=1          # job
start_stage=3    # scp
run_stage=3      # training
col_name=section # outlier or section
threshold=1
seed=0
machine=fan
# ("fan" "pump" "slider" "ToyCar" "ToyConveyor" "valve")

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
set -euo pipefail
no=audioset_v000
valid_ratio=0.15
epochs="100"

if [ "${stage}" -le 1 ] && [ "${stage}" -ge 1 ]; then
    echo "Start model training ${machine}/${no}_${col_name}${threshold}_${valid_ratio}."
    sbatch --mail-type=END --mail-user=kuroyanagi.ibuki@g.sp.m.is.nagoya-u.ac.jp -J "${machine}_${col_name}${threshold}_seed${seed}_${valid_ratio}" ./local/use_outlier.sh \
        --stage "${start_stage}" \
        --run_stage "${run_stage}" \
        --stop_stage "3" \
        --pos_machine "${machine}" \
        --no "${no}" \
        --epochs "${epochs}" \
        --valid_ratio "${valid_ratio}" \
        --col_name "${col_name}" \
        --threshold "${threshold}" \
        --seed "${seed}"
fi

if [ "${stage}" -le 2 ] && [ "${stage}" -ge 2 ]; then
    ./local/scoring.sh \
        --no "${no}_${col_name}${threshold}_${valid_ratio}_seed${seed}" \
        --feature "_embed" \
        --epochs "${epochs}"
fi
