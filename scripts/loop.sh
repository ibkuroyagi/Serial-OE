#!/bin/bash

# shellcheck disable=SC1091
. utils/original_funcs.sh || exit 1
available_gpus=16

no=019
valid_ratio=0.15
n_anomaly=-1
seed=0

for seed in 1 2 3 4; do
    for no in 120 220; do
        for n_anomaly in 0 1 2 4 8 16 32; do
            slurm_gpu_scheduler "${available_gpus}"
            ./job.sh \
                --no "audioset_v${no}" \
                --audioset_pow "0" \
                --stage "1" \
                --start_stage "3" \
                --valid_ratio "${valid_ratio}" \
                --use_uav "false" \
                --use_idmt "false" \
                --n_anomaly "${n_anomaly}" \
                --seed "${seed}"
        done
    done
done

sleep 10800
for seed in 1 2 3 4; do
    for no in 120 220; do
        for n_anomaly in 0 1 2 4 8 16 32; do
            ./job.sh \
                --no "audioset_v${no}" \
                --audioset_pow "0" \
                --stage "2" \
                --valid_ratio "${valid_ratio}" \
                --use_uav "false" \
                --use_idmt "false" \
                --n_anomaly "${n_anomaly}" \
                --seed "${seed}"
        done
    done
done
