#!/bin/bash

# shellcheck disable=SC1091
. utils/original_funcs.sh || exit 1
available_gpus=15
no=audioset_v000
seed=1
col_name=machine
for threshold in 0.001 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
    slurm_gpu_scheduler "${available_gpus}"
    ./use_outliers_job.sh --start_stage 3 --stop_stage 3 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "${no}"
done
for threshold in 0 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
    sbatch ./use_outliers_job.sh --stage 2 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "${no}"
done

for seed in 2 3 4; do
    for threshold in 0 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
        slurm_gpu_scheduler "${available_gpus}"
        ./use_outliers_job.sh --start_stage 3 --stop_stage 3 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "${no}"
    done
    for threshold in 0 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
        sbatch ./use_outliers_job.sh --stage 2 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "${no}"
    done
done
