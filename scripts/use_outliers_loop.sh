#!/bin/bash

# shellcheck disable=SC1091
. utils/original_funcs.sh || exit 1
available_gpus=22

# 0 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 0.995 0.999

# for no in 030 029; do
#     for seed in 0 1 2 3 4; do
#         slurm_gpu_scheduler "${available_gpus}"
#         ./use_outliers_job.sh --start_stage 1 --stop_stage 2 --run_stage 3 --seed "${seed}" --no "audioset_v${no}"
#     done
# done
# for no in 030 029; do
#     col_name=machine
#     for threshold in 0 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
#         slurm_gpu_scheduler "${available_gpus}"
#         ./use_outliers_job.sh --start_stage 3 --stop_stage 4 --run_stage 3 --threshold "${threshold}" --col_name "${col_name}" --seed "0" --no "audioset_v${no}"
#     done
#     col_name=machine2
#     for threshold in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 0.995 0.999 1; do
#         slurm_gpu_scheduler "${available_gpus}"
#         ./use_outliers_job.sh --start_stage 3 --stop_stage 4 --run_stage 3 --threshold "${threshold}" --col_name "${col_name}" --seed "0" --no "audioset_v${no}"
#     done

#     col_name=machine
#     for seed in 1 2 3 4; do
#         for threshold in 0 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
#             slurm_gpu_scheduler "${available_gpus}"
#             ./use_outliers_job.sh --start_stage 3 --stop_stage 3 --run_stage 3 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "audioset_v${no}"
#         done
#     done
#     col_name=machine2
#     for seed in 1 2 3 4; do
#         for threshold in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 0.995 0.999 1; do
#             slurm_gpu_scheduler "${available_gpus}"
#             ./use_outliers_job.sh --start_stage 3 --stop_stage 3 --run_stage 3 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "audioset_v${no}"
#         done
#     done
# done

# #########
for no in 029 030; do
    col_name=machine
    for seed in 0 1 2 3 4; do
        for threshold in 0 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
            ./use_outliers_job.sh --stage 2 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "audioset_v${no}"
        done
    done
    col_name=machine2
    for seed in 0 1 2 3 4; do
        for threshold in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 0.995 0.999 1; do
            ./use_outliers_job.sh --stage 2 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "audioset_v${no}"
        done
    done
done
