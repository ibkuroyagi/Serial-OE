#!/bin/bash

# shellcheck disable=SC1091
. utils/original_funcs.sh || exit 1
available_gpus=15

# 0 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 0.995 0.999
# seed=0
# no=020
# col_name=section2
# for threshold in 0.7 1; do
#     slurm_gpu_scheduler "${available_gpus}"
#     ./use_outliers_job.sh --start_stage 3 --stop_stage 4 --run_stage 3 --threshold "${threshold}" --col_name "${col_name}" --seed "0" --no "audioset_v${no}"
# done

# no=019
# col_name=outlier2
# for threshold in 0 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.3 0.5 0.7 1; do
#     slurm_gpu_scheduler "${available_gpus}"
#     ./use_outliers_job.sh --start_stage 3 --stop_stage 4 --run_stage 3 --threshold "${threshold}" --col_name "${col_name}" --seed "0" --no "audioset_v${no}"
# done

# no=020
# col_name=section2
# for seed in 1 2 3 4; do
#     for threshold in 0 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.3 0.5 0.7 1; do
#         slurm_gpu_scheduler "${available_gpus}"
#         ./use_outliers_job.sh --start_stage 3 --stop_stage 3 --run_stage 3 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "audioset_v${no}"
#     done
# done

# no=019
# col_name=outlier2
# for seed in 1 2 3 4; do
#     for threshold in 0 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.3 0.5 0.7 1; do
#         slurm_gpu_scheduler "${available_gpus}"
#         ./use_outliers_job.sh --start_stage 3 --stop_stage 3 --run_stage 3 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "audioset_v${no}"
#     done
# done

# col_name=machine
# for seed in 1 2; do
#     for threshold in 0 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
#         sbatch ./use_outliers_job.sh --stage 2 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "audioset_v${no}"
#     done
#     sleep 60
# done

# for no in 019 020; do
#     for col_name in outlier section; do
#         for threshold in 0 0.1 0.3 0.5 0.7 0.9 0.95 0.99 0.995 0.999 0.9995 0.9999; do
#             sbatch ./use_outliers_job.sh --stage 2 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "audioset_v${no}"
#         done
#         sleep 60
#     done
# # done
# for no in 020 019; do
#     for col_name in section2 outlier2; do
#         for threshold in 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.3 0.5 0.7 1; do
#             ./use_outliers_job.sh --stage 2 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "audioset_v${no}"
#         done
#     done
# done
# for seed in 1 2 3 4; do
#     for no in 019 020; do
#         sbatch ./use_outliers_job.sh --start_stage 2 --stop_stage 2 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "audioset_v${no}"
#     done
# done
# scoring stage
# no=020
# col_name=section2
# for seed in 0 1 2 3 4; do
#     for threshold in 0 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.3 0.5 0.7 1; do
#         sbatch ./use_outliers_job.sh --stage 2 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "audioset_v${no}"
#     done
#     sleep 60
# done
# no=019
# col_name=outlier2
# for seed in 0 1 2 3 4; do
#     for threshold in 0 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.3 0.5 0.7 1; do
#         ./use_outliers_job.sh --stage 2 --threshold "${threshold}" --col_name "${col_name}" --seed "${seed}" --no "audioset_v${no}"
#     done
#     sleep 60
# done

no=008
for no in 121 108 208; do
    for n_anomaly in 0 32; do
        ./use_outliers_job.sh --start_stage 1 --stop_stage 2 --seed "0" --no "audioset_v${no}" --n_anomaly "${n_anomaly}"
    done
done
