#!/bin/bash

# shellcheck disable=SC1091
. utils/original_funcs.sh || exit 1
available_gpus=16

# valid_ratio=0.15
# n_anomaly=-1

# ./sub_job.sh --no audioset_v226 --stage 1 --start_stage 4 --n_anomaly 8 --seed 0 --machine pump
# ./sub_job.sh --no audioset_v226 --stage 1 --start_stage 4 --n_anomaly 32 --seed 4 --machine pump
# ./sub_job.sh --no audioset_v227 --stage 1 --start_stage 4 --n_anomaly 0 --seed 0 --machine pump

# for no in 028 027 026; do
#     for seed in 0 1 2 3 4; do
#         slurm_gpu_scheduler "${available_gpus}"
#         ./job.sh \
#             --no "audioset_v${no}" \
#             --stage "1" \
#             --start_stage "3" \
#             --seed "${seed}"
#     done
# done
# for no in 128 228 127 227 126 226; do
#     for seed in 0 1 2 3 4; do
#         for n_anomaly in 0 1 2 4 8 16 32; do
#             slurm_gpu_scheduler "${available_gpus}"
#             ./job.sh \
#                 --no "audioset_v${no}" \
#                 --audioset_pow "0" \
#                 --stage "1" \
#                 --start_stage "3" \
#                 --valid_ratio "${valid_ratio}" \
#                 --use_uav "false" \
#                 --use_idmt "false" \
#                 --n_anomaly "${n_anomaly}" \
#                 --seed "${seed}"
#         done
#     done
# done

for no in 008 020 019 021; do
  for seed in 0 1 2 3 4; do
    sbatch ./job.sh \
      --no "audioset_v${no}" \
      --stage "2" \
      --start_stage "3" \
      --seed "${seed}"
  done
done
for no in 108 120 119 121 208 220 219 221; do
  for seed in 0 1 2 3 4; do
    for n_anomaly in 0 1 2 4 8 16 32; do
      ./job.sh \
        --no "audioset_v${no}" \
        --audioset_pow "0" \
        --stage "2" \
        --start_stage "3" \
        --use_uav "false" \
        --use_idmt "false" \
        --n_anomaly "${n_anomaly}" \
        --seed "${seed}"
    done
  done
done
